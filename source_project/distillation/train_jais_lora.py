#!/usr/bin/env python3
"""LoRA SFT for the project's Jais-3 base; saves an adapter, not a merged model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def read_jsonl(path: str) -> list[dict[str, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            prompt = str(item.get("prompt", "")).strip()
            response = str(item.get("response", "")).strip()
            if prompt and response:
                rows.append({"prompt": prompt, "response": response})
    if not rows:
        raise ValueError(f"No usable prompt/response rows in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Dhunun/Jais-3.0.9B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", default="adapters/ownedlm-arabic-v1")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.max_length < 128 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("max-length must be >=128 and batch/accumulation must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("LoRA training requires a CUDA GPU; run this script on a GPU training machine.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def encode_rows(rows: list[dict[str, str]]) -> Dataset:
        encoded = []
        for row in rows:
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            ids = list(ids)[-args.max_length:]
            if len(ids) >= 2:
                encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        if not encoded:
            raise ValueError("No examples fit the selected max-length")
        return Dataset.from_list(encoded)

    train_rows = read_jsonl(args.train_file)
    valid_rows = read_jsonl(args.validation_file)
    if args.limit:
        train_rows = train_rows[:args.limit]
        valid_rows = valid_rows[:max(1, min(len(valid_rows), args.limit // 10 or 1))]

    train_dataset = encode_rows(train_rows)
    valid_dataset = encode_rows(valid_rows)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
    )
    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved OwnedLM Arabic LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
