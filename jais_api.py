"""Jais 3 chat API for the OwnedLM project.

Run:
  pip install -r requirements-jais.txt
  JAIS_MODEL_ID=Dhunun/Jais-3.0.9B python jais_api.py

The API uses the model's native K2 Horizon chat template and returns only
the assistant answer, not the internal reasoning segment.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("JAIS_MODEL_ID", "Dhunun/Jais-3.0.9B")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

app = FastAPI(title="OwnedLM Jais API", version="0.2.0")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=24)
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.05, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def extract_answer(text: str) -> str:
    """Remove the K2 Horizon reasoning section and trailing turn markers."""
    for marker in ("</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
            break

    for marker in ("<|ifm|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if marker in text:
            text = text.split(marker, 1)[0]

    return text.strip()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE.type,
    }


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    try:
        tokenizer, model = load_model()
        messages = [
            {"role": item.role, "content": item.content}
            for item in request.messages
        ]

        # Use Jais 3's own K2 Horizon template. The previous generic pipeline
        # call did not set reasoning_effort and could expose an unsuitable
        # prompt/completion format to the user.
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort="high",
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        inputs.pop("token_type_ids", None)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=20,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = output[0, prompt_length:]
        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False,
        )
        answer = extract_answer(generated_text)

        if not answer:
            raise RuntimeError("The model returned an empty assistant answer.")

        return {
            "model": MODEL_ID,
            "choices": [
                {"message": {"role": "assistant", "content": answer}}
            ],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {type(exc).__name__}: {exc}",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
