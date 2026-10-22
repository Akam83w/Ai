# SPDX-License-Identifier: MIT
"""Train TinyLM on local text files and write a resumable checkpoint.

No network access. The optimizer, step, config, and tokenizer are stored
beside the weights so training can continue later on the same machine.

Metrics
-------
train_loss
    Mean cross-entropy over the training batches in the current log window
    (or the single step being logged). Measures fit to training documents only.

val_loss
    Mean cross-entropy over up to --val-batches from the validation loader.
    Validation documents never appear in the training loader (document-level
    split). A falling train_loss with a rising val_loss is a sign of
    memorization, not generalization. With only the seed corpus, val_loss is
    usually empty or too noisy to interpret.

val_ppl
    exp(min(val_loss, 20)). Bounded so extreme loss does not overflow.
    Not comparable to commercial model perplexities (different tokenization
    and data scale).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    CausalWindowDataset,
    describe_split,
    encode_documents,
    load_text_files,
    split_documents,
)
from model import PRESETS, ModelConfig, TinyLM, config_from_preset
from tokenizer import CharTokenizer

CHECKPOINT_FORMAT = 1


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cycle_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_loader(
    token_ids: list[int], seq_len: int, stride: int, batch_size: int, seed: int
) -> DataLoader:
    dataset = CausalWindowDataset(token_ids, seq_len=seq_len, stride=stride)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )


def evaluate(
    model: TinyLM, loader: DataLoader, device: torch.device, max_batches: int
) -> tuple[float, int]:
    """Return (mean_loss, n_batches). mean_loss is nan if no batches ran."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if i >= max_batches:
                break
            _logits, loss = model(inputs.to(device), targets.to(device))
            losses.append(float(loss.item()))
    model.train()
    if not losses:
        return float("nan"), 0
    return sum(losses) / len(losses), len(losses)


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    documents = load_text_files(args.data)
    start_step = 0
    resume_ckpt = None

    if args.resume:
        resume_path = Path(args.resume)
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format in {resume_path}")
        tokenizer = CharTokenizer.from_dict(resume_ckpt["tokenizer"])
        cfg = ModelConfig.from_dict(resume_ckpt["config"])
        start_step = int(resume_ckpt["step"])
    else:
        tokenizer = CharTokenizer.from_texts([text for _name, text in documents])
        cfg = config_from_preset(args.preset, tokenizer.vocab_size, pad_id=tokenizer.pad_id)

    if args.seq_len > cfg.max_seq_len:
        raise ValueError(f"--seq-len {args.seq_len} is above the model max_seq_len {cfg.max_seq_len}")

    train_docs, val_docs = split_documents(documents, args.val_ratio, args.seed)
    split_info = describe_split(train_docs, val_docs)
    train_ids = encode_documents(tokenizer, train_docs)
    unk = sum(1 for token in train_ids if token == tokenizer.unk_id)
    usable = len(train_ids)
    seq_len = args.seq_len
    if usable < seq_len + 1:
        seq_len = max(1, usable - 1)
        print(f"warning: corpus is short, reducing seq_len from {args.seq_len} to {seq_len}")
    if seq_len < 1 or usable < seq_len + 1:
        raise ValueError("training text is too short to build one window")

    train_loader = build_loader(train_ids, seq_len, args.stride, args.batch_size, args.seed)
    val_loader = None
    if val_docs:
        val_ids = encode_documents(tokenizer, val_docs)
        if len(val_ids) >= seq_len + 1:
            val_loader = build_loader(val_ids, seq_len, args.stride, args.batch_size, args.seed + 1)
        else:
            print(
                f"warning: validation has {len(val_ids)} tokens; "
                f"need at least {seq_len + 1} for val_loss — skipping val"
            )

    model = TinyLM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    print(
        f"device={device} preset={args.preset if not args.resume else 'from-checkpoint'} "
        f"params={model.parameter_count():,} vocab={tokenizer.vocab_size} "
        f"train_docs={split_info['train_docs']} val_docs={split_info['val_docs']} "
        f"train_tokens={len(train_ids):,} seq_len={seq_len} "
        f"unk_tokens={unk} start_step={start_step}"
    )
    if split_info["val_names"]:
        print(f"val files: {', '.join(split_info['val_names'])}")
    if unk:
        print("warning: some characters are outside the checkpoint vocabulary and became <unk>")

    model.train()
    batches = cycle_batches(train_loader)
    running = 0.0
    started = time.time()
    last_loss = float("nan")
    steps_to_run = args.steps
    end_step = start_step + steps_to_run
    last_val_loss = float("nan")

    for step in range(start_step + 1, end_step + 1):
        inputs, targets = next(batches)
        _logits, loss = model(inputs.to(device), targets.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())
        running += last_loss

        if step % args.log_every == 0 or step == end_step:
            elapsed = max(time.time() - started, 1e-6)
            done = step - start_step
            message = f"step {step} train_loss {running / max(done, 1):.4f}"
            if val_loader is not None and (step % args.log_every == 0 or step == end_step):
                val_loss, n_val = evaluate(model, val_loader, device, args.val_batches)
                last_val_loss = val_loss
                if n_val:
                    ppl = math.exp(min(val_loss, 20))
                    message += f" val_loss {val_loss:.4f} val_ppl {ppl:.2f} (batches={n_val})"
            message += f" steps/s {done / elapsed:.2f}"
            print(message)

        if args.save_every and step % args.save_every == 0:
            payload = checkpoint_payload(model, optimizer, tokenizer, step, last_loss, last_val_loss)
            save_checkpoint(Path(args.out), payload)

    payload = checkpoint_payload(model, optimizer, tokenizer, end_step, last_loss, last_val_loss)
    save_checkpoint(Path(args.out), payload)
    print(f"saved {args.out}")
    return payload


def checkpoint_payload(
    model, optimizer, tokenizer: CharTokenizer, step: int, loss: float, val_loss: float = float("nan")
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "loss": loss,
        "val_loss": val_loss,
        "config": model.cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small Transformer from random weights")
    parser.add_argument("--data", default="data/corpus", help="directory of .txt files you may use")
    parser.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--resume", default="", help="checkpoint to continue from")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="fraction of *documents* held out for validation (0 disables; needs >=2 files)",
    )
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    train(args)


if __name__ == "__main__":
    main()
