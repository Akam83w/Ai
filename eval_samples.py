# SPDX-License-Identifier: MIT
"""Fixed-prompt generation snapshots for qualitative comparison.

Saves plain-text samples so you can compare the same prompts before and after
a training run. This is **not** a quality score and does not prove that the
model understands Arabic or generalizes.

Typical use:
  python eval_samples.py --checkpoint checkpoints/before.pt --out samples/before.txt
  python train.py ... --out checkpoints/after.pt
  python eval_samples.py --checkpoint checkpoints/after.pt --out samples/after.txt
  diff -u samples/before.txt samples/after.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from model import ModelConfig, TinyLM
from tokenizer import CharTokenizer

# Fixed Arabic prompts. Keep them short and stable across runs.
DEFAULT_PROMPTS = (
    "في ورشة",
    "قال سامي",
    "هطل المطر",
    "الخبز يحتاج",
    "القطة",
)


def sample_prompt(
    model: TinyLM,
    tokenizer: CharTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, add_bos=True)
    if not ids:
        ids = [tokenizer.bos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=None,
    )
    return tokenizer.decode(out[0].tolist())


def run_eval(args: argparse.Namespace) -> str:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    cfg = ModelConfig.from_dict(checkpoint["config"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompts = list(DEFAULT_PROMPTS)
    if args.prompts:
        prompts = [line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]

    top_k = None if args.top_k <= 0 else args.top_k
    lines: list[str] = []
    lines.append(f"# generation snapshot")
    lines.append(f"# checkpoint: {args.checkpoint}")
    lines.append(f"# step: {checkpoint.get('step', '?')}")
    lines.append(f"# train_loss (last): {checkpoint.get('loss', float('nan'))}")
    lines.append(f"# val_loss (last): {checkpoint.get('val_loss', float('nan'))}")
    lines.append(f"# temperature: {args.temperature}  top_k: {args.top_k}  max_new: {args.max_new_tokens}")
    lines.append(f"# seed: {args.seed}")
    lines.append(f"# time_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(
        "# Note: readable Arabic here does not mean the model is an assistant. "
        "With a tiny seed corpus, outputs often remain garbled or copy training phrases."
    )
    lines.append("")

    for i, prompt in enumerate(prompts):
        text = sample_prompt(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            device=device,
            seed=args.seed + i,
        )
        lines.append(f"## prompt {i + 1}")
        lines.append(f"prompt: {prompt}")
        lines.append(f"output: {text}")
        lines.append("")

    report = "\n".join(lines).rstrip() + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save fixed-prompt generation samples from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="samples/snapshot.txt")
    parser.add_argument("--prompts", default="", help="optional UTF-8 file, one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_eval(args)
    print(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
