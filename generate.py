# SPDX-License-Identifier: MIT
"""Sample text from a local checkpoint.

This process does not contact a model provider. It only reads the checkpoint
you trained and prints characters.
"""
from __future__ import annotations

import argparse

import torch

from model import ModelConfig, TinyLM
from tokenizer import CharTokenizer


def generate(args: argparse.Namespace, include_prompt: bool = True) -> str:
    """Generate text locally; optionally return only tokens generated after the prompt."""
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    cfg = ModelConfig.from_dict(checkpoint["config"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = tokenizer.encode(args.prompt, add_bos=True)
    if not ids:
        ids = [tokenizer.bos_id]
    prompt_token_count = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    top_k = None if args.top_k <= 0 else args.top_k
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id if args.stop_at_eos else None,
    )
    output_ids = out[0].tolist()
    if not include_prompt:
        output_ids = output_ids[prompt_token_count:]
    return tokenizer.decode(output_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a local TinyLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stop-at-eos", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")
    print(generate(args))


if __name__ == "__main__":
    main()
