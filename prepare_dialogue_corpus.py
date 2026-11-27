# SPDX-License-Identifier: MIT
"""Format multi-turn dialogues into consistent training text.

Reads المستخدم/الصديق blocks (or HuggingFace-style prepared files) and writes
clean conversation documents so causal training sees full turns, not mixed
sales noise. Use the output folder with train.py --data.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


BLOCK_SPLIT = re.compile(r"\n\s*\n+")


def iter_blocks(text: str) -> list[str]:
    return [b.strip() for b in BLOCK_SPLIT.split(text) if b.strip()]


def is_dialogue_block(block: str) -> bool:
    has_user = any(line.strip().startswith("المستخدم:") for line in block.splitlines())
    has_friend = any(line.strip().startswith("الصديق:") for line in block.splitlines())
    return has_user and has_friend


def clean_block(block: str) -> str:
    lines = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("المستخدم:") or line.startswith("الصديق:"):
            speaker, content = line.split(":", 1)
            content = " ".join(content.split())
            if content:
                lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean dialogue corpus for OwnedLM training")
    parser.add_argument("--input", default="data/corpus", help="input directory of .txt files")
    parser.add_argument("--output", default="data/private/dialogue_train", help="output directory")
    parser.add_argument("--min-turns", type=int, default=2, help="minimum speaker lines per block")
    args = parser.parse_args()

    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for path in sorted(root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8")
        clean_blocks = []
        for block in iter_blocks(text):
            if not is_dialogue_block(block):
                skipped += 1
                continue
            cleaned = clean_block(block)
            turns = cleaned.count("المستخدم:") + cleaned.count("الصديق:")
            if turns < args.min_turns:
                skipped += 1
                continue
            clean_blocks.append(cleaned)
            kept += 1
        if clean_blocks:
            target = out / path.name
            target.write_text("\n\n".join(clean_blocks) + "\n", encoding="utf-8")

    report = out / "prepare_report.txt"
    report.write_text(
        f"kept_blocks={kept}\nskipped_blocks={skipped}\ninput={root}\noutput={out}\n",
        encoding="utf-8",
    )
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
