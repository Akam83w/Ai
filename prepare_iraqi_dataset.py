# SPDX-License-Identifier: MIT
"""Import a licensed Iraqi Arabic dialogue dataset into local UTF-8 text files.

The source dataset is synthetic and template-generated. This script keeps
train/validation splits separate, removes the disclosed scraped-car subset,
filters system messages, and deduplicates exact conversations. It does not
train a model or upload any resulting data to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PHONE_RE = re.compile(r"(?<!\d)(?:\+?964|0)7\d{9}(?!\d)")


def clean_content(value: str) -> str:
    value = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    value = PHONE_RE.sub("[رقم محذوف]", value)
    return "\n".join(line.strip() for line in value.splitlines()).strip()


def convert_rows(rows, max_rows: int = 0) -> tuple[list[str], dict]:
    conversations: list[str] = []
    seen: set[str] = set()
    skipped_scraped = 0
    skipped_invalid = 0
    duplicates = 0

    for row in rows:
        source_file = str(row.get("source_file") or "").lower()
        if "cars_plus_scraped" in source_file or row.get("real_data"):
            skipped_scraped += 1
            continue

        messages = []
        for item in row.get("messages") or []:
            role = str(item.get("role") or "").lower()
            if role not in ("user", "assistant"):
                continue
            content = clean_content(item.get("content", ""))
            if not content:
                continue
            speaker = "المستخدم" if role == "user" else "الصديق"
            messages.append(f"{speaker}: {content}")

        if len(messages) < 2 or not any(line.startswith("المستخدم:") for line in messages) or not any(line.startswith("الصديق:") for line in messages):
            skipped_invalid += 1
            continue

        block = "\n".join(messages)
        digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        conversations.append(block)

        if max_rows and len(conversations) >= max_rows:
            break

    stats = {
        "kept_conversations": len(conversations),
        "skipped_scraped_records": skipped_scraped,
        "skipped_invalid_records": skipped_invalid,
        "removed_exact_duplicates": duplicates,
    }
    return conversations, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Iraqi Arabic dialogue text locally")
    parser.add_argument("--dataset", default="ameer4wisam/iraqi-arabic-sales-dialogue-dataset")
    parser.add_argument("--config", default="v4", help="dataset version/config, e.g. v4")
    parser.add_argument("--output-dir", default="data/private/iraqi_sales")
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means all available rows")
    parser.add_argument("--max-validation-rows", type=int, default=0, help="0 means all available rows")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, name=args.config)
    output = Path(args.output_dir)
    train_dir = output / "train"
    validation_dir = output / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_stats = convert_rows(dataset["train"], args.max_train_rows)
    validation_rows, validation_stats = convert_rows(dataset["validation"], args.max_validation_rows)

    (train_dir / "iraqi_dialogues.txt").write_text("\n\n".join(train_rows) + "\n", encoding="utf-8")
    (validation_dir / "iraqi_validation.txt").write_text("\n\n".join(validation_rows) + "\n", encoding="utf-8")
    report = {
        "dataset": args.dataset,
        "config": args.config,
        "license": "CC-BY-4.0",
        "attribution": "Ameer Wisam Abdulsattar, Iraqi Arabic Sales Dialogue Dataset",
        "train": train_stats,
        "validation": validation_stats,
        "note": "Synthetic template-generated Baghdadi-based Iraqi Arabic; review samples before training.",
    }
    (output / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
