#!/usr/bin/env python3
"""Validate, deduplicate, and split Jais-generated Arabic SFT examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def normalize(value: str) -> str:
    return " ".join(value.split()).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--min-response-chars", type=int, default=100)
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 0.5:
        raise SystemExit("validation-ratio must be greater than 0 and less than 0.5")

    source = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    rejected = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
                prompt = normalize(str(item.get("prompt", "")))
                response = normalize(str(item.get("response", "")))
            except (json.JSONDecodeError, AttributeError):
                rejected += 1
                continue
            if not prompt or len(response) < args.min_response_chars:
                rejected += 1
                continue
            digest = hashlib.sha256((prompt + "\\0" + response).encode("utf-8")).hexdigest()
            if digest in seen:
                rejected += 1
                continue
            seen.add(digest)
            rows.append({
                "prompt": prompt,
                "response": response,
                "area": str(item.get("area", "غير مصنف")),
                "language": "ar",
                "register": "modern_standard_arabic",
            })

    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 valid unique examples; found {len(rows)}")

    random.Random(args.seed).shuffle(rows)
    validation_count = max(1, round(len(rows) * args.validation_ratio))
    validation = rows[:validation_count]
    training = rows[validation_count:]

    for name, items in (("train.jsonl", training), ("validation.jsonl", validation)):
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input": str(source),
        "accepted": len(rows),
        "training": len(training),
        "validation": len(validation),
        "rejected_or_duplicate": rejected,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
