#!/usr/bin/env python3
"""Collect Arabic teaching examples from the project's Jais-compatible chat API.

This creates a local distillation corpus; it does not copy model weights.
Generated files are written under data/private and should not be committed.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ROOT / "distillation" / "arabic_topics.json"


def ask_api(base_url: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "أنت معلّم عربي دقيق. أجب بالعربية فقط. اشرح بوضوح وبأمثلة صحيحة. "
                    "ميّز الحقيقة من الرأي، ولا تخترع مصادر أو أرقاماً. عند عدم التأكد صرّح بذلك. "
                    "استخدم الفصحى أساساً، وأضف مثالاً عراقياً عند ملاءمة الموضوع."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Arabic distillation examples from Jais")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the running Jais API")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of prompts; 0 means all")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--output", default="data/private/jais_arabic_distilled.jsonl")
    args = parser.parse_args()

    if args.samples_per_task < 1 or args.max_new_tokens < 1:
        raise SystemExit("samples-per-task and max-new-tokens must be positive")

    topics = json.loads(TOPICS.read_text(encoding="utf-8"))
    jobs = [(group["area"], task) for group in topics for task in group["tasks"]]
    random.Random(args.seed).shuffle(jobs)
    if args.limit:
        jobs = jobs[: args.limit]

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_path = output_path.with_suffix(".txt")
    count = 0

    with output_path.open("a", encoding="utf-8") as jsonl, text_path.open("a", encoding="utf-8") as corpus:
        for area, task in jobs:
            for sample_index in range(args.samples_per_task):
                prompt = (
                    f"المجال: {area}.\nالمطلوب: {task}\n"
                    "اجعل الإجابة درساً مستقلاً منظماً بعناوين وأمثلة وتمرين قصير مع الحل. "
                    "لا تذكر أنك نموذج ذكاء اصطناعي ولا تكرر السؤال."
                )
                try:
                    answer = ask_api(args.base_url, prompt, args.max_new_tokens, args.temperature)
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
                    print(f"FAILED [{area}] {task}: {exc}")
                    continue
                if len(answer) < 80:
                    print(f"SKIPPED short answer [{area}] {task}")
                    continue
                row = {
                    "source": "Jais teacher-generated Arabic example",
                    "area": area,
                    "prompt": prompt,
                    "response": answer,
                }
                jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl.flush()
                corpus.write(f"سؤال: {prompt}\nجواب: {answer}\n\n")
                corpus.flush()
                count += 1
                print(f"saved {count}: {area} / sample {sample_index + 1}")
                if args.pause:
                    time.sleep(args.pause)

    print(f"Done. Added {count} examples.")
    print(f"JSONL: {output_path}")
    print(f"Training text: {text_path}")
    print("These are generated teaching examples, not the full Arabic language or copied model weights.")


if __name__ == "__main__":
    main()
