# SPDX-License-Identifier: MIT
"""Local CLI chat for OwnedLM knowledge + optional HTTP API.

Examples:
  python chat_cli.py
  python chat_cli.py --api http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from conversation import decide_policy, polish_reply
from knowledge import clear_pair_cache, retrieve_answer

ROOT = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE = ROOT / "data" / "corpus"


def local_reply(user_text: str, knowledge_dir: Path, prior_user: list[str]) -> tuple[str, str]:
    policy = decide_policy(user_text, history_len=len(prior_user) + 1)
    match = retrieve_answer(user_text, knowledge_dir, context_turns=prior_user or None)
    if match is not None:
        answer = polish_reply(match["answer"], block_sales=policy.block_sales, user_text=user_text)
        return answer, f"knowledge:{match.get('intent') or match.get('score')}"
    # Honest fallback when no strong local match and no remote model.
    if policy.intent == "greeting":
        return "هلا بيك! أني بخير، إنت شلونك؟", "fallback:greeting"
    if policy.intent == "thanks":
        return "العفو، أي خدمة.", "fallback:thanks"
    if policy.intent == "goodbye":
        return "الله وياك.", "fallback:goodbye"
    if policy.intent == "identity":
        return "أني OwnedLM، صديق محادثة محلي. أجاوب من بيانات المشروع والمعرفة المحلية.", "fallback:identity"
    return (
        "ما لقيت جواب جاهز بهالمعرفة المحلية. "
        "إذا شغّال عندك api_server مع نموذج Jais، استخدم --api عشان توليد أوسع.",
        "fallback:none",
    )


def api_reply(base_url: str, session_id: str, messages: list[dict]) -> tuple[str, str]:
    payload = {
        "messages": messages,
        "session_id": session_id,
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answer = data["choices"][0]["message"]["content"]
    mode = data.get("mode", "api")
    return answer, str(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="OwnedLM local chat")
    parser.add_argument("--knowledge", default=str(DEFAULT_KNOWLEDGE))
    parser.add_argument("--api", default="", help="optional API base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--session", default="cli-session")
    args = parser.parse_args()

    clear_pair_cache()
    knowledge_dir = Path(args.knowledge)
    print("OwnedLM chat — اكتب exit للخروج")
    print(f"knowledge={knowledge_dir} api={args.api or 'local-only'}")
    history: list[dict] = []
    prior_user: list[str] = []

    while True:
        try:
            user_text = input("أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "خروج"}:
            break
        if args.api:
            history.append({"role": "user", "content": user_text})
            try:
                answer, mode = api_reply(args.api, args.session, history)
            except Exception as exc:
                print(f"[خطأ API] {type(exc).__name__}: {exc}")
                continue
            history.append({"role": "assistant", "content": answer})
        else:
            answer, mode = local_reply(user_text, knowledge_dir, prior_user)
            prior_user.append(user_text)
        print(f"OwnedLM ({mode}): {answer}")


if __name__ == "__main__":
    main()
