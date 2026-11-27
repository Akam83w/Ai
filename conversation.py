# SPDX-License-Identifier: MIT
"""Conversation helpers: memory, response polish, and ChatGPT-like turn policy.

These utilities do not claim human-level understanding. They keep replies
on-topic, short when needed, and consistent with prior turns using only
local rules and the project's own corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge import detect_intent, normalize

# Soft limits so replies stay readable in chat UIs.
_MAX_REPLY_CHARS = 900
_MAX_REPLY_SENTENCES = 6

_SALES_LEAK = re.compile(
    r"(\u0633\u0639\u0631\s*\d+|\u0633\u0639\u0631\u0647\s*\d+|\d+\s*\u062f\u064a\u0646\u0627\u0631|\u0628\u0636\u0627\u0639\u0629|\u0641\u0627\u062a\u0648\u0631\u0629|\u0642\u0633\u0637|\u062e\u0635\u0645\s*\d+)",
    re.IGNORECASE,
)

_REPEAT_FILLER = re.compile(r"(\b[\w\u0600-\u06FF]{2,}\b)(?:\s+\1){2,}")


@dataclass
class TurnPolicy:
    """Lightweight policy decided before generation or retrieval override."""

    intent: str | None
    prefer_short: bool
    block_sales: bool
    max_tokens: int
    temperature: float


def decide_policy(user_text: str, history_len: int) -> TurnPolicy:
    intent = detect_intent(user_text)
    if intent in {"greeting", "thanks", "goodbye"}:
        return TurnPolicy(
            intent=intent,
            prefer_short=True,
            block_sales=True,
            max_tokens=80,
            temperature=0.4,
        )
    if intent == "identity":
        return TurnPolicy(
            intent=intent,
            prefer_short=True,
            block_sales=True,
            max_tokens=120,
            temperature=0.5,
        )
    return TurnPolicy(
        intent=None,
        prefer_short=False,
        block_sales=False,
        max_tokens=512 if history_len < 12 else 384,
        temperature=0.65,
    )


def polish_reply(text: str, *, block_sales: bool = False, user_text: str = "") -> str:
    """Clean model output into a single usable chat message."""
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^(?:\u0627\u0644\u0635\u062f\u064a\u0642|\u0627\u0644\u0645\u0633\u0627\u0639\u062f|Assistant)\s*[:\uff1a]\s*", "", text)
    text = re.sub(r"\n(?:\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645|User)\s*[:\uff1a].*", "", text, flags=re.DOTALL)
    text = _REPEAT_FILLER.sub(r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if block_sales and _SALES_LEAK.search(text):
        intent = detect_intent(user_text) if user_text else None
        if intent == "greeting":
            return "\u0647\u0644\u0627 \u0628\u064a\u0643! \u0623\u0646\u064a \u0628\u062e\u064a\u0631\u060c \u0625\u0646\u062a \u0634\u0644\u0648\u0646\u0643 \u0627\u0644\u064a\u0648\u0645\u061f"
        if intent == "thanks":
            return "\u0627\u0644\u0639\u0641\u0648\u060c \u0623\u064a \u062e\u062f\u0645\u0629."
        if intent == "goodbye":
            return "\u0627\u0644\u0644\u0647 \u0648\u064a\u0627\u0643\u060c \u0623\u0634\u0648\u0641\u0643 \u0639\u0644\u0649 \u062e\u064a\u0631."
        if intent == "identity":
            return "\u0623\u0646\u064a \u0635\u062f\u064a\u0642\u0643 \u0627\u0644\u0631\u0642\u0645\u064a OwnedLM. \u0623\u06af\u062f\u0631 \u0623\u0633\u0627\u0639\u062f\u0643 \u0628\u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629 \u0636\u0645\u0646 \u062d\u062f\u0648\u062f \u0648\u0627\u0636\u062d\u0629."
        text = _SALES_LEAK.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" \u060c,.-\u060c")

    if block_sales:
        parts = re.split(r"(?<=[\.\!\?\u061F])\s+", text)
        text = " ".join(parts[:2]).strip()

    if len(text) > _MAX_REPLY_CHARS:
        cut = text[:_MAX_REPLY_CHARS]
        for sep in ("\u06d4", ".", "!", "?", "\u061F", "\n"):
            pos = cut.rfind(sep)
            if pos > _MAX_REPLY_CHARS // 2:
                cut = cut[: pos + 1]
                break
        text = cut.strip()
    return text


def build_system_prompt(rag_block: str = "", style_hint: str = "") -> str:
    base = (
        "\u0623\u0646\u062a OwnedLM\u060c \u0635\u062f\u064a\u0642 \u0645\u062d\u0627\u062f\u062b\u0629 \u0639\u0631\u0628\u064a \u0648\u062f\u0648\u062f \u0648\u0648\u0627\u0636\u062d. "
        "\u0623\u062c\u0628 \u0628\u0646\u0641\u0633 \u0623\u0633\u0644\u0648\u0628 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 (\u0639\u0631\u0627\u0642\u064a/\u0639\u0627\u0645\u064a \u0623\u0648 \u0641\u0635\u062d\u0649) \u0628\u062f\u0648\u0646 \u062a\u0635\u0646\u0651\u0639. "
        "\u0642\u0648\u0627\u0639\u062f \u0645\u0647\u0645\u0629:\n"
        "1) \u062c\u0627\u0648\u0628 \u0639\u0644\u0649 \u0622\u062e\u0631 \u0633\u0624\u0627\u0644 \u0645\u0628\u0627\u0634\u0631\u0629\u061b \u0644\u0627 \u062a\u063a\u064a\u0651\u0631 \u0627\u0644\u0645\u0648\u0636\u0648\u0639.\n"
        "2) \u0625\u0630\u0627 \u0643\u0627\u0646\u062a \u0627\u0644\u0631\u0633\u0627\u0644\u0629 \u062a\u062d\u064a\u0629 \u0641\u0642\u0637\u060c \u0631\u062f \u0628\u062a\u062d\u064a\u0629 \u0642\u0635\u064a\u0631\u0629 \u0648\u062f\u0648\u062f\u0629 \u0628\u062f\u0648\u0646 \u0623\u0633\u0639\u0627\u0631 \u0623\u0648 \u0645\u0646\u062a\u062c\u0627\u062a.\n"
        "3) \u0625\u0630\u0627 \u0645\u0627 \u062a\u0639\u0631\u0641 \u0627\u0644\u062c\u0648\u0627\u0628\u060c \u0642\u0644 \u0630\u0644\u0643 \u0628\u0635\u0631\u0627\u062d\u0629 \u0648\u0644\u0627 \u062a\u062e\u062a\u0631\u0639 \u0623\u0631\u0642\u0627\u0645\u0627\u064b \u0623\u0648 \u062d\u0642\u0627\u0626\u0642.\n"
        "4) \u062d\u0627\u0641\u0638 \u0639\u0644\u0649 \u0633\u064a\u0627\u0642 \u0627\u0644\u0631\u0633\u0627\u0626\u0644 \u0627\u0644\u0633\u0627\u0628\u0642\u0629 \u0641\u064a \u0647\u0630\u0647 \u0627\u0644\u0645\u062d\u0627\u062f\u062b\u0629.\n"
        "5) \u0643\u0646 \u0645\u062e\u062a\u0635\u0631\u0627\u064b \u0648\u0645\u0641\u064a\u062f\u0627\u064b\u061b \u0641\u0642\u0631\u0627\u062a \u0642\u0635\u064a\u0631\u0629 \u0623\u0641\u0636\u0644 \u0645\u0646 \u0643\u0644\u0627\u0645 \u0637\u0648\u064a\u0644 \u0628\u0644\u0627 \u0645\u0639\u0646\u0649.\n"
        "6) \u0644\u0627 \u062a\u0630\u0643\u0631 \u0623\u0646\u0643 ChatGPT \u0623\u0648 \u0646\u0645\u0627\u0630\u062c \u0634\u0631\u0643\u0627\u062a \u0623\u062e\u0631\u0649\u061b \u0623\u0646\u062a OwnedLM \u0627\u0644\u0645\u062d\u0644\u064a."
    )
    if style_hint:
        base += "\n" + style_hint
    if rag_block:
        base += "\n\n" + rag_block
    return base


def style_hint_from_user(user_text: str) -> str:
    n = normalize(user_text)
    iraqi_markers = ("\u0634\u0644\u0648\u0646", "\u0634\u0646\u0648", "\u0647\u0633\u0647", "\u06af", "\u0686", "\u0645\u0627\u0643\u0648", "\u0627\u0643\u0648", "\u0647\u0627\u064a", "\u0648\u064a\u0646")
    if any(m in n for m in iraqi_markers):
        return "\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 \u064a\u062a\u0643\u0644\u0645 \u0628\u0644\u0647\u062c\u0629 \u0639\u0631\u0627\u0642\u064a\u0629\u061b \u0631\u062f \u0628\u0644\u0647\u062c\u0629 \u0639\u0631\u0627\u0642\u064a\u0629 \u0637\u0628\u064a\u0639\u064a\u0629 \u0648\u0645\u0641\u0647\u0648\u0645\u0629."
    return "\u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 \u064a\u0645\u064a\u0644 \u0644\u0644\u0641\u0635\u062d\u0649 \u0623\u0648 \u0639\u0631\u0628\u064a\u0629 \u0639\u0627\u0645\u0629\u061b \u0631\u062f \u0628\u0648\u0636\u0648\u062d \u0648\u0628\u0644\u063a\u0629 \u0642\u0631\u064a\u0628\u0629 \u0645\u0646\u0647."


def format_history_for_prompt(messages: list[dict], max_turns: int = 10) -> list[dict]:
    """Keep system + last N user/assistant turns for the generative model."""
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(rest) > max_turns:
        rest = rest[-max_turns:]
    return system[:1] + rest


def continuity_note(prior_user: list[str], prior_assistant: list[str]) -> str:
    """One-line memory cue injected into system prompt (not shown to user)."""
    if not prior_user:
        return ""
    last_u = prior_user[-1][:120]
    last_a = prior_assistant[-1][:120] if prior_assistant else ""
    note = f"\u0622\u062e\u0631 \u0631\u0633\u0627\u0644\u0629 \u0645\u0646 \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 \u0643\u0627\u0646\u062a \u062d\u0648\u0644: {last_u}"
    if last_a:
        note += f" | \u0622\u062e\u0631 \u0631\u062f\u0643 \u0643\u0627\u0646 \u064a\u0628\u062f\u0623 \u0628\u0640: {last_a}"
    return note
