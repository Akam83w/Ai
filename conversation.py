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

_MAX_REPLY_CHARS = 900
_MAX_REPLY_SENTENCES = 6

_SALES_LEAK = re.compile(
    r"(سعر\s*\d+|سعره\s*\d+|\d+\s*دينار|بضاعة|فاتورة|قسط|خصم\s*\d+)",
    re.IGNORECASE,
)

_REPEAT_FILLER = re.compile(r"(\b[\w\u0600-\u06FF]{2,}\b)(?:\s+\1){2,}")


@dataclass
class TurnPolicy:
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
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^(?:الصديق|المساعد|Assistant)\s*[:：]\s*", "", text)
    text = re.sub(r"\n(?:المستخدم|User)\s*[:：].*", "", text, flags=re.DOTALL)
    text = _REPEAT_FILLER.sub(r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if block_sales and _SALES_LEAK.search(text):
        intent = detect_intent(user_text) if user_text else None
        if intent == "greeting":
            return "هلا بيك! أني بخير، إنت شلونك اليوم؟"
        if intent == "thanks":
            return "العفو، أي خدمة."
        if intent == "goodbye":
            return "الله وياك، أشوفك على خير."
        if intent == "identity":
            return "أني صديقك الرقمي OwnedLM. أگدر أساعدك بالمحادثة ضمن حدود واضحة."
        text = _SALES_LEAK.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ،,.-")

    if block_sales:
        parts = re.split(r"(?<=[\.\!\?\u061F])\s+", text)
        text = " ".join(parts[:2]).strip()

    if len(text) > _MAX_REPLY_CHARS:
        cut = text[:_MAX_REPLY_CHARS]
        for sep in ("۔", ".", "!", "?", "\u061F", "\n"):
            pos = cut.rfind(sep)
            if pos > _MAX_REPLY_CHARS // 2:
                cut = cut[: pos + 1]
                break
        text = cut.strip()
    return text


def build_system_prompt(rag_block: str = "", style_hint: str = "") -> str:
    base = (
        "أنت OwnedLM، صديق محادثة عربي ودود وواضح. "
        "أجب بنفس أسلوب المستخدم (عراقي/عامي أو فصحى) بدون تصنّع. "
        "قواعد مهمة:\n"
        "1) جاوب على آخر سؤال مباشرة؛ لا تغيّر الموضوع.\n"
        "2) إذا كانت الرسالة تحية فقط، رد بتحية قصيرة ودودة بدون أسعار أو منتجات.\n"
        "3) إذا ما تعرف الجواب، قل ذلك بصراحة ولا تخترع أرقاماً أو حقائق.\n"
        "4) حافظ على سياق الرسائل السابقة في هذه المحادثة.\n"
        "5) كن مختصراً ومفيداً؛ فقرات قصيرة أفضل من كلام طويل بلا معنى.\n"
        "6) لا تذكر أنك ChatGPT أو نماذج شركات أخرى؛ أنت OwnedLM المحلي."
    )
    if style_hint:
        base += "\n" + style_hint
    if rag_block:
        base += "\n\n" + rag_block
    return base


def style_hint_from_user(user_text: str) -> str:
    n = normalize(user_text)
    iraqi_markers = ("شلون", "شنو", "هسه", "گ", "چ", "ماكو", "اكو", "هاي", "وين")
    if any(m in n for m in iraqi_markers):
        return "المستخدم يتكلم بلهجة عراقية؛ رد بلهجة عراقية طبيعية ومفهومة."
    return "المستخدم يميل للفصحى أو عربية عامة؛ رد بوضوح وبلغة قريبة منه."


def format_history_for_prompt(messages: list[dict], max_turns: int = 10) -> list[dict]:
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(rest) > max_turns:
        rest = rest[-max_turns:]
    return system[:1] + rest


def continuity_note(prior_user: list[str], prior_assistant: list[str]) -> str:
    if not prior_user:
        return ""
    last_u = prior_user[-1][:120]
    last_a = prior_assistant[-1][:120] if prior_assistant else ""
    note = f"آخر رسالة من المستخدم كانت حول: {last_u}"
    if last_a:
        note += f" | آخر ردك كان يبدأ بـ: {last_a}"
    return note
