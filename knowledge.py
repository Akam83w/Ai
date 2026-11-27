# SPDX-License-Identifier: MIT
"""Local Arabic dialogue retriever with intent rules and light context awareness.

This is extractive lookup over text you supplied, not a full language model.
It prefers high-confidence matches and explicit intents (greeting, identity,
thanks, goodbye) so short social turns do not fall through to free generation
that was trained on sales talk.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_ARABIC_STOPWORDS = {
    "شنو", "شن", "شلون", "كيف", "ماذا", "هل", "من", "هو", "هي", "اني",
    "انت", "انتي", "انا", "اريد", "ممكن", "لو", "اذا", "يعني", "على",
    "عن", "في", "منو", "شنهي", "هاي", "هذا", "هذه", "هسه", "بعد",
    "يا", "لي", "لك", "له", "لها", "بي", "بينا", "ويا", "بس", "مو",
}

_GREETING_QUERY = re.compile(
    r"^(?:(?:هلا|هلو|مرحبا|مرحباً|السلام عليكم|السلام|شلونك|شلونچ|كيف حالك|شخبارك|"
    r"شلون الحال|صباح الخير|مساء الخير|اهلا|أهلا)"
    r"(?:\s+(?:بيك|بيچ|عليكم|ورحمه|ورحمة|الله|وبركاته|اخي|عزيزي|حبيب|حبيبي|غالي))?"
    r"(?:\s+(?:شلونك|شلونچ|كيف حالك|شخبارك|شلون الحال))?)+$"
)

_THANKS_QUERY = re.compile(
    r"^(?:(?:شكرا|شكراً|مشكور|مشكورة|تسلم|تسلمين|يعطيكم العافيه|الله يخليك|"
    r"ممنون|ممنونه)(?:\s+\w+){0,4})$"
)

_GOODBYE_QUERY = re.compile(
    r"^(?:(?:مع السلامه|مع السلامة|باي|يلا باي|اشوفك|نشوفك|الى اللقاء|"
    r"إلى اللقاء|تصبح على خير|الله وياك)(?:\s+\w+){0,3})$"
)

_IDENTITY_QUERY = re.compile(
    r"(?:منو انت|من انت|شنو اسمك|شو اسمك|من تكون|تعرفني|انت منو|"
    r"منو انتو|شنو صفتك|انت شنو)"
)

_INTENT_FALLBACKS = {
    "greeting": "هلا بيك! أني بخير، إنت شلونك اليوم؟",
    "thanks": "العفو، أي خدمة. إذا تحتاج شي ثاني گلي.",
    "goodbye": "الله وياك، أي وقت تحتاجني أنا موجود.",
    "identity": "أني صديقك الرقمي OwnedLM. أگدر أساعدك بالمحادثة والتفكير، وحدودي واضحة.",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي",
        "ؤ": "و", "ئ": "ي", "ة": "ه",
    }))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _terms(text: str) -> set[str]:
    return {
        token for token in normalize(text).split()
        if len(token) > 1 and token not in _ARABIC_STOPWORDS
    }


def _trigrams(text: str) -> set[str]:
    compact = normalize(text).replace(" ", "")
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i:i + 3] for i in range(len(compact) - 2)}


def _similarity(query: str, candidate: str) -> float:
    q_terms, c_terms = _terms(query), _terms(candidate)
    if not q_terms or not c_terms:
        q_tri, c_tri = _trigrams(query), _trigrams(candidate)
        if not q_tri or not c_tri:
            return 0.0
        return len(q_tri & c_tri) / max(1, len(q_tri | c_tri))
    overlap = len(q_terms & c_terms)
    dice = (2.0 * overlap) / (len(q_terms) + len(c_terms))
    q_tri, c_tri = _trigrams(query), _trigrams(candidate)
    tri = len(q_tri & c_tri) / max(1, len(q_tri | c_tri))
    return 0.7 * dice + 0.3 * tri


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_QUERY.fullmatch(normalize(text)))


def _is_thanks(text: str) -> bool:
    return bool(_THANKS_QUERY.fullmatch(normalize(text)))


def _is_goodbye(text: str) -> bool:
    return bool(_GOODBYE_QUERY.fullmatch(normalize(text)))


def _is_identity(text: str) -> bool:
    return bool(_IDENTITY_QUERY.search(normalize(text)))


def detect_intent(text: str) -> str | None:
    if _is_greeting(text):
        return "greeting"
    if _is_thanks(text):
        return "thanks"
    if _is_goodbye(text):
        return "goodbye"
    if _is_identity(text):
        return "identity"
    return None


def _looks_like_sales_answer(answer: str) -> bool:
    n = normalize(answer)
    sales_markers = (
        "سعر", "دينار", "فلوس", "بضاع", "زبون", "بيع", "شراء",
        "خصم", "قسط", "مورد", "مخزون", "فاتوره", "فاتورة",
    )
    return any(marker in n for marker in sales_markers)


@lru_cache(maxsize=4)
def load_pairs(corpus_dir: str) -> tuple[tuple[str, str, str], ...]:
    root = Path(corpus_dir)
    pairs: list[tuple[str, str, str]] = []
    if not root.is_dir():
        return tuple()
    for path in sorted(root.rglob("*.txt")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        pending_question = None
        for line in lines:
            line = line.strip()
            if line.startswith("المستخدم:"):
                pending_question = line.split(":", 1)[1].strip()
            elif line.startswith("الصديق:") and pending_question:
                answer = line.split(":", 1)[1].strip()
                if pending_question and answer:
                    pairs.append((pending_question, answer, str(path.relative_to(root))))
                pending_question = None
    return tuple(pairs)


def clear_pair_cache() -> None:
    load_pairs.cache_clear()


def _with_context(query: str, context_turns: list[str] | None) -> str:
    if not context_turns:
        return query
    recent = [t.strip() for t in context_turns if t and t.strip()][-2:]
    if not recent:
        return query
    return " ".join(recent + [query])


def retrieve_answer(
    query: str,
    corpus_dir: str | Path,
    min_score: float = 0.48,
    context_turns: list[str] | None = None,
) -> dict | None:
    if not query.strip():
        return None

    corpus_key = str(Path(corpus_dir).resolve()) if Path(corpus_dir).exists() else str(corpus_dir)
    pairs = list(load_pairs(corpus_key))
    intent = detect_intent(query)

    if intent is not None:
        intent_checkers = {
            "greeting": _is_greeting,
            "thanks": _is_thanks,
            "goodbye": _is_goodbye,
            "identity": _is_identity,
        }
        checker = intent_checkers[intent]
        candidates = [
            (q, a, src) for q, a, src in pairs
            if checker(q) and not (intent == "greeting" and _looks_like_sales_answer(a))
        ]
        if candidates:
            def _rank(pair: tuple[str, str, str]) -> tuple[float, int]:
                q, a, _src = pair
                return (_similarity(query, q), -len(a))

            question, answer, source = max(candidates, key=_rank)
            if intent in {"thanks", "goodbye", "greeting"} and len(answer) > 120:
                return {
                    "answer": _INTENT_FALLBACKS[intent],
                    "matched_question": query,
                    "source": "intent_fallback",
                    "score": 1.0,
                    "intent": intent,
                }
            return {
                "answer": answer,
                "matched_question": question,
                "source": source,
                "score": 1.0,
                "intent": intent,
            }
        return {
            "answer": _INTENT_FALLBACKS[intent],
            "matched_question": query,
            "source": "intent_fallback",
            "score": 1.0,
            "intent": intent,
        }

    search_query = _with_context(query, context_turns)
    best = None
    for question, answer, source in pairs:
        score = _similarity(search_query, question)
        if context_turns:
            score = max(score, _similarity(query, question))
        if best is None or score > best["score"]:
            best = {
                "answer": answer,
                "matched_question": question,
                "source": source,
                "score": round(score, 4),
                "intent": None,
            }
    if best is None or best["score"] < min_score:
        return None
    return best


def build_rag_context(
    query: str,
    corpus_dir: str | Path,
    top_k: int = 3,
    min_score: float = 0.35,
) -> list[dict]:
    corpus_key = str(Path(corpus_dir).resolve()) if Path(corpus_dir).exists() else str(corpus_dir)
    scored: list[dict] = []
    for question, answer, source in load_pairs(corpus_key):
        score = _similarity(query, question)
        if score >= min_score:
            scored.append({
                "question": question,
                "answer": answer,
                "source": source,
                "score": round(score, 4),
            })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
