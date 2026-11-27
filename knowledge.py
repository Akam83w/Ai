# SPDX-License-Identifier: MIT
"""Small extractive retriever over the project's own Arabic dialogue corpus.

This is a local knowledge lookup, not a language model and not a claim of
semantic understanding. It returns an answer only when lexical similarity
to a stored user question is strong enough.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_ARABIC_STOPWORDS = {
    "شنو", "شن", "شلون", "كيف", "ماذا", "هل", "من", "هو", "هي", "اني",
    "انت", "انتي", "انا", "اريد", "ممكن", "لو", "اذا", "يعني", "على",
    "عن", "في", "منو", "شنهي", "هاي", "هذا", "هذه", "هسه", "بعد",
}

# Greetings need a small intent rule: words such as "شلونك" are otherwise
# removed as stopwords, leaving no lexical terms to compare with the corpus.
_GREETING_QUERY = re.compile(
    r"^(?:(?:هلا|هلو|مرحبا|السلام عليكم|شلونك|كيف حالك|شخبارك|شلون الحال)"
    r"(?:\s+(?:بيك|عليكم|ورحمه|الله|وبركاته|اخي|عزيزي|حبيبي))?"
    r"(?:\s+(?:شلونك|كيف حالك|شخبارك))?)+$"
)


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
        return 0.0
    overlap = len(q_terms & c_terms)
    dice = (2.0 * overlap) / (len(q_terms) + len(c_terms))
    q_tri, c_tri = _trigrams(query), _trigrams(candidate)
    tri = len(q_tri & c_tri) / max(1, len(q_tri | c_tri))
    return 0.75 * dice + 0.25 * tri


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_QUERY.fullmatch(normalize(text)))


def load_pairs(corpus_dir: str | Path) -> list[tuple[str, str, str]]:
    """Read adjacent المستخدم/الصديق turns from UTF-8 .txt corpus files."""
    root = Path(corpus_dir)
    pairs: list[tuple[str, str, str]] = []
    if not root.is_dir():
        return pairs
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
    return pairs


def retrieve_answer(
    query: str,
    corpus_dir: str | Path,
    min_score: float = 0.48,
) -> dict | None:
    """Return a stored answer only for a clear greeting or strong text match."""
    if not query.strip():
        return None

    pairs = load_pairs(corpus_dir)
    # Handle short greetings explicitly. The normal scorer treats "شلون" and
    # "كيف" as stopwords, so a plain "شلونك؟" otherwise never matches the
    # greeting example and gets sent to free-form generation instead.
    if _is_greeting(query):
        greeting_pairs = [(question, answer, source) for question, answer, source in pairs
                          if _is_greeting(question)]
        if greeting_pairs:
            question, answer, source = max(
                greeting_pairs, key=lambda pair: _similarity(query, pair[0])
            )
            return {
                "answer": answer,
                "matched_question": question,
                "source": source,
                "score": 1.0,
            }

    best = None
    for question, answer, source in pairs:
        score = _similarity(query, question)
        if best is None or score > best["score"]:
            best = {
                "answer": answer,
                "matched_question": question,
                "source": source,
                "score": round(score, 4),
            }
    if best is None or best["score"] < min_score:
        return None
    return best
