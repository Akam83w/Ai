# SPDX-License-Identifier: MIT
"""Tests for local Arabic corpus retrieval and intent handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from knowledge import detect_intent, load_pairs, retrieve_answer, clear_pair_cache


class KnowledgeRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_pair_cache()

    def test_loads_adjacent_arabic_dialogue_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text(
                "المستخدم: شلون أعرف إذا المشروع مربح؟\n"
                "الصديق: احسب الربح الإجمالي واطرح المصاريف.\n",
                encoding="utf-8",
            )
            pairs = load_pairs(str(root.resolve()))
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][1], "احسب الربح الإجمالي واطرح المصاريف.")

    def test_exact_question_returns_stored_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text(
                "المستخدم: شلون أعرف إذا المشروع مربح؟\n"
                "الصديق: احسب الربح الإجمالي واطرح المصاريف.\n",
                encoding="utf-8",
            )
            result = retrieve_answer("شلون أعرف إذا المشروع مربح؟", root)
            self.assertIsNotNone(result)
            self.assertEqual(result["answer"], "احسب الربح الإجمالي واطرح المصاريف.")
            self.assertGreaterEqual(result["score"], 0.48)

    def test_unrelated_question_does_not_get_false_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text(
                "المستخدم: شلون أعرف إذا المشروع مربح؟\n"
                "الصديق: احسب الربح الإجمالي واطرح المصاريف.\n",
                encoding="utf-8",
            )
            result = retrieve_answer("من بنى سور الصين العظيم؟", root)
            self.assertIsNone(result)

    def test_greeting_never_returns_sales_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text(
                "المستخدم: هلا شلونك\n"
                "الصديق: هلا بيك! أني بخير، إنت شلونك؟\n\n"
                "المستخدم: شكد السعر؟\n"
                "الصديق: سعره 30 دينار.\n",
                encoding="utf-8",
            )
            result = retrieve_answer("هلا شلونك", root)
            self.assertIsNotNone(result)
            self.assertEqual(result["intent"], "greeting")
            self.assertNotIn("30", result["answer"])
            self.assertNotIn("سعر", result["answer"])

    def test_greeting_fallback_when_corpus_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = retrieve_answer("هلا شلونك", root)
            self.assertIsNotNone(result)
            self.assertEqual(result["intent"], "greeting")
            self.assertIn("هلا", result["answer"])

    def test_detect_intent_variants(self) -> None:
        self.assertEqual(detect_intent("هلا شلونك"), "greeting")
        self.assertEqual(detect_intent("شكرا"), "thanks")
        self.assertEqual(detect_intent("مع السلامة"), "goodbye")
        self.assertEqual(detect_intent("منو انت؟"), "identity")
        self.assertIsNone(detect_intent("شلون أحسب الربح؟"))


if __name__ == "__main__":
    unittest.main()
