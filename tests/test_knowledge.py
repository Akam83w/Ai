# SPDX-License-Identifier: MIT
"""Tests for local Arabic corpus retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from knowledge import load_pairs, retrieve_answer


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_loads_adjacent_arabic_dialogue_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text(
                "المستخدم: شلون أعرف إذا المشروع مربح؟\n"
                "الصديق: احسب الربح الإجمالي واطرح المصاريف.\n",
                encoding="utf-8",
            )
            pairs = load_pairs(root)
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


if __name__ == "__main__":
    unittest.main()
