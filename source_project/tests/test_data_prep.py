# SPDX-License-Identifier: MIT
"""Tests for preparing Iraqi dialogue data."""

from __future__ import annotations

import unittest

from prepare_iraqi_dataset import convert_rows


class IraqiDatasetPreparationTests(unittest.TestCase):
    def test_filters_scraped_subset_and_removes_phone_numbers(self) -> None:
        rows = [
            {
                "source_file": "iraqi_train_04_cars_plus_scraped.json",
                "messages": [
                    {"role": "user", "content": "هلا"},
                    {"role": "assistant", "content": "أهلاً"},
                ],
            },
            {
                "source_file": "iraqi_train_02_food.json",
                "messages": [
                    {"role": "system", "content": "تعليمات داخلية"},
                    {"role": "user", "content": "رقمي 07801234567"},
                    {"role": "assistant", "content": "وصلت رسالتك"},
                ],
            },
        ]
        conversations, stats = convert_rows(rows)
        self.assertEqual(len(conversations), 1)
        self.assertNotIn("07801234567", conversations[0])
        self.assertIn("[رقم محذوف]", conversations[0])
        self.assertNotIn("تعليمات داخلية", conversations[0])
        self.assertEqual(stats["skipped_scraped_records"], 1)


if __name__ == "__main__":
    unittest.main()
