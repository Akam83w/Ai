# SPDX-License-Identifier: MIT
"""Tests for conversation policy and reply polish."""

from __future__ import annotations

import unittest

from conversation import (
    build_system_prompt,
    continuity_note,
    decide_policy,
    format_history_for_prompt,
    polish_reply,
    style_hint_from_user,
)


class ConversationPolicyTests(unittest.TestCase):
    def test_greeting_policy_is_short_and_blocks_sales(self) -> None:
        policy = decide_policy("هلا شلونك", history_len=1)
        self.assertEqual(policy.intent, "greeting")
        self.assertTrue(policy.prefer_short)
        self.assertTrue(policy.block_sales)
        self.assertLessEqual(policy.max_tokens, 80)

    def test_open_question_allows_longer_reply(self) -> None:
        policy = decide_policy("شلون أركز بالشغل؟", history_len=3)
        self.assertIsNone(policy.intent)
        self.assertFalse(policy.block_sales)
        self.assertGreaterEqual(policy.max_tokens, 384)

    def test_polish_strips_sales_on_greeting(self) -> None:
        dirty = "هلا بيك، سعره 30 دينار اليوم."
        clean = polish_reply(dirty, block_sales=True, user_text="هلا شلونك")
        self.assertNotIn("30", clean)
        self.assertNotIn("سعر", clean)
        self.assertIn("هلا", clean)

    def test_polish_removes_role_echo(self) -> None:
        text = "الصديق: تمام، نكمل.\nالمستخدم: شنو؟"
        out = polish_reply(text, block_sales=False)
        self.assertFalse(out.startswith("الصديق:"))
        self.assertNotIn("المستخدم:", out)

    def test_system_prompt_includes_rag_and_style(self) -> None:
        prompt = build_system_prompt(
            rag_block="سؤال مشابه: اختبار\nجواب مناسب: نعم",
            style_hint="المستخدم يتكلم بلهجة عراقية؛ رد بلهجة عراقية طبيعية ومفهومة.",
        )
        self.assertIn("OwnedLM", prompt)
        self.assertIn("اختبار", prompt)
        self.assertIn("عراقي", prompt)

    def test_style_hint_detects_iraqi(self) -> None:
        hint = style_hint_from_user("شلونك هسه؟")
        self.assertIn("عراق", hint)

    def test_format_history_keeps_system_and_trims(self) -> None:
        messages = [{"role": "system", "content": "sys"}]
        for i in range(20):
            messages.append({"role": "user", "content": f"u{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})
        out = format_history_for_prompt(messages, max_turns=6)
        self.assertEqual(out[0]["role"], "system")
        self.assertLessEqual(len([m for m in out if m["role"] != "system"]), 6)

    def test_continuity_note_uses_last_turns(self) -> None:
        note = continuity_note(["موضوع النوم"], ["جرب تقلل الشاشات"])
        self.assertIn("النوم", note)
        self.assertIn("الشاشات", note)


if __name__ == "__main__":
    unittest.main()
