# SPDX-License-Identifier: MIT
"""API chat tests with mocked generative model (no Jais download)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Prevent startup from trying to download Jais during import/TestClient lifespan.
os.environ.setdefault("JAIS_MODEL_ID", "mock/does-not-exist")


class ApiChatKnowledgeFirstTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.corpus = Path(self._tmpdir.name)
        (self.corpus / "sample.txt").write_text(
            "المستخدم: هلا شلونك\n"
            "الصديق: هلا بيك! أني بخير، إنت شلونك؟\n\n"
            "المستخدم: شلون أركز بالشغل؟\n"
            "الصديق: قسم الشغل لقطع صغيرة واقفل الإشعارات.\n",
            encoding="utf-8",
        )

        import knowledge
        import api_server

        knowledge.clear_pair_cache()
        self.api = api_server
        self.api.KNOWLEDGE_DIR = self.corpus
        self.api.save_exchange = lambda *args, **kwargs: None
        self.api._model = None
        self.api._tokenizer = None
        self.api._load_error = "mocked: no weights"

        from fastapi.testclient import TestClient

        # Disable lifespan model load
        with patch.object(self.api, "load_checkpoint", return_value=(None, None)):
            self.client = TestClient(self.api.app)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_greeting_returns_knowledge_without_loading_model(self) -> None:
        load_calls = {"n": 0}

        def _fake_load():
            load_calls["n"] += 1
            return None, None

        with patch.object(self.api, "load_checkpoint", side_effect=_fake_load):
            resp = self.client.post(
                "/v1/chat/completions",
                json={
                    "session_id": "t1",
                    "messages": [{"role": "user", "content": "هلا شلونك"}],
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["mode"], "knowledge_retrieval")
        self.assertIn("هلا", body["choices"][0]["message"]["content"])
        self.assertEqual(load_calls["n"], 0, "load_checkpoint must not run on knowledge hit")

    def test_stored_question_works_when_model_unavailable(self) -> None:
        with patch.object(self.api, "load_checkpoint", return_value=(None, None)):
            resp = self.client.post(
                "/v1/chat/completions",
                json={
                    "session_id": "t2",
                    "messages": [{"role": "user", "content": "شلون أركز بالشغل؟"}],
                },
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["mode"], "knowledge_retrieval")
        self.assertIn("قطع", body["choices"][0]["message"]["content"])

    def test_unknown_question_without_model_returns_503(self) -> None:
        with patch.object(self.api, "load_checkpoint", return_value=(None, None)):
            self.api._load_error = "mocked missing weights"
            resp = self.client.post(
                "/v1/chat/completions",
                json={
                    "session_id": "t3",
                    "messages": [{"role": "user", "content": "ما عاصمة منغوليا؟"}],
                },
            )
        self.assertEqual(resp.status_code, 503)

    def test_rag_context_builder_returns_related_pairs(self) -> None:
        from knowledge import build_rag_context, clear_pair_cache

        clear_pair_cache()
        hits = build_rag_context("شلون أركز", self.corpus, top_k=2, min_score=0.2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("question", hits[0])
        self.assertIn("answer", hits[0])


class CorpusCountTests(unittest.TestCase):
    def test_pair_counts_are_documented_honestly(self) -> None:
        from knowledge import clear_pair_cache, load_pairs
        from collections import Counter

        root = Path(__file__).resolve().parents[1] / "data" / "corpus"
        if not root.is_dir():
            self.skipTest("corpus not present")
        clear_pair_cache()
        pairs = load_pairs(str(root.resolve()))
        by_file = Counter(src for _q, _a, src in pairs)
        everyday = by_file.get("everyday_chat.txt", 0)
        companion = by_file.get("iraqi_companion.txt", 0)
        # everyday_chat is a small topical supplement (~48 pairs), not the whole corpus.
        self.assertGreaterEqual(everyday, 40)
        self.assertLess(everyday, 100)
        self.assertGreater(companion, 500)
        total_named = everyday + companion
        other = sum(v for k, v in by_file.items() if k not in ("everyday_chat.txt", "iraqi_companion.txt"))
        self.assertEqual(len(pairs), total_named + other)


if __name__ == "__main__":
    unittest.main()
