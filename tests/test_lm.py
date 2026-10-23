# SPDX-License-Identifier: MIT
"""Create the model, train it for a few steps, resume, and sample text.

Also checks document-level train/val split (no leakage) and that train_loss
and val_loss are computed from different loaders.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    CausalWindowDataset,
    describe_split,
    encode_documents,
    load_text_files,
    split_documents,
)
from eval_samples import DEFAULT_PROMPTS, sample_prompt  # noqa: E402
from generate import generate  # noqa: E402
from model import ModelConfig, TinyLM  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from train import evaluate, train  # noqa: E402

BANNED_IMPORTS = (
    "openai",
    "anthropic",
    "requests",
    "httpx",
    "huggingface_hub",
    "transformers",
    "torch.hub",
)


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_finite_loss(self) -> None:
        cfg = ModelConfig(vocab_size=40, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0)
        model = TinyLM(cfg)
        idx = torch.randint(1, cfg.vocab_size, (3, 10))
        targets = torch.randint(1, cfg.vocab_size, (3, 10))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (3, 10, cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.parameter_count(), 1000)

    def test_future_tokens_do_not_change_past_logits(self) -> None:
        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=30, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0)
        model = TinyLM(cfg).eval()
        left = torch.randint(1, cfg.vocab_size, (2, 12))
        right = left.clone()
        right[:, 8:] = torch.randint(1, cfg.vocab_size, (2, 4))
        with torch.no_grad():
            logits_left, _ = model(left)
            logits_right, _ = model(right)
        torch.testing.assert_close(logits_left[:, :8], logits_right[:, :8], atol=1e-5, rtol=1e-5)

    def test_loss_falls_on_repeated_text(self) -> None:
        torch.manual_seed(1)
        text = "القطة تأكل السمك ثم تنام. " * 40
        tokenizer = CharTokenizer.from_texts([text])
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            max_seq_len=32,
            dropout=0.0,
            pad_id=tokenizer.pad_id,
        )
        model = TinyLM(cfg)
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        dataset = CausalWindowDataset(ids, seq_len=16, stride=8)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        def epoch_loss() -> float:
            total = 0.0
            count = 0
            for inputs, targets in loader:
                _logits, loss = model(inputs, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item())
                count += 1
            return total / count

        first = epoch_loss()
        last = first
        for _ in range(6):
            last = epoch_loss()
        self.assertLess(last, first * 0.7)
        self.assertTrue(math_isfinite(last))

    def test_tokenizer_roundtrip_and_unknown(self) -> None:
        tokenizer = CharTokenizer.from_texts(["مرحبا hello"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("مرحبا")), "مرحبا")
        self.assertIn("<?>", tokenizer.decode(tokenizer.encode("مرحبا!")))

    def test_no_external_model_clients(self) -> None:
        for path in ROOT.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for name in BANNED_IMPORTS:
                self.assertNotIn(f"import {name}", source)
                self.assertNotIn(f"from {name}", source)


class SplitAndMetricsTests(unittest.TestCase):
    def test_document_split_no_leakage(self) -> None:
        docs = [(f"f{i}.txt", f"content-{i}-" * 20) for i in range(5)]
        train, val = split_documents(docs, val_ratio=0.4, seed=7)
        train_names = {n for n, _ in train}
        val_names = {n for n, _ in val}
        self.assertTrue(train_names)
        self.assertTrue(val_names)
        self.assertEqual(train_names & val_names, set())
        self.assertEqual(train_names | val_names, {n for n, _ in docs})
        info = describe_split(train, val)
        self.assertEqual(info["train_docs"], len(train))
        self.assertEqual(info["val_docs"], len(val))

    def test_single_document_yields_empty_val(self) -> None:
        docs = [("only.txt", "نص قصير " * 50)]
        train, val = split_documents(docs, val_ratio=0.5, seed=1)
        self.assertEqual(len(train), 1)
        self.assertEqual(val, [])

    def test_train_and_val_loss_use_disjoint_docs(self) -> None:
        """Train loss on train docs; val loss on val docs — separate loaders."""
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text(("ألفا " * 80) + "\n", encoding="utf-8")
            (root / "b.txt").write_text(("بيتا " * 80) + "\n", encoding="utf-8")
            documents = load_text_files(root)
            train_docs, val_docs = split_documents(documents, val_ratio=0.5, seed=0)
            self.assertEqual(len(train_docs), 1)
            self.assertEqual(len(val_docs), 1)
            self.assertNotEqual(train_docs[0][0], val_docs[0][0])

            tokenizer = CharTokenizer.from_texts([t for _, t in documents])
            train_ids = encode_documents(tokenizer, train_docs)
            val_ids = encode_documents(tokenizer, val_docs)
            seq_len = 16
            train_ds = CausalWindowDataset(train_ids, seq_len=seq_len, stride=8)
            val_ds = CausalWindowDataset(val_ids, seq_len=seq_len, stride=8)
            train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

            cfg = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                d_model=32,
                n_layers=1,
                n_heads=4,
                d_ff=64,
                max_seq_len=32,
                dropout=0.0,
                pad_id=tokenizer.pad_id,
            )
            model = TinyLM(cfg)
            train_loss, n_train = evaluate(model, train_loader, torch.device("cpu"), max_batches=10)
            val_loss, n_val = evaluate(model, val_loader, torch.device("cpu"), max_batches=10)
            self.assertGreater(n_train, 0)
            self.assertGreater(n_val, 0)
            self.assertTrue(math_isfinite(train_loss))
            self.assertTrue(math_isfinite(val_loss))
            self.assertIsInstance(train_loss, float)
            self.assertIsInstance(val_loss, float)


class TrainGenerateTests(unittest.TestCase):
    def test_train_resume_and_generate(self) -> None:
        torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "corpus"
            data.mkdir()
            sample = "مرحبا بالعالم المحلي. hello local model. " * 30
            (data / "sample.txt").write_text(sample, encoding="utf-8")
            out = Path(tmp) / "model.pt"
            train(_args(data, out, steps=3))
            first = torch.load(out, map_location="cpu", weights_only=False)
            self.assertEqual(first["step"], 3)
            self.assertEqual(first["format"], 1)
            self.assertIn("itos", first["tokenizer"])

            resumed = _args(data, out, steps=2)
            resumed.resume = str(out)
            train(resumed)
            second = torch.load(out, map_location="cpu", weights_only=False)
            self.assertEqual(second["step"], 5)

            text = generate(
                Namespace(
                    checkpoint=str(out),
                    prompt="مرحبا",
                    max_new_tokens=16,
                    temperature=0.8,
                    top_k=5,
                    seed=0,
                    device="cpu",
                    stop_at_eos=False,
                )
            )
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
            self.assertNotIn("<bos>", text)

    def test_train_with_val_ratio_logs_val_loss(self) -> None:
        torch.manual_seed(4)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "corpus"
            data.mkdir()
            (data / "one.txt").write_text("واحد " * 100, encoding="utf-8")
            (data / "two.txt").write_text("اثنان " * 100, encoding="utf-8")
            out = Path(tmp) / "model.pt"
            args = _args(data, out, steps=4)
            args.val_ratio = 0.5
            args.val_batches = 2
            args.log_every = 2
            payload = train(args)
            self.assertIn("val_loss", payload)
            self.assertTrue(
                math_isfinite(payload["val_loss"]) or math.isnan(payload["val_loss"])
            )

    def test_fixed_prompts_produce_snapshot_text(self) -> None:
        torch.manual_seed(5)
        text = "في ورشة صغيرة قال سامي هطل المطر الخبز يحتاج القطة. " * 20
        tokenizer = CharTokenizer.from_texts([text])
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=64,
            max_seq_len=32,
            dropout=0.0,
            pad_id=tokenizer.pad_id,
        )
        model = TinyLM(cfg).eval()
        self.assertGreaterEqual(len(DEFAULT_PROMPTS), 3)
        for i, prompt in enumerate(DEFAULT_PROMPTS[:3]):
            out = sample_prompt(
                model,
                tokenizer,
                prompt,
                max_new_tokens=12,
                temperature=0.8,
                top_k=10,
                device=torch.device("cpu"),
                seed=10 + i,
            )
            self.assertIsInstance(out, str)
            self.assertGreater(len(out), 0)

    def test_missing_corpus_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_text_files(tmp)


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _args(data: Path, out: Path, steps: int) -> Namespace:
    return Namespace(
        data=str(data),
        preset="tiny",
        steps=steps,
        batch_size=4,
        seq_len=32,
        stride=8,
        lr=3e-4,
        weight_decay=0.0,
        grad_clip=1.0,
        out=str(out),
        resume="",
        log_every=1,
        save_every=0,
        val_ratio=0.0,
        val_batches=1,
        seed=2,
        device="cpu",
    )


if __name__ == "__main__":
    unittest.main()
