# SPDX-License-Identifier: MIT
"""Create the model, train it for a few steps, resume, and sample text."""

from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import CausalWindowDataset, load_text_files  # noqa: E402
from generate import generate  # noqa: E402
from model import ModelConfig, TinyLM  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from train import train  # noqa: E402

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
