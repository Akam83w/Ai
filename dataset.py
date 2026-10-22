# SPDX-License-Identifier: MIT
"""Read local .txt files and slice them into causal language-model windows."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer


def load_text_files(directory: str | Path) -> list[tuple[str, str]]:
    """Return (relative path, text) for every .txt file under directory."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*.txt") if path.is_file())
    documents: list[tuple[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if text.strip():
            documents.append((str(path.relative_to(root)), text))
    if not documents:
        raise FileNotFoundError(
            f"no non-empty .txt files in {root}. Add text you have the right to use."
        )
    return documents


def split_documents(
    documents: list[tuple[str, str]], val_ratio: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if val_ratio <= 0 or len(documents) < 2:
        return documents, []
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(len(documents) * val_ratio)))
    n_val = min(n_val, len(documents) - 1)
    val_idx = set(order[:n_val])
    train, val = [], []
    for i, doc in enumerate(documents):
        (val if i in val_idx else train).append(doc)
    return train, val


def encode_documents(tokenizer: CharTokenizer, documents: list[tuple[str, str]]) -> list[int]:
    ids: list[int] = []
    for _name, text in documents:
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return ids


class CausalWindowDataset(Dataset):
    """Each item is (input, target) with target shifted one character to the right."""

    def __init__(self, token_ids: list[int], seq_len: int, stride: int = 1) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if len(token_ids) < seq_len + 1:
            raise ValueError(
                f"need at least {seq_len + 1} tokens, found {len(token_ids)}. "
                "Add more text or lower --seq-len."
            )
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.starts = list(range(0, len(token_ids) - seq_len, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        window = self.ids[start : start + self.seq_len + 1]
        return window[:-1], window[1:]
