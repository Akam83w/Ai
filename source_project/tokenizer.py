# SPDX-License-Identifier: MIT
"""Local character and byte-level BPE tokenizers built only from text you provide.

There is no downloaded vocabulary. Characters absent from the saved vocab
become the unknown token at encode time.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)


class CharTokenizer:
    def __init__(self, itos: list[str]) -> None:
        if len(itos) < len(SPECIALS) or tuple(itos[: len(SPECIALS)]) != SPECIALS:
            raise ValueError("vocabulary must start with the four special tokens")
        self.itos = list(itos)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({ch for text in texts for ch in text})
        return cls(list(SPECIALS) + chars)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        skipped = {self.pad_id, self.bos_id, self.eos_id}
        pieces: list[str] = []
        for idx in ids:
            if skip_special and idx in skipped:
                continue
            if idx == self.unk_id:
                pieces.append("<?>")
                continue
            if 0 <= idx < len(self.itos):
                pieces.append(self.itos[idx])
        return "".join(pieces)

    def to_dict(self) -> dict:
        return {"version": 1, "type": "char", "itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        if data.get("type") not in (None, "char"):
            raise ValueError("unsupported tokenizer type")
        return cls(list(data["itos"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))



class BpeTokenizer:
    """Byte-level BPE tokenizer trained only on the supplied local corpus."""

    def __init__(self, backend: Tokenizer) -> None:
        self.backend = backend
        self.pad_id = self._required_id(PAD)
        self.bos_id = self._required_id(BOS)
        self.eos_id = self._required_id(EOS)
        self.unk_id = self._required_id(UNK)

    def _required_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required token {token}")
        return token_id

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    @classmethod
    def from_texts(cls, texts: list[str], vocab_size: int = 2048) -> "BpeTokenizer":
        if vocab_size < 260:
            raise ValueError("BPE vocab_size must be at least 260 for byte-level coverage")
        backend = Tokenizer(models.BPE(unk_token=UNK))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=[PAD, BOS, EOS, UNK],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.backend.decode([int(idx) for idx in ids], skip_special_tokens=skip_special)

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "type": "bpe",
            "tokenizer_json": self.backend.to_str(),
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BpeTokenizer":
        return cls(Tokenizer.from_str(data["tokenizer_json"]))


def tokenizer_from_dict(data: dict):
    """Load a versioned tokenizer while retaining support for old char checkpoints."""
    if data.get("type") == "bpe":
        return BpeTokenizer.from_dict(data)
    return CharTokenizer.from_dict(data)


def tokenizer_from_texts(texts: list[str], tokenizer_type: str = "bpe", vocab_size: int = 2048):
    if tokenizer_type == "bpe":
        return BpeTokenizer.from_texts(texts, vocab_size=vocab_size)
    if tokenizer_type == "char":
        return CharTokenizer.from_texts(texts)
    raise ValueError("tokenizer_type must be 'bpe' or 'char'")
