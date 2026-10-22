# SPDX-License-Identifier: MIT
"""Character tokenizer built only from text you provide.

There is no downloaded vocabulary. Characters absent from the saved vocab
become the unknown token at encode time.
"""

from __future__ import annotations

import json
from pathlib import Path

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
