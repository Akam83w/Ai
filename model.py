# SPDX-License-Identifier: MIT
"""Decoder-only Transformer language model.

Weights are created with a random initialization. This module does not
download pretrained weights and does not call any external model API.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


PRESETS: dict[str, dict[str, float | int]] = {
    # Laptop CPU. A few hundred thousand parameters at a typical char vocab.
    "tiny": {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 256,
        "max_seq_len": 128,
        "dropout": 0.1,
    },
    # Still small (single-digit millions of parameters). Prefer a GPU.
    "small": {
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 1024,
        "max_seq_len": 256,
        "dropout": 0.1,
    },
}


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.1
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1 or self.max_seq_len < 2:
            raise ValueError("n_layers >= 1 and max_seq_len >= 2 are required")
        if self.d_ff < self.d_model:
            raise ValueError("d_ff must be >= d_model")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        fields = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**fields)


def config_from_preset(name: str, vocab_size: int, pad_id: int = 0) -> ModelConfig:
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose one of: {known}")
    params = dict(PRESETS[name])
    return ModelConfig(vocab_size=vocab_size, pad_id=pad_id, **params)  # type: ignore[arg-type]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.max_seq_len, cfg.max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        mixed = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_drop(self.proj(mixed))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """Small causal language model. Token embeddings and the output head are tied."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _batch, seq_len = idx.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. temperature <= 0 selects the greedy token."""
        self.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(cond)
            next_logits = logits[:, -1, :]
            if temperature <= 1e-4:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None and top_k > 0:
                    k = min(top_k, next_logits.size(-1))
                    cutoff = torch.topk(next_logits, k).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
