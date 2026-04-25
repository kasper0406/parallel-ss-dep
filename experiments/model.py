"""
Minimal transformer scaffold for the kill-gate experiment.

Standard pre-norm decoder block: RMSNorm + attention + RMSNorm + MLP.
Pluggable attention via the constructor's `attention_cls` argument —
one of `LinearAttention` or `HeisenbergAttention` from `experiments.layers`.

We keep the scaffold intentionally tiny — no positional encoding (the
attention itself is causal and order-aware via the cumsum), no dropout,
no tied embeddings. The point is to compare two attentions, not to
build a competitive small LM.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class GLU(nn.Module):
    """SwiGLU MLP — d_model -> d_ff -> d_model."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W_g = nn.Linear(d_model, d_ff, bias=False)
        self.W_u = nn.Linear(d_model, d_ff, bias=False)
        self.W_d = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_d(F.silu(self.W_g(x)) * self.W_u(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int,
                 attention_cls: Callable[..., nn.Module]):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = attention_cls(d_model=d_model, n_heads=n_heads, d_head=d_head)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = GLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_head: int = 32,
        d_ff: int | None = None,
        attention_cls: Callable[..., nn.Module] | None = None,
    ):
        super().__init__()
        if attention_cls is None:
            raise ValueError("attention_cls is required")
        if d_ff is None:
            d_ff = 4 * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, n_heads=n_heads, d_head=d_head,
                  d_ff=d_ff, attention_cls=attention_cls)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.out_norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
