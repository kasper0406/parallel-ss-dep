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

import math
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

    def forward(self, x: torch.Tensor,
                input_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Symbol-grounded attention needs the raw token IDs to key its
        # sparse table. Other attentions ignore input_ids.
        if getattr(self.attn, "needs_input_ids", False):
            x = x + self.attn(self.attn_norm(x), input_ids=input_ids)
        else:
            x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Cross-layer top-down feedback — novel direction.
#
# Each layer's output o_L is shifted right by 1 along T to make
# `state_{L, t-1}` accessible to layer L-1 at position t. This breaks
# the cross-layer recurrence cycle without sequential per-t dispatch,
# so we keep parallel-scan friendliness.
#
# Implementation: 2-pass forward through the network.
#   Pass 1: vanilla forward — collect each layer's output as the "state"
#   Pass 2: forward with feedback. Layer L's input is augmented by
#           a TopDown projection of layer L+1's pass-1 output, shifted
#           right by 1.
#
# Cost: 2× forward compute. Comparable to multi-pass at K=2.
#
# Modes (selected via `feedback_mode`):
#   - none:      no feedback; identity (control)
#   - additive:  x_in += α · W_fb · state_above_lagged
#   - film:      x_in = x_in · (1 + α·scale) + α·shift  (FiLM modulation)
#   - predictive: TopDown predicts lower layer's pass-1 output; the
#                 ERROR (pass-1 - prediction) is what propagates to L+1.
#                 Optional surprise_loss aux: MSE on prediction error.
# ---------------------------------------------------------------------------


def _shift_right_by_1(o: torch.Tensor) -> torch.Tensor:
    """(B, T, d) -> (B, T, d), o[:, t] := original o[:, t-1]; o[:, 0] := 0."""
    pad = torch.zeros_like(o[:, :1])
    return torch.cat([pad, o[:, :-1]], dim=1)


class FeedbackProjection(nn.Module):
    """Per-layer top-down feedback module — mode-specific projection.

    Input: state_above_lagged (B, T, d) — pass-1 output of layer L+1,
           shifted right by 1.
    Output: feedback contribution to layer L's residual input.
    """

    def __init__(self, d_model: int, mode: str):
        super().__init__()
        self.mode = mode
        # IMPORTANT: keep W_fb / W_scale / W_shift at *normal* init so the
        # gradient on α is non-zero from the first step. Earlier bug: with
        # both α=0 AND W_*=0, gradient on α is zero and α never moves.
        if mode in ("additive", "predictive"):
            self.W_fb = nn.Linear(d_model, d_model, bias=False)
            # Use a smallish gain so initial feedback magnitude isn't
            # huge once α grows.
            nn.init.normal_(self.W_fb.weight, std=1.0 / math.sqrt(d_model))
        elif mode == "film":
            self.W_scale = nn.Linear(d_model, d_model, bias=False)
            self.W_shift = nn.Linear(d_model, d_model, bias=False)
            nn.init.normal_(self.W_scale.weight, std=1.0 / math.sqrt(d_model))
            nn.init.normal_(self.W_shift.weight, std=1.0 / math.sqrt(d_model))
        elif mode != "none":
            raise ValueError(f"unknown feedback mode: {mode}")
        # Per-layer learnable feedback strength α_L. Init zero so the model
        # starts as a pure feedforward stack and has to *earn* feedback use,
        # but the gradient on α is non-zero (because W_* are non-zero) so
        # the optimizer can move it.
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, state_above_lagged: torch.Tensor) -> torch.Tensor:
        if self.mode == "none" or state_above_lagged is None:
            return x
        if self.mode in ("additive", "predictive"):
            fb = self.W_fb(state_above_lagged)
            return x + self.alpha * fb
        if self.mode == "film":
            scale = self.W_scale(state_above_lagged)
            shift = self.W_shift(state_above_lagged)
            return x * (1.0 + self.alpha * scale) + self.alpha * shift
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
        max_T: int = 0,                       # 0 = no positional encoding
        attention_cls_per_layer: list[Callable[..., nn.Module]] | None = None,
        aux_dim: int = 0,                     # 0 = no aux head
        feedback_mode: str = "none",          # none / additive / film / predictive
    ):
        super().__init__()
        # Per-layer attention class list (for hybrid architectures) takes
        # precedence over the single attention_cls argument.
        if attention_cls_per_layer is not None:
            cls_list = attention_cls_per_layer
            assert len(cls_list) == n_layers, \
                f"attention_cls_per_layer has {len(cls_list)} entries but n_layers={n_layers}"
        else:
            if attention_cls is None:
                raise ValueError("attention_cls or attention_cls_per_layer is required")
            cls_list = [attention_cls] * n_layers

        if d_ff is None:
            d_ff = 4 * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        # Optional learnable absolute positional embedding (max_T > 0).
        # Off by default; needed for MQAR-style recall where the
        # architecture must distinguish lookup-phase vs query-phase
        # positions.
        self.max_T = max_T
        if max_T > 0:
            self.pos_embed = nn.Embedding(max_T, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, n_heads=n_heads, d_head=d_head,
                  d_ff=d_ff, attention_cls=cls)
            for cls in cls_list
        ])
        self.out_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Optional aux head — e.g. for bracket-depth supervision (direction E).
        self.aux_dim = aux_dim
        if aux_dim > 0:
            self.aux_head = nn.Linear(d_model, aux_dim, bias=True)

        # Cross-layer top-down feedback. One FeedbackProjection per layer;
        # layer L receives feedback from layer L+1 (so the topmost layer
        # has a no-op feedback module). For mode='none', everything is a
        # no-op and we degrade to the original 1-pass forward.
        self.feedback_mode = feedback_mode
        if feedback_mode != "none":
            self.feedback = nn.ModuleList([
                FeedbackProjection(d_model, mode=feedback_mode)
                for _ in range(n_layers)
            ])

    def forward(self, input_ids: torch.Tensor,
                return_aux: bool = False,
                return_surprise: bool = False
                ) -> torch.Tensor | tuple:
        x = self.embed(input_ids)
        if self.max_T > 0:
            T = input_ids.shape[1]
            pos = torch.arange(T, device=input_ids.device)
            x = x + self.pos_embed(pos)

        if self.feedback_mode == "none":
            for blk in self.blocks:
                x = blk(x, input_ids=input_ids)
            h = self.out_norm(x)
            lm_logits = self.lm_head(h)
            if return_aux and self.aux_dim > 0:
                return lm_logits, self.aux_head(h)
            return lm_logits

        # 2-pass forward with cross-layer feedback.
        # Pass 1: vanilla forward, collect each layer's output.
        pass1_outs: list[torch.Tensor] = []
        h = x
        for blk in self.blocks:
            h = blk(h, input_ids=input_ids)
            pass1_outs.append(h)

        # Pass 2: forward with top-down feedback from pass-1 outputs,
        # shifted right by 1 along T.
        h = x
        surprise_loss = torch.zeros((), device=x.device)
        for L, blk in enumerate(self.blocks):
            if L + 1 < len(self.blocks):
                state_above_lagged = _shift_right_by_1(pass1_outs[L + 1])
            else:
                state_above_lagged = None
            if self.feedback_mode == "predictive" and state_above_lagged is not None:
                # Predictive coding flavor: TopDown predicts the lower
                # layer's pass-1 output; layer L's input gets the prediction
                # added and we accumulate the surprise (prediction error)
                # for an optional aux loss.
                proj = self.feedback[L]
                pred = proj.W_fb(state_above_lagged)
                h_input = h + proj.alpha * pred
                err = pass1_outs[L] - pred                 # surprise
                surprise_loss = surprise_loss + (err.detach() ** 2).mean()
                h = blk(h_input, input_ids=input_ids)
            else:
                h_input = self.feedback[L](h, state_above_lagged) \
                    if self.feedback_mode != "none" else h
                h = blk(h_input, input_ids=input_ids)

        h = self.out_norm(h)
        lm_logits = self.lm_head(h)
        outs = (lm_logits,)
        if return_aux and self.aux_dim > 0:
            outs = outs + (self.aux_head(h),)
        if return_surprise:
            outs = outs + (surprise_loss,)
        return outs[0] if len(outs) == 1 else outs

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
