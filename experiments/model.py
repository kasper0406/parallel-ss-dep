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
        # Diagnostic: freeze α at 0 (ablation — film with feedback machinery
        # but no actual feedback application). Tests whether "dead weights"
        # alone hurt PPL.
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x: torch.Tensor, state_above_lagged: torch.Tensor) -> torch.Tensor:
        if self.mode == "none" or state_above_lagged is None:
            return x
        a = self.get_alpha()
        if self.mode in ("additive", "predictive"):
            fb = self.W_fb(state_above_lagged)
            return x + a * fb
        if self.mode == "film":
            scale = self.W_scale(state_above_lagged)
            shift = self.W_shift(state_above_lagged)
            return x * (1.0 + a * scale) + a * shift
        return x


class MultiSourceFiLMFeedbackMLP(nn.Module):
    """Multi-source FiLM-sum with an MLP nonlinearity in the feedback path.

    Same as MultiSourceFiLMFeedback (sum across sources, multiplicative form),
    but each source's projection is an MLP (Linear → GELU → Linear) rather
    than a single Linear, giving the feedback module nonlinear expressivity.

    Tests whether the linear feedback projections were leaving margin on
    the table.
    """

    def __init__(self, d_model: int, n_sources: int, d_hidden: int | None = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = 2 * d_model
        self.n_sources = n_sources
        self.W_scales = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden, bias=False),
                nn.GELU(),
                nn.Linear(d_hidden, d_model, bias=False),
            ) for _ in range(n_sources)
        ])
        self.W_shifts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden, bias=False),
                nn.GELU(),
                nn.Linear(d_hidden, d_model, bias=False),
            ) for _ in range(n_sources)
        ])
        # Init last linear of each MLP small so feedback magnitude is modest.
        for seq in [*self.W_scales, *self.W_shifts]:
            nn.init.normal_(seq[0].weight, std=1.0 / math.sqrt(d_model))
            nn.init.normal_(seq[2].weight, std=1.0 / math.sqrt(d_hidden))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x: torch.Tensor,
                source_states_lagged: list) -> torch.Tensor:
        a = self.get_alpha()
        scale = sum(W(s) for W, s in zip(self.W_scales, source_states_lagged))
        shift = sum(W(s) for W, s in zip(self.W_shifts, source_states_lagged))
        return x * (1.0 + a * scale) + a * shift


class MultiSourceFiLMFeedback(nn.Module):
    """Multi-source FiLM with summed contributions (no softmax routing).

    For sources s1, ..., sN with t-1 lagged hidden states, computes
        scale = sum_i W_scale[i](s_i)
        shift = sum_i W_shift[i](s_i)
    and applies multiplicative FiLM:
        x_out = x * (1 + alpha * scale) + alpha * shift

    Same multiplicative form as the proven sparse-1-pair (2, 28) finding,
    but with N source layers contributing additively. Tests whether
    multi-source structure alone (without softmax) lands in the negative-α
    predictive-coding basin. If it does, softmax was the culprit; if it
    still falls into positive-α, the basin is specifically tied to
    single-source FiLM.
    """

    def __init__(self, d_model: int, n_sources: int):
        super().__init__()
        self.n_sources = n_sources
        self.W_scales = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        self.W_shifts = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        for m in [*self.W_scales, *self.W_shifts]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d_model))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x: torch.Tensor,
                source_states_lagged: list) -> torch.Tensor:
        a = self.get_alpha()
        scale = sum(W(s) for W, s in zip(self.W_scales, source_states_lagged))
        shift = sum(W(s) for W, s in zip(self.W_shifts, source_states_lagged))
        return x * (1.0 + a * scale) + a * shift


class FiLMAttentionFeedback(nn.Module):
    """Cross-layer attention with FiLM-form multiplicative output.

    Like CrossLayerAttentionFeedback, but the value vectors carry a
    (scale, shift) pair, and the attention-weighted mixture is applied
    via FiLM modulation rather than additive residual:

        Q from x at target
        K_i, V_scale_i, V_shift_i from each source layer
        attn = softmax(QK)
        scale = out_proj_scale(attn @ V_scale)
        shift = out_proj_shift(attn @ V_shift)
        x_out = x * (1 + alpha * scale) + alpha * shift

    Combines softmax routing across sources with the multiplicative form
    that produced the −3 % win. Tests whether the negative-α basin is
    reachable when attention has the right output structure.
    """

    def __init__(self, d_model: int, n_sources: int,
                 n_heads: int = 4, d_head: int | None = None):
        super().__init__()
        if d_head is None:
            d_head = d_model // n_heads
        assert n_heads * d_head == d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_sources = n_sources
        self.scale = d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        # V emits 2*d_model per source: scale-half and shift-half.
        self.v_projs = nn.ModuleList([
            nn.Linear(d_model, 2 * d_model, bias=False) for _ in range(n_sources)
        ])
        self.out_proj_scale = nn.Linear(d_model, d_model, bias=False)
        self.out_proj_shift = nn.Linear(d_model, d_model, bias=False)
        for m in [self.q_proj, *self.k_projs, *self.v_projs,
                  self.out_proj_scale, self.out_proj_shift]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d_model))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x: torch.Tensor,
                source_states_lagged: list) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        S = self.n_sources
        q = self.q_proj(x).view(B, T, H, Dh)
        k_stack = torch.stack(
            [kp(s).view(B, T, H, Dh) for kp, s in zip(self.k_projs, source_states_lagged)],
            dim=2,
        )
        # V split into scale-half and shift-half.
        v_pairs = [vp(s) for vp, s in zip(self.v_projs, source_states_lagged)]
        v_scales = torch.stack(
            [v[..., :D].view(B, T, H, Dh) for v in v_pairs], dim=2,
        )
        v_shifts = torch.stack(
            [v[..., D:].view(B, T, H, Dh) for v in v_pairs], dim=2,
        )
        scores = torch.einsum("bthd,btshd->bths", q, k_stack) * self.scale
        attn = F.softmax(scores, dim=-1)
        scale_mix = torch.einsum("bths,btshd->bthd", attn, v_scales).reshape(B, T, D)
        shift_mix = torch.einsum("bths,btshd->bthd", attn, v_shifts).reshape(B, T, D)
        scale = self.out_proj_scale(scale_mix)
        shift = self.out_proj_shift(shift_mix)
        a = self.get_alpha()
        return x * (1.0 + a * scale) + a * shift


class CrossLayerAttentionFeedback(nn.Module):
    """Cross-layer attention top-down feedback.

    A single target layer's input attends over the (t-1 lagged) outputs of
    several "above" source layers. The attention is per-token across the
    source dimension only — no temporal mixing — so parallel-scan
    friendliness is preserved (the lag on source states preserves causality).

    Inputs:
      x:                       (B, T, d) — target layer's input (pass-2)
      source_states_lagged:    list of (B, T, d) — pass-1 outputs of each
                               source layer, already shifted right by 1.
    Output:
      x + alpha * out_proj(attn_over_sources)
    """

    def __init__(self, d_model: int, n_sources: int,
                 n_heads: int = 4, d_head: int | None = None):
        super().__init__()
        if d_head is None:
            d_head = d_model // n_heads
        assert n_heads * d_head == d_model, \
            f"n_heads*d_head must equal d_model; got {n_heads}*{d_head} != {d_model}"
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_sources = n_sources
        self.scale = d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # One K/V projection per source layer (learns source-specific channels).
        self.k_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Init at non-zero magnitude so α has gradient on step 1.
        for m in [self.q_proj, *self.k_projs, *self.v_projs, self.out_proj]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d_model))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x: torch.Tensor,
                source_states_lagged: list) -> torch.Tensor:
        # x: (B, T, d). source_states_lagged: list of (B, T, d).
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        S = self.n_sources
        # Q: (B, T, H, Dh)
        q = self.q_proj(x).view(B, T, H, Dh)
        # K, V stacked across sources: (B, T, S, H, Dh)
        k_stack = torch.stack(
            [kp(s).view(B, T, H, Dh) for kp, s in zip(self.k_projs, source_states_lagged)],
            dim=2,
        )
        v_stack = torch.stack(
            [vp(s).view(B, T, H, Dh) for vp, s in zip(self.v_projs, source_states_lagged)],
            dim=2,
        )
        # Attention scores per (b, t, h) over S sources: (B, T, H, S)
        scores = torch.einsum("bthd,btshd->bths", q, k_stack) * self.scale
        attn = F.softmax(scores, dim=-1)
        # Weighted sum of values over S: (B, T, H, Dh)
        out = torch.einsum("bths,btshd->bthd", attn, v_stack)
        out = out.reshape(B, T, D)
        return x + self.get_alpha() * self.out_proj(out)


class MultiScaleFeedbackProjection(nn.Module):
    """Multi-distance top-down feedback for one layer.

    Holds K FeedbackProjections, one per distance d in `distances`.
    In pass 2, each consumes the pass-1 output of layer L+d (shifted
    right by 1), produces a modulation of x; modulations are applied
    serially.

    For a 30-layer stack with distances=(1, 2, 4, 8, 16):
      - layer 0  sees feedback from layers {1, 2, 4, 8, 16}
      - layer 13 sees feedback from layers {14, 15, 17, 21, 29}
      - layer 29 sees nothing (no layers above)
    Distances that exceed n_layers - 1 - L are skipped at runtime
    (caller passes None for those slots).
    """

    def __init__(self, d_model: int, mode: str,
                 distances: tuple[int, ...] = (1,)):
        super().__init__()
        self.distances = distances
        self.projs = nn.ModuleList([
            FeedbackProjection(d_model, mode) for _ in distances
        ])

    def forward(self, x: torch.Tensor,
                states_above: list) -> torch.Tensor:
        """states_above: list of K tensors (or None) aligned with self.distances."""
        for proj, state in zip(self.projs, states_above):
            x = proj(x, state)
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
        feedback_distances: tuple[int, ...] = (1,),  # multi-scale top-down
        feedback_pairs: tuple = (),           # sparse (target_L, source_L) pairs;
                                              # if non-empty overrides distances
        feedback_xattn_pairs: tuple = (),     # cross-layer attention: tuple of
                                              # (target_L, (src1, src2, ...));
                                              # if non-empty, overrides BOTH the
                                              # dense and the sparse-FiLM modes
                                              # and uses CrossLayerAttentionFeedback.
        feedback_xattn_heads: int = 4,        # heads inside the cross-layer attention
        feedback_xattn_form: str = "attn",    # 'attn'      = additive residual (Q-K-V)
                                              # 'film_sum'  = multi-source FiLM, sum
                                              # 'film_attn' = softmax routing + FiLM out
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

        # Cross-layer top-down feedback. Two modes:
        # (a) Dense (default): one MultiScaleFeedbackProjection per layer
        #     with `feedback_distances` covering nearby layers above.
        # (b) Sparse: only specific (target, source) pairs get feedback.
        #     Used to test "high-level layer modulates one early layer"
        #     hypothesis — fewer connections = less compounding divergence
        #     and tiny param overhead.
        self.feedback_mode = feedback_mode
        self.feedback_distances = tuple(feedback_distances)
        self.feedback_pairs = tuple((int(t), int(s)) for t, s in feedback_pairs)
        self.feedback_xattn_pairs = tuple(
            (int(t), tuple(int(s) for s in srcs)) for t, srcs in feedback_xattn_pairs
        )
        self.feedback_xattn_form = feedback_xattn_form
        if self.feedback_xattn_pairs:
            # Cross-layer attention mode. Three forms:
            #   attn      — additive residual via Q-K-V softmax (default)
            #   film_sum  — multi-source FiLM, contributions summed (no softmax)
            #   film_attn — softmax routing + multiplicative FiLM-form output
            def _make_xattn(srcs):
                if feedback_xattn_form == "attn":
                    return CrossLayerAttentionFeedback(
                        d_model=d_model, n_sources=len(srcs),
                        n_heads=feedback_xattn_heads,
                    )
                if feedback_xattn_form == "film_sum":
                    return MultiSourceFiLMFeedback(
                        d_model=d_model, n_sources=len(srcs),
                    )
                if feedback_xattn_form == "film_sum_mlp":
                    return MultiSourceFiLMFeedbackMLP(
                        d_model=d_model, n_sources=len(srcs),
                    )
                if feedback_xattn_form == "film_attn":
                    return FiLMAttentionFeedback(
                        d_model=d_model, n_sources=len(srcs),
                        n_heads=feedback_xattn_heads,
                    )
                raise ValueError(f"unknown feedback_xattn_form: {feedback_xattn_form!r}")
            self.xattn_feedback = nn.ModuleDict({
                str(t): _make_xattn(srcs)
                for t, srcs in self.feedback_xattn_pairs
            })
            self.xattn_target_to_sources = {t: srcs for t, srcs in self.feedback_xattn_pairs}
        elif feedback_mode != "none" and self.feedback_pairs:
            # Sparse mode — one FeedbackProjection per (target, source) pair,
            # indexed by target layer string for ModuleDict.
            self.sparse_feedback = nn.ModuleDict({
                str(t): FeedbackProjection(d_model, mode=feedback_mode)
                for t, _ in self.feedback_pairs
            })
            # Inverse map: target_layer -> source_layer (for fast lookup).
            self.sparse_target_to_source = {t: s for t, s in self.feedback_pairs}
        elif feedback_mode != "none":
            self.feedback = nn.ModuleList([
                MultiScaleFeedbackProjection(
                    d_model, mode=feedback_mode,
                    distances=self.feedback_distances,
                )
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

        if self.feedback_mode == "none" and not self.feedback_xattn_pairs:
            for blk in self.blocks:
                x = blk(x, input_ids=input_ids)
            h = self.out_norm(x)
            lm_logits = self.lm_head(h)
            if return_aux and self.aux_dim > 0:
                return lm_logits, self.aux_head(h)
            return lm_logits

        # Cross-layer attention feedback (Idea 1 / Idea 2). Target layers attend
        # over the t-1 lagged pass-1 outputs of multiple source layers.
        if self.feedback_xattn_pairs:
            # Pass 1: vanilla. Collect outputs at every layer that any target
            # references as a source (union across all targets).
            needed_sources = set()
            for _, srcs in self.feedback_xattn_pairs:
                needed_sources.update(srcs)
            pass1_at_sources: dict = {}
            h1 = x
            for L, blk in enumerate(self.blocks):
                h1 = blk(h1, input_ids=input_ids)
                if L in needed_sources:
                    pass1_at_sources[L] = h1
            # Pass 2: at each target layer, gather lagged source states and
            # apply CrossLayerAttentionFeedback before the block's forward.
            h = x
            for L, blk in enumerate(self.blocks):
                if L in self.xattn_target_to_sources:
                    srcs = self.xattn_target_to_sources[L]
                    src_states = [_shift_right_by_1(pass1_at_sources[s]) for s in srcs]
                    h = self.xattn_feedback[str(L)](h, src_states)
                h = blk(h, input_ids=input_ids)
            h = self.out_norm(h)
            lm_logits = self.lm_head(h)
            if return_aux and self.aux_dim > 0:
                return lm_logits, self.aux_head(h)
            return lm_logits

        # Sparse-pair feedback: only specific target layers receive feedback
        # from specific source layers. Avoids compounding divergence.
        if self.feedback_pairs:
            # Pass 1: vanilla, collect outputs only at source layers.
            source_layers = set(s for _, s in self.feedback_pairs)
            pass1_at_sources: dict = {}
            h1 = x
            for L, blk in enumerate(self.blocks):
                h1 = blk(h1, input_ids=input_ids)
                if L in source_layers:
                    pass1_at_sources[L] = h1
            # Pass 2: forward with sparse modulation at target layers.
            h = x
            for L, blk in enumerate(self.blocks):
                if L in self.sparse_target_to_source:
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_1(pass1_at_sources[src])
                    h = self.sparse_feedback[str(L)](h, state_above_lagged)
                h = blk(h, input_ids=input_ids)
            h = self.out_norm(h)
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
        # shifted right by 1 along T. Multi-scale: each layer L sees
        # states from layers {L+d for d in feedback_distances}.
        h = x
        surprise_loss = torch.zeros((), device=x.device)
        N = len(self.blocks)
        for L, blk in enumerate(self.blocks):
            if self.feedback_mode == "none":
                h = blk(h, input_ids=input_ids)
                continue
            # Gather multi-scale lagged states for layer L.
            states_above: list = []
            for d in self.feedback_distances:
                src = L + d
                if src < N:
                    states_above.append(_shift_right_by_1(pass1_outs[src]))
                else:
                    states_above.append(None)
            if self.feedback_mode == "predictive":
                # Predictive coding (Ali/Kietzmann lineage). For multi-scale,
                # surprise loss accumulates over each distance's prediction.
                # Apply each modulation serially (same as MultiScale forward),
                # but track surprise per distance.
                multi = self.feedback[L]
                for proj, state in zip(multi.projs, states_above):
                    if state is None:
                        continue
                    pred = proj.W_fb(state)
                    h = h + proj.get_alpha() * pred
                    err = pass1_outs[L].detach() - pred
                    surprise_loss = surprise_loss + (err ** 2).mean()
                h = blk(h, input_ids=input_ids)
            else:
                h_input = self.feedback[L](h, states_above)
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

    def feedback_alphas(self) -> list:
        """Return current α values. Format depends on mode:
        - xattn:   list of (target, sources_tuple, alpha) triples.
        - sparse:  list of (target, source, alpha) triples.
        - dense single: flat list of n_layers floats.
        - dense multi:  list of n_layers lists.
        - none:    empty list.
        """
        if self.feedback_xattn_pairs:
            return [(t, srcs, float(self.xattn_feedback[str(t)].alpha.detach().item()))
                    for t, srcs in self.feedback_xattn_pairs]
        if self.feedback_mode == "none":
            return []
        if self.feedback_pairs:
            return [(t, s, float(self.sparse_feedback[str(t)].alpha.detach().item()))
                    for t, s in self.feedback_pairs]
        if len(self.feedback_distances) == 1:
            return [float(fb.projs[0].alpha.detach().item())
                    for fb in self.feedback]
        return [[float(p.alpha.detach().item()) for p in fb.projs]
                for fb in self.feedback]

    def freeze_alpha(self) -> None:
        """Diagnostic: lock α at 0 across all layers (no active feedback)."""
        if self.feedback_xattn_pairs:
            for m in self.xattn_feedback.values():
                m.freeze_alpha = True
            return
        if self.feedback_mode == "none":
            return
        if self.feedback_pairs:
            for proj in self.sparse_feedback.values():
                proj.freeze_alpha = True
        else:
            for fb in self.feedback:
                for p in fb.projs:
                    p.freeze_alpha = True
