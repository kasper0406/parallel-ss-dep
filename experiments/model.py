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


def _shift_right_by_k(o: torch.Tensor, k: int) -> torch.Tensor:
    """(B, T, d) -> (B, T, d), o[:, t] := original o[:, t-k]; first k zeroed."""
    if k <= 0:
        return o
    pad = torch.zeros_like(o[:, :k])
    return torch.cat([pad, o[:, :-k]], dim=1)


class FeedbackProjection(nn.Module):
    """Per-layer top-down feedback module — mode-specific projection.

    Input: state_above_lagged (B, T, d) — pass-1 output of layer L+1,
           shifted right by 1.
    Output: feedback contribution to layer L's residual input.

    Alpha modes (`alpha_mode`):
      - 'scalar' (default):  one learnable α (or per-channel d_model
        learnable αs when `per_channel_alpha` is True). FiLM:
            x * (1 + α · scale) + α · shift
      - 'surprise_modulated' (Phase 22 / structural-surprise PoC):
        per-token α(t) = α₀ · σ(scale · surprise_z(t) + bias) where
        α₀, scale, bias are three learnable scalars. The surprise tensor
        is supplied at forward time (B, T) — the caller computes it from
        inter-iter source-state deltas in K=3 self-feeding. Internal
        normalization: per-batch z-score over (B, T) before the sigmoid,
        so the inputs to σ are roughly unit-variance regardless of the
        absolute scale of the surprise signal.
    """

    def __init__(self, d_model: int, mode: str,
                 per_channel_alpha: bool = False,
                 alpha_mode: str = "scalar"):
        super().__init__()
        self.mode = mode
        self.per_channel_alpha = per_channel_alpha
        if alpha_mode not in ("scalar", "surprise_modulated"):
            raise ValueError(
                f"alpha_mode must be 'scalar' or 'surprise_modulated' "
                f"(got {alpha_mode!r})"
            )
        self.alpha_mode = alpha_mode
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
        if alpha_mode == "scalar":
            # Per-layer learnable feedback strength α_L. Init zero so the
            # model starts as a pure feedforward stack and has to *earn*
            # feedback use, but the gradient on α is non-zero (because
            # W_* are non-zero) so the optimizer can move it.
            # Scalar (1,) by default; per-channel (d_model,) when requested.
            alpha_shape = (d_model,) if per_channel_alpha else (1,)
            self.alpha = nn.Parameter(torch.zeros(alpha_shape))
        else:
            # alpha_mode == 'surprise_modulated'.
            # α(t) = alpha_zero · σ(surprise_scale · surprise_z(t) + surprise_bias).
            # Init: α_zero = 0 so model starts as a pure feedforward stack
            # (matches scalar-α init), surprise_scale = 1 (unit slope),
            # surprise_bias = 0 (σ(0)=0.5 baseline, symmetric around mean
            # surprise).  Three scalars total (extra params are negligible).
            # NOTE: per_channel_alpha is ignored in this mode — α(t) is
            # always per-token-scalar (broadcast across channels). A future
            # extension could add a per-channel α₀ tensor if needed.
            self.alpha_zero = nn.Parameter(torch.zeros(1))
            self.surprise_scale = nn.Parameter(torch.ones(1))
            self.surprise_bias = nn.Parameter(torch.zeros(1))
        # Diagnostic: freeze α at 0 (ablation — film with feedback machinery
        # but no actual feedback application). Tests whether "dead weights"
        # alone hurt PPL.
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        # For backwards-compat with external callers that summarise the
        # learned strength. In scalar mode this is the per-layer (or
        # per-channel) α; in surprise_modulated mode this is α_zero
        # (the unmodulated maximum strength).
        if self.alpha_mode == "scalar":
            return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha
        return torch.zeros_like(self.alpha_zero) if self.freeze_alpha else self.alpha_zero

    def get_per_token_alpha(self, surprise: torch.Tensor) -> torch.Tensor:
        """Compute α(t) for surprise-modulated mode.

        Args:
          surprise: (B, T) non-negative tensor of per-token L2 deltas
                    between consecutive iterations' source states.
        Returns:
          α(t): (B, T, 1) tensor — broadcasts across channels in the
                FiLM modulation.
        """
        assert self.alpha_mode == "surprise_modulated"
        if self.freeze_alpha:
            return torch.zeros(*surprise.shape, 1,
                                device=surprise.device, dtype=surprise.dtype)
        # Per-batch z-score: (s - mean) / (std + eps). Computed over the
        # whole (B, T) flat for stability — per-batch normalisation, not
        # per-position. This makes the surprise scale invariant to the
        # absolute layer-output magnitude.
        s = surprise.detach()                  # surprise is a measurement,
                                                # not a learnable signal — block
                                                # gradient flow through it.
        mu = s.mean()
        sd = s.std().clamp(min=1e-6)
        s_z = (s - mu) / sd
        # σ(scale · z + bias) ∈ (0, 1). Multiply by α_zero.
        gate = torch.sigmoid(self.surprise_scale * s_z + self.surprise_bias)
        a_t = self.alpha_zero * gate           # (B, T)
        return a_t.unsqueeze(-1)               # (B, T, 1)

    def forward(self, x: torch.Tensor, state_above_lagged: torch.Tensor,
                surprise: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "none" or state_above_lagged is None:
            return x
        if self.alpha_mode == "scalar":
            a = self.get_alpha()
        else:
            # surprise_modulated. surprise must be provided by caller.
            if surprise is None:
                # Fallback for callers (e.g. cold-start lagged-cached eval)
                # that lack a surprise signal: use σ(bias) as the gate so
                # the model degrades to a fixed-α form rather than
                # crashing. This is the deployment-without-surprise mode.
                if self.freeze_alpha:
                    a = torch.zeros_like(self.alpha_zero)
                else:
                    a = self.alpha_zero * torch.sigmoid(self.surprise_bias)
            else:
                a = self.get_per_token_alpha(surprise)   # (B, T, 1)
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


class MultiSourceFiLMTargetGated(nn.Module):
    """Target-gated cross-layer feedback (no Q-K-V attention).

    For each source layer i, the TARGET's hidden state x produces a
    per-channel gate (via SwiGLU on a wide projection), and the SOURCE's
    hidden state provides values:

        gate_i  = silu(W_gate_i(x))            # from target, what to keep
        value_i = W_value_i(source_i_lagged)   # from source, what's there
        contribution_i_scale = gate_i_scale ⊙ value_i_scale
        contribution_i_shift = gate_i_shift ⊙ value_i_shift

    Sum across sources, apply multiplicatively as FiLM.

    Differences from previously tested forms:
      - vs film_sum:      gating depends on target (per-token routing)
      - vs film_sum_glu:  gate is from TARGET, value is from SOURCE
                          (vs both from source, which failed)
      - vs sigmoid attn:  no Q-K dot product; per-channel gate directly
                          from target's projection (more expressive than
                          one scalar gate per source).
    """

    def __init__(self, d_model: int, n_sources: int):
        super().__init__()
        self.n_sources = n_sources
        # Per source: target → 2d (gate_scale | gate_shift)
        #             source → 2d (value_scale | value_shift)
        self.W_target_gates = nn.ModuleList([
            nn.Linear(d_model, 2 * d_model, bias=False) for _ in range(n_sources)
        ])
        self.W_source_values = nn.ModuleList([
            nn.Linear(d_model, 2 * d_model, bias=False) for _ in range(n_sources)
        ])
        for m in [*self.W_target_gates, *self.W_source_values]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d_model))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x: torch.Tensor,
                source_states_lagged: list) -> torch.Tensor:
        scale_acc = x.new_zeros(x.shape)
        shift_acc = x.new_zeros(x.shape)
        for Wg, Wv, src in zip(self.W_target_gates, self.W_source_values,
                                source_states_lagged):
            gate = Wg(x)             # (B, T, 2d) per-channel target-side gate
            value = Wv(src)          # (B, T, 2d) per-channel source value
            g_scale, g_shift = gate.chunk(2, dim=-1)
            v_scale, v_shift = value.chunk(2, dim=-1)
            scale_acc = scale_acc + F.silu(g_scale) * v_scale
            shift_acc = shift_acc + F.silu(g_shift) * v_shift
        a = self.get_alpha()
        return x * (1.0 + a * scale_acc) + a * shift_acc


class MultiSourceFiLMFeedbackGLU(nn.Module):
    """Multi-source FiLM-sum with SwiGLU-style per-channel gating.

    Per source, project to 4·d_model and split into
        [scale_value | scale_gate | shift_value | shift_gate]
    then per-channel gate: v ⊙ silu(g). Sum across sources, then
    apply multiplicatively as in FiLM:
        x_out = x · (1 + α · scale) + α · shift

    Differs from `film_sum_mlp` in *what* the nonlinearity is —
    multiplicative gating (GLU) instead of additive nonlinearity
    (Linear-GELU-Linear). Tests the hypothesis that the source
    state has channel-dependent reliability that linear projections
    can't express but per-channel gates can.
    """

    def __init__(self, d_model: int, n_sources: int):
        super().__init__()
        self.n_sources = n_sources
        # One 4·d output projection per source: [s_v | s_g | h_v | h_g].
        self.projs = nn.ModuleList([
            nn.Linear(d_model, 4 * d_model, bias=False) for _ in range(n_sources)
        ])
        for m in self.projs:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d_model))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x: torch.Tensor,
                source_states_lagged: list) -> torch.Tensor:
        scale_acc = x.new_zeros(x.shape)
        shift_acc = x.new_zeros(x.shape)
        for proj, src in zip(self.projs, source_states_lagged):
            mix = proj(src)
            sv, sg, hv, hg = mix.chunk(4, dim=-1)
            scale_acc = scale_acc + sv * F.silu(sg)
            shift_acc = shift_acc + hv * F.silu(hg)
        a = self.get_alpha()
        return x * (1.0 + a * scale_acc) + a * shift_acc


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


class SigmoidGatedFiLMAttentionFeedback(nn.Module):
    """Cross-layer routing with sigmoid (not softmax) gates + FiLM output.

    Each source layer has an independent per-token gate in [0, 1] computed
    as sigmoid(Q · K_i / sqrt(d_head)). Avoids the softmax 1/K-dilution
    failure mode of FiLMAttentionFeedback while preserving per-token
    routing flexibility. Output is multiplicative FiLM (which we know
    finds the negative-α basin from film_sum).

        Q from x[t]
        for each source i:
            g_i = sigmoid(Q · K_i)   ∈ [0, 1] independently per source
            scale_i = W_scale[i](source_i)
            shift_i = W_shift[i](source_i)
        scale = sum_i g_i · scale_i
        shift = sum_i g_i · shift_i
        x_out = x * (1 + alpha · out_proj_scale(scale))
              +     alpha · out_proj_shift(shift)

    At init g ≈ 0.5 → magnitude is 0.5·K per source vs softmax's 1/K
    per source. The gradient on α is therefore ≥ K² stronger than
    film_attn at init.
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
        self.scale_qk = d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        # FiLM-form values: each source emits a scale and shift (d_model each).
        self.W_scales = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        self.W_shifts = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_sources)
        ])
        self.out_proj_scale = nn.Linear(d_model, d_model, bias=False)
        self.out_proj_shift = nn.Linear(d_model, d_model, bias=False)
        for m in [self.q_proj, *self.k_projs, *self.W_scales, *self.W_shifts,
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
        # Q at target. (B, T, H, Dh)
        q = self.q_proj(x).view(B, T, H, Dh)
        scale_acc = x.new_zeros((B, T, D))
        shift_acc = x.new_zeros((B, T, D))
        for i, src in enumerate(source_states_lagged):
            # K_i. (B, T, H, Dh)
            k = self.k_projs[i](src).view(B, T, H, Dh)
            # Per-source per-token per-head sigmoid gate.
            score = (q * k).sum(dim=-1) * self.scale_qk     # (B, T, H)
            g = torch.sigmoid(score)                        # (B, T, H)
            # Repeat each head's gate across its head_dim slice. (B, T, D)
            g_exp = g.unsqueeze(-1).expand(-1, -1, -1, Dh).reshape(B, T, D)
            scale_acc = scale_acc + g_exp * self.W_scales[i](src)
            shift_acc = shift_acc + g_exp * self.W_shifts[i](src)
        scale = self.out_proj_scale(scale_acc)
        shift = self.out_proj_shift(shift_acc)
        a = self.get_alpha()
        return x * (1.0 + a * scale) + a * shift


class AllToAllSigmoidFeedback(nn.Module):
    """All-to-all sigmoid-gated FiLM cross-layer attention with parameter
    sharing across targets and sources.

    Every target layer can attend over every later (or earlier — caller
    decides) source layer with an independent sigmoid gate. To keep param
    count tractable, projections are shared by role:
      - per-target:   q_proj, alpha, out_proj_scale, out_proj_shift
      - per-source:   k_proj, W_scale, W_shift   (shared across all targets
                                                  that read this source)

    Total params ≈ n_layers · 4·d² + n_layers · 3·d²  =  7·n_layers·d²
    (vs naive (target,source) — quadratic in n_layers and prohibitive).

    The user passes a target_to_sources mapping. For Idea 2 ("every layer
    attends over every later layer") that mapping is
        {L: [L+1, L+2, …, n_layers-1] for L in 0..n_layers-2}
    """

    def __init__(self, d_model: int, target_to_sources: dict,
                 n_heads: int = 4, d_head: int | None = None):
        super().__init__()
        if d_head is None:
            d_head = d_model // n_heads
        assert n_heads * d_head == d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale_qk = d_head ** -0.5
        self.target_to_sources = {int(t): [int(s) for s in srcs]
                                  for t, srcs in target_to_sources.items()}
        # Union of source layers (anyone needs them).
        all_sources = sorted({s for srcs in self.target_to_sources.values() for s in srcs})
        all_targets = sorted(self.target_to_sources.keys())
        self.all_sources = all_sources
        self.all_targets = all_targets
        # Per-target projections (Q, out_scale, out_shift, alpha).
        self.q_projs = nn.ModuleDict({
            str(t): nn.Linear(d_model, d_model, bias=False) for t in all_targets
        })
        self.out_proj_scales = nn.ModuleDict({
            str(t): nn.Linear(d_model, d_model, bias=False) for t in all_targets
        })
        self.out_proj_shifts = nn.ModuleDict({
            str(t): nn.Linear(d_model, d_model, bias=False) for t in all_targets
        })
        self.alphas = nn.ParameterDict({
            str(t): nn.Parameter(torch.zeros(1)) for t in all_targets
        })
        # Per-source projections (K, W_scale, W_shift) — shared across targets.
        self.k_projs_src = nn.ModuleDict({
            str(s): nn.Linear(d_model, d_model, bias=False) for s in all_sources
        })
        self.W_scales_src = nn.ModuleDict({
            str(s): nn.Linear(d_model, d_model, bias=False) for s in all_sources
        })
        self.W_shifts_src = nn.ModuleDict({
            str(s): nn.Linear(d_model, d_model, bias=False) for s in all_sources
        })
        for m in [*self.q_projs.values(), *self.out_proj_scales.values(),
                  *self.out_proj_shifts.values(), *self.k_projs_src.values(),
                  *self.W_scales_src.values(), *self.W_shifts_src.values()]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d_model))
        self.freeze_alpha = False

    def get_alpha(self, target: int) -> torch.Tensor:
        a = self.alphas[str(target)]
        return torch.zeros_like(a) if self.freeze_alpha else a

    def forward_at_target(self, x: torch.Tensor, target: int,
                          pass1_at_sources: dict) -> torch.Tensor:
        """Apply the all-to-all sigmoid feedback at one target layer."""
        srcs = self.target_to_sources[target]
        if not srcs:
            return x
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.q_projs[str(target)](x).view(B, T, H, Dh)
        scale_acc = x.new_zeros((B, T, D))
        shift_acc = x.new_zeros((B, T, D))
        for s in srcs:
            src = pass1_at_sources[s]   # already lagged by caller
            k = self.k_projs_src[str(s)](src).view(B, T, H, Dh)
            score = (q * k).sum(dim=-1) * self.scale_qk        # (B, T, H)
            g = torch.sigmoid(score).unsqueeze(-1).expand(-1, -1, -1, Dh).reshape(B, T, D)
            scale_acc = scale_acc + g * self.W_scales_src[str(s)](src)
            shift_acc = shift_acc + g * self.W_shifts_src[str(s)](src)
        scale = self.out_proj_scales[str(target)](scale_acc)
        shift = self.out_proj_shifts[str(target)](shift_acc)
        a = self.get_alpha(target)
        return x * (1.0 + a * scale) + a * shift


class SurpriseScratchpadFeedback(nn.Module):
    """Surprise-gated cross-layer attention scratchpad with FiLM output.

    For target layer t with source layer s:
      - At each query position p, the scratchpad is the *causal* sequence
        of source-layer pass-1 outputs at positions ≤ p (with the
        standard t−1 lag for parallel-scan friendliness).
      - Attention scores are *biased* by per-position surprise scores —
        high-surprise positions in the past get larger weight.
      - Output is FiLM-form (multiplicative scale + additive shift) since
        the mechanism analysis (Phase 14b–g) showed multiplicative form
        is required for the negative-α basin.

    Surprise is computed externally from the pass-1 logits and passed in
    as a (B, T) tensor with non-negative values (clamped at 0).

    Memory at training time: O(T) (full causal attention over T keys).
    At inference time: only the *latest* source state is needed per token,
    so memory is O(K) for a fixed-budget K-slot cache; we use full T
    here for simplicity in this prototype.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 d_head: int | None = None,
                 routing: str = "softmax"):
        super().__init__()
        if d_head is None:
            d_head = d_model // n_heads
        assert n_heads * d_head == d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale_qk = d_head ** -0.5
        # Routing modes — disentangling write vs read contributions:
        #   'softmax'           — content (Q·K) softmax + surprise log-bias
        #   'sigmoid'           — content (Q·K) sigmoid + surprise multiplicative
        #   'sigmoid_uniform'   — content (Q·K) sigmoid, surprise IGNORED
        #                         (tests pure content addressing on the write set)
        #   'uniform_surprise'  — Q·K IGNORED, just surprise-weighted average
        #                         (tests pure saliency-based retrieval)
        assert routing in ("softmax", "sigmoid", "sigmoid_uniform",
                            "uniform_surprise"), f"unknown routing: {routing!r}"
        self.routing = routing
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # V emits 2·d_model per source — split into scale-half and shift-half.
        self.v_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj_scale = nn.Linear(d_model, d_model, bias=False)
        self.out_proj_shift = nn.Linear(d_model, d_model, bias=False)
        for m in [self.q_proj, self.k_proj, self.v_proj,
                  self.out_proj_scale, self.out_proj_shift]:
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d_model))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.freeze_alpha = False

    def get_alpha(self) -> torch.Tensor:
        return torch.zeros_like(self.alpha) if self.freeze_alpha else self.alpha

    def forward(self, x_target: torch.Tensor,
                source_states_lagged: torch.Tensor,
                surprise: torch.Tensor) -> torch.Tensor:
        # x_target:              (B, T, d) — pass-2 hidden state at target layer
        # source_states_lagged:  (B, T, d) — pass-1 output of source layer, t−1 lagged
        # surprise:              (B, T)    — per-position surprise score, ≥ 0
        B, T, D = x_target.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x_target).view(B, T, H, Dh)            # (B, T_q, H, Dh)
        k = self.k_proj(source_states_lagged).view(B, T, H, Dh)  # (B, T_k, H, Dh)
        v_pair = self.v_proj(source_states_lagged)              # (B, T_k, 2D)
        v_scale, v_shift = v_pair.chunk(2, dim=-1)              # (B, T_k, D) each
        v_scale = v_scale.view(B, T, H, Dh)
        v_shift = v_shift.view(B, T, H, Dh)

        # Attention scores: (B, T_q, T_k, H)
        scores = torch.einsum("bthd,bshd->btsh", q, k) * self.scale_qk

        # Causal mask: query position t can attend to key positions ≤ t.
        mask = torch.ones(T, T, dtype=torch.bool, device=x_target.device).triu(diagonal=1)

        if self.routing == "softmax":
            # Content (Q·K) softmax + surprise log-bias.
            surprise_bias = torch.log(surprise.clamp(min=0) + 1e-4)  # (B, T_k)
            scores = scores + surprise_bias.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(-1), float("-inf"))
            attn = F.softmax(scores, dim=2)
        elif self.routing == "sigmoid":
            # Content (Q·K) sigmoid + surprise multiplicative.
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(-1), -50.0)
            gates = torch.sigmoid(scores)
            attn = gates * surprise.unsqueeze(1).unsqueeze(-1)
            attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-6)
        elif self.routing == "sigmoid_uniform":
            # Content-only — Q·K sigmoid, surprise IGNORED (uniform writes).
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(-1), -50.0)
            gates = torch.sigmoid(scores)
            # Causal mask in float (1 inside causal region, 0 outside).
            causal_float = (~mask).float().unsqueeze(0).unsqueeze(-1)
            attn = gates * causal_float
            attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-6)
        else:  # 'uniform_surprise'
            # Saliency-only — Q·K IGNORED, surprise-weighted average.
            # Per query position t, just weighted-avg by surprise of keys 0..t.
            causal_float = (~mask).float()  # (T_q, T_k)
            attn_flat = surprise.unsqueeze(1) * causal_float.unsqueeze(0)  # (B, T_q, T_k)
            attn_flat = attn_flat / (attn_flat.sum(dim=-1, keepdim=True) + 1e-6)
            # Broadcast same weights over heads.
            attn = attn_flat.unsqueeze(-1).expand(-1, -1, -1, self.n_heads)

        # Weighted sums of v_scale and v_shift.
        scale_mix = torch.einsum("btsh,bshd->bthd", attn, v_scale).reshape(B, T, D)
        shift_mix = torch.einsum("btsh,bshd->bthd", attn, v_shift).reshape(B, T, D)
        scale = self.out_proj_scale(scale_mix)
        shift = self.out_proj_shift(shift_mix)
        a = self.get_alpha()
        return x_target * (1.0 + a * scale) + a * shift


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


class WorkingMemory(nn.Module):
    """Bounded, write-gated working memory read at thinking positions.

    Pipeline (single forward, fully differentiable):
        Per position t (input is `h ∈ ℝ^{B×T×d}`, already out-normed):
          g_t = σ(W_write(h_t)) ∈ [0,1]              # write gate
          v_t = W_v(h_t)            ∈ ℝ^{d_mem}      # value to write
          q_t = W_q(h_t)            ∈ ℝ^{d_mem}      # query
        Per row, select the top-K positions by g_t (K = min(T, mem_size)):
          buf_k = v[top_idx[k]],  g_buf_k = g[top_idx[k]]
        For each position p:
          score_{p,k} = (q_p · buf_k)/√d_mem + log(g_buf_k + ε)
          mask: -∞ if top_idx[k] ≥ p (strict-causal) or src was a pad token
          α = softmax_k(score)
          read_p = Σ_k α_{p,k} buf_k
        At think positions only:
          h_p ← h_p + W_proj(read_p)

    Cost is O(B·T·K·d_mem) — linear in T with constant K, matching the
    SSM ethos. No O(T²) attention is introduced.

    Gradient flows through values, queries, gate (via log-bias and value
    scaling), and W_proj — none of which start at zero, so the path can
    bootstrap.
    """

    def __init__(
        self,
        d_model: int,
        d_mem: int,
        mem_size: int,
        thinking_token_id: int,
        pad_token_id: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.mem_size = int(mem_size)
        self.thinking_token_id = int(thinking_token_id)
        self.pad_token_id = pad_token_id

        # Write side
        self.W_write = nn.Linear(d_model, 1, bias=True)
        self.W_v = nn.Linear(d_model, d_mem, bias=False)
        # Read side
        self.W_q = nn.Linear(d_model, d_mem, bias=False)
        self.W_proj = nn.Linear(d_mem, d_model, bias=False)

        # Init — small-random everywhere, *zero bias on W_write* so the gate
        # starts at σ(0)=0.5 (model can write or not at start). W_proj is
        # explicitly small-random (NOT zero), so the read path has a
        # non-vanishing gradient signal from step 0.
        for lin in (self.W_v, self.W_q, self.W_proj):
            nn.init.normal_(lin.weight, std=0.02)
        nn.init.normal_(self.W_write.weight, std=0.02)
        nn.init.zeros_(self.W_write.bias)

    def forward(self, h: torch.Tensor, input_ids: torch.Tensor,
                read_mask: torch.Tensor | None = None) -> torch.Tensor:
        """`read_mask` (B, T) bool/float: 1 where memory should be injected.
        If None, derived from `input_ids == thinking_token_id`."""
        B, T, _ = h.shape
        device = h.device

        # ---- Write side: compute gate + value at every position --------------
        write_logits = self.W_write(h).squeeze(-1)              # (B, T)
        g = torch.sigmoid(write_logits)                          # (B, T)
        v = self.W_v(h)                                          # (B, T, d_mem)
        # Stash for diagnostics (mirrors TinyLM._last_gate_logits pattern).
        self._last_write_gate = g.detach()

        # Mask pad-position contributions out before top-K so the buffer never
        # contains padding rows.
        if self.pad_token_id is not None:
            is_pad = input_ids == int(self.pad_token_id)          # (B, T)
            g = g.masked_fill(is_pad, 0.0)

        K_eff = min(T, self.mem_size)

        # ---- Top-K positions per row by write-gate ---------------------------
        # top_idx: (B, K_eff). Gradient flows through `v` & `g` at the
        # *selected* positions exactly as torch.topk semantics dictate.
        _, top_idx = torch.topk(g, k=K_eff, dim=-1)              # (B, K_eff)
        gather_idx_v = top_idx.unsqueeze(-1).expand(-1, -1, self.d_mem)  # (B, K, d_mem)
        buf_v = torch.gather(v, dim=1, index=gather_idx_v)        # (B, K, d_mem)
        buf_g = torch.gather(g, dim=1, index=top_idx)             # (B, K)

        # ---- Read side: query for every position -----------------------------
        # We only USE the result at think positions (mask later); cheaper than
        # gather-only-think indices but keeps the implementation uniform.
        q = self.W_q(h)                                          # (B, T, d_mem)
        scale = 1.0 / math.sqrt(self.d_mem)
        # scores: (B, T, K)
        scores = torch.einsum("btd,bkd->btk", q, buf_v) * scale
        # Log-gate bias: tiny ε to keep log finite when a row's K-th slot has g=0.
        scores = scores + torch.log(buf_g.clamp_min(1e-6)).unsqueeze(1)

        # Causal mask: position t can attend to buffer-slot k iff top_idx[k] < t.
        # top_idx: (B, K) → (B, 1, K). pos: (1, T, 1).
        pos = torch.arange(T, device=device).view(1, T, 1)
        src_pos = top_idx.unsqueeze(1)                           # (B, 1, K)
        causal_mask = src_pos >= pos                              # (B, T, K)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Some rows (t < min source position) get all -inf. Softmax of all
        # -inf is NaN; replace those rows with zero attention so the read is
        # zero (and the injection is zero).
        all_masked = causal_mask.all(dim=-1, keepdim=True)        # (B, T, 1)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        read = torch.einsum("btk,bkd->btd", attn, buf_v)          # (B, T, d_mem)
        injection = self.W_proj(read)                             # (B, T, d_model)

        # Inject only at "read positions" — either an explicit mask the
        # caller provided (e.g. MQAR query positions) or the default
        # thinking-token-based mask.
        if read_mask is None:
            inj = (input_ids == self.thinking_token_id).unsqueeze(-1).to(h.dtype)
        else:
            inj = read_mask.to(h.dtype).unsqueeze(-1)
        return h + injection * inj


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
        feedback_lag: int = 1,                # Lag (in tokens) for source state
                                              # before feeding to target. 1 = t-1
                                              # (default, parallel-scan friendly).
        feedback_position: str = "pre",       # 'pre' = modulate target's input
                                              # 'post' = modulate target's output
        feedback_per_channel_alpha: bool = False,  # per-channel α for sparse FiLM
        tie_embeddings: bool = False,         # share weights between embedding
                                              # and lm_head (~halves params for
                                              # large-vocab Qwen tokenizer).
        feedback_scratchpad_pairs: tuple = (),  # surprise-gated scratchpad
                                              # tuple of (target_L, source_L);
                                              # at each target the cross-layer
                                              # attention is biased by per-pos
                                              # surprise (computed from pass-1
                                              # logits, stop-grad).
        feedback_scratchpad_heads: int = 4,
        feedback_scratchpad_routing: str = "softmax",  # 'softmax' or 'sigmoid'
        feedback_self_k: int = 0,             # K-iteration self-feeding training.
                                              # 0 = off, 2 = K=2 (cold start +
                                              # one self-feed), 3 = K=3
                                              # (cold start + two self-feeds).
                                              # Only meaningful with sparse
                                              # feedback_pairs + feedback_mode
                                              # 'film' / 'additive'. Trains
                                              # pass K to be self-consistent —
                                              # closes the train/inference gap
                                              # for the lagged-cached deploy
                                              # protocol.
        feedback_alpha_mode: str = "scalar",  # 'scalar' = single learnable α
                                              # per (target,source) pair (the
                                              # default, matches Phase 21c).
                                              # 'surprise_modulated' = per-token
                                              # α(t) = α₀·σ(scale·s_z(t)+bias)
                                              # where s_z(t) is the per-batch
                                              # z-scored inter-iter delta of
                                              # source-state norms (free
                                              # signal from K=3 self-feeding).
                                              # Adds 3 learnable scalars per
                                              # FiLM target. Requires
                                              # feedback_self_k >= 2.
        semantic_loss_dim: int = 0,           # Phase 22 / structural-surprise
                                              # full PoC: dimensionality of the
                                              # oracle's embedding space (0 =
                                              # off; non-zero = construct a
                                              # learnable W: d_model -> dim
                                              # projection used by the trainer
                                              # to align pooled hidden states
                                              # to the frozen oracle's E(s_t).
                                              # Saved with the checkpoint so
                                              # eval can reuse the W.
        output_gate: bool = False,            # Learned per-position output gate.
                                              # When True, adds a gate_head
                                              # (d_model → 1) whose sigmoid
                                              # output g_t ∈ [0,1] controls
                                              # the emit/think tradeoff.
                                              # The caller uses g to compute:
                                              # L = mean(g*CE + (1-g)*λ).
                                              # Init: zero weight, bias=+2.0
                                              # (g ≈ 0.88 at start so training
                                              # starts near standard LM).
        use_memory: bool = False,             # Enable bounded working memory:
                                              # write-gated, top-K-bounded
                                              # store of past hidden states,
                                              # soft-attention-read at think
                                              # positions only.
        mem_size: int = 1024,
        mem_dim: int | None = None,
        thinking_token_id: int | None = None,  # Required when use_memory=True.
        pad_token_id: int | None = None,      # Optional; used to keep padding
                                              # rows out of the memory.
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
        if tie_embeddings:
            # Share parameters between embedding and lm_head. With tied
            # embeddings the embedding init also determines the logit
            # magnitude — default N(0, 1) gives logit std ≈ √d_model
            # (huge for d=576). Rescale to N(0, 1/√d_model) so initial
            # logits have unit-ish scale and softmax doesn't degenerate.
            nn.init.normal_(self.embed.weight, mean=0.0,
                            std=1.0 / math.sqrt(d_model))
            self.lm_head.weight = self.embed.weight
        self.tie_embeddings = bool(tie_embeddings)
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
        self.feedback_lag = int(feedback_lag)
        self.feedback_position = feedback_position
        self.feedback_per_channel_alpha = bool(feedback_per_channel_alpha)
        if feedback_position not in ("pre", "post"):
            raise ValueError(f"feedback_position must be 'pre' or 'post', got {feedback_position!r}")
        self.feedback_self_k = int(feedback_self_k)
        if self.feedback_self_k not in (0, 2, 3):
            raise ValueError(
                f"feedback_self_k must be 0, 2, or 3 (got {feedback_self_k!r})"
            )
        if self.feedback_self_k > 0 and not feedback_pairs:
            raise ValueError(
                "feedback_self_k requires sparse feedback_pairs to be non-empty"
            )
        if feedback_alpha_mode not in ("scalar", "surprise_modulated"):
            raise ValueError(
                f"feedback_alpha_mode must be 'scalar' or "
                f"'surprise_modulated' (got {feedback_alpha_mode!r})"
            )
        self.feedback_alpha_mode = feedback_alpha_mode
        if (feedback_alpha_mode == "surprise_modulated"
                and self.feedback_self_k < 2):
            raise ValueError(
                "feedback_alpha_mode='surprise_modulated' requires "
                "feedback_self_k >= 2 (we need at least 2 no-grad "
                "iterations to compute the inter-iter delta surprise "
                "signal before the loss-bearing pass)."
            )
        self.feedback_xattn_pairs = tuple(
            (int(t), tuple(int(s) for s in srcs)) for t, srcs in feedback_xattn_pairs
        )
        self.feedback_xattn_form = feedback_xattn_form
        self.xattn_target_to_sources = {t: srcs for t, srcs in self.feedback_xattn_pairs}
        # Surprise-gated scratchpad pairs (target → source; one source per target).
        self.feedback_scratchpad_pairs = tuple(
            (int(t), int(s)) for t, s in feedback_scratchpad_pairs
        )
        self.feedback_scratchpad_routing = feedback_scratchpad_routing
        if self.feedback_scratchpad_pairs:
            self.scratchpad_feedback = nn.ModuleDict({
                str(t): SurpriseScratchpadFeedback(
                    d_model=d_model, n_heads=feedback_scratchpad_heads,
                    routing=feedback_scratchpad_routing,
                )
                for t, _ in self.feedback_scratchpad_pairs
            })
            self.scratchpad_target_to_source = {
                t: s for t, s in self.feedback_scratchpad_pairs
            }
        else:
            self.scratchpad_target_to_source = {}
        if self.feedback_xattn_pairs and feedback_xattn_form == "all_sigmoid":
            # Single shared module across all (target, source) pairs with
            # parameter sharing per source layer. Used when feedback_xattn=='all'
            # or any large-target setting where naive instantiation would
            # explode the param count.
            self.xattn_all_module = AllToAllSigmoidFeedback(
                d_model=d_model,
                target_to_sources=self.xattn_target_to_sources,
                n_heads=feedback_xattn_heads,
            )
        elif self.feedback_xattn_pairs:
            # Per-target instantiation. Used by the 1- to ~10-target cases.
            #   attn          — additive Q-K-V softmax residual (default)
            #   film_sum      — multi-source FiLM, contributions summed (no softmax)
            #   film_sum_mlp  — film_sum but each W is an MLP
            #   film_attn     — softmax routing + multiplicative FiLM-form output
            #   film_sigmoid  — independent sigmoid gates + FiLM output
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
                if feedback_xattn_form == "film_sum_glu":
                    return MultiSourceFiLMFeedbackGLU(
                        d_model=d_model, n_sources=len(srcs),
                    )
                if feedback_xattn_form == "film_target_gated":
                    return MultiSourceFiLMTargetGated(
                        d_model=d_model, n_sources=len(srcs),
                    )
                if feedback_xattn_form == "film_attn":
                    return FiLMAttentionFeedback(
                        d_model=d_model, n_sources=len(srcs),
                        n_heads=feedback_xattn_heads,
                    )
                if feedback_xattn_form == "film_sigmoid":
                    return SigmoidGatedFiLMAttentionFeedback(
                        d_model=d_model, n_sources=len(srcs),
                        n_heads=feedback_xattn_heads,
                    )
                raise ValueError(f"unknown feedback_xattn_form: {feedback_xattn_form!r}")
            self.xattn_feedback = nn.ModuleDict({
                str(t): _make_xattn(srcs)
                for t, srcs in self.feedback_xattn_pairs
            })
        elif feedback_mode != "none" and self.feedback_pairs:
            # Sparse mode — one FeedbackProjection per (target, source) pair,
            # indexed by target layer string for ModuleDict.
            self.sparse_feedback = nn.ModuleDict({
                str(t): FeedbackProjection(
                    d_model, mode=feedback_mode,
                    per_channel_alpha=feedback_per_channel_alpha,
                    alpha_mode=feedback_alpha_mode,
                )
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

        # Phase 22 / structural-surprise full PoC: linear projection
        # W: R^{d_model} -> R^{semantic_loss_dim}, used by the trainer to
        # align pooled hidden states to the frozen oracle's E(s_t).
        self.semantic_loss_dim = int(semantic_loss_dim)
        if self.semantic_loss_dim > 0:
            # Identity-like init when the dims match (576 -> 576). Helps
            # the projection start near the encoder's representation space.
            self.W_semantic = nn.Linear(d_model, self.semantic_loss_dim,
                                         bias=False)
            if self.semantic_loss_dim == d_model:
                with torch.no_grad():
                    self.W_semantic.weight.copy_(torch.eye(d_model))
            else:
                nn.init.normal_(self.W_semantic.weight,
                                std=1.0 / math.sqrt(d_model))

        # Output gate (emit/think per-position gate, Phase 23).
        # gate_head: d_model → 1; sigmoid gives g_t ∈ (0, 1).
        # Bias = +2.0 → sigmoid(2) ≈ 0.88 so all positions start near
        # "always emit", making early training identical to a standard LM.
        self.output_gate = bool(output_gate)
        if self.output_gate:
            self.gate_head = nn.Linear(d_model, 1, bias=True)
            # Small-random weight (NOT zero) so the gate has position-dependent
            # output at init — the gate logit varies across the sequence even
            # before any training. The previous zero-weight init was tuned for
            # BPTT thinking-curriculum (which keeps the gate near uniform on
            # purpose to ease early training); for RL the result was a flat
            # gate with zero gradient signal across positions — RL could
            # never differentiate hard from easy tokens.
            #
            # Bias = +2.0 still puts mean σ(gate) ≈ 0.88 at init (mostly
            # emit), but each position is now perturbed by the W·h term,
            # giving non-trivial gate_std and a usable RL gradient.
            nn.init.normal_(self.gate_head.weight, std=0.02)
            nn.init.constant_(self.gate_head.bias, 2.0)

        # Bounded working memory: write-gated buffer + soft-attention read
        # at thinking positions only. See WorkingMemory docstring.
        self.use_memory = bool(use_memory)
        self.thinking_token_id = thinking_token_id
        self.pad_token_id = pad_token_id
        if self.use_memory:
            if thinking_token_id is None:
                raise ValueError("use_memory=True requires thinking_token_id")
            self.memory = WorkingMemory(
                d_model=d_model,
                d_mem=int(mem_dim) if mem_dim is not None else d_model,
                mem_size=int(mem_size),
                thinking_token_id=int(thinking_token_id),
                pad_token_id=pad_token_id,
            )
            # Re-init the thinking-token embedding row to the mean of the
            # other rows. PyTorch's default init makes it random noise, which
            # corrupts the recurrence every time a think token is appended.
            with torch.no_grad():
                mean_row = self.embed.weight.mean(dim=0)
                self.embed.weight[int(thinking_token_id)].copy_(mean_row)

    def _sparse_pass_collect_sources(self,
                                      x: torch.Tensor,
                                      source_layers: set,
                                      film_sources_lagged: dict | None,
                                      input_ids: torch.Tensor | None,
                                      surprise: torch.Tensor | None = None,
                                      ) -> dict:
        """One sparse-FiLM forward pass — collect source-layer outputs for
        use as next iteration's FiLM input.

        - x: input embeddings (B, T, d).
        - film_sources_lagged: dict[source_L -> (B, T, d)] — lagged source
          states to feed at FiLM target layers. If None, no FiLM (cold-start
          iteration 0).
        - surprise: optional (B, T) tensor — passed to FeedbackProjection in
          surprise-modulated mode to compute per-token α(t). For scalar α
          mode, ignored.

        Returns dict[source_L -> (B, T, d)] of THIS pass's source-layer
        outputs (un-lagged; the caller is responsible for applying the lag
        before feeding to the next iteration).
        """
        h = x
        out_src: dict = {}
        for L, blk in enumerate(self.blocks):
            if (film_sources_lagged is not None
                    and L in self.sparse_target_to_source
                    and self.feedback_position == "pre"):
                src = self.sparse_target_to_source[L]
                h = self.sparse_feedback[str(L)](h, film_sources_lagged[src],
                                                  surprise=surprise)
            h = blk(h, input_ids=input_ids)
            if (film_sources_lagged is not None
                    and L in self.sparse_target_to_source
                    and self.feedback_position == "post"):
                src = self.sparse_target_to_source[L]
                h = self.sparse_feedback[str(L)](h, film_sources_lagged[src],
                                                  surprise=surprise)
            if L in source_layers:
                out_src[L] = h
        return out_src

    def _apply_memory(self, h_normed: torch.Tensor,
                      input_ids: torch.Tensor,
                      read_mask: torch.Tensor | None = None) -> torch.Tensor:
        """No-op unless use_memory; else inject working-memory read at the
        positions selected by read_mask (default: think-token positions)."""
        if not self.use_memory:
            return h_normed
        return self.memory(h_normed, input_ids, read_mask=read_mask)

    def forward(self, input_ids: torch.Tensor,
                return_aux: bool = False,
                return_surprise: bool = False,
                return_hidden: bool = False,
                return_gate: bool = False,
                mem_read_mask: torch.Tensor | None = None,
                ) -> torch.Tensor | tuple:
        x = self.embed(input_ids)

        if self.max_T > 0:
            T = input_ids.shape[1]
            pos = torch.arange(T, device=input_ids.device)
            x = x + self.pos_embed(pos)

        def _maybe_gate(h_normed: torch.Tensor) -> torch.Tensor | None:
            """Compute gate (B, T) from normed hidden, store as _last_gate."""
            if not self.output_gate:
                return None
            gate_logits = self.gate_head(h_normed).squeeze(-1)
            g = torch.sigmoid(gate_logits)
            self._last_gate_logits = gate_logits
            self._last_gate = g  # side-effect: accessible after forward()
            return g

        if (self.feedback_mode == "none"
                and not self.feedback_xattn_pairs
                and not self.feedback_scratchpad_pairs):
            for blk in self.blocks:
                x = blk(x, input_ids=input_ids)
            h = self.out_norm(x)
            h = self._apply_memory(h, input_ids, read_mask=mem_read_mask)
            lm_logits = self.lm_head(h)
            outs = (lm_logits,)
            if return_aux and self.aux_dim > 0:
                outs = outs + (self.aux_head(h),)
            if return_hidden:
                outs = outs + (h,)
            if return_gate:
                outs = outs + (_maybe_gate(h),)
            else:
                _maybe_gate(h)   # still populate _last_gate as side effect
            return outs[0] if len(outs) == 1 else outs

        # Surprise-gated scratchpad feedback. Pass 1 vanilla, pass 1 logits to
        # compute per-position surprise (stop-grad), pass 2 applies the
        # scratchpad attention biased by surprise at the target layer(s).
        if self.feedback_scratchpad_pairs:
            needed_sources = set(s for _, s in self.feedback_scratchpad_pairs)
            pass1_at_sources: dict = {}
            h1 = x
            for L, blk in enumerate(self.blocks):
                h1 = blk(h1, input_ids=input_ids)
                if L in needed_sources:
                    pass1_at_sources[L] = h1
            # Pass-1 logits → per-position CE → surprise = CE − running mean.
            pass1_top = self.out_norm(h1)
            pass1_logits = self.lm_head(pass1_top)             # (B, T, V)
            B, T, V = pass1_logits.shape
            with torch.no_grad():
                if T > 1:
                    ce_pp = F.cross_entropy(
                        pass1_logits[:, :-1].reshape(-1, V),
                        input_ids[:, 1:].reshape(-1),
                        reduction="none",
                    ).reshape(B, T - 1)
                    # Causal cumulative mean.
                    cum_ce = ce_pp.cumsum(dim=-1)
                    pos = torch.arange(1, T, device=x.device, dtype=ce_pp.dtype)
                    running_mean = cum_ce / pos
                    surprise = (ce_pp - running_mean).clamp(min=0.0)
                    # Pad position 0 (no prediction yet) with 0 surprise.
                    surprise = F.pad(surprise, (1, 0), value=0.0)
                else:
                    surprise = torch.zeros(B, T, device=x.device)
            # Pass 2 with scratchpad feedback.
            h = x
            for L, blk in enumerate(self.blocks):
                if L in self.scratchpad_target_to_source:
                    src = self.scratchpad_target_to_source[L]
                    state_above_lagged = _shift_right_by_1(pass1_at_sources[src])
                    h = self.scratchpad_feedback[str(L)](
                        h, state_above_lagged, surprise=surprise,
                    )
                h = blk(h, input_ids=input_ids)
            h = self.out_norm(h)
            h = self._apply_memory(h, input_ids, read_mask=mem_read_mask)
            lm_logits = self.lm_head(h)
            outs = (lm_logits,)
            if return_aux and self.aux_dim > 0:
                outs = outs + (self.aux_head(h),)
            if return_gate:
                outs = outs + (_maybe_gate(h),)
            else:
                _maybe_gate(h)
            return outs[0] if len(outs) == 1 else outs
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
            # apply the chosen feedback module before the block's forward.
            h = x
            # Pre-compute lagged source states once.
            lagged_pass1 = {s: _shift_right_by_1(pass1_at_sources[s])
                            for s in needed_sources}
            for L, blk in enumerate(self.blocks):
                if L in self.xattn_target_to_sources:
                    if self.feedback_xattn_form == "all_sigmoid":
                        h = self.xattn_all_module.forward_at_target(
                            h, target=L, pass1_at_sources=lagged_pass1,
                        )
                    else:
                        srcs = self.xattn_target_to_sources[L]
                        src_states = [lagged_pass1[s] for s in srcs]
                        h = self.xattn_feedback[str(L)](h, src_states)
                h = blk(h, input_ids=input_ids)
            h = self.out_norm(h)
            h = self._apply_memory(h, input_ids, read_mask=mem_read_mask)
            lm_logits = self.lm_head(h)
            outs = (lm_logits,)
            if return_aux and self.aux_dim > 0:
                outs = outs + (self.aux_head(h),)
            if return_gate:
                outs = outs + (_maybe_gate(h),)
            else:
                _maybe_gate(h)
            return outs[0] if len(outs) == 1 else outs
        if self.feedback_pairs:
            source_layers = set(s for _, s in self.feedback_pairs)

            # ----------------------------------------------------------------
            # Self-feeding training (K-iteration fixed-point).
            #
            # At convergence the FiLM input the model uses at iteration K
            # equals the FiLM input it would generate at deployment under the
            # lagged-cached protocol — so deployment is a single forward.
            #
            # K=2: cold start (FiLM=0) → produce source state → feed it (lagged)
            #      to second pass; loss on second pass.
            # K=3: cold start → pass-2 with lag(pass-1) → pass-3 with lag(pass-2);
            #      loss on third pass.
            # All `.detach()`s mean backward only flows through the FINAL pass,
            # giving us 1× backward cost (same as current 2-pass training)
            # but K× forward cost (K=2 same as today; K=3 ~50 % more).
            # ----------------------------------------------------------------
            if self.feedback_self_k > 0:
                K = self.feedback_self_k
                # Iteration 0: cold start — no FiLM, vanilla forward.
                # We only need to collect source-layer outputs (for use as the
                # next iter's FiLM input).
                with torch.no_grad():
                    src_states = self._sparse_pass_collect_sources(
                        x, source_layers, film_sources_lagged=None,
                        input_ids=input_ids,
                    )
                # Iterations 1 .. K-2 (if K>2): self-feed. Detached so backward
                # never reaches them.
                # In surprise_modulated mode we also need the source state
                # from the SECOND-TO-LAST no_grad iteration (so we can compute
                # the inter-iter delta surprise = ||iter_{K-1} - iter_{K-2}||
                # before the loss-bearing pass starts). Track it explicitly.
                #
                # Iter naming (1-indexed in comments to match design doc):
                #   K=2: pass 1 (cold, no_grad), pass 2 (loss-bearing).
                #        Cannot compute inter-iter delta surprise (only one
                #        no_grad pass). [validated above by feedback_self_k>=2
                #        gating + the K=2 case relies on `prev_src_states`
                #        being a zero-init "previous" stand-in.]
                #   K=3: pass 1 (cold, no_grad), pass 2 (no_grad self-feed),
                #        pass 3 (loss-bearing). Surprise = ||pass2.src - pass1.src||,
                #        normalised then sigmoid'd → α(t) for pass 3's FiLM.
                prev_src_states = None     # iter k-2 source states (for delta)
                for _ in range(K - 2):
                    with torch.no_grad():
                        film_in = {s: _shift_right_by_k(v, self.feedback_lag)
                                    for s, v in src_states.items()}
                        prev_src_states = src_states   # save iter K-2 (= prev)
                        src_states = self._sparse_pass_collect_sources(
                            x, source_layers, film_sources_lagged=film_in,
                            input_ids=input_ids,
                        )
                # Final iteration K-1: this is the loss-bearing pass. Backprop
                # flows through this forward only. FiLM inputs come from the
                # PREVIOUS iter's source outputs (already-detached / no_grad
                # produced), shifted right by feedback_lag.
                film_in = {s: _shift_right_by_k(v, self.feedback_lag)
                            for s, v in src_states.items()}

                # Compute the per-token surprise signal for surprise_modulated
                # FiLM. We use the inter-iter delta between the two most-recent
                # no_grad iterations: ||src_states[s] - prev_src_states[s]||₂
                # at each (B, T) position, summed/averaged across the source
                # layers in self.sparse_target_to_source. The signal is a
                # MEASUREMENT of how much the model's own source-layer
                # representation shifted between the two no_grad iterations,
                # so it's already a self-surprise proxy without needing a
                # separate predictive head.
                surprise_per_target: dict = {}
                if self.feedback_alpha_mode == "surprise_modulated":
                    if prev_src_states is None:
                        # K=2 fallback (no two no_grad iterations available).
                        # Use a zero-surprise tensor so α(t) = α₀·σ(bias) —
                        # equivalent to a fixed-α form for the entire batch.
                        # Caller is warned by the constructor's K>=2 check; in
                        # practice we expect K=3 for surprise_modulated.
                        for t, s in self.feedback_pairs:
                            surprise_per_target[t] = x.new_zeros(x.shape[:2])
                    else:
                        for t, s in self.feedback_pairs:
                            # L2-norm across channels at each (B, T) position.
                            delta = src_states[s] - prev_src_states[s]
                            surprise_per_target[t] = (
                                delta.float().norm(dim=-1)        # (B, T)
                                .to(x.dtype)
                                .detach()                          # measurement
                            )
                # Run the final forward end-to-end with grad enabled.
                final_src_states: dict = {}
                h = x
                for L, blk in enumerate(self.blocks):
                    if L in self.sparse_target_to_source and self.feedback_position == "pre":
                        src = self.sparse_target_to_source[L]
                        sup = surprise_per_target.get(L) if surprise_per_target else None
                        h = self.sparse_feedback[str(L)](h, film_in[src], surprise=sup)
                    h = blk(h, input_ids=input_ids)
                    if L in self.sparse_target_to_source and self.feedback_position == "post":
                        src = self.sparse_target_to_source[L]
                        sup = surprise_per_target.get(L) if surprise_per_target else None
                        h = self.sparse_feedback[str(L)](h, film_in[src], surprise=sup)
                    if L in source_layers:
                        final_src_states[L] = h
                h_norm = self.out_norm(h)
                h_norm = self._apply_memory(h_norm, input_ids, read_mask=mem_read_mask)
                lm_logits = self.lm_head(h_norm)
                # Cache final pass's source states so the eval/diagnostic
                # callers can compute self-consistency norms cheaply.
                self._last_self_feed_final_src = {
                    s: v.detach() for s, v in final_src_states.items()
                }
                self._last_self_feed_prev_src = {
                    s: v.detach() for s, v in src_states.items()
                }
                # Cache surprise + per-token α(t) for diagnostic/eval use.
                # Only meaningful in surprise_modulated mode.
                if surprise_per_target:
                    self._last_surprise_per_target = {
                        t: s.detach() for t, s in surprise_per_target.items()
                    }
                    self._last_alpha_t_per_target = {
                        t: self.sparse_feedback[str(t)]
                            .get_per_token_alpha(s).detach()
                        for t, s in surprise_per_target.items()
                    }
                else:
                    self._last_surprise_per_target = {}
                    self._last_alpha_t_per_target = {}
                outs = (lm_logits,)
                if return_aux and self.aux_dim > 0:
                    outs = outs + (self.aux_head(h_norm),)
                if return_hidden:
                    outs = outs + (h_norm,)
                if return_gate:
                    outs = outs + (_maybe_gate(h_norm),)
                else:
                    _maybe_gate(h_norm)
                return outs[0] if len(outs) == 1 else outs
            # ----------------------------------------------------------------
            # Standard 2-pass sparse FiLM (existing path).
            # ----------------------------------------------------------------
            # Pass 1: vanilla, collect outputs only at source layers.
            pass1_at_sources: dict = {}
            h1 = x
            for L, blk in enumerate(self.blocks):
                h1 = blk(h1, input_ids=input_ids)
                if L in source_layers:
                    pass1_at_sources[L] = h1
            # Pass 2: forward with sparse modulation at target layers.
            # feedback_position: 'pre' = modulate input, 'post' = modulate output.
            # feedback_lag: how many tokens to shift the source state right.
            h = x
            for L, blk in enumerate(self.blocks):
                if L in self.sparse_target_to_source and self.feedback_position == "pre":
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_k(pass1_at_sources[src],
                                                            self.feedback_lag)
                    h = self.sparse_feedback[str(L)](h, state_above_lagged)
                h = blk(h, input_ids=input_ids)
                if L in self.sparse_target_to_source and self.feedback_position == "post":
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_k(pass1_at_sources[src],
                                                            self.feedback_lag)
                    h = self.sparse_feedback[str(L)](h, state_above_lagged)
            h = self.out_norm(h)
            h = self._apply_memory(h, input_ids, read_mask=mem_read_mask)
            lm_logits = self.lm_head(h)
            outs = (lm_logits,)
            if return_aux and self.aux_dim > 0:
                outs = outs + (self.aux_head(h),)
            if return_hidden:
                outs = outs + (h,)
            if return_gate:
                outs = outs + (_maybe_gate(h),)
            else:
                _maybe_gate(h)
            return outs[0] if len(outs) == 1 else outs
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
        h = self._apply_memory(h, input_ids, read_mask=mem_read_mask)
        lm_logits = self.lm_head(h)
        outs = (lm_logits,)
        if return_aux and self.aux_dim > 0:
            outs = outs + (self.aux_head(h),)
        if return_surprise:
            outs = outs + (surprise_loss,)
        if return_gate:
            outs = outs + (_maybe_gate(h),)
        else:
            _maybe_gate(h)
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
        # Use .mean().item() throughout so per-channel α (if enabled in
        # sparse_feedback) reduces to a scalar summary cleanly; for scalar
        # α it's a no-op.
        if self.feedback_xattn_pairs:
            if self.feedback_xattn_form == "all_sigmoid":
                return [(t, srcs,
                         float(self.xattn_all_module.alphas[str(t)].detach().mean().item()))
                        for t, srcs in self.feedback_xattn_pairs]
            return [(t, srcs, float(self.xattn_feedback[str(t)].alpha.detach().mean().item()))
                    for t, srcs in self.feedback_xattn_pairs]
        if self.feedback_mode == "none":
            return []
        if self.feedback_pairs:
            if self.feedback_alpha_mode == "surprise_modulated":
                # Report (target, source, α₀_zero); the per-token α(t) is
                # logged separately via _last_alpha_t_per_target. The
                # `_alpha_zero` magnitude is the analogue of the scalar α
                # in the unmodulated baseline (max α achievable when σ→1).
                return [
                    (t, s,
                     float(self.sparse_feedback[str(t)]
                           .alpha_zero.detach().mean().item()))
                    for t, s in self.feedback_pairs
                ]
            return [(t, s, float(self.sparse_feedback[str(t)].alpha.detach().mean().item()))
                    for t, s in self.feedback_pairs]
        if len(self.feedback_distances) == 1:
            return [float(fb.projs[0].alpha.detach().mean().item())
                    for fb in self.feedback]
        return [[float(p.alpha.detach().mean().item()) for p in fb.projs]
                for fb in self.feedback]

    def freeze_alpha(self) -> None:
        """Diagnostic: lock α at 0 across all layers (no active feedback)."""
        if self.feedback_xattn_pairs:
            if self.feedback_xattn_form == "all_sigmoid":
                self.xattn_all_module.freeze_alpha = True
            else:
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
