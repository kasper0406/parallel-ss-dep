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

from experiments.gist_loss import trunk_gist_loss


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


class ThinkAdapter(nn.Module):
    """Phase B thinking-specialized adapter (2026-05-26).

    A small 2-layer MLP `d_model → hidden_mult·d_model → d_model` with a
    learnable scalar α (init 0). At think positions, the adapter's
    contribution is added to the residual stream:

        h_out = h_in + α · think_mask · ThinkAdapter(h_in)

    α=0 at init means the adapter is byte-identical to "no adapter" at
    cold start; the optimizer opts in only via gradient as the adapter
    proves useful. Mirrors the FiLM-α / PKM-α / retrieval-α pattern.

    Why: the trunk runs IDENTICAL computation at think vs emit positions
    today — the only difference is the input embedding. Dedicated
    adapter params give the model the capacity for think-time-specialized
    processing. Routing: adapter weights + α → AdamW (not Muon — α is a
    scalar, and the dedicated function makes Newton-Schulz orthogonalization
    of the small Linear matrices conceptually wrong).
    """

    def __init__(self, d_model: int, hidden_mult: int = 2):
        super().__init__()
        d_hidden = int(hidden_mult) * d_model
        self.fc1 = nn.Linear(d_model, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=True)
        # α init = 0 so the adapter contributes 0 at cold start. Gradient
        # on α is non-zero because fc1/fc2 are at their normal init (the
        # FiLM-α lesson: zero α + zero W = no gradient). Single learnable
        # scalar; no weight decay (routed via is_film_alpha-style check).
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, think_mask: torch.Tensor) -> torch.Tensor:
        """Compute the adapter contribution (NOT added to h here; the caller
        does the residual add). think_mask is (B, T) bool/float.
        """
        # NOTE: we let fc2 run on every position (the matmul cost is the
        # same as masking afterwards on modern GPUs) and gate the result
        # by think_mask + α. Output is zero at non-think positions.
        adapter_out = self.fc2(F.gelu(self.fc1(h)))
        mask = think_mask.to(adapter_out.dtype).unsqueeze(-1)
        return self.alpha * mask * adapter_out


class LatentFeedbackAdapter(nn.Module):
    """Input adapter for LATENT (Coconut-style) thinking feedback (2026-06-01).

    The latent-thinking loop feeds the trunk's OWN `out_norm(h)` hidden state
    back as the next think-step `inputs_embeds`. But the input layer was only
    ever trained on embedding-TABLE vectors — a different manifold — so feeding
    a raw out_norm hidden is OOD and produces near-garbage (the documented
    Δlogp ≈ -4 to -6 "thinking HURTS" failure mode). This module learns the
    mapping `out_norm-hidden → input-embedding space` applied BEFORE the fed-back
    hidden is consumed as the next think input.

        z_adapted = z + α · Linear(RMSNorm(z))

    Init so behaviour is UNCHANGED at start: the `Linear` is zero-init AND a
    learnable scalar α (init 0) wraps it, so a fresh / untrained adapter is the
    identity `z_adapted == z` — i.e. byte-identical to the no-adapter latent
    path, and an existing ckpt without the adapter loads identically (the module
    is only built when the flag is on, and its zero-init makes its first forward
    a no-op). The optimizer opts in only via gradient (the FiLM-α / ThinkAdapter
    pattern). The Linear weights route to AdamW (matched by
    `optim_utils._is_latent_feedback_adapter`).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        # Zero-init the projection so the residual `z + α·proj(norm(z))` is the
        # identity at cold start regardless of α. α init 0 is belt-and-braces
        # (and gives the same opt-in-via-gradient curriculum the rest of the
        # stack uses); because proj.weight is also zero, the GRADIENT on α is
        # zero until proj has moved — so we DON'T zero proj.bias-only; instead
        # we keep proj fully zero and rely on α's gradient bootstrapping once
        # proj.weight gets gradient from the LM loss flowing through z.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map a fed-back hidden `z` (…, d_model) into input-embedding space.

        Identity at cold start (proj zero-init). `z` may be (N, 1, d) or (N, d).
        """
        return z + self.alpha * self.proj(self.norm(z))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int,
                 attention_cls: Callable[..., nn.Module],
                 use_think_adapter: bool = False,
                 think_adapter_hidden_mult: int = 2):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = attention_cls(d_model=d_model, n_heads=n_heads, d_head=d_head)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = GLU(d_model, d_ff)
        # Phase B thinking-specialized adapter (2026-05-26). Default off —
        # when off, no parameters are added, so existing ckpts load
        # byte-identically. When on, applied AFTER the attention + MLP
        # residual stream, gated by think_mask and α (init 0).
        self.use_think_adapter = bool(use_think_adapter)
        if self.use_think_adapter:
            self.think_adapter = ThinkAdapter(
                d_model=d_model,
                hidden_mult=int(think_adapter_hidden_mult),
            )

    def forward(self, x: torch.Tensor,
                input_ids: torch.Tensor | None = None,
                cu_seqlens: torch.Tensor | None = None,
                think_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
        # Symbol-grounded attention needs the raw token IDs to key its
        # sparse table. Other attentions ignore input_ids.
        # cu_seqlens (cross-document isolation) is passed only to attentions
        # that advertise `accepts_cu_seqlens` — FLA recurrent kernels.
        # think_mask (state-readonly-at-think, Phase 2 / 2026-05-26) is
        # passed only to attentions that advertise `accepts_think_mask` —
        # currently the DeltaNet _FlaWrapper with state_readonly enabled.
        attn = self.attn
        if getattr(attn, "needs_input_ids", False):
            x = x + attn(self.attn_norm(x), input_ids=input_ids)
        else:
            kw: dict = {}
            if cu_seqlens is not None and getattr(attn, "accepts_cu_seqlens", False):
                kw["cu_seqlens"] = cu_seqlens
            if think_mask is not None and getattr(attn, "accepts_think_mask", False):
                kw["think_mask"] = think_mask
            x = x + attn(self.attn_norm(x), **kw)
        x = x + self.mlp(self.mlp_norm(x))
        # Phase B think adapter (2026-05-26). Applied AFTER the standard
        # attention + MLP residual updates. The adapter's contribution is
        # gated by both the think_mask (B, T) and the learnable α scalar
        # init at 0 — so a cold start is byte-identical to no adapter.
        # think_mask is None when the caller doesn't have a thinking_token_id
        # (e.g. plain LM eval / tests with no think tokens); skip silently.
        if self.use_think_adapter and think_mask is not None:
            x = x + self.think_adapter(x, think_mask)
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
# ---------------------------------------------------------------------------


def _run_block(blk: nn.Module, h: torch.Tensor,
               input_ids: torch.Tensor | None,
               cu_seqlens: torch.Tensor | None = None,
               think_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Free function so torch.utils.checkpoint can pickle / re-run it
    cleanly (closures over `self` cause occasional issues with the
    non-reentrant checkpoint implementation)."""
    return blk(h, input_ids=input_ids, cu_seqlens=cu_seqlens,
               think_mask=think_mask)


def _ckpt_run_block(blk: nn.Module, h: torch.Tensor,
                    input_ids: torch.Tensor | None,
                    cu_seqlens: torch.Tensor | None = None,
                    think_mask: torch.Tensor | None = None) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint
    return checkpoint(_run_block, blk, h, input_ids, cu_seqlens, think_mask,
                      use_reentrant=False)


def _build_cu_seqlens(doc_ids: torch.Tensor | None) -> torch.Tensor | None:
    """Ragged `cu_seqlens` (int32, shape [N_segments + 1]) over the
    row-major-flattened (B, T) batch, for FLA's packed-sequence kernels.

    A new segment starts at every row boundary and wherever `doc_ids`
    changes within a row — so multiple documents packed into one T-length
    sequence each become an independent segment, and the DeltaNet recurrent
    state never flows across a document boundary.

    Returns None when `doc_ids` is None: the caller then takes the plain
    batched path, where each row is already an independent sequence (state
    does not flow across the batch dim) — the leak-free default for any
    stream that does not pack multiple documents per row.
    """
    if doc_ids is None:
        return None
    B, T = doc_ids.shape
    device = doc_ids.device
    if B * T == 0:
        return torch.zeros(1, dtype=torch.int32, device=device)
    flat = doc_ids.reshape(-1)
    idx = torch.arange(1, B * T, device=device)
    change = flat[1:] != flat[:-1]
    row_start = (idx % T) == 0
    is_start = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        change | row_start,
    ])
    starts = is_start.nonzero(as_tuple=False).flatten()
    cu = torch.cat([
        starts,
        torch.tensor([B * T], dtype=starts.dtype, device=device),
    ]).to(torch.int32)
    return cu


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
        if mode == "additive":
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
        if self.mode == "additive":
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
        read_alpha_init: float = 1.0,
        read_alpha_floor_start: float = 0.0,
        read_alpha_floor_warmup_steps: int = 0,
        decoupled_kv: bool = False,
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

        # DKV-WM (decoupled key/value, 2026-06-04). The default WM addresses
        # BY VALUE: scores = W_q(h)·W_v(h_src) — one projection does double duty
        # (be-matchable AND carry-content) and the raw dot-product has a
        # magnitude degeneracy (small-norm → flat softmax → the measured
        # top_mass=0.156 diffuse attention / mis-ranking → unreliable
        # addressing). With decoupled_kv: a dedicated match-KEY W_k is stored
        # per slot (W_v becomes content-only), scoring is cosine with a learned
        # CLAMPED temperature (sharpness no longer needs growing weight norms),
        # and the write-gate log-bias is β-scaled so write-recency can't swamp
        # query-match. This is the minimal structural requirement for semantic
        # (non-token-identity) recall. Default off → byte-identical legacy path.
        self.decoupled_kv = bool(decoupled_kv)
        if self.decoupled_kv:
            self.W_k = nn.Linear(d_model, d_mem, bias=False)
            nn.init.normal_(self.W_k.weight, std=0.02)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(math.sqrt(d_mem))))
            self.gate_bias_beta = nn.Parameter(torch.tensor(0.1))

        # Output α gate on the READ injection (added 2026-06-04). Every other
        # side-module in this repo (FiLM, PKM, RefinementHead, LineSelector)
        # has a learned scalar α that lets the model dial its contribution to
        # zero and fall back to the baseline trunk. WM was the lone exception:
        # it injected W_proj(read) UNCONDITIONALLY at read positions, so when
        # the injection hurt there was no escape hatch — at the saturating
        # MQAR regime (K=256/T=1024/lr=3e-3) this drove training to collapse
        # (recall 0.014, near-uniform loss) while plain DeltaNet hit 0.999.
        # read_alpha_init=0.0 → zero-init-residual (FiLM-α pattern): cold start
        # is byte-identical to no-WM, α moves first under loss gradient (∂L/∂α
        # is non-zero because W_proj is non-zero), the W_* weights follow once
        # α drifts off zero, and α→0 always recovers the baseline if WM hurts.
        # Default 1.0 preserves the pre-gate behaviour AND keeps old ckpts
        # (which lack this param) byte-identical when loaded with strict=False.
        self.read_alpha = nn.Parameter(torch.tensor(float(read_alpha_init)))

        # Sign-preserving additive α-FLOOR curriculum (mirrors PKM FIX 1B).
        # The utilization probe (2026-06-04) showed WM's seed-variance is an
        # ADDRESSING-learning failure, NOT a capacity one: failing seeds have
        # full value-coverage in the buffer but diffuse read attention (top
        # mass 0.16 vs 0.99) and a read_alpha that drifted DOWN (0.35 vs 0.55)
        # — the starvation loop "weak read → looks useless → α shrinks → read
        # gets less gradient → addressing never locks in". Holding the
        # EFFECTIVE contribution magnitude ≥ floor during a warmup window keeps
        # strong gradient on W_q/W_v/W_proj so the sharp addressing locks in,
        # then the floor decays to 0 and the learned α takes over. Default 0.0
        # = off (backwards-compat, compile-safe — no counter math runs).
        self.read_alpha_floor_start = float(read_alpha_floor_start)
        self.read_alpha_floor_warmup_steps = int(read_alpha_floor_warmup_steps)
        self.register_buffer("_fwd_count", torch.zeros((), dtype=torch.long),
                             persistent=False)

        # Diagnostics: when _capture_read is set True (probe-only, default off
        # to avoid the B·T·K activation cost during training), forward stashes
        # the per-position read-attention and the buffer's source positions so
        # a utilization probe can measure write concentration + read-addressing
        # accuracy. See experiments/probe_wm_utilization.py.
        self._capture_read = False
        self._last_read_attn = None
        self._last_top_idx = None

    def forward(self, h: torch.Tensor, input_ids: torch.Tensor,
                read_mask: torch.Tensor | None = None,
                doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        """`read_mask` (B, T) bool/float: 1 where memory should be injected.
        If None, derived from `input_ids == thinking_token_id`.

        `doc_ids` (B, T): when given, a query position may only read buffer
        slots from its own document — a think token in document 2 never
        attends to document 1's hidden states. None → no document masking
        (behaviour unchanged)."""
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
        if self.decoupled_kv:
            buf_k = torch.gather(self.W_k(h), dim=1, index=gather_idx_v)  # (B, K, d_mem)

        # ---- Read side: query for every position -----------------------------
        # We only USE the result at think positions (mask later); cheaper than
        # gather-only-think indices but keeps the implementation uniform.
        q = self.W_q(h)                                          # (B, T, d_mem)
        # scores: (B, T, K)
        if self.decoupled_kv:
            # Cosine addressing: match query to the slot KEY (not the value),
            # sharpness via a learned clamped temperature; β-scaled gate-bias.
            qn = F.normalize(q, dim=-1)
            kn = F.normalize(buf_k, dim=-1)
            tau = self.logit_scale.exp().clamp(2.0, 100.0)
            scores = torch.einsum("btd,bkd->btk", qn, kn) * tau
            scores = scores + self.gate_bias_beta * torch.log(buf_g.clamp_min(1e-6)).unsqueeze(1)
        elif getattr(self, "cosine_address", False):
            # No-train DKV-spirit probe: cosine(query, VALUE) with a fixed
            # temperature, using the EXISTING W_q/W_v (no W_k, no training).
            # Tests whether L2-normalization alone fixes the legacy
            # dot-product's magnitude-degeneracy / diffuse-softmax failure
            # before committing to a full DKV continuation. Default OFF
            # (getattr) so production paths are byte-identical.
            qn = F.normalize(q, dim=-1)
            vn = F.normalize(buf_v, dim=-1)
            tau = float(getattr(self, "cosine_address_tau", 20.0))
            scores = torch.einsum("btd,bkd->btk", qn, vn) * tau
            scores = scores + torch.log(buf_g.clamp_min(1e-6)).unsqueeze(1)
        else:
            scale = 1.0 / math.sqrt(self.d_mem)
            scores = torch.einsum("btd,bkd->btk", q, buf_v) * scale
            # Log-gate bias: tiny ε to keep log finite when a row's K-th slot has g=0.
            scores = scores + torch.log(buf_g.clamp_min(1e-6)).unsqueeze(1)

        # Causal mask: position t can attend to buffer-slot k iff top_idx[k] < t.
        # top_idx: (B, K) → (B, 1, K). pos: (1, T, 1).
        pos = torch.arange(T, device=device).view(1, T, 1)
        src_pos = top_idx.unsqueeze(1)                           # (B, 1, K)
        causal_mask = src_pos >= pos                              # (B, T, K)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Document mask: a query at position t may only read buffer slots whose
        # source token lies in the same document. Without this the memory
        # leaks hidden states across packed-document boundaries.
        blocked = causal_mask
        if doc_ids is not None:
            buf_doc = torch.gather(doc_ids, 1, top_idx)           # (B, K)
            doc_mask = buf_doc.unsqueeze(1) != doc_ids.unsqueeze(-1)  # (B, T, K)
            scores = scores.masked_fill(doc_mask, float("-inf"))
            blocked = causal_mask | doc_mask

        # Some rows (t < min source position, or no in-document predecessor)
        # get all -inf. Softmax of all -inf is NaN; replace those rows with
        # zero attention so the read is zero (and the injection is zero).
        all_masked = blocked.all(dim=-1, keepdim=True)            # (B, T, 1)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        if self._capture_read:
            self._last_read_attn = attn.detach()                  # (B, T, K)
            self._last_top_idx = top_idx.detach()                 # (B, K)

        read = torch.einsum("btk,bkd->btd", attn, buf_v)          # (B, T, d_mem)
        injection = self.W_proj(read)                             # (B, T, d_model)
        # Stash per-position pre-mask injection. Two versions:
        #   `_last_injection_grad` — keeps the autograd graph, used by
        #     auxiliary training losses (e.g. Option-A future-emb-pred
        #     through WM, 2026-05-19) so gradient on the aux loss flows
        #     to W_v / W_q / W_proj / W_write.
        #   `_last_injection` — detached, safe for use in inference
        #     generators and diagnostic probes.
        self._last_injection_grad = injection
        self._last_injection = injection.detach()

        # Inject only at "read positions" — either an explicit mask the
        # caller provided (e.g. MQAR query positions) or the default
        # thinking-token-based mask.
        if read_mask is None:
            inj = (input_ids == self.thinking_token_id).unsqueeze(-1).to(h.dtype)
        else:
            inj = read_mask.to(h.dtype).unsqueeze(-1)

        # Effective α = learned α (+ decaying sign-preserving floor in training).
        alpha = self.read_alpha
        if (self.training and self.read_alpha_floor_start > 0.0
                and self.read_alpha_floor_warmup_steps > 0):
            frac = (1.0 - self._fwd_count.float()
                    / self.read_alpha_floor_warmup_steps).clamp_min(0.0)
            floor = self.read_alpha_floor_start * frac
            self._fwd_count += 1
            sign = torch.sign(alpha.detach())
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            alpha = alpha + sign * floor
        return h + alpha * injection * inj


class RefinementHead(nn.Module):
    """Phase D — STRUCTURALLY DIFFERENT computation at think positions.

    The diagnosis from v8/v9 (Phase A/B failures): every "think" in the
    existing stack is just another forward pass through the SAME trunk
    with a slightly different input embedding. Supervising it doesn't
    help because there's no specialised computation to optimise.

    This module IS that specialised computation:
      - 1 layer of causal local-window self-attention (n_heads heads,
        sliding window of last `window` positions). The attention can
        see positions VERBATIM in the recent past — something the
        DeltaNet trunk's bounded recurrent state structurally cannot.
      - 2-layer GELU MLP for further mixing.
      - LayerNorm + residual on each sub-block (Pre-LN).
      - Scalar α (init 0) gating the entire contribution.

        h_out = h + α · (attn_residual + mlp_residual)

    With α=0 the output is bit-identical to the input — so a ckpt
    loaded WITHOUT refinement-head training is byte-identical to its
    pre-Phase-D self at decode. α moves first under loss gradient
    (matches the FiLM-α / Phase-B pattern); fc1/fc2/attn weights
    follow once α drifts off zero.

    Caller (`TinyLM._apply_refinement_head`) handles the gate-mix:
    `h_final = σ · h_trunk + (1-σ) · refinement_head(h_trunk)`.
    σ = P(emit); σ high → keep trunk; σ low (uncertain / would have
    wanted to think) → use refinement.
    """

    def __init__(self, d_model: int, n_heads: int = 8, window: int = 128,
                 mlp_mult: int = 2, alpha_init: float = 0.3):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads "
                f"({n_heads})")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = int(d_model // n_heads)
        self.window = int(window)
        self.mlp_mult = int(mlp_mult)

        self.attn_norm = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.mlp_norm = nn.LayerNorm(d_model)
        d_ff = mlp_mult * d_model
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

        # Alpha init: 0.0 → byte-identical at cold start but the head
        # stays inert (v10 lesson — α moved only 2.6e-4 over 960 steps,
        # MLPs never trained). 0.3 → head contributes from step 1, MLP
        # weights get real gradient, model learns whether to keep or
        # suppress the contribution. The α=0 short-circuit in
        # TinyLM._apply_refinement_head still preserves byte-identity
        # for ckpts that explicitly want zero contribution.
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

    def _build_window_mask(self, T: int, device, dtype) -> torch.Tensor:
        """Additive SDPA mask: 0 inside the causal sliding window, -inf
        outside. (T, T)."""
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        allowed = (j <= i) & ((i - j) < self.window)
        mask = torch.zeros((T, T), device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask

    def _local_attn(self, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        x = self.attn_norm(h)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        mask = self._build_window_mask(T, device=h.device, dtype=q.dtype)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(o)

    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
        x = self.mlp_norm(h)
        return self.W_down(F.gelu(self.W_up(x)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d). Returns h + α · refined_residual.

        With α=0, returns h exactly (the load-bearing invariant).
        """
        refined = self._local_attn(h)
        refined = refined + self._mlp(h + refined)
        return h + self.alpha * refined


class LineSelectorAttn(nn.Module):
    """Think-time op-selector — position-addressable VERBATIM line content (2026-06-03).

    At latent-thinking step `j`, softly select program line `j` (by a learned
    per-burst-step query) from the non-think prompt prefix and return its
    VERBATIM mean-pooled INPUT EMBEDDING. The selected content is injected as a
    zero-init-α additive side-channel into the trunk input at the think
    position only. This gives latent thinking position-addressable verbatim
    content access WITHOUT overwriting the carried latent thread (additive, not
    replacement; and never touches non-think positions).

    Cold start: `out_proj.weight` is zero-init AND a learnable scalar `alpha`
    (init 0) wraps the output, so a fresh / untrained selector returns EXACTLY
    zero at every position — the additive term is a no-op and the trunk is
    byte-identical to "no selector". The model opts in only via gradient (the
    FiLM-α / ThinkAdapter / LatentFeedbackAdapter pattern).

    The prompt is segmented into "lines" by a running cumsum of the newline
    token over the non-think prefix; each line's value is the mean of its
    constituent token INPUT embeddings (verbatim content, not trunk hidden).
    Keys add a per-line index embedding so the query can address by position
    (line 0, line 1, ...). Queries are a per-think-step embedding indexed by the
    token's position within its consecutive-think burst. The whole prompt
    precedes the think tokens, so strict causality is unnecessary.
    """

    def __init__(self, d_model: int, max_lines: int = 64, max_burst: int = 32):
        super().__init__()
        self.d_model = int(d_model)
        self.max_lines = int(max_lines)
        self.max_burst = int(max_burst)
        # Per-line index key (addressable by absolute line position) and the
        # per-think-step query table.
        self.line_key_emb = nn.Embedding(self.max_lines, d_model)
        self.burst_q_emb = nn.Embedding(self.max_burst, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Zero-init the OUTPUT projection -> cold-start no-op (output = 0, byte-
        # identical load). alpha is a FIXED scale of 1.0, NOT a zero gate. With
        # out_proj zero-init, d_loss/d_out_proj ∝ alpha = 1 (nonzero) so out_proj
        # learns the right direction from zero — the standard zero-init-residual
        # trick. (Two earlier inits FAILED: zero-init BOTH alpha & out_proj = dead
        # gradient on both; alpha=0 + nonzero out_proj = a scalar can only scale a
        # RANDOM out_proj direction -> sign-oscillating grad -> alpha stuck at 0,
        # selector never activated through step 5k. 2026-06-04.)
        nn.init.zeros_(self.out_proj.weight)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor,
                embeds: torch.Tensor, thinking_token_id: int,
                newline_token_id: int, burst_idx: torch.Tensor) -> torch.Tensor:
        """Returns an additive (B, T, d) term, ZERO everywhere except think
        positions (and exactly zero at cold start).

        x:        (B, T, d) current trunk input (unused for content; kept for
                  signature symmetry with the other think-time modules).
        input_ids:(B, T) long token ids.
        embeds:   (B, T, d) raw token INPUT embeddings = model.embed(input_ids).
        burst_idx:(B, T) long index of each token within its consecutive-think
                  burst (0 elsewhere); the per-think-step query index.
        """
        B, T, d = embeds.shape
        device = embeds.device
        think_mask = (input_ids == thinking_token_id)             # (B, T) bool
        # No-think (all-prompt) batch: nothing to inject -> return zeros. The
        # final output is gated by think_mask anyway, so a plain zeros tensor is
        # the correct early-out.
        if not bool(think_mask.any()):
            return embeds.new_zeros(B, T, d)
        prompt_mask = ~think_mask                                 # (B, T) bool

        # Per-token line id: cumsum of newline occurrences within the prompt.
        # Counting BEFORE a newline keeps the newline token itself on the line
        # it terminates; we just need a stable per-line bucket. Clamp to the
        # last bucket so over-long programs don't index OOB.
        is_nl = (input_ids == newline_token_id) & prompt_mask     # (B, T) bool
        line_id = is_nl.to(torch.int64).cumsum(dim=1)             # (B, T)
        line_id = line_id.clamp(max=self.max_lines - 1)           # (B, T)

        # Vectorized scatter-mean of `embeds` over line_id (prompt tokens only).
        # line_sum[b, l] = sum of embeds[b, t] for prompt tokens t on line l.
        # line_cnt[b, l] = number of such tokens.
        pm = prompt_mask.unsqueeze(-1).to(embeds.dtype)           # (B, T, 1)
        masked_emb = embeds * pm                                  # (B, T, d)
        line_sum = embeds.new_zeros(B, self.max_lines, d)
        idx_d = line_id.unsqueeze(-1).expand(B, T, d)             # (B, T, d)
        line_sum.scatter_add_(1, idx_d, masked_emb)              # (B, max_lines, d)
        line_cnt = embeds.new_zeros(B, self.max_lines)
        line_cnt.scatter_add_(1, line_id, prompt_mask.to(embeds.dtype))
        line_valid = line_cnt > 0                                 # (B, max_lines) bool
        denom = line_cnt.clamp(min=1.0).unsqueeze(-1)             # (B, max_lines, 1)
        line_val = line_sum / denom                              # (B, max_lines, d)

        # Keys: projected line content + a per-line absolute-index embedding.
        line_ids = torch.arange(self.max_lines, device=device)
        keys = self.k_proj(line_val) + self.line_key_emb(line_ids).unsqueeze(0)
        vals = self.v_proj(line_val)                             # (B, max_lines, d)

        # Query per think position from the burst-step table.
        q_idx = burst_idx.clamp(max=self.max_burst - 1)          # (B, T)
        query = self.burst_q_emb(q_idx)                          # (B, T, d)

        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(d)
        # Mask invalid (empty) lines so softmax never attends to padding rows.
        invalid = (~line_valid).unsqueeze(1)                     # (B, 1, max_lines)
        scores = scores.masked_fill(invalid, float("-inf"))      # (B, T, max_lines)
        attn = torch.softmax(scores, dim=-1)                     # (B, T, max_lines)
        sel = torch.matmul(attn, vals)                          # (B, T, d)
        out = self.out_proj(sel) * self.alpha                    # zero at cold start
        # Additive term: ZERO at non-think positions.
        return out * think_mask.unsqueeze(-1).to(out.dtype)


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
        feedback_mode: str = "none",          # none / additive / film
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
        mem_read_alpha_init: float = 1.0,     # α gate on WM read injection;
                                              # 0.0 = zero-init-residual boot.
        mem_read_alpha_floor_start: float = 0.0,   # PKM-style α-floor for WM
        mem_read_alpha_floor_warmup_steps: int = 0,  # addressing bootstrap.
        mem_decoupled_kv: bool = False,       # DKV-WM: decoupled key/value +
                                              # cosine addressing (reliable
                                              # semantic addressing).
        cooperative_latent_wm: bool = False,  # WM×latent cooperation: register a
                                              # learned `mem_alpha` so a latent
                                              # think step can add an α-gated WM
                                              # retrieval (adapter(h_premem)+α·inj).
                                              # Must be a real ctor param (not a
                                              # runtime attr) so it round-trips
                                              # through save/load.
        thinking_token_id: int | None = None,  # Required when use_memory=True.
        pad_token_id: int | None = None,      # Optional; used to keep padding
                                              # rows out of the memory.
        layer_drop_max: float = 0.0,          # Stochastic Depth: linearly
                                              # increasing per-block drop
                                              # prob 0 → layer_drop_max
                                              # with depth. 0 = off.
        activation_checkpointing: bool = False,  # Wrap each Block's
                                              # loss-bearing forward in
                                              # torch.utils.checkpoint to
                                              # trade ~30% compute for a
                                              # large reduction in
                                              # activation memory. No effect
                                              # under torch.no_grad (early
                                              # K-self-feed passes), so
                                              # only the final loss-bearing
                                              # pass pays the cost. Output
                                              # is bit-identical because
                                              # the model has no dropout
                                              # and preserve_rng_state=True
                                              # by default.
        use_pkm: bool = False,                # Persistent learned-RAG /
                                              # Product-Key Memory layer.
                                              # Drop-in residual side-table
                                              # at one mid-depth block. See
                                              # PKM_PLAN.md and
                                              # experiments/memory_layer.py.
        pkm_after_layer: int = 14,
        pkm_n_keys: int = 256,
        pkm_n_heads: int = 4,
        pkm_k_dim: int = 128,
        pkm_top_k: int = 32,
        pkm_value_bf16: bool = True,
        # v7 PKM-bootstrap-fix package (2026-05-17). See PKMLayer.__init__.
        pkm_score_norm: str = "layer",
        pkm_value_init_std: float = 1.0,
        pkm_use_output_gate: bool = True,
        # Phase 2 thinking fix (2026-05-26): force DeltaNet β=0 at think
        # positions so think tokens can read the recurrent state but never
        # write to it. Preserves long-range bindings across multi-think
        # bursts (the documented 100% → 20% recall-at-512 drop). Only
        # applies to plain DeltaNet attention blocks. Default OFF (every
        # existing ckpt behaves byte-identically without the flag).
        state_readonly_at_think: bool = False,
        # Phase 3 thinking fix (2026-05-26): per-position think-index
        # embedding. Each think token in a consecutive burst gets a small
        # learned positional embedding added on top of the [THINKING]
        # input embedding (position 0, 1, ..., size-1). Without this, an
        # 8-think burst feeds the same embedding 8 times → the resulting
        # hidden states have median pairwise cos +0.146 vs +0.060 at emit
        # and effective rank ~210 vs ~560 (diag_think_position_diversity).
        # Default 0 (disabled) keeps existing ckpts byte-identical. The
        # table is init to zero so a cold start has no effect; the model
        # opts in only via gradient.
        think_index_emb_size: int = 0,
        # Phase B thinking-specialized adapter (2026-05-26). When True,
        # each Block grows a small 2-layer MLP `d_model → hidden_mult·d_model
        # → d_model` whose contribution is gated by the per-position
        # think_mask AND a learnable scalar α (init 0). With α=0 a cold
        # start is byte-identical to "no adapter"; existing ckpts without
        # adapter weights load with strict=False. Adapter params route to
        # AdamW (not Muon — see optim_utils._is_think_adapter).
        use_think_adapter: bool = False,
        think_adapter_hidden_mult: int = 2,
        # Phase D (THINKING_PLAN v5, 2026-05-27): RefinementHead — a
        # dedicated module with windowed local attention + MLP whose
        # output is soft-mixed with the trunk hidden by σ(gate). Gives
        # the gate σ a REAL job: weight two STRUCTURALLY DIFFERENT
        # predictions, not "compute / don't compute." Default OFF; with
        # use_refinement_head=True the head's α starts at 0 so a fresh
        # ckpt is byte-identical to the no-head trunk at decode.
        use_refinement_head: bool = False,
        refinement_head_window: int = 128,
        refinement_head_n_heads: int = 8,
        refinement_head_mlp_mult: int = 2,
        refinement_head_alpha_init: float = 0.3,
        # Latent-thinking input adapter (2026-06-01). When True, build a
        # `LatentFeedbackAdapter` that maps the fed-back out_norm hidden into
        # the input-embedding manifold before it is consumed as the next
        # latent think-step input. Zero-init → identity → a fresh ckpt is
        # byte-identical to the no-adapter latent path. Default OFF (existing
        # ckpts load identically; they simply lack the adapter keys).
        use_latent_feedback_adapter: bool = False,
        # Think-time op-selector (2026-06-03). When True, build a
        # `LineSelectorAttn` that softly selects a program LINE from the prompt
        # by a learned per-think-step query and injects that line's verbatim
        # mean-pooled input embedding as a zero-init-α additive side-channel at
        # think positions only. Zero-init out_proj + α(0) → byte-identical to
        # OFF at cold start, so an existing ckpt loads identically (it simply
        # lacks the `line_selector.*` keys). `newline_token_id` is required for
        # line segmentation.
        use_line_selector: bool = False,
        line_selector_max_lines: int = 64,
        newline_token_id: int | None = None,
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
        # Phase B (2026-05-26) — thread `use_think_adapter` into each Block.
        # Stored here too so TinyLM.forward knows whether to build the
        # per-position think_mask even when state_readonly_at_think is OFF.
        self.use_think_adapter = bool(use_think_adapter)
        self.think_adapter_hidden_mult = int(think_adapter_hidden_mult)
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, n_heads=n_heads, d_head=d_head,
                  d_ff=d_ff, attention_cls=cls,
                  use_think_adapter=self.use_think_adapter,
                  think_adapter_hidden_mult=self.think_adapter_hidden_mult)
            for cls in cls_list
        ])
        # Phase-2 state-readonly thinking (2026-05-26). Install β-mask
        # wrapping on every plain-DeltaNet block. Other FLA variants
        # (GatedDeltaNet, etc.) and non-FLA blocks (Symbol-grounded,
        # OrthogonalScan, ...) silently skip — we only have the recipe
        # for plain DeltaNet's `b_proj`.
        self.state_readonly_at_think = bool(state_readonly_at_think)
        if self.state_readonly_at_think:
            from experiments.layers import _FlaWrapper
            for blk in self.blocks:
                attn = blk.attn
                if isinstance(attn, _FlaWrapper) and hasattr(attn.layer, "b_proj"):
                    attn.enable_state_readonly_at_think()
        # Phase 3 think-index embedding (2026-05-26). Created here so
        # state-dict round-trip preserves it; zero-init so an unused
        # table contributes 0 at every position.
        self.think_index_emb_size = int(think_index_emb_size)
        if self.think_index_emb_size > 0:
            self.think_index_emb = nn.Embedding(self.think_index_emb_size, d_model)
            nn.init.zeros_(self.think_index_emb.weight)
        # Phase D RefinementHead (2026-05-27) — built here so state-dict
        # round-trip preserves weights. α init 0 (inside the module) → a
        # freshly-attached head is byte-identical to no-head at decode.
        self.use_refinement_head = bool(use_refinement_head)
        self.refinement_head_window = int(refinement_head_window)
        self.refinement_head_n_heads = int(refinement_head_n_heads)
        self.refinement_head_mlp_mult = int(refinement_head_mlp_mult)
        self.refinement_head_alpha_init = float(refinement_head_alpha_init)
        if self.use_refinement_head:
            self.refinement_head = RefinementHead(
                d_model=d_model,
                n_heads=self.refinement_head_n_heads,
                window=self.refinement_head_window,
                mlp_mult=self.refinement_head_mlp_mult,
                alpha_init=self.refinement_head_alpha_init,
            )
        # Latent-thinking input adapter (2026-06-01). Built here so state-dict
        # round-trip preserves weights; zero-init → identity → a freshly
        # attached adapter is byte-identical to the no-adapter latent path.
        self.use_latent_feedback_adapter = bool(use_latent_feedback_adapter)
        if self.use_latent_feedback_adapter:
            self.latent_feedback_adapter = LatentFeedbackAdapter(d_model)
        # Think-time op-selector (2026-06-03). Built here so state-dict
        # round-trip preserves weights; zero-init out_proj + α(0) → byte-identical
        # to OFF at cold start. `newline_token_id` drives line segmentation.
        self.use_line_selector = bool(use_line_selector)
        self.line_selector_max_lines = int(line_selector_max_lines)
        self.newline_token_id = newline_token_id
        if self.use_line_selector:
            self.line_selector = LineSelectorAttn(
                d_model, max_lines=self.line_selector_max_lines)
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
        # Runtime curriculum flag: when True, forward() takes the plain
        # block-loop path (1 forward, no FiLM passes) regardless of
        # feedback_mode. The trainer flips this off after a warmup so the
        # expensive K-self-feed isn't paid while early grad is just noise.
        self._film_bypass = False
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
        # Activation checkpointing on the loss-bearing block loop. See the
        # forward() helper _block_fwd() for usage.
        self.activation_checkpointing = bool(activation_checkpointing)
        self.layer_drop_max = float(layer_drop_max)
        if self.use_memory:
            if thinking_token_id is None:
                raise ValueError("use_memory=True requires thinking_token_id")
            self.memory = WorkingMemory(
                d_model=d_model,
                d_mem=int(mem_dim) if mem_dim is not None else d_model,
                mem_size=int(mem_size),
                thinking_token_id=int(thinking_token_id),
                pad_token_id=pad_token_id,
                read_alpha_init=float(mem_read_alpha_init),
                read_alpha_floor_start=float(mem_read_alpha_floor_start),
                read_alpha_floor_warmup_steps=int(mem_read_alpha_floor_warmup_steps),
                decoupled_kv=bool(mem_decoupled_kv),
            )
            # Re-init the thinking-token embedding row to the mean of the
            # other rows. PyTorch's default init makes it random noise, which
            # corrupts the recurrence every time a think token is appended.
            with torch.no_grad():
                mean_row = self.embed.weight.mean(dim=0)
                self.embed.weight[int(thinking_token_id)].copy_(mean_row)
            # v7 (2026-05-20): additive α-gated retrieval-as-input.
            # The retrieval-as-input design (v5/v6) REPLACED the
            # think-token input embedding with the WM retrieval — a
            # destructive overwrite. The v6 long-context eval showed it
            # corrupts precise recall (99%→61% as distance grows): a
            # blurry retrieval fed in as the input wipes the precise
            # binding the DeltaNet trunk was carrying. v7 makes the
            # injection ADDITIVE and scalar-gated:
            #     input[think] = think_embed + α · retrieval
            # A useless retrieval contributes ≈0 (gradient shrinks α);
            # a useful one is gated in. The think_embed baseline is
            # always present, so a bad retrieval can never corrupt.
            # Init 0.1 so the retrieval has a small effect from step 0
            # and receives gradient. Mirrors the FiLM-α / PKM-α pattern.
            # Consumed by sft_code.py (training) and
            # eval_humaneval.generate_with_retrieval_as_input (eval) —
            # the model.forward path itself is unchanged (the caller
            # builds inputs_embeds). Keep WD off this parameter.
            self.retrieval_input_alpha = nn.Parameter(torch.tensor(0.1))

            # WM×latent cooperation scalar (M0). Registered as a real parameter
            # (init 0.1, no-WD) ONLY when cooperative_latent_wm=True, so it
            # round-trips through state_dict/build_model_from_ckpt — fixing the
            # audit bug where trainers set `model.mem_alpha` as a runtime attr
            # that load_state_dict(strict=False) silently dropped, disabling the
            # coupling at eval. At a latent think step the next input embedding is
            #   adapter(h_premem) + mem_alpha · WM_injection
            # so a useless retrieval contributes ≈0 and the clean adapter
            # baseline always survives (FiLM-α curriculum). Default OFF →
            # byte-identical to pre-cooperation ckpts.
            if cooperative_latent_wm:
                self.mem_alpha = nn.Parameter(torch.tensor(0.1))

        # Persistent learned-RAG (Product-Key Memory). Drop-in residual
        # at one mid-depth block. See PKM_PLAN.md.
        self.use_pkm = bool(use_pkm)
        self.pkm_after_layer = int(pkm_after_layer)
        if self.use_pkm:
            from experiments.memory_layer import PKMLayer
            if not (0 <= self.pkm_after_layer < n_layers):
                raise ValueError(
                    f"pkm_after_layer={self.pkm_after_layer} out of range "
                    f"for n_layers={n_layers}"
                )
            self.pkm_layer = PKMLayer(
                d_model=d_model,
                n_heads=int(pkm_n_heads),
                n_keys=int(pkm_n_keys),
                k_dim=int(pkm_k_dim),
                top_k=int(pkm_top_k),
                value_bf16=bool(pkm_value_bf16),
                score_norm=str(pkm_score_norm),
                value_init_std=float(pkm_value_init_std),
                use_output_gate=bool(pkm_use_output_gate),
            )

    def _compute_think_index_emb(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-position additive think-index embedding (Phase 3).

        Returns a (B, T, d_model) tensor: at think positions the embedding
        for that token's index within its consecutive-think burst (0 for
        the first think in a burst, 1 for the second, ...), clamped to
        `think_index_emb_size - 1`. At non-think positions the contribution
        is zero. Vectorized via the cumsum-reset trick:
        let `c = cumsum(think_mask)` and `reset = cummax(c * (1-mask))` —
        the cumsum value just before the current burst began — then the
        per-position 0-indexed burst-position is `(c - reset - 1).clamp(0)`.
        """
        think_mask = (input_ids == int(self.thinking_token_id))
        burst_idx = self._compute_burst_index(input_ids)
        burst_idx = burst_idx.clamp(max=self.think_index_emb_size - 1)
        idx_emb = self.think_index_emb(burst_idx)
        # Mask to think positions only (non-think positions contribute 0).
        return idx_emb * think_mask.unsqueeze(-1).to(idx_emb.dtype)

    def _compute_burst_index(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-token 0-indexed position within its consecutive-think burst.

        Returns a (B, T) long tensor: at think positions, the token's index
        within the run of consecutive think tokens it belongs to (0 for the
        first think in a burst, 1 for the second, ...); 0 at non-think
        positions. Vectorized via the cumsum-reset trick: let
        `c = cumsum(think_mask)` and `reset = cummax(c * (1-mask))` (the cumsum
        value just before the current burst began), then the burst-position is
        `(c - reset - 1).clamp(0)`. Shared by `_compute_think_index_emb` (Phase
        3 index embedding) and `LineSelectorAttn` (per-think-step query index).
        """
        think_mask = (input_ids == int(self.thinking_token_id))
        m_int = think_mask.to(torch.int64)
        c = m_int.cumsum(dim=1)
        reset = (c * (1 - m_int)).cummax(dim=1).values
        return (c - reset - 1).clamp(min=0)

    def _maybe_pkm(self, h: torch.Tensor, L: int) -> torch.Tensor:
        """Apply the PKM residual side-table after layer L iff configured.

        Called after every Block forward in the loss-bearing path AND in
        the K-self-feed no-grad warmup passes — without the latter, the
        FiLM source-state collected in passes 1/2 differs from what the
        deployed model sees, breaking K=3 self-consistency.
        """
        if self.use_pkm and L == self.pkm_after_layer:
            h = h + self.pkm_layer(h)
        return h

    def _sparse_pass_collect_sources(self,
                                      x: torch.Tensor,
                                      source_layers: set,
                                      film_sources_lagged: dict | None,
                                      input_ids: torch.Tensor | None,
                                      surprise: torch.Tensor | None = None,
                                      cu_seqlens: torch.Tensor | None = None,
                                      think_mask: torch.Tensor | None = None,
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
            h = blk(h, input_ids=input_ids, cu_seqlens=cu_seqlens,
                    think_mask=think_mask)
            if (film_sources_lagged is not None
                    and L in self.sparse_target_to_source
                    and self.feedback_position == "post"):
                src = self.sparse_target_to_source[L]
                h = self.sparse_feedback[str(L)](h, film_sources_lagged[src],
                                                  surprise=surprise)
            h = self._maybe_pkm(h, L)
            if L in source_layers:
                out_src[L] = h
        return out_src

    def _block_fwd(self, blk: nn.Module, h: torch.Tensor,
                   input_ids: torch.Tensor | None,
                   L: int = -1,
                   cu_seqlens: torch.Tensor | None = None,
                   think_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run one Block; optionally checkpoint and/or stochastically drop.

        Activation checkpointing trades ~30% extra compute for a large
        reduction in stored activations; skipped when grad is disabled.

        LayerDrop (Huang et al. 2016 / Fan et al. 2020): with probability
        `layer_drop_max · L / (n_layers - 1)`, return `h` unchanged —
        i.e., the block's contribution is dropped. Only active during
        training. No survivor rescaling (Fan-et-al. convention).
        """
        if (self.training and self.layer_drop_max > 0.0 and L >= 0
                and self.layer_drop_max * L
                / max(1, len(self.blocks) - 1) > 0.0):
            p_L = self.layer_drop_max * L / max(1, len(self.blocks) - 1)
            if torch.rand((), device=h.device).item() < p_L:
                return h
        if self.activation_checkpointing and torch.is_grad_enabled():
            h = _ckpt_run_block(blk, h, input_ids, cu_seqlens, think_mask)
        else:
            h = blk(h, input_ids=input_ids, cu_seqlens=cu_seqlens,
                    think_mask=think_mask)
        return self._maybe_pkm(h, L)

    def _apply_memory(self, h_normed: torch.Tensor,
                      input_ids: torch.Tensor,
                      read_mask: torch.Tensor | None = None,
                      doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        """No-op unless use_memory; else inject working-memory read at the
        positions selected by read_mask (default: think-token positions).
        `doc_ids` (when given) confines reads/writes to within a document."""
        if not self.use_memory:
            return h_normed
        return self.memory(h_normed, input_ids, read_mask=read_mask,
                           doc_ids=doc_ids)

    def apply_latent_feedback_adapter(self, z: torch.Tensor) -> torch.Tensor:
        """Map a fed-back latent hidden `z` into the input-embedding manifold.

        Called by the latent-thinking loop (`thinking._latent_think_logits`
        and its grad twin) on the trunk's own `out_norm(h)` hidden BEFORE it is
        used as the next think-step `inputs_embeds`. No-op (identity) when the
        adapter is not built — so every existing latent callsite keeps its
        current behaviour unless the model was constructed with
        `use_latent_feedback_adapter=True`. With the adapter built but untrained
        (zero-init proj) it is also the identity, so a fresh ckpt is unchanged.
        """
        if not getattr(self, "use_latent_feedback_adapter", False):
            return z
        return self.latent_feedback_adapter(z)

    def _apply_refinement_head(self, h: torch.Tensor) -> torch.Tensor:
        """Phase D soft-mixture: σ · h + (1-σ) · refinement_head(h).

        σ = sigmoid(gate_head(out_norm(h))) — P(emit). σ high (confident
        emit) keeps the trunk output; σ low (would have wanted to
        think) routes to the refinement-head output. Refinement head
        starts with α=0 so the mix is identity at cold start.

        Tests can override σ via `model._force_gate_sigma = <float>`.

        No-op when use_refinement_head is False (byte-identical).
        """
        if not getattr(self, "use_refinement_head", False):
            return h
        # Short-circuit when α is exactly 0 — preserves byte-identity
        # against the no-head trunk for freshly-attached heads, since
        # the soft-mix `σ·h + (1-σ)·h` accumulates fp32 rounding even
        # when h_refined == h.
        if self.refinement_head.alpha.item() == 0.0 and not self.training:
            return h
        if not getattr(self, "output_gate", False):
            # Without a gate head we can't soft-mix; just apply the head
            # additively (α=0 → identity). Useful for non-thinking ckpts
            # that experiment with the head alone.
            return self.refinement_head(h)
        h_norm = self.out_norm(h)
        gate_logit = self.gate_head(h_norm)
        override = getattr(self, "_force_gate_sigma", None)
        sigma = (torch.sigmoid(gate_logit) if override is None
                 else torch.full_like(gate_logit, float(override)))
        self._last_refinement_sigma = sigma.detach()
        h_refined = self.refinement_head(h)
        return sigma * h + (1.0 - sigma) * h_refined

    def _finalize(self, h_raw: torch.Tensor,
                  input_ids: torch.Tensor,
                  mem_read_mask: torch.Tensor | None,
                  return_aux: bool,
                  return_hidden: bool,
                  return_gate: bool,
                  maybe_gate,
                  extra: tuple = (),
                  doc_ids: torch.Tensor | None = None,
                  skip_lm_head: bool = False):
        """Shared exit-tail for every forward branch:
        out_norm → memory → lm_head → (aux, hidden, gate) packing.

        `maybe_gate` is the closure defined in forward() so the gate side
        effect (`self._last_gate*`) lands consistently.
        `extra` is appended to the outputs after aux/hidden/gate — used by
        the dense-feedback path to thread `surprise_loss` through.
        `doc_ids` confines working-memory reads/writes to within a document.
        """
        # Phase D: soft-mix refinement-head output with the trunk hidden
        # BEFORE out_norm / memory / lm_head. The mix is
        # `σ · h_raw + (1-σ) · refinement_head(h_raw)` where σ = P(emit)
        # comes from the gate_head (computed on out_norm(h_raw)). At
        # cold-start refinement_head's α=0 → output == h_raw → mix is
        # identity for any σ → byte-identical to no-head training.
        h_raw = self._apply_refinement_head(h_raw)
        h = self.out_norm(h_raw)
        # Pre-memory hidden = the CLEAN latent-thread source. With WM on, the
        # post-memory `h` (below) carries a WM retrieval injected at think
        # positions; feeding THAT back as the latent thread overwrites the
        # precise value the thread must carry (audit 2026-06-03). When
        # `_latent_feedback_premem` is set, return_hidden hands back this
        # pre-memory hidden so the thread stays clean while WM still shapes the
        # emitted logits below. Default off -> byte-identical to old behaviour.
        h_premem = h
        self._last_premem_hidden = h_premem
        h = self._apply_memory(h, input_ids, read_mask=mem_read_mask,
                               doc_ids=doc_ids)
        if skip_lm_head:
            # Memory-lean RL policy-update path: return the pre-lm_head hidden
            # (post out_norm + memory) so the caller applies lm_head ONLY at the
            # positions it needs (emit-prediction indices), avoiding the full
            # (B, T, V) logits tensor AND its backward graph — V≈49k ≫ d, so
            # that tensor dominates the policy-update activation memory and
            # grows with rollout length T. The gate side-effect still fires so
            # `_last_gate` is populated for the stochastic-gate policy gradient.
            maybe_gate(h)
            return h
        lm_logits = self.lm_head(h)
        outs = (lm_logits,)
        if return_aux and self.aux_dim > 0:
            outs = outs + (self.aux_head(h),)
        if return_hidden:
            hid = (h_premem if getattr(self, "_latent_feedback_premem", False)
                   else h)
            outs = outs + (hid,)
        if return_gate:
            outs = outs + (maybe_gate(h),)
        else:
            maybe_gate(h)
        outs = outs + tuple(extra)
        # Trunk gist loss — computed INSIDE the forward and returned as a
        # SCALAR. This is the torch.compile-safe wiring: handing the
        # hidden state `h` OUT of the compiled forward (via the return
        # tuple OR an attribute stash) trips AOTAutograd's output-alias
        # replay (garbage-shape RuntimeError in gen_alias_from_base).
        # A scalar reduction output cannot alias anything, so computing
        # the loss here and exposing only the scalar sidesteps the bug.
        # Gated on self.training so eval/validation forwards
        # (model.eval()) return plain logits with no extra output.
        if self.training and getattr(self, "_gist_loss_enabled", False):
            outs = outs + (trunk_gist_loss(h, self.gist_heads,
                                           self._gist_horizons),)
        return outs[0] if len(outs) == 1 else outs

    def forward(self, input_ids: torch.Tensor,
                return_aux: bool = False,
                return_hidden: bool = False,
                return_gate: bool = False,
                mem_read_mask: torch.Tensor | None = None,
                doc_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                skip_lm_head: bool = False,
                ) -> torch.Tensor | tuple:
        """
        inputs_embeds (B, T, d_model) | None: when provided, BYPASSES the
        embedding-table lookup for `input_ids`. The model uses these
        embeddings directly as the trunk input. `input_ids` MUST still
        be provided (used by WorkingMemory's think-position mask, by
        cu_seqlens construction, by max_T positional embed, and for
        loss-target alignment) but its embedding contribution is
        replaced. This is the entry-point for the retrieval-as-input
        thinking-token design (2026-05-19): at think positions, the
        caller substitutes the retrieved WM/PKM value as the input
        embedding so each think step gets a unique input signal
        (avoids the homogeneous-think-position pathology that motivated
        FIX A and then disproved it).
        """
        if inputs_embeds is not None:
            if inputs_embeds.shape[:2] != input_ids.shape[:2]:
                raise ValueError(
                    f"inputs_embeds shape {inputs_embeds.shape[:2]} must "
                    f"match input_ids shape {input_ids.shape[:2]}")
            x = inputs_embeds
        else:
            x = self.embed(input_ids)

        # Phase 3 (2026-05-26): per-position think-index embedding. Adds
        # a small learned vector to each think token based on its index
        # within the consecutive-think burst it belongs to, so a burst
        # of 8 thinks produces 8 distinct inputs instead of 8 identical
        # ones. Applied AFTER inputs_embeds so retrieval-as-input mode
        # also benefits (additive — diversifies the think input either
        # way the embedding was constructed).
        if (self.think_index_emb_size > 0
                and self.thinking_token_id is not None
                and input_ids is not None):
            x = x + self._compute_think_index_emb(input_ids)

        # Think-time op-selector (2026-06-03). Inject a position-addressable
        # verbatim program-line embedding as a zero-init-α additive side-channel
        # at think positions only. Uses the raw token embeddings (NOT x) for the
        # KV content so retrieval-as-input mode (x != embeds) still selects from
        # verbatim prompt content. Zero at non-think positions / cold start.
        if (getattr(self, "use_line_selector", False)
                and self.thinking_token_id is not None
                and input_ids is not None
                and self.newline_token_id is not None):
            embeds = self.embed(input_ids)
            burst_idx = self._compute_burst_index(input_ids)
            x = x + self.line_selector(
                x, input_ids, embeds, int(self.thinking_token_id),
                int(self.newline_token_id), burst_idx)

        # Cross-document isolation: when `doc_ids` marks multiple documents
        # packed into one T-length row, `cu_seqlens` makes the DeltaNet
        # recurrent kernel reset state at each document boundary. None →
        # plain batched path (each row already an independent sequence).
        cu_seqlens = _build_cu_seqlens(doc_ids)

        # State-readonly-at-think (Phase 2, 2026-05-26). Precompute a
        # boolean mask once per forward; threaded into Block → _FlaWrapper
        # so DeltaNet's β is forced to 0 at think positions. Only built
        # when both the flag is set AND we know the thinking_token_id.
        # The thread is a no-op on every other attention kind (Block
        # checks `accepts_think_mask`).
        # Phase B (2026-05-26): the think adapter ALSO needs the mask —
        # so we build it whenever EITHER mechanism is on.
        need_think_mask = (
            (self.state_readonly_at_think or self.use_think_adapter)
            and self.thinking_token_id is not None
            and input_ids is not None
        )
        if need_think_mask:
            think_mask = (input_ids == int(self.thinking_token_id))
        else:
            think_mask = None

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

        if ((self.feedback_mode == "none" or self._film_bypass)
                and not self.feedback_xattn_pairs):
            # Plain block loop: 1 forward, no FiLM passes. Reached when
            # feedback is genuinely off, OR when the trainer has set
            # _film_bypass during the K-self-feed warmup. FiLM params get
            # no gradient on bypassed steps — intentional (early grad is
            # noise; the K-self-feed tax isn't worth paying yet).
            for L, blk in enumerate(self.blocks):
                x = self._block_fwd(blk, x, input_ids, L=L,
                                    cu_seqlens=cu_seqlens,
                                    think_mask=think_mask)
            return self._finalize(x, input_ids, mem_read_mask,
                                   return_aux, return_hidden, return_gate,
                                   _maybe_gate, doc_ids=doc_ids,
                                   skip_lm_head=skip_lm_head)

        if self.feedback_xattn_pairs:
            # Pass 1: vanilla. Collect outputs at every layer that any target
            # references as a source (union across all targets).
            # We use _block_fwd (not raw blk) so activation checkpointing —
            # if enabled — applies to BOTH passes. Without checkpointing the
            # two-pass forward would double the activation peak and OOM at
            # batch+sequence-length combos that fit fine in the one-pass
            # baseline.
            #
            # IMPORTANT: pass-1 runs UNDER no_grad. The xattn forward is the
            # cross-layer analogue of FiLM K=3 self-feed, where K-1 warmup
            # passes are no_grad and only the final pass carries gradient.
            # Source-layer params still receive gradient via pass-2 (which
            # also runs the full block stack); pass-1's only function is to
            # *produce* source-layer hidden states for pass-2's xattn modules
            # to attend over. Making pass-1 no_grad saves ~25% per-step
            # (no autograd tape + no backward through pass-1) and brings
            # xattn's per-step cost in line with FiLM K=3. Validated against
            # the v5/v4 FiLM-K=3 protocol that proved no_grad warmup passes
            # don't hurt quality.
            needed_sources = set()
            for _, srcs in self.feedback_xattn_pairs:
                needed_sources.update(srcs)
            pass1_at_sources: dict = {}
            with torch.no_grad():
                h1 = x
                for L, blk in enumerate(self.blocks):
                    h1 = self._block_fwd(blk, h1, input_ids, L=L,
                                         cu_seqlens=cu_seqlens,
                                         think_mask=think_mask)
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
                h = self._block_fwd(blk, h, input_ids, L=L,
                                    cu_seqlens=cu_seqlens,
                                    think_mask=think_mask)
            return self._finalize(h, input_ids, mem_read_mask,
                                   return_aux, return_hidden, return_gate,
                                   _maybe_gate, doc_ids=doc_ids,
                                   skip_lm_head=skip_lm_head)
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
                        input_ids=input_ids, cu_seqlens=cu_seqlens,
                        think_mask=think_mask,
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
                            input_ids=input_ids, cu_seqlens=cu_seqlens,
                            think_mask=think_mask,
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
                    h = self._block_fwd(blk, h, input_ids, L=L,
                                    cu_seqlens=cu_seqlens,
                                    think_mask=think_mask)
                    if L in self.sparse_target_to_source and self.feedback_position == "post":
                        src = self.sparse_target_to_source[L]
                        sup = surprise_per_target.get(L) if surprise_per_target else None
                        h = self.sparse_feedback[str(L)](h, film_in[src], surprise=sup)
                    if L in source_layers:
                        final_src_states[L] = h
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
                return self._finalize(h, input_ids, mem_read_mask,
                                       return_aux, return_hidden, return_gate,
                                       _maybe_gate, doc_ids=doc_ids,
                                       skip_lm_head=skip_lm_head)
            # ----------------------------------------------------------------
            # Standard 2-pass sparse FiLM (existing path).
            # ----------------------------------------------------------------
            # Pass 1: vanilla, collect outputs only at source layers.
            pass1_at_sources: dict = {}
            h1 = x
            for L, blk in enumerate(self.blocks):
                h1 = blk(h1, input_ids=input_ids, cu_seqlens=cu_seqlens,
                         think_mask=think_mask)
                h1 = self._maybe_pkm(h1, L)
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
                h = self._block_fwd(blk, h, input_ids, L=L,
                                    cu_seqlens=cu_seqlens,
                                    think_mask=think_mask)
                if L in self.sparse_target_to_source and self.feedback_position == "post":
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_k(pass1_at_sources[src],
                                                            self.feedback_lag)
                    h = self.sparse_feedback[str(L)](h, state_above_lagged)
            return self._finalize(h, input_ids, mem_read_mask,
                                   return_aux, return_hidden, return_gate,
                                   _maybe_gate, doc_ids=doc_ids,
                                   skip_lm_head=skip_lm_head)
        # Pass 1: vanilla forward, collect each layer's output.
        pass1_outs: list[torch.Tensor] = []
        h = x
        for L, blk in enumerate(self.blocks):
            h = blk(h, input_ids=input_ids, think_mask=think_mask)
            h = self._maybe_pkm(h, L)
            pass1_outs.append(h)

        # Pass 2: forward with top-down feedback from pass-1 outputs,
        # shifted right by 1 along T. Multi-scale: each layer L sees
        # states from layers {L+d for d in feedback_distances}.
        h = x
        N = len(self.blocks)
        for L, blk in enumerate(self.blocks):
            if self.feedback_mode == "none":
                h = blk(h, input_ids=input_ids, cu_seqlens=cu_seqlens,
                        think_mask=think_mask)
                h = self._maybe_pkm(h, L)
                continue
            # Gather multi-scale lagged states for layer L.
            states_above: list = []
            for d in self.feedback_distances:
                src = L + d
                if src < N:
                    states_above.append(_shift_right_by_1(pass1_outs[src]))
                else:
                    states_above.append(None)
            h_input = self.feedback[L](h, states_above)
            h = blk(h_input, input_ids=input_ids, cu_seqlens=cu_seqlens,
                    think_mask=think_mask)
            h = self._maybe_pkm(h, L)

        return self._finalize(h, input_ids, mem_read_mask,
                               return_aux, return_hidden, return_gate,
                               _maybe_gate, skip_lm_head=skip_lm_head)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # State-passing incremental decoding (2026-05-23).
    #
    # `prefill(input_ids)` runs ONE full forward over the prompt and
    # extracts everything `forward_step` needs to continue the
    # generation one token at a time at constant per-token cost.
    #
    # The cache is a plain dict (call it `IncrementalCache` for clarity)
    # holding:
    #
    #   fla_cache: fla.models.utils.Cache
    #       Per-layer recurrent + ShortConv state. Each DeltaNet layer
    #       writes its state to slot `layer_idx` (we set `layer_idx`
    #       per-layer at prefill / step time via _FlaWrapper.forward_step).
    #
    #   seen: int
    #       Total tokens processed so far (= position of the NEXT token
    #       when forward_step is called).
    #
    #   lagged_sources: dict[int, Tensor (B, 1, d_model)] | None
    #       For FiLM (--feedback_pairs with K-self-feed): the previous
    #       step's source-layer outputs, used as FiLM input for THIS
    #       step's target layer (matches `decode_step_film` semantics —
    #       at convergence of K-self-feed, the pass-2 lagged source
    #       state is the right deploy-time input).
    #       None when _film_bypass=True (the default at decode for the
    #       v7 ckpt family, since generators set bypass=True).
    #
    #   wm_buf: dict with growing tensors
    #       'gate':  (B, t_cur)        per-position write-gate sigmoid
    #       'value': (B, t_cur, d_mem) per-position W_v(h)
    #       'pos':   (B, t_cur) int    position index (for causal mask
    #                                   in a future read)
    #       'doc':   (B, t_cur) long | None    doc_id (always None at
    #                                   inference)
    #       'tok':   (B, t_cur) long   input_ids (only used to recompute
    #                                   pad/think masks during a read)
    #       Build is "append every step"; reads happen lazily at the
    #       per-step `_apply_memory_incremental` call. For non-think
    #       emit steps the read is masked out so we skip the read entirely
    #       and just append to the buffer.
    #
    # Limitations / what's intentionally NOT supported in forward_step:
    #   - Cross-document doc_ids — None at inference; not threaded.
    #   - feedback_xattn (cross-layer attention) — v7 uses FiLM, not xattn;
    #     the deployed generators set _film_bypass=True so even FiLM is
    #     bypassed. xattn forward_step raises NotImplementedError.
    #   - K-self-feed at decode time: ALWAYS run K=1 (`_film_bypass=True`
    #     is the convention shipped in eval_humaneval / train_rl_grader;
    #     when bypass is True, no FiLM is applied either, mirroring the
    #     full forward's "plain block loop" branch).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor,
                inputs_embeds: torch.Tensor | None = None) -> dict:
        """Run the prompt through a full forward, build an incremental
        cache, and return (cache, last_logits).

        The prompt's logits at the LAST position are also returned (the
        generator typically samples the first emit from these without
        ever calling forward_step on the first new token).

        `inputs_embeds` is forwarded to support the retrieval-as-input
        generator (which substitutes the embedding at the new position).
        For prefill the prompt is plain input_ids and inputs_embeds is
        usually None.

        Side effect (preserves the existing public interface): the
        usual `self._last_gate*` stashes are populated by the full
        forward, so callers that read them keep working unchanged.
        """
        from fla.models.utils import Cache as FLACache

        B, T_prompt = input_ids.shape

        # Use a plain bypass forward over the prompt for state extraction.
        # We need to populate (a) per-layer FLA cache, (b) FiLM lagged
        # source state if feedback is active and NOT bypassed, (c) WM
        # buffer entries for the entire prompt.
        #
        # The cleanest extraction: mirror the "_film_bypass / plain
        # block loop" branch of forward(), but pass `past_key_values`
        # into each block's attention so it caches the recurrent state.
        if inputs_embeds is not None:
            if inputs_embeds.shape[:2] != input_ids.shape[:2]:
                raise ValueError(
                    f"inputs_embeds shape {inputs_embeds.shape[:2]} must "
                    f"match input_ids shape {input_ids.shape[:2]}")
            x = inputs_embeds
        else:
            x = self.embed(input_ids)

        # Phase 3 think-index embedding (mirror of full forward).
        if (self.think_index_emb_size > 0
                and self.thinking_token_id is not None):
            x = x + self._compute_think_index_emb(input_ids)

        # Phase B think_mask — fires the ThinkAdapter at think positions
        # during prefill (mirror of Block.forward path). Also drives the
        # Phase-2 state-readonly β-masking on the inner DeltaNet b_proj
        # (2026-05-28: previously only `use_think_adapter` built this mask,
        # so the prefill chunk-kernel path left β UNMASKED at think
        # positions — the decode-path bug GEMINI flagged). Built whenever
        # either consumer is active and we know the thinking token id.
        prefill_think_mask = None
        if ((self.use_think_adapter or self.state_readonly_at_think)
                and self.thinking_token_id is not None):
            prefill_think_mask = (input_ids == int(self.thinking_token_id))

        if self.max_T > 0:
            pos = torch.arange(T_prompt, device=input_ids.device)
            x = x + self.pos_embed(pos)

        fla_cache = FLACache(seen_tokens=0)
        # We RUN the full plain-bypass block stack with use_cache=True
        # so each attention layer populates fla_cache[layer_idx].
        lagged_sources: dict[int, torch.Tensor] = {}
        source_layers = set(s for _, s in self.feedback_pairs) \
            if (self.feedback_pairs and not self._film_bypass) else set()

        # If FiLM is bypassed (the standard deploy convention), we just
        # run plain blocks. Otherwise (rare in practice for our v7
        # generators), we run the K=1 lagged-cached FiLM at prefill so
        # the FIRST forward step has correct lagged inputs.
        use_film_at_decode = (self.feedback_pairs
                              and not self._film_bypass
                              and not self.feedback_xattn_pairs)

        h = x
        if use_film_at_decode:
            # Pass 1: vanilla, collect source layer outputs (no cache —
            # pass-1 state isn't used downstream, only the source-layer
            # outputs matter for the FiLM lag).
            h1 = x
            pass1_at_sources: dict = {}
            for L, blk in enumerate(self.blocks):
                h1 = self._step_block(blk, h1, past=None, layer_idx=L,
                                     think_mask=prefill_think_mask)
                h1 = self._maybe_pkm(h1, L)
                if L in source_layers:
                    pass1_at_sources[L] = h1
            # Pass 2: FiLM at targets, populate the real cache.
            for L, blk in enumerate(self.blocks):
                if L in self.sparse_target_to_source and self.feedback_position == "pre":
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_k(
                        pass1_at_sources[src], self.feedback_lag)
                    h = self.sparse_feedback[str(L)](h, state_above_lagged)
                h = self._step_block(blk, h, past=fla_cache, layer_idx=L,
                                    think_mask=prefill_think_mask)
                if L in self.sparse_target_to_source and self.feedback_position == "post":
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_k(
                        pass1_at_sources[src], self.feedback_lag)
                    h = self.sparse_feedback[str(L)](h, state_above_lagged)
                h = self._maybe_pkm(h, L)
            # Lagged source for the NEXT decode step: the LAST position of
            # pass-2's source-layer output (since pass-2 is what training
            # actually computed; pass-1 only feeds FiLM). Match
            # decode_step_film semantics.
            #
            # Subtle: decode_step_film actually caches PASS-2's source
            # output for the NEXT step's FiLM input (it's a proxy for
            # the true pass-1 lagged input). We mirror that.
            for L in source_layers:
                lagged_sources[L] = h.new_zeros(B, 1, h.shape[-1])
            # Walk pass-2 again? No — `pass1_at_sources` already has them
            # to lag from. But for the NEXT decode step we want pass-2's
            # output at the last prompt position. Recompute by saving
            # pass-2 source outputs:
            # → do it in-line above. Repeating that here cleanly:
            pass2_at_sources: dict = {}
            h_p2 = x
            for L, blk in enumerate(self.blocks):
                if L in self.sparse_target_to_source and self.feedback_position == "pre":
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_k(
                        pass1_at_sources[src], self.feedback_lag)
                    h_p2 = self.sparse_feedback[str(L)](h_p2, state_above_lagged)
                # NOTE: this is a wasteful 2nd full forward; for now this
                # branch is mainly correctness fallback. Production usage
                # has _film_bypass=True so we never hit this code.
                h_p2 = blk(h_p2, input_ids=input_ids)
                if L in self.sparse_target_to_source and self.feedback_position == "post":
                    src = self.sparse_target_to_source[L]
                    state_above_lagged = _shift_right_by_k(
                        pass1_at_sources[src], self.feedback_lag)
                    h_p2 = self.sparse_feedback[str(L)](h_p2, state_above_lagged)
                h_p2 = self._maybe_pkm(h_p2, L)
                if L in source_layers:
                    lagged_sources[L] = h_p2[:, -1:].clone()
        else:
            # Plain block loop (= _film_bypass branch). The single,
            # primary code path for the v7 generator.
            for L, blk in enumerate(self.blocks):
                h = self._step_block(blk, h, past=fla_cache, layer_idx=L,
                                    think_mask=prefill_think_mask)
                h = self._maybe_pkm(h, L)

        # Phase D — apply refinement head soft-mixture before out_norm
        # (mirror _finalize). No-op when use_refinement_head is False.
        h = self._apply_refinement_head(h)
        # Gate + memory + lm_head (mirror _finalize).
        h_normed = self.out_norm(h)
        # Build the WM buffer entries for the prompt BEFORE applying
        # memory injection — the buffer is computed from h_normed (same
        # as WorkingMemory.forward does internally).
        wm_buf = self._wm_init_buffer_from_prompt(h_normed, input_ids) \
            if self.use_memory else None

        # Memory injection for the prompt itself — needed so the prompt's
        # last-position logits match the full forward exactly (used by
        # the generator's "first sample comes from prefill" path).
        h_normed = self._apply_memory(h_normed, input_ids, read_mask=None,
                                       doc_ids=None)
        lm_logits = self.lm_head(h_normed)
        # Gate stash (matches _finalize / _maybe_gate behaviour).
        if self.output_gate:
            gate_logits = self.gate_head(h_normed).squeeze(-1)
            g = torch.sigmoid(gate_logits)
            self._last_gate_logits = gate_logits
            self._last_gate = g

        # Phase 3: running consecutive-think count ending at the last
        # processed position, per row. forward_step needs this to assign
        # the right think-index embedding to a new think token without
        # re-seeing the prompt.
        if self.think_index_emb_size > 0 and self.thinking_token_id is not None:
            tm_int = (input_ids == int(self.thinking_token_id)).to(torch.int64)
            c = tm_int.cumsum(dim=1)
            reset = (c * (1 - tm_int)).cummax(dim=1).values
            burst_idx = (c - reset - 1).clamp(min=0)  # (B, T)
            # Run-length at last position: 0 if last token not a think,
            # else burst_idx[-1] + 1.
            last_run = (burst_idx[:, -1] + 1) * tm_int[:, -1]  # (B,)
        else:
            last_run = None

        cache = {
            "fla_cache": fla_cache,
            "seen": int(T_prompt),
            "lagged_sources": lagged_sources if use_film_at_decode else None,
            "wm_buf": wm_buf,
            "think_run_len": last_run,
        }
        return cache, lm_logits

    def _step_block(self, blk: nn.Module, x: torch.Tensor,
                    past, layer_idx: int,
                    think_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run one Block, threading `past` through the attention layer's
        cache. The attention layer is `blk.attn` (a `_FlaWrapper`); we
        use its `.layer` directly so the kernel can write into the
        cache. This matches `_block_with_cache` in decode_bench.py.

        Caller decides whether `past` is from a prefill (full T) or a
        single-token forward_step (T=1). For the FLA DeltaNet family,
        the chunk kernel handles full sequences with use_cache=True,
        AND the fused_recurrent kernel handles T=1 — we pick which by
        x.shape[1].

        If `think_mask` is provided AND the block has a ThinkAdapter
        (Phase B), the adapter is applied after the MLP residual — so
        the adapter ALSO fires during incremental decode, not just
        during the full-sequence training forward. Without this thread,
        a ckpt trained with the adapter would have the adapter silently
        inert at inference (caught by the v5 code review).
        """
        attn_in = blk.attn_norm(x)
        # State-readonly-at-think (Phase 2) in the decode path (2026-05-28).
        # The b_proj forward-hook reads `_current_think_mask` off the
        # `_FlaWrapper`; the full-sequence `forward` sets it, but the decode
        # entry points (forward_step / chunk-with-cache) bypass `forward`
        # entirely. Stash it here around the call so β is forced to 0 at
        # think positions in BOTH decode sub-paths, then clear it. No-op
        # when the wrapper isn't state-readonly (`_current_think_mask` stays
        # None and the hook returns logits unchanged).
        attn = blk.attn
        sr_active = (think_mask is not None
                     and getattr(attn, "state_readonly_at_think", False))
        if sr_active:
            attn._current_think_mask = think_mask
        try:
            # T==1 → use forward_step (fused_recurrent kernel; correct for
            #         incremental decoding with a populated cache).
            # T>1  → run the full chunk kernel with use_cache=True so the
            #         cache slot is initialized from the whole prompt.
            if x.shape[1] == 1 and past is not None:
                attn_out, _ = blk.attn.forward_step(attn_in, past, layer_idx)
            else:
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                    layer = blk.attn.layer
                    saved_layer_idx = getattr(layer, "layer_idx", None)
                    layer.layer_idx = int(layer_idx)
                    try:
                        out = layer(
                            hidden_states=attn_in,
                            past_key_values=past,
                            use_cache=(past is not None),
                        )
                    finally:
                        if saved_layer_idx is not None:
                            layer.layer_idx = saved_layer_idx
                if isinstance(out, tuple):
                    attn_out = out[0]
                else:
                    attn_out = out
                attn_out = attn_out.to(x.dtype)
        finally:
            if sr_active:
                attn._current_think_mask = None
        x = x + attn_out
        x = x + blk.mlp(blk.mlp_norm(x))
        # Phase B: apply ThinkAdapter at think positions, mirroring the
        # Block.forward path. Silent no-op when adapter absent or mask
        # not provided.
        if (think_mask is not None
                and getattr(blk, "use_think_adapter", False)):
            x = x + blk.think_adapter(x, think_mask)
        return x

    def _wm_init_buffer_from_prompt(self, h_normed: torch.Tensor,
                                     input_ids: torch.Tensor) -> dict:
        """Compute the per-position WM write contributions for the
        prompt and stash them in the growing-buffer dict.

        We don't trim to top-K here — the buffer is the FULL list of
        (gate, value, pos) for every position seen so far. The
        `forward_step` read path takes top-K from this buffer at READ
        time (same semantics as WorkingMemory.forward, just with the
        topk applied lazily over the growing list).
        """
        mem = self.memory
        B, T, _ = h_normed.shape
        device = h_normed.device

        write_logits = mem.W_write(h_normed).squeeze(-1)   # (B, T)
        g = torch.sigmoid(write_logits)                     # (B, T)
        v = mem.W_v(h_normed)                               # (B, T, d_mem)

        if mem.pad_token_id is not None:
            is_pad = input_ids == int(mem.pad_token_id)
            g = g.masked_fill(is_pad, 0.0)

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T).contiguous()
        return {
            "gate": g.detach(),       # (B, T)
            "value": v.detach(),      # (B, T, d_mem)
            "pos": pos,                # (B, T)
            "tok": input_ids.clone(),  # (B, T)
        }

    def _wm_append_one(self, buf: dict, h_normed_new: torch.Tensor,
                       input_ids_new: torch.Tensor) -> None:
        """Append ONE new position's (gate, value, pos, tok) to the
        growing WM buffer in-place.

        h_normed_new: (B, 1, d_model)
        input_ids_new: (B, 1)
        """
        mem = self.memory
        write_logit = mem.W_write(h_normed_new).squeeze(-1)   # (B, 1)
        g_new = torch.sigmoid(write_logit)
        v_new = mem.W_v(h_normed_new)                          # (B, 1, d_mem)

        if mem.pad_token_id is not None:
            is_pad = input_ids_new == int(mem.pad_token_id)
            g_new = g_new.masked_fill(is_pad, 0.0)

        B = h_normed_new.shape[0]
        cur_T = int(buf["gate"].shape[1])
        new_pos = torch.full((B, 1), cur_T, dtype=buf["pos"].dtype,
                              device=h_normed_new.device)

        buf["gate"]  = torch.cat([buf["gate"],  g_new.detach()],   dim=1)
        buf["value"] = torch.cat([buf["value"], v_new.detach()],   dim=1)
        buf["pos"]   = torch.cat([buf["pos"],   new_pos],          dim=1)
        buf["tok"]   = torch.cat([buf["tok"],   input_ids_new],    dim=1)

    def _wm_read_one(self, buf: dict, h_normed_new: torch.Tensor,
                     read_mask_new: torch.Tensor | None,
                     input_ids_new: torch.Tensor) -> torch.Tensor:
        """Compute the WM injection at the NEW (single) position.

        Returns (B, 1, d_model) — the per-position post-memory hidden
        state (residual already added).

        Semantics match WorkingMemory.forward but for a single read
        position: topk over the entire current buffer's write-gates,
        then soft-attention with causal+pad masking.

        Cheap-skip: if the read is masked out (the default-think-mask
        case: input_ids_new != thinking_token_id, AND read_mask_new is
        None or 0), we return h_normed_new unchanged — no read compute.
        """
        import math
        mem = self.memory
        B, _, d_model = h_normed_new.shape
        device = h_normed_new.device

        # Decide read activation mask FIRST.
        # NOTE: we always compute the injection unconditionally — the
        # retrieval-as-input generator reads `mem._last_injection` at
        # every emit step (not just think positions) to feed back as
        # the next think-token's input embedding. The original
        # `WorkingMemory.forward` also stashes the injection PRE-mask,
        # so we must too. The mask only gates whether the injection is
        # ADDED back to h.
        if read_mask_new is None:
            inj_mask = (input_ids_new == mem.thinking_token_id).to(h_normed_new.dtype)
        else:
            inj_mask = read_mask_new.to(h_normed_new.dtype)

        gate_full  = buf["gate"]                    # (B, t_cur)
        value_full = buf["value"]                   # (B, t_cur, d_mem)
        pos_full   = buf["pos"]                     # (B, t_cur)
        t_cur = gate_full.shape[1]
        K_eff = min(t_cur, mem.mem_size)

        _, top_idx = torch.topk(gate_full, k=K_eff, dim=-1)    # (B, K)
        gather_idx_v = top_idx.unsqueeze(-1).expand(-1, -1, mem.d_mem)
        buf_v = torch.gather(value_full, dim=1, index=gather_idx_v)  # (B, K, d_mem)
        buf_g = torch.gather(gate_full, dim=1, index=top_idx)        # (B, K)
        buf_pos = torch.gather(pos_full, dim=1, index=top_idx)       # (B, K)

        # Query at the single new position.
        q = mem.W_q(h_normed_new)                                    # (B, 1, d_mem)
        scale = 1.0 / math.sqrt(mem.d_mem)
        scores = torch.einsum("btd,bkd->btk", q, buf_v) * scale       # (B, 1, K)
        scores = scores + torch.log(buf_g.clamp_min(1e-6)).unsqueeze(1)

        # Causal mask: the new position is at index t_cur (we've ALREADY
        # appended this token to the buffer before reading — see
        # forward_step ordering). So allow buffer slot k iff
        # buf_pos[k] < t_cur (strict). Equivalently: top_idx[k] != the
        # new token's position index.
        new_pos_val = t_cur - 1   # we appended first; the read happens
                                   # AFTER append, so the new token is at
                                   # index t_cur - 1.
        src_pos = buf_pos.unsqueeze(1)                                # (B, 1, K)
        causal_mask = src_pos >= new_pos_val                          # (B, 1, K)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        all_masked = causal_mask.all(dim=-1, keepdim=True)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)
        read = torch.einsum("btk,bkd->btd", attn, buf_v)              # (B, 1, d_mem)
        injection = mem.W_proj(read)                                   # (B, 1, d_model)
        mem._last_injection = injection.detach()

        # Apply the read-injection α gate (matches WorkingMemory.forward) so
        # incremental decode reproduces the full-forward magnitude exactly.
        return h_normed_new + mem.read_alpha * injection * inj_mask.unsqueeze(-1)

    @torch.no_grad()
    def forward_step(self, input_id: torch.Tensor, cache: dict, *,
                     return_hidden: bool = False,
                     inputs_embeds: torch.Tensor | None = None,
                     mem_read_mask: torch.Tensor | None = None,
                     ) -> tuple:
        """Incremental one-token forward.

        input_id: (B, 1) — the next token to process. (For inputs_embeds
            substitution at think positions, pass `inputs_embeds=...` and
            input_id is still used for the WM think-position mask.)

        cache: the dict returned by `prefill(...)` (or a previous
            `forward_step` call). MUTATED in place.

        Returns `(logits, cache)` where `logits` is (B, 1, V) — the
        next-token distribution for the position just processed.

        Honors the same `self._last_gate / _last_gate_logits` stashes as
        the full forward so callers that read them keep working.

        Limitations: see the IncrementalCache docstring above.
        """
        if input_id.dim() == 1:
            input_id = input_id.unsqueeze(-1)
        assert input_id.shape[1] == 1, \
            f"forward_step expects (B, 1) input, got {tuple(input_id.shape)}"
        B = input_id.shape[0]
        device = input_id.device

        if self.feedback_xattn_pairs:
            raise NotImplementedError(
                "forward_step does not support cross-layer-attention "
                "feedback (use the full-forward fallback)")

        # Embedding (or substitute).
        if inputs_embeds is not None:
            assert inputs_embeds.shape[:2] == (B, 1), \
                f"inputs_embeds must be (B, 1, d); got {tuple(inputs_embeds.shape)}"
            x = inputs_embeds
        else:
            x = self.embed(input_id)

        # Phase 3 think-index embedding: maintain a per-row running
        # consecutive-think counter across forward_step calls in
        # cache["think_run_len"]. At a new think token, the index is
        # the prior run length (clamped to table size); at a non-think
        # token the counter resets to 0.
        if (self.think_index_emb_size > 0
                and self.thinking_token_id is not None):
            tm = (input_id.squeeze(-1) == int(self.thinking_token_id))  # (B,)
            prior_run = cache.get("think_run_len")
            if prior_run is None:
                prior_run = torch.zeros(B, dtype=torch.int64, device=device)
            idx = prior_run.clamp(max=self.think_index_emb_size - 1)
            idx_emb = self.think_index_emb(idx).unsqueeze(1)  # (B, 1, d)
            x = x + idx_emb * tm.view(B, 1, 1).to(idx_emb.dtype)
            # Update running counter: increment if think else reset to 0.
            cache["think_run_len"] = torch.where(
                tm, prior_run + 1, torch.zeros_like(prior_run))

        if self.max_T > 0:
            # Per-row absolute position. When `cache["seen_per_row"]` is set
            # (cross-problem batched decode: rows have DIFFERENT prompt
            # lengths, so a single scalar `seen` would give every row the
            # wrong positional embedding), index pos_embed per row. The
            # legacy scalar `cache["seen"]` path is unchanged (all rows share
            # one prompt length, e.g. `rollout_group_batched`).
            seen_per_row = cache.get("seen_per_row")
            if seen_per_row is not None:
                pos = seen_per_row.to(device=device, dtype=torch.long)  # (B,)
                x = x + self.pos_embed(pos).unsqueeze(1)                # (B,1,d)
            else:
                pos = torch.tensor([cache["seen"]], device=device,
                                   dtype=torch.long)
                x = x + self.pos_embed(pos)

        fla_cache = cache["fla_cache"]
        use_film_at_decode = (self.feedback_pairs
                              and not self._film_bypass
                              and not self.feedback_xattn_pairs
                              and cache.get("lagged_sources") is not None)
        source_layers = set(s for _, s in self.feedback_pairs) \
            if use_film_at_decode else set()

        # Phase B think_mask — fires the ThinkAdapter at think positions
        # during forward_step (mirror of Block.forward path). Also drives
        # the Phase-2 state-readonly β-masking on the inner DeltaNet b_proj
        # in the T=1 fused_recurrent decode path (2026-05-28 fix: this path
        # previously left β unmasked at think positions, so an incremental
        # decoder corrupted long-range bindings exactly like the broken
        # full-forward path used to before Phase 2).
        step_think_mask = None
        if ((self.use_think_adapter or self.state_readonly_at_think)
                and self.thinking_token_id is not None):
            step_think_mask = (input_id == int(self.thinking_token_id))

        if use_film_at_decode:
            h = x
            new_lagged: dict = {}
            for L, blk in enumerate(self.blocks):
                if L in self.sparse_target_to_source and self.feedback_position == "pre":
                    src = self.sparse_target_to_source[L]
                    h = self.sparse_feedback[str(L)](
                        h, cache["lagged_sources"][src])
                h = self._step_block(blk, h, past=fla_cache, layer_idx=L,
                                    think_mask=step_think_mask)
                if L in self.sparse_target_to_source and self.feedback_position == "post":
                    src = self.sparse_target_to_source[L]
                    h = self.sparse_feedback[str(L)](
                        h, cache["lagged_sources"][src])
                h = self._maybe_pkm(h, L)
                if L in source_layers:
                    new_lagged[L] = h.clone()
            cache["lagged_sources"] = new_lagged
        else:
            h = x
            for L, blk in enumerate(self.blocks):
                h = self._step_block(blk, h, past=fla_cache, layer_idx=L,
                                    think_mask=step_think_mask)
                h = self._maybe_pkm(h, L)

        # Phase D — apply refinement head soft-mixture (mirror _finalize).
        # No-op when use_refinement_head is False. NOTE: at T=1 the local
        # attention only sees its own token; effectively becomes the MLP
        # branch only. For real benefit during incremental decode the
        # head would need a KV cache over recent positions — TODO.
        h = self._apply_refinement_head(h)
        # Gate + memory + lm_head (mirror _finalize).
        h_normed = self.out_norm(h)
        if self.use_memory:
            # Append THIS token to the WM buffer first, then read (the
            # read mask is causal anyway — strict < t_cur).
            self._wm_append_one(cache["wm_buf"], h_normed, input_id)
            h_normed = self._wm_read_one(cache["wm_buf"], h_normed,
                                          mem_read_mask, input_id)

        lm_logits = self.lm_head(h_normed)

        if self.output_gate:
            gate_logits = self.gate_head(h_normed).squeeze(-1)
            g = torch.sigmoid(gate_logits)
            self._last_gate_logits = gate_logits
            self._last_gate = g

        cache["seen"] = int(cache["seen"]) + 1
        if cache.get("seen_per_row") is not None:
            cache["seen_per_row"] = cache["seen_per_row"] + 1

        if return_hidden:
            return lm_logits, h_normed, cache
        return lm_logits, cache

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
