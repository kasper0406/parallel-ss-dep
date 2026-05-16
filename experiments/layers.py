"""
Attention modules for the kill-gate experiment.

Two architectures, identical residual-stream API:

    LinearAttention(d_model, n_heads, d_head)
    HeisenbergAttention(d_model, n_heads, d_head)

Both consume (B, T, d_model) and produce (B, T, d_model) via:
  - learned per-head projections from the residual stream,
  - a vectorized cumsum-based scan (autograd-friendly PyTorch ops),
  - per-head output projection back to d_model.

For kill-gate scale (T≤512), PyTorch ops are fast enough and we get
autograd for free. We can swap in the Triton kernel + a custom backward
later once an architecture clears the gate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, n_heads * d_head) -> (B, n_heads, T, d_head)."""
    B, T, _ = x.shape
    return x.view(B, T, n_heads, -1).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, n_heads, T, d_head) -> (B, T, n_heads * d_head)."""
    B, H, T, D = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)


def linear_attn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal linear attention via cumsum.

    o_t = q_t · S_t  where  S_t = Σ_{i≤t} k_i ⊗ v_i.

    q, k, v: (B, H, T, D). Output: (B, H, T, D).
    Materializes (B, H, T, D, D) — fine at kill-gate scale.
    """
    kv = k.unsqueeze(-1) * v.unsqueeze(-2)             # (B, H, T, D, D)
    S = torch.cumsum(kv, dim=-3)                        # (B, H, T, D, D)
    o = torch.einsum("bhtd,bhtde->bhte", q, S)
    return o


def heisenberg_attn_forward(q: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Causal Heisenberg attention via cumsum.

    o_t = q_t · c_t  where  c_t = Σ_{i<j≤t} a_i ⊗ b_j  (strict i<j).

    q, a, b: (B, H, T, D). Output: (B, H, T, D).
    """
    a_excl = torch.cumsum(a, dim=-2) - a                # (B, H, T, D), exclusive prefix
    rank1 = a_excl.unsqueeze(-1) * b.unsqueeze(-2)      # (B, H, T, D, D)
    c = torch.cumsum(rank1, dim=-3)                     # (B, H, T, D, D)
    o = torch.einsum("bhtd,bhtde->bhte", q, c)
    return o


class LinearAttention(nn.Module):
    """Causal linear attention: o_t = q_t · Σ_{i≤t} k_i ⊗ v_i."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.qkv_dim = n_heads * d_head
        self.W_q = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_o = nn.Linear(self.qkv_dim, d_model, bias=False)
        # Tame state magnitude — pure linear attention with no
        # normalization explodes after a few hundred steps.
        self.q_norm = nn.LayerNorm(d_head)
        self.k_norm = nn.LayerNorm(d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.W_q(x), self.n_heads)
        k = _split_heads(self.W_k(x), self.n_heads)
        v = _split_heads(self.W_v(x), self.n_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        o = linear_attn_forward(q, k, v)
        return self.W_o(_merge_heads(o))


class HeisenbergAttention(nn.Module):
    """Heisenberg attention: o_t = q_t · Σ_{i<j≤t} a_i ⊗ b_j."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.qab_dim = n_heads * d_head
        self.W_q = nn.Linear(d_model, self.qab_dim, bias=False)
        self.W_a = nn.Linear(d_model, self.qab_dim, bias=False)
        self.W_b = nn.Linear(d_model, self.qab_dim, bias=False)
        self.W_o = nn.Linear(self.qab_dim, d_model, bias=False)
        self.q_norm = nn.LayerNorm(d_head)
        self.a_norm = nn.LayerNorm(d_head)
        self.b_norm = nn.LayerNorm(d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.W_q(x), self.n_heads)
        a = _split_heads(self.W_a(x), self.n_heads)
        b = _split_heads(self.W_b(x), self.n_heads)
        q = self.q_norm(q)
        a = self.a_norm(a)
        b = self.b_norm(b)
        o = heisenberg_attn_forward(q, a, b)
        return self.W_o(_merge_heads(o))


class SoftmaxAttention(nn.Module):
    """Causal softmax attention — the ceiling baseline."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.qkv_dim = n_heads * d_head
        self.W_q = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_o = nn.Linear(self.qkv_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.W_q(x), self.n_heads)
        k = _split_heads(self.W_k(x), self.n_heads)
        v = _split_heads(self.W_v(x), self.n_heads)
        # SDPA expects (B, H, T, D), causal=True for triangular mask.
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True,
        )
        return self.W_o(_merge_heads(o))


# ---------------------------------------------------------------------------
# fla wrappers — flash-linear-attention provides full layer modules that take
# (B, T, d_model) → (B, T, d_model). We just forward to them and shape the
# return into our (B, T, d_model) contract.
# ---------------------------------------------------------------------------


class _FlaWrapper(nn.Module):
    """Minimal wrapper: build the fla layer in __init__, drop the tuple
    structure on the forward output.

    fla's chunked kernels (chunk_delta_rule, etc.) assert non-fp32 inputs.
    We cast hidden_states to bf16 on entry and back to the caller's dtype
    on exit so the rest of the (fp32) model is unaffected.

    `accepts_cu_seqlens`: FLA's recurrent kernels take a `cu_seqlens` ragged
    index for packed variable-length sequences (cross-document isolation).
    When `Block.forward` passes one, we flatten (B, T, d) -> (1, B*T, d) so
    the kernel sees a single ragged batch, then reshape back. Default False
    (conservative — an unverified fla layer could choke on the kwarg);
    subclasses whose underlying fla layer is verified to accept `cu_seqlens`
    opt in by setting this True.
    """

    accepts_cu_seqlens = False

    def __init__(self, fla_layer: nn.Module):
        super().__init__()
        self.layer = fla_layer

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None
                ) -> torch.Tensor:
        # fla's chunked kernels assert non-fp32 inputs. The trainer wraps
        # model.forward in bf16 autocast via apply_speed_knobs, but eval
        # callers (eval_humaneval.py, diag_ckpt.py, ...) don't — we keep
        # the explicit autocast here as a safety net so any caller works.
        in_dtype = x.dtype
        if cu_seqlens is not None and self.accepts_cu_seqlens:
            B, T, d = x.shape
            x_flat = x.reshape(1, B * T, d)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                out = self.layer(x_flat, cu_seqlens=cu_seqlens)
            if isinstance(out, tuple):
                out = out[0]
            return out.reshape(B, T, d).to(in_dtype)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            out = self.layer(x)
        if isinstance(out, tuple):
            out = out[0]
        return out.to(in_dtype)

    def forward_step(self, x: torch.Tensor, past_key_values, layer_idx: int):
        """Incremental-decoding forward for one timestep.

        `x`: (B, 1, d_model) — typically one new token's hidden state.
        `past_key_values`: an `fla.models.utils.Cache` (or None for the
            first step). Provides this layer's recurrent state from the
            previous step; the FLA kernel reads it via `layer_idx`.
        `layer_idx`: which entry of `past_key_values` this wrapper's
            inner FLA layer owns. Must be unique per attention layer in
            the model.

        Returns `(out, past_key_values)` — `out` is (B, 1, d_model);
        `past_key_values` is the updated cache (same object, mutated
        in-place by FLA).

        Implementation notes:
        - FLA's DeltaNet (and family) reads `self.layer.layer_idx` and
          `self.layer.mode` at forward time. We monkey-patch both for
          the duration of this call so the SAME instance can serve both
          full-sequence training (`mode="chunk"`) and single-step
          incremental decoding (`mode="fused_recurrent"`). The original
          mode is restored on the way out so subsequent training-path
          forwards behave identically.
        - The cache is bit-identical to having run the chunked kernel
          on the full sequence so far (verified in
          `experiments/test_incremental_decode.py`).
        """
        # Lazy-initialise a Cache the first time we're called.
        if past_key_values is None:
            from fla.models.utils import Cache
            past_key_values = Cache()
        in_dtype = x.dtype
        layer = self.layer
        saved_mode = getattr(layer, "mode", None)
        saved_layer_idx = getattr(layer, "layer_idx", None)
        try:
            layer.layer_idx = int(layer_idx)
            # Switch this instance to the incremental kernel for the call.
            if saved_mode is not None and saved_mode != "fused_recurrent":
                layer.mode = "fused_recurrent"
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                out = layer(x, past_key_values=past_key_values, use_cache=True)
        finally:
            # Restore — never leave the model in a non-training-friendly state.
            if saved_mode is not None:
                layer.mode = saved_mode
            if saved_layer_idx is None:
                # Best effort: leave it set; FLA tolerates a stale layer_idx
                # as long as we keep calling with the same Cache.
                pass
            else:
                layer.layer_idx = saved_layer_idx
        # FLA's DeltaNet returns (hidden, attentions, past_key_values).
        if isinstance(out, tuple):
            new_pkvs = out[-1] if len(out) >= 3 else past_key_values
            out_t = out[0]
        else:
            new_pkvs = past_key_values
            out_t = out
        return out_t.to(in_dtype), new_pkvs


class DeltaNetAttention(_FlaWrapper):
    """fla DeltaNet — same KV-state size as our linear_attn, plus delta updates."""

    # fla's DeltaNet.forward reads cu_seqlens from kwargs and threads it into
    # chunk_delta_rule + all three ShortConvolutions (fwd + bwd). Verified in
    # the local fork — see CROSS_DOC_ISOLATION_PLAN.md.
    accepts_cu_seqlens = True

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        from fla.layers import DeltaNet
        assert d_model == n_heads * d_head, \
            f"DeltaNet expects d_model == n_heads*d_head; got {d_model} vs {n_heads*d_head}"
        super().__init__(DeltaNet(
            mode="chunk",
            d_model=d_model,
            hidden_size=d_model,
            num_heads=n_heads,
            expand_k=1.0,
            expand_v=1.0,
            use_beta=True,
            use_gate=False,
            use_short_conv=True,
            qk_activation="silu",
            qk_norm="l2",
            allow_neg_eigval=False,
        ))


class DeltaNetNegEigAttention(_FlaWrapper):
    """fla DeltaNet with `allow_neg_eigval=True` — Grazzi-clean variant.

    Rescales β so the Householder reflector `(I − β k kᵀ)` can have eigenvalue
    in (−1, 1) along k. Per Grazzi et al. ICLR'25, this lifts DeltaNet from
    chance to 100% on parity at T ≤ 256. The strongest published single-cell
    baseline against our hybrid.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        from fla.layers import DeltaNet
        assert d_model == n_heads * d_head, \
            f"DeltaNet expects d_model == n_heads*d_head; got {d_model} vs {n_heads*d_head}"
        super().__init__(DeltaNet(
            mode="chunk",
            d_model=d_model,
            hidden_size=d_model,
            num_heads=n_heads,
            expand_k=1.0,
            expand_v=1.0,
            use_beta=True,
            use_gate=False,
            use_short_conv=True,
            qk_activation="silu",
            qk_norm="l2",
            allow_neg_eigval=True,
        ))


class GatedDeltaNetAttention(_FlaWrapper):
    """fla GatedDeltaNet — DeltaNet + RetNet-style scalar gate."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        from fla.layers import GatedDeltaNet
        assert d_model == n_heads * d_head, \
            f"GatedDeltaNet expects d_model == n_heads*d_head; got {d_model} vs {n_heads*d_head}"
        super().__init__(GatedDeltaNet(
            hidden_size=d_model,
            expand_v=1.0,
            head_dim=d_head,
            num_heads=n_heads,
            num_v_heads=n_heads,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
        ))


class DeltaNetForgetGateAttention(_FlaWrapper):
    """Forget-gate-only DeltaNet (Phase 21b H2 test).

    Plain `DeltaNet` plus a per-token learnable forget gate (via fla's
    `chunk_gated_delta_rule` kernel: `g = a_proj(x)` modulated through
    per-head `A_log` and `dt_bias`), but **no output gate**. This is the
    cleanest isolation of the forget-gate mechanism vs plain DN —
    instantiated as `GatedDeltaNet(use_gate=False, allow_neg_eigval=False)`.

    Used to test H2 (forget-gate redundancy): if adding the forget gate
    to plain DN reduces the sparse-FiLM lift toward GDP's −1.9 % range
    (vs plain DN's −3.1 %), the cross-cell pattern in Phase 17 is driven
    by forget-gate redundancy with FiLM.

    Note: applies the sm_120 TileLang smem-alignment workaround on import
    (forget-gate kernel needs it on RTX 5090). See `BUG_sm120_forget_gate.md`.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        # Apply sm_120 workaround BEFORE the GatedDeltaNet import triggers
        # any TileLang JIT in the gated-delta-rule backward path.
        try:
            import scripts.sm120_tilelang_workaround  # noqa: F401
        except ImportError:
            pass  # Non-sm120 hardware doesn't need it.
        from fla.layers import GatedDeltaNet
        assert d_model == n_heads * d_head, \
            f"DeltaNetForgetGate expects d_model == n_heads*d_head; got {d_model} vs {n_heads*d_head}"
        super().__init__(GatedDeltaNet(
            hidden_size=d_model,
            expand_v=1.0,
            head_dim=d_head,
            num_heads=n_heads,
            num_v_heads=n_heads,
            mode="chunk",
            use_gate=False,         # NO output gate — forget-gate-only variant
            use_short_conv=True,
            allow_neg_eigval=False, # match plain DN baseline
        ))


class GatedDeltaProductAttention(_FlaWrapper):
    """fla GatedDeltaProduct — DeltaNet + K Householder products + gating.

    DeltaProduct ([Yang et al. NeurIPS 2025][deltaproduct]) generalizes
    DeltaNet by using K Householder transformations per token (vs DeltaNet's
    single rank-1 erase). With allow_neg_eigval=True the spectrum extends
    beyond DeltaNet's, escaping Grazzi's TC⁰ wall at the cell level.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 num_householder: int = 2):
        from fla.layers import GatedDeltaProduct
        assert d_model == n_heads * d_head, \
            f"GatedDeltaProduct expects d_model == n_heads*d_head; got {d_model} vs {n_heads*d_head}"
        super().__init__(GatedDeltaProduct(
            hidden_size=d_model,
            expand_v=1.0,
            head_dim=d_head,
            num_heads=n_heads,
            num_v_heads=n_heads,
            mode="chunk",
            use_output_gate=True,
            use_short_conv=True,
            use_forget_gate=False,  # forget-gate kernel hits sm_120 misalign bug
            allow_neg_eigval=True,
            num_householder=num_householder,
        ))


class OrthogonalScanAttention(nn.Module):
    """SO(n) scan — Grazzi-clean NC¹-accessible parallel-scan primitive.

    Per-channel state is an n×n orthogonal matrix (SO(n)). Transition is
    `O_t = exp(X_t)` for X_t skew-symmetric (n(n-1)/2 floats from the
    input). Composition is matrix multiplication; exp is differentiable
    via `torch.linalg.matrix_exp`.

    Why this escapes Grazzi's TC⁰ wall:
      - X_t skew-symmetric ⇒ O_t = exp(X_t) ∈ SO(n) ⇒ eigenvalues are
        on the unit circle: {e^{iθ_k}} including −1 at θ=π.
      - SO(n) for n≥3 contains the icosahedral group A₅, which is
        non-solvable. Per Barrington's theorem, scans over non-solvable
        groups are NC¹-complete.

    Engineering options (matched to DeltaNet's defaults — these are what
    close the LM PPL gap on real text without touching the algebra):
      use_short_conv : kernel-`conv_size` 1D causal conv on the input
                       embedding before the projections (gives the layer
                       4-gram local context per token, consistent with
                       fla's `use_short_conv=True`).
      use_silu_input : SiLU activation on the input before the conv.
      use_v_norm     : L2-normalise the per-token rotation target vector
                       (analog of qk_norm="l2" in DeltaNet).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 ortho_dim: int = 4,
                 use_short_conv: bool = True,
                 conv_size: int = 4,
                 use_silu_input: bool = True,
                 use_v_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n = ortho_dim
        self.n_skew = ortho_dim * (ortho_dim - 1) // 2
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_silu_input = use_silu_input
        self.use_v_norm = use_v_norm

        # Optional kernel-K causal 1D conv before projections.
        if use_short_conv:
            self.short_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_size,
                padding=conv_size - 1, groups=d_model, bias=False,
            )

        # Skew-symmetric generator params per head per token.
        self.W_skew = nn.Linear(d_model, n_heads * self.n_skew, bias=False)
        # "Input" vector that gets rotated by accumulated state.
        self.W_v = nn.Linear(d_model, n_heads * ortho_dim, bias=False)
        # Output projection (n_heads · ortho_dim → d_model).
        self.W_o = nn.Linear(n_heads * ortho_dim, d_model, bias=False)

        # Cached upper-triangular index pattern for skew-symmetric build.
        idx_i, idx_j = torch.triu_indices(ortho_dim, ortho_dim, offset=1)
        self.register_buffer("_idx_i", idx_i, persistent=False)
        self.register_buffer("_idx_j", idx_j, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, n = self.n_heads, self.n

        # Optional pre-conv (causal): mixes a kernel-`conv_size` window of
        # past tokens before the projections, giving the layer local n-gram
        # context. Same trick as fla's `use_short_conv=True` on DeltaNet.
        x_in = x
        if self.use_short_conv:
            # (B, T, D) → (B, D, T) for Conv1d, then back.
            x_perm = x.transpose(1, 2)
            x_conv = self.short_conv(x_perm)              # (B, D, T + conv_size − 1)
            x_conv = x_conv[..., : T]                     # causal trim
            x_in = x_conv.transpose(1, 2)
        if self.use_silu_input:
            x_in = F.silu(x_in)

        # Build skew-symmetric matrix per (B, T, H).
        skew_flat = self.W_skew(x_in).view(B, T, H, self.n_skew)
        skew = torch.zeros(B, T, H, n, n, device=x.device, dtype=x.dtype)
        skew[..., self._idx_i, self._idx_j] = skew_flat
        skew[..., self._idx_j, self._idx_i] = -skew_flat

        # Per-step orthogonal transition: O_t = exp(X_t) ∈ SO(n).
        # matrix_exp handles arbitrary batch dims.
        O_t = torch.linalg.matrix_exp(skew)            # (B, T, H, n, n)

        # Cumulative left-multiplication along T: state_t = O_t · O_{t-1} · ... · O_0.
        # Use the Triton-backed differentiable scan kernel — eliminates the
        # Python for-loop overhead that dominated wall-clock at scale.
        # Layout: (B, T, H, n, n) → (B, H, T, n, n) for the kernel, then back.
        if x.is_cuda:
            from kernels.ortho_son import kernel as _ortho_kernel
            O_perm = O_t.permute(0, 2, 1, 3, 4).contiguous()       # (B, H, T, n, n)
            states_perm = _ortho_kernel.matmul_scan(O_perm, block_t=64)
            states = states_perm.permute(0, 2, 1, 3, 4).contiguous()
        else:
            # CPU fallback (Mac / debugging).
            states = torch.empty_like(O_t)
            states[:, 0] = O_t[:, 0]
            for t in range(1, T):
                states[:, t] = O_t[:, t] @ states[:, t - 1]

        # Apply accumulated rotation to learned per-token input vector.
        v = self.W_v(x_in).view(B, T, H, n)
        if self.use_v_norm:
            v = F.normalize(v, dim=-1, eps=1e-6)
        rotated = (states @ v.unsqueeze(-1)).squeeze(-1)       # (B, T, H, n)

        return self.W_o(rotated.reshape(B, T, H * n))


class RotConjAttention(nn.Module):
    """Semidirect-product scan `SO(n) ⋉ ℝ^{n×n}`.

    Per channel, state is `(R_t, c_t)` with R ∈ SO(n) and c ∈ ℝ^{n×n}.
    Per-token transition:
        R_t = O_t · R_{t-1}                          (rotation, like ortho)
        c_t = O_t · c_{t-1} · O_tᵀ + k_t ⊗ v_t       (conjugated KV memory)
    where O_t = exp(skew(W_skew · x_t)).

    Composition (provably associative):
        (R_a, c_a) · (R_b, c_b) = (R_b R_a,  R_b c_a R_bᵀ + c_b)

    This bypasses Grazzi's TC⁰ wall *on the memory slot itself*: the
    transition `c → R c Rᵀ` has spectrum `{e^{i(θ_a + θ_b)}}` which
    includes −1 (when θ_a + θ_b = π). The c slot is unbounded, giving
    KV-style memory growth that AUSSM-style pure-unitary architectures lack.

    Readout: `o_t = q_t · c_t` (linear-attention style).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 ortho_dim: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n = ortho_dim
        self.n_skew = ortho_dim * (ortho_dim - 1) // 2

        # Skew-symmetric generator (gives R via matrix_exp).
        self.W_skew = nn.Linear(d_model, n_heads * self.n_skew, bias=False)
        # K, V for the c memory (per-token outer product k_t ⊗ v_t).
        self.W_k = nn.Linear(d_model, n_heads * ortho_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * ortho_dim, bias=False)
        # Q for the readout (q_t · c_t).
        self.W_q = nn.Linear(d_model, n_heads * ortho_dim, bias=False)
        # Output projection.
        self.W_o = nn.Linear(n_heads * ortho_dim, d_model, bias=False)

        idx_i, idx_j = torch.triu_indices(ortho_dim, ortho_dim, offset=1)
        self.register_buffer("_idx_i", idx_i, persistent=False)
        self.register_buffer("_idx_j", idx_j, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, n = self.n_heads, self.n

        # Build skew-symmetric and orthogonal matrices per token.
        skew_flat = self.W_skew(x).view(B, T, H, self.n_skew)
        skew = torch.zeros(B, T, H, n, n, device=x.device, dtype=x.dtype)
        skew[..., self._idx_i, self._idx_j] = skew_flat
        skew[..., self._idx_j, self._idx_i] = -skew_flat
        O_t = torch.linalg.matrix_exp(skew)            # (B, T, H, n, n)

        # K, V — per-token outer products kv_t = k_t ⊗ v_t : (B, T, H, n, n).
        k = self.W_k(x).view(B, T, H, n)
        v = self.W_v(x).view(B, T, H, n)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)         # (B, T, H, n, n)

        # Sequential scan: c_t = O_t @ c_{t-1} @ O_tᵀ + kv_t.
        c = torch.zeros(B, H, n, n, device=x.device, dtype=x.dtype)
        c_states = torch.empty(B, T, H, n, n, device=x.device, dtype=x.dtype)
        for t in range(T):
            O = O_t[:, t]                              # (B, H, n, n)
            c = O @ c @ O.transpose(-1, -2) + kv[:, t]
            c_states[:, t] = c

        # Readout: o_t = q_t · c_t (linear-attn style; q is row vector).
        q = self.W_q(x).view(B, T, H, n)               # (B, T, H, n)
        # q (..., n) treated as (..., 1, n); c (..., n, n); result (..., 1, n) → (..., n).
        o = (q.unsqueeze(-2) @ c_states).squeeze(-2)   # (B, T, H, n)

        return self.W_o(o.reshape(B, T, H * n))


class RotDeltaAttention(nn.Module):
    """Rotation-conjugated DeltaNet — variant (α) of the rotation+delta family.

    State per channel: c ∈ ℝ^{n×n}. Per-step update combines (a) two-sided
    rotation conjugation (Grazzi-clean — eigenvalues of `R ⊗ R` include −1),
    (b) DeltaNet-style rank-1 erase (recall mechanism), and (c) rank-1 write:

        c_t = (I − β_t k_t k_tᵀ) · (O_t c_{t-1} O_tᵀ) + β_t k_t v_tᵀ

    Per-token transition factors as `c → A_t c B_t + d_t` with:
        A_t = (I − β_t k_t k_tᵀ) O_t
        B_t = O_tᵀ
        d_t = β_t k_t v_tᵀ

    The triple-monoid `(A, B, d) · (A', B', d') = (A'A, BB', A'dB' + d')`
    is associative — verified analytically. Distinct from DeltaProduct
    (which uses left-multiplication only); the two-sided action `c → AcB`
    preserves trace/eigenvalues/rank of c, which left-multiplication does
    not. Per the literature search, this combination is genuinely novel.

    β_t = sigmoid(W_β · x_t) ∈ (0, 1) — standard DeltaNet erase strength.
    Readout: o_t = q_t · c_t.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 ortho_dim: int = 4, max_skew_angle: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n = ortho_dim
        self.n_skew = ortho_dim * (ortho_dim - 1) // 2
        # Bound the per-step rotation angle so cumulative rotation over T
        # doesn't scramble the state. With max_skew_angle=0.5, each O_t is
        # within ~30° of identity per axis, and over T=64 the cumulative
        # rotation stays in a controlled regime.
        self.max_skew_angle = max_skew_angle

        self.W_skew = nn.Linear(d_model, n_heads * self.n_skew, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * ortho_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * ortho_dim, bias=False)
        self.W_q = nn.Linear(d_model, n_heads * ortho_dim, bias=False)
        # Erase strength gate β ∈ (0, 1) per token per head.
        self.W_beta = nn.Linear(d_model, n_heads, bias=True)
        self.W_o = nn.Linear(n_heads * ortho_dim, d_model, bias=False)

        idx_i, idx_j = torch.triu_indices(ortho_dim, ortho_dim, offset=1)
        self.register_buffer("_idx_i", idx_i, persistent=False)
        self.register_buffer("_idx_j", idx_j, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, n = self.n_heads, self.n

        # Build skew-symmetric and orthogonal matrices, with bounded angle.
        skew_flat = self.W_skew(x).view(B, T, H, self.n_skew)
        skew_flat = self.max_skew_angle * torch.tanh(skew_flat)
        skew = torch.zeros(B, T, H, n, n, device=x.device, dtype=x.dtype)
        skew[..., self._idx_i, self._idx_j] = skew_flat
        skew[..., self._idx_j, self._idx_i] = -skew_flat
        O_t = torch.linalg.matrix_exp(skew)            # (B, T, H, n, n)

        # K, V, β per token per head. Normalize k to unit norm so the
        # erase factor (I − β k kᵀ) is well-conditioned (eigenvalues
        # in {1, …, 1, 1−β} along k).
        k = self.W_k(x).view(B, T, H, n)
        k = F.normalize(k, dim=-1, eps=1e-6)            # unit-norm key
        v = self.W_v(x).view(B, T, H, n)
        beta = torch.sigmoid(self.W_beta(x))            # (B, T, H), in (0, 1)

        # Sequential scan: c_t = (I − β k kᵀ)(O c_{t-1} Oᵀ) + β k vᵀ.
        c = torch.zeros(B, H, n, n, device=x.device, dtype=x.dtype)
        c_states = torch.empty(B, T, H, n, n, device=x.device, dtype=x.dtype)
        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        for t in range(T):
            O = O_t[:, t]                              # (B, H, n, n)
            k_t = k[:, t]                              # (B, H, n)
            v_t = v[:, t]                              # (B, H, n)
            b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            # 1. Rotate: c' = O · c · Oᵀ.
            c_rot = O @ c @ O.transpose(-1, -2)
            # 2. Erase: (I − β k kᵀ) · c'  via Sherman-Morrison form.
            kkT_c = k_t.unsqueeze(-1) * (k_t.unsqueeze(-2) @ c_rot).squeeze(-2).unsqueeze(-2)
            # Cleaner: build erase matrix and multiply.
            erase = eye - b_t * (k_t.unsqueeze(-1) * k_t.unsqueeze(-2))   # (B, H, n, n)
            c_erased = erase @ c_rot
            # 3. Write: + β k vᵀ.
            write = b_t * (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
            c = c_erased + write
            c_states[:, t] = c

        # Readout: o_t = q_t · c_t.
        q = self.W_q(x).view(B, T, H, n)
        o = (q.unsqueeze(-2) @ c_states).squeeze(-2)   # (B, T, H, n)

        return self.W_o(o.reshape(B, T, H * n))


class Mamba2Attention(_FlaWrapper):
    """fla Mamba2 — modern SSM, state-of-the-art for many state-tracking tasks."""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        from fla.layers import Mamba2
        # Mamba2 requires `num_heads * head_dim == expand * hidden_size`. We
        # want hidden_size = d_model and num_heads * head_dim = d_model, so
        # set expand=1 (rather than fla's default 2).
        super().__init__(Mamba2(
            num_heads=n_heads,
            head_dim=d_head,
            hidden_size=d_model,
            state_size=64,           # SSM state size
            expand=1,
            n_groups=1,
            chunk_size=64,
        ))


# ---------------------------------------------------------------------------
# Symbol-Grounded scan — novel direction A.
#
# Per-head state S ∈ ℝ^{V × D}, keyed by INPUT TOKEN ID. On binding events
# (gated), the cell writes the current value to S[id]. On every token, the
# cell reads S[id] (last-write-wins). Different from kNN-cache and PKM:
# keys are token *identities*, not opaque content vectors.
#
# Refresher on associativity:
#   For a sequence of (id_i, gate_i, value_i), the prefix-state is the
#   "last-write-wins" semilattice — for each id, take the latest write
#   whose gate fired hardest (we use a soft mix). Composing two prefix
#   tables (A, B) where B is later: result[id] = B[id] if id ∈ B else A[id].
#   This is associative ⇒ parallel-scan compatible.
#
# This reference impl is sequential in T (Python loop) for clarity. Speed
# is fine at synthetic-task scale (T ≤ 256). A Triton scan kernel comes
# later if the cell wins on var_binding.
# ---------------------------------------------------------------------------


class SymbolGroundedAttention(nn.Module):
    """Sequence-aware sparse symbol table keyed by token-id.

    Forward signature requires `input_ids` (the token IDs, B×T int64).
    Block.forward must thread these through.

    State (per head, per example): S ∈ ℝ^{V × D}.
        Initialized to zeros at t=0.
        On token t with id_t, writes S[id_t] ← gate · v_t + (1-gate) · S[id_t].
        Reads happen *before* the write (causal).

    Output at token t:  o_t = q_t · read_t + b_t
        where read_t = S_{t-1}[id_t], q_t/b_t are per-token projections.
        b_t (bypass) keeps the layer useful when id_t hasn't been bound.
    """

    needs_input_ids = True       # signal to Block

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 vocab_size: int, n_symbols: int | None = None):
        """
        Args:
            vocab_size: tokenizer vocab size (for the embed dim story).
            n_symbols: size of the hashed symbol table. If None, uses
                vocab_size directly (state is (B,H,V,D)). For real LM use
                with V=50K, set n_symbols ~ 256-2048 to keep state small;
                IDs are hashed via `id % n_symbols`.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.vocab_size = vocab_size
        self.n_symbols = n_symbols if n_symbols is not None else vocab_size
        self.qkv_dim = n_heads * d_head
        self.W_v = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_w = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_q = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_b = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_o = nn.Linear(self.qkv_dim, d_model, bias=False)
        # Bias the write-gate toward "don't write" so the cell starts as a
        # near-pure bypass and learns to turn on writes as needed.
        nn.init.zeros_(self.W_w.weight)
        self.write_bias = nn.Parameter(torch.full((self.qkv_dim,), -2.0))

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D, V = self.n_heads, self.d_head, self.n_symbols

        v = _split_heads(self.W_v(x), H)                          # (B, H, T, D)
        w_logit = _split_heads(self.W_w(x) + self.write_bias, H)  # (B, H, T, D)
        q = _split_heads(self.W_q(x), H)                          # (B, H, T, D)
        b = _split_heads(self.W_b(x), H)                          # (B, H, T, D)

        S = torch.zeros(B, H, V, D, device=x.device, dtype=x.dtype)
        outs = []

        # Hash token ids into the (smaller) symbol-table dimension.
        if self.n_symbols != self.vocab_size:
            hashed_ids = input_ids % self.n_symbols
        else:
            hashed_ids = input_ids

        for t in range(T):
            id_t = hashed_ids[:, t]                               # (B,)
            # Index for gather/scatter along V dim.
            id_idx = id_t.view(B, 1, 1, 1).expand(B, H, 1, D)     # (B, H, 1, D)

            # Read S[id_t] before write.
            read_t = S.gather(2, id_idx).squeeze(2)               # (B, H, D)

            # Output mixes lookup with bypass.
            o_t = q[:, :, t] * read_t + b[:, :, t]                # (B, H, D)
            outs.append(o_t)

            # Gated write.
            gate = torch.sigmoid(w_logit[:, :, t])                # (B, H, D)
            new_val = gate * v[:, :, t] + (1.0 - gate) * read_t   # (B, H, D)
            S = S.scatter(2, id_idx, new_val.unsqueeze(2))

        out = torch.stack(outs, dim=2)                            # (B, H, T, D)
        return self.W_o(_merge_heads(out))


# ---------------------------------------------------------------------------
# Multi-pass parallel scans — novel direction B.
#
# Within a single layer, run K different cells on the SAME input residual,
# fuse outputs with a learned mixture. The mechanistic intent: instead of
# alternating (rotation, delta, rotation, delta) at the LAYER level (which
# loses each cell's state across the gap), every reading mode is
# simultaneously available at every token.
#
# Compute cost is K× a single cell. Pair this with reduced layer count
# for a fair compute-matched comparison.
# ---------------------------------------------------------------------------


class MultiPassAttention(nn.Module):
    """K cells in parallel on the same input; learned softmax mixture.

    The cells argument is a tuple of (constructor, kwargs_dict) pairs, so
    the user can mix cells that need different signatures (e.g. one that
    requires `vocab_size`).

    Output: weighted sum of cell outputs, with weights g = softmax(α)
    where α is a learned (K,) parameter starting at zeros (uniform 1/K).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 cells: list):
        """
        Args:
            cells: list of cell constructors. Each must accept
                (d_model=, n_heads=, d_head=) at minimum. If a cell needs
                `input_ids`, it must declare `needs_input_ids = True`.
                For SymbolGrounded use a closure that pre-binds vocab_size.
        """
        super().__init__()
        self.K = len(cells)
        self.cells = nn.ModuleList([
            c(d_model=d_model, n_heads=n_heads, d_head=d_head)
            for c in cells
        ])
        # Per-cell mixture logit; softmax over K.
        self.alpha = nn.Parameter(torch.zeros(self.K))

    @property
    def needs_input_ids(self) -> bool:
        return any(getattr(c, "needs_input_ids", False) for c in self.cells)

    def forward(self, x: torch.Tensor,
                input_ids: torch.Tensor | None = None) -> torch.Tensor:
        outs = []
        for c in self.cells:
            if getattr(c, "needs_input_ids", False):
                outs.append(c(x, input_ids=input_ids))
            else:
                outs.append(c(x))
        gates = torch.softmax(self.alpha, dim=0)                  # (K,)
        # outs is list of (B, T, d_model); stack to (K, B, T, d_model)
        stacked = torch.stack(outs, dim=0)
        return (gates.view(self.K, 1, 1, 1) * stacked).sum(dim=0)
