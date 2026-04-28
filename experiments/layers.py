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
    """

    def __init__(self, fla_layer: nn.Module):
        super().__init__()
        self.layer = fla_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fla's chunked kernels assert non-fp32 inputs. Use mixed-precision
        # autocast so weights stay fp32 (master copy) and the kernel sees bf16.
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            out = self.layer(x)
        if isinstance(out, tuple):
            out = out[0]
        return out.to(in_dtype)


class DeltaNetAttention(_FlaWrapper):
    """fla DeltaNet — same KV-state size as our linear_attn, plus delta updates."""

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


class PDScanAttention(nn.Module):
    """PD-SSM (Permutation × Diagonal) state-space scan.

    Per channel, state is a vector h ∈ ℝ^N. The per-token transition is

        A_t = P_t · D_t

    where P_t ∈ {0,1}^{N×N} is a column-one-hot matrix encoding a function
    σ_t : {0..N-1} → {0..N-1} (an element of the full transformation monoid
    T_N), and D_t is real-diagonal with entries in [-1, 1]. Recurrence:

        h_t[i] = D_t[σ_t⁻¹(i)] · h_{t-1}[σ_t⁻¹(i)] + w_t[i]

    Closure: (P_a D_a)(P_b D_b) = (P_a P_b)(D_a' · D_b) with D_a'[j] = D_a[σ_b(j)],
    so cumulative products stay in the PD class — O(N) per compose,
    O(L · N) parallel scan.

    Why this escapes both walls of the repo's analysis:
      - Eigenvalues lie on the closed real interval [-1, 1] in this
        implementation (D real); negative real ⇒ parity, mod-2.
        Cycles of length k in σ_t with D = -1 give k-th roots of unity
        in the cumulative spectrum ⇒ mod-k counting is reachable.
      - σ_t is the full transformation monoid T_N per token (not just
        Householder reflections), so by Cayley's theorem one layer
        recognises any FSA with ≤ N states (Terzić et al. 2025
        Proposition 2). This includes non-solvable groups (A_5, S_5)
        when N ≥ 5 — strictly more than DeltaProduct's n_h reflections.

    Differentiability: σ_t is discrete. We use straight-through:
    forward = argmax (one-hot), backward = softmax. This is the standard
    Bengio-Léonard-Courville trick. The N×N logits tensor has size
    O(B · T · H · N²); keep N modest (8–32) at the kill-gate scale.

    Reference: Terzić et al., "Structured Sparse Transition Matrices to
    Enable State Tracking in State-Space Models" (arXiv:2509.22284).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 state_dim: int | None = None,
                 use_short_conv: bool = True,
                 conv_size: int = 4,
                 use_silu_input: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        # Per-head state dim. Default N = d_head — the per-head output
        # is the full state vector at time t.
        self.N = state_dim if state_dim is not None else d_head
        N = self.N
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_silu_input = use_silu_input

        # Optional kernel-K causal 1D conv before projections — same
        # short-conv trick as fla DeltaNet / OrthogonalScanAttention.
        if use_short_conv:
            self.short_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_size,
                padding=conv_size - 1, groups=d_model, bias=False,
            )

        # Per-output-slot logits over N input candidates. Picks σ_t⁻¹.
        self.W_sigma = nn.Linear(d_model, n_heads * N * N, bias=False)
        # Real diagonal in [-1, 1] via 2·sigmoid − 1 ⇒ negative-eig unlocked.
        self.W_D = nn.Linear(d_model, n_heads * N, bias=False)
        # Per-token additive write into the state.
        self.W_w = nn.Linear(d_model, n_heads * N, bias=False)
        # Output projection (n_heads · N → d_model).
        self.W_o = nn.Linear(n_heads * N, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, N = self.n_heads, self.N

        # Optional pre-conv (causal): kernel-`conv_size` 1D conv mixes a
        # local window before projections. Same trick as DeltaNet's
        # `use_short_conv=True`.
        x_in = x
        if self.use_short_conv:
            x_perm = x.transpose(1, 2)
            x_conv = self.short_conv(x_perm)              # (B, D, T + conv_size − 1)
            x_conv = x_conv[..., : T]                     # causal trim
            x_in = x_conv.transpose(1, 2)
        if self.use_silu_input:
            x_in = F.silu(x_in)

        # Permutation logits: (B, T, H, N, N). Each output slot i has a
        # softmax over N input candidates j; argmax picks σ_t⁻¹(i).
        sigma_logits = self.W_sigma(x_in).view(B, T, H, N, N)
        sigma_soft = F.softmax(sigma_logits, dim=-1)
        sigma_idx = sigma_logits.argmax(dim=-1)            # (B, T, H, N)
        sigma_hard = F.one_hot(sigma_idx, num_classes=N).to(x.dtype)
        # Straight-through: forward = hard, backward = soft.
        P_inv = sigma_hard + (sigma_soft - sigma_soft.detach())

        # Diagonal in [-1, 1].
        D = 2.0 * torch.sigmoid(self.W_D(x_in).view(B, T, H, N)) - 1.0

        # Additive write per token.
        w = self.W_w(x_in).view(B, T, H, N)

        # Sequential scan: h_0 = 0; h_t = (P_inv_t · (D_t * h_{t-1}_gathered)) + w_t.
        # Vectorised inside batch + heads via batched matmul.
        h = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)
        out = torch.empty(B, T, H, N, device=x.device, dtype=x.dtype)
        for t in range(T):
            P_inv_t = P_inv[:, t]                          # (B, H, N, N)
            D_t = D[:, t]                                  # (B, H, N)
            w_t = w[:, t]                                  # (B, H, N)
            # gather_h[i] = h_prev[σ⁻¹(i)] = (P_inv · h_prev)[i].
            gather_h = (P_inv_t @ h.unsqueeze(-1)).squeeze(-1)
            # gather_D[i] = D[σ⁻¹(i)] — same gather pattern via P_inv.
            gather_D = (P_inv_t @ D_t.unsqueeze(-1)).squeeze(-1)
            h = gather_D * gather_h + w_t
            out[:, t] = h

        return self.W_o(out.reshape(B, T, H * N))


class PDKVScanAttention(nn.Module):
    """PD-KV — matrix-state PD-SSM with rank-1 KV write.

    Generalises PDScanAttention by lifting the vector state h ∈ ℝ^N to a
    matrix state S ∈ ℝ^{N × D}, and adding a DeltaNet-style rank-1 KV
    write. Combines PD-SSM's transformation-monoid transitions on rows
    of S with linear-attention's outer-product memory writes.

    Recurrence:
        S_t = (P_t · D_t) S_{t-1} + k_t v_tᵀ
        o_t = q_tᵀ S_t                       (∈ ℝ^D)

    where:
        P_t ∈ {0,1}^{N×N} is column-one-hot (full T_N monoid),
        D_t ∈ ℝ^{N} is real-diagonal in [-1, 1] (acts on rows),
        k_t ∈ ℝ^N is the per-slot write key,
        v_t ∈ ℝ^D is the value,
        q_t ∈ ℝ^N is the read query.

    Closure: (P_a D_a)(P_b D_b) is still PD, so the cumulative-transition
    monoid stays PD; rank-1 writes accumulate as in linear attention.
    Per-token cost: O(N · D) for both the gather-transition and rank-1
    write — strictly cheaper than DeltaNet's O(d²) when N · D ≤ d².

    Why this is the natural PD/DeltaNet hybrid:
      - From PD-SSM: transformation monoid T_N per token ⇒ all FSAs ≤ N
        states recognised in one layer (Terzić et al. 2025).
      - From DeltaNet: matrix state with rank-1 KV write ⇒ associative
        recall up to ~N · D bindings (Arora et al. 2023, MQAR).
      - Combined: closes the capacity gap of pure PD-SSM (vector state
        ⇒ ~N bindings) without giving up its non-solvable group reach.

    This combination does not appear in the published literature as of
    arXiv:2509.22284 (PD-SSM uses pure vector state) or arXiv:2406.06484
    (DeltaNet uses Householder I − β k kᵀ rather than P · D).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 state_dim: int | None = None,
                 use_short_conv: bool = True,
                 conv_size: int = 4,
                 use_silu_input: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        # N = per-head permutation dim. Default N = d_head matches the
        # square-state convention of DeltaNet (state size = d_head²).
        self.N = state_dim if state_dim is not None else d_head
        N, D = self.N, d_head
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_silu_input = use_silu_input

        if use_short_conv:
            self.short_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_size,
                padding=conv_size - 1, groups=d_model, bias=False,
            )

        # Per-output-slot logits over N input candidates (σ_t⁻¹).
        self.W_sigma = nn.Linear(d_model, n_heads * N * N, bias=False)
        # Real diagonal in [-1, 1] via 2·sigmoid − 1.
        self.W_D = nn.Linear(d_model, n_heads * N, bias=False)
        # KV: k picks slot, v is value.
        self.W_k = nn.Linear(d_model, n_heads * N, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * D, bias=False)
        # Q: read slot.
        self.W_q = nn.Linear(d_model, n_heads * N, bias=False)
        # Output projection (n_heads · D → d_model).
        self.W_o = nn.Linear(n_heads * D, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, N, D = self.n_heads, self.N, self.d_head

        x_in = x
        if self.use_short_conv:
            x_perm = x.transpose(1, 2)
            x_conv = self.short_conv(x_perm)[..., : T]
            x_in = x_conv.transpose(1, 2)
        if self.use_silu_input:
            x_in = F.silu(x_in)

        # Permutation logits and straight-through softmax.
        sigma_logits = self.W_sigma(x_in).view(B, T, H, N, N)
        sigma_soft = F.softmax(sigma_logits, dim=-1)
        sigma_idx = sigma_logits.argmax(dim=-1)
        sigma_hard = F.one_hot(sigma_idx, num_classes=N).to(x.dtype)
        P_inv = sigma_hard + (sigma_soft - sigma_soft.detach())  # (B, T, H, N, N)

        # Diagonal in [-1, 1].
        Dvec = 2.0 * torch.sigmoid(self.W_D(x_in).view(B, T, H, N)) - 1.0

        # KVQ.
        k = self.W_k(x_in).view(B, T, H, N)            # (B, T, H, N)
        v = self.W_v(x_in).view(B, T, H, D)            # (B, T, H, D)
        q = self.W_q(x_in).view(B, T, H, N)            # (B, T, H, N)

        # Sequential scan over T. State S ∈ (B, H, N, D).
        S = torch.zeros(B, H, N, D, device=x.device, dtype=x.dtype)
        out = torch.empty(B, T, H, D, device=x.device, dtype=x.dtype)
        for t in range(T):
            P_inv_t = P_inv[:, t]                       # (B, H, N, N)
            D_t = Dvec[:, t]                            # (B, H, N)
            k_t = k[:, t]                               # (B, H, N)
            v_t = v[:, t]                               # (B, H, D)
            q_t = q[:, t]                               # (B, H, N)

            # Gather rows of S via P_inv: gathered_S[b, h, i, :] = S[b, h, σ⁻¹(i), :].
            gathered_S = P_inv_t @ S                                   # (B, H, N, D)
            gathered_D = (P_inv_t @ D_t.unsqueeze(-1)).squeeze(-1)     # (B, H, N)
            # Apply diagonal scaling per row.
            transitioned = gathered_D.unsqueeze(-1) * gathered_S       # (B, H, N, D)
            # Add rank-1 KV write.
            S = transitioned + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)    # (B, H, N, D)

            # Read: o_t = qᵀ S → (B, H, D).
            out[:, t] = (q_t.unsqueeze(-2) @ S).squeeze(-2)

        return self.W_o(out.reshape(B, T, H * D))


class ComplexPDScanAttention(nn.Module):
    """Complex-PD — PD-SSM with complex unit-disk diagonal.

    Same recurrence as PDScanAttention but with D_t complex-valued
    (|D_t[j]| ≤ 1, arbitrary phase). State is complex too, represented
    as a real tensor of shape (..., N, 2) where the last axis holds
    (real, imag) — MPS-compatible without native complex tensor support.

    Why this generalisation matters:
      Real-only D constrains achievable group orders to those embeddable
      in S_N as permutation cycles. By Landau's function g(N), the
      maximum element-order is g(4)=4, g(5)=6, g(6)=6, g(7)=12, g(8)=15.
      So PDScanAttention with N=4 cannot do mod-p for p ≥ 5; with N=6
      cannot do mod-7; etc.

      With complex D = e^{2πi k/p}, a single non-trivial element of D on
      any 1-cycle of σ realises Z_p exactly via |D|=1, θ=2π/p — for ANY
      p, regardless of N. This lifts PD-SSM out of the Landau bound:
      mod-p becomes solvable at N=2 for any p.

    Real-pair complex multiplication:
      (a + bi)(c + di) = (ac − bd) + (ad + bc)i

    Recurrence (component-wise):
      h_re_t[i] = D_re·h_re − D_im·h_im   (gathered via σ⁻¹) + w_re_t[i]
      h_im_t[i] = D_re·h_im + D_im·h_re   (gathered via σ⁻¹) + w_im_t[i]

    Output: real part of h_t (or concatenation of re+im, then projected).
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 state_dim: int | None = None,
                 use_short_conv: bool = True,
                 conv_size: int = 4,
                 use_silu_input: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.N = state_dim if state_dim is not None else d_head
        N = self.N
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_silu_input = use_silu_input

        if use_short_conv:
            self.short_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_size,
                padding=conv_size - 1, groups=d_model, bias=False,
            )

        self.W_sigma = nn.Linear(d_model, n_heads * N * N, bias=False)
        # Complex D parameterised as |D| ∈ (0, 1) and angle θ ∈ ℝ.
        # |D| via sigmoid; θ via raw linear (any real → wrapped phase).
        self.W_D_mag = nn.Linear(d_model, n_heads * N, bias=False)
        self.W_D_phase = nn.Linear(d_model, n_heads * N, bias=False)
        # Complex write — (real, imag) pair.
        self.W_w_re = nn.Linear(d_model, n_heads * N, bias=False)
        self.W_w_im = nn.Linear(d_model, n_heads * N, bias=False)
        # Output projection: read both re+im of state ⇒ 2N per head.
        self.W_o = nn.Linear(n_heads * 2 * N, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, N = self.n_heads, self.N

        x_in = x
        if self.use_short_conv:
            x_perm = x.transpose(1, 2)
            x_conv = self.short_conv(x_perm)[..., : T]
            x_in = x_conv.transpose(1, 2)
        if self.use_silu_input:
            x_in = F.silu(x_in)

        sigma_logits = self.W_sigma(x_in).view(B, T, H, N, N)
        sigma_soft = F.softmax(sigma_logits, dim=-1)
        sigma_idx = sigma_logits.argmax(dim=-1)
        sigma_hard = F.one_hot(sigma_idx, num_classes=N).to(x.dtype)
        P_inv = sigma_hard + (sigma_soft - sigma_soft.detach())

        # Complex D in unit disk: |D| ∈ (0, 1), phase ∈ ℝ.
        D_mag = torch.sigmoid(self.W_D_mag(x_in).view(B, T, H, N))
        D_phase = self.W_D_phase(x_in).view(B, T, H, N)
        D_re = D_mag * torch.cos(D_phase)
        D_im = D_mag * torch.sin(D_phase)

        w_re = self.W_w_re(x_in).view(B, T, H, N)
        w_im = self.W_w_im(x_in).view(B, T, H, N)

        # State carries (re, im) — sequential scan.
        h_re = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)
        h_im = torch.zeros(B, H, N, device=x.device, dtype=x.dtype)
        out = torch.empty(B, T, H, 2 * N, device=x.device, dtype=x.dtype)
        for t in range(T):
            P_inv_t = P_inv[:, t]                      # (B, H, N, N)
            D_re_t = D_re[:, t]                        # (B, H, N)
            D_im_t = D_im[:, t]
            # Gather state and D via σ⁻¹ — same gather pattern as PD-SSM.
            g_h_re = (P_inv_t @ h_re.unsqueeze(-1)).squeeze(-1)
            g_h_im = (P_inv_t @ h_im.unsqueeze(-1)).squeeze(-1)
            g_D_re = (P_inv_t @ D_re_t.unsqueeze(-1)).squeeze(-1)
            g_D_im = (P_inv_t @ D_im_t.unsqueeze(-1)).squeeze(-1)
            # Complex multiplication: (D_re + i·D_im)(h_re + i·h_im).
            new_h_re = g_D_re * g_h_re - g_D_im * g_h_im + w_re[:, t]
            new_h_im = g_D_re * g_h_im + g_D_im * g_h_re + w_im[:, t]
            h_re, h_im = new_h_re, new_h_im
            out[:, t, :, :N] = h_re
            out[:, t, :, N:] = h_im

        return self.W_o(out.reshape(B, T, H * 2 * N))
