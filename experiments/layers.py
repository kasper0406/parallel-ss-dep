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
      - Compare to plain Heisenberg whose transition spec ⊂ {1}: stuck TC⁰.

    State per channel: n² floats. Composition: n×n matmul (fast for small n).
    For n=4: 16-float state per channel; small enough to fit alongside
    Heisenberg's d²-state at matched parameter count.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 ortho_dim: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.n = ortho_dim
        self.n_skew = ortho_dim * (ortho_dim - 1) // 2

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

        # Build skew-symmetric matrix per (B, T, H).
        skew_flat = self.W_skew(x).view(B, T, H, self.n_skew)
        skew = torch.zeros(B, T, H, n, n, device=x.device, dtype=x.dtype)
        skew[..., self._idx_i, self._idx_j] = skew_flat
        skew[..., self._idx_j, self._idx_i] = -skew_flat

        # Per-step orthogonal transition: O_t = exp(X_t) ∈ SO(n).
        # matrix_exp handles arbitrary batch dims.
        O_t = torch.linalg.matrix_exp(skew)            # (B, T, H, n, n)

        # Cumulative left-multiplication along T: state_t = O_t · O_{t-1} · ... · O_0.
        # Sequential scan — replace with Blelloch on GPU later if hot.
        states = torch.empty_like(O_t)
        states[:, 0] = O_t[:, 0]
        for t in range(1, T):
            states[:, t] = O_t[:, t] @ states[:, t - 1]

        # Apply accumulated rotation to learned per-token input vector.
        v = self.W_v(x).view(B, T, H, n)
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
