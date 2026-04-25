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
        # `expand_k=expand_v=1.0` keeps head_dim aligned to our d_head when
        # `hidden_size = n_heads * d_head` — but DeltaNet derives head_dim from
        # hidden_size / num_heads, so we set hidden_size = n_heads * d_head and
        # let the model project from/to d_model itself if the dims differ.
        # In our setup d_model == n_heads * d_head (matched arch), so this works.
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
