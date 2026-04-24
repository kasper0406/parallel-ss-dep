"""
Linear attention as a monoid scan — PyTorch reference implementations.

This is the *baseline* all novel kernels are compared against. The monoid
is trivially abelian: state S ∈ R^{d_k × d_v} accumulates under plain
matrix addition:

    S_t = Σ_{i ≤ t} k_i v_iᵀ       (outer products summed along i)
    o_t = q_t @ S_t                 (query reads the current KV state)

Because `+` is commutative, any scan schedule works — so the real content
here is the chunkwise blocking strategy, which mirrors what
`flash-linear-attention` (and hence our Triton kernel) uses.

The two reference implementations are:

  naive_loop : exact T-step Python loop. Ground truth. O(T·d_k·d_v).
  chunked    : chunked chunks-attend-to-state form. Same complexity but
               organised into (C×d_k)·(d_k×d_v) matmuls, which is what
               the kernel will launch onto tensor cores.

These must agree numerically on any input. `test.py` verifies that.
"""
from __future__ import annotations
import torch
from torch import Tensor


def naive_loop(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """
    q, k: (T, d_k). v: (T, d_v). Returns o: (T, d_v).

    Exact T-step Python loop — unambiguously correct, slow.
    """
    T, d_k = q.shape
    _, d_v = v.shape
    S = torch.zeros(d_k, d_v, dtype=q.dtype, device=q.device)
    out = torch.empty(T, d_v, dtype=q.dtype, device=q.device)
    for t in range(T):
        S = S + torch.outer(k[t], v[t])   # rank-1 update to KV state
        out[t] = q[t] @ S                  # read
    return out


def chunked(
    q: Tensor, k: Tensor, v: Tensor, chunk_size: int = 64
) -> Tensor:
    """
    Chunkwise form, exactly matching the Triton kernel's blocking.

    Within a chunk of size C, compute:
      inter[t]  = q[t] @ S_prev                          (C×d_k · d_k×d_v)
      attn[s,t] = q[t] @ k[s] for s ≤ t, else 0          (causal C×C)
      intra[t]  = Σ_s attn[s,t] · v[s]                   (C×C · C×d_v)
      out[t]    = inter[t] + intra[t]

    Between chunks, advance state:
      S ← S + k_c^T @ v_c                                (d_k×C · C×d_v)

    All three matrix products are dense GEMMs — ideal for tensor cores.
    """
    T, d_k = q.shape
    _, d_v = v.shape
    S = torch.zeros(d_k, d_v, dtype=q.dtype, device=q.device)
    out = torch.empty(T, d_v, dtype=q.dtype, device=q.device)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        qc, kc, vc = q[start:end], k[start:end], v[start:end]
        C = qc.shape[0]

        # Pre-chunk state contribution.
        inter = qc @ S                                    # (C, d_v)

        # Intra-chunk dense causal attention.
        attn = qc @ kc.transpose(-1, -2)                  # (C, C)
        mask = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
        attn = attn.masked_fill(~mask, 0.0)
        intra = attn @ vc                                 # (C, d_v)

        out[start:end] = inter + intra

        # Advance state by the full chunk.
        S = S + kc.transpose(-1, -2) @ vc                 # (d_k, d_v)

    return out
