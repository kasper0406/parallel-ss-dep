"""
SO(n) scan — PyTorch reference implementation.

State per channel: orthogonal n×n matrix in SO(n).
Transition: input-dependent rotation `O_t = exp(X_t)` where X_t is
skew-symmetric (built from n(n-1)/2 floats per token).
Composition: matrix multiplication.

Why this primitive:
  - Per-step transition has eigenvalues `e^{iθ_k}` on the unit circle —
    can hit −1 at θ=π. So spec(O_t) ⊂ unit circle, including −1.
  - For n ≥ 3, SO(n) contains the icosahedral group A₅ (non-solvable),
    so any sequence of input-dependent rotations generates a non-solvable
    transformation semigroup — provably escapes Grazzi's TC⁰ wall.

Two implementations:
  naive_loop : exact T-step Python loop (slow, ground truth).
  chunked    : chunkwise form mirroring the eventual Triton kernel.
"""
from __future__ import annotations

import torch
from torch import Tensor


def build_skew(skew_flat: Tensor, n: int) -> Tensor:
    """Build skew-symmetric n×n matrix from n(n-1)/2 upper-triangular floats.

    skew_flat: (..., n(n-1)/2). Returns: (..., n, n).
    """
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    out = torch.zeros(*skew_flat.shape[:-1], n, n,
                      dtype=skew_flat.dtype, device=skew_flat.device)
    out[..., idx_i, idx_j] = skew_flat
    out[..., idx_j, idx_i] = -skew_flat
    return out


def naive_loop(skew_flat: Tensor) -> Tensor:
    """T-step sequential SO(n) scan.

    skew_flat: (T, n(n-1)/2). Returns running orthogonal state at each step.

    Output shape: (T, n, n) — running cumulative product of rotations,
    `O_T = O_T · O_{T-1} · … · O_0`.
    """
    T, k = skew_flat.shape
    # n from k = n(n-1)/2: solve n² − n − 2k = 0 → n = (1 + sqrt(1+8k)) / 2.
    n = int((1 + (1 + 8 * k) ** 0.5) / 2 + 0.5)
    assert n * (n - 1) // 2 == k, f"k={k} doesn't match any n via n(n-1)/2"

    skew = build_skew(skew_flat, n)                    # (T, n, n)
    O = torch.linalg.matrix_exp(skew)                  # (T, n, n)

    state = torch.empty_like(O)
    state[0] = O[0]
    for t in range(1, T):
        state[t] = O[t] @ state[t - 1]
    return state


def chunked(skew_flat: Tensor, chunk_size: int = 64) -> Tensor:
    """Chunkwise SO(n) scan — mirrors the eventual Triton kernel.

    Within each chunk: compute per-step `O_t = exp(X_t)`, then build the
    chunk-local prefix product via Hillis-Steele (or sequential within-chunk).
    Across chunks: maintain a running cumulative product `R_run`, then
    each chunk's local result `L_t` becomes `state_t = L_t · R_run`.
    After the chunk, `R_run ← chunk_total · R_run`.

    Equivalent to `naive_loop` because matrix multiplication is associative.
    """
    T, k = skew_flat.shape
    n = int((1 + (1 + 8 * k) ** 0.5) / 2 + 0.5)
    skew = build_skew(skew_flat, n)
    O = torch.linalg.matrix_exp(skew)

    R_run = torch.eye(n, dtype=O.dtype, device=O.device)
    state = torch.empty_like(O)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = O[start:end]
        # Within-chunk left-fold (sequential; the Triton kernel will do this
        # as a parallel Hillis-Steele scan).
        local_state = torch.empty_like(chunk)
        local_state[0] = chunk[0]
        for i in range(1, end - start):
            local_state[i] = chunk[i] @ local_state[i - 1]

        # Each chunk-local state composes with R_run on the right
        # (since the cumulative product is `O_t · … · O_0 = chunk · R_run`).
        state[start:end] = local_state @ R_run

        # Advance R_run by the chunk total.
        R_run = local_state[-1] @ R_run

    return state
