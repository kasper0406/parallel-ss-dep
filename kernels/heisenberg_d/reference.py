"""
d-dimensional Heisenberg monoid scan — PyTorch reference implementations.

The Heisenberg monoid element is a triple `(a, b, c)` with `a, b ∈ R^d` and
`c ∈ R^{d×d}`. Composition (see `StateDep/StateDep/HeisenbergD.lean`):

    (a₁, b₁, c₁) * (a₂, b₂, c₂) = (a₁ + a₂, b₁ + b₂, c₁ + c₂ + a₁ ⊗ b₂)

where `(a₁ ⊗ b₂)_{ij} = a₁_i · b₂_j`. The operation is associative but
*not* commutative — the cross term `a₁ ⊗ b₂` sees the left operand's `a`
and the right operand's `b`, so order matters. The identity is `(0, 0, 0)`.

Given an input sequence of "atoms" `[(a_t, b_t, 0)]_{t=1..T}`, the inclusive
scan produces, at step t:

    running_a[t] = Σ_{i ≤ t} a_i
    running_b[t] = Σ_{i ≤ t} b_i
    running_c[t] = Σ_{i < j ≤ t} a_i ⊗ b_j        ← strict `i < j`

The strict inequality is the novel content: `c_T` is the causal
"cross-pair outer product" statistic. A neural cell consumes `(a_t, b_t)`
as per-token features and reads off `c_t` (or `q_t @ c_t`) like linear
attention reads its KV state.

Two implementations live here:

  naive_loop : exact T-step Python loop. Ground truth. O(T·d²).
  chunked    : chunkwise form mirroring the Triton kernel's blocking.
               Same complexity, organised into dense GEMMs on (C, d) and
               (d, d) tiles so it maps onto tensor cores.

Both are fp64-safe and shape-general in `d`, `T`.
"""
from __future__ import annotations
import torch
from torch import Tensor


def naive_loop(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    a, b: (T, d). Returns (running_a, running_b, running_c) with shapes
    (T, d), (T, d), (T, d, d).

    Indexing convention — MUST match the chunked form exactly:

        running_a starts at 0 and is updated *after* writing c[t], so
        at the point c[t] is written, `running_a` still equals Σ_{i<t} a_i.
        Therefore out_c[t] = Σ_{i<j≤t} a_i ⊗ b_j (strict `i<j`).

        out_a[t] = Σ_{i≤t} a_i (inclusive on a — set after the update).
        out_b[t] = Σ_{i≤t} b_i (inclusive on b — trivial cumsum).
    """
    T, d = a.shape
    running_a = torch.zeros(d, dtype=a.dtype, device=a.device)
    running_b = torch.zeros(d, dtype=b.dtype, device=b.device)
    c = torch.zeros(d, d, dtype=a.dtype, device=a.device)

    out_a = torch.empty(T, d, dtype=a.dtype, device=a.device)
    out_b = torch.empty(T, d, dtype=b.dtype, device=b.device)
    out_c = torch.empty(T, d, d, dtype=a.dtype, device=a.device)

    for t in range(T):
        # Cross-term contribution from pairs (i, t) with i < t:
        # before updating running_a we have running_a = Σ_{i<t} a_i, so this
        # adds Σ_{i<t} a_i ⊗ b_t to c — exactly the missing (·, t) pairs.
        c = c + torch.outer(running_a, b[t])
        # Now fold in a_t, b_t into the running sums.
        running_a = running_a + a[t]
        running_b = running_b + b[t]

        out_a[t] = running_a
        out_b[t] = running_b
        out_c[t] = c
    return out_a, out_b, out_c


def chunked(
    a: Tensor, b: Tensor, chunk_size: int = 64
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Chunkwise form, matching the Triton kernel's blocking exactly.

    Carry across chunks: `prev_A ∈ R^d` (= Σ_{i<chunk_start} a_i) and
    `prev_C ∈ R^{d×d}` (= Σ_{i<j<chunk_start} a_i ⊗ b_j).

    Within a chunk of length C, the running c_t at the global index
    `t = chunk_start + l` covers all pairs `(i, j)` with `i < j ≤ t`:

        c_t = Σ_{i<j≤t} a_i ⊗ b_j
            = prev_C                                                 (pairs with j < chunk_start)
            + prev_A ⊗ (Σ_{j'≤l} b_c[j'])                            (pairs with i<chunk_start, j≥chunk_start)
            + Σ_{j'≤l, s<j'} a_c[s] ⊗ b_c[j']                        (pairs with both i,j in chunk)

    Rewriting the last two lines in chunk-local prefix-sum form:

        B_incl[l]  = Σ_{j'≤l} b_c[j']                                (inclusive cumsum of b)
        A_excl[j'] = Σ_{s<j'} a_c[s]                                 (exclusive cumsum of a)
        C_intra[l] = Σ_{j'≤l} A_excl[j'] ⊗ b_c[j']                   (inclusive "Heisenberg cumsum")

    so the per-token c output is

        out_c[chunk_start + l] = prev_C + prev_A ⊗ B_incl[l] + C_intra[l]

    and after the chunk the carries advance as

        prev_C += prev_A ⊗ (Σ_t b_c[t]) + C_intra[C-1]               (= full-chunk Heisenberg update)
        prev_A += Σ_t a_c[t]

    `C_intra` is itself a Heisenberg scan at block level; we compute it
    directly with an inclusive cumsum over the (C, d, d) tensor
    `A_excl[:, :, None] * b_c[:, None, :]`. On GPU the Triton kernel
    computes the same thing as a small sequential scan within the chunk
    (`BLOCK_T` steps of a rank-1 update). The final block-level aggregate
    `C_intra[C-1]` equals the `(d, C)·(C, d)` GEMM `A_exclᵀ @ b_c` — used
    for advancing `prev_C`.
    """
    T, d = a.shape
    prev_A = torch.zeros(d, dtype=a.dtype, device=a.device)
    prev_B = torch.zeros(d, dtype=b.dtype, device=b.device)
    prev_C = torch.zeros(d, d, dtype=a.dtype, device=a.device)

    out_a = torch.empty(T, d, dtype=a.dtype, device=a.device)
    out_b = torch.empty(T, d, dtype=b.dtype, device=b.device)
    out_c = torch.empty(T, d, d, dtype=a.dtype, device=a.device)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        ac = a[start:end]                                       # (C, d)
        bc = b[start:end]                                       # (C, d)

        # Exclusive prefix sum of `a` inside the chunk: A_excl[t] = Σ_{s<t} a_c[s].
        # torch.cumsum is inclusive, so subtract `ac` to make it exclusive.
        A_excl = torch.cumsum(ac, dim=0) - ac                   # (C, d)
        # Inclusive prefix sums for the running_a / running_b outputs and
        # for the inter-chunk inclusive `b` cumsum used in the c output.
        A_incl = torch.cumsum(ac, dim=0)                        # (C, d)
        B_incl = torch.cumsum(bc, dim=0)                        # (C, d)

        # Intra-chunk per-token cross contributions, as a rank-1-update cumsum:
        #   C_intra[l]  = Σ_{j'≤l} A_excl[j'] ⊗ b_c[j']
        # shape (C, d, d). For each j', the outer product A_excl[j'] ⊗ b_c[j']
        # is a (d, d) rank-1 matrix; cumsum along the chunk axis gives the
        # strict-`i<j` running Heisenberg statistic within the chunk.
        rank1 = A_excl.unsqueeze(-1) * bc.unsqueeze(-2)         # (C, d, d)
        C_intra = torch.cumsum(rank1, dim=0)                    # (C, d, d)

        # Inter-chunk term per token: prev_A ⊗ B_incl[l]. (C, d, d) via broadcast.
        inter = prev_A.unsqueeze(0).unsqueeze(-1) * B_incl.unsqueeze(-2)   # (C, d, d)

        out_c[start:end] = prev_C.unsqueeze(0) + inter + C_intra
        out_a[start:end] = prev_A.unsqueeze(0) + A_incl
        out_b[start:end] = prev_B.unsqueeze(0) + B_incl

        # Advance carries by the full chunk:
        #   prev_C  += prev_A ⊗ ΣB_chunk + C_intra[-1]
        # `C_intra[-1]` equals `A_exclᵀ @ bc` — the same `(d, C)·(C, d)` GEMM
        # flash-linear-attention uses to advance its KV state.
        chunk_a_sum = A_incl[-1]                                # (d,)
        chunk_b_sum = B_incl[-1]                                # (d,)

        prev_C = prev_C + torch.outer(prev_A, chunk_b_sum) + C_intra[-1]
        prev_A = prev_A + chunk_a_sum
        prev_B = prev_B + chunk_b_sum

    return out_a, out_b, out_c
