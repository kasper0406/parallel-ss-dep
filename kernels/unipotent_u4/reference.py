"""
U_4 unipotent scan as a monoid scan — PyTorch reference implementations.

The monoid is the 4×4 upper-triangular unipotent group `U_4(R)`, whose
elements are parametrised by six scalars — the strictly-above-diagonal
entries of the matrix — arranged as `(x12, x13, x14, x23, x24, x34)`.
Composition is (`*` is scalar multiply in R):

    (M1 * M2).x12 = M1.x12 + M2.x12
    (M1 * M2).x23 = M1.x23 + M2.x23
    (M1 * M2).x34 = M1.x34 + M2.x34
    (M1 * M2).x13 = M1.x13 + M2.x13 + M1.x12 · M2.x23
    (M1 * M2).x24 = M1.x24 + M2.x24 + M1.x23 · M2.x34
    (M1 * M2).x14 = M1.x14 + M2.x14 + M1.x12 · M2.x24 + M1.x13 · M2.x34

This is non-commutative but associative; Lean proof lives at
`StateDep/StateDep/Unipotent.lean`. After an inclusive scan of length T,
the (1,4) slot carries the trilinear cross-contribution
`Σ_{i<j<k} x12_i · x23_j · x34_k` (plus bilinear pair terms — those
come from the `x12·x24` and `x13·x34` products in the combine).

Vectorisation strategy: we run `d` independent U_4 scans in parallel
across the last axis, so the six "state scalars" become `(d,)` vectors
and every multiplication is elementwise. Inputs and outputs have shape
`(T, 6, d)` where axis 1 indexes the six components in the canonical
order `[x12, x13, x14, x23, x24, x34]`.

The two reference implementations are:

  naive_loop : exact T-step Python loop. Ground truth. O(T·d).
  chunked    : chunkwise form mirroring what the Triton kernel does.
               Within a chunk of `chunk_size` we run a Hillis-Steele
               associative scan over six scalar channels (log-depth,
               and — crucially for U_4 — it only ever combines
               `(earlier, later)` which is the argument order the
               non-commutative monoid requires). Between chunks we
               combine the chunk summary with a running cross-chunk
               state via one U_4 multiplication. Same O(T·d) complexity
               as the naive loop but the intra-chunk work has
               O(log chunk) depth — which is what the kernel exploits.

These must agree numerically on any input. `test.py` verifies that.
"""
from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor

from kernels.shared.scan import hillis_steele_scan


# Canonical channel ordering in the (T, 6, d) layout.
IDX_X12 = 0
IDX_X13 = 1
IDX_X14 = 2
IDX_X23 = 3
IDX_X24 = 4
IDX_X34 = 5


def _combine_six(
    a: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    b: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    U_4 monoid product, channel-by-channel, broadcast elementwise over
    the trailing `d` axis. Each of `a`, `b` is a 6-tuple
    `(x12, x13, x14, x23, x24, x34)` of `(d,)` tensors (any shape works
    as long as both operands broadcast).
    """
    a12, a13, a14, a23, a24, a34 = a
    b12, b13, b14, b23, b24, b34 = b

    # Additive channels (length-1 paths).
    c12 = a12 + b12
    c23 = a23 + b23
    c34 = a34 + b34

    # Length-2 paths (bilinear cross-terms).
    c13 = a13 + b13 + a12 * b23
    c24 = a24 + b24 + a23 * b34

    # Length-3 path lives here. The trilinear `a12 · b23 · c34`
    # correlation emerges *across two consecutive combines* — within a
    # single combine only these two bilinear pair-terms appear, and
    # chaining them is exactly what reproduces the triple sum.
    c14 = a14 + b14 + a12 * b24 + a13 * b34

    return (c12, c13, c14, c23, c24, c34)


def naive_loop(x: Tensor) -> Tensor:
    """
    x : (T, 6, d). Each x[t, k, :] is the per-channel scalar values at
    step t for component k in the canonical order
        [x12, x13, x14, x23, x24, x34].

    Returns y of the same shape (T, 6, d) containing the *inclusive*
    running U_4 scan, i.e. y[t] = x[0] * x[1] * ... * x[t].

    Plain T-step Python loop. Unambiguously correct, slow.
    """
    T, six, d = x.shape
    assert six == 6, f"expected 6 channels, got {six}"

    out = torch.empty_like(x)
    # Running accumulator as six (d,) tensors.
    s12 = torch.zeros(d, dtype=x.dtype, device=x.device)
    s13 = torch.zeros(d, dtype=x.dtype, device=x.device)
    s14 = torch.zeros(d, dtype=x.dtype, device=x.device)
    s23 = torch.zeros(d, dtype=x.dtype, device=x.device)
    s24 = torch.zeros(d, dtype=x.dtype, device=x.device)
    s34 = torch.zeros(d, dtype=x.dtype, device=x.device)

    for t in range(T):
        a12, a13, a14, a23, a24, a34 = s12, s13, s14, s23, s24, s34
        b12 = x[t, IDX_X12]
        b13 = x[t, IDX_X13]
        b14 = x[t, IDX_X14]
        b23 = x[t, IDX_X23]
        b24 = x[t, IDX_X24]
        b34 = x[t, IDX_X34]

        # U_4 combine, written out channel-by-channel.
        s12 = a12 + b12
        s23 = a23 + b23
        s34 = a34 + b34
        s13 = a13 + b13 + a12 * b23
        s24 = a24 + b24 + a23 * b34
        s14 = a14 + b14 + a12 * b24 + a13 * b34

        out[t, IDX_X12] = s12
        out[t, IDX_X13] = s13
        out[t, IDX_X14] = s14
        out[t, IDX_X23] = s23
        out[t, IDX_X24] = s24
        out[t, IDX_X34] = s34

    return out


def chunked(x: Tensor, chunk_size: int = 64) -> Tensor:
    """
    Chunkwise scan — mirrors the Triton kernel's blocking strategy.

    For each chunk of `chunk_size` consecutive tokens:

      1. Build a list of 6-tuples (per-step elements along the chunk).
      2. Run an O(log C) Hillis-Steele associative scan with the U_4
         combine — giving the inclusive chunk-local scan. (A Blelloch
         sweep would also work, but we use Hillis-Steele here because
         it places the earlier element on the left of every combine,
         which is the argument order a non-commutative monoid like
         U_4 demands — see the monoid-correctness note in `kernel.py`.)
      3. Left-multiply each chunk-local partial product by the
         running cross-chunk state `S`: that is `S * local_t`, one
         scalar U_4 combine per token (and elementwise across `d`).
      4. Advance `S` by composing with the chunk's *total* product
         (the last entry of the chunk-local scan).

    All multiplications are scalar-elementwise over the trailing `d`
    axis. No GEMMs — the parallelism is over `d` independent U_4 scans.
    """
    T, six, d = x.shape
    assert six == 6, f"expected 6 channels, got {six}"

    out = torch.empty_like(x)

    # Running cross-chunk state S — six (d,) tensors starting at identity (0).
    S12 = torch.zeros(d, dtype=x.dtype, device=x.device)
    S13 = torch.zeros(d, dtype=x.dtype, device=x.device)
    S14 = torch.zeros(d, dtype=x.dtype, device=x.device)
    S23 = torch.zeros(d, dtype=x.dtype, device=x.device)
    S24 = torch.zeros(d, dtype=x.dtype, device=x.device)
    S34 = torch.zeros(d, dtype=x.dtype, device=x.device)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        C = end - start

        # Pull out the chunk as a list of 6-tuples, one per step.
        chunk_steps = []
        for t in range(start, end):
            chunk_steps.append(
                (
                    x[t, IDX_X12], x[t, IDX_X13], x[t, IDX_X14],
                    x[t, IDX_X23], x[t, IDX_X24], x[t, IDX_X34],
                )
            )

        # Log-depth associative scan inside the chunk. Hillis-Steele only
        # ever composes (earlier, later) → later — exactly the argument
        # order U_4's non-commutative combine requires.
        local_scan = hillis_steele_scan(chunk_steps, _combine_six)

        # Prefix each local partial product by the running state S, and
        # write to `out`.
        for i, local in enumerate(local_scan):
            g12, g13, g14, g23, g24, g34 = _combine_six(
                (S12, S13, S14, S23, S24, S34), local
            )
            t = start + i
            out[t, IDX_X12] = g12
            out[t, IDX_X13] = g13
            out[t, IDX_X14] = g14
            out[t, IDX_X23] = g23
            out[t, IDX_X24] = g24
            out[t, IDX_X34] = g34

        # Advance S by the chunk total (the last entry of the inclusive
        # local scan — which is the U_4 product of all C chunk steps).
        total = local_scan[-1]
        S12, S13, S14, S23, S24, S34 = _combine_six(
            (S12, S13, S14, S23, S24, S34), total
        )

    return out
