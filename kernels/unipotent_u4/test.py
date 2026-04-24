"""Correctness tests for the U_4 unipotent-scan reference.

Runs on Mac (CPU). Compares `chunked` against `naive_loop` on random
inputs. This is the correctness anchor for the Triton kernel — the
kernel's outputs on GPU must match `chunked` to within fp atol.

Includes a targeted test that checks the trilinear
`x12_i · x23_j · x34_k` cross-contribution to `x14` — the signature
non-trivial component of U_4 composition. A naive "add everything"
implementation would agree on the five linear/bilinear channels but
miss this triple product, so we verify it directly.
"""
from __future__ import annotations
import sys
import pathlib

# Let `python kernels/unipotent_u4/test.py` work without PYTHONPATH gymnastics.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from kernels.unipotent_u4.reference import (
    naive_loop,
    chunked,
    IDX_X12,
    IDX_X13,
    IDX_X14,
    IDX_X23,
    IDX_X24,
    IDX_X34,
)


def _check(T: int, d: int, chunk: int, seed: int = 0, atol: float = 1e-10) -> float:
    torch.manual_seed(seed)
    x = torch.randn(T, 6, d, dtype=torch.float64)
    y_naive = naive_loop(x)
    y_chunk = chunked(x, chunk_size=chunk)
    diff = (y_naive - y_chunk).abs().max().item()
    assert diff < atol, f"T={T} d={d} chunk={chunk} seed={seed} diff={diff}"
    return diff


def _check_trilinear_x14():
    """
    Targeted check: with a specially-crafted input where the *only*
    non-zero per-step channels are x12 at step 0, x23 at step 1, and
    x34 at step 2, the running x14 at step 2 should equal exactly
    x12_0 · x23_1 · x34_2 (the length-3 path through the monoid).

    This isolates the trilinear cross-contribution — it cannot be
    produced by simple additive accumulation or by any single bilinear
    term. An implementation that only sums components, or that only
    covers one of the two bilinear terms in the x14 line, will fail
    this test.
    """
    T, d = 5, 3
    x = torch.zeros(T, 6, d, dtype=torch.float64)

    # Three distinct values, one per "path position".
    a = torch.tensor([2.0, -3.0, 1.5], dtype=torch.float64)   # x12 at t=0
    b = torch.tensor([5.0,  0.5, -1.0], dtype=torch.float64)  # x23 at t=1
    c = torch.tensor([-1.0, 7.0,  2.0], dtype=torch.float64)  # x34 at t=2

    x[0, IDX_X12] = a
    x[1, IDX_X23] = b
    x[2, IDX_X34] = c

    y_naive = naive_loop(x)
    y_chunk = chunked(x, chunk_size=4)

    expected_x14 = a * b * c  # elementwise product over d
    got_naive = y_naive[2, IDX_X14]
    got_chunk = y_chunk[2, IDX_X14]

    d_naive = (got_naive - expected_x14).abs().max().item()
    d_chunk = (got_chunk - expected_x14).abs().max().item()
    assert d_naive < 1e-12, f"naive_loop missed trilinear term: {d_naive}"
    assert d_chunk < 1e-12, f"chunked missed trilinear term: {d_chunk}"

    # And make sure the *only* non-zero output channel at t=2 is x14
    # (everything else must be zero for this specific input, because
    # each of the three active input channels appears at only one step
    # and they live in *different* channels).
    for k in (IDX_X12, IDX_X13, IDX_X23, IDX_X24, IDX_X34):
        if k == IDX_X12:
            # Running x12 at t=2 should be a (only step 0 non-zero).
            assert (y_chunk[2, k] - a).abs().max().item() < 1e-12
        elif k == IDX_X23:
            # Running x23 at t=2 should be b.
            assert (y_chunk[2, k] - b).abs().max().item() < 1e-12
        elif k == IDX_X34:
            # Running x34 at t=2 should be c.
            assert (y_chunk[2, k] - c).abs().max().item() < 1e-12
        else:
            # x13 at t=2: a12_0 * b23_1 = a * b (the only non-zero bilinear path).
            # x24 at t=2: a23_1 * b34_2 = b * c.
            if k == IDX_X13:
                assert (y_chunk[2, k] - a * b).abs().max().item() < 1e-12
            elif k == IDX_X24:
                assert (y_chunk[2, k] - b * c).abs().max().item() < 1e-12

    # Also verify across a chunk boundary (chunk_size < 3 wouldn't be
    # a valid power-of-two-scan input, so use chunk=2 which forces
    # step 2 into the second chunk and exercises the cross-chunk
    # combine's trilinear propagation).
    y_chunk_boundary = chunked(x, chunk_size=2)
    d_boundary = (y_chunk_boundary[2, IDX_X14] - expected_x14).abs().max().item()
    assert d_boundary < 1e-12, (
        f"chunk-boundary trilinear term wrong: {d_boundary}"
    )
    return max(d_naive, d_chunk, d_boundary)


def main():
    cases = [
        # (T, d, chunk)
        (128, 16, 32),
        (256, 32, 64),
        (100, 24, 16),     # non-power-of-two T and chunk
        (7, 4, 16),        # T < chunk
        (1, 8, 8),         # degenerate length-1
        (3, 8, 2),         # T straddling a small chunk boundary
        (13, 5, 4),        # odd T, odd d, small chunk
        (512, 64, 128),    # larger
    ]
    for T, d, chunk in cases:
        diff = _check(T, d, chunk)
        print(
            f"unipotent_u4  T={T:4d}  d={d:3d}  chunk={chunk:3d}   "
            f"max|Δ|={diff:.2e}"
        )

    # Targeted trilinear-propagation check.
    diff_tri = _check_trilinear_x14()
    print(f"unipotent_u4  trilinear x14 cross-term check                max|Δ|={diff_tri:.2e}")

    print("OK")


if __name__ == "__main__":
    main()
