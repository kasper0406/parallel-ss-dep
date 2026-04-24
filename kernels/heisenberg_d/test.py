"""Correctness tests for the d-dimensional Heisenberg reference.

Runs on Mac (CPU). Compares `chunked` against `naive_loop` on random
inputs. This is the correctness anchor for the Triton kernel — the
kernel's outputs on GPU must match `chunked` to within fp atol.

fp64 is used throughout so the reference-vs-reference comparison can use
atol ~1e-10 (matmul reassociation is the only source of drift).
"""
from __future__ import annotations
import sys
import pathlib

# Let `python kernels/heisenberg_d/test.py` work without PYTHONPATH gymnastics.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from kernels.heisenberg_d.reference import naive_loop, chunked


def _check(T: int, d: int, chunk: int, seed: int = 0, atol: float = 1e-10):
    torch.manual_seed(seed)
    a = torch.randn(T, d, dtype=torch.float64)
    b = torch.randn(T, d, dtype=torch.float64)

    a_naive, b_naive, c_naive = naive_loop(a, b)
    a_chunk, b_chunk, c_chunk = chunked(a, b, chunk_size=chunk)

    da = (a_naive - a_chunk).abs().max().item()
    db = (b_naive - b_chunk).abs().max().item()
    dc = (c_naive - c_chunk).abs().max().item()

    worst = max(da, db, dc)
    assert worst < atol, (
        f"T={T} d={d} chunk={chunk}  Δa={da:.2e} Δb={db:.2e} Δc={dc:.2e}"
    )
    return worst


def main():
    # (T, d, chunk_size)
    cases = [
        (128, 16, 32),
        (256, 32, 64),
        (100, 24, 16),          # non-power-of-two T and chunk
        (97, 13, 17),            # all-odd
        (7, 4, 16),              # T < chunk
        (1, 8, 8),               # degenerate length-1
        (2, 8, 8),               # length-2 (first non-trivial cross-pair)
        (64, 8, 1),              # chunk=1: every step is its own chunk
        (65, 8, 64),             # chunk boundary just past the edge
        (512, 32, 128),          # larger
    ]
    for T, d, chunk in cases:
        diff = _check(T, d, chunk)
        print(f"heisenberg_d  T={T:4d}  d={d:3d}  chunk={chunk:3d}   max|Δ|={diff:.2e}")
    print("OK")


if __name__ == "__main__":
    main()
