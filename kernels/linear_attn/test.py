"""Correctness tests for the linear-attention reference.

Runs on Mac (CPU). Compares `chunked` against `naive_loop` on random
inputs. This is the correctness anchor for the Triton kernel — the
kernel's outputs on GPU must match `chunked` to within fp atol.
"""
from __future__ import annotations
import sys
import pathlib

# Let `python kernels/linear_attn/test.py` work without PYTHONPATH gymnastics.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from kernels.linear_attn.reference import naive_loop, chunked


def _check(T: int, d_k: int, d_v: int, chunk: int, seed: int = 0, atol: float = 1e-5):
    torch.manual_seed(seed)
    q = torch.randn(T, d_k, dtype=torch.float64)
    k = torch.randn(T, d_k, dtype=torch.float64)
    v = torch.randn(T, d_v, dtype=torch.float64)
    o_naive = naive_loop(q, k, v)
    o_chunk = chunked(q, k, v, chunk_size=chunk)
    diff = (o_naive - o_chunk).abs().max().item()
    assert diff < atol, f"T={T} d_k={d_k} d_v={d_v} chunk={chunk} diff={diff}"
    return diff


def main():
    cases = [
        (128, 16, 16, 32),
        (256, 32, 32, 64),
        (100, 24, 24, 16),      # non-power-of-two T and chunk
        (7, 4, 4, 16),          # T < chunk
        (1, 8, 8, 8),           # degenerate length-1
        (512, 64, 64, 128),     # larger
    ]
    for T, d_k, d_v, chunk in cases:
        d = _check(T, d_k, d_v, chunk)
        print(f"linear_attn  T={T:4d}  d_k={d_k:3d}  d_v={d_v:3d}  chunk={chunk:3d}   max|Δ|={d:.2e}")
    print("OK")


if __name__ == "__main__":
    main()
