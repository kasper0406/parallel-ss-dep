"""Mac-runnable correctness tests for the SO(n) scan reference."""
from __future__ import annotations

import sys
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernels.ortho_son.reference import naive_loop, chunked


def test_one(T: int, n: int, chunk: int = 64, atol: float = 1e-10) -> None:
    torch.manual_seed(0)
    k = n * (n - 1) // 2
    skew_flat = torch.randn(T, k, dtype=torch.float64) * 0.1
    a = naive_loop(skew_flat)
    b = chunked(skew_flat, chunk_size=chunk)
    diff = (a - b).abs().max().item()
    assert diff < atol, f"T={T} n={n} chunk={chunk}: max|Δ|={diff:.2e}"
    # Verify orthogonality: each state is in SO(n).
    eye = torch.eye(n, dtype=torch.float64)
    for t in range(T):
        ortho_err = (a[t] @ a[t].T - eye).abs().max().item()
        assert ortho_err < 1e-10, f"non-orthogonal at t={t}: err={ortho_err:.2e}"
    print(f"ortho_son  T={T:4d}  n={n}  chunk={chunk:3d}   max|Δ|={diff:.2e}")


def main() -> None:
    test_one(T=128, n=4)
    test_one(T=256, n=4, chunk=32)
    test_one(T=100, n=3, chunk=16)
    test_one(T=7, n=4, chunk=16)
    test_one(T=1, n=4, chunk=8)
    test_one(T=512, n=8, chunk=128)
    print("OK")


if __name__ == "__main__":
    main()
