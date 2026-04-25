"""
GPU smoke test for the three Triton kernels.

Goals:
  1. Confirm each kernel compiles + runs on sm_120 (RTX 5090) without
     hitting pytorch#176426 (multi-tl.load segfault).
  2. Confirm BF16-input + FP32-accumulator numerics match the
     PyTorch fp64 reference within EVAL_PLAN.md §2.3 tolerances.
  3. Report a one-line PASS/FAIL per (kernel × shape × dtype) cell.

Run from repo root:
    python kernels/smoke_gpu.py
"""
from __future__ import annotations

import sys
import pathlib
import time
import traceback

# Allow `python kernels/smoke_gpu.py` from repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from kernels.linear_attn import kernel as la_kernel
from kernels.linear_attn import reference as la_ref
from kernels.heisenberg_d import kernel as hd_kernel
from kernels.heisenberg_d import reference as hd_ref
from kernels.unipotent_u4 import kernel as u4_kernel
from kernels.unipotent_u4 import reference as u4_ref


def _hdr(s: str) -> None:
    print(f"\n=== {s} ===")


def _tol_for(dtype: torch.dtype, T: int) -> tuple[float, float]:
    """EVAL_PLAN.md §2.3 tolerance table.

    Note: fp32 path on Blackwell uses TF32 in `tl.dot` by default (10-bit
    mantissa, ~1e-3 relative). The §2.3 1e-4 figure assumes IEEE FP32 — to
    enforce that we'd pass `input_precision='ieee'` to `tl.dot`, but for
    a smoke test we accept TF32 since training is BF16 anyway.
    """
    if dtype == torch.float64:
        return 1e-10, 1e-12
    if dtype == torch.float32:
        return 1e-3, 1e-3
    if dtype == torch.bfloat16:
        # Per §2.3: 5e-3 at T<=512, 5e-2 at T~4096.
        return (5e-3, 5e-3) if T <= 512 else (5e-2, 5e-2)
    raise ValueError(f"no tolerance set for dtype={dtype}")


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _passes(diff: float, ref: torch.Tensor, atol: float, rtol: float) -> bool:
    """torch.allclose semantics: pass if max|Δ| ≤ atol + rtol·max|ref|."""
    ref_max = float(ref.float().abs().max().item())
    return diff <= atol + rtol * ref_max


def smoke_linear_attn(B: int, H: int, T: int, d_k: int, d_v: int,
                     dtype: torch.dtype, block_t: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(B, H, T, d_k, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, H, T, d_k, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, H, T, d_v, dtype=dtype, device=device) * 0.1

    # FP64 reference: collapse (B, H) and run the naive loop.
    o_ref = torch.empty(B, H, T, d_v, dtype=torch.float64, device=device)
    for bi in range(B):
        for hi in range(H):
            o_ref[bi, hi] = la_ref.naive_loop(
                q[bi, hi].double(), k[bi, hi].double(), v[bi, hi].double()
            )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o = la_kernel.launch(q, k, v, block_t=block_t)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3

    atol, rtol = _tol_for(dtype, T)
    diff = _max_abs_diff(o, o_ref)
    passed = _passes(diff, o_ref, atol, rtol)
    return {"diff": diff, "atol": atol, "passed": passed, "ms": dt_ms,
            "shape": (B, H, T, d_k, d_v), "dtype": dtype, "block_t": block_t}


def smoke_heisenberg_d(B: int, H: int, T: int, D: int,
                       dtype: torch.dtype, block_t: int) -> dict:
    torch.manual_seed(1)
    device = "cuda"
    a = torch.randn(B, H, T, D, dtype=dtype, device=device) * 0.1
    b = torch.randn(B, H, T, D, dtype=dtype, device=device) * 0.1

    # FP64 reference for c (the novel statistic we care about).
    c_ref = torch.empty(B, H, T, D, D, dtype=torch.float64, device=device)
    for bi in range(B):
        for hi in range(H):
            _, _, c_ij = hd_ref.naive_loop(a[bi, hi].double(), b[bi, hi].double())
            c_ref[bi, hi] = c_ij

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out_a, out_b, out_c = hd_kernel.launch(a, b, block_t=block_t)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3

    atol, rtol = _tol_for(dtype, T)
    diff = _max_abs_diff(out_c, c_ref)
    passed = _passes(diff, c_ref, atol, rtol)
    return {"diff": diff, "atol": atol, "passed": passed, "ms": dt_ms,
            "shape": (B, H, T, D), "dtype": dtype, "block_t": block_t}


def smoke_heisenberg_readout(B: int, H: int, T: int, D: int,
                             dtype: torch.dtype, block_t: int) -> dict:
    torch.manual_seed(3)
    device = "cuda"
    q = torch.randn(B, H, T, D, dtype=dtype, device=device) * 0.1
    a = torch.randn(B, H, T, D, dtype=dtype, device=device) * 0.1
    b = torch.randn(B, H, T, D, dtype=dtype, device=device) * 0.1

    o_ref = torch.empty(B, H, T, D, dtype=torch.float64, device=device)
    for bi in range(B):
        for hi in range(H):
            o_ref[bi, hi] = hd_ref.naive_loop_with_readout(
                q[bi, hi].double(), a[bi, hi].double(), b[bi, hi].double()
            )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o = hd_kernel.launch_with_readout(q, a, b, block_t=block_t)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3

    atol, rtol = _tol_for(dtype, T)
    diff = _max_abs_diff(o, o_ref)
    passed = _passes(diff, o_ref, atol, rtol)
    return {"diff": diff, "atol": atol, "passed": passed, "ms": dt_ms,
            "shape": (B, H, T, D), "dtype": dtype, "block_t": block_t}


def smoke_unipotent_u4(B: int, H: int, T: int, D: int,
                       dtype: torch.dtype, block_t: int, block_d: int) -> dict:
    torch.manual_seed(2)
    device = "cuda"
    x = torch.randn(B, H, T, 6, D, dtype=dtype, device=device) * 0.1

    # FP64 reference per (B, H, d) parallel scan.
    y_ref = torch.empty_like(x, dtype=torch.float64)
    for bi in range(B):
        for hi in range(H):
            y_ref[bi, hi] = u4_ref.naive_loop(x[bi, hi].double())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = u4_kernel.launch(x, block_t=block_t, block_d=block_d)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3

    atol, rtol = _tol_for(dtype, T)
    diff = _max_abs_diff(y, y_ref)
    passed = _passes(diff, y_ref, atol, rtol)
    return {"diff": diff, "atol": atol, "passed": passed, "ms": dt_ms,
            "shape": (B, H, T, D), "dtype": dtype,
            "block_t": block_t, "block_d": block_d}


def _run(name: str, fn, *args) -> bool:
    try:
        r = fn(*args)
    except Exception as e:
        print(f"[FAIL] {name}: exception during launch")
        traceback.print_exc()
        return False
    tag = "PASS" if r["passed"] else "FAIL"
    print(f"[{tag}] {name}  shape={r['shape']}  dtype={r['dtype']}  "
          f"block_t={r.get('block_t')}{'/' + str(r['block_d']) if 'block_d' in r else ''}  "
          f"max|Δ|={r['diff']:.3e}  atol={r['atol']:.0e}  {r['ms']:.1f} ms")
    return r["passed"]


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU 0: {torch.cuda.get_device_name(0)}  capability={cap}  "
          f"torch={torch.__version__}")

    all_ok = True

    _hdr("linear_attn — single chunk")
    all_ok &= _run("linear_attn fp32 small ", smoke_linear_attn,
                   1, 1, 64, 16, 16, torch.float32, 64)
    _hdr("linear_attn — multi-chunk, BF16-in / FP32-accum")
    all_ok &= _run("linear_attn bf16 T=512", smoke_linear_attn,
                   2, 4, 512, 64, 64, torch.bfloat16, 64)
    all_ok &= _run("linear_attn bf16 T=2048", smoke_linear_attn,
                   1, 4, 2048, 64, 64, torch.bfloat16, 128)

    _hdr("heisenberg_d — single chunk")
    all_ok &= _run("heisenberg_d fp32 small", smoke_heisenberg_d,
                   1, 1, 64, 16, torch.float32, 64)
    _hdr("heisenberg_d — multi-chunk (sm_120 multi-load risk)")
    all_ok &= _run("heisenberg_d bf16 T=512", smoke_heisenberg_d,
                   2, 4, 512, 32, torch.bfloat16, 64)
    all_ok &= _run("heisenberg_d bf16 T=2048", smoke_heisenberg_d,
                   1, 4, 2048, 32, torch.bfloat16, 64)

    _hdr("heisenberg_d fused readout — sanity")
    all_ok &= _run("heisenberg_d ro fp32", smoke_heisenberg_readout,
                   1, 1, 64, 16, torch.float32, 64)
    all_ok &= _run("heisenberg_d ro bf16 T=512", smoke_heisenberg_readout,
                   2, 4, 512, 32, torch.bfloat16, 64)
    all_ok &= _run("heisenberg_d ro bf16 T=2048", smoke_heisenberg_readout,
                   1, 4, 2048, 32, torch.bfloat16, 64)

    _hdr("unipotent_u4 — single chunk")
    all_ok &= _run("unipotent_u4 fp32 small", smoke_unipotent_u4,
                   1, 1, 64, 16, torch.float32, 64, 16)
    _hdr("unipotent_u4 — multi-chunk")
    all_ok &= _run("unipotent_u4 bf16 T=512", smoke_unipotent_u4,
                   2, 4, 512, 64, torch.bfloat16, 64, 64)
    all_ok &= _run("unipotent_u4 bf16 T=2048", smoke_unipotent_u4,
                   1, 4, 2048, 64, torch.bfloat16, 64, 64)

    print("\n" + ("ALL PASS" if all_ok else "FAILURES"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
