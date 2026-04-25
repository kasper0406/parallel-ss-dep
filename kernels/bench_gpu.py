"""
Throughput micro-benchmark for the three Triton kernels.

Goals:
  1. Wall-clock ms/iter for each kernel across realistic (B, H, T, D) shapes.
  2. Tokens/sec and effective memory-bandwidth (input + state-output bytes
     ÷ time) per shape.
  3. Ratio vs the `linear_attn` baseline at matched (B, H, T) — the
     kill-gate comparison is only apples-to-apples if these ratios are
     known and reasonable (≤ ~5x).
  4. Reference: 5090 BF16 peak ≈ 209 TFLOPS, HBM peak ≈ 1.8 TB/s.

Usage:
    python kernels/bench_gpu.py                # default shape grid
    python kernels/bench_gpu.py --quick        # tiny grid, smoke
    python kernels/bench_gpu.py --csv out.csv  # also dump CSV
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
import sys
import time
from typing import Any, Callable

# Allow running from repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from kernels.linear_attn import kernel as la_kernel
from kernels.heisenberg_d import kernel as hd_kernel
from kernels.unipotent_u4 import kernel as u4_kernel


# ---------------------------- timing helpers ---------------------------------


def _time_kernel(fn: Callable[[], Any], iters: int, warmup: int) -> list[float]:
    """Run `fn` `warmup + iters` times; return ms per iter for the iters."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


# ---------------------------- per-kernel runners -----------------------------


@dataclasses.dataclass
class Bench:
    name: str
    shape: dict
    dtype: torch.dtype
    median_ms: float
    p10_ms: float
    p90_ms: float
    tokens_per_sec: float
    in_gbps: float
    state_out_gb: float
    notes: str = ""


def _bench_linear_attn(B: int, H: int, T: int, D: int, dtype: torch.dtype,
                       block_t: int, iters: int, warmup: int) -> Bench:
    q = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    k = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    v = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    fn = lambda: la_kernel.launch(q, k, v, block_t=block_t)
    ms = _time_kernel(fn, iters, warmup)
    median = _percentile(ms, 0.5)
    bytes_in = (q.numel() + k.numel() + v.numel()) * q.element_size()
    bytes_state_out = B * H * T * D * 2  # bf16 output
    return Bench(
        name="linear_attn",
        shape={"B": B, "H": H, "T": T, "D_K": D, "D_V": D, "BLK_T": block_t},
        dtype=dtype,
        median_ms=median,
        p10_ms=_percentile(ms, 0.1),
        p90_ms=_percentile(ms, 0.9),
        tokens_per_sec=(B * H * T) / (median * 1e-3),
        in_gbps=bytes_in / (median * 1e-3) / 1e9,
        state_out_gb=bytes_state_out / 1e9,
    )


def _bench_heisenberg_d(B: int, H: int, T: int, D: int, dtype: torch.dtype,
                        block_t: int, iters: int, warmup: int) -> Bench:
    a = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    b = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    fn = lambda: hd_kernel.launch(a, b, block_t=block_t)
    ms = _time_kernel(fn, iters, warmup)
    median = _percentile(ms, 0.5)
    bytes_in = (a.numel() + b.numel()) * a.element_size()
    # Output: out_a, out_b in bf16 + out_c in fp32 (the dominant term).
    bytes_state_out = B * H * T * D * 2 * 2 + B * H * T * D * D * 4
    return Bench(
        name="heisenberg_d",
        shape={"B": B, "H": H, "T": T, "D": D, "BLK_T": block_t},
        dtype=dtype,
        median_ms=median,
        p10_ms=_percentile(ms, 0.1),
        p90_ms=_percentile(ms, 0.9),
        tokens_per_sec=(B * H * T) / (median * 1e-3),
        in_gbps=bytes_in / (median * 1e-3) / 1e9,
        state_out_gb=bytes_state_out / 1e9,
    )


def _bench_heisenberg_readout(B: int, H: int, T: int, D: int, dtype: torch.dtype,
                              block_t: int, iters: int, warmup: int) -> Bench:
    q = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    a = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    b = torch.randn(B, H, T, D, dtype=dtype, device="cuda") * 0.1
    fn = lambda: hd_kernel.launch_with_readout(q, a, b, block_t=block_t)
    ms = _time_kernel(fn, iters, warmup)
    median = _percentile(ms, 0.5)
    bytes_in = (q.numel() + a.numel() + b.numel()) * q.element_size()
    bytes_state_out = B * H * T * D * 2  # bf16 D-vector readout
    return Bench(
        name="heisenberg_ro",
        shape={"B": B, "H": H, "T": T, "D": D, "BLK_T": block_t},
        dtype=dtype,
        median_ms=median,
        p10_ms=_percentile(ms, 0.1),
        p90_ms=_percentile(ms, 0.9),
        tokens_per_sec=(B * H * T) / (median * 1e-3),
        in_gbps=bytes_in / (median * 1e-3) / 1e9,
        state_out_gb=bytes_state_out / 1e9,
    )


def _bench_unipotent_u4(B: int, H: int, T: int, D: int, dtype: torch.dtype,
                        block_t: int, block_d: int, iters: int, warmup: int) -> Bench:
    x = torch.randn(B, H, T, 6, D, dtype=dtype, device="cuda") * 0.1
    fn = lambda: u4_kernel.launch(x, block_t=block_t, block_d=block_d)
    ms = _time_kernel(fn, iters, warmup)
    median = _percentile(ms, 0.5)
    bytes_in = x.numel() * x.element_size()
    bytes_state_out = B * H * T * 6 * D * 4  # fp32 output
    return Bench(
        name="unipotent_u4",
        shape={"B": B, "H": H, "T": T, "D": D, "BLK_T": block_t, "BLK_D": block_d},
        dtype=dtype,
        median_ms=median,
        p10_ms=_percentile(ms, 0.1),
        p90_ms=_percentile(ms, 0.9),
        tokens_per_sec=(B * H * T) / (median * 1e-3),
        in_gbps=bytes_in / (median * 1e-3) / 1e9,
        state_out_gb=bytes_state_out / 1e9,
    )


# ---------------------------- driver -----------------------------------------


def _print_row(b: Bench, base_ms: float | None) -> None:
    ratio = f"{b.median_ms / base_ms:5.2f}x" if base_ms is not None else "    -"
    shape_s = " ".join(f"{k}={v}" for k, v in b.shape.items())
    print(f"{b.name:<14}  {shape_s:<46}  "
          f"{b.median_ms:7.2f} ms  "
          f"({b.p10_ms:6.2f}-{b.p90_ms:6.2f})  "
          f"{b.tokens_per_sec/1e6:6.2f} Mtok/s  "
          f"in={b.in_gbps:6.1f} GB/s  "
          f"vs la {ratio}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="tiny shape grid")
    p.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  capability={cap}  "
          f"torch={torch.__version__}\n"
          f"(reference: 5090 BF16 peak ≈ 209 TFLOPS, HBM peak ≈ 1.8 TB/s)\n")

    # Shape grid. Default targets are realistic for the kill-gate (~30M params,
    # T~512-2048) and the SmolLM2-135M distillation phase (T=2048-4096).
    if args.quick:
        bhtds = [
            (1, 4, 512, 32),
            (1, 4, 2048, 64),
        ]
    else:
        bhtds = [
            # (B, H, T, D)
            (1, 4, 512, 32),    # MQAR-like
            (4, 4, 512, 32),    # batch 4
            (1, 8, 1024, 64),   # mid
            (4, 8, 2048, 64),   # SmolLM2 ballpark
            (1, 16, 4096, 64),  # long-context probe
            (1, 16, 4096, 32),  # long-context, narrow head
        ]

    rows: list[Bench] = []
    print(f"{'kernel':<14}  {'shape':<46}  {'median':>7}      "
          f"{'p10-p90':<13}  {'tok/s':>13}  {'data-in':>13}  {'vs LA':>8}")
    print("-" * 140)

    for (B, H, T, D) in bhtds:
        # Match (B, H, T) across kernels; use BLOCK_T=64 by default (autotune later).
        la = _bench_linear_attn(B, H, T, D, torch.bfloat16, 64,
                                args.iters, args.warmup)
        rows.append(la)
        _print_row(la, base_ms=None)

        hd = _bench_heisenberg_d(B, H, T, D, torch.bfloat16, 64,
                                 args.iters, args.warmup)
        rows.append(hd)
        _print_row(hd, base_ms=la.median_ms)

        hr = _bench_heisenberg_readout(B, H, T, D, torch.bfloat16, 64,
                                       args.iters, args.warmup)
        rows.append(hr)
        _print_row(hr, base_ms=la.median_ms)

        u4 = _bench_unipotent_u4(B, H, T, D, torch.bfloat16, 64, 64,
                                 args.iters, args.warmup)
        rows.append(u4)
        _print_row(u4, base_ms=la.median_ms)
        print()

    if args.csv:
        path = pathlib.Path(args.csv)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "shape", "dtype", "median_ms",
                        "p10_ms", "p90_ms", "tokens_per_sec", "in_gbps",
                        "state_out_gb"])
            for r in rows:
                w.writerow([r.name, str(r.shape), str(r.dtype), r.median_ms,
                            r.p10_ms, r.p90_ms, r.tokens_per_sec, r.in_gbps,
                            r.state_out_gb])
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
