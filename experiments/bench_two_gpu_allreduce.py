"""Two-GPU manual-allreduce validation (IDEAS Tier-0, systems lens idea 1).

Measures, WITHOUT DistributedDataParallel (whose autograd hooks are what
break with the latent path):
  1. raw all_reduce bandwidth for a flat bf16 grad-sized bucket (574-804 MB
     at 287-402M params) over the rig's PCIe (gloo AND nccl backends) — the
     single number that decides manual-allreduce vs DiLoCo;
  2. overlap potential: bucketed reduction latency vs one flat tensor.

Run when BOTH GPUs are free:
  PYTHONPATH=. .venv/bin/python experiments/bench_two_gpu_allreduce.py \
      --params_m 402 --backend nccl
(spawns 2 processes itself; ~2 min). Decision rule (pre-registered): if
all_reduce of the full grad takes < ~15% of a 5s step even UNoverlapped
(i.e. < 750ms), manual bucketed allreduce wins and DiLoCo is unnecessary;
if > 50%, go DiLoCo (H=32 inner steps). Between: overlap engineering call.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int, params_m: float, backend: str,
           iters: int, bucket_mb: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29571"
    dist.init_process_group(backend, rank=rank, world_size=world)
    dev = f"cuda:{rank}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
    n_elem = int(params_m * 1e6)
    flat = torch.randn(n_elem, dtype=torch.bfloat16, device=dev)

    # warmup
    for _ in range(3):
        dist.all_reduce(flat)
    if backend == "nccl":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(flat)
    if backend == "nccl":
        torch.cuda.synchronize()
    dt_flat = (time.perf_counter() - t0) / iters

    # bucketed (as backward-overlap would issue them)
    n_bucket = max(1, (2 * n_elem) // (bucket_mb * 1024 * 1024))
    chunks = list(torch.chunk(flat, n_bucket))
    t0 = time.perf_counter()
    for _ in range(iters):
        for c in chunks:
            dist.all_reduce(c)
    if backend == "nccl":
        torch.cuda.synchronize()
    dt_bucket = (time.perf_counter() - t0) / iters

    if rank == 0:
        gb = 2 * n_elem / 1e9
        print(f"[{backend}] flat {gb:.2f} GB: {dt_flat*1e3:.1f} ms "
              f"({gb/dt_flat:.1f} GB/s algo-bw)  |  "
              f"{n_bucket} x {bucket_mb}MB buckets: {dt_bucket*1e3:.1f} ms")
        step_ms = 5000.0
        frac = dt_flat * 1e3 / step_ms
        verdict = ("MANUAL-ALLREDUCE (comm trivial)" if frac < 0.15 else
                   "DiLoCo (comm dominates)" if frac > 0.5 else
                   "OVERLAP-ENGINEERING (in between)")
        print(f"  unoverlapped comm = {100*frac:.1f}% of a 5s step -> {verdict}")
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params_m", type=float, default=402.0)
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--bucket_mb", type=int, default=64)
    args = ap.parse_args()
    mp.spawn(worker, args=(2, args.params_m, args.backend, args.iters,
                           args.bucket_mb), nprocs=2, join=True)


if __name__ == "__main__":
    main()
