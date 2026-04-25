"""
Profile hybrid vs DeltaNet at SmolLM2-135M scale.

The point: decide whether the current PyTorch ortho layer (sequential
matrix_exp + left-fold) is fast enough at ~135M params for distillation,
or whether we need to wire the Triton kernel + custom autograd first.

SmolLM2-135M architecture (HuggingFaceTB/SmolLM2-135M):
  d_model=576, n_heads=9, d_head=64, n_layers=30, vocab=49152.

Profile setups:
  - Pure DeltaNet (the strongest single-cell baseline)
  - Hybrid [ortho, deltanet, ortho, deltanet] × 7 + 2 = 30 layers
  - Pure ortho (for comparison)

Reports forward + backward time per step at T=512, batch=4.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import (
    DeltaNetAttention, DeltaNetNegEigAttention,
    OrthogonalScanAttention,
)
from experiments.model import TinyLM


def profile_model(model, input_ids, n_warmup=3, n_iter=10) -> tuple[float, float, int]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Warmup.
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = logits.float().sum()       # dummy loss
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    # Forward + backward time.
    t0 = time.perf_counter()
    fwd_time, bwd_time = 0.0, 0.0
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        logits = model(input_ids)
        loss = logits.float().sum()
        torch.cuda.synchronize()
        fwd_time += time.perf_counter() - t_fwd_start

        torch.cuda.synchronize()
        t_bwd_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time += time.perf_counter() - t_bwd_start
        opt.step()
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    n_params = model.num_params()
    return fwd_time / n_iter * 1e3, bwd_time / n_iter * 1e3, n_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--n_iter", type=int, default=5)
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Profile shape: T={args.T} batch={args.batch} d_model={args.d_model} "
          f"n_heads={args.n_heads} d_head={args.d_head} n_layers={args.n_layers}")
    print(f"vocab={args.vocab} (smaller than SmolLM2's 49152 to fit memory)\n")

    torch.manual_seed(0)
    input_ids = torch.randint(0, args.vocab, (args.batch, args.T), device="cuda")

    # 1. Pure DeltaNet (baseline).
    print("--- pure DeltaNet ---")
    torch.manual_seed(0)
    m = TinyLM(
        vocab_size=args.vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head,
        attention_cls=DeltaNetAttention,
    ).to("cuda")
    fwd, bwd, n_p = profile_model(m, input_ids, n_iter=args.n_iter)
    print(f"  params={n_p/1e6:.1f}M  fwd={fwd:.1f} ms  bwd={bwd:.1f} ms  "
          f"step={(fwd+bwd):.1f} ms")
    del m; torch.cuda.empty_cache()

    # 2. Hybrid [ortho, deltanet, ortho, deltanet, ...] alternating.
    print("\n--- hybrid alternating [ortho, deltanet] ---")
    torch.manual_seed(0)
    cls_per_layer = []
    for i in range(args.n_layers):
        cls_per_layer.append(OrthogonalScanAttention if i % 2 == 0 else DeltaNetAttention)
    m = TinyLM(
        vocab_size=args.vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head,
        attention_cls_per_layer=cls_per_layer,
    ).to("cuda")
    fwd, bwd, n_p = profile_model(m, input_ids, n_iter=args.n_iter)
    print(f"  params={n_p/1e6:.1f}M  fwd={fwd:.1f} ms  bwd={bwd:.1f} ms  "
          f"step={(fwd+bwd):.1f} ms")
    del m; torch.cuda.empty_cache()

    # 3. Pure ortho (for the slowdown source).
    print("\n--- pure ortho ---")
    torch.manual_seed(0)
    m = TinyLM(
        vocab_size=args.vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head,
        attention_cls=OrthogonalScanAttention,
    ).to("cuda")
    fwd, bwd, n_p = profile_model(m, input_ids, n_iter=args.n_iter)
    print(f"  params={n_p/1e6:.1f}M  fwd={fwd:.1f} ms  bwd={bwd:.1f} ms  "
          f"step={(fwd+bwd):.1f} ms")
    del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
