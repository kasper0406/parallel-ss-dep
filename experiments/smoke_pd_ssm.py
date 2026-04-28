"""Smoke test for PDScanAttention on parity (MPS-friendly).

Trains a small TinyLM with PDScanAttention on running-parity at T=64 and
compares against LinearAttention. Parity is a star-free-vs-Z_2 separator:
LinearAttention (eigvals = {1}) provably cannot solve it; PDScanAttention
with [-1, 1] diagonal should.

Predicted by the 5-tuple framework:
  - LinearAttention: stuck at 50% (chance) — eigenvalue range = {1}.
  - PDScanAttention: 100% — eigenvalue range = [-1, 1] via D + σ has 2-cycle.

Usage:
    python experiments/smoke_pd_ssm.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import LinearAttention, PDScanAttention
from experiments.model import TinyLM
from experiments.tasks.parity import make_batch as parity_batch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one(arch_cls, label: str, T: int, steps: int, batch: int,
              d_model: int, n_layers: int, n_heads: int, d_head: int,
              lr: float, device: str, log_every: int) -> dict:
    torch.manual_seed(0)
    model = TinyLM(
        vocab_size=2, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=arch_cls,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.1)
    print(f"\n[{label}] T={T}  params={n_params:,}  device={device}")
    print(f"{'step':>5}  {'loss':>7}  {'val_acc':>7}  {'end_acc':>7}  q1/q2/q3/q4")
    t0 = time.perf_counter()
    last = float("nan")
    for step in range(1, steps + 1):
        x, y = parity_batch(batch, T, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = loss.item()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                xv, yv = parity_batch(256, T, device=device)
                logv = model(xv)
                pred = logv.argmax(-1)
                v_acc = (pred == yv).float().mean().item()
                e_acc = (pred[:, -1] == yv[:, -1]).float().mean().item()
                per_pos = (pred == yv).float().mean(0)
                q = T // 4
                qs = "/".join(f"{per_pos[i*q:(i+1)*q].mean():.2f}" for i in range(4))
            model.train()
            print(f"{step:>5d}  {last:>7.4f}  {v_acc:>7.3f}  {e_acc:>7.3f}  {qs}")
    secs = time.perf_counter() - t0
    return dict(label=label, params=n_params, final_loss=last,
                end_acc=e_acc, val_acc=v_acc, secs=secs)


def main() -> int:
    device = pick_device()
    print(f"device: {device}")

    common = dict(T=64, steps=500, batch=64, d_model=64,
                  n_layers=2, n_heads=4, d_head=16, lr=3e-3,
                  device=device, log_every=100)

    results = []
    results.append(train_one(LinearAttention, "linear", **common))
    results.append(train_one(PDScanAttention, "pd_ssm", **common))

    print("\n" + "=" * 60)
    print(f"{'arch':<10} {'params':>8} {'val_acc':>8} {'end_acc':>8} {'secs':>7}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<10} {r['params']:>8,} "
              f"{r['val_acc']:>8.3f} {r['end_acc']:>8.3f} {r['secs']:>7.1f}")

    pd = next(r for r in results if r["label"] == "pd_ssm")
    lin = next(r for r in results if r["label"] == "linear")
    margin = (pd["end_acc"] - lin["end_acc"]) * 100
    print(f"\npd_ssm − linear end-token margin: {margin:+.1f} pp")
    print(f"pass: {'yes' if margin >= 30 else 'no'} (≥ +30 pp = clean parity solve)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
