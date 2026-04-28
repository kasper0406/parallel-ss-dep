"""Mod-p-above-Landau test: PD-SSM (real D) vs Complex-PD vs baselines.

The algebra-gap test: PD-SSM with N=4 cannot solve mod-p for p ≥ 5
because Z_p ≤ S_4 requires an element of order p in S_4, and Landau's
function g(4) = 4 (max element-order in S_4 is 4).

Complex-PD with the same N=4 *should* solve any mod-p via a complex
diagonal entry e^{2πi/p} on a 1-cycle of σ.

Predicted by the framework + Grazzi negative-eigenvalues theorem:
  arch          mod-2 (parity)   mod-3      mod-5      mod-7
  linear         chance           chance     chance     chance
  pd_ssm  N=4    100%             100%       fail       fail
  complex_pd N=4 100%             100%       100%       100%

This is the cleanest sharp algebraic prediction of the framework: a
specific value of N where the architectures diverge based on Landau's
function. We test N=4, where g(4)=4 puts the boundary between p=4
(reachable) and p=5 (not reachable for real-only D).

Usage:
    python experiments/smoke_complex_pd_landau.py
    python experiments/smoke_complex_pd_landau.py --p 2 3 5 7 --steps 1500
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
    LinearAttention, PDScanAttention, ComplexPDScanAttention,
)
from experiments.model import TinyLM
from experiments.tasks.modular import make_batch as modular_batch
from experiments.optim_muon import Muon, muon_param_groups


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_optimizer(model, name: str, lr: float):
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=0.0)]
    if name == "muon":
        muon_p, adam_p = muon_param_groups(model)
        return [Muon(muon_p, lr=lr, momentum=0.95),
                torch.optim.AdamW(adam_p, lr=lr * 0.3,
                                  betas=(0.9, 0.95), weight_decay=0.0)]
    raise ValueError(name)


def step(opts):
    for o in opts: o.step()


def zero(opts):
    for o in opts: o.zero_grad(set_to_none=True)


def train_one(arch_cls, label: str, p_val: int, T: int, steps: int,
              batch: int, d_model: int, n_layers: int, n_heads: int,
              d_head: int, state_dim: int | None, lr: float,
              optim_name: str, device: str, log_every: int) -> dict:
    torch.manual_seed(0)

    def make_attn(d_model, n_heads, d_head):
        if state_dim is not None and arch_cls in (PDScanAttention, ComplexPDScanAttention):
            return arch_cls(d_model=d_model, n_heads=n_heads, d_head=d_head,
                            state_dim=state_dim)
        return arch_cls(d_model=d_model, n_heads=n_heads, d_head=d_head)

    model = TinyLM(
        vocab_size=p_val, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=make_attn,
    ).to(device)
    opts = build_optimizer(model, optim_name, lr)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
        o, T_max=steps, eta_min=lr * 0.1) for o in opts]
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n[{label}] mod-{p_val} T={T} N={state_dim} optim={optim_name} "
          f"params={n_params:,}", flush=True)
    t0 = time.perf_counter()
    last_loss = float("nan")
    last_e = float("nan")
    for s in range(1, steps + 1):
        x, y = modular_batch(batch, T, p=p_val, device=device)
        nll = F.cross_entropy(model(x).reshape(-1, p_val), y.reshape(-1))
        zero(opts)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step(opts)
        for sch in schedulers: sch.step()
        last_loss = nll.item()
        if s % log_every == 0 or s == steps:
            model.eval()
            with torch.no_grad():
                xv, yv = modular_batch(256, T, p=p_val, device=device)
                pv = model(xv).argmax(-1)
                e_acc = (pv[:, -1] == yv[:, -1]).float().mean().item()
                last_e = e_acc
            model.train()
            print(f"  step {s:>4d}  loss={last_loss:.4f}  end_acc={e_acc:.3f}",
                  flush=True)
    return dict(label=label, p=p_val, params=n_params,
                end_acc=last_e, final_loss=last_loss,
                secs=time.perf_counter() - t0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--p", type=int, nargs="+", default=[2, 3, 5, 7])
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=8)
    p.add_argument("--state_dim", type=int, default=4)   # N=4: Landau bound g(4)=4
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--optim", type=str, default="adamw",
                   choices=("adamw", "muon"))
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--arches", type=str, default="linear,pd_ssm,complex_pd")
    args = p.parse_args()

    device = pick_device()
    print(f"device: {device}  optim: {args.optim}  N (state_dim): {args.state_dim}")

    name_to_cls = {
        "linear":     LinearAttention,
        "pd_ssm":     PDScanAttention,
        "complex_pd": ComplexPDScanAttention,
    }
    arches = [(n, name_to_cls[n]) for n in args.arches.split(",")]

    results = []
    for p_val in args.p:
        for label, cls in arches:
            r = train_one(
                cls, label, p_val=p_val, T=args.T, steps=args.steps,
                batch=args.batch, d_model=args.d_model,
                n_layers=args.n_layers, n_heads=args.n_heads,
                d_head=args.d_head, state_dim=args.state_dim,
                lr=args.lr, optim_name=args.optim,
                device=device, log_every=args.log_every,
            )
            results.append(r)

    print("\n" + "=" * 60)
    print(f"Mod-p above Landau — N={args.state_dim} (g({args.state_dim})="
          f"{ {2:2,3:3,4:4,5:6,6:6,7:12,8:15}.get(args.state_dim,'?') })  "
          f"optim={args.optim}")
    print("-" * 60)
    print(f"{'p':>3}  {'arch':<12}  {'end_acc':>8}  {'chance':>7}  {'verdict'}")
    for p_val in args.p:
        chance = 1.0 / p_val
        for r in results:
            if r["p"] == p_val:
                ok = r["end_acc"] > chance + 0.20
                print(f"{p_val:>3}  {r['label']:<12}  {r['end_acc']:>8.3f}  "
                      f"{chance:>7.3f}  {'✓' if ok else '✗'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
