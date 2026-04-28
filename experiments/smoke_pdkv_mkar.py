"""MKAR (Many-Key Associative Recall) head-to-head: PD-SSM vs PD-KV vs baselines.

The capacity-gap test: as the number of key-value bindings K grows past
the per-head state dim N, vector-state architectures (PD-SSM) saturate
while matrix-state architectures (DeltaNet, PD-KV, LinearAttention) do not.

Predicted by the 5-tuple framework + Arora et al. Zoology:
  - LinearAttention: handles K up to ~ d_head² via outer-product memory.
  - PDScanAttention (vector state, dim N): saturates around K ≈ N.
  - PDKVScanAttention (matrix state N × D): handles K up to ~ N · D.

Sweeps K = {4, 8, 16}; N = 8 fixed. With d_head=8, expected pattern:
  K=4   all should solve
  K=8   pd_ssm saturates, others fine
  K=16  pd_ssm clearly worse than pd_kv / linear

Optimizer: AdamW or Muon (--optim muon).

Usage:
    python experiments/smoke_pdkv_mkar.py
    python experiments/smoke_pdkv_mkar.py --optim muon
    python experiments/smoke_pdkv_mkar.py --K 4 16 32 --steps 1000
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
    LinearAttention, PDScanAttention, PDKVScanAttention,
)
from experiments.model import TinyLM
from experiments.tasks.mqar import make_batch as mqar_batch
from experiments.optim_muon import Muon, muon_param_groups


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_optimizer(model, optim_name: str, lr: float):
    if optim_name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=0.0)]
    if optim_name == "muon":
        muon_p, adam_p = muon_param_groups(model)
        muon = Muon(muon_p, lr=lr, momentum=0.95, nesterov=True)
        # Give the AdamW group a smaller LR — embeddings need calmer updates.
        adam = torch.optim.AdamW(adam_p, lr=lr * 0.3, betas=(0.9, 0.95),
                                 weight_decay=0.0)
        return [muon, adam]
    raise ValueError(f"unknown optim: {optim_name}")


def step_optimizers(opts):
    for o in opts:
        o.step()


def zero_optimizers(opts):
    for o in opts:
        o.zero_grad(set_to_none=True)


def train_one(arch_cls, label: str, K: int, T: int, steps: int, batch: int,
              vocab: int, d_model: int, n_layers: int, n_heads: int,
              d_head: int, lr: float, optim_name: str,
              device: str, log_every: int,
              state_dim: int | None = None) -> dict:
    torch.manual_seed(0)

    # Allow per-arch state_dim override (PD-SSM, PD-KV).
    def make_attn(d_model, n_heads, d_head):
        if state_dim is not None and arch_cls in (PDScanAttention, PDKVScanAttention):
            return arch_cls(d_model=d_model, n_heads=n_heads, d_head=d_head,
                            state_dim=state_dim)
        return arch_cls(d_model=d_model, n_heads=n_heads, d_head=d_head)

    model = TinyLM(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head,
        attention_cls=make_attn, max_T=T,
    ).to(device)

    opts = build_optimizer(model, optim_name, lr)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(
        o, T_max=steps, eta_min=lr * 0.1) for o in opts]
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n[{label}] K={K} T={T} N={state_dim} optim={optim_name} "
          f"params={n_params:,}", flush=True)
    print(f"{'step':>5}  {'loss':>7}  {'q_acc':>7}")

    t0 = time.perf_counter()
    last_loss = float("nan")
    last_q = float("nan")
    for step in range(1, steps + 1):
        x, y, mask = mqar_batch(batch, T, vocab_size=vocab,
                                n_pairs=K, device=device)
        logits = model(x)                                     # (B, T, V)
        # Loss only at query positions.
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs[mask], y[mask], reduction="mean")
        zero_optimizers(opts)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_optimizers(opts)
        for s in schedulers:
            s.step()
        last_loss = nll.item()

        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                xv, yv, mv = mqar_batch(256, T, vocab_size=vocab,
                                        n_pairs=K, device=device)
                pv = model(xv).argmax(dim=-1)
                q_acc = (pv[mv] == yv[mv]).float().mean().item()
                last_q = q_acc
            model.train()
            print(f"{step:>5d}  {last_loss:>7.4f}  {q_acc:>7.3f}", flush=True)
    secs = time.perf_counter() - t0
    return dict(label=label, K=K, params=n_params,
                final_loss=last_loss, q_acc=last_q, secs=secs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=8)
    p.add_argument("--state_dim", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--optim", type=str, default="adamw",
                   choices=("adamw", "muon"))
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--arches", type=str, default="linear,pd_ssm,pd_kv")
    args = p.parse_args()

    device = pick_device()
    print(f"device: {device}  optim: {args.optim}")

    name_to_cls = {
        "linear": LinearAttention,
        "pd_ssm": PDScanAttention,
        "pd_kv":  PDKVScanAttention,
    }
    arches = [(n, name_to_cls[n]) for n in args.arches.split(",")]

    results: list[dict] = []
    for K in args.K:
        # Need T >= 2K + a few queries; pad up if necessary.
        T = max(args.T, 2 * K + 16)
        for label, cls in arches:
            r = train_one(
                cls, label, K=K, T=T, steps=args.steps, batch=args.batch,
                vocab=args.vocab, d_model=args.d_model,
                n_layers=args.n_layers, n_heads=args.n_heads,
                d_head=args.d_head, lr=args.lr, optim_name=args.optim,
                device=device, log_every=args.log_every,
                state_dim=args.state_dim,
            )
            results.append(r)

    # Summary by K, ranked by query-accuracy.
    print("\n" + "=" * 70)
    print(f"MKAR results — N={args.state_dim}, d_head={args.d_head}, "
          f"optim={args.optim}, steps={args.steps}")
    print("-" * 70)
    print(f"{'K':>3}  {'arch':<10}  {'q_acc':>7}  {'loss':>8}  {'secs':>7}")
    for K in args.K:
        for r in results:
            if r["K"] == K:
                marker = "  ✓" if r["q_acc"] > 0.85 else ("  ~" if r["q_acc"] > 0.5 else "  ✗")
                print(f"{K:>3}  {r['label']:<10}  {r['q_acc']:>7.3f}  "
                      f"{r['final_loss']:>8.4f}  {r['secs']:>7.1f}{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
