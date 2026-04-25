"""
Kill-gate training driver — parity at small scale.

Trains both `LinearAttention` and `HeisenbergAttention` from scratch on the
running-parity task and compares final per-token accuracy on a held-out batch.

Pass criterion (per EVAL_PLAN.md §3.6):
  HeisenbergAttention beats LinearAttention by ≥ 10 percentage points
  on per-position parity accuracy.

Usage:
    python experiments/train.py
    python experiments/train.py --T 256 --steps 5000
    python experiments/train.py --arches linear,heisenberg --T 64 256 512
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

# Allow `python experiments/train.py` from repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import (
    LinearAttention, HeisenbergAttention, SoftmaxAttention,
    DeltaNetAttention, GatedDeltaNetAttention, Mamba2Attention,
    OrthogonalScanAttention, RotConjAttention,
)
from experiments.model import TinyLM
from experiments.tasks.parity import make_batch


ARCHES = {
    "linear":      LinearAttention,
    "heisenberg":  HeisenbergAttention,
    "softmax":     SoftmaxAttention,
    "deltanet":    DeltaNetAttention,
    "gateddelta":  GatedDeltaNetAttention,
    "mamba2":      Mamba2Attention,
    "ortho":       OrthogonalScanAttention,        # SO(n) scan — Grazzi-clean
    "rotconj":     RotConjAttention,               # SO(n) ⋉ ℝ^{n×n} — semidirect, novel
}


@dataclasses.dataclass
class RunResult:
    arch: str
    T: int
    steps: int
    final_train_loss: float
    final_val_loss: float
    final_val_acc: float          # per-token accuracy on val batch
    end_token_acc: float          # accuracy at the *last* position
    secs: float
    params: int


def _val(model: torch.nn.Module, T: int, batch_size: int,
         device: torch.device | str
         ) -> tuple[float, float, float, list[float]]:
    """Returns (loss, per-token acc, end-token acc, quartile accs)."""
    model.eval()
    with torch.no_grad():
        x, y = make_batch(batch_size, T, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1)).item()
        preds = logits.argmax(dim=-1)
        per_tok_acc = (preds == y).float().mean().item()
        end_tok_acc = (preds[:, -1] == y[:, -1]).float().mean().item()
        # Quartile accuracies — last quarter is the actual parity test.
        per_pos = (preds == y).float().mean(dim=0)              # (T,)
        q = T // 4
        if q == 0:
            quartiles = [per_tok_acc] * 4
        else:
            quartiles = [
                per_pos[0:q].mean().item(),
                per_pos[q:2*q].mean().item(),
                per_pos[2*q:3*q].mean().item(),
                per_pos[3*q:].mean().item(),
            ]
    model.train()
    return loss, per_tok_acc, end_tok_acc, quartiles


def train_one(arch: str, T: int, steps: int, batch_size: int,
              d_model: int, n_layers: int, n_heads: int, d_head: int,
              lr: float, log_every: int,
              device: torch.device | str = "cuda",
              seed: int = 0) -> RunResult:
    torch.manual_seed(seed)
    cls = ARCHES[arch]
    model = TinyLM(
        vocab_size=2, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=cls,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                           eta_min=lr * 0.1)

    print(f"\n[{arch}] T={T}  params={model.num_params():,}  "
          f"d_model={d_model}  n_layers={n_layers}  H={n_heads}  D={d_head}")
    print(f"{'step':>6}  {'train_loss':>11}  {'val_loss':>9}  "
          f"{'val_acc':>8}  {'end_acc':>8}  {'q1/q2/q3/q4 (last quartile is parity)':<40}")

    t0 = time.perf_counter()
    last_train_loss = float("nan")
    for step in range(1, steps + 1):
        x, y = make_batch(batch_size, T, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        last_train_loss = loss.item()

        if step % log_every == 0 or step == steps:
            v_loss, v_acc, e_acc, q = _val(model, T=T, batch_size=512,
                                           device=device)
            qs = "/".join(f"{x:.2f}" for x in q)
            print(f"{step:>6d}  {last_train_loss:>11.4f}  "
                  f"{v_loss:>9.4f}  {v_acc:>8.3f}  {e_acc:>8.3f}  {qs}")

    # Final eval, larger batch — capped to avoid Mamba2's per-step
    # O(T·B·d_state·d_head) memory blow-up at T=128.
    final_bs = 2048 if T <= 64 else 512
    v_loss, v_acc, e_acc, _ = _val(model, T=T, batch_size=final_bs, device=device)
    secs = time.perf_counter() - t0
    return RunResult(
        arch=arch, T=T, steps=steps,
        final_train_loss=last_train_loss, final_val_loss=v_loss,
        final_val_acc=v_acc, end_token_acc=e_acc, secs=secs,
        params=model.num_params(),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arches", type=str, default="linear,heisenberg")
    p.add_argument("--T", type=int, nargs="+", default=[64])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arches = args.arches.split(",")
    for a in arches:
        if a not in ARCHES:
            raise SystemExit(f"unknown arch: {a}; choices: {list(ARCHES)}")

    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"capability={torch.cuda.get_device_capability(0)}")

    results: list[RunResult] = []
    for T in args.T:
        for a in arches:
            r = train_one(
                arch=a, T=T, steps=args.steps, batch_size=args.batch,
                d_model=args.d_model, n_layers=args.n_layers,
                n_heads=args.n_heads, d_head=args.d_head,
                lr=args.lr, log_every=args.log_every, seed=args.seed,
            )
            results.append(r)

    print("\n" + "=" * 80)
    print(f"{'arch':<12} {'T':>5} {'params':>10} {'val_loss':>9} "
          f"{'val_acc':>8} {'end_acc':>8} {'secs':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r.arch:<12} {r.T:>5} {r.params:>10,} {r.final_val_loss:>9.4f} "
              f"{r.final_val_acc:>8.3f} {r.end_token_acc:>8.3f} {r.secs:>7.1f}")

    # Kill-gate summary: heisenberg vs linear, per T (preserved).
    print("\nKill-gate margins (heisenberg − linear, percentage points):")
    by_T: dict[int, dict[str, RunResult]] = {}
    for r in results:
        by_T.setdefault(r.T, {})[r.arch] = r
    for T, archs in sorted(by_T.items()):
        if "heisenberg" in archs and "linear" in archs:
            d_acc = (archs["heisenberg"].final_val_acc
                     - archs["linear"].final_val_acc) * 100
            d_end = (archs["heisenberg"].end_token_acc
                     - archs["linear"].end_token_acc) * 100
            verdict = "PASS" if d_acc >= 10 else "FAIL"
            print(f"  T={T:>4}: Δ per-tok = {d_acc:+6.2f} pp,  "
                  f"Δ end-tok = {d_end:+6.2f} pp   [{verdict}]")

    # SOTA ranking — sort by end-token accuracy at each T.
    print("\nSOTA ranking by end-token accuracy:")
    for T, archs in sorted(by_T.items()):
        ordered = sorted(archs.values(),
                         key=lambda r: r.end_token_acc, reverse=True)
        print(f"  T={T}:")
        for r in ordered:
            print(f"    {r.arch:<12}  end-tok={r.end_token_acc:.3f}  "
                  f"per-tok={r.final_val_acc:.3f}  loss={r.final_val_loss:.4f}  "
                  f"params={r.params/1e6:.2f}M")

    return 0


if __name__ == "__main__":
    sys.exit(main())
