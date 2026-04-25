"""
MQAR training driver — tests recall ability across architectures.

The killer test for the `(R, c)` semidirect-product scan: can the
conjugated KV memory actually retrieve past tokens, the way DeltaNet/
linear-attention can but pure SO(n) and AUSSM cannot?

Usage:
    python experiments/train_mqar.py --arches linear,ortho,rotconj,deltanet \\
        --T 256 --steps 5000 --batch 256 --n_pairs 16 --vocab 64
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import (
    LinearAttention, HeisenbergAttention, SoftmaxAttention,
    DeltaNetAttention, GatedDeltaNetAttention, Mamba2Attention,
    OrthogonalScanAttention, RotConjAttention,
)
from experiments.model import TinyLM
from experiments.tasks.mqar import make_batch as mqar_batch


ARCHES = {
    "linear":     LinearAttention,
    "heisenberg": HeisenbergAttention,
    "softmax":    SoftmaxAttention,
    "deltanet":   DeltaNetAttention,
    "gateddelta": GatedDeltaNetAttention,
    "mamba2":     Mamba2Attention,
    "ortho":      OrthogonalScanAttention,
    "rotconj":    RotConjAttention,
}


@dataclasses.dataclass
class RunResult:
    arch: str
    T: int
    n_pairs: int
    vocab: int
    steps: int
    final_train_loss: float
    final_val_loss: float
    recall_acc: float
    secs: float
    params: int


def _val(model, T, n_pairs, vocab, batch_size, device):
    model.eval()
    with torch.no_grad():
        x, y, mask = mqar_batch(batch_size, T, vocab_size=vocab,
                                n_pairs=n_pairs, device=device)
        logits = model(x)
        # CE only on query positions.
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction="none",
        ).reshape(batch_size, T)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        preds = logits.argmax(dim=-1)
        correct = ((preds == y) & mask).float().sum()
        total = mask.sum().clamp_min(1)
        recall = (correct / total).item()
    model.train()
    return loss.item(), recall


def train_one(arch, T, n_pairs, vocab, steps, batch_size,
              d_model, n_layers, n_heads, d_head, lr, log_every,
              device="cuda", seed=0):
    torch.manual_seed(seed)
    cls = ARCHES[arch]
    model = TinyLM(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=cls,
        max_T=T,                              # learnable abs pos embed
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1,
    )

    print(f"\n[{arch}]  T={T}  n_pairs={n_pairs}  vocab={vocab}  "
          f"params={model.num_params():,}")
    print(f"{'step':>6}  {'train_loss':>11}  {'val_loss':>9}  {'recall':>8}")

    t0 = time.perf_counter()
    last_train_loss = float("nan")
    for step in range(1, steps + 1):
        x, y, mask = mqar_batch(batch_size, T, vocab_size=vocab,
                                n_pairs=n_pairs, device=device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction="none",
        ).reshape(batch_size, T)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        last_train_loss = loss.item()

        if step % log_every == 0 or step == steps:
            v_loss, recall = _val(model, T, n_pairs, vocab, 512, device)
            print(f"{step:>6d}  {last_train_loss:>11.4f}  "
                  f"{v_loss:>9.4f}  {recall:>8.3f}")

    v_loss, recall = _val(model, T, n_pairs, vocab, 1024, device)
    secs = time.perf_counter() - t0
    return RunResult(
        arch=arch, T=T, n_pairs=n_pairs, vocab=vocab, steps=steps,
        final_train_loss=last_train_loss, final_val_loss=v_loss,
        recall_acc=recall, secs=secs, params=model.num_params(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", type=str, default="linear,ortho,rotconj,deltanet")
    p.add_argument("--T", type=int, nargs="+", default=[256])
    p.add_argument("--n_pairs", type=int, default=16)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arches = args.arches.split(",")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"capability={torch.cuda.get_device_capability(0)}")

    results = []
    for T in args.T:
        for a in arches:
            r = train_one(
                arch=a, T=T, n_pairs=args.n_pairs, vocab=args.vocab,
                steps=args.steps, batch_size=args.batch,
                d_model=args.d_model, n_layers=args.n_layers,
                n_heads=args.n_heads, d_head=args.d_head,
                lr=args.lr, log_every=args.log_every, seed=args.seed,
            )
            results.append(r)

    print("\n" + "=" * 80)
    print(f"{'arch':<12} {'T':>5} {'pairs':>5} {'recall':>8} "
          f"{'val_loss':>9} {'params':>10} {'secs':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r.arch:<12} {r.T:>5} {r.n_pairs:>5} {r.recall_acc:>8.3f} "
              f"{r.final_val_loss:>9.4f} {r.params:>10,} {r.secs:>7.1f}")

    print("\nMQAR ranking by recall:")
    by_T = {}
    for r in results:
        by_T.setdefault(r.T, []).append(r)
    for T, rs in sorted(by_T.items()):
        ordered = sorted(rs, key=lambda r: r.recall_acc, reverse=True)
        print(f"  T={T}:")
        for r in ordered:
            print(f"    {r.arch:<12}  recall={r.recall_acc:.3f}  "
                  f"loss={r.final_val_loss:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
