"""
Hybrid-layer-stack training driver.

Tests the "specialist layers" hypothesis: a single network with some
ortho layers (parity-capable, Grazzi-clean) and some DeltaNet layers
(recall-capable, fixed-frame delta erase) might escape both walls at
the network level — even though no single cell does both.

Architecture spec via --layers, e.g. `--layers ortho,deltanet,ortho,deltanet`.
Tests on either parity or induction (--task).
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
    OrthogonalScanAttention, RotConjAttention, RotDeltaAttention,
)
from experiments.model import TinyLM
from experiments.tasks.parity import make_batch as parity_batch
from experiments.tasks.induction import make_batch as induction_batch


ARCHES = {
    "linear":     LinearAttention,
    "heisenberg": HeisenbergAttention,
    "softmax":    SoftmaxAttention,
    "deltanet":   DeltaNetAttention,
    "gateddelta": GatedDeltaNetAttention,
    "mamba2":     Mamba2Attention,
    "ortho":      OrthogonalScanAttention,
    "rotconj":    RotConjAttention,
    "rotdelta":   RotDeltaAttention,
}


def parse_layers(spec: str) -> list:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not all(p in ARCHES for p in parts):
        bad = [p for p in parts if p not in ARCHES]
        raise SystemExit(f"unknown arch(es): {bad}; choices: {list(ARCHES)}")
    return [ARCHES[p] for p in parts]


def _val_parity(model, T, batch_size, device):
    model.eval()
    with torch.no_grad():
        x, y = parity_batch(batch_size, T, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1)).item()
        preds = logits.argmax(dim=-1)
        per_tok_acc = (preds == y).float().mean().item()
        end_tok_acc = (preds[:, -1] == y[:, -1]).float().mean().item()
        per_pos = (preds == y).float().mean(dim=0)
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


def _val_induction(model, T, vocab, batch_size, device):
    model.eval()
    with torch.no_grad():
        x, y, mask = induction_batch(batch_size, T, vocab_size=vocab, device=device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction="none",
        ).reshape(batch_size, T)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        preds = logits.argmax(dim=-1)
        correct = ((preds == y) & mask).float().sum()
        total = mask.sum().clamp_min(1)
        accuracy = (correct / total).item()
    model.train()
    return loss.item(), accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="parity",
                   choices=["parity", "induction"])
    p.add_argument("--layers", type=str, required=True,
                   help="comma-separated arch list, e.g. 'ortho,deltanet,ortho,deltanet'")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--vocab", type=int, default=2,
                   help="vocab size (auto: 2 for parity, default 32 for induction)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.task == "induction" and args.vocab == 2:
        args.vocab = 32

    cls_list = parse_layers(args.layers)
    n_layers = len(cls_list)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"task={args.task}  T={args.T}  vocab={args.vocab}  "
          f"n_layers={n_layers}  layer_arch={args.layers}")

    torch.manual_seed(args.seed)
    model = TinyLM(
        vocab_size=args.vocab, d_model=args.d_model, n_layers=n_layers,
        n_heads=args.n_heads, d_head=args.d_head,
        attention_cls_per_layer=cls_list,
        max_T=args.T if args.task == "induction" else 0,
    ).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1,
    )

    print(f"params: {model.num_params():,}\n")
    if args.task == "parity":
        print(f"{'step':>6}  {'tloss':>8}  {'vloss':>8}  {'val_acc':>8}  "
              f"{'end_acc':>8}  q1/q2/q3/q4")
    else:
        print(f"{'step':>6}  {'tloss':>8}  {'vloss':>8}  {'recall':>8}")

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        if args.task == "parity":
            x, y = parity_batch(args.batch, args.T, device="cuda")
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))
        else:
            x, y, mask = induction_batch(args.batch, args.T, vocab_size=args.vocab, device="cuda")
            logits = model(x)
            ce = F.cross_entropy(
                logits.reshape(-1, args.vocab), y.reshape(-1), reduction="none",
            ).reshape(args.batch, args.T)
            loss = (ce * mask).sum() / mask.sum().clamp_min(1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % args.log_every == 0 or step == args.steps:
            if args.task == "parity":
                v_loss, v_acc, e_acc, q = _val_parity(model, args.T, 512, "cuda")
                qs = "/".join(f"{x:.2f}" for x in q)
                print(f"{step:>6d}  {loss.item():>8.4f}  {v_loss:>8.4f}  "
                      f"{v_acc:>8.3f}  {e_acc:>8.3f}  {qs}")
            else:
                v_loss, recall = _val_induction(model, args.T, args.vocab, 512, "cuda")
                print(f"{step:>6d}  {loss.item():>8.4f}  {v_loss:>8.4f}  "
                      f"{recall:>8.3f}")

    secs = time.perf_counter() - t0
    print(f"\ndone in {secs:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
