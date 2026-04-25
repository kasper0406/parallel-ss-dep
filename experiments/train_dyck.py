"""
Dyck depth-tracking training driver.

Tests architectures' ability to track nesting depth in bracket sequences
— the synthetic analog of code scope/indentation tracking.

Usage:
    python experiments/train_dyck.py --arches deltanet \\
        --layers ortho,deltanet,ortho,deltanet \\
        --T 128 --steps 5000
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
    DeltaNetAttention, DeltaNetNegEigAttention,
    GatedDeltaNetAttention, Mamba2Attention,
    OrthogonalScanAttention, RotConjAttention, RotDeltaAttention,
)
from experiments.model import TinyLM
from experiments.tasks.dyck import make_batch as dyck_batch


ARCHES = {
    "linear":     LinearAttention,
    "heisenberg": HeisenbergAttention,
    "softmax":    SoftmaxAttention,
    "deltanet":   DeltaNetAttention,
    "deltanet_negeig": DeltaNetNegEigAttention,
    "gateddelta": GatedDeltaNetAttention,
    "mamba2":     Mamba2Attention,
    "ortho":      OrthogonalScanAttention,
    "rotconj":    RotConjAttention,
    "rotdelta":   RotDeltaAttention,
}


@dataclasses.dataclass
class RunResult:
    arch: str
    T: int
    steps: int
    final_train_loss: float
    final_val_loss: float
    end_token_acc: float
    per_tok_acc: float
    secs: float
    params: int


def _val(model, T, n_classes, batch_size, device):
    model.eval()
    with torch.no_grad():
        x, y = dyck_batch(batch_size, T, n_types=2, max_depth=n_classes - 1,
                          device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), y.reshape(-1)).item()
        preds = logits.argmax(dim=-1)
        per_tok_acc = (preds == y).float().mean().item()
        end_tok_acc = (preds[:, -1] == y[:, -1]).float().mean().item()
        # Quartile breakdown — q4 is the late-position tracking accuracy.
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


def train_one(arch_or_layers, T, steps, batch_size, d_model, n_layers,
              n_heads, d_head, lr, log_every, max_depth=15, device="cuda",
              seed=0):
    n_classes = max_depth + 1                          # 0..max_depth
    n_tokens = 4                                       # 2 bracket types × 2

    torch.manual_seed(seed)
    if "," in arch_or_layers:
        cls_list = [ARCHES[p.strip()] for p in arch_or_layers.split(",")]
        n_layers = len(cls_list)
        attn_kw = dict(attention_cls_per_layer=cls_list)
    else:
        attn_kw = dict(attention_cls=ARCHES[arch_or_layers])

    # Model has n_tokens vocab on input but we need n_classes output dims.
    # Build with vocab=max(n_tokens, n_classes) and slice if needed; or just
    # use a separate linear head. Simpler: vocab=n_classes (>= n_tokens).
    vocab = max(n_tokens, n_classes)
    model = TinyLM(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, max_T=T, **attn_kw,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1,
    )

    print(f"\n[{arch_or_layers}]  T={T}  max_depth={max_depth}  "
          f"params={model.num_params():,}")
    print(f"{'step':>6}  {'tloss':>8}  {'vloss':>8}  {'val_acc':>8}  "
          f"{'end_acc':>8}  q1/q2/q3/q4")

    t0 = time.perf_counter()
    last_train_loss = float("nan")
    for step in range(1, steps + 1):
        x, y = dyck_batch(batch_size, T, n_types=2, max_depth=max_depth,
                          device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        last_train_loss = loss.item()

        if step % log_every == 0 or step == steps:
            v_loss, v_acc, e_acc, q = _val(model, T, n_classes, 512, device)
            qs = "/".join(f"{x:.2f}" for x in q)
            print(f"{step:>6d}  {last_train_loss:>8.4f}  {v_loss:>8.4f}  "
                  f"{v_acc:>8.3f}  {e_acc:>8.3f}  {qs}")

    v_loss, v_acc, e_acc, _ = _val(model, T, n_classes, 1024, device)
    secs = time.perf_counter() - t0
    return RunResult(
        arch=arch_or_layers, T=T, steps=steps,
        final_train_loss=last_train_loss, final_val_loss=v_loss,
        end_token_acc=e_acc, per_tok_acc=v_acc,
        secs=secs, params=model.num_params(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", type=str, default=None)
    p.add_argument("--layers", type=str, default=None)
    p.add_argument("--T", type=int, default=128)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--max_depth", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.arches is None and args.layers is None:
        raise SystemExit("specify --arches or --layers")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    runs = []
    if args.arches:
        for a in args.arches.split(","):
            runs.append((a.strip(), args.n_layers))
    if args.layers:
        runs.append((args.layers, len(args.layers.split(","))))

    results = []
    for arch_or_layers, n_layers in runs:
        r = train_one(
            arch_or_layers=arch_or_layers, T=args.T,
            steps=args.steps, batch_size=args.batch,
            d_model=args.d_model, n_layers=n_layers,
            n_heads=args.n_heads, d_head=args.d_head,
            lr=args.lr, log_every=args.log_every,
            max_depth=args.max_depth, seed=args.seed,
        )
        results.append(r)

    print("\n" + "=" * 90)
    print(f"{'arch':<32} {'T':>4} {'end_acc':>8} {'per_tok':>8} "
          f"{'val_loss':>9} {'params':>10} {'secs':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r.arch:<32} {r.T:>4} {r.end_token_acc:>8.3f} "
              f"{r.per_tok_acc:>8.3f} {r.final_val_loss:>9.4f} "
              f"{r.params:>10,} {r.secs:>7.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
