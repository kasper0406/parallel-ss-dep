"""
S₅ word problem training driver.

Tests the architectures' ability to recognise non-solvable group state-
tracking. Hybrid (with SO(n) layers, n ≥ 3 contains A₅) should succeed;
deltanet_negeig (Z₂ only via Householder eigenvalue ±1) should fail.

Usage:
    python experiments/train_s5.py --arches deltanet,deltanet_negeig \\
        --layers ortho,deltanet,ortho,deltanet --T 128 --steps 5000

Note: this is binary classification per position (running composition =
identity yes/no). Class is heavily imbalanced (mostly "no") at long T.
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
from experiments.tasks.s5 import make_batch as s5_batch


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
    pos_class_recall: float           # accuracy on positions where label = 1
    secs: float
    params: int


def _val(model, T, batch_size, device):
    model.eval()
    with torch.no_grad():
        x, y = s5_batch(batch_size, T, vocab_size=5, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1)).item()
        preds = logits.argmax(dim=-1)
        per_tok_acc = (preds == y).float().mean().item()
        end_tok_acc = (preds[:, -1] == y[:, -1]).float().mean().item()
        # Recall on the positive class (positions where running comp = identity).
        pos_mask = (y == 1)
        if pos_mask.any():
            pos_class_recall = ((preds == y) & pos_mask).float().sum() / pos_mask.float().sum()
            pos_class_recall = pos_class_recall.item()
        else:
            pos_class_recall = float("nan")
    model.train()
    return loss, per_tok_acc, end_tok_acc, pos_class_recall


def train_one(arch_or_layers, T, steps, batch_size, d_model, n_layers,
              n_heads, d_head, lr, log_every, device="cuda", seed=0):
    torch.manual_seed(seed)
    if "," in arch_or_layers:
        cls_list = [ARCHES[p.strip()] for p in arch_or_layers.split(",")]
        n_layers = len(cls_list)
        attn_kw = dict(attention_cls_per_layer=cls_list)
    else:
        attn_kw = dict(attention_cls=ARCHES[arch_or_layers])

    model = TinyLM(
        vocab_size=5, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, max_T=T, **attn_kw,
    ).to(device)
    # The lm_head outputs over vocab=5, but we want a binary classifier.
    # Replace lm_head with a 2-class head.
    model.lm_head = torch.nn.Linear(d_model, 2, bias=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1,
    )

    print(f"\n[{arch_or_layers}]  T={T}  params={model.num_params():,}")
    print(f"{'step':>6}  {'tloss':>8}  {'vloss':>8}  {'val_acc':>8}  "
          f"{'end_acc':>8}  {'pos_rec':>8}")

    t0 = time.perf_counter()
    last_train_loss = float("nan")
    # Class-imbalance weights: positive class (=identity) is rare,
    # weight it inversely to the empirical frequency to keep gradient
    # signal balanced. Empirically ~2% positives at T=128.
    class_weight = torch.tensor([1.0, 50.0], device=device)
    for step in range(1, steps + 1):
        x, y = s5_batch(batch_size, T, vocab_size=5, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1),
                               weight=class_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        last_train_loss = loss.item()

        if step % log_every == 0 or step == steps:
            v_loss, v_acc, e_acc, pr = _val(model, T, 512, device)
            print(f"{step:>6d}  {last_train_loss:>8.4f}  {v_loss:>8.4f}  "
                  f"{v_acc:>8.3f}  {e_acc:>8.3f}  {pr:>8.3f}")

    v_loss, v_acc, e_acc, pr = _val(model, T, 1024, device)
    secs = time.perf_counter() - t0
    return RunResult(
        arch=arch_or_layers, T=T, steps=steps,
        final_train_loss=last_train_loss, final_val_loss=v_loss,
        end_token_acc=e_acc, per_tok_acc=v_acc, pos_class_recall=pr,
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
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.arches is None and args.layers is None:
        raise SystemExit("specify either --arches or --layers")

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
            lr=args.lr, log_every=args.log_every, seed=args.seed,
        )
        results.append(r)

    print("\n" + "=" * 90)
    print(f"{'arch':<28} {'T':>4} {'end_acc':>8} {'per_tok':>8} "
          f"{'pos_rec':>8} {'val_loss':>9} {'params':>10} {'secs':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r.arch:<28} {r.T:>4} {r.end_token_acc:>8.3f} "
              f"{r.per_tok_acc:>8.3f} {r.pos_class_recall:>8.3f} "
              f"{r.final_val_loss:>9.4f} {r.params:>10,} {r.secs:>7.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
