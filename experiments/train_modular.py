"""
Modular addition mod p training driver.

The cleanest theoretical separator we have between SO(n)-based
architectures (ortho, hybrid) and DeltaNet/DeltaProduct: for p > 2,
SO(n) handles any Z_p (rotation by 2π/p) but DeltaNet's eigenvalues are
limited to ±1, so it can only do p=2 (parity).

Usage:
    python experiments/train_modular.py --arches linear,ortho,deltanet,mamba2 \\
        --p 3 5 7 --T 128 --steps 5000 --batch 256

For hybrid (ortho/deltanet alternating), use --layers explicitly:
    python experiments/train_modular.py --layers ortho,deltanet,ortho,deltanet \\
        --p 3 5 7 --T 128 --steps 5000 --batch 256
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
from experiments.tasks.modular import make_batch as modular_batch


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
    p: int
    steps: int
    final_train_loss: float
    final_val_loss: float
    end_token_acc: float
    per_tok_acc: float
    secs: float
    params: int


def _val(model, T, p, batch_size, device):
    model.eval()
    with torch.no_grad():
        x, y = modular_batch(batch_size, T, p=p, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, p), y.reshape(-1)).item()
        preds = logits.argmax(dim=-1)
        per_tok_acc = (preds == y).float().mean().item()
        end_tok_acc = (preds[:, -1] == y[:, -1]).float().mean().item()
        # Quartile breakdown for parity-style diagnostic.
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


def train_one(arch_or_layers, T, p, steps, batch_size, d_model, n_layers,
              n_heads, d_head, lr, log_every, device="cuda", seed=0):
    torch.manual_seed(seed)

    # arch_or_layers may be either a single arch name (homogeneous model)
    # or a comma-separated list (hybrid/heterogeneous).
    if "," in arch_or_layers:
        cls_list = [ARCHES[p_.strip()] for p_ in arch_or_layers.split(",")]
        n_layers = len(cls_list)
        attn_kw = dict(attention_cls_per_layer=cls_list)
        label = arch_or_layers
    else:
        attn_kw = dict(attention_cls=ARCHES[arch_or_layers])
        label = arch_or_layers

    model = TinyLM(
        vocab_size=p, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, **attn_kw,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1,
    )

    print(f"\n[{label}]  T={T}  p={p}  params={model.num_params():,}  "
          f"n_layers={n_layers}")
    print(f"{'step':>6}  {'tloss':>8}  {'vloss':>8}  {'val_acc':>8}  "
          f"{'end_acc':>8}  q1/q2/q3/q4")

    t0 = time.perf_counter()
    last_train_loss = float("nan")
    for step in range(1, steps + 1):
        x, y = modular_batch(batch_size, T, p=p, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, p), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        last_train_loss = loss.item()

        if step % log_every == 0 or step == steps:
            v_loss, v_acc, e_acc, q = _val(model, T, p, 512, device)
            qs = "/".join(f"{x:.2f}" for x in q)
            print(f"{step:>6d}  {last_train_loss:>8.4f}  {v_loss:>8.4f}  "
                  f"{v_acc:>8.3f}  {e_acc:>8.3f}  {qs}")

    v_loss, v_acc, e_acc, _ = _val(model, T, p, 1024, device)
    secs = time.perf_counter() - t0
    return RunResult(
        arch=label, T=T, p=p, steps=steps,
        final_train_loss=last_train_loss, final_val_loss=v_loss,
        end_token_acc=e_acc, per_tok_acc=v_acc,
        secs=secs, params=model.num_params(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", type=str, default=None,
                   help="comma-separated arch names for homogeneous models, e.g. 'linear,ortho,deltanet'")
    p.add_argument("--layers", type=str, default=None,
                   help="comma-separated arch list for ONE hybrid model, e.g. 'ortho,deltanet,ortho,deltanet'")
    p.add_argument("--p", type=int, nargs="+", default=[3, 5, 7])
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

    # Build the list of (arch_label, n_layers) pairs to run.
    runs = []
    if args.arches:
        for a in args.arches.split(","):
            runs.append((a.strip(), args.n_layers))
    if args.layers:
        runs.append((args.layers, len(args.layers.split(","))))

    results = []
    for p_val in args.p:
        for arch_or_layers, n_layers in runs:
            r = train_one(
                arch_or_layers=arch_or_layers, T=args.T, p=p_val,
                steps=args.steps, batch_size=args.batch,
                d_model=args.d_model, n_layers=n_layers,
                n_heads=args.n_heads, d_head=args.d_head,
                lr=args.lr, log_every=args.log_every, seed=args.seed,
            )
            results.append(r)

    print("\n" + "=" * 90)
    print(f"{'arch':<28} {'T':>4} {'p':>3} {'end_acc':>8} {'per_tok':>8} "
          f"{'val_loss':>9} {'params':>10} {'secs':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r.arch:<28} {r.T:>4} {r.p:>3} {r.end_token_acc:>8.3f} "
              f"{r.per_tok_acc:>8.3f} {r.final_val_loss:>9.4f} "
              f"{r.params:>10,} {r.secs:>7.1f}")

    print("\nModular counting ranking by end-token accuracy:")
    by_p = {}
    for r in results:
        by_p.setdefault(r.p, []).append(r)
    for pv, rs in sorted(by_p.items()):
        ordered = sorted(rs, key=lambda r: r.end_token_acc, reverse=True)
        chance = 1.0 / pv
        print(f"  p={pv}  (chance = {chance:.3f}):")
        for r in ordered:
            verdict = "✓" if r.end_token_acc > chance + 0.10 else "✗"
            print(f"    {verdict}  {r.arch:<28}  end_acc={r.end_token_acc:.3f}  "
                  f"per_tok={r.per_tok_acc:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
