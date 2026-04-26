"""
Variable-binding pointer-chasing training driver.

Tests whether SymbolGroundedAttention's id-keyed sparse table actually
delivers what the design promises: O(1) lookup of the latest binding for
a queried variable, beating linear-RNN cells (DeltaNet) which must
compress the entire binding history into their fixed state.

Usage:
    python experiments/train_var_binding.py --arches deltanet,symgrounded \\
        --T 128 --steps 3000 --batch 128
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
    LinearAttention, SoftmaxAttention,
    DeltaNetAttention, GatedDeltaNetAttention,
    OrthogonalScanAttention, SymbolGroundedAttention,
)
from experiments.model import TinyLM
from experiments.tasks.var_binding import make_batch, vocab_size


def build_attention_cls(arch: str, vocab_sz: int):
    """Return a callable that the TinyLM scaffold can use as attention_cls.

    For SymbolGrounded we partially apply vocab_size; for everything else
    we just return the class.
    """
    if arch == "linear":      return LinearAttention
    if arch == "softmax":     return SoftmaxAttention
    if arch == "deltanet":    return DeltaNetAttention
    if arch == "gateddelta":  return GatedDeltaNetAttention
    if arch == "ortho":       return OrthogonalScanAttention
    if arch == "symgrounded":
        def _factory(**kw):
            return SymbolGroundedAttention(vocab_size=vocab_sz, **kw)
        return _factory
    if arch == "hybrid_so":
        # Alternating SymbolGrounded + DeltaNet — analog of our prior
        # hybrid but with symbol-grounded as the "specialist."
        def _so(**kw):
            return SymbolGroundedAttention(vocab_size=vocab_sz, **kw)
        return None  # signal to caller to use per-layer list
    raise ValueError(arch)


def per_layer_list(arch: str, n_layers: int, vocab_sz: int):
    """For arches that need a per-layer schedule, return the list."""
    if arch == "hybrid_so":
        def _so(**kw):
            return SymbolGroundedAttention(vocab_size=vocab_sz, **kw)
        # Standard 4 layers: alternating delta/symgrounded.
        out = []
        for i in range(n_layers):
            out.append(_so if i % 2 == 0 else DeltaNetAttention)
        return out
    return None


@dataclasses.dataclass
class RunResult:
    arch: str
    T: int
    n_vars: int
    n_vals: int
    steps: int
    final_train_loss: float
    final_val_loss: float
    val_acc: float
    secs: float
    params: int


def _val(model, T, n_vars, n_vals, batch_size, device, V):
    model.eval()
    with torch.no_grad():
        x, y, mask = make_batch(batch_size, T, n_vars=n_vars, n_vals=n_vals,
                                device=device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction="none",
        ).reshape(batch_size, T)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        preds = logits.argmax(dim=-1)
        correct = ((preds == y) & mask).float().sum()
        total = mask.sum().clamp_min(1)
        acc = (correct / total).item()
    model.train()
    return loss.item(), acc


def train_one(arch, T, n_vars, n_vals, steps, batch_size,
              d_model, n_layers, n_heads, d_head, lr, log_every,
              device="cuda", seed=0):
    torch.manual_seed(seed)
    V = vocab_size(n_vars=n_vars, n_vals=n_vals)

    cls = build_attention_cls(arch, V)
    cls_list = per_layer_list(arch, n_layers, V)
    if cls_list is not None:
        model = TinyLM(
            vocab_size=V, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head,
            attention_cls_per_layer=cls_list, max_T=T,
        ).to(device)
    else:
        model = TinyLM(
            vocab_size=V, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head,
            attention_cls=cls, max_T=T,
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1,
    )

    print(f"\n[{arch}]  T={T}  n_vars={n_vars}  n_vals={n_vals}  V={V}  "
          f"params={model.num_params():,}")
    print(f"{'step':>6}  {'train_loss':>11}  {'val_loss':>9}  {'acc':>6}")

    t0 = time.perf_counter()
    last_train_loss = float("nan")
    for step in range(1, steps + 1):
        x, y, mask = make_batch(batch_size, T, n_vars=n_vars, n_vals=n_vals,
                                device=device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1), reduction="none",
        ).reshape(batch_size, T)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        last_train_loss = loss.item()

        if step % log_every == 0 or step == steps:
            v_loss, acc = _val(model, T, n_vars, n_vals, 256, device, V)
            print(f"{step:>6d}  {last_train_loss:>11.4f}  "
                  f"{v_loss:>9.4f}  {acc:>6.3f}")

    v_loss, acc = _val(model, T, n_vars, n_vals, 1024, device, V)
    secs = time.perf_counter() - t0
    return RunResult(
        arch=arch, T=T, n_vars=n_vars, n_vals=n_vals, steps=steps,
        final_train_loss=last_train_loss, final_val_loss=v_loss,
        val_acc=acc, secs=secs, params=model.num_params(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", type=str, default="deltanet,symgrounded")
    p.add_argument("--T", type=int, nargs="+", default=[128])
    p.add_argument("--n_vars", type=int, default=8)
    p.add_argument("--n_vals", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arches = args.arches.split(",")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    for T in args.T:
        for a in arches:
            r = train_one(
                arch=a, T=T, n_vars=args.n_vars, n_vals=args.n_vals,
                steps=args.steps, batch_size=args.batch,
                d_model=args.d_model, n_layers=args.n_layers,
                n_heads=args.n_heads, d_head=args.d_head,
                lr=args.lr, log_every=args.log_every, seed=args.seed,
            )
            results.append(r)

    print("\n" + "=" * 80)
    print(f"{'arch':<14} {'T':>5} {'acc':>6} {'val_loss':>9} "
          f"{'params':>10} {'secs':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r.arch:<14} {r.T:>5} {r.val_acc:>6.3f} "
              f"{r.final_val_loss:>9.4f} {r.params:>10,} {r.secs:>7.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
