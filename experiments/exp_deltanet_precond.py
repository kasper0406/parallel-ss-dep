"""FAIR head-to-head: DeltaNet-tailored matrix optimizer vs plain Muon, on MQAR.

ARMS (matrix optimizer on the DeltaNet q/k/v/b projections; EVERYTHING else
identical):
  A  muon          : plain torch.optim.Muon on ALL 2D hidden matrices.
  B  perhead       : DeltaNetProjMuon(per-head NS) on q/k/v/b; Muon on o_proj+MLP.
  C  qk_coupled    : DeltaNetProjMuon(qk-coupled NS) on q/k/v/b; Muon on o_proj+MLP.

Fairness:
  * IDENTICAL model init (same --seed -> torch.manual_seed before build).
  * IDENTICAL data + order: per-step batch drawn from a generator re-seeded to
    --data_seed at the start of EVERY run; a fixed val set drawn from --val_seed.
  * IDENTICAL step budget / batch / T / n_pairs / LR-schedule.
  * IDENTICAL AdamW group (embeddings/lm_head/pos/1D/conv) at the SAME --lr.
  * The matrix LR (--lr_mat) is the swept knob; sweep >=3 per arm, report BEST.
  * fp32 (MQAR's masked loss collapses in bf16 -- documented). The FLA kernel
    still runs bf16 internally (the wrapper autocasts just the kernel), as in
    train_mqar.py, so this is the canonical MQAR setup.

Per-run JSON records loss/recall vs STEP and vs WALLCLOCK + isolated optimizer
ms/step, so the driver can compare iso-step AND iso-wallclock.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from experiments.model import TinyLM
from experiments.layers import DeltaNetAttention
from experiments.tasks.mqar import make_batch
from experiments.exp_deltanet_precond_optim import (
    DeltaNetProjMuon, build_units_from_model,
)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["muon", "perhead", "qk_coupled"])
    ap.add_argument("--lr_mat", type=float, required=True, help="matrix-opt LR (SWEPT)")
    ap.add_argument("--lr", type=float, default=3e-3, help="AdamW LR (held constant)")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.95)
    # arch (small DeltaNet, discriminating MQAR regime)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=64)
    # task: T=256 / K=32 sweet-spot (NOT the saturation regime)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--n_pairs", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=64)
    # run
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="model-init seed")
    ap.add_argument("--data_seed", type=int, default=1234)
    ap.add_argument("--val_seed", type=int, default=999)
    ap.add_argument("--val_batch", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=40)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="runs/deltanet_precond")
    return ap.parse_args()


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    import math
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, val, vocab):
    model.eval()
    x, y, mask = val
    out = model(x)
    logits = (out[0] if isinstance(out, tuple) else out)[..., :vocab]
    ce = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                         reduction="none").reshape(x.shape[0], x.shape[1])
    ce = (ce * mask).sum() / mask.sum().clamp_min(1)
    preds = logits.argmax(-1)
    recall = (((preds == y) & mask).float().sum() / mask.sum().clamp_min(1)).item()
    model.train()
    return ce.item(), recall


def build_opts(model, arm, lr_mat, lr_adamw, wd, momentum, steps):
    """Return list of optimizers. The MATRIX optimizer(s) use lr_mat; the
    AdamW group uses lr_adamw. Returns (opts, matrix_opt_indices)."""
    # AdamW set: embeddings/lm_head/pos + every non-2D param (1D norms, 3D conv).
    adamw_params, embed_like = [], {"embed.weight", "pos_embed.weight", "lm_head.weight"}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in embed_like or p.ndim != 2:
            adamw_params.append(p)
    opts = [torch.optim.AdamW(adamw_params, lr=lr_adamw, betas=(0.9, 0.95),
                              weight_decay=0.0)]
    matrix_idx = []
    if arm == "muon":
        # ALL 2D hidden matrices (attn q/k/v/b/o + MLP) -> plain Muon.
        muon_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad or p.ndim != 2 or n in embed_like:
                continue
            muon_params.append(p)
        opts.append(torch.optim.Muon(muon_params, lr=lr_mat, momentum=momentum,
                                     weight_decay=wd, nesterov=True))
        matrix_idx.append(1)
    else:
        tailored_params, units, other_2d = build_units_from_model(model, mode=arm)
        opts.append(torch.optim.Muon(other_2d, lr=lr_mat, momentum=momentum,
                                     weight_decay=wd, nesterov=True))
        opts.append(DeltaNetProjMuon(units, lr=lr_mat, momentum=momentum,
                                     weight_decay=wd, nesterov=True))
        matrix_idx.extend([1, 2])
    return opts, matrix_idx


def main():
    args = get_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- identical model init across arms ----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = TinyLM(vocab_size=args.vocab, d_model=args.d_model,
                   n_layers=args.n_layers, n_heads=args.n_heads,
                   d_head=args.d_head, attention_cls=DeltaNetAttention,
                   max_T=args.T, feedback_mode="none").to(dev)
    n_params = model.num_params()

    opts, matrix_idx = build_opts(model, args.arm, args.lr_mat, args.lr,
                                  args.wd, args.momentum, args.steps)

    # ---- fixed val set ----
    vg = torch.Generator(device=dev).manual_seed(args.val_seed)
    val = make_batch(args.val_batch, args.T, vocab_size=args.vocab,
                     n_pairs=args.n_pairs, device=dev, generator=vg)

    # ---- identical per-step data across arms ----
    dg = torch.Generator(device=dev).manual_seed(args.data_seed)

    hist = []
    opt_ms_acc, opt_ms_n = 0.0, 0
    t_train0 = None
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for step in range(args.steps):
        lr_mat = cosine_lr(step, args.steps, args.warmup_steps, args.lr_mat)
        lr_ad = cosine_lr(step, args.steps, args.warmup_steps, args.lr)
        for i in matrix_idx:
            for g in opts[i].param_groups:
                g["lr"] = lr_mat
        for g in opts[0].param_groups:
            g["lr"] = lr_ad

        x, y, mask = make_batch(args.batch, args.T, vocab_size=args.vocab,
                                n_pairs=args.n_pairs, device=dev, generator=dg)
        if step == args.warmup_steps:
            torch.cuda.synchronize(); t_train0 = time.time()

        for o in opts:
            o.zero_grad(set_to_none=True)
        out = model(x)
        logits = (out[0] if isinstance(out, tuple) else out)[..., :args.vocab]
        ce = F.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1),
                             reduction="none").reshape(args.batch, args.T)
        loss = (ce * mask).sum() / mask.sum().clamp_min(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        torch.cuda.synchronize(); t_opt = time.time()
        for o in opts:
            o.step()
        torch.cuda.synchronize()
        if step >= args.warmup_steps:
            opt_ms_acc += (time.time() - t_opt) * 1e3
            opt_ms_n += 1

        if step % args.eval_every == 0 or step == args.steps - 1:
            ce_v, rec = evaluate(model, val, args.vocab)
            wall = (time.time() - t_train0) if t_train0 is not None else 0.0
            hist.append({"step": step, "train_loss": loss.item(),
                         "val_ce": ce_v, "val_recall": rec, "wall": wall})
            print(f"[{args.tag}] step {step:5d}  train {loss.item():.4f}  "
                  f"val_ce {ce_v:.4f}  recall {rec:.3f}  wall {wall:6.1f}s")

    torch.cuda.synchronize()
    total_wall = time.time() - t_train0
    summary = {
        "tag": args.tag, "arm": args.arm, "lr_mat": args.lr_mat,
        "lr_adamw": args.lr, "wd": args.wd, "seed": args.seed,
        "n_params": n_params, "steps": args.steps, "batch": args.batch,
        "T": args.T, "n_pairs": args.n_pairs, "vocab": args.vocab,
        "d_model": args.d_model, "n_layers": args.n_layers,
        "n_heads": args.n_heads, "d_head": args.d_head,
        "total_wall_s": total_wall,
        "ms_per_step": total_wall / max(1, args.steps - args.warmup_steps) * 1e3,
        "opt_ms_per_step": opt_ms_acc / max(1, opt_ms_n),
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 1e9,
        "final_val_recall": hist[-1]["val_recall"],
        "final_val_ce": hist[-1]["val_ce"],
        "best_val_recall": max(h["val_recall"] for h in hist),
        "history": hist,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUMMARY {args.tag}: final_recall={summary['final_val_recall']:.3f} "
          f"best_recall={summary['best_val_recall']:.3f} "
          f"opt_ms/step={summary['opt_ms_per_step']:.2f} "
          f"ms/step={summary['ms_per_step']:.1f} -> {out}")


if __name__ == "__main__":
    main()
