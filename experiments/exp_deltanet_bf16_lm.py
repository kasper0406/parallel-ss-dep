"""FAIR head-to-head on a DENSE next-token LM objective, in the PRODUCTION
bf16 regime vs an fp32 control: DeltaNet per-head NS matrix optimizer vs Muon.

WHY a dense-LM probe (not MQAR): MQAR's masked/sparse loss collapses to uniform
in bf16 (documented in AGENTS.md). So MQAR can only be run fp32 and cannot test
"does the per-head win survive how we ACTUALLY train" (autocast bf16 + fp32
masters + bf16 optimizer state). A pure dense next-token loss on real text is
bf16-valid, so it is the faithful test.

REGIMES (mirrors `speed_knobs.apply_speed_knobs` + `bf16_optim`):
  fp32 : fp32 fwd/bwd, fp32 optimizer state. (reproduces the known fp32 gap.)
  bf16 : torch.autocast(bf16) fwd/bwd + fp32 MASTER weights + bf16 optim state.

ARMS (matrix optimizer on q/k/v/b; everything else identical):
  A  muon    : plain Muon on ALL 2D hidden matrices.
  B  perhead : per-head NS on q/k/v/b + Muon on o_proj+MLP.

FAIRNESS: identical init (same --seed), identical data + order (cached fixed
pool + a --seed-seeded index generator), identical step budget / batch / T /
schedule, identical AdamW group at a held-constant --lr. The matrix LR
(--lr_mat) is the swept knob (>=3 per arm; report best). Val CE is evaluated in
PURE fp32 in BOTH regimes (eval autocast disabled) so the convergence metric
measures the fp32 master weights, isolating "what the optimizer learned" from
eval-time rounding -- the bf16 noise enters only through TRAINING.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from experiments.model import TinyLM
from experiments.layers import DeltaNetAttention
from experiments.exp_deltanet_precond_bf16 import build_dense_opts


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["muon", "perhead"])
    ap.add_argument("--regime", required=True, choices=["fp32", "bf16"])
    ap.add_argument("--lr_mat", type=float, required=True)
    ap.add_argument("--lr", type=float, default=3e-3, help="AdamW LR (held)")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.95)
    # arch
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=64)
    # run
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=20)
    ap.add_argument("--eval_seqs", type=int, default=256)
    ap.add_argument("--warmup_steps", type=int, default=40)
    ap.add_argument("--pool", default="runs/deltanet_bf16/pool.pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="runs/deltanet_bf16")
    return ap.parse_args()


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def evaluate(model, val, vocab, regime):
    """Val CE in PURE fp32 (autocast disabled) in BOTH regimes."""
    model.eval()
    x, y = val[:, :-1], val[:, 1:]
    tot, n = 0.0, 0
    # Small chunk: logits are (chunk*T, vocab) and vocab~49k makes this large
    # (chunk=8, T=512 -> ~0.77 GB fp32). Keeps eval peak well under the
    # training peak so the probe leaves headroom for a co-resident agent.
    with torch.autocast("cuda", enabled=False):
        for i in range(0, x.shape[0], 8):
            xb, yb = x[i:i + 8], y[i:i + 8]
            out = model(xb)
            logits = (out[0] if isinstance(out, tuple) else out)[..., :vocab]
            ce = F.cross_entropy(logits.reshape(-1, vocab), yb.reshape(-1),
                                 reduction="sum")
            tot += ce.item(); n += yb.numel()
    model.train()
    return tot / n


def main():
    args = get_args()
    dev = "cuda"
    torch.set_float32_matmul_precision("high")          # TF32 (production always on)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bf16 = args.regime == "bf16"

    blob = torch.load(args.pool, map_location=dev, weights_only=False)
    train_pool = blob["train_pool"].to(dev)             # (Ntr, T+1)
    val_pool = blob["val_pool"][: args.eval_seqs].to(dev)
    vocab = int(blob["vocab"])
    T = int(blob["T"])
    assert train_pool.shape[1] == T + 1

    # ---- identical model init across arms/regimes ----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = TinyLM(vocab_size=vocab, d_model=args.d_model,
                   n_layers=args.n_layers, n_heads=args.n_heads,
                   d_head=args.d_head, attention_cls=DeltaNetAttention,
                   max_T=T, feedback_mode="none").to(dev)
    n_params = model.num_params()

    opts, matrix_idx = build_dense_opts(
        model, arm=args.arm, regime=args.regime, lr_mat=args.lr_mat,
        lr_adamw=args.lr, wd=args.wd, momentum=args.momentum)

    # ---- identical per-step data order across arms/regimes (varies by seed) ----
    dg = torch.Generator(device=dev).manual_seed(args.seed)
    Ntr = train_pool.shape[0]

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

        idx = torch.randint(0, Ntr, (args.batch,), generator=dg, device=dev)
        seq = train_pool[idx]                            # (B, T+1)
        x, y = seq[:, :-1], seq[:, 1:]

        if step == args.warmup_steps:
            torch.cuda.synchronize(); t_train0 = time.time()

        for o in opts:
            o.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(x)
            logits = (out[0] if isinstance(out, tuple) else out)[..., :vocab]
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
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
            ce_v = evaluate(model, val_pool, vocab, args.regime)
            wall = (time.time() - t_train0) if t_train0 is not None else 0.0
            hist.append({"step": step, "train_loss": loss.item(),
                         "val_ce": ce_v, "wall": wall})
            print(f"[{args.tag}] step {step:5d}  train {loss.item():.4f}  "
                  f"val_ce {ce_v:.4f}  wall {wall:6.1f}s")

    torch.cuda.synchronize()
    total_wall = time.time() - t_train0
    summary = {
        "tag": args.tag, "arm": args.arm, "regime": args.regime,
        "lr_mat": args.lr_mat, "lr_adamw": args.lr, "wd": args.wd,
        "seed": args.seed, "n_params": n_params, "steps": args.steps,
        "batch": args.batch, "T": T, "vocab": vocab,
        "d_model": args.d_model, "n_layers": args.n_layers,
        "n_heads": args.n_heads, "d_head": args.d_head,
        "total_wall_s": total_wall,
        "ms_per_step": total_wall / max(1, args.steps - args.warmup_steps) * 1e3,
        "opt_ms_per_step": opt_ms_acc / max(1, opt_ms_n),
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 1e9,
        "step0_train_loss": hist[0]["train_loss"],
        "final_val_ce": hist[-1]["val_ce"],
        "best_val_ce": min(h["val_ce"] for h in hist),
        "history": hist,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUMMARY {args.tag}: final_ce={summary['final_val_ce']:.4f} "
          f"best_ce={summary['best_val_ce']:.4f} "
          f"opt_ms/step={summary['opt_ms_per_step']:.2f} -> {out}")


if __name__ == "__main__":
    main()
