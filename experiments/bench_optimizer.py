"""Controlled optimizer benchmark: Muon+AdamW vs SOAP+AdamW (vs others).

Fair-baseline discipline:
  * SAME model init (seed), SAME cached data + order, SAME batch/T/schedule
    across every arm — only the optimizer (and its LR) changes.
  * The matrix optimizer (Muon or SOAP) operates on EXACTLY the same 2D
    hidden-matrix params (see optim_utils.build_optimizer); embeddings /
    lm_head / 1D / α route to AdamW identically in both arms. So a
    muon-vs-soap A/B isolates the matrix optimizer.
  * Faithfully-scaled-down DeltaNet+FiLM trunk (real arch, fewer layers /
    narrower / self_k=1, T=512) so a full per-optimizer LR sweep finishes
    quickly. The 2D matrices the optimizer preconditions (d_model and d_ff
    sides) are at production-relevant sizes so SOAP's eigh/qr cost is
    representative.
  * Reports loss-vs-STEP and loss-vs-WALLCLOCK, steps/seconds-to-target,
    per-step ms, ISOLATED optimizer-step ms (cuda-synced — compile-
    independent), and peak memory. torch.compile OFF for the sweep so the
    forward/backward (identical across arms) is plain eager and the
    optimizer.step() delta is the only moving part.

Usage (single arm):
  python experiments/bench_optimizer.py --optimizer soap --lr_muon 3e-3 \
      --tag soap_lr3e-3 --steps 800
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from experiments.build_arch import build_arch
from experiments.model import TinyLM
from experiments.model_builder import _parse_feedback_pairs
from experiments.optim_utils import build_optimizer


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="muon",
                    choices=["muon", "soap", "adamw"])
    ap.add_argument("--lr", type=float, default=1.4e-3,
                    help="AdamW LR (embeddings/lm_head/1D/α). HELD CONSTANT "
                         "across arms — both Muon and SOAP arms use AdamW for "
                         "these, so this is a controlled, not swept, knob.")
    ap.add_argument("--lr_muon", type=float, default=5e-3,
                    help="Matrix-optimizer LR (Muon OR SOAP). THE SWEPT KNOB.")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--soap_precond_freq", type=int, default=10)
    ap.add_argument("--soap_normalize_grads", action="store_true")
    ap.add_argument("--lr_schedule", default="wsd")
    ap.add_argument("--warmup_steps", type=int, default=40)
    ap.add_argument("--decay_frac", type=float, default=0.15)
    # arch (faithfully-scaled-down DeltaNet + dense FiLM)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--feedback_pairs", default="0,4;1,5;2,6;3,7")
    ap.add_argument("--feedback_self_k", type=int, default=1)
    # run
    ap.add_argument("--data", default="runs/bench_optim_data.pt")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=40)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="runs/bench_optim")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, Xva, Yva, batch, device):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, Xva.shape[0], batch):
        x = Xva[i:i + batch].to(device).long()
        y = Yva[i:i + batch].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             y.reshape(-1), ignore_index=-100)
        tot += ce.item() * x.shape[0]
        n += x.shape[0]
    model.train()
    return tot / max(1, n)


def main():
    args = get_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    blob = torch.load(args.data, weights_only=False)
    Xtr, Ytr, Xva, Yva = blob["Xtr"], blob["Ytr"], blob["Xva"], blob["Yva"]
    vocab = blob["model_vocab_size"]
    T = blob["T"]
    print(f"data: train {tuple(Xtr.shape)} val {tuple(Xva.shape)} "
          f"vocab={vocab} T={T}")

    # Deterministic, identical init across arms.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    attn_kw = build_arch("deltanet", args.n_layers)
    fb_pairs = _parse_feedback_pairs(args.feedback_pairs)
    model = TinyLM(
        vocab_size=vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head,
        feedback_mode="film", feedback_pairs=fb_pairs,
        feedback_self_k=args.feedback_self_k,
        **attn_kw,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params "
          f"(d_model={args.d_model} n_layers={args.n_layers} "
          f"fb_pairs={args.feedback_pairs} self_k={args.feedback_self_k})")

    opts, scheds = build_optimizer(
        model, optimizer=args.optimizer, lr=args.lr, lr_muon=args.lr_muon,
        alpha_wd=0.0, steps=args.steps, wd=args.wd,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup_steps,
        decay_frac=args.decay_frac, soap_precond_freq=args.soap_precond_freq,
        soap_normalize_grads=args.soap_normalize_grads,
    )

    nb = Xtr.shape[0] // args.batch
    history = []
    opt_ms_acc, opt_ms_n = 0.0, 0
    step_ms_acc, step_ms_n = 0.0, 0   # pure train step (excl. eval/log)
    t_train0 = time.time()            # overwritten at step 5 (warmup excl.)
    torch.cuda.reset_peak_memory_stats()  # so peak_mem is this run only
    model.train()

    for step in range(args.steps):
        bi = step % nb
        x = Xtr[bi * args.batch:(bi + 1) * args.batch].to(device).long()
        y = Ytr[bi * args.batch:(bi + 1) * args.batch].to(device)

        # Warmup steps (compile/cudnn autotune) excluded from wall-clock.
        if step == 5:
            torch.cuda.synchronize()
            t_train0 = time.time()
        torch.cuda.synchronize()
        t_step = time.time()

        for o in opts:
            o.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Isolated, cuda-synced optimizer-step time (compile-independent).
        torch.cuda.synchronize()
        t_opt = time.time()
        for o in opts:
            o.step()
        torch.cuda.synchronize()
        if step >= 5:
            opt_ms_acc += (time.time() - t_opt) * 1e3
            opt_ms_n += 1
        for s in scheds:
            s.step()

        torch.cuda.synchronize()
        if step >= 5:
            step_ms_acc += (time.time() - t_step) * 1e3
            step_ms_n += 1

        if step % args.log_every == 0 or step == args.steps - 1:
            wall = (time.time() - t_train0) if t_train0 is not None else 0.0
            rec = {"step": step, "train_loss": loss.item(), "wall": wall}
            if step % args.eval_every == 0 or step == args.steps - 1:
                rec["val_loss"] = evaluate(model, Xva, Yva, args.batch, device)
            history.append(rec)
            vl = rec.get("val_loss")
            print(f"step {step:4d}  train {loss.item():.4f}"
                  + (f"  val {vl:.4f}" if vl is not None else "")
                  + f"  wall {wall:6.1f}s  lr_mat {scheds[0].get_last_lr()[0]:.2e}")

    torch.cuda.synchronize()
    total_wall = time.time() - t_train0
    steps_timed = args.steps - 5
    peak_mem_gib = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "tag": args.tag, "optimizer": args.optimizer,
        "lr_muon": args.lr_muon, "lr_adamw": args.lr, "wd": args.wd,
        "soap_precond_freq": args.soap_precond_freq,
        "n_params_M": n_params / 1e6,
        "steps": args.steps, "batch": args.batch, "T": T,
        "total_wall_s": total_wall,
        "ms_per_step": step_ms_acc / max(1, step_ms_n),       # pure train step
        "ms_per_step_incl_eval": total_wall / steps_timed * 1e3,
        "opt_ms_per_step": opt_ms_acc / max(1, opt_ms_n),
        "peak_mem_gib": peak_mem_gib,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1].get("val_loss"),
        "history": history,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUMMARY {args.tag}: ms/step={summary['ms_per_step']:.1f} "
          f"opt_ms/step={summary['opt_ms_per_step']:.2f} "
          f"peak_mem={peak_mem_gib:.2f}GiB "
          f"final_val={summary['final_val_loss']:.4f} → {out}")


if __name__ == "__main__":
    main()
