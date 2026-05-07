"""
DN-4B distillation pilot — full-scale training (Phase B).

Scales the validated KL+CE recipe from `distill_pilot.py` to the production
target: a ~3 B plain-DN student trained on a ~50 M-token teacher-aligned
corpus from Qwen3.6-35B-A3B-AWQ.

Differences vs the validation script (`distill_pilot.py`):

  - Muon optimizer for ≥2D matrices, AdamW for 1D + embeddings/lm_head
    (per `train_lm.py` convention). lr_muon=1e-3, lr_adamw=3e-4.
  - Gradient accumulation for an effective batch larger than the
    activation-fitted micro-batch.
  - Streaming shard load (still all in RAM, but as fp16/int32 to halve
    memory; cast to fp32 only on the per-batch path).
  - Periodic checkpoint saving every val_every steps (overwrite a
    "latest" file plus a separate "best by val PPL" file).
  - Robust to long-running interrupts: each val emits a JSONL line that
    can be inspected mid-run; the running summary is written every
    val_every too.

Recipe (defaults match the brief):

    arch=deltanet (no FiLM, no K=3, no L_sem)
    optimizer=muon (Muon for ≥2D, AdamW for 1D/embed/head)
    lr_muon=1e-3, lr=3e-4
    T=512 batch=2 grad_accum=2 (effective batch=4)
    steps=30000-50000
    KL+CE: alpha=0.9 (90% CE + 10% KL)
    top_K=20 teacher logprobs (vLLM 0.20 cap)
    log_every=500 val_every=2500 seed=0

Usage:
  CUDA_VISIBLE_DEVICES=0 \\
    /home/knielsen/ml/parallel-ss-dep/.venv/bin/python -u \\
    experiments/distill_pilot_full.py \\
    --shards data/distill_pilot_50M \\
    --mode kl_ce --alpha 0.9 --top_k 20 \\
    --d_model 2048 --n_heads 32 --d_head 64 --n_layers 32 \\
    --batch 2 --grad_accum 2 --steps 30000 \\
    --log_every 500 --val_every 2500 \\
    --save_ckpt checkpoints/dn_4B_distilled_qwen3p6.pt \\
    --save_metrics logs/distill_pilot_full/dn_4B_train.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


def load_shards(shard_dir: str, dtype_logprob=np.float16):
    """Load all NPZ shards into RAM. Returns concatenated arrays.

    Keeps top_lps as fp16 to halve RAM (it's converted to float on the
    per-batch path, which is small enough to be cheap).
    """
    sd = pathlib.Path(shard_dir)
    shards = sorted(sd.glob("shard_*.npz"))
    print(f"Loading {len(shards)} shards from {sd} ...", flush=True)
    token_ids_list, top_ids_list, top_lps_list = [], [], []
    for s in shards:
        z = np.load(s)
        token_ids_list.append(z["token_ids"])
        top_ids_list.append(z["top_k_ids"])
        top_lps_list.append(z["top_k_logprobs"].astype(dtype_logprob))
    token_ids = np.concatenate(token_ids_list, axis=0)
    top_ids = np.concatenate(top_ids_list, axis=0)
    top_lps = np.concatenate(top_lps_list, axis=0)
    print(f"  token_ids:      {token_ids.shape} {token_ids.dtype} "
          f"({token_ids.nbytes/1e9:.2f} GB)", flush=True)
    print(f"  top_k_ids:      {top_ids.shape} {top_ids.dtype} "
          f"({top_ids.nbytes/1e9:.2f} GB)", flush=True)
    print(f"  top_k_logprobs: {top_lps.shape} {top_lps.dtype} "
          f"({top_lps.nbytes/1e9:.2f} GB)", flush=True)
    return token_ids, top_ids, top_lps


def compute_loss(
    student_logits: torch.Tensor,    # (B, T, V)
    target_token_ids: torch.Tensor,  # (B, T) int — actual next-token at each pos
    teacher_top_ids: torch.Tensor,   # (B, T, K) int — teacher's top-K at each pos
    teacher_top_lps: torch.Tensor,   # (B, T, K) float — teacher logprobs
    kl_weight: float,
    ce_weight: float,
):
    """KL distillation loss + CE on ground-truth next token.

    Identical to `distill_pilot.compute_loss`. Position semantics: at
    position t, teacher's slot t is the teacher's distribution over what
    should appear AT t. For autoregressive student: student_logits[b, t-1]
    predicts x[b, t]. Shift student left and align with teacher slots
    [1..T-1].
    """
    sl = student_logits[:, :-1, :]        # (B, T-1, V)
    tgt = target_token_ids[:, 1:]         # (B, T-1)
    tk_ids = teacher_top_ids[:, 1:, :]    # (B, T-1, K)
    tk_lps = teacher_top_lps[:, 1:, :]    # (B, T-1, K)

    B, Tm1, V = sl.shape
    ce = F.cross_entropy(sl.reshape(-1, V), tgt.reshape(-1))

    valid = torch.isfinite(tk_lps[..., 0])  # (B, T-1)
    if not valid.any() or kl_weight == 0.0:
        return ce_weight * ce, ce, torch.zeros_like(ce)

    gathered = sl.gather(2, tk_ids.long())  # (B, T-1, K)
    tk_lps_safe = torch.where(torch.isfinite(tk_lps), tk_lps,
                              torch.full_like(tk_lps, -1e3))
    teacher_log_p = F.log_softmax(tk_lps_safe.float(), dim=-1)
    teacher_p = teacher_log_p.exp()
    student_log_p = F.log_softmax(gathered.float(), dim=-1)

    kl_per_pos = (teacher_p * (teacher_log_p - student_log_p)).sum(dim=-1)
    kl_per_pos = torch.where(valid, kl_per_pos,
                             torch.zeros_like(kl_per_pos))
    kl = kl_per_pos.sum() / valid.float().sum().clamp(min=1)

    total = kl_weight * kl + ce_weight * ce
    return total, ce, kl


def build_optimizer_muon(model, lr_muon=1e-3, lr_adamw=3e-4, weight_decay=0.1):
    """Muon for 2D non-embed matrices, AdamW for embeddings/lm_head/1D
    params (RMSNorm scales etc.). Matches the convention in train_lm.py."""
    embed_or_head_names = {"embed.weight", "pos_embed.weight", "lm_head.weight"}
    muon_params, adamw_params = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if name in embed_or_head_names or p.ndim != 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    print(f"  optimizer split: {len(muon_params)} Muon params "
          f"({sum(p.numel() for p in muon_params)/1e6:.1f}M), "
          f"{len(adamw_params)} AdamW params "
          f"({sum(p.numel() for p in adamw_params)/1e6:.1f}M)", flush=True)
    opts = [
        torch.optim.Muon(muon_params, lr=lr_muon,
                         momentum=0.95, weight_decay=weight_decay),
        torch.optim.AdamW(adamw_params, lr=lr_adamw,
                          betas=(0.9, 0.95), weight_decay=weight_decay),
    ]
    return opts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=str, required=True,
                   help="Directory of pre-extracted NPZ shards (teacher data).")
    p.add_argument("--mode", type=str, default="kl_ce",
                   choices=["kl_ce", "ce"])
    p.add_argument("--alpha", type=float, default=0.9,
                   help="CE weight (kl_weight = 1 - alpha for kl_ce mode).")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--d_model", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=32)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=32)
    p.add_argument("--tie_embeddings", action="store_true", default=True)
    p.add_argument("--no_tie_embeddings", action="store_false",
                   dest="tie_embeddings")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accumulation steps. Effective batch = "
                        "batch * grad_accum.")
    p.add_argument("--steps", type=int, default=30000,
                   help="Number of optimizer steps (gradient updates).")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="AdamW lr (embeddings/lm_head/1D).")
    p.add_argument("--lr_muon", type=float, default=1e-3,
                   help="Muon lr (≥2D hidden matrices).")
    p.add_argument("--optimizer", type=str, default="muon",
                   choices=["muon", "adamw"])
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="Linear warmup, after which cosine decay to 10%.")
    p.add_argument("--amp_dtype", type=str, default="bf16",
                   choices=["fp32", "bf16"])
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--val_every", type=int, default=2500)
    p.add_argument("--val_chunks", type=int, default=128,
                   help="How many chunks to hold out for validation.")
    p.add_argument("--save_metrics", type=str, default=None)
    p.add_argument("--save_ckpt", type=str, default=None,
                   help="Final checkpoint path. A '.latest' suffix is also "
                        "saved every val_every and overwritten.")
    p.add_argument("--save_jsonl", type=str, default=None,
                   help="Per-val JSONL emission for live monitoring.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"GPU: {torch.cuda.get_device_name(0)}  mode={args.mode}  "
          f"alpha={args.alpha}  steps={args.steps}  seed={args.seed}", flush=True)

    if args.mode == "ce":
        kl_w, ce_w = 0.0, 1.0
    else:
        ce_w = float(args.alpha)
        kl_w = 1.0 - ce_w
    print(f"  loss weights: kl={kl_w}, ce={ce_w}", flush=True)
    print(f"  effective batch = batch={args.batch} * grad_accum="
          f"{args.grad_accum} = {args.batch * args.grad_accum}", flush=True)

    token_ids, top_ids, top_lps = load_shards(args.shards)
    n_chunks_total, T = token_ids.shape
    K_avail = top_ids.shape[2]
    K = min(args.top_k, K_avail)
    if K < args.top_k:
        print(f"  warning: requested top_k={args.top_k} but shard only has "
              f"K={K_avail}; using K={K}", flush=True)
    if K < K_avail:
        top_ids = top_ids[:, :, :K]
        top_lps = top_lps[:, :, :K]

    manifest_path = pathlib.Path(args.shards) / "manifest.npz"
    manifest_vs = 0
    teacher_model = "(unknown)"
    dataset_name = "(unknown)"
    if manifest_path.exists():
        m = np.load(manifest_path, allow_pickle=True)
        manifest_vs = int(m["vocab_size"])
        teacher_model = str(m["model"])
        dataset_name = str(m["dataset"])
    obs_max = max(int(token_ids.max()), int(top_ids.max()))
    vocab_size = max(manifest_vs, obs_max + 1)
    vocab_size = ((vocab_size + 63) // 64) * 64
    print(f"  vocab_size={vocab_size}  T={T}  K={K}", flush=True)
    print(f"  teacher: {teacher_model}", flush=True)
    print(f"  dataset: {dataset_name}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_chunks_total)
    val_idx = perm[: args.val_chunks]
    train_idx = perm[args.val_chunks:]
    print(f"  train chunks: {len(train_idx)}, val chunks: {len(val_idx)}",
          flush=True)

    # 2) Build student.
    print(f"\nBuilding student: vocab={vocab_size} d_model={args.d_model} "
          f"n_heads={args.n_heads} d_head={args.d_head} "
          f"n_layers={args.n_layers} tie_emb={args.tie_embeddings}",
          flush=True)
    model = TinyLM(
        vocab_size=vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, d_head=args.d_head, n_layers=args.n_layers,
        attention_cls=DeltaNetAttention,
        feedback_mode="none", feedback_pairs=(),
        tie_embeddings=args.tie_embeddings,
    ).cuda()
    n_params = model.num_params()
    print(f"  params: {n_params / 1e6:.1f} M  ({n_params / 1e9:.3f} B)",
          flush=True)

    # 3) Optimizer.
    if args.optimizer == "muon":
        opts = build_optimizer_muon(model, lr_muon=args.lr_muon,
                                     lr_adamw=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.95),
                                   weight_decay=args.weight_decay)]

    # Warmup + cosine to 10% (per LR group).
    def make_sched(opt, base_lr):
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(
                1, args.steps - args.warmup_steps)
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cos     # decay to 10% of base lr
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scheds = []
    for opt in opts:
        # base lr per group
        base = opt.param_groups[0]["lr"]
        scheds.append(make_sched(opt, base))

    print(f"  total train steps: {args.steps}, warmup={args.warmup_steps}",
          flush=True)

    token_ids_t = torch.from_numpy(token_ids).long()
    top_ids_t = torch.from_numpy(top_ids).long()
    # Keep teacher logprobs in fp16 on CPU; cast to float when on GPU.
    top_lps_t = torch.from_numpy(top_lps)
    print(f"\nData tensors built (CPU): {token_ids_t.shape}, "
          f"{top_ids_t.shape}, {top_lps_t.shape} (fp16 on CPU)", flush=True)

    def get_batch(idx_array, step_idx, micro):
        # micro is the current micro-batch index within a global step
        eff = step_idx * args.grad_accum + micro
        i = (eff * args.batch) % max(1, len(idx_array) - args.batch + 1)
        sel = idx_array[i : i + args.batch]
        return (token_ids_t[sel].cuda(non_blocking=True),
                top_ids_t[sel].cuda(non_blocking=True),
                top_lps_t[sel].cuda(non_blocking=True).float())

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32
    use_amp = amp_dtype != torch.float32

    @torch.no_grad()
    def run_validation(step):
        model.eval()
        v_loss = v_ce = v_kl = 0.0
        v_n = 0
        for vi in range(0, len(val_idx) - args.batch + 1, args.batch):
            sel = val_idx[vi : vi + args.batch]
            x = token_ids_t[sel].cuda()
            ti = top_ids_t[sel].cuda()
            tp = top_lps_t[sel].cuda().float()
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                 enabled=use_amp):
                logits = model(x)
                loss, ce, kl = compute_loss(
                    logits, x, ti, tp, kl_weight=kl_w, ce_weight=ce_w)
            v_loss += loss.item() * args.batch
            v_ce += ce.item() * args.batch
            v_kl += kl.item() * args.batch
            v_n += args.batch
        if v_n == 0:
            return None
        v_loss /= v_n; v_ce /= v_n; v_kl /= v_n
        ppl = math.exp(v_ce)
        print(f"    [VAL @ step {step}]  loss={v_loss:.4f}  "
              f"ce={v_ce:.4f}  kl={v_kl:.4f}  ppl={ppl:.2f}  (n={v_n})",
              flush=True)
        model.train()
        return {"step": step, "val_loss": v_loss, "val_ce": v_ce,
                "val_kl": v_kl, "val_ppl": ppl, "n": v_n,
                "wall_s": time.time() - t_global0}

    def save_ckpt(path, step, summary_so_far):
        if path is None:
            return
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": summary_so_far,
            "step": step,
        }, path)
        print(f"    [ckpt] saved {path} ({path.stat().st_size/1e9:.2f} GB)",
              flush=True)

    def emit_jsonl(d):
        if args.save_jsonl is None:
            return
        path = pathlib.Path(args.save_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(d) + "\n")

    # 4) Train.
    print(f"\nStarting {args.mode} training for {args.steps} steps "
          f"(× {args.grad_accum} grad-accum micro-steps each) ...", flush=True)
    t_global0 = time.time()
    t0 = t_global0
    model.train()
    losses, ces, kls = [], [], []
    val_history = []
    best_val_ppl = float("inf")

    init_val = run_validation(0)
    if init_val is not None:
        val_history.append(init_val)
        emit_jsonl({"event": "val", **init_val})

    summary_template = {
        "mode": args.mode, "alpha": args.alpha,
        "kl_weight": kl_w, "ce_weight": ce_w,
        "top_k": K, "steps": args.steps,
        "batch": args.batch, "grad_accum": args.grad_accum,
        "lr": args.lr, "lr_muon": args.lr_muon,
        "warmup_steps": args.warmup_steps,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "arch": "deltanet",
        "tokenizer": "QuantTrio/Qwen3.6-35B-A3B-AWQ",  # via vLLM; matches the teacher
        "feedback_mode": "none",
        "feedback_pairs": (),
        "feedback_self_k": 0,
        "d_model": args.d_model, "n_heads": args.n_heads,
        "d_head": args.d_head, "n_layers": args.n_layers,
        "tie_embeddings": args.tie_embeddings,
        "vocab_size": vocab_size, "T": T,
        "max_T": 0,                   # DN has no positional embedding
        "params_M": n_params / 1e6,
        "shards": str(args.shards),
        "teacher": teacher_model, "dataset": dataset_name,
    }

    for step in range(args.steps):
        # Gradient accumulation: micro_grad_accum micro-steps before opt.step()
        micro_loss = micro_ce = micro_kl = 0.0
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x_ids, tk_ids, tk_lps = get_batch(train_idx, step, micro)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                 enabled=use_amp):
                logits = model(x_ids)
                loss, ce, kl = compute_loss(
                    logits, x_ids, tk_ids, tk_lps,
                    kl_weight=kl_w, ce_weight=ce_w,
                )
            (loss / args.grad_accum).backward()
            micro_loss += loss.item()
            micro_ce += ce.item()
            micro_kl += kl.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in opts:
            opt.step()
        for sched in scheds:
            sched.step()
        losses.append(micro_loss / args.grad_accum)
        ces.append(micro_ce / args.grad_accum)
        kls.append(micro_kl / args.grad_accum)

        if step == 0 or (step + 1) % args.log_every == 0:
            n_recent = min(args.log_every, len(losses))
            ll = sum(losses[-n_recent:]) / n_recent
            cc = sum(ces[-n_recent:]) / n_recent
            kk = sum(kls[-n_recent:]) / n_recent
            elapsed = time.time() - t0
            tps = ((step + 1) * args.batch * args.grad_accum * T
                   / max(elapsed, 1e-3))
            lrs = [s.get_last_lr()[0] for s in scheds]
            lrs_s = " / ".join(f"{l:.2e}" for l in lrs)
            print(f"  step {step+1:>6d}/{args.steps}  "
                  f"loss={ll:.4f}  ce={cc:.4f}  kl={kk:.4f}  "
                  f"tok/s={tps:.0f}  lr={lrs_s}", flush=True)
            emit_jsonl({"event": "train", "step": step + 1,
                        "loss": ll, "ce": cc, "kl": kk,
                        "tps": tps, "wall_s": elapsed})

        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            v = run_validation(step + 1)
            if v is not None:
                val_history.append(v)
                emit_jsonl({"event": "val", **v})
                # Save latest + best (by val PPL).
                summary_so_far = {
                    **summary_template,
                    "wallclock_s": time.time() - t_global0,
                    "final_val": v,
                    "val_history": val_history,
                }
                if args.save_ckpt:
                    latest = pathlib.Path(args.save_ckpt).with_suffix(
                        ".latest.pt")
                    save_ckpt(latest, step + 1, summary_so_far)
                    if v["val_ppl"] < best_val_ppl:
                        best_val_ppl = v["val_ppl"]
                        best = pathlib.Path(args.save_ckpt).with_suffix(
                            ".best.pt")
                        save_ckpt(best, step + 1, summary_so_far)

                # Always also write a running summary JSON.
                if args.save_metrics:
                    out = pathlib.Path(args.save_metrics)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(summary_so_far, indent=2))

    # 5) Save final.
    final_val = val_history[-1] if val_history else None
    summary = {
        **summary_template,
        "wallclock_s": time.time() - t_global0,
        "final_val": final_val,
        "best_val_ppl": best_val_ppl,
        "val_history": val_history,
    }

    print("\n=== summary ===", flush=True)
    if final_val is not None:
        print(f"  mode={args.mode}  final_val_ce={final_val['val_ce']:.4f}  "
              f"final_val_ppl={final_val['val_ppl']:.2f}  "
              f"final_val_kl={final_val['val_kl']:.4f}", flush=True)
    print(f"  best val PPL across run: {best_val_ppl:.4f}", flush=True)
    print(f"  wallclock: {summary['wallclock_s']:.1f} s "
          f"({summary['wallclock_s']/3600:.2f} h)", flush=True)

    if args.save_metrics:
        out = pathlib.Path(args.save_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"  metrics written to {out}", flush=True)

    if args.save_ckpt:
        save_ckpt(args.save_ckpt, args.steps, summary)


if __name__ == "__main__":
    main()
