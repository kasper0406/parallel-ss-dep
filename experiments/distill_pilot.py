"""
DN-4B distillation pilot — validation training script.

Validates the KL+CE distillation pipeline on a ~1B plain-DN student against
a ~1M-token teacher-aligned corpus from Qwen3.6-35B-A3B-AWQ. Two modes:

  --mode kl_ce  : KL(student || teacher_topK) + alpha * CE(student, gt)
  --mode ce     : pure CE baseline (kl_weight=0). Same data, same schedule.

Both share data, schedule, optimizer, seed → the only variable is the loss.
This is exactly the comparison Phase 15 used. Difference here is the data
(teacher-aligned, not codeparrot).

Architectural choice for the pilot: PLAIN DN, no FiLM, no self-feeding K.
The architectural variable is held constant; we are stress-testing the
distillation recipe end-to-end.

Usage:
  /home/knielsen/ml/parallel-ss-dep/.venv/bin/python experiments/distill_pilot.py \
    --shards /home/knielsen/ml/parallel-ss-dep-distill/data/distill_pilot_1M \
    --mode kl_ce --steps 1000 --batch 4 \
    --d_model 1280 --n_heads 20 --d_head 64 --n_layers 24
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


def load_shards(shard_dir: str):
    """Load all NPZ shards into RAM. Returns concatenated arrays."""
    sd = pathlib.Path(shard_dir)
    shards = sorted(sd.glob("shard_*.npz"))
    print(f"Loading {len(shards)} shards from {sd} ...")
    token_ids_list, top_ids_list, top_lps_list = [], [], []
    for s in shards:
        z = np.load(s)
        token_ids_list.append(z["token_ids"])
        top_ids_list.append(z["top_k_ids"])
        top_lps_list.append(z["top_k_logprobs"])
    token_ids = np.concatenate(token_ids_list, axis=0)
    top_ids = np.concatenate(top_ids_list, axis=0)
    top_lps = np.concatenate(top_lps_list, axis=0)
    print(f"  token_ids:      {token_ids.shape} {token_ids.dtype}")
    print(f"  top_k_ids:      {top_ids.shape} {top_ids.dtype}")
    print(f"  top_k_logprobs: {top_lps.shape} {top_lps.dtype}")
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

    Position semantics: at position t in the input, teacher's prompt-logprobs
    slot t is the teacher's distribution over what should appear AT t given
    the prefix [0..t-1]. For autoregressive student: student_logits[b, t-1]
    predicts x[b, t]. So we shift student left and align with teacher slots
    [1..T-1].
    """
    sl = student_logits[:, :-1, :]        # (B, T-1, V)
    tgt = target_token_ids[:, 1:]         # (B, T-1)
    tk_ids = teacher_top_ids[:, 1:, :]    # (B, T-1, K)
    tk_lps = teacher_top_lps[:, 1:, :]    # (B, T-1, K)

    B, Tm1, V = sl.shape
    ce = F.cross_entropy(sl.reshape(-1, V), tgt.reshape(-1))

    # Mask positions where teacher top-K is invalid (all -inf or zero entries
    # at tk_ids indicating a missing slot). Slot 0 (pos=0 in original,
    # before shift) was already removed; here we still need to mask any
    # padding positions that ended up with no teacher distribution.
    valid = torch.isfinite(tk_lps[..., 0])  # (B, T-1)
    if not valid.any() or kl_weight == 0.0:
        return ce_weight * ce, ce, torch.zeros_like(ce)

    # Gather student logits at top-K positions.
    gathered = sl.gather(2, tk_ids.long())  # (B, T-1, K)

    # Teacher distribution restricted to its top-K. Re-normalise so that the
    # K probabilities sum to 1 within the support set; tail mass beyond top-K
    # is negligible (Qwen's top-32 typically covers >0.99 mass on code).
    #
    # Replace -inf entries (invalid positions, prompt mask) with a finite
    # placeholder before log_softmax to avoid NaN propagation. We'll mask
    # the resulting per-position KL via `valid` below, so the placeholder
    # values don't matter mathematically.
    tk_lps_safe = torch.where(torch.isfinite(tk_lps), tk_lps,
                              torch.full_like(tk_lps, -1e3))
    teacher_log_p = F.log_softmax(tk_lps_safe.float(), dim=-1)
    teacher_p = teacher_log_p.exp()
    student_log_p = F.log_softmax(gathered.float(), dim=-1)

    # KL(P_teacher || P_student) over top-K support.
    kl_per_pos = (teacher_p * (teacher_log_p - student_log_p)).sum(dim=-1)
    # Zero out per-position KL at invalid positions (prompt-mask) BEFORE
    # the average, so any NaN that slipped through doesn't poison the sum.
    kl_per_pos = torch.where(valid, kl_per_pos,
                             torch.zeros_like(kl_per_pos))
    kl = kl_per_pos.sum() / valid.float().sum().clamp(min=1)

    total = kl_weight * kl + ce_weight * ce
    return total, ce, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=str, required=True,
                   help="Directory of pre-extracted NPZ shards (teacher data).")
    p.add_argument("--mode", type=str, default="kl_ce",
                   choices=["kl_ce", "ce"],
                   help="kl_ce: KL+CE distillation; ce: CE-only baseline.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="CE weight (kl_weight = 1 - alpha for kl_ce mode).")
    p.add_argument("--top_k", type=int, default=32,
                   help="Top-K teacher positions to use (capped at shard's K).")
    p.add_argument("--d_model", type=int, default=1280)
    p.add_argument("--n_heads", type=int, default=20)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=24)
    p.add_argument("--tie_embeddings", action="store_true", default=True)
    p.add_argument("--no_tie_embeddings", action="store_false",
                   dest="tie_embeddings")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--amp_dtype", type=str, default="bf16",
                   choices=["fp32", "bf16"])
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_chunks", type=int, default=64,
                   help="How many chunks to hold out for validation.")
    p.add_argument("--save_metrics", type=str, default=None,
                   help="Path to write JSON metrics summary.")
    p.add_argument("--save_ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"GPU: {torch.cuda.get_device_name(0)}  mode={args.mode}  "
          f"alpha={args.alpha}  steps={args.steps}  seed={args.seed}")

    # Loss weights from mode.
    if args.mode == "ce":
        kl_w, ce_w = 0.0, 1.0
    else:
        # alpha is the CE weight; KL weight is (1 - alpha) so they sum to 1.
        ce_w = float(args.alpha)
        kl_w = 1.0 - ce_w
    print(f"  loss weights: kl={kl_w}, ce={ce_w}")

    # 1) Load shards.
    token_ids, top_ids, top_lps = load_shards(args.shards)
    n_chunks_total, T = token_ids.shape
    K_avail = top_ids.shape[2]
    K = min(args.top_k, K_avail)
    if K < args.top_k:
        print(f"  warning: requested top_k={args.top_k} but shard only has K={K_avail}; "
              f"using K={K}")
    if K < K_avail:
        top_ids = top_ids[:, :, :K]
        top_lps = top_lps[:, :, :K]

    # Vocab size from manifest (or observed max).
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
    print(f"  vocab_size={vocab_size}  T={T}  K={K}")
    print(f"  teacher: {teacher_model}")
    print(f"  dataset: {dataset_name}")

    # Train/val split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_chunks_total)
    val_idx = perm[: args.val_chunks]
    train_idx = perm[args.val_chunks:]
    print(f"  train chunks: {len(train_idx)}, val chunks: {len(val_idx)}")

    # 2) Build student. Plain DN, no feedback.
    print(f"\nBuilding student: vocab={vocab_size} d_model={args.d_model} "
          f"n_heads={args.n_heads} d_head={args.d_head} n_layers={args.n_layers} "
          f"tie_emb={args.tie_embeddings}")
    model = TinyLM(
        vocab_size=vocab_size, d_model=args.d_model,
        n_heads=args.n_heads, d_head=args.d_head, n_layers=args.n_layers,
        attention_cls=DeltaNetAttention,
        feedback_mode="none", feedback_pairs=(),
        tie_embeddings=args.tie_embeddings,
    ).cuda()
    print(f"  params: {model.num_params() / 1e6:.1f} M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.steps), eta_min=args.lr * 0.1)
    print(f"  total train steps: {args.steps}")

    token_ids_t = torch.from_numpy(token_ids).long()
    top_ids_t = torch.from_numpy(top_ids).long()
    top_lps_t = torch.from_numpy(top_lps).float()
    print(f"\nData tensors built (CPU): {token_ids_t.shape}, "
          f"{top_ids_t.shape}, {top_lps_t.shape}")

    def get_batch(idx_array, step_idx):
        i = (step_idx * args.batch) % max(1, len(idx_array) - args.batch + 1)
        sel = idx_array[i : i + args.batch]
        return (token_ids_t[sel].cuda(non_blocking=True),
                top_ids_t[sel].cuda(non_blocking=True),
                top_lps_t[sel].cuda(non_blocking=True))

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
            tp = top_lps_t[sel].cuda()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
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
              f"ce={v_ce:.4f}  kl={v_kl:.4f}  ppl={ppl:.2f}  (n={v_n})")
        model.train()
        return {"step": step, "val_loss": v_loss, "val_ce": v_ce,
                "val_kl": v_kl, "val_ppl": ppl, "n": v_n}

    # 3) Train.
    print(f"\nStarting {args.mode} training for {args.steps} steps ...")
    t0 = time.time()
    model.train()
    losses, ces, kls = [], [], []
    val_history = []

    # Initial val (step 0).
    init_val = run_validation(0)
    if init_val is not None:
        val_history.append(init_val)

    for step in range(args.steps):
        x_ids, tk_ids, tk_lps = get_batch(train_idx, step)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x_ids)
            loss, ce, kl = compute_loss(
                logits, x_ids, tk_ids, tk_lps,
                kl_weight=kl_w, ce_weight=ce_w,
            )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()
        losses.append(loss.item()); ces.append(ce.item()); kls.append(kl.item())

        if step == 0 or (step + 1) % args.log_every == 0:
            n_recent = min(args.log_every, len(losses))
            ll = sum(losses[-n_recent:]) / n_recent
            cc = sum(ces[-n_recent:]) / n_recent
            kk = sum(kls[-n_recent:]) / n_recent
            tps = (step + 1) * args.batch * T / (time.time() - t0)
            print(f"  step {step+1:>5d}/{args.steps}  "
                  f"loss={ll:.4f}  ce={cc:.4f}  kl={kk:.4f}  "
                  f"tok/s={tps:.0f}  lr={sched.get_last_lr()[0]:.2e}")

        if (step + 1) % args.val_every == 0 or step + 1 == args.steps:
            v = run_validation(step + 1)
            if v is not None:
                val_history.append(v)

    # 4) Save metrics.
    final_val = val_history[-1] if val_history else None
    summary = {
        "mode": args.mode,
        "alpha": args.alpha,
        "kl_weight": kl_w,
        "ce_weight": ce_w,
        "top_k": K,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_head": args.d_head,
        "n_layers": args.n_layers,
        "tie_embeddings": args.tie_embeddings,
        "vocab_size": vocab_size,
        "T": T,
        "params_M": model.num_params() / 1e6,
        "shards": str(args.shards),
        "teacher": teacher_model,
        "dataset": dataset_name,
        "wallclock_s": time.time() - t0,
        "final_val": final_val,
        "val_history": val_history,
    }

    print("\n=== summary ===")
    if final_val is not None:
        print(f"  mode={args.mode}  final_val_ce={final_val['val_ce']:.4f}  "
              f"final_val_ppl={final_val['val_ppl']:.2f}  "
              f"final_val_kl={final_val['val_kl']:.4f}")
    print(f"  wallclock: {summary['wallclock_s']:.1f} s")

    if args.save_metrics:
        out = pathlib.Path(args.save_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"  metrics written to {out}")

    if args.save_ckpt:
        pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": summary,
        }, args.save_ckpt)
        print(f"  ckpt saved to {args.save_ckpt}")


if __name__ == "__main__":
    main()
