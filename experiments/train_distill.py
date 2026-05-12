"""
Train a sparse-(2,28)-FiLM student via KL distillation against
pre-extracted Qwen3.6-35B-A3B-AWQ teacher logprobs.

Reads NPZ shards produced by experiments/extract_teacher_logprobs.py
and trains a TinyLM with the proven sparse-feedback architecture.

Usage:
  CUDA_VISIBLE_DEVICES=1 python experiments/train_distill.py \\
    --shards /home/knielsen/ml/parallel-ss-dep/data/distill_1M \\
    --epochs 3 --d_model 576 --n_layers 30
"""
from __future__ import annotations

import argparse
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


def distill_loss(
    student_logits: torch.Tensor,    # (B, T, V)
    target_token_ids: torch.Tensor,  # (B, T) int — actual next-token at each pos
    teacher_top_ids: torch.Tensor,   # (B, T, K) int — teacher's top-K at each pos
    teacher_top_lps: torch.Tensor,   # (B, T, K) float — teacher logprobs
    kl_weight: float = 1.0,
    ce_weight: float = 0.5,
):
    """KL distillation loss + CE on ground-truth next token.

    Position semantics: at position t, teacher predicted the token AT t (i.e.
    teacher_top_ids[b, t, :] are predictions for x[b, t]). For autoregressive
    student: student_logits[b, t-1, :] predicts x[b, t]. So we shift.
    """
    # Shift so student predictions align with teacher predictions.
    # student_logits[:, :-1] predicts positions [1..T-1].
    # Teacher predictions [1..T-1] are at teacher_top_ids[:, 1:].
    sl = student_logits[:, :-1, :]        # (B, T-1, V)
    tgt = target_token_ids[:, 1:]         # (B, T-1)
    tk_ids = teacher_top_ids[:, 1:, :]    # (B, T-1, K)
    tk_lps = teacher_top_lps[:, 1:, :]    # (B, T-1, K)

    # CE on actual next token.
    B, Tm1, V = sl.shape
    ce = F.cross_entropy(sl.reshape(-1, V), tgt.reshape(-1))

    # Mask out positions where teacher top-K is invalid (all -inf logprobs).
    # We use -inf rather than -1e9 in extraction; here we detect by checking
    # the rank-1 logprob is finite. Float16 saturates very-negative to -inf.
    valid = torch.isfinite(tk_lps[..., 0])  # (B, T-1)
    # Avoid catastrophic edge case where ALL positions are invalid.
    if not valid.any():
        return ce, ce, torch.zeros_like(ce)

    # Gather student logits at top-K positions.
    # sl: (B, T-1, V), tk_ids: (B, T-1, K) → gather over V
    gathered = sl.gather(2, tk_ids.long())  # (B, T-1, K)

    # Local distributions over the top-K (valid since teacher's prob mass is
    # heavily on these tokens; a missed-tail correction is negligible).
    teacher_log_p = F.log_softmax(tk_lps.float(), dim=-1)
    teacher_p = teacher_log_p.exp()
    student_log_p = F.log_softmax(gathered.float(), dim=-1)

    # KL = sum(p_t * (log p_t - log p_s))
    kl_per_pos = (teacher_p * (teacher_log_p - student_log_p)).sum(dim=-1)
    # Apply mask, take mean over valid positions.
    kl = (kl_per_pos * valid.float()).sum() / valid.float().sum().clamp(min=1)

    total = kl_weight * kl + ce_weight * ce
    return total, ce, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=str, required=True,
                   help="Directory of pre-extracted NPZ shards.")
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--feedback_pairs", type=str, default="2,28",
                   help="Sparse FiLM pairs.")
    p.add_argument("--feedback_mode", type=str, default="film")
    p.add_argument("--feedback_self_k", type=int, default=3,
                   help="K=3 self-feeding closes the 1-pass deployment gap; "
                        "default ON for distillation per the validated stack.")
    p.add_argument("--tie_embeddings", action="store_true", default=True)
    p.add_argument("--no_tie_embeddings", action="store_false",
                   dest="tie_embeddings")
    p.add_argument("--output_gate", action="store_true",
                   help="Enable per-position emit/think gate. Optional during "
                        "distillation; useful if a downstream RL pass will use "
                        "the gate.")
    p.add_argument("--use_memory", action="store_true",
                   help="Enable bounded working memory. Reads happen at every "
                        "position during distillation (every token is a target "
                        "→ many read events → favourable regime).")
    p.add_argument("--mem_size", type=int, default=1024)
    p.add_argument("--mem_dim", type=int, default=0,
                   help="0 = use d_model.")
    p.add_argument("--teacher_tokenizer", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ",
                   help="Saved to ckpt config so eval scripts pick the right tokenizer.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--ce_weight", type=float, default=0.5)
    p.add_argument("--amp_dtype", type=str, default="bf16",
                   choices=["fp32", "bf16"],
                   help="bf16 mixed precision (default) halves activation memory.")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_chunks", type=int, default=64,
                   help="How many chunks to hold out for validation.")
    p.add_argument("--save_ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) Load shards.
    token_ids, top_ids, top_lps = load_shards(args.shards)
    n_chunks_total, T = token_ids.shape
    K = top_ids.shape[2]

    # Infer vocab size: take max over BOTH manifest and observed data,
    # since some Qwen-style tokenizers have added/special tokens whose
    # IDs exceed `tokenizer.vocab_size` but still appear in the data.
    manifest_path = pathlib.Path(args.shards) / "manifest.npz"
    manifest_vs = 0
    teacher_model = "(unknown)"
    if manifest_path.exists():
        m = np.load(manifest_path, allow_pickle=True)
        manifest_vs = int(m["vocab_size"])
        teacher_model = str(m["model"])
    obs_max = max(int(token_ids.max()), int(top_ids.max()))
    vocab_size = max(manifest_vs, obs_max + 1)
    # Round up to a multiple of 64 so embedding/lm_head dims are tidy.
    vocab_size = ((vocab_size + 63) // 64) * 64
    print(f"  vocab_size = {vocab_size} (manifest={manifest_vs}, obs_max={obs_max}, "
          f"rounded to multiple of 64), T = {T}, top-K = {K}, teacher = {teacher_model}")

    # Train/val split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_chunks_total)
    val_idx = perm[: args.val_chunks]
    train_idx = perm[args.val_chunks:]
    print(f"  train chunks: {len(train_idx)}, val chunks: {len(val_idx)}")

    # 2) Build student.
    fb_pairs = tuple(
        tuple(int(x) for x in pair.split(","))
        for pair in args.feedback_pairs.split(";") if pair
    )
    # Reserve a single extra vocab slot for the thinking-token id when memory
    # or output_gate is enabled. The slot doesn't appear in the teacher's
    # token_ids stream, so we just slice student logits to teacher_vocab for
    # KL/CE losses. (Same pattern as MQAR / Induction / Dyck wiring.)
    teacher_vocab = vocab_size
    needs_think_slot = args.use_memory or args.output_gate
    if needs_think_slot:
        vocab_size = teacher_vocab + 1
        thinking_token_id = teacher_vocab  # the new last id
    else:
        thinking_token_id = None
    print(f"\nBuilding student: vocab={vocab_size}"
          f"{' (+1 think slot)' if needs_think_slot else ''}  "
          f"d_model={args.d_model} n_layers={args.n_layers} sparse={fb_pairs} "
          f"self_k={args.feedback_self_k} gate={args.output_gate} "
          f"memory={args.use_memory} tie_emb={args.tie_embeddings}")
    model = TinyLM(
        vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        d_head=args.d_head, n_layers=args.n_layers,
        attention_cls=DeltaNetAttention,
        feedback_mode=args.feedback_mode, feedback_pairs=fb_pairs,
        feedback_self_k=int(args.feedback_self_k),
        output_gate=bool(args.output_gate),
        use_memory=bool(args.use_memory),
        mem_size=int(args.mem_size),
        mem_dim=int(args.mem_dim) if args.mem_dim > 0 else args.d_model,
        thinking_token_id=thinking_token_id,
        tie_embeddings=args.tie_embeddings,
    ).cuda()
    print(f"  params: {model.num_params() / 1e6:.1f} M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=0.1)
    n_steps = (len(train_idx) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, n_steps), eta_min=args.lr * 0.1)
    print(f"  total train steps: {n_steps}")

    # Convert data tensors to fp16 / int32 for memory efficiency.
    token_ids_t = torch.from_numpy(token_ids).long()         # int64 for embed
    top_ids_t = torch.from_numpy(top_ids).long()             # int64 for gather
    top_lps_t = torch.from_numpy(top_lps).float()            # fp32 for safety
    print(f"\nData tensors built (CPU): {token_ids_t.shape}, "
          f"{top_ids_t.shape}, {top_lps_t.shape}")

    def get_batch(idx_array, step_idx):
        i = (step_idx * args.batch) % (len(idx_array) - args.batch + 1)
        sel = idx_array[i : i + args.batch]
        return (token_ids_t[sel].cuda(non_blocking=True),
                top_ids_t[sel].cuda(non_blocking=True),
                top_lps_t[sel].cuda(non_blocking=True))

    # 3) Train.
    print("\nStarting distillation training ...")
    t0 = time.time()
    model.train()
    losses, ces, kls = [], [], []
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32
    use_amp = amp_dtype != torch.float32

    for step in range(n_steps):
        x_ids, tk_ids, tk_lps = get_batch(train_idx, step)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x_ids)
            # If we widened the vocab for a thinking slot, slice it off for
            # the KL/CE losses (teacher never emits the thinking-token).
            if needs_think_slot:
                logits = logits[..., :teacher_vocab]
            loss, ce, kl = distill_loss(
                logits, x_ids, tk_ids, tk_lps,
                kl_weight=args.kl_weight, ce_weight=args.ce_weight,
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
            alpha_str = ""
            if hasattr(model, "feedback_alphas"):
                fa = model.feedback_alphas()
                if fa and isinstance(fa[0], tuple):
                    alpha_str = "  α=[" + ",".join(
                        f"{t}<-{s}:{a:+.3f}" for t, s, a in fa) + "]"
            print(f"  step {step+1:>5d}/{n_steps}  loss={ll:.4f}  "
                  f"ce={cc:.4f}  kl={kk:.4f}  tok/s={tps:.0f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}{alpha_str}")

    # 4) Final val.
    print("\nFinal validation ...")
    model.eval()
    with torch.no_grad():
        val_loss, val_ce, val_kl, n_seen = 0.0, 0.0, 0.0, 0
        for vi in range(0, len(val_idx) - args.batch + 1, args.batch):
            sel = val_idx[vi : vi + args.batch]
            x = token_ids_t[sel].cuda()
            ti = top_ids_t[sel].cuda()
            tp = top_lps_t[sel].cuda()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                if needs_think_slot:
                    logits = logits[..., :teacher_vocab]
                loss, ce, kl = distill_loss(
                    logits, x, ti, tp,
                    kl_weight=args.kl_weight, ce_weight=args.ce_weight,
                )
            val_loss += loss.item() * args.batch
            val_ce += ce.item() * args.batch
            val_kl += kl.item() * args.batch
            n_seen += args.batch
        if n_seen > 0:
            print(f"  VAL  loss={val_loss/n_seen:.4f}  "
                  f"ce={val_ce/n_seen:.4f}  kl={val_kl/n_seen:.4f}  "
                  f"ppl={math.exp(val_ce/n_seen):.2f}")

    if args.save_ckpt:
        pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size, "d_model": args.d_model,
                "n_heads": args.n_heads, "d_head": args.d_head,
                "n_layers": args.n_layers, "max_T": T,
                "feedback_mode": args.feedback_mode,
                "feedback_pairs": fb_pairs,
                "feedback_self_k": int(args.feedback_self_k),
                "tie_embeddings": args.tie_embeddings,
                "output_gate": bool(args.output_gate),
                "use_memory": bool(args.use_memory),
                "mem_size": int(args.mem_size),
                "mem_dim": int(args.mem_dim) if args.mem_dim > 0 else args.d_model,
                "thinking_token_id": thinking_token_id,
                "teacher_model": teacher_model,
                "tokenizer": args.teacher_tokenizer,
                "arch": "deltanet",
                "shards": args.shards,
            },
        }, args.save_ckpt)
        print(f"Checkpoint saved to {args.save_ckpt}")


if __name__ == "__main__":
    main()
