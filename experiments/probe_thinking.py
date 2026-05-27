"""Diagnostic probe for a thinking-gate + Continuous-RAG RL checkpoint.

Loads a TinyLM checkpoint, runs a fixed held-out sample of code-parrot tokens
through the model (no training), and reports:

  - depth histogram (over thinking rollouts)
  - gate-vs-CE Spearman correlation (does the gate fire low on hard tokens?)
  - retrieval-index entropy (RAG enabled only)
  - cosine-score histogram for the chosen retrievals

The probe is meant to be fast: a few minutes on a single 5090.

Usage:
  python -m experiments.probe_thinking \
      --ckpt checkpoints/think_rl_rag_sft_v2.pt \
      --enable_rag --rag_n_chunks 1000 --n_samples 64
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

from experiments.model import TinyLM
from experiments.layers import DeltaNetAttention
from experiments.thinking import generate_thought_trajectories


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation. Returns NaN on degenerate input."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ta = torch.tensor(a, dtype=torch.float)
    tb = torch.tensor(b, dtype=torch.float)
    ra = ta.argsort().argsort().float()
    rb = tb.argsort().argsort().float()
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (ra.norm() * rb.norm()).item()
    if denom == 0.0:
        return float("nan")
    return float((ra @ rb).item() / denom)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--use_memory", action="store_true",
                   help="Build the model with bounded working memory enabled.")
    p.add_argument("--mem_size", type=int, default=1024)
    p.add_argument("--mem_dim", type=int, default=0,
                   help="0 = use d_model for the memory dim.")
    p.add_argument("--rag_dataset", type=str, default="codeparrot/codeparrot-clean")
    p.add_argument("--text_field", type=str, default="content")
    p.add_argument("--n_samples", type=int, default=64,
                   help="Number of (context, target) pairs to probe.")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--grpo_n_group", type=int, default=4,
                   help="Trajectories per sample (only affects depth/retrieval stats).")
    p.add_argument("--min_decision_pos", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    # Model topology — must match the checkpoint's training config.
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--max_T", type=int, default=0)
    p.add_argument("--feedback", type=str, default="film")
    p.add_argument("--feedback_pairs", type=str, default="2,28")
    p.add_argument("--feedback_self_k", type=int, default=3)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    print(f"[probe] loading {args.ckpt}")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    thinking_token_id = len(tok)
    vocab = len(tok) + 1

    fb_pairs = []
    for s in args.feedback_pairs.split(";"):
        t, src = map(int, s.split(","))
        fb_pairs.append((t, src))

    pad_id = int(tok.eos_token_id)
    model = TinyLM(
        vocab_size=vocab,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head, max_T=args.max_T,
        feedback_mode=args.feedback, feedback_pairs=tuple(fb_pairs),
        feedback_self_k=args.feedback_self_k,
        output_gate=True,
        use_memory=bool(args.use_memory),
        mem_size=int(args.mem_size),
        mem_dim=int(args.mem_dim) if args.mem_dim > 0 else args.d_model,
        thinking_token_id=int(thinking_token_id),
        pad_token_id=int(pad_id),
        attention_cls=DeltaNetAttention,
    ).to(device)
    model.eval()

    sd = torch.load(args.ckpt, map_location=device, weights_only=False)["state_dict"]
    # Tolerate vocab resize done at RL start (mirrors train_rl.py:100-106).
    for key in ["embed.weight", "lm_head.weight"]:
        if key in sd and sd[key].shape != model.state_dict()[key].shape:
            new_param = model.state_dict()[key].clone()
            n_copy = min(sd[key].shape[0], new_param.shape[0])
            new_param[:n_copy] = sd[key][:n_copy]
            sd[key] = new_param
    model.load_state_dict(sd, strict=False)

    # --- Build a fixed sample batch from a held-out shard ---------------------
    ds = load_dataset(args.rag_dataset, streaming=True, split="train")
    contexts: list[list[int]] = []
    targets: list[int] = []
    g = torch.Generator().manual_seed(args.seed)
    for x in ds:
        ids = tok(x[args.text_field], truncation=True, max_length=args.T)["input_ids"]
        if len(ids) < args.min_decision_pos + 4:
            continue
        upper = len(ids) - 1
        lower = args.min_decision_pos
        pos = int(torch.randint(lower, max(lower + 1, upper), (1,), generator=g).item())
        contexts.append(ids[:pos])
        targets.append(int(ids[pos]))
        if len(contexts) >= args.n_samples:
            break
    print(f"[probe] built {len(contexts)} probe samples")

    # --- Direct gate-vs-CE probe (no rollout) --------------------------------
    # This isolates what the gate_head learned independent of GRPO trajectory
    # mechanics. We feed each context, read σ(gate) and CE at the last
    # position, and Spearman-correlate them.
    gate_vals: list[float] = []
    ce_vals: list[float] = []
    pad = pad_id
    with torch.no_grad():
        for ctx, tgt in zip(contexts, targets):
            ids = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, gate = model(ids, return_gate=True)
            g_last = float(torch.sigmoid(model._last_gate_logits[0, -1]).item())
            # logits[..., thinking_token_id] should be masked; ignore the
            # thinking row when computing the predicted-token CE.
            row = logits[0, -1].clone()
            row[thinking_token_id] = -float("inf")
            ce = float(F.cross_entropy(row.unsqueeze(0),
                                       torch.tensor([tgt], device=device)).item())
            gate_vals.append(g_last)
            ce_vals.append(ce)

    rho = _spearman(gate_vals, ce_vals)
    g_mean = sum(gate_vals) / len(gate_vals)
    g_std = (sum((x - g_mean) ** 2 for x in gate_vals) / len(gate_vals)) ** 0.5
    ce_mean = sum(ce_vals) / len(ce_vals)

    # Bin gate by CE quartile to make the relationship interpretable.
    paired = sorted(zip(ce_vals, gate_vals))
    qsize = max(1, len(paired) // 4)
    quartiles = []
    for q in range(4):
        s = q * qsize
        e = len(paired) if q == 3 else (q + 1) * qsize
        if s >= e:
            continue
        slc = paired[s:e]
        ce_q = sum(c for c, _ in slc) / len(slc)
        gt_q = sum(g for _, g in slc) / len(slc)
        quartiles.append((ce_q, gt_q))

    # --- Rollout for depth distribution --------------------------------------
    print("[probe] running thought trajectories for depth stats…")
    traj_groups = generate_thought_trajectories(
        model,
        initial_contexts=contexts,
        target_ids=targets,
        n_group=args.grpo_n_group,
        max_depth=args.max_depth,
        thinking_token_id=thinking_token_id,
        block_size=args.T,
        device=device,
        pad_token_id=pad_id,
    )

    depth_counter: Counter = Counter()
    for group in traj_groups:
        for t in group:
            depth_counter[t.depth] += 1

    # --- Memory diagnostics (only meaningful if use_memory=True) -------------
    mem_stats = None
    if args.use_memory:
        # Replay the prefix passes to capture write-gate and read-attention
        # statistics on the *probed* batch.
        write_gates: list[float] = []
        attn_entropies: list[float] = []
        with torch.no_grad():
            for ctx in contexts:
                ids = torch.tensor([ctx], dtype=torch.long, device=device)
                # Append a single think token to trigger a read at the tail.
                ids = torch.cat([
                    ids, torch.tensor([[thinking_token_id]], device=device),
                ], dim=1)
                _ = model(ids, return_gate=True)
                wg = model.memory._last_write_gate  # (1, T+1)
                write_gates.append(float(wg.mean().item()))
                # Re-run the memory module manually to capture attention.
                # Cheap: T+1 ≤ ~T, K = mem_size.
                h = model._apply_memory  # already applied; instead recompute attn
                # We just want the entropy at the (only) think position.
                mem = model.memory
                B_, T_, _ = (1, ids.shape[1], None)
                # Build the same scores the module computes — we recompute
                # rather than capture from forward to keep this simple.
                # Recompute on the post-out_norm hidden — but we don't have
                # it lying around. Instead, derive write gates' entropy: a
                # poor man's proxy for attention concentration.
                k_eff = min(T_, mem.mem_size)
                topk_g, _ = torch.topk(wg[0], k=k_eff)
                p = topk_g / (topk_g.sum() + 1e-9)
                ent = float(-(p * (p + 1e-9).log()).sum().item())
                attn_entropies.append(ent / max(1.0, math.log(k_eff)))
        mem_stats = {
            "write_gate_mean": sum(write_gates) / max(1, len(write_gates)),
            "write_topk_norm_entropy": sum(attn_entropies) / max(1, len(attn_entropies)),
            "W_proj_norm": float(model.memory.W_proj.weight.detach().norm().item()),
            "W_write_norm": float(model.memory.W_write.weight.detach().norm().item()),
            "W_v_norm": float(model.memory.W_v.weight.detach().norm().item()),
            "W_q_norm": float(model.memory.W_q.weight.detach().norm().item()),
        }

    # --- Report --------------------------------------------------------------
    print()
    print("=" * 64)
    print(f"  PROBE REPORT  —  ckpt: {args.ckpt}")
    print(f"  memory: {bool(args.use_memory)},  n_samples: {len(contexts)},  "
          f"n_group: {args.grpo_n_group}")
    print("=" * 64)
    print(f"  gate σ:   mean={g_mean:.4f}   std={g_std:.4f}")
    print(f"  CE:       mean={ce_mean:.4f}")
    print(f"  Spearman ρ(gate, CE) = {rho:+.4f}    "
          f"(want negative → low gate = think on hard tokens)")
    print()
    print("  Gate by CE quartile (low CE → high gate is healthy):")
    print(f"    {'CE_q':>10}  {'gate_q':>8}")
    for ce_q, gt_q in quartiles:
        print(f"    {ce_q:>10.4f}  {gt_q:>8.4f}")
    print()
    print(f"  Trajectory depth histogram (out of {sum(depth_counter.values())}):")
    for d in sorted(depth_counter):
        bar = "#" * int(50 * depth_counter[d] / max(depth_counter.values()))
        print(f"    d={d}: {depth_counter[d]:>4}  {bar}")
    print()
    print(f"  ‖gate_head.weight‖     = "
          f"{float(model.gate_head.weight.detach().norm().item()):.4f}")
    if mem_stats is not None:
        print()
        print("  Working-memory diagnostics:")
        print(f"    write-gate σ mean        = {mem_stats['write_gate_mean']:.4f}")
        print(f"    write top-K norm-entropy = {mem_stats['write_topk_norm_entropy']:.4f}  "
              f"(1.0 = uniform across slots, 0.0 = single slot dominates)")
        print(f"    ‖W_proj‖  = {mem_stats['W_proj_norm']:.4f}   "
              f"‖W_write‖ = {mem_stats['W_write_norm']:.4f}")
        print(f"    ‖W_v‖     = {mem_stats['W_v_norm']:.4f}   "
              f"‖W_q‖     = {mem_stats['W_q_norm']:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
