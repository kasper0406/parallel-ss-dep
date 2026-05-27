"""Direct Preference Optimization (DPO) on rollout pairs.

Given a JSONL of GRADED rollouts (output of gen_rejection_data.py
--keep_all), group by underlying problem (task_id prefix), build
preference pairs of (passing, failing) completions, and train via the
DPO loss:

    L = -log sigmoid( beta * (
            log_pi(chosen) - log_pi(rejected)
          - log_pi_ref(chosen) + log_pi_ref(rejected)
        ))

The reference policy is a frozen copy of --load_ckpt — the same
mechanism we use for KL in train_rl_grader. Loss is computed over the
COMPLETION tokens only (the prompt is excluded, so we don't train the
model to fit the problem statement).

Usage:
  PYTHONPATH=. .venv/bin/python experiments/train_dpo.py \\
      --load_ckpt checkpoints/rl_grader_phase_c_v2_step200.pt \\
      --rollouts data/rejection_v2_step200_all.jsonl \\
      --save_ckpt checkpoints/dpo_phase_c_v2.pt \\
      --beta 0.1 --epochs 2 --lr 5e-6
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def base_task_id(task_id: str) -> str:
    """Strip the rollout-index suffix `/j` that gen_rejection_data adds."""
    parts = task_id.rstrip("/").split("/")
    # Format: reject/<orig_task_id>/<j>. The last segment is always the
    # rollout index; everything before it identifies the problem.
    if len(parts) >= 3 and parts[-1].isdigit():
        return "/".join(parts[:-1])
    return task_id


def load_pairs(rollouts_path: str, max_pairs_per_problem: int = 4,
                seed: int = 0) -> list[tuple[str, str, str, float, float]]:
    """Load graded rollouts and build (prompt, chosen, rejected, c_score,
    r_score) tuples. Chosen has score > rejected (i.e. preference). A
    'pass' beats a 'partial' beats an 'exec_error' beats a 'syntax_error'
    via their numeric scores.
    """
    by_prob: dict[str, list[dict]] = collections.defaultdict(list)
    with open(rollouts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_prob[base_task_id(r["task_id"])].append(r)
    rng = random.Random(seed)
    pairs: list[tuple[str, str, str, float, float]] = []
    n_groups_with_pairs = 0
    for tid, rows in by_prob.items():
        # Need at least one high-score and one low-score in the group.
        rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
        if rows_sorted[0]["score"] <= rows_sorted[-1]["score"]:
            continue                                       # all-equal — skip
        # Pair top-scoring vs bottom-scoring rollouts, up to a cap.
        hi = [r for r in rows_sorted
              if r["score"] >= rows_sorted[0]["score"]]
        lo = [r for r in rows_sorted
              if r["score"] <= rows_sorted[-1]["score"]]
        # All hi-rows tie at the top score; same for lo. Pair them.
        rng.shuffle(hi); rng.shuffle(lo)
        k = min(max_pairs_per_problem, len(hi), len(lo))
        for i in range(k):
            ch = hi[i]
            rj = lo[i]
            if ch["score"] <= rj["score"]:
                continue
            pairs.append((ch["problem_prompt"], ch["qwen_completion"],
                          rj["qwen_completion"],
                          float(ch["score"]), float(rj["score"])))
        n_groups_with_pairs += 1
    print(f"[dpo] {len(pairs)} pairs from {n_groups_with_pairs} problems "
          f"(of {len(by_prob)} total)")
    return pairs


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def encode_pair(prompt: str, completion: str, tokenizer,
                 max_len: int) -> tuple[list[int], int]:
    """Tokenise prompt+completion, return (ids, prompt_len). The
    completion log-prob is summed over ids[prompt_len:]."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        comp_ids = comp_ids + [int(eos)]
    full = prompt_ids + comp_ids
    if len(full) > max_len:
        # Truncate the completion FROM the right; we'd rather train
        # on the first half of the solution than truncate the problem.
        full = full[:max_len]
    return full, len(prompt_ids)


# ---------------------------------------------------------------------------
# Completion log-prob over a forward
# ---------------------------------------------------------------------------
def completion_logprob(model, input_ids: torch.Tensor, prompt_len: int,
                        thinking_token_id: int | None = None,
                        ) -> torch.Tensor:
    """Sum of log p(token_t | token_<t) for t in [prompt_len, len). One
    forward pass over the full sequence. Skips the thinking token in the
    target space (its logit is masked to -inf before the log-softmax) so
    DPO can't game it.

    Returns a 0-dim tensor (the log-prob sum). Gradient flows."""
    # input_ids: (T,) — we treat as a single-row batch.
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids.unsqueeze(0)).float()    # (1, T, V)
    # logits[:, t-1] predicts input_ids[t].
    targets = input_ids[prompt_len:]                       # (T - prompt_len,)
    pred = logits[0, prompt_len - 1:-1, :].clone()         # (T - prompt_len, V)
    if thinking_token_id is not None:
        pred[:, int(thinking_token_id)] = -float("inf")
    lp = F.log_softmax(pred, dim=-1)                       # (L, V)
    return lp.gather(1, targets.unsqueeze(1)).squeeze(1).sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--load_ckpt", required=True)
    p.add_argument("--rollouts", required=True,
                   help="Output of gen_rejection_data --keep_all (JSONL).")
    p.add_argument("--save_ckpt", required=True)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature. Lower = closer to reference.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_pairs_per_problem", type=int, default=4)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=0,
                   help="Snapshot ckpt every N steps. 0 = only save final. "
                        "Used to capture the early-training sweet spot "
                        "before DPO over-fits (the 2026-05-23 v1 over-train "
                        "lesson: 3908 steps drove winrate to 0.97 / "
                        "log_ratio +200 and regressed HumanEval to 9/164).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    from experiments.eval_bracket_structure import build_model_from_ckpt
    print(f"[dpo] loading policy from {args.load_ckpt}")
    model, cfg = build_model_from_ckpt(args.load_ckpt)
    model = model.to("cuda").train()
    thinking_token_id = cfg.get("thinking_token_id")

    print(f"[dpo] loading reference (frozen copy of policy)")
    ref_model, _ = build_model_from_ckpt(args.load_ckpt)
    ref_model = ref_model.to("cuda").eval()
    for _p in ref_model.parameters():
        _p.requires_grad = False

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    pairs = load_pairs(args.rollouts, args.max_pairs_per_problem, args.seed)
    if not pairs:
        raise SystemExit("[dpo] no preference pairs built — check rollouts file")

    # Pre-tokenise so we don't redo it every epoch.
    encoded = []
    for (prompt, chosen, rejected, c_s, r_s) in pairs:
        ch_ids, ch_pl = encode_pair(prompt, chosen, tok, args.max_len)
        rj_ids, rj_pl = encode_pair(prompt, rejected, tok, args.max_len)
        if len(ch_ids) <= ch_pl or len(rj_ids) <= rj_pl:
            continue                                       # no completion left
        encoded.append((ch_ids, ch_pl, rj_ids, rj_pl, c_s, r_s))
    print(f"[dpo] {len(encoded)} tokenised pairs ready")

    # WD=0.01 is the project default (GEMINI.md, 2026-05-14). DPO
    # runs are short so impact is small, but matches sft_code +
    # train_rl_grader for consistency across the post-training stack.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.95), weight_decay=0.01)
    n_steps = len(encoded) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, n_steps), eta_min=args.lr * 0.1)
    print(f"[dpo] total train steps: {n_steps}  beta={args.beta}")

    rng = random.Random(args.seed)
    losses = []
    accs = []                           # fraction of pairs where chosen wins
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for idx in order:
            ch_ids, ch_pl, rj_ids, rj_pl, c_s, r_s = encoded[idx]
            ch = torch.tensor(ch_ids, dtype=torch.long, device="cuda")
            rj = torch.tensor(rj_ids, dtype=torch.long, device="cuda")
            lp_pi_ch = completion_logprob(model, ch, ch_pl,
                                            thinking_token_id)
            lp_pi_rj = completion_logprob(model, rj, rj_pl,
                                            thinking_token_id)
            with torch.no_grad():
                lp_ref_ch = completion_logprob(ref_model, ch, ch_pl,
                                                 thinking_token_id)
                lp_ref_rj = completion_logprob(ref_model, rj, rj_pl,
                                                 thinking_token_id)
            log_ratio = (lp_pi_ch - lp_pi_rj) - (lp_ref_ch - lp_ref_rj)
            loss = -F.logsigmoid(args.beta * log_ratio)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
            accs.append(1.0 if log_ratio.item() > 0 else 0.0)
            step += 1
            if step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                wnd = min(args.log_every * 4, len(losses))
                print(f"  step {step:>5}/{n_steps}  "
                      f"loss={sum(losses[-wnd:])/wnd:.4f}  "
                      f"chosen_winrate={sum(accs[-wnd:])/wnd:.3f}  "
                      f"log_ratio={log_ratio.item():+.3f}  "
                      f"lr={lr:.2e}")
            if args.save_every > 0 and step % args.save_every == 0:
                snap_path = (args.save_ckpt
                             .replace(".pt", f"_step{step}.pt"))
                pathlib.Path(snap_path).parent.mkdir(
                    parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(),
                            "config": dict(cfg),
                            "step": step}, snap_path)
                print(f"  [saved snapshot → {snap_path}]")
    print(f"\n[dpo] done in {(time.time()-t0)/60:.1f}m. "
          f"Final loss: {losses[-1]:.4f}, winrate: "
          f"{sum(accs[-200:])/min(200,len(accs)):.3f}")
    pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": dict(cfg)}, args.save_ckpt)
    print(f"[dpo] saved → {args.save_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
