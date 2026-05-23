"""Rejection-sampling data generator for self-distillation SFT.

For each problem, roll out N solutions at temperature τ, grade each via
code_grader, keep only the PASSING ones, and emit a JSONL row per pass
in the schema sft_code.load_distilled_jsonl already consumes:

    {task_id, sample_idx, problem_prompt, qwen_completion,
     extracted_code, has_tests, tier, score}

This is the (STaR / RAFT) approach: use the model's own successful
generations as training data for itself. Cheaper and more stable than
on-policy RL, and uses every passing rollout as a supervised example.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_rejection_data.py \\
      --ckpt checkpoints/rl_grader_phase_c_v2_step200.pt \\
      --dataset mbpp_combined \\
      --n_rollouts 16 --temperature 0.9 --max_gen 384 \\
      --out data/rejection_v2_step200.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.train_rl_grader import (
    rollout_group_batched, build_mbpp_prompt,
)
from experiments.code_grader import (
    grade, truncate_at_stop, LOADERS,
)
from experiments.distill_solutions import extract_code_block


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="mbpp_combined",
                   help="A key in code_grader.LOADERS.")
    p.add_argument("--out", required=True)
    p.add_argument("--n_rollouts", type=int, default=16,
                   help="Rollouts per problem (matches RL --grpo_n_group).")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_gen", type=int, default=384)
    p.add_argument("--max_think_per_step", type=int, default=4)
    p.add_argument("--total_think_budget", type=int, default=120)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--max_problems", type=int, default=None,
                   help="Cap problems for a quick smoke; default = all.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep_partial", action="store_true",
                   help="Also keep partial-credit rollouts (default: pass only).")
    p.add_argument("--keep_all", action="store_true",
                   help="Keep every rollout (pass + partial + exec_error + "
                        "syntax_error). Needed for DPO pair-building "
                        "downstream — each row has its tier+score so the "
                        "DPO trainer can pair passes with fails per "
                        "problem.")
    p.add_argument("--grade_workers", type=int, default=16,
                   help="Thread-pool size for parallel grading. Each "
                        "grade() spawns a subprocess that runs the "
                        "candidate code + MBPP tests; subprocess.run "
                        "releases the GIL, so threads scale linearly "
                        "until subprocess-spawn becomes the floor. "
                        "Defaults to n_rollouts (16); use 1 for the "
                        "old sequential behaviour.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"[reject-gen] loading {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.to("cuda").eval()
    thinking_token_id = int(cfg.get("thinking_token_id"))
    # Compile + bf16 — the rollout loop is autoregressive, so compile pays
    # for itself many times over across 10k+ rollouts. Without it the
    # baseline rate was ~0.6 rollouts/s — far too slow for a full sweep.
    from experiments.speed_knobs import apply_speed_knobs
    apply_speed_knobs(model, bf16=True, tf32=True, compile_model=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    if args.dataset not in LOADERS:
        raise SystemExit(
            f"unknown --dataset {args.dataset!r}; choices: {list(LOADERS)}")
    problems = LOADERS[args.dataset]()
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    print(f"[reject-gen] {len(problems)} problems, "
          f"N={args.n_rollouts}/problem, T={args.temperature}, "
          f"max_gen={args.max_gen}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pad_id = int(tok.eos_token_id) if tok.eos_token_id is not None else 0

    n_total_rollouts = 0
    n_passes = 0
    n_partials = 0
    tier_hist: dict[str, int] = {}
    t0 = time.time()

    # Parallel grading: each grade() spawns a Python subprocess to exec
    # the candidate + MBPP tests. The candidate code is the SLOW part
    # (often hits the per-call timeout); subprocess.run releases the
    # GIL, so a thread pool gives near-linear speedup until subprocess-
    # spawn overhead becomes the floor. Sequential grading was 96 % of
    # per-problem wall time at 16 rollouts/problem.
    grade_pool = (_cf.ThreadPoolExecutor(max_workers=args.grade_workers)
                  if args.grade_workers > 1 else None)

    with open(out_path, "w") as f:
        for i, prob in enumerate(problems):
            prompt = build_mbpp_prompt(prob)
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                     device="cuda").unsqueeze(0)
            with torch.no_grad():
                rollouts = rollout_group_batched(
                    model, tok, prompt_t,
                    n_rollouts=args.n_rollouts,
                    thinking_token_id=thinking_token_id,
                    eos_token_id=tok.eos_token_id,
                    max_gen=args.max_gen,
                    max_think_per_step=args.max_think_per_step,
                    total_think_budget=args.total_think_budget,
                    emit_threshold=args.emit_threshold,
                    gate_floor=args.gate_floor,
                    temperature=args.temperature,
                    min_emit_before_eos=args.min_emit_before_eos,
                )
            # First pass: lightweight CPU work per rollout (tokenize +
            # extract). Keep on main thread — HF tokenizers are not
            # reliably thread-safe across versions.
            per_roll: list[tuple[int, str, str]] = []
            for j, r in enumerate(rollouts):
                gen_only = [t for t in r.full_ids[r.prompt_len:]
                            if t != thinking_token_id]
                gen_text = tok.decode(gen_only, skip_special_tokens=True)
                code = extract_code_block(gen_text)
                full_code = (code if code is not None
                             else truncate_at_stop(gen_text))
                per_roll.append((j, gen_text, full_code))
            # Second pass: grade in parallel (or sequentially if the
            # pool is disabled). Order is preserved by iterating the
            # futures in submission order.
            if grade_pool is not None:
                futs = [grade_pool.submit(grade, prob, fc)
                        for (_, _, fc) in per_roll]
                grade_results = [fut.result() for fut in futs]
            else:
                grade_results = [grade(prob, fc) for (_, _, fc) in per_roll]
            # Third pass: account + write (single-threaded, preserves
            # JSONL order).
            for (j, gen_text, full_code), gr in zip(per_roll, grade_results):
                n_total_rollouts += 1
                tier_hist[gr.tier] = tier_hist.get(gr.tier, 0) + 1
                keep = (args.keep_all or
                        gr.tier == "pass" or
                        (args.keep_partial and gr.tier == "partial"))
                if not keep:
                    continue
                if gr.tier == "pass":
                    n_passes += 1
                else:
                    n_partials += 1
                rec = {
                    "task_id": f"reject/{prob.task_id}/{j}",
                    "sample_idx": j,
                    "problem_prompt": prompt,
                    "qwen_completion": gen_text,
                    "extracted_code": full_code,
                    "has_tests": True,
                    "tier": gr.tier,
                    "score": float(gr.score),
                }
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = n_total_rollouts / max(1, elapsed)
                eta = (len(problems) - i - 1) * args.n_rollouts / max(1, rate)
                print(f"  [{i+1:>4}/{len(problems)}] "
                      f"passes={n_passes} partials={n_partials} "
                      f"({n_total_rollouts} rollouts, {rate:.1f}/s, "
                      f"ETA {eta/60:.0f}m)",
                      flush=True)
    if grade_pool is not None:
        grade_pool.shutdown(wait=False)

    print(f"\n[reject-gen] done in {(time.time()-t0)/60:.1f}m")
    print(f"  total rollouts: {n_total_rollouts}")
    print(f"  passes kept:    {n_passes} ({100*n_passes/max(1,n_total_rollouts):.1f}%)")
    if args.keep_partial:
        print(f"  partials kept:  {n_partials}")
    print(f"  tier histogram: {sorted(tier_hist.items())}")
    print(f"  wrote → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
