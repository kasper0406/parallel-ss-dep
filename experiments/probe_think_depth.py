"""Measure the THINK-DEPTH distribution at generation, not just think_frac.

think_frac says "X% of tokens are thinks" but conflates "thinks 1 step often"
with "thinks 8 steps (the cap) rarely". The user's concern is the latter:
unbounded depth per think-event. This probe reports, per emit step, how many
consecutive latent thinks preceded it (`think_steps_used`), so we can see the
actual depth histogram and the fraction of emit steps that SATURATE the
max_think_per_step cap.
"""
import argparse
import json
import sys

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate_with_retrieval_as_input
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/longctx_recall_heldout.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_gen", type=int, default=24)
    ap.add_argument("--max_think_per_step", type=int, default=8)
    ap.add_argument("--emit_threshold", type=float, default=0.5)
    ap.add_argument("--gate_floor", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.to(args.device).eval()
    thinking_id = cfg.get("thinking_token_id", tok.vocab_size)
    additive = bool(cfg.get("retrieval_input_additive", False))
    eos = tok.eos_token_id

    prompts = []
    with open(args.data) as f:
        for line in f:
            prompts.append(json.loads(line)["problem_prompt"])
            if len(prompts) >= args.n:
                break

    all_depths = []           # think_steps_used flattened across all emit steps
    per_task_total_think = []
    per_task_emit = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(args.device)
        out, diag = generate_with_retrieval_as_input(
            model, ids, max_gen=args.max_gen,
            temperature=0.0, eos_token_id=eos,
            thinking_token_id=thinking_id,
            max_think_per_step=args.max_think_per_step,
            emit_threshold=args.emit_threshold,
            gate_floor=args.gate_floor,
            additive=additive,
        )
        all_depths.extend(diag["think_steps_used"])
        per_task_total_think.append(diag["think_total"])
        per_task_emit.append(diag["emit_count"])

    import statistics as st
    n_steps = len(all_depths)
    cap = args.max_think_per_step
    buckets = {"0": 0, "1-2": 0, "3-5": 0, f"6-{cap-1}": 0, f"{cap}(cap)": 0}
    for d in all_depths:
        if d == 0:
            buckets["0"] += 1
        elif d <= 2:
            buckets["1-2"] += 1
        elif d <= 5:
            buckets["3-5"] += 1
        elif d < cap:
            buckets[f"6-{cap-1}"] += 1
        else:
            buckets[f"{cap}(cap)"] += 1

    print(f"=== think-depth probe: {args.ckpt}")
    print(f"  tasks={len(prompts)} emit_steps={n_steps} "
          f"total_thinks={sum(per_task_total_think)}")
    tf = sum(per_task_total_think) / max(1, sum(per_task_total_think) + sum(per_task_emit))
    print(f"  think_frac={tf:.3f}  "
          f"mean_depth_per_emit={st.mean(all_depths):.2f}  "
          f"max_depth={max(all_depths)}  "
          f"median={st.median(all_depths)}")
    print("  depth histogram (per emit step):")
    for k, v in buckets.items():
        pct = 100 * v / max(1, n_steps)
        bar = "#" * int(pct / 2)
        print(f"    {k:>8}: {v:5d}  {pct:5.1f}%  {bar}")
    sat = buckets[f"{cap}(cap)"] / max(1, n_steps)
    print(f"  >>> {100*sat:.1f}% of emit steps SATURATE the {cap}-think cap "
          f"({'BAD: unbounded depth' if sat > 0.1 else 'ok: depth is bounded/shallow'})")


if __name__ == "__main__":
    main()
