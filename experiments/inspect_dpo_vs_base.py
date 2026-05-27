"""Side-by-side inspection: same HumanEval problems on two ckpts.

For each problem, runs greedy generation with the same flags and dumps
(task_id, passed, think_rate, gen_text_preview, full_code) per ckpt.
Then prints a comparison highlighting problems where base PASSES and
candidate FAILS — the regression cases we need to understand.

Designed to answer: did DPO change WHAT THE MODEL EMITS, or just HOW it
thinks/formats?
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import (
    generate_with_retrieval_as_input,
    _extract_code_block, _truncate_at_stop, _run_test_in_subprocess,
)


def run_one(ckpt_path: str, problem_indices: list[int]):
    print(f"\n========== loading {ckpt_path}", flush=True)
    model, cfg = build_model_from_ckpt(ckpt_path)
    model = model.to("cuda").eval()
    thinking_token_id = cfg.get("thinking_token_id")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")

    results = []
    for i in problem_indices:
        problem = ds[i]
        raw_prompt = problem["prompt"]
        prompt = "# Complete the following Python function.\n" + raw_prompt
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                  device="cuda").unsqueeze(0)
        with torch.no_grad():
            gen, diag = generate_with_retrieval_as_input(
                model, prompt_t, max_gen=256,
                temperature=0.0, eos_token_id=tok.eos_token_id,
                thinking_token_id=thinking_token_id,
                max_think_per_step=8,
                total_think_budget=512,
                emit_threshold=0.5,
                min_emit_before_eos=30,
                gate_floor=0.0,
                additive=cfg.get("retrieval_input_additive", True),
            )
        gen_only_full = gen[0, len(prompt_ids):].tolist()
        gen_only = [t for t in gen_only_full
                    if t != int(thinking_token_id)] if thinking_token_id is not None else gen_only_full
        gen_text = tok.decode(gen_only, skip_special_tokens=True)
        code = _extract_code_block(gen_text)
        full_code = code if code is not None else gen_text
        passed = _run_test_in_subprocess(
            full_code, problem["test"], problem["entry_point"])
        emit_count = max(1, diag["emit_count"])
        think_total = diag["think_total"]
        results.append({
            "task_id": problem["task_id"],
            "i": i,
            "passed": passed,
            "think_count": think_total,
            "emit_count": emit_count,
            "think_per_emit": think_total / emit_count,
            "gen_text": gen_text,
            "full_code": full_code,
        })
        print(f"  [{i:3}] {problem['task_id']:24s} pass={passed!s:5s} "
              f"thinks={think_total:>4} emits={emit_count:>3}", flush=True)
    del model
    torch.cuda.empty_cache()
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True,
                   help="The reference (e.g. v2_step300).")
    p.add_argument("--cand_ckpt", required=True,
                   help="The candidate to compare (e.g. dpo_v2_step250).")
    p.add_argument("--first_n", type=int, default=40,
                   help="HumanEval indices [0, first_n) — first N problems "
                        "(where most passes happen for our models).")
    p.add_argument("--out", default="runs/inspect_dpo_vs_base.json")
    args = p.parse_args()

    indices = list(range(args.first_n))

    base = run_one(args.base_ckpt, indices)
    cand = run_one(args.cand_ckpt, indices)

    # Build comparison.
    base_by_tid = {r["task_id"]: r for r in base}
    cand_by_tid = {r["task_id"]: r for r in cand}
    base_pass = {tid for tid, r in base_by_tid.items() if r["passed"]}
    cand_pass = {tid for tid, r in cand_by_tid.items() if r["passed"]}
    regressed = base_pass - cand_pass
    gained = cand_pass - base_pass
    print()
    print(f"=========================================================")
    print(f"BASE ({args.base_ckpt}):    {len(base_pass)}/{len(base)} passed")
    print(f"CAND ({args.cand_ckpt}):    {len(cand_pass)}/{len(cand)} passed")
    print(f"  regressed (base passed, cand failed): {len(regressed)}")
    print(f"  gained    (base failed, cand passed): {len(gained)}")
    print(f"=========================================================")

    # Dump full per-problem records.
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "base": base, "cand": cand,
            "regressed_tids": sorted(regressed),
            "gained_tids": sorted(gained),
        }, f, indent=2)
    print(f"\nDetailed per-problem JSON: {args.out}")

    # Per-regression head-to-head with code preview.
    for tid in sorted(regressed):
        b = base_by_tid[tid]
        c = cand_by_tid[tid]
        print()
        print(f"=========================================================")
        print(f"REGRESSED: {tid}")
        print(f"  BASE  thinks={b['think_count']:>3} "
              f"emits={b['emit_count']:>3} "
              f"per_emit={b['think_per_emit']:.2f}")
        print(f"  CAND  thinks={c['think_count']:>3} "
              f"emits={c['emit_count']:>3} "
              f"per_emit={c['think_per_emit']:.2f}")
        print(f"  BASE code (first 400):")
        print("    " + (b["full_code"][:400].replace("\n", "\n    ")))
        print(f"  CAND code (first 400):")
        print("    " + (c["full_code"][:400].replace("\n", "\n    ")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
