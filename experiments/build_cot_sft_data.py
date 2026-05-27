"""Convert CoT-triple JSONL (from gen_cot_distill_data.py) into a
distill-shaped JSONL that sft_code.load_distilled_jsonl can consume,
with metadata flagging the CoT span for downstream think-token
materialisation (THINKING_PLAN.md Phase 4, Option A).

Option A (the implemented choice): the SFT target is
`<CoT prose> + ```python``` block`. A `prepare_for_thinking: true` flag
on each row tells a downstream SFT pass to wrap EVERY token of the CoT
portion in [THINKING] markers (gate-on, loss masked to -100). The model
gets gradient ONLY on the solution code — so the gate learns "thinking
bursts precede answer emission" without us having to invent a per-step
target.

Rationale vs Option B (one think per numbered step):
Option B requires new architectural plumbing (mapping each step to a
fixed-length retrieval-as-input embedding) and an open call on how to
target a step's content. Option A re-uses the EXISTING insert_think_bursts
machinery and the existing load_distilled_jsonl reader — zero new
plumbing — while still teaching the gate the temporal "think then
emit" structure. If Option A teaches the gate to thinking-burst before
code, we then have a cheap escalation path to Option B.

Output schema (consumed by sft_code.load_distilled_jsonl):
  {task_id, problem_prompt, qwen_completion, extracted_code,
   has_tests, tier, score, sample_idx,
   source_tier, source_score,         # provenance (always pass / 1.0)
   prepare_for_thinking: true,        # flag for downstream SFT
   cot_text}                          # the CoT prose alone, for the
                                      # downstream wrapper to identify
                                      # which token-span to mark as
                                      # think positions.

Usage:
  python experiments/build_cot_sft_data.py \\
      --in data/cot_distill.jsonl \\
      --out data/cot_distill_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


# ---------------------------------------------------------------------------
# Row conversion: triple -> distill-shaped SFT row.
# ---------------------------------------------------------------------------

def build_sft_row(triple: dict) -> dict:
    """Convert one (problem, cot, solution) triple to a distill-shaped
    row. The qwen_completion field reconstructs `CoT\n\n```python\nCODE\n```\n`
    so the existing load_distilled_jsonl reader (which targets the
    full completion) trains over the right span. extracted_code is the
    raw solution so a downstream caller can choose code-only targets
    via --distilled_code_only."""
    cot_text = triple["cot_text"].rstrip()
    solution_code = triple["solution_code"].rstrip()
    qwen_completion = f"{cot_text}\n\n```python\n{solution_code}\n```\n"
    return {
        "task_id": triple["task_id"],
        "sample_idx": triple.get("sample_idx", 0),
        "problem_prompt": triple["problem_text"],
        "qwen_completion": qwen_completion,
        "extracted_code": solution_code,
        "has_tests": False,            # we did not re-grade here
        "tier": "pass",                # trusted-teacher convention
        "score": 1.0,
        "source_tier": "cot",          # provenance: came from CoT generator
        "source_score": 1.0,
        "prepare_for_thinking": True,  # downstream: wrap CoT in [THINKING]
        "cot_text": cot_text,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, required=True,
                   help="Input JSONL from gen_cot_distill_data.py")
    p.add_argument("--out", type=str, required=True,
                   help="Output JSONL (sft-shaped, with cot metadata).")
    args = p.parse_args()

    in_path = pathlib.Path(args.in_path)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_dropped = 0
    with in_path.open() as fi, out_path.open("w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                triple = json.loads(line)
            except json.JSONDecodeError:
                n_dropped += 1
                continue
            if not triple.get("cot_text") or not triple.get("solution_code"):
                n_dropped += 1
                continue
            row = build_sft_row(triple)
            fo.write(json.dumps(row) + "\n")
            n_out += 1

    print(f"[build-cot-sft] in={n_in}  out={n_out}  dropped={n_dropped}  "
          f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
