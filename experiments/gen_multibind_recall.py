"""Multi-binding recall tasks — the headroom-bearing WM probe.

Single-binding recall (`gen_longctx_recall_tasks.py`) is trivial for a DeltaNet
linear-RNN: it recalls ONE never-overwritten binding at 100% out to 512+ tokens,
so WorkingMemory has no job there (measured 2026-06-13: base = 100% no-think at
every in-context distance). The regime where WM has demonstrated headroom is
MULTI-KEY SATURATION (MQAR): hold N simultaneous key→value bindings, then query
ONE — the fixed-size recurrent state can't carry them all.

This generator dresses that regime in realistic CODE: a program assigns N
variables (interleaved with distractor lines), then `print(v_query)` for a
random one. The model must hold all N bindings until it sees which is asked.
Measured base no-think recall: N=8 → 0.15, N=24 → 0.05 (vs 1.00 single-binding).
That gap is the headroom WM×latent cooperation (Stage A) is trained to close.

Same JSONL schema as the single-binding set (problem_prompt + answer +
extracted_code + approx_distance_tokens) so eval_longctx_recall and
latent_code_cotrain --wm_on consume it unchanged. Adds `n_vars`.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_multibind_recall.py \
      --out data/multibind_recall_train.jsonl --n_examples 8000 --seed 0
  PYTHONPATH=. .venv/bin/python experiments/gen_multibind_recall.py \
      --out data/multibind_recall_heldout.jsonl --n_examples 600 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random

_DISTRACTOR_POOL = [
    "_z = sum([i for i in range(10)])", "_z = abs(-42)", "_z = max(2, 9, 5)",
    "_z = sorted([5, 2, 9, 1])", "_z = bool(True)", "_z = ord('A')",
    "_z = bin(255)", "_z = round(3.14159, 2)", "_z = tuple([10, 20, 30])",
    "_z = ''.join(['a', 'b', 'c'])", "_z = type({}).__name__", "_z = hex(4096)",
    "_z = list(range(0, 20, 4))", "_z = divmod(17, 5)", "_z = chr(65) + chr(66)",
    "_z = 2 ** 8 - 1", "_z = [j * j for j in range(8)]", "_z = set([1, 1, 2, 3])",
]


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _gen_multibind(rng: random.Random, n_vars: int,
                   distractors_per_var: int) -> dict:
    """Assign n_vars distinct variables to 4-digit values (interleaved with
    distractors), then print one. The answer is the queried var's value."""
    vals = {f"v{i}": rng.randint(1000, 9999) for i in range(n_vars)}
    items = list(vals.items())
    rng.shuffle(items)
    lines: list[str] = []
    for name, v in items:
        lines.append(f"{name} = {v}")
        for _ in range(distractors_per_var):
            lines.append(rng.choice(_DISTRACTOR_POOL))
    query = rng.choice(list(vals.keys()))
    body = "\n".join(lines)
    problem = (
        "Run the following Python program and report what it prints.\n\n"
        f"```python\n{body}\nprint({query})\n```\n")
    ans = vals[query]
    solution = (
        f"The program assigns {n_vars} variables. `{query}` is set to {ans}, "
        f"and no later line modifies it, so `print({query})` outputs {ans}.\n\n"
        f"Answer: {ans}\n")
    return {
        "problem_prompt": problem,
        "qwen_completion": solution,
        "extracted_code": f"print({ans})",
        "answer": str(ans),
        "has_tests": False,
        "tier": "multibind_recall",
        "score": 1.0,
        "n_vars": n_vars,
        "approx_distance_tokens": _approx_tokens(body),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_examples", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    # Difficulty curriculum: mix of N (live bindings). All kept in-window
    # (≤~700 tokens) so the binding survives the eval/latent prefix window.
    p.add_argument("--n_vars_choices", type=str, default="4,8,12,16,24",
                   help="comma list of N (live-binding counts) sampled uniformly")
    p.add_argument("--distractors_per_var", type=int, default=3)
    args = p.parse_args()

    n_choices = [int(x) for x in args.n_vars_choices.split(",") if x.strip()]
    rng = random.Random(args.seed)
    n_by = {n: 0 for n in n_choices}
    with open(args.out, "w") as f:
        for e in range(args.n_examples):
            n_vars = rng.choice(n_choices)
            rec = _gen_multibind(rng, n_vars, args.distractors_per_var)
            rec["task_id"] = f"multibind/{e}"
            rec["sample_idx"] = e
            n_by[n_vars] += 1
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {args.n_examples} → {args.out}  N-distribution={n_by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
