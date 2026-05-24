"""Generate LONG-CONTEXT synthetic recall tasks where DeltaNet's
bounded recurrent state will plausibly lose track of a binding made
early in the sequence — making WorkingMemory mandatory for solving.

The architectural motivation: gen_synthetic_memory_tasks.py produces
short (≤500 token) recall problems that DeltaNet's trunk can solve by
attending across the recurrent state. WM has no opportunity to
demonstrate value. This generator makes the distractor SO long that
the trunk's state HAS to forget; only the explicit memory mechanism
can recover the answer.

Format mirrors distill_solutions output JSONL so sft_code's
load_distilled_jsonl consumes it directly.

Task family: var_binding_long
  - Define `x = <int>` near the top.
  - Insert N lines of unrelated Python distractor (1000-4000 tokens).
  - End with `print(x)` and the expected answer.

Difficulty knobs:
  - --distance_tokens controls the gap between binding and recall.
    1000 ≈ DeltaNet boundary; 2000-4000 forces WM use.
  - --n_distractor_vars controls how many OTHER variables get defined
    in between (potential interference for the recurrent state).

We deliberately produce ONLY this one family for now (var_binding_long).
Diversity-via-other-families can come in v2 once we know it's working.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_longctx_recall_tasks.py \\
      --n_examples 5000 --out data/longctx_recall.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import string
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# Distractor lines — fully self-contained, each statement compiles in
# isolation, no inter-line references. Mixing variable defs (interfere
# with the recurrent state) and function defs / comments (don't).
# Each line is GUARANTEED standalone — no inter-line variable refs.
# Variables defined here all start with _z and use only built-ins or
# literals on the RHS. The pool is sampled with replacement; any order
# must compile + exec.
_DISTRACTOR_POOL = [
    "# Some unrelated bookkeeping...",
    "_z = 3 * 7 + 1",
    "_z = sum([i for i in range(10)])",
    "_z = [j*j for j in range(8)]",
    "_z = {'a': 1, 'b': 2, 'c': 3}",
    "_z = max(2, 9, 5)",
    "_z = (lambda n: n * 2)(7)",
    "_z = 'unused string'.upper()",
    "_z = abs(-42)",
    "_z = sorted([5, 2, 9, 1])",
    "_z = any(x > 0 for x in [1, 2, 3])",
    "# More unrelated computation:",
    "_z = (lambda y: y + 1)(99)",
    "_z = ''.join(['a','b','c'])",
    "_z = bin(255)",
    "_z = hex(4096)",
    "_z = round(3.14159, 2)",
    "_z = divmod(17, 5)",
    "_z = list(range(0, 20, 4))",
    "_z = tuple([10, 20, 30])",
    "_z = set([1, 1, 2, 2, 3])",
    "_z = frozenset({4, 5, 6})",
    "_z = isinstance(42, int)",
    "_z = callable(len)",
    "_z = hash(('a', 1))",
    "_z = 2 ** 8 - 1",
    "_z = bool(True)",
    "_z = chr(65) + chr(66)",
    "_z = ord('A')",
    "_z = type({}).__name__",
    "_z = repr({1: 'one'})",
    "# (continued)",
]


def _rand_var(rng: random.Random, exclude: set[str]) -> str:
    """Single-char var name not in `exclude`."""
    while True:
        c = rng.choice(string.ascii_lowercase)
        if c not in exclude:
            return c


def _build_distractor_block(rng: random.Random, target_lines: int) -> str:
    """Pick lines from the distractor pool, sampling WITH replacement so
    we can reach arbitrary lengths. Returns a joined string."""
    if target_lines == 0:
        return ""
    # Shuffle the pool, take groups of pool-size, repeat until enough.
    out: list[str] = []
    while len(out) < target_lines:
        chunk = list(_DISTRACTOR_POOL)
        rng.shuffle(chunk)
        out.extend(chunk)
    return "\n".join(out[:target_lines])


def _approx_tokens(s: str) -> int:
    """Crude token estimator (chars / 4) for distance-token planning."""
    return max(1, len(s) // 4)


def _gen_var_binding_long(rng: random.Random,
                           target_distance_tokens: int) -> dict:
    """Define x = N near the top, distract for target_distance_tokens
    worth of code, then print(x)."""
    var = _rand_var(rng, set())
    val = rng.randint(1, 9999)
    # Estimate lines needed to hit target distance (each distractor line
    # is ~10-20 chars ≈ 3-5 tokens). Slightly overestimate.
    lines_needed = max(1, target_distance_tokens // 4)
    distractor = _build_distractor_block(rng, lines_needed)
    problem = (
        f"Run the following Python program and report what it prints.\n\n"
        f"```python\n"
        f"{var} = {val}\n"
        f"{distractor}\n"
        f"print({var})\n"
        f"```\n"
    )
    solution = (
        f"At the top of the program, {var} is assigned the value {val}. "
        f"None of the subsequent unrelated lines modify {var}. The final "
        f"`print({var})` therefore outputs {val}.\n\n"
        f"Answer: {val}\n"
    )
    return {
        "problem": problem,
        "solution": solution,
        "answer": str(val),
        "approx_distance_tokens": _approx_tokens(distractor),
    }


def _to_jsonl_record(task_id: str, ex: dict) -> dict:
    """Same schema as distill_solutions / gen_synthetic_memory_tasks.
    The explicit `answer` field is added so a recall eval can score the
    model without re-parsing `print(N)` out of `extracted_code`."""
    extracted = f"print({ex['answer']})"
    return {
        "task_id": task_id,
        "sample_idx": 0,
        "problem_prompt": ex["problem"],
        "qwen_completion": ex["solution"],
        "extracted_code": extracted,
        "answer": ex["answer"],
        "has_tests": False,
        "tier": "longctx_recall",
        "score": 1.0,
        "approx_distance_tokens": ex["approx_distance_tokens"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=5000)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    # Distance buckets — sample uniformly across these so the model
    # sees a curriculum.
    p.add_argument("--distance_tokens_min", type=int, default=512)
    p.add_argument("--distance_tokens_max", type=int, default=3500,
                   help="Max distance. Capped so the full sequence fits "
                        "in our SFT max_len (1024 → 3500 won't fit; "
                        "consider --max_len 4096 for the SFT run).")
    # Held-out eval mode: generate a FIXED number of examples at each of
    # several EXACT distances, so recall accuracy can be reported as a
    # curve vs distance. When set, overrides --n_examples / random
    # distance sampling.
    p.add_argument("--distance_buckets", type=str, default="",
                   help="Comma-separated exact distances (tokens). When "
                        "set, generates --per_bucket examples at each "
                        "distance instead of uniform random sampling. "
                        "Use for the held-out recall eval set.")
    p.add_argument("--per_bucket", type=int, default=100,
                   help="Examples per distance bucket (bucket mode only).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.distance_buckets.strip():
        buckets = [int(b) for b in args.distance_buckets.split(",")
                   if b.strip()]
        with open(out_path, "w") as f:
            i = 0
            for dist in buckets:
                for _ in range(args.per_bucket):
                    ex = _gen_var_binding_long(rng, dist)
                    rec = _to_jsonl_record(f"longctx/d{dist}/{i}", ex)
                    f.write(json.dumps(rec) + "\n")
                    i += 1
        print(f"wrote {i} examples to {out_path} "
              f"({len(buckets)} buckets × {args.per_bucket}: {buckets})")
        return

    with open(out_path, "w") as f:
        for i in range(args.n_examples):
            dist = rng.randint(args.distance_tokens_min,
                                args.distance_tokens_max)
            ex = _gen_var_binding_long(rng, dist)
            rec = _to_jsonl_record(f"longctx/{i}", ex)
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {args.n_examples} examples to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
