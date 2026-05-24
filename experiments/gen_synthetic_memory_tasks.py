"""Generate synthetic memory-required training tasks.

The diagnostic (diag_thinking_machinery.py, 2026-05-18) showed:
  - WM reads ARE sharp at think positions (eff-k of 3-11 vs 20-24 emit)
  - PKM routes differently for think vs emit
  - BUT the write gate fires uniformly across positions — buffer holds noise
  - And the SFT data (Qwen completions) has NO task pressure to make the
    write gate discriminate

This generator produces tasks where:
  1. Earlier positions hold VALUES that the model needs later
  2. Structured think tokens are placed at write points (after a value is
    set) and read points (just before the value is needed)
  3. The loss is on EMIT tokens only (think targets masked to -100 by
    sft_code's existing think-burst pipeline)
  4. Distractor text intervenes so straightforward attention recall is
    insufficient — the model must use its bounded WorkingMemory

Output format matches distill_solutions.py JSONL so the existing SFT
loader (sft_code.load_distilled_jsonl) consumes it transparently:
  {task_id, sample_idx, problem_prompt, qwen_completion, extracted_code,
   has_tests, tier, score}

We set extracted_code = the entire completion (no fence stripping needed
since this is plain Python), has_tests=False, tier="synthetic_memory".

Task families:
  - var_binding: "x = 7. ... distractor ... print(x)" → "7"
  - chain_arithmetic: "x = a + b; y = x * c; ... print(y)"
  - list_index_recall: "L = [...]; ... print(L[k])"
  - dict_lookup: "d = {...}; ... print(d['key'])"

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_synthetic_memory_tasks.py \\
      --n_per_family 2500 --out data/synthetic_memory.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import string
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Distractor generators (force long-range dependency)
# ---------------------------------------------------------------------------

# Every distractor line MUST be syntactically standalone (no continuation
# from a previous line). We sample without replacement and the order is
# random, so each line is exec'd as its own statement.
# Distractor lines MUST be fully self-contained: each line is execable
# in isolation, with no references to other distractor lines (since we
# sample a random subset and order). Each defines a fresh _-prefixed
# variable to avoid collisions with the task's own variables.
_DISTRACTOR_LINES = [
    "# Some unrelated computation:",
    "_a1 = 3 * 7 + 1",
    "# Continuing with helper code...",
    "_b1 = [i for i in range(5)]",
    "_b2 = sum([i for i in range(4)])",
    "# More setup:",
    "_c1 = (1 + 1) if True else 0",
    "_c2 = sum(i for i in range(3))",
    "# (continued)",
    "_d1 = 'abc'.upper()",
    "_d2 = max(1, 2, 3)",
    "# Almost done with the irrelevant block.",
    "_e1 = len('python')",
    "_e2 = abs(-7)",
]


def _rand_distractor(rng: random.Random, n_lines: int) -> str:
    """Pick n_lines from the distractor pool to bloat the context."""
    lines = rng.sample(_DISTRACTOR_LINES, k=min(n_lines, len(_DISTRACTOR_LINES)))
    return "\n".join(lines)


def _rand_var(rng: random.Random, used: set[str]) -> str:
    """Random single-char variable name that hasn't been used yet."""
    while True:
        c = rng.choice(string.ascii_lowercase)
        if c not in used:
            used.add(c)
            return c


# ---------------------------------------------------------------------------
# Think-burst helpers — embed structured WM-write / WM-read hints inline.
# These are NOT thinking tokens; the actual <THINK> token insertion happens
# at training time via the sft_code think-burst pipeline. Here we just
# leave a clear textual marker that the SFT pipeline can latch on to —
# or, even simpler, we leave the raw computation traces in place and let
# the WM read-mask infer think positions from the canonical thinking token
# (which the sft_code pipeline already inserts randomly around our output).
# ---------------------------------------------------------------------------

# We render solutions as raw Python WITHOUT explicit think markers — the
# sft_code think-burst injector will scatter them around the value-binding
# and recall positions naturally. The training signal we care about is:
# (a) value is stored early, (b) distractor intervenes, (c) value is
# recalled at the end. The architectural FIX A + the think bursts the
# trainer scatters will fill the WM buffer with the value-binding-time
# hidden states; reads at the recall position will retrieve those.


# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------

def _gen_var_binding(rng: random.Random, n_distractor_lines: int = 4) -> dict:
    """x = <int>; ... distractor ... print(x) → <int>"""
    used: set[str] = set()
    var = _rand_var(rng, used)
    val = rng.randint(1, 999)
    problem = (f"Run the following Python program and report what it prints.\n\n"
               f"```python\n"
               f"{var} = {val}\n"
               f"{_rand_distractor(rng, n_distractor_lines)}\n"
               f"print({var})\n"
               f"```\n")
    answer = str(val)
    solution = (f"The variable {var} is assigned {val} at the top, then "
                f"unrelated code runs, then we print {var}, which is still "
                f"{val}.\n\nAnswer: {val}\n")
    return {"problem": problem, "solution": solution, "answer": answer}


def _gen_chain_arithmetic(rng: random.Random,
                            n_distractor_lines: int = 4) -> dict:
    """a = X; b = Y; c = a + b; ... distractor ... print(c) → X+Y"""
    used: set[str] = set()
    a = _rand_var(rng, used); b = _rand_var(rng, used); c = _rand_var(rng, used)
    va = rng.randint(1, 50); vb = rng.randint(1, 50)
    op = rng.choice(["+", "-", "*"])
    if op == "+": vc = va + vb
    elif op == "-": vc = va - vb
    else: vc = va * vb
    problem = (f"Run the following Python program and report what it prints.\n\n"
               f"```python\n"
               f"{a} = {va}\n"
               f"{b} = {vb}\n"
               f"{c} = {a} {op} {b}\n"
               f"{_rand_distractor(rng, n_distractor_lines)}\n"
               f"print({c})\n"
               f"```\n")
    solution = (f"{a} = {va}, {b} = {vb}, then {c} = {a} {op} {b} = "
                f"{va} {op} {vb} = {vc}. The distractor doesn't change "
                f"{c}, so the print shows {vc}.\n\nAnswer: {vc}\n")
    return {"problem": problem, "solution": solution, "answer": str(vc)}


def _gen_list_index_recall(rng: random.Random,
                             n_distractor_lines: int = 3) -> dict:
    """L = [v0, v1, ...]; ... distractor ... print(L[k]) → vk"""
    used: set[str] = set()
    name = _rand_var(rng, used)
    n = rng.randint(4, 8)
    L = [rng.randint(1, 99) for _ in range(n)]
    k = rng.randint(0, n - 1)
    problem = (f"Run the following Python program and report what it prints.\n\n"
               f"```python\n"
               f"{name} = {L!r}\n"
               f"{_rand_distractor(rng, n_distractor_lines)}\n"
               f"print({name}[{k}])\n"
               f"```\n")
    solution = (f"{name} is a list with {n} elements. Index {k} (0-based) "
                f"picks element {L[k]}. The distractor doesn't change "
                f"{name}, so the print shows {L[k]}.\n\nAnswer: {L[k]}\n")
    return {"problem": problem, "solution": solution, "answer": str(L[k])}


def _gen_dict_lookup(rng: random.Random,
                      n_distractor_lines: int = 3) -> dict:
    """d = {k0: v0, k1: v1, ...}; ... distractor ... print(d[k_chosen])"""
    used: set[str] = set()
    name = _rand_var(rng, used)
    n = rng.randint(3, 6)
    keys = rng.sample(["alpha", "beta", "gamma", "delta", "epsilon",
                        "zeta", "eta", "theta"], k=n)
    vals = [rng.randint(1, 99) for _ in range(n)]
    d_lit = "{" + ", ".join(f"'{k}': {v}" for k, v in zip(keys, vals)) + "}"
    pick = rng.randrange(n)
    problem = (f"Run the following Python program and report what it prints.\n\n"
               f"```python\n"
               f"{name} = {d_lit}\n"
               f"{_rand_distractor(rng, n_distractor_lines)}\n"
               f"print({name}['{keys[pick]}'])\n"
               f"```\n")
    solution = (f"{name} maps '{keys[pick]}' to {vals[pick]}. The distractor "
                f"doesn't change {name}, so the print shows "
                f"{vals[pick]}.\n\nAnswer: {vals[pick]}\n")
    return {"problem": problem, "solution": solution, "answer": str(vals[pick])}


def _gen_multi_step_arithmetic(rng: random.Random,
                                 n_distractor_lines: int = 4) -> dict:
    """4-step compositional: a, b, c, d, then result depends on all four."""
    used: set[str] = set()
    a, b, c, d = (_rand_var(rng, used) for _ in range(4))
    va, vb, vc, vd = (rng.randint(1, 20) for _ in range(4))
    # result = (a + b) * (c - d) — exercises subtree composition
    result = (va + vb) * (vc - vd)
    problem = (f"Run the following Python program and report what it prints.\n\n"
               f"```python\n"
               f"{a} = {va}\n"
               f"{b} = {vb}\n"
               f"{c} = {vc}\n"
               f"{d} = {vd}\n"
               f"{_rand_distractor(rng, n_distractor_lines)}\n"
               f"print(({a} + {b}) * ({c} - {d}))\n"
               f"```\n")
    solution = (f"{a}={va}, {b}={vb}, {c}={vc}, {d}={vd}. "
                f"({a} + {b}) = {va+vb}, ({c} - {d}) = {vc-vd}, "
                f"product = {result}.\n\nAnswer: {result}\n")
    return {"problem": problem, "solution": solution, "answer": str(result)}


_FAMILIES = {
    "var_binding": _gen_var_binding,
    "chain_arithmetic": _gen_chain_arithmetic,
    "list_index_recall": _gen_list_index_recall,
    "dict_lookup": _gen_dict_lookup,
    "multi_step_arithmetic": _gen_multi_step_arithmetic,
}


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _to_jsonl_record(task_id: str, sample_idx: int, ex: dict) -> dict:
    """Match distill_solutions.py JSONL schema so sft_code.load_distilled_jsonl
    consumes it transparently."""
    # The full "qwen_completion" field is the solution prose + answer.
    # Wrap the final answer in a ```python ... ``` block so the
    # extract_code_block path picks it up (consistency with the distill
    # corpus). The "code" is just `print(<answer>)` — a 1-line program
    # whose execution would print the right value.
    completion = ex["solution"]
    extracted = f"print({ex['answer']})"
    return {
        "task_id": task_id,
        "sample_idx": sample_idx,
        "problem_prompt": ex["problem"],
        "qwen_completion": completion,
        "extracted_code": extracted,
        "has_tests": False,
        "tier": "synthetic_memory",
        "score": 1.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_per_family", type=int, default=2500,
                   help="Number of examples per task family. "
                        "Total = n_per_family × n_families.")
    p.add_argument("--out", type=str, required=True,
                   help="Output JSONL path. Overwrites.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n_written = 0
    with open(out_path, "w") as f:
        for family_name, generator in _FAMILIES.items():
            for i in range(args.n_per_family):
                ex = generator(rng)
                rec = _to_jsonl_record(f"synthmem/{family_name}/{i}", 0, ex)
                f.write(json.dumps(rec) + "\n")
                n_written += 1
            print(f"  {family_name:<24} wrote {args.n_per_family} examples")
    print(f"\ntotal: {n_written} examples → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
