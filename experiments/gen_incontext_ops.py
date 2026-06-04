"""In-context operator ladders (2026-06-03) — the operation-generality probe.

Pointer-chase (`gen_pointer_chase.py`) already generalizes over OPERANDS: a random
table every example forces "read the table from context, apply it" — and it works on
held-out tables. What it does NOT generalize over is the OPERATION: trained on one
operator (table-apply), the latent step learns "operator = constant" and collapses on
any other operator. That is the exact reason latent thinking doesn't transfer.

This bank attacks operation-generality directly. Every example is a single-digit map
  f : {0..9} -> {0..9}
DEFINED IN-CONTEXT as real Python (`def f(x): return <BODY>`), then applied n times
from a start s. The surface skeleton is IDENTICAL across families — only the BODY
changes — so the model cannot use surface cues to route to a memorized sub-iterator;
it must read the actual operator and apply it. Generality is then testable by holding
out whole operator FAMILIES (bodies the model never trained on) and asking whether
autonomous latent thinking still lifts them. This is literally def-then-call, the
same shape as code, so the meta-skill it forces ("read the def, apply with depth") is
the one that should bridge to coding.

Three splits, per rung n (= depth), file `{prefix}_{split}_n{n}.jsonl`:
  - train     : TRAIN_FAMILIES, random params/start.
  - heldoutOp : HELDOUT_FAMILIES (disjoint operator bodies) — the MONEY metric.
                A lift here = the latent step learned to APPLY a context operator,
                not to BE a memorized one.
  - heldoutPar: TRAIN_FAMILIES, fresh params/start (operand-generalization CONTROL).
                Expected high if anything was learned; distinguishes "memorized the
                trained operators" from "learned nothing".

All maps are mod 10 so state stays a single digit (single-token answers AND
intermediates -> compatible with --per_hop and the single-int eval parse). The
none-vs-R `none` column in the trainer's eval is the fairness control: a family the
single forward already solves shows none~R and simply isn't depth-bound; the
generality claim rests on held-out families where none is low and R is high.

Output record matches `latent_arith_real.py::_load_rung` (prompt/answer/intermediates
/rung) and `gen_pointer_chase.py`.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_incontext_ops.py \
      --out_prefix data/icops --rungs 0,1,2,3,4,5,6 --n_train 3000 --n_heldout 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

M = 10  # digit domain {0..9} -> single-token values


# --------------------------------------------------------------------------- #
# Operator families. Each returns (table, body_src) where:
#   table    : the length-10 lookup f[0..9] (ground truth, used to compute orbits)
#   body_src : the Python expression for `def f(x): return <body_src>` shown to the
#              model. The model must parse body_src and apply it — the table is never
#              shown except where the family IS an explicit table (list/dict).
# Families are deliberately NON-COLLAPSING where possible (arbitrary tables, parity
# branches, quadratics) so f^n genuinely needs n sequential steps; affine is included
# as a known-collapsing family so the `none` control can show the difference.
# --------------------------------------------------------------------------- #
def _fam_list(rng):
    t = [rng.randrange(M) for _ in range(M)]
    body = "[" + ", ".join(str(v) for v in t) + "][x]"
    return t, body


def _fam_affine(rng):
    a, b = rng.randint(1, 9), rng.randrange(M)
    t = [(a * x + b) % M for x in range(M)]
    return t, f"({a} * x + {b}) % 10"


def _fam_parity(rng):
    a, b = rng.randrange(M), rng.randrange(M)
    t = [((x + a) % M) if x % 2 == 0 else ((x + b) % M) for x in range(M)]
    return t, f"(x + {a}) % 10 if x % 2 == 0 else (x + {b}) % 10"


def _fam_quad(rng):
    a, b, c = rng.randint(1, 9), rng.randrange(M), rng.randrange(M)
    t = [(a * x * x + b * x + c) % M for x in range(M)]
    return t, f"({a} * x * x + {b} * x + {c}) % 10"


def _fam_dict(rng):
    """Held-out: a random table in DICT syntax (table semantics, novel surface)."""
    t = [rng.randrange(M) for _ in range(M)]
    body = "{" + ", ".join(f"{i}: {v}" for i, v in enumerate(t)) + "}[x]"
    return t, body


def _fam_modindex(rng):
    """Held-out: index a short table by (x % k) — non-affine, non-collapsing, and a
    surface form (short list + x%k) the train families never present."""
    k = rng.choice([3, 4])
    short = [rng.randrange(M) for _ in range(k)]
    t = [short[x % k] for x in range(M)]
    body = "[" + ", ".join(str(v) for v in short) + f"][x % {k}]"
    return t, body


def _fam_threshold(rng):
    """Held-out: a value-threshold branch (a different conditional structure than the
    parity branch the model trained on)."""
    k = rng.randint(3, 6)
    a, b = rng.randint(1, 9), rng.randrange(M)
    t = [((x + a) % M) if x < k else ((b * x) % M) for x in range(M)]
    return t, f"(x + {a}) % 10 if x < {k} else ({b} * x) % 10"


TRAIN_FAMILIES = {
    "list": _fam_list,
    "affine": _fam_affine,
    "parity": _fam_parity,
    "quad": _fam_quad,
}
HELDOUT_FAMILIES = {
    "dict": _fam_dict,
    "modindex": _fam_modindex,
    "threshold": _fam_threshold,
}


def _prompt(body_src: str, s: int, n: int) -> str:
    return (
        f"You are given a function f on digits 0-9, defined in Python:\n"
        f"def f(x):\n    return {body_src}\n\n"
        f"Start with x = {s}. Apply f to x a total of {n} times. Return the final "
        f"value of x.\n\n"
        f"Write a Python function `solve()` that takes no arguments and returns the "
        f"final value of x.\n"
    )


def _gen_one(rng, families, fam_names, n_steps, idx):
    fam = rng.choice(fam_names)
    t, body = families[fam](rng)
    s = rng.randrange(M)
    x = s
    intermediates = []
    for _ in range(n_steps):
        x = t[x]
        intermediates.append(x)
    answer = x
    return {
        "task_id": f"icops/{fam}/n{n_steps}/{idx}",
        "family": fam,
        "prompt": _prompt(body, s, n_steps),
        "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {answer}\n",
        "rung": n_steps,
        "answer": answer,
        "intermediates": intermediates,
    }


def _gen_split(families, n_steps, n, seed, banned_prompts=None):
    rng = random.Random(seed)
    fam_names = list(families.keys())
    seen, out, idx, attempts = set(), [], 0, 0
    banned = banned_prompts or set()
    while len(out) < n and attempts < n * 80:
        rec = _gen_one(rng, families, fam_names, n_steps, idx)
        attempts += 1
        p = rec["prompt"]
        if p in seen or p in banned:
            continue
        seen.add(p)
        out.append(rec)
        idx += 1
    return out


def _write(recs, path):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="data/icops")
    ap.add_argument("--rungs", default="0,1,2,3,4,5,6")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    print(f"train families:   {list(TRAIN_FAMILIES)}")
    print(f"heldout families: {list(HELDOUT_FAMILIES)}  (operators NEVER trained)")
    for n in rungs:
        tr = _gen_split(TRAIN_FAMILIES, n, args.n_train, seed=args.seed + n)
        tr_keys = {r["prompt"] for r in tr}
        # operand-generalization control: train families, fresh params, disjoint prompts
        par = _gen_split(TRAIN_FAMILIES, n, args.n_heldout,
                         seed=args.seed + 50000 + n, banned_prompts=tr_keys)
        # operation-generalization test: disjoint operator families
        op = _gen_split(HELDOUT_FAMILIES, n, args.n_heldout, seed=args.seed + 90000 + n)
        _write(tr, f"{args.out_prefix}_train_n{n}.jsonl")
        _write(par, f"{args.out_prefix}_heldoutPar_n{n}.jsonl")
        _write(op, f"{args.out_prefix}_heldoutOp_n{n}.jsonl")
        print(f"  n={n:>2}  train={len(tr):>5}  heldoutPar={len(par):>4}  "
              f"heldoutOp={len(op):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
