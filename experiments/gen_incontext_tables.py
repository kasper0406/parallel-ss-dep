"""Syntax-generality probe (2026-06-03) — corrected, calibrated to the regime where
latent thinking has ANY positive value.

The first operator-family probe (`gen_incontext_ops.py`) was confounded: digit-domain
10 + depth<=6 made the tasks forward-EASY (no-think baseline 0.3-0.78), so there was
no depth-gap for thinking to fill and one latent step merely corrupted an already-
competent forward (R=n < none even on TRAINED families). The validated pointer-chase
worked only because it was forward-IMPOSSIBLE (none~0.1, m=20).

This bank keeps the operation genuinely depth-bound by making EVERY map an arbitrary
table over a larger domain m (default 20) — exactly the regime that validated — and
varies ONLY the presentation SYNTAX. The underlying task ("apply this map n times")
is identical across families; what differs is how the map is written:
  TRAIN syntaxes:   list  ( [v0,..][x] ),  ifchain ( v0 if x==0 else .. )
  HELD-OUT syntaxes: dict ( {0:v0,..}[x] ), var ( T=[..]; T[x] )
So a lift on the HELD-OUT syntaxes = the latent step learned to APPLY an arbitrary
in-context map regardless of how it's written (the def-then-call meta-skill), not to
parse one specific syntax. All families are arbitrary tables -> all depth-bound ->
none~0, so thinking is operating in the regime where it can actually help.

Answers/intermediates are in [0,m) (may be multi-token at m>10) -> use the FINAL-
ANSWER recipe (latent_sft_loss / autonomous_halt), NOT --per_hop.

Splits per rung n: `{prefix}_{train,heldoutSyn,heldoutPar}_n{n}.jsonl`.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_incontext_tables.py \
      --out_prefix data/ictab --m 20 --rungs 1,2,3,4,5,6,7,8 \
      --n_train 4000 --n_heldout 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random


def _syn_list(t):
    return "def f(x):\n    return [" + ", ".join(str(v) for v in t) + "][x]"


def _syn_ifchain(t):
    parts = [f"{t[i]} if x == {i} else" for i in range(len(t) - 1)] + [str(t[-1])]
    return "def f(x):\n    return " + " ".join(parts)


def _syn_dict(t):
    return "def f(x):\n    return {" + ", ".join(f"{i}: {v}" for i, v in enumerate(t)) + "}[x]"


def _syn_var(t):
    return "T = [" + ", ".join(str(v) for v in t) + "]\ndef f(x):\n    return T[x]"


TRAIN_SYNTAXES = {"list": _syn_list, "ifchain": _syn_ifchain}
HELDOUT_SYNTAXES = {"dict": _syn_dict, "var": _syn_var}


def _prompt(defsrc: str, s: int, n: int, m: int) -> str:
    return (
        f"You are given a function f on the integers 0 to {m-1}, defined in Python:\n"
        f"{defsrc}\n\n"
        f"Start with x = {s}. Apply f to x a total of {n} times. Return the final "
        f"value of x.\n\n"
        f"Write a Python function `solve()` that takes no arguments and returns the "
        f"final value of x.\n"
    )


def _gen_one(rng, syntaxes, syn_names, n_steps, m, idx):
    t = [rng.randrange(m) for _ in range(m)]
    syn = rng.choice(syn_names)
    defsrc = syntaxes[syn](t)
    s = rng.randrange(m)
    x = s
    intermediates = []
    for _ in range(n_steps):
        x = t[x]
        intermediates.append(x)
    answer = x
    return {
        "task_id": f"ictab/{syn}/m{m}/n{n_steps}/{idx}",
        "family": syn,
        "prompt": _prompt(defsrc, s, n_steps, m),
        "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {answer}\n",
        "rung": n_steps,
        "answer": answer,
        "intermediates": intermediates,
    }


def _gen_split(syntaxes, n_steps, n, m, seed, banned=None):
    rng = random.Random(seed)
    syn_names = list(syntaxes.keys())
    seen, out, idx, attempts = set(), [], 0, 0
    banned = banned or set()
    while len(out) < n and attempts < n * 80:
        rec = _gen_one(rng, syntaxes, syn_names, n_steps, m, idx)
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
    ap.add_argument("--out_prefix", default="data/ictab")
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--rungs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    print(f"m={args.m}  train syntaxes={list(TRAIN_SYNTAXES)}  "
          f"heldout syntaxes={list(HELDOUT_SYNTAXES)} (NEVER trained)")
    for n in rungs:
        tr = _gen_split(TRAIN_SYNTAXES, n, args.n_train, args.m, seed=args.seed + n)
        tr_keys = {r["prompt"] for r in tr}
        par = _gen_split(TRAIN_SYNTAXES, n, args.n_heldout, args.m,
                         seed=args.seed + 50000 + n, banned=tr_keys)
        syn = _gen_split(HELDOUT_SYNTAXES, n, args.n_heldout, args.m,
                         seed=args.seed + 90000 + n)
        _write(tr, f"{args.out_prefix}_train_n{n}.jsonl")
        _write(par, f"{args.out_prefix}_heldoutPar_n{n}.jsonl")
        _write(syn, f"{args.out_prefix}_heldoutSyn_n{n}.jsonl")
        print(f"  n={n:>2}  train={len(tr):>5}  heldoutPar={len(par):>4}  "
              f"heldoutSyn={len(syn):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
