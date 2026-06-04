"""Framing-breadth probe (2026-06-03).

Validated so far: a model trained on ONE framing of table-lookup pointer-chase
(prose `f(i)=v`) transfers across OPERANDS (every table new) and across the LOCAL
table SYNTAX under the same framing (dict, +0.37) — but the lift COLLAPSES when the
whole prompt FRAMING changes (def-style `def f(x): return [...][x]`, +0.08) and is
ZERO across a different OPERATION (affine formula). So generality is bounded by the
trained DISTRIBUTION on two axes: framing and operation.

This bank tests whether BREADTH on the framing axis widens the boundary. The
operation is held fixed (arbitrary table-lookup, m=10, genuinely depth-bound) and we
present it in SEVERAL distinct prompt FRAMINGS. Train on a subset of framings, hold
out the rest. If autonomous/forced thinking lifts the HELD-OUT framings after multi-
framing training, framing-invariance is learnable -> breadth works -> scale to the
operation axis. If held-out framings stay flat, framing-invariance is not induced by
breadth and we rethink.

Each framing is a COMPLETE prompt renderer over the same (table t, start s, depth n);
the underlying map and orbit are identical, only the surface framing differs:
  TRAIN framings:  prose  ( f(0)=7 ... Apply f )
                   list   ( def f(x): return [..][x] )
                   ifchain( def f(x): return v0 if x==0 else .. )
  HELD-OUT:        dict   ( def f(x): return {..}[x] )
                   var    ( T=[..]; def f(x): return T[x] )
                   assign ( d = {..}; repeatedly x = d[x] )

m=10 -> single-token answers AND intermediates -> --per_hop compatible (the recipe
that makes the latent thread load-bearing on the real model).

Splits per rung n: `{prefix}_{train,heldout}_n{n}.jsonl`.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_framings.py \
      --out_prefix data/fram --m 10 --rungs 1,2,3,4,5,6,7,8 --n_train 4000 --n_heldout 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

_TAIL = ("\n\nWrite a Python function `solve()` that takes no arguments and returns "
         "the final value of x.\n")


def _fr_prose(t, s, n, m):
    lines = "\n".join(f"f({i}) = {t[i]}" for i in range(m))
    return (f"You are given a function f defined on the integers 0 to {m-1}:\n{lines}"
            f"\n\nStart with x = {s}. Apply f to x a total of {n} times." + _TAIL)


def _fr_list(t, s, n, m):
    body = "[" + ", ".join(str(v) for v in t) + "][x]"
    return (f"You are given a function f on the integers 0 to {m-1}, defined in "
            f"Python:\ndef f(x):\n    return {body}\n\nStart with x = {s}. Apply f to "
            f"x a total of {n} times." + _TAIL)


def _fr_ifchain(t, s, n, m):
    parts = [f"{t[i]} if x == {i} else" for i in range(m - 1)] + [str(t[-1])]
    body = " ".join(parts)
    return (f"You are given a function f on the integers 0 to {m-1}, defined in "
            f"Python:\ndef f(x):\n    return {body}\n\nStart with x = {s}. Apply f to "
            f"x a total of {n} times." + _TAIL)


def _fr_dict(t, s, n, m):
    body = "{" + ", ".join(f"{i}: {v}" for i, v in enumerate(t)) + "}[x]"
    return (f"You are given a function f on the integers 0 to {m-1}, defined in "
            f"Python:\ndef f(x):\n    return {body}\n\nStart with x = {s}. Apply f to "
            f"x a total of {n} times." + _TAIL)


def _fr_var(t, s, n, m):
    arr = "[" + ", ".join(str(v) for v in t) + "]"
    return (f"You are given a function f on the integers 0 to {m-1}, defined in "
            f"Python:\nT = {arr}\ndef f(x):\n    return T[x]\n\nStart with x = {s}. "
            f"Apply f to x a total of {n} times." + _TAIL)


def _fr_assign(t, s, n, m):
    d = "{" + ", ".join(f"{i}: {v}" for i, v in enumerate(t)) + "}"
    return (f"You are given a Python dictionary d mapping integers to integers:\n"
            f"d = {d}\n\nStart with x = {s}. Apply the update x = d[x] a total of {n} "
            f"times. Return the final value of x." + _TAIL)


TRAIN_FRAMINGS = {"prose": _fr_prose, "list": _fr_list, "ifchain": _fr_ifchain}
HELDOUT_FRAMINGS = {"dict": _fr_dict, "var": _fr_var, "assign": _fr_assign}


def _gen_one(rng, framings, names, n_steps, m, idx):
    t = [rng.randrange(m) for _ in range(m)]
    fr = rng.choice(names)
    s = rng.randrange(m)
    x = s
    intermediates = []
    for _ in range(n_steps):
        x = t[x]
        intermediates.append(x)
    return {
        "task_id": f"fram/{fr}/m{m}/n{n_steps}/{idx}",
        "family": fr,
        "prompt": framings[fr](t, s, n_steps, m),
        "tests": f"def check(candidate):\n    assert candidate() == {x}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {x}\n",
        "rung": n_steps,
        "answer": x,
        "intermediates": intermediates,
    }


def _gen_split(framings, n_steps, n, m, seed, banned=None):
    rng = random.Random(seed)
    names = list(framings.keys())
    seen, out, idx, attempts = set(), [], 0, 0
    banned = banned or set()
    while len(out) < n and attempts < n * 80:
        rec = _gen_one(rng, framings, names, n_steps, m, idx)
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
    ap.add_argument("--out_prefix", default="data/fram")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--rungs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    print(f"m={args.m}  TRAIN framings={list(TRAIN_FRAMINGS)}  "
          f"HELD-OUT framings={list(HELDOUT_FRAMINGS)} (NEVER trained)")
    for n in rungs:
        tr = _gen_split(TRAIN_FRAMINGS, n, args.n_train, args.m, seed=args.seed + n)
        he = _gen_split(HELDOUT_FRAMINGS, n, args.n_heldout, args.m,
                        seed=args.seed + 90000 + n)
        _write(tr, f"{args.out_prefix}_train_n{n}.jsonl")
        _write(he, f"{args.out_prefix}_heldout_n{n}.jsonl")
        print(f"  n={n:>2}  train={len(tr):>5}  heldout={len(he):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
