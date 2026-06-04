"""Operator-complexity (compositional) probe (2026-06-03).

At single-digit domain every depth-bound task reduces to "iterate a finite map", so a
bare operation swap collapses into the framing axis (already validated: breadth over
framings -> framing-invariance). The genuinely harder, CODE-RELEVANT axis is operator
COMPLEXITY: operators built by NESTED COMPOSITION of sub-maps,
    f(x) = g1[g2[...gk[x]...]]
which is more computation per step and is exactly the shape of nested function calls
in code. The per-application map f is still a finite map (so f^n stays genuinely
depth-bound), but recovering f(x) from the spec now requires k sequential sub-lookups.

Test: train on k in {1,2} (single + double composition), HOLD OUT k=3 (triple). If
forced/gated thinking lifts the held-out 3-composition, the apply-with-depth skill
extends to a more complex operator spec it never trained on -> the breadth story
reaches operator complexity, the closest synthetic analog to code's nested calls.
If k=3 stays flat while k in {1,2} lift, the skill is pinned to the spec complexity
it saw.

m=10 -> single-token answers AND intermediates -> --per_hop compatible. Framing is
held FIXED (Python list sub-tables) to isolate the composition-depth axis.

Splits per rung n: `{prefix}_{train,heldout}_n{n}.jsonl` (train = k in {1,2},
heldout = k=3).

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_compose.py \
      --out_prefix data/comp --m 10 --rungs 1,2,3,4,5,6,7,8 --n_train 4000 --n_heldout 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

_TAIL = ("\n\nWrite a Python function `solve()` that takes no arguments and returns "
         "the final value of x.\n")


def _compose_table(subs, m):
    # f(x) = subs[0][ subs[1][ ... subs[k-1][x] ... ] ]  (apply LAST sub first)
    f = []
    for x in range(m):
        v = x
        for t in reversed(subs):
            v = t[v]
        f.append(v)
    return f


def _gen_one(rng, k, n_steps, m, idx):
    subs = [[rng.randrange(m) for _ in range(m)] for _ in range(k)]
    f = _compose_table(subs, m)
    s = rng.randrange(m)
    x = s
    intermediates = []
    for _ in range(n_steps):
        x = f[x]
        intermediates.append(x)
    names = [f"g{i+1}" for i in range(k)]
    decl = "\n".join(f"{names[i]} = [" + ", ".join(str(v) for v in subs[i]) + "]"
                     for i in range(k))
    expr = "x"
    for nm in reversed(names):
        expr = f"{nm}[{expr}]"
    prompt = (
        f"You are given functions on the integers 0 to {m-1}, defined in Python:\n"
        f"{decl}\ndef f(x):\n    return {expr}\n\n"
        f"Start with x = {s}. Apply f to x a total of {n_steps} times. Return the "
        f"final value of x." + _TAIL)
    return {
        "task_id": f"comp/k{k}/m{m}/n{n_steps}/{idx}",
        "family": f"k{k}",
        "prompt": prompt,
        "tests": f"def check(candidate):\n    assert candidate() == {x}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {x}\n",
        "rung": n_steps,
        "answer": x,
        "intermediates": intermediates,
    }


def _gen_split(ks, n_steps, n, m, seed, banned=None):
    rng = random.Random(seed)
    seen, out, idx, attempts = set(), [], 0, 0
    banned = banned or set()
    while len(out) < n and attempts < n * 80:
        k = rng.choice(ks)
        rec = _gen_one(rng, k, n_steps, m, idx)
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
    ap.add_argument("--out_prefix", default="data/comp")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--rungs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--train_k", default="1,2")
    ap.add_argument("--heldout_k", default="3")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_ks = [int(x) for x in args.train_k.split(",") if x.strip()]
    heldout_ks = [int(x) for x in args.heldout_k.split(",") if x.strip()]
    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    print(f"m={args.m}  TRAIN compositions k={train_ks}  "
          f"HELD-OUT k={heldout_ks} (NEVER trained)")
    for n in rungs:
        tr = _gen_split(train_ks, n, args.n_train, args.m, seed=args.seed + n)
        he = _gen_split(heldout_ks, n, args.n_heldout, args.m, seed=args.seed + 90000 + n)
        _write(tr, f"{args.out_prefix}_train_n{n}.jsonl")
        _write(he, f"{args.out_prefix}_heldout_n{n}.jsonl")
        print(f"  n={n:>2}  train={len(tr):>5}  heldout={len(he):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
