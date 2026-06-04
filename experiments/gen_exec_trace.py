"""Code-execution-trace task (2026-06-03) — decomposition applied to CODE.

The validated boundary: latent thinking extends DEPTH via a chain of ~1-hop steps,
generalizes over presentation, but cannot do heavy per-step computation. The escape
(and the way to make it help code): pre-decompose reasoning into a CHAIN OF CHEAP
SINGLE-HOP STEPS and let the gate allocate the chain length.

This is the code-native instance: a real Python snippet that threads a single value
x through n statements, each ONE cheap op (single hop):
    x = L0[x]            # arbitrary-table index  (depth-bound, non-collapsing)
    x = (x + c) % 10     # add
    x = (x * c) % 10     # mul
    x = (x - c) % 10     # sub
Tables L0/L1 are declared in-context. Answer = final x; per-step intermediate = x
after each line -> latent step j mirrors executing line j. Op TYPE varies per line,
so this is also the clean test of the one unresolved axis: can the thread learn a
GENERAL "apply this line's single-hop op" primitive (breadth over op types), as
opposed to the ONE fixed op it learns from a single-family corpus? Arbitrary-table
index ops keep the composite genuinely depth-bound (a single forward can't trace n
data-dependent lines); arithmetic-only would collapse.

m=10 -> single-token answers AND intermediates -> --per_hop compatible (per-hop is
exactly the decomposition: supervise each cheap step).

Splits per rung n (= #statements): `{prefix}_{train,heldout}_n{n}.jsonl`.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_exec_trace.py \
      --out_prefix data/trace --rungs 1,2,3,4,5,6,7,8 --n_train 4000 --n_heldout 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

M = 10


def _gen_one(rng, n_steps, idx, n_tables=2):
    tables = {f"L{i}": [rng.randrange(M) for _ in range(M)] for i in range(n_tables)}
    decl = "\n".join(f"{name} = [" + ", ".join(str(v) for v in tbl) + "]"
                     for name, tbl in tables.items())
    s = rng.randrange(M)
    x = s
    lines = [f"x = {s}"]
    intermediates = []
    n_index = 0
    for _ in range(n_steps):
        op = rng.choice(["index", "index", "add", "mul", "sub"])  # bias to index (depth)
        if op == "index":
            nm = rng.choice(list(tables))
            x = tables[nm][x]
            lines.append(f"x = {nm}[x]")
            n_index += 1
        elif op == "add":
            c = rng.randint(1, 9); x = (x + c) % M; lines.append(f"x = (x + {c}) % {M}")
        elif op == "mul":
            c = rng.randint(2, 9); x = (x * c) % M; lines.append(f"x = (x * {c}) % {M}")
        else:
            c = rng.randint(1, 9); x = (x - c) % M; lines.append(f"x = (x - {c}) % {M}")
        intermediates.append(x)
    body = "\n".join(lines)
    prompt = (
        f"Given the following Python program:\n{decl}\n{body}\n\n"
        f"After the program runs, what is the final value of x?\n\n"
        f"Write a Python function `solve()` that takes no arguments and returns the "
        f"final value of x.\n")
    return {
        "task_id": f"trace/n{n_steps}/{idx}",
        "family": "exec_trace",
        "n_index": n_index,
        "prompt": prompt,
        "tests": f"def check(candidate):\n    assert candidate() == {x}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {x}\n",
        "rung": n_steps,
        "answer": x,
        "intermediates": intermediates,
    }


def _gen_split(n_steps, n, seed):
    rng = random.Random(seed)
    seen, out, idx, attempts = set(), [], 0, 0
    while len(out) < n and attempts < n * 80:
        rec = _gen_one(rng, n_steps, idx)
        attempts += 1
        if rec["prompt"] in seen:
            continue
        seen.add(rec["prompt"])
        out.append(rec)
        idx += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="data/trace")
    ap.add_argument("--rungs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for n in rungs:
        tr = _gen_split(n, args.n_train, seed=args.seed + n)
        tr_keys = {r["prompt"] for r in tr}
        he = [r for r in _gen_split(n, args.n_heldout, seed=args.seed + 90000 + n)
              if r["prompt"] not in tr_keys]
        for split, recs in (("train", tr), ("heldout", he)):
            with open(f"{args.out_prefix}_{split}_n{n}.jsonl", "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        print(f"  n={n:>2}  train={len(tr):>5}  heldout={len(he):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
