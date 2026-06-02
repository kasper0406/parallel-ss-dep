"""In-context pointer-chase ladders (2026-06-01) — the genuinely depth-bound probe.

Why not arithmetic: every affine chain (plain +/-, AND modular (v OP a) mod m for
OP in +/-/*) collapses under composition to a single affine map, computable by a
parallel O(log n) reduction. A capable (10-layer) model can therefore do it in ONE
forward, so none-accuracy races R=n upward and there is no depth gap to measure
(empirically confirmed: +/- gave a small noisy lift, mod-100 was too hard at 0%).

Pointer-chase escapes this. Each example presents its OWN random function table
f: [0,m) -> [0,m) in the prompt, a start s, and asks for f^n(s). Composition of an
ARBITRARY table does NOT collapse to a closed form — computing f^n(s) genuinely
requires n sequential lookups, each indexed by the previous result. One forward can
trace only a bounded number of hops (depth-limited); the latent ponder supplies the
missing sequential steps (R = n). This is the task `latent_think.py`/`latent_mem.py`
validated ("needs exactly R=K steps"). The table is per-example, so the model must
LOOK UP from context, not memorize. Answer in [0,m) -> 1-2 tokens.

Output matches `latent_arith_real.py` / `eval_arith_ladder_thinking.py` expectations.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_pointer_chase.py \
      --out_prefix data/ptr --rungs 1,2,3,4,5,6,7,8,9,10 \
      --n_train 4000 --n_heldout 200 --m 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random


def _gen_one(rng: random.Random, n_steps: int, m: int, idx: int) -> dict:
    f = [rng.randrange(m) for _ in range(m)]
    s = rng.randrange(m)
    x = s
    intermediates = []                       # f^1(s), f^2(s), ..., f^n(s)
    for _ in range(n_steps):
        x = f[x]
        intermediates.append(x)
    answer = x
    table_lines = "\n".join(f"f({i}) = {f[i]}" for i in range(m))
    prompt = (
        f"You are given a function f defined on the integers 0 to {m-1}:\n"
        f"{table_lines}\n\n"
        f"Start with x = {s}. Apply f to x a total of {n_steps} times "
        f"(compute f(f(...f(x)...)) with {n_steps} applications).\n\n"
        f"Write a Python function `solve()` that takes no arguments and returns "
        f"the final value of x.\n"
    )
    return {
        "task_id": f"ptr/m{m}/n{n_steps}/{idx}",
        "prompt": prompt,
        "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {answer}\n",
        "rung": n_steps,
        "answer": answer,
        "intermediates": intermediates,      # per-hop targets for latent supervision
    }


def _gen_one_fixed_point(rng: random.Random, d: int, m: int, idx: int) -> dict:
    """IMPLICIT-depth task: x = f(x) repeated until a fixed point f(t)=t, return t.

    The depth d (#hops to the fixed point) is NOT stated in the prompt — the model
    must trace and RECOGNISE the halt condition (current value maps to itself). We
    construct a distinct orbit x_0..x_d with x_d a fixed point so the true depth is
    exactly d (used only for gate supervision, never shown to the model)."""
    perm = list(range(m))
    rng.shuffle(perm)
    orbit = perm[:d + 1]                 # distinct x_0..x_d
    s, t = orbit[0], orbit[-1]
    f = [rng.randrange(m) for _ in range(m)]
    for i in range(d):
        f[orbit[i]] = orbit[i + 1]       # x_i -> x_{i+1} (never a fixed point, distinct)
    f[t] = t                             # the only fixed point the orbit reaches
    answer = t
    intermediates = orbit[1:]            # f^1(s)..f^d(s)
    table_lines = "\n".join(f"f({i}) = {f[i]}" for i in range(m))
    prompt = (
        f"You are given a function f defined on the integers 0 to {m-1}:\n"
        f"{table_lines}\n\n"
        f"Start with x = {s}. Repeatedly replace x with f(x) until you reach a "
        f"value that maps to itself (a value x with f(x) = x). Return that final "
        f"value.\n\n"
        f"Write a Python function `solve()` that takes no arguments and returns "
        f"the final value of x.\n"
    )
    return {
        "task_id": f"ptrfp/m{m}/d{d}/{idx}",
        "prompt": prompt,
        "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {answer}\n",
        "rung": d,
        "answer": answer,
        "intermediates": intermediates,
    }


def _gen_split(n_steps: int, n: int, m: int, seed: int,
               fixed_point: bool = False) -> list[dict]:
    rng = random.Random(seed)
    seen, out, idx, attempts = set(), [], 0, 0
    while len(out) < n and attempts < n * 50:
        rec = (_gen_one_fixed_point(rng, n_steps, m, idx) if fixed_point
               else _gen_one(rng, n_steps, m, idx))
        attempts += 1
        if rec["prompt"] in seen:
            continue
        seen.add(rec["prompt"])
        out.append(rec)
        idx += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="data/ptr")
    ap.add_argument("--rungs", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--m", type=int, default=20, help="table size (answers in [0,m))")
    ap.add_argument("--fixed_point", action="store_true",
                    help="IMPLICIT-depth task: iterate f until a fixed point f(x)=x "
                         "(depth NOT stated; model must recognise the halt condition)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for n in rungs:
        tr = _gen_split(n, args.n_train, args.m, seed=args.seed + n,
                        fixed_point=args.fixed_point)
        he = _gen_split(n, args.n_heldout, args.m, seed=args.seed + 10000 + n,
                        fixed_point=args.fixed_point)
        tr_keys = {r["prompt"] for r in tr}
        he = [r for r in he if r["prompt"] not in tr_keys]
        tp = f"{args.out_prefix}_train_n{n}.jsonl"
        hp = f"{args.out_prefix}_heldout_n{n}.jsonl"
        with open(tp, "w") as f:
            for r in tr:
                f.write(json.dumps(r) + "\n")
        with open(hp, "w") as f:
            for r in he:
                f.write(json.dumps(r) + "\n")
        print(f"  n={n:>2}  train={len(tr):>5} -> {tp}   heldout={len(he):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
