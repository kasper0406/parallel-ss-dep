"""Depth-bound arithmetic ladders, +/- only, bounded answers (2026-06-01).

Faithful real-text port of `latent_arith.py::gen_arith_batch` (which uses +/-
ONLY, answers bounded so a single decode position carries the answer). The
existing `synth_arith_ladder_n{n}.jsonl` files use +/-/* and therefore explode
to 10+ digit answers at deep rungs, confounding "reasoning depth" with "emit a
huge number". Here the SOLE difficulty axis is sequential depth n: the model
must propagate a running value through n add/subtract steps. The answer stays in
[-max_abs, max_abs] (1-2 tokens), so end-to-end exact-match accuracy isolates
whether the latent ponder supplied the missing sequential compute.

Output format matches `eval_arith_ladder_thinking.py` (reads `prompt`, `tests`
`== <int>`, `gold_solution`). Train/heldout splits use disjoint seeds so the
heldout chains are never seen in training.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_arith_ladder_pm.py \
      --out_prefix data/arith_pm --rungs 1,2,3,4,5,6,7,8,9,10,11,12 \
      --n_train 4000 --n_heldout 200 --max_abs 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random


def _apply(running: int, op: str, operand: int, mod: int) -> int:
    if op == "+":
        v = running + operand
    elif op == "-":
        v = running - operand
    else:  # "*"
        v = running * operand
    return v % mod if mod else v


def _gen_one(rng: random.Random, n_steps: int, max_abs: int, idx: int,
             ops_pool=("+", "-"), mod: int = 0) -> dict:
    """One arithmetic chain of exactly n_steps.

    Two regimes:
      - mod == 0 (default, +/- pool): plain integer +/-. Bounded by rejection
        to +/- max_abs. ASSOCIATIVE → in principle parallelizable.
      - mod > 0 (mixed +/-/* pool): every step is `(v OP a) mod mod`. With `*`
        in the pool the composition is NON-ASSOCIATIVE → genuinely O(n)
        sequential depth, while the answer stays in [0, mod) (bounded, 1-2
        tokens). This is the clean depth-bound probe for a deep trunk.
    """
    while True:
        seed = rng.randint(0, (mod - 1) if mod else 9)
        running = seed
        ops, operands = [], []
        ok = True
        for _ in range(n_steps):
            op = rng.choice(list(ops_pool))
            operand = rng.randint(1, 9)
            running = _apply(running, op, operand, mod)
            ops.append(op)
            operands.append(operand)
            if not mod and abs(running) > max_abs:
                ok = False
                break
        if ok:
            break
    answer = running
    var = [f"v{i}" for i in range(n_steps + 1)]
    lines = [f"Let {var[0]} = {seed}."]
    for i in range(n_steps):
        if mod:
            lines.append(
                f"Let {var[i+1]} = ({var[i]} {ops[i]} {operands[i]}) mod {mod}.")
        else:
            lines.append(f"Let {var[i+1]} = {var[i]} {ops[i]} {operands[i]}.")
    mod_note = (f" All arithmetic is performed modulo {mod} (so every value "
                f"stays in the range 0 to {mod-1}).") if mod else ""
    prompt = (
        "Solve the following chain of arithmetic operations step by step."
        + mod_note + "\n\n"
        + "\n".join(lines)
        + f"\n\nWrite a Python function `solve()` that takes no arguments and "
        f"returns the final value of {var[-1]}.\n"
    )
    return {
        "task_id": f"arith_pm/n{n_steps}/{idx}",
        "prompt": prompt,
        "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {answer}\n",
        "rung": n_steps,
        "answer": answer,
    }


def _gen_split(n_steps: int, n: int, max_abs: int, seed: int,
               ops_pool=("+", "-"), mod: int = 0) -> list[dict]:
    rng = random.Random(seed)
    # Dedup on the (prompt) so train and heldout don't accidentally share a
    # chain even at small n where the space is limited.
    seen, out, idx, attempts = set(), [], 0, 0
    while len(out) < n and attempts < n * 50:
        rec = _gen_one(rng, n_steps, max_abs, idx, ops_pool=ops_pool, mod=mod)
        attempts += 1
        key = rec["prompt"]
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
        idx += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="data/arith_pm")
    ap.add_argument("--rungs", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--max_abs", type=int, default=60)
    ap.add_argument("--ops", default="+,-",
                    help="comma-separated op pool, e.g. '+,-' or '+,-,*'")
    ap.add_argument("--mod", type=int, default=0,
                    help="if >0, every step is (v OP a) mod MOD; with '*' in the "
                         "op pool this makes the chain genuinely sequential "
                         "(non-associative) with bounded answers in [0,MOD)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ops_pool = tuple(x.strip() for x in args.ops.split(",") if x.strip())
    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for n in rungs:
        # Disjoint seed streams: train uses seed n, heldout uses seed 10000+n.
        tr = _gen_split(n, args.n_train, args.max_abs, seed=args.seed + n,
                        ops_pool=ops_pool, mod=args.mod)
        he = _gen_split(n, args.n_heldout, args.max_abs, seed=args.seed + 10000 + n,
                        ops_pool=ops_pool, mod=args.mod)
        # Guarantee disjoint chains across splits.
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
        print(f"  n={n:>2}  train={len(tr):>5} -> {tp}   heldout={len(he):>4} -> {hp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
