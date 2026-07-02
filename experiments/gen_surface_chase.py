"""Surface-augmentation probe: the SAME pointer-chase operation (x = T[x], iterate
n times) rendered in MULTIPLE SURFACE FORMS, to test whether training across
surfaces induces a surface-INVARIANT latent retrieval that transfers to a HELD-OUT
surface. (2026-06-27 — follows the finding that 0 cross-type transfer is
surface-overfitting, not computational: same op transfers fully across the SAME
notation, fails only across notation.)

Single-token (m=10) values + per-hop `intermediates` so latent_arith_real --per_hop
works. Output: data/surf_<surface>_{train,heldout}_n<k>.jsonl, schema matching
the other latent corpora (prompt, answer, rung, intermediates, tests, ...).
"""
import argparse, json, random, pathlib

SOLVE_TAIL = ("\n\nWrite a Python function `solve()` that takes no arguments and "
              "returns the final value of x.\n")


def _table(f):  # surface renderers of the same map f: i -> f[i]
    m = len(f)
    return {
        "function": ("You are given a function f defined on the integers 0 to "
                     f"{m-1}:\n" + "\n".join(f"f({i}) = {f[i]}" for i in range(m)),
                     "Apply f to x a total of {n} times (x becomes f(x) each time)."),
        "dict": ("You are given a Python dictionary d:\nd = {\n"
                 + "".join(f"    {i}: {f[i]},\n" for i in range(m)) + "}",
                 "Apply the update x = d[x] a total of {n} times."),
        "arrow": ("You are given a mapping on the integers 0 to "
                  f"{m-1}:\n" + "\n".join(f"{i} -> {f[i]}" for i in range(m)),
                  "Starting from x, replace x with the value it maps to, a total "
                  "of {n} times."),
        "list": ("You are given a list a of integers:\na = ["
                 + ", ".join(str(v) for v in f) + "]",
                 "Apply the update x = a[x] a total of {n} times."),
        "prose": ("On the integers 0 to " f"{m-1}, the image of each value is:\n"
                  + "\n".join(f"the image of {i} is {f[i]}." for i in range(m)),
                  "Replace x with its image a total of {n} times."),
    }


def gen_one(rng, surface, n_steps, m, idx):
    f = [rng.randrange(m) for _ in range(m)]
    s = rng.randrange(m)
    x = s
    inter = []
    for _ in range(n_steps):
        x = f[x]
        inter.append(x)
    table_str, instr = _table(f)[surface]
    prompt = (f"{table_str}\n\nStart with x = {s}. "
              + instr.format(n=n_steps) + " Return the final value of x." + SOLVE_TAIL)
    return {"task_id": f"surf/{surface}/n{n_steps}/{idx}", "prompt": prompt,
            "tests": f"def check(candidate):\n    assert candidate() == {x}\n",
            "entry_point": "solve", "prompt_is_code": False,
            "gold_solution": f"def solve():\n    return {x}\n",
            "rung": n_steps, "answer": x, "intermediates": inter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfaces", default="function,dict,arrow,list,prose")
    ap.add_argument("--rungs", default="2,3,4,5,6")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", default="data/surf")
    a = ap.parse_args()
    rungs = [int(x) for x in a.rungs.split(",")]
    for surface in a.surfaces.split(","):
        for n in rungs:
            for split, cnt, sd in [("train", a.n_train, a.seed),
                                   ("heldout", a.n_heldout, a.seed + 99991)]:
                rng = random.Random(hash((surface, n, split, sd)) & 0xffffffff)
                recs = [gen_one(rng, surface, n, a.m, i) for i in range(cnt)]
                p = f"{a.out_prefix}_{surface}_{split}_n{n}.jsonl"
                with open(p, "w") as fh:
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
            print(f"  {surface} n={n}: train={a.n_train} heldout={a.n_heldout}", flush=True)


if __name__ == "__main__":
    main()
