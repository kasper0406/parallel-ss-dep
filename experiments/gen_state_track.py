"""Agentic long-horizon STATE-TRACKING task (2026-06-27).

The intersection of CODING and AGENTIC reasoning: track ONE entity's mutable
state through a LONG, INTERLEAVED trajectory of updates amid DISTRACTOR entities,
then answer its final value. This is the use case latent thinking should nail:
 - LONG-HORIZON  -> plays to the O(1)-decode recurrent moat (cheap long context).
 - DEPTH-BOUND   -> the queried var's value is a K-deep table-lookup chain
                    (non-foldable; a single forward can't trace K dependent hops).
 - RANDOM-ACCESS SELECTION -> the K relevant updates are scattered among distractor
                    updates, so the latent loop must select the queried var's j-th
                    update at step j (content-addressed, the hard realistic regime).
 - AGENTIC SHAPE -> entities + a trajectory of actions + a state query.

Surface: dict table (label-based, clean retrieval) + Python variable updates.
m=10 -> single-token values + per-hop `intermediates` (queried var's value after
each of its K updates) so latent_arith_real --per_hop works.
Output: data/<prefix>_{train,heldout}_n<K>.jsonl (rung = K = queried var's depth).
"""
import argparse, json, random

VARS = ["a", "b", "c", "d", "e", "g", "h", "k"]
SOLVE_TAIL = ("\n\nWrite a Python function `solve()` that takes no arguments and "
              "returns the final value of {q}.\n")


def gen_one(rng, K, m, n_vars, distractor_density, idx):
    tbl = [rng.randrange(m) for _ in range(m)]
    names = rng.sample(VARS, n_vars)
    q = names[0]                              # queried entity
    distractors = names[1:]
    init = {nm: rng.randrange(m) for nm in names}

    # queried var's K-deep chain + per-hop intermediates
    qv = init[q]
    inter = []
    for _ in range(K):
        qv = tbl[qv]
        inter.append(qv)
    answer = qv

    # build interleaved trajectory: K queried-updates (in order) + D distractor-updates
    D = int(round(distractor_density * K))
    dstate = {nm: init[nm] for nm in distractors}
    q_updates = [("q", None) for _ in range(K)]
    d_updates = []
    for _ in range(D):
        nm = rng.choice(distractors) if distractors else q
        d_updates.append(("d", nm))
    # interleave: keep queried updates in order, scatter distractors around them
    seq = q_updates + d_updates
    # stable-shuffle: assign each item a random key but it's fine if queried reorder
    # among themselves -> we must keep queried in order, so shuffle positions only
    positions = list(range(len(seq)))
    rng.shuffle(positions)
    # place queried updates at the sorted-first K of a random position assignment
    qpos = sorted(rng.sample(range(len(seq)), K))
    order = []
    di = 0
    for i in range(len(seq)):
        if i in qpos:
            order.append(("q", q))
        else:
            nm = rng.choice(distractors) if distractors else q
            order.append(("d", nm))

    lines = []
    for nm in names:
        lines.append(f"{nm} = {init[nm]}")
    for kind, nm in order:
        lines.append(f"{nm} = tbl[{nm}]")

    tbl_repr = "{" + ", ".join(f"{i}: {tbl[i]}" for i in range(m)) + "}"
    prompt = (
        f"You are given a lookup table and a sequence of variable updates.\n"
        f"tbl = {tbl_repr}\n"
        + "\n".join(lines)
        + f"\n\nAfter running all the updates in order, what is the final value of {q}?"
        + SOLVE_TAIL.format(q=q)
    )
    return {"task_id": f"state/n{K}/{idx}", "prompt": prompt,
            "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
            "entry_point": "solve", "prompt_is_code": False,
            "gold_solution": f"def solve():\n    return {answer}\n",
            "rung": K, "answer": answer, "intermediates": inter,
            "horizon": len(lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="2,3,4,5,6")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--n_vars", type=int, default=4)
    ap.add_argument("--distractor_density", type=float, default=3.0,
                    help="distractor updates per queried update (horizon multiplier)")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", default="data/state")
    a = ap.parse_args()
    import statistics
    for K in [int(x) for x in a.rungs.split(",")]:
        for split, cnt, sd in [("train", a.n_train, a.seed),
                               ("heldout", a.n_heldout, a.seed + 99991)]:
            rng = random.Random(hash((K, split, sd)) & 0xffffffff)
            recs = [gen_one(rng, K, a.m, a.n_vars, a.distractor_density, i)
                    for i in range(cnt)]
            with open(f"{a.out_prefix}_{split}_n{K}.jsonl", "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
        hz = statistics.median(r["horizon"] for r in recs)
        print(f"  K={K}: train={a.n_train} heldout={a.n_heldout} median_horizon={hz} lines", flush=True)


if __name__ == "__main__":
    main()
