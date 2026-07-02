"""
Runner + report generator for the depth-via-iteration grid.

Spawns one `depth_via_iteration.py` subprocess per (task, variant, model) cell,
up to `--parallel` at a time on a single GPU (cells are tiny), collects the
appended JSONL, and prints a skimmable markdown report with the load-bearing
comparison tables.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
      experiments/depth_via_iteration_run.py --steps 3000 --parallel 4

  # report only from an existing jsonl:
  PYTHONPATH=. .venv/bin/python experiments/depth_via_iteration_run.py \
      --report_only --out /tmp/depthiter_results.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (n_layers, d_model) model shapes.  d_head=32 -> n_heads=d_model//32.
MODELS = [
    (1, 128),   # very shallow narrow
    (2, 128),   # shallow narrow
    (4, 128),   # mid narrow
    (8, 128),   # deep narrow         <- iso-param vs (2,256)
    (2, 256),   # shallow wide (iso-param to (8,128))
    (2, 384),   # shallow wider (width-help probe)
]
# full model grid runs on the two NON-foldable, directly-comparable tasks
GRID_TASKS = ["homo", "hetero_mt"]
VARIANTS = ["nothink", "latent"]

# supplementary cells: the positional-folding caveat + the learnability control
SUPP = [
    ("hetero", "nothink", 2, 128),     # baked-op program, s-last (foldable along positions)
    ("hetero", "nothink", 8, 128),
    ("hetero", "latent", 2, 128),
    ("hetero_fold", "nothink", 1, 128),  # s-first: trivially foldable (ops learnable?)
    ("hetero_fold", "nothink", 2, 128),
]


def cells():
    for task, variant, (L, d) in itertools.product(GRID_TASKS, VARIANTS, MODELS):
        yield task, variant, L, d
    for c in SUPP:
        yield c


def run_grid(args):
    open(args.out, "w").close()   # truncate
    jobs = list(cells())
    print(f"[runner] {len(jobs)} cells, parallel={args.parallel}", flush=True)
    running = []
    t0 = time.time()

    def launch(task, variant, L, d):
        save = (f"{ROOT}/checkpoints/depthiter_{task}_{variant}_L{L}_d{d}.pt"
                if args.save_ckpts else "")
        log = f"/tmp/depthiter_logs/{task}_{variant}_L{L}_d{d}.log"
        cmd = [sys.executable, f"{HERE}/depth_via_iteration.py",
               "--task", task, "--variant", variant,
               "--n_layers", str(L), "--d_model", str(d),
               "--N", str(args.N), "--K", str(args.K),
               "--eval_K_max", str(args.eval_K_max), "--L_ops", str(args.L_ops),
               "--batch", str(args.batch), "--steps", str(args.steps),
               "--n_eval", str(args.n_eval), "--out", args.out]
        if save:
            cmd += ["--save", save]
        env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="1")
        lf = open(log, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        return (p, lf, f"{task}/{variant}/L{L}/d{d}")

    idx = 0
    while idx < len(jobs) or running:
        while len(running) < args.parallel and idx < len(jobs):
            running.append(launch(*jobs[idx]))
            print(f"[runner] launched {running[-1][2]}  ({idx+1}/{len(jobs)})", flush=True)
            idx += 1
        time.sleep(3)
        still = []
        for p, lf, name in running:
            if p.poll() is None:
                still.append((p, lf, name))
            else:
                lf.close()
                ok = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
                print(f"[runner] done {name}: {ok}  ({time.time()-t0:.0f}s)", flush=True)
        running = still
    print(f"[runner] all cells finished in {time.time()-t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def load(out):
    rows = {}
    with open(out) as f:
        for line in f:
            r = json.loads(line)
            rows[(r["task"], r["variant"], r["n_layers"], r["d_model"])] = r
    return rows


def fmt_row(label, accmap, K_list):
    cells = " | ".join(f"{accmap.get(k, accmap.get(str(k), float('nan'))):.2f}"
                       for k in K_list)
    return f"| {label} | {cells} |"


def _get(rows, key, accname):
    r = rows.get(key)
    if r is None or accname not in r:
        return None
    return {int(k): v for k, v in r[accname].items()}


def report(out):
    rows = load(out)
    if not rows:
        print("(no results)")
        return
    any_r = next(iter(rows.values()))
    K_list = any_r["K_list"]
    hdr = "| model | " + " | ".join(f"K={k}" for k in K_list) + " |"
    sep = "|" + "---|" * (len(K_list) + 1)

    L = []
    P = L.append
    P("# Depth-via-iteration: can latent-R simulate trunk depth?\n")
    # params table
    P("## Model sizes (params)\n")
    P("| (L, d) | params |")
    P("|---|---|")
    seen = set()
    for (t, v, nl, d), r in sorted(rows.items()):
        if (nl, d) in seen:
            continue
        seen.add((nl, d))
        P(f"| L{nl} d{d} | {r['params']:,} |")
    P("")

    titles = {
        "homo": "HOMOGENEOUS pointer-chase (f^K, SAME op each hop; non-foldable)",
        "homo_mt": ("HOMOGENEOUS multi-table CONTROL (L_ops recalled tables, SAME "
                    "op repeated K times; matches hetero_mt recall+context burden)"),
        "hetero_mt": ("HETEROGENEOUS multi-table chase (DISTINCT recalled op each "
                      "hop; non-foldable depth-at-position = the true test)"),
    }
    for task in ["homo", "homo_mt", "hetero_mt"]:
        if not any(k[0] == task for k in rows):
            continue
        P(f"## {titles[task]}\n")

        # --- no-think depth scaling: each depth at R=0 ---
        P("### Pure depth (no-think, R=0): accuracy vs K by trunk depth/width\n")
        P(hdr); P(sep)
        for (nl, d) in [(1, 128), (2, 128), (4, 128), (8, 128), (2, 256), (2, 384)]:
            acc = _get(rows, (task, "nothink", nl, d), "acc_R0")
            if acc:
                P(fmt_row(f"L{nl} d{d} R0", acc, K_list))
        P("")

        # --- shallow latent R-sweep ---
        P("### Shallow (L2 d128) + latent: accuracy vs K at each R\n")
        P(hdr); P(sep)
        for R in [0, 2, 4, 8]:
            acc = _get(rows, (task, "latent", 2, 128), f"acc_R{R}")
            if acc:
                P(fmt_row(f"L2 d128 R{R}", acc, K_list))
        diag = _get(rows, (task, "latent", 2, 128), "acc_ReqK")
        if diag:
            P(fmt_row("L2 d128 R=K", diag, K_list))
        P("")

        # --- KEY: shallow+R=K vs deep+R=0 ---
        P("### KEY: shallow+latent(R=K) vs deep+R=0\n")
        P(hdr); P(sep)
        sh = _get(rows, (task, "latent", 2, 128), "acc_ReqK")
        if sh:
            P(fmt_row("L2 d128 latent R=K", sh, K_list))
        dn = _get(rows, (task, "nothink", 8, 128), "acc_R0")
        if dn:
            P(fmt_row("L8 d128 nothink R0", dn, K_list))
        # also deep latent R=K for completeness
        dl = _get(rows, (task, "latent", 8, 128), "acc_ReqK")
        if dl:
            P(fmt_row("L8 d128 latent R=K", dl, K_list))
        P("")

        # --- ISO-PARAM: shallow-wide+R=K vs deep-narrow+R=0 ---
        P("### ISO-PARAM: shallow-wide (L2 d256) vs deep-narrow (L8 d128)\n")
        P(hdr); P(sep)
        sw = _get(rows, (task, "latent", 2, 256), "acc_ReqK")
        if sw:
            P(fmt_row("L2 d256 latent R=K", sw, K_list))
        swn = _get(rows, (task, "nothink", 2, 256), "acc_R0")
        if swn:
            P(fmt_row("L2 d256 nothink R0", swn, K_list))
        dn = _get(rows, (task, "nothink", 8, 128), "acc_R0")
        if dn:
            P(fmt_row("L8 d128 nothink R0", dn, K_list))
        P("")

    # ---- positional-folding caveat + learnability control ----
    P("## CAVEAT: heterogeneous OPS-PROGRAM (baked ops over K sequence positions)\n")
    P("When the K distinct ops are laid out as K *sequence positions*, the linear-RNN\n"
      "recurrence folds the composite ALONG the sequence — no depth-at-position needed.\n")
    P(hdr); P(sep)
    fold1 = _get(rows, ("hetero_fold", "nothink", 1, 128), "acc_R0")
    if fold1:
        P(fmt_row("hetero_fold (s-first) L1 nothink R0", fold1, K_list))
    fold2 = _get(rows, ("hetero_fold", "nothink", 2, 128), "acc_R0")
    if fold2:
        P(fmt_row("hetero_fold (s-first) L2 nothink R0", fold2, K_list))
    hn2 = _get(rows, ("hetero", "nothink", 2, 128), "acc_R0")
    if hn2:
        P(fmt_row("hetero (s-last) L2 nothink R0", hn2, K_list))
    hn8 = _get(rows, ("hetero", "nothink", 8, 128), "acc_R0")
    if hn8:
        P(fmt_row("hetero (s-last) L8 nothink R0", hn8, K_list))
    for R in [0, 2, 4, 8]:
        hl = _get(rows, ("hetero", "latent", 2, 128), f"acc_R{R}")
        if hl:
            P(fmt_row(f"hetero (s-last) L2 latent R{R}", hl, K_list))
    hld = _get(rows, ("hetero", "latent", 2, 128), "acc_ReqK")
    if hld:
        P(fmt_row("hetero (s-last) L2 latent R=K", hld, K_list))
    P("")

    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="/tmp/depthiter_results.jsonl")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--eval_K_max", type=int, default=8)
    ap.add_argument("--L_ops", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--save_ckpts", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()
    os.makedirs("/tmp/depthiter_logs", exist_ok=True)
    if not args.report_only:
        run_grid(args)
    report(args.out)


if __name__ == "__main__":
    main()
