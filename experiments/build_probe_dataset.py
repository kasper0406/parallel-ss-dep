"""
Build a tiny held-out HumanEval probe (first 50 problems).

Idempotent: re-runs produce byte-identical JSONL. Used by the in-process
pretrain probe in `probe_humaneval.py`.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/build_probe_dataset.py \
        [--out data/probe_humaneval_50.jsonl] [--n 50]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def build(out_path: str, n: int = 50) -> int:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    out_path_p = pathlib.Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(out_path_p, "w") as f:
        for i, problem in enumerate(ds):
            if i >= n:
                break
            row = {
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "test": problem["test"],
                "entry_point": problem["entry_point"],
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1
    print(f"Wrote {n_written} probe rows to {out_path_p}")
    return n_written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/probe_humaneval_50.jsonl")
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()
    build(args.out, args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
