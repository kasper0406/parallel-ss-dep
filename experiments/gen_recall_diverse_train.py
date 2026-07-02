"""DIVERSE recall training stream — fix the format-overfit found by the
zero-training generalization probe (string values + natural-code distractors did
NOT transfer). Combines varied CUES x VALUE forms x (synthetic + natural)
distractors into one leak-free / first-occurrence training stream.

Leak-free: the bound value appears EXACTLY once (at the binding) and is supervised
at the query (`...cue VALUE`); never restated before the query.

CUES (sampled): pyout `print(x)  # `, assert `assert x == ` (numeric only),
  prose `# the value of x is `, indirect `_a = x\nprint(_a)  # `.
VALUE forms (sampled): int4, bigint(7-digit), string(4-letter), dict-element,
  func-return, list-element.
DISTRACTORS: 50% canned synthetic + 50% NATURAL real Python lines from Magicoder
  (held-out-able vs codeparrot at eval → cross-source generalization test).
Distances 128..1900 (fit T=2048).

NOTE the deliberate held-out gap for the eval: the NOVEL cues `arrow`/`eq` and the
codeparrot natural-distractor source are NOT produced here, so the strict eval
(`gen_recall_variants.py --strict --natural_source codeparrot`) measures TRUE
generalization, not memorization of the trained distribution.

Usage:
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python experiments/gen_recall_diverse_train.py \
      --out data/recall_diverse_train.jsonl --n_examples 40000 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.gen_recall_variants import _SYNTH, build_natural_pool, _ntok


def _build(tok, rng, pools, buckets):
    key = "v0"
    vform = rng.choices(
        ["int4", "bigint", "string", "dict", "func", "list"],
        weights=[3, 2, 3, 2, 2, 2], k=1)[0]
    val_int = rng.randint(1000, 9999)
    if vform == "int4":
        head = f"{key} = {val_int}"; access = key; gold = str(val_int)
    elif vform == "bigint":
        v = rng.randint(1000000, 9999999)
        head = f"{key} = {v}"; access = key; gold = str(v)
    elif vform == "string":
        s = "".join(rng.choice("bcdfghjklmnpqrstvwxyz") for _ in range(4))
        head = f'{key} = "{s}"'; access = key; gold = s
    elif vform == "dict":
        head = f'cfg = {{"slot": {val_int}}}'; access = 'cfg["slot"]'; gold = str(val_int)
    elif vform == "func":
        head = f"def get_{key}():\n    return {val_int}"; access = f"get_{key}()"; gold = str(val_int)
    elif vform == "list":
        head = f"arr = [{val_int}]"; access = "arr[0]"; gold = str(val_int)

    numeric = vform != "string"
    cue_choices = ["pyout", "prose", "indirect"] + (["assert"] if numeric else [])
    cue = rng.choice(cue_choices)
    if cue == "pyout":
        tail = f"print({access})  # "
    elif cue == "prose":
        tail = f"# the value of {access} is "
    elif cue == "indirect":
        tail = f"_a = {access}\nprint(_a)  # "
    elif cue == "assert":
        tail = f"assert {access} == "

    pool = rng.choice(pools)            # synthetic or natural
    L = rng.choice(buckets)
    # fill between head and tail to ~L
    lines = []
    cur = _ntok(tok, head + "\n" + tail + gold)
    while cur < L:
        line = rng.choice(pool)
        lines.append(line)
        cur += _ntok(tok, line + "\n")
    while lines and _ntok(tok, head + "\n" + "\n".join(lines) + "\n" + tail + gold) > L:
        lines.pop()
    body = head + ("\n" + "\n".join(lines) if lines else "")
    text = body + "\n" + tail + gold + "\n"
    return {"text": text, "vform": vform, "cue": cue,
            "distractor": "natural" if pool is not _SYNTH else "synthetic", "bucket": L}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_examples", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--buckets", default="128,256,384,512,768,1024,1536,1900")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    rng = random.Random(args.seed)

    natural = build_natural_pool(5000, args.seed, source="magicoder")
    pools = [_SYNTH, natural]   # 50/50

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hist = {}
    with open(args.out, "w") as f:
        for e in range(args.n_examples):
            rec = _build(tok, rng, pools, buckets)
            rec["task_id"] = f"divrecall/{e}"
            f.write(json.dumps(rec) + "\n")
            kk = (rec["vform"], rec["cue"], rec["distractor"])
            hist[kk] = hist.get(kk, 0) + 1
    print(f"wrote {args.n_examples} -> {args.out}")
    print("combo histogram (sample):", dict(list(sorted(hist.items()))[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
