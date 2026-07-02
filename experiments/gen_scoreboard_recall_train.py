"""Generate a RECALL-cotrain training stream that EXACTLY matches the scoreboard
eval format (`experiments/scoreboard_longctx_cost.py::prompt_pyout`).

WHY this generator (2026-06-30, recall-cotrain feasibility probe). The lean
linearized base scores 0% gen_acc on the scoreboard at every distance: it
inherited SmolLM2's knowledge via MOHAWK+KD but NOT recurrent-state long-range
recall. We test whether a short recall-heavy continue-train can teach it to USE
its bounded recurrent state for recall WITHOUT a format mismatch.

CRITICAL — train/eval format MUST match (the repo's documented "recall=0 was a
format artifact" trap, project_wm_recall_probe_broken_and_routed_around). The
scoreboard elicits with the raw-Python output-comment cue `print(vQ)  # ` and
extracts the first 4-digit number. The existing prose recall streams
(multibind_recall_pretrain, code_recall_train) use a DIFFERENT prose format AND
RESTATE the value before the query ("Answer: NNNN") -> they train recency-copy,
not long-range recall, and in the wrong format. So this stream:
  - reuses the scoreboard's OWN body builder (flagship_recall_probe_gen._build_body)
    so the binding+distractor distribution is identical to the eval;
  - appends the query in the EXACT `print(vQ)  # VALUE` form;
  - NEVER restates the value before its query line (leak-free / first-occurrence)
    so the supervision is genuine long-range recall across the recurrent state.

Curriculum: bias to SINGLE-binding (the clean "can the recurrent state carry ONE
binding" test, where a from-scratch DeltaNet is ~100% at every distance) plus a
multi-binding minority (matches the harder 6-binding scoreboard). Distances
sampled to fit the T=2048 training window (recall at L=4096/8192 is tested at
eval via length-generalization of the position-agnostic recurrent mechanism).
For multi-binding bodies we query EVERY bound key once (each first-occurrence)
to raise answer-token density -> stronger recall gradient per sequence under
plain full-sequence LM CE (no answer-span masking infra in the pretrain path).

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_scoreboard_recall_train.py \
      --out data/scoreboard_recall_train.jsonl --n_examples 40000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.flagship_recall_probe_gen import _build_body


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_examples", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    # distance buckets (target token lengths) and their sampling weights —
    # biased toward shorter (denser recall gradient + the foundation) but
    # covering up to ~1900 so the in-window recurrence learns long carries.
    p.add_argument("--buckets", default="128,256,384,512,768,1024,1536,1900")
    p.add_argument("--bucket_weights", default="3,3,2,2,2,1,1,1")
    # n_keys curriculum (sampled per example). Bias to single-binding.
    p.add_argument("--nkey_choices", default="1,1,1,2,3,6")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    bweights = [float(x) for x in args.bucket_weights.split(",") if x.strip()]
    assert len(buckets) == len(bweights)
    nkeys = [int(x) for x in args.nkey_choices.split(",") if x.strip()]

    rng = random.Random(args.seed)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    nkey_hist: dict[int, int] = {}
    with open(out_path, "w") as f:
        for e in range(args.n_examples):
            n_keys = rng.choice(nkeys)
            L = rng.choices(buckets, weights=bweights, k=1)[0]
            body, keys, answers, ntok = _build_body(rng, n_keys, L, tok)
            # Query lines in the EXACT scoreboard `print(vQ)  # ` form, then the
            # value (the recall target). For multi-binding query every key once
            # (random order); each value is first-occurrence (never restated
            # before its own query) so every query line is a genuine recall.
            qkeys = keys[:]
            rng.shuffle(qkeys)
            if n_keys == 1:
                qkeys = qkeys[:1]
            query_block = "\n".join(
                f"print({k})  # {answers[k]}" for k in qkeys)
            text = body + "\n" + query_block + "\n"
            rec = {
                "text": text,
                "bucket": L,
                "n_keys": n_keys,
                "task_id": f"sbtrain/{e}",
            }
            f.write(json.dumps(rec) + "\n")
            nkey_hist[n_keys] = nkey_hist.get(n_keys, 0) + 1
            n += 1
    print(f"wrote {n} -> {out_path}  n_keys_hist={dict(sorted(nkey_hist.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
