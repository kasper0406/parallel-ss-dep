"""Generate the FLAGSHIP multi-key long-context recall probe set.

Capability probe for the agent-economics thesis: can a bounded-state
DeltaNet + WorkingMemory recall facts bound EARLY in a long sequence and
queried LATE — at sequence lengths BEYOND a fixed transformer's training
window — where a fixed-window transformer goes blind?

TASK (multi-key recall, generation-graded, leak-free by construction):
  - Bind K (default 6) facts near the START as `vN = <4-digit int>`. The
    VALUES are 4-digit so they are digit-tokenized and discriminable; chance
    of guessing one ≈ 1/9000 ≈ 0. Variable names are `v0..v{K-1}` — the exact
    format the v12 pretrain saw via `multibind_recall_pretrain` (matched so
    OURS gets its best shot).
  - Distractor-fill with innocuous Python-ish lines (each compiles in
    isolation, no inter-line refs, no 4-digit constants) until the TOTAL
    program reaches a controlled token length L (measured in OURS's
    tokenizer; the *text* is identical across arms so each model's own
    tokenizer just yields a different token count, reported separately).
  - The query (`print(vQ)`) is appended by the EVAL, one key at a time, so a
    single body record supports K independent queries. The binding is at the
    START and the query at the END → the recall distance ≈ L.

Output JSONL record (one per task): task_id, bucket (target L), n_keys,
body (the program text WITHOUT the trailing print), keys (list), answers
(dict name->str), and the char length. The EVAL forms the per-key prompt:

    Run the following Python program and report what it prints.

    ```python
    {body}
    print({key})
    ```

This generator does NOT add a `print()` line itself — that is the eval's job
(so the same body serves all K queries with identical text across all model
arms). Leak-free: the bound value is NEVER restated before the query, so a
recency-copy cannot answer it; the model must actually recall.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/flagship_recall_probe_gen.py \\
      --out data/flagship_recall.jsonl \\
      --buckets 1024,2048,4096,8192 --per_bucket 150 --n_keys 6 --seed 0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Distractor lines: each compiles standalone, no inter-line variable refs, no
# 4-digit numeric literals (so they cannot be confused with a binding value),
# and never assign a `vN` name (so they cannot clobber a binding).
_DISTRACTOR_POOL = [
    "# bookkeeping step",
    "_z = sum([i for i in range(10)])",
    "_z = abs(-42)",
    "_z = max(2, 9, 5)",
    "_z = sorted([5, 2, 9, 1])",
    "_z = bool(True)",
    "_z = ord('A')",
    "_z = round(3.14159, 2)",
    "_z = tuple([10, 20, 30])",
    "_z = ''.join(['a', 'b', 'c'])",
    "_z = type({}).__name__",
    "_z = list(range(0, 20, 4))",
    "_z = divmod(17, 5)",
    "_z = chr(65) + chr(66)",
    "_z = 2 ** 8 - 1",
    "_z = [j * j for j in range(8)]",
    "_z = set([1, 1, 2, 3])",
    "_z = len('abcdefg')",
    "_z = 'x' * 3",
    "_z = {'k': 7}",
    "# continued",
    "_z = (lambda n: n + 1)(7)",
    "_z = isinstance(3, int)",
    "_z = bool([])",
]


def _prompt_for(body: str, key: str) -> str:
    """The exact per-key eval prompt (shared text across all model arms)."""
    return ("Run the following Python program and report what it prints.\n\n"
            "```python\n" + body + f"\nprint({key})\n```\n\n")


def _build_body(rng: random.Random, n_keys: int, target_tokens: int,
                tok) -> tuple[str, list[str], dict, int]:
    """Build a program body: K bindings near the top + distractor fill, then
    TRIM the fill so the per-key prompt lands at or just under target_tokens
    (measured exactly with `tok`). Returns (body, keys, answers, ntok)."""
    keys = [f"v{i}" for i in range(n_keys)]
    pool = rng.sample(range(1000, 10000), n_keys)            # distinct 4-digit
    answers = {k: str(v) for k, v in zip(keys, pool)}

    lines: list[str] = []
    # Bindings at the very top, lightly interleaved with one distractor each so
    # the binding block matches the training distribution but stays within the
    # first ~120 tokens of the program (binding -> query distance ≈ L).
    for k in keys:
        lines.append(f"{k} = {answers[k]}")
        lines.append(rng.choice(_DISTRACTOR_POOL))
    head_lines = list(lines)

    # Overshoot, then trim to land in [0.94*L, L]. ~3.2 chars/token for this
    # digit-heavy code; overshoot by 15% then bisect-trim by lines.
    target_chars = int(target_tokens * 3.2 * 1.15)
    fill: list[str] = []
    cur = sum(len(x) + 1 for x in head_lines)
    while cur < target_chars:
        line = rng.choice(_DISTRACTOR_POOL)
        fill.append(line)
        cur += len(line) + 1

    def _ntok(nfill: int) -> int:
        body = "\n".join(head_lines + fill[:nfill])
        return len(tok.encode(_prompt_for(body, keys[0]),
                              add_special_tokens=False))

    # Binary-search the largest fill count whose prompt is <= target_tokens.
    lo, hi = 0, len(fill)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _ntok(mid) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    body = "\n".join(head_lines + fill[:lo])
    ntok = len(tok.encode(_prompt_for(body, keys[0]), add_special_tokens=False))
    return body, keys, answers, ntok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--buckets", type=str, default="1024,2048,4096,8192",
                   help="comma list of target TOTAL token lengths L")
    p.add_argument("--per_bucket", type=int, default=150)
    p.add_argument("--n_keys", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", type=str,
                   default="HuggingFaceTB/SmolLM2-135M",
                   help="tokenizer used to MEASURE/calibrate target length "
                        "(OURS's tokenizer). Text is shared across arms.")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    buckets = [int(b) for b in args.buckets.split(",") if b.strip()]
    rng = random.Random(args.seed)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    tok_counts: dict[int, list[int]] = {}
    with open(out_path, "w") as f:
        for L in buckets:
            tok_counts[L] = []
            for j in range(args.per_bucket):
                body, keys, answers, ntok = _build_body(
                    rng, args.n_keys, L, tok)
                tok_counts[L].append(ntok)
                rec = {
                    "task_id": f"flagship/L{L}/{j}",
                    "bucket": L,
                    "n_keys": args.n_keys,
                    "body": body,
                    "keys": keys,
                    "answers": answers,
                    "ours_tok_len": ntok,
                }
                f.write(json.dumps(rec) + "\n")
                n += 1
    for L in buckets:
        c = tok_counts[L]
        print(f"bucket L={L}: {len(c)} tasks, ours-tok mean={sum(c)/len(c):.0f} "
              f"min={min(c)} max={max(c)}")
    print(f"wrote {n} tasks -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
