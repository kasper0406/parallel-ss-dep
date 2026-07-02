"""Held-out recall GENERALIZATION probe sets — does the recall skill the
recall-cotrain taught transfer beyond the trained synthetic format, or is it
format-overfit?

Leak-free / first-occurrence throughout: the bound value appears EXACTLY once
(at the binding near the top) and is NEVER restated before the query, so a
recency copy cannot answer — the model must recall across the distance L.

THREE orthogonal axes, each isolated against the TRAINED-FORMAT control
(cue=pyout, value=int4, distractor=synthetic — i.e. exactly what
gen_scoreboard_recall_train.py produced):

  (a) CUE / FORMAT  (value=int4, distractor=synthetic):
        pyout    : `print(vQ)  # `            (CONTROL = trained)
        assert   : `assert vQ == `
        prose    : `# the value of vQ is `
        indirect : `answer = vQ\nprint(answer)  # `
  (b) VALUE / BINDING FORM (cue=pyout, distractor=synthetic):
        int4     : `vQ = 5440`                 (CONTROL = trained)
        bigint   : `vQ = 5839271`              (7-digit, multi-token)
        string   : `vQ = "zqfk"`               (alpha string value)
        dict     : `cfg = {"slot": 5440}` -> `print(cfg["slot"])  # `
        func     : `def get_vQ():\n    return 5440` -> `print(get_vQ())  # `
  (c) DISTRACTOR (cue=pyout, value=int4):
        synthetic: canned `_z = ...` pool        (CONTROL = trained)
        natural  : real Python lines sampled from Magicoder (filtered: no 4-digit
                   run, never assigns/uses a vN, len 8..80) — genuine code idioms.

Each record: prompt (full text ENDING at the cue, ready to feed), gold (answer
string), match (extractor: int4 | int_any | string), axis, variant, bucket L.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_recall_variants.py \
      --out data/recall_variants_heldout.jsonl --per_bucket 25 \
      --buckets 512,2048,4096 --seed 123
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Synthetic distractor pool (same family as the trained set): each line compiles
# standalone, no inter-line refs, NO 4-digit literal, never assigns a vN.
_SYNTH = [
    "# bookkeeping step", "_z = sum([i for i in range(10)])", "_z = abs(-42)",
    "_z = max(2, 9, 5)", "_z = sorted([5, 2, 9, 1])", "_z = bool(True)",
    "_z = ord('A')", "_z = round(3.14159, 2)", "_z = tuple([10, 20, 30])",
    "_z = ''.join(['a', 'b', 'c'])", "_z = type({}).__name__",
    "_z = list(range(0, 20, 4))", "_z = divmod(17, 5)", "_z = chr(65) + chr(66)",
    "_z = 2 ** 8 - 1", "_z = [j * j for j in range(8)]", "_z = set([1, 1, 2, 3])",
    "_z = len('abcdefg')", "_z = 'x' * 3", "_z = {'k': 7}", "# continued",
    "_z = (lambda n: n + 1)(7)", "_z = isinstance(3, int)", "_z = bool([])",
]

_FOURPLUS = re.compile(r"\d{4,}")
_VN = re.compile(r"\bv\d\b")


def build_natural_pool(n_target: int, seed: int, source: str = "magicoder",
                       allow_synth_fallback: bool = False) -> list[str]:
    """Sample real Python lines from cached code datasets; filter so they
    can't introduce a 4-digit literal or clobber/leak a vN binding. `source`
    selects the dataset so a CROSS-SOURCE held-out (train on one, eval on a
    different one) is possible — a true natural-code generalization test.

    On failure to build a natural pool (dataset load error, OR the extracted
    pool coming back empty), the default is to HARD-FAIL (raise
    RuntimeError) rather than silently substitute synthetic lines. A prior
    version returned a *copy* of `_SYNTH` here while downstream labeling
    (this file's own `distractor=natural` axis, and
    `gen_recall_diverse_train.py`'s `pool is not _SYNTH` identity check)
    kept recording those rows as genuine natural-code distractors — silently
    corrupting the natural-vs-synthetic axis of any probe/training file
    built from a failed load. Pass `allow_synth_fallback=True`
    (CLI: `--allow_synth_fallback`) to opt back into the old
    degrade-gracefully behavior for local iteration where the distinction
    doesn't matter.

    When the fallback fires, this function returns the literal `_SYNTH`
    object (NOT a copy), so identity-based downstream checks — notably
    `gen_recall_diverse_train.py:87`'s `pool is not _SYNTH` — correctly see
    the fallback as synthetic and label those rows "synthetic" without
    needing to edit that consumer. This file's own `main()` additionally
    detects `build_natural_pool(...) is _SYNTH` and relabels its
    `("distractor", "natural")` plan entry as `("distractor",
    "synthetic_fallback")` so records generated here are never silently
    mislabeled either."""
    def _fail_or_fallback(reason: str) -> list[str]:
        if not allow_synth_fallback:
            raise RuntimeError(
                f"[natural pool/{source}] {reason}. Refusing to silently "
                "substitute the synthetic pool here — a prior version did "
                "this and downstream code kept labeling the resulting rows "
                "'natural', corrupting the natural-vs-synthetic axis. Pass "
                "allow_synth_fallback=True / --allow_synth_fallback to "
                "opt into the old (now explicitly labeled) fallback "
                "behavior.")
        print(f"  [natural pool/{source}] {reason}; falling back to "
              "synthetic (--allow_synth_fallback set)")
        return _SYNTH   # literal object, not a copy — see docstring above.

    try:
        from datasets import load_dataset
        if source == "codeparrot":
            ds = load_dataset("codeparrot/codeparrot-clean", split="train",
                              streaming=True)
            field = "content"
            it = iter(ds)
        else:
            ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
            field = "solution"
            it = None
    except Exception as e:  # noqa: BLE001
        return _fail_or_fallback(f"dataset load failed ({e!r})")
    rng = random.Random(seed)
    pool: list[str] = []
    if it is not None:   # streaming source (codeparrot)
        seen_docs = 0
        for ex in it:
            seen_docs += 1
            sol = ex.get(field) or ""
            for raw in sol.splitlines():
                line = raw.rstrip(); s = line.strip()
                if not (8 <= len(s) <= 80):
                    continue
                if _FOURPLUS.search(line) or _VN.search(line):
                    continue
                if "```" in line or "print(" in line:
                    continue
                pool.append(line.lstrip())
            if len(pool) >= n_target or seen_docs > 4000:
                break
        seen = set(); out = []
        for x in pool:
            if x not in seen:
                seen.add(x); out.append(x)
        print(f"  [natural pool/{source}] {len(out)} unique real Python lines")
        return out or _fail_or_fallback(
            f"extracted 0 usable lines from {seen_docs} streamed docs")
    idxs = list(range(min(len(ds), 20000)))
    rng.shuffle(idxs)
    for i in idxs:
        sol = ds[i].get(field) or ""
        for raw in sol.splitlines():
            line = raw.rstrip()
            s = line.strip()
            if not (8 <= len(s) <= 80):
                continue
            if _FOURPLUS.search(line) or _VN.search(line):
                continue
            if "```" in line or "print(" in line:   # avoid fences / stray prints
                continue
            # keep realistic statements (assignments, calls, defs, comments, ifs)
            pool.append(line.lstrip())   # de-indent so it packs as top-level-ish
        if len(pool) >= n_target:
            break
    # dedup, keep order
    seen = set(); out = []
    for x in pool:
        if x not in seen:
            seen.add(x); out.append(x)
    print(f"  [natural pool] {len(out)} unique real Python lines")
    return out or _fail_or_fallback(
        f"extracted 0 usable lines from {source} (dataset had {len(ds)} rows)")


def _ntok(tok, s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))


def _fill_to_length(tok, head: str, cue: str, L: int, pool, rng) -> str:
    """Append distractor lines between head and cue until tok(head+fill+cue) ~ L."""
    lines = []
    cur = _ntok(tok, head + "\n" + cue)
    # coarse fill then stop just under L
    while cur < L:
        line = rng.choice(pool)
        lines.append(line)
        cur += _ntok(tok, line + "\n")
    # trim last few if overshoot
    while lines and _ntok(tok, head + "\n" + "\n".join(lines) + "\n" + cue) > L:
        lines.pop()
    body = head + ("\n" + "\n".join(lines) if lines else "")
    return body + "\n" + cue


def make_record(tok, axis, variant, L, pool, rng):
    key = "v0"
    val_int = rng.randint(1000, 9999)
    # ---- binding head + cue + gold per (axis, variant) ----
    if axis == "value":
        if variant == "int4":
            head = f"{key} = {val_int}"; cue = f"print({key})  # "
            gold = str(val_int); match = "int4"
        elif variant == "bigint":
            v = rng.randint(1000000, 9999999)
            head = f"{key} = {v}"; cue = f"print({key})  # "
            gold = str(v); match = "int_any"
        elif variant == "string":
            s = "".join(rng.choice("bcdfghjklmnpqrstvwxyz") for _ in range(4))
            head = f'{key} = "{s}"'; cue = f"print({key})  # "
            gold = s; match = "string"
        elif variant == "dict":
            head = f'cfg = {{"slot": {val_int}}}'; cue = 'print(cfg["slot"])  # '
            gold = str(val_int); match = "int4"
        elif variant == "func":
            head = f"def get_{key}():\n    return {val_int}"
            cue = f"print(get_{key}())  # "
            gold = str(val_int); match = "int4"
        else:
            raise ValueError(variant)
    else:  # axis in {cue, distractor}: value is always int4
        head = f"{key} = {val_int}"
        gold = str(val_int); match = "int4"
        if axis == "distractor":
            cue = f"print({key})  # "                      # pyout (trained)
        else:  # axis == "cue"
            if variant == "pyout":
                cue = f"print({key})  # "
            elif variant == "assert":
                cue = f"assert {key} == "
            elif variant == "prose":
                cue = f"# the value of {key} is "
            elif variant == "indirect":
                cue = f"answer = {key}\nprint(answer)  # "
            elif variant == "arrow":            # NOVEL cue (held-out, not trained)
                cue = f"# {key} -> "
            elif variant == "eq":               # NOVEL cue (held-out, not trained)
                cue = f"# {key} == "
            else:
                raise ValueError(variant)
    prompt = _fill_to_length(tok, head, cue, L, pool, rng)
    return {"prompt": prompt, "gold": gold, "match": match,
            "axis": axis, "variant": variant, "bucket": L,
            "ntok": _ntok(tok, prompt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", default="512,2048,4096")
    ap.add_argument("--per_bucket", type=int, default=25)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--no_natural", action="store_true",
                    help="skip the natural-distractor axis")
    ap.add_argument("--natural_source", default="magicoder",
                    choices=["magicoder", "codeparrot"])
    ap.add_argument("--allow_synth_fallback", action="store_true",
                    help="on natural-pool build failure, degrade to the "
                         "synthetic pool instead of hard-failing (rows are "
                         "then labeled distractor=synthetic_fallback, not "
                         "natural — see build_natural_pool docstring)")
    ap.add_argument("--strict", action="store_true",
                    help="STRICT cross-distribution held-out plan: NOVEL cues "
                         "(arrow/eq, never trained) + natural distractors from "
                         "the given --natural_source (use codeparrot != the "
                         "magicoder the diverse-train uses).")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    rng = random.Random(args.seed)

    natural = None if args.no_natural else build_natural_pool(
        4000, args.seed, source=args.natural_source,
        allow_synth_fallback=args.allow_synth_fallback)
    # If build_natural_pool degraded to synthetic (only possible with
    # --allow_synth_fallback), `natural` is the literal `_SYNTH` object (see
    # its docstring) -- detect that here and relabel the plan entry so THIS
    # file's own (axis, variant) records aren't silently mislabeled
    # "natural" either.
    natural_is_fallback = natural is _SYNTH
    natural_variant = "synthetic_fallback" if natural_is_fallback else "natural"
    if natural_is_fallback:
        print(f"  WARNING: natural pool build fell back to synthetic lines; "
              f"distractor rows for this axis are labeled "
              f"'{natural_variant}', not 'natural'.")

    if args.strict:
        # everything here is OUT of the diverse-train distribution:
        plan = [("cue", "arrow"), ("cue", "eq"),  # novel cues
                ("value", "string"), ("value", "int4")]
        if natural is not None:
            plan.append(("distractor", natural_variant))   # cross-source (codeparrot)
    else:
        plan = [
            ("value", "int4"), ("value", "bigint"), ("value", "string"),
            ("value", "dict"), ("value", "func"),
            ("cue", "pyout"), ("cue", "assert"), ("cue", "prose"), ("cue", "indirect"),
            ("distractor", "synthetic"),
        ]
        if natural is not None:
            plan.append(("distractor", natural_variant))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for axis, variant in plan:
            is_natural_variant = axis == "distractor" and variant == natural_variant
            pool = natural if is_natural_variant else _SYNTH
            for L in buckets:
                for _ in range(args.per_bucket):
                    rec = make_record(tok, axis, variant, L, pool, rng)
                    f.write(json.dumps(rec) + "\n")
                    n += 1
    print(f"wrote {n} -> {args.out}  variants={len(plan)} buckets={buckets} per_bucket={args.per_bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
