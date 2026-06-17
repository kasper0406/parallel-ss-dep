"""ALIAS-CHAIN recall data — the decisive test for SEMANTIC vs SURFACE addressing.

THE QUESTION (research, fair-baseline discipline — a clean negative is a valid
result): every WM addressing key we have tried keys on SURFACE FORM (exact
hash, normalized hash, learned name-span embedding key). They break on
aliases/paraphrase where the recall query does NOT textually match the binding
that actually holds the value. The GENERAL primitive is addressing by
MEANING/REFERENCE: a learned attention whose Q/K come from the trunk's
CONTEXTUAL hidden states (what the context says a token refers to). This data
isolates exactly that gap.

FORMAT (one record → tokenized by experiments/alias_addressing_probe.py):
  A block of `name = value` bindings (distractors + the real binding), then —
  for the alias variants — one or two ALIAS lines `B = A` / `C = B` that make a
  later name REFER to the value-holding name WITHOUT repeating the value, then a
  recall query referencing the *terminal* alias name. The answer is the value,
  which appears textually ONLY in the value-holder's binding line.

  kind="exact"  : query reuses the value-holder's EXACT spelling (no alias).
                  SURFACE addressing CAN solve this — the harness sanity arm.
  kind="alias1" : `A = 1234` ... `B = A` ... query B → 1234.
                  SURFACE key on B matches B's OWN slot (whose value is the NAME
                  `A`, not the digits) → copies a name → FAILS. Only addressing
                  that RESOLVES the reference B→A→value succeeds.
  kind="alias2" : `A = 1234` ... `B = A` ... `C = B` ... query C → 1234.

LEAK-FREE BY CONSTRUCTION:
  - the value digits appear exactly ONCE (the value-holder's binding line), far
    back; the probe scores the model PREDICTING the value at the recall position
    (before the teacher-forced restatement) → genuine long-range recall.
  - the value-holder is placed in the FIRST ~40% of the binding block and there
    are always later (distractor / alias) lines, so it is NEVER the most-recent
    binding → no recency shortcut.
  - all values are distinct 4-digit numbers and all names distinct, so the
    correct binding (and the answer) is unambiguous.
  - train / heldout NAME pools are DISJOINT (no entity leakage) so any learned
    addresser must generalize, not memorize name→slot.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_alias_recall.py
  PYTHONPATH=. .venv/bin/python experiments/gen_alias_recall.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import random

_CONS = "bcdfghjklmnpqrstvwxz"
_VOW = "aeiou"


def _word(rng):
    n = rng.randint(2, 3)
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(n))


def make_name_pool(rng, n, forbidden):
    """n distinct 2-part snake bases (`word_word`), disjoint from `forbidden`."""
    out, seen = [], set(forbidden)
    while len(out) < n:
        base = _word(rng) + "_" + _word(rng)
        if base in seen:
            continue
        seen.add(base)
        out.append(base)
    return out


def _form(snake_base: str, style: str) -> str:
    parts = snake_base.split("_")
    if style == "snake":
        return "_".join(parts)
    if style == "upper":
        return "_".join(parts).upper()
    if style == "camel":
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    raise ValueError(style)


def make_record(pool, rng, *, n_vars, kind):
    """One alias-recall record.

    `lines` is the ORDERED list of (name, value) bindings as they appear in the
    text. For an alias line the "value" is another NAME (a reference), flagged in
    `value_is_ref`. `qb_line` is the index of the binding that actually HOLDS the
    answer digits (the correct addressing target). `alias_match_line` is the
    index of the slot a SURFACE/form key would match for the query (== qb_line
    for exact; the terminal alias's own slot for aliases).
    """
    hops = {"exact": 0, "alias1": 1, "alias2": 2}[kind]
    # distinct value-holder names + values (all 4-digit, distinct)
    names = rng.sample(pool, n_vars)
    vals, seen_v = [], set()
    while len(vals) < n_vars:
        v = rng.randint(1000, 9999)
        if v not in seen_v:
            seen_v.add(v)
            vals.append(v)
    # the value-holder (identity chosen at random — position decided below)
    qi = rng.randrange(0, n_vars)
    style = rng.choice(["snake", "camel"])
    holder_form = _form(names[qi], style)

    # ---- ordered binding block ----
    digit_lines = []
    for i in range(n_vars):
        digit_lines.append([_form(names[i], rng.choice(["snake", "camel"])),
                            str(vals[i]), False])
    # Place the holder at a UNIFORMLY-RANDOM block slot (review fix: removes the
    # positional confound the contextual arm could otherwise exploit), BUT never
    # the LAST block slot → there is always >=1 distractor DIGIT binding AFTER
    # the holder, so "grab the most-recent value" is never the answer (no recency
    # shortcut). For alias kinds the alias lines (whose values are NAMES, not
    # digits) are appended after the block, adding further distance.
    holder_line = [holder_form, str(vals[qi]), False]
    others = [digit_lines[i] for i in range(n_vars) if i != qi]
    rng.shuffle(others)
    holder_pos = rng.randrange(0, n_vars - 1)        # [0, n_vars-2] → >=1 after
    block = others[:holder_pos] + [holder_line] + others[holder_pos:]

    # ---- alias chain (appended AFTER all digit lines, in chain order) -------
    alias_names = make_name_pool(rng, hops, forbidden=set(names))
    alias_lines = []
    prev_name = holder_form
    chain_forms = []
    for h in range(hops):
        a_form = _form(alias_names[h], rng.choice(["snake", "camel"]))
        alias_lines.append([a_form, prev_name, True])   # `a_form = prev_name`
        chain_forms.append(a_form)
        prev_name = a_form
    query_name = chain_forms[-1] if hops > 0 else holder_form

    lines = block + alias_lines
    # locate the value-holder + the surface-match slot in the final ordering
    qb_line = next(i for i, ln in enumerate(lines) if ln is holder_line)
    alias_match_line = next(i for i, ln in enumerate(lines)
                            if ln[0] == query_name)

    return dict(
        kind=kind, n_vars=n_vars, hops=hops,
        lines=[[n, v] for (n, v, _r) in lines],
        value_is_ref=[bool(r) for (_n, _v, r) in lines],
        query_name=query_name, answer=str(vals[qi]),
        qb_line=qb_line, alias_match_line=alias_match_line,
    )


def gen_split(pool, rng, *, kinds, n_per, n_vars_choices):
    out = []
    for kind in kinds:
        for _ in range(n_per):
            out.append(make_record(pool, rng, n_vars=rng.choice(n_vars_choices),
                                   kind=kind))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_per_kind", type=int, default=1500)
    ap.add_argument("--n_eval_per_kind", type=int, default=200)
    ap.add_argument("--n_vars", default="16,32,64")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_train_per_kind = 60
        args.n_eval_per_kind = 30
        args.n_vars = "8,16"

    rng = random.Random(args.seed)
    nvs = [int(x) for x in args.n_vars.split(",")]
    kinds = ["exact", "alias1", "alias2"]

    # DISJOINT name pools (no entity leakage train↔heldout).
    train_pool = make_name_pool(rng, 6000, forbidden=set())
    eval_pool = make_name_pool(rng, 3000, forbidden=set(train_pool))
    assert not (set(train_pool) & set(eval_pool))

    os.makedirs(args.out_dir, exist_ok=True)
    train = gen_split(train_pool, rng, kinds=kinds,
                      n_per=args.n_train_per_kind, n_vars_choices=nvs)
    tp = os.path.join(args.out_dir, "alias_recall_train.jsonl")
    with open(tp, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    print(f"[train] {len(train)} → {tp}")

    # per (kind, N) heldout files for a clean per-axis table
    for kind in kinds:
        for N in nvs:
            recs = gen_split(eval_pool, rng, kinds=[kind], n_per=args.n_eval_per_kind,
                             n_vars_choices=[N])
            p = os.path.join(args.out_dir, f"alias_recall_{kind}_N{N}_heldout.jsonl")
            with open(p, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"[{kind} N={N}] {len(recs)} → {p}")


if __name__ == "__main__":
    main()
