"""Generate COMPACT name-recall data for the soft-namekey transfer gate (M2).

Two axes, reported separately (mirrors experiments/wm_vqkey_probe.py):
  EXACT   : the recall query reuses the EXACT bound identifier spelling.
  VARIANT : the query uses a surface VARIANT (case: snake↔UPPER; camel:
            snake↔camelCase) of the bound name — the deployed lexical HASH
            scores ~0 here (spelling-locked); this is where a learned soft key
            must win.

FORMAT (compact, deliberately prose-free between the query name and its value):
  prompt =  a config block of `name = value` bindings (distractors + the
            queried binding) + a short question.
  completion = "<prose>`<QUERY_FORM>`: <value>"  — the query name (in backticks,
            so it is an isolated identifier run) sits immediately before the
            value with NO alphabetic word in between. This matters because the
            soft path's name-key carry walks back to the most-recent identifier
            run; a prose word ("the value 1994") would clobber the key. (The
            deployed PROSE format — `code_recall_const` — is therefore out of
            scope for the soft path's any-name carry; that is an honest M2
            limitation, fixed in M3 by a normalized-bound carry.)

Leak-free by construction: the answer value appears (1) in the prompt binding
(the genuine source, ~hundreds of tokens back) and (2) at the completion recall
position — the FIRST completion occurrence, which is what the teacher-forced
kill-gate scores. WM-OFF (recurrence) must fail at the recall position because
the binding is far + there are N competing bindings.

Train / heldout word pools are DISJOINT (no entity leakage): variant robustness
must be learned from the char-level form, not memorized.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_variant_recall.py
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
    w = "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(n))
    return w


def make_word_pool(rng, n, forbidden):
    """n distinct 2-part snake bases (`word_word`), disjoint from `forbidden`."""
    out = []
    seen = set(forbidden)
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


_QUERY_PROSE = [
    "Reading the configuration above.\n",
    "Checking the module settings.\n",
    "From the constants block.\n",
    "Per the definitions listed.\n",
]


def make_record(pool, rng, *, n_vars, kind):
    """One compact recall record. kind in {exact, case, camel}."""
    names = rng.sample(pool, n_vars)                       # distinct snake bases
    values = []
    seen_v = set()
    while len(values) < n_vars:
        v = rng.randint(1000, 9999)
        if v in seen_v:
            continue
        seen_v.add(v)
        values.append(v)
    # the queried binding: NOT the last config line (keep distance + defeat the
    # hash's "carried code = last binding" locality shortcut on variants).
    qi = rng.randrange(0, n_vars - 1)
    q_base = names[qi]
    q_val = values[qi]

    if kind == "exact":
        bound_style = rng.choice(["snake", "upper", "camel"])
        bound_form = _form(q_base, bound_style)
        query_form = bound_form                            # exact reuse
        # all bindings use the same (random) style family for realism
        styles = [bound_style] * n_vars
    elif kind == "case":
        bound_form = _form(q_base, "snake")
        query_form = _form(q_base, "upper")
        styles = ["snake"] * n_vars
    elif kind == "camel":
        bound_form = _form(q_base, "snake")
        query_form = _form(q_base, "camel")
        styles = ["snake"] * n_vars
    else:
        raise ValueError(kind)

    lines = []
    for i in range(n_vars):
        form = (bound_form if i == qi else _form(names[i], styles[i]))
        lines.append(f"{form} = {values[i]}")
    block = "\n".join(lines)
    prompt = (
        "You are reviewing a Python module's constants.\n\n"
        "```python\n" + block + "\n```\n\n"
        f"Question: what value is bound to `{query_form}`?")
    completion = rng.choice(_QUERY_PROSE) + f"`{query_form}`: {q_val}"
    return dict(
        problem_prompt=prompt,
        qwen_completion=completion,
        answer=str(q_val),
        family=f"compact_{kind}",
        n_vars=n_vars,
        recall_key=query_form,
        bound_form=bound_form,
    )


def gen_split(pool, rng, *, kinds, n_per_kind, n_vars_choices):
    out = []
    for kind in kinds:
        for _ in range(n_per_kind):
            nv = rng.choice(n_vars_choices)
            out.append(make_record(pool, rng, n_vars=nv, kind=kind))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train_per_kind", type=int, default=1200)
    ap.add_argument("--n_eval_per_kind", type=int, default=250)
    ap.add_argument("--n_vars", default="16,24,32")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nvs = [int(x) for x in args.n_vars.split(",")]

    # DISJOINT word pools (no entity leakage train↔heldout).
    train_pool = make_word_pool(rng, 4000, forbidden=set())
    eval_pool = make_word_pool(rng, 1500, forbidden=set(train_pool))
    assert not (set(train_pool) & set(eval_pool))

    train = gen_split(train_pool, rng, kinds=["exact", "case", "camel"],
                      n_per_kind=args.n_train_per_kind, n_vars_choices=nvs)
    # separate heldout files per kind (for a clean per-axis table)
    splits = {
        "compact_exact_heldout": (["exact"], args.n_eval_per_kind),
        "compact_case_heldout": (["case"], args.n_eval_per_kind),
        "compact_camel_heldout": (["camel"], args.n_eval_per_kind),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    tp = os.path.join(args.out_dir, "variant_recall_train.jsonl")
    with open(tp, "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    print(f"[train] {len(train)} → {tp}")
    for name, (kinds, n) in splits.items():
        recs = gen_split(eval_pool, rng, kinds=kinds, n_per_kind=n,
                         n_vars_choices=nvs)
        p = os.path.join(args.out_dir, name + ".jsonl")
        with open(p, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"[{name}] {len(recs)} → {p}")


if __name__ == "__main__":
    main()
