"""Realistic long-range CODE recall tasks — the WM-load-bearing probe on REAL code.

WHY (the gap this fills, 2026-06-15).
  The validated WM recall mechanism (name-keyed addressing + copy/pointer readout,
  see project_recall_discrete_key_direction + wm_namekey_probe.py /
  wm_multitok_readout.py) gets 100% exact recall on the SYNTHETIC multibind probe
  (`vN = MMMM`, recall one) where the bare DeltaNet recurrence scores ~0. But
  `vN = MMMM` is NOT representative of the project's real workload (CODING). To
  decide whether WM earns its place we need recall tasks built from REAL coding
  structure, with the recall point at a distance/interference regime that EXCEEDS
  the recurrence.

KEY DESIGN INSIGHT (from the memory: project_recall_no_headroom +
project_wm_recall_probe_broken_and_routed_around).
  SINGLE-binding recall at distance is TRIVIAL for the linear-RNN recurrence (~100%
  even at 512+ tokens — the one binding fits in the recurrent state). Headroom for
  WM appears only when the task is BOTH (1) capacity-exceeding — MANY simultaneous
  competing bindings of the same kind (N constants / N signatures / N imports), and
  (2) non-memorizable — fresh random content. So these realistic tasks plant N
  competing bindings (a realistic config block / utils module / import header) and
  query ONE, with the binding→query gap padded by REAL Python so the distance is
  realistic. Distance is the reported curve axis; N (interference) is the real
  headroom driver — both are knobs.

MECHANISM MAPPING (how the validated WM maps onto each family — see module
HONEST CAVEATS at the bottom and the agent report):
  - KEY (addressing): the identifier token span at the query (`recall_key`), keyed
    on its INPUT-EMBEDDING window (name_emb_key) — exactly the validated addressing.
  - VALUE (readout): the concrete answer token span at the binding
    (`source_value_text`) — copied by the copy/pointer head.
  - mem_read_mask fires over the ANSWER span in qwen_completion (`answer_char_span`).

FAMILIES (addressing type recorded per-record in `addressing`):
  const      : config-constant value recall.   addressing="name"  (DIRECT map)
  signature  : function param-name recall.      addressing="name+ordinal"
  import     : import-alias recall.             addressing="name"  (DIRECT map)
  fname      : function-name-by-PURPOSE recall. addressing="content" (HARDER — flagged:
               the query gives a semantic role, NOT the name, so name-key addressing
               does NOT apply; this needs content/role addressing.)

REALISM (flagged honestly):
  - The recall STRUCTURE (binding + query + answer) is synthetic & controlled so it
    is annotatable + scorable.
  - The DISTRACTOR body is REAL Python (magicoder OSS solutions, data/_realpy_pool.json)
    so the interference (real identifiers, real idioms) is representative.
  - The binding idioms themselves are realistic code (config sections, def signatures,
    import headers).
  See `_build_python_pool` for the real-code source; pass --synthetic_distractors to
  fall back to a built-in pool (no HF needed).

OUTPUT SCHEMA (superset of data/multibind_recall_*.jsonl so it drops into data_mix
`text_field: [problem_prompt, qwen_completion]` AND eval_longctx_recall / eval_code_recall):
  problem_prompt, qwen_completion, answer, extracted_code, has_tests, tier, score,
  task_id (= "{family}/d{bucket}/{i}"), sample_idx, approx_distance_tokens,
  n_bindings, family, addressing,
  recall_key, source_value_text, query_key_text,
  binding_char_span, query_key_char_span (in problem_prompt),
  answer_char_span (in qwen_completion).

Usage:
  # train set (uniform distance + N)
  PYTHONPATH=. .venv/bin/python experiments/gen_code_recall_tasks.py \
      --out data/code_recall_train.jsonl --n_examples 6000 --seed 0
  # heldout distance curve
  PYTHONPATH=. .venv/bin/python experiments/gen_code_recall_tasks.py \
      --out data/code_recall_heldout.jsonl --seed 7 \
      --distance_buckets 256,512,768,1024,1536 --per_bucket 60
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

POOL_PATH = "data/_realpy_pool.json"

# ----------------------------------------------------------------------- naming
_CONST_WORDS = [
    "CACHE", "BUFFER", "RETRY", "BATCH", "WINDOW", "TIMEOUT", "POOL", "QUEUE",
    "CHUNK", "THREAD", "WORKER", "SHARD", "BLOCK", "STRIDE", "MARGIN", "BACKLOG",
    "LIMIT", "QUOTA", "DEPTH", "OFFSET", "EPOCH", "STEP", "PAGE", "FRAME",
]
_CONST_SUFFIX = ["SIZE", "LIMIT", "COUNT", "MAX", "MIN", "BYTES", "MS", "SECONDS",
                 "FACTOR", "THRESHOLD", "CAPACITY", "INTERVAL"]

_FUNC_VERBS = ["compute", "parse", "build", "validate", "encode", "decode",
               "merge", "resolve", "fetch", "render", "serialize", "normalize",
               "aggregate", "filter", "transform", "dispatch", "schedule",
               "compress", "checksum", "tokenize"]
_FUNC_NOUNS = ["record", "payload", "manifest", "session", "ledger", "packet",
               "digest", "snapshot", "registry", "header", "frame", "token",
               "config", "buffer", "request", "response", "node", "edge",
               "schema", "index"]
_PARAM_WORDS = ["record_id", "timestamp", "payload", "checksum", "retries",
                "verbose", "encoding", "max_depth", "session_token", "offset",
                "chunk_size", "callback", "dry_run", "namespace", "priority",
                "shard_key", "deadline", "fallback", "cursor", "batch_id"]

_MODULES = [
    ("numpy", "np"), ("pandas", "pd"), ("matplotlib.pyplot", "plt"),
    ("tensorflow", "tf"), ("collections", "col"), ("itertools", "it"),
    ("functools", "ft"), ("datetime", "dt"), ("networkx", "nx"),
    ("seaborn", "sns"), ("sqlalchemy", "sa"), ("scipy.stats", "st"),
    ("multiprocessing", "mp"), ("statistics", "stats"), ("operator", "op"),
    ("xml.etree.ElementTree", "ET"), ("os.path", "osp"), ("pickle", "pk"),
    ("threading", "th"), ("subprocess", "sp"),
]

_PURPOSE = [
    ("compute_file_checksum", "computes the SHA-256 checksum of a file"),
    ("parse_iso_timestamp", "parses an ISO-8601 timestamp string into a datetime"),
    ("merge_sorted_streams", "merges two sorted iterators into one sorted stream"),
    ("resolve_symlink_chain", "resolves a chain of symbolic links to the real path"),
    ("normalize_unicode_text", "normalizes unicode text to NFC form"),
    ("encode_base62", "encodes an integer into a base-62 string"),
    ("build_adjacency_list", "builds an adjacency list from an edge list"),
    ("validate_email_address", "validates that a string is a well-formed email"),
    ("compress_run_length", "run-length compresses a byte string"),
    ("schedule_retry_backoff", "computes an exponential backoff delay for retries"),
    ("dispatch_event_handler", "dispatches an event to the registered handler"),
    ("aggregate_daily_totals", "aggregates a list of records into daily totals"),
]


def _uniq_names(rng, pool, n, joiner="_"):
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        nm = joiner.join(rng.sample(pool, 2)) if len(pool) > 1 else rng.choice(pool)
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


# ----------------------------------------------------------------- real distractors
_FALLBACK_SNIPPETS = [
    "def _clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
    "def _flatten(xs):\n    return [y for x in xs for y in x]",
    "class _Counter:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1",
    "def _safe_div(a, b):\n    return a / b if b else 0.0",
    "def _running_mean(xs):\n    s = 0.0\n    for i, x in enumerate(xs, 1):\n        s += (x - s) / i\n    return s",
    "def _retry(fn, attempts=3):\n    for _ in range(attempts):\n        try:\n            return fn()\n        except Exception:\n            continue\n    raise RuntimeError('gave up')",
    "def _chunked(seq, size):\n    for i in range(0, len(seq), size):\n        yield seq[i:i + size]",
    "def _dedupe(seq):\n    seen = set()\n    return [x for x in seq if not (x in seen or seen.add(x))]",
]


def _build_python_pool(synthetic: bool) -> list[str]:
    if not synthetic and os.path.exists(POOL_PATH):
        with open(POOL_PATH) as f:
            pool = json.load(f)
        if pool:
            return pool
    # fallback: tile the small built-in pool (varied indentation tweaks)
    return list(_FALLBACK_SNIPPETS)


_ANSWER_GUARD = re.compile(r"answer\s*:", re.IGNORECASE)


def _sanitize_snippet(s: str, forbidden: set[str]) -> str | None:
    """Drop snippets that would corrupt scoring: an 'Answer:' literal, or the
    UNIQUE answer source value (so the copied/recalled span is unambiguous).

    Only tokens of length >= 4 are filtered as substrings — short identifiers
    (2-char aliases like `np`/`pd`) occur in nearly all real Python, so forbidding
    them as substrings starves the pool. Short-token collisions are a REAL code
    difficulty (the alias may appear elsewhere); we document that rather than
    engineer it away."""
    if _ANSWER_GUARD.search(s):
        return None
    for f in forbidden:
        if f and len(f) >= 4 and f in s:
            return None
    return s


def _pad_distractor(rng, pool, tok, target_tokens, forbidden):
    """Concatenate real snippets until ~target_tokens, as a realistic
    'utility module' body. Returns (text, token_len). Accepts a snippet after a
    few rejects rather than starving (collisions tolerated — see _sanitize)."""
    if target_tokens <= 0:
        return "", 0
    out, ntok, guard = [], 0, 0
    while ntok < target_tokens and guard < 2000:
        guard += 1
        s = None
        for _ in range(5):
            cand = rng.choice(pool)
            s = _sanitize_snippet(cand, forbidden)
            if s is not None:
                break
        if s is None:
            continue
        out.append(s)
        ntok += len(tok.encode("\n\n" + s, add_special_tokens=False))
    text = "\n\n".join(out)
    return text, len(tok.encode(text, add_special_tokens=False))


# ----------------------------------------------------------------- families
def _span(haystack: str, needle: str, start: int = 0):
    i = haystack.find(needle, start)
    return None if i < 0 else (i, i + len(needle))


def _wrap(intro, body, question):
    return f"{intro}\n\n```python\n{body}\n```\n\nQuestion: {question}"


def _make_const(rng, pool, tok, dist, n):
    names = [f"{rng.choice(_CONST_WORDS)}_{rng.choice(_CONST_SUFFIX)}"
             for _ in range(n * 3)]
    names = list(dict.fromkeys(names))[:n]
    while len(names) < n:
        names.append(f"{rng.choice(_CONST_WORDS)}_{rng.choice(_CONST_SUFFIX)}_{len(names)}")
    # distinct 4-digit values, all unique strings
    vals, seen = [], set()
    while len(vals) < n:
        v = rng.randint(1000, 9999)
        if str(v) not in seen:
            seen.add(str(v))
            vals.append(v)
    qi = rng.randrange(n)
    qname, qval = names[qi], vals[qi]
    forbidden = {str(v) for v in vals} | set(names)
    distractor, _ = _pad_distractor(rng, pool, tok, dist, forbidden)
    cfg_lines = [f"{nm} = {v}" for nm, v in zip(names, vals)]
    header = ("import os\nimport sys\nimport logging\n\n"
              "# ---- module configuration ----\n" + "\n".join(cfg_lines))
    body = header + "\n\n" + distractor
    intro = "You are reviewing a Python module. Read it and answer the question."
    question = f"What integer value is assigned to the constant `{qname}`?"
    prompt = _wrap(intro, body, question)
    completion = (f"Scanning the configuration block, `{qname}` is assigned the "
                  f"value {qval}, and no later line reassigns it.\n\n"
                  f"Answer: {qval}")
    bind = _span(prompt, f"{qname} = {qval}")
    qkey = _span(prompt, f"`{qname}`")
    # query_key span: the identifier inside the backticks
    qkey_id = (qkey[0] + 1, qkey[1] - 1) if qkey else None
    ans = _span(completion, str(qval), completion.rfind("Answer:"))
    return dict(problem_prompt=prompt, qwen_completion=completion,
                answer=str(qval), extracted_code=f"print({qval})",
                family="const", addressing="name", n_bindings=n,
                recall_key=qname, source_value_text=str(qval),
                query_key_text=qname,
                binding_char_span=bind, query_key_char_span=qkey_id,
                answer_char_span=ans)


def _make_signature(rng, pool, tok, dist, n):
    fnames = _uniq_names(rng, [f"{v}_{nn}" for v in _FUNC_VERBS for nn in _FUNC_NOUNS], n)
    sigs = []
    for fn in fnames:
        k = rng.randint(3, 5)
        params = rng.sample(_PARAM_WORDS, k)
        sigs.append((fn, params))
    qi = rng.randrange(n)
    qfn, qparams = sigs[qi]
    ord_i = rng.randrange(len(qparams))
    ordinal = ["first", "second", "third", "fourth", "fifth"][ord_i]
    qparam = qparams[ord_i]
    forbidden = set(fnames)
    distractor, _ = _pad_distractor(rng, pool, tok, dist, forbidden)
    defs = []
    for fn, params in sigs:
        defs.append(f"def {fn}({', '.join(params)}):\n"
                    f"    \"\"\"Helper {fn.replace('_', ' ')}.\"\"\"\n"
                    f"    return None")
    header = "# ---- helper functions ----\n" + "\n\n".join(defs)
    body = header + "\n\n" + distractor
    intro = "You are reading a Python utility module. Answer the question about it."
    question = (f"In the function `{qfn}`, what is the name of the {ordinal} "
                f"parameter?")
    prompt = _wrap(intro, body, question)
    completion = (f"The definition `def {qfn}({', '.join(qparams)})` has "
                  f"`{qparam}` as its {ordinal} parameter.\n\n"
                  f"Answer: {qparam}")
    bind = _span(prompt, f"def {qfn}(")
    qkey = _span(prompt, f"`{qfn}`")
    qkey_id = (qkey[0] + 1, qkey[1] - 1) if qkey else None
    ans = _span(completion, qparam, completion.rfind("Answer:"))
    # the value source span = the parameter token at the binding signature
    return dict(problem_prompt=prompt, qwen_completion=completion,
                answer=qparam, extracted_code=f"print('{qparam}')",
                family="signature", addressing="name+ordinal", n_bindings=n,
                recall_key=qfn, source_value_text=qparam, query_key_text=qfn,
                binding_char_span=bind, query_key_char_span=qkey_id,
                answer_char_span=ans)


def _make_import(rng, pool, tok, dist, n):
    mods = rng.sample(_MODULES, min(n, len(_MODULES)))
    # make aliases unique
    used = set()
    pairs = []
    for mod, alias in mods:
        a = alias
        j = 0
        while a in used:
            j += 1
            a = f"{alias}{j}"
        used.add(a)
        pairs.append((mod, a))
    qi = rng.randrange(len(pairs))
    qmod, qalias = pairs[qi]
    forbidden = {a for _, a in pairs}
    distractor, _ = _pad_distractor(rng, pool, tok, dist, forbidden)
    imp_lines = [f"import {m} as {a}" if "." not in m or m.count(".") == 0
                 else f"import {m} as {a}" for m, a in pairs]
    header = "# ---- imports ----\n" + "\n".join(imp_lines)
    body = header + "\n\n" + distractor
    intro = "You are reading the top of a Python module. Answer the question."
    question = f"Which alias is the module `{qmod}` imported under?"
    prompt = _wrap(intro, body, question)
    completion = (f"The import header contains `import {qmod} as {qalias}`, "
                  f"so the module `{qmod}` is available under the alias "
                  f"`{qalias}`.\n\nAnswer: {qalias}")
    bind = _span(prompt, f"import {qmod} as {qalias}")
    qkey = _span(prompt, f"`{qmod}`")
    qkey_id = (qkey[0] + 1, qkey[1] - 1) if qkey else None
    ans = _span(completion, qalias, completion.rfind("Answer:"))
    return dict(problem_prompt=prompt, qwen_completion=completion,
                answer=qalias, extracted_code=f"print('{qalias}')",
                family="import", addressing="name", n_bindings=len(pairs),
                recall_key=qmod, source_value_text=qalias, query_key_text=qmod,
                binding_char_span=bind, query_key_char_span=qkey_id,
                answer_char_span=ans)


def _make_fname(rng, pool, tok, dist, n):
    chosen = rng.sample(_PURPOSE, min(n, len(_PURPOSE)))
    qi = rng.randrange(len(chosen))
    qfn, qpurpose = chosen[qi]
    forbidden = {fn for fn, _ in chosen}
    distractor, _ = _pad_distractor(rng, pool, tok, dist, forbidden)
    defs = []
    for fn, purpose in chosen:
        defs.append(f"def {fn}(*args, **kwargs):\n"
                    f"    \"\"\"This function {purpose}.\"\"\"\n"
                    f"    raise NotImplementedError")
    header = "# ---- library functions ----\n" + "\n\n".join(defs)
    body = header + "\n\n" + distractor
    intro = "You are reading a Python library. Answer the question about it."
    question = (f"Which function in this module {qpurpose}? Give its name.")
    prompt = _wrap(intro, body, question)
    completion = (f"The function whose docstring says it {qpurpose} is "
                  f"`{qfn}`.\n\nAnswer: {qfn}")
    bind = _span(prompt, f"def {qfn}(")
    # NOTE: content addressing — the query key is the PURPOSE phrase, not a name.
    # The phrase appears first in the docstring (next to the def); the RECALL
    # query is its occurrence in the trailing Question, so locate it there.
    qkey = _span(prompt, qpurpose, prompt.rfind("Question:"))
    ans = _span(completion, qfn, completion.rfind("Answer:"))
    return dict(problem_prompt=prompt, qwen_completion=completion,
                answer=qfn, extracted_code=f"print('{qfn}')",
                family="fname", addressing="content", n_bindings=len(chosen),
                recall_key=qpurpose, source_value_text=qfn,
                query_key_text=qpurpose,
                binding_char_span=bind, query_key_char_span=qkey,
                answer_char_span=ans)


_BUILDERS = {"const": _make_const, "signature": _make_signature,
             "import": _make_import, "fname": _make_fname}


def _measured_distance(prompt, rec, tok):
    """True token gap from the END of the binding source span to the START of
    the query-key mention (the recall distance the model must bridge)."""
    b = rec.get("binding_char_span")
    q = rec.get("query_key_char_span")
    if not b or not q:
        return rec.get("approx_distance_tokens", 0)
    lo, hi = (b[1], q[0]) if b[1] <= q[0] else (q[1], b[0])
    return len(tok.encode(prompt[lo:hi], add_special_tokens=False))


def build_one(rng, pool, tok, family, dist, n):
    rec = _BUILDERS[family](rng, pool, tok, dist, n)
    rec["approx_distance_tokens"] = _measured_distance(rec["problem_prompt"], rec, tok)
    rec["has_tests"] = False
    rec["score"] = 1.0
    rec["tier"] = f"code_recall_{family}"
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_examples", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", type=str,
                   default="const,signature,import,fname")
    p.add_argument("--n_bindings_choices", type=str, default="8,16,24",
                   help="competing-binding counts sampled uniformly (the "
                        "capacity/interference knob — the real headroom driver)")
    p.add_argument("--distance_min", type=int, default=256)
    p.add_argument("--distance_max", type=int, default=1536)
    p.add_argument("--distance_buckets", type=str, default="",
                   help="heldout curve mode: comma distances; per_bucket each")
    p.add_argument("--per_bucket", type=int, default=60)
    p.add_argument("--max_total_tokens", type=int, default=1850,
                   help="skip records whose prompt+completion exceeds this "
                        "(must leave room under the 2048 train/eval window)")
    p.add_argument("--synthetic_distractors", action="store_true")
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pool = _build_python_pool(args.synthetic_distractors)
    print(f"[pool] {len(pool)} distractor snippets "
          f"({'synthetic' if args.synthetic_distractors else 'real magicoder'})")

    families = [f for f in args.families.split(",") if f.strip()]
    n_choices = sorted(int(x) for x in args.n_bindings_choices.split(",") if x.strip())
    rng = random.Random(args.seed)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # approx token cost PER competing binding (the config-block / def-block / import
    # header line), so we can cap N to fit `dist` under max_total_tokens on the
    # FIRST try instead of burning retries (the headroom driver is still N, but the
    # token budget at long distance forces smaller N — documented tension).
    _PER_BIND = {"const": 7, "signature": 30, "import": 9, "fname": 32}
    _OVERHEAD = 170  # intro + question + completion

    def _feasible_n(family, dist):
        budget = args.max_total_tokens - dist - _OVERHEAD
        per = _PER_BIND[family]
        feas = [n for n in n_choices if n * per <= budget]
        return feas or [n_choices[0]]

    def _bucket_possible(family, dist):
        """A (family, dist) is feasible only if even the SMALLEST binding block
        fits under max_total_tokens — else skip the bucket instead of dead-looping
        (e.g. signature defs are large, so d1536 cannot fit 8 defs + 1536 + body)."""
        return n_choices[0] * _PER_BIND[family] + dist + _OVERHEAD <= args.max_total_tokens

    def emit(f, family, dist, nominal_bucket, idx):
        feas = _feasible_n(family, dist)
        for _try in range(3):
            n = rng.choice(feas)
            rec = build_one(rng, pool, tok, family, dist, n)
            full = rec["problem_prompt"] + "\n\n" + rec["qwen_completion"]
            ntok = len(tok.encode(full, add_special_tokens=False))
            if ntok <= args.max_total_tokens and rec.get("binding_char_span"):
                rec["task_id"] = f"{family}/d{nominal_bucket}/{idx}"
                rec["sample_idx"] = idx
                rec["total_tokens"] = ntok
                f.write(json.dumps(rec) + "\n")
                return True
        return False

    written = 0
    dist_hist = {}
    fam_hist = {fam: 0 for fam in families}
    with open(out_path, "w") as f:
        if args.distance_buckets.strip():
            buckets = [int(b) for b in args.distance_buckets.split(",") if b.strip()]
            idx = 0
            for fam in families:
                for b in buckets:
                    if not _bucket_possible(fam, b):
                        print(f"  [skip] {fam} d{b}: even min N={n_choices[0]} "
                              f"exceeds max_total_tokens={args.max_total_tokens}")
                        continue
                    got, tries = 0, 0
                    while got < args.per_bucket and tries < args.per_bucket * 6:
                        tries += 1
                        if emit(f, fam, b, b, idx):
                            written += 1
                            got += 1
                            fam_hist[fam] += 1
                            dist_hist[b] = dist_hist.get(b, 0) + 1
                        idx += 1
        else:
            idx = 0
            while written < args.n_examples:
                fam = rng.choice(families)
                dist = rng.randint(args.distance_min, args.distance_max)
                nominal = min(buckets_floor(dist), 1536)
                if emit(f, fam, dist, nominal, idx):
                    written += 1
                    fam_hist[fam] += 1
                    dist_hist[nominal] = dist_hist.get(nominal, 0) + 1
                idx += 1
                if idx > args.n_examples * 30:
                    break
    print(f"wrote {written} -> {out_path}")
    print(f"  families: {fam_hist}")
    print(f"  nominal-distance histogram: {dict(sorted(dist_hist.items()))}")


def buckets_floor(d):
    for b in (1536, 1024, 768, 512, 256):
        if d >= b:
            return b
    return 256


if __name__ == "__main__":
    sys.exit(main())
