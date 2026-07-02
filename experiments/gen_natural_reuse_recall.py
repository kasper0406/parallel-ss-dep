"""Generate natural-code far-identifier-reuse recall supervision.

MISSION (2026-07-02 forensics follow-up): the dependency-distance-stratified
probe (`experiments/probe_depdist_stratified_ce.py`, `runs/depdist_probe/`,
`SESSION_FINDINGS.md` 2026-07-02) found REAL headroom on natural code: an
identifier whose previous occurrence was >=256 LM tokens back costs 2.0-3.4
extra nats vs a fresh/near token, and WM's copy head is confirmed
"structurally zero on code CE in every deployed ckpt" because no pretrain
data source ever raises `mem_read_mask` at a natural far-reuse position (the
existing recall streams are all synthetic var=value / lookup-table tasks).
This script builds that missing data source: real codeparrot-clean Python
files, with the WM read mask placed at the token position where a real
identifier (function/class/variable name) is reused >=256 LM tokens after its
previous mention in the SAME file.

Repo lesson (project_const_recall_mask_mismatch / const-recall mask
mismatch): supervise recall where recurrence FAILS, at the position of
actual use — not on a restated answer. Here there is no restated answer at
all; the "answer" IS the natural reuse position in the running code, so the
mask is placed exactly where recall is structurally required.

SCHEMA (matches experiments/data_mix.py `_build_read_mask` with ZERO code
changes — see the module docstring below for why):
    text                 : str   — a contiguous natural-code chunk (~1-2k LM
                                    tokens) containing BOTH the identifier's
                                    prior occurrence (the "binding" site) and
                                    the far reuse.
    answer               : str   — the identifier text (informational).
    answer_char_span     : [c0, c1] — char offsets of the REUSE occurrence
                                    inside `text` (0-based from the start of
                                    `text`, since `text_field`==`answer_field`
                                    == "text" so data_mix's `base` offset is
                                    always 0 — see docstring below).
    binding_char_span    : [c0, c1] — char offsets of the PRIOR occurrence
                                    inside `text` (informational + used by
                                    the validation harness; NOT consumed by
                                    data_mix, mirrors code_recall_train's
                                    field of the same name).
    approx_distance_tokens: int  — LM-token distance from the prior
                                    occurrence to the reuse (computed on the
                                    FULL original file, not the cut chunk).
    family / tier / task_id / sample_idx / source_row / repo_name / path /
    score / has_tests / total_tokens : informational metadata, mirrors the
    conventions of data/code_recall_train.jsonl / multibind_recall_pretrain.jsonl.

WHY THIS SCHEMA NEEDS ZERO data_mix.py CHANGES:
    `_build_read_mask` computes `base = len(text) - len(ans_text)` where
    `ans_text = ex[answer_field]`. Existing recall sources use a TWO-field
    `text_field: [problem_prompt, qwen_completion]` (joined with "\n\n") with
    `answer_field: qwen_completion` (the LAST joined part) so `base` recovers
    the join offset. Natural code has no such prompt/completion split — and
    joining two char-sliced halves of one file with an artificial "\n\n"
    would corrupt real code structure. Instead this generator uses a SINGLE
    `text_field: text` string AND sets `answer_field: text` (the SAME field)
    in the YAML — so `ans_text is text`, `base` is always 0, and
    `answer_char_span` is a plain absolute char offset into the emitted
    chunk. `mask_first_occurrence` is left at its default False: unlike
    code_recall_train (where the annotated span points at a recency-trivial
    RESTATED answer and first-occurrence must be used instead), here the
    annotated `answer_char_span` ALREADY IS the hard (far, non-first, only
    genuinely-recallable) occurrence — re-deriving "first occurrence of the
    value" would in fact walk back to the EASY (near-start) original
    binding, which is exactly wrong. So the direct-span path is correct here.
    See configs/natural_reuse_recall_mini.yaml for the exact source stanza.

USAGE (generate + write sidecar meta):
    PYTHONPATH=. .venv/bin/python experiments/gen_natural_reuse_recall.py \\
        --n_train 45000 --n_heldout 2000 \\
        --out data/natural_reuse_recall.jsonl \\
        --out_heldout data/natural_reuse_recall_heldout.jsonl

VALIDATION (decode the emitted read_mask through the real MixedSourceStream
pipeline, print samples + stats):
    PYTHONPATH=. .venv/bin/python experiments/gen_natural_reuse_recall.py \\
        --validate --config configs/natural_reuse_recall_mini.yaml
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import keyword
import pathlib
import random
import sys
import time
import tokenize as _tokenize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"

# Depdist-probe eval window (SESSION_FINDINGS.md 2026-07-02 / mission brief):
# stream rows ~50000-52877 with files >=3072 tokens were used as the probe's
# scoring set. We must never train on those files, so this row window is
# unconditionally skipped regardless of how far the scan has to run.
PROBE_EVAL_ROW_LO = 50000
PROBE_EVAL_ROW_HI = 52877

# Small stoplist of near-universal short names that are almost always
# same-scope-only (self/cls) or too generic to carry a real "recall" lesson
# even when they happen to clear the length floor coincidentally.
NAME_STOPLIST = {"self", "cls", "args", "kwargs"}


# ---------------------------------------------------------------------------
# Python NAME-token extraction (real identifiers, not LM sub-word pieces).

def python_name_occurrences(source: str, min_name_len: int
                            ) -> list[tuple[str, int, int]]:
    """Return [(name, char_start, char_end), ...] in source order for every
    Python NAME token that is not a keyword/soft-keyword, not in the small
    generic stoplist, and has length >= min_name_len. Uses the stdlib
    `tokenize` module (real Python lexer, not the LM tokenizer) so matches
    are genuine identifiers, not coincidental sub-word overlaps. Returns []
    on any parse failure (non-UTF8-clean / malformed source — common enough
    in a large streamed corpus that this must be a soft failure, not a
    crash)."""
    out: list[tuple[str, int, int]] = []
    # Precompute cumulative char offset at the start of each (1-indexed) line
    # so tokenize's (row, col) positions convert to flat char offsets.
    lines = source.splitlines(keepends=True)
    line_start = [0] * (len(lines) + 2)
    for i, ln in enumerate(lines):
        line_start[i + 1] = line_start[i] + len(ln)
    try:
        for tok in _tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != _tokenize.NAME:
                continue
            name = tok.string
            if (keyword.iskeyword(name) or keyword.issoftkeyword(name)
                    or name in NAME_STOPLIST or len(name) < min_name_len):
                continue
            srow, scol = tok.start
            erow, ecol = tok.end
            if srow >= len(line_start) or erow >= len(line_start):
                continue
            c0 = line_start[srow - 1] + scol
            c1 = line_start[erow - 1] + ecol
            if c1 <= c0:
                continue
            out.append((name, c0, c1))
    except Exception:
        # Broad catch is deliberate: this walks arbitrary streamed source
        # files (tokenize.TokenError, IndentationError, SyntaxError,
        # ValueError, UnicodeDecodeError, RecursionError on pathological
        # nesting, etc. all show up in practice) — any parse failure is a
        # soft "no candidates from this file", not a generator crash.
        return []
    return out


# ---------------------------------------------------------------------------
# Char-offset <-> LM-token-index mapping (bisect over a single whole-file
# fast-tokenizer offset_mapping call — O(log n) per identifier occurrence).

def _tok_idx_at_char(starts: list[int], ends: list[int], c: int) -> int:
    """Index of the LM token covering (or immediately following) char c."""
    i = bisect.bisect_right(starts, c)
    if i > 0 and ends[i - 1] > c:
        return i - 1
    return min(i, len(starts) - 1)


# ---------------------------------------------------------------------------
# Per-file candidate extraction.

class Candidate:
    __slots__ = ("name", "prev_tok", "prev_span", "cur_tok", "cur_span",
                "distance")

    def __init__(self, name, prev_tok, prev_span, cur_tok, cur_span,
                distance):
        self.name = name
        self.prev_tok = prev_tok
        self.prev_span = prev_span
        self.cur_tok = cur_tok
        self.cur_span = cur_span
        self.distance = distance


def find_reuse_candidates(text: str, tok, min_name_len: int,
                          min_distance: int, max_distance: int
                          ) -> tuple[list[Candidate], list[int], list[int]]:
    """Return (candidates, starts, ends) for one file's text. `starts`/`ends`
    are the LM offset_mapping arrays (reused by the caller to cut the chunk
    without re-tokenizing)."""
    names = python_name_occurrences(text, min_name_len)
    if not names:
        return [], [], []
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    if not offs:
        return [], [], []
    starts = [a for a, b in offs]
    ends = [b for a, b in offs]
    last: dict[str, tuple[int, int, int]] = {}   # name -> (tok_idx, c0, c1)
    cands: list[Candidate] = []
    for name, c0, c1 in names:
        tidx = _tok_idx_at_char(starts, ends, c0)
        prev = last.get(name)
        if prev is not None:
            p_tidx, p_c0, p_c1 = prev
            dist = tidx - p_tidx
            if min_distance <= dist <= max_distance:
                cands.append(Candidate(name, p_tidx, (p_c0, p_c1), tidx,
                                       (c0, c1), dist))
        last[name] = (tidx, c0, c1)
    return cands, starts, ends


# ---------------------------------------------------------------------------
# Chunk construction: cut a contiguous char window (from the ORIGINAL file
# text, not a re-decoded token slice — avoids any BPE roundtrip risk) that
# contains both occurrences plus a little natural lead/trail context.

def build_example(text: str, starts: list[int], ends: list[int],
                  cand: Candidate, rng: random.Random,
                  lead_margin_range: tuple[int, int],
                  trail_margin_range: tuple[int, int]) -> dict | None:
    n_tok = len(starts)
    lead = rng.randint(*lead_margin_range)
    trail = rng.randint(*trail_margin_range)
    start_tok = max(0, cand.prev_tok - lead)
    # end-of-answer token index: last LM token overlapping cur_span, via the
    # same bisect convention as _tok_idx_at_char but anchored on the END char.
    end_tok = _tok_idx_at_char(starts, ends, max(cand.cur_span[1] - 1,
                                                  cand.cur_span[0]))
    end_tok = min(n_tok - 1, end_tok + trail)
    chunk_c0 = starts[start_tok]
    chunk_c1 = ends[end_tok]
    if chunk_c0 >= cand.prev_span[0] or chunk_c1 <= cand.cur_span[1]:
        return None
    chunk_text = text[chunk_c0:chunk_c1]
    rel_answer = [cand.cur_span[0] - chunk_c0, cand.cur_span[1] - chunk_c0]
    rel_binding = [cand.prev_span[0] - chunk_c0, cand.prev_span[1] - chunk_c0]
    if not (0 <= rel_answer[0] < rel_answer[1] <= len(chunk_text)):
        return None
    if not (0 <= rel_binding[0] < rel_binding[1] <= len(chunk_text)):
        return None
    if chunk_text[rel_answer[0]:rel_answer[1]] != cand.name:
        return None
    if chunk_text[rel_binding[0]:rel_binding[1]] != cand.name:
        return None
    return {
        "text": chunk_text,
        "answer": cand.name,
        "answer_char_span": rel_answer,
        "binding_char_span": rel_binding,
        "approx_distance_tokens": int(cand.distance),
        "total_tokens": int(end_tok - start_tok + 1),
    }


# ---------------------------------------------------------------------------
# Streaming generation loop.

def _stream_codeparrot():
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train",
                      streaming=True)
    return iter(ds)


def generate(args) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)

    it = _stream_codeparrot()
    row_idx = -1
    seen_hashes: set = set()

    targets = [("train", args.out, args.n_train),
               ("heldout", args.out_heldout, args.n_heldout)]

    overall_stats = {"n_files_scanned": 0, "n_files_with_candidate": 0,
                      "row_range": {}}
    dist_hist_all: list[int] = []

    for split_name, out_path, n_target in targets:
        t0 = time.time()
        n_written = 0
        n_files_scanned = 0
        n_files_hit = 0
        dist_hist: list[int] = []
        row_start = row_idx + 1
        out_f = open(out_path, "w")
        try:
            while n_written < n_target:
                try:
                    ex = next(it)
                except StopIteration:
                    print(f"[{split_name}] stream exhausted at row {row_idx} "
                          f"with {n_written}/{n_target} written")
                    break
                row_idx += 1
                if PROBE_EVAL_ROW_LO <= row_idx <= PROBE_EVAL_ROW_HI:
                    continue  # reserved for probe_depdist_stratified_ce.py
                content = ex.get("content") or ""
                if not (args.min_chars <= len(content) <= args.max_chars):
                    continue
                h = ex.get("hash")
                if h is not None:
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                n_files_scanned += 1
                cands, starts, ends = find_reuse_candidates(
                    content, tok, args.min_name_len, args.min_distance,
                    args.max_distance)
                if not cands:
                    continue
                # Diversity: at most one candidate per distinct name per
                # file, then cap total picks per file.
                by_name: dict[str, list[Candidate]] = {}
                for c in cands:
                    by_name.setdefault(c.name, []).append(c)
                picked_names = list(by_name.keys())
                rng.shuffle(picked_names)
                picked_names = picked_names[:args.max_per_file]
                file_hit = False
                for name in picked_names:
                    c = rng.choice(by_name[name])
                    example = build_example(
                        content, starts, ends, c, rng,
                        (args.lead_margin_min, args.lead_margin_max),
                        (args.trail_margin_min, args.trail_margin_max))
                    if example is None:
                        continue
                    example.update({
                        "family": "natural_identifier_reuse",
                        "tier": "natural_reuse_recall",
                        "task_id": f"natural_reuse/{row_idx}/{n_written}",
                        "sample_idx": n_written,
                        "source_row": row_idx,
                        "repo_name": ex.get("repo_name", ""),
                        "path": ex.get("path", ""),
                        "score": 1.0,
                        "has_tests": False,
                    })
                    out_f.write(json.dumps(example) + "\n")
                    n_written += 1
                    dist_hist.append(example["approx_distance_tokens"])
                    file_hit = True
                    if n_written >= n_target:
                        break
                if file_hit:
                    n_files_hit += 1
                if n_files_scanned % args.report_every == 0:
                    el = time.time() - t0
                    print(f"[{split_name}] row={row_idx} scanned="
                          f"{n_files_scanned} hit_files={n_files_hit} "
                          f"written={n_written}/{n_target} "
                          f"hit_rate={n_files_hit/max(1,n_files_scanned):.3f} "
                          f"({el:.0f}s, {n_written/max(el,1e-6):.1f} ex/s)",
                          flush=True)
        finally:
            out_f.close()
        row_end = row_idx
        print(f"[{split_name}] DONE: {n_written} examples from "
              f"{n_files_scanned} scanned files ({n_files_hit} hit), "
              f"rows [{row_start}, {row_end}] -> {out_path}")
        overall_stats["row_range"][split_name] = [row_start, row_end]
        overall_stats["n_files_scanned"] += n_files_scanned
        overall_stats["n_files_with_candidate"] += n_files_hit
        overall_stats[f"{split_name}_n_examples"] = n_written
        overall_stats[f"{split_name}_n_files_scanned"] = n_files_scanned
        overall_stats[f"{split_name}_n_files_hit"] = n_files_hit
        dist_hist_all.extend(dist_hist)
        meta_path = str(pathlib.Path(out_path).with_suffix("")) + ".meta.json"
        with open(meta_path, "w") as mf:
            json.dump({
                "split": split_name,
                "source_dataset": "codeparrot/codeparrot-clean",
                "source_row_range": [row_start, row_end],
                "probe_eval_row_range_excluded":
                    [PROBE_EVAL_ROW_LO, PROBE_EVAL_ROW_HI],
                "n_examples": n_written,
                "n_files_scanned": n_files_scanned,
                "n_files_hit": n_files_hit,
                "params": vars(args),
                "distance_histogram_summary": _hist_summary(dist_hist),
            }, mf, indent=1)
        print(f"[{split_name}] wrote meta -> {meta_path}")

    print("\n=== overall distance distribution ===")
    print(_hist_summary(dist_hist_all))


def _hist_summary(dists: list[int]) -> dict:
    if not dists:
        return {}
    buckets = {"256-511": 0, "512-1023": 0, "1024-1535": 0, "1536+": 0}
    for d in dists:
        if d < 512:
            buckets["256-511"] += 1
        elif d < 1024:
            buckets["512-1023"] += 1
        elif d < 1536:
            buckets["1024-1535"] += 1
        else:
            buckets["1536+"] += 1
    s = sorted(dists)
    n = len(s)
    return {
        "n": n,
        "min": s[0], "max": s[-1],
        "mean": sum(s) / n,
        "median": s[n // 2],
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# Validation harness: run the REAL MixedSourceStream over a mini YAML that
# points only at this source, decode the emitted read_mask, and check it
# lands exactly on the intended identifier tokens.

def validate(args) -> None:
    from transformers import AutoTokenizer

    from experiments.data_mix import MixedSourceStream, load_sources_from_yaml

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    base_vocab = tok.vocab_size
    thinking_token_id = base_vocab

    sources = load_sources_from_yaml(args.config)
    print(f"Loaded {len(sources)} source(s) from {args.config}")
    for s in sources:
        print(f"  - {s.name} jsonl_path={s.jsonl_path} "
              f"text_field={s.text_field} answer_field={s.answer_field} "
              f"answer_span_field={s.answer_span_field} "
              f"mask_first_occurrence={s.mask_first_occurrence} "
              f"emit_read_mask={s.emit_read_mask}")

    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.block_size,
        thinking_token_id=thinking_token_id,
        think_burst_prob=0.0,           # isolate the mask behaviour first
        emit_read_mask=True,
    )
    it = iter(ds)
    n_chunks = args.n_chunks
    n_with_mask = 0
    n_mask_tokens = 0
    distances = []
    printed = 0
    for i in range(n_chunks):
        x, y, doc_ids, read_mask = next(it)
        ids = x.tolist()
        rm = read_mask.tolist()
        mask_positions = [p for p, m in enumerate(rm) if m == 1]
        if mask_positions:
            n_with_mask += 1
            n_mask_tokens += len(mask_positions)
        if printed < args.n_print and mask_positions:
            printed += 1
            # Group contiguous mask positions into runs (one run per
            # answer-token-span; mask marks predicting positions p, i.e. the
            # answer token itself is ids[p+1]).
            runs = []
            cur = [mask_positions[0]]
            for p in mask_positions[1:]:
                if p == cur[-1] + 1:
                    cur.append(p)
                else:
                    runs.append(cur)
                    cur = [p]
            runs.append(cur)
            doc = doc_ids.tolist()
            for run in runs:
                p0 = run[0]
                answer_ids = [ids[p + 1] for p in run]
                answer_text = tok.decode(answer_ids)
                # Trim the printed tail to the SAME document as the masked
                # position (MixedSourceStream packs several short documents
                # per 2048-tok chunk; without this the tail can visually
                # splice across an unrelated doc boundary — cosmetic only,
                # doc_ids/cu_seqlens already isolate the recurrent state
                # correctly at train time).
                this_doc = doc[p0]
                tail_start = p0
                while tail_start > 0 and doc[tail_start - 1] == this_doc \
                        and p0 - tail_start < 40:
                    tail_start -= 1
                tail_ids = [t for t in ids[tail_start:p0 + 1]
                           if t != thinking_token_id]
                tail_text = tok.decode(tail_ids)
                print(f"\n[chunk {i}] masked run at pos {run[0]}-{run[-1]} "
                      f"(len={len(run)}, distance-in-chunk n/a — see jsonl "
                      f"approx_distance_tokens)")
                print(f"  context tail: ...{tail_text!r}")
                print(f"  masked token(s) decode to: {answer_text!r}")
    print(f"\nProcessed {n_chunks} chunks: {n_with_mask} contained a "
          f"read_mask, {n_mask_tokens} total masked positions "
          f"({n_mask_tokens/max(1,n_chunks):.2f} masked tok/chunk avg).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    p.add_argument("--config", default="configs/natural_reuse_recall_mini.yaml")
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--n_chunks", type=int, default=200)
    p.add_argument("--n_print", type=int, default=5)

    p.add_argument("--out", default="data/natural_reuse_recall.jsonl")
    p.add_argument("--out_heldout",
                   default="data/natural_reuse_recall_heldout.jsonl")
    p.add_argument("--n_train", type=int, default=45000)
    p.add_argument("--n_heldout", type=int, default=2000)
    p.add_argument("--tokenizer", default=TOKENIZER_NAME)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--min_distance", type=int, default=256)
    p.add_argument("--max_distance", type=int, default=1536)
    p.add_argument("--min_name_len", type=int, default=4)
    p.add_argument("--max_per_file", type=int, default=3)
    p.add_argument("--min_chars", type=int, default=1200)
    p.add_argument("--max_chars", type=int, default=120000)
    p.add_argument("--lead_margin_min", type=int, default=20)
    p.add_argument("--lead_margin_max", type=int, default=120)
    p.add_argument("--trail_margin_min", type=int, default=10)
    p.add_argument("--trail_margin_max", type=int, default=100)
    p.add_argument("--report_every", type=int, default=2000)
    return p


def main():
    args = build_parser().parse_args()
    if args.validate:
        validate(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
