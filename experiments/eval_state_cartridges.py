"""State-cartridges eval harness (pre-registered — STATE_CARTRIDGES_PLAN_2026_07_19.md).

EVAL-ONLY. No training, no checkpoint modification. For each episode in the
frozen `data/repo_episodes` eval split, computes teacher-forced task-span CE
under several *ingestion* arms and reports the pre-registered retention metric.

Arms (per episode)
------------------
  sequential          full repo context prepended to the task (the ceiling; ==
                      eval_repo_adaptive.py's `real` arm — reused VERBATIM so
                      tokenization + span-CE placement are pinned equal to that
                      harness, hence comparable to the +0.246 incidental-lift
                      anchor).
  none                task only, zero state (the floor; == eval_repo_adaptive's
                      `none` arm, reused verbatim).
  cartridge@K         context split into K token-balanced, line-aligned
                      segments; each segment run INDEPENDENTLY from zero state;
                      the per-segment final DeltaNet recurrent states are
                      MEAN-MERGED (conv state taken from the LAST segment — the
                      validated choice in probe_state_algebra.py); the merged
                      state is injected as the task's initial recurrent state
                      and the task is run with NO context tokens.
  shuffled@K          arm cartridge@K but with each segment's LINES permuted
                      (token-count preserving, deterministic per-episode seed):
                      does the cartridge carry STRUCTURE or a token bag?

`--segments_per_repo K` is the swept parallelism axis (default 2 4 8); it is the
NUMBER of segments the context is split into. Segment size ~= total_ctx / K
(~2k tokens for a typical 8k context at K=4 — the plan's "~2k segment" ballpark).
`sequential`/`none` are computed once; `cartridge`/`shuffled` are computed per K.

Metrics (span-CE is the pre-registered primary; line-CE reported alongside)
--------------------------------------------------------------------------
  lift(X)     = CE(none) - CE(X)              (paired per episode; +ve = helps)
  retention@K = lift(cartridge@K) / lift(sequential)
  Sanity gate : lift(sequential) >= +0.15 span-CE   (else the run is VOID)
  Pre-registered bands (reported, NOT decided here — this is the instrument):
     retention >= 0.75  STRONG PASS       >= 0.60  PASS
     retention <  0.35  KILL              [0.35,0.60) inconclusive
  Structure check: lift(shuffled@K) should be < lift(cartridge@K) by a clear
     margin for a structural claim; a tie => token-bag prior.
  Bootstrap 95% CI on retention (1000 resamples over episodes).

The state-injection path (the load-bearing new machinery) reuses:
  * probe_state_algebra.py : snapshot_states / states_to_cache / conv handling.
  * meta_ttt_train.py      : _block_stack / _embed_chunk / _finalize_logits_at
                             (the exact prefill-mirroring block loop + forward
                             tail), and FLA's initial_state read
                             (fla/layers/delta_net.py: `recurrent_state =
                             last_state['recurrent_state']`, read in BOTH the
                             chunk and fused_recurrent modes -> works for task
                             windows of any length).
  * eval_repo_adaptive.py  : load_model / context_ids / task_line_span_tokens /
                             arm_ce (the sequential + none arms, verbatim).

Usage
-----
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \
      experiments/eval_state_cartridges.py \
      --ckpt checkpoints/production_lean_soup3.pt \
      --eval data/repo_episodes/eval.jsonl \
      --out results/state_cartridges.json

  # 3-episode plumbing smoke:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \
      experiments/eval_state_cartridges.py --ckpt checkpoints/production_lean_soup3.pt \
      --eval data/repo_episodes/eval.jsonl --limit 3 --out /tmp/sc_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Reused verbatim from the sibling harness (identical tokenization + arm CE).
from experiments.eval_repo_adaptive import (
    _autocast,
    _load_jsonl,
    arm_ce,
    context_ids,
    load_model,
    task_line_span_tokens,
)
from experiments.gen_repo_episodes import perline_ids, split_lines_nl
# The prefill-mirroring block loop + forward tail (state-carry validated).
from experiments.meta_ttt_train import (
    _block_stack,
    _embed_chunk,
    _finalize_logits_at,
)
# State snapshot / cache-wrap / conv handling (mutation-isolated copies).
from experiments.probe_state_algebra import (
    _clone_any,
    snapshot_states,
    states_to_cache,
)


# --------------------------------------------------------------------------- #
# Context line assembly + segmentation.
# --------------------------------------------------------------------------- #

def build_context_lines(episode: dict, tok) -> list[list[int]]:
    """Per-line token-id chunks for all context files EXCEPT the task file, in
    file order. Each element is one physical line's ids (`split_lines_nl` +
    `tok`). The FLATTENED concatenation is EXACTLY
    `eval_repo_adaptive.context_ids(episode, tok)` — so segmenting on these
    line chunks partitions the identical token stream (checked in the tests).
    Keeping line granularity is what lets the shuffled arm permute lines while
    preserving each segment's token multiset."""
    lines: list[list[int]] = []
    for f in episode["context_files"][:-1]:
        for ln in split_lines_nl(f["text"]):
            lines.append(tok(ln, add_special_tokens=False)["input_ids"])
    return lines


def segment_lines(lines: list[list[int]], K: int) -> list[list[list[int]]]:
    """Split the per-line chunk list into K contiguous, token-balanced segments
    at LINE boundaries. Cut points are placed at cumulative-token fractions
    total*j/K (j=1..K-1) snapped to the next line boundary, then clamped so every
    segment is non-empty. Covers all lines exactly once, in order.

    K is capped at len(lines) (cannot have more segments than lines); an empty
    context returns []. With total ~= 8000 and K=4 each segment is ~2000 tokens
    (the plan's ~2k sizing)."""
    n = len(lines)
    if n == 0:
        return []
    K = max(1, min(int(K), n))
    if K == 1:
        return [list(lines)]
    cum: list[int] = []
    s = 0
    for l in lines:
        s += len(l)
        cum.append(s)
    total = cum[-1] if cum[-1] > 0 else n     # all-empty-lines fallback: by count
    weights = cum if cum[-1] > 0 else list(range(1, n + 1))

    segs: list[list[list[int]]] = []
    prev = 0
    for j in range(1, K):
        target = total * j / K
        # first line index (exclusive cut) whose cumulative weight >= target
        cut = prev
        while cut < n and weights[cut] < target:
            cut += 1
        cut = cut + 1                              # include that line in this seg
        cut = max(cut, prev + 1)                   # segment must be non-empty
        # leave enough lines for the remaining (K-j) segments
        cut = min(cut, n - (K - j))
        segs.append(lines[prev:cut])
        prev = cut
    segs.append(lines[prev:])
    return segs


def shuffle_segment_lines(segs: list[list[list[int]]],
                          rng: random.Random) -> list[list[list[int]]]:
    """Permute the LINE order WITHIN each segment (token-count preserving — each
    segment's token multiset is unchanged; only line order is scrambled)."""
    out: list[list[list[int]]] = []
    for seg in segs:
        seg2 = list(seg)
        rng.shuffle(seg2)
        out.append(seg2)
    return out


def _episode_seed(base_seed: int, episode_key, K: int) -> int:
    """Deterministic, process-stable per-(episode, K) seed (plain sha256, NOT
    the builtin hash() which is salted per process)."""
    h = hashlib.sha256(f"{base_seed}:{episode_key}:{K}".encode()).hexdigest()
    return int(h[:16], 16)


# --------------------------------------------------------------------------- #
# K-way mean merge of recurrent states (conv from the last segment).
# --------------------------------------------------------------------------- #

def mean_merge_states(states_list: list[list[dict]],
                      conv_from: str = "last") -> list[dict]:
    """Mean-merge K per-layer state-dict lists (each from `snapshot_states`).

    ONLY `recurrent_state` is merged — elementwise arithmetic mean over the K
    segments, accumulated in fp32 then cast back. `conv_state` is a raw
    last-few-token convolution buffer (local, flushed within conv_size query
    steps) and is taken verbatim from the LAST segment (`conv_from="last"`, the
    probe's validated choice); attn/ffn states pass through from the last
    segment. Output tensors are fresh copies (mutation-isolated from inputs, so
    FLA's in-place state update during the task forward cannot corrupt them).

    Mean of K IDENTICAL states == that state; mean of differing states == exact
    arithmetic mean (both unit-tested)."""
    assert states_list, "mean_merge_states needs >= 1 state list"
    K = len(states_list)
    n_layers = len(states_list[0])
    for st in states_list:
        assert len(st) == n_layers, (len(st), n_layers)
    conv_src = {"last": states_list[-1], "first": states_list[0]}[conv_from] \
        if conv_from in ("last", "first") else states_list[-1]

    merged: list[dict] = []
    for L in range(n_layers):
        rs0 = states_list[0][L].get("recurrent_state")
        if rs0 is None:
            rec = None
        else:
            acc = None
            for st in states_list:
                rs = st[L]["recurrent_state"]
                acc = rs.float() if acc is None else acc + rs.float()
            rec = (acc / K).to(rs0.dtype)
        entry = {
            "recurrent_state": rec,
            "conv_state": _clone_any(conv_src[L].get("conv_state")),
            "attn_state": _clone_any(states_list[-1][L].get("attn_state")),
            "ffn_state": _clone_any(states_list[-1][L].get("ffn_state")),
        }
        merged.append(entry)
    return merged


# --------------------------------------------------------------------------- #
# Ingest one segment; run the task with an injected initial state.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ingest_segment(model, seg_ids: list[int], device: str,
                   bf16: bool) -> tuple[list[dict], int]:
    """Full prefill of one segment FROM ZERO STATE; returns (pristine per-layer
    states, n tokens). Mirrors probe_state_algebra.ingest but honours --no_bf16
    for precision parity with the eval's arm_ce."""
    t = torch.tensor([seg_ids], dtype=torch.long, device=device)
    with _autocast(device, bf16):
        cache, _ = model.prefill(t)
    return snapshot_states(cache["fla_cache"]), int(cache["seen"])


@torch.no_grad()
def arm_ce_injected(model, init_states: list[dict] | None, seen: int,
                    task_prefix_ids: list[int], task_line_ids: list[int],
                    span_local_idx: list[int], device: str, bf16: bool) -> dict:
    """Teacher-forced CE on the task_line tokens, running the task window
    [task_prefix + task_line] with `init_states` injected as the model's initial
    recurrent state (and NO context tokens). CE math + span placement mirror
    eval_repo_adaptive.arm_ce EXACTLY (same predictor positions [P-1, P+L),
    same span index selection); the ONLY difference is the initial state.

    `init_states=None` -> empty cache -> byte-identical to a plain forward
    (validated by the none-arm test)."""
    from fla.models.utils import Cache as FLACache

    P = len(task_prefix_ids)
    L = len(task_line_ids)
    if L == 0 or P < 1:
        return {"line_ce": None, "span_ce": None, "n_line": 0, "n_span": 0}
    full = task_prefix_ids + task_line_ids
    pred_pos = list(range(P - 1, P + L - 1))          # predictors for task_line
    ids_t = torch.tensor([full], device=device, dtype=torch.long)
    tgt = torch.tensor(full[P:P + L], device=device, dtype=torch.long)

    if init_states is None:
        fla = FLACache(seen_tokens=0)
    else:
        fla = states_to_cache(init_states, seen)["fla_cache"]

    with _autocast(device, bf16):
        x = _embed_chunk(model, ids_t, pos_offset=0)      # pos-free base -> inert
        hidden = _block_stack(model, x, fla)              # threads injected state
        logits = _finalize_logits_at(model, hidden, ids_t, pred_pos)   # (L, V)

    ce_tok = F.cross_entropy(logits.float(), tgt, reduction="none")    # (L,)
    line_ce = float(ce_tok.mean().item())
    if span_local_idx:
        span_ce = float(ce_tok[torch.tensor(span_local_idx)].mean().item())
        n_span = len(span_local_idx)
    else:
        span_ce = None
        n_span = 0
    return {"line_ce": line_ce, "span_ce": span_ce, "per_tok": ce_tok.tolist(),
            "n_line": L, "n_span": n_span}


@torch.no_grad()
def cartridge_merged_state(model, segs: list[list[list[int]]], device: str,
                           bf16: bool, conv_from: str = "last"
                           ) -> tuple[list[dict] | None, int, int]:
    """Ingest every segment independently from zero state, mean-merge the final
    recurrent states. Returns (merged_states|None, total_seen, n_used_segments).
    Empty segments are skipped; if none remain merged is None (-> none arm)."""
    seg_states: list[list[dict]] = []
    total_seen = 0
    for seg in segs:
        seg_ids = [tid for ln in seg for tid in ln]
        if not seg_ids:
            continue
        st, seen = ingest_segment(model, seg_ids, device, bf16)
        seg_states.append(st)
        total_seen += seen
    if not seg_states:
        return None, 0, 0
    merged = mean_merge_states(seg_states, conv_from=conv_from)
    return merged, total_seen, len(seg_states)


# --------------------------------------------------------------------------- #
# Per-episode eval.
# --------------------------------------------------------------------------- #

_ARM_FIELDS = ("line_ce", "span_ce", "n_line", "n_span")


def _pick(d: dict) -> dict:
    return {k: d.get(k) for k in _ARM_FIELDS}


def eval_episode(model, episode: dict, tok, device: str, bf16: bool,
                 seg_counts: list[int], seed: int, ep_index: int,
                 conv_from: str = "last") -> dict:
    task_prefix_ids = perline_ids(episode["task_prefix"], tok)
    line_ids, span_idx = task_line_span_tokens(episode, tok)
    ctx = context_ids(episode, tok)

    # sequential (== eval_repo_adaptive real arm) and none (== its none arm).
    seq = arm_ce(model, ctx + task_prefix_ids, line_ids, span_idx, device, bf16)
    none = arm_ce(model, task_prefix_ids, line_ids, span_idx, device, bf16)

    lines = build_context_lines(episode, tok)
    ep_key = episode.get("episode_id", f"idx{ep_index}")

    arms = {"sequential": _pick(seq), "none": _pick(none)}
    n_segments: dict[str, int] = {}
    for K in seg_counts:
        segs = segment_lines(lines, K)
        n_segments[str(K)] = len(segs)
        # cartridge@K
        merged, seen, _ = cartridge_merged_state(model, segs, device, bf16,
                                                 conv_from)
        arms[f"cartridge@{K}"] = _pick(arm_ce_injected(
            model, merged, seen, task_prefix_ids, line_ids, span_idx,
            device, bf16))
        # shuffled@K (deterministic per-episode line permutation)
        rng = random.Random(_episode_seed(seed, ep_key, K))
        segs_sh = shuffle_segment_lines(segs, rng)
        merged_sh, seen_sh, _ = cartridge_merged_state(model, segs_sh, device,
                                                       bf16, conv_from)
        arms[f"shuffled@{K}"] = _pick(arm_ce_injected(
            model, merged_sh, seen_sh, task_prefix_ids, line_ids, span_idx,
            device, bf16))

    return {
        "episode_id": episode.get("episode_id"),
        "repo_name": episode.get("repo_name"),
        "identifier": episode.get("link", {}).get("identifier"),
        "bucket": episode.get("bucket"),
        "n_ctx_tokens": episode.get("n_ctx_tokens"),
        "n_ctx_tokens_actual": len(ctx),
        "n_segments": n_segments,
        "arms": arms,
    }


# --------------------------------------------------------------------------- #
# Aggregation: lifts, retention, bootstrap CI.
# --------------------------------------------------------------------------- #

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.fmean(xs)) if xs else None


def _arm_mean(results: list[dict], arm: str, field: str):
    return _mean([r["arms"].get(arm, {}).get(field) for r in results])


def _paired_lift(results: list[dict], arm: str, field: str):
    """mean_ep( CE(none) - CE(arm) ), paired over episodes where both present."""
    vals = []
    for r in results:
        none = r["arms"]["none"].get(field)
        x = r["arms"].get(arm, {}).get(field)
        if none is not None and x is not None:
            vals.append(none - x)
    return (_mean(vals), len(vals))


def _retention_rows(results: list[dict], K: int, field: str):
    """(none, seq, cart) triples over episodes where none/sequential/cartridge@K
    all have a valid `field` — the paired set retention is computed on."""
    cart = f"cartridge@{K}"
    rows = []
    for r in results:
        a = r["arms"]
        vals = [a["none"].get(field), a["sequential"].get(field),
                a.get(cart, {}).get(field)]
        if all(v is not None for v in vals):
            rows.append(tuple(vals))
    return rows


def _retention(rows) -> float | None:
    if not rows:
        return None
    none = statistics.fmean(r[0] for r in rows)
    seq = statistics.fmean(r[1] for r in rows)
    cart = statistics.fmean(r[2] for r in rows)
    denom = none - seq
    return (none - cart) / denom if denom != 0 else None


def _bootstrap_retention_ci(rows, n_boot: int, rng: np.random.Generator):
    if len(rows) < 2:
        return (None, None)
    arr = np.asarray(rows, dtype=np.float64)              # (N, 3)
    N = arr.shape[0]
    idx = rng.integers(0, N, size=(n_boot, N))
    boot = arr[idx]                                        # (n_boot, N, 3)
    m = boot.mean(axis=1)                                  # (n_boot, 3)
    denom = m[:, 0] - m[:, 1]
    good = denom != 0
    ret = (m[good, 0] - m[good, 2]) / denom[good]
    if ret.size == 0:
        return (None, None)
    lo, hi = np.percentile(ret, [2.5, 97.5])
    return (float(lo), float(hi))


def aggregate(results: list[dict], seg_counts: list[int], n_boot: int,
              seed: int) -> dict:
    arms = ["sequential", "none"]
    for K in seg_counts:
        arms += [f"cartridge@{K}", f"shuffled@{K}"]

    ce = {}
    for arm in arms:
        ce[arm] = {"line_ce": _arm_mean(results, arm, "line_ce"),
                   "span_ce": _arm_mean(results, arm, "span_ce")}

    lifts = {}
    for arm in arms:
        if arm == "none":
            continue
        for field in ("line_ce", "span_ce"):
            m, n = _paired_lift(results, arm, field)
            lifts.setdefault(arm, {})[field] = {"lift": m, "n": n}

    rng = np.random.default_rng(seed)
    retention = {}
    for K in seg_counts:
        retention[str(K)] = {}
        for field in ("span_ce", "line_ce"):
            rows = _retention_rows(results, K, field)
            r = _retention(rows)
            lo, hi = _bootstrap_retention_ci(rows, n_boot, rng)
            retention[str(K)][field] = {
                "retention": r, "n": len(rows),
                "ci95": [lo, hi],
                "lift_sequential": (statistics.fmean(x[0] - x[1] for x in rows)
                                    if rows else None),
                "lift_cartridge": (statistics.fmean(x[0] - x[2] for x in rows)
                                   if rows else None),
            }

    seq_lift_span = lifts.get("sequential", {}).get("span_ce", {}).get("lift")
    seq_lift_line = lifts.get("sequential", {}).get("line_ce", {}).get("lift")
    sanity = {
        "lift_sequential_span": seq_lift_span,
        "lift_sequential_line": seq_lift_line,
        "gate_threshold": 0.15,
        "sanity_gate_pass": (seq_lift_span is not None
                             and seq_lift_span >= 0.15),
        "note": "span-CE gate per the pre-registered plan; run VOID if False.",
    }

    return {"ce": ce, "lifts": lifts, "retention": retention,
            "sanity": sanity, "n_episodes": len(results),
            "seg_counts": list(seg_counts),
            "bands": {"strong_pass": 0.75, "pass": 0.60, "kill": 0.35}}


# --------------------------------------------------------------------------- #
# Readable summary.
# --------------------------------------------------------------------------- #

def _f(x, w=7, p=4):
    return f"{x:.{p}f}".rjust(w) if isinstance(x, (int, float)) else " n/a ".rjust(w)


def _band(r):
    if r is None:
        return "n/a"
    if r >= 0.75:
        return "STRONG-PASS band"
    if r >= 0.60:
        return "PASS band"
    if r < 0.35:
        return "KILL band"
    return "inconclusive band"


def print_summary(agg: dict):
    seg_counts = agg["seg_counts"]
    ce = agg["ce"]
    print("\n=== task-CE per arm (lower = better) ===")
    print(f"{'arm':<14} {'span_ce':>9} {'line_ce':>9}")
    print("-" * 34)
    for arm in ["sequential", "none"] + \
            [f"cartridge@{K}" for K in seg_counts] + \
            [f"shuffled@{K}" for K in seg_counts]:
        d = ce.get(arm, {})
        print(f"{arm:<14} {_f(d.get('span_ce'), 9):>9} {_f(d.get('line_ce'), 9):>9}")

    s = agg["sanity"]
    print("\n=== sanity gate (pre-registered: lift(sequential) span-CE >= 0.15) ===")
    print(f"  lift(sequential) span = {_f(s['lift_sequential_span'])}   "
          f"line = {_f(s['lift_sequential_line'])}   "
          f"gate_pass = {s['sanity_gate_pass']}"
          + ("" if s['sanity_gate_pass'] else "   <-- RUN VOID, not a result"))

    print("\n=== retention = lift(cartridge@K) / lift(sequential)  [SPAN-CE, primary] ===")
    hdr = (f"{'K':>3} {'n':>4} | {'lift_seq':>9} {'lift_cart':>9} | "
           f"{'retention':>9} {'ci95_lo':>8} {'ci95_hi':>8} | band")
    print(hdr)
    print("-" * len(hdr))
    for K in seg_counts:
        d = agg["retention"][str(K)]["span_ce"]
        lo, hi = d["ci95"]
        print(f"{K:>3} {d['n']:>4} | {_f(d['lift_sequential'], 9):>9} "
              f"{_f(d['lift_cartridge'], 9):>9} | {_f(d['retention'], 9):>9} "
              f"{_f(lo, 8):>8} {_f(hi, 8):>8} | {_band(d['retention'])}")

    print("\n=== retention [LINE-CE, secondary] ===")
    print(hdr)
    print("-" * len(hdr))
    for K in seg_counts:
        d = agg["retention"][str(K)]["line_ce"]
        lo, hi = d["ci95"]
        print(f"{K:>3} {d['n']:>4} | {_f(d['lift_sequential'], 9):>9} "
              f"{_f(d['lift_cartridge'], 9):>9} | {_f(d['retention'], 9):>9} "
              f"{_f(lo, 8):>8} {_f(hi, 8):>8} | {_band(d['retention'])}")

    print("\n=== structure check: lift(shuffled@K) vs lift(cartridge@K) [span-CE] ===")
    lf = agg["lifts"]
    print(f"{'K':>3} | {'lift_cart':>9} {'lift_shuf':>9} {'cart-shuf':>9}")
    print("-" * 40)
    for K in seg_counts:
        lc = lf.get(f"cartridge@{K}", {}).get("span_ce", {}).get("lift")
        ls = lf.get(f"shuffled@{K}", {}).get("span_ce", {}).get("lift")
        diff = (lc - ls) if (lc is not None and ls is not None) else None
        print(f"{K:>3} | {_f(lc, 9):>9} {_f(ls, 9):>9} {_f(diff, 9):>9}")
    print("\n(bands are the pre-registered thresholds, printed for reference; "
          "this harness is the instrument, not the verdict.)")


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/production_lean_soup3.pt")
    ap.add_argument("--eval", default="data/repo_episodes/eval.jsonl",
                    help="frozen eval-split episodes (jsonl)")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--segments_per_repo", type=int, nargs="+",
                    default=[2, 4, 8],
                    help="number(s) of segments the context is split into "
                         "(the swept parallelism axis)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N episodes (smoke/debug)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--conv_from", default="last", choices=["last", "first"],
                    help="which segment supplies the merged conv state "
                         "(probe's validated choice: last)")
    ap.add_argument("--max_ctx_tokens", type=int, default=32000,
                    help="skip episodes whose context exceeds this (safety)")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    episodes = _load_jsonl(args.eval)
    if args.limit is not None:
        episodes = episodes[:args.limit]
    bf16 = not args.no_bf16
    seg_counts = sorted(set(int(k) for k in args.segments_per_repo))

    t0 = time.time()
    model, cfg, tok = load_model(args.ckpt, args.device)
    # The cartridge/shuffled arms inject the merged state and run the task
    # window from absolute position 0, while the sequential arm runs the task
    # at position len(context). These agree ONLY for a position-free base
    # (max_T==0, the production lean DeltaNet). For a pos-embedded (max_T>0)
    # checkpoint the arms would use inconsistent positions AND the recurrent-
    # state injection equivalence would not hold — silently wrong retention.
    # Guard it (fixes/guards are defaults, not silent assumptions).
    max_T = int(getattr(model, "max_T", 0) or 0)
    assert max_T == 0, (
        f"eval_state_cartridges requires a position-free base (max_T==0); "
        f"this checkpoint has max_T={max_T}. The injected-state arms assume "
        f"positional invariance (see _embed_chunk pos_offset).")
    print(f"[loaded] {args.ckpt}  n_layers={cfg.get('n_layers')} "
          f"d_model={cfg.get('d_model')} max_T={max_T} "
          f"use_memory={getattr(model, 'use_memory', False)} "
          f"film_bypass={getattr(model, '_film_bypass', None)} "
          f"segments_per_repo={seg_counts} bf16={bf16}", flush=True)

    results = []
    n_skipped = 0
    for i, ep in enumerate(episodes):
        # +64 headroom for the task window, matching eval_repo_adaptive's skip.
        if int(ep.get("n_ctx_tokens", 0)) + 64 > args.max_ctx_tokens:
            n_skipped += 1
            print(f"  [skip {i}] n_ctx {ep.get('n_ctx_tokens')} > "
                  f"max_ctx_tokens {args.max_ctx_tokens}", file=sys.stderr)
            continue
        r = eval_episode(model, ep, tok, args.device, bf16, seg_counts,
                         args.seed, i, conv_from=args.conv_from)
        results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(episodes)}  ({time.time() - t0:.0f}s)",
                  file=sys.stderr)

    agg = aggregate(results, seg_counts, args.n_bootstrap, args.seed)
    agg["meta"] = {
        "ckpt": args.ckpt, "eval": args.eval, "device": args.device,
        "bf16": bf16, "conv_from": args.conv_from, "seed": args.seed,
        "n_bootstrap": args.n_bootstrap, "n_episodes": len(results),
        "n_skipped": n_skipped, "elapsed_s": round(time.time() - t0, 1),
    }
    print_summary(agg)
    print(f"\nevaluated {len(results)} episodes "
          f"({n_skipped} skipped over max_ctx_tokens) in "
          f"{time.time() - t0:.0f}s")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"aggregate": agg, "per_episode": results}, f, indent=2,
                      default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
