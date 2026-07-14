"""State Algebra probe — do DeltaNet recurrent states from DISJOINT content
merge usefully? (ideas_2026_07_13/09_wildcards.md idea 1, Tier-0 probe.)

The delta rule is Widrow-Hoff: the per-layer recurrent state is (approximately)
the running least-squares solution of an associative key->value regression.
Sums of rank-1 updates from DISJOINT key subspaces should be nearly additive,
so merging two independently-ingested states might approximate sequential
ingestion (State Soup, arXiv 2406.08423, showed this for Mamba-2.8b). But the
delta rule is interference-aware — each write first SUBTRACTS the state's
current prediction for that key — so updates don't commute and merging may
produce off-manifold states, especially under key overlap (the LASP inter-chunk
correction term is exactly what naive merging discards).

Task (minimal inline saturating-multibind, the headroom regime from
gen_multibind_recall.py): context A binds keys 1..N (``name=value`` lines,
4-digit values), context B binds keys N+1..2N. Arms per trial:

  seq       sequential A+B ingest (ceiling)
  mean      elementwise mean of per-layer final recurrent states of A and B
  sum       elementwise sum
  normmax   per-head select: for each (layer, head), take the state slice
            from whichever context has the larger Frobenius norm there
  b_only    context B's state alone (floor for A-keys)

Recall metric: TEACHER-FORCED accuracy, first-occurrence protocol — from the
arm's state, feed the query prefix ``name=`` via `forward_step` and greedily
argmax-check each value token (true tokens are fed forward, correctness =
ALL value tokens argmax-correct).

Overlap arm (interference probe): contexts A' and B' share
``--overlap_frac`` (default 25%) of their keys with DIFFERENT values; under
sequential ingest the LATER (B') value wins, so shared keys are scored
against the B' value; we also report how often the merged state predicts
the A' value instead (first-token diagnostic).

PRE-REGISTERED READ:
  * merged recall within ~10pp of sequential on DISJOINT keys => delta-rule
    states are approximately additive — unlocks parallel prefill across
    shards/GPUs, zero-training state "cartridges", and state-mixing search
    operators (feeds the meta-TTT program directly).
  * merged recall ~= the b_only floor on A-keys (catastrophic) => states are
    order-entangled and unmixable — a hard constraint: state must always be
    built sequentially, which constrains sleep-consolidation / repo-ingestion
    / meta-TTT design.
  * overlap arm: graceful degradation vs disjoint = interference is local to
    the shared keys; global collapse = interference is not key-local.

The WM buffer is EXCLUDED (lean no-WM ckpts; `use_memory` is forced off like
the other exec-trace evals) — this probes the raw FLA recurrent state only.

Usage (ready to fire; ~minutes on one 5090):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/probe_state_algebra.py \\
      --ckpt checkpoints/feature_pilot_A.pt \\
      --n_bindings 32 --n_trials 5 --seed 0 \\
      --out runs/probe_state_algebra.json

  # smoke (8 bindings, 1 trial):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/probe_state_algebra.py --ckpt checkpoints/feature_pilot_A.pt \\
      --smoke
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

MERGE_ARMS = ("seq", "mean", "sum", "normmax", "b_only")


# --------------------------------------------------------------------------- #
# Wilson CI (shared with probe_latent_exposure_bias; kept dependency-free).
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
# Binding-context construction (deterministic in the rng).
# --------------------------------------------------------------------------- #

def _mk_names(idxs) -> list[str]:
    # Trained multibind naming (gen_multibind_recall.py): v0, v1, ...
    return [f"v{i}" for i in idxs]


# The TRAINED multibind rendering (gen_multibind_recall.py::_gen_multibind).
# The first probe run used an ad-hoc `vaNNN=DDDD` format and read a 0.000
# SEQUENTIAL ceiling — the probe-format trap (same ckpt, wrong rendering →
# recall reads 0; cf. the WM-recall-probe post-mortem). Everything below
# matches the trained format exactly: program wrapper, `vN = DDDD` lines,
# `print(vN)` close, `Answer: NNNN` answer cue.
_WRAPPER_HEAD = ("Run the following Python program and report what it "
                 "prints.\n\n```python\n")


def build_binding_contexts(n_bindings: int, rng: random.Random,
                           overlap_frac: float = 0.0) -> dict:
    """Build two binding contexts of `n_bindings` NAME=value lines each.

    overlap_frac == 0.0 -> DISJOINT keys: A binds names 0..N-1, B binds
    names N..2N-1.
    overlap_frac > 0.0  -> round(overlap_frac*N) of B's keys are shared with
    A (same name, DIFFERENT value); the rest of B's keys are fresh.

    Returns dict with `a` / `b`: list of (name, value_str) in shuffled line
    order, plus `shared`: list of (name, a_value_str, b_value_str).
    """
    n_shared = int(round(overlap_frac * n_bindings))
    names_a = _mk_names(range(n_bindings))
    fresh_b = _mk_names(range(n_bindings, 2 * n_bindings - n_shared))
    shared_names = names_a[:n_shared]

    def _val() -> str:
        return str(rng.randint(1000, 9999))

    a_vals = {nm: _val() for nm in names_a}
    b_vals = {nm: _val() for nm in fresh_b}
    shared_b_vals = {}
    for nm in shared_names:
        v = _val()
        while v == a_vals[nm]:               # shared keys must DIFFER
            v = _val()
        shared_b_vals[nm] = v

    a_items = [(nm, a_vals[nm]) for nm in names_a]
    b_items = [(nm, b_vals[nm]) for nm in fresh_b] + \
              [(nm, shared_b_vals[nm]) for nm in shared_names]
    rng.shuffle(a_items)
    rng.shuffle(b_items)
    return {
        "a": a_items,
        "b": b_items,
        "shared": [(nm, a_vals[nm], shared_b_vals[nm]) for nm in shared_names],
    }


def line_and_query_ids(tok, name: str, value: str) -> tuple[list[int],
                                                            list[int],
                                                            list[int]]:
    """Tokenize one binding line (TRAINED format `vN = DDDD\\n`) plus the
    trained query/answer rendering: the query CLOSES the program with
    `print(vN)\\n```\\n\\nAnswer:` and the value tokens are the teacher-forced
    answer digits. The query/value split is chosen so query_ids is an exact
    token prefix of query+value (two candidate splits tried — BPE may merge
    the space after 'Answer:' into the first digit token).

    Returns (line_ids, query_ids, value_ids)."""
    line = f"{name} = {value}\n"
    l_ids = tok.encode(line, add_special_tokens=False)
    # First-occurrence answer cue in the TRAINED completion is the prose
    # restatement ("`vN` is set to NNNN"), which precedes "Answer:".
    stem = f"print({name})\n```\nThe program assigns variables. `{name}` is set to"
    tail = f"{stem} {value},\n"
    full_ids = tok.encode(tail, add_special_tokens=False)
    nl_ids = tok.encode(",\n", add_special_tokens=False)
    for query in (f"{stem} ", stem):
        q_ids = tok.encode(query, add_special_tokens=False)
        if full_ids[: len(q_ids)] == q_ids:
            v_ids = full_ids[len(q_ids):]
            if v_ids[-len(nl_ids):] == nl_ids:
                v_ids = v_ids[: -len(nl_ids)]
            if v_ids:
                return l_ids, q_ids, v_ids
    raise ValueError(
        f"no query split is a token prefix of {tail!r} for this tokenizer")


def build_trial(tok, n_bindings: int, rng: random.Random,
                overlap_frac: float = 0.0) -> dict:
    """Token-level trial: context id streams + query records.

    Query record: dict(name, query_ids, value_ids, group, alt_value_ids).
    group in {"A", "B", "shared"}; `value_ids` is the SCORED value (for
    shared keys: the B/later value — sequential-ingest semantics);
    `alt_value_ids` (shared only) is the A/earlier value for the
    interference diagnostic."""
    ctx = build_binding_contexts(n_bindings, rng, overlap_frac)
    head_ids = tok.encode(_WRAPPER_HEAD, add_special_tokens=False)
    ids_a: list[int] = list(head_ids)
    ids_b: list[int] = list(head_ids)
    queries: list[dict] = []
    shared_names = {nm for nm, _, _ in ctx["shared"]}

    per_name: dict[str, dict] = {}
    for nm, val in ctx["a"]:
        l_ids, q_ids, v_ids = line_and_query_ids(tok, nm, val)
        ids_a.extend(l_ids)
        per_name[nm] = {"query_ids": q_ids, "a_value_ids": v_ids}
    for nm, val in ctx["b"]:
        l_ids, q_ids, v_ids = line_and_query_ids(tok, nm, val)
        ids_b.extend(l_ids)
        per_name.setdefault(nm, {"query_ids": q_ids})["b_value_ids"] = v_ids

    for nm, rec in per_name.items():
        if nm in shared_names:
            queries.append({"name": nm, "query_ids": rec["query_ids"],
                            "value_ids": rec["b_value_ids"],   # later wins
                            "alt_value_ids": rec["a_value_ids"],
                            "group": "shared"})
        elif "a_value_ids" in rec:
            queries.append({"name": nm, "query_ids": rec["query_ids"],
                            "value_ids": rec["a_value_ids"],
                            "alt_value_ids": None, "group": "A"})
        else:
            queries.append({"name": nm, "query_ids": rec["query_ids"],
                            "value_ids": rec["b_value_ids"],
                            "alt_value_ids": None, "group": "B"})
    # seq = ONE well-formed program (wrapper + A lines + B lines); the merged
    # arms each ingest their own full wrapped doc (that's what shard-merging
    # means in practice).
    n_head = len(head_ids)
    ids_seq = list(head_ids) + ids_a[n_head:] + ids_b[n_head:]
    return {"ids_a": ids_a, "ids_b": ids_b, "ids_seq": ids_seq,
            "queries": queries}


# --------------------------------------------------------------------------- #
# State extraction / merge ops (pure tensor math — CPU-testable).
#
# Cache structure (surveyed from TinyLM.prefill + fla.models.utils):
#   TinyLM.prefill(ids) -> (cache, last_logits) with
#     cache = {"fla_cache": fla.models.utils.Cache, "seen": int,
#              "lagged_sources": None (film_bypass), "wm_buf": None (no WM),
#              "think_run_len": None}
#   fla_cache[layer] -> dict(recurrent_state=Tensor (B, H, d_k, d_v),
#                            attn_state=None, conv_state=tuple of 3 Tensors
#                            (the q/k/v ShortConvolution buffers), ffn_state=None)
# --------------------------------------------------------------------------- #

_STATE_KEYS = ("recurrent_state", "attn_state", "conv_state", "ffn_state")


def _clone_any(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, (tuple, list)):
        return tuple(_clone_any(t) for t in x)
    return x


def snapshot_states(fla_cache) -> list[dict]:
    """Deep-copied per-layer state dicts (pristine — later mutation of any
    derived cache cannot touch these)."""
    out = []
    for i in range(len(fla_cache)):
        entry = fla_cache[i]
        out.append({k: _clone_any(entry.get(k)) for k in _STATE_KEYS})
    return out


def _map_pair(a, b, fn):
    """Apply fn(ta, tb) across matching tensors in tensor/tuple/list states."""
    if a is None or b is None:
        return _clone_any(b if a is None else a)
    if torch.is_tensor(a):
        return fn(a, b)
    if isinstance(a, (tuple, list)):
        return tuple(_map_pair(x, y, fn) for x, y in zip(a, b))
    raise TypeError(f"unsupported state type {type(a)}")


def _normmax_select(ta: torch.Tensor, tb: torch.Tensor) -> torch.Tensor:
    """Per-head select: with state shape (B, H, ...), take the (B, H) slice
    from whichever tensor has the larger Frobenius norm over the trailing
    dims. Falls back to whole-tensor select for dim < 3."""
    if ta.dim() < 3:
        na = ta.float().norm()
        nb = tb.float().norm()
        return (ta if na >= nb else tb).clone()
    dims = tuple(range(2, ta.dim()))
    na = ta.float().norm(dim=dims)             # (B, H)
    nb = tb.float().norm(dim=dims)
    mask = (na >= nb).view(*na.shape, *([1] * (ta.dim() - 2)))
    return torch.where(mask, ta, tb)


def merge_state_dicts(states_a: list[dict], states_b: list[dict],
                      op: str, conv_from: str = "b") -> list[dict]:
    """Merge two per-layer state-dict lists (each from `snapshot_states`).

    `op`: "mean" | "sum" | "normmax" | "b_only". Only `recurrent_state` is
    merged; `conv_state` (a raw last-few-token convolution buffer — local,
    flushed within conv_size steps of query feeding) is taken verbatim from
    `conv_from` ("a"|"b"|"mean"). attn/ffn states pass through from B.
    Output tensors are fresh copies (mutation-isolated from inputs)."""
    assert len(states_a) == len(states_b), (len(states_a), len(states_b))
    ops = {
        "mean": lambda x, y: (x.float() + y.float()).div_(2.0).to(x.dtype),
        "sum": lambda x, y: (x.float() + y.float()).to(x.dtype),
        "normmax": _normmax_select,
        "b_only": lambda x, y: y.clone(),
    }
    if op not in ops:
        raise ValueError(f"unknown merge op {op!r}")
    fn = ops[op]
    merged = []
    for sa, sb in zip(states_a, states_b):
        entry = {}
        entry["recurrent_state"] = _map_pair(sa.get("recurrent_state"),
                                             sb.get("recurrent_state"), fn)
        if conv_from == "mean":
            entry["conv_state"] = _map_pair(
                sa.get("conv_state"), sb.get("conv_state"),
                lambda x, y: (x.float() + y.float()).div_(2.0).to(x.dtype))
        else:
            src = sa if conv_from == "a" else sb
            entry["conv_state"] = _clone_any(src.get("conv_state"))
        entry["attn_state"] = _clone_any(sb.get("attn_state"))
        entry["ffn_state"] = _clone_any(sb.get("ffn_state"))
        merged.append(entry)
    return merged


def replicate_states(states: list[dict], n: int) -> list[dict]:
    """Repeat every tensor's batch dim (dim 0, must be 1) to `n` — fresh
    contiguous copies, so decoding one replica never touches another (FLA's
    fused_recurrent step updates the cache state tensors)."""

    def _rep(t):
        assert t.shape[0] == 1, f"expected batch 1, got {tuple(t.shape)}"
        return t.repeat(n, *([1] * (t.dim() - 1)))

    out = []
    for s in states:
        entry = {}
        for k in _STATE_KEYS:
            v = s.get(k)
            if v is None:
                entry[k] = None
            elif torch.is_tensor(v):
                entry[k] = _rep(v)
            else:
                entry[k] = tuple(_rep(t) for t in v)
        out.append(entry)
    return out


def states_to_cache(states: list[dict], seen: int) -> dict:
    """Wrap per-layer state dicts back into the TinyLM incremental-decode
    cache contract (the dict `prefill` returns). Uses `Cache.update` (not
    `from_legacy_cache`) so both transformers layer-replication paths work."""
    from fla.models.utils import Cache as FLACache
    fla = FLACache(seen_tokens=int(seen))
    for i, s in enumerate(states):
        fla.update(recurrent_state=s.get("recurrent_state"),
                   attn_state=s.get("attn_state"),
                   conv_state=s.get("conv_state"),
                   ffn_state=s.get("ffn_state"),
                   layer_idx=i, offset=int(seen))
    return {"fla_cache": fla, "seen": int(seen), "lagged_sources": None,
            "wm_buf": None, "think_run_len": None}


# --------------------------------------------------------------------------- #
# Query running (teacher-forced, batched over same-shape queries).
# --------------------------------------------------------------------------- #

def _autocast(device):
    dt = device.type if hasattr(device, "type") else str(device).split(":")[0]
    if dt == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def run_queries(model, base_states: list[dict], seen: int, queries: list[dict],
                device) -> list[dict]:
    """Teacher-forced recall for every query, each from a FRESH replica of
    `base_states` (queries never contaminate each other or the source).

    Batches queries with identical (len(query_ids), len(value_ids)) shape;
    for each: feed query tokens via forward_step, then at each value
    position compare argmax to the true token and feed the TRUE token
    (teacher forcing). Returns per-query dicts:
      {name, group, correct (all value tokens argmax-correct),
       first_tok_pred, per_tok_correct}."""
    dev = torch.device(device)
    groups: dict[tuple[int, int], list[dict]] = {}
    for q in queries:
        groups.setdefault((len(q["query_ids"]), len(q["value_ids"])),
                          []).append(q)

    results = []
    for (lq, lv), qs in sorted(groups.items()):
        G = len(qs)
        cache = states_to_cache(replicate_states(base_states, G), seen)
        q_ids = torch.tensor([q["query_ids"] for q in qs], dtype=torch.long,
                             device=dev)                       # (G, lq)
        v_ids = torch.tensor([q["value_ids"] for q in qs], dtype=torch.long,
                             device=dev)                       # (G, lv)
        with _autocast(dev):
            logits = None
            for t in range(lq):
                logits, cache = model.forward_step(q_ids[:, t:t + 1], cache)
            per_tok = torch.zeros(G, lv, dtype=torch.bool, device=dev)
            first_pred = None
            for t in range(lv):
                pred = logits[:, -1, :].float().argmax(dim=-1)  # (G,)
                if t == 0:
                    first_pred = pred.clone()
                per_tok[:, t] = pred == v_ids[:, t]
                if t < lv - 1:
                    logits, cache = model.forward_step(v_ids[:, t:t + 1],
                                                       cache)
        for i, q in enumerate(qs):
            results.append({
                "name": q["name"], "group": q["group"],
                "correct": int(bool(per_tok[i].all().item())),
                "per_tok_correct": [int(x) for x in per_tok[i].tolist()],
                "first_tok_pred": int(first_pred[i].item()),
                "first_tok_true": int(q["value_ids"][0]),
                "first_tok_alt": (int(q["alt_value_ids"][0])
                                  if q.get("alt_value_ids") else None),
            })
    return results


# --------------------------------------------------------------------------- #
# Model loading + ingestion.
# --------------------------------------------------------------------------- #

def load_model(ckpt_path: str, device: str = "cuda"):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    model, cfg = build_model_from_ckpt(ckpt_path)
    model = model.to(device).eval()
    if getattr(model, "use_memory", False):
        # WM buffer explicitly excluded — this probes the raw FLA state.
        model.use_memory = False
    model._film_bypass = True                  # deploy convention
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    return model, cfg, tok


@torch.no_grad()
def ingest(model, ids: list[int], device) -> tuple[list[dict], int]:
    """Full prefill of a token stream; returns (pristine per-layer states,
    n tokens seen)."""
    t = torch.tensor([ids], dtype=torch.long, device=device)
    with _autocast(torch.device(device)):
        cache, _logits = model.prefill(t)
    return snapshot_states(cache["fla_cache"]), int(cache["seen"])


@torch.no_grad()
def advance(model, states: list[dict], seen: int, ids: list[int],
            device) -> tuple[list[dict], int]:
    """Advance a state through a short token stream (e.g. the uniform
    query-separator preamble) via forward_step; returns fresh states."""
    cache = states_to_cache(replicate_states(states, 1), seen)
    dev = torch.device(device)
    t = torch.tensor([ids], dtype=torch.long, device=dev)
    with _autocast(dev):
        for i in range(t.shape[1]):
            _logits, cache = model.forward_step(t[:, i:i + 1], cache)
    return snapshot_states(cache["fla_cache"]), int(cache["seen"])


# --------------------------------------------------------------------------- #
# Trial evaluation.
# --------------------------------------------------------------------------- #

def build_arm_states(model, trial: dict, device) -> dict:
    """(arm -> (states, seen)) for all MERGE_ARMS on one trial."""
    st_a, seen_a = ingest(model, trial["ids_a"], device)
    st_b, seen_b = ingest(model, trial["ids_b"], device)
    st_seq, seen_seq = ingest(model, trial["ids_seq"], device)
    seen_m = seen_a + seen_b                   # bookkeeping only (max_T==0)
    return {
        "seq": (st_seq, seen_seq),
        "mean": (merge_state_dicts(st_a, st_b, "mean"), seen_m),
        "sum": (merge_state_dicts(st_a, st_b, "sum"), seen_m),
        "normmax": (merge_state_dicts(st_a, st_b, "normmax"), seen_m),
        "b_only": (st_b, seen_b),
    }


def eval_trial(model, tok, trial: dict, device,
               preamble: str = "\n") -> dict:
    """Run every arm on one trial; returns arm -> per-query result list."""
    pre_ids = tok.encode(preamble, add_special_tokens=False)
    arms = build_arm_states(model, trial, device)
    out = {}
    for arm, (states, seen) in arms.items():
        # Uniform preamble flushes the conv buffer / line boundary equally
        # for every arm before the first query.
        st, sn = advance(model, states, seen, pre_ids, device)
        out[arm] = run_queries(model, st, sn, trial["queries"], device)
    return out


def aggregate(per_trial: list[dict]) -> dict:
    """arm -> group -> {n, k, acc, wilson95}; plus shared-key first-token
    interference diagnostics."""
    agg: dict = {}
    for trial in per_trial:
        for arm, results in trial.items():
            a = agg.setdefault(arm, {})
            for r in results:
                g = a.setdefault(r["group"],
                                 {"n": 0, "k": 0, "pred_a_first": 0,
                                  "pred_b_first": 0})
                g["n"] += 1
                g["k"] += r["correct"]
                if r["group"] == "shared":
                    if r["first_tok_pred"] == r["first_tok_true"]:
                        g["pred_b_first"] += 1
                    elif (r["first_tok_alt"] is not None
                          and r["first_tok_pred"] == r["first_tok_alt"]):
                        g["pred_a_first"] += 1
    for arm, by_group in agg.items():
        for g, d in by_group.items():
            d["acc"] = d["k"] / d["n"] if d["n"] else None
            d["wilson95"] = wilson_ci(d["k"], d["n"])
    return agg


def print_table(title: str, agg: dict, groups: list[str]):
    print(f"\n=== {title} ===")
    hdr = f"{'arm':>8} |" + "".join(
        f" {g + ' acc [95% CI]':>18} {'n':>6} |" for g in groups)
    print(hdr)
    print("-" * len(hdr))
    for arm in MERGE_ARMS:
        if arm not in agg:
            continue
        row = f"{arm:>8} |"
        for g in groups:
            d = agg[arm].get(g)
            if d is None or d["n"] == 0:
                row += f" {'n/a':>18} {'':>6} |"
            else:
                lo, hi = d["wilson95"]
                cell = f"{d['acc']:.3f} [{lo:.2f},{hi:.2f}]"
                row += f" {cell:>18} {d['n']:>6} |"
        print(row)
    if any("shared" in agg.get(arm, {}) for arm in MERGE_ARMS):
        print("  shared keys scored against the B/LATER value "
              "(sequential-ingest semantics); first-token pick:")
        for arm in MERGE_ARMS:
            d = agg.get(arm, {}).get("shared")
            if d and d["n"]:
                print(f"    {arm:>8}: B-value {d['pred_b_first']}/{d['n']}, "
                      f"A-value {d['pred_a_first']}/{d['n']}, "
                      f"neither {d['n'] - d['pred_a_first'] - d['pred_b_first']}"
                      f"/{d['n']}")


def apply_smoke(args) -> None:
    """--smoke: 8 bindings, 1 trial (fast end-to-end plumbing check)."""
    args.n_bindings = 8
    args.n_trials = 1


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/feature_pilot_A.pt",
                    help="lean no-WM ckpt (feature_pilot_A or stageA_executor)")
    ap.add_argument("--n_bindings", type=int, default=32,
                    help="bindings PER context (A and B each)")
    ap.add_argument("--n_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap_frac", type=float, default=0.25,
                    help="key-overlap fraction for the interference arm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/probe_state_algebra.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        apply_smoke(args)

    assert torch.cuda.is_available() or args.device == "cpu", "needs CUDA"
    t0 = time.time()
    model, cfg, tok = load_model(args.ckpt, args.device)
    print(f"[loaded] {args.ckpt}  n_layers={cfg['n_layers']} "
          f"d_model={cfg['d_model']} "
          f"use_memory={getattr(model, 'use_memory', False)} "
          f"film_bypass={model._film_bypass}", flush=True)

    disjoint_trials, overlap_trials = [], []
    for trial_i in range(args.n_trials):
        # stable arithmetic seeds (string-tuple hashes vary per process)
        rng_d = random.Random(args.seed * 10_000 + trial_i * 2)
        rng_o = random.Random(args.seed * 10_000 + trial_i * 2 + 1)
        tr_d = build_trial(tok, args.n_bindings, rng_d, overlap_frac=0.0)
        tr_o = build_trial(tok, args.n_bindings, rng_o,
                           overlap_frac=args.overlap_frac)
        disjoint_trials.append(eval_trial(model, tok, tr_d, args.device))
        overlap_trials.append(eval_trial(model, tok, tr_o, args.device))
        print(f"[trial {trial_i + 1}/{args.n_trials}] done "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)

    agg_d = aggregate(disjoint_trials)
    agg_o = aggregate(overlap_trials)
    print_table(f"DISJOINT keys (N={args.n_bindings} per context, "
                f"{args.n_trials} trials)", agg_d, ["A", "B"])
    print_table(f"OVERLAP {args.overlap_frac:.0%} of keys "
                f"(later/B value wins)", agg_o, ["A", "B", "shared"])

    # Pre-registered read, mechanically applied on disjoint A-keys.
    seq_a = agg_d["seq"]["A"]["acc"]
    floor_a = agg_d["b_only"]["A"]["acc"]
    print("\n=== PRE-REGISTERED READ (disjoint A-keys) ===")
    print(f"  sequential ceiling: {seq_a:.3f}   b_only floor: {floor_a:.3f}")
    for arm in ("mean", "sum", "normmax"):
        acc = agg_d[arm]["A"]["acc"]
        gap_pp = 100.0 * (seq_a - acc)
        if gap_pp <= 10.0:
            verdict = "APPROX-ADDITIVE (within 10pp of sequential)"
        elif acc <= floor_a + 0.05:
            verdict = "CATASTROPHIC (~= b_only floor: order-entangled)"
        else:
            verdict = "PARTIAL (between floor and ceiling)"
        print(f"  {arm:>8}: {acc:.3f}  (gap to seq {gap_pp:+.1f}pp)  -> {verdict}")

    results = {
        "ckpt": args.ckpt, "n_bindings": args.n_bindings,
        "n_trials": args.n_trials, "seed": args.seed,
        "overlap_frac": args.overlap_frac,
        "disjoint": agg_d, "overlap": agg_o,
        "runtime_s": time.time() - t0,
    }
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[saved] {args.out}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
