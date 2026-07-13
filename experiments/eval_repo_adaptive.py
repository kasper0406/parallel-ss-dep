"""Three-arm repo-adaptive eval (Meta-TTT Phase P0 kill-test harness).

For each held-out cross-file usage-prediction episode
(`gen_repo_episodes.py`), teacher-forced CE on the `task_line` tokens under
three ingestion arms:

  real      [all context files concatenated in order] + task_prefix → task_line
  shuffled  the shuffled-repo control (file order shuffled + non-task files
            line-permuted, EXACT same token count) + task_prefix → task_line
  none      task_prefix only → task_line   (task-local context, no repo)

Reported per bucket (n_ctx: 4-8k / 8-16k / 16-32k) and overall:
  * CE per arm (whole task-line AND the identifier `task_char_span` tokens only
    — the knowledge-bearing tokens; the rest of the line dilutes),
  * lift(none − real) and lift(shuffled − real)  [positive = ingestion helps],
  * `none`-arm stratified by whether the identifier appears in task_prefix
    (in-file hint present vs the model is truly blind cross-file).

The kill-test (META_TTT_PLAN_2026_07_13.md) reads lift(real − shuffled): does
reading the STRUCTURED repo beat reading the same tokens scrambled. This script
measures it on ANY ckpt with NO training — the incidental-state-learning
baseline the kill-test compares the meta-trained model against.

DeltaNet has NO positional encoding (linear RNN), so full forwards at 8-32k are
architecturally fine. `skip_lm_head=True` returns the post-out_norm hidden so
lm_head runs only on the ~task-line positions (the (T,V) logits tensor is never
materialised). Validated single-forward up to 32k tokens (GPU).

Usage
-----
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \\
      experiments/eval_repo_adaptive.py \\
      --ckpt checkpoints/stageA_executor.pt \\
      --eval data/repo_episodes/eval.jsonl \\
      --controls data/repo_episodes/eval_controls.jsonl \\
      --out results/repo_adaptive_stageA.json

  # tiny smoke:
  PYTHONPATH=. .venv/bin/python experiments/eval_repo_adaptive.py \\
      --ckpt <ckpt> --eval data/repo_episodes/eval.jsonl --smoke --device cpu
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from experiments.gen_repo_episodes import perline_ids

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
BUCKET_ORDER = ["4-8k", "8-16k", "16-32k"]


# --------------------------------------------------------------------------- #
# Model loading (device-aware; bf16 autocast on cuda, fp32 on cpu).
# --------------------------------------------------------------------------- #

def load_model(ckpt_path: str, device: str = "cuda"):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    model, cfg = build_model_from_ckpt(ckpt_path)
    model = model.to(device).eval()
    if getattr(model, "use_memory", False):
        model.use_memory = False           # no think tokens → WM inert anyway
    model._film_bypass = True
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", TOKENIZER_NAME))
    return model, cfg, tok


def _autocast(device: str, bf16: bool):
    if bf16 and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# Token assembly.
# --------------------------------------------------------------------------- #

def _file_ids(f: dict, tok) -> list[int]:
    """Context-file token ids — per-line tokenization (see gen_repo_episodes
    header). Shuffling file/line order is a pure reordering of these per-line
    chunks, so real and shuffled contexts are EXACTLY length-matched."""
    return perline_ids(f["text"], tok)


def context_ids(episode: dict, tok) -> list[int]:
    """Concatenated ids for all context files EXCEPT the task file (the last
    one, whose prefix is supplied separately as task_prefix)."""
    ids: list[int] = []
    for f in episode["context_files"][:-1]:
        ids.extend(_file_ids(f, tok))
    return ids


def task_line_span_tokens(episode: dict, tok) -> tuple[list[int], list[int]]:
    """Tokenise task_line; return (task_line_ids, span_local_indices) where the
    span indices select the identifier's `task_char_span` tokens."""
    enc = tok(episode["task_line"], add_special_tokens=False,
              return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    s0, s1 = episode["task_char_span"]
    span_idx = [i for i, (a, b) in enumerate(offs) if a < s1 and b > s0]
    return ids, span_idx


# --------------------------------------------------------------------------- #
# Teacher-forced CE for one arm.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def arm_ce(model, prefix_ids: list[int], task_line_ids: list[int],
           span_local_idx: list[int], device: str, bf16: bool) -> dict:
    """CE on the task_line tokens given `prefix_ids` (everything before the
    task line). Returns per-token CE list, line-mean, and span-mean.

    Uses skip_lm_head → hidden, then lm_head only on the predicting positions,
    so the full (T, V) logits tensor is never built."""
    P = len(prefix_ids)
    L = len(task_line_ids)
    if L == 0:
        return {"line_ce": None, "span_ce": None, "n_line": 0, "n_span": 0}
    full = prefix_ids + task_line_ids
    x = torch.tensor([full], device=device, dtype=torch.long)
    # predicting positions: to predict full[t] use hidden at t-1.
    # task-line targets occupy [P, P+L); predictors are [P-1, P+L-1).
    pred_pos = list(range(P - 1, P + L - 1))
    tgt = torch.tensor(full[P:P + L], device=device, dtype=torch.long)
    with _autocast(device, bf16):
        h = model(x, skip_lm_head=True, doc_ids=None)
        if isinstance(h, tuple):
            h = h[0]
        logits = model.lm_head(h[0, pred_pos, :])          # (L, V)
    ce_tok = F.cross_entropy(logits.float(), tgt, reduction="none")  # (L,)
    ce_list = ce_tok.tolist()
    line_ce = float(ce_tok.mean().item())
    if span_local_idx:
        span_ce = float(ce_tok[torch.tensor(span_local_idx)].mean().item())
        n_span = len(span_local_idx)
    else:
        span_ce = None
        n_span = 0
    return {"line_ce": line_ce, "span_ce": span_ce, "per_tok": ce_list,
            "n_line": L, "n_span": n_span}


# --------------------------------------------------------------------------- #
# Episode-level three-arm eval.
# --------------------------------------------------------------------------- #

def eval_episode(model, episode: dict, control: dict | None, tok,
                 device: str, bf16: bool) -> dict:
    prefix_ctx = context_ids(episode, tok)
    task_prefix_ids = perline_ids(episode["task_prefix"], tok)
    line_ids, span_idx = task_line_span_tokens(episode, tok)

    real_prefix = prefix_ctx + task_prefix_ids
    none_prefix = task_prefix_ids

    out = {
        "repo_name": episode["repo_name"],
        "identifier": episode["link"]["identifier"],
        "bucket": episode["bucket"],
        "n_ctx_tokens": episode["n_ctx_tokens"],
        "identifier_in_task_prefix": episode.get("identifier_in_task_prefix"),
        "identifier_in_prefix_nonimport":
            episode.get("identifier_in_prefix_nonimport"),
    }
    real = arm_ce(model, real_prefix, line_ids, span_idx, device, bf16)
    none = arm_ce(model, none_prefix, line_ids, span_idx, device, bf16)
    out["real"] = {k: real[k] for k in ("line_ce", "span_ce", "n_line", "n_span")}
    out["none"] = {k: none[k] for k in ("line_ce", "span_ce", "n_line", "n_span")}

    if control is not None:
        ctrl_ctx = context_ids(control, tok)
        shuf = arm_ce(model, ctrl_ctx + task_prefix_ids, line_ids, span_idx,
                      device, bf16)
        out["shuffled"] = {k: shuf[k] for k in
                           ("line_ce", "span_ce", "n_line", "n_span")}
        # the control MUST be length-matched to real; a residual mismatch (e.g.
        # an odd tokenizer edge case) would confound lift(real−shuffled), so
        # flag it — aggregate() excludes non-matched episodes from the metric.
        out["_ctx_len_real"] = len(prefix_ctx)
        out["_ctx_len_shuffled"] = len(ctrl_ctx)
        out["ctx_length_match"] = (len(prefix_ctx) == len(ctrl_ctx))
    return out


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.fmean(xs)) if xs else None


def _has_shuffled(r: dict) -> bool:
    """Shuffled arm is usable only when it exists AND is length-matched to real
    (an unmatched control would confound lift(real−shuffled))."""
    return "shuffled" in r and r.get("ctx_length_match", False)


def aggregate(results: list[dict]) -> dict:
    def agg(subset: list[dict]) -> dict:
        d = {}
        for arm in ("real", "none"):
            d[f"{arm}_line_ce"] = _mean([r.get(arm, {}).get("line_ce")
                                         for r in subset])
            d[f"{arm}_span_ce"] = _mean([r.get(arm, {}).get("span_ce")
                                         for r in subset])
        # shuffled arm: only length-matched episodes
        shuf_sub = [r for r in subset if _has_shuffled(r)]
        d["shuffled_line_ce"] = _mean([r["shuffled"]["line_ce"]
                                       for r in shuf_sub])
        d["shuffled_span_ce"] = _mean([r["shuffled"]["span_ce"]
                                       for r in shuf_sub])
        d["n_shuffled"] = len(shuf_sub)
        # lifts (paired means; positive => ingestion helps). The shuffled lift
        # is paired only over length-matched episodes.
        def paired(a, b, key, sub=None):
            vs = []
            for r in (sub if sub is not None else subset):
                ra, rb = r.get(a, {}).get(key), r.get(b, {}).get(key)
                if ra is not None and rb is not None:
                    vs.append(rb - ra)          # (worse arm) - (real)
            return _mean(vs) if vs else None
        d["lift_none_minus_real_line"] = paired("real", "none", "line_ce")
        d["lift_shuffled_minus_real_line"] = paired("real", "shuffled",
                                                     "line_ce", shuf_sub)
        d["lift_none_minus_real_span"] = paired("real", "none", "span_ce")
        d["lift_shuffled_minus_real_span"] = paired("real", "shuffled",
                                                     "span_ce", shuf_sub)
        d["n"] = len(subset)
        return d

    out = {"overall": agg(results), "by_bucket": {}}
    for b in BUCKET_ORDER:
        sub = [r for r in results if r.get("bucket") == b]
        if sub:
            out["by_bucket"][b] = agg(sub)
    # none-arm stratification. Two cuts:
    #   in_prefix        : identifier mentioned anywhere in task_prefix
    #                      (mostly True for imported links — name-only hint).
    #   in_prefix_nonimp : identifier used OUTSIDE imports (real local signal;
    #                      the discriminating "none arm truly blind" cut).
    def _strat(key):
        s = {}
        for flag in (True, False):
            sub = [r for r in results if r.get(key) is flag]
            if sub:
                s[str(flag)] = {
                    "n": len(sub),
                    "none_line_ce": _mean([r.get("none", {}).get("line_ce")
                                           for r in sub]),
                    "real_line_ce": _mean([r.get("real", {}).get("line_ce")
                                           for r in sub]),
                    "lift_none_minus_real_line": _mean(
                        [r["none"]["line_ce"] - r["real"]["line_ce"]
                         for r in sub if r.get("none", {}).get("line_ce")
                         is not None and r.get("real", {}).get("line_ce")
                         is not None]),
                    "none_span_ce": _mean([r.get("none", {}).get("span_ce")
                                           for r in sub]),
                    "real_span_ce": _mean([r.get("real", {}).get("span_ce")
                                           for r in sub]),
                }
        return s
    out["none_stratified_by_identifier_in_prefix"] = _strat(
        "identifier_in_task_prefix")
    out["none_stratified_by_identifier_nonimport"] = _strat(
        "identifier_in_prefix_nonimport")
    return out


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else " n/a "


def print_table(agg: dict):
    print("\n=== repo-adaptive three-arm CE (lower = better) ===")
    hdr = (f"{'scope':<10} {'n':>4} | {'real':>7} {'shuf':>7} {'none':>7} | "
           f"{'shuf-real':>9} {'none-real':>9}  (line)")
    print(hdr)
    print("-" * len(hdr))

    def row(name, d):
        print(f"{name:<10} {d['n']:>4} | {_fmt(d['real_line_ce']):>7} "
              f"{_fmt(d['shuffled_line_ce']):>7} {_fmt(d['none_line_ce']):>7} | "
              f"{_fmt(d['lift_shuffled_minus_real_line']):>9} "
              f"{_fmt(d['lift_none_minus_real_line']):>9}")
    row("overall", agg["overall"])
    for b in BUCKET_ORDER:
        if b in agg["by_bucket"]:
            row(b, agg["by_bucket"][b])

    print("\n=== span-restricted (identifier tokens only) ===")
    print(f"{'scope':<10} {'n':>4} | {'real':>7} {'shuf':>7} {'none':>7} | "
          f"{'shuf-real':>9} {'none-real':>9}  (span)")
    d = agg["overall"]
    print(f"{'overall':<10} {d['n']:>4} | {_fmt(d['real_span_ce']):>7} "
          f"{_fmt(d['shuffled_span_ce']):>7} {_fmt(d['none_span_ce']):>7} | "
          f"{_fmt(d['lift_shuffled_minus_real_span']):>9} "
          f"{_fmt(d['lift_none_minus_real_span']):>9}")

    for label, key in (("identifier-in-task_prefix (name hint / mostly import)",
                        "none_stratified_by_identifier_in_prefix"),
                       ("identifier-used-outside-imports (local signal)",
                        "none_stratified_by_identifier_nonimport")):
        strat = agg.get(key, {})
        if strat:
            print(f"\n=== none-arm stratified by {label} ===")
            for flag, s in strat.items():
                print(f"  ={flag:<5} n={s['n']:>4}  "
                      f"none_line={_fmt(s['none_line_ce'])}  "
                      f"real_line={_fmt(s['real_line_ce'])}  "
                      f"none-real_line={_fmt(s.get('lift_none_minus_real_line'))}  "
                      f"none_span={_fmt(s['none_span_ce'])}  "
                      f"real_span={_fmt(s['real_span_ce'])}")


# --------------------------------------------------------------------------- #

def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _ep_key(ep: dict):
    """Unique episode key for episode↔control matching. Prefer `episode_id`
    (unique even when the same identifier is used in several files of one repo);
    fall back to (repo, identifier) for older corpora without it."""
    eid = ep.get("episode_id")
    if eid is not None:
        return eid
    return (ep["repo_name"], ep["link"]["identifier"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", required=True, help="eval.jsonl episodes")
    ap.add_argument("--controls", default=None,
                    help="eval_controls.jsonl (shuffled arm); optional")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--n", type=int, default=None, help="limit episodes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--max_ctx_tokens", type=int, default=32000,
                    help="skip episodes whose input would exceed this (safety)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    episodes = _load_jsonl(args.eval)
    if args.smoke:
        episodes = episodes[:5]
    if args.n is not None:
        episodes = episodes[:args.n]

    controls_by_key = {}
    if args.controls:
        for c in _load_jsonl(args.controls):
            controls_by_key[_ep_key(c)] = c

    bf16 = not args.no_bf16
    model, cfg, tok = load_model(args.ckpt, args.device)

    results = []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        ctrl = controls_by_key.get(_ep_key(ep))
        # safety: skip pathologically long episodes
        approx = ep["n_ctx_tokens"] + 64
        if approx > args.max_ctx_tokens:
            print(f"  [skip {i}] n_ctx {ep['n_ctx_tokens']} > "
                  f"max_ctx_tokens {args.max_ctx_tokens}", file=sys.stderr)
            continue
        r = eval_episode(model, ep, ctrl, tok, args.device, bf16)
        results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(episodes)}  ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    agg = aggregate(results)
    agg["meta"] = {
        "ckpt": args.ckpt, "n_episodes": len(results),
        "device": args.device, "bf16": bf16,
        "has_controls": bool(controls_by_key),
        "elapsed_s": round(time.time() - t0, 1),
    }
    # length-match audit for controls (mismatched episodes are EXCLUDED from
    # the shuffled metric by aggregate(); this reports how many).
    n_ctrl = sum(1 for r in results if "_ctx_len_shuffled" in r)
    mism = [r for r in results if "_ctx_len_shuffled" in r
            and not r.get("ctx_length_match", False)]
    agg["meta"]["ctx_length_mismatches"] = len(mism)
    agg["meta"]["n_shuffled_used"] = agg["overall"].get("n_shuffled")

    print_table(agg)
    print(f"\ncontrol ctx length mismatches (excluded from shuffled metric): "
          f"{len(mism)} / {n_ctrl}   "
          f"(shuffled arm used on {agg['overall'].get('n_shuffled')} episodes)")

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"aggregate": agg, "per_episode": results}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
