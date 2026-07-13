"""CRUXEval-O transfer probe — does the synthetic exec-trace training
(Stage A text scratchpad / Stage B latent trace, EXEC_TRACE_LATENT_PLAN.md)
transfer to REAL-code output prediction (CRUXEval-O, arXiv 2401.03065)?

Absolute scores will be LOW at 402M — the probe's value is the CONTRASTS:
  (a) trace/latent arms vs the same-ckpt `direct` arm (does the trained
      scratchpad/latent machinery help on real code at all?), and
  (b) the executor ckpts vs the untrained-executor control
      (`checkpoints/feature_pilot_A.pt`).

Data: HuggingFace `cruxeval-org/cruxeval` (test split, 800 problems; fields
`code` / `input` / `output` / `id`). `--n` problems after a seeded shuffle
(default 200, seed 0). If the dataset is neither cached nor downloadable
(HF_HUB_OFFLINE=1 without a cache), we error out with a clear message.

Prompt templates (EXACT — tests pin these):
  direct      rec["code"].rstrip() + f"\\nassert f({rec['input']}) == "
              (the CRUXEval-O output-prediction convention: complete the
              assert with a literal; no few-shot examples, no [PYTHON] tags —
              our 402M models never saw that chat scaffolding).
  text_trace  direct-prompt .rstrip() (drops the trailing space after '==')
              + "\\n# trace:\\n" — the Stage-A trained scratchpad cue.
  latent_R{N} the text_trace prompt, then N latent (hidden-feedback) slots
              injected via the Stage-B growing-thread machinery
              (`eval_exec_trace_latent_trace.latent_greedy_answer`), then
              greedy text continuation. On a ckpt WITHOUT a trained
              LatentFeedbackAdapter (stageA_executor / feature_pilot_A)
              `apply_latent_feedback_adapter` is the identity, so the arm
              degrades to raw-hidden feedback — an interpretable control,
              flagged in the output as `latent_adapter: false`.

Parse rules (EXACT — tests pin these):
  direct arm      candidate = the SHORTEST newline-terminated prefix of the
                  completion that `ast.literal_eval`s (`extract_literal_
                  prefix`); if none, fall back to the first non-empty line,
                  stripped (feeds the string-strip equality fallback only).
  trace/latent    in priority order (`extract_trace_answer`):
                    1. first "# final: <rest-of-line>" match -> rest-of-line
                       (the Stage-A/B trained answer cue; value is an
                       arbitrary literal here, unlike the int-only synthetic
                       parser in eval_exec_trace_text.py);
                    2. else first "assert ... == <rest-of-line>" match ->
                       rest-of-line (the model re-completing the assert
                       after its scratchpad);
                    3. else the literal-prefix rule (model answered
                       immediately). NO first-line fallback here — trace
                       output legitimately starts with "# step" lines.
  scoring         `score_answer(candidate, gold)`: if BOTH candidate and
                  gold `ast.literal_eval` -> Python `==` on the values (the
                  official CRUXEval comparison); else string-strip equality.
                  `parsed` (the parse-rate numerator) = candidate extracted
                  AND it literal-evals; a fallback string match can still
                  count as correct with parsed=False.

CRUXEval outputs are arbitrary multi-token Python values, so the synthetic
per-hop (slot-decode) machinery does NOT apply — final-answer scoring only.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \\
      experiments/eval_cruxeval_transfer.py \\
      --ckpt checkpoints/stageB_latent_trace.pt --n 200 \\
      --out results/cruxeval_transfer_stageB.json

  # fast harness self-check (8 problems):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \\
      experiments/eval_cruxeval_transfer.py --ckpt <ckpt> --smoke
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import random
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

# Reuse (import, don't copy) the existing generator machinery. Text arms ride
# the Stage-A incremental prefill/forward_step greedy generator; latent arms
# ride the Stage-B growing-thread injector + greedy continue. Both are
# referenced through THIS module's globals so tests can monkeypatch them.
from experiments.eval_exec_trace_text import greedy_generate
from experiments.eval_exec_trace_latent_trace import (
    latent_greedy_answer,
    load_eval_model,
)

DEFAULT_ARMS = ("direct", "text_trace", "latent_R4", "latent_R8")

# --------------------------------------------------------------------------- #
# Prompt templates — pure functions, pinned by tests.
# --------------------------------------------------------------------------- #

def build_direct_prompt(rec: dict) -> str:
    """CRUXEval-O output-prediction prompt: the function source, then an
    assert whose right-hand side the model must complete with a literal."""
    return rec["code"].rstrip() + f"\nassert f({rec['input']}) == "


def build_trace_prompt(rec: dict) -> str:
    """The Stage-A trained scratchpad cue appended to the direct prompt
    (rstrip drops the dangling space after '==' so the cue starts a fresh
    line exactly like the training format's `<prompt>\\n# trace:\\n`)."""
    return build_direct_prompt(rec).rstrip() + "\n# trace:\n"


# --------------------------------------------------------------------------- #
# Parsing + scoring — pure functions, no torch/model dependency.
# --------------------------------------------------------------------------- #

# "# final: <anything to end of line>" — arbitrary-literal variant of the
# int-only FINAL_RE in eval_exec_trace_text.py.
FINAL_LINE_RE = re.compile(r"#\s*final\s*:\s*([^\n]+)", re.IGNORECASE)
# "assert ... == <anything to end of line>" — the model re-completing the
# assert after its scratchpad.
ASSERT_LINE_RE = re.compile(r"assert\s+[^\n]*?==\s*([^\n]+)")


def try_literal(s: str):
    """(value, ok): `ast.literal_eval` on the stripped string; ok=False on
    ANY failure (never raises)."""
    try:
        return ast.literal_eval(s.strip()), True
    except Exception:
        return None, False


def extract_literal_prefix(text: str) -> str | None:
    """The SHORTEST cumulative-line prefix of `text` that literal-evals
    (so `'[1,\\n 2]\\njunk'` -> '[1,\\n 2]'), or None. Scanning whole-line
    prefixes keeps multi-line literals intact while cutting greedy-decode
    junk that follows the value on later lines."""
    lines = text.splitlines()
    for i in range(1, len(lines) + 1):
        cand = "\n".join(lines[:i]).strip()
        if cand and try_literal(cand)[1]:
            return cand
    return None


def first_nonempty_line(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def extract_direct_answer(text: str) -> str | None:
    """Direct-arm candidate: literal prefix, else first non-empty line
    (the latter only ever feeds the string-strip fallback)."""
    cand = extract_literal_prefix(text)
    if cand is not None:
        return cand
    return first_nonempty_line(text)


def extract_trace_answer(text: str) -> str | None:
    """Trace/latent-arm candidate — priority: '# final:' line, then
    assert-completion, then a leading literal. See module docstring."""
    m = FINAL_LINE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = ASSERT_LINE_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return extract_literal_prefix(text)


def score_answer(cand: str | None, gold_str: str) -> tuple[bool, bool]:
    """(correct, parsed). Literal-equality when both sides literal-eval
    (Python `==`, matching the official CRUXEval comparison — so
    '[1,2]' == '[1, 2]' but a *string* '(1, 2)' != the *tuple* (1, 2));
    else string-strip equality with parsed=False."""
    if cand is None:
        return False, False
    pv, pok = try_literal(cand)
    gv, gok = try_literal(gold_str)
    if pok and gok:
        return bool(pv == gv), True
    return cand.strip() == gold_str.strip(), pok


# Early-stop predicates for greedy_generate (a false negative only means we
# generate up to the token budget — output stays parseable either way).

def _has_complete_literal(text: str) -> bool:
    """True once some newline-TERMINATED whole-line prefix literal-evals
    (the value token stream looks finished, not mid-emission)."""
    lines = text.split("\n")
    for i in range(1, len(lines)):  # only prefixes followed by a newline
        cand = "\n".join(lines[:i]).strip()
        if cand and try_literal(cand)[1]:
            return True
    return False


def _has_complete_trace_answer(text: str) -> bool:
    """True once a '# final:' or assert-completion line is newline-
    terminated (its rest-of-line value looks finished)."""
    for rx in (FINAL_LINE_RE, ASSERT_LINE_RE):
        m = rx.search(text)
        if m and m.end() < len(text) and text[m.end()] == "\n":
            return True
    return False


# --------------------------------------------------------------------------- #
# Data loading.
# --------------------------------------------------------------------------- #

def load_cruxeval() -> list[dict]:
    """The 800-problem CRUXEval test split as a list of plain dicts. Uses
    the HF cache when present (works under HF_HUB_OFFLINE=1 once cached);
    otherwise downloads; otherwise raises with a clear message."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cruxeval-org/cruxeval", split="test")
    except Exception as e:
        raise RuntimeError(
            "Could not load 'cruxeval-org/cruxeval' from HuggingFace. "
            "Either the dataset is not in the local HF cache and the network "
            "is unavailable (or HF_HUB_OFFLINE=1), or `datasets` is broken. "
            "Fix: run once WITH network (no HF_HUB_OFFLINE) to populate the "
            f"cache. Underlying error: {type(e).__name__}: {e}") from e
    return [dict(r) for r in ds]


def select_problems(records: list[dict], n: int, seed: int) -> list[dict]:
    """Seeded-shuffle then take the first `n` (n<=0 or n>=len -> all,
    still shuffled). Pure + deterministic — pinned by tests."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    if n and 0 < n < len(idx):
        idx = idx[:n]
    return [records[i] for i in idx]


# --------------------------------------------------------------------------- #
# Per-record evaluation.
# --------------------------------------------------------------------------- #

def _latent_arm_r(arm: str) -> int | None:
    """'latent_R8' -> 8; None for non-latent arm names."""
    m = re.fullmatch(r"latent_R(\d+)", arm)
    return int(m.group(1)) if m else None


@torch.no_grad()
def eval_record(model, tok, eos_id: int, thinking_id: int, rec: dict, device,
                arms=DEFAULT_ARMS, max_gen_direct: int = 64,
                max_gen_trace: int = 256, max_gen_latent: int = 64,
                grace: int = 4) -> dict:
    gold = rec["output"]
    res: dict = {"id": rec.get("id")}
    for arm in arms:
        if arm == "direct":
            ids = tok.encode(build_direct_prompt(rec), add_special_tokens=False)
            txt = greedy_generate(model, tok, ids, max_gen_direct, eos_id,
                                  stop_fn=_has_complete_literal, grace=grace)
            cand = extract_direct_answer(txt)
        elif arm == "text_trace":
            ids = tok.encode(build_trace_prompt(rec), add_special_tokens=False)
            txt = greedy_generate(model, tok, ids, max_gen_trace, eos_id,
                                  stop_fn=_has_complete_trace_answer,
                                  grace=grace)
            cand = extract_trace_answer(txt)
        else:
            r = _latent_arm_r(arm)
            if r is None:
                raise ValueError(f"unknown arm {arm!r}")
            ids = tok.encode(build_trace_prompt(rec), add_special_tokens=False)
            txt = latent_greedy_answer(model, ids, r, thinking_id, tok,
                                       max_gen_latent, device, grace)
            cand = extract_trace_answer(txt)
        correct, parsed = score_answer(cand, gold)
        res[arm] = {"pred": cand, "correct": int(correct),
                    "parsed": int(parsed), "text": txt}
    return res


# --------------------------------------------------------------------------- #
# Aggregation + reporting.
# --------------------------------------------------------------------------- #

def aggregate(per_record: list[dict], arms=DEFAULT_ARMS) -> dict:
    out = {}
    for arm in arms:
        rows = [r[arm] for r in per_record if arm in r]
        n = len(rows)
        out[arm] = {
            "n": n,
            "acc": (sum(r["correct"] for r in rows) / n) if n else None,
            "parse_rate": (sum(r["parsed"] for r in rows) / n) if n else None,
        }
    return out


def print_table(title: str, agg: dict):
    print(f"\n=== {title} ===")
    hdr = f"{'arm':>12} {'n':>5} | {'acc':>7} | {'parse%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for arm, r in agg.items():
        acc = f"{r['acc']:.3f}" if r["acc"] is not None else "  n/a"
        pr = f"{100 * r['parse_rate']:.1f}" if r["parse_rate"] is not None \
            else "  n/a"
        print(f"{arm:>12} {r['n']:>5} | {acc:>7} | {pr:>6}%")
    print("  (acc = exact-match vs CRUXEval gold output, literal-equality "
          "with string-strip fallback; parse% = a literal candidate was "
          "extracted — separates format compliance from capability)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200,
                    help="problems evaluated (seeded shuffle of the 800)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma list from: direct, text_trace, latent_R<N>")
    ap.add_argument("--max_gen_direct", type=int, default=64)
    ap.add_argument("--max_gen_trace", type=int, default=256)
    ap.add_argument("--max_gen_latent", type=int, default=64)
    ap.add_argument("--grace_tokens", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--save_texts", action="store_true",
                    help="keep full generated texts in the per-record JSON "
                         "(default: truncate to 600 chars)")
    ap.add_argument("--smoke", action="store_true",
                    help="8 problems, fast harness self-check")
    args = ap.parse_args()

    if args.smoke:
        args.n = 8

    assert torch.cuda.is_available(), "needs CUDA"
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for a in arms:
        if a not in ("direct", "text_trace") and _latent_arm_r(a) is None:
            raise SystemExit(f"unknown arm {a!r}")

    records = select_problems(load_cruxeval(), args.n, args.seed)
    print(f"[data] cruxeval-org/cruxeval test: {len(records)} problems "
          f"(seed={args.seed})")

    t0 = time.time()
    model, cfg, thinking_id, tok, eos_id = load_eval_model(
        args.ckpt, args.device, bf16=not args.no_bf16,
        tf32=not args.no_tf32)
    has_adapter = bool(getattr(model, "use_latent_feedback_adapter", False))
    print(f"[loaded] {args.ckpt} ({cfg.get('n_layers')}L x "
          f"{cfg.get('d_model')}d) thinking_id={thinking_id} eos_id={eos_id} "
          f"latent_adapter={has_adapter}", flush=True)
    if not has_adapter and any(_latent_arm_r(a) is not None for a in arms):
        print("[warn] ckpt has NO trained LatentFeedbackAdapter — latent "
              "arms run as the raw-hidden-feedback control "
              "(apply_latent_feedback_adapter is the identity).")

    per_record = []
    for i, rec in enumerate(records):
        per_record.append(eval_record(
            model, tok, eos_id, thinking_id, rec, args.device, arms=arms,
            max_gen_direct=args.max_gen_direct,
            max_gen_trace=args.max_gen_trace,
            max_gen_latent=args.max_gen_latent, grace=args.grace_tokens))
        if (i + 1) % 25 == 0 or (i + 1) == len(records):
            print(f"  [{i + 1}/{len(records)}] {time.time() - t0:.0f}s "
                  f"elapsed", flush=True)

    agg = aggregate(per_record, arms)
    print_table(f"CRUXEval-O transfer ({args.ckpt}, n={len(records)})", agg)

    elapsed = time.time() - t0
    print(f"\n[done] total runtime {elapsed:.1f}s")

    if args.out:
        if not args.save_texts:
            for r in per_record:
                for arm in arms:
                    if arm in r and isinstance(r[arm].get("text"), str):
                        r[arm]["text"] = r[arm]["text"][:600]
        results = {
            "ckpt": args.ckpt, "n": len(records), "seed": args.seed,
            "arms": list(arms), "latent_adapter": has_adapter,
            "aggregate": agg, "per_record": per_record,
            "runtime_s": elapsed,
        }
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
