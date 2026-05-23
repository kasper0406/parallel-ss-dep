"""Held-out long-context recall eval — the RIGHT probe for WorkingMemory.

Motivation (2026-05-20). The HumanEval ablation cannot show whether WM
is useful: HumanEval problems are short enough to fit inside DeltaNet's
recurrent state, so long-range memory is never needed, and all v1/v5/v6
headline numbers sit at 9-11/164 regardless. WM's actual job — recall a
binding made many tokens earlier — is structurally invisible there.

This eval is the unconfounded probe. Each task (from
gen_longctx_recall_tasks.py, bucket mode) binds `x = N` at the TOP of a
program, inserts a fixed token-distance of distractor lines, then asks
the model what the program prints. Accuracy is reported as a CURVE vs
distance: if WM helps, the WM ckpts should hold accuracy at distances
where a bare DeltaNet state has decayed.

Two comparisons:
  * cross-ckpt baseline (v1 vs v5 vs v6) — fully unconfounded; three
    different trained models, same eval. Answers "did the gist target
    help WM learn".
  * within-ckpt --wm_ablate zero — zeroes memory.W_proj.weight. NOTE:
    for retrieval-as-input ckpts this is CONFOUNDED (zeroing W_proj
    feeds think tokens a zero input embedding), so a drop conflates
    "retrieval useless" with "think mechanism broken". The honest
    signal is the cross-ckpt baseline curve.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/eval_longctx_recall.py \\
      --ckpt checkpoints/sft_v7_pkm_film_combined_v6.pt \\
      --tasks data/longctx_recall_heldout.jsonl \\
      --generator retrieval_as_input
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")
_BUCKET_RE = re.compile(r"/d(\d+)/")


def extract_predicted_answer(text: str) -> str | None:
    """Pull the model's predicted integer answer out of generated text.

    The SFT target prose ends with `Answer: N`, so that pattern wins.
    Fallback: the last bare integer in the text (the model may have
    produced the value without the `Answer:` scaffold). Returns the
    integer as a string, or None when the text has no integer at all."""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1)
    ints = _INT_RE.findall(text)
    return ints[-1] if ints else None


def bucket_of(record: dict) -> int:
    """Recover the exact distance bucket. gen_longctx_recall_tasks.py
    bucket mode encodes it in the task_id (`longctx/d768/123`); fall
    back to the rounded approx_distance_tokens."""
    m = _BUCKET_RE.search(record.get("task_id", ""))
    if m:
        return int(m.group(1))
    return int(record.get("approx_distance_tokens", 0))


def gold_answer(record: dict) -> str:
    """Ground-truth answer string. Explicit `answer` field if present,
    else parsed from `extracted_code` (`print(N)`)."""
    if record.get("answer") is not None:
        return str(record["answer"])
    m = _INT_RE.search(record.get("extracted_code", ""))
    return m.group(0) if m else ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tasks", type=str,
                   default="data/longctx_recall_heldout.jsonl")
    p.add_argument("--generator", type=str, default="standard",
                   choices=["standard", "retrieval_as_input"],
                   help="Use 'retrieval_as_input' for ckpts trained "
                        "with --retrieval_as_input_thinking (v5+).")
    p.add_argument("--wm_ablate", type=str, default="none",
                   choices=["none", "zero"],
                   help="'zero' zeroes memory.W_proj.weight. CONFOUNDED "
                        "for retrieval-as-input ckpts — see module doc.")
    p.add_argument("--max_gen", type=int, default=120)
    p.add_argument("--total_think_budget", type=int, default=200)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--out_log", type=str, default=None)
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.eval_humaneval import (
        generate, generate_with_retrieval_as_input)
    from experiments.sft_code import _flatten_to_oneline
    from transformers import AutoTokenizer

    # --- resolve checkpoint (optionally WM-ablated) ---------------------
    ckpt_path = args.ckpt
    tmpdir = None
    if args.wm_ablate == "zero":
        from experiments.ablate_memory_mechanisms import _write_ablated_ckpt
        tmpdir = tempfile.TemporaryDirectory()
        ckpt_path = str(pathlib.Path(tmpdir.name) / "wm_off.pt")
        _write_ablated_ckpt(args.ckpt, ckpt_path,
                             disable_wm=True, disable_pkm=False)
        print(f"[wm_ablate=zero] zeroed memory.W_proj.weight "
              f"(CONFOUNDED for retrieval-as-input — see module doc)")

    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    model.eval()
    thinking_token_id = (getattr(model, "thinking_token_id", None)
                         or cfg.get("thinking_token_id"))
    if thinking_token_id is None:
        raise SystemExit("ckpt has no thinking_token_id — needs a "
                          "thinking-gate model.")
    if args.generator == "retrieval_as_input" and not hasattr(model, "memory"):
        raise SystemExit("--generator retrieval_as_input needs a model "
                          "with WorkingMemory.")
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eff_max_T = cfg["max_T"] if cfg["max_T"] > 0 else 2048

    # --- load held-out tasks -------------------------------------------
    records = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.max_problems is not None:
        records = records[:args.max_problems]
    print(f"  {len(records)} held-out recall tasks  generator={args.generator}")

    # bucket -> [n_correct, n_total]
    per_bucket: dict[int, list[int]] = {}
    n_correct = n_total = n_truncated = 0
    agg_think = agg_emit = 0

    for rec in records:
        prompt = f"# {_flatten_to_oneline(rec['problem_prompt'])}\n"
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        room = args.max_gen + args.total_think_budget
        if len(prompt_ids) + room > eff_max_T:
            # Left-truncation would drop the `x = N` binding — the whole
            # point of the task. Skip instead of silently mis-scoring.
            n_truncated += 1
            continue
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                 device="cuda").unsqueeze(0)
        with torch.no_grad():
            if args.generator == "retrieval_as_input":
                gen, diag = generate_with_retrieval_as_input(
                    model, prompt_t, max_gen=args.max_gen,
                    temperature=0.0, eos_token_id=tok.eos_token_id,
                    thinking_token_id=thinking_token_id,
                    total_think_budget=args.total_think_budget,
                    emit_threshold=args.emit_threshold,
                    gate_floor=args.gate_floor,
                    additive=cfg.get("retrieval_input_additive", False))
            else:
                gen, diag = generate(
                    model, prompt_t, max_gen=args.max_gen,
                    temperature=0.0, eos_token_id=tok.eos_token_id,
                    use_thinking=True,
                    thinking_token_id=thinking_token_id,
                    total_think_budget=args.total_think_budget,
                    emit_threshold=args.emit_threshold,
                    gate_floor=args.gate_floor)
        gen_only = [t for t in gen[0, len(prompt_ids):].tolist()
                    if t != int(thinking_token_id)]
        text = tok.decode(gen_only, skip_special_tokens=True)
        pred = extract_predicted_answer(text)
        gold = gold_answer(rec)
        ok = (pred is not None and pred == gold)

        b = bucket_of(rec)
        per_bucket.setdefault(b, [0, 0])
        per_bucket[b][1] += 1
        per_bucket[b][0] += int(ok)
        n_total += 1
        n_correct += int(ok)
        agg_think += diag.get("think_total", 0)
        agg_emit += diag.get("emit_count", 0)
        if n_total % 50 == 0:
            print(f"  {n_total} done  acc={n_correct/n_total:.3f}")

    # --- report ---------------------------------------------------------
    print(f"\n{'='*60}\nLONG-CONTEXT RECALL — {pathlib.Path(args.ckpt).name}")
    if args.wm_ablate != "none":
        print(f"  wm_ablate={args.wm_ablate}")
    print(f"{'='*60}")
    print(f"{'distance':>10} {'acc':>8} {'(correct/total)':>18}")
    for b in sorted(per_bucket):
        c, t = per_bucket[b]
        print(f"{b:>10} {c/max(1,t)*100:>7.1f}% {f'{c}/{t}':>18}")
    overall = n_correct / max(1, n_total)
    print(f"{'-'*40}")
    print(f"{'OVERALL':>10} {overall*100:>7.1f}% {f'{n_correct}/{n_total}':>18}")
    think_rate = agg_think / max(1, agg_think + agg_emit)
    print(f"  think_rate={think_rate:.3f}  skipped(too long)={n_truncated}")

    if args.out_log:
        with open(args.out_log, "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "wm_ablate": args.wm_ablate,
                "generator": args.generator,
                "overall_acc": overall,
                "n_correct": n_correct,
                "n_total": n_total,
                "per_bucket": {str(b): per_bucket[b] for b in per_bucket},
                "think_rate": think_rate,
                "n_truncated": n_truncated,
            }, f, indent=2)
        print(f"  summary → {args.out_log}")

    if tmpdir is not None:
        tmpdir.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
