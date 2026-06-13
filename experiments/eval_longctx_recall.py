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


import contextlib


@contextlib.contextmanager
def _coop_off_ablation(model):
    """Temporarily zero `mem_alpha` — the WM×latent COOPERATION channel.

    Stage A kill-gate ablation. In the cooperation latent step the next think
    input is `adapter(h) + mem_alpha·WM_inj`; setting mem_alpha→0 collapses it to
    the adapter-only path (WM retrieval contributes nothing) WITHOUT touching the
    adapter, the trunk, or the gate. So WM-on (trained mem_alpha) minus WM-off
    (this ablation) isolates exactly 'did the trained WM addressing help recall
    through the cooperation channel'. NOTE: `_wm_mean_ablation` (read_alpha→0)
    does NOT ablate this channel — the cooperation read uses `_last_injection`,
    which is stashed BEFORE read_alpha scaling. This is the correct ablation for
    a cooperation ckpt."""
    ma = getattr(model, "mem_alpha", None)
    if ma is None:
        yield
        return
    orig = ma.data.clone()
    try:
        ma.data.zero_()
        yield
    finally:
        ma.data.copy_(orig)


@contextlib.contextmanager
def _wm_mean_ablation(model):
    """Temporarily force WorkingMemory's read injection off (read_alpha→0).

    The 'mean'-ablation framing from feature_probe._wm_mean_ablation: with
    read_alpha=0 the residual `alpha * injection * inj` vanishes, so think
    tokens still drive a normal forward (the think pathway is present) but the
    RETRIEVED content contributes nothing. A recall-accuracy DROP under this
    ablation therefore means "the retrieved content was load-bearing", not
    "the think mechanism is broken" (which the older `--wm_ablate zero` that
    zeroes W_proj cannot disentangle for retrieval-as-input ckpts).
    """
    mem = getattr(model, "memory", None)
    if mem is None or not hasattr(mem, "read_alpha"):
        yield
        return
    orig = mem.read_alpha.data.clone()
    try:
        mem.read_alpha.data.zero_()
        yield
    finally:
        mem.read_alpha.data.copy_(orig)


def eval_longctx_recall(
    model,
    tok,
    path: str = "data/longctx_recall_heldout.jsonl",
    n: int | None = None,
    *,
    generator: str = "retrieval_as_input",
    wm_ablate: str = "none",
    max_gen: int = 96,
    total_think_budget: int = 200,
    emit_threshold: float = 0.5,
    gate_floor: float = 0.0,
    additive: bool | None = None,
    device: str = "cuda",
) -> dict:
    """Importable long-context recall eval — the WM-load-bearing probe.

    Runs the same greedy generation loop as the CLI ``main`` but as a callable
    so the train_lm feature-probe can invoke it periodically WITHOUT shelling
    out. Returns recall accuracy (optionally with the WM read mean-ablated).

    Args:
      model: a loaded TinyLM (eval mode handled internally; train mode restored
        on exit). Must have ``thinking_token_id``.
      tok: tokenizer.
      path: held-out recall JSONL (``problem_prompt`` + ``answer``/``print(N)``).
      n: cap on number of tasks (None = all). Small (e.g. 64) for an in-train
        probe.
      generator: 'retrieval_as_input' (v5+/v10 WM ckpts), 'standard', or
        'latent_think'.
      wm_ablate: 'none', 'mean' (zero read_alpha — see ``_wm_mean_ablation``),
        or 'coop_off' (zero mem_alpha — the WM×latent cooperation channel, the
        Stage A kill-gate WM-off arm).
      additive: retrieval-as-input additive mode; None → read ckpt cfg via
        ``model.retrieval_input_additive`` if present else False.

    Returns a flat dict of finite floats: ``recall`` (accuracy in [0,1]),
    ``n_correct``, ``n_total``, ``think_rate``, ``think_frac`` (= think_rate;
    a >0 guard that the probe actually exercised the think path), and
    ``wm_ablate`` echoed as 0/1 in ``wm_ablated``.

    Never raises on a missing WM (just runs without ablation effect). Runs
    under ``torch.no_grad``.
    """
    from experiments.eval_humaneval import (
        generate, generate_with_retrieval_as_input, generate_latent_think)
    from experiments.sft_code import _flatten_to_oneline

    was_training = getattr(model, "training", False)
    thinking_token_id = (getattr(model, "thinking_token_id", None))
    if thinking_token_id is None:
        raise ValueError("model has no thinking_token_id — needs a "
                         "thinking-gate model for the recall probe.")
    if additive is None:
        additive = bool(getattr(model, "retrieval_input_additive", False))
    eff_max_T = int(getattr(model, "max_T", 0) or 2048)

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if n is not None:
        records = records[:n]

    n_correct = n_total = 0
    agg_think = agg_emit = 0
    abl_cm = (_wm_mean_ablation(model) if wm_ablate == "mean"
              else _coop_off_ablation(model) if wm_ablate == "coop_off"
              else contextlib.nullcontext())
    try:
        model.eval()
        with torch.no_grad(), abl_cm:
            for rec in records:
                prompt = f"# {_flatten_to_oneline(rec['problem_prompt'])}\n"
                prompt_ids = tok.encode(prompt, add_special_tokens=False)
                if len(prompt_ids) + max_gen + total_think_budget > eff_max_T:
                    continue
                prompt_t = torch.tensor(
                    prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                if generator == "latent_think":
                    gen, diag = generate_latent_think(
                        model, prompt_t, max_gen=max_gen, temperature=0.0,
                        eos_token_id=tok.eos_token_id,
                        thinking_token_id=thinking_token_id,
                        total_think_budget=total_think_budget,
                        emit_threshold=emit_threshold, gate_floor=gate_floor)
                elif generator == "retrieval_as_input":
                    gen, diag = generate_with_retrieval_as_input(
                        model, prompt_t, max_gen=max_gen, temperature=0.0,
                        eos_token_id=tok.eos_token_id,
                        thinking_token_id=thinking_token_id,
                        total_think_budget=total_think_budget,
                        emit_threshold=emit_threshold, gate_floor=gate_floor,
                        additive=additive)
                else:
                    gen, diag = generate(
                        model, prompt_t, max_gen=max_gen, temperature=0.0,
                        eos_token_id=tok.eos_token_id, use_thinking=True,
                        thinking_token_id=thinking_token_id,
                        total_think_budget=total_think_budget,
                        emit_threshold=emit_threshold, gate_floor=gate_floor)
                gen_only = [t for t in gen[0, len(prompt_ids):].tolist()
                            if t != int(thinking_token_id)]
                text = tok.decode(gen_only, skip_special_tokens=True)
                pred = extract_predicted_answer(text)
                gold = gold_answer(rec)
                n_total += 1
                n_correct += int(pred is not None and pred == gold)
                agg_think += diag.get("think_total", 0)
                agg_emit += diag.get("emit_count", 0)
    finally:
        if was_training:
            model.train()

    think_rate = agg_think / max(1, agg_think + agg_emit)
    return {
        "recall": n_correct / max(1, n_total),
        "n_correct": float(n_correct),
        "n_total": float(n_total),
        "think_rate": float(think_rate),
        "think_frac": float(think_rate),
        "wm_ablated": 1.0 if wm_ablate in ("mean", "coop_off", "zero") else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tasks", type=str,
                   default="data/longctx_recall_heldout.jsonl")
    p.add_argument("--generator", type=str, default="standard",
                   choices=["standard", "retrieval_as_input", "latent_think"],
                   help="'latent_think' = unified hybrid (hidden + α·WM "
                        "retrieval, β=0); 'retrieval_as_input' for v5+ ckpts.")
    p.add_argument("--force_prefix_think", type=int, default=0,
                   help="latent_think: force N think steps before the answer "
                        "(matches latent_sft think-before-answer training).")
    p.add_argument("--wm_ablate", type=str, default="none",
                   choices=["none", "zero", "mean", "coop_off"],
                   help="'zero' zeroes memory.W_proj.weight (CONFOUNDED for "
                        "retrieval-as-input ckpts — see module doc). 'mean' "
                        "zeroes read_alpha so think tokens still drive a "
                        "forward but the retrieved content contributes nothing "
                        "(the unconfounded WM-content ablation). 'coop_off' "
                        "zeroes mem_alpha — the WM×latent cooperation channel "
                        "(Stage A kill-gate WM-off arm; the read_alpha 'mean' "
                        "ablation does NOT disable cooperation).")
    p.add_argument("--max_gen", type=int, default=120)
    p.add_argument("--total_think_budget", type=int, default=200)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--out_log", type=str, default=None)
    # 3-way state-readonly probe (2026-05-28): test whether thinking that
    # READS but never WRITES the recurrent state preserves recall vs
    # thinking that writes (and corrupts) it.
    p.add_argument("--force_state_readonly", action="store_true",
                   help="Override the ckpt cfg: install the β=0 b_proj hook "
                        "at think positions so thinks can't write the state.")
    p.add_argument("--no_think", action="store_true",
                   help="Disable thinking entirely (total_think_budget=0 → "
                        "force-emit before any think token). The control "
                        "baseline for the 3-way recall probe.")
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.eval_humaneval import (
        generate, generate_with_retrieval_as_input, generate_latent_think)
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
    model, cfg = build_model_from_ckpt(
        ckpt_path,
        force_state_readonly=(True if (args.force_state_readonly
                                       or args.generator == "latent_think")
                              else None))
    model.eval()
    # no-think → zero think budget forces emit before any think token.
    eff_think_budget = 0 if args.no_think else args.total_think_budget
    print(f"  think_mode={'NO-THINK' if args.no_think else ('THINK+state-readonly' if args.force_state_readonly else 'THINK+state-write')}"
          f"  think_budget={eff_think_budget}")
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

    # 'mean' ablation: zero read_alpha for the whole loop (the unconfounded
    # WM-content ablation; 'zero' is handled above by a rewritten ckpt).
    # 'coop_off': zero mem_alpha (the WM×latent cooperation channel; Stage A
    # kill-gate WM-off arm — the only ablation that disables cooperation).
    _abl = (_wm_mean_ablation(model) if args.wm_ablate == "mean"
            else _coop_off_ablation(model) if args.wm_ablate == "coop_off"
            else __import__("contextlib").nullcontext())
    with _abl:
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
            if args.generator == "latent_think":
                gen, diag = generate_latent_think(
                    model, prompt_t, max_gen=args.max_gen,
                    temperature=0.0, eos_token_id=tok.eos_token_id,
                    thinking_token_id=thinking_token_id,
                    total_think_budget=eff_think_budget,
                    emit_threshold=args.emit_threshold,
                    gate_floor=args.gate_floor,
                    force_prefix_think=(0 if args.no_think else args.force_prefix_think))
            elif args.generator == "retrieval_as_input":
                gen, diag = generate_with_retrieval_as_input(
                    model, prompt_t, max_gen=args.max_gen,
                    temperature=0.0, eos_token_id=tok.eos_token_id,
                    thinking_token_id=thinking_token_id,
                    total_think_budget=eff_think_budget,
                    emit_threshold=args.emit_threshold,
                    gate_floor=args.gate_floor,
                    additive=cfg.get("retrieval_input_additive", False))
            else:
                gen, diag = generate(
                    model, prompt_t, max_gen=args.max_gen,
                    temperature=0.0, eos_token_id=tok.eos_token_id,
                    use_thinking=True,
                    thinking_token_id=thinking_token_id,
                    total_think_budget=eff_think_budget,
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
