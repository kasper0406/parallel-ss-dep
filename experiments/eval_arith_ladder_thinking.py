"""Reasoning-payoff eval: does latent thinking help on arithmetic ladders?

For a given ckpt, for each rung n=1..5 we sample up to ~200 problems from
`data/synth_arith_ladder_n{n}.jsonl`, build the prompt, and measure EXACT-MATCH
answer accuracy under three decode settings:

  - none : no thinking (plain greedy decode)
  - R=1  : one forced latent-think step before the first emit (adapter on)
  - R=2  : two forced latent-think steps before the first emit

Latent thinking uses `eval_humaneval.generate_latent_think` with
`force_prefix_think=R` — exactly R state-readonly Coconut-style latent steps
(the trunk feeds its own out_norm hidden back through the learned
LatentFeedbackAdapter as the next input embedding) before emitting the answer.
This is the validated "think-before-solution" layout from
THINKING_LATENT_2026_05_28.md.

Arithmetic ladders are the RIGHT probe for sequential reasoning (rung n =
reasoning depth); HumanEval is capacity-bound. The answer is a single integer,
so EXACT-MATCH on the emitted `return <int>` is an unambiguous accuracy signal.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/eval_arith_ladder_thinking.py \
      --ckpt checkpoints/pretrain_v9_latent_adapter_step18311_tok2400059392.pt \
      --max_problems 200
"""
import argparse
import json
import re
import sys

import torch
from transformers import AutoTokenizer

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate_latent_think


def _parse_gold(rec: dict) -> int | None:
    """The gold answer is the integer the solution returns.

    `tests` looks like `def check(candidate):\n    assert candidate() == 16\n`.
    `gold_solution` looks like `def solve():\n    return 16\n`. Prefer tests;
    fall back to gold_solution.
    """
    m = re.search(r"==\s*(-?\d+)", rec.get("tests", ""))
    if m:
        return int(m.group(1))
    m = re.search(r"return\s+(-?\d+)", rec.get("gold_solution", ""))
    if m:
        return int(m.group(1))
    return None


def _extract_pred(text: str) -> int | None:
    """Pull the integer the model's emitted code returns.

    The model is prompted to write `def solve(): return <value>`. We look for
    the FIRST `return <int>` in the generation; if none, fall back to the first
    standalone integer.
    """
    m = re.search(r"return\s+(-?\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(-?\d+)", text)
    if m:
        return int(m.group(1))
    return None


def _decode_new(tokenizer, out_ids: torch.Tensor, prompt_len: int,
                thinking_token_id: int) -> str:
    """Decode only the newly generated (non-prompt, non-think) tokens."""
    new = out_ids[0, prompt_len:].tolist()
    new = [t for t in new if t != thinking_token_id]
    return tokenizer.decode(new, skip_special_tokens=True)


@torch.no_grad()
def eval_rung(model, tokenizer, records, thinking_token_id, eos_token_id,
              max_gen, max_problems, verbose=False):
    """Return dict with acc for none / R=1 / R=2 over the records."""
    n_none = n_r1 = n_r2 = 0
    total = 0
    for rec in records[:max_problems]:
        gold = _parse_gold(rec)
        if gold is None:
            continue
        # Append the code cue so the model completes the function body the
        # gold_solution has (`def solve():\n    return <int>`) rather than
        # continuing the prompt as prose. This is the natural completion and
        # makes the emitted integer an unambiguous answer.
        prompt = rec["prompt"] + "\ndef solve():\n    return "
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        plen = prompt_ids.shape[1]
        total += 1
        for R, counter_name in ((0, "none"), (1, "r1"), (2, "r2")):
            out, _ = generate_latent_think(
                model, prompt_ids, max_gen=max_gen, temperature=0.0,
                eos_token_id=eos_token_id, thinking_token_id=thinking_token_id,
                force_prefix_think=R,
            )
            text = _decode_new(tokenizer, out, plen, thinking_token_id)
            pred = _extract_pred(text)
            ok = (pred is not None and pred == gold)
            if R == 0 and ok:
                n_none += 1
            elif R == 1 and ok:
                n_r1 += 1
            elif R == 2 and ok:
                n_r2 += 1
            if verbose and R == 1:
                print(f"    gold={gold} pred={pred} ok={ok} "
                      f"text={text[:60]!r}", file=sys.stderr)
    if total == 0:
        return dict(total=0, none=0.0, r1=0.0, r2=0.0)
    return dict(
        total=total,
        none=n_none / total,
        r1=n_r1 / total,
        r2=n_r2 / total,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--data_glob",
                    default="data/synth_arith_ladder_n{n}.jsonl")
    ap.add_argument("--rungs", default="1,2,3,4,5")
    ap.add_argument("--max_problems", type=int, default=200)
    ap.add_argument("--max_gen", type=int, default=24)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token_id = tokenizer.eos_token_id

    # force_state_readonly=True is required by generate_latent_think; these
    # ckpts already train with it on, but force it for safety/control ckpts.
    model, cfg = build_model_from_ckpt(args.ckpt, force_state_readonly=True)
    thinking_token_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    has_adapter = bool(getattr(model, "use_latent_feedback_adapter", False))
    print(f"# ckpt={args.ckpt}")
    print(f"# thinking_token_id={thinking_token_id} "
          f"latent_feedback_adapter={has_adapter} "
          f"state_readonly_at_think={getattr(model, 'state_readonly_at_think', False)}")
    print(f"# {'rung':>4} {'n':>4} {'none':>7} {'R1':>7} {'R2':>7} "
          f"{'d(R1-none)':>11} {'d(R2-none)':>11}")

    rungs = [int(x) for x in args.rungs.split(",")]
    rows = []
    for n in rungs:
        path = args.data_glob.format(n=n)
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        res = eval_rung(model, tokenizer, records, thinking_token_id,
                        eos_token_id, args.max_gen, args.max_problems,
                        verbose=args.verbose)
        d1 = res["r1"] - res["none"]
        d2 = res["r2"] - res["none"]
        rows.append(dict(rung=n, **res, d1=d1, d2=d2))
        print(f"  {n:>4} {res['total']:>4} {res['none']:>7.3f} "
              f"{res['r1']:>7.3f} {res['r2']:>7.3f} {d1:>+11.3f} {d2:>+11.3f}",
              flush=True)

    # Machine-readable summary line for the orchestration script.
    print("RESULT_JSON " + json.dumps(dict(ckpt=args.ckpt, rows=rows)))


if __name__ == "__main__":
    main()
