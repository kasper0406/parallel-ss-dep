"""Thinking-demonstration harness: pass@1 with vs without thinking across a
difficulty ladder of arithmetic-chain reasoning tasks.

The single most important question for the thinking primitive: is there ANY
task where the model's "thinking" mechanism measurably improves task
performance? The hypothesis is that thinking is EMERGENT — it cannot help
where the base fails completely (floor) or succeeds trivially (ceiling), but
should help in the COMPETENCE BAND (tasks the model can partially do).

This harness sweeps `data/synth_arith_ladder_n{1..6}.jsonl` (chain lengths
1..6) and, for each rung, runs the SAME ckpt twice with the SAME generation
config except the think flag:

  * without-think : plain greedy decode (no gate, no [THINKING] tokens)
  * with-think    : retrieval_as_input generator (gate decides when to think)

Both are greedy (temperature 0), graded deterministically by code_grader.
Per-rung pass counts (not just rates) are reported because n=80 is small.

CONFOUND NOTE: with-think gives the model extra forward passes (the think
steps) that the without-think baseline lacks. A lift could therefore be
"more compute" rather than "thinking specifically". To separate these, pass
`--extra_compute_baseline`: a third condition that grants the without-think
decode the same number of *extra emit positions* the think condition spent
on thinking is NOT meaningful for a closed-form answer, so instead we report
the mean think tokens spent per rung so the confound magnitude is visible.

Usage:
    PYTHONPATH=. .venv/bin/python -u experiments/eval_thinking_ladder.py \
        --ckpt checkpoints/rl_grader_phase_c_v2_step300.pt \
        --rungs 1,2,3,4,5,6 --max_gen 128 \
        --out THINKING_LADDER_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.code_grader import grade, load_synth_reasoning
from experiments.distill_solutions import extract_code_block as _extract_code
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate, generate_with_retrieval_as_input
from experiments.layers import _FlaWrapper


def _set_state_readonly(model, enabled: bool) -> None:
    """Toggle the Phase-2 state-readonly β-masking on every DeltaNet block
    at inference. The model must have been built with the b_proj hook
    installed (force_state_readonly=True); this flips the per-wrapper
    `state_readonly_at_think` bool the hook reads. When False the hook is a
    no-op (byte-identical to a model that never had it), so the same model
    instance serves both the with-think (state-write) and
    with-think+state-readonly conditions."""
    model.state_readonly_at_think = bool(enabled)
    for blk in model.blocks:
        attn = blk.attn
        if isinstance(attn, _FlaWrapper) and hasattr(attn, "_b_proj_hook_handle"):
            attn.state_readonly_at_think = bool(enabled)


def _run_condition(
    model, cfg, tok, problems, *, use_thinking, generator,
    max_gen, temperature, total_think_budget, max_think_per_step,
    emit_threshold, gate_floor, min_emit_before_eos, prompt_style,
    extract_code_block, thinking_token_id, grader_timeout_s,
):
    """Run a single (with/without think) pass over a rung. Returns a dict
    with pass count, tier histogram, and mean think tokens spent."""
    n_passed = 0
    n_total = 0
    tier_hist: dict[str, int] = {}
    think_tokens_total = 0
    # When the prompt itself opens a ```python fence, the model's
    # continuation is the fence BODY; we must re-prepend the opener before
    # running extract_code_block (which searches for a fence).
    fence_opened = prompt_style == "code_fence"
    for prob in problems:
        raw_prompt = prob.prompt
        if prompt_style == "sft_comment":
            prompt = f"# {raw_prompt.strip()}\n"
        elif prompt_style == "code_fence":
            prompt = f"# {raw_prompt.strip()}\n\n```python\n"
        else:
            prompt = raw_prompt
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        eff_max_T = cfg["max_T"] if cfg["max_T"] > 0 else 2048
        thinks_reserve = (total_think_budget if use_thinking else 0)
        room = max_gen + thinks_reserve
        if len(prompt_ids) + room > eff_max_T:
            prompt_ids = prompt_ids[-(eff_max_T - room):]
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                device="cuda").unsqueeze(0)

        gen_kw = dict(
            max_gen=max_gen, temperature=temperature,
            eos_token_id=tok.eos_token_id,
            use_thinking=use_thinking,
            thinking_token_id=thinking_token_id,
            max_think_per_step=max_think_per_step,
            total_think_budget=total_think_budget,
            emit_threshold=emit_threshold,
            min_emit_before_eos=min_emit_before_eos,
            gate_floor=gate_floor,
        )
        if use_thinking and generator == "retrieval_as_input":
            gen_kw.pop("use_thinking")
            gen_kw["additive"] = cfg.get("retrieval_input_additive", False)
            gen, _ = generate_with_retrieval_as_input(model, prompt_t, **gen_kw)
        else:
            # without-think (or standard generator). Force use_thinking off
            # so no gate / [THINKING] tokens are inserted.
            gen_kw["use_thinking"] = use_thinking
            gen, _ = generate(model, prompt_t, **gen_kw)

        gen_only = gen[0, len(prompt_ids):].tolist()
        if thinking_token_id is not None:
            n_think = sum(1 for t in gen_only if t == int(thinking_token_id))
            think_tokens_total += n_think
            gen_only = [t for t in gen_only if t != int(thinking_token_id)]
        gen_text = tok.decode(gen_only, skip_special_tokens=True)
        if extract_code_block:
            search_text = ("```python\n" + gen_text) if fence_opened else gen_text
            code = _extract_code(search_text)
            full_code = code if code is not None else gen_text
        else:
            full_code = prompt + gen_text

        g_res = grade(prob, full_code, timeout_s=grader_timeout_s)
        tier_hist[g_res.tier] = tier_hist.get(g_res.tier, 0) + 1
        n_total += 1
        if g_res.tier == "pass":
            n_passed += 1

    return dict(
        n_passed=n_passed,
        n_total=n_total,
        pass_rate=n_passed / max(1, n_total),
        tier_hist=tier_hist,
        mean_think_tokens=think_tokens_total / max(1, n_total),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--rungs", type=str, default="1,2,3,4,5,6",
                   help="Comma-separated chain lengths to eval.")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--data_prefix", type=str, default="synth_arith_ladder_n")
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_gen", type=int, default=128)
    p.add_argument("--max_think_per_step", type=int, default=8)
    p.add_argument("--total_think_budget", type=int, default=400)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--prompt_style", type=str, default="code_fence",
                   choices=["raw", "sft_comment", "code_fence"])
    p.add_argument("--extract_code_block", action="store_true", default=True)
    p.add_argument("--no_extract_code_block", dest="extract_code_block",
                   action="store_false")
    p.add_argument("--generator", type=str, default="retrieval_as_input",
                   choices=["standard", "retrieval_as_input"])
    p.add_argument("--grader_timeout_s", type=int, default=5)
    p.add_argument("--out", type=str, default="THINKING_LADDER_RESULTS.json")
    p.add_argument("--state_readonly_3way", action="store_true",
                   help="Add a THIRD condition: with-think + state-readonly "
                        "(β=0 at think positions). Forces state-readonly ON "
                        "at inference regardless of how the ckpt was trained "
                        "— the off-distribution rescue probe for the "
                        "'thinking corrupts recall' failure mode.")
    args = p.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    print(f"[ladder] ckpt={args.ckpt}  rungs={rungs}")
    print(f"[ladder] generator(with-think)={args.generator}  "
          f"max_gen={args.max_gen}  temp={args.temperature}")

    # When running the 3-way comparison, build with the b_proj β-masking
    # hook installed (force_state_readonly=True). We then toggle the
    # per-wrapper flag OFF for the state-write condition and ON for the
    # state-readonly condition — the same model instance serves all three.
    model, cfg = build_model_from_ckpt(
        args.ckpt,
        force_state_readonly=True if args.state_readonly_3way else None)
    model = model.to("cuda").eval()
    if args.state_readonly_3way:
        # Default state for the non-SR conditions: OFF.
        _set_state_readonly(model, False)
    has_gate = hasattr(model, "gate_head")
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")
    if not has_gate or thinking_token_id is None:
        raise RuntimeError("ckpt lacks gate / thinking_token_id; cannot run "
                           "the with-think condition")

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M",
                                        trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    common = dict(
        max_gen=args.max_gen, temperature=args.temperature,
        total_think_budget=args.total_think_budget,
        max_think_per_step=args.max_think_per_step,
        emit_threshold=args.emit_threshold, gate_floor=args.gate_floor,
        min_emit_before_eos=args.min_emit_before_eos,
        prompt_style=args.prompt_style,
        extract_code_block=args.extract_code_block,
        thinking_token_id=thinking_token_id,
        grader_timeout_s=args.grader_timeout_s,
    )

    results = []
    t0 = time.time()
    for n in rungs:
        path = pathlib.Path(args.data_dir) / f"{args.data_prefix}{n}.jsonl"
        problems = load_synth_reasoning(str(path))
        if args.max_problems is not None:
            problems = problems[:args.max_problems]

        with torch.no_grad():
            no_think = _run_condition(
                model, cfg, tok, problems,
                use_thinking=False, generator="standard", **common)
            # with-think (state-WRITE — the broken baseline).
            if args.state_readonly_3way:
                _set_state_readonly(model, False)
            with_think = _run_condition(
                model, cfg, tok, problems,
                use_thinking=True, generator=args.generator, **common)

            with_think_sr = None
            if args.state_readonly_3way:
                # with-think + state-READONLY (β=0 at think positions).
                _set_state_readonly(model, True)
                with_think_sr = _run_condition(
                    model, cfg, tok, problems,
                    use_thinking=True, generator=args.generator, **common)
                _set_state_readonly(model, False)

        delta = with_think["n_passed"] - no_think["n_passed"]
        rec = dict(
            n_steps=n, n=len(problems),
            no_think=no_think, with_think=with_think, delta=delta,
        )
        if with_think_sr is not None:
            rec["with_think_sr"] = with_think_sr
            rec["delta_sr_vs_write"] = (
                with_think_sr["n_passed"] - with_think["n_passed"])
            rec["delta_sr_vs_nothink"] = (
                with_think_sr["n_passed"] - no_think["n_passed"])
        results.append(rec)
        line = (f"  n_steps={n:>1}  n={len(problems):>3}  "
                f"no_think={no_think['n_passed']:>3}/{no_think['n_total']} "
                f"({no_think['pass_rate']:.3f})   "
                f"with_think={with_think['n_passed']:>3}/{with_think['n_total']} "
                f"({with_think['pass_rate']:.3f})")
        if with_think_sr is not None:
            line += (f"   with_think_sr="
                     f"{with_think_sr['n_passed']:>3}/{with_think_sr['n_total']} "
                     f"({with_think_sr['pass_rate']:.3f})")
        line += (f"   delta={delta:+d}   "
                 f"think_tok/prob={with_think['mean_think_tokens']:.1f}  "
                 f"({time.time()-t0:.0f}s)")
        print(line)

    has_sr = args.state_readonly_3way
    print("\n=== HEADLINE: pass@1 with vs without thinking ===")
    hdr = (f"{'n_steps':>7} | {'n':>3} | {'no_think':>10} | "
           f"{'with_think':>11}")
    if has_sr:
        hdr += f" | {'wt+state_ro':>12}"
    hdr += f" | {'delta':>6} | {'think_tok':>9}"
    print(hdr)
    print("-" * (len(hdr)))
    for r in results:
        row = (f"{r['n_steps']:>7} | {r['n']:>3} | "
               f"{r['no_think']['n_passed']:>3}/{r['no_think']['n_total']:<3} "
               f"{r['no_think']['pass_rate']:>5.3f}" + " | "
               f"{r['with_think']['n_passed']:>3}/{r['with_think']['n_total']:<3} "
               f"{r['with_think']['pass_rate']:>5.3f}")
        if has_sr and "with_think_sr" in r:
            row += (" | "
                    f"{r['with_think_sr']['n_passed']:>4}/"
                    f"{r['with_think_sr']['n_total']:<3} "
                    f"{r['with_think_sr']['pass_rate']:>5.3f}")
        row += (" | "
                f"{r['delta']:>+6d} | "
                f"{r['with_think']['mean_think_tokens']:>9.1f}")
        print(row)

    out_path = pathlib.Path(args.out)
    with open(out_path, "w") as f:
        json.dump(dict(ckpt=args.ckpt, config=vars(args), results=results),
                  f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
