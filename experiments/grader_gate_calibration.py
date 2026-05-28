"""Grader-grounded gate calibration — the execution-grounded BCE gate target.

This is the synthesis of the long thinking-gate debugging arc
(THINKING_AUDIT_2026_05_28.md flaw C, PLAN_FLAW_C.md, THINKING_REVIEW_2026_05_28.md).

The problem it fixes
--------------------
`process_reward.compute_gate_calibration_loss` trained the gate with a BCE
target derived from **next-token logp** (`logp_think > logp_no_think`). The
audit found this FATALLY wrong (flaw C): it rewards sharpening locally-
uncertain SURFACE tokens (variable names, literals, whitespace), is nearly
free extra compute for an autoregressive model, and is structurally blind to
whether the emitted *function passes tests* (the terminal HumanEval/MBPP
objective). It is essentially a laundered entropy prior.

The two ways to train a DISCRETE gate decision
-----------------------------------------------
(a) Stochastic exploration (Bernoulli draw + log-prob policy gradient) — the
    `train_rl_grader.py --stochastic_gate` path. FAILED TWICE: exploration
    noise drowns the terminal reward (flat on weak base, regression on strong
    base). DO NOT rebuild it.
(b) SUPERVISED target via BCE. This is what we want, but the existing
    `gate_calibration` used the WRONG target (next-token logp). This module is
    option (b) with the CORRECT, task-grounded target.

The mechanism (this module)
---------------------------
For each (problem, gold) example with a thinking-capable model:
  1. Roll out a completion WITH the gate's think budget using the REAL deploy
     generator (`generate_with_retrieval_as_input`, additive-α, iterative
     re-decode — the same generator used at eval). Grade it → `score_with`.
     Record the absolute positions in the rollout id-sequence where the gate
     FIRED (chose to think).
  2. Roll out WITHOUT thinking (gate forced never-fire via emit_threshold=1.0
     and zero think budget). Grade → `score_without`.
  3. Rollout-level credit assignment:
       Δ = score_with − score_without
       Δ > 0  → thinking helped this problem → gate-fire decisions get BCE
                target = 1 (fire more).
       Δ < 0  → thinking hurt → BCE target = 0 (fire less).
       Δ == 0 → no signal → the problem is SKIPPED entirely (no loss).
     Optionally each problem's BCE term is weighted by |Δ| (`weight_by_delta`).
  4. A single teacher-forced GRAD forward over the with-think rollout's full
     id-sequence yields gate logits at the recorded fire positions; BCE them
     against the Δ-derived target.

Why this is correct where the logp proxy was wrong:
  * The target is the TERMINAL grader score on the *completed function* run
    through the verifier — not a local next-token distribution.
  * It is DETERMINISTIC (no Bernoulli noise → avoids the stochastic-gate
    failure mode that drowned the reward twice).
  * Train == deploy: the rollout uses the exact inference generator, so there
    is no K-at-once / iterative-re-decode mismatch (audit flaw B), and no
    pretrain-vs-SFT mechanism mismatch (flaw A).

Compute cost: 2 rollouts per problem (with-think + no-think) + 1 grad forward.
Bounded by `max_problems_per_step` and `max_gate_positions_per_problem`.

Credit-assignment coarseness (documented concern): the target is the SAME for
every fire decision in a with-think rollout — the rollout-level Δ cannot tell
*which* of the thinks helped. This is the standard rejection-style trade-off
(terminal credit, no per-decision attribution). It is the honest, low-variance
counterpart to mechanism-B PPO from PLAN_FLAW_C §1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@dataclass
class GraderGateCalibrationStats:
    """Diagnostics from one call to compute_grader_gate_calibration_loss."""
    n_problems: int = 0            # problems attempted this step
    n_used: int = 0               # problems with Δ != 0 (contributed a target)
    n_skipped_equal: int = 0      # problems with Δ == 0 (no signal)
    n_helped: int = 0             # problems where thinking helped (target 1)
    n_hurt: int = 0               # problems where thinking hurt (target 0)
    n_gate_positions: int = 0     # total fire positions BCE'd across problems
    mean_delta: float = 0.0       # mean Δ score over attempted problems
    mean_abs_delta: float = 0.0   # mean |Δ| over used problems
    mean_score_with: float = 0.0
    mean_score_without: float = 0.0
    mean_fire_sigma: float = 0.0  # mean σ(gate) at fire positions (pre-update)


# ---------------------------------------------------------------------------
# Rollout helpers (reuse the real deploy generator)
# ---------------------------------------------------------------------------
@dataclass
class _WithThinkRollout:
    """Result of one with-think rollout that records gate-fire positions."""
    full_ids: list[int]          # prompt + (emits interleaved with thinks)
    fire_positions: list[int]    # absolute idx in full_ids of the position
                                 # whose hidden state DECIDED to think (i.e. the
                                 # token immediately BEFORE an appended THINK).
    gen_text: str                # decoded emit-only text (think tokens stripped)
    think_total: int


def _rollout_with_think_record_fires(
    model, tok, problem, *,
    prompt_style: str,
    max_gen: int,
    total_think_budget: int,
    max_think_per_step: int,
    emit_threshold: float,
    gate_floor: float,
    min_emit_before_eos: int,
    thinking_token_id: int,
    additive: bool,
    max_T: int,
) -> _WithThinkRollout:
    """Run the REAL deploy generator with thinking on, recording the absolute
    positions where the gate fired (chose to think).

    We re-implement the thin generation loop here (rather than calling
    `generate_with_retrieval_as_input`) ONLY so we can record the precise
    id-sequence index of each gate-fire decision. The decode mechanics
    (additive-α retrieval-as-input, iterative re-decode, think/emit gating,
    EOS suppression) are kept byte-for-byte identical to
    `eval_humaneval.generate_with_retrieval_as_input`'s FULL-FORWARD path so
    train == deploy.
    """
    device = next(model.parameters()).device
    raw_prompt = problem.prompt
    if prompt_style == "sft_comment":
        prompt = "# Complete the following Python function.\n" + raw_prompt
    else:
        prompt = raw_prompt
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    room = max_gen + total_think_budget
    if len(prompt_ids) + room > max_T:
        prompt_ids = prompt_ids[-(max_T - room):]
    out = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = model.embed(out).clone()

    _alpha_p = getattr(model, "retrieval_input_alpha", None)
    alpha = float(_alpha_p.detach()) if _alpha_p is not None else 0.1

    fire_positions: list[int] = []
    emit_count = 0
    think_total = 0
    with torch.no_grad():
        while emit_count < max_gen:
            thinks_this_step = 0
            while True:
                logits = model(out, inputs_embeds=inputs_embeds)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[:, -1, :]
                gate_t = getattr(model, "_last_gate", None)
                gate_val = (1.0 if gate_t is None
                            else float(gate_t[0, -1].item()))
                if gate_floor > 0.0:
                    gate_val = max(gate_val, gate_floor)
                force_emit = (
                    thinks_this_step >= max_think_per_step
                    or think_total >= total_think_budget
                )
                if gate_val >= emit_threshold or force_emit:
                    break
                # FIRE: the gate at the current last position chose to think.
                # That deciding position is `out.shape[1] - 1`.
                fire_positions.append(int(out.shape[1] - 1))
                inj = getattr(model.memory, "_last_injection", None)
                if inj is None:
                    raise RuntimeError(
                        "memory._last_injection missing — needs WorkingMemory "
                        "stash (added 2026-05-19).")
                retrieved = inj[:, -1:, :].to(inputs_embeds.dtype)
                think_tok = torch.full((1, 1), int(thinking_token_id),
                                       dtype=out.dtype, device=device)
                out = torch.cat([out, think_tok], dim=1)
                if additive:
                    think_emb = model.embed(think_tok).to(inputs_embeds.dtype)
                    inj_input = think_emb + alpha * retrieved
                else:
                    inj_input = retrieved
                inputs_embeds = torch.cat([inputs_embeds, inj_input], dim=1)
                thinks_this_step += 1
                think_total += 1
            # EMIT
            next_logits = next_logits.clone()
            next_logits[..., int(thinking_token_id)] = -float("inf")
            if (tok.eos_token_id is not None
                    and min_emit_before_eos > 0
                    and emit_count < min_emit_before_eos):
                next_logits[..., int(tok.eos_token_id)] = -float("inf")
            next_tok = next_logits.argmax(dim=-1, keepdim=True)  # greedy (τ=0)
            out = torch.cat([out, next_tok], dim=1)
            emit_emb = model.embed(next_tok).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, emit_emb], dim=1)
            emit_count += 1
            if tok.eos_token_id is not None and (next_tok == tok.eos_token_id).all():
                break

    full_ids = out[0].tolist()
    prompt_len = len(prompt_ids)
    gen_only = [t for t in full_ids[prompt_len:] if t != int(thinking_token_id)]
    gen_text = tok.decode(gen_only, skip_special_tokens=True)
    return _WithThinkRollout(
        full_ids=full_ids,
        fire_positions=fire_positions,
        gen_text=gen_text,
        think_total=think_total,
    )


def _rollout_no_think_text(
    model, tok, problem, *,
    prompt_style: str,
    max_gen: int,
    min_emit_before_eos: int,
    thinking_token_id: int,
    additive: bool,
    max_T: int,
    generate_fn,
) -> str:
    """Run the deploy generator with thinking FORCED OFF (emit_threshold=1.0,
    zero budget) → decoded emit-only text. Reuses the real generator."""
    device = next(model.parameters()).device
    raw_prompt = problem.prompt
    if prompt_style == "sft_comment":
        prompt = "# Complete the following Python function.\n" + raw_prompt
    else:
        prompt = raw_prompt
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) + max_gen > max_T:
        prompt_ids = prompt_ids[-(max_T - max_gen):]
    prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                            device=device).unsqueeze(0)
    with torch.no_grad():
        gen, _ = generate_fn(
            model, prompt_t, max_gen=max_gen,
            temperature=0.0, eos_token_id=tok.eos_token_id,
            thinking_token_id=thinking_token_id,
            max_think_per_step=0,
            total_think_budget=0,
            emit_threshold=1.0,
            min_emit_before_eos=min_emit_before_eos,
            gate_floor=0.0,
            additive=additive,
        )
    gen_only = [t for t in gen[0, len(prompt_ids):].tolist()
                if t != int(thinking_token_id)]
    return tok.decode(gen_only, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Teacher-forced gate-logit forward (the GRAD path)
# ---------------------------------------------------------------------------
def _gate_logits_at_positions(
    model,
    full_ids: torch.Tensor,         # (1, L) recorded rollout id-sequence
    fire_positions: torch.Tensor,   # (P,) absolute indices into full_ids
    *,
    thinking_token_id: int,
    retrieval_as_input: bool,
    additive: bool,
) -> torch.Tensor:
    """Single GRAD-enabled forward over the recorded id-sequence; return the
    pre-sigmoid gate logits at `fire_positions`.

    In retrieval-as-input mode we reconstruct `inputs_embeds` the same way
    `process_reward._retrieval_input_embeds` does: one no_grad forward to
    populate `memory._last_injection`, then add the lagged injection (scaled
    by α) at think positions. The SECOND, grad-enabled forward feeds those
    embeds so the gate logits carry gradient to the gate head + trunk.
    """
    if retrieval_as_input:
        with torch.no_grad():
            _ = model(full_ids)
        inj = model.memory._last_injection             # (1, L, d), detached
        base_emb = model.embed(full_ids)
        is_think = (full_ids == int(thinking_token_id)).unsqueeze(-1)
        shifted_inj = torch.cat(
            [torch.zeros_like(inj[:, :1]), inj[:, :-1]], dim=1)
        if additive:
            alpha = model.retrieval_input_alpha
            inputs_embeds = (
                base_emb
                + is_think.to(base_emb.dtype) * alpha
                * shifted_inj.to(base_emb.dtype))
        else:
            # v5/v6 destructive replacement at think positions.
            inputs_embeds = torch.where(
                is_think, shifted_inj.to(base_emb.dtype), base_emb)
        out = model(full_ids, inputs_embeds=inputs_embeds, return_gate=True)
    else:
        out = model(full_ids, return_gate=True)
    # _last_gate_logits is stashed by _maybe_gate during the forward.
    gate_logits = getattr(model, "_last_gate_logits", None)
    if gate_logits is None:
        raise RuntimeError(
            "model._last_gate_logits not populated — the model must have a "
            "gate head (use_gate=True) for grader-gate-calibration.")
    return gate_logits[0, fire_positions]               # (P,)


# ---------------------------------------------------------------------------
# Main loss
# ---------------------------------------------------------------------------
def compute_grader_gate_calibration_loss(
    model,
    tok,
    problems: list,
    *,
    grade_fn: Callable,                 # code_grader.grade-compatible
    generate_fn: Callable,              # generate_with_retrieval_as_input-compat
    thinking_token_id: int,
    prompt_style: str = "sft_comment",
    extract_code_block: bool = True,
    retrieval_as_input: bool = True,
    additive: bool = True,
    max_gen: int = 256,
    total_think_budget: int = 120,
    max_think_per_step: int = 8,
    emit_threshold: float = 0.5,
    gate_floor: float = 0.0,
    min_emit_before_eos: int = 30,
    max_T: int = 2048,
    timeout_s: int = 7,
    max_problems_per_step: int = 4,
    max_gate_positions_per_problem: int = 64,
    weight_by_delta: bool = False,
    build_completion_fn: Optional[Callable] = None,
) -> tuple[torch.Tensor, GraderGateCalibrationStats]:
    """Compute the grader-grounded gate-calibration BCE loss for a batch of
    problems.

    Loss (per problem p with Δ_p != 0, target t_p ∈ {0,1}, fire positions
    F_p, gate logits g):
        L_p = mean_{i in F_p} BCE_with_logits(g_i, t_p)   [optionally · |Δ_p|]
        L   = mean_p L_p

    Returns `(loss, stats)`. When no problem yields a usable target / fire
    position, the loss is a zero scalar with requires_grad=False (caller
    should still call .backward()-safe code: it just contributes nothing).
    """
    if build_completion_fn is None:
        from experiments.probe_think_grader_reward import (
            _build_completion_for_grading as build_completion_fn)

    device = next(model.parameters()).device
    problems = problems[:max_problems_per_step]

    was_training = model.training
    model.eval()  # rollouts are deterministic; eval disables any dropout

    per_problem_terms: list[torch.Tensor] = []
    stats = GraderGateCalibrationStats(n_problems=len(problems))
    deltas: list[float] = []
    abs_deltas: list[float] = []
    scores_with: list[float] = []
    scores_without: list[float] = []
    fire_sigmas: list[float] = []

    for problem in problems:
        # --- with-think rollout (records fire positions).
        wt = _rollout_with_think_record_fires(
            model, tok, problem,
            prompt_style=prompt_style, max_gen=max_gen,
            total_think_budget=total_think_budget,
            max_think_per_step=max_think_per_step,
            emit_threshold=emit_threshold, gate_floor=gate_floor,
            min_emit_before_eos=min_emit_before_eos,
            thinking_token_id=thinking_token_id, additive=additive,
            max_T=max_T)
        # --- no-think rollout.
        nt_text = _rollout_no_think_text(
            model, tok, problem,
            prompt_style=prompt_style, max_gen=max_gen,
            min_emit_before_eos=min_emit_before_eos,
            thinking_token_id=thinking_token_id, additive=additive,
            max_T=max_T, generate_fn=generate_fn)

        # --- grade both.
        wt_prob, wt_comp = build_completion_fn(
            problem, wt.gen_text, extract_code_block=extract_code_block)
        nt_prob, nt_comp = build_completion_fn(
            problem, nt_text, extract_code_block=extract_code_block)
        score_with = grade_fn(wt_prob, wt_comp, timeout_s=timeout_s).score
        score_without = grade_fn(nt_prob, nt_comp, timeout_s=timeout_s).score
        delta = float(score_with - score_without)
        deltas.append(delta)
        scores_with.append(score_with)
        scores_without.append(score_without)

        if abs(delta) < 1e-9:
            stats.n_skipped_equal += 1
            continue
        if len(wt.fire_positions) == 0:
            # Thinking changed the score but no recorded fire? Only possible
            # if the score delta came from a different source (shouldn't, but
            # guard): no gate decision to attribute → skip.
            stats.n_skipped_equal += 1
            continue

        target_val = 1.0 if delta > 0 else 0.0
        if delta > 0:
            stats.n_helped += 1
        else:
            stats.n_hurt += 1
        abs_deltas.append(abs(delta))

        fire_positions = wt.fire_positions[:max_gate_positions_per_problem]
        full_ids = torch.tensor(wt.full_ids, dtype=torch.long,
                                device=device).unsqueeze(0)
        fp = torch.tensor(fire_positions, dtype=torch.long, device=device)

        gate_logits_at = _gate_logits_at_positions(
            model, full_ids, fp,
            thinking_token_id=thinking_token_id,
            retrieval_as_input=retrieval_as_input, additive=additive)  # (P,)
        target = torch.full_like(gate_logits_at, target_val)
        term = F.binary_cross_entropy_with_logits(
            gate_logits_at.float(), target.float())
        if weight_by_delta:
            term = term * abs(delta)
        per_problem_terms.append(term)
        stats.n_used += 1
        stats.n_gate_positions += len(fire_positions)
        with torch.no_grad():
            fire_sigmas.append(
                float(torch.sigmoid(gate_logits_at).mean().item()))

    if was_training:
        model.train()

    stats.mean_delta = float(sum(deltas) / max(1, len(deltas)))
    stats.mean_abs_delta = float(sum(abs_deltas) / max(1, len(abs_deltas)))
    stats.mean_score_with = float(sum(scores_with) / max(1, len(scores_with)))
    stats.mean_score_without = float(
        sum(scores_without) / max(1, len(scores_without)))
    stats.mean_fire_sigma = float(sum(fire_sigmas) / max(1, len(fire_sigmas)))

    if not per_problem_terms:
        zero = torch.zeros((), device=device, requires_grad=False)
        return zero, stats
    loss = torch.stack(per_problem_terms).mean()
    return loss, stats


# ---------------------------------------------------------------------------
# Optional KL-to-reference penalty (cheap stability anchor, mirrors
# train_rl_grader.py's v2 KL term). Computed on the with-think rollout's full
# id-sequence: one no_grad forward on the frozen reference, one grad forward on
# the policy, token-level KL averaged over the sequence.
# ---------------------------------------------------------------------------
def _kl_to_ref(model, ref_model, full_ids: torch.Tensor,
               base_vocab: int | None) -> torch.Tensor:
    pol = model(full_ids)
    pol = pol[0] if isinstance(pol, tuple) else pol
    with torch.no_grad():
        ref = ref_model(full_ids)
        ref = ref[0] if isinstance(ref, tuple) else ref
    if base_vocab is not None:
        pol = pol[..., :base_vocab]
        ref = ref[..., :base_vocab]
    log_p = F.log_softmax(pol.float(), dim=-1)
    log_q = F.log_softmax(ref.float(), dim=-1)
    # KL(policy || ref) = sum p (log p - log q)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    return kl


# ---------------------------------------------------------------------------
# Standalone trainer
# ---------------------------------------------------------------------------
def main():
    import argparse
    import os
    import pathlib
    import sys
    import time

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--load_ckpt", type=str, required=True)
    p.add_argument("--save_ckpt", type=str,
                   default="checkpoints/grader_gate_calibration.pt")
    p.add_argument("--dataset", type=str, default="mbpp_combined")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--max_problems_per_step", type=int, default=4)
    p.add_argument("--max_gate_positions_per_problem", type=int, default=64)
    p.add_argument("--weight_by_delta", action="store_true", default=False)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--total_think_budget", type=int, default=120)
    p.add_argument("--max_think_per_step", type=int, default=8)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--prompt_style", type=str, default="sft_comment",
                   choices=["humaneval", "sft_comment"])
    p.add_argument("--extract_code_block", action="store_true", default=True)
    p.add_argument("--no_extract_code_block", dest="extract_code_block",
                   action="store_false")
    p.add_argument("--timeout_s", type=int, default=7)
    p.add_argument("--kl_coef", type=float, default=0.05,
                   help="KL-to-frozen-reference penalty (ref = --load_ckpt). "
                        "0 disables (skips loading the reference, saves mem).")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--activation_checkpointing", action="store_true",
                   default=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 2
        args.max_problems_per_step = 2
        args.max_gen = 32
        args.total_think_budget = 16
        args.save_ckpt = ""

    import torch as _torch
    from experiments.code_grader import grade as grade_fn, LOADERS
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.eval_humaneval import generate_with_retrieval_as_input
    from experiments.probe_think_grader_reward import (
        _build_completion_for_grading)
    from transformers import AutoTokenizer

    _torch.manual_seed(args.seed)
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    print(f"[ggc] loading {args.load_ckpt}")
    model, cfg = build_model_from_ckpt(args.load_ckpt)
    model = model.to(device)
    if args.activation_checkpointing and hasattr(
            model, "activation_checkpointing"):
        model.activation_checkpointing = True
    thinking_token_id = int(cfg.get("thinking_token_id"))
    additive = bool(cfg.get("retrieval_input_additive", True))
    base_vocab = cfg.get("vocab_size", None)
    max_T = cfg["max_T"] if cfg.get("max_T", 0) and cfg["max_T"] > 0 else 2048

    ref_model = None
    if args.kl_coef > 0.0:
        print(f"[ggc] loading frozen reference (kl_coef={args.kl_coef})")
        ref_model, _ = build_model_from_ckpt(args.load_ckpt)
        ref_model = ref_model.to(device).eval()
        for _pp in ref_model.parameters():
            _pp.requires_grad = False

    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    if args.dataset not in LOADERS:
        raise SystemExit(f"unknown --dataset {args.dataset!r}")
    problems = LOADERS[args.dataset]()
    print(f"[ggc] dataset={args.dataset} n={len(problems)}  "
          f"thinking_token_id={thinking_token_id} additive={additive}")

    opt = _torch.optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.95), weight_decay=0.01)
    rng = _torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    print(f"\n{'step':>5}  {'loss':>9}  {'Δmean':>8}  {'used':>5}  "
          f"{'help/hurt':>10}  {'firePos':>8}  {'fireσ':>7}  {'kl':>8}")
    for step in range(1, args.steps + 1):
        idx = _torch.randint(0, len(problems),
                             (args.max_problems_per_step,),
                             generator=rng).tolist()
        batch = [problems[i] for i in idx]

        loss, stats = compute_grader_gate_calibration_loss(
            model, tok, batch,
            grade_fn=grade_fn,
            generate_fn=generate_with_retrieval_as_input,
            thinking_token_id=thinking_token_id,
            prompt_style=args.prompt_style,
            extract_code_block=args.extract_code_block,
            retrieval_as_input=True, additive=additive,
            max_gen=args.max_gen,
            total_think_budget=args.total_think_budget,
            max_think_per_step=args.max_think_per_step,
            emit_threshold=args.emit_threshold,
            gate_floor=args.gate_floor,
            min_emit_before_eos=args.min_emit_before_eos,
            max_T=max_T, timeout_s=args.timeout_s,
            max_problems_per_step=args.max_problems_per_step,
            max_gate_positions_per_problem=args.max_gate_positions_per_problem,
            weight_by_delta=args.weight_by_delta,
            build_completion_fn=_build_completion_for_grading,
        )

        kl_val = 0.0
        total = loss
        if (ref_model is not None and args.kl_coef > 0.0
                and stats.n_used > 0):
            # KL on the last used problem's sequence is enough as a cheap
            # anchor; recompute it on a fresh forward (the loss path used
            # eval mode + no_grad reconstruct). We re-roll the first batch
            # problem deterministically for a stable KL anchor.
            model.eval()
            anchor = _rollout_with_think_record_fires(
                model, tok, batch[0],
                prompt_style=args.prompt_style, max_gen=args.max_gen,
                total_think_budget=args.total_think_budget,
                max_think_per_step=args.max_think_per_step,
                emit_threshold=args.emit_threshold, gate_floor=args.gate_floor,
                min_emit_before_eos=args.min_emit_before_eos,
                thinking_token_id=thinking_token_id, additive=additive,
                max_T=max_T)
            model.train()
            fids = _torch.tensor(anchor.full_ids, dtype=_torch.long,
                                 device=device).unsqueeze(0)
            kl = _kl_to_ref(model, ref_model, fids, base_vocab)
            kl_val = float(kl.detach().item())
            total = total + args.kl_coef * kl

        if total.requires_grad:
            opt.zero_grad(set_to_none=True)
            total.backward()
            if args.grad_clip > 0:
                _torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                args.grad_clip)
            opt.step()

        if step % args.log_every == 0:
            print(f"{step:>5}  {float(loss.detach()):>9.4f}  "
                  f"{stats.mean_delta:>+8.4f}  {stats.n_used:>5}  "
                  f"{stats.n_helped:>4}/{stats.n_hurt:<5}  "
                  f"{stats.n_gate_positions:>8}  "
                  f"{stats.mean_fire_sigma:>7.3f}  {kl_val:>8.4f}")

        if (args.save_ckpt and args.save_every > 0
                and step % args.save_every == 0):
            os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
            _torch.save({"state_dict": model.state_dict(), "step": step,
                         "config": cfg}, args.save_ckpt)
            print(f"[ggc] saved {args.save_ckpt} @ step {step}")

    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
        _torch.save({"state_dict": model.state_dict(), "step": args.steps,
                     "config": cfg}, args.save_ckpt)
        print(f"[ggc] done in {time.time()-t0:.1f}s → {args.save_ckpt}")
    else:
        print(f"[ggc] smoke done in {time.time()-t0:.1f}s (no save)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
