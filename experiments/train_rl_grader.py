"""GRPO with execution-grounded code reward.

Each "step" of training:
  1. Sample a batch of B problems from MBPP.
  2. For each problem, generate N rollouts at temperature τ with the
     thinking gate engaged. Each rollout decides per-token whether to
     emit or insert a think token; emit tokens are sampled from the
     softmax (with the think-token slot masked out).
  3. Extract the generated code and run it through `code_grader.grade`.
     Reward = grader score ∈ [0, 1] (dense tier ladder gives variance
     even when no rollout passes).
  4. Compute GRPO advantages per group (problem) with ponder cost
     shaping (matches `thinking.compute_grpo_advantages` flag set).
  5. Policy gradient update with PPO-style ratio clip + KL to the
     base policy.

Distinct from `train_rl.py` (prediction-as-reward on codeparrot): the
reward here measures whether the emitted code actually works, which
is the only signal that can teach productive thinking.

Smoke test: `--steps 2 --batch 2 --grpo_n_group 2 --max_gen 32`
should complete in <2 minutes on a 5090.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.code_grader import (
    Problem, grade, load_mbpp, truncate_at_stop, _STOP_SEQUENCES, LOADERS,
)
from experiments.distill_solutions import extract_code_block
from experiments.eval_bracket_structure import build_model_from_ckpt


# ---------------------------------------------------------------------------
# Rollout: greedy or sampled, with thinking gate
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    prompt_len: int                 # how many tokens were prompt
    emit_token_ids: list[int]       # emitted tokens (excluding think tokens)
    emit_log_probs: list[float]     # log p of each sampled emit token
    emit_positions: list[int]       # absolute position (in the full id sequence)
                                    # of each emit token; useful for re-fetching
                                    # log-probs from the offline forward pass.
    full_ids: list[int]             # full id sequence: prompt + (emits + thinks)
    depth: int                      # total think tokens injected (over the rollout)
    text: str                       # decoded emit_token_ids
    reward: float = 0.0             # set by the grader
    tier: str = ""
    n_passed: int = 0
    n_tests: int = 0


@torch.no_grad()
def rollout_group_batched(model, tokenizer, prompt_ids: torch.Tensor, *,
                           n_rollouts: int,
                           thinking_token_id: int,
                           eos_token_id: int | None,
                           max_gen: int,
                           max_think_per_step: int,
                           total_think_budget: int,
                           emit_threshold: float,
                           gate_floor: float,
                           temperature: float,
                           min_emit_before_eos: int,
                           ) -> list[Rollout]:
    """Roll out `n_rollouts` parallel completions of the same prompt in a
    SINGLE batched forward per iteration.

    Each row decides per-step whether to emit or insert a think token (gate
    + ponder logic identical to `rollout_one`). The batch dimension is kept
    rectangular by lock-stepping: rows that "finished" (hit max_gen emits
    or EOS) append a harmless pad token (we re-use EOS) on every subsequent
    iteration, but we don't record those appendages for the policy update.

    Returns a list of N Rollout objects, one per row.
    """
    device = prompt_ids.device
    N = int(n_rollouts)
    # Tile prompt across N rows.
    ids = prompt_ids.expand(N, -1).contiguous()
    prompt_len = ids.shape[1]
    pad_id = int(eos_token_id) if eos_token_id is not None else 0
    # SPEEDUP: force single-pass FiLM at decode (T=1 makes K=3 pointless).
    _orig_film_bypass = getattr(model, "_film_bypass", False)
    if hasattr(model, "_film_bypass"):
        model._film_bypass = True

    emit_counts = torch.zeros(N, dtype=torch.long, device=device)
    think_counts_this_step = torch.zeros(N, dtype=torch.long, device=device)
    think_totals = torch.zeros(N, dtype=torch.long, device=device)
    finished = torch.zeros(N, dtype=torch.bool, device=device)

    emit_token_ids_per_row: list[list[int]] = [[] for _ in range(N)]
    emit_log_probs_per_row: list[list[float]] = [[] for _ in range(N)]
    emit_positions_per_row: list[list[int]] = [[] for _ in range(N)]

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    # Cap total iterations: even worst case, N*(max_gen + total_think_budget)
    # bounded by max_gen + total_think_budget (each iteration adds 1 token).
    max_iter = max_gen + total_think_budget + 4
    with autocast:
        for _ in range(max_iter):
            if bool(finished.all().item()):
                break
            logits = model(ids)
            next_logits = logits[:, -1, :].float()    # (N, V)

            gate_t = getattr(model, "_last_gate", None)
            if gate_t is None:
                gate = torch.ones(N, device=device)
            else:
                gate = gate_t[:, -1]
            gate_clamped = (gate if gate_floor <= 0
                            else gate.clamp_min(gate_floor))
            force_emit = (
                (think_counts_this_step >= max_think_per_step)
                | (think_totals >= total_think_budget)
                | finished  # finished rows always "emit" pad
            )
            want_emit = (gate_clamped >= emit_threshold) | force_emit

            # Sample emit tokens. Mask think token from softmax.
            next_logits[:, thinking_token_id] = -float("inf")
            # Per-row min_emit_before_eos.
            if min_emit_before_eos > 0 and eos_token_id is not None:
                mask = emit_counts < min_emit_before_eos
                if mask.any():
                    next_logits_clone = next_logits.clone()
                    next_logits_clone[mask, int(eos_token_id)] = -float("inf")
                    next_logits = next_logits_clone

            if temperature <= 0.0:
                emit_toks = next_logits.argmax(dim=-1)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                emit_toks = torch.multinomial(probs, num_samples=1).squeeze(-1)
            log_probs_full = F.log_softmax(
                next_logits / max(temperature, 1e-8), dim=-1)
            emit_lp = log_probs_full.gather(
                1, emit_toks.unsqueeze(1)).squeeze(1)        # (N,)

            # Per row append: think token or emit token.
            think_tok_t = torch.full_like(emit_toks, int(thinking_token_id))
            appended = torch.where(want_emit, emit_toks, think_tok_t)
            ids = torch.cat([ids, appended.unsqueeze(1)], dim=1)
            new_pos = ids.shape[1] - 1   # position of the just-appended token

            # Update per-row bookkeeping (host loop — small N).
            for i in range(N):
                if bool(finished[i].item()):
                    continue
                if bool(want_emit[i].item()):
                    tok = int(emit_toks[i].item())
                    is_eos = (eos_token_id is not None
                              and tok == int(eos_token_id))
                    if is_eos:
                        # Finish without recording the EOS token (avoids the
                        # "<|endoftext|>" -> syntax_error bug we hit before).
                        # Roll back: replace the appended EOS with a pad
                        # token so the batch stays clean — but actually,
                        # we're going to keep appending pad on finished rows
                        # anyway, and the appended token at this position is
                        # not used downstream for this row, so leave as-is.
                        finished[i] = True
                        continue
                    emit_token_ids_per_row[i].append(tok)
                    emit_log_probs_per_row[i].append(float(emit_lp[i].item()))
                    emit_positions_per_row[i].append(int(new_pos))
                    emit_counts[i] = emit_counts[i] + 1
                    think_counts_this_step[i] = 0
                    if int(emit_counts[i].item()) >= max_gen:
                        finished[i] = True
                else:
                    think_counts_this_step[i] = think_counts_this_step[i] + 1
                    think_totals[i] = think_totals[i] + 1

    if hasattr(model, "_film_bypass"):
        model._film_bypass = _orig_film_bypass
    # Build Rollout objects.
    out_rollouts = []
    full_ids_host = ids.cpu().tolist()
    for i in range(N):
        text = tokenizer.decode(emit_token_ids_per_row[i])
        out_rollouts.append(Rollout(
            prompt_len=prompt_len,
            emit_token_ids=emit_token_ids_per_row[i],
            emit_log_probs=emit_log_probs_per_row[i],
            emit_positions=emit_positions_per_row[i],
            full_ids=full_ids_host[i],
            depth=int(think_totals[i].item()),
            text=text,
        ))
    return out_rollouts


@torch.no_grad()
def rollout_one(model, tokenizer, prompt_ids: torch.Tensor, *,
                 thinking_token_id: int,
                 eos_token_id: int | None,
                 max_gen: int,
                 max_think_per_step: int,
                 total_think_budget: int,
                 emit_threshold: float,
                 gate_floor: float,
                 temperature: float,
                 min_emit_before_eos: int,
                 ) -> Rollout:
    """Run one rollout. `prompt_ids` is (1, T_prompt).

    Per emit step: forward the current sequence, read σ(gate) at the last
    position, decide emit vs think (with the same gate-floor clamp used at
    eval), sample the emit token from the temperature-softmax with the
    think-token slot masked out.
    """
    device = prompt_ids.device
    ids = prompt_ids.clone()
    prompt_len = ids.shape[1]

    emit_token_ids: list[int] = []
    emit_log_probs: list[float] = []
    emit_positions: list[int] = []
    think_total = 0

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast:
        for emit_idx in range(max_gen):
            # Inner think loop.
            thinks_this_step = 0
            while True:
                logits = model(ids)
                next_logits = logits[:, -1, :].float()  # fp32 for sampling

                gate_t = getattr(model, "_last_gate", None)
                if gate_t is None:
                    gate_val = 1.0
                else:
                    gate_val = float(gate_t[0, -1].item())
                gate_clamped = max(gate_val, gate_floor) if gate_floor > 0 else gate_val
                force_emit = (
                    thinks_this_step >= max_think_per_step
                    or think_total >= total_think_budget
                )
                if gate_clamped >= emit_threshold or force_emit:
                    break
                # Insert think token, loop.
                think_tok = torch.full((1, 1), thinking_token_id,
                                        dtype=ids.dtype, device=device)
                ids = torch.cat([ids, think_tok], dim=1)
                thinks_this_step += 1
                think_total += 1

            # Mask the think token slot before sampling — model should not
            # emit a think token via the normal vocab path.
            next_logits[..., thinking_token_id] = -float("inf")
            if (eos_token_id is not None and min_emit_before_eos > 0
                    and emit_idx < min_emit_before_eos):
                next_logits[..., int(eos_token_id)] = -float("inf")

            # Sample.
            if temperature <= 0.0:
                next_tok = int(next_logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())

            # Record log prob of the sampled token (under the temperature-
            # softmax we ACTUALLY sampled from — that's the policy ratio's
            # denominator).
            # EOS: stop WITHOUT including it in emit_token_ids — its decoded
            # text ("<|endoftext|>") would otherwise appear in the graded
            # code and produce SyntaxError (we saw this in the first smoke
            # run: every rollout was syntax_error because of trailing EOS).
            if eos_token_id is not None and next_tok == int(eos_token_id):
                break
            log_probs_full = F.log_softmax(next_logits / max(temperature, 1e-8),
                                            dim=-1)
            emit_log_probs.append(float(log_probs_full[0, next_tok].item()))
            emit_positions.append(int(ids.shape[1]))  # position where this token will sit
            emit_token_ids.append(next_tok)
            # Append.
            ids = torch.cat([ids, torch.tensor([[next_tok]], device=device,
                                                dtype=ids.dtype)], dim=1)

    text = tokenizer.decode(emit_token_ids)
    return Rollout(
        prompt_len=prompt_len,
        emit_token_ids=emit_token_ids,
        emit_log_probs=emit_log_probs,
        emit_positions=emit_positions,
        full_ids=ids[0].cpu().tolist(),
        depth=think_total,
        text=text,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_mbpp_prompt(problem: Problem) -> str:
    """Build a self-explanatory prompt for an MBPP problem.

    Format matches the SFT training format (sft_code.build_example):
        # {one-line description}
        {solution code}
    We additionally surface the first test as a Python-comment example
    so the model knows the function name and signature shape it should
    commit to (without seeing the answer).
    """
    desc = problem.prompt.strip().replace("\n", " ")
    # First test line, as a hint:
    first_test = ""
    try:
        ts = problem.tests.strip().split("\n")
        for line in ts:
            s = line.strip()
            if s.startswith("assert"):
                first_test = s
                break
    except Exception:
        pass
    parts = [f"# {desc}"]
    if first_test:
        parts.append(f"# Example: {first_test}")
    parts.append("")  # trailing newline
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Advantage computation (from scalar rewards)
# ---------------------------------------------------------------------------

def compute_grpo_advantages_from_rewards(
    rewards: torch.Tensor, depths: torch.Tensor, *,
    ponder_cost: float,
    ponder_shape: str = "quadratic",
    counterfactual: bool = True,
    ponder_warmup_scale: float = 1.0,
) -> torch.Tensor:
    """GRPO advantages from scalar (B, N) rewards + (B, N) depths.

    Counterfactual mode: clamps the task-reward component at zero (no
    rollout can "earn negative task reward" from thinking — but the
    depth cost still always applies). Then group-normalizes.

    ponder_warmup_scale: scales the depth cost by a curriculum factor
    in [0, 1] (ramped from 0 at the start of training to 1 after
    warmup steps; matches the prediction-as-reward setup's
    --grpo_ponder_warmup_steps).
    """
    if ponder_shape not in ("linear", "quadratic"):
        raise ValueError(f"unknown ponder_shape: {ponder_shape!r}")
    depth_cost = ponder_cost * ponder_warmup_scale * (
        depths ** 2 if ponder_shape == "quadratic" else depths
    )
    if counterfactual:
        # Task component: max(reward_d, 0) — thinking never makes the
        # task reward worse than not having thought. (The depth-0
        # baseline is implicit at 0 here since reward already
        # incorporates the rollout, and we don't have a separate
        # "no-think" measurement per problem.)
        task = torch.clamp_min(rewards, 0.0)
        full_reward = task - depth_cost
    else:
        full_reward = rewards - depth_cost
    means = full_reward.mean(dim=1, keepdim=True)
    stds = full_reward.std(dim=1, unbiased=False, keepdim=True) + 1e-8
    return (full_reward - means) / stds


# ---------------------------------------------------------------------------
# Policy loss with PPO clip
# ---------------------------------------------------------------------------

def policy_loss_for_rollouts_batched(
    rollouts: list[Rollout], advantages: list[float], model,
    *, clip_eps: float, thinking_token_id: int, temperature: float,
    pad_id: int,
    ref_model=None, kl_coef: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO clipped loss for a list of rollouts in ONE batched forward.

    Pads all rollout sequences to the max length, then runs a single forward
    pass on the (R, T_max) batch. Per-rollout, indexes out the logits at each
    rollout's `emit_positions`, computes new log-probs, forms the PPO ratio,
    and adds to the total loss.

    When `ref_model` and `kl_coef > 0` are set, also computes the KL
    penalty E[new_lp − ref_lp] at the same emit positions (single batched
    no_grad forward on the reference policy) and adds `kl_coef * KL` to
    the loss. This is the principled stability mechanism missing from
    the original implementation.

    Returns (loss, mean_ratio, mean_kl) — mean ratio and KL averaged
    across all emit positions across all rollouts.
    """
    device = next(model.parameters()).device
    R = len(rollouts)
    if R == 0:
        return (torch.zeros((), device=device, requires_grad=True),
                torch.tensor(1.0), torch.tensor(0.0))
    T_max = max(len(r.full_ids) for r in rollouts)
    padded = torch.full((R, T_max), int(pad_id), dtype=torch.long, device=device)
    for i, r in enumerate(rollouts):
        L = len(r.full_ids)
        padded[i, :L] = torch.tensor(r.full_ids, dtype=torch.long, device=device)

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast:
        logits = model(padded).float()              # (R, T_max, V)
    # KL-to-reference: one extra forward through the frozen ref model.
    ref_logits = None
    if ref_model is not None and kl_coef > 0.0:
        with torch.no_grad(), autocast:
            ref_logits = ref_model(padded).float()
    # Per-rollout, extract log-probs at emit positions.
    all_surrs = []
    all_ratios = []
    all_kls = []
    for i, r in enumerate(rollouts):
        if not r.emit_token_ids:
            continue
        adv = float(advantages[i])
        positions = torch.tensor(r.emit_positions, dtype=torch.long,
                                  device=device)
        # logits that PREDICTED the token at position p are at logits[i, p-1, :].
        pred_idx = positions - 1
        pred_logits = logits[i, pred_idx, :].clone()   # (n_emit, V)
        pred_logits[:, thinking_token_id] = -float("inf")
        new_log_probs = F.log_softmax(
            pred_logits / max(temperature, 1e-8), dim=-1)
        tok_ids = torch.tensor(r.emit_token_ids, dtype=torch.long,
                                device=device)
        new_lp = new_log_probs.gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        old_lp = torch.tensor(r.emit_log_probs, dtype=torch.float32,
                               device=device)
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surr = -torch.minimum(ratio * adv, clipped * adv).mean()
        all_surrs.append(surr)
        all_ratios.append(ratio.detach().mean())
        if ref_logits is not None:
            ref_pred_logits = ref_logits[i, pred_idx, :].clone()
            ref_pred_logits[:, thinking_token_id] = -float("inf")
            ref_log_probs = F.log_softmax(
                ref_pred_logits / max(temperature, 1e-8), dim=-1)
            ref_lp = ref_log_probs.gather(
                1, tok_ids.unsqueeze(1)).squeeze(1)
            # Schulman k3 KL estimator: (r - 1) - log r where
            # r = π_ref / π_new (so log r = ref_lp - new_lp). Always
            # non-negative, low-variance, unbiased for KL(new||ref)
            # under samples from π_new. The earlier k1 form
            # `(new_lp - ref_lp).mean()` is biased and can go negative
            # (turning the stability penalty into a small reward).
            # http://joschu.net/blog/kl-approx.html
            log_r = ref_lp - new_lp
            kl_k3 = (torch.exp(log_r) - 1.0) - log_r
            all_kls.append(kl_k3.mean())
    if not all_surrs:
        return (torch.zeros((), device=device, requires_grad=True),
                torch.tensor(1.0), torch.tensor(0.0))
    surr_loss = torch.stack(all_surrs).mean()
    mean_ratio = torch.stack(all_ratios).mean()
    if all_kls:
        mean_kl = torch.stack(all_kls).mean()
        total_loss = surr_loss + kl_coef * mean_kl
        return total_loss, mean_ratio, mean_kl.detach()
    return surr_loss, mean_ratio, torch.tensor(0.0, device=device)


def policy_loss_for_rollout(
    rollout: Rollout, advantage: float, model, prompt_ids: torch.Tensor,
    *, clip_eps: float, thinking_token_id: int, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-compute log-probs under the CURRENT policy for a rollout's emit
    tokens, then form the PPO clipped objective.

    Returns (loss, mean_ratio_for_logging). Mean is averaged across
    emit positions.

    Implementation detail: we run ONE forward pass on the full rollout
    sequence (prompt + all emit/think tokens), then index out the logits
    at each emit position to get new log-probs.
    """
    device = prompt_ids.device
    full_ids = torch.tensor(rollout.full_ids, dtype=prompt_ids.dtype,
                             device=device).unsqueeze(0)
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast:
        logits = model(full_ids).float()
    # For each emit at position p (where the emit token sits at full_ids[p]),
    # the logits that predicted it are at logits[:, p-1, :].
    losses = []
    ratios_for_log = []
    for emit_idx, (pos, old_logp, tok_id) in enumerate(
            zip(rollout.emit_positions,
                rollout.emit_log_probs,
                rollout.emit_token_ids)):
        pred_logits = logits[0, pos - 1, :].clone()
        pred_logits[thinking_token_id] = -float("inf")
        new_logp = F.log_softmax(pred_logits / max(temperature, 1e-8),
                                  dim=-1)[tok_id]
        # PPO ratio.
        ratio = torch.exp(new_logp - float(old_logp))
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        # Both surrogates; take the worse (PPO objective).
        surr = -torch.minimum(ratio * advantage, clipped * advantage)
        losses.append(surr)
        ratios_for_log.append(ratio.detach())
    if not losses:
        return (torch.zeros((), device=device, requires_grad=True),
                torch.tensor(1.0))
    return (torch.stack(losses).mean(),
            torch.stack(ratios_for_log).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load_ckpt", type=str, required=True)
    p.add_argument("--save_ckpt", type=str,
                    default="checkpoints/rl_grader_v5_pkm.pt")
    p.add_argument("--steps", type=int, default=200,
                    help="Number of GRPO updates.")
    p.add_argument("--batch", type=int, default=4,
                    help="Problems per step (B).")
    p.add_argument("--grpo_n_group", type=int, default=4,
                    help="Rollouts per problem (N).")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max_gen", type=int, default=80)
    p.add_argument("--max_think_per_step", type=int, default=4)
    p.add_argument("--total_think_budget", type=int, default=120)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.0,
                   help="KL-to-reference penalty coefficient (KL between "
                        "the current policy and a frozen reference policy "
                        "= the starting --load_ckpt). 0 = disabled. The "
                        "principled stability mechanism for grader-RL — "
                        "without it the policy can drift arbitrarily far "
                        "from the SFT base and catastrophically collapse "
                        "(observed at v1 step ~350, see GEMINI.md). "
                        "Recommended 0.05.")
    p.add_argument("--ponder_cost", type=float, default=0.005)
    p.add_argument("--ponder_shape", type=str, default="quadratic",
                    choices=["linear", "quadratic"])
    p.add_argument("--counterfactual", action="store_true", default=True)
    p.add_argument("--no_counterfactual", dest="counterfactual",
                    action="store_false")
    p.add_argument("--ponder_warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", type=str, default="mbpp",
                    help="Code-grader dataset: mbpp (374, legacy), "
                         "mbpp_all (974), mbpp_plus (378), "
                         "mbpp_combined (1352 = all+plus).")
    p.add_argument("--extract_code_block", action="store_true",
                    help="For distilled-SFT students that emit CoT + "
                         "```python ... ``` blocks: extract the python "
                         "fence and grade ONLY the extracted code "
                         "(otherwise the CoT prose breaks exec). Required "
                         "when --load_ckpt is a Qwen-distilled SFT ckpt.")
    p.add_argument("--smoke", action="store_true",
                    help="Run a tiny end-to-end smoke (2 steps, B=N=2, max_gen=32).")
    args = p.parse_args()

    if args.smoke:
        args.steps = 2
        args.batch = 2
        args.grpo_n_group = 2
        args.max_gen = 32
        args.save_ckpt = ""  # don't pollute checkpoints/

    torch.manual_seed(args.seed)

    # ---- Load model + tokenizer
    print(f"[rl-grader] loading {args.load_ckpt}")
    model, cfg = build_model_from_ckpt(args.load_ckpt)
    model = model.to("cuda")
    model.train()  # we'll switch to eval for rollouts then back for updates.
    thinking_token_id = int(cfg.get("thinking_token_id"))

    # Frozen reference policy for KL penalty (the principled stability
    # mechanism — without it, GRPO can drift arbitrarily far from the SFT
    # base and catastrophically collapse). Lazy-loaded only when needed.
    ref_model = None
    if args.kl_coef > 0.0:
        print(f"[rl-grader] loading reference policy from {args.load_ckpt}"
              f" (frozen, kl_coef={args.kl_coef})")
        ref_model, _ = build_model_from_ckpt(args.load_ckpt)
        ref_model = ref_model.to("cuda").eval()
        for _p in ref_model.parameters():
            _p.requires_grad = False

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    print(f"  thinking_token_id={thinking_token_id}  "
          f"vocab={cfg['vocab_size']}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ---- Load problems
    print(f"[rl-grader] loading dataset: {args.dataset}")
    if args.dataset not in LOADERS:
        raise SystemExit(
            f"unknown --dataset {args.dataset!r}; choose from {list(LOADERS)}")
    problems = LOADERS[args.dataset]()
    print(f"  loaded {len(problems)} problems; will sample {args.batch} per step")

    # ---- Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                             weight_decay=0.01)

    # ---- Loop
    rng = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    print(f"\n{'step':>5}  {'reward_mean':>10}  {'reward_max':>10}  "
          f"{'depth_mean':>10}  {'think_rate':>10}  {'pass_n':>6}  "
          f"{'ratio':>6}  {'loss':>8}  {'tier_hist':>30}")
    for step in range(1, args.steps + 1):
        # Sample B problems.
        idx = torch.randint(0, len(problems), (args.batch,), generator=rng).tolist()
        batch_problems = [problems[i] for i in idx]

        # Generate N rollouts per problem in ONE batched forward per iter.
        model.eval()
        all_rollouts: list[list[Rollout]] = []
        for prob in batch_problems:
            prompt_text = build_mbpp_prompt(prob)
            prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.cuda()
            group = rollout_group_batched(
                model, tok, prompt_ids,
                n_rollouts=args.grpo_n_group,
                thinking_token_id=thinking_token_id,
                eos_token_id=tok.eos_token_id,
                max_gen=args.max_gen,
                max_think_per_step=args.max_think_per_step,
                total_think_budget=args.total_think_budget,
                emit_threshold=args.emit_threshold,
                gate_floor=args.gate_floor,
                temperature=args.temperature,
                min_emit_before_eos=args.min_emit_before_eos,
            )
            all_rollouts.append(group)
        # Grade.
        tier_hist = {}
        for prob, group in zip(batch_problems, all_rollouts):
            for r in group:
                if args.extract_code_block:
                    # Distilled-student path: model emits CoT + ```python
                    # ... ```. Pull the code out of the fence; if no
                    # fence, fall back to raw text (will likely
                    # syntax_error).
                    extracted = extract_code_block(r.text)
                    code = extracted if extracted is not None else r.text
                else:
                    # Legacy / non-distilled path: model emits raw code
                    # continuation of the comment-style prompt header.
                    code = build_mbpp_prompt(prob) + r.text
                g_res = grade(prob, code, timeout_s=5)
                r.reward = float(g_res.score)
                r.tier = g_res.tier
                r.n_passed = g_res.n_passed
                r.n_tests = g_res.n_tests
                tier_hist[g_res.tier] = tier_hist.get(g_res.tier, 0) + 1

        # Build (B, N) reward + depth tensors.
        B, N = args.batch, args.grpo_n_group
        rewards = torch.tensor(
            [[r.reward for r in group] for group in all_rollouts])
        depths = torch.tensor(
            [[float(r.depth) for r in group] for group in all_rollouts])
        # Curriculum.
        warmup_scale = min(1.0, step / max(1, args.ponder_warmup_steps))
        advantages = compute_grpo_advantages_from_rewards(
            rewards, depths,
            ponder_cost=args.ponder_cost,
            ponder_shape=args.ponder_shape,
            counterfactual=args.counterfactual,
            ponder_warmup_scale=warmup_scale,
        )  # (B, N)

        # Policy update — one batched forward over ALL rollouts (B*N) in
        # the step, padded to the longest sequence. This replaces the prior
        # sequential per-rollout forward; expect ~10x speedup at B=4 N=4.
        model.train()
        opt.zero_grad(set_to_none=True)
        flat_rollouts = []
        flat_advantages = []
        for group, group_adv in zip(all_rollouts, advantages):
            for r, adv in zip(group, group_adv):
                if not r.emit_token_ids:
                    continue
                flat_rollouts.append(r)
                flat_advantages.append(float(adv.item()))
        if flat_rollouts:
            loss, mean_ratio, mean_kl = policy_loss_for_rollouts_batched(
                flat_rollouts, flat_advantages, model,
                clip_eps=args.clip_eps,
                thinking_token_id=thinking_token_id,
                temperature=args.temperature,
                pad_id=int(tok.eos_token_id) if tok.eos_token_id is not None else 0,
                ref_model=ref_model,
                kl_coef=args.kl_coef,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_val = float(loss.item())
            ratio_val = float(mean_ratio.item())
            kl_val = float(mean_kl.item())
        else:
            loss_val, ratio_val, kl_val = float("nan"), float("nan"), float("nan")

        # Logging.
        if step % args.log_every == 0:
            think_rate = (depths > 0).float().mean().item()
            pass_n = sum(1 for g in all_rollouts for r in g if r.tier == "pass")
            tier_str = "  ".join(f"{k}={v}" for k, v in sorted(tier_hist.items()))
            print(f"{step:>5}  {rewards.mean():>10.4f}  "
                  f"{rewards.max():>10.4f}  {depths.mean():>10.2f}  "
                  f"{think_rate:>10.3f}  {pass_n:>6d}  "
                  f"{ratio_val:>6.3f}  {loss_val:>8.4f}  "
                  f"kl={kl_val:+.4f}  "
                  f"{tier_str:>30}")

        if args.save_ckpt and step % args.save_every == 0:
            pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "step": step, "config": cfg},
                        args.save_ckpt)
            print(f"  [saved {args.save_ckpt}]")

    print(f"\n[rl-grader] done in {time.time() - t0:.0f}s")
    if args.save_ckpt:
        torch.save({"state_dict": model.state_dict(),
                    "step": args.steps, "config": cfg}, args.save_ckpt)
        print(f"final ckpt → {args.save_ckpt}")


if __name__ == "__main__":
    main()
