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
import concurrent.futures as _cf
import math
import os
import pathlib
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Distributed (torchrun) helpers
# ---------------------------------------------------------------------------
def setup_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed from torchrun env vars if present.

    Returns `(rank, world_size, local_rank)`. When not launched under
    torchrun, returns `(0, 1, 0)` and skips init_process_group so the
    script runs identically to the single-GPU path.
    """
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def all_reduce_mean(x: float, world_size: int) -> float:
    """Average a Python float across ranks (no-op if world_size == 1)."""
    if world_size <= 1:
        return x
    t = torch.tensor([x], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / world_size


def all_reduce_sum_int(x: int, world_size: int) -> int:
    """Sum an integer across ranks (no-op if world_size == 1)."""
    if world_size <= 1:
        return x
    t = torch.tensor([x], device="cuda", dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())

from experiments.code_grader import (
    Problem, grade, load_mbpp, load_synth_reasoning,
    truncate_at_stop, _STOP_SEQUENCES, LOADERS,
)
from experiments.bf16_optim import BF16StateAdamW
from experiments.curriculum import ProblemDifficultyEMA, merge_rank_updates
from experiments.distill_solutions import extract_code_block
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.iterative_repair import (
    build_repair_prompt, group_became_variance_bearing, select_repair_targets,
)
from experiments.rl_multiturn import (
    JudgeCandidate, Trajectory, Turn, apply_judge_to_group,
    assemble_flat_rollouts, compute_trajectory_reward,
    group_is_variance_bearing,
)


def all_gather_object_list(obj, world_size: int) -> list:
    """Gather a Python object from every rank into a list of length world_size."""
    if world_size <= 1:
        return [obj]
    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj)
    return out


def grade_in_parallel(jobs, *, timeout_s: int = 5, max_workers: int = 8):
    """Grade a list of (Problem, code) pairs in parallel via ThreadPoolExecutor.

    Each grade() call spawns its own Python subprocess for execution — the
    GIL is irrelevant, so threads give real parallelism on the subprocess
    fork/wait. Results are returned in the SAME ORDER as `jobs`.
    """
    n = len(jobs)
    if n == 0:
        return []
    if max_workers <= 1 or n == 1:
        return [grade(prob, code, timeout_s=timeout_s) for prob, code in jobs]
    workers = min(max_workers, n)
    out: list = [None] * n
    with _cf.ThreadPoolExecutor(max_workers=workers) as pool:
        fut_to_idx = {
            pool.submit(grade, prob, code, timeout_s): i
            for i, (prob, code) in enumerate(jobs)
        }
        for fut in _cf.as_completed(fut_to_idx):
            out[fut_to_idx[fut]] = fut.result()
    return out


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
    # --- stochastic gate (only populated when --stochastic_gate is set) ---
    # `gate_decisions[i]` is True for "emit", False for "think". Positions
    # where `force_emit` was True (budget exhausted / finished row) are
    # EXCLUDED — the model had no policy choice, so they get no gradient.
    gate_decisions: list[bool] | None = None
    gate_log_probs: list[float] | None = None
    gate_positions: list[int] | None = None   # absolute pos in full_ids of the
                                              # token (think OR emit) whose
                                              # selection was a Bernoulli draw.
    # --- selective-sampling diagnostics (Phase A, 2026-05-26). Populated
    # alongside the gate_* fields when stochastic_gate=True. Counts are
    # accumulated across the rollout's non-force-emit positions. ---
    # σ buckets: <0.1, [0.1, 0.5), [0.5, 0.9), >=0.9
    gate_sigma_bucket_counts: list[int] | None = None
    # Number of positions where Bernoulli was actually sampled (σ in the
    # configured sample range and not force_emit).
    gate_n_sampled: int = 0
    # Number of positions where the deterministic threshold was used
    # (σ outside the sample range and not force_emit).
    gate_n_decisive: int = 0


def compute_think_budget_spread(budget: int, n: int,
                                diversity: float) -> list[int]:
    """Per-rollout think budgets for a GRPO group of size ``n``.

    Within-group think-budget diversity (FIX 2, THINKING_GATE_SELECTIVITY):
    the N rollouts of one problem get a SPREAD of think budgets so the group
    contains "thought less" and "thought more" variants of the SAME problem.
    With the existing counterfactual ponder shaping, the group-relative
    advantage then directly compares depth levels — a rollout that thought
    less but scored the same gets higher advantage, so the gate learns to
    think less where thinking doesn't pay.

    Scheme: linearly spaced budgets in ``[max(1, round(budget*(1-diversity))),
    budget]`` (inclusive, ascending). ``diversity == 0`` -> all budgets equal
    to ``budget`` (byte-identical to the pre-change single-scalar behavior).
    ``diversity`` clamped to [0, 1]; the floor is clamped to >= 1 so no rollout
    gets a zero (or negative) budget. With ``diversity == 1`` the lowest
    rollout gets budget 1 (think at most once) and the highest gets the full
    ``budget``.

    Returns a list of ``n`` ints, each in ``[1, budget]``.
    """
    n = int(n)
    budget = int(budget)
    if n <= 0:
        return []
    d = float(max(0.0, min(1.0, diversity)))
    if d == 0.0 or n == 1:
        return [budget] * n
    lo = max(1, int(round(budget * (1.0 - d))))
    hi = max(lo, budget)
    if hi == lo:
        return [lo] * n
    out = []
    for i in range(n):
        frac = i / (n - 1)                       # 0 .. 1 ascending
        b = int(round(lo + frac * (hi - lo)))
        out.append(max(1, min(budget, b)))
    return out


@torch.no_grad()
def rollout_group_batched(model, tokenizer, prompt_ids: torch.Tensor, *,
                           n_rollouts: int,
                           thinking_token_id: int,
                           eos_token_id: int | None,
                           max_gen: int,
                           max_think_per_step: int,
                           total_think_budget,
                           emit_threshold: float,
                           gate_floor: float,
                           temperature: float,
                           min_emit_before_eos: int,
                           stochastic_gate: bool = False,
                           gate_sample_range_low: float = 0.0,
                           gate_sample_range_high: float = 1.0,
                           ) -> list[Rollout]:
    """Roll out `n_rollouts` parallel completions of the same prompt in a
    SINGLE batched forward per iteration.

    Each row decides per-step whether to emit or insert a think token (gate
    + ponder logic identical to `rollout_one`). The batch dimension is kept
    rectangular by lock-stepping: rows that "finished" (hit max_gen emits
    or EOS) append a harmless pad token (we re-use EOS) on every subsequent
    iteration, but we don't record those appendages for the policy update.

    `total_think_budget` is either an int (broadcast to every row — the
    pre-change behavior) OR a per-row sequence/tensor of length N (FIX 2,
    within-group think-budget diversity: each rollout of the group can get a
    DIFFERENT budget so the group contains "thought less" / "thought more"
    variants of the same problem). Only the per-row force-emit cutoff changes;
    the advantage/loss path is untouched.

    Returns a list of N Rollout objects, one per row.
    """
    device = prompt_ids.device
    N = int(n_rollouts)
    # Tile prompt across N rows.
    ids = prompt_ids.expand(N, -1).contiguous()
    prompt_len = ids.shape[1]
    pad_id = int(eos_token_id) if eos_token_id is not None else 0
    # Per-row think budget (FIX 2). Accept a scalar (broadcast) or a length-N
    # sequence/tensor. Built as a (N,) long tensor for the vectorized
    # force-emit comparison; `budget_ceiling` bounds the iteration cap.
    if isinstance(total_think_budget, torch.Tensor):
        budget_per_row = total_think_budget.to(device=device,
                                               dtype=torch.long).reshape(-1)
    elif isinstance(total_think_budget, (list, tuple)):
        budget_per_row = torch.tensor([int(b) for b in total_think_budget],
                                      dtype=torch.long, device=device)
    else:
        budget_per_row = torch.full((N,), int(total_think_budget),
                                    dtype=torch.long, device=device)
    if budget_per_row.numel() != N:
        raise ValueError(
            f"total_think_budget length {budget_per_row.numel()} != "
            f"n_rollouts {N}")
    budget_ceiling = int(budget_per_row.max().item())
    # DO NOT set `_film_bypass = True`. The supposed "decode speedup"
    # (single-pass FiLM at T=1) catastrophically breaks the model at
    # temperature sampling. Diagnosed 2026-05-23: with bypass=True at
    # T=0.9 the model produces mode-collapsed garbage tokens
    # (`(24, 24)(24, 24)...`) on every prompt → ~0 % pass rate vs
    # ~6 % with FiLM ON. v2 was trained WITH FiLM ON, so removing it
    # at inference removes a load-bearing architectural component.
    # Greedy decode (HumanEval default) happens to work without it
    # because there's no entropy to amplify; the bug is invisible at
    # T=0 and ruinous at T=0.9. See the doc-fix note in this commit.
    emit_counts = torch.zeros(N, dtype=torch.long, device=device)
    think_counts_this_step = torch.zeros(N, dtype=torch.long, device=device)
    think_totals = torch.zeros(N, dtype=torch.long, device=device)
    finished = torch.zeros(N, dtype=torch.bool, device=device)

    emit_token_ids_per_row: list[list[int]] = [[] for _ in range(N)]
    emit_log_probs_per_row: list[list[float]] = [[] for _ in range(N)]
    emit_positions_per_row: list[list[int]] = [[] for _ in range(N)]
    # Per-row gate-decision record (only populated when stochastic_gate=True;
    # force_emit positions are excluded — no policy choice was made there).
    gate_decisions_per_row: list[list[bool]] = [[] for _ in range(N)]
    gate_log_probs_per_row: list[list[float]] = [[] for _ in range(N)]
    gate_positions_per_row: list[list[int]] = [[] for _ in range(N)]
    # Selective-sampling diagnostics. Sigma buckets are <0.1, [0.1,0.5),
    # [0.5,0.9), >=0.9 (4 buckets); n_sampled/decisive count the partition
    # between "Bernoulli draw" and "deterministic threshold" decisions.
    gate_sigma_buckets_per_row: list[list[int]] = [[0, 0, 0, 0] for _ in range(N)]
    gate_n_sampled_per_row: list[int] = [0 for _ in range(N)]
    gate_n_decisive_per_row: list[int] = [0 for _ in range(N)]

    # STATE-PASSING INCREMENTAL DECODE (2026-05-23). At T=prompt + 96
    # gen steps, the full-forward path's per-step cost grows linearly
    # while the incremental path is constant ~6 ms/tok. The cache is
    # batched naturally (FLA Cache + our WM buffer both have a batch
    # dimension), so all N rollout rows advance in lockstep.
    can_incremental = (hasattr(model, "forward_step")
                       and hasattr(model, "prefill"))
    if can_incremental:
        cache, last_logits = model.prefill(ids)
        pending_logits = last_logits[:, -1:, :]
    else:
        cache = None
        pending_logits = None

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    # Cap total iterations: even worst case, N*(max_gen + total_think_budget)
    # bounded by max_gen + total_think_budget (each iteration adds 1 token).
    max_iter = max_gen + budget_ceiling + 4
    with autocast:
        for _ in range(max_iter):
            if bool(finished.all().item()):
                break
            if can_incremental:
                next_logits = pending_logits[:, -1, :].float()
            else:
                logits = model(ids)
                next_logits = logits[:, -1, :].float()    # (N, V)

            gate_t = getattr(model, "_last_gate", None)
            if gate_t is None:
                gate = torch.ones(N, device=device)
            else:
                gate = gate_t[:, -1]
            # Apply gate_floor as a hard floor on p_emit (also used as the
            # threshold-compare value in the deterministic path).
            gate_clamped = (gate if gate_floor <= 0
                            else gate.clamp_min(gate_floor))
            force_emit = (
                (think_counts_this_step >= max_think_per_step)
                | (think_totals >= budget_per_row)
                | finished  # finished rows always "emit" pad
            )
            if stochastic_gate:
                # p_emit comes from the gate (already sigmoided into _last_gate);
                # clamp to floor so exploration above the floor is unconstrained
                # but we never assign Bernoulli probability below the floor.
                p_emit = gate_clamped.clamp(1e-6, 1.0 - 1e-6)
                # Selective stochastic gate (Phase A, 2026-05-26): only
                # Bernoulli-sample at uncertain positions; use deterministic
                # threshold at decisive positions. "Strictly inside" the
                # range — σ == low or σ == high is OUT (decisive). Default
                # range [0.0, 1.0] degenerates to sampling-everywhere.
                in_sample_range = (p_emit > gate_sample_range_low) & \
                                   (p_emit < gate_sample_range_high)
                sampled_emit = torch.bernoulli(p_emit).to(torch.bool)
                det_emit = gate_clamped >= emit_threshold
                chosen_emit = torch.where(in_sample_range, sampled_emit, det_emit)
                want_emit = chosen_emit | force_emit
                # log-prob of the SAMPLED action — only meaningful at sampled
                # positions. The downstream record step gates on
                # `in_sample_range` (AND not force_emit), so log-probs at
                # decisive / forced positions are never appended.
                gate_lp_emit = torch.log(p_emit)
                gate_lp_think = torch.log(1.0 - p_emit)
                gate_lp = torch.where(sampled_emit, gate_lp_emit, gate_lp_think)
            else:
                want_emit = (gate_clamped >= emit_threshold) | force_emit
                gate_lp = None
                sampled_emit = None
                in_sample_range = None
                p_emit = None

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

            if can_incremental:
                # Advance the cache by the just-appended row of tokens.
                # Finished rows feed garbage (their `appended` is whatever
                # `emit_toks` happened to be on this iter, since
                # want_emit is True for finished rows); we don't read
                # their downstream logits so the garbage state is fine.
                pending_logits, cache = model.forward_step(
                    appended.unsqueeze(1), cache)

            # Update per-row bookkeeping (host loop — small N).
            for i in range(N):
                if bool(finished[i].item()):
                    continue
                # Record stochastic gate decision BEFORE the EOS-finalize
                # short-circuit. Only when the model actually had a choice
                # (force_emit positions are masked out — no gradient there).
                # Phase A: ALSO mask out decisive positions (σ outside the
                # sample range) — those used the deterministic threshold,
                # no policy choice happened there either.
                if stochastic_gate and not bool(force_emit[i].item()):
                    p_emit_i = float(p_emit[i].item())
                    # σ histogram bucket (4-bin: <0.1, [0.1,0.5), [0.5,0.9), >=0.9).
                    if p_emit_i < 0.1:
                        gate_sigma_buckets_per_row[i][0] += 1
                    elif p_emit_i < 0.5:
                        gate_sigma_buckets_per_row[i][1] += 1
                    elif p_emit_i < 0.9:
                        gate_sigma_buckets_per_row[i][2] += 1
                    else:
                        gate_sigma_buckets_per_row[i][3] += 1
                    in_range_i = bool(in_sample_range[i].item())
                    if in_range_i:
                        gate_decisions_per_row[i].append(
                            bool(sampled_emit[i].item()))
                        gate_log_probs_per_row[i].append(float(gate_lp[i].item()))
                        gate_positions_per_row[i].append(int(new_pos))
                        gate_n_sampled_per_row[i] += 1
                    else:
                        gate_n_decisive_per_row[i] += 1
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
            gate_decisions=(gate_decisions_per_row[i] if stochastic_gate else None),
            gate_log_probs=(gate_log_probs_per_row[i] if stochastic_gate else None),
            gate_positions=(gate_positions_per_row[i] if stochastic_gate else None),
            gate_sigma_bucket_counts=(
                list(gate_sigma_buckets_per_row[i]) if stochastic_gate else None),
            gate_n_sampled=(
                gate_n_sampled_per_row[i] if stochastic_gate else 0),
            gate_n_decisive=(
                gate_n_decisive_per_row[i] if stochastic_gate else 0),
        ))
    return out_rollouts


def _merge_prefill_caches(caches: list[dict], group_sizes: list[int],
                          prompt_lens: list[int]) -> dict:
    """Concatenate B per-problem prefill caches (each already tiled to that
    problem's N rows) into ONE batched cache over R = sum(group_sizes) rows,
    for a single joint `forward_step` decode loop.

    Correctness contract — each per-problem cache was built by `prefill` over
    ONLY that problem's prompt (NO cross-problem padding), so the DeltaNet
    recurrent state is byte-identical to the single-problem path. Merging only
    stacks along the batch dim, which is independent across rows in every
    consumer (FLA recurrent kernel, WM read, pos-embed). The two things that
    legitimately differ per row — absolute position (`seen`) and WM buffer
    width — are handled explicitly:

      * `seen_per_row`  (R,) long: each row's true prompt length, so
        `forward_step` indexes `pos_embed` per row (model.py change).
      * WM buffer: per-problem buffers have DIFFERENT widths (= prompt_len_i).
        We LEFT-pad each to the common max width with `gate=0, value=0,
        tok=pad, pos=+SENTINEL`. The sentinel makes the causal mask
        (`buf_pos >= new_pos_val`) exclude pad slots with a true `-inf`
        (NOT a finite log-gate penalty), and only the RELATIVE order of
        `pos` values matters for the causal comparison — so this is
        bit-identical to the per-problem buffer (proof in the module note).
    """
    from fla.models.utils import Cache as FLACache

    R = sum(group_sizes)
    # --- FLA recurrent cache: concat each layer's state along batch dim. ---
    # Use the PUBLIC cache interface (`cache[L]` -> per-layer state dict,
    # `from_legacy_cache(tuple_of_state_dicts)` to rebuild). This is robust to
    # the FLA Cache implementation (new `.layers` FLALayer vs legacy `.states`)
    # — both expose `__getitem__`, `__len__`, and `from_legacy_cache`.
    n_layers = max(len(c["fla_cache"]) for c in caches)

    def _cat_field(states_for_layer, key):
        vals = [st.get(key) for st in states_for_layer]
        if any(v is None for v in vals):
            return None
        if isinstance(vals[0], (tuple, list)):
            return tuple(torch.cat([v[j] for v in vals], dim=0)
                         for j in range(len(vals[0])))
        return torch.cat(vals, dim=0)

    merged_state_dicts = []
    for L in range(n_layers):
        states_for_layer = [c["fla_cache"][L] for c in caches]
        merged_state_dicts.append(dict(
            recurrent_state=_cat_field(states_for_layer, "recurrent_state"),
            attn_state=_cat_field(states_for_layer, "attn_state"),
            conv_state=_cat_field(states_for_layer, "conv_state"),
            ffn_state=_cat_field(states_for_layer, "ffn_state"),
        ))
    seen_tokens = max(int(c["fla_cache"]._seen_tokens) for c in caches)
    merged_fla = FLACache.from_legacy_cache(
        tuple(merged_state_dicts), seen_tokens=seen_tokens)

    # Resolve a device for the small bookkeeping tensors below.
    _rec0 = merged_state_dicts[0].get("recurrent_state")
    if _rec0 is not None:
        cache_device = (_rec0[0] if isinstance(_rec0, (tuple, list)) else _rec0).device
    elif caches[0].get("wm_buf") is not None:
        cache_device = caches[0]["wm_buf"]["gate"].device
    else:
        cache_device = torch.device("cpu")

    # --- WM buffer: left-pad each problem's buffer to common width. ---
    wm_buf = None
    if caches[0].get("wm_buf") is not None:
        widths = [c["wm_buf"]["gate"].shape[1] for c in caches]
        T_max = max(widths)
        SENT = 10 ** 9  # pos sentinel: always >= any real new_pos_val
        device = caches[0]["wm_buf"]["gate"].device
        gate_rows, val_rows, pos_rows, tok_rows = [], [], [], []
        for c in caches:
            b = c["wm_buf"]
            w = b["gate"].shape[1]
            pad = T_max - w
            Bi = b["gate"].shape[0]
            d_mem = b["value"].shape[-1]
            if pad > 0:
                gpad = torch.zeros(Bi, pad, dtype=b["gate"].dtype, device=device)
                vpad = torch.zeros(Bi, pad, d_mem, dtype=b["value"].dtype,
                                   device=device)
                ppad = torch.full((Bi, pad), SENT, dtype=b["pos"].dtype,
                                  device=device)
                tpad = torch.zeros(Bi, pad, dtype=b["tok"].dtype, device=device)
                gate_rows.append(torch.cat([gpad, b["gate"]], dim=1))
                val_rows.append(torch.cat([vpad, b["value"]], dim=1))
                pos_rows.append(torch.cat([ppad, b["pos"]], dim=1))
                tok_rows.append(torch.cat([tpad, b["tok"]], dim=1))
            else:
                gate_rows.append(b["gate"]); val_rows.append(b["value"])
                pos_rows.append(b["pos"]);   tok_rows.append(b["tok"])
        wm_buf = {
            "gate": torch.cat(gate_rows, dim=0),
            "value": torch.cat(val_rows, dim=0),
            "pos": torch.cat(pos_rows, dim=0),
            "tok": torch.cat(tok_rows, dim=0),
        }

    # --- think_run_len (Phase 3): concat per-row counters if present. ---
    think_run_len = None
    if any(c.get("think_run_len") is not None for c in caches):
        parts = []
        for c, gs in zip(caches, group_sizes):
            tr = c.get("think_run_len")
            if tr is None:
                tr = torch.zeros(gs, dtype=torch.int64, device=cache_device)
            parts.append(tr)
        think_run_len = torch.cat(parts, dim=0)

    # FiLM lagged_sources: only present when _film_bypass is False. The
    # production generators run with film bypass OFF here? No — rollout keeps
    # FiLM ON (see rollout_group_batched note). lagged_sources are per-row
    # (B_i, 1, d) so they concat the same way.
    lagged_sources = None
    if caches[0].get("lagged_sources") is not None:
        lagged_sources = {}
        src_layers = caches[0]["lagged_sources"].keys()
        for L in src_layers:
            lagged_sources[L] = torch.cat(
                [c["lagged_sources"][L] for c in caches], dim=0)

    seen_per_row = torch.tensor(
        [pl for pl, gs in zip(prompt_lens, group_sizes) for _ in range(gs)],
        dtype=torch.long, device=cache_device,
    )
    return {
        "fla_cache": merged_fla,
        "seen": int(max(prompt_lens)),       # scalar fallback (unused when
                                             # seen_per_row is set)
        "seen_per_row": seen_per_row,
        "lagged_sources": lagged_sources,
        "wm_buf": wm_buf,
        "think_run_len": think_run_len,
        "_group_sizes": list(group_sizes),
        "_prompt_lens": list(prompt_lens),
    }


@torch.no_grad()
def rollout_turn0_batched_across_problems(
        model, tokenizer, prompt_ids_list: list[torch.Tensor], *,
        n_rollouts: int,
        budgets_per_problem: list,
        thinking_token_id: int,
        eos_token_id: int | None,
        max_gen: int,
        max_think_per_step: int,
        emit_threshold: float,
        gate_floor: float,
        temperature: float,
        min_emit_before_eos: int,
        stochastic_gate: bool = False,
        gate_sample_range_low: float = 0.0,
        gate_sample_range_high: float = 1.0,
) -> list[tuple[int, Rollout]]:
    """Turn-0 rollouts for ALL B problems in a SINGLE batched decode loop.

    This is the cross-problem extension of `rollout_group_batched`. Instead of
    B separate decode loops of N rows each, it runs ONE decode loop over
    R = B*N rows. DeltaNet decode is memory-bandwidth-bound, so the single
    R-row loop costs ~the same wall-time as one N-row loop → ~B× speedup on
    the dominant (turn-0) phase.

    `prompt_ids_list[b]` is `(1, T_b)` — problems have DIFFERENT prompt
    lengths. `budgets_per_problem[b]` is the per-row think-budget list of
    length N for problem b (from `compute_think_budget_spread`).

    Correctness — leakage-free by construction. Two decode paths:

      * Incremental (production, model has prefill/forward_step): each problem
        is prefilled OVER ITS OWN PROMPT ONLY (no cross-problem padding → the
        DeltaNet recurrent state is byte-identical to the single-problem
        path). The B caches are concatenated along the batch dim
        (`_merge_prefill_caches`) and ONE joint `forward_step` loop advances
        all R rows. Per-row absolute position uses `cache["seen_per_row"]`.

      * Full-forward fallback (test models / no incremental support): rows are
        RIGHT-padded to a common width and each row keeps its own `frontier`
        column (= prompt_len + tokens appended). Logits are read at
        `frontier-1` and the new token written at `frontier`. Causal
        attention means a row's real tokens never attend to the right-pad that
        follows its frontier → each row's logits are bit-identical to
        forwarding that row alone (no leakage).

    Returns a FLAT list of `(problem_index, Rollout)` so the caller can
    regroup into `trajectories_per_group`. Within a problem the N rollouts
    appear in row order (rollout r of problem b uses `budgets_per_problem[b][r]`).
    """
    device = prompt_ids_list[0].device
    B = len(prompt_ids_list)
    N = int(n_rollouts)
    R = B * N
    pad_id = int(eos_token_id) if eos_token_id is not None else 0

    # Per-row metadata: which problem, that problem's prompt length, the row's
    # think budget.
    prob_of_row: list[int] = []
    prompt_len_of_row: list[int] = []
    budget_of_row: list[int] = []
    prompt_lens = [int(p.shape[1]) for p in prompt_ids_list]
    for b in range(B):
        bl = budgets_per_problem[b]
        if isinstance(bl, (int,)):
            bl = [int(bl)] * N
        if len(bl) != N:
            raise ValueError(
                f"budgets_per_problem[{b}] length {len(bl)} != n_rollouts {N}")
        for r in range(N):
            prob_of_row.append(b)
            prompt_len_of_row.append(prompt_lens[b])
            budget_of_row.append(int(bl[r]))
    budget_per_row = torch.tensor(budget_of_row, dtype=torch.long, device=device)
    prompt_len_t = torch.tensor(prompt_len_of_row, dtype=torch.long,
                                device=device)
    budget_ceiling = int(budget_per_row.max().item())

    # Per-row bookkeeping (identical semantics to rollout_group_batched).
    emit_counts = torch.zeros(R, dtype=torch.long, device=device)
    think_counts_this_step = torch.zeros(R, dtype=torch.long, device=device)
    think_totals = torch.zeros(R, dtype=torch.long, device=device)
    finished = torch.zeros(R, dtype=torch.bool, device=device)

    emit_token_ids_per_row: list[list[int]] = [[] for _ in range(R)]
    emit_log_probs_per_row: list[list[float]] = [[] for _ in range(R)]
    emit_positions_per_row: list[list[int]] = [[] for _ in range(R)]
    gate_decisions_per_row: list[list[bool]] = [[] for _ in range(R)]
    gate_log_probs_per_row: list[list[float]] = [[] for _ in range(R)]
    gate_positions_per_row: list[list[int]] = [[] for _ in range(R)]
    gate_sigma_buckets_per_row: list[list[int]] = [[0, 0, 0, 0] for _ in range(R)]
    gate_n_sampled_per_row: list[int] = [0 for _ in range(R)]
    gate_n_decisive_per_row: list[int] = [0 for _ in range(R)]
    # full token sequence per row (row-local: prompt + emits + thinks, NO pad).
    full_ids_per_row: list[list[int]] = [
        prompt_ids_list[prob_of_row[i]][0].cpu().tolist() for i in range(R)
    ]

    can_incremental = (hasattr(model, "forward_step")
                       and hasattr(model, "prefill"))

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    max_iter = max_gen + budget_ceiling + 4

    if can_incremental:
        # ---- Per-problem isolated prefill, then concat into one cache. ----
        # Each problem is prefilled over ONLY its own prompt (tiled to N rows),
        # so its DeltaNet recurrent state is byte-identical to the single-
        # problem path. The B caches are concatenated along the batch dim.
        caches = []
        first_logits_per_row = []
        first_gate_per_row = []   # step-0 gate per row (from prefill)
        with autocast:
            for b in range(B):
                ids_b = prompt_ids_list[b].expand(N, -1).contiguous()
                cache_b, last_logits_b = model.prefill(ids_b)
                caches.append(cache_b)
                # (N, T_b, V) -> last prompt position per row (N, V).
                first_logits_per_row.append(last_logits_b[:, -1, :])
                # The gate that drives step-0 is the prompt's LAST-position
                # gate. `prefill` stashes it on `model._last_gate` (N, T_b);
                # capture it NOW (before the next problem's prefill overwrites
                # the stash) so the merged step-0 gate is per-row correct.
                g_b = getattr(model, "_last_gate", None)
                if g_b is not None:
                    first_gate_per_row.append(g_b[:, -1:].detach().clone())
                else:
                    first_gate_per_row.append(torch.ones(N, 1, device=device))
            cache = _merge_prefill_caches(caches, [N] * B, prompt_lens)
        pending = torch.cat(first_logits_per_row, dim=0)   # (R, V)
        step0_gate = torch.cat(first_gate_per_row, dim=0)   # (R, 1)

        with autocast:
            first_step = True
            for _ in range(max_iter):
                if bool(finished.all().item()):
                    break
                next_logits = pending.float()    # (R, V)
                # Step 0 reads the merged prefill gate (model._last_gate holds
                # only the LAST problem's prefill stash); later steps read the
                # (R,1) gate that forward_step correctly set for all rows.
                gate_for_step = step0_gate if first_step else None
                first_step = False
                _decode_one_step(
                    model, next_logits, gate_full=gate_for_step,
                    R=R, device=device, thinking_token_id=thinking_token_id,
                    eos_token_id=eos_token_id, gate_floor=gate_floor,
                    emit_threshold=emit_threshold, temperature=temperature,
                    min_emit_before_eos=min_emit_before_eos,
                    max_gen=max_gen, max_think_per_step=max_think_per_step,
                    budget_per_row=budget_per_row,
                    stochastic_gate=stochastic_gate,
                    gate_sample_range_low=gate_sample_range_low,
                    gate_sample_range_high=gate_sample_range_high,
                    emit_counts=emit_counts,
                    think_counts_this_step=think_counts_this_step,
                    think_totals=think_totals, finished=finished,
                    emit_token_ids_per_row=emit_token_ids_per_row,
                    emit_log_probs_per_row=emit_log_probs_per_row,
                    emit_positions_per_row=emit_positions_per_row,
                    gate_decisions_per_row=gate_decisions_per_row,
                    gate_log_probs_per_row=gate_log_probs_per_row,
                    gate_positions_per_row=gate_positions_per_row,
                    gate_sigma_buckets_per_row=gate_sigma_buckets_per_row,
                    gate_n_sampled_per_row=gate_n_sampled_per_row,
                    gate_n_decisive_per_row=gate_n_decisive_per_row,
                    full_ids_per_row=full_ids_per_row,
                    frontier=None,
                )
                # Advance the cache with the just-appended row of tokens.
                appended = torch.tensor(
                    [full_ids_per_row[i][-1] for i in range(R)],
                    dtype=torch.long, device=device).unsqueeze(1)
                pending_logits, cache = model.forward_step(appended, cache)
                pending = pending_logits[:, -1, :]
    else:
        # ---- Full-forward fallback: right-padded joint tensor + per-row
        # frontier. Leakage-free (causal attn never reads a row's right-pad). ----
        T0 = max(prompt_lens)
        ids = torch.full((R, T0), pad_id, dtype=torch.long, device=device)
        for i in range(R):
            pl = prompt_len_of_row[i]
            ids[i, :pl] = torch.tensor(full_ids_per_row[i], dtype=torch.long,
                                       device=device)
        frontier = prompt_len_t.clone()   # (R,) next write column per row
        with autocast:
            for _ in range(max_iter):
                if bool(finished.all().item()):
                    break
                logits = model(ids)                          # (R, T_cur, V)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # Read each row's logits at frontier-1.
                idx = (frontier - 1).clamp(min=0)
                next_logits = logits[torch.arange(R, device=device), idx, :].float()
                gate_full = getattr(model, "_last_gate", None)
                appended = _decode_one_step(
                    model, next_logits,
                    R=R, device=device, thinking_token_id=thinking_token_id,
                    eos_token_id=eos_token_id, gate_floor=gate_floor,
                    emit_threshold=emit_threshold, temperature=temperature,
                    min_emit_before_eos=min_emit_before_eos,
                    max_gen=max_gen, max_think_per_step=max_think_per_step,
                    budget_per_row=budget_per_row,
                    stochastic_gate=stochastic_gate,
                    gate_sample_range_low=gate_sample_range_low,
                    gate_sample_range_high=gate_sample_range_high,
                    emit_counts=emit_counts,
                    think_counts_this_step=think_counts_this_step,
                    think_totals=think_totals, finished=finished,
                    emit_token_ids_per_row=emit_token_ids_per_row,
                    emit_log_probs_per_row=emit_log_probs_per_row,
                    emit_positions_per_row=emit_positions_per_row,
                    gate_decisions_per_row=gate_decisions_per_row,
                    gate_log_probs_per_row=gate_log_probs_per_row,
                    gate_positions_per_row=gate_positions_per_row,
                    gate_sigma_buckets_per_row=gate_sigma_buckets_per_row,
                    gate_n_sampled_per_row=gate_n_sampled_per_row,
                    gate_n_decisive_per_row=gate_n_decisive_per_row,
                    full_ids_per_row=full_ids_per_row,
                    frontier=frontier,
                    gate_full=gate_full,
                )
                # Write each row's appended token at its frontier column,
                # growing the tensor if any frontier reached the width.
                if int(frontier.max().item()) >= ids.shape[1]:
                    grow = torch.full((R, 1), pad_id, dtype=torch.long,
                                      device=device)
                    ids = torch.cat([ids, grow], dim=1)
                ids[torch.arange(R, device=device), frontier] = appended
                frontier = frontier + 1

    # Build Rollout objects tagged with problem index.
    out: list[tuple[int, Rollout]] = []
    for i in range(R):
        text = tokenizer.decode(emit_token_ids_per_row[i])
        r = Rollout(
            prompt_len=prompt_len_of_row[i],
            emit_token_ids=emit_token_ids_per_row[i],
            emit_log_probs=emit_log_probs_per_row[i],
            emit_positions=emit_positions_per_row[i],
            full_ids=full_ids_per_row[i],
            depth=int(think_totals[i].item()),
            text=text,
            gate_decisions=(gate_decisions_per_row[i] if stochastic_gate else None),
            gate_log_probs=(gate_log_probs_per_row[i] if stochastic_gate else None),
            gate_positions=(gate_positions_per_row[i] if stochastic_gate else None),
            gate_sigma_bucket_counts=(
                list(gate_sigma_buckets_per_row[i]) if stochastic_gate else None),
            gate_n_sampled=(gate_n_sampled_per_row[i] if stochastic_gate else 0),
            gate_n_decisive=(gate_n_decisive_per_row[i] if stochastic_gate else 0),
        )
        out.append((prob_of_row[i], r))
    return out


def _decode_one_step(
        model, next_logits, *, R, device,
        thinking_token_id, eos_token_id, gate_floor, emit_threshold,
        temperature, min_emit_before_eos, max_gen, max_think_per_step,
        budget_per_row, stochastic_gate, gate_sample_range_low,
        gate_sample_range_high, emit_counts, think_counts_this_step,
        think_totals, finished, emit_token_ids_per_row, emit_log_probs_per_row,
        emit_positions_per_row, gate_decisions_per_row, gate_log_probs_per_row,
        gate_positions_per_row, gate_sigma_buckets_per_row,
        gate_n_sampled_per_row, gate_n_decisive_per_row, full_ids_per_row,
        frontier, gate_full=None):
    """One emit/think decision + record step, shared by both decode paths.

    Mirrors the inner body of `rollout_group_batched` EXACTLY (same gate /
    sampling / bookkeeping logic) but operates on R rows with PER-ROW prompt
    lengths: each row's recorded `emit_positions` / `gate_positions` are
    indices into THAT ROW's own `full_ids` (row-local), so they are identical
    to what the single-problem path records and re-fetchable by the offline
    policy forward.

    For the incremental path, `gate_full` is read from `model._last_gate`
    (shape (R, 1)); for the full-forward path the caller passes the (R, T)
    gate and we take the per-row frontier-1 column. Appends the chosen token
    to each row's `full_ids_per_row[i]` and returns the (R,) appended tensor.
    """
    N_rows = R
    if gate_full is None:
        gate_t = getattr(model, "_last_gate", None)
        if gate_t is None:
            gate = torch.ones(N_rows, device=device)
        else:
            gate = gate_t[:, -1]
    else:
        if frontier is not None:
            idx = (frontier - 1).clamp(min=0)
            gate = gate_full[torch.arange(N_rows, device=device), idx]
        else:
            gate = gate_full[:, -1]

    gate_clamped = (gate if gate_floor <= 0 else gate.clamp_min(gate_floor))
    force_emit = (
        (think_counts_this_step >= max_think_per_step)
        | (think_totals >= budget_per_row)
        | finished
    )
    if stochastic_gate:
        p_emit = gate_clamped.clamp(1e-6, 1.0 - 1e-6)
        in_sample_range = (p_emit > gate_sample_range_low) & \
                          (p_emit < gate_sample_range_high)
        sampled_emit = torch.bernoulli(p_emit).to(torch.bool)
        det_emit = gate_clamped >= emit_threshold
        chosen_emit = torch.where(in_sample_range, sampled_emit, det_emit)
        want_emit = chosen_emit | force_emit
        gate_lp_emit = torch.log(p_emit)
        gate_lp_think = torch.log(1.0 - p_emit)
        gate_lp = torch.where(sampled_emit, gate_lp_emit, gate_lp_think)
    else:
        want_emit = (gate_clamped >= emit_threshold) | force_emit
        gate_lp = None
        sampled_emit = None
        in_sample_range = None
        p_emit = None

    next_logits[:, thinking_token_id] = -float("inf")
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
    log_probs_full = F.log_softmax(next_logits / max(temperature, 1e-8), dim=-1)
    emit_lp = log_probs_full.gather(1, emit_toks.unsqueeze(1)).squeeze(1)

    think_tok_t = torch.full_like(emit_toks, int(thinking_token_id))
    appended = torch.where(want_emit, emit_toks, think_tok_t)

    # Per-row record + append to row-local full_ids. `new_pos` is the index
    # the appended token WILL occupy in this row's own full_ids — identical to
    # the single-problem path (where full_ids has no cross-problem padding).
    for i in range(N_rows):
        if bool(finished[i].item()):
            # Finished rows still "append" in the incremental cache lock-step,
            # but we do NOT record them and do NOT grow their full_ids (so the
            # rollout's full_ids stays exactly what the single-problem path
            # produced). The cache advance feeds garbage for these rows; their
            # logits are never read again.
            continue
        new_pos = len(full_ids_per_row[i])   # row-local index of next token
        if stochastic_gate and not bool(force_emit[i].item()):
            p_emit_i = float(p_emit[i].item())
            if p_emit_i < 0.1:
                gate_sigma_buckets_per_row[i][0] += 1
            elif p_emit_i < 0.5:
                gate_sigma_buckets_per_row[i][1] += 1
            elif p_emit_i < 0.9:
                gate_sigma_buckets_per_row[i][2] += 1
            else:
                gate_sigma_buckets_per_row[i][3] += 1
            in_range_i = bool(in_sample_range[i].item())
            if in_range_i:
                gate_decisions_per_row[i].append(bool(sampled_emit[i].item()))
                gate_log_probs_per_row[i].append(float(gate_lp[i].item()))
                gate_positions_per_row[i].append(int(new_pos))
                gate_n_sampled_per_row[i] += 1
            else:
                gate_n_decisive_per_row[i] += 1
        if bool(want_emit[i].item()):
            tok = int(emit_toks[i].item())
            is_eos = (eos_token_id is not None and tok == int(eos_token_id))
            if is_eos:
                finished[i] = True
                # Still append so the cache lock-step has a token, but mark
                # finished so it isn't recorded. Match single-problem path:
                # there, the EOS token IS appended to ids (the row's full_ids
                # would include it) but not to emit_*. To stay byte-identical
                # to rollout_group_batched's full_ids (which DOES keep the
                # appended emit token at the EOS step), append it here too.
                full_ids_per_row[i].append(tok)
                continue
            emit_token_ids_per_row[i].append(tok)
            emit_log_probs_per_row[i].append(float(emit_lp[i].item()))
            emit_positions_per_row[i].append(int(new_pos))
            full_ids_per_row[i].append(tok)
            emit_counts[i] = emit_counts[i] + 1
            think_counts_this_step[i] = 0
            if int(emit_counts[i].item()) >= max_gen:
                finished[i] = True
        else:
            full_ids_per_row[i].append(int(thinking_token_id))
            think_counts_this_step[i] = think_counts_this_step[i] + 1
            think_totals[i] = think_totals[i] + 1

    return appended


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
# Multi-turn trajectory rollout (RL_NEXT_DESIGN §2.1, §5)
# ---------------------------------------------------------------------------

def run_trajectories_for_group(
    problem, prompt_text: str, *,
    rollout_fn: Callable, grade_fn: Callable, extract_fn: Callable,
    n_rollouts: int, max_turns: int, carry_history: bool,
    pass_threshold: float = 0.5,
    grade_batch_fn: Callable | None = None,
    precomputed_turn0: list | None = None,
) -> list[Trajectory]:
    """Run N feedback-guided revision trajectories on one problem.

    `rollout_fn(prompt_text, n_rollouts) -> list[Rollout]` does the (batched)
    generation; `grade_fn(problem, code) -> GradingResult` grades; `extract_fn`
    pulls the gradeable code from a rollout's text. These are injected so the
    GPU/grader work stays in the trainer and this orchestration is testable.

    Turn 0 generates N parallel rollouts (one batched call). Each subsequent
    turn re-rolls ONLY the still-unsolved lineages, one at a time, with a
    repair prompt built from the latest (or full, if carry_history) failure.

    `precomputed_turn0` (optional): a list of N Rollout objects for turn 0,
    already generated by the cross-problem batched decode
    (`rollout_turn0_batched_across_problems`). When provided, `rollout_fn` is
    NOT called for turn 0 — the rollouts are used as-is. Turns ≥1 still call
    `rollout_fn`. This is the only behavioural difference and it is exactly
    semantics-preserving: the supplied turn-0 rollouts are byte-identical (on
    the production incremental path, by construction) to what `rollout_fn`
    would have produced per-problem.

    Returns N Trajectory objects (one per lineage). `max_turns == 1` produces
    one-turn trajectories → byte-identical to the legacy single-shot path.
    """
    # Turn 0: N parallel rollouts. Grade all N in parallel (each grade() forks
    # a subprocess, so threads give real parallelism) when a batch grader is
    # supplied — otherwise serial via grade_fn (keeps the function mock-testable).
    if precomputed_turn0 is not None:
        rollouts0 = precomputed_turn0
    else:
        rollouts0 = rollout_fn(prompt_text, n_rollouts)
    codes0 = [extract_fn(r) for r in rollouts0]
    if grade_batch_fn is not None:
        grades0 = grade_batch_fn([(problem, c) for c in codes0])
    else:
        grades0 = [grade_fn(problem, c) for c in codes0]
    trajectories: list[Trajectory] = []
    for r, code, g in zip(rollouts0, codes0, grades0):
        traj = Trajectory(problem_id=str(problem.task_id))
        traj.turns.append(Turn(prompt_text=prompt_text, rollout=r,
                               score=float(g.score), tier=g.tier,
                               error_text=g.error_text))
        # Stash the gradeable code on the turn for later judge use.
        traj.turns[-1].__dict__["_code"] = code
        trajectories.append(traj)

    # Turns 1..max_turns-1: revise still-unsolved lineages.
    for _turn_idx in range(1, max_turns):
        any_open = False
        for traj in trajectories:
            if traj.passed(pass_threshold):
                continue  # early-stop solved lineages (§6.2 item 2)
            any_open = True
            last = traj.turns[-1]
            failed_code = last.__dict__.get("_code", "")
            err_text = last.error_text or ""
            if carry_history:
                # Stack all prior failures into the prompt context.
                base_prompt = prompt_text
                for t in traj.turns:
                    base_prompt = build_repair_prompt(
                        base_prompt, t.__dict__.get("_code", ""),
                        t.error_text or "")
                repair_prompt = base_prompt
            else:
                repair_prompt = build_repair_prompt(
                    prompt_text, failed_code, err_text)
            rr = rollout_fn(repair_prompt, 1)[0]
            code = extract_fn(rr)
            g = grade_fn(problem, code)
            traj.turns.append(Turn(prompt_text=repair_prompt, rollout=rr,
                                   score=float(g.score), tier=g.tier,
                                   error_text=g.error_text))
            traj.turns[-1].__dict__["_code"] = code
        if not any_open:
            break
    return trajectories


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
    stochastic_gate: bool = False, gate_entropy_bonus: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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

    Returns (loss, mean_ratio, mean_kl, gate_stats) — mean ratio and KL
    averaged across all emit positions across all rollouts; gate_stats is
    a dict with `gate_ratio`, `gate_entropy`, `gate_fire_rate` (mean of
    the BERNOULLI-decision think share — i.e. fraction of stochastic
    decisions that landed on think). When stochastic_gate=False, the
    gate_stats entries are NaN.
    """
    device = next(model.parameters()).device
    R = len(rollouts)
    nan = float("nan")
    empty_gate_stats = {"gate_ratio": nan, "gate_entropy": nan,
                         "gate_fire_rate": nan}
    if R == 0:
        return (torch.zeros((), device=device, requires_grad=True),
                torch.tensor(1.0), torch.tensor(0.0), empty_gate_stats)
    T_max = max(len(r.full_ids) for r in rollouts)
    padded = torch.full((R, T_max), int(pad_id), dtype=torch.long, device=device)
    for i, r in enumerate(rollouts):
        L = len(r.full_ids)
        padded[i, :L] = torch.tensor(r.full_ids, dtype=torch.long, device=device)

    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    with autocast:
        logits = model(padded)                      # (R, T_max, V) bf16
    # Grab the per-position gate stash. With DDP, `model._last_gate` is set
    # on the underlying module, accessed via .module (forward propagates it).
    inner = getattr(model, "module", model)
    new_gate = getattr(inner, "_last_gate", None)        # (R, T_max) post-sigmoid
    ref_logits = None
    if ref_model is not None and kl_coef > 0.0:
        with torch.no_grad(), autocast:
            ref_logits = ref_model(padded)          # bf16; small slices cast to fp32 below
    # Per-rollout, extract log-probs at emit positions.
    all_surrs = []
    all_ratios = []
    all_kls = []
    all_gate_surrs = []
    all_gate_ratios = []
    all_gate_entropies = []
    all_gate_emit_shares = []  # for fire-rate logging
    for i, r in enumerate(rollouts):
        if not r.emit_token_ids:
            continue
        adv = float(advantages[i])
        positions = torch.tensor(r.emit_positions, dtype=torch.long,
                                  device=device)
        # logits that PREDICTED the token at position p are at logits[i, p-1, :].
        pred_idx = positions - 1
        pred_logits = logits[i, pred_idx, :].float()   # (n_emit, V), bf16 → fp32 for softmax
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

        # --- Stochastic gate policy gradient (when rollout recorded
        # Bernoulli draws). Same group-relative `adv` rewards the gate's
        # decisions: if this rollout earned high advantage, push the
        # probability of WHAT THE GATE CHOSE upward.
        if (stochastic_gate and new_gate is not None
                and r.gate_decisions is not None and len(r.gate_decisions) > 0):
            gate_pos = torch.tensor(r.gate_positions, dtype=torch.long,
                                     device=device)
            # The gate value at position p drove the decision for the token
            # APPENDED at position p (the gate is read from logits[..., -1, :]
            # of the prefix ending just before p was appended; in the
            # batched forward, `_last_gate[i, p]` is the gate at position p
            # which used h[i, p-1] — same prefix). Use index `p-1` to match
            # the prefix-prediction convention of the emit branch.
            new_p_at_dec = new_gate[i, gate_pos - 1].float().clamp(1e-6, 1.0 - 1e-6)
            dec_bool = torch.tensor(r.gate_decisions, dtype=torch.bool,
                                     device=device)
            new_gate_lp = torch.where(
                dec_bool, torch.log(new_p_at_dec),
                torch.log(1.0 - new_p_at_dec))
            old_gate_lp = torch.tensor(r.gate_log_probs, dtype=torch.float32,
                                        device=device)
            g_ratio = torch.exp(new_gate_lp - old_gate_lp)
            g_clipped = torch.clamp(g_ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            g_surr = -torch.minimum(g_ratio * adv, g_clipped * adv).mean()
            all_gate_surrs.append(g_surr)
            all_gate_ratios.append(g_ratio.detach().mean())
            # Bernoulli entropy on the CURRENT policy's probabilities at the
            # same decision positions (for the entropy bonus + logging).
            ent = -(new_p_at_dec * torch.log(new_p_at_dec)
                    + (1.0 - new_p_at_dec) * torch.log(1.0 - new_p_at_dec))
            all_gate_entropies.append(ent.mean())
            # Fire rate = fraction of decisions that emitted (decoded; for logging).
            all_gate_emit_shares.append(dec_bool.float().mean().detach())
        if ref_logits is not None:
            ref_pred_logits = ref_logits[i, pred_idx, :].float()
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
                torch.tensor(1.0), torch.tensor(0.0), empty_gate_stats)
    surr_loss = torch.stack(all_surrs).mean()
    mean_ratio = torch.stack(all_ratios).mean()
    total_loss = surr_loss
    if all_kls:
        mean_kl = torch.stack(all_kls).mean()
        total_loss = total_loss + kl_coef * mean_kl
        kl_out = mean_kl.detach()
    else:
        kl_out = torch.tensor(0.0, device=device)
    # Stochastic-gate contributions. Per-rollout averages then mean across
    # rollouts — matches the per-rollout-mean convention used for emit-surr
    # so a single gate decision in a short rollout doesn't dominate.
    if all_gate_surrs:
        gate_surr = torch.stack(all_gate_surrs).mean()
        gate_ent = torch.stack(all_gate_entropies).mean()
        total_loss = total_loss + gate_surr - gate_entropy_bonus * gate_ent
        gate_stats = {
            "gate_ratio": float(torch.stack(all_gate_ratios).mean().item()),
            "gate_entropy": float(gate_ent.detach().item()),
            "gate_fire_rate": float(
                torch.stack(all_gate_emit_shares).mean().item()),
        }
    else:
        gate_stats = empty_gate_stats
    return total_loss, mean_ratio, kl_out, gate_stats


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
# Entropy-bonus curriculum (Phase B)
# ---------------------------------------------------------------------------

def compute_entropy_bonus(step: int, *, static: float, start: float,
                          end: float, total: int) -> float:
    if start <= 0:
        return static
    frac = min(1.0, step / max(1, total))
    return start + (end - start) * frac


# ---------------------------------------------------------------------------
# Multi-turn step orchestration (RL_NEXT_DESIGN §2, §5, §8)
# ---------------------------------------------------------------------------

def _run_multiturn_step(args, batch_problems, model_module, tok, device, *,
                        thinking_token_id: int, warmup_scale: float,
                        judge_backend=None):
    """Run one multi-turn training step's rollout→grade→reward→advantage
    pipeline (RL_NEXT_DESIGN §2, §5, §8). Returns the SAME downstream
    structures the legacy single-shot path produces, so the rest of the
    training loop (curriculum update, policy update, logging) is unchanged:

        (all_rollouts, combined_groups, per_group_advantages,
         first_pass_rewards_per_group, tier_hist, n_zero_var_before,
         n_lift_to_var, repair_n, repair_pass_n, n_judge_fired)

    `all_rollouts[gi]` are the FIRST-PASS (turn-0) rollouts per group — what
    logging + the curriculum EMA consume (the model's unprompted behaviour).
    `combined_groups[gi]` flattens every turn of every trajectory into the
    GRPO group (separate-rows assembly, §5.3); `per_group_advantages[gi]` is
    the duplicated trajectory advantage aligned 1:1 with combined_groups[gi].
    """
    rep_temp = (args.temperature if args.repair_temperature < 0
                else args.repair_temperature)

    def _extract(r: Rollout) -> str:
        if args.extract_code_block:
            extracted = extract_code_block(r.text)
            return extracted if extracted is not None else r.text
        return r.text

    def _rollout_fn(prompt_text: str, n: int):
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
        temp = args.temperature if n > 1 else rep_temp  # turn-0 vs repair temp
        # Within-group budget diversity only applies to the full turn-0 group
        # (n > 1); single repair rollouts (n == 1) keep the base budget.
        budget = compute_think_budget_spread(
            args.total_think_budget, n, args.think_budget_diversity)
        return rollout_group_batched(
            model_module, tok, prompt_ids, n_rollouts=n,
            thinking_token_id=thinking_token_id, eos_token_id=tok.eos_token_id,
            max_gen=args.max_gen, max_think_per_step=args.max_think_per_step,
            total_think_budget=budget,
            emit_threshold=args.emit_threshold, gate_floor=args.gate_floor,
            temperature=temp, min_emit_before_eos=args.min_emit_before_eos,
            stochastic_gate=args.stochastic_gate,
            gate_sample_range_low=args.gate_sample_range_low,
            gate_sample_range_high=args.gate_sample_range_high)

    # The grader runs a subprocess per call; the orchestrator grades turns
    # synchronously (per RL_NEXT_DESIGN §6.2: at grade_workers≥12 these stay
    # off the critical path). We keep it a thin closure so the pure
    # orchestration logic (run_trajectories_for_group) stays GPU/grader-free
    # and unit-testable with mocks.
    def _grade_fn(problem, code):
        return grade(problem, code, timeout_s=5)

    def _grade_batch_fn(jobs):
        # Grade the N turn-0 rollouts concurrently (subprocess fork/wait).
        return grade_in_parallel(jobs, timeout_s=5, max_workers=8)

    # --- Turn-0 rollouts for ALL problems in ONE batched decode loop. ---
    # DeltaNet decode is memory-bandwidth-bound, so a single B*N-row loop costs
    # ~the same wall-time as one N-row loop → ~B× speedup on the dominant
    # (turn-0) phase. Turns ≥1 (revision) stay per-trajectory below. The
    # per-row rollouts are byte-identical (on the incremental path, by
    # construction: each problem is prefilled over ONLY its own prompt) to what
    # the per-problem `_rollout_fn` would have produced. `--no_batch_turn0`
    # falls back to the legacy per-problem turn-0 path.
    prompt_texts = [build_mbpp_prompt(prob) for prob in batch_problems]
    precomputed_per_group: list[list | None] = [None] * len(batch_problems)
    if (getattr(args, "batch_turn0", True)
            and len(batch_problems) > 1):
        prompt_ids_list = [
            tok(pt, return_tensors="pt").input_ids.to(device)
            for pt in prompt_texts
        ]
        budgets_per_problem = [
            compute_think_budget_spread(
                args.total_think_budget, args.grpo_n_group,
                args.think_budget_diversity)
            for _ in batch_problems
        ]
        flat = rollout_turn0_batched_across_problems(
            model_module, tok, prompt_ids_list,
            n_rollouts=args.grpo_n_group,
            budgets_per_problem=budgets_per_problem,
            thinking_token_id=thinking_token_id, eos_token_id=tok.eos_token_id,
            max_gen=args.max_gen, max_think_per_step=args.max_think_per_step,
            emit_threshold=args.emit_threshold, gate_floor=args.gate_floor,
            temperature=args.temperature,
            min_emit_before_eos=args.min_emit_before_eos,
            stochastic_gate=args.stochastic_gate,
            gate_sample_range_low=args.gate_sample_range_low,
            gate_sample_range_high=args.gate_sample_range_high)
        precomputed_per_group = [[] for _ in batch_problems]
        for pi, r in flat:
            precomputed_per_group[pi].append(r)

    trajectories_per_group: list[list[Trajectory]] = []
    for prob, prompt_text, pre in zip(
            batch_problems, prompt_texts, precomputed_per_group):
        trajs = run_trajectories_for_group(
            prob, prompt_text,
            rollout_fn=_rollout_fn, grade_fn=_grade_fn, extract_fn=_extract,
            n_rollouts=args.grpo_n_group, max_turns=args.max_turns,
            carry_history=args.carry_history, grade_batch_fn=_grade_batch_fn,
            precomputed_turn0=pre)
        trajectories_per_group.append(trajs)

    # Per-trajectory improvement-shaped reward (§2.2).
    for trajs in trajectories_per_group:
        for traj in trajs:
            traj.R_traj = compute_trajectory_reward(
                [t.score for t in traj.turns],
                lambda_improve=args.turn_improvement_weight,
                turn_cost=args.turn_cost)

    # Diagnostics + first-pass (turn-0) structures for logging/curriculum.
    all_rollouts: list[list[Rollout]] = []
    first_pass_rewards_per_group: list[list[float]] = []
    tier_hist: dict[str, int] = {}
    repair_n = 0
    repair_pass_n = 0
    for trajs in trajectories_per_group:
        all_rollouts.append([traj.turns[0].rollout for traj in trajs])
        first_pass_rewards_per_group.append(
            [traj.turns[0].score for traj in trajs])
        for traj in trajs:
            for ti, turn in enumerate(traj.turns):
                # Stamp grade results back onto the rollout for logging
                # (reward_mean / tier_hist reflect first-pass behaviour).
                turn.rollout.reward = float(turn.score)
                turn.rollout.tier = turn.tier
                tier_hist[turn.tier] = tier_hist.get(turn.tier, 0) + 1
                if ti > 0:  # turns ≥1 are revisions ("repair")
                    repair_n += 1
                    if turn.tier == "pass":
                        repair_pass_n += 1

    # Per-group trajectory advantages + zero-variance hygiene (§1.2 C) +
    # optional LLM-judge tie-break on tied groups (§8).
    n_zero_var_before = 0
    n_lift_to_var = 0
    n_judge_fired = 0
    n_var_bearing = 0
    n_dropped_groups = 0
    combined_groups: list[list[Rollout]] = []
    per_group_advantages: list[torch.Tensor] = []
    for trajs in trajectories_per_group:
        if not trajs:
            combined_groups.append([])
            per_group_advantages.append(torch.zeros(0))
            continue
        traj_rewards = [traj.R_traj for traj in trajs]
        traj_depths = [float(traj.total_depth) for traj in trajs]
        # First-pass zero-variance bookkeeping (the model's unprompted signal).
        fp = [traj.turns[0].score for traj in trajs]
        fp_passes = sum(1 for r in fp if r >= 0.5)
        if fp_passes == 0 or fp_passes == len(fp):
            n_zero_var_before += 1
            if group_is_variance_bearing(traj_rewards, 1e-9):
                n_lift_to_var += 1  # multi-turn rescued an all-fail group

        var_bearing = group_is_variance_bearing(
            traj_rewards, args.group_var_floor)
        used_judge = False
        if (not var_bearing and judge_backend is not None
                and len(trajs) >= 2):
            # Execution-tied group → the judge is exactly its scope (§8.2).
            cands = [
                JudgeCandidate(
                    code=traj.turns[-1].__dict__.get("_code", ""),
                    error_text=traj.turns[-1].error_text,
                    tier_base=traj.terminal_score)
                for traj in trajs
            ]
            folded = apply_judge_to_group(
                cands, judge_backend, build_mbpp_prompt(batch_problems[
                    len(combined_groups)]),
                eps_judge=args.judge_eps, tier_margin=args.judge_tier_margin)
            if folded is not None:
                traj_rewards = folded
                var_bearing = group_is_variance_bearing(
                    traj_rewards, 1e-9)
                used_judge = True
                n_judge_fired += 1

        # Build the GRPO group as separate rows (one per turn), shared adv.
        adv_t = compute_grpo_advantages_from_rewards(
            torch.tensor([traj_rewards]), torch.tensor([traj_depths]),
            ponder_cost=args.ponder_cost, ponder_shape=args.ponder_shape,
            counterfactual=args.counterfactual,
            ponder_warmup_scale=warmup_scale)[0]
        # Zero-variance hygiene (§1.2 C): with a positive floor, drop the
        # group from the update (zero advantage) if it is STILL flat after the
        # optional judge. The judge (when it fired and produced a ranking) has
        # already re-injected within-tier variance, so `var_bearing` reflects
        # the post-judge state.
        if var_bearing:
            n_var_bearing += 1
        if args.group_var_floor > 0.0 and not var_bearing:
            adv_t = torch.zeros(len(trajs))
            n_dropped_groups += 1

        group_rollouts: list[Rollout] = []
        group_advs: list[float] = []
        for traj, a in zip(trajs, adv_t):
            for turn in traj.turns:
                group_rollouts.append(turn.rollout)
                group_advs.append(float(a.item()))
        combined_groups.append(group_rollouts)
        per_group_advantages.append(torch.tensor(group_advs))

    return (all_rollouts, combined_groups, per_group_advantages,
            first_pass_rewards_per_group, tier_hist, n_zero_var_before,
            n_lift_to_var, repair_n, repair_pass_n, n_judge_fired,
            n_var_bearing, n_dropped_groups)


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
    p.add_argument("--batch", type=int, default=3,
                    help="Problems per step per rank (B). 3 is the long-run-"
                         "robust value: batch=6 fits on typical steps but "
                         "OOMed in v11 at step 26 (PKM cast on long-prompt "
                         "MBPP draw); batch=4 OOMed in v11b at step ~95 "
                         "(lm_head 5.7 GiB allocation on a long sequence). "
                         "batch=3 leaves ~5-6 GiB headroom — enough for the "
                         "longest-tail prompts. Push to 4-6 only for short "
                         "(<100 step) runs where you can babysit OOMs.")
    p.add_argument("--grpo_n_group", type=int, default=4,
                    help="Rollouts per problem (N). v5 crashed at 8 → 4 is "
                         "the validated upper bound.")
    p.add_argument("--lr", type=float, default=1.5e-6,
                    help="Learning rate. v3 used 2e-6 (peak 17/164); v8 "
                         "ran 500 steps stably at 1e-6. 1.5e-6 is the "
                         "compromise default; lower for very long runs.")
    p.add_argument("--max_gen", type=int, default=384,
                    help="Max generated tokens per rollout (excl. thinks). "
                         "Below ~256 truncates valid MBPP solutions.")
    p.add_argument("--max_think_per_step", type=int, default=4)
    p.add_argument("--total_think_budget", type=int, default=120)
    p.add_argument("--think_budget_diversity", type=float, default=0.0,
                   help="Within-group think-budget diversity (FIX 2). 0 = OFF "
                        "(all N rollouts share --total_think_budget, "
                        "byte-identical to today). >0 spreads the N rollouts' "
                        "budgets linearly over [max(1, budget*(1-d)), budget] "
                        "so the group compares 'thought less' vs 'thought "
                        "more' variants; with the counterfactual ponder this "
                        "sharpens the cut-short signal toward selective "
                        "thinking. Clamped to [0,1].")
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0,
                    help="Gate output floor before emit threshold compare. "
                         "MUST be < --emit_threshold or thinking is silently "
                         "disabled (test_rl_grader_gate_floor pins this). "
                         "0.0 = let the trained gate decide freely.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--clip_eps", type=float, default=0.1,
                    help="PPO clip range. v2/v3/v8 stable at 0.1; 0.2 was "
                         "the original GRPO paper value but our small "
                         "batches favor tighter clipping.")
    p.add_argument("--kl_coef", type=float, default=0.05,
                   help="KL-to-reference penalty coefficient (KL between "
                        "the current policy and a frozen reference policy "
                        "= the starting --load_ckpt). The mandatory "
                        "stability mechanism for grader-RL — without it "
                        "the policy can drift arbitrarily far from the SFT "
                        "base and catastrophically collapse (v1 step ~350). "
                        "0.05 is v2/v3's validated value for ~200-step "
                        "runs. For 500+ step runs, increase to 0.15 (v7b "
                        "showed 0.08 was outrun at 500 steps; v8 with "
                        "0.15 stayed bounded).")
    p.add_argument("--ponder_cost", type=float, default=0.001,
                    help="Per-thinking-token penalty. 0 = always-think gate "
                         "collapse (v7b). 0.005 = catastrophic shutdown of "
                         "gate (v1 step 350). 0.001 = gentle pressure that "
                         "preserves selectivity (validated in v8).")
    p.add_argument("--ponder_shape", type=str, default="quadratic",
                    choices=["linear", "quadratic"])
    p.add_argument("--counterfactual", action="store_true", default=True)
    p.add_argument("--no_counterfactual", dest="counterfactual",
                    action="store_false")
    p.add_argument("--ponder_warmup_steps", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=25,
                    help="Milestone ckpt cadence. Fine granularity helps "
                         "identify the HumanEval peak post-hoc since RL "
                         "trajectories are often noisy near the optimum.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", type=str, default="mbpp_combined",
                    help="Code-grader dataset: mbpp (374, legacy), "
                         "mbpp_all (974), mbpp_plus (378), "
                         "mbpp_combined (1352 = all+plus, validated for "
                         "curriculum coverage).")
    p.add_argument("--extract_code_block", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="For distilled-SFT students that emit CoT + "
                         "```python ... ``` blocks: extract the python "
                         "fence and grade ONLY the extracted code "
                         "(otherwise the CoT prose breaks exec). Required "
                         "when --load_ckpt is a Qwen-distilled SFT ckpt — "
                         "without it HumanEval is structurally 0/164.")
    p.add_argument("--batch_turn0", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="EXPERIMENTAL (default OFF until a rollout-level GPU "
                         "equivalence diff confirms the real-kernel cache "
                         "merge; CPU-equivalent + temp-0 A/B was inconclusive). "
                         "Generate turn-0 rollouts for ALL B problems in ONE "
                         "B*N-row decode loop (vs B separate N-row loops). "
                         "DeltaNet decode is bandwidth-bound so this is a "
                         "~B× speedup on the dominant rollout phase, and is "
                         "byte-identical to the per-problem path on the "
                         "incremental decode path (each problem is prefilled "
                         "over only its own prompt). --no-batch_turn0 reverts "
                         "to the legacy per-problem turn-0 loop.")
    p.add_argument("--smoke", action="store_true",
                    help="Run a tiny end-to-end smoke (2 steps, B=N=2, max_gen=32).")
    p.add_argument("--state_readonly_at_think", action="store_true",
                    default=False,
                    help="Force the DeltaNet per-token write-gate beta to 0 at "
                         "think positions (think tokens READ the recurrent state "
                         "but never WRITE to it). Installs the state-readonly hook "
                         "on BOTH the policy and the frozen KL reference via "
                         "build_model_from_ckpt(force_state_readonly=True). "
                         "Default OFF (backwards-compat). The decisive lever for "
                         "training read-only thinking so thinks don't corrupt the "
                         "carried bindings the chain reasoning needs (D18).")
    p.add_argument("--dataset_jsonl", type=str, default=None,
                    help="Path to a synth_reasoning-schema JSONL (task_id, "
                         "prompt, tests, entry_point, prompt_is_code, "
                         "gold_solution). When set, problems are loaded from this "
                         "file via code_grader.load_synth_reasoning instead of "
                         "the --dataset LOADERS registry. Default None = use "
                         "--dataset (backwards-compat).")
    p.add_argument("--curriculum_filter", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Weight problem sampling by a per-problem pass-rate EMA "
                         "(peaks at p≈0.5 → variance-bearing zone). "
                         "Use --no-curriculum_filter for the uniform-sampling ablation.")
    p.add_argument("--curriculum_alpha", type=float, default=0.1)
    p.add_argument("--curriculum_init_pass_rate", type=float, default=0.25,
                    help="Pessimistic prior for unseen problems; lower → new "
                         "problems sampled more often until their EMA converges.")
    p.add_argument("--progressive_curriculum", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Progressive (easy→hard) curriculum: weight Gaussian-peaks "
                         "at target_p(step), where target_p decays linearly from "
                         "--curriculum_target_start to --curriculum_target_end "
                         "across --steps. Lets the model consolidate on achievable "
                         "problems before facing harder ones. Addresses the v7b "
                         "drift-into-suboptimal-equilibrium failure mode where "
                         "variance-only sampling converges to a narrow band of "
                         "lucky middle-difficulty problems from step 1.")
    p.add_argument("--curriculum_target_start", type=float, default=0.7,
                    help="Initial target pass-rate (easy problems weighted highest).")
    p.add_argument("--curriculum_target_end", type=float, default=0.2,
                    help="Final target pass-rate (hard problems weighted highest "
                         "by end of training).")
    p.add_argument("--curriculum_target_sigma", type=float, default=0.15,
                    help="Gaussian width of the target band. Wider = more spread "
                         "across difficulties at each step.")
    p.add_argument("--adaptive_curriculum", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Closed-loop curriculum: target_p = max(adaptive_floor, "
                         "1 - mean_p_seen). Parameter-free except the floor; "
                         "tracks the model's capability frontier automatically. "
                         "Mutually exclusive with --progressive_curriculum. "
                         "Default on — strictly better than the variance-only "
                         "fallback (v7b drifted into a narrow band of lucky "
                         "middle-difficulty problems; v8 progressive over-shot "
                         "the model's actual capability).")
    p.add_argument("--curriculum_adaptive_floor", type=float, default=0.3,
                    help="Lower bound on the adaptive target_p — prevents the "
                         "curriculum from sampling impossibly-hard problems even "
                         "when the model becomes strong (mean_p high).")
    p.add_argument("--iterative_repair", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="After grading, do a second rollout on failed attempts "
                         "with the grader's error_text as repair context. Both "
                         "rollouts contribute to the GRPO group → expanded "
                         "effective group size, fewer zero-variance groups.")
    # ---- Multi-turn agentic revision (RL_NEXT_DESIGN §2, §5) ----
    p.add_argument("--max_turns", type=int, default=1,
                    help="Multi-turn revision depth (RL_NEXT_DESIGN §2). "
                         "1 = byte-identical to today's single-shot trainer "
                         "(default). >1 enables the agentic loop: after grading "
                         "a non-passing rollout, build a feedback-augmented "
                         "repair prompt from error_text and regenerate, up to "
                         "N turns. Each turn becomes a SEPARATE ROW in the "
                         "policy update sharing one trajectory advantage "
                         "(§5.3). Recommended: 3.")
    p.add_argument("--turn_improvement_weight", type=float, default=0.0,
                    help="λ on the per-turn improvement bonus Σ max(0, Δ_turn) "
                         "in the trajectory reward (RL_NEXT_DESIGN §2.2). "
                         "0 = pure terminal reward (default, reproduces "
                         "single-turn). The bonus is HARD-CAPPED at "
                         "0.5·min_adjacent_gap so it can never cross a terminal "
                         "tier (terminal-dominates clamp); it only breaks "
                         "within-tier ties.")
    p.add_argument("--turn_cost", type=float, default=0.0,
                    help="Per-extra-turn penalty in the trajectory reward "
                         "(gentle pressure to solve in fewer turns). "
                         "0 = off (default). Recommended with max_turns>1: 0.02.")
    p.add_argument("--carry_history",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Carry the FULL turn history into each repair prompt "
                         "vs only the latest failed attempt + its error "
                         "(default off → bounded context, RL_NEXT_DESIGN §2.1).")
    # ---- Zero-variance hygiene (RL_NEXT_DESIGN §1.2 filter C) ----
    p.add_argument("--group_var_floor", type=float, default=0.0,
                    help="Drop groups whose trajectory-reward std is below "
                         "this floor from the policy update (RL_NEXT_DESIGN "
                         "§1.2 C) so the GRPO +1e-8 epsilon never injects "
                         "noise. 0 = off (default; legacy keeps every group, "
                         "and flat groups already advantage→0). Recommended: "
                         "1e-3.")
    # ---- LLM-judge tie-breaker (RL_NEXT_DESIGN §8) ----
    p.add_argument("--llm_judge",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Enable the LLM tie-breaker (RL_NEXT_DESIGN §8, "
                         "Phase 4). Fires ONLY on execution-tied groups that "
                         "would otherwise be dropped by --group_var_floor; asks "
                         "Qwen (vLLM) to listwise-rank the tied candidates by "
                         "closeness-to-correct, folded into the reward via "
                         "--judge_eps clamped by --judge_tier_margin so it "
                         "CANNOT cross an execution tier. Default OFF. Requires "
                         "a free second GPU (--judge_url) — never co-resident "
                         "with the trainer on one card.")
    p.add_argument("--judge_url", type=str, default=None,
                    help="vLLM OpenAI-compatible server endpoint for the judge "
                         "(Qwen on GPU 1). When set, the trainer POSTs ranking "
                         "requests over HTTP. None → in-process vLLM (only if a "
                         "GPU is genuinely free).")
    p.add_argument("--judge_model", type=str,
                    default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--judge_eps", type=float, default=0.02,
                    help="Within-tier judge perturbation magnitude "
                         "(RL_NEXT_DESIGN §8.2; ≤0.04 hard cap to stay inside "
                         "the smallest adjacent tier gap).")
    p.add_argument("--judge_tier_margin", type=float, default=0.025,
                    help="Clamp |reward − tier_base| so the judge provably "
                         "can't cross a tier (= 0.5·min_adjacent_gap).")
    p.add_argument("--judge_strip_comments",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Strip comments/docstrings before sending code to the "
                         "judge so it ranks behaviour-bearing code only "
                         "(RL_NEXT_DESIGN §8.5 anti-hacking guardrail).")
    p.add_argument("--repair_max_per_group", type=int, default=2)
    p.add_argument("--repair_min_failed", type=int, default=1,
                    help="Skip repair when fewer than this many rollouts in the "
                         "group failed (group is already variance-bearing).")
    p.add_argument("--repair_temperature", type=float, default=-1.0,
                    help="Sampling temperature for repair rollouts. -1 → "
                         "reuse --temperature.")
    p.add_argument("--grade_workers", type=int, default=8,
                    help="ThreadPoolExecutor size for code grading. Each "
                         "grade() spawns a subprocess (GIL irrelevant); "
                         "values above 8-12 don't help and risk fork "
                         "pressure on the host. 1 = sequential (legacy).")
    p.add_argument("--activation_checkpointing",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Wrap each Block's loss-bearing forward in "
                         "torch.utils.checkpoint during the policy update. "
                         "Trades ~30% extra compute for ~Nlayers× activation "
                         "memory reduction → enables higher --batch. Rollouts "
                         "use torch.no_grad so they're unaffected.")
    p.add_argument("--stochastic_gate",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Treat the thinking gate as a Bernoulli policy "
                        "variable during rollouts: sample emit/think from "
                        "sigmoid(gate) instead of thresholding at "
                        "emit_threshold. Same group-relative advantage that "
                        "rewards emit-token PPO also rewards the gate's "
                        "decisions, so the model can DISCOVER its own "
                        "optimal thinking pattern instead of imitating "
                        "Qwen's CoT. Default OFF — backwards-compat with "
                        "existing launchers.")
    p.add_argument("--gate_entropy_bonus", type=float, default=0.01,
                   help="Bernoulli-entropy regularization on the gate's "
                        "decisions (only used when --stochastic_gate). "
                        "Prevents collapse to always-think / never-think. "
                        "Subtracted from the loss → higher entropy lowers "
                        "loss. 0 disables.")
    p.add_argument("--gate_sample_range_low", type=float, default=0.0,
                   help="Lower bound of the σ(gate) range at which Bernoulli "
                        "sampling actually fires. σ values STRICTLY between "
                        "low and high are sampled; values outside use the "
                        "deterministic emit_threshold. Default 0.0 → "
                        "samples everywhere (legacy behaviour). v4 "
                        "recommended setting: 0.1 (with high=0.9).")
    p.add_argument("--gate_sample_range_high", type=float, default=1.0,
                   help="Upper bound of the σ(gate) sample range. See "
                        "--gate_sample_range_low. Default 1.0 → samples "
                        "everywhere. v4 recommended setting: 0.9.")
    p.add_argument("--gate_entropy_bonus_start", type=float, default=0.0,
                   help="Phase B: starting value of the entropy-bonus "
                        "curriculum (linear schedule). 0 disables the "
                        "curriculum and the static --gate_entropy_bonus "
                        "is used instead (backwards-compat). When > 0, "
                        "the bonus linearly decays from this value at "
                        "step 0 to --gate_entropy_bonus_end at "
                        "--gate_entropy_curriculum_steps, then stays at "
                        "the end value.")
    p.add_argument("--gate_entropy_bonus_end", type=float, default=0.0,
                   help="Phase B: final value of the entropy-bonus "
                        "curriculum. Only used when "
                        "--gate_entropy_bonus_start > 0.")
    p.add_argument("--gate_entropy_curriculum_steps", type=int, default=200,
                   help="Phase B: number of steps over which the entropy "
                        "bonus linearly decays from start to end.")
    p.add_argument("--policy_film_bypass",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="During the policy-update forward, skip FiLM K=3 "
                         "self-feed (run as K=1 single pass). Tested but "
                         "produces a log-prob mismatch with rollouts (which "
                         "use K=3) → PPO ratio drops to ~0.5 → all gradients "
                         "are clipped. NOT USEFUL as-is. Kept as a flag for "
                         "future experiments where rollouts also use K=1 "
                         "(currently blocked by the CLAUDE.md note that "
                         "_film_bypass at decode catastrophically breaks "
                         "T>0 sampling, commit 0d29d94).")
    args = p.parse_args()

    if args.smoke:
        args.steps = 2
        args.batch = 2
        args.grpo_n_group = 2
        args.max_gen = 32
        args.save_ckpt = ""  # don't pollute checkpoints/

    # ---- Distributed setup (torchrun) ----
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else "cuda")
    # Per-rank seed so each rank samples a DIFFERENT slice of problems
    # → effective batch = world_size × args.batch unique problems/step.
    torch.manual_seed(args.seed + rank)
    if is_main(rank):
        print(f"[rl-grader] world_size={world_size}  rank={rank}  "
              f"local_rank={local_rank}")

    # ---- Load model + tokenizer
    if is_main(rank):
        print(f"[rl-grader] loading {args.load_ckpt}")
    _force_sr = True if args.state_readonly_at_think else None
    model, cfg = build_model_from_ckpt(args.load_ckpt,
                                       force_state_readonly=_force_sr)
    if args.state_readonly_at_think:
        # Make the saved ckpt self-describing so build_model_from_ckpt
        # auto-enables the state-readonly hook on reload (eval / continue).
        cfg["state_readonly_at_think"] = True
    model = model.to(device)
    model.train()
    if args.activation_checkpointing and hasattr(model, "activation_checkpointing"):
        model.activation_checkpointing = True
    thinking_token_id = int(cfg.get("thinking_token_id"))

    # Frozen reference policy for KL penalty (the principled stability
    # mechanism — without it, GRPO can drift arbitrarily far from the SFT
    # base and catastrophically collapse). Lazy-loaded only when needed.
    ref_model = None
    if args.kl_coef > 0.0:
        if is_main(rank):
            print(f"[rl-grader] loading reference policy from {args.load_ckpt}"
                  f" (frozen, kl_coef={args.kl_coef})")
        ref_model, _ = build_model_from_ckpt(args.load_ckpt,
                                             force_state_readonly=_force_sr)
        # bf16 reference params: halves the resident KL-ref copy (~1.2GB). The
        # ref only supplies detached log-probs for a soft KL penalty, so bf16
        # precision is ample. (The policy stays fp32-master under autocast.)
        ref_model = ref_model.to(device).eval().bfloat16()
        for _p in ref_model.parameters():
            _p.requires_grad = False

    # Wrap policy in DDP if multi-rank. Reference stays unwrapped (frozen,
    # no grads). Rollouts call `model_module` directly (no DDP overhead
    # under no_grad); the policy-update forward calls `model` (DDP-wrapped)
    # so `.backward()` triggers automatic gradient all-reduce.
    model_module = model
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # `static_graph=False` because the model has data-dependent control
        # flow (FiLM bypass toggling, etc.). `find_unused_parameters=True`
        # is a safety net for the same reason — most params do receive
        # gradient on every step, but the FiLM α / PKM α / retrieval α
        # paths might not on every microbatch.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    if is_main(rank):
        print(f"  thinking_token_id={thinking_token_id}  "
              f"vocab={cfg['vocab_size']}  "
              f"params={sum(p.numel() for p in model_module.parameters())/1e6:.1f}M")
        if args.think_budget_diversity > 0.0:
            spread = compute_think_budget_spread(
                args.total_think_budget, args.grpo_n_group,
                args.think_budget_diversity)
            print(f"  think-budget diversity ON (d={args.think_budget_diversity}): "
                  f"per-rollout budgets = {spread}")

    # ---- Load problems
    if args.dataset_jsonl is not None:
        if is_main(rank):
            print(f"[rl-grader] loading dataset from JSONL: {args.dataset_jsonl}")
        problems = load_synth_reasoning(path=args.dataset_jsonl)
    else:
        if is_main(rank):
            print(f"[rl-grader] loading dataset: {args.dataset}")
        if args.dataset not in LOADERS:
            raise SystemExit(
                f"unknown --dataset {args.dataset!r}; choose from {list(LOADERS)}")
        problems = LOADERS[args.dataset]()
    if is_main(rank):
        print(f"  loaded {len(problems)} problems; will sample "
              f"{args.batch} per step per rank "
              f"({args.batch * world_size} effective)")

    # ---- Curriculum: per-problem pass-rate EMA used to weight sampling
    # toward the variance-bearing zone. Same `--seed` across ranks so
    # all ranks start with identical EMA tables; we then sync per-step
    # updates so they stay identical.
    import random
    cur_rng = random.Random(args.seed)
    problem_ids = [p.task_id for p in problems]
    curriculum = ProblemDifficultyEMA(
        problem_ids=problem_ids,
        alpha=args.curriculum_alpha,
        init_pass_rate=args.curriculum_init_pass_rate,
        progressive=args.progressive_curriculum,
        target_start=args.curriculum_target_start,
        target_end=args.curriculum_target_end,
        target_sigma=args.curriculum_target_sigma,
        total_steps=args.steps if args.progressive_curriculum else None,
        adaptive=args.adaptive_curriculum,
        adaptive_floor=args.curriculum_adaptive_floor,
    )
    try:
        _ckpt_blob = torch.load(args.load_ckpt, map_location="cpu",
                                 weights_only=False)
        if isinstance(_ckpt_blob, dict) and "curriculum_state_dict" in _ckpt_blob:
            curriculum.load_state_dict(_ckpt_blob["curriculum_state_dict"])
            if is_main(rank):
                stats = curriculum.stats()
                print(f"[rl-grader] resumed curriculum: "
                      f"n_seen={stats['n_seen']} mean_p={stats['mean_p']:.3f}")
        del _ckpt_blob
    except Exception as e:
        if is_main(rank):
            print(f"[rl-grader] curriculum: no resume ({e}); starting fresh")

    # ---- LLM-judge backend (RL_NEXT_DESIGN §8). Constructed lazily and only
    # on the main rank — it needs a free second GPU / a vLLM server. The
    # import is kept local so the unit tests never trigger vLLM.
    judge_backend = None
    if args.llm_judge and is_main(rank):
        from experiments.rl_multiturn import VLLMJudgeBackend
        print(f"[rl-grader] LLM judge ENABLED (url={args.judge_url}, "
              f"eps={args.judge_eps}, tier_margin={args.judge_tier_margin})")
        judge_backend = VLLMJudgeBackend(
            model=args.judge_model, server_url=args.judge_url,
            strip_comments_first=args.judge_strip_comments)

    # ---- Optimizer. BF16StateAdamW stores exp_avg/exp_avg_sq in bf16 (math
    # in fp32, lift→step→cast back) — lossless vs stock AdamW, saves ~half the
    # optimizer state (~2.4GB for 600M) → headroom for the co-resident judge.
    opt = BF16StateAdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                         weight_decay=0.01)

    # ---- Loop
    rng = torch.Generator().manual_seed(args.seed + rank)
    t0 = time.time()
    if is_main(rank):
        print(f"\n{'step':>5}  {'reward_mean':>10}  {'reward_max':>10}  "
              f"{'depth_mean':>10}  {'think_rate':>10}  {'pass_n':>6}  "
              f"{'ratio':>6}  {'loss':>8}  {'tier_hist':>30}")
    for step in range(1, args.steps + 1):
        # Sample B problems. Weighted by curriculum EMA when enabled
        # (peaks at p≈0.5); uniform otherwise. Both ranks use the same
        # cur_rng seed BUT step counter, plus per-rank salt → different
        # problems on different ranks each step.
        if args.curriculum_filter:
            weights = curriculum.sampling_weights(problem_ids, step=step)
            cur_rng.seed(args.seed + step * 1000003 + rank)
            idx = [problem_ids.index(pid) for pid in
                   cur_rng.choices(problem_ids, weights=weights, k=args.batch)]
        else:
            idx = torch.randint(0, len(problems), (args.batch,),
                                 generator=rng).tolist()
        batch_problems = [problems[i] for i in idx]

        # Generate N rollouts per problem in ONE batched forward per iter.
        # Use the UNWRAPPED `model_module` so DDP doesn't try to manage the
        # backward pass for no_grad rollouts (and so `prefill` /
        # `forward_step` attribute lookups work — DDP wrap masks them).
        model_module.eval()
        warmup_scale = min(1.0, step / max(1, args.ponder_warmup_steps))
        # Repair / multi-turn diagnostics shared by both code paths.
        repair_n = 0
        repair_pass_n = 0
        n_zero_var_before = 0
        n_lift_to_var = 0
        n_judge_fired = 0
        n_var_bearing = 0
        n_dropped_groups = 0

        if args.max_turns > 1:
            # ===== Multi-turn trajectory path (RL_NEXT_DESIGN §2, §5) =====
            (all_rollouts, combined_groups, per_group_advantages,
             first_pass_rewards_per_group, tier_hist, n_zero_var_before,
             n_lift_to_var, repair_n, repair_pass_n, n_judge_fired,
             n_var_bearing, n_dropped_groups) = \
                _run_multiturn_step(
                    args, batch_problems, model_module, tok, device,
                    thinking_token_id=thinking_token_id,
                    warmup_scale=warmup_scale, judge_backend=judge_backend)
        else:
            # ===== Legacy single-shot path (byte-identical to pre-change) =====
            all_rollouts: list[list[Rollout]] = []
            for prob in batch_problems:
                prompt_text = build_mbpp_prompt(prob)
                prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
                group_budget = compute_think_budget_spread(
                    args.total_think_budget, args.grpo_n_group,
                    args.think_budget_diversity)
                group = rollout_group_batched(
                    model_module, tok, prompt_ids,
                    n_rollouts=args.grpo_n_group,
                    thinking_token_id=thinking_token_id,
                    eos_token_id=tok.eos_token_id,
                    max_gen=args.max_gen,
                    max_think_per_step=args.max_think_per_step,
                    total_think_budget=group_budget,
                    emit_threshold=args.emit_threshold,
                    gate_floor=args.gate_floor,
                    temperature=args.temperature,
                    min_emit_before_eos=args.min_emit_before_eos,
                    stochastic_gate=args.stochastic_gate,
                    gate_sample_range_low=args.gate_sample_range_low,
                gate_sample_range_high=args.gate_sample_range_high,
            )
                all_rollouts.append(group)
            tier_hist = {}
            first_pass_error_texts: list[list[str | None]] = [
                [None] * len(group) for group in all_rollouts
            ]
            flat_jobs: list[tuple] = []
            flat_index: list[tuple[int, int]] = []
            for gi, (prob, group) in enumerate(zip(batch_problems, all_rollouts)):
                for ri, r in enumerate(group):
                    if args.extract_code_block:
                        extracted = extract_code_block(r.text)
                        code = extracted if extracted is not None else r.text
                    else:
                        code = build_mbpp_prompt(prob) + r.text
                    flat_jobs.append((prob, code))
                    flat_index.append((gi, ri))
            grade_results = grade_in_parallel(
                flat_jobs, timeout_s=5, max_workers=args.grade_workers)
            for (gi, ri), g_res in zip(flat_index, grade_results):
                r = all_rollouts[gi][ri]
                r.reward = float(g_res.score)
                r.tier = g_res.tier
                r.n_passed = g_res.n_passed
                r.n_tests = g_res.n_tests
                tier_hist[g_res.tier] = tier_hist.get(g_res.tier, 0) + 1
                first_pass_error_texts[gi][ri] = g_res.error_text

            # Snapshot first-pass rewards BEFORE running repair — these are
            # what feeds the curriculum EMA (the model's unprompted pass-rate;
            # repair would inflate the estimate).
            first_pass_rewards_per_group = [
                [float(r.reward) for r in group] for group in all_rollouts]

            # Iterative repair: for failed rollouts, do a second rollout
            # with the grader's error_text as repair context. Repair rollouts
            # join the same GRPO group (expanding N for that problem).
            repair_groups: list[list[Rollout]] = [[] for _ in batch_problems]
            if args.iterative_repair:
                rep_temp = (args.temperature if args.repair_temperature < 0
                            else args.repair_temperature)
                repair_jobs: list[tuple] = []
                repair_meta: list[tuple[int, str, Rollout]] = []
                zero_var_per_group: list[bool] = [False] * len(batch_problems)
                for gi, (prob, group) in enumerate(zip(batch_problems, all_rollouts)):
                    rewards_here = [float(r.reward) for r in group]
                    targets = select_repair_targets(
                        rewards_here,
                        max_per_group=args.repair_max_per_group,
                        min_failed=args.repair_min_failed,
                    )
                    if not targets:
                        continue
                    orig_passes = sum(1 for r in rewards_here if r >= 0.5)
                    orig_zero_var = orig_passes == 0 or orig_passes == len(group)
                    zero_var_per_group[gi] = orig_zero_var
                    if orig_zero_var:
                        n_zero_var_before += 1
                    orig_prompt = build_mbpp_prompt(prob)
                    for ti in targets:
                        failed_code = (extract_code_block(group[ti].text)
                                       if args.extract_code_block
                                       else group[ti].text) or group[ti].text
                        err_text = first_pass_error_texts[gi][ti] or ""
                        repair_prompt = build_repair_prompt(
                            orig_prompt, failed_code, err_text)
                        repair_ids = tok(repair_prompt,
                                          return_tensors="pt").input_ids.to(device)
                        single = rollout_group_batched(
                            model_module, tok, repair_ids,
                            n_rollouts=1,
                            thinking_token_id=thinking_token_id,
                            eos_token_id=tok.eos_token_id,
                            max_gen=args.max_gen,
                            max_think_per_step=args.max_think_per_step,
                            total_think_budget=args.total_think_budget,
                            emit_threshold=args.emit_threshold,
                            gate_floor=args.gate_floor,
                            temperature=rep_temp,
                            min_emit_before_eos=args.min_emit_before_eos,
                            stochastic_gate=args.stochastic_gate,
                            gate_sample_range_low=args.gate_sample_range_low,
                            gate_sample_range_high=args.gate_sample_range_high,
                        )
                        rr = single[0]
                        if args.extract_code_block:
                            extracted = extract_code_block(rr.text)
                            code = extracted if extracted is not None else rr.text
                        else:
                            code = repair_prompt + rr.text
                        repair_jobs.append((prob, code))
                        repair_meta.append((gi, repair_prompt, rr))
                if repair_jobs:
                    rep_grade_results = grade_in_parallel(
                        repair_jobs, timeout_s=5,
                        max_workers=args.grade_workers)
                    per_group_repair_rewards: list[list[float]] = [
                        [] for _ in batch_problems]
                    for (gi, _, rr), g_res in zip(repair_meta, rep_grade_results):
                        rr.reward = float(g_res.score)
                        rr.tier = g_res.tier
                        rr.n_passed = g_res.n_passed
                        rr.n_tests = g_res.n_tests
                        per_group_repair_rewards[gi].append(rr.reward)
                        repair_n += 1
                        if g_res.tier == "pass":
                            repair_pass_n += 1
                        repair_groups[gi].append(rr)
                    for gi, group in enumerate(all_rollouts):
                        if not zero_var_per_group[gi]:
                            continue
                        rewards_here = [float(r.reward) for r in group]
                        if group_became_variance_bearing(
                                rewards_here, per_group_repair_rewards[gi]):
                            n_lift_to_var += 1

            # Combined (per-group ragged) rollout sets for advantage computation.
            combined_groups: list[list[Rollout]] = [
                list(orig) + list(rep)
                for orig, rep in zip(all_rollouts, repair_groups)
            ]

            # Per-group advantages: each group may have a different N after
            # repair, so we compute (1, K) tensors group-by-group.
            per_group_advantages: list[torch.Tensor] = []
            for group in combined_groups:
                if not group:
                    per_group_advantages.append(torch.zeros(0))
                    continue
                rew = torch.tensor([[r.reward for r in group]])
                dep = torch.tensor([[float(r.depth) for r in group]])
                adv = compute_grpo_advantages_from_rewards(
                    rew, dep,
                    ponder_cost=args.ponder_cost,
                    ponder_shape=args.ponder_shape,
                    counterfactual=args.counterfactual,
                    ponder_warmup_scale=warmup_scale,
                )
                per_group_advantages.append(adv[0])

        # ---- Zero-variance hygiene (RL_NEXT_DESIGN §1.2 C) for the SINGLE-SHOT
        # legacy path: drop near-flat groups from the policy update so the GRPO
        # +1e-8 epsilon never injects noise. group_var_floor=0 (default) is a
        # no-op. The multi-turn path already computed n_var_bearing /
        # n_dropped_groups internally (against the trajectory reward), so we
        # only run this for max_turns<=1.
        if args.max_turns <= 1:
            for gi, group in enumerate(combined_groups):
                rews = [float(r.reward) for r in group]
                if group_is_variance_bearing(rews, args.group_var_floor):
                    n_var_bearing += 1
                elif args.group_var_floor > 0.0:
                    per_group_advantages[gi] = torch.zeros(len(group))
                    n_dropped_groups += 1

        # Flattened (B, N_orig) tensors kept for logging (depth_mean,
        # think_rate, reward_mean reflect the FIRST-PASS distribution,
        # i.e. the model's unprompted behaviour — what we care about).
        rewards = torch.tensor(
            [[r.reward for r in group] for group in all_rollouts])
        depths = torch.tensor(
            [[float(r.depth) for r in group] for group in all_rollouts])

        # ---- Curriculum EMA update (DDP-synced; use FIRST-PASS rewards
        # only, never repair). Each rank gathers its local
        # (problem_id, rewards) updates, all_gather_object across ranks,
        # then everyone applies the merged update — keeping EMA tables
        # bit-identical across ranks.
        local_updates: list[tuple[str, list[float]]] = []
        for prob, fp_rewards in zip(batch_problems, first_pass_rewards_per_group):
            local_updates.append((prob.task_id, fp_rewards))
        all_rank_updates = all_gather_object_list(local_updates, world_size)
        merged_updates = merge_rank_updates(all_rank_updates)
        for pid, merged_rewards in merged_updates:
            curriculum.update(pid, merged_rewards)

        # Policy update — one batched forward over ALL rollouts (B*N) in
        # the step, padded to the longest sequence. This replaces the prior
        # sequential per-rollout forward; expect ~10x speedup at B=4 N=4.
        model.train()
        opt.zero_grad(set_to_none=True)
        flat_rollouts = []
        flat_advantages = []
        for group, group_adv in zip(combined_groups, per_group_advantages):
            for r, adv in zip(group, group_adv):
                if not r.emit_token_ids:
                    continue
                flat_rollouts.append(r)
                flat_advantages.append(float(adv.item()))
        # DDP requires the SAME number of forward passes on every rank or
        # the all-reduce will hang. Synchronise "do we have rollouts this
        # step" across ranks: skip the policy update only if NO rank has
        # any rollouts to update on.
        has_rollouts = 1 if flat_rollouts else 0
        any_has_rollouts = all_reduce_sum_int(has_rollouts, world_size) > 0
        effective_entropy_bonus = compute_entropy_bonus(
            step,
            static=args.gate_entropy_bonus,
            start=args.gate_entropy_bonus_start,
            end=args.gate_entropy_bonus_end,
            total=args.gate_entropy_curriculum_steps,
        )
        if flat_rollouts:
            prev_film_bypass_policy = getattr(model_module, "_film_bypass", False)
            prev_film_bypass_ref = (getattr(ref_model, "_film_bypass", False)
                                     if ref_model is not None else False)
            if args.policy_film_bypass:
                model_module._film_bypass = True
                if ref_model is not None:
                    ref_model._film_bypass = True
            try:
                loss, mean_ratio, mean_kl, gate_stats = policy_loss_for_rollouts_batched(
                    flat_rollouts, flat_advantages, model,
                    clip_eps=args.clip_eps,
                    thinking_token_id=thinking_token_id,
                    temperature=args.temperature,
                    pad_id=int(tok.eos_token_id) if tok.eos_token_id is not None else 0,
                    ref_model=ref_model,
                    kl_coef=args.kl_coef,
                    stochastic_gate=args.stochastic_gate,
                    gate_entropy_bonus=effective_entropy_bonus,
                )
                loss.backward()
            finally:
                if args.policy_film_bypass:
                    model_module._film_bypass = prev_film_bypass_policy
                    if ref_model is not None:
                        ref_model._film_bypass = prev_film_bypass_ref
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_val = float(loss.item())
            ratio_val = float(mean_ratio.item())
            kl_val = float(mean_kl.item())
        elif any_has_rollouts and world_size > 1:
            # Some other rank has rollouts; we must still participate in
            # the all-reduce. Do a trivial forward+backward that produces
            # zero-norm gradients on all params, then take an
            # opt.step() that's a no-op for this rank.
            dummy = torch.zeros(
                1, device=device, requires_grad=True
            ) * sum(p.sum() for p in model.parameters() if p.requires_grad)
            dummy.backward()
            opt.zero_grad(set_to_none=True)
            loss_val, ratio_val, kl_val = float("nan"), float("nan"), float("nan")
            gate_stats = {"gate_ratio": float("nan"),
                          "gate_entropy": float("nan"),
                          "gate_fire_rate": float("nan")}
        else:
            loss_val, ratio_val, kl_val = float("nan"), float("nan"), float("nan")
            gate_stats = {"gate_ratio": float("nan"),
                          "gate_entropy": float("nan"),
                          "gate_fire_rate": float("nan")}

        # Logging.
        if step % args.log_every == 0:
            think_rate = (depths > 0).float().mean().item()
            pass_n = sum(1 for g in all_rollouts for r in g if r.tier == "pass")
            reward_mean_g = all_reduce_mean(rewards.mean().item(), world_size)
            reward_max_g = all_reduce_mean(rewards.max().item(), world_size)
            depth_mean_g = all_reduce_mean(depths.mean().item(), world_size)
            think_rate_g = all_reduce_mean(think_rate, world_size)
            pass_n_g = all_reduce_sum_int(pass_n, world_size)
            tier_str = "  ".join(f"{k}={v}" for k, v in sorted(tier_hist.items()))
            cur_stats = curriculum.stats(step=step)
            cur_mean_p_g = cur_stats["mean_p"]
            cur_pct_in_band_g = cur_stats["pct_in_band"]
            cur_target_p_str = (f" tgt={cur_stats['target_p']:.2f}"
                                if "target_p" in cur_stats else "")
            repair_n_g = all_reduce_sum_int(int(repair_n), world_size)
            repair_pass_n_g = all_reduce_sum_int(int(repair_pass_n), world_size)
            n_zero_var_before_g = all_reduce_sum_int(
                int(n_zero_var_before), world_size)
            n_lift_to_var_g = all_reduce_sum_int(int(n_lift_to_var), world_size)
            repair_pass_rate = (repair_pass_n_g / repair_n_g
                                if repair_n_g > 0 else 0.0)
            repair_lift = (n_lift_to_var_g / n_zero_var_before_g
                           if n_zero_var_before_g > 0 else 0.0)
            # Multi-turn / hygiene / judge diagnostics (RL_NEXT_DESIGN
            # §1, §2, §8). frac_var_bearing is the §7 Phase-0 gate metric.
            n_var_bearing_g = all_reduce_sum_int(int(n_var_bearing), world_size)
            n_dropped_g = all_reduce_sum_int(int(n_dropped_groups), world_size)
            n_judge_fired_g = all_reduce_sum_int(int(n_judge_fired), world_size)
            n_groups_g = all_reduce_sum_int(len(combined_groups), world_size)
            frac_var_bearing = (n_var_bearing_g / n_groups_g
                                if n_groups_g > 0 else 0.0)
            mt_str = ""
            if args.max_turns > 1 or args.group_var_floor > 0 or args.llm_judge:
                mt_str = (f"  mt(var={frac_var_bearing:.2f}, "
                          f"drop={n_dropped_g}, judge={n_judge_fired_g})")
            gate_str = ""
            sigma_hist_line = ""
            if args.stochastic_gate:
                # Phase A diagnostics: aggregate per-step σ-histogram +
                # sampled-vs-decisive partition across every rollout in the
                # step. Only meaningful when stochastic_gate is on (other
                # rollouts have no bucket counts).
                n_sampled_step = 0
                n_decisive_step = 0
                bucket_totals = [0, 0, 0, 0]
                for grp in all_rollouts:
                    for r in grp:
                        n_sampled_step += int(r.gate_n_sampled)
                        n_decisive_step += int(r.gate_n_decisive)
                        if r.gate_sigma_bucket_counts is not None:
                            for bi, c in enumerate(r.gate_sigma_bucket_counts):
                                bucket_totals[bi] += int(c)
                total_decisions = n_sampled_step + n_decisive_step
                if total_decisions > 0:
                    frac_sampled = n_sampled_step / total_decisions
                    frac_decisive = n_decisive_step / total_decisions
                else:
                    frac_sampled = float("nan")
                    frac_decisive = float("nan")
                # Stash log-only floats; nan when no gate decisions this step.
                ent_bonus_field = ""
                if args.gate_entropy_bonus_start > 0:
                    ent_bonus_field = f", ent_bonus={effective_entropy_bonus:.3f}"
                gate_str = (f"  gate(fire={gate_stats['gate_fire_rate']:.2f}, "
                            f"H={gate_stats['gate_entropy']:.3f}, "
                            f"ratio={gate_stats['gate_ratio']:.3f}, "
                            f"samp={frac_sampled:.2f}, "
                            f"dec={frac_decisive:.2f}"
                            f"{ent_bonus_field})")
                # Periodic σ-histogram (one extra line every 20 steps).
                if step % 20 == 0 and sum(bucket_totals) > 0:
                    total = sum(bucket_totals)
                    pcts = [100.0 * b / total for b in bucket_totals]
                    sigma_hist_line = (
                        f"  sigma_hist: <0.1: {pcts[0]:.0f}%  "
                        f"[0.1,0.5): {pcts[1]:.0f}%  "
                        f"[0.5,0.9): {pcts[2]:.0f}%  "
                        f">=0.9: {pcts[3]:.0f}%")
            if is_main(rank):
                print(f"{step:>5}  {reward_mean_g:>10.4f}  "
                      f"{reward_max_g:>10.4f}  {depth_mean_g:>10.2f}  "
                      f"{think_rate_g:>10.3f}  {pass_n_g:>6d}  "
                      f"{ratio_val:>6.3f}  {loss_val:>8.4f}  "
                      f"kl={kl_val:+.4f}{gate_str}  "
                      f"cur(n={cur_stats['n_seen']}, "
                      f"p={cur_mean_p_g:.2f}, "
                      f"band={cur_pct_in_band_g:.2f}{cur_target_p_str})  "
                      f"rep(n={repair_n_g}, pass={repair_pass_rate:.2f}, "
                      f"lift={repair_lift:.2f}){mt_str}  "
                      f"{tier_str:>30}")
                if sigma_hist_line:
                    print(sigma_hist_line)

        if args.save_ckpt and step % args.save_every == 0 and is_main(rank):
            pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
            # Save with `_step{N}` suffix so we keep every milestone
            # ckpt — RL peaks before collapse have happened before
            # (v1 → step-100 peak, v2 → step-300 peak); we need the
            # actual peak weights, not just the final ones.
            base = args.save_ckpt
            if base.endswith(".pt"):
                step_path = f"{base[:-3]}_step{step}.pt"
            else:
                step_path = f"{base}_step{step}"
            torch.save({"state_dict": model_module.state_dict(),
                        "step": step, "config": cfg,
                        "curriculum_state_dict": curriculum.state_dict()},
                        step_path)
            print(f"  [saved {step_path}]")

    if is_main(rank):
        print(f"\n[rl-grader] done in {time.time() - t0:.0f}s")
        if args.save_ckpt:
            torch.save({"state_dict": model_module.state_dict(),
                        "step": args.steps, "config": cfg,
                        "curriculum_state_dict": curriculum.state_dict()},
                        args.save_ckpt)
            print(f"final ckpt → {args.save_ckpt}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
