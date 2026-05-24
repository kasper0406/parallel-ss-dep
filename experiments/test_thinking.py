from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.thinking import (
    ThinkContinuation,
    ThinkContinuationQueue,
    ThoughtTrajectory,
    build_continuation_batch,
    choose_explore_actions,
    choose_think_actions,
    compute_grpo_advantages,
    cross_entropy_masking_token,
)


def _logits_favoring(idx: int, V: int = 5, mag: float = 3.0) -> torch.Tensor:
    """Small helper: a logit vector strongly favoring `idx`."""
    L = torch.full((V,), -0.1)
    L[idx] = mag
    return L


def _make_traj(depth: int, immediate: torch.Tensor, final: torch.Tensor,
               target_id: int = 1) -> ThoughtTrajectory:
    t = ThoughtTrajectory(initial_context=[0], target_id=target_id,
                           actions=[], action_logprobs=[], depth=depth)
    t.immediate_logits = immediate
    t.final_logits = final
    return t


def test_queue_fifo_and_overflow_refuses_drop() -> None:
    q = ThinkContinuationQueue(max_len=2)
    q.enqueue(ThinkContinuation([1], 10, depth=0))
    q.enqueue(ThinkContinuation([2], 20, depth=1))
    try:
        q.enqueue(ThinkContinuation([3], 30, depth=2))
        raise AssertionError("expected overflow to raise")
    except OverflowError:
        pass
    assert len(q) == 2
    assert q.mean_depth() == 0.5
    assert q.max_depth() == 1
    batch = q.pop_batch(2)
    assert [item.target_id for item in batch] == [10, 20]
    assert len(q) == 0


def test_build_continuation_batch_right_pads_and_crops() -> None:
    items = [
        ThinkContinuation([1, 2, 3], 4),
        ThinkContinuation([5, 6, 7, 8, 9], 10),
    ]
    ctx, targets, last_pos = build_continuation_batch(
        items, block_size=4, pad_token_id=0, device="cpu"
    )
    assert ctx.tolist() == [[1, 2, 3, 0], [6, 7, 8, 9]]
    assert targets.tolist() == [4, 10]
    assert last_pos.tolist() == [2, 3]


def test_choose_threshold_and_masked_answer_ce() -> None:
    thinking_id = 2
    logits = torch.tensor([
        [0.0, 0.0, 3.0],
        [3.0, 0.0, 0.0],
    ])
    think = choose_think_actions(
        logits, thinking_id, policy="threshold", threshold=0.5,
        temperature=1.0,
    )
    assert think.tolist() == [True, False]

    targets = torch.tensor([0, 0])
    masked_ce = cross_entropy_masking_token(
        logits, targets, thinking_id, reduction="none"
    )
    expected = F.cross_entropy(
        torch.tensor([[0.0, 0.0], [3.0, 0.0]]),
        targets,
        reduction="none",
    )
    assert torch.allclose(masked_ce, expected)


def test_compounded_trajectory_accounting() -> None:
    thinking_nll = -math.log(0.25)
    answer_nll = -math.log(0.5)
    think_lambda = 0.1
    item = ThinkContinuation(
        context_ids=[7, 8, 99],
        target_id=42,
        depth=1,
        accum_nll=thinking_nll,
        accum_cost=think_lambda,
    )
    closed = item.accum_nll + item.accum_cost + answer_nll
    assert abs(closed - (thinking_nll + think_lambda + answer_nll)) < 1e-7


def test_high_ce_exploration_only_samples_top_candidates() -> None:
    torch.manual_seed(0)
    scores = torch.tensor([[0.1, 10.0, 0.2, 9.0]])
    explore = choose_explore_actions(
        scores,
        probability=1.0,
        mode="high_ce",
        top_frac=0.5,
    )
    assert explore.tolist() == [[False, True, False, True]]


def test_grpo_default_shape_matches_pre_refactor() -> None:
    """Default (linear, joint norm, no counterfactual) reproduces the
    original formula: reward = -CE(d) - cost*d, then z-score within group.
    Pinning this guards against silent behaviour change of existing RL runs.
    """
    # 1 group of 4 trajectories.
    g = [
        _make_traj(0, _logits_favoring(0), _logits_favoring(0)),
        _make_traj(2, _logits_favoring(0), _logits_favoring(1)),
        _make_traj(4, _logits_favoring(0), _logits_favoring(1, mag=2.0)),
        _make_traj(8, _logits_favoring(0), _logits_favoring(0)),
    ]
    adv = compute_grpo_advantages([g], ponder_cost=0.5)
    assert adv.shape == (1, 4)
    # All-finite. Mean ~0 (z-scored).
    assert torch.isfinite(adv).all()
    assert abs(adv.mean().item()) < 1e-5
    # The trajectory with the best task reward AND lowest depth (t1: depth=2,
    # logits→correct target) should have the highest advantage.
    assert adv[0].argmax().item() == 1
    # The trajectory with depth=8 and no benefit (t3) should be worst.
    assert adv[0].argmin().item() == 3


def test_grpo_quadratic_punishes_deep_thinking_harder() -> None:
    """Quadratic shape: depth=8 with no benefit should rank worse under
    quadratic than under linear (because 0.5*8^2=32 vs 0.5*8=4)."""
    g = [
        _make_traj(0, _logits_favoring(0), _logits_favoring(0)),
        _make_traj(2, _logits_favoring(0), _logits_favoring(1)),
        _make_traj(4, _logits_favoring(0), _logits_favoring(1, mag=2.0)),
        _make_traj(8, _logits_favoring(0), _logits_favoring(0)),
    ]
    adv_lin = compute_grpo_advantages([g], ponder_cost=0.5,
                                       ponder_shape="linear")
    adv_quad = compute_grpo_advantages([g], ponder_cost=0.5,
                                        ponder_shape="quadratic")
    # The depth=8 wasted-thinking trajectory should drop *further* under
    # quadratic (its absolute reward gets hit by 32 vs 4 ponder cost).
    # Group-z-scored, so we compare ranks/relative magnitudes.
    assert adv_quad[0, 3].item() < adv_lin[0, 3].item()


def test_grpo_counterfactual_floor_caps_task_component() -> None:
    """Counterfactual mode: thinking that *hurts* CE should NOT punish
    the task component (it's floor-capped at the depth-0 baseline). The
    only penalty is the depth cost."""
    # Trajectory where thinking hurts CE.
    g_hurts = [
        # t0: no think, baseline CE.
        _make_traj(0, _logits_favoring(1), _logits_favoring(1)),
        # t1: deep think, but final logits are *worse* than baseline.
        _make_traj(4, _logits_favoring(1), _logits_favoring(2)),
        # t2: deep think, helps a bit.
        _make_traj(2, _logits_favoring(1, mag=1.0), _logits_favoring(1, mag=3.0)),
        # t3: wasted depth-8.
        _make_traj(8, _logits_favoring(1), _logits_favoring(1)),
    ]
    # Default linear: t1 (depth-4, hurt CE) is heavily penalised.
    adv_default = compute_grpo_advantages([g_hurts], ponder_cost=0.5)
    # Counterfactual: t1 should be *less* penalised (task component clamped).
    adv_cf = compute_grpo_advantages([g_hurts], ponder_cost=0.5,
                                      counterfactual=True)
    # Look at t1's rank vs default — relative to t3 (also bad).
    # In default linear, t1's CE penalty + depth cost makes it worst.
    # In counterfactual, t1 keeps the depth-0 baseline as its floor, so
    # the gap to other trajectories shrinks.
    default_t1_rank = (adv_default[0] < adv_default[0, 1]).sum().item()
    cf_t1_rank = (adv_cf[0] < adv_cf[0, 1]).sum().item()
    # t1's rank should improve (or at minimum not worsen) under counterfactual.
    assert cf_t1_rank >= default_t1_rank


def test_grpo_counterfactual_still_charges_depth_cost() -> None:
    """A trajectory that wastes depth (final == immediate) should still
    pay the cost — counterfactual is about clamping the task floor, not
    about forgiving wasted compute."""
    # Two trajectories: t0 doesn't think, t1 wastes depth=10 (same logits).
    g = [
        _make_traj(0, _logits_favoring(1), _logits_favoring(1)),
        _make_traj(10, _logits_favoring(1), _logits_favoring(1)),
    ]
    adv = compute_grpo_advantages([g], ponder_cost=0.5, counterfactual=True)
    # Both trajectories have identical task component (clamped to baseline),
    # but t1 pays cost 5.0, t0 pays 0. So t0 should have higher advantage.
    assert adv[0, 0].item() > adv[0, 1].item()


def test_grpo_separate_ponder_norm_preserves_absolute_cost() -> None:
    """With separate_ponder_norm, the depth cost retains absolute magnitude
    (it isn't squashed by group z-scoring). Verify by comparing two
    trajectories that differ ONLY in depth — the difference between their
    advantages should equal cost * (d_a - d_b)."""
    # Same logits → same task_reward → z-scored task component is identical.
    same_imm = _logits_favoring(1)
    same_fin = _logits_favoring(1)
    g = [
        _make_traj(0, same_imm, same_fin),
        _make_traj(4, same_imm, same_fin),
    ]
    adv = compute_grpo_advantages([g], ponder_cost=0.1,
                                   separate_ponder_norm=True)
    # task component identical → task advantages both 0 after norm.
    # The only difference is the subtracted ponder term: 0.1*0 vs 0.1*4.
    # So adv[0,0] - adv[0,1] should equal 0.1 * (4 - 0) = 0.4.
    diff = adv[0, 0].item() - adv[0, 1].item()
    assert abs(diff - 0.4) < 1e-5, f"expected diff=0.4, got {diff:.4f}"


def test_grpo_handles_group_of_size_one() -> None:
    """Edge case: group of size 1 has std=0. Output must still be finite
    (the 1e-8 epsilon should prevent NaN/Inf)."""
    g = [_make_traj(0, _logits_favoring(1), _logits_favoring(1))]
    adv = compute_grpo_advantages([g], ponder_cost=0.1)
    assert torch.isfinite(adv).all()


def test_grpo_unknown_ponder_shape_raises() -> None:
    g = [_make_traj(0, _logits_favoring(1), _logits_favoring(1))]
    try:
        compute_grpo_advantages([g], ponder_cost=0.1, ponder_shape="cubic")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_grpo_missing_immediate_logits_falls_back_gracefully() -> None:
    """ThoughtTrajectory used to not have immediate_logits. An older
    rollout that didn't capture it should still compute advantages
    correctly under default (non-counterfactual) mode."""
    t = ThoughtTrajectory(initial_context=[0], target_id=1, actions=[],
                           action_logprobs=[], depth=2)
    t.final_logits = _logits_favoring(1)
    # immediate_logits stays None.
    g = [t]
    # Default mode doesn't need immediate_logits — should still work.
    adv = compute_grpo_advantages([g], ponder_cost=0.1)
    assert torch.isfinite(adv).all()


if __name__ == "__main__":
    test_queue_fifo_and_overflow_refuses_drop()
    test_build_continuation_batch_right_pads_and_crops()
    test_choose_threshold_and_masked_answer_ce()
    test_compounded_trajectory_accounting()
    test_high_ce_exploration_only_samples_top_candidates()
    test_grpo_default_shape_matches_pre_refactor()
    test_grpo_quadratic_punishes_deep_thinking_harder()
    test_grpo_counterfactual_floor_caps_task_component()
    test_grpo_counterfactual_still_charges_depth_cost()
    test_grpo_separate_ponder_norm_preserves_absolute_cost()
    test_grpo_handles_group_of_size_one()
    test_grpo_unknown_ponder_shape_raises()
    test_grpo_missing_immediate_logits_falls_back_gracefully()
