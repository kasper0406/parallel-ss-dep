"""Tests for ProblemDifficultyEMA."""
from __future__ import annotations

import math

import pytest

from experiments.curriculum import ProblemDifficultyEMA, merge_rank_updates


def test_init_unseen_pass_rate():
    cur = ProblemDifficultyEMA(["a", "b", "c"], init_pass_rate=0.3)
    assert cur.ema["a"] == pytest.approx(0.3)
    assert cur.sampling_weight("brand_new") == pytest.approx(
        4 * 0.3 * 0.7 + 0.05)


def test_update_moves_toward_one():
    cur = ProblemDifficultyEMA(["x"], alpha=0.5, init_pass_rate=0.0)
    cur.update("x", [1.0])
    assert cur.ema["x"] == pytest.approx(0.5)
    cur.update("x", [1.0])
    assert cur.ema["x"] == pytest.approx(0.75)


def test_converges_to_one_under_constant_success():
    cur = ProblemDifficultyEMA(["x"], alpha=0.1, init_pass_rate=0.0)
    for _ in range(200):
        cur.update("x", [1.0])
    assert cur.ema["x"] == pytest.approx(1.0, abs=1e-6)


def test_converges_to_zero_under_constant_failure():
    cur = ProblemDifficultyEMA(["x"], alpha=0.1, init_pass_rate=1.0)
    for _ in range(200):
        cur.update("x", [0.0])
    assert cur.ema["x"] == pytest.approx(0.0, abs=1e-6)


def test_partial_pass_rate_from_mixed_rewards():
    cur = ProblemDifficultyEMA(["x"], alpha=1.0, init_pass_rate=0.0)
    cur.update("x", [1.0, 0.0, 1.0, 0.0])
    assert cur.ema["x"] == pytest.approx(0.5)


def test_threshold_is_half():
    cur = ProblemDifficultyEMA(["x"], alpha=1.0, init_pass_rate=0.0)
    cur.update("x", [0.4, 0.6])  # only 0.6 counts as "pass"
    assert cur.ema["x"] == pytest.approx(0.5)


def test_sampling_weight_peaks_at_half():
    cur = ProblemDifficultyEMA(["a", "b", "c"], eps=0.05)
    cur.ema["a"] = 0.0
    cur.ema["b"] = 0.5
    cur.ema["c"] = 1.0
    w_a = cur.sampling_weight("a")
    w_b = cur.sampling_weight("b")
    w_c = cur.sampling_weight("c")
    assert w_b > w_a
    assert w_b > w_c
    assert w_b == pytest.approx(1.0 + 0.05)


def test_sampling_weight_never_zero():
    cur = ProblemDifficultyEMA(["a", "b"], eps=0.05)
    cur.ema["a"] = 0.0
    cur.ema["b"] = 1.0
    for w in [cur.sampling_weight("a"), cur.sampling_weight("b")]:
        assert w >= 0.05


def test_sampling_weights_batch():
    cur = ProblemDifficultyEMA(["a", "b"], eps=0.01)
    cur.ema["a"] = 0.5
    cur.ema["b"] = 1.0
    ws = cur.sampling_weights(["a", "b", "a"])
    assert len(ws) == 3
    assert ws[0] == ws[2]


def test_state_dict_roundtrip_identical():
    cur1 = ProblemDifficultyEMA(["a", "b", "c"], alpha=0.2,
                                 init_pass_rate=0.3, eps=0.07)
    cur1.update("a", [1.0, 0.0])
    cur1.update("b", [0.0])
    state = cur1.state_dict()

    cur2 = ProblemDifficultyEMA([], alpha=0.0, init_pass_rate=0.0)
    cur2.load_state_dict(state)
    assert cur2.ema == cur1.ema
    assert cur2.alpha == cur1.alpha
    assert cur2.eps == cur1.eps
    assert cur2.init_pass_rate == cur1.init_pass_rate
    assert cur2.seen == cur1.seen
    for pid in cur1.ema:
        assert cur2.sampling_weight(pid) == cur1.sampling_weight(pid)


def test_stats_reports_in_band():
    cur = ProblemDifficultyEMA(["a", "b", "c", "d"], alpha=1.0)
    cur.update("a", [1.0])  # → 1.0, out of band
    cur.update("b", [0.0])  # → 0.0, out of band
    cur.update("c", [1.0, 0.0])  # → 0.5, in band
    s = cur.stats()
    assert s["n_seen"] == 3
    assert s["pct_in_band"] == pytest.approx(1.0 / 3.0)


def test_empty_rewards_noop():
    cur = ProblemDifficultyEMA(["x"], init_pass_rate=0.5)
    cur.update("x", [])
    assert cur.ema["x"] == pytest.approx(0.5)
    assert "x" not in cur.seen


def test_ddp_merge_disjoint_ranks_produces_identical_state():
    """Simulate two ranks updating disjoint problem sets; verify that
    once merged-via-broadcast each rank reaches the same EMA state."""
    pids = ["p1", "p2", "p3", "p4"]
    rank0_local = [("p1", [1.0, 0.0]), ("p2", [1.0])]
    rank1_local = [("p3", [0.0]), ("p4", [1.0, 1.0])]

    merged = merge_rank_updates([rank0_local, rank1_local])
    pid_to_rewards = dict(merged)
    assert set(pid_to_rewards.keys()) == {"p1", "p2", "p3", "p4"}
    assert pid_to_rewards["p1"] == [1.0, 0.0]
    assert pid_to_rewards["p4"] == [1.0, 1.0]

    cur_rank0 = ProblemDifficultyEMA(pids, alpha=0.3, init_pass_rate=0.25)
    cur_rank1 = ProblemDifficultyEMA(pids, alpha=0.3, init_pass_rate=0.25)
    for pid, rewards in merged:
        cur_rank0.update(pid, rewards)
        cur_rank1.update(pid, rewards)
    assert cur_rank0.ema == cur_rank1.ema
    assert cur_rank0.seen == cur_rank1.seen


def test_ddp_merge_overlapping_problem_pools_rewards():
    """When the same problem id appears on multiple ranks, merged rewards
    are concatenated so the single per-step EMA update sees the full
    cross-rank batch."""
    rank0 = [("p1", [1.0])]
    rank1 = [("p1", [0.0, 0.0])]
    merged = merge_rank_updates([rank0, rank1])
    pid_to_rewards = dict(merged)
    assert sorted(pid_to_rewards["p1"]) == [0.0, 0.0, 1.0]
    cur = ProblemDifficultyEMA(["p1"], alpha=1.0, init_pass_rate=0.0)
    cur.update("p1", pid_to_rewards["p1"])
    assert cur.ema["p1"] == pytest.approx(1.0 / 3.0)


def test_unseen_problem_gets_init_weight():
    cur = ProblemDifficultyEMA(["a"], init_pass_rate=0.25, eps=0.05)
    expected = 4.0 * 0.25 * 0.75 + 0.05
    assert cur.sampling_weight("a") == pytest.approx(expected)
    assert cur.sampling_weight("never_seen") == pytest.approx(expected)


def test_progressive_requires_total_steps():
    with pytest.raises(ValueError):
        ProblemDifficultyEMA(["a"], progressive=True)


def test_progressive_target_linear_schedule():
    cur = ProblemDifficultyEMA(
        ["a"], progressive=True, total_steps=100,
        target_start=0.7, target_end=0.2)
    assert cur.target_at(0) == pytest.approx(0.7)
    assert cur.target_at(50) == pytest.approx(0.45)
    assert cur.target_at(100) == pytest.approx(0.2)
    assert cur.target_at(200) == pytest.approx(0.2)


def test_progressive_weight_peaks_at_target():
    cur = ProblemDifficultyEMA(
        ["easy", "mid", "hard"],
        progressive=True, total_steps=100,
        target_start=0.7, target_end=0.2, target_sigma=0.1, eps=0.0)
    cur.ema["easy"] = 0.7
    cur.ema["mid"] = 0.45
    cur.ema["hard"] = 0.2

    w_step0 = cur.sampling_weights(["easy", "mid", "hard"], step=0)
    assert w_step0[0] > w_step0[1] > w_step0[2]

    w_step100 = cur.sampling_weights(["easy", "mid", "hard"], step=100)
    assert w_step100[2] > w_step100[1] > w_step100[0]

    w_step50 = cur.sampling_weights(["easy", "mid", "hard"], step=50)
    assert w_step50[1] > w_step50[0]
    assert w_step50[1] > w_step50[2]


def test_progressive_state_dict_roundtrip():
    cur1 = ProblemDifficultyEMA(
        ["a"], progressive=True, total_steps=200,
        target_start=0.6, target_end=0.1, target_sigma=0.2)
    cur1.update("a", [1.0])
    state = cur1.state_dict()

    cur2 = ProblemDifficultyEMA([], alpha=0.1, init_pass_rate=0.25)
    cur2.load_state_dict(state)
    assert cur2.progressive is True
    assert cur2.total_steps == 200
    assert cur2.target_start == 0.6
    assert cur2.target_end == 0.1
    assert cur2.target_sigma == 0.2
    assert cur2.sampling_weight("a", step=100) == cur1.sampling_weight("a", step=100)


def test_progressive_off_falls_back_to_variance_weighting():
    cur = ProblemDifficultyEMA(["a"], progressive=False, eps=0.05)
    cur.ema["a"] = 0.5
    assert cur.sampling_weight("a", step=0) == pytest.approx(4*0.5*0.5 + 0.05)
    assert cur.sampling_weight("a", step=999) == pytest.approx(4*0.5*0.5 + 0.05)


def test_progressive_stats_includes_target():
    cur = ProblemDifficultyEMA(
        ["a"], progressive=True, total_steps=100,
        target_start=0.7, target_end=0.2)
    cur.update("a", [1.0])
    s = cur.stats(step=50)
    assert "target_p" in s
    assert s["target_p"] == pytest.approx(0.45)


def test_adaptive_and_progressive_are_mutually_exclusive():
    with pytest.raises(ValueError):
        ProblemDifficultyEMA(
            ["a"], progressive=True, adaptive=True, total_steps=100)


def test_adaptive_target_at_cold_start():
    cur = ProblemDifficultyEMA(
        ["a"], adaptive=True, init_pass_rate=0.25, adaptive_floor=0.3)
    # No problems seen yet: mean_p = init_pass_rate = 0.25
    # target = max(0.3, 1 - 0.25) = max(0.3, 0.75) = 0.75
    assert cur.target_at(step=0) == pytest.approx(0.75)


def test_adaptive_target_lowers_as_model_improves():
    cur = ProblemDifficultyEMA(
        ["a"], adaptive=True, alpha=1.0, adaptive_floor=0.3)
    cur.update("a", [1.0])  # mean_p_seen = 1.0
    # target = max(0.3, 1 - 1.0) = max(0.3, 0.0) = 0.3 (floor)
    assert cur.target_at(step=0) == pytest.approx(0.3)


def test_adaptive_target_clamps_at_floor():
    cur = ProblemDifficultyEMA(
        ["a", "b"], adaptive=True, alpha=1.0, adaptive_floor=0.4)
    cur.update("a", [1.0])
    cur.update("b", [1.0])
    # mean_p = 1.0 → 1 - mean_p = 0.0 → clamped to floor 0.4
    assert cur.target_at(step=0) == pytest.approx(0.4)


def test_adaptive_weight_peaks_at_dynamic_target():
    cur = ProblemDifficultyEMA(
        ["easy", "mid", "hard"], adaptive=True, alpha=1.0,
        adaptive_floor=0.2, target_sigma=0.1, eps=0.0)
    cur.ema["easy"] = 0.8
    cur.ema["mid"] = 0.5
    cur.ema["hard"] = 0.2
    # Update once with all-fail so seen set is populated; mean_p_seen still
    # tracks current EMA values
    cur.seen.add("easy")
    cur.seen.add("mid")
    cur.seen.add("hard")
    # mean_p = (0.8+0.5+0.2)/3 = 0.5 → target = max(0.2, 0.5) = 0.5
    assert cur.target_at(step=0) == pytest.approx(0.5)
    ws = cur.sampling_weights(["easy", "mid", "hard"], step=0)
    # mid (p=0.5) is at target, should weigh highest
    assert ws[1] > ws[0]
    assert ws[1] > ws[2]


def test_adaptive_state_dict_roundtrip():
    cur1 = ProblemDifficultyEMA(
        ["a"], adaptive=True, adaptive_floor=0.25, target_sigma=0.2)
    cur1.update("a", [1.0, 0.0])
    state = cur1.state_dict()

    cur2 = ProblemDifficultyEMA([])
    cur2.load_state_dict(state)
    assert cur2.adaptive is True
    assert cur2.adaptive_floor == 0.25
    assert cur2.target_at(step=0) == cur1.target_at(step=0)
