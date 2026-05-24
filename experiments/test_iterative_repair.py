"""Tests for iterative-repair helpers + end-to-end with code_grader."""
from __future__ import annotations

import pytest
import torch

from experiments.code_grader import Problem, grade
from experiments.iterative_repair import (
    REPAIR_HEADER,
    build_repair_prompt,
    group_became_variance_bearing,
    select_repair_targets,
)
from experiments.train_rl_grader import compute_grpo_advantages_from_rewards


def test_repair_prompt_snapshot():
    prompt = "# Write a function that adds two numbers.\n"
    failed = "def add(a, b):\n    return a - b\n"
    err = "1/1 assertions failed:\n  assert add(2, 3) == 5  ->  AssertionError"
    out = build_repair_prompt(prompt, failed, err)
    expected = (
        "# The previous attempt failed. Fix the code below.\n"
        "# Write a function that adds two numbers.\n"
        "# Attempted solution:\n"
        "def add(a, b):\n    return a - b\n"
        "# Error:\n"
        "# 1/1 assertions failed:\n"
        "#   assert add(2, 3) == 5  ->  AssertionError\n"
        "# Fix the code:\n"
    )
    assert out == expected


def test_repair_prompt_handles_empty_error():
    prompt = "# Foo.\n"
    out = build_repair_prompt(prompt, "x = 1", "")
    assert REPAIR_HEADER.strip() in out
    assert "# (no error text)" in out


def test_select_repair_targets_skips_saturated_group():
    assert select_repair_targets([1.0, 1.0, 1.0, 1.0], max_per_group=2,
                                  min_failed=1) == []
    assert select_repair_targets([1.0, 1.0, 1.0, 0.0], max_per_group=2,
                                  min_failed=2) == []


def test_select_repair_targets_picks_failed_indices():
    out = select_repair_targets([0.0, 1.0, 0.0, 0.2], max_per_group=2,
                                 min_failed=1)
    assert out == [0, 2]


def test_select_repair_targets_caps_at_max():
    out = select_repair_targets([0.0, 0.0, 0.0, 0.0, 0.0], max_per_group=2,
                                 min_failed=1)
    assert len(out) == 2


def test_select_repair_targets_threshold_inclusive():
    out = select_repair_targets([0.4, 0.5, 0.6], max_per_group=4,
                                 min_failed=1, pass_threshold=0.5)
    assert out == [0]


def test_group_variance_bearing_after_repair():
    assert group_became_variance_bearing([0.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0])
    assert not group_became_variance_bearing([0.0, 0.0, 0.0, 0.0],
                                              [0.0, 0.0])
    assert not group_became_variance_bearing([1.0, 0.0, 1.0, 0.0],
                                              [0.0, 1.0])
    assert not group_became_variance_bearing([1.0, 1.0, 1.0, 1.0],
                                              [1.0, 1.0])


def test_grade_repair_pair_fixes_name_error():
    problem = Problem(
        task_id="repair_test",
        prompt="Write a function `add(a, b)` returning a+b.\n",
        tests="def check(candidate):\n    assert candidate(2, 3) == 5\n",
        entry_point="add",
        prompt_is_code=False,
    )
    bad = "def add(a, b):\n    return a + bb\n"
    good = "def add(a, b):\n    return a + b\n"
    bad_res = grade(problem, bad, timeout_s=5)
    good_res = grade(problem, good, timeout_s=5)
    assert bad_res.tier != "pass"
    assert good_res.tier == "pass"
    assert good_res.score > bad_res.score


def test_zero_variance_group_with_repair_produces_nonzero_advantage():
    """A 4-rollout group of [0,0,0,0] + 2 repair rollouts [0,1] becomes a
    6-element group with non-zero advantage on the successful repair.
    Verifies the load-bearing claim: repair turns zero-gradient groups
    into gradient-bearing ones.
    """
    rewards_orig = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    depths_orig = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    adv_orig = compute_grpo_advantages_from_rewards(
        rewards_orig, depths_orig, ponder_cost=0.0,
        counterfactual=True, ponder_warmup_scale=1.0,
    )
    assert torch.allclose(adv_orig, torch.zeros_like(adv_orig), atol=1e-5)

    combined_rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    combined_depths = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    adv_combined = compute_grpo_advantages_from_rewards(
        combined_rewards, combined_depths, ponder_cost=0.0,
        counterfactual=True, ponder_warmup_scale=1.0,
    )
    assert adv_combined[0, -1].item() > 0.5
    assert (adv_combined[0, :-1] < 0).all()


def test_repair_prompt_includes_original_prompt_verbatim():
    p = "# Compute the factorial of n.\n"
    out = build_repair_prompt(p, "def f(): pass", "TypeError")
    assert p.rstrip("\n") in out
