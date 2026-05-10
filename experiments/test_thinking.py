from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from experiments.thinking import (
    ThinkContinuation,
    ThinkContinuationQueue,
    build_continuation_batch,
    choose_explore_actions,
    choose_think_actions,
    cross_entropy_masking_token,
)


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


if __name__ == "__main__":
    test_queue_fifo_and_overflow_refuses_drop()
    test_build_continuation_batch_right_pads_and_crops()
    test_choose_threshold_and_masked_answer_ce()
    test_compounded_trajectory_accounting()
    test_high_ce_exploration_only_samples_top_candidates()
