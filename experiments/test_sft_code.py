"""Tests for experiments.sft_code.insert_think_bursts.

Used by:
  - sft_code.py itself (SFT-with-thinking bootstrap)
  - data_mix.MixedSourceStream (mixed-corpus pretrain inserts bursts
    at chunk boundaries)

The label-masking semantics are critical: think positions must be -100
so the model doesn't learn to *predict* think tokens (which would be
nonsensical at random insertion sites).
"""
from __future__ import annotations

import torch

from experiments.sft_code import insert_think_bursts


def _g(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_returns_unmodified_when_max_bursts_zero() -> None:
    ids = [10, 11, 12, 13, 14]
    lbls = [10, 11, 12, 13, 14]
    out_ids, out_lbls = insert_think_bursts(
        ids, lbls, thinking_token_id=999, max_len=10, max_bursts=0,
        rng=_g(0),
    )
    assert out_ids == ids
    assert out_lbls == lbls


def test_too_short_input_passes_through() -> None:
    """The function refuses to operate on very short inputs (<4 tokens)
    because there's nowhere meaningful to insert."""
    ids = [10, 11, 12]
    lbls = [10, 11, 12]
    out_ids, out_lbls = insert_think_bursts(
        ids, lbls, thinking_token_id=999, max_len=10, max_bursts=2,
        rng=_g(0),
    )
    assert out_ids == ids and out_lbls == lbls


def test_inserted_think_positions_are_labeled_minus_100() -> None:
    """Every inserted think token in input_ids must have -100 at the
    corresponding label position."""
    ids = list(range(100, 200))   # 100 tokens
    lbls = list(range(100, 200))
    THINK = 9999
    # Force at least one burst by using a seed where the RNG returns
    # n_bursts > 0; iterate a few seeds until we find one.
    for seed in range(20):
        out_ids, out_lbls = insert_think_bursts(
            ids, lbls, thinking_token_id=THINK, max_len=200,
            max_bursts=3, max_burst_depth=5, rng=_g(seed),
        )
        n_think = sum(1 for x in out_ids if x == THINK)
        if n_think > 0:
            # Verify every think position has -100 in labels.
            for i, x in enumerate(out_ids):
                if x == THINK:
                    assert out_lbls[i] == -100, (
                        f"think token at index {i} but label is "
                        f"{out_lbls[i]} != -100")
            # And every non-think position keeps its original label
            # (no -100 leaks).
            for i, x in enumerate(out_ids):
                if x != THINK:
                    assert out_lbls[i] != -100, (
                        f"non-think token at index {i} got -100 leak")
            return
    raise AssertionError("could not find a seed producing think bursts")


def test_input_and_label_length_match() -> None:
    """input_ids and labels must always have the same length, even
    after truncation."""
    ids = list(range(50))
    lbls = list(range(50))
    for seed in range(20):
        out_ids, out_lbls = insert_think_bursts(
            ids, lbls, thinking_token_id=9999, max_len=80,
            max_bursts=3, max_burst_depth=8, rng=_g(seed),
        )
        assert len(out_ids) == len(out_lbls), (
            f"len mismatch at seed={seed}: ids={len(out_ids)}, "
            f"lbls={len(out_lbls)}"
        )


def test_truncation_caps_at_max_len() -> None:
    """When the inserted think tokens push total length past max_len,
    output must be truncated to exactly max_len."""
    ids = list(range(100, 150))
    lbls = list(range(100, 150))
    out_ids, out_lbls = insert_think_bursts(
        ids, lbls, thinking_token_id=9999, max_len=40,
        max_bursts=5, max_burst_depth=10, rng=_g(7),
    )
    assert len(out_ids) <= 40
    assert len(out_lbls) <= 40


def test_real_token_order_preserved() -> None:
    """Real tokens must appear in their original relative order. The
    function only INSERTS think tokens — never deletes or reorders real
    tokens (except via end-truncation)."""
    ids = list(range(10, 110))
    lbls = list(range(10, 110))
    for seed in range(15):
        out_ids, _ = insert_think_bursts(
            ids, lbls, thinking_token_id=9999, max_len=200,
            max_bursts=2, max_burst_depth=4, rng=_g(seed),
        )
        real_tokens = [x for x in out_ids if x != 9999]
        # `real_tokens` must be a prefix of `ids` (since insertions
        # don't reorder, only end-truncation can drop real tokens).
        assert real_tokens == ids[: len(real_tokens)], (
            f"seed={seed}: real-token order changed: "
            f"{real_tokens[:5]} ... vs original {ids[:5]} ..."
        )


def test_n_bursts_within_max() -> None:
    """The number of insertion *sites* should not exceed max_bursts.
    Count distinct runs of THINK tokens in the output."""
    ids = list(range(10, 110))
    lbls = list(range(10, 110))
    MAX_BURSTS = 3
    THINK = 9999
    for seed in range(15):
        out_ids, _ = insert_think_bursts(
            ids, lbls, thinking_token_id=THINK, max_len=200,
            max_bursts=MAX_BURSTS, max_burst_depth=4, rng=_g(seed),
        )
        # Count maximal runs of consecutive THINK tokens.
        runs = 0
        in_run = False
        for x in out_ids:
            if x == THINK and not in_run:
                runs += 1
                in_run = True
            elif x != THINK:
                in_run = False
        assert runs <= MAX_BURSTS, (
            f"seed={seed}: {runs} burst runs but max_bursts={MAX_BURSTS}"
        )


def test_burst_depth_within_max() -> None:
    """No single burst run should exceed max_burst_depth tokens."""
    ids = list(range(10, 110))
    lbls = list(range(10, 110))
    MAX_DEPTH = 5
    THINK = 9999
    for seed in range(15):
        out_ids, _ = insert_think_bursts(
            ids, lbls, thinking_token_id=THINK, max_len=200,
            max_bursts=3, max_burst_depth=MAX_DEPTH, rng=_g(seed),
        )
        # Walk runs.
        run_len = 0
        for x in out_ids:
            if x == THINK:
                run_len += 1
                assert run_len <= MAX_DEPTH, (
                    f"seed={seed}: burst run of length {run_len} > "
                    f"max_burst_depth={MAX_DEPTH}"
                )
            else:
                run_len = 0


def test_determinism_with_same_seed() -> None:
    """Same input + same seed → same output. Critical for reproducible
    pretrain runs."""
    ids = list(range(50, 100))
    lbls = list(range(50, 100))
    a_ids, a_lbls = insert_think_bursts(
        ids, lbls, thinking_token_id=9999, max_len=100,
        max_bursts=3, max_burst_depth=5, rng=_g(42),
    )
    b_ids, b_lbls = insert_think_bursts(
        ids, lbls, thinking_token_id=9999, max_len=100,
        max_bursts=3, max_burst_depth=5, rng=_g(42),
    )
    assert a_ids == b_ids
    assert a_lbls == b_lbls


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
