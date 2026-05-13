"""Tests for experiments.eval_callback.EvalStopController.

The auto-stop logic decides when the entire pretrain ends. Bugs here
either burn a day of compute (false-negative) or stop too early
(false-positive) — both are expensive.
"""
from __future__ import annotations

import math

from experiments.eval_callback import EvalResult, EvalStopController


def _r(rate: float, tokens: int = 0, step: int = 0) -> EvalResult:
    return EvalResult(
        humaneval_pass_rate=rate, mbpp_pass_rate=None,
        tokens_seen=tokens, step=step,
        ckpt_path="", raw_log_tail="",
    )


def test_controller_never_stops_before_k_plus_one_evals() -> None:
    """k_consecutive_flat=2 needs 3 evals (1 anchor + 2 flat intervals).
    Don't stop on the first or second eval, even if both are 0."""
    c = EvalStopController(stop_threshold=0.01, k_consecutive_flat=2)
    c.append(_r(0.0))
    assert c.should_stop() is False
    c.append(_r(0.0))
    assert c.should_stop() is False, \
        "k=2 needs 3 evals before it can possibly stop"


def test_controller_stops_on_two_consecutive_flat_intervals() -> None:
    c = EvalStopController(stop_threshold=0.01, k_consecutive_flat=2)
    c.append(_r(0.10))   # anchor
    c.append(_r(0.105))  # Δ=+0.005 < 0.01 → flat
    assert c.should_stop() is False, "1 flat interval is not enough"
    c.append(_r(0.108))  # Δ=+0.003 < 0.01 → flat #2
    assert c.should_stop() is True


def test_controller_does_not_stop_when_one_interval_improves() -> None:
    c = EvalStopController(stop_threshold=0.01, k_consecutive_flat=2)
    c.append(_r(0.10))
    c.append(_r(0.10))   # flat
    c.append(_r(0.13))   # +0.03 ≥ threshold → resets the counter
    assert c.should_stop() is False
    c.append(_r(0.13))   # only 1 flat interval after the gain
    assert c.should_stop() is False


def test_controller_negative_delta_counts_as_flat() -> None:
    """If the eval goes DOWN, that's even-more-flat. Should count toward
    the consecutive-flat counter."""
    c = EvalStopController(stop_threshold=0.01, k_consecutive_flat=2)
    c.append(_r(0.20))
    c.append(_r(0.19))   # Δ=-0.01 (noise) → flat
    c.append(_r(0.17))   # Δ=-0.02 → flat
    assert c.should_stop() is True


def test_controller_clearly_above_threshold_does_not_stop() -> None:
    """Δ clearly above threshold (×3) must not stop. The rule is
    `delta >= threshold` → don't-stop, so we use a margin big enough
    to avoid IEEE-754 imprecision (note: `0.11 - 0.10 ≈ 0.009999...
    < 0.01` in FP, which is a corner case we don't try to defend
    against — eval pass-rates are quantized in 1/N steps so this
    won't bite at realistic thresholds)."""
    c = EvalStopController(stop_threshold=0.01, k_consecutive_flat=2)
    c.append(_r(0.10))
    c.append(_r(0.14))   # Δ=+0.04 — clearly above threshold
    c.append(_r(0.18))   # Δ=+0.04 — clearly above threshold
    assert c.should_stop() is False


def test_controller_nan_in_history_prevents_stop() -> None:
    """If an eval failed and returned NaN, don't stop on that data point
    — assume something went wrong and keep training (the user can
    investigate and force-stop)."""
    c = EvalStopController(stop_threshold=0.01, k_consecutive_flat=2)
    c.append(_r(0.10))
    c.append(_r(0.10))
    c.append(_r(float("nan")))
    assert c.should_stop() is False


def test_controller_k_equals_1_stops_immediately_on_first_flat() -> None:
    c = EvalStopController(stop_threshold=0.01, k_consecutive_flat=1)
    c.append(_r(0.20))
    assert c.should_stop() is False
    c.append(_r(0.20))   # 1 flat interval, k=1 → stop.
    assert c.should_stop() is True


def test_controller_rejects_k_less_than_one() -> None:
    try:
        EvalStopController(stop_threshold=0.01, k_consecutive_flat=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_controller_latest_delta_none_until_two_evals() -> None:
    c = EvalStopController()
    assert c.latest_delta() is None
    c.append(_r(0.1))
    assert c.latest_delta() is None
    c.append(_r(0.13))
    delta = c.latest_delta()
    assert delta is not None and abs(delta - 0.03) < 1e-9


def test_controller_summary_line_renders_with_and_without_history() -> None:
    c = EvalStopController()
    assert "no evals" in c.summary_line()
    c.append(_r(0.05, tokens=500_000_000, step=61_000))
    s = c.summary_line()
    # Should mention the token count, step, and rate; delta is NA on first.
    assert "61000" in s and "0.050" in s and "NA" in s
    c.append(_r(0.08, tokens=1_000_000_000, step=122_000))
    s = c.summary_line()
    assert "+0.030" in s


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
