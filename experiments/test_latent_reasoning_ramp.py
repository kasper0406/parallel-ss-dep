"""Tests for the latent-reasoning co-train weight ramp + start-step gating
(the v12-destabilization safety knob, added 2026-06-14 for v13).

The ramp keeps the depth-matched latent-reasoning aux gradient negligible while
PKM bootstraps, then brings it to full --latent_reasoning_weight. Default
(warmup=0, start=0) must be byte-identical to the pre-ramp path (always 1.0).
"""
from experiments.train_lm import _latent_reasoning_ramp
from experiments.train_lm_args import build_parser


def test_default_off_is_full_weight_immediately():
    # warmup=0, start=0 → 1.0 at every step (old behaviour preserved).
    for step in (0, 1, 100, 5000, 19000):
        assert _latent_reasoning_ramp(step, 0, 0) == 1.0


def test_ramp_is_zero_before_start_step():
    assert _latent_reasoning_ramp(0, 3000, 3000) == 0.0
    assert _latent_reasoning_ramp(2999, 3000, 3000) == 0.0
    # exactly at start with a warmup → still 0 (no progress yet).
    assert _latent_reasoning_ramp(3000, 3000, 3000) == 0.0


def test_ramp_linear_between_start_and_warmup_end():
    # start=0, warmup=3000 → linear 0→1 over [0, 3000].
    assert _latent_reasoning_ramp(0, 0, 3000) == 0.0
    assert abs(_latent_reasoning_ramp(1500, 0, 3000) - 0.5) < 1e-9
    assert abs(_latent_reasoning_ramp(750, 0, 3000) - 0.25) < 1e-9


def test_ramp_clamps_to_one_after_warmup():
    assert _latent_reasoning_ramp(3000, 0, 3000) == 1.0
    assert _latent_reasoning_ramp(3001, 0, 3000) == 1.0
    assert _latent_reasoning_ramp(19000, 0, 3000) == 1.0


def test_ramp_respects_nonzero_start_offset():
    # start=2000, warmup=3000 → 0 until 2000, 0.5 at 3500, 1.0 at >=5000.
    assert _latent_reasoning_ramp(1999, 2000, 3000) == 0.0
    assert _latent_reasoning_ramp(2000, 2000, 3000) == 0.0
    assert abs(_latent_reasoning_ramp(3500, 2000, 3000) - 0.5) < 1e-9
    assert _latent_reasoning_ramp(5000, 2000, 3000) == 1.0
    assert _latent_reasoning_ramp(9000, 2000, 3000) == 1.0


def test_ramp_monotonic_nondecreasing():
    prev = -1.0
    for step in range(0, 6001, 50):
        v = _latent_reasoning_ramp(step, 0, 3000)
        assert v >= prev - 1e-12
        prev = v


def test_new_args_exist_with_backward_compatible_defaults():
    p = build_parser()
    ns = p.parse_args([])
    # Defaults must keep the old (no-ramp, from-step-0) behaviour.
    assert ns.latent_reasoning_start_step == 0
    assert ns.latent_reasoning_weight_warmup_steps == 0


def test_new_args_parse_v13_recipe():
    p = build_parser()
    ns = p.parse_args([
        "--latent_reasoning_weight", "0.05",
        "--latent_reasoning_start_step", "0",
        "--latent_reasoning_weight_warmup_steps", "3000",
    ])
    assert ns.latent_reasoning_weight == 0.05
    assert ns.latent_reasoning_start_step == 0
    assert ns.latent_reasoning_weight_warmup_steps == 3000
    # With this recipe the ramp covers the PKM α-floor window exactly.
    assert _latent_reasoning_ramp(
        0, ns.latent_reasoning_start_step,
        ns.latent_reasoning_weight_warmup_steps) == 0.0
    assert _latent_reasoning_ramp(
        3000, ns.latent_reasoning_start_step,
        ns.latent_reasoning_weight_warmup_steps) == 1.0
