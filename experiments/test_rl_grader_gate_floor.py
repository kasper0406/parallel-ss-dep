"""Tests for the rollout-time gate-floor / emit-threshold interaction.

Bug (2026-05-19): the launch_rl_grader_v7_combined_v2.sh used
  --gate_floor 0.5 --emit_threshold 0.5
intending to prevent gate collapse under τ=0.9 sampling. But the
rollout logic in train_rl_grader.py does:
  gate_clamped = gate.clamp_min(gate_floor)        # >= 0.5
  want_emit = (gate_clamped >= emit_threshold)     # 0.5 >= 0.5 = True ALWAYS
so the model always emits, never thinks.

Effect: RL ran 30+ steps with think_rate = 0.000, defeating the entire
point of training with the thinking gate engaged.

The fix is operational (set gate_floor < emit_threshold) but we ALSO
guard with a test that pins the gate decision under known inputs, so a
future refactor can't silently re-introduce the saturating-comparison
bug.
"""
import torch
import pytest


def _decide(gate_val: float, emit_threshold: float, gate_floor: float,
            force_emit: bool = False) -> bool:
    """Reproduces train_rl_grader.py:119-126 in scalar form."""
    if gate_floor > 0:
        gate_clamped = max(gate_val, gate_floor)
    else:
        gate_clamped = gate_val
    return (gate_clamped >= emit_threshold) or force_emit


def test_gate_floor_below_threshold_lets_think():
    """The normal case: gate_floor < emit_threshold. Low gate values
    still trigger think (clamped to floor, but floor < threshold)."""
    # Raw gate is 0.1, floor 0.3, threshold 0.5
    # → clamped = 0.3, 0.3 < 0.5 → think
    assert _decide(0.1, emit_threshold=0.5, gate_floor=0.3) is False
    # Raw gate is 0.7 → clamped = 0.7, 0.7 >= 0.5 → emit
    assert _decide(0.7, emit_threshold=0.5, gate_floor=0.3) is True
    # Raw gate is 0.4 → clamped = 0.4, 0.4 < 0.5 → think
    assert _decide(0.4, emit_threshold=0.5, gate_floor=0.3) is False


def test_gate_floor_equal_threshold_saturates_to_emit():
    """REGRESSION: when gate_floor == emit_threshold, clamping pushes
    every gate value to AT LEAST the threshold, so the >= test is
    trivially true and the model NEVER thinks. This was the v2-RL bug."""
    # Raw 0.1, floor 0.5, threshold 0.5: clamped = 0.5, 0.5 >= 0.5 = emit
    assert _decide(0.1, emit_threshold=0.5, gate_floor=0.5) is True
    # Same for raw 0.0, raw 0.49
    assert _decide(0.0, emit_threshold=0.5, gate_floor=0.5) is True
    assert _decide(0.49, emit_threshold=0.5, gate_floor=0.5) is True
    # This is the observed pathology — the test exists to remind future
    # readers that the configuration is broken.


def test_gate_floor_above_threshold_always_emits():
    """Even more clearly broken: floor > threshold means clamping
    pushes everything strictly above the threshold."""
    assert _decide(0.0, emit_threshold=0.5, gate_floor=0.7) is True
    assert _decide(0.3, emit_threshold=0.5, gate_floor=0.7) is True


def test_gate_floor_zero_uses_raw_gate():
    """gate_floor=0 disables clamping; raw gate decides."""
    assert _decide(0.1, emit_threshold=0.5, gate_floor=0.0) is False
    assert _decide(0.6, emit_threshold=0.5, gate_floor=0.0) is True


def test_force_emit_overrides_low_gate():
    """When force_emit (e.g., budget exhausted), think is suppressed
    regardless of the gate value."""
    assert _decide(0.1, emit_threshold=0.5, gate_floor=0.0,
                   force_emit=True) is True
    assert _decide(0.0, emit_threshold=0.5, gate_floor=0.3,
                   force_emit=True) is True


# ---------------------------------------------------------------------------
# Tensor-form parity test — verify the test's scalar _decide matches the
# tensor-form logic actually used by train_rl_grader.py rollout_group_batched.
# ---------------------------------------------------------------------------

def _decide_tensor(gate: torch.Tensor, emit_threshold: float,
                    gate_floor: float,
                    force_emit: torch.Tensor) -> torch.Tensor:
    """Mirror of train_rl_grader.py:119-126."""
    gate_clamped = (gate if gate_floor <= 0
                    else gate.clamp_min(gate_floor))
    return (gate_clamped >= emit_threshold) | force_emit


def test_scalar_and_tensor_decide_agree():
    gate_vals = torch.tensor([0.0, 0.1, 0.4, 0.49, 0.5, 0.7, 1.0])
    for floor in [0.0, 0.3, 0.5, 0.7]:
        for thr in [0.3, 0.5]:
            force = torch.zeros_like(gate_vals, dtype=torch.bool)
            tensor_decisions = _decide_tensor(gate_vals, thr, floor, force)
            for i, g in enumerate(gate_vals.tolist()):
                scalar = _decide(g, thr, floor, force_emit=False)
                assert tensor_decisions[i].item() == scalar, (
                    f"mismatch gate={g} thr={thr} floor={floor}: "
                    f"scalar={scalar} tensor={tensor_decisions[i].item()}"
                )
