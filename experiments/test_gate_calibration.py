"""CPU-only tests for the SFT gate-calibration aux loss (FIX 1,
THINKING_GATE_SELECTIVITY_2026_05_30.md).

Validates the LOSS machinery (not a real ckpt): a tiny CPU TinyLM with
CPU-friendly SoftmaxAttention exercises the snapshot + extra-forward + BCE
target construction. Synthetic separable cases validate that the BCE drives
the gate logit toward the target.

Run ONLY this file (the full suite has CUDA tests that would OOM a co-resident
training run):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \\
        experiments/test_gate_calibration.py -v
"""
from __future__ import annotations

import pytest
import torch

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.gate_calibration import (
    PAD_ID,
    compute_gate_calibration_loss,
    GateCalibrationResult,
)

THINK_ID = 5  # != PAD_ID (0)
EOS_ID = 1
VOCAB = 64
D_MODEL = 32


def _tiny_model(seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=2, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=128,
        output_gate=True, state_readonly_at_think=True,
    )
    m.thinking_token_id = THINK_ID
    return m


def _clean_batch(B=3, T=24, seed=7):
    g = torch.Generator().manual_seed(seed)
    # ids in [2, VOCAB-8): no think/pad/eos.
    input_ids = torch.randint(2, VOCAB - 8, (B, T), generator=g)
    targets = input_ids[:, 1:].clone()
    targets = torch.cat([targets, torch.full((B, 1), -100)], dim=1)
    return input_ids, targets


def _run_main_forward(model, input_ids):
    """Run the (grad-enabled) main forward and return the gate-logits snapshot
    exactly as sft_code.py does (model._last_gate_logits after forward)."""
    model.train()
    _ = model(input_ids)
    return model._last_gate_logits


# ---------------------------------------------------------------------------
# Pad != think guard.
# ---------------------------------------------------------------------------

def test_pad_equals_think_asserts():
    m = _tiny_model()
    input_ids, targets = _clean_batch()
    snap = _run_main_forward(m, input_ids)
    with pytest.raises(AssertionError):
        compute_gate_calibration_loss(
            m, input_ids, targets, snap,
            thinking_token_id=PAD_ID,  # == PAD_ID -> must raise
        )


# ---------------------------------------------------------------------------
# Shape / sample bounds.
# ---------------------------------------------------------------------------

def test_returns_result_and_bounds_count():
    m = _tiny_model()
    input_ids, targets = _clean_batch(B=3, T=24)
    snap = _run_main_forward(m, input_ids)
    gen = torch.Generator().manual_seed(0)
    res = compute_gate_calibration_loss(
        m, input_ids, targets, snap,
        thinking_token_id=THINK_ID, K=4, eos_id=EOS_ID,
        sample_frac=1.0, max_positions=8, generator=gen)
    assert isinstance(res, GateCalibrationResult)
    # max_positions caps the count.
    assert res.n_positions <= 8
    assert res.n_positions >= 1
    assert 0.0 <= res.target_frac_pos <= 1.0
    assert 0.0 <= res.mean_sigma <= 1.0
    assert res.loss.requires_grad


def test_sample_frac_bounds_count():
    m = _tiny_model()
    input_ids, targets = _clean_batch(B=4, T=40)
    snap = _run_main_forward(m, input_ids)
    gen = torch.Generator().manual_seed(1)
    # Many valid positions; sample_frac=0.1 should keep ~10% (well below cap).
    res_small = compute_gate_calibration_loss(
        m, input_ids, targets, snap, thinking_token_id=THINK_ID,
        sample_frac=0.1, max_positions=1000, eos_id=EOS_ID, generator=gen)
    res_full = compute_gate_calibration_loss(
        m, input_ids, targets, snap, thinking_token_id=THINK_ID,
        sample_frac=1.0, max_positions=1000, eos_id=EOS_ID, generator=gen)
    assert res_small.n_positions < res_full.n_positions


def test_no_valid_positions_returns_none():
    m = _tiny_model()
    B, T = 2, 8
    input_ids = torch.full((B, T), THINK_ID)   # all think -> no clean positions
    targets = torch.full((B, T), -100)
    snap = _run_main_forward(m, input_ids)
    res = compute_gate_calibration_loss(
        m, input_ids, targets, snap, thinking_token_id=THINK_ID, eos_id=EOS_ID)
    assert res is None


def test_weight_zero_is_noop_in_caller_contract():
    """The caller multiplies by the weight; weight=0 must make the loss a
    no-op contribution. We assert the helper itself returns a finite loss so
    that `weight * loss` is exactly 0 when weight==0 (no NaN/inf leak)."""
    m = _tiny_model()
    input_ids, targets = _clean_batch()
    snap = _run_main_forward(m, input_ids)
    res = compute_gate_calibration_loss(
        m, input_ids, targets, snap, thinking_token_id=THINK_ID, eos_id=EOS_ID)
    assert res is not None
    assert torch.isfinite(res.loss)
    assert float((0.0 * res.loss).item()) == 0.0


# ---------------------------------------------------------------------------
# Snapshot is used (not the post-extra-forward gate logits).
# ---------------------------------------------------------------------------

def test_uses_snapshot_not_post_extra_forward_gate():
    """The BCE must flow into the SNAPSHOT tensor we pass, even though the
    extra forward overwrites model._last_gate_logits. We pass a snapshot that
    requires grad and is a *different object*; after backward, only the
    snapshot must carry a gradient — and the gradient must be confined to the
    sampled positions."""
    m = _tiny_model()
    input_ids, targets = _clean_batch(B=2, T=20)
    model_out = m(input_ids)  # populate _last_gate_logits
    snap = m._last_gate_logits  # grad-carrying snapshot from the MAIN forward
    snap.retain_grad()
    obj_id_before = id(m._last_gate_logits)

    gen = torch.Generator().manual_seed(2)
    res = compute_gate_calibration_loss(
        m, input_ids, targets, snap, thinking_token_id=THINK_ID,
        sample_frac=1.0, max_positions=16, eos_id=EOS_ID, generator=gen)
    assert res is not None
    # The extra forward must have overwritten _last_gate_logits (different
    # object / shape) — proving we did NOT rely on it for the BCE.
    assert id(m._last_gate_logits) != obj_id_before or \
        m._last_gate_logits.shape != snap.shape
    res.loss.backward()
    assert snap.grad is not None
    # Gradient confined to sampled positions: at least one nonzero, and the
    # number of nonzero entries <= number of scored positions.
    nz = (snap.grad.abs() > 0).sum().item()
    assert nz >= 1
    assert nz <= res.n_positions


# ---------------------------------------------------------------------------
# BCE drives the gate logit toward the target on a synthetic separable case.
# ---------------------------------------------------------------------------

def test_bce_drives_gate_toward_target():
    """Freeze the trunk; train ONLY the gate head with the calibration loss on
    a FIXED set of (position, target) pairs and check the gate sigma moves
    toward the per-position target across steps.

    We fabricate a separable target by overriding the Δlogp sign via a
    monkeypatched extra-forward: positions with even index -> target 1, odd ->
    target 0. The gate head (linear on h) should learn to raise σ on the
    even positions relative to odd ones."""
    import experiments.gate_calibration as gc_mod

    m = _tiny_model()
    input_ids, targets = _clean_batch(B=2, T=16, seed=11)

    # Freeze everything except the gate head.
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.gate_head.parameters():
        p.requires_grad_(True)

    # Deterministic separable target: even absolute position -> helps.
    orig = gc_mod._post_think_logp

    def fake_post_think(model, prefixes, true_next, *, K, thinking_token_id):
        # Return a logp whose value (vs the baseline lp0 ~ -log V) makes
        # Δ>margin on exactly half the rows. We can't see absolute position
        # here, so encode the target in the prefix LENGTH parity is not
        # available; instead alternate by row index within the chunk.
        n = prefixes.shape[0]
        out = torch.full((n,), -1.0)  # high logp -> Δ>0 -> y=1
        out[1::2] = -1e3              # very low logp -> Δ<0 -> y=0
        return out

    gc_mod._post_think_logp = fake_post_think
    try:
        opt = torch.optim.Adam(m.gate_head.parameters(), lr=0.05)
        # Capture the y labels once (deterministic) by running a single call
        # with a generator that yields a stable position order.
        def step_once():
            _ = m(input_ids)
            snap = m._last_gate_logits
            res = compute_gate_calibration_loss(
                m, input_ids, targets, snap, thinking_token_id=THINK_ID,
                sample_frac=1.0, max_positions=64, eos_id=EOS_ID,
                generator=torch.Generator().manual_seed(0))
            return res

        res0 = step_once()
        loss0 = float(res0.loss.item())
        for _ in range(120):
            res = step_once()
            opt.zero_grad()
            res.loss.backward()
            opt.step()
        res1 = step_once()
        loss1 = float(res1.loss.item())
        # The BCE loss must drop substantially as the gate fits the target.
        assert loss1 < loss0 - 0.05
    finally:
        gc_mod._post_think_logp = orig
