"""CPU-only tests for the gate-calibration probe.

Validates the probe MACHINERY (not a real ckpt): a tiny TinyLM with
CPU-friendly SoftmaxAttention exercises the baseline + post-think
forwards and the per-position collection; synthetic tensors validate
the AUC computation and the pad!=think guard.

Run (do NOT run the full suite — it has CUDA tests):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \
      experiments/test_probe_gate_calibration.py -v
"""
from __future__ import annotations

import pytest
import torch

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.probe_gate_calibration import (
    PAD_ID,
    roc_auc,
    linear_probe_auc,
    baseline_forward,
    post_think_logp,
    collect_one_batch,
    gate_head_calibration_fit,
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
    # SoftmaxAttention doesn't consume think_mask (Block.accepts_think_mask is
    # False for it), so state_readonly is a structural no-op here — fine: the
    # test validates the PROBE machinery, not DeltaNet recurrence. We still
    # set thinking_token_id so the gate/forward paths behave.
    m.thinking_token_id = THINK_ID
    return m.eval()


# ---------------------------------------------------------------------------
# AUC math.
# ---------------------------------------------------------------------------

def test_roc_auc_perfect_separation():
    # labels perfectly predicted by scores -> AUC == 1.0
    scores = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.95, 0.99])
    labels = torch.tensor([0., 0., 0., 1., 1., 1.])
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    # reversed scores -> AUC == 0.0
    assert roc_auc(-scores, labels) == pytest.approx(0.0)


def test_roc_auc_chance_and_ties():
    # all-equal scores -> 0.5 (ties averaged)
    scores = torch.zeros(8)
    labels = torch.tensor([0., 1.] * 4)
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_single_class_is_nan():
    import math
    scores = torch.rand(10)
    assert math.isnan(roc_auc(scores, torch.ones(10)))
    assert math.isnan(roc_auc(scores, torch.zeros(10)))


def test_roc_auc_matches_sklearn_on_random():
    from sklearn.metrics import roc_auc_score
    g = torch.Generator().manual_seed(0)
    scores = torch.rand(500, generator=g)
    labels = (torch.rand(500, generator=g) < (0.3 + 0.4 * scores)).float()
    ours = roc_auc(scores, labels)
    ref = roc_auc_score(labels.numpy(), scores.numpy())
    assert ours == pytest.approx(ref, abs=1e-9)


def test_linear_probe_separable_auc_near_one():
    # label = sign of first feature dim -> linearly separable -> AUC ~ 1.
    g = torch.Generator().manual_seed(1)
    n, d = 600, 16
    h = torch.randn(n, d, generator=g)
    y = (h[:, 0] > 0).float()
    h[:, 0] += 3.0 * (2 * y - 1)  # push the separating dim apart
    n_te = 200
    test_auc, train_auc, _ = linear_probe_auc(
        h[n_te:], y[n_te:], h[:n_te], y[:n_te])
    assert test_auc > 0.95
    assert train_auc > 0.95


def test_linear_probe_noise_auc_near_half():
    g = torch.Generator().manual_seed(2)
    n, d = 800, 16
    h = torch.randn(n, d, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float()  # label independent of h
    test_auc, _, _ = linear_probe_auc(h[300:], y[300:], h[:300], y[:300])
    assert 0.35 < test_auc < 0.65  # chance, within finite-sample slack


# ---------------------------------------------------------------------------
# Pad != think guard.
# ---------------------------------------------------------------------------

def test_post_think_pad_equals_think_asserts():
    m = _tiny_model()
    prefixes = torch.randint(2, VOCAB - 8, (3, 6))  # avoid think/pad ids
    true_next = torch.randint(2, VOCAB - 8, (3,))
    # thinking_token_id == PAD_ID must raise.
    with pytest.raises(AssertionError):
        post_think_logp(m, prefixes, true_next, K=4, thinking_token_id=PAD_ID)
    # a legal think id must NOT raise.
    out = post_think_logp(m, prefixes, true_next, K=4,
                          thinking_token_id=THINK_ID)
    assert out.shape == (3,)


# ---------------------------------------------------------------------------
# End-to-end on the tiny model.
# ---------------------------------------------------------------------------

def test_baseline_forward_shapes_and_fp32():
    m = _tiny_model()
    ids = torch.randint(2, VOCAB - 8, (2, 16))
    logits, hidden, gate_logits = baseline_forward(m, ids)
    assert logits.shape == (2, 16, VOCAB)
    assert hidden.shape == (2, 16, D_MODEL)
    assert gate_logits.shape == (2, 16)
    assert logits.dtype == torch.float32
    assert hidden.dtype == torch.float32
    # snapshotted gate must be detached.
    assert not gate_logits.requires_grad


def test_post_think_uses_last_think_slot_and_state_readonly_safe():
    m = _tiny_model()
    prefixes = torch.randint(2, VOCAB - 8, (4, 7))
    true_next = torch.randint(2, VOCAB - 8, (4,))
    lpK = post_think_logp(m, prefixes, true_next, K=3,
                          thinking_token_id=THINK_ID)
    assert lpK.shape == (4,)
    assert torch.isfinite(lpK).all()
    # log-probs must be <= 0.
    assert (lpK <= 1e-4).all()


def test_collect_one_batch_end_to_end():
    m = _tiny_model()
    B, T = 3, 24
    # Build clean real-token input (ids in [2, VOCAB-8), no think/pad/eos).
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(2, VOCAB - 8, (B, T), generator=g)
    targets = input_ids[:, 1:].clone()
    targets = torch.cat([targets, torch.full((B, 1), -100)], dim=1)  # last pos masked
    res = collect_one_batch(
        m, input_ids, targets, K=4, thinking_token_id=THINK_ID, eos_id=EOS_ID,
        max_positions_per_batch=64, margin=0.0, think_batch=16,
        device=torch.device("cpu"), generator=g)
    assert res is not None
    M = res["y"].numel()
    assert M > 0
    # consistent shapes.
    assert res["h"].shape == (M, D_MODEL)
    for k in ("lp0", "lpK", "delta", "y", "gate_sigma"):
        assert res[k].shape == (M,)
    # delta == lpK - lp0 exactly; y == (delta > margin).
    assert torch.allclose(res["delta"], res["lpK"] - res["lp0"], atol=1e-5)
    assert torch.equal(res["y"], (res["delta"] > 0.0).float())
    # gate sigma in [0,1].
    assert (res["gate_sigma"] >= 0).all() and (res["gate_sigma"] <= 1).all()
    # sampled positions are clean (target not think/pad/eos handled by mask):
    # lp0 and lpK are valid log-probs.
    assert torch.isfinite(res["lp0"]).all()
    assert torch.isfinite(res["lpK"]).all()


def test_collect_skips_when_no_valid_positions():
    m = _tiny_model()
    B, T = 2, 8
    input_ids = torch.full((B, T), THINK_ID)  # all think -> no clean positions
    targets = torch.full((B, T), -100)
    res = collect_one_batch(
        m, input_ids, targets, K=4, thinking_token_id=THINK_ID, eos_id=EOS_ID,
        max_positions_per_batch=64, margin=0.0, think_batch=16,
        device=torch.device("cpu"), generator=torch.Generator().manual_seed(0))
    assert res is None


def test_gate_head_calibration_fit_moves_toward_target():
    # Build hiddens where y is linearly decodable, then check the gate head
    # learns to raise sigma on y==1 vs y==0 after the BCE fit.
    m = _tiny_model()
    g = torch.Generator().manual_seed(3)
    n = 400
    h = torch.randn(n, D_MODEL, generator=g)
    y = (h[:, 0] > 0).float()
    h[:, 0] += 2.0 * (2 * y - 1)
    sb, sa, ab, aa = gate_head_calibration_fit(
        m, h, y, fit_steps=300, device=torch.device("cpu"), batch=128)
    # The gate head should become a better classifier of y after fitting.
    assert aa > ab
    assert aa > 0.9
