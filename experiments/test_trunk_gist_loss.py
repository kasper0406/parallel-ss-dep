"""Tests for the shared trunk multi-horizon gist loss
(experiments/gist_loss.py) — used by both sft_code.py and train_lm.py.

Mechanism: each per-horizon head predicts, from the trunk hidden state
h[t], the GIST of the upcoming window — the mean-pooled hidden state
over h[t+1 : t+1+K], stop-grad'd. Supervises the trunk to encode
"high-level direction".

These tests pin the actual shared functions (not a reference copy), so
both trainers are covered by one test.
"""
import pytest
import torch

from experiments.gist_loss import (
    windowed_future_gist, build_gist_heads, trunk_gist_loss, parse_horizons,
)


# --------------------------------------------------------------------
# parse_horizons
# --------------------------------------------------------------------
def test_parse_horizons_basic():
    assert parse_horizons("16,64,256") == [16, 64, 256]


def test_parse_horizons_dedups_and_sorts():
    assert parse_horizons("64, 16,16, 64,8") == [8, 16, 64]


def test_parse_horizons_rejects_empty_and_nonpositive():
    for bad in ("", "  ", "0,16", "-4,8"):
        with pytest.raises(ValueError):
            parse_horizons(bad)


# --------------------------------------------------------------------
# windowed_future_gist
# --------------------------------------------------------------------
def test_gist_matches_explicit_window_mean():
    """gist[:, t] must equal the explicit mean of h[t+1 : t+1+K]."""
    torch.manual_seed(0)
    B, T, d, K = 2, 12, 8, 3
    h = torch.randn(B, T, d)
    gist, vlen = windowed_future_gist(h, K)
    assert vlen == T - K
    assert gist.shape == (B, T - K, d)
    for t in range(vlen):
        expected = h[:, t + 1:t + 1 + K].mean(dim=1)
        assert torch.allclose(gist[:, t], expected, atol=1e-5)


def test_gist_none_when_window_too_large():
    h = torch.randn(1, 6, 4)
    assert windowed_future_gist(h, K=6) == (None, 0)
    assert windowed_future_gist(h, K=99) == (None, 0)


def test_gist_valid_len_shrinks_with_horizon():
    h = torch.randn(1, 20, 4)
    _, v_small = windowed_future_gist(h, K=4)
    _, v_large = windowed_future_gist(h, K=16)
    assert v_small == 16 and v_large == 4


# --------------------------------------------------------------------
# build_gist_heads
# --------------------------------------------------------------------
def test_build_gist_heads_one_per_horizon():
    heads = build_gist_heads(32, [16, 64, 256])
    assert set(heads.keys()) == {"16", "64", "256"}
    for k in ("16", "64", "256"):
        assert heads[k].weight.shape == (32, 32)
        assert heads[k].bias is None


# --------------------------------------------------------------------
# trunk_gist_loss
# --------------------------------------------------------------------
def test_loss_positive_with_random_head():
    """Random head vs the (stop-grad) gist target → cos < 1 → loss>0."""
    torch.manual_seed(1)
    B, T, d = 1, 24, 8
    horizons = [4, 8]
    h = torch.randn(B, T, d)
    loss = trunk_gist_loss(h, build_gist_heads(d, horizons), horizons)
    assert loss.item() > 0.1, f"expected ~0.5-1.0, got {loss.item()}"


def test_loss_zero_when_head_predicts_gist_exactly():
    """Identity head + constant-over-time h → every windowed gist equals
    h, so an identity prediction matches the target and loss is ~0."""
    torch.manual_seed(2)
    B, T, d, K = 1, 12, 8, 3
    heads = build_gist_heads(d, [K])
    heads[str(K)].weight.data = torch.eye(d)
    h = torch.randn(B, 1, d).expand(B, T, d).contiguous()
    loss = trunk_gist_loss(h, heads, [K])
    assert loss.item() < 1e-5, f"expected ~0, got {loss.item()}"


def test_loss_gradient_reaches_heads_and_trunk():
    """Gradient must reach every horizon head AND the trunk h via the
    prediction path; the gist target is stop-grad'd."""
    torch.manual_seed(3)
    B, T, d = 1, 24, 8
    horizons = [4, 8]
    heads = build_gist_heads(d, horizons)
    h = torch.randn(B, T, d, requires_grad=True)
    trunk_gist_loss(h, heads, horizons).backward()
    assert h.grad is not None and h.grad.abs().sum().item() > 0
    for k in horizons:
        g = heads[str(k)].weight.grad
        assert g is not None and g.abs().sum().item() > 0


def test_loss_skips_horizons_larger_than_sequence():
    """A horizon K >= T contributes nothing; the loss averages only the
    horizons that produced a valid window."""
    torch.manual_seed(4)
    B, T, d = 1, 12, 8
    heads = build_gist_heads(d, [4, 99])
    h = torch.randn(B, T, d)
    mixed = trunk_gist_loss(h, heads, [4, 99])
    only4 = trunk_gist_loss(h, heads, [4])
    assert torch.allclose(mixed, only4)


def test_loss_safe_when_all_horizons_too_large():
    """Every horizon >= T → loss 0, no NaN / crash."""
    torch.manual_seed(5)
    h = torch.randn(1, 6, 8)
    loss = trunk_gist_loss(h, build_gist_heads(8, [6, 10]), [6, 10])
    assert loss.item() == 0.0


def test_multihorizon_loss_is_mean_of_per_horizon():
    """The combined loss equals the mean of the per-horizon losses."""
    torch.manual_seed(6)
    B, T, d = 1, 40, 8
    horizons = [4, 12, 24]
    heads = build_gist_heads(d, horizons)
    h = torch.randn(B, T, d)
    combined = trunk_gist_loss(h, heads, horizons)
    per = [trunk_gist_loss(h, heads, [k]) for k in horizons]
    assert torch.allclose(combined, sum(per) / len(per), atol=1e-5)
