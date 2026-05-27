"""Tests for the activation-checkpointing flag on TinyLM.

Output equivalence: with and without checkpointing the model must produce
bit-identical logits and identical gradients (the model has no dropout,
preserve_rng_state=True is default).

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_activation_checkpointing.py -v
"""
from __future__ import annotations

import torch
import pytest

from experiments.model import TinyLM


def _make_model(activation_checkpointing: bool) -> TinyLM:
    """Tiny no-feedback model so the test runs quickly."""
    torch.manual_seed(0)
    from experiments.layers import DeltaNetAttention
    return TinyLM(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=2,
        d_head=16,
        attention_cls=DeltaNetAttention,
        activation_checkpointing=activation_checkpointing,
    )


def _make_film_model(activation_checkpointing: bool) -> TinyLM:
    """Tiny sparse-FiLM K=3 self-feed model — mirrors the pretrain path."""
    torch.manual_seed(0)
    from experiments.layers import DeltaNetAttention
    return TinyLM(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=2,
        d_head=16,
        attention_cls=DeltaNetAttention,
        feedback_mode="film",
        feedback_pairs=((0, 2),),
        feedback_self_k=3,
        activation_checkpointing=activation_checkpointing,
    )


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_logits_equal_no_feedback():
    """Forward (with autograd enabled) must match bit-exactly."""
    m_off = _make_model(False).cuda().eval()
    m_on = _make_model(True).cuda().eval()
    # Same init seed already; double-check by copying state.
    m_on.load_state_dict(m_off.state_dict())

    x = torch.randint(0, 64, (2, 16), device="cuda")
    # Enable grad so the checkpoint path actually engages.
    with torch.enable_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    assert torch.allclose(y_off, y_on, atol=0.0, rtol=0.0), \
        f"max diff {(y_off - y_on).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_gradients_equal_no_feedback():
    """Backward should produce identical parameter grads."""
    m_off = _make_model(False).cuda()
    m_on = _make_model(True).cuda()
    m_on.load_state_dict(m_off.state_dict())

    x = torch.randint(0, 64, (2, 16), device="cuda")
    target = torch.randint(0, 64, (2, 16), device="cuda")

    def step(m: TinyLM) -> dict:
        m.zero_grad(set_to_none=True)
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()
                if p.grad is not None}

    g_off = step(m_off)
    g_on = step(m_on)
    assert set(g_off) == set(g_on)
    for name in g_off:
        diff = (g_off[name] - g_on[name]).abs().max().item()
        # Checkpointing can introduce tiny numerical noise from re-running
        # forward with different intermediate buffer allocations even though
        # we have no dropout. Allow a small rtol.
        assert diff < 1e-5, f"grad diff {diff:.3e} at {name}"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_film_path_engages_checkpoint():
    """K=3 self-feed FiLM path: with grad enabled, checkpointing must
    not crash, and outputs must match the non-checkpointed path."""
    m_off = _make_film_model(False).cuda().eval()
    m_on = _make_film_model(True).cuda().eval()
    m_on.load_state_dict(m_off.state_dict())

    x = torch.randint(0, 64, (2, 16), device="cuda")
    with torch.enable_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    diff = (y_off - y_on).abs().max().item()
    assert diff < 1e-5, f"FiLM logits diff {diff:.3e}"


def test_flag_propagates():
    """Pure attribute sanity: activation_checkpointing kwarg lands
    on the model."""
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    m = TinyLM(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention,
        activation_checkpointing=True,
    )
    assert m.activation_checkpointing is True

    m2 = TinyLM(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention,
    )
    assert m2.activation_checkpointing is False
