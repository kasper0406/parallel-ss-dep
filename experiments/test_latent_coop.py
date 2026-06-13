"""CPU tests for the WM×latent-thinking cooperation plumbing (M0).

Guards the load-bearing invariants of the cooperation foundation:
  - `mem_alpha` is a REAL ctor-registered param that round-trips through
    state_dict / build (fixes the audit bug where it was a runtime attr that
    load_state_dict(strict=False) silently dropped → coupling disabled at eval).
  - the cooperation ctor flag does NOT perturb the plain (no-think) forward
    (mem_alpha is used only in the latent think-step builder, never in forward).
  - the shared think-step builder formula (adapter(z) [+ mem_alpha·wm_inj]) and
    its byte-identity to the legacy adapter-only path when cooperation is off.

Run: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \
        experiments/test_latent_coop.py -v
"""
from __future__ import annotations

import torch

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.thinking import latent_think_step_input, latent_wm_injection

THINK_ID = 5
VOCAB = 64
D = 32


def _coop_model(seed=0, cooperative=True):
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D, n_layers=2, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=128,
        use_memory=True, thinking_token_id=THINK_ID, mem_size=64,
        use_latent_feedback_adapter=True,
        cooperative_latent_wm=cooperative,
        state_readonly_at_think=True,
    )
    m.eval()
    return m


def test_mem_alpha_registered_and_roundtrips():
    m = _coop_model(cooperative=True)
    assert hasattr(m, "mem_alpha"), "cooperative_latent_wm=True must register mem_alpha"
    assert isinstance(m.mem_alpha, torch.nn.Parameter)
    assert "mem_alpha" in m.state_dict(), "mem_alpha must be in state_dict"
    with torch.no_grad():
        m.mem_alpha.fill_(0.37)
    sd = m.state_dict()
    # reconstruct a fresh cooperative model and load → value preserved
    m2 = _coop_model(seed=1, cooperative=True)
    m2.load_state_dict(sd, strict=False)
    assert abs(float(m2.mem_alpha) - 0.37) < 1e-6, "mem_alpha must round-trip"


def test_no_cooperation_has_no_mem_alpha():
    m = _coop_model(cooperative=False)
    assert not hasattr(m, "mem_alpha"), \
        "default (cooperative_latent_wm=False) must NOT create mem_alpha"


def test_cooperation_ctor_inert_in_plain_forward():
    """mem_alpha is used only in the latent think-step builder, never in the
    forward → a plain (no-think) forward must be unaffected by its value."""
    m = _coop_model(cooperative=True)
    ids = torch.randint(0, VOCAB - 10, (2, 12))   # no think tokens
    with torch.no_grad():
        out0 = m(ids)
        l0 = (out0[0] if isinstance(out0, tuple) else out0).clone()
        m.mem_alpha.fill_(9.9)                    # huge — must not matter
        out1 = m(ids)
        l1 = out1[0] if isinstance(out1, tuple) else out1
    assert torch.allclose(l0, l1, atol=1e-6), \
        "mem_alpha must not affect the plain forward (it's a latent-loop scalar)"


def test_step_input_byte_identical_without_mem_alpha():
    """No mem_alpha (cooperation off) → builder == adapter(z) exactly."""
    m = _coop_model(cooperative=False)
    z = torch.randn(1, 1, D)
    got = latent_think_step_input(m, z, wm_inj=torch.randn(1, 1, D))
    want = m.apply_latent_feedback_adapter(z)
    assert torch.allclose(got, want, atol=1e-6), \
        "with no mem_alpha the WM term must be ignored (legacy path)"


def test_step_input_adds_alpha_weighted_injection():
    m = _coop_model(cooperative=True)
    with torch.no_grad():
        m.mem_alpha.fill_(0.5)
    z = torch.randn(1, 1, D)
    inj = torch.randn(1, 1, D)
    got = latent_think_step_input(m, z, wm_inj=inj)
    want = m.apply_latent_feedback_adapter(z) + 0.5 * inj
    assert torch.allclose(got, want, atol=1e-6)
    # and None injection collapses to adapter-only even with mem_alpha present
    got_none = latent_think_step_input(m, z, wm_inj=None)
    assert torch.allclose(got_none, m.apply_latent_feedback_adapter(z), atol=1e-6)


def test_latent_wm_injection_off_paths_return_none():
    m = _coop_model(cooperative=True)
    # use_memory off → None even with mem_alpha present
    m.use_memory = False
    assert latent_wm_injection(m, grad=False) is None
    # cooperation off (no mem_alpha) → None even with WM on
    m2 = _coop_model(cooperative=False)
    assert latent_wm_injection(m2, grad=False) is None
