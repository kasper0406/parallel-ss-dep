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

import pytest
import torch

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.optim_utils import is_film_alpha
from experiments.thinking import (
    _latent_think_logits_grad,
    latent_think_step_input,
    latent_wm_injection,
)

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
    assert abs(float(m2.mem_alpha.detach()) - 0.37) < 1e-6, "mem_alpha must round-trip"


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


def test_mem_alpha_routes_to_no_weight_decay_group():
    """`mem_alpha`/`retrieval_input_alpha` are FiLM-α-style opt-in scalars (init
    0.1, grown by gradient only if useful). WD on them fights that curriculum, so
    they must match `is_film_alpha` → the alpha_wd group, like every other α.
    The cooperative model's mem_alpha must actually land there end-to-end."""
    assert is_film_alpha("mem_alpha")
    assert is_film_alpha("retrieval_input_alpha")
    assert is_film_alpha("feedback.2.projs.0.alpha")
    assert not is_film_alpha("memory.W_proj.weight")
    assert not is_film_alpha("lm_head.weight")
    # end-to-end: the cooperative model's mem_alpha is matched by the predicate
    m = _coop_model(cooperative=True)
    alpha_names = [n for n, _ in m.named_parameters() if is_film_alpha(n)]
    assert "mem_alpha" in alpha_names


def test_latent_wm_injection_off_paths_return_none():
    m = _coop_model(cooperative=True)
    # use_memory off → None even with mem_alpha present
    m.use_memory = False
    assert latent_wm_injection(m, grad=False) is None
    # cooperation off (no mem_alpha) → None even with WM on
    m2 = _coop_model(cooperative=False)
    assert latent_wm_injection(m2, grad=False) is None


def test_latent_wm_injection_positive_path():
    """After a forward, a cooperative WM model stashes the per-position
    injection; `latent_wm_injection` returns the LAST-position slice — detached
    for grad=False, graph-carrying for grad=True (so cooperation co-train can
    backprop into the WM read path + mem_alpha)."""
    m = _coop_model(cooperative=True)
    m.train()
    ids = torch.randint(0, VOCAB - 10, (2, 12))
    # run a grad-enabled forward so memory._last_injection_grad carries a graph
    _ = m(ids)
    inj_det = latent_wm_injection(m, grad=False)
    inj_grad = latent_wm_injection(m, grad=True)
    assert inj_det is not None and inj_grad is not None
    assert inj_det.shape == (2, 1, D), "must be the last-position (B,1,d) slice"
    assert inj_grad.shape == (2, 1, D)
    assert not inj_det.requires_grad, "grad=False slice must be detached"
    assert inj_grad.requires_grad, "grad=True slice must carry the autograd graph"


def test_grad_twin_uses_wm_and_backprops_to_mem_alpha():
    """The grad twin's think step is adapter(z)+mem_alpha·WM_inj. With cooperation
    ON: (a) a non-zero mem_alpha changes the logits vs mem_alpha=0 (the WM term
    actually flows), and (b) gradient reaches mem_alpha through the WM read."""
    m = _coop_model(cooperative=True)
    m.train()
    prefixes = torch.randint(0, VOCAB - 10, (3, 10))

    with torch.no_grad():
        m.mem_alpha.fill_(0.0)
        logits_off = _latent_think_logits_grad(
            m, prefixes, R=2, thinking_token_id=THINK_ID).clone()
        m.mem_alpha.fill_(0.6)
        logits_on = _latent_think_logits_grad(
            m, prefixes, R=2, thinking_token_id=THINK_ID).clone()
    assert not torch.allclose(logits_off, logits_on, atol=1e-5), \
        "non-zero mem_alpha must change the grad-twin logits (WM term must flow)"

    # gradient must reach mem_alpha through the WM read path
    m.zero_grad(set_to_none=True)
    logits = _latent_think_logits_grad(m, prefixes, R=2, thinking_token_id=THINK_ID)
    loss = logits.float().pow(2).mean()
    loss.backward()
    assert m.mem_alpha.grad is not None, "mem_alpha must receive gradient"
    assert torch.isfinite(m.mem_alpha.grad).all()
    assert float(m.mem_alpha.grad.abs().sum()) > 0.0, \
        "mem_alpha gradient must be non-zero (WM read contributes to the loss)"


def _save_coop_ckpt(path, cooperative=True, mem_alpha_val=0.37):
    m = _coop_model(cooperative=cooperative)
    if cooperative:
        with torch.no_grad():
            m.mem_alpha.fill_(mem_alpha_val)
    cfg = dict(
        arch="transformer", vocab_size=VOCAB, d_model=D, n_layers=2,
        n_heads=2, d_head=16, mem_size=64, thinking_token_id=THINK_ID,
        state_readonly_at_think=True,
    )
    torch.save({"state_dict": m.state_dict(), "step": 0, "config": cfg}, path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="build_model_from_ckpt is CUDA-only")
def test_build_from_ckpt_roundtrips_mem_alpha_and_sets_premem(tmp_path):
    """A cooperation ckpt (has mem_alpha) must reconstruct with the param value
    preserved AND `_latent_feedback_premem` forced on — cooperation implies
    premem, else the adapter would run on the contaminated post-memory hidden
    (the 2026-06-05 train/eval divergence the M0 review flagged)."""
    from experiments.eval_bracket_structure import build_model_from_ckpt
    ck = tmp_path / "coop.pt"
    _save_coop_ckpt(ck, cooperative=True, mem_alpha_val=0.37)
    model, cfg = build_model_from_ckpt(str(ck))
    assert hasattr(model, "mem_alpha"), "reconstructed model must create mem_alpha"
    assert abs(float(model.mem_alpha.detach()) - 0.37) < 1e-5, "mem_alpha must round-trip"
    assert getattr(model, "_latent_feedback_premem", False) is True, \
        "cooperation must imply premem on reload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="build_model_from_ckpt is CUDA-only")
def test_force_cooperative_attaches_mem_alpha_to_legacy_ckpt(tmp_path):
    """force_cooperative_latent_wm=True attaches a fresh (init 0.1) mem_alpha to a
    ckpt that lacks one (Stage A bootstrap), and still sets premem."""
    from experiments.eval_bracket_structure import build_model_from_ckpt
    ck = tmp_path / "legacy.pt"
    _save_coop_ckpt(ck, cooperative=False)   # no mem_alpha in state_dict
    model, cfg = build_model_from_ckpt(str(ck), force_cooperative_latent_wm=True)
    assert hasattr(model, "mem_alpha"), "force flag must attach a fresh mem_alpha"
    assert abs(float(model.mem_alpha.detach()) - 0.1) < 1e-6, "fresh mem_alpha inits at 0.1"
    assert getattr(model, "_latent_feedback_premem", False) is True
