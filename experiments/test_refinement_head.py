"""Tests for Phase D RefinementHead (THINKING_PLAN v5, 2026-05-27).

Validates the integration into the dev-branch TinyLM:
  1. Default off → no submodule, no `.refinement_head.` params
  2. Default off → forward bit-identical to no-head baseline
  3. With head + α=0 → byte-identical to no-head (load-bearing)
  4. With head + α set + force σ=1.0 → matches no-head (gate fully to trunk)
  5. With head + α set + force σ=0.0 → DIFFERS from no-head (refinement dominant)
  6. Refinement-head params route to AdamW (not Muon)
  7. State-dict round-trip preserves output
  8. Local-window attention is causal
  9. Local-window attention respects window bound

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_refinement_head.py -v
"""
from __future__ import annotations

import pytest
import torch

from experiments.layers import DeltaNetAttention
from experiments.model import RefinementHead, TinyLM
from experiments.optim_utils import _is_refinement_head, build_optimizer


def _make_model(*, use_refinement_head: bool = False, seed: int = 0,
                output_gate: bool = True,
                vocab_size: int = 64, d_model: int = 16, n_layers: int = 2,
                n_heads: int = 2, d_head: int = 8,
                refinement_head_window: int = 8,
                refinement_head_n_heads: int = 2,
                refinement_head_mlp_mult: int = 2,
                refinement_head_alpha_init: float = 0.3) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        attention_cls=DeltaNetAttention,
        output_gate=output_gate,
        use_refinement_head=use_refinement_head,
        refinement_head_window=refinement_head_window,
        refinement_head_n_heads=refinement_head_n_heads,
        refinement_head_mlp_mult=refinement_head_mlp_mult,
        refinement_head_alpha_init=refinement_head_alpha_init,
    )


# --------------------------------------------------------------------------
# 1. Default off → no params
# --------------------------------------------------------------------------

def test_default_off_no_params():
    m = _make_model(use_refinement_head=False)
    assert not hasattr(m, "refinement_head")
    rh_params = [n for n, _ in m.named_parameters()
                 if "refinement_head" in n]
    assert rh_params == []


def test_on_creates_expected_params():
    m = _make_model(use_refinement_head=True)
    assert hasattr(m, "refinement_head")
    names = {n for n, _ in m.named_parameters() if "refinement_head" in n}
    # Expect alpha + W_q/k/v/o + W_up/down + 2 LayerNorms (w,b each)
    expected_suffixes = {
        "refinement_head.alpha",
        "refinement_head.W_q.weight", "refinement_head.W_k.weight",
        "refinement_head.W_v.weight", "refinement_head.W_o.weight",
        "refinement_head.W_up.weight", "refinement_head.W_down.weight",
        "refinement_head.attn_norm.weight",
        "refinement_head.attn_norm.bias",
        "refinement_head.mlp_norm.weight",
        "refinement_head.mlp_norm.bias",
    }
    assert names == expected_suffixes
    # Default alpha init is 0.3 (v10 lesson — α=0 stays inert).
    assert abs(m.refinement_head.alpha.item() - 0.3) < 1e-6


def test_alpha_init_zero_supported():
    """Explicit alpha_init=0 preserves the byte-identity invariant
    for callers that want to recover the pre-v11 cold-start behaviour."""
    m = _make_model(use_refinement_head=True,
                    refinement_head_n_heads=2)
    # Default is 0.3 — but the kwarg flows through TinyLM
    m_zero = TinyLM(
        vocab_size=64, d_model=16, n_layers=2, n_heads=2, d_head=8,
        attention_cls=DeltaNetAttention,
        output_gate=True,
        use_refinement_head=True,
        refinement_head_window=8,
        refinement_head_n_heads=2,
        refinement_head_mlp_mult=2,
        refinement_head_alpha_init=0.0,
    )
    assert m_zero.refinement_head.alpha.item() == 0.0


# --------------------------------------------------------------------------
# 2. Default off → byte-identical
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_default_off_forward_byte_identical():
    m_a = _make_model(use_refinement_head=False, seed=0).cuda().eval()
    m_b = _make_model(use_refinement_head=False, seed=0).cuda().eval()
    x = torch.randint(0, 64, (1, 24), device="cuda")
    with torch.no_grad():
        y_a = m_a(x)
        y_b = m_b(x)
    assert torch.equal(y_a, y_b)


# --------------------------------------------------------------------------
# 3. With head + α=0 → byte-identical
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_alpha_zero_byte_identical_to_no_head():
    """The load-bearing invariant: a RefinementHead with explicit
    alpha_init=0 contributes nothing to the forward output."""
    m_off = _make_model(use_refinement_head=False, seed=0).cuda().eval()
    m_on = _make_model(use_refinement_head=True, seed=0,
                       refinement_head_alpha_init=0.0).cuda().eval()
    # Sync non-head weights so any difference is from the head.
    on_state = m_on.state_dict()
    off_state = m_off.state_dict()
    for k in off_state:
        if k in on_state and on_state[k].shape == off_state[k].shape:
            on_state[k].copy_(off_state[k])
    m_on.load_state_dict(on_state, strict=True)
    # Confirm α is still 0.
    assert m_on.refinement_head.alpha.item() == 0.0

    x = torch.randint(0, 64, (1, 24), device="cuda")
    with torch.no_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    # When alpha=0, refinement_head(h) = h, so mix = σ·h + (1-σ)·h = h.
    assert torch.allclose(y_off, y_on, atol=0, rtol=0), \
        f"alpha=0 must give byte-identical output; max diff {(y_off-y_on).abs().max().item()}"


# --------------------------------------------------------------------------
# 4-5. Forced σ semantics
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_force_sigma_one_matches_trunk():
    """σ=1 → mix is all trunk → output matches no-head baseline even
    when α is non-zero (the gate fully suppresses refinement)."""
    m_off = _make_model(use_refinement_head=False, seed=0).cuda().eval()
    m_on = _make_model(use_refinement_head=True, seed=0).cuda().eval()
    on_state = m_on.state_dict()
    off_state = m_off.state_dict()
    for k in off_state:
        if k in on_state and on_state[k].shape == off_state[k].shape:
            on_state[k].copy_(off_state[k])
    m_on.load_state_dict(on_state, strict=True)
    with torch.no_grad():
        m_on.refinement_head.alpha.fill_(0.5)  # non-zero head contribution
    m_on._force_gate_sigma = 1.0  # gate routes everything to trunk

    x = torch.randint(0, 64, (1, 24), device="cuda")
    with torch.no_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    assert torch.allclose(y_off, y_on, atol=1e-5, rtol=1e-5), \
        f"σ=1 must give trunk output regardless of α; max diff {(y_off-y_on).abs().max().item()}"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_force_sigma_zero_differs_from_trunk():
    """σ=0 → mix is all refinement → output DIFFERS from no-head when α>0."""
    m_off = _make_model(use_refinement_head=False, seed=0).cuda().eval()
    m_on = _make_model(use_refinement_head=True, seed=0).cuda().eval()
    on_state = m_on.state_dict()
    off_state = m_off.state_dict()
    for k in off_state:
        if k in on_state and on_state[k].shape == off_state[k].shape:
            on_state[k].copy_(off_state[k])
    m_on.load_state_dict(on_state, strict=True)
    with torch.no_grad():
        m_on.refinement_head.alpha.fill_(0.5)
    m_on._force_gate_sigma = 0.0  # gate routes everything to refinement

    x = torch.randint(0, 64, (1, 24), device="cuda")
    with torch.no_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    assert not torch.allclose(y_off, y_on, atol=1e-3), \
        "σ=0 with α=0.5 must differ from no-head (refinement is contributing)"


# --------------------------------------------------------------------------
# 6. Optimizer routing
# --------------------------------------------------------------------------

def test_is_refinement_head_predicate():
    assert _is_refinement_head("refinement_head.W_q.weight")
    assert _is_refinement_head("refinement_head.alpha")
    assert _is_refinement_head("refinement_head.attn_norm.bias")
    assert not _is_refinement_head("blocks.0.attn.W_q.weight")
    assert not _is_refinement_head("think_adapter.fc1.weight")
    assert not _is_refinement_head("memory.W_proj.weight")


def test_refinement_head_params_route_to_adamw_with_muon():
    """Under build_optimizer(optimizer='muon'), refinement-head params
    must all land in AdamW (the local attention matrices are NOT
    candidates for Newton-Schulz orthogonalisation)."""
    m = _make_model(use_refinement_head=True)
    opts, _ = build_optimizer(
        m, optimizer="muon", lr=1e-3, lr_muon=1e-3, alpha_wd=0.0,
        steps=10, wd=0.01, lr_schedule="cosine",
        warmup_steps=1, decay_frac=0.1)
    muon_opt, adamw_opt = opts[0], opts[1]
    refinement_param_ids = {id(p) for n, p in m.named_parameters()
                            if _is_refinement_head(n)}
    muon_ids = {id(p) for g in muon_opt.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in adamw_opt.param_groups for p in g["params"]}
    overlap_muon = refinement_param_ids & muon_ids
    overlap_adamw = refinement_param_ids & adamw_ids
    assert overlap_muon == set(), \
        f"Refinement-head params must NOT be in Muon: {overlap_muon}"
    assert overlap_adamw == refinement_param_ids, \
        f"All refinement-head params must be in AdamW; missing: " \
        f"{refinement_param_ids - overlap_adamw}"


# --------------------------------------------------------------------------
# 7. State-dict round-trip
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_state_dict_round_trip_preserves_output():
    m_save = _make_model(use_refinement_head=True, seed=0).cuda().eval()
    with torch.no_grad():
        m_save.refinement_head.alpha.fill_(0.3)
        for p in m_save.refinement_head.parameters():
            if p.ndim == 2:
                p.normal_(0, 0.02)
    sd = m_save.state_dict()

    m_load = _make_model(use_refinement_head=True, seed=99).cuda().eval()
    m_load.load_state_dict(sd, strict=True)

    x = torch.randint(0, 64, (1, 24), device="cuda")
    with torch.no_grad():
        y_save = m_save(x)
        y_load = m_load(x)
    assert torch.allclose(y_save, y_load, atol=0, rtol=0)


# --------------------------------------------------------------------------
# 8-9. RefinementHead module unit tests (CPU)
# --------------------------------------------------------------------------

def test_refinement_head_causality():
    """Perturbing position t must not affect output at positions < t."""
    torch.manual_seed(0)
    head = RefinementHead(d_model=16, n_heads=2, window=8, mlp_mult=2)
    with torch.no_grad():
        head.alpha.fill_(0.5)
        # Random non-trivial weights so the perturbation actually propagates.
        for p in head.parameters():
            if p.ndim == 2:
                p.normal_(0, 0.1)
    h = torch.randn(1, 12, 16)
    out_a = head(h)
    # Perturb position 6.
    h_perturbed = h.clone()
    h_perturbed[0, 6] = h[0, 6] + torch.randn(16) * 5.0
    out_b = head(h_perturbed)
    # Positions 0..5 must be unchanged.
    for t in range(6):
        diff = (out_a[0, t] - out_b[0, t]).abs().max().item()
        assert diff < 1e-5, \
            f"Position {t} should be unaffected by perturbing position 6 " \
            f"(causality), got max diff {diff}"
    # Position 6 onwards MAY change.
    diff_6 = (out_a[0, 6] - out_b[0, 6]).abs().max().item()
    assert diff_6 > 0, "Position 6 should change when its own input is perturbed"


def test_refinement_head_window_bound():
    """Perturbing position t must not affect positions ≥ t + window."""
    torch.manual_seed(0)
    window = 4
    head = RefinementHead(d_model=16, n_heads=2, window=window, mlp_mult=2)
    with torch.no_grad():
        head.alpha.fill_(0.5)
        for p in head.parameters():
            if p.ndim == 2:
                p.normal_(0, 0.1)
    h = torch.randn(1, 16, 16)
    out_a = head(h)
    h_perturbed = h.clone()
    h_perturbed[0, 5] = h[0, 5] + torch.randn(16) * 5.0
    out_b = head(h_perturbed)
    # Positions 5..(5+window-1)=8 are inside the window from t=5 and may change.
    # Position 5+window=9 onwards: position 5 is outside their window (9-5=4
    # is NOT < window=4), so they should NOT see the perturbation.
    for t in range(5 + window, 16):
        diff = (out_a[0, t] - out_b[0, t]).abs().max().item()
        assert diff < 1e-5, \
            f"Position {t} should be outside window from perturbed position " \
            f"5 (5 + window = {5 + window}); got max diff {diff}"
