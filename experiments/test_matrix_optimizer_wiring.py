"""Tests for the `--matrix_optimizer {muon,fused_deltanet_ns}` wiring in
`optim_utils.build_optimizer`.

The fairness guarantee this pins (the whole point of the production A/B):
  1. default / `"muon"` is BYTE-IDENTICAL to the legacy path (every existing
     launcher + checkpoint unaffected — the flag is purely additive).
  2. `"fused_deltanet_ns"` swaps ONLY the q/k/v/b orthogonalization: the
     matrix-optimizer PARAM SET is identical to the muon arm, the AdamW groups
     are identical, and the o_proj/MLP whole-matrix update is byte-identical to
     Muon — so a muon-vs-fused A/B isolates exactly the per-head NS boundary.

CPU-only + tiny DeltaNet so it runs in CI without a GPU (no forward needed —
the optimizer math is fed fake grads, mirroring test_deltanet_precond_fused.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.optim_utils import build_optimizer
from experiments.exp_deltanet_precond_fused import (
    FusedDeltaNetMuon, build_production_model, matrix_param_set)
from experiments.exp_deltanet_precond_optim import build_units_from_model


_OPT_KW = dict(optimizer="muon", lr=1e-3, lr_muon=5e-3, alpha_wd=0.0,
               steps=100, wd=0.01, lr_schedule="wsd", warmup_steps=10,
               verbose=False)


def _model(seed=0):
    torch.manual_seed(seed)
    return build_production_model(d_model=128, n_layers=3, n_heads=4,
                                  d_head=32, vocab=256, max_T=64, device="cpu")


def _name_of(model):
    return {id(p): n for n, p in model.named_parameters()}


def _matrix_opt_names(model, matrix_opt):
    name_of = _name_of(model)
    out = set()
    for g in matrix_opt.param_groups:
        for p in g["params"]:
            out.add(name_of[id(p)])
    return out


def _adamw_named_groups(model, adamw_opt):
    name_of = _name_of(model)
    return [sorted(name_of[id(p)] for p in g["params"])
            for g in adamw_opt.param_groups]


def _qkvb_names(model):
    _, units, _ = build_units_from_model(model, mode="perhead")
    name_of = _name_of(model)
    return {name_of[id(p)] for u in units for p in u["params"]}


def _set_fake_grads(model, gen):
    for _, p in matrix_param_set(model):
        p.grad = torch.randn(p.shape, generator=gen) * 0.01


# ---------------------------------------------------------------------------
# 1. backwards-compat: default == explicit "muon" == plain torch.optim.Muon
# ---------------------------------------------------------------------------
def test_default_matrix_optimizer_is_plain_muon():
    m = _model()
    opts, _ = build_optimizer(m, **_OPT_KW)            # no matrix_optimizer kwarg
    assert isinstance(opts[0], torch.optim.Muon)
    assert not isinstance(opts[0], FusedDeltaNetMuon)


def test_explicit_muon_matches_default_byte_identical():
    """Default flag and explicit --matrix_optimizer muon must produce
    bit-for-bit identical parameter updates (additive, default-off)."""
    mA, mB = _model(0), _model(0)
    optA, _ = build_optimizer(mA, **_OPT_KW)                      # default
    optB, _ = build_optimizer(mB, matrix_optimizer="muon", **_OPT_KW)
    gA = torch.Generator().manual_seed(123)
    gB = torch.Generator().manual_seed(123)
    for _ in range(4):
        _set_fake_grads(mA, gA)
        _set_fake_grads(mB, gB)
        for o in optA:
            o.step()
        for o in optB:
            o.step()
    byB = dict(matrix_param_set(mB))
    max_d = max((pA - byB[n]).abs().max().item()
                for n, pA in matrix_param_set(mA))
    assert max_d == 0.0, f"default vs explicit muon drifted: max|Δ|={max_d}"


# ---------------------------------------------------------------------------
# 2. fused selects FusedDeltaNetMuon with the SAME matrix param set
# ---------------------------------------------------------------------------
def test_fused_uses_fused_optimizer_class():
    m = _model()
    opts, _ = build_optimizer(m, matrix_optimizer="fused_deltanet_ns", **_OPT_KW)
    assert isinstance(opts[0], FusedDeltaNetMuon)


def test_fused_matrix_param_set_identical_to_muon():
    """Fairness: the fused arm's matrix optimizer must touch EXACTLY the
    same params as the muon arm (only the orthogonalization differs)."""
    m1, m2 = _model(0), _model(0)
    o_muon, _ = build_optimizer(m1, matrix_optimizer="muon", **_OPT_KW)
    o_fused, _ = build_optimizer(m2, matrix_optimizer="fused_deltanet_ns",
                                 **_OPT_KW)
    assert _matrix_opt_names(m1, o_muon[0]) == _matrix_opt_names(m2, o_fused[0])


def test_fused_excludes_pkm_and_adapter_2d_tables():
    """Regression for the leak path the wiring guards against: PKM value tables
    and ThinkAdapter fc weights are 2D and would be swept into
    `build_units_from_model`'s OWN other_2d (it only excludes embed/pos/lm_head),
    but the fused branch DISCARDS that and re-derives from `muon_params` (which
    routes them to AdamW). So neither the muon NOR the fused matrix optimizer may
    touch them, and the two matrix sets must still be identical."""
    m1, m2 = _model(0), _model(0)
    # Attach a fake PKM value table (name pkm_layer.values.0.weight) and a fake
    # ThinkAdapter under a block (name blocks.0.think_adapter.fc1.weight) to BOTH.
    for m in (m1, m2):
        pkm = nn.Module()
        pkm.values = nn.ModuleList([nn.Embedding(64, 32)])
        m.add_module("pkm_layer", pkm)
        ta = nn.Module()
        ta.fc1 = nn.Linear(32, 32, bias=False)
        m.blocks[0].add_module("think_adapter", ta)
    o_muon, _ = build_optimizer(m1, matrix_optimizer="muon", **_OPT_KW)
    o_fused, _ = build_optimizer(m2, matrix_optimizer="fused_deltanet_ns",
                                 **_OPT_KW)
    muon_names = _matrix_opt_names(m1, o_muon[0])
    fused_names = _matrix_opt_names(m2, o_fused[0])
    assert muon_names == fused_names, "fused matrix set must equal muon set"
    for bad in ("pkm_layer.values.0.weight", "blocks.0.think_adapter.fc1.weight"):
        assert bad not in fused_names, f"{bad} leaked into fused matrix optimizer"
        assert bad not in muon_names, f"{bad} leaked into muon matrix optimizer"


def test_fused_adamw_groups_identical_to_muon():
    """The non-matrix groups (AdamW regular/alpha/pkm) must be byte-identical
    across arms — only the q/k/v/b orthogonalization may differ."""
    m1, m2 = _model(0), _model(0)
    o_muon, _ = build_optimizer(m1, matrix_optimizer="muon", **_OPT_KW)
    o_fused, _ = build_optimizer(m2, matrix_optimizer="fused_deltanet_ns",
                                 **_OPT_KW)
    assert _adamw_named_groups(m1, o_muon[1]) == _adamw_named_groups(m2,
                                                                     o_fused[1])


# ---------------------------------------------------------------------------
# 3. fused differs from muon ONLY on q/k/v/b (per-head NS), identical on
#    o_proj/MLP (both whole-matrix Muon)
# ---------------------------------------------------------------------------
def test_fused_differs_on_qkvb_matches_on_other():
    mA, mB = _model(0), _model(0)
    optA, _ = build_optimizer(mA, matrix_optimizer="muon", **_OPT_KW)
    optB, _ = build_optimizer(mB, matrix_optimizer="fused_deltanet_ns",
                              **_OPT_KW)
    qkvb = _qkvb_names(mA)
    gA = torch.Generator().manual_seed(7)
    gB = torch.Generator().manual_seed(7)
    _set_fake_grads(mA, gA)
    _set_fake_grads(mB, gB)
    for o in optA:
        o.step()
    for o in optB:
        o.step()
    byB = dict(matrix_param_set(mB))
    n_diff_qkvb = n_same_other = 0
    for n, pA in matrix_param_set(mA):
        d = (pA - byB[n]).abs().max().item()
        if n in qkvb:
            assert d > 0.0, f"q/k/v/b {n} should differ (per-head vs whole NS)"
            n_diff_qkvb += 1
        else:
            assert d == 0.0, f"non-qkvb {n} must match Muon exactly, got {d}"
            n_same_other += 1
    assert n_diff_qkvb > 0 and n_same_other > 0


# ---------------------------------------------------------------------------
# 4. guards
# ---------------------------------------------------------------------------
def test_fused_requires_muon_optimizer():
    # both soap and adamw must reject the fused matrix optimizer (the adamw
    # branch returns early, so this also guards that the validation is hoisted
    # above it rather than silently ignored).
    for bad_opt in ("soap", "adamw"):
        m = _model()
        try:
            build_optimizer(m, **{**_OPT_KW, "optimizer": bad_opt,
                                  "matrix_optimizer": "fused_deltanet_ns"})
        except ValueError as e:
            assert "requires --optimizer muon" in str(e)
        else:
            raise AssertionError(
                f"expected ValueError for fused + {bad_opt} optimizer")


def test_unknown_matrix_optimizer_raises():
    m = _model()
    try:
        build_optimizer(m, **{**_OPT_KW, "matrix_optimizer": "bogus"})
    except ValueError as e:
        assert "unknown matrix_optimizer" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown matrix_optimizer")
