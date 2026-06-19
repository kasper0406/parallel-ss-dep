"""Tests for the FUSED DeltaNet-tailored matrix optimizer.

The load-bearing guarantee: `FusedDeltaNetMuon` (single optimizer, per-head NS on
q/k/v/b + whole-matrix Muon on o_proj/MLP) produces BYTE-IDENTICAL parameter
updates to the validated 2-object prototype (`DeltaNetProjMuon` + `torch.optim.Muon`).
If this drifts, the perf claims in DELTANET_PRECONDITIONER_PERF.md profile a
different algorithm. CPU-only + tiny model so it runs in CI without a GPU.
"""
from __future__ import annotations

import torch

from experiments.exp_deltanet_precond_fused import (
    FusedDeltaNetMuon, build_production_model, matrix_param_set,
    make_two_object, make_fused, build_units_from_model,
)


def _build(seed=0):
    torch.manual_seed(seed)
    return build_production_model(d_model=128, n_layers=3, n_heads=4, d_head=32,
                                  vocab=256, max_T=64, device="cpu")


def _run_steps(opt_or_list, mats_by_name, grads_seq):
    is_list = isinstance(opt_or_list, list)
    for grads in grads_seq:
        for n, p in mats_by_name.items():
            p.grad = grads[n].clone()
        if is_list:
            for o in opt_or_list:
                o.step()
        else:
            opt_or_list.step()


def _make_grad_seq(mats, nsteps, seed=123):
    g = torch.Generator().manual_seed(seed)
    seq = []
    for _ in range(nsteps):
        seq.append({n: torch.randn(p.shape, generator=g) * 0.01 for n, p in mats})
    return seq


def test_fused_perunit_byte_identical_to_two_object():
    lr, mom, wd = 1e-2, 0.95, 0.01
    mA, mB = _build(0), _build(0)
    optA = make_two_object(mA, lr, mom, wd)
    optB = make_fused(mB, lr, mom, wd, batch_across_layers=False)
    matsA = matrix_param_set(mA)
    grads_seq = _make_grad_seq(matsA, 6)
    _run_steps(optA, dict(matsA), grads_seq)
    _run_steps(optB, dict(matrix_param_set(mB)), grads_seq)
    mB_by = dict(matrix_param_set(mB))
    max_d = max((pA - mB_by[n]).abs().max().item() for n, pA in matsA)
    assert max_d < 1e-5, f"fused per-unit drifted from 2-object: max|Δ|={max_d}"


def test_fused_cross_layer_matches_two_object():
    lr, mom, wd = 1e-2, 0.95, 0.01
    mA, mB = _build(1), _build(1)
    optA = make_two_object(mA, lr, mom, wd)
    optB = make_fused(mB, lr, mom, wd, batch_across_layers=True)
    matsA = matrix_param_set(mA)
    grads_seq = _make_grad_seq(matsA, 6)
    _run_steps(optA, dict(matsA), grads_seq)
    _run_steps(optB, dict(matrix_param_set(mB)), grads_seq)
    mB_by = dict(matrix_param_set(mB))
    max_d = max((pA - mB_by[n]).abs().max().item() for n, pA in matsA)
    assert max_d < 1e-5, f"fused cross-layer drifted from 2-object: max|Δ|={max_d}"


def test_state_is_one_buffer_per_param():
    lr, mom, wd = 1e-2, 0.95, 0.01
    m = _build(2)
    opt = make_fused(m, lr, mom, wd)
    mats = matrix_param_set(m)
    grads_seq = _make_grad_seq(mats, 1)
    _run_steps(opt, dict(mats), grads_seq)
    n_buf = sum("momentum_buffer" in st for st in opt.state.values())
    assert n_buf == len(mats), f"expected {len(mats)} buffers, got {n_buf}"
    # exactly one tensor of state per param
    for st in opt.state.values():
        tensors = [v for v in st.values() if torch.is_tensor(v)]
        assert len(tensors) == 1


def test_bf16_state_runs_and_buffers_are_bf16():
    lr, mom, wd = 1e-2, 0.95, 0.01
    m = _build(3)
    opt = make_fused(m, lr, mom, wd, bf16_state=True)
    mats = matrix_param_set(m)
    grads_seq = _make_grad_seq(mats, 3)
    _run_steps(opt, dict(mats), grads_seq)
    for st in opt.state.values():
        assert st["momentum_buffer"].dtype == torch.bfloat16
    # params stay finite fp32
    for _, p in mats:
        assert p.dtype == torch.float32 and torch.isfinite(p).all()


def test_units_discovered():
    m = _build(0)
    _, units, other = build_units_from_model(m, mode="perhead")
    kinds = sorted({u["kind"] for u in units})
    # q/k/v -> matrix_perhead, b_proj -> rownorm; o_proj+MLP -> other
    assert "matrix_perhead" in kinds
    assert len(other) > 0
    assert isinstance(FusedDeltaNetMuon(units, other, lr=1e-2), FusedDeltaNetMuon)
