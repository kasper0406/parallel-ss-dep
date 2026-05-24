"""Tests for BF16StateAdamW / BF16StateMuon — bf16 optimizer-state storage.

Asserts that loss curves track the stock fp32-state optimizers within a
tight tolerance over 100 steps, that state actually lives in bf16, and
that the memory footprint is the expected ~half.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.bf16_optim import BF16StateAdamW, BF16StateMuon


CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                           reason="CUDA required")


def _build_tiny_mlp(d_in=64, d_hidden=128, d_out=32, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


def _train_and_get_losses(opt_factory, n_steps=100, batch=8, d_in=64, d_out=32,
                           seed=42, device="cpu"):
    """Train an MLP on a fixed regression task; return loss trajectory."""
    model = _build_tiny_mlp(d_in=d_in, d_out=d_out, seed=seed).to(device)
    opt = opt_factory(model.parameters())
    rng = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for _ in range(n_steps):
        x = torch.randn(batch, d_in, generator=rng, device=device)
        y = torch.randn(batch, d_out, generator=rng, device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model, opt


def test_bf16_adamw_state_dtype_is_bf16():
    """After the first step, exp_avg / exp_avg_sq must be stored as bf16."""
    losses, _, opt = _train_and_get_losses(
        lambda ps: BF16StateAdamW(ps, lr=1e-3),
        n_steps=2,
    )
    n_state_tensors = 0
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            for k in ("exp_avg", "exp_avg_sq"):
                assert k in st, f"missing {k} in state"
                assert st[k].dtype == torch.bfloat16, (
                    f"{k} dtype is {st[k].dtype}, expected bfloat16"
                )
                n_state_tensors += 1
    assert n_state_tensors > 0


def test_bf16_adamw_matches_stock_adamw_loss():
    """bf16-state AdamW must track stock AdamW closely over 100 steps."""
    bf16_losses, _, _ = _train_and_get_losses(
        lambda ps: BF16StateAdamW(ps, lr=1e-3),
        n_steps=100,
    )
    fp32_losses, _, _ = _train_and_get_losses(
        lambda ps: torch.optim.AdamW(ps, lr=1e-3, betas=(0.9, 0.999),
                                       eps=1e-8, weight_decay=1e-2,
                                       foreach=False),
        n_steps=100,
    )
    bf = torch.tensor(bf16_losses)
    fp = torch.tensor(fp32_losses)
    # Final-loss diff: bf16 storage should not measurably hurt convergence
    # on this small task. Generous bound (4 % rel) since we compare against
    # PyTorch's reference implementation, which has a slightly different
    # math ordering.
    assert abs(bf[-1] - fp[-1]) / fp[-1].abs() < 0.04, (
        f"bf16 AdamW final loss {bf[-1]:.4f} vs fp32 {fp[-1]:.4f}"
    )
    # Trajectory tracking — last 20 steps mean abs diff is small.
    assert (bf[-20:] - fp[-20:]).abs().mean() < 0.02


def test_bf16_adamw_state_memory_is_half():
    """exp_avg + exp_avg_sq footprint should be ~half vs stock fp32 AdamW."""
    _, _, opt_bf = _train_and_get_losses(
        lambda ps: BF16StateAdamW(ps, lr=1e-3), n_steps=2,
    )
    _, _, opt_fp = _train_and_get_losses(
        lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=False), n_steps=2,
    )
    def state_bytes(opt):
        n = 0
        for g in opt.param_groups:
            for p in g["params"]:
                for v in opt.state[p].values():
                    if isinstance(v, torch.Tensor):
                        n += v.element_size() * v.numel()
        return n
    bf = state_bytes(opt_bf)
    fp = state_bytes(opt_fp)
    # bf16 should be ~half of fp32 (the `step` int counter is not a tensor;
    # only exp_avg / exp_avg_sq contribute, both halved).
    assert 0.45 < bf / fp < 0.55, f"bf16 state {bf} vs fp32 {fp}"


@CUDA
def test_bf16_muon_state_dtype_is_bf16():
    """After the first step, momentum_buffer must be stored as bf16."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 16, device="cuda"))
    opt = BF16StateMuon([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    buf = opt.state[p]["momentum_buffer"]
    assert buf.dtype == torch.bfloat16
    assert buf.shape == p.shape


@CUDA
def test_bf16_muon_matches_stock_muon():
    """bf16-momentum Muon must track torch.optim.Muon closely.

    Compares loss trajectories on the same quadratic-with-noise objective
    starting from identical params. Muon's Newton-Schulz makes per-step
    descent small on tiny problems, so we don't assert big loss drops —
    we assert that bf16-state Muon ≈ fp32-state Muon.
    """
    def run(opt_cls):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        target = torch.randn(16, 16, device="cuda")
        p = nn.Parameter(torch.randn(16, 16, device="cuda") * 0.1)
        opt = opt_cls([p], lr=1e-2, momentum=0.9)
        losses = []
        for _ in range(50):
            opt.zero_grad(set_to_none=True)
            loss = ((p - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    bf = torch.tensor(run(BF16StateMuon))
    fp = torch.tensor(run(torch.optim.Muon))
    # Trajectories should be within 1 % rel of each other in the tail.
    diff = (bf[-20:] - fp[-20:]).abs().mean().item()
    assert diff < 0.005, (
        f"bf16 vs fp32 Muon last-20 mean |Δ|={diff:.5f}; "
        f"bf={bf[-1]:.4f} fp={fp[-1]:.4f}"
    )


@CUDA
def test_compiled_adamw_matches_eager_adamw():
    """Regression: compile_step=True must produce bit-identical updates
    (up to fp32 rounding from operator reordering inside the fused
    kernel) to compile_step=False over many steps. Guards against silent
    numerical drift introduced when the per-param-step kernel is fused
    via torch.compile."""
    def run(compile_step: bool):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        # Mix of shapes — exercises the per-shape compile cache.
        params = [
            nn.Parameter(torch.randn(8, 16, device="cuda")),
            nn.Parameter(torch.randn(32, device="cuda")),       # 1-D
            nn.Parameter(torch.randn(16, 16, device="cuda")),
        ]
        opt = BF16StateAdamW(params, lr=1e-3, weight_decay=0.01,
                              compile_step=compile_step)
        # Same gradients per step across both runs.
        torch.manual_seed(123)
        for _ in range(20):
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()
        return [p.detach().clone() for p in params]

    eager_params = run(compile_step=False)
    comp_params = run(compile_step=True)
    for i, (pe, pc) in enumerate(zip(eager_params, comp_params)):
        diff = (pe - pc).abs().max().item()
        assert diff < 1e-5, (
            f"param[{i}] shape={tuple(pe.shape)}: compiled vs eager "
            f"max |Δ|={diff:.3e} (>1e-5; numerical drift from compile fusion)"
        )


@CUDA
def test_compile_step_flag_defaults_on():
    """The compile-step optimization is on by default; passing
    compile_step=False is the kill-switch. Guards against accidentally
    flipping the default (which would silently regress wall-clock)."""
    import inspect
    sig = inspect.signature(BF16StateAdamW.__init__)
    assert sig.parameters["compile_step"].default is True, (
        "compile_step default must be True; pass compile_step=False to disable"
    )
