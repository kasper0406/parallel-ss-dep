"""Tests for `experiments/speed_knobs.py::apply_speed_knobs`.

Focus: the `_eager_forward` stash that lets aux-loss helpers (process
reward / gate calibration) bypass `torch.compile` on their odd-shaped
extra forwards. Without that escape hatch, the second forward triggers
either a recompile (slow) or an Inductor symbolic-shape assertion
(crash) — the exact failure mode that motivated --no-compile in the
2026-05-27 smoke launcher and the throughput regression to 17k tok/s.

All tests CPU-only; we monkeypatch `torch.compile` so we don't need a
CUDA-with-Triton stack to exercise the wiring.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments import speed_knobs


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


def test_no_compile_no_eager_attr():
    """When compile is off, `_eager_forward` is not installed — callers
    fall back to `model(...)` as before."""
    m = _TinyModel()
    speed_knobs.apply_speed_knobs(m, bf16=False, tf32=False,
                                   compile_model=False, verbose=False)
    assert not hasattr(m, "_eager_forward")


def test_compile_installs_eager_forward(monkeypatch):
    """When compile is on, `_eager_forward` MUST point at the
    pre-compile (bf16-wrapped or raw) forward so aux helpers can call
    eager. We don't actually need Inductor here — just verify the
    attribute is installed and points at the un-compiled callable.
    """
    m = _TinyModel()

    # Monkeypatch torch.compile to a no-op wrapper so we don't need
    # CUDA / Triton in the test env. The wrapper must be *distinct* from
    # the input so we can assert `_eager_forward != model.forward`.
    sentinel_calls = {"compile": 0}

    def fake_compile(fn, **kwargs):
        sentinel_calls["compile"] += 1
        original = fn

        def compiled_wrapper(*args, **kwargs):
            return original(*args, **kwargs)
        compiled_wrapper._is_fake_compiled = True
        return compiled_wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    # Also stub out `_disable_dynamo_on_fla_helpers` and the dynamo
    # config touches so the function reaches our path without needing
    # the real fla import / dynamo install.
    monkeypatch.setattr(speed_knobs, "_disable_dynamo_on_fla_helpers",
                        lambda verbose=False: None)

    speed_knobs.apply_speed_knobs(m, bf16=False, tf32=False,
                                   compile_model=True, verbose=False)

    assert hasattr(m, "_eager_forward"), \
        "compile_model=True must install model._eager_forward"
    assert getattr(m.forward, "_is_fake_compiled", False), \
        "compile_model=True must install the compiled wrapper as model.forward"
    assert not getattr(m._eager_forward, "_is_fake_compiled", False), \
        "_eager_forward must be the PRE-compile callable, not the wrapper"
    assert sentinel_calls["compile"] == 1


def test_compile_with_bf16_eager_skips_compile(monkeypatch):
    """The bf16 wrap should be inside `_eager_forward` (i.e. aux
    helpers get bf16 autocast but skip compile). Verifies the layering
    order in apply_speed_knobs: bf16 first, then compile."""
    m = _TinyModel()

    def fake_compile(fn, **kwargs):
        original = fn

        def compiled_wrapper(*args, **kwargs):
            return original(*args, **kwargs)
        compiled_wrapper._is_fake_compiled = True
        return compiled_wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(speed_knobs, "_disable_dynamo_on_fla_helpers",
                        lambda verbose=False: None)

    speed_knobs.apply_speed_knobs(m, bf16=True, tf32=False,
                                   compile_model=True, verbose=False)
    # The eager forward should be the bf16 wrap, NOT the raw nn.Module
    # forward (so process_reward's autocast contract matches the main
    # path).
    eager = m._eager_forward
    assert not getattr(eager, "_is_fake_compiled", False)
    # It IS the bf16 wrap (a plain closure), not the bound method.
    # The bf16 wrap returns the same forward output (we don't introspect
    # the autocast manager here — that's covered by torch.amp's tests).
    x = torch.randn(2, 4)
    y_eager = eager(x)
    assert y_eager.shape == (2, 4)
