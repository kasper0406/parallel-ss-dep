"""Regression tests for the WorkingMemory read-injection α gate (2026-06-04).

Background: WM was the only side-module in the repo whose read injection
(`h + W_proj(read)` at read positions) had no learned scalar gate. Without an
α escape hatch the module could not fall back to the baseline trunk, and at the
saturating MQAR regime (K=256/T=1024/lr=3e-3) the un-gated injection drove
training to collapse (recall 0.014 vs plain DeltaNet's 0.999). The fix adds a
learned scalar `read_alpha` (init configurable):
  - default 1.0 preserves the legacy un-gated behaviour AND keeps old ckpts
    (which lack the param) byte-identical when loaded with strict=False;
  - 0.0 gives the zero-init-residual bootstrap (FiLM-α pattern) — inert at cold
    start, α moves first under loss gradient, weights follow.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_wm_read_alpha.py -v
"""
from __future__ import annotations

import torch

from experiments.model import WorkingMemory


def _make_inputs(B=2, T=16, d_model=16, think_id=7, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(B, T, d_model)
    input_ids = torch.full((B, T), 3, dtype=torch.long)
    # read at the last few positions (they have causal predecessors to read).
    read_mask = torch.zeros(B, T, dtype=torch.bool)
    read_mask[:, -4:] = True
    return h, input_ids, read_mask


def test_default_alpha_is_one():
    """Default construction keeps the legacy un-gated magnitude (α=1)."""
    mem = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7)
    assert float(mem.read_alpha.item()) == 1.0


def test_alpha_zero_injection_is_inert():
    """With α=0 the read injection contributes nothing: out == input exactly,
    regardless of the read mask. This is the baseline-recovery escape hatch."""
    mem = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7,
                        read_alpha_init=0.0).eval()
    h, input_ids, read_mask = _make_inputs()
    with torch.no_grad():
        out = mem(h, input_ids, read_mask=read_mask)
    torch.testing.assert_close(out, h)


def test_alpha_one_injects():
    """With α=1 the injection is active: read positions are perturbed."""
    mem = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7,
                        read_alpha_init=1.0).eval()
    h, input_ids, read_mask = _make_inputs()
    with torch.no_grad():
        out = mem(h, input_ids, read_mask=read_mask)
    assert not torch.allclose(out, h), "α=1 should change read positions"
    # ...and only at read positions (non-read rows untouched).
    nonread = ~read_mask
    torch.testing.assert_close(out[nonread], h[nonread])


def test_alpha_grad_nonzero_at_zero():
    """Even at α=0 the gradient on read_alpha is non-zero, so the gate can
    bootstrap off zero (the FiLM-α pattern). If it were zero, α-init-0 would
    be a dead parameter and the module could never turn on."""
    mem = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7,
                        read_alpha_init=0.0)
    h, input_ids, read_mask = _make_inputs()
    out = mem(h, input_ids, read_mask=read_mask)
    out.pow(2).sum().backward()
    assert mem.read_alpha.grad is not None
    assert mem.read_alpha.grad.abs().item() > 0.0


def test_old_ckpt_loads_without_read_alpha():
    """A state_dict produced before the gate existed (no `read_alpha` key)
    loads with strict=False and the param keeps its construction default —
    so legacy checkpoints behave byte-identically (α=1)."""
    mem_new = WorkingMemory(d_model=16, d_mem=16, mem_size=16,
                            thinking_token_id=7)
    legacy_sd = {k: v for k, v in mem_new.state_dict().items()
                 if k != "read_alpha"}
    assert "read_alpha" not in legacy_sd
    fresh = WorkingMemory(d_model=16, d_mem=16, mem_size=16,
                          thinking_token_id=7)
    missing, unexpected = fresh.load_state_dict(legacy_sd, strict=False)
    assert "read_alpha" in missing
    assert not unexpected
    assert float(fresh.read_alpha.item()) == 1.0
