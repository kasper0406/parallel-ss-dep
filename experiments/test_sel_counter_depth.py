"""Tests for the counter-gather depth experiment (2026-06-25).

Covers:
  1. Fairness / no answer leak: counter_gather only ever gathers the program
     OP-TOKEN at program-position r (in [OP_BASE, OP_BASE+L_ops)) from the input
     -- never the answer (a node id < N) nor table content.  And the gathered
     op-identity is IDENTICAL to what the oracle injects (same fairness bar).
  2. Cold-start no-op: with a zero-init adapter the counter_gather latent loop is
     byte-identical to the validated baseline latent loop (think_forward
     mode='latent').
  3. counter_gather (raw additive, no adapter) is byte-identical to the oracle
     (it injects the same op-identity embedding by construction).
  4. An active (non-zero) adapter makes counter_gather differ from baseline.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
      experiments/test_sel_counter_depth.py -v
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from experiments.depth_via_iteration import (
    build, make_multitable_chase_batch, task_meta,
)
from experiments.latent_think import think_forward
from experiments.op_selector_depth import think_forward_oracle
from experiments.sel_counter_depth import (
    think_forward_counter_gather, _prog_start,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _model(N=8, L_ops=2, K_max=8, d=64, L=2):
    thinking_id, vocab, max_T = task_meta("hetero_mt", N, K_max, L_ops)
    torch.manual_seed(0)
    model = build(vocab, thinking_id, d, L, max(1, d // 32), 32, max_T, device=DEV)
    return model, thinking_id, N, L_ops


def test_counter_gather_only_gathers_op_identity_no_leak():
    """The tokens counter_gather reads at the counter positions are exactly the
    program op-tokens OP_BASE+prog -- op IDENTITY only, never the answer."""
    N, L_ops, K, B = 8, 2, 6, 16
    g = torch.Generator().manual_seed(3)
    ids, ans, _chain, prog, _vocab = make_multitable_chase_batch(
        B, N, K, L_ops, DEV, g, homogeneous=False)
    OP_BASE = N
    prog_start = _prog_start("hetero_mt", N, L_ops)
    for r in range(K):
        gathered = ids[:, prog_start + r].cpu()
        # must equal the op token OP_BASE+prog[:,r]
        assert torch.equal(gathered, (OP_BASE + prog[:, r]).cpu())
        # must be an OP token, never a node/answer (< N) nor QUERY/THINK/PAD
        assert torch.all(gathered >= OP_BASE)
        assert torch.all(gathered < OP_BASE + L_ops)
    # the answer is a node id in [0,N): it is structurally NOT in the op range,
    # so it can never be injected by the gather.
    assert torch.all(ans.cpu() < N)


def test_counter_gather_raw_matches_oracle():
    """counter_gather with no adapter injects embed(OP_BASE+prog[:,r]) -- exactly
    the oracle's injection -> byte-identical full-forward logits."""
    if DEV != "cuda":
        return
    model, thinking_id, N, L_ops = _model()
    model.eval()
    g = torch.Generator().manual_seed(7)
    K = 5
    ids, _ans, _chain, prog, _vocab = make_multitable_chase_batch(
        8, N, K, L_ops, DEV, g, homogeneous=False)
    prog_start = _prog_start("hetero_mt", N, L_ops)
    with torch.no_grad():
        oracle = think_forward_oracle(model, ids, K, thinking_id, prog, N)
        cg = think_forward_counter_gather(model, ids, K, thinking_id,
                                          prog_start, K, adapter=None)
    assert torch.allclose(oracle, cg, atol=1e-5), \
        (oracle - cg).abs().max().item()


def test_counter_gather_cold_start_is_noop_vs_baseline_latent():
    """Zero-init adapter => counter_gather latent loop == baseline latent loop."""
    if DEV != "cuda":
        return
    model, thinking_id, N, L_ops = _model()
    model.eval()
    d = model.embed.embedding_dim
    adapter = nn.Linear(d, d, bias=False).to(DEV)
    nn.init.zeros_(adapter.weight)
    assert torch.all(adapter.weight == 0)
    g = torch.Generator().manual_seed(11)
    K = 4
    ids, _a, _c, _p, _v = make_multitable_chase_batch(
        8, N, K, L_ops, DEV, g, homogeneous=False)
    prog_start = _prog_start("hetero_mt", N, L_ops)
    with torch.no_grad():
        base = think_forward(model, ids, K, thinking_id, mode="latent")
        cg = think_forward_counter_gather(model, ids, K, thinking_id,
                                          prog_start, K, adapter=adapter)
    assert torch.allclose(base, cg, atol=1e-5), (base - cg).abs().max().item()


def test_active_adapter_changes_output():
    """A non-zero adapter makes counter_gather differ from baseline latent."""
    if DEV != "cuda":
        return
    model, thinking_id, N, L_ops = _model()
    model.eval()
    d = model.embed.embedding_dim
    adapter = nn.Linear(d, d, bias=False).to(DEV)
    with torch.no_grad():
        nn.init.normal_(adapter.weight, std=0.5)
    g = torch.Generator().manual_seed(13)
    K = 4
    ids, _a, _c, _p, _v = make_multitable_chase_batch(
        8, N, K, L_ops, DEV, g, homogeneous=False)
    prog_start = _prog_start("hetero_mt", N, L_ops)
    with torch.no_grad():
        base = think_forward(model, ids, K, thinking_id, mode="latent")
        cg = think_forward_counter_gather(model, ids, K, thinking_id,
                                          prog_start, K, adapter=adapter)
    assert not torch.allclose(base, cg, atol=1e-4)
