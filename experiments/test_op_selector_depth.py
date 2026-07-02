"""Tests for the latent op-selector depth experiment (2026-06-25).

Covers:
  1. Cold-start no-op: at init (zero out_proj) the op-selector latent loop is
     byte-identical to the validated baseline latent loop (latent_think
     think_forward, mode='latent').
  2. The op-selector attention CAN select program position r (mechanism is
     capable when its per-step query is aligned to the per-position key).
  3. prog_start indexing matches the actual program tokens in the batch.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
      experiments/test_op_selector_depth.py -v
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.depth_via_iteration import (
    build, make_multitable_chase_batch, task_meta, hetero_layout,
)
from experiments.latent_think import think_forward
from experiments.op_selector_depth import OpSelectorAttn, think_forward_opsel

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _model(N=8, L_ops=2, K_max=8, d=64, L=2):
    thinking_id, vocab, max_T = task_meta("hetero_mt", N, K_max, L_ops)
    torch.manual_seed(0)
    model = build(vocab, thinking_id, d, L, max(1, d // 32), 32, max_T, device=DEV)
    return model, thinking_id, N, L_ops


def test_prog_start_indexes_program_tokens():
    """base_emb[:, prog_start:prog_start+K] must be exactly the op tokens
    OP_BASE+prog, i.e. the program."""
    N, L_ops, K = 8, 2, 5
    g = torch.Generator().manual_seed(3)
    ids, _ans, _chain, prog, _vocab = make_multitable_chase_batch(
        4, N, K, L_ops, DEV, g, homogeneous=False)
    OP_BASE = N
    prog_start = L_ops * (2 * N + 1) + 1
    got = ids[:, prog_start:prog_start + K].cpu()
    expect = (OP_BASE + prog).cpu()
    assert torch.equal(got, expect), (got, expect)
    # s is the last token
    assert ids.shape[1] == prog_start + K + 1


def test_cold_start_is_noop_vs_baseline_latent():
    """Zero-init out_proj => op-selector latent loop == baseline latent loop."""
    if DEV != "cuda":
        return
    model, thinking_id, N, L_ops = _model()
    model.eval()
    opsel = OpSelectorAttn(model.embed.embedding_dim, max_steps=10).to(DEV).eval()
    # confirm cold-start invariants
    assert torch.all(opsel.out_proj.weight == 0)
    assert torch.all(opsel.alpha == 1)
    g = torch.Generator().manual_seed(7)
    K = 4
    ids, _ans, _chain, _prog, _vocab = make_multitable_chase_batch(
        8, N, K, L_ops, DEV, g, homogeneous=False)
    prog_start = L_ops * (2 * N + 1) + 1
    with torch.no_grad():
        base = think_forward(model, ids, K, thinking_id, mode="latent")
        opsel_out = think_forward_opsel(
            model, ids, K, thinking_id, opsel, prog_start, K)
    assert torch.allclose(base, opsel_out, atol=1e-5), \
        (base - opsel_out).abs().max().item()


def test_op_selector_can_select_position_r():
    """With per-step query = per-position key, the attention concentrates on
    program position r (mechanism is capable of per-step selection)."""
    if DEV != "cuda":
        return
    d = 64
    opsel = OpSelectorAttn(d, max_steps=8).to(DEV)
    K = 6
    torch.manual_seed(1)
    prog_embeds = torch.randn(3, K, d, device=DEV)
    # Force k_proj to ignore content (zero) so keys = pos_key_emb, and make the
    # per-step query table equal the per-position key table -> q_r . key_p peaks
    # at p == r.
    with torch.no_grad():
        opsel.k_proj.weight.zero_()
        opsel.pos_key_emb.weight.normal_()
        opsel.step_q_emb.weight.copy_(opsel.pos_key_emb.weight)
    for r in range(K):
        _out, attn = opsel(prog_embeds, r)            # (B,K)
        sel = attn.argmax(dim=-1)
        assert torch.all(sel == r), (r, sel.tolist())


def test_active_selector_changes_output():
    """A non-zero out_proj makes the op-selector loop differ from baseline."""
    if DEV != "cuda":
        return
    model, thinking_id, N, L_ops = _model()
    model.eval()
    opsel = OpSelectorAttn(model.embed.embedding_dim, max_steps=10).to(DEV).eval()
    with torch.no_grad():
        torch.nn.init.normal_(opsel.out_proj.weight, std=0.5)
    g = torch.Generator().manual_seed(11)
    K = 4
    ids, _a, _c, _p, _v = make_multitable_chase_batch(
        8, N, K, L_ops, DEV, g, homogeneous=False)
    prog_start = L_ops * (2 * N + 1) + 1
    with torch.no_grad():
        base = think_forward(model, ids, K, thinking_id, mode="latent")
        act = think_forward_opsel(model, ids, K, thinking_id, opsel,
                                  prog_start, K)
    assert not torch.allclose(base, act, atol=1e-4)
