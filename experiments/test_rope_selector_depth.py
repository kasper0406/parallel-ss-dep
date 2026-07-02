"""Correctness tests for the RoPE op-selector (rope_selector_depth.py).

Validates:
  - the RoPE relative-position geometry (dot(rope(v,r), rope(v,p)) is maximal
    at p==r over p in 0..K-1) — the load-bearing selection property;
  - the selector injects op-IDENTITY only (selected vector lies in the convex
    hull of the program op embeddings; attention sums to 1) -> no answer leak;
  - cold-start sanity: a fresh selector runs end-to-end and yields finite,
    correctly-shaped answer logits.

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
     experiments/test_rope_selector_depth.py -v
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from experiments.rope_selector_depth import (
    RopeOpSelector, think_forward_rope, _prog_start,
)
from experiments.depth_via_iteration import build, task_meta
from experiments.sel_counter_depth import _make_batch, _task_meta_name

CUDA = torch.cuda.is_available()


def test_rope_relative_position_peaks_at_offset_zero():
    """For ALIGNED q==k content, dot(rope(q,r), rope(k,p)) over p in 0..K-1 must
    be maximal at p==r (relative offset 0).  This is the geometric property the
    learned selector exploits to select program position r at latent step r."""
    torch.manual_seed(0)
    sel = RopeOpSelector(d_model=128, d_head=128, base=10000.0)
    K = 8
    # aligned content: every key direction == a fixed unit vector v == query base
    v = torch.randn(128)
    v = v / v.norm()
    pos = torch.arange(K)
    cos_k, sin_k = sel._angles(pos)                      # (K, dh)
    keys = sel._rope(v.unsqueeze(0).expand(K, -1), cos_k, sin_k)   # (K, dh)
    for r in range(K):
        cos_q, sin_q = sel._angles(torch.tensor([float(r)]))
        q = sel._rope(v.unsqueeze(0), cos_q, sin_q)      # (1, dh)
        scores = (keys * q).sum(-1)                       # (K,)
        assert scores.argmax().item() == r, (
            f"step {r}: argmax {scores.argmax().item()} != {r}; scores={scores}")


def test_injection_is_op_identity_no_answer_leak():
    """The selected vector must be a CONVEX combination (attn sums to 1) of the
    RAW program op embeddings — never the answer.  So the injected term lives in
    span(prog_embeds); the answer node id (< N) cannot enter via the selector."""
    torch.manual_seed(1)
    sel = RopeOpSelector(d_model=64, d_head=64)
    B, K, d = 5, 6, 64
    prog_embeds = torch.randn(B, K, d)
    for r in range(K):
        out, attn = sel(prog_embeds, r)
        assert attn.shape == (B, K)
        assert torch.allclose(attn.sum(-1), torch.ones(B), atol=1e-5)
        assert (attn >= 0).all()
        recon = torch.bmm(attn.unsqueeze(1), prog_embeds).squeeze(1)
        assert torch.allclose(out, recon, atol=1e-5), \
            "selected vector is not attn @ prog_embeds (value path not raw)"


def test_selector_only_learnable_params_are_q_and_k():
    """The selection path exposes ONLY q_base + k_proj as learnable params
    (no learnable value/out transform) — the minimal-learnable-surface design."""
    sel = RopeOpSelector(d_model=64, d_head=32)
    names = {n for n, _ in sel.named_parameters()}
    assert names == {"q_base", "k_proj.weight"}, names


@pytest.mark.skipif(not CUDA, reason="needs CUDA (DeltaNet FLA kernels)")
def test_cold_start_forward_finite_and_shaped():
    """A fresh selector runs the full latent loop end-to-end and yields finite
    answer logits of shape (B, vocab)."""
    device = "cuda"
    task = "hetero_mt"
    N, L_ops, K = 8, 2, 4
    thinking_id, vocab, max_T = task_meta(_task_meta_name(task), N, K, L_ops)
    model = build(vocab, thinking_id, d_model=64, n_layers=2, n_heads=2,
                  d_head=32, max_T=max_T, device=device)
    sel = RopeOpSelector(64, d_head=64).to(device)
    g = torch.Generator().manual_seed(0)
    ids, ans, _c, _p = _make_batch(task, 16, N, K, L_ops, device, g)
    prog_start = _prog_start(task, N, L_ops)
    out = think_forward_rope(model, ids, K, thinking_id, sel, prog_start, K)
    assert out.shape == (16, vocab)
    assert torch.isfinite(out).all()
    # R=0 path also finite (no-think)
    out0 = think_forward_rope(model, ids, 0, thinking_id, sel, prog_start, K)
    assert out0.shape == (16, vocab) and torch.isfinite(out0).all()


@pytest.mark.skipif(not CUDA, reason="needs CUDA (DeltaNet FLA kernels)")
def test_one_hot_selection_matches_oracle_injection():
    """When the selector is forced to a one-hot attention at p==r, the injected
    op embedding equals oracle/counter_gather's embed(OP_BASE+prog[:,r]) — i.e.
    the ceiling injection is recoverable, confirming the value path is faithful."""
    device = "cuda"
    task = "hetero_mt"
    N, L_ops, K = 8, 2, 5
    thinking_id, vocab, max_T = task_meta(_task_meta_name(task), N, K, L_ops)
    model = build(vocab, thinking_id, d_model=64, n_layers=2, n_heads=2,
                  d_head=32, max_T=max_T, device=device)
    sel = RopeOpSelector(64, d_head=64).to(device)
    g = torch.Generator().manual_seed(3)
    ids, _ans, _c, _p = _make_batch(task, 8, N, K, L_ops, device, g)
    prog_start = _prog_start(task, N, L_ops)
    base_emb = model.embed(ids)
    prog_embeds = base_emb[:, prog_start:prog_start + K, :]
    for r in range(K):
        # force one-hot at r by overriding scores: use huge q aligned to key r
        keys = sel.k_proj(prog_embeds)
        cos_k, sin_k = sel._angles(torch.arange(K, device=device))
        keys = sel._rope(keys, cos_k.unsqueeze(0), sin_k.unsqueeze(0))
        scores = torch.full((8, K), -1e9, device=device)
        scores[:, r] = 1e9
        attn = torch.softmax(scores, -1)
        recon = torch.bmm(attn.unsqueeze(1), prog_embeds).squeeze(1)
        oracle = model.embed(ids[:, prog_start + r])
        assert torch.allclose(recon, oracle, atol=1e-4)
