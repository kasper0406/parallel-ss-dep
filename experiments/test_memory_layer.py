"""Tests for PKMLayer (experiments/memory_layer.py).

Covers shape, top-k correctness vs naive lookup, gradient sparsity,
cold-start spread under BatchNorm, and determinism in eval mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.memory_layer import PKMLayer


def test_pkm_forward_shape():
    layer = PKMLayer(d_model=64, n_heads=2, n_keys=8, k_dim=16, top_k=4,
                      value_bf16=False)
    x = torch.randn(3, 5, 64)
    y = layer(x)
    assert y.shape == (3, 5, 64), f"got {y.shape}"


def test_pkm_topk_matches_naive():
    """The PKM-decoded top-k must match a naive (n_keys^2)-row argmax —
    same scores, same value contribution."""
    torch.manual_seed(0)
    H, K, kd, tk, v = 1, 8, 16, 4, 16
    layer = PKMLayer(d_model=v, n_heads=H, n_keys=K, k_dim=kd, top_k=tk,
                      value_bf16=False)
    layer.eval()  # disable BN running-mean updates so we get a clean math eq
    x = torch.randn(2, 3, v)
    h_n = layer.norm(x)
    q = layer.query_proj(h_n).float().view(2, 3, H, 2, kd)
    q1, q2 = q[..., 0, :], q[..., 1, :]
    sk1 = layer.subkeys[:, 0].float()
    sk2 = layer.subkeys[:, 1].float()
    s1 = torch.einsum("bthk,hnk->bthn", q1, sk1)
    s2 = torch.einsum("bthk,hnk->bthn", q2, sk2)
    s1 = layer.bn_s1(s1.reshape(-1, K)).reshape(2, 3, H, K)
    s2 = layer.bn_s2(s2.reshape(-1, K)).reshape(2, 3, H, K)
    # Naive: for every (i, j) ∈ [K]² compute s1[i] + s2[j]; pick top-tk.
    naive_full = s1.unsqueeze(-1) + s2.unsqueeze(-2)   # (B, T, H, K, K)
    naive_flat = naive_full.view(2, 3, H, K * K)
    naive_top, naive_idx = naive_flat.topk(tk, dim=-1)

    # PKM-side decode (same as forward, but we just want the chosen scores).
    s1_top, i1 = s1.topk(tk, dim=-1)
    s2_top, i2 = s2.topk(tk, dim=-1)
    scores = s1_top.unsqueeze(-1) + s2_top.unsqueeze(-2)
    scores_flat = scores.view(2, 3, H, tk * tk)
    final_scores, final_idx_in_grid = scores_flat.topk(tk, dim=-1)
    i_in_top1 = final_idx_in_grid // tk
    j_in_top2 = final_idx_in_grid %  tk
    sel1 = torch.gather(i1, dim=-1, index=i_in_top1)
    sel2 = torch.gather(i2, dim=-1, index=j_in_top2)
    pkm_idx = sel1 * K + sel2

    # Sort both for comparison (top-k order from outer-product can differ
    # from naive scan when ties are broken differently, but the *scores*
    # of the selected entries must match.)
    naive_sorted, _ = naive_top.sort(dim=-1)
    pkm_sorted, _ = final_scores.sort(dim=-1)
    assert torch.allclose(naive_sorted, pkm_sorted, atol=1e-5), (
        f"top-k score mismatch:\n naive={naive_sorted}\n pkm={pkm_sorted}"
    )
    # Also check the indices selected are a subset of the naive top-k:
    # for the strict outer-product PKM, the selected (i,j) must always
    # have score equal to one of the naive top-k scores.
    for b in range(2):
        for t in range(3):
            for h_idx in range(H):
                pkm_set = set(pkm_idx[b, t, h_idx].tolist())
                naive_set = set(naive_idx[b, t, h_idx].tolist())
                # Every PKM-chosen index must be in the naive top-k set
                # (or have the same score as one in it).
                for i in pkm_set:
                    naive_score_set = set(naive_top[b, t, h_idx].tolist())
                    s_pkm = naive_full.view(2, 3, H, K * K)[b, t, h_idx, i].item()
                    assert any(abs(s_pkm - n) < 1e-5 for n in naive_score_set), (
                        f"PKM chose idx {i} score {s_pkm}; not in naive top-k {naive_score_set}"
                    )


def test_pkm_gradient_to_topk_only():
    """Backprop through PKM should give gradient only to retrieved value
    rows (sparse update), not to every row in the table."""
    torch.manual_seed(1)
    H, K, kd, tk, v = 1, 16, 16, 4, 8
    layer = PKMLayer(d_model=v, n_heads=H, n_keys=K, k_dim=kd, top_k=tk,
                      value_bf16=False)
    x = torch.randn(2, 4, v, requires_grad=False)
    out = layer(x)
    out.sum().backward()
    grad = layer.values[0].weight.grad   # (K*K, v_dim)
    nonzero_rows = (grad.abs().sum(dim=-1) > 0).sum().item()
    total_rows = K * K
    # 2 batch * 4 positions * top_k = at most 32 rows touched
    max_touched = 2 * 4 * tk
    assert nonzero_rows <= max_touched, (
        f"got {nonzero_rows} nonzero rows in grad, expected ≤ {max_touched} "
        f"(out of {total_rows})"
    )
    # And at least one row got gradient.
    assert nonzero_rows > 0, "no value rows got gradient"


def test_pkm_cold_start_bn_spreads_lookup():
    """With BatchNorm on scores, repeated lookups should rotate through
    most of the table within a few steps (the warmup property). Without
    BN the same test collapses to a few hot keys.

    We assert only the BN-on direction (the failure-mode test for BN-off
    would need a BN-removed variant which we don't expose).
    """
    torch.manual_seed(2)
    H, K, kd, tk, v = 1, 16, 8, 4, 8
    layer = PKMLayer(d_model=v, n_heads=H, n_keys=K, k_dim=kd, top_k=tk,
                      value_bf16=False)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    seen_slots = set()
    for step in range(20):
        x = torch.randn(8, 8, v)
        out = layer(x)
        # Inspect retrieved slot indices via a dry forward (re-run a partial
        # forward to extract slot_idx). Simpler: rely on the gradient
        # signature on values.weight.
        loss = out.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        grad_rows = (layer.values[0].weight.grad.abs().sum(dim=-1) > 0)
        seen_slots.update(grad_rows.nonzero(as_tuple=False).flatten().tolist())
        opt.step()
    coverage = len(seen_slots) / (K * K)
    assert coverage >= 0.5, (
        f"BN-warmup coverage was {coverage:.2%} after 20 steps; "
        f"expected ≥ 50%."
    )


def test_pkm_eval_determinism():
    layer = PKMLayer(d_model=32, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)
    layer.eval()
    x = torch.randn(2, 3, 32)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_pkm_with_bf16_storage():
    """Smoke-test that bf16 value storage works end-to-end (forward+
    backward) and produces non-NaN output."""
    layer = PKMLayer(d_model=32, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=True)
    assert layer.values[0].weight.dtype == torch.bfloat16
    x = torch.randn(2, 3, 32)
    y = layer(x)
    assert y.shape == (2, 3, 32)
    assert torch.isfinite(y).all()
    y.sum().backward()
    # Gradient should populate (in fp32, since autograd lifts).
    assert layer.values[0].weight.grad is not None
    assert torch.isfinite(layer.values[0].weight.grad).all()


def test_pkm_param_count_matches_design():
    """Sanity-check the param accounting in PKM_PLAN.md."""
    layer = PKMLayer(d_model=576, n_heads=4, n_keys=256, k_dim=128, top_k=32)
    total = sum(p.numel() for p in layer.parameters())
    # ~38.9M total per the plan.
    assert 38_500_000 < total < 39_500_000, (
        f"param count {total/1e6:.2f}M outside expected ~38.9M range"
    )
    # Value table dominates.
    assert layer.n_value_params == 4 * 256 * 256 * 144
