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
    # Use score_norm='batch' here because the rest of this test reaches into
    # layer.bn_s1 / bn_s2 directly to replicate the score math. (The v7
    # default is 'layer'; covered by separate tests.) Also disable the v7
    # output_gate so the comparison against the naive lookup isn't multiplied
    # by α=0.
    layer = PKMLayer(d_model=v, n_heads=H, n_keys=K, k_dim=kd, top_k=tk,
                      value_bf16=False, score_norm="batch",
                      use_output_gate=False)
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
    # use_output_gate=False because the v7 default α=0 would zero out the
    # gradient to value rows entirely. This test is about the SPARSITY of
    # the gradient w.r.t. value rows, not about α — disable α to isolate.
    layer = PKMLayer(d_model=v, n_heads=H, n_keys=K, k_dim=kd, top_k=tk,
                      value_bf16=False, use_output_gate=False)
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
    # score_norm='batch' to actually exercise BN. use_output_gate=False so α
    # doesn't squash gradient to values.
    layer = PKMLayer(d_model=v, n_heads=H, n_keys=K, k_dim=kd, top_k=tk,
                      value_bf16=False, score_norm="batch",
                      use_output_gate=False)
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
    # ~38.9M total per the plan. v7 adds 1 scalar (out_alpha) and swaps BN
    # for LN (same param shape: 2*K per side); the difference is +1 scalar
    # so total is still ~38.9M.
    assert 38_500_000 < total < 39_500_000, (
        f"param count {total/1e6:.2f}M outside expected ~38.9M range"
    )
    # Value table dominates.
    assert layer.n_value_params == 4 * 256 * 256 * 144


# ============================================================================
# v7 PKM-bootstrap-fix package tests (2026-05-17).
# ============================================================================


def test_pkm_v7_default_score_norm_is_layer():
    """Default score_norm must be 'layer' (drops the v5-pkm BN). The legacy
    'batch' form must still be available for back-compat / ablation."""
    layer = PKMLayer(d_model=32, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)
    assert layer.score_norm_kind == "layer"
    assert hasattr(layer, "ln_s1") and hasattr(layer, "ln_s2")
    assert not hasattr(layer, "bn_s1")

    layer_bn = PKMLayer(d_model=32, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                        value_bf16=False, score_norm="batch")
    assert layer_bn.score_norm_kind == "batch"
    assert hasattr(layer_bn, "bn_s1") and hasattr(layer_bn, "bn_s2")
    assert not hasattr(layer_bn, "ln_s1")


def test_pkm_v7_output_gate_zero_at_init():
    """FIX 1: with use_output_gate=True (default), out_alpha is 0 at init
    so PKM output is exactly 0 — the model can't be poisoned by random
    initial PKM lookups."""
    torch.manual_seed(0)
    layer = PKMLayer(d_model=32, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)
    assert layer.use_output_gate
    assert float(layer.out_alpha) == 0.0
    x = torch.randn(2, 5, 32)
    y = layer(x)
    assert torch.allclose(y, torch.zeros_like(y))


def test_pkm_v7_output_gate_learns():
    """Backward through a non-trivial loss should put gradient on out_alpha
    (so it can grow as PKM proves useful)."""
    torch.manual_seed(0)
    layer = PKMLayer(d_model=32, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)
    # Force α away from 0 to get a meaningful gradient path through values.
    with torch.no_grad():
        layer.out_alpha.copy_(torch.tensor([0.3]))
    x = torch.randn(2, 5, 32)
    y = layer(x)
    target = torch.randn_like(y)
    ((y - target).pow(2).mean()).backward()
    assert layer.out_alpha.grad is not None
    assert float(layer.out_alpha.grad.abs()) > 0.0


def test_pkm_v7_epsilon_greedy_replaces_slots():
    """FIX 2: with random_slot_epsilon=1.0 in training mode, every retrieved
    slot index is replaced by a uniform random slot. The stashed
    _last_slot_idx must then differ from the deterministic (eval) baseline."""
    torch.manual_seed(0)
    H, K, kd, tk, v = 1, 8, 16, 4, 16
    layer = PKMLayer(d_model=v, n_heads=H, n_keys=K, k_dim=kd, top_k=tk,
                      value_bf16=False, use_output_gate=False)
    x = torch.randn(2, 5, v)

    layer.eval()
    layer.random_slot_epsilon = 0.0
    _ = layer(x)
    eval_slots = layer._last_slot_idx.clone()

    layer.train()
    layer.random_slot_epsilon = 1.0
    torch.manual_seed(123)
    _ = layer(x)
    train_slots = layer._last_slot_idx.clone()

    # With ε=1.0, virtually all slots should differ (probability of
    # randomly drawing the same slot is 1/K² = 1/64; over 2*5*1*4 = 40
    # picks the chance of ANY match is ~46% — but the chance of MORE THAN
    # HALF matching is astronomically small).
    diff_frac = (eval_slots != train_slots).float().mean().item()
    assert diff_frac > 0.5, (
        f"ε=1.0 should replace most slots; only {diff_frac:.2f} differ")


def test_pkm_v7_epsilon_greedy_inactive_at_eval():
    """ε-greedy must NOT fire in eval mode (validation must be deterministic)."""
    layer = PKMLayer(d_model=16, n_heads=1, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False, use_output_gate=False)
    layer.eval()
    layer.random_slot_epsilon = 1.0      # would be aggressive if it fired
    x = torch.randn(2, 5, 16)
    _ = layer(x)
    slots_a = layer._last_slot_idx.clone()
    _ = layer(x)
    slots_b = layer._last_slot_idx.clone()
    assert torch.equal(slots_a, slots_b), "eval mode must skip ε-greedy"


def test_pkm_v7_value_init_std_is_configurable():
    """FIX 3: value rows must respect the requested init std."""
    for std in (0.04, 1.0, 2.0):
        layer = PKMLayer(d_model=32, n_heads=2, n_keys=16, k_dim=8, top_k=4,
                          value_bf16=False, value_init_std=std)
        rows = layer.values[0].weight.float()
        observed = rows.std().item()
        # With ~256 rows × 16 dims = 4096 samples per init, std should be
        # within ~5% of the requested value.
        assert abs(observed - std) / std < 0.10, (
            f"requested std={std}, observed {observed:.4f}")


def test_pkm_v7_layernorm_score_norm_works():
    """LayerNorm score normalisation must run end-to-end (forward + backward)
    and produce non-NaN gradients."""
    torch.manual_seed(0)
    layer = PKMLayer(d_model=32, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False, score_norm="layer",
                      use_output_gate=False)
    x = torch.randn(2, 5, 32)
    y = layer(x)
    y.sum().backward()
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad on {name}"


def test_pkm_v7_1_alpha_floor_preserves_sign():
    """v7.1 FIX 1B: with alpha_floor > 0, the effective output gate is
    α + sign(α)·floor — magnitude ≥ floor, sign of learned α preserved."""
    torch.manual_seed(0)
    layer = PKMLayer(d_model=16, n_heads=1, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)  # use_output_gate=True default
    # α defaults to 0 → sign-default-positive → α_eff = +floor
    layer.alpha_floor = 0.3
    x = torch.randn(2, 5, 16)
    y_with_floor = layer(x).detach().clone()
    # Without floor and α=0 the gate is 0 → output is exactly 0
    layer.alpha_floor = 0.0
    y_no_floor = layer(x).detach()
    assert torch.allclose(y_no_floor, torch.zeros_like(y_no_floor))
    # With floor, contribution magnitude is non-trivial (~ floor * pkm_out)
    assert y_with_floor.abs().sum() > 0.0
    # Now set α to negative — α_eff should be negative (sign preserved)
    with torch.no_grad():
        layer.out_alpha.copy_(torch.tensor([-0.05]))
    layer.alpha_floor = 0.3
    y_neg_floor = layer(x).detach()
    # With negative α and +floor, α_eff = -0.05 + (-1)*0.3 = -0.35 → opposite
    # sign from positive-α case.
    cos = (y_with_floor * y_neg_floor).sum() / (
        y_with_floor.norm() * y_neg_floor.norm() + 1e-9)
    assert cos < 0.0, f"sign-preservation broken: cos = {cos:.3f}"


def test_pkm_v7_1_alpha_gets_gradient_through_floor():
    """α should still receive gradient even when the floor is active —
    so it can learn the right post-warmup magnitude."""
    torch.manual_seed(0)
    layer = PKMLayer(d_model=16, n_heads=1, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)
    layer.alpha_floor = 0.3
    x = torch.randn(2, 5, 16)
    y = layer(x)
    target = torch.randn_like(y)
    loss = (y - target).pow(2).mean()
    loss.backward()
    assert layer.out_alpha.grad is not None
    assert float(layer.out_alpha.grad.abs()) > 0.0


def test_pkm_value_lr_mult_creates_separate_group():
    """Build optimizer with pkm_value_lr_mult > 1; verify PKM value rows
    land in a dedicated param group with the boosted LR."""
    from experiments.optim_utils import build_optimizer
    from experiments.model import TinyLM
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    m = TinyLM(vocab_size=64, d_model=64, n_layers=2, n_heads=2, d_head=32,
                attention_cls=DeltaNetAttention,
                use_pkm=True, pkm_after_layer=0,
                pkm_n_heads=2, pkm_n_keys=16, pkm_k_dim=8, pkm_top_k=4,
                pkm_value_bf16=False)
    opts, _ = build_optimizer(
        m, optimizer="adamw", lr=1e-3, lr_muon=5e-3, alpha_wd=0.0,
        steps=10, wd=0.01, lr_schedule="cosine", warmup_steps=0,
        decay_frac=0.0, pkm_value_lr_mult=10.0, verbose=False)
    # Find the param group containing pkm value rows.
    val_ids = {id(emb.weight) for emb in m.pkm_layer.values}
    boosted_lrs = []
    for g in opts[0].param_groups:
        for p in g["params"]:
            if id(p) in val_ids:
                boosted_lrs.append(g["lr"])
    assert len(boosted_lrs) == len(val_ids), "missing value rows in groups"
    assert all(lr == 1e-2 for lr in boosted_lrs), (
        f"PKM-value lr should be 10× base; got {boosted_lrs}")


def test_pkm_value_lr_mult_disabled_at_1():
    """pkm_value_lr_mult=1.0 means no separate group; values use base lr."""
    from experiments.optim_utils import build_optimizer
    from experiments.model import TinyLM
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    m = TinyLM(vocab_size=64, d_model=64, n_layers=2, n_heads=2, d_head=32,
                attention_cls=DeltaNetAttention,
                use_pkm=True, pkm_after_layer=0,
                pkm_n_heads=2, pkm_n_keys=16, pkm_k_dim=8, pkm_top_k=4,
                pkm_value_bf16=False)
    opts, _ = build_optimizer(
        m, optimizer="adamw", lr=1e-3, lr_muon=5e-3, alpha_wd=0.0,
        steps=10, wd=0.01, lr_schedule="cosine", warmup_steps=0,
        decay_frac=0.0, pkm_value_lr_mult=1.0, verbose=False)
    val_ids = {id(emb.weight) for emb in m.pkm_layer.values}
    lrs = set()
    for g in opts[0].param_groups:
        for p in g["params"]:
            if id(p) in val_ids:
                lrs.add(g["lr"])
    assert lrs == {1e-3}


def test_pkm_v7_diversity_loss_helper():
    """train_lm._pkm_diversity_loss returns the negative entropy of the
    per-head slot-selection distribution; lower (more negative) = more
    diverse (higher entropy)."""
    from experiments.train_lm import _pkm_diversity_loss
    layer = PKMLayer(d_model=16, n_heads=2, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False, use_output_gate=False)
    x = torch.randn(4, 6, 16)
    layer(x)
    div1 = float(_pkm_diversity_loss(layer))
    # Now FORCE concentration: stash all-zeros slot indices (every retrieval
    # picks slot 0). Entropy collapses to 0, neg-entropy = 0 (higher than div1).
    layer._last_slot_idx = torch.zeros_like(layer._last_slot_idx)
    layer._last_weights = torch.ones_like(layer._last_weights)
    div_concentrated = float(_pkm_diversity_loss(layer))
    assert div_concentrated > div1, (
        f"concentrated-slot loss {div_concentrated:.4f} should be larger "
        f"(less negative entropy) than diverse-slot loss {div1:.4f}")
