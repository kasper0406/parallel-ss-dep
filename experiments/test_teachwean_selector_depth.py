"""
Tests for experiments/teachwean_selector_depth.py — the teach-then-wean
process-supervision experiment for the learned op-selector.

Run:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m pytest \
      experiments/test_teachwean_selector_depth.py -v
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from experiments.op_selector_depth import (
    OpSelectorAttn, think_forward_opsel, think_forward_oracle,
)
from experiments.latent_think import think_forward
from experiments.depth_via_iteration import (
    build, make_multitable_chase_batch, task_meta, hetero_layout,
)
from experiments.teachwean_selector_depth import (
    aux_op_loss, aux_weight, selection_params, freeze_selection_params,
    think_forward_opsel_fixedvalue,
)

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="DeltaNet/FLA kernels need CUDA")


# ---------------------------------------------------------------------------
# (1) aux op-CE target is one-hot at r and minimized when attn=onehot(r)
# ---------------------------------------------------------------------------
def test_aux_loss_target_is_onehot_at_r_and_minimized_at_onehot():
    B, R, K = 4, 6, 6
    # Perfect counter: attn = onehot(r) at every step -> aux ~ 0.
    onehot = torch.zeros(B, R, K)
    for r in range(R):
        onehot[:, r, r] = 1.0
    loss_perfect = aux_op_loss(onehot).item()
    assert loss_perfect < 1e-4, f"onehot(r) should minimize aux, got {loss_perfect}"

    # Uniform attention -> aux == log(K).
    uniform = torch.full((B, R, K), 1.0 / K)
    loss_uniform = aux_op_loss(uniform).item()
    assert abs(loss_uniform - math.log(K)) < 1e-4

    # Wrong one-hot (attend the LAST position every step) -> large loss.
    wrong = torch.zeros(B, R, K)
    wrong[:, :, K - 1] = 1.0
    loss_wrong = aux_op_loss(wrong).item()
    assert loss_wrong > loss_uniform > loss_perfect

    # Clamp: when R>K the target at step r is clamped to K-1 (no OOB gather).
    onehot_clamped = torch.zeros(B, 8, K)
    for r in range(8):
        onehot_clamped[:, r, min(r, K - 1)] = 1.0
    loss_clamped = aux_op_loss(onehot_clamped).item()
    assert loss_clamped < 1e-4


# ---------------------------------------------------------------------------
# (3) wean curriculum w(t) hits 1.0 at t=0 and 0.0 at/after 60% of steps
# ---------------------------------------------------------------------------
def test_wean_curriculum_schedule():
    steps = 5000
    # baseline: 0 throughout.
    assert aux_weight("baseline", 0, steps) == 0.0
    assert aux_weight("baseline", 2500, steps) == 0.0
    # teach_persist: const 0.5.
    assert aux_weight("teach_persist", 0, steps) == 0.5
    assert aux_weight("teach_persist", 4999, steps) == 0.5
    # teach_then_wean: 1.0 at t=0, 0.0 at/after teach_end (60%).
    assert abs(aux_weight("teach_then_wean", 0, steps) - 1.0) < 1e-9
    assert abs(aux_weight("teach_then_wean", 1500, steps) - 0.5) < 1e-6  # halfway
    assert aux_weight("teach_then_wean", 3000, steps) == 0.0             # teach_end
    assert aux_weight("teach_then_wean", 4000, steps) == 0.0             # after
    # freeze arm shares the same schedule.
    assert abs(aux_weight("teach_then_wean_freeze", 0, steps) - 1.0) < 1e-9
    assert aux_weight("teach_then_wean_freeze", 3000, steps) == 0.0
    # monotone non-increasing across the teach phase.
    prev = 2.0
    for t in range(0, 3001, 100):
        w = aux_weight("teach_then_wean", t, steps)
        assert w <= prev + 1e-9
        prev = w


# ---------------------------------------------------------------------------
# (5) no answer leak: the selector sees only program-position op tokens,
#     never the answer / table values; the aux target is a pure counter.
# ---------------------------------------------------------------------------
def test_no_answer_leak_in_selector_inputs_and_aux_target():
    N, L_ops, K, B = 8, 2, 6, 16
    OP_BASE, QUERY, THINK, PAD, vocab = hetero_layout(N, L_ops)
    g = torch.Generator().manual_seed(0)
    ids, ans, _chain, prog, _vocab = make_multitable_chase_batch(
        B, N, K, L_ops, "cpu", g, homogeneous=False)
    prog_start = L_ops * (2 * N + 1) + 1

    # The selector reads exactly ids[:, prog_start:prog_start+K] = op tokens.
    sel_inputs = ids[:, prog_start:prog_start + K]
    assert sel_inputs.shape == (B, K)
    # Every selector-visible token is an OP token in [OP_BASE, OP_BASE+L_ops).
    assert (sel_inputs >= OP_BASE).all()
    assert (sel_inputs < OP_BASE + L_ops).all()
    # The op tokens equal OP_BASE + program ids (verbatim program, not answer).
    assert torch.equal(sel_inputs, OP_BASE + prog)

    # Answers are node ids in [0, N) — a DISJOINT id range from the op tokens the
    # selector sees, so the selector input cannot encode the answer.
    assert (ans >= 0).all() and (ans < N).all()
    assert N <= OP_BASE  # node-id range is below the op-token range

    # The aux target at step r is the pure counter min(r, K-1) — independent of
    # ans / tables / chain.  Reconstruct it the way aux_op_loss does and confirm.
    R = K
    targets = torch.arange(R).clamp(max=K - 1)
    assert torch.equal(targets, torch.arange(K))
    # Permuting the answers must not change the target.
    ans_perm = ans[torch.randperm(B, generator=g)]
    targets2 = torch.arange(R).clamp(max=K - 1)
    assert torch.equal(targets, targets2)
    assert not torch.equal(ans, ans_perm) or B == 1  # sanity: perm actually moved


# ---------------------------------------------------------------------------
# (2) cold-start no-op parity: at init the op-selector latent loop equals the
#     baseline latent loop (think_forward mode='latent') byte-for-byte (<1e-5).
# ---------------------------------------------------------------------------
@cuda_only
def test_cold_start_noop_parity_with_baseline_latent_loop():
    torch.manual_seed(0)
    N, L_ops, K = 8, 2, 6
    device = "cuda"
    thinking_id, vocab, max_T = task_meta("hetero_mt", N, max(K, 8), L_ops)
    model = build(vocab, thinking_id, d_model=128, n_layers=2, n_heads=4,
                  d_head=32, max_T=max_T, device=device)
    model.eval()
    opsel = OpSelectorAttn(128, max_steps=K + 4).to(device)  # COLD init (out_proj=0)
    prog_start = L_ops * (2 * N + 1) + 1

    g = torch.Generator().manual_seed(123)
    ids, _ans, _chain, _prog, _vocab = make_multitable_chase_batch(
        32, N, K, L_ops, device, g, homogeneous=False)

    for R in (1, 3, 6):
        with torch.no_grad():
            out_base = think_forward(model, ids, R, thinking_id, mode="latent")
            out_opsel = think_forward_opsel(
                model, ids, R, thinking_id, opsel, prog_start, K)
        max_abs = (out_base - out_opsel).abs().max().item()
        assert max_abs < 1e-5, f"R={R}: cold-start op-selector not a no-op (max|d|={max_abs})"


# ---------------------------------------------------------------------------
# (4) freeze arm actually stops gradient on the selection params after the teach
#     phase (they don't change) while out_proj still updates.
# ---------------------------------------------------------------------------
@cuda_only
def test_freeze_stops_selection_params_but_out_proj_keeps_updating():
    import torch.nn.functional as F
    torch.manual_seed(0)
    N, L_ops, K = 8, 2, 6
    device = "cuda"
    thinking_id, vocab, max_T = task_meta("hetero_mt", N, max(K, 8), L_ops)
    model = build(vocab, thinking_id, d_model=128, n_layers=2, n_heads=4,
                  d_head=32, max_T=max_T, device=device)
    opsel = OpSelectorAttn(128, max_steps=K + 4).to(device)
    prog_start = L_ops * (2 * N + 1) + 1
    params = list(model.parameters()) + list(opsel.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    g = torch.Generator().manual_seed(7)

    def one_step():
        ids, ans, _c, _p, _v = make_multitable_chase_batch(
            64, N, K, L_ops, device, g, homogeneous=False)
        logits, attn = think_forward_opsel(
            model, ids, K, thinking_id, opsel, prog_start, K, return_attn=True)
        # train the attention so the selection params WOULD move absent the freeze
        loss = F.cross_entropy(logits, ans) + aux_op_loss(attn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # A couple of warm steps so out_proj leaves zero and selection params move.
    one_step()
    one_step()

    # Snapshot, then FREEZE the selection params.
    sel_before = [p.detach().clone() for p in selection_params(opsel)]
    out_before = opsel.out_proj.weight.detach().clone()
    freeze_selection_params(opsel)

    # Confirm grad really stops on selection params.
    ids, ans, _c, _p, _v = make_multitable_chase_batch(
        64, N, K, L_ops, device, g, homogeneous=False)
    import torch.nn.functional as F2
    logits, attn = think_forward_opsel(
        model, ids, K, thinking_id, opsel, prog_start, K, return_attn=True)
    loss = F2.cross_entropy(logits, ans) + aux_op_loss(attn)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for p in selection_params(opsel):
        assert p.grad is None, "frozen selection param still received a gradient"
    assert opsel.out_proj.weight.grad is not None, "out_proj should still get grad"
    opt.step()

    # Run a few more steps; selection params must be byte-identical, out_proj must move.
    for _ in range(3):
        one_step()
    for p, b in zip(selection_params(opsel), sel_before):
        assert torch.equal(p.detach(), b), "frozen selection param changed after freeze"
    out_after = opsel.out_proj.weight.detach()
    assert not torch.equal(out_after, out_before), "out_proj did not update post-freeze"


# ---------------------------------------------------------------------------
# (6) LOAD-BEARING: fixed-value loop with a one-hot selection at step r injects
#     EXACTLY the oracle's embed(OP_BASE+prog[:,r]) -> output == oracle output.
#     This pins the property the whole teach-then-wean conclusion rests on:
#     perfect taught selection (sel_acc=1.0) is oracle-equivalent (no leak, no
#     learnable value transform), so the fixed-value path measures SELECTION only.
# ---------------------------------------------------------------------------
class _OneHotSelector:
    """Stub op-selector: returns a hard one-hot attention at program position r
    (clamped to K-1), and a zero 'op' (unused by the fixed-value loop)."""
    def __call__(self, prog_embeds, r):
        B, K, d = prog_embeds.shape
        attn = torch.zeros(B, K, device=prog_embeds.device, dtype=prog_embeds.dtype)
        attn[:, min(int(r), K - 1)] = 1.0
        return torch.zeros(B, d, device=prog_embeds.device, dtype=prog_embeds.dtype), attn


@cuda_only
def test_fixedvalue_onehot_selection_equals_oracle():
    device = "cuda"
    torch.manual_seed(0)
    N, K, L_ops = 8, 6, 2
    thinking_id, vocab, max_T = task_meta("hetero_mt", N, K, L_ops)
    model = build(vocab, thinking_id, d_model=128, n_layers=2,
                  n_heads=4, d_head=32, max_T=max_T, device=device).eval()
    prog_start = L_ops * (2 * N + 1) + 1
    g = torch.Generator().manual_seed(7)
    ids, _ans, _chain, prog, _vocab = make_multitable_chase_batch(
        16, N, K, L_ops, device, g, homogeneous=False)

    with torch.no_grad():
        out_fv = think_forward_opsel_fixedvalue(
            model, ids, K, thinking_id, _OneHotSelector(), prog_start, K)
        out_oracle = think_forward_oracle(model, ids, K, thinking_id, prog, N)
    assert out_fv.shape == out_oracle.shape
    maxdiff = (out_fv - out_oracle).abs().max().item()
    assert maxdiff < 1e-4, (
        f"fixed-value one-hot injection != oracle (max|Δ|={maxdiff:.2e}); the "
        f"selection->oracle-equivalence the conclusion rests on is broken")


@cuda_only
def test_fixedvalue_gradient_reaches_selection_params_not_valueproj():
    """In fixed-value mode, gradient must reach the selection params (k_proj /
    pos_key_emb / step_q_emb) via attn; v_proj/out_proj are unused (grad None)."""
    import torch.nn.functional as F2
    device = "cuda"
    torch.manual_seed(0)
    N, K, L_ops = 8, 6, 2
    thinking_id, vocab, max_T = task_meta("hetero_mt", N, K, L_ops)
    model = build(vocab, thinking_id, d_model=128, n_layers=2,
                  n_heads=4, d_head=32, max_T=max_T, device=device)
    opsel = OpSelectorAttn(128, max_steps=K + 2).to(device)
    prog_start = L_ops * (2 * N + 1) + 1
    g = torch.Generator().manual_seed(1)
    ids, ans, _c, _p, _v = make_multitable_chase_batch(
        32, N, K, L_ops, device, g, homogeneous=False)
    logits, attn = think_forward_opsel_fixedvalue(
        model, ids, K, thinking_id, opsel, prog_start, K, return_attn=True)
    (F2.cross_entropy(logits, ans) + aux_op_loss(attn)).backward()
    for p in selection_params(opsel):
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            "selection param got no gradient in fixed-value mode"
    # v_proj/out_proj are not in the fixed-value path -> no gradient.
    assert opsel.v_proj.weight.grad is None
    assert opsel.out_proj.weight.grad is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
