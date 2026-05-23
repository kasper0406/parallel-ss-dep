"""Tests for the FIX-A write_only_at_think flag on WorkingMemory.

Added 2026-05-18 after the diag_thinking_machinery.py probe showed that
the v7.1-distilled SFT model writes uniformly across think/emit
positions (buffer composition matches baseline % think). The new flag
masks the write-gate to a very-negative value at non-think positions
before the top-K selection, so the buffer always holds think-content.

These tests pin the masking behaviour exactly so future refactors don't
silently regress it.
"""
import math

import pytest
import torch

from experiments.model import WorkingMemory


THINK_TOKEN_ID = 99


@pytest.fixture
def wm_pair():
    """A (off, on) pair of WorkingMemory instances sharing identical
    weights, differing only in write_only_at_think."""
    d_model, d_mem, mem_size = 16, 8, 4
    torch.manual_seed(0)
    off = WorkingMemory(d_model=d_model, d_mem=d_mem, mem_size=mem_size,
                         thinking_token_id=THINK_TOKEN_ID,
                         write_only_at_think=False)
    on = WorkingMemory(d_model=d_model, d_mem=d_mem, mem_size=mem_size,
                        thinking_token_id=THINK_TOKEN_ID,
                        write_only_at_think=True)
    on.load_state_dict(off.state_dict())
    return off, on, d_model


def test_flag_stored(wm_pair):
    off, on, _ = wm_pair
    assert off.write_only_at_think is False
    assert on.write_only_at_think is True


def test_off_buffer_can_be_any_position(wm_pair):
    """With the flag OFF, the top-K buffer may include emit-token
    positions whenever their (random-init) write-gate is highest."""
    off, _, d_model = wm_pair
    B, T = 2, 16
    torch.manual_seed(1)
    h = torch.randn(B, T, d_model)
    # Build an input_ids with NO think tokens — pure emit positions.
    input_ids = torch.zeros(B, T, dtype=torch.long)
    # Run forward — should not raise. Buffer is implicit; we just check
    # that nothing crashes and the output has the right shape.
    out = off(h, input_ids)
    assert out.shape == (B, T, d_model)


def test_on_buffer_is_all_think_when_enough_thinks(wm_pair):
    """With the flag ON and n_think >= K, every buffer slot must be a
    think-token position."""
    _, on, d_model = wm_pair
    B, T = 1, 16
    K = on.mem_size  # 4
    torch.manual_seed(2)
    h = torch.randn(B, T, d_model)
    # Place 8 think tokens at positions [2, 4, 6, 8, 9, 12, 13, 15]
    think_positions = [2, 4, 6, 8, 9, 12, 13, 15]
    input_ids = torch.zeros(B, T, dtype=torch.long)
    for p in think_positions:
        input_ids[0, p] = THINK_TOKEN_ID
    # Run forward and grab the buffer via the same mechanism the
    # forward uses internally.
    # We can't easily extract top_idx from forward() output, so we
    # mirror the topk logic with monkey-patching.
    captured = {}
    orig_topk = torch.topk
    def _capture(g, k, dim=-1):
        captured["g"] = g.clone()
        captured["k"] = k
        return orig_topk(g, k, dim=dim)
    torch.topk = _capture
    try:
        on(h, input_ids)
    finally:
        torch.topk = orig_topk
    # Verify that the masked g has -1.0 at non-think positions
    g = captured["g"]
    for t in range(T):
        if t in think_positions:
            assert g[0, t] >= 0.0, f"think pos {t} should keep its sigmoid g"
        else:
            assert g[0, t] == -1.0, (
                f"non-think pos {t} should be masked to -1.0, got {g[0, t]}"
            )
    # And topk(g, k=K) should select think positions (top-K largest)
    _, top_idx = orig_topk(g, k=K)
    selected = set(top_idx[0].tolist())
    assert selected.issubset(set(think_positions)), (
        f"buffer slots {selected} must all be think positions "
        f"{set(think_positions)}"
    )


def test_on_falls_back_when_fewer_think_than_k(wm_pair):
    """When n_think < K, the remaining slots must be filled (topk
    returns K values regardless). They'll have masked-g = -1.0, which
    log-bias drives their attention score to -inf — they effectively
    don't contribute to reads."""
    _, on, d_model = wm_pair
    B, T = 1, 16
    K = on.mem_size  # 4
    torch.manual_seed(3)
    h = torch.randn(B, T, d_model)
    # Only 2 think tokens — less than K=4
    think_positions = [3, 10]
    input_ids = torch.zeros(B, T, dtype=torch.long)
    for p in think_positions:
        input_ids[0, p] = THINK_TOKEN_ID
    # Forward must not crash even when n_think < K.
    out = on(h, input_ids)
    assert out.shape == (B, T, d_model)
    # And the read injection at emit positions should be zero by the
    # read-mask (only think positions inject), so the only effect of
    # think-only writes is at the 2 think positions.


def test_on_vs_off_differ_when_emit_g_higher(wm_pair):
    """The flag's whole point: when emit positions have higher random
    g than think positions, the OFF version picks emit, the ON version
    picks think. The outputs should then differ at the think positions
    (where the read injects)."""
    off, on, d_model = wm_pair
    B, T = 1, 8
    torch.manual_seed(4)
    h = torch.randn(B, T, d_model)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    # One think token at the end so it can read from earlier positions
    input_ids[0, 7] = THINK_TOKEN_ID
    # Also put one think mid-sequence so the buffer can include it
    input_ids[0, 2] = THINK_TOKEN_ID
    out_off = off(h, input_ids)
    out_on = on(h, input_ids)
    # At think positions, the injections differ because the buffer
    # content differs (off may include emit positions, on includes
    # only positions 2).
    delta_think = (out_on[0, 7] - out_off[0, 7]).norm().item()
    delta_emit = (out_on[0, 4] - out_off[0, 4]).norm().item()  # an emit pos
    # Think positions should see a real difference (the injection changes)
    assert delta_think > 1e-4, (
        f"think position should see different injection with the flag; "
        f"got delta={delta_think:.6f}"
    )
    # Emit positions get NO injection either way (mask gates it off)
    # so they should be identical
    assert delta_emit < 1e-6, (
        f"emit position should be identical (no injection); got "
        f"delta={delta_emit:.6f}"
    )


def test_on_with_zero_thinks_does_not_crash(wm_pair):
    """Edge case: input has no think tokens at all (e.g., pretrain on
    plain text). With the flag ON, all positions get masked to -1.0,
    but topk still returns K positions (just all with g=-1.0).
    """
    _, on, d_model = wm_pair
    B, T = 1, 8
    torch.manual_seed(5)
    h = torch.randn(B, T, d_model)
    input_ids = torch.zeros(B, T, dtype=torch.long)  # no think tokens
    out = on(h, input_ids)
    assert out.shape == (B, T, d_model)
    # And since there are no think positions, no read mask fires, so the
    # output should be exactly h (no injection).
    assert torch.allclose(out, h, atol=1e-5), (
        "without any think tokens, no injection should happen — output "
        "should equal input"
    )
