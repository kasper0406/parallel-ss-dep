"""Tests for Phase 3 per-position think-index embedding (2026-05-26).

The flag `think_index_emb_size=N` enables a small zero-init learned
embedding table of shape (N, d_model). At every think position the
table's row corresponding to that token's index within its consecutive
burst (0 for the first think in a burst, 1 for the second, ...) is
added to the input embedding. Bursts longer than N share the row at
index N-1.

This breaks the homogenization observed in
`diag_think_position_diversity.py`: 8 consecutive [THINKING] tokens
have the same input embedding -> hidden-state pairwise cosine +0.146
vs +0.060 at emit positions, effective rank 210 vs 560.

Tests:
  1. think_index_emb_size=0 (default): no parameter created, forward
     is byte-identical to the pre-Phase-3 path.
  2. think_index_emb_size=8 + burst of 5 thinks: the input rows fed
     to the trunk at the 5 think positions are DISTINCT (max pairwise
     cosine < 1.0 - eps).
  3. Bursts longer than the table size clamp to the last index — the
     9th, 10th, ... consecutive thinks all share the position-(N-1)
     embedding.
  4. A think token NOT in a burst (one think between emits) gets the
     position-0 embedding.
  5. State-dict round-trip preserves the embedding weights.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m pytest \
    experiments/test_think_index_emb.py -v
"""
from __future__ import annotations

import pytest
import torch

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


THINK_ID = 1
PAD_ID = 0


def _make_model(*, vocab_size=64, n_layers=2, d_model=16, n_heads=2,
                d_head=8, think_index_emb_size: int = 0, seed: int = 0,
                ) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(
        vocab_size=vocab_size,
        d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head,
        attention_cls=DeltaNetAttention,
        max_T=128,
        think_index_emb_size=think_index_emb_size,
    )


def _set_thinking_id(model: TinyLM, tid: int) -> None:
    model.thinking_token_id = int(tid)


# --------------------------------------------------------------------------
# 1. Default (size=0): no parameter, byte-identical forward
# --------------------------------------------------------------------------

def test_default_off_no_param():
    """think_index_emb_size=0 must not create the embedding parameter."""
    model = _make_model(think_index_emb_size=0)
    assert not hasattr(model, "think_index_emb"), \
        "think_index_emb_size=0 should not create think_index_emb"
    assert model.think_index_emb_size == 0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_default_off_forward_unchanged():
    """With size=0, forward output matches the no-flag baseline exactly."""
    torch.manual_seed(0)
    m0 = _make_model(think_index_emb_size=0, seed=0).cuda().eval()
    torch.manual_seed(0)
    m1 = _make_model(think_index_emb_size=0, seed=0).cuda().eval()
    _set_thinking_id(m0, THINK_ID)
    _set_thinking_id(m1, THINK_ID)
    x = torch.randint(2, 64, (1, 24), device="cuda")
    x[0, 5] = THINK_ID
    x[0, 6] = THINK_ID
    with torch.no_grad():
        y0 = m0(x)
        y1 = m1(x)
    assert torch.allclose(y0, y1, atol=0, rtol=0)


# --------------------------------------------------------------------------
# 2 + 3 + 4. The mathematical core: index assignment + clamp
# --------------------------------------------------------------------------

def test_index_assignment_inside_burst():
    """5-think burst -> indices [0, 1, 2, 3, 4]; isolated think -> [0]."""
    model = _make_model(think_index_emb_size=8)
    _set_thinking_id(model, THINK_ID)
    # Layout: [emit, emit, T, T, T, T, T, emit, T, emit]
    #          0     1   2  3  4  5  6   7   8   9
    # Expected burst_idx at think positions: 0,1,2,3,4 then 0.
    ids = torch.tensor(
        [[5, 5, THINK_ID, THINK_ID, THINK_ID, THINK_ID, THINK_ID, 5,
          THINK_ID, 5]],
        dtype=torch.long)
    # Recreate the internal computation by calling the helper directly.
    contrib = model._compute_think_index_emb(ids)  # (1, 10, d_model)
    # Non-think positions contribute zero.
    nonthink = [0, 1, 7, 9]
    for t in nonthink:
        assert torch.all(contrib[0, t] == 0), \
            f"non-think position {t} should contribute 0"
    # Think positions in the burst should equal the table rows 0..4 .
    table = model.think_index_emb.weight  # (8, d_model)
    # Manually set table to a non-zero pattern so equality is meaningful.
    with torch.no_grad():
        model.think_index_emb.weight.copy_(
            torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 16))
    contrib = model._compute_think_index_emb(ids)
    expected_idx = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 0}
    for t, k in expected_idx.items():
        assert torch.allclose(contrib[0, t], model.think_index_emb.weight[k]), \
            f"think position {t} should map to table row {k}"


def test_burst_clamp_at_table_size():
    """Bursts longer than table size: indices >=N share row N-1."""
    N = 4
    model = _make_model(think_index_emb_size=N)
    _set_thinking_id(model, THINK_ID)
    with torch.no_grad():
        model.think_index_emb.weight.copy_(
            torch.arange(N, dtype=torch.float32).view(N, 1)
            .expand(N, 16).contiguous())
    # 7-think burst (longer than table N=4).
    ids = torch.tensor(
        [[9, THINK_ID, THINK_ID, THINK_ID, THINK_ID,
              THINK_ID, THINK_ID, THINK_ID, 9]],
        dtype=torch.long)
    contrib = model._compute_think_index_emb(ids)
    # Think positions are 1..7 -> burst_idx 0..6 -> clamped to 0,1,2,3,3,3,3.
    expected = [0, 1, 2, 3, 3, 3, 3]
    for t, k in enumerate(expected, start=1):
        assert torch.allclose(contrib[0, t], model.think_index_emb.weight[k]), \
            f"position {t} should clamp to row {k}"


def test_distinct_inputs_in_burst():
    """5 consecutive thinks should produce 5 DISTINCT additive vectors."""
    model = _make_model(think_index_emb_size=8)
    _set_thinking_id(model, THINK_ID)
    # Randomize the table so distinctness is meaningful.
    with torch.no_grad():
        torch.manual_seed(42)
        model.think_index_emb.weight.normal_(std=0.5)
    ids = torch.tensor(
        [[5, 5, THINK_ID, THINK_ID, THINK_ID, THINK_ID, THINK_ID, 5]],
        dtype=torch.long)
    contrib = model._compute_think_index_emb(ids)
    vecs = contrib[0, 2:7]  # (5, d_model)
    # All pairs distinct.
    for i in range(5):
        for j in range(i + 1, 5):
            assert not torch.allclose(vecs[i], vecs[j], atol=1e-6), \
                f"think rows {i} and {j} of burst should differ"


def test_isolated_think_gets_index_zero():
    """A lone think token between emits is the 0th in its 1-long burst."""
    model = _make_model(think_index_emb_size=8)
    _set_thinking_id(model, THINK_ID)
    with torch.no_grad():
        model.think_index_emb.weight.copy_(
            torch.arange(8, dtype=torch.float32).view(8, 1)
            .expand(8, 16).contiguous())
    ids = torch.tensor([[7, THINK_ID, 7, 7, THINK_ID, 7]], dtype=torch.long)
    contrib = model._compute_think_index_emb(ids)
    # Both think positions are isolated -> index 0 -> row 0 (all zeros).
    assert torch.allclose(contrib[0, 1], model.think_index_emb.weight[0])
    assert torch.allclose(contrib[0, 4], model.think_index_emb.weight[0])


# --------------------------------------------------------------------------
# 5. State-dict round-trip
# --------------------------------------------------------------------------

def test_state_dict_round_trip():
    """Save / load preserves think_index_emb weights."""
    m_save = _make_model(think_index_emb_size=8, seed=0)
    _set_thinking_id(m_save, THINK_ID)
    with torch.no_grad():
        m_save.think_index_emb.weight.normal_(std=0.5)
    saved = m_save.state_dict()
    assert "think_index_emb.weight" in saved

    m_load = _make_model(think_index_emb_size=8, seed=1)  # different init
    _set_thinking_id(m_load, THINK_ID)
    # Pre-load: weights should differ (different seed).
    assert not torch.allclose(
        m_save.think_index_emb.weight, m_load.think_index_emb.weight)
    m_load.load_state_dict(saved)
    assert torch.allclose(
        m_save.think_index_emb.weight, m_load.think_index_emb.weight)


# --------------------------------------------------------------------------
# Bonus: confirm the additive embedding actually reaches the trunk input
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_nontrivial_table_changes_output():
    """A non-zero think_index_emb table must change the forward output
    at think positions (otherwise we'd silently be a no-op).
    """
    torch.manual_seed(0)
    model = _make_model(think_index_emb_size=8, seed=0).cuda().eval()
    _set_thinking_id(model, THINK_ID)
    ids = torch.tensor(
        [[5, 5, THINK_ID, THINK_ID, THINK_ID, 5, 5]],
        dtype=torch.long, device="cuda")
    with torch.no_grad():
        y_zero = model(ids)  # table is zero-init, no contribution
        # Now set non-zero and re-run.
        model.think_index_emb.weight.normal_(std=0.5)
        y_nonzero = model(ids)
    # Outputs at non-think positions can still differ because the trunk
    # mixes earlier think positions into them via the recurrence — so
    # just check the global tensors differ.
    assert not torch.allclose(y_zero, y_nonzero, atol=1e-5)
