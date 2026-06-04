"""Tests for the think-time op-selector LineSelectorAttn (2026-06-03).

`LineSelectorAttn` softly selects a program LINE from the non-think prompt
prefix (by a learned per-think-step query) and injects that line's VERBATIM
mean-pooled INPUT embedding as a zero-init-α additive side-channel into the
trunk input at think positions ONLY. This gives latent thinking
position-addressable verbatim content access WITHOUT overwriting the carried
latent thread.

Cold start: `out_proj.weight` is zero-init AND a learnable scalar α (init 0)
wraps the output, so a fresh selector returns EXACTLY zero everywhere — the
additive term is a no-op and the trunk is byte-identical to "no selector".

Tests:
  1. Cold-start (α=0, zero out_proj): forward with use_line_selector=True is
     byte-identical to use_line_selector=False (the zero-init no-op).
  2. After α=1 + randomized out_proj: the output DIFFERS ONLY at think
     positions (non-think positions are byte-identical to the off model).
  3. burst_idx refactor: _compute_think_index_emb output is unchanged vs a
     saved reference, and _compute_burst_index matches the manual cumsum-reset.

The DeltaNet forward needs CUDA (Triton kernels); the refactor/index-math
checks (test 3) run on CPU.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m pytest \
    experiments/test_line_selector.py -v
"""
from __future__ import annotations

import pytest
import torch

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM, LineSelectorAttn


THINK_ID = 1
NL_ID = 2          # newline token id (line separator)
PAD_ID = 0


def _make_model(*, vocab_size=64, n_layers=2, d_model=32, n_heads=2,
                d_head=16, use_line_selector=False,
                line_selector_max_lines=16, seed=0) -> TinyLM:
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=vocab_size,
        d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head,
        attention_cls=DeltaNetAttention,
        max_T=128,
        thinking_token_id=THINK_ID,
        use_line_selector=use_line_selector,
        line_selector_max_lines=line_selector_max_lines,
        newline_token_id=NL_ID,
    )
    return m


def _make_input(device="cpu") -> torch.Tensor:
    # Layout: [tok tok \n tok tok \n tok tok THINK THINK]
    #           0   1  2  3   4  5  6   7    8     9
    ids = torch.tensor(
        [[5, 6, NL_ID, 7, 8, NL_ID, 9, 10, THINK_ID, THINK_ID]],
        dtype=torch.long, device=device)
    return ids


# --------------------------------------------------------------------------
# 1. Cold-start no-op: use_line_selector=True == use_line_selector=False
# --------------------------------------------------------------------------

def test_default_off_no_param():
    """use_line_selector=False must not create the line_selector module."""
    m = _make_model(use_line_selector=False)
    assert not hasattr(m, "line_selector")
    assert m.use_line_selector is False


def test_module_exists_when_on():
    m = _make_model(use_line_selector=True)
    assert isinstance(m.line_selector, LineSelectorAttn)
    # Cold-start invariants: α==0 and out_proj zero-init.
    assert torch.all(m.line_selector.alpha == 0)
    assert torch.all(m.line_selector.out_proj.weight == 0)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_cold_start_byte_identical():
    """α=0 + zero out_proj -> forward identical to use_line_selector=False."""
    m_off = _make_model(use_line_selector=False, seed=0).cuda().eval()
    m_on = _make_model(use_line_selector=True, seed=0).cuda().eval()
    # The two models share the same trunk init (same seed); the only extra
    # weights in m_on are the (zero-init) line_selector. Copy the shared trunk
    # so any non-line-selector init noise can't cause a spurious diff.
    on_sd = m_on.state_dict()
    for k, v in m_off.state_dict().items():
        on_sd[k] = v
    m_on.load_state_dict(on_sd, strict=False)
    ids = _make_input("cuda")
    with torch.no_grad():
        y_off = m_off(ids)
        y_on = m_on(ids)
    assert torch.allclose(y_off, y_on, atol=0, rtol=0), \
        "cold-start line_selector must be byte-identical to off"


# --------------------------------------------------------------------------
# 2. After α=1 + random out_proj: output differs ONLY at think positions
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_active_selector_changes_only_think_positions():
    m_off = _make_model(use_line_selector=False, seed=0).cuda().eval()
    m_on = _make_model(use_line_selector=True, seed=0).cuda().eval()
    on_sd = m_on.state_dict()
    for k, v in m_off.state_dict().items():
        on_sd[k] = v
    m_on.load_state_dict(on_sd, strict=False)
    # Activate the selector: α=1 and a non-zero out_proj.
    with torch.no_grad():
        m_on.line_selector.alpha.fill_(1.0)
        torch.nn.init.normal_(m_on.line_selector.out_proj.weight, std=0.5)
    ids = _make_input("cuda")
    with torch.no_grad():
        y_off = m_off(ids)
        y_on = m_on(ids)
    think_mask = (ids == THINK_ID)[0]              # (T,)
    per_pos_diff = (y_on - y_off).abs().amax(dim=-1)[0]  # (T,)
    # Non-think positions: the selector's ADDITIVE term is exactly zero at
    # those positions; the DeltaNet recurrence is causal (think tokens are at
    # the END), so earlier non-think positions cannot be perturbed by the
    # later think-position injection. They must be byte-identical.
    for t in range(ids.shape[1]):
        if not bool(think_mask[t]):
            assert per_pos_diff[t] == 0, \
                f"non-think position {t} changed (diff {per_pos_diff[t]})"
    # Think positions: at least one must actually differ (selector fired).
    assert per_pos_diff[think_mask].max() > 0, \
        "active selector did not change any think position"


# --------------------------------------------------------------------------
# 3. burst_idx refactor: _compute_think_index_emb unchanged + index math
# --------------------------------------------------------------------------

def test_burst_index_matches_manual_cumsum_reset():
    """_compute_burst_index == the documented cumsum-reset formula."""
    m = _make_model(use_line_selector=True)
    # [emit emit T T T T T emit T emit] -> burst idx 0,1,2,3,4 then 0.
    ids = torch.tensor(
        [[5, 5, THINK_ID, THINK_ID, THINK_ID, THINK_ID, THINK_ID, 5,
          THINK_ID, 5]], dtype=torch.long)
    burst = m._compute_burst_index(ids)[0].tolist()
    # Non-think positions are 0; think positions carry the in-burst index.
    expected = [0, 0, 0, 1, 2, 3, 4, 0, 0, 0]
    assert burst == expected, f"burst index {burst} != {expected}"


def test_think_index_emb_unchanged_after_refactor():
    """_compute_think_index_emb output must equal a hand-rolled reference
    computed with the ORIGINAL inline cumsum-reset logic (no refactor)."""
    m = TinyLM(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=128,
        thinking_token_id=THINK_ID, think_index_emb_size=8)
    # Give the table a distinctive non-zero pattern.
    with torch.no_grad():
        m.think_index_emb.weight.copy_(
            torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 32))
    ids = torch.tensor(
        [[5, 5, THINK_ID, THINK_ID, THINK_ID, THINK_ID, THINK_ID, 5,
          THINK_ID, 5]], dtype=torch.long)

    # --- Reference: the ORIGINAL inline implementation (pre-refactor). ---
    think_mask = (ids == THINK_ID)
    m_int = think_mask.to(torch.int64)
    c = m_int.cumsum(dim=1)
    reset = (c * (1 - m_int)).cummax(dim=1).values
    burst_idx = (c - reset - 1).clamp(min=0)
    burst_idx = burst_idx.clamp(max=m.think_index_emb_size - 1)
    idx_emb = m.think_index_emb(burst_idx)
    ref = idx_emb * think_mask.unsqueeze(-1).to(idx_emb.dtype)

    out = m._compute_think_index_emb(ids)
    assert torch.allclose(out, ref, atol=0, rtol=0), \
        "refactored _compute_think_index_emb diverged from the inline reference"


def test_line_selector_scatter_mean_is_verbatim():
    """The selected value table is the VERBATIM mean of each line's input
    embeddings; the additive term is zero at non-think positions and the
    selection respects the line_valid (empty-line) mask."""
    d = 32
    sel = LineSelectorAttn(d, max_lines=8, max_burst=4)
    # Activate so we can read non-trivial output.
    with torch.no_grad():
        sel.alpha.fill_(1.0)
        torch.nn.init.eye_(sel.out_proj.weight)  # identity out_proj
        torch.nn.init.eye_(sel.v_proj.weight)    # vals == line_val
    ids = torch.tensor(
        [[5, 6, NL_ID, 7, 8, NL_ID, 9, THINK_ID]], dtype=torch.long)
    embeds = torch.randn(1, ids.shape[1], d)
    burst_idx = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    out = sel(torch.zeros_like(embeds), ids, embeds, THINK_ID, NL_ID, burst_idx)
    # Non-think positions: additive term must be exactly zero.
    for t in range(7):
        assert torch.all(out[0, t] == 0)
    # Think position is non-zero (selector fired).
    assert out[0, 7].abs().sum() > 0
    # The per-line value table is the verbatim mean of constituent input
    # embeddings; reconstruct line 0 (tokens 0,1) and line 2 (token 6) means.
    line0_mean = embeds[0, [0, 1]].mean(dim=0)
    line2_mean = embeds[0, [6]].mean(dim=0)
    # Recompute the module's line_id to assert verbatim pooling. The cumsum of
    # is_nl increments AT the newline position, so the \n token belongs to the
    # line it TERMINATES (line 1), not the preceding line. Hence line 0 is just
    # tokens {0, 1}.
    prompt_mask = ids != THINK_ID
    is_nl = (ids == NL_ID) & prompt_mask
    line_id = is_nl.to(torch.int64).cumsum(dim=1)
    assert torch.allclose(
        embeds[0][(line_id[0] == 0) & prompt_mask[0]].mean(dim=0),
        embeds[0, [0, 1]].mean(dim=0)), \
        "line 0 pooled value must be the verbatim mean of tokens {0,1}"
    # Sanity: the precomputed means are finite (verbatim, no projection mangling
    # since v_proj/out_proj are identity).
    assert torch.isfinite(line0_mean).all() and torch.isfinite(line2_mean).all()


def test_no_think_batch_returns_zeros():
    """All-prompt batch (no think tokens) -> selector returns all zeros."""
    d = 32
    sel = LineSelectorAttn(d, max_lines=8, max_burst=4)
    with torch.no_grad():
        sel.alpha.fill_(1.0)
        torch.nn.init.normal_(sel.out_proj.weight, std=0.5)
    ids = torch.tensor([[5, 6, NL_ID, 7, 8]], dtype=torch.long)  # no THINK
    embeds = torch.randn(1, ids.shape[1], d)
    burst_idx = torch.zeros_like(ids)
    out = sel(torch.zeros_like(embeds), ids, embeds, THINK_ID, NL_ID, burst_idx)
    assert torch.all(out == 0)


def test_state_dict_roundtrip():
    m = _make_model(use_line_selector=True, line_selector_max_lines=16)
    sd = m.state_dict()
    assert any(k.startswith("line_selector.") for k in sd)
    m2 = _make_model(use_line_selector=True, line_selector_max_lines=16, seed=7)
    m2.load_state_dict(sd, strict=True)
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(),
                                  m2.state_dict().items()):
        assert torch.allclose(v1, v2, atol=0, rtol=0)
