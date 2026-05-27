"""Tests for Phase B ThinkAdapter (2026-05-26).

The flag `use_think_adapter=True` adds a small 2-layer MLP per Block
(`d_model -> hidden_mult * d_model -> d_model`, GELU) plus a learnable
scalar alpha (init 0). At think positions, the adapter contribution is
added to the residual stream AFTER attn + MLP:

    h_out = h_in + alpha * think_mask * ThinkAdapter(h_in)

Invariants under test:
  1. Default off -> no params added (no `think_adapter` submodule).
  2. Default off -> forward output bit-identical to the no-adapter trunk.
  3. use_think_adapter=True, alpha=0 (init) -> forward still bit-identical
     to no-adapter (the zero-alpha byte-identity guarantee).
  4. use_think_adapter=True, alpha=0.5 (manually set) -> forward DIFFERS
     from no-adapter at and after think positions.
  5. State-dict round-trip preserves adapter weights AND reproduces output.
  6. Adapter params route to AdamW (not Muon) under build_optimizer:
     the Linear weights and the alpha scalar all appear in AdamW groups.

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_think_adapter.py -v

The CUDA-requiring tests skip cleanly without a GPU (DeltaNet's Triton
kernels need CUDA); the CPU-friendly tests cover the parameter-creation,
state-dict, and optimizer-routing invariants.
"""
from __future__ import annotations

import copy

import pytest
import torch

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM
from experiments.optim_utils import _is_think_adapter, build_optimizer


THINK_ID = 1


def _make_model(
    *,
    vocab_size: int = 64,
    n_layers: int = 2,
    d_model: int = 16,
    n_heads: int = 2,
    d_head: int = 8,
    use_think_adapter: bool = False,
    think_adapter_hidden_mult: int = 2,
    seed: int = 0,
) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        attention_cls=DeltaNetAttention,
        max_T=128,
        use_think_adapter=use_think_adapter,
        think_adapter_hidden_mult=think_adapter_hidden_mult,
    )


def _set_thinking_id(model: TinyLM, tid: int) -> None:
    model.thinking_token_id = int(tid)


# --------------------------------------------------------------------------
# 1. Default off -> no parameters added
# --------------------------------------------------------------------------

def test_default_off_no_params():
    """use_think_adapter=False (default) must not create think_adapter."""
    model = _make_model(use_think_adapter=False)
    assert model.use_think_adapter is False
    for L, blk in enumerate(model.blocks):
        assert not hasattr(blk, "think_adapter"), (
            f"block {L} unexpectedly has think_adapter when "
            f"use_think_adapter=False"
        )
    # No parameter name should match the adapter pattern.
    for name, _ in model.named_parameters():
        assert ".think_adapter." not in name, (
            f"unexpected adapter param {name!r} when use_think_adapter=False"
        )


def test_on_creates_expected_params():
    """use_think_adapter=True creates exactly fc1/fc2/alpha per block."""
    n_layers = 3
    d_model = 24
    hidden_mult = 3
    model = _make_model(
        n_layers=n_layers, d_model=d_model, n_heads=2, d_head=12,
        use_think_adapter=True,
        think_adapter_hidden_mult=hidden_mult,
    )
    adapter_names = [
        name for name, _ in model.named_parameters()
        if ".think_adapter." in name
    ]
    # Per block: fc1.weight, fc1.bias, fc2.weight, fc2.bias, alpha = 5.
    assert len(adapter_names) == 5 * n_layers, adapter_names
    # alpha is initialised to zero.
    for L, blk in enumerate(model.blocks):
        assert torch.all(blk.think_adapter.alpha == 0.0), (
            f"block {L}.think_adapter.alpha should init to 0, "
            f"got {blk.think_adapter.alpha.tolist()}"
        )
        # fc1 shape: (d_model -> hidden_mult * d_model).
        assert blk.think_adapter.fc1.weight.shape == (
            hidden_mult * d_model, d_model,
        )
        # fc2 shape: (hidden_mult * d_model -> d_model).
        assert blk.think_adapter.fc2.weight.shape == (
            d_model, hidden_mult * d_model,
        )


# --------------------------------------------------------------------------
# 2 + 3. Byte-identity invariants
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_default_off_forward_byte_identical():
    """Two independently-built models with use_think_adapter=False at the
    same seed must produce identical outputs."""
    m0 = _make_model(use_think_adapter=False, seed=0).cuda().eval()
    m1 = _make_model(use_think_adapter=False, seed=0).cuda().eval()
    _set_thinking_id(m0, THINK_ID)
    _set_thinking_id(m1, THINK_ID)
    x = torch.randint(2, 64, (1, 24), device="cuda")
    x[0, 5] = THINK_ID
    x[0, 6] = THINK_ID
    with torch.no_grad():
        y0 = m0(x)
        y1 = m1(x)
    assert torch.equal(y0, y1)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_alpha_zero_byte_identical_to_no_adapter():
    """use_think_adapter=True with alpha=0 (init) -> forward output must
    be byte-identical to the same trunk with use_think_adapter=False.

    This is the load-bearing invariant: an existing ckpt loaded into a
    Phase-B model with strict=False keeps its original behaviour because
    the adapter contributes 0 at every position with alpha=0.
    """
    # Build adapter-OFF reference and adapter-ON variant from the SAME
    # weights (we manually copy non-adapter state across).
    m_off = _make_model(use_think_adapter=False, seed=0).cuda().eval()
    m_on = _make_model(use_think_adapter=True, seed=0).cuda().eval()
    _set_thinking_id(m_off, THINK_ID)
    _set_thinking_id(m_on, THINK_ID)
    # Copy the off model's state into the on model (skipping adapter keys).
    on_state = m_on.state_dict()
    off_state = m_off.state_dict()
    for k in off_state:
        if k in on_state and on_state[k].shape == off_state[k].shape:
            on_state[k].copy_(off_state[k])
    m_on.load_state_dict(on_state, strict=True)
    # Confirm alpha is zero (init).
    for blk in m_on.blocks:
        assert torch.all(blk.think_adapter.alpha == 0.0)

    x = torch.randint(2, 64, (1, 24), device="cuda")
    x[0, 5] = THINK_ID
    x[0, 6] = THINK_ID
    with torch.no_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    assert torch.allclose(y_off, y_on, atol=0, rtol=0), (
        f"alpha=0 invariant broken: max diff "
        f"{(y_off - y_on).abs().max().item()}"
    )


# --------------------------------------------------------------------------
# 4. Setting alpha != 0 changes the output
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_nonzero_alpha_changes_output():
    """With alpha=0.5 set manually, the forward output must differ from
    the no-adapter baseline (otherwise the adapter is a structural
    no-op)."""
    m_off = _make_model(use_think_adapter=False, seed=0).cuda().eval()
    m_on = _make_model(use_think_adapter=True, seed=0).cuda().eval()
    _set_thinking_id(m_off, THINK_ID)
    _set_thinking_id(m_on, THINK_ID)
    # Sync non-adapter weights so the only difference is the adapter
    # contribution at think positions.
    on_state = m_on.state_dict()
    off_state = m_off.state_dict()
    for k in off_state:
        if k in on_state and on_state[k].shape == off_state[k].shape:
            on_state[k].copy_(off_state[k])
    m_on.load_state_dict(on_state, strict=True)
    # Set alpha to 0.5 in every block.
    with torch.no_grad():
        for blk in m_on.blocks:
            blk.think_adapter.alpha.fill_(0.5)

    x = torch.randint(2, 64, (1, 24), device="cuda")
    x[0, 5] = THINK_ID
    x[0, 6] = THINK_ID
    with torch.no_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    # The trunk is recurrent; even a perturbation at think positions
    # propagates to all subsequent tokens via the recurrence, so it's
    # safe to compare the full tensors.
    assert not torch.allclose(y_off, y_on, atol=1e-5), (
        "alpha=0.5 must change the forward output relative to no-adapter"
    )


# --------------------------------------------------------------------------
# 5. State-dict round-trip
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_state_dict_round_trip_preserves_output():
    """Save adapter weights, load into a fresh model, reproduce output."""
    m_save = _make_model(use_think_adapter=True, seed=0).cuda().eval()
    _set_thinking_id(m_save, THINK_ID)
    # Set the adapter weights to a non-default pattern so equality is meaningful.
    with torch.no_grad():
        for blk in m_save.blocks:
            blk.think_adapter.fc1.weight.normal_(std=0.1)
            blk.think_adapter.fc2.weight.normal_(std=0.1)
            blk.think_adapter.alpha.fill_(0.3)
    saved = {k: v.cpu().clone() for k, v in m_save.state_dict().items()}
    # Confirm at least one adapter key made it in.
    adapter_keys = [k for k in saved if ".think_adapter." in k]
    assert len(adapter_keys) > 0, "expected adapter keys in state dict"

    # Fresh model from a different seed, then load.
    m_load = _make_model(use_think_adapter=True, seed=99).cuda().eval()
    _set_thinking_id(m_load, THINK_ID)
    m_load.load_state_dict({k: v.cuda() for k, v in saved.items()}, strict=True)
    # Adapter weights match exactly.
    for blk_s, blk_l in zip(m_save.blocks, m_load.blocks):
        assert torch.equal(blk_s.think_adapter.alpha, blk_l.think_adapter.alpha)
        assert torch.equal(blk_s.think_adapter.fc1.weight,
                           blk_l.think_adapter.fc1.weight)
        assert torch.equal(blk_s.think_adapter.fc2.weight,
                           blk_l.think_adapter.fc2.weight)

    x = torch.randint(2, 64, (1, 24), device="cuda")
    x[0, 5] = THINK_ID
    x[0, 6] = THINK_ID
    with torch.no_grad():
        y_save = m_save(x)
        y_load = m_load(x)
    assert torch.allclose(y_save, y_load, atol=0, rtol=0)


def test_state_dict_contains_adapter_keys():
    """CPU-friendly: confirm fc1/fc2/alpha names appear in state_dict."""
    n_layers = 2
    model = _make_model(n_layers=n_layers, use_think_adapter=True, seed=0)
    sd = model.state_dict()
    for L in range(n_layers):
        for sub in ("fc1.weight", "fc1.bias",
                    "fc2.weight", "fc2.bias", "alpha"):
            key = f"blocks.{L}.think_adapter.{sub}"
            assert key in sd, f"missing {key} in state_dict"


# --------------------------------------------------------------------------
# 6. Optimizer routing — adapter params go to AdamW, not Muon
# --------------------------------------------------------------------------

def test_is_think_adapter_predicate():
    """_is_think_adapter should match exactly the adapter param names."""
    yes = [
        "blocks.0.think_adapter.fc1.weight",
        "blocks.7.think_adapter.fc2.bias",
        "blocks.3.think_adapter.alpha",
    ]
    no = [
        "blocks.0.attn.q_proj.weight",
        "embed.weight",
        "lm_head.weight",
        "sparse_feedback.0.alpha",
        "pkm_layer.values.0.weight",
        "memory.W_proj.weight",
    ]
    for n in yes:
        assert _is_think_adapter(n), f"should match adapter: {n!r}"
    for n in no:
        assert not _is_think_adapter(n), f"should NOT match adapter: {n!r}"


def test_adapter_params_routed_to_adamw_with_muon():
    """build_optimizer(optimizer='muon'): adapter params must NOT land in
    the Muon optimizer; they must be in one of the AdamW param groups.
    """
    model = _make_model(n_layers=2, d_model=16, n_heads=2, d_head=8,
                        use_think_adapter=True)
    # Collect the adapter param tensors (we'll check they're in AdamW).
    adapter_param_ids = {
        id(p) for name, p in model.named_parameters()
        if _is_think_adapter(name)
    }
    assert len(adapter_param_ids) > 0, "expected some adapter params"

    opts, _ = build_optimizer(
        model, optimizer="muon", lr=1e-3, lr_muon=1e-2,
        alpha_wd=0.0, steps=10, wd=0.01,
        lr_schedule="cosine", warmup_steps=0, decay_frac=0.15,
        verbose=False,
    )
    assert len(opts) == 2, "muon mode returns [Muon, AdamW]"
    muon_opt, adamw_opt = opts[0], opts[1]
    # Muon's param ids — none should be an adapter param.
    muon_param_ids = {
        id(p) for group in muon_opt.param_groups for p in group["params"]
    }
    overlap_muon = adapter_param_ids & muon_param_ids
    assert not overlap_muon, (
        f"adapter params landed in Muon: {len(overlap_muon)} found "
        f"(should all be in AdamW)"
    )
    # AdamW's param ids — every adapter param should be there.
    adamw_param_ids = {
        id(p) for group in adamw_opt.param_groups for p in group["params"]
    }
    missing = adapter_param_ids - adamw_param_ids
    assert not missing, (
        f"adapter params missing from AdamW: {len(missing)} not routed"
    )


def test_adapter_params_routed_to_adamw_with_pure_adamw():
    """build_optimizer(optimizer='adamw'): adapter params land in the
    'regular' AdamW group (not the alpha-WD-0 group or PKM-value group).
    """
    model = _make_model(n_layers=2, d_model=16, n_heads=2, d_head=8,
                        use_think_adapter=True)
    adapter_param_ids = {
        id(p) for name, p in model.named_parameters()
        if _is_think_adapter(name)
    }
    opts, _ = build_optimizer(
        model, optimizer="adamw", lr=1e-3, lr_muon=1e-2,
        alpha_wd=0.0, steps=10, wd=0.01,
        lr_schedule="cosine", warmup_steps=0, decay_frac=0.15,
        verbose=False,
    )
    assert len(opts) == 1
    adamw_opt = opts[0]
    seen_ids = {
        id(p) for group in adamw_opt.param_groups for p in group["params"]
    }
    missing = adapter_param_ids - seen_ids
    assert not missing, (
        f"adapter params missing from AdamW (adamw-only mode): "
        f"{len(missing)} not routed"
    )


# --------------------------------------------------------------------------
# 11. Regression: adapter fires during forward_step (caught by v5 review)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_adapter_fires_during_incremental_decode():
    """The state-passing incremental-decode path (`_step_block` invoked
    from `prefill` and `forward_step`) used to BYPASS Block.forward
    entirely — meaning the adapter was silently inert at inference
    even for ckpts trained with it. Regression: with alpha != 0, the
    incremental decode output at a think position must DIFFER from
    an identical decode with alpha temporarily set to 0.
    """
    m = _make_model(use_think_adapter=True, seed=0).cuda().eval()
    _set_thinking_id(m, THINK_ID)
    with torch.no_grad():
        for blk in m.blocks:
            blk.think_adapter.alpha.fill_(0.5)

    # Build a short prompt with a think token at position 4.
    ids = torch.tensor([[2, 3, 4, 5, THINK_ID, 6]], device="cuda")
    cache_on, _prefill_logits_on = m.prefill(ids)
    logits_on, cache_on = m.forward_step(
        torch.tensor([[THINK_ID]], device="cuda"), cache_on)

    # Now zero alpha and rerun.
    with torch.no_grad():
        for blk in m.blocks:
            blk.think_adapter.alpha.fill_(0.0)
    cache_off, _prefill_logits_off = m.prefill(ids)
    logits_off, cache_off = m.forward_step(
        torch.tensor([[THINK_ID]], device="cuda"), cache_off)

    # If the adapter fired during prefill AND/OR forward_step, the
    # logits must differ. The previous bug had the adapter silently
    # ignored in BOTH paths so logits matched.
    assert not torch.allclose(logits_on, logits_off, atol=1e-5), (
        "Adapter must fire during incremental decode — got identical "
        "logits at a think position with alpha=0.5 vs alpha=0, which "
        "is the symptom of the _step_block bypass bug."
    )
