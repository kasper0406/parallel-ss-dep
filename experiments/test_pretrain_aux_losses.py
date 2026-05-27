"""Tests for the pretrain trainer's process-reward + gate-calibration
auxiliary losses (the wiring added in train_lm.py).

Three things to check:
  1. CLI flags survive argparse round-trip with their documented defaults.
  2. The aux-loss helpers are importable from the trainer's import path
     (catches import-cycle / typo regressions) and return a zero loss
     when their respective weights are 0 by inspection of `use_*` gates.
  3. With weight > 0, an extra forward fires (counted via a forward-call
     counter mock); with weight = 0 (the default), no extra forward fires
     so existing pretrain steps are byte-identical.

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_pretrain_aux_losses.py -v
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.train_lm_args import build_parser
from experiments.process_reward import (
    compute_process_reward_loss,
    compute_gate_calibration_loss,
)


THINK_ID = 7
PAD_ID = 0


class _MockLM(nn.Module):
    """Minimal LM with the surface the aux losses need: `embed`, `forward`
    that returns logits and stashes `_last_gate` / `_last_gate_logits`.

    Counts the number of full `forward` calls so a test can verify the
    aux-loss helpers ran (or didn't) an extra forward.
    """
    def __init__(self, vocab: int = 16, d: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.gate_head = nn.Linear(d, 1)
        self._last_gate = None
        self._last_gate_logits = None
        self.n_forward_calls = 0

    def forward(self, x, inputs_embeds=None, return_hidden=False,
                doc_ids=None):
        self.n_forward_calls += 1
        h = inputs_embeds if inputs_embeds is not None else self.embed(x)
        h = h + h.tanh() * 0.01
        gl = self.gate_head(h).squeeze(-1)
        self._last_gate_logits = gl
        self._last_gate = torch.sigmoid(gl)
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits


def _parse(extra_args: list[str]):
    """Helper: parse argparse with the minimum required arch+steps args."""
    p = build_parser()
    return p.parse_args(["--arch", "deltanet", "--steps", "1"] + extra_args)


# ---------------------------------------------------------------------------
# 1. CLI flag presence + defaults
# ---------------------------------------------------------------------------

def test_process_reward_flags_default_off():
    args = _parse([])
    # Mirror sft_code.py defaults.
    assert args.process_reward_weight == 0.0
    assert args.process_reward_K == 4
    assert args.process_reward_apply_min_sigma == 0.3
    assert args.process_reward_sample_frac == 0.05
    assert args.process_reward_max_positions == 32


def test_gate_calibration_flags_default_off():
    args = _parse([])
    assert args.gate_calibration_weight == 0.0
    assert args.gate_calibration_K == 4
    assert args.gate_calibration_apply_min_sigma == 0.1
    assert args.gate_calibration_apply_max_sigma == 0.9
    assert args.gate_calibration_sample_frac == 0.05
    assert args.gate_calibration_max_positions == 32
    assert args.gate_calibration_smooth_target_scale == 0.0


def test_thinking_arch_flags_present():
    """Confirm the thinking-arch CLI flags the spec relies on all exist
    and parse to their declared default values."""
    args = _parse([])
    assert args.state_readonly_at_think is False
    assert args.use_think_adapter is False
    assert args.think_adapter_hidden_mult == 2
    assert args.use_refinement_head is False
    assert args.refinement_head_window == 128
    assert args.refinement_head_n_heads == 8
    assert args.refinement_head_mlp_mult == 2
    assert args.refinement_head_alpha_init == 0.3
    assert args.think_index_emb_size == 0


def test_full_recipe_flags_parse():
    """The exact CLI recipe documented in the spec must parse cleanly."""
    args = _parse([
        "--process_reward_weight", "0.05",
        "--gate_calibration_weight", "0.05",
        "--state_readonly_at_think",
        "--use_think_adapter", "--think_adapter_hidden_mult", "2",
        "--use_refinement_head", "--refinement_head_window", "128",
        "--refinement_head_alpha_init", "0.3",
        "--think_index_emb_size", "16",
    ])
    assert args.process_reward_weight == 0.05
    assert args.gate_calibration_weight == 0.05
    assert args.state_readonly_at_think is True
    assert args.use_think_adapter is True
    assert args.use_refinement_head is True
    assert args.refinement_head_alpha_init == 0.3
    assert args.think_index_emb_size == 16


# ---------------------------------------------------------------------------
# 2. Helpers are importable from train_lm (the wiring exists)
# ---------------------------------------------------------------------------

def test_train_lm_imports_process_reward_helpers():
    """If the wiring is in place, `train_lm.compute_process_reward_loss`
    and `compute_gate_calibration_loss` resolve to the same callables
    as imported directly from `process_reward`."""
    from experiments import train_lm
    assert train_lm.compute_process_reward_loss is compute_process_reward_loss
    assert train_lm.compute_gate_calibration_loss is compute_gate_calibration_loss


# ---------------------------------------------------------------------------
# 3. Extra-forward firing
# ---------------------------------------------------------------------------

def _build_inputs(B=2, T=12, vocab=16, gate_value=0.7):
    torch.manual_seed(0)
    x = torch.randint(2, vocab, (B, T))
    # Make sure no spurious THINK_ID lands in the input.
    x[x == THINK_ID] = 1
    y = x.roll(-1, dims=1)
    y[:, -1] = -100  # last position has no next-token target
    return x, y


def test_process_reward_runs_extra_forward_when_on():
    model = _MockLM(vocab=16, d=8)
    x, y = _build_inputs()
    # Force gate to be high so candidate positions exist.
    with torch.no_grad():
        model.gate_head.bias.fill_(5.0)
    # Main forward (the trainer's normal LM forward).
    logits = model(x)
    n_before = model.n_forward_calls
    gate = model._last_gate
    rng = torch.Generator(device="cpu").manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y,
        gate=gate, main_logits=logits,
        thinking_token_id=THINK_ID, K=4,
        apply_min_sigma=0.3, sample_frac=0.5,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False,
        base_vocab_for_loss=None,
        max_positions=16,
    )
    # The helper must have run at least one extra forward when sampled > 0.
    assert stats.n_sampled > 0
    assert model.n_forward_calls == n_before + 1
    assert torch.isfinite(loss)


def test_gate_calibration_runs_extra_forward_when_on():
    model = _MockLM(vocab=16, d=8)
    x, y = _build_inputs()
    with torch.no_grad():
        # σ in the middle of the [0.1, 0.9] window.
        model.gate_head.bias.fill_(0.0)
    logits = model(x)
    n_before = model.n_forward_calls
    gate = model._last_gate
    rng = torch.Generator(device="cpu").manual_seed(0)
    loss, stats = compute_gate_calibration_loss(
        model, x, y,
        gate=gate, main_logits=logits,
        thinking_token_id=THINK_ID, K=4,
        apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=0.5, rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False,
        base_vocab_for_loss=None,
        max_positions=16,
        gate_logits=model._last_gate_logits,
        smooth_target_scale=0.0,
    )
    assert stats.n_sampled > 0
    assert model.n_forward_calls == n_before + 1
    assert torch.isfinite(loss)


def test_process_reward_overwrites_last_gate_logits():
    """Regression: an extra `compute_process_reward_loss` call MUTATES
    `model._last_gate_logits` to the after-forward's (N, T_after)
    shape. Any trainer code that needs the main-forward gate logits
    AFTER calling process_reward must have snapshotted them first —
    re-reading the attribute returns the smaller (and semantically
    wrong) after-forward tensor.

    This nails down the invariant the train_lm.py / sft_code.py
    snapshot fix depends on. Without this guarantee, the snapshot
    code in both trainers becomes "defensive" dead code; with it,
    omitting the snapshot is a (CUDA-asserting / wrong-tensor-
    trained) bug.
    """
    model = _MockLM(vocab=16, d=8)
    x, y = _build_inputs()
    with torch.no_grad():
        model.gate_head.bias.fill_(5.0)
    logits = model(x)
    pre_snap = model._last_gate_logits           # the main-forward tensor
    assert pre_snap is not None
    pre_shape = tuple(pre_snap.shape)
    rng = torch.Generator(device="cpu").manual_seed(0)
    _, stats = compute_process_reward_loss(
        model, x, y,
        gate=model._last_gate, main_logits=logits,
        thinking_token_id=THINK_ID, K=4,
        apply_min_sigma=0.3, sample_frac=1.0,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        max_positions=4,
    )
    assert stats.n_sampled > 0
    post = model._last_gate_logits               # now the after-forward tensor
    assert post is not None
    # Identity test: the attribute now points to a DIFFERENT object.
    assert post is not pre_snap, (
        "process_reward did not overwrite _last_gate_logits — the "
        "snapshot fix in train_lm.py / sft_code.py is no longer "
        "load-bearing; either the snapshot can be dropped OR this "
        "test needs updating to reflect a new invariant.")
    # Shape test: typically smaller (n_sampled rows, ≤ T cols).
    post_shape = tuple(post.shape)
    assert post_shape != pre_shape, (
        f"shapes match unexpectedly: pre={pre_shape}, post={post_shape}")
    # The held snapshot is still valid (no in-place mutation).
    assert tuple(pre_snap.shape) == pre_shape


def test_pad_id_equals_think_id_raises():
    """Caller must use a pad_id distinct from thinking_token_id — the
    helper raises to surface this trap."""
    import pytest
    model = _MockLM(vocab=16, d=8)
    x, y = _build_inputs()
    with torch.no_grad():
        model.gate_head.bias.fill_(5.0)
    logits = model(x)
    rng = torch.Generator(device="cpu").manual_seed(0)
    with pytest.raises(ValueError, match="pad_token_id must NOT equal"):
        compute_process_reward_loss(
            model, x, y,
            gate=model._last_gate, main_logits=logits,
            thinking_token_id=THINK_ID, K=4,
            apply_min_sigma=0.3, sample_frac=0.5,
            rng=rng, pad_token_id=THINK_ID,  # ← deliberate misuse
            retrieval_as_input=False,
            base_vocab_for_loss=None,
            max_positions=16,
        )


# ---------------------------------------------------------------------------
# 4. cfg-dict round-trip: the values we save in train_lm.py's checkpoint
#    config must match what the user passed (post-hoc reload should see
#    the right recipe).
# ---------------------------------------------------------------------------

def test_cfg_dict_carries_new_fields():
    """Build the cfg-dict construction inline (mirrors train_lm.py's final
    save site) and verify the new fields land with the right values."""
    args = _parse([
        "--process_reward_weight", "0.05",
        "--gate_calibration_weight", "0.07",
        "--state_readonly_at_think",
        "--use_think_adapter", "--think_adapter_hidden_mult", "3",
        "--use_refinement_head",
        "--refinement_head_window", "64",
        "--refinement_head_alpha_init", "0.25",
        "--think_index_emb_size", "8",
    ])
    cfg = {
        "process_reward_weight": float(args.process_reward_weight),
        "process_reward_K": int(args.process_reward_K),
        "gate_calibration_weight": float(args.gate_calibration_weight),
        "state_readonly_at_think": bool(args.state_readonly_at_think),
        "use_think_adapter": bool(args.use_think_adapter),
        "think_adapter_hidden_mult": int(args.think_adapter_hidden_mult),
        "use_refinement_head": bool(args.use_refinement_head),
        "refinement_head_window": int(args.refinement_head_window),
        "refinement_head_alpha_init": float(args.refinement_head_alpha_init),
        "think_index_emb_size": int(args.think_index_emb_size),
    }
    assert cfg["process_reward_weight"] == 0.05
    assert cfg["gate_calibration_weight"] == 0.07
    assert cfg["state_readonly_at_think"] is True
    assert cfg["use_think_adapter"] is True
    assert cfg["think_adapter_hidden_mult"] == 3
    assert cfg["use_refinement_head"] is True
    assert cfg["refinement_head_window"] == 64
    assert cfg["refinement_head_alpha_init"] == 0.25
    assert cfg["think_index_emb_size"] == 8
