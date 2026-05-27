"""Tests for the gate-calibration auxiliary loss.

Spec recap:
  At sampled positions where σ_t ∈ [min, max]:
    1. compute log p_no_think from existing main-forward logits
    2. run an extra forward with K inserted THINK tokens, compute
       log p_think
    3. target_t = 1 if log p_think > log p_no_think else 0
    4. loss term = BCE(σ_t, target_t)

  Symmetric: punishes "gate fired but thinking didn't help" AND "gate
  didn't fire but thinking would have helped".

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_gate_calibration.py -v
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.process_reward import (
    compute_gate_calibration_loss,
    _select_candidate_positions_window,
)


THINK_ID = 7
PAD_ID = 0


class _MockLM(nn.Module):
    """Tiny stand-in: same surface as the process_reward _MockLM but
    also exposes a `forward_mode` toggle that lets us *forcibly* make
    thinking help (`'think_helps'`) or hurt (`'think_hurts'`) the
    target-token log-prob.

    Mechanism: in the 'think_helps' mode the second forward (which sees
    K think tokens at the right) returns logits that put a large bias
    on the target id 5; the main forward does not. In 'think_hurts' it
    flips the sign.
    """
    def __init__(self, vocab: int = 16, d: int = 8,
                 forward_mode: str = "neutral"):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.gate_head = nn.Linear(d, 1)
        self._last_gate = None
        self._last_gate_logits = None
        self.forward_mode = forward_mode

    def forward(self, x, inputs_embeds=None, return_hidden=False):
        h = inputs_embeds if inputs_embeds is not None else self.embed(x)
        # Cheap trunk: nonlinearity so gradients flow.
        h = h + h.tanh() * 0.01
        gl = self.gate_head(h).squeeze(-1)
        self._last_gate_logits = gl
        self._last_gate = torch.sigmoid(gl)
        logits = self.head(h)
        # Detect "this is the think-forward" by checking if the LAST
        # token in the sequence is THINK_ID. Mock-only assumption — the
        # gate-calibration helper always constructs the think-forward
        # with K think tokens at the right.
        if x.shape[1] >= 1:
            is_think_forward = (x[:, -1] == THINK_ID)
        else:
            is_think_forward = torch.zeros(
                x.shape[0], dtype=torch.bool, device=x.device)
        if self.forward_mode == "think_helps":
            bias = torch.zeros_like(logits)
            bias[..., 5] = 10.0
            logits = logits + is_think_forward.float().view(-1, 1, 1) * bias
        elif self.forward_mode == "think_hurts":
            bias = torch.zeros_like(logits)
            bias[..., 5] = -10.0
            logits = logits + is_think_forward.float().view(-1, 1, 1) * bias
        if return_hidden:
            return logits, h
        return logits


# ---------- 1. Default off — helper returns zero on no-candidates ------

def test_default_off_zero_loss_no_candidates():
    """When the gate σ is outside the [min, max] window everywhere, the
    helper returns a non-grad scalar zero and does nothing."""
    torch.manual_seed(0)
    model = _MockLM()
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    # All σ saturated to 1.0 (outside max=0.9 window).
    gate = torch.full((B, T), 0.99)
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_gate_calibration_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None)
    assert loss.item() == 0.0
    assert not loss.requires_grad
    assert stats.n_sampled == 0


# ---------- 2. Pad collision raises -----------------------------------

def test_pad_eq_think_raises():
    import pytest
    torch.manual_seed(0)
    model = _MockLM()
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = x.clone()
    main_logits = model(x)
    gate = torch.full((B, T), 0.5)
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="pad_token_id must NOT equal"):
        compute_gate_calibration_loss(
            model, x, y, gate=gate, main_logits=main_logits,
            thinking_token_id=THINK_ID,
            K=2, apply_min_sigma=0.0, apply_max_sigma=1.0,
            sample_frac=1.0, rng=rng,
            pad_token_id=THINK_ID,  # collision
            retrieval_as_input=False, base_vocab_for_loss=None)


def test_max_less_than_min_raises():
    import pytest
    torch.manual_seed(0)
    model = _MockLM()
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = x.clone()
    main_logits = model(x)
    gate = torch.full((B, T), 0.5)
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="apply_max_sigma"):
        compute_gate_calibration_loss(
            model, x, y, gate=gate, main_logits=main_logits,
            thinking_token_id=THINK_ID,
            K=2, apply_min_sigma=0.9, apply_max_sigma=0.1,
            sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
            retrieval_as_input=False, base_vocab_for_loss=None)


# ---------- 3. Window selector --------------------------------------

def test_selector_respects_window():
    B, T = 2, 8
    gate = torch.full((B, T), 0.5)
    gate[0, 0] = 0.05    # below min
    gate[0, 1] = 0.95    # above max
    y_shift = torch.zeros(B, T - 1, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)
    b_idx, t_idx = _select_candidate_positions_window(
        gate, y_shift, apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=1.0, rng=rng, max_positions=128)
    # Excluded: (0, 0) and (0, 1). Total = B*(T-1) - 2 = 12.
    assert b_idx.numel() == B * (T - 1) - 2


def test_selector_respects_max_positions():
    B, T = 8, 16
    gate = torch.full((B, T), 0.5)
    y_shift = torch.zeros(B, T - 1, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)
    b_idx, _ = _select_candidate_positions_window(
        gate, y_shift, apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=1.0, rng=rng, max_positions=10)
    assert b_idx.numel() == 10


# ---------- 4. target_frac_one plausible -----------------------------

def test_target_frac_one_in_range():
    torch.manual_seed(0)
    model = _MockLM()
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    gate = torch.full((B, T), 0.5)  # all in [0.1, 0.9]
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_gate_calibration_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        max_positions=64)
    assert stats.n_sampled > 0
    assert 0.0 <= stats.target_frac_one <= 1.0


# ---------- 5. Loss is scalar with grad on the gate -----------------

def test_loss_scalar_grad_on_gate():
    torch.manual_seed(0)
    model = _MockLM()
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    gate = model._last_gate                   # ← grad-enabled tensor
    gate_logits = model._last_gate_logits
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_gate_calibration_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        gate_logits=gate_logits, max_positions=64)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert stats.n_sampled > 0
    loss.backward()
    # gate_head must receive non-trivial gradient.
    g = model.gate_head.weight.grad
    assert g is not None
    assert g.abs().sum() > 0


# ---------- 6. Forced target=1 → gate σ rises after a step ---------

def _run_one_step(forward_mode: str, n_steps: int = 5):
    """Train one mock-LM for n_steps with ONLY the gate-calibration
    loss and report change in mean σ on a fixed batch.
    """
    torch.manual_seed(0)
    model = _MockLM(forward_mode=forward_mode)
    # Set up: target label is 5 at every position. think_helps mode
    # makes the K-think forward put 10.0 logit bias on id 5; think_hurts
    # mode flips that to -10.0. With a hard {0,1} target this gives a
    # uniform target of 1 (or 0) across all sampled positions.
    B, T = 4, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.full((B, T), 5, dtype=torch.long)

    # Initialise gate_head bias so σ starts in the middle of the window.
    with torch.no_grad():
        model.gate_head.weight.zero_()
        model.gate_head.bias.fill_(0.0)  # σ(0) = 0.5

    opt = torch.optim.SGD(
        [model.gate_head.weight, model.gate_head.bias], lr=1.0)

    sigma_before = None
    sigma_after = None
    for step in range(n_steps):
        rng = torch.Generator().manual_seed(step)
        main_logits = model(x)
        gate = model._last_gate
        gate_logit = model._last_gate_logits
        loss, stats = compute_gate_calibration_loss(
            model, x, y, gate=gate, main_logits=main_logits,
            thinking_token_id=THINK_ID,
            K=2, apply_min_sigma=0.0, apply_max_sigma=1.0,
            sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
            retrieval_as_input=False, base_vocab_for_loss=None,
            gate_logits=gate_logit, max_positions=128)
        if step == 0:
            sigma_before = float(gate.detach().mean().item())
            assert stats.n_sampled > 0
            if forward_mode == "think_helps":
                assert stats.target_frac_one > 0.9
            elif forward_mode == "think_hurts":
                assert stats.target_frac_one < 0.1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _ = model(x)
        sigma_after = float(model._last_gate.mean().item())
    return sigma_before, sigma_after


def test_forced_target_one_pushes_sigma_up():
    """If thinking ALWAYS helps, BCE pushes σ → 1 (target=1)."""
    s_before, s_after = _run_one_step("think_helps", n_steps=5)
    assert s_before is not None and s_after is not None
    assert s_after > s_before, (
        f"expected σ to rise; got {s_before:.3f} → {s_after:.3f}")


def test_forced_target_zero_pushes_sigma_down():
    """If thinking ALWAYS hurts, BCE pushes σ → 0 (target=0)."""
    s_before, s_after = _run_one_step("think_hurts", n_steps=5)
    assert s_before is not None and s_after is not None
    assert s_after < s_before, (
        f"expected σ to fall; got {s_before:.3f} → {s_after:.3f}")


# ---------- 7. SFT CLI flag wiring ----------------------------------

def test_sft_cli_flag_wired():
    """The new flag is declared with default=0.0 in sft_code.py and
    the legacy forward path guards on `use_gate_calibration`."""
    import pathlib
    src = pathlib.Path("experiments/sft_code.py").read_text()
    assert '"--gate_calibration_weight"' in src
    section = src.split('"--gate_calibration_weight"', 1)[1].split(")", 1)[0]
    assert "default=0.0" in section
    assert "use_gate_calibration = (" in src
    assert "args.gate_calibration_weight > 0.0" in src
    assert "compute_gate_calibration_loss" in src


def test_train_lm_args_flag_wired():
    """And it's also in the shared train-time argparse."""
    import pathlib
    src = pathlib.Path("experiments/train_lm_args.py").read_text()
    assert '"--gate_calibration_weight"' in src
    section = src.split('"--gate_calibration_weight"', 1)[1].split(")", 1)[0]
    assert "default=0.0" in section


# ---------- 8. Smooth-target mode produces real-valued labels --------

def test_smooth_target_scale_uses_sigmoid_target():
    torch.manual_seed(0)
    model = _MockLM(forward_mode="think_helps")
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.full((B, T), 5, dtype=torch.long)
    main_logits = model(x)
    gate = model._last_gate
    gate_logits = model._last_gate_logits
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_gate_calibration_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.0, apply_max_sigma=1.0,
        sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        gate_logits=gate_logits, max_positions=64,
        smooth_target_scale=1.0)
    # Smooth target is σ(diff*scale); with think_helps the diff is
    # uniformly positive → target ~ 0.99 (close to 1 but not 1.0).
    assert stats.n_sampled > 0
    assert 0.5 < stats.target_frac_one < 1.0


# ---------- 9. CUDA equivalence (skipped if no CUDA) ----------------

def test_cuda_runs_and_produces_grad():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    # Pin a non-zero device when present so we don't bump GPU 0 (busy
    # with training). Fall back to current device if only one is up.
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda:0")
    torch.manual_seed(0)
    model = _MockLM().to(device)
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T), device=device)
    y = torch.randint(0, 16, (B, T), device=device)
    main_logits = model(x)
    gate = model._last_gate
    gate_logits = model._last_gate_logits
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_gate_calibration_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        gate_logits=gate_logits, max_positions=64)
    assert loss.device.type == "cuda"
    assert stats.n_sampled > 0
    loss.backward()
    assert model.gate_head.weight.grad is not None


# ---------- 10. End-to-end one step is finite -----------------------

def test_end_to_end_one_step_finite():
    import torch.nn.functional as F
    torch.manual_seed(0)
    model = _MockLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    gate = model._last_gate
    gate_logits = model._last_gate_logits
    rng = torch.Generator().manual_seed(0)
    shift_logits = main_logits[:, :-1].contiguous()
    shift_labels = y[:, 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1))
    gc_loss, _ = compute_gate_calibration_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.1, apply_max_sigma=0.9,
        sample_frac=1.0, rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        gate_logits=gate_logits, max_positions=64)
    total = lm_loss + 0.1 * gc_loss
    assert torch.isfinite(total)
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()
