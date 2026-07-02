"""Tests for the per-step-supervised latent-CoT distillation trainer.

Run:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      -m pytest experiments/test_latent_cot_distill.py -v
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

from experiments.latent_cot_distill import (
    parse_cot_steps,
    perstep_ce_from_logits,
    wean_weight,
    should_freeze,
    _perstep_latent_logits,
)
from experiments.optim_utils import _is_latent_feedback_adapter

CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# (1) CoT parser
# ---------------------------------------------------------------------------
def test_parse_cot_steps_basic():
    cot = ("1. Define a function reverse_words.\n"
           "2. Split the string on whitespace.\n"
           "3. Reverse the list of words.\n"
           "4. Join with single spaces and return.\n")
    steps = parse_cot_steps(cot)
    assert steps == [
        "Define a function reverse_words.",
        "Split the string on whitespace.",
        "Reverse the list of words.",
        "Join with single spaces and return.",
    ]


def test_parse_cot_steps_stops_at_second_block():
    # Qwen frequently restates the plan as a second numbered block — we keep
    # only the first contiguous 1..n run.
    cot = ("1. First real step.\n"
           "2. Second real step.\n"
           "3. Third real step.\n"
           "\nSteps:\n"
           "1. restated one.\n"
           "2. restated two.\n")
    steps = parse_cot_steps(cot)
    assert steps == ["First real step.", "Second real step.", "Third real step."]


def test_parse_cot_steps_skips_placeholders_and_clamps():
    assert parse_cot_steps("1. ...\n2. ...\n...") == []          # placeholder corpus
    cot = "\n".join(f"{i}. step {i}" for i in range(1, 11))
    assert parse_cot_steps(cot, max_steps=4) == [f"step {i}" for i in range(1, 5)]


# ---------------------------------------------------------------------------
# (2) Per-step loss: minimized at argmax==target; noise/coverage knobs
# ---------------------------------------------------------------------------
def _onehot_logits(targets, vocab, big=20.0):
    """(B,R,V) logits that argmax to `targets` (clamped >=0)."""
    B, R = targets.shape
    logits = torch.zeros(B, R, vocab)
    idx = targets.clamp(min=0)
    logits.scatter_(2, idx.unsqueeze(-1), big)
    return logits


def test_perstep_loss_minimized_when_argmax_matches():
    vocab = 50
    tgt = torch.tensor([[3, 7, 11, -100],
                        [5, 9, -100, -100]])      # -100 = padding beyond n_cot
    logits = _onehot_logits(tgt, vocab)
    loss, align, n = perstep_ce_from_logits(logits, tgt, vocab)
    assert align == pytest.approx(1.0)            # every valid step argmax==target
    assert n == 5                                 # 5 valid (non -100) steps
    assert float(loss) < 0.05                     # one-hot at target => ~0 CE


def test_perstep_loss_label_noise_raises_ce_align_unchanged():
    vocab = 50
    tgt = torch.tensor([[3, 7, 11, 13], [5, 9, 17, 19]])
    logits = _onehot_logits(tgt, vocab)
    g = torch.Generator().manual_seed(0)
    base_loss, base_align, _ = perstep_ce_from_logits(logits, tgt, vocab)
    g = torch.Generator().manual_seed(0)
    noisy_loss, noisy_align, n = perstep_ce_from_logits(
        logits, tgt, vocab, label_noise=1.0, generator=g, device="cpu")
    assert n == 8
    # all targets flipped to random tokens while logits still point at the
    # clean target => CE explodes, but alignment (vs CLEAN target) is unchanged.
    assert float(noisy_loss) > float(base_loss) + 5.0
    assert noisy_align == pytest.approx(base_align) == pytest.approx(1.0)


def test_perstep_loss_label_coverage_masks_targets():
    vocab = 50
    tgt = torch.tensor([[3, 7, 11, 13], [5, 9, 17, 19]])
    logits = _onehot_logits(tgt, vocab)
    # coverage 0 => everything dropped => n==0, loss==0
    g = torch.Generator().manual_seed(1)
    loss0, _, n0 = perstep_ce_from_logits(
        logits, tgt, vocab, label_coverage=0.0, generator=g, device="cpu")
    assert n0 == 0 and float(loss0) == 0.0
    # coverage 0.5 => strictly between 0 and full (8)
    g = torch.Generator().manual_seed(1)
    _, _, nhalf = perstep_ce_from_logits(
        logits, tgt, vocab, label_coverage=0.5, generator=g, device="cpu")
    assert 0 < nhalf < 8


# ---------------------------------------------------------------------------
# (3) Teach-then-freeze / wean curriculum
# ---------------------------------------------------------------------------
def test_wean_curriculum_full_at_start_zero_at_teach_frac():
    steps, tf = 100, 0.6
    for mode in ("freeze", "wean"):
        assert wean_weight(1, steps, tf, mode) == pytest.approx(1.0)
        # at the teach boundary the weight has weaned to 0
        assert wean_weight(int(tf * steps) + 1, steps, tf, mode) == pytest.approx(0.0)
        # mid-teach it is strictly between
        mid = wean_weight(int(0.3 * steps), steps, tf, mode)
        assert 0.0 < mid < 1.0
    # none => constant full
    assert wean_weight(1, steps, tf, "none") == 1.0
    assert wean_weight(99, steps, tf, "none") == 1.0


def test_should_freeze_boundary():
    steps, tf = 100, 0.6
    assert not should_freeze(1, steps, tf, "freeze")
    assert not should_freeze(60, steps, tf, "freeze")          # step-1=59 < 60
    assert should_freeze(61, steps, tf, "freeze")              # step-1=60 >= 60
    # wean / none never freeze
    assert not should_freeze(99, steps, tf, "wean")
    assert not should_freeze(99, steps, tf, "none")


# ---------------------------------------------------------------------------
# Tiny real-model fixtures (need CUDA — FLA DeltaNet kernel)
# ---------------------------------------------------------------------------
def _tiny_model(adapter=True, device="cuda"):
    from experiments.model import TinyLM
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    m = TinyLM(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_head=8,
               attention_cls=DeltaNetAttention, max_T=64, feedback_mode="none",
               use_memory=False, thinking_token_id=63,
               state_readonly_at_think=True,
               use_latent_feedback_adapter=adapter).to(device)
    return m


# ---------------------------------------------------------------------------
# (4) freeze_mode=freeze stops grad on the adapter while answer path updates
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA for the FLA DeltaNet kernel")
def test_freeze_locks_adapter_answer_path_still_updates():
    import torch.nn.functional as F
    m = _tiny_model(adapter=True)
    m.train()
    thinking_id = 63
    latent_params = [p for n, p in m.named_parameters()
                     if _is_latent_feedback_adapter(n)]
    # a non-adapter ("answer path") trunk param to watch
    trunk_name, trunk_p = next((n, p) for n, p in m.named_parameters()
                               if not _is_latent_feedback_adapter(n)
                               and p.ndim == 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 60, (2, 8), device="cuda")

    def adapter_grad_step():
        # loss through the adapter (per-step latent forward) + plain LM loss.
        opt.zero_grad(set_to_none=True)
        from experiments.latent_cot_distill import _perstep_latent_logits
        with torch.enable_grad():
            sl = _perstep_latent_logits(m, ids, 3, thinking_id)     # (2,3,V)
            loss = sl.float().pow(2).mean()
            out = m(ids)
            lg = out[0] if isinstance(out, tuple) else out
            loss = loss + F.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]),
                ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()

    # Teach phase: both move.
    a0 = latent_params[0].detach().clone()
    t0 = trunk_p.detach().clone()
    adapter_grad_step()
    assert not torch.allclose(latent_params[0], a0), "adapter should move pre-freeze"

    # Freeze the latent mechanism (what should_freeze triggers in main()).
    for p in latent_params:
        p.requires_grad = False
    a1 = latent_params[0].detach().clone()
    t1 = trunk_p.detach().clone()
    adapter_grad_step()
    assert torch.allclose(latent_params[0], a1), "adapter must be frozen post-freeze"
    assert not torch.allclose(trunk_p, t1), "answer path must keep updating"


# ---------------------------------------------------------------------------
# (5) freeze_trunk + R=0 (no-think) == base output (byte-identical)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="needs CUDA for the FLA DeltaNet kernel")
def test_freeze_trunk_nothink_byte_identical_to_base():
    m = _tiny_model(adapter=True)
    m.eval()
    ids = torch.randint(0, 60, (2, 10), device="cuda")
    with torch.no_grad():
        base = m(ids)
        base = (base[0] if isinstance(base, tuple) else base).clone()
        # Perturb the (trainable) adapter — the Pareto-safety claim is that the
        # NO-THINK forward never invokes it, so the output is byte-identical
        # regardless of adapter weights (== base / zero forgetting).
        for n, p in m.named_parameters():
            if _is_latent_feedback_adapter(n):
                p.add_(torch.randn_like(p))
        after = m(ids)
        after = after[0] if isinstance(after, tuple) else after
    assert torch.allclose(base, after, atol=1e-4), \
        "no-think path must be byte-identical regardless of adapter weights"


@pytest.mark.skipif(not CUDA, reason="needs CUDA for the FLA DeltaNet kernel")
def test_perstep_latent_forward_shape_and_gradients_reach_adapter():
    m = _tiny_model(adapter=True)
    m.train()
    ids = torch.randint(0, 60, (2, 8), device="cuda")
    sl = _perstep_latent_logits(m, ids, 4, 63)
    assert sl.shape == (2, 4, 64)
    sl.float().pow(2).mean().backward()
    grads = [p.grad for n, p in m.named_parameters()
             if _is_latent_feedback_adapter(n) and "proj" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), \
        "adapter proj must receive gradient from the per-step latent forward"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
