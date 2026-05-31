"""CPU-only tests for the grader-RL speedups (2026-05-30):

  1. **Emit-only KL gather** in `policy_loss_for_rollouts_batched`. The KL-to-
     reference term now runs the ref model with `return_hidden=True` and
     applies `lm_head` ONLY at each rollout's emit positions, instead of
     materializing the full `(R, T, V)` ref-logit tensor. This test proves the
     resulting KL (and total loss) is NUMERICALLY IDENTICAL to a reference
     implementation that materializes the full ref logits and slices — i.e.
     the optimization is correct-by-construction, only cheaper.

  2. **Gate σ-histogram bucketing** is unchanged by the vectorized
     `torch.bucketize(..., right=True)` rewrite of the per-token host loop —
     a boundary-value equivalence test against the old strict-`<` Python
     bucketing.

Run ONLY this file (the full suite has CUDA tests that would OOM a co-resident
training run):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \\
        experiments/test_rl_kl_gather.py -v
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.train_rl_grader import (
    Rollout,
    policy_loss_for_rollouts_batched,
)


# ---------------------------------------------------------------------------
# A stub model exposing the TinyLM surface the gather path needs:
#   - forward(ids, return_hidden=False) -> logits OR (logits, hidden)
#   - .lm_head (so lm_head(hidden) == logits)
#   - ._last_gate stash
# The hidden state is a per-(row,pos) learned-but-deterministic vector so that
# lm_head(hidden) reproduces logits exactly, AND the hidden depends on the
# token content so a wrong gather index would change the KL.
# ---------------------------------------------------------------------------

class HiddenStubLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 8, *, seed: int = 0,
                 gate_value: float = 0.7):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self._gate_value = gate_value
        g = torch.Generator().manual_seed(seed)
        # Embedding-like content map: each token id -> a d_model vector. The
        # per-position hidden is a causal-running-mean of these, so hidden at
        # position p depends on ids[:, :p+1] (any wrong gather index diverges).
        self.content = nn.Embedding(vocab_size, d_model)
        with torch.no_grad():
            self.content.weight.copy_(torch.randn(vocab_size, d_model, generator=g))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.randn(vocab_size, d_model, generator=g) * 0.3)
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, ids: torch.Tensor, return_hidden: bool = False,
                skip_lm_head: bool = False):
        B, T = ids.shape
        emb = self.content(ids)                       # (B, T, d)
        # Causal running mean -> hidden depends on the row's own prefix.
        csum = torch.cumsum(emb, dim=1)
        denom = torch.arange(1, T + 1, device=ids.device).float().view(1, T, 1)
        hidden = csum / denom                          # (B, T, d)
        g = torch.full((B, T), float(self._gate_value), device=ids.device)
        self._last_gate = g
        self._last_gate_logits = torch.logit(g.clamp(1e-6, 1 - 1e-6))
        if skip_lm_head:
            # Mirror TinyLM: return pre-lm_head hidden, gate stash still set.
            return hidden
        logits = self.lm_head(hidden)                  # (B, T, V)
        if return_hidden:
            return logits, hidden
        return logits


def _reference_kl_full_logits(rollouts, advantages, model, ref_model, *,
                              clip_eps, thinking_token_id, temperature, kl_coef):
    """Independent KL reference: materialize FULL ref logits and slice at emit
    positions (the OLD implementation's behavior). Returns mean KL only."""
    device = next(model.parameters()).device
    R = len(rollouts)
    T_max = max(len(r.full_ids) for r in rollouts)
    padded = torch.full((R, T_max), 0, dtype=torch.long, device=device)
    for i, r in enumerate(rollouts):
        padded[i, :len(r.full_ids)] = torch.tensor(r.full_ids, device=device)
    with torch.no_grad():
        logits = model(padded).float()
        ref_logits = ref_model(padded).float()
    all_kls = []
    for i, r in enumerate(rollouts):
        if not r.emit_token_ids:
            continue
        positions = torch.tensor(r.emit_positions, device=device)
        pred_idx = positions - 1
        pred_logits = logits[i, pred_idx, :].clone()
        pred_logits[:, thinking_token_id] = -float("inf")
        new_lp = F.log_softmax(pred_logits / max(temperature, 1e-8), dim=-1)
        tok_ids = torch.tensor(r.emit_token_ids, device=device)
        new_lp = new_lp.gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        ref_pred = ref_logits[i, pred_idx, :].clone()
        ref_pred[:, thinking_token_id] = -float("inf")
        ref_lp = F.log_softmax(ref_pred / max(temperature, 1e-8), dim=-1)
        ref_lp = ref_lp.gather(1, tok_ids.unsqueeze(1)).squeeze(1)
        log_r = ref_lp - new_lp
        all_kls.append(((torch.exp(log_r) - 1.0) - log_r).mean())
    return torch.stack(all_kls).mean()


def _make_rollouts(model, thinking_id, temperature):
    """Two rollouts of differing lengths so padding + per-row gather are
    exercised. emit_log_probs filled from a throwaway forward."""
    rollouts = [
        Rollout(prompt_len=2, emit_token_ids=[5, 7], emit_log_probs=[0.0, 0.0],
                emit_positions=[2, 4],
                full_ids=[3, 4, 5, 9, 7], depth=1, text=""),
        Rollout(prompt_len=3, emit_token_ids=[6, 8, 4],
                emit_log_probs=[0.0, 0.0, 0.0],
                emit_positions=[3, 4, 6],
                full_ids=[2, 3, 4, 6, 8, 9, 4], depth=1, text=""),
    ]
    # Fill emit_log_probs from the model so old/new ratio is well-defined.
    for r in rollouts:
        ids = torch.tensor(r.full_ids).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).float()
        lps = []
        for pos, tok in zip(r.emit_positions, r.emit_token_ids):
            pl = logits[0, pos - 1].clone()
            pl[thinking_id] = -float("inf")
            lps.append(float(F.log_softmax(pl / temperature, dim=-1)[tok].item()))
        r.emit_log_probs = lps
    return rollouts


def test_emit_only_gather_kl_matches_full_logits_reference():
    """The emit-only-gather KL must EQUAL a full-ref-logits-slice reference."""
    vocab, thinking_id, temp = 16, 9, 1.0
    model = HiddenStubLM(vocab, seed=1, gate_value=0.7)
    ref_model = HiddenStubLM(vocab, seed=2, gate_value=0.7).eval()
    rollouts = _make_rollouts(model, thinking_id, temp)
    advantages = [1.0, -0.5]

    _, _, kl_gather, _ = policy_loss_for_rollouts_batched(
        rollouts, advantages, model,
        clip_eps=0.2, thinking_token_id=thinking_id, temperature=temp,
        pad_id=0, ref_model=ref_model, kl_coef=0.05)

    kl_ref = _reference_kl_full_logits(
        rollouts, advantages, model, ref_model,
        clip_eps=0.2, thinking_token_id=thinking_id, temperature=temp,
        kl_coef=0.05)

    assert torch.allclose(kl_gather, kl_ref, atol=1e-6), (kl_gather, kl_ref)
    # And the KL is a real (non-trivial) positive value here.
    assert float(kl_gather.item()) > 1e-4


def test_kl_zero_when_ref_equals_policy():
    """If ref_model IS the policy (identical params), KL must be ~0."""
    vocab, thinking_id, temp = 16, 9, 1.0
    model = HiddenStubLM(vocab, seed=5, gate_value=0.6)
    rollouts = _make_rollouts(model, thinking_id, temp)
    _, _, kl, _ = policy_loss_for_rollouts_batched(
        rollouts, [1.0, 1.0], model,
        clip_eps=0.2, thinking_token_id=thinking_id, temperature=temp,
        pad_id=0, ref_model=model, kl_coef=0.05)
    assert abs(float(kl.item())) < 1e-5, kl


def test_no_ref_model_gives_zero_kl():
    vocab, thinking_id, temp = 16, 9, 1.0
    model = HiddenStubLM(vocab, seed=3)
    rollouts = _make_rollouts(model, thinking_id, temp)
    _, _, kl, _ = policy_loss_for_rollouts_batched(
        rollouts, [1.0, 1.0], model,
        clip_eps=0.2, thinking_token_id=thinking_id, temperature=temp,
        pad_id=0, ref_model=None, kl_coef=0.0)
    assert float(kl.item()) == 0.0


def test_kl_loss_increases_total_loss_via_coef():
    """Total loss with kl_coef>0 differs from kl_coef=0 by exactly coef*KL."""
    vocab, thinking_id, temp = 16, 9, 1.0
    model = HiddenStubLM(vocab, seed=7, gate_value=0.7)
    ref_model = HiddenStubLM(vocab, seed=8, gate_value=0.7).eval()
    rollouts = _make_rollouts(model, thinking_id, temp)
    adv = [1.0, -0.5]
    loss0, _, _, _ = policy_loss_for_rollouts_batched(
        rollouts, adv, model, clip_eps=0.2, thinking_token_id=thinking_id,
        temperature=temp, pad_id=0, ref_model=ref_model, kl_coef=0.0)
    coef = 0.05
    loss1, _, kl1, _ = policy_loss_for_rollouts_batched(
        rollouts, adv, model, clip_eps=0.2, thinking_token_id=thinking_id,
        temperature=temp, pad_id=0, ref_model=ref_model, kl_coef=coef)
    assert torch.allclose(loss1 - loss0, coef * kl1, atol=1e-6)


# ---------------------------------------------------------------------------
# Gate σ-histogram bucketing: torch.bucketize(right=True) == old strict-`<`.
# ---------------------------------------------------------------------------

def _old_bucket(x: float) -> int:
    if x < 0.1:
        return 0
    elif x < 0.5:
        return 1
    elif x < 0.9:
        return 2
    return 3


def test_sigma_bucket_float64_matches_old_per_row_item():
    """The vectorized float64 σ-bucket sum reproduces the OLD per-row
    `float(p_emit[i].item()) < 0.x` cascade EXACTLY — including at the
    float32 boundary values where the input came from a float32 gate."""
    xs = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.49, 0.5, 0.7, 0.89, 0.9,
                       0.95, 1.0, 1e-6, 1 - 1e-6], dtype=torch.float32)
    pe64 = xs.double()
    got = ((pe64 >= 0.1).long() + (pe64 >= 0.5).long()
           + (pe64 >= 0.9).long()).tolist()
    # Old code did exactly: float(p_emit[i].item()) compared to python doubles.
    want = [_old_bucket(float(x)) for x in xs]
    assert got == want, (got, want)
