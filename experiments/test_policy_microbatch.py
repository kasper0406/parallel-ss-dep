"""Microbatched policy update == full-batch policy update (gradient equivalence).

`train_rl_grader.py` can now microbatch the policy update over rollout ROWS
(`--policy_micro_chunk N`): forward+backward each chunk of N rollouts,
ACCUMULATING gradients (no zero between chunks), then a single opt.step().
Each chunk's loss is weighted by chunk_rows/total_rows.

Because the batched surrogate + KL are MEANS OVER ROLLOUTS
(`surr_loss = torch.stack(all_surrs).mean()`), the weighted-accumulation
gradient must equal the full-batch gradient EXACTLY (up to fp error). These
tests pin that — the whole point of the feature is "same gradient, smaller
peak memory."
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.train_rl_grader import policy_loss_for_rollouts_batched, Rollout
from experiments.test_rl_kl_gather import HiddenStubLM


THINK_ID = 9
TEMP = 1.0


def _make_rollouts(model):
    """Four rollouts of differing lengths/advantages so padding + per-row
    gather + chunk boundaries are all exercised."""
    rollouts = [
        Rollout(prompt_len=2, emit_token_ids=[5, 7], emit_log_probs=[0.0, 0.0],
                emit_positions=[2, 4], full_ids=[3, 4, 5, 9, 7], depth=1, text=""),
        Rollout(prompt_len=3, emit_token_ids=[6, 8, 4],
                emit_log_probs=[0.0, 0.0, 0.0], emit_positions=[3, 4, 6],
                full_ids=[2, 3, 4, 6, 8, 9, 4], depth=1, text=""),
        Rollout(prompt_len=1, emit_token_ids=[1, 2, 3, 5],
                emit_log_probs=[0.0, 0.0, 0.0, 0.0], emit_positions=[1, 2, 3, 4],
                full_ids=[7, 1, 2, 3, 5], depth=0, text=""),
        Rollout(prompt_len=2, emit_token_ids=[8],
                emit_log_probs=[0.0], emit_positions=[2],
                full_ids=[4, 6, 8], depth=0, text=""),
    ]
    # Fill emit_log_probs from the model so the old/new PPO ratio != 1.
    for r in rollouts:
        ids = torch.tensor(r.full_ids).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).float()
        lps = []
        for pos, tok in zip(r.emit_positions, r.emit_token_ids):
            pl = logits[0, pos - 1].clone()
            pl[THINK_ID] = -float("inf")
            lps.append(torch.log_softmax(pl, -1)[tok].item())
        r.emit_log_probs = lps
    return rollouts


def _grad_snapshot(model):
    return {n: (p.grad.clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


def _full_grad(model, rollouts, advantages, **kw):
    model.zero_grad(set_to_none=True)
    loss, _, _, _ = policy_loss_for_rollouts_batched(
        rollouts, advantages, model, clip_eps=0.2, thinking_token_id=THINK_ID,
        temperature=TEMP, pad_id=0, **kw)
    loss.backward()
    return float(loss.item()), _grad_snapshot(model)


def _micro_grad(model, rollouts, advantages, chunk, **kw):
    """Replicate the trainer's microbatch loop: weighted-accumulate."""
    model.zero_grad(set_to_none=True)
    n_total = len(rollouts)
    loss_val = 0.0
    for c0 in range(0, n_total, chunk):
        sr = rollouts[c0:c0 + chunk]
        sa = advantages[c0:c0 + chunk]
        w = len(sr) / n_total
        loss_c, _, _, _ = policy_loss_for_rollouts_batched(
            sr, sa, model, clip_eps=0.2, thinking_token_id=THINK_ID,
            temperature=TEMP, pad_id=0, **kw)
        (loss_c * w).backward()
        loss_val += float(loss_c.item()) * w
    return loss_val, _grad_snapshot(model)


def _assert_grads_close(g_full, g_micro, atol=1e-5, rtol=1e-4):
    for n in g_full:
        a, b = g_full[n], g_micro[n]
        if a is None and b is None:
            continue
        assert a is not None and b is not None, f"grad presence mismatch for {n}"
        assert torch.allclose(a, b, atol=atol, rtol=rtol), (
            f"grad mismatch for {n}: max|Δ|={ (a - b).abs().max().item():.2e}")


def test_microbatch_grad_equals_full_surrogate_only():
    """kl=0, no stochastic gate -> pure surrogate (mean over rollouts).
    Weighted accumulation must reproduce the full-batch gradient exactly."""
    model = HiddenStubLM(vocab_size=12, d_model=8, seed=1)
    rollouts = _make_rollouts(model)
    advantages = [1.5, -0.5, 0.8, -1.2]

    lf, gf = _full_grad(model, rollouts, advantages)
    for chunk in (1, 2, 3):
        lm, gm = _micro_grad(model, rollouts, advantages, chunk)
        assert abs(lf - lm) < 1e-5, f"loss mismatch chunk={chunk}: {lf} vs {lm}"
        _assert_grads_close(gf, gm)


def test_microbatch_grad_equals_full_with_kl():
    """With a KL term (also a mean over rollouts) the equivalence still holds."""
    model = HiddenStubLM(vocab_size=12, d_model=8, seed=2)
    ref = HiddenStubLM(vocab_size=12, d_model=8, seed=3).eval()
    rollouts = _make_rollouts(model)
    advantages = [0.9, -0.4, 1.1, -0.7]
    kw = dict(ref_model=ref, kl_coef=0.05)

    lf, gf = _full_grad(model, rollouts, advantages, **kw)
    for chunk in (1, 2, 3):
        lm, gm = _micro_grad(model, rollouts, advantages, chunk, **kw)
        assert abs(lf - lm) < 1e-5, f"loss mismatch chunk={chunk}: {lf} vs {lm}"
        _assert_grads_close(gf, gm)


def _full_logits_surrogate_grad(model, rollouts, advantages, *, clip_eps=0.2):
    """Reference surrogate that materializes the FULL (R,T,V) logits and slices
    at emit positions — the pre-gather behavior. Loss + grad must match the
    gather path (lm_head only at emit positions) exactly."""
    model.zero_grad(set_to_none=True)
    R = len(rollouts)
    T_max = max(len(r.full_ids) for r in rollouts)
    padded = torch.zeros((R, T_max), dtype=torch.long)
    for i, r in enumerate(rollouts):
        padded[i, :len(r.full_ids)] = torch.tensor(r.full_ids)
    logits = model(padded)                              # (R,T,V) FULL
    all_surrs = []
    for i, r in enumerate(rollouts):
        if not r.emit_token_ids:
            continue
        adv = float(advantages[i])
        pos = torch.tensor(r.emit_positions)
        pred = logits[i, pos - 1, :].float().clone()
        pred[:, THINK_ID] = -float("inf")
        nlp = F.log_softmax(pred / max(TEMP, 1e-8), dim=-1)
        tok = torch.tensor(r.emit_token_ids)
        new_lp = nlp.gather(1, tok.unsqueeze(1)).squeeze(1)
        old_lp = torch.tensor(r.emit_log_probs)
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        all_surrs.append(-torch.minimum(ratio * adv, clipped * adv).mean())
    loss = torch.stack(all_surrs).mean()
    loss.backward()
    return float(loss.item()), _grad_snapshot(model)


def test_gather_lmhead_equals_full_logits():
    """The skip_lm_head + gather path (lm_head only at emit positions) must
    produce the IDENTICAL surrogate loss + gradient as materializing the full
    (R,T,V) logits and slicing. This is the T-axis memory win's correctness
    guarantee: lm_head(h[i,p]) == full_logits[i,p] by construction."""
    model = HiddenStubLM(vocab_size=12, d_model=8, seed=7)
    rollouts = _make_rollouts(model)
    advantages = [1.3, -0.6, 0.4, -0.9]

    lf, gf = _full_logits_surrogate_grad(model, rollouts, advantages)
    lg, gg = _full_grad(model, rollouts, advantages)   # uses gather internally
    assert abs(lf - lg) < 1e-5, f"loss mismatch: full={lf} gather={lg}"
    _assert_grads_close(gf, gg)


def test_full_chunk_is_identity():
    """chunk >= n_total must be byte-identical to the one-shot full path."""
    model = HiddenStubLM(vocab_size=12, d_model=8, seed=4)
    rollouts = _make_rollouts(model)
    advantages = [1.0, -1.0, 0.5, -0.5]
    lf, gf = _full_grad(model, rollouts, advantages)
    lm, gm = _micro_grad(model, rollouts, advantages, chunk=99)
    assert abs(lf - lm) < 1e-7
    _assert_grads_close(gf, gm, atol=1e-7, rtol=1e-6)
