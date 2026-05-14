"""Tests for the v4 pretrain knobs: gradient accumulation, z-loss, grad-clip.

- Gradient accumulation: accumulating N microbatches of size B/N must give
  the same parameter gradients as a single batch of size B (the LM loss is a
  token-mean, so (1/N)·Σ mean_i == mean over all tokens).
- z-loss: _z_loss_term is mean(logsumexp(logits)^2) scaled by the weight, and
  is exactly zero when the weight is zero.
- The three new CLI flags parse with the documented defaults.

CPU-only: uses a tiny mock LM so we don't need the DeltaNet CUDA kernels.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_pretrain_knobs.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.train_lm import _nonthink_forward_loss, _z_loss_term
from experiments.train_lm_args import build_parser


class _MockLM(torch.nn.Module):
    """Minimal stand-in for TinyLM: embed -> linear head, plus a gate head.

    `_nonthink_forward_loss` only needs `model(x)` to return logits and, when
    `output_gate` is on, `model._last_gate` to hold per-position gate values.
    """

    def __init__(self, vocab: int = 32, d: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab)
        self.gate = torch.nn.Linear(d, 1)
        self._last_gate = None

    def forward(self, x, return_aux: bool = False, doc_ids=None):
        h = self.emb(x)
        self._last_gate = torch.sigmoid(self.gate(h).squeeze(-1))  # (B, T)
        return self.head(h)


def _args(output_gate: bool) -> SimpleNamespace:
    return SimpleNamespace(
        aux_brackets=False,
        aux_max_depth=4,
        output_gate=output_gate,
        gate_warmup_steps=0,
        gate_floor_min=0.5,
        gate_lambda=2.0,
        aux_weight=0.1,
        z_loss=0.0,
    )


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def _run_grad_accum_equivalence(output_gate: bool):
    torch.manual_seed(0)
    vocab, d, B, T = 32, 16, 4, 12
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    args = _args(output_gate)

    # Single full batch.
    m_full = _MockLM(vocab, d)
    m_full.zero_grad(set_to_none=True)
    _, _, lm_loss, aux_loss = _nonthink_forward_loss(m_full, x, y, args, 0, None)
    (lm_loss + args.aux_weight * aux_loss).backward()
    g_full = _grads(m_full)

    # Same data, accumulated over B microbatches of size 1.
    m_accum = _MockLM(vocab, d)
    m_accum.load_state_dict(m_full.state_dict())
    m_accum.zero_grad(set_to_none=True)
    n_micro = B
    for i in range(n_micro):
        _, _, lm_loss_i, aux_loss_i = _nonthink_forward_loss(
            m_accum, x[i:i + 1], y[i:i + 1], args, 0, None)
        ((lm_loss_i + args.aux_weight * aux_loss_i) / n_micro).backward()
    g_accum = _grads(m_accum)

    assert g_full.keys() == g_accum.keys()
    for name in g_full:
        torch.testing.assert_close(
            g_full[name], g_accum[name], rtol=1e-4, atol=1e-5,
            msg=f"grad mismatch on {name} (output_gate={output_gate})",
        )


def test_grad_accum_equivalence_plain():
    _run_grad_accum_equivalence(output_gate=False)


def test_grad_accum_equivalence_gated():
    _run_grad_accum_equivalence(output_gate=True)


def test_z_loss_zero_when_off():
    logits = torch.randn(2, 5, 7)
    assert float(_z_loss_term(logits, 0.0)) == 0.0
    assert _z_loss_term(logits, 0.0).shape == ()


def test_z_loss_value():
    logits = torch.randn(2, 5, 7)
    expected = 1e-4 * (torch.logsumexp(logits, dim=-1) ** 2).mean()
    torch.testing.assert_close(_z_loss_term(logits, 1e-4), expected)
    assert float(_z_loss_term(logits, 1e-4)) > 0.0


def test_z_loss_shrinks_logit_magnitude():
    """One gradient step on pure z-loss must reduce ||logits||."""
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 10, requires_grad=True)
    before = logits.detach().norm().item()
    _z_loss_term(logits, 1.0).backward()
    with torch.no_grad():
        stepped = logits - 0.1 * logits.grad
    assert stepped.norm().item() < before


def test_new_flags_defaults():
    a = build_parser().parse_args([])
    assert a.grad_accum == 1
    assert a.grad_clip == 1.0
    assert a.z_loss == 1e-4
    assert a.wd == 0.01


def test_new_flags_parse():
    a = build_parser().parse_args(
        ["--grad_accum", "8", "--grad_clip", "0", "--z_loss", "1e-4"])
    assert a.grad_accum == 8
    assert a.grad_clip == 0.0
    assert a.z_loss == 1e-4
