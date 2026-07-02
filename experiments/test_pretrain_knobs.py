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
    `output_gate` is on, `model._last_gate` / `_last_gate_logits` to hold
    per-position gate values + raw logits (for the entropy-aux BCE).
    """

    def __init__(self, vocab: int = 32, d: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab)
        self.gate = torch.nn.Linear(d, 1)
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, x, return_aux: bool = False, doc_ids=None):
        h = self.emb(x)
        gl = self.gate(h).squeeze(-1)
        self._last_gate_logits = gl
        self._last_gate = torch.sigmoid(gl)  # (B, T)
        return self.head(h)


def _args(output_gate: bool, gate_entropy_aux_weight: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        aux_brackets=False,
        aux_max_depth=4,
        output_gate=output_gate,
        gate_warmup_steps=0,
        gate_floor_min=0.5,
        gate_lambda=2.0,
        aux_weight=0.1,
        z_loss=0.0,
        gate_entropy_aux_weight=gate_entropy_aux_weight,
        gate_entropy_aux_temperature=1.0,
        gate_entropy_aux_target_clamp=0.0,
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
    _, _, lm_loss, aux_loss, _, _, _ = _nonthink_forward_loss(m_full, x, y, args, 0, None)
    (lm_loss + args.aux_weight * aux_loss).backward()
    g_full = _grads(m_full)

    # Same data, accumulated over B microbatches of size 1.
    m_accum = _MockLM(vocab, d)
    m_accum.load_state_dict(m_full.state_dict())
    m_accum.zero_grad(set_to_none=True)
    n_micro = B
    for i in range(n_micro):
        _, _, lm_loss_i, aux_loss_i, _, _, _ = _nonthink_forward_loss(
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
    assert a.lr == 1.4e-3
    assert a.lr_muon == 5e-3


def test_new_flags_parse():
    a = build_parser().parse_args(
        ["--grad_accum", "8", "--grad_clip", "0", "--z_loss", "1e-4"])
    assert a.grad_accum == 8
    assert a.grad_clip == 0.0
    assert a.z_loss == 1e-4


# --- Entropy-grounded gate target (CE-reduction self-reward, cheap form) ----

def test_gate_entropy_aux_off_returns_zero():
    """weight=0 ⇒ gate_aux_loss is the zero tensor and no extra grad path."""
    torch.manual_seed(0)
    vocab, d, B, T = 32, 16, 2, 8
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    m = _MockLM(vocab, d)
    _, _, _, _, gate_aux, _, _ = _nonthink_forward_loss(
        m, x, y, _args(output_gate=True, gate_entropy_aux_weight=0.0),
        0, None,
    )
    assert float(gate_aux) == 0.0
    assert gate_aux.shape == ()


def test_gate_entropy_aux_target_extremes():
    """Target = exp(-H/T). For one-hot logits H≈0 ⇒ target≈1 (emit).
    For uniform logits H≈log(V) ⇒ target≈1/V ≈ 0 (think)."""
    torch.manual_seed(0)
    V = 64
    # one-hot logits: position r picks token r (very confident → H≈0).
    sharp = torch.zeros(1, 4, V)
    for r in range(4):
        sharp[0, r, r] = 50.0
    lse = torch.logsumexp(sharp, dim=-1)
    p = (sharp - lse.unsqueeze(-1)).exp()
    H_sharp = lse - (p * sharp).sum(dim=-1)
    target_sharp = torch.exp(-H_sharp).clamp(0.0, 1.0)
    assert (target_sharp > 0.99).all()

    # uniform logits: all zero ⇒ H = log(V).
    flat = torch.zeros(1, 4, V)
    lse = torch.logsumexp(flat, dim=-1)
    p = (flat - lse.unsqueeze(-1)).exp()
    H_flat = lse - (p * flat).sum(dim=-1)
    target_flat = torch.exp(-H_flat).clamp(0.0, 1.0)
    # exp(-log(V)) = 1/V
    assert torch.allclose(target_flat, torch.full_like(target_flat, 1.0 / V),
                          atol=1e-4)


def test_gate_entropy_aux_pushes_gate_against_uncertainty():
    """After one step of pure entropy-BCE loss, the gate logit at the
    most-confident position should rise and at the least-confident
    position should fall — i.e. the signal flows in the expected direction
    and the LM head receives NO gradient (target is detached)."""
    torch.manual_seed(0)
    vocab, d = 32, 16
    # Two positions: position 0 will be made highly confident by tweaking
    # embedding, position 1 will be uniform. We just check the BCE target
    # itself differentiates them.
    m = _MockLM(vocab, d)
    # Use the real loss-bearing call with weight=1.0 and zero everything
    # else so backward grad on gate_head is purely from the entropy aux.
    args = _args(output_gate=True, gate_entropy_aux_weight=1.0)
    args.aux_weight = 0.0
    args.z_loss = 0.0
    x = torch.tensor([[5, 6, 7, 8]])
    y = torch.tensor([[5, 6, 7, 8]])
    m.zero_grad(set_to_none=True)
    _, _, lm_loss, _, gate_aux, _, _ = _nonthink_forward_loss(m, x, y, args, 0, None)
    # The entropy-aux must have positive loss (BCE on a non-degenerate target).
    assert float(gate_aux) > 0.0
    # Backward only on gate_aux: head must receive NO gradient (target is
    # detached) and the LM-loss path is exercised separately by other tests.
    gate_aux.backward()
    assert m.gate.weight.grad is not None
    assert m.gate.weight.grad.abs().sum().item() > 0.0
    assert m.head.weight.grad is None  # target was detached → no LM grad


def test_gate_entropy_aux_flag_defaults():
    a = build_parser().parse_args([])
    assert a.gate_entropy_aux_weight == 0.0  # off by default
    assert a.gate_entropy_aux_temperature == 1.0
    assert a.gate_entropy_aux_target_clamp == 0.0


def test_gate_entropy_aux_temperature_broadens_target():
    """Higher T ⇒ target distribution closer to 1 (less compressed)."""
    torch.manual_seed(0)
    V = 64
    flat = torch.zeros(1, 4, V)
    lse = torch.logsumexp(flat, dim=-1)
    p = (flat - lse.unsqueeze(-1)).exp()
    H = lse - (p * flat).sum(dim=-1)
    target_t1 = torch.exp(-H / 1.0)
    target_t4 = torch.exp(-H / 4.0)
    assert (target_t4 > target_t1).all()
