"""Tests for experiments/kd_objectives.py — the LFM2-style tempered decoupled
top-K KD objective over the stored teacher-logit format, and its trainer
dispatch (`--kd_objective {legacy,decoupled}`).

Covered:
- decoupled loss matches an analytic hand-computed value on a tiny
  distribution (B=T=1, V=4, K=2), including temperature scaling at T=2.
- legacy default byte-identity: the module's legacy copy is bitwise-equal to
  train_lm._kd_loss_term_topk (drift guard), and _nonthink_forward_loss
  WITHOUT a kd_objective attr (every pre-existing caller) still routes to the
  legacy term.
- T=1 reduction: the within-top-K component at T=1 equals the plain
  (unscaled, untempered) KL over the renormalised top-K support.
- mass term direction: student mass on the teacher's top-K set pulled toward
  the teacher's (≈1) — more off-set mass ⇒ larger loss, and the gradient
  pushes off-set logits down.
- mass term is untempered (invariant to --kd_temperature).
- CE-mix composition: ce_mix=1 → plain masked CE; ce_mix=0.5 → exact convex
  mix of the two calls.
- top-K-miss handling: hard labels outside the teacher's top-K set are exact
  in the CE term; targets are ignored entirely at ce_mix=0.
- valid-mask handling incl. -100 / OOB targets at masked positions.
- mass_weight=0 + ce_mix=0 collapses to the legacy objective exactly.
- trainer dispatch end-to-end through _nonthink_forward_loss + a real
  LogitStoreReader.

CPU-only. Run:
    PYTHONPATH=. .venv/bin/python -m pytest experiments/test_kd_objectives.py -v
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.kd_objectives import (
    decoupled_topk_kd_loss,
    legacy_topk_kd_loss,
)
from experiments.train_lm import _kd_loss_term_topk, _nonthink_forward_loss
from experiments.train_lm_args import build_parser
from experiments.teacher_logits_io import LogitStoreReader, LogitStoreWriter


def _rand_case(B=2, T=3, V=16, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    student = torch.randn(B, T, V, generator=g)
    teacher_full = torch.randn(B, T, V, generator=g)
    t_logits, t_ids = torch.topk(teacher_full, k, dim=-1)
    valid = torch.ones(B, T)
    return student, t_ids, t_logits, valid


# ---------------------------------------------------------------------------
# Analytic hand-computed value on a tiny distribution.

def test_decoupled_matches_hand_computed_tiny():
    # V=4 student logits, teacher top-K=2 at ids {0, 2}, T=2, all valid.
    s = torch.tensor([[[1.0, 0.0, -1.0, 0.5]]])           # (1,1,4)
    t_ids = torch.tensor([[[0, 2]]])                       # (1,1,2)
    t_logits = torch.tensor([[[2.0, 1.0]]])                # (1,1,2)
    valid = torch.ones(1, 1)
    T = 2.0

    # Hand computation with plain python math -----------------------------
    # student full softmax
    zs = [1.0, 0.0, -1.0, 0.5]
    Z = sum(math.exp(z) for z in zs)
    p_full = [math.exp(z) / Z for z in zs]
    m_s = p_full[0] + p_full[2]
    mass = -math.log(m_s)
    # within-top-K, tempered: student gathered at ids (1.0, -1.0)
    sg = [1.0 / T, -1.0 / T]
    Zk = sum(math.exp(z) for z in sg)
    q = [math.exp(z) / Zk for z in sg]                     # student over K
    tg = [2.0 / T, 1.0 / T]
    Zt = sum(math.exp(z) for z in tg)
    p = [math.exp(z) / Zt for z in tg]                     # teacher over K
    kl = sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))
    expected = (T * T) * kl + 1.0 * mass

    got = decoupled_topk_kd_loss(s, t_ids, t_logits, valid,
                                 temperature=T, mass_weight=1.0, ce_mix=0.0)
    torch.testing.assert_close(got, torch.tensor(expected),
                               rtol=1e-6, atol=1e-6)


def test_decoupled_hand_computed_with_ce_mix():
    # Same tiny case + hard label y=3 (a top-K MISS: 3 not in {0,2}) at
    # ce_mix=0.25 → loss = 0.75*kd + 0.25*CE.
    s = torch.tensor([[[1.0, 0.0, -1.0, 0.5]]])
    t_ids = torch.tensor([[[0, 2]]])
    t_logits = torch.tensor([[[2.0, 1.0]]])
    valid = torch.ones(1, 1)
    y = torch.tensor([[3]])
    kd = decoupled_topk_kd_loss(s, t_ids, t_logits, valid,
                                temperature=2.0, mass_weight=1.0, ce_mix=0.0)
    zs = [1.0, 0.0, -1.0, 0.5]
    Z = sum(math.exp(z) for z in zs)
    ce = -math.log(math.exp(0.5) / Z)
    got = decoupled_topk_kd_loss(s, t_ids, t_logits, valid, targets=y,
                                 temperature=2.0, mass_weight=1.0, ce_mix=0.25)
    torch.testing.assert_close(
        got, 0.75 * kd + 0.25 * torch.tensor(ce), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Legacy identity / drift guards.

def test_legacy_copy_bitwise_matches_train_lm():
    student, t_ids, t_logits, valid = _rand_case(seed=1)
    a = _kd_loss_term_topk(student, t_ids, t_logits, valid, 2.0)
    b = legacy_topk_kd_loss(student, t_ids, t_logits, valid, 2.0)
    assert torch.equal(a, b), "kd_objectives.legacy drifted from train_lm"


def test_decoupled_mass0_ce0_equals_legacy():
    student, t_ids, t_logits, valid = _rand_case(seed=2)
    for T in (1.0, 2.0, 4.0):
        leg = legacy_topk_kd_loss(student, t_ids, t_logits, valid, T)
        dec = decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                                     temperature=T, mass_weight=0.0,
                                     ce_mix=0.0)
        torch.testing.assert_close(dec, leg, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Temperature behaviour.

def test_within_kl_T1_reduction():
    """At T=1 the within component is the plain KL over the renormalised
    top-K support — no tempering, no Hinton scaling (T²=1)."""
    student, t_ids, t_logits, valid = _rand_case(seed=3)
    _, comps = decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                                      temperature=1.0, mass_weight=0.0,
                                      ce_mix=0.0, return_components=True)
    s_g = torch.gather(student, -1, t_ids)
    q = F.log_softmax(s_g, dim=-1)
    p = F.softmax(t_logits, dim=-1)
    kl = (p * (p.log() - q)).sum(-1).mean()
    torch.testing.assert_close(torch.tensor(comps["kl_within"]), kl,
                               rtol=1e-5, atol=1e-6)


def test_mass_term_is_untempered():
    student, t_ids, t_logits, valid = _rand_case(seed=4)
    masses = []
    for T in (1.0, 2.0, 5.0):
        _, comps = decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                                          temperature=T, return_components=True)
        masses.append(comps["mass"])
    assert masses[0] == masses[1] == masses[2], (
        f"mass term must not depend on temperature, got {masses}")


# ---------------------------------------------------------------------------
# Mass-term direction.

def test_mass_term_direction_pulls_student_onto_teacher_set():
    # Teacher's set is {0, 1}. Student A concentrates its mass there; student
    # B leaks it to id 3. Same within-set SHAPE for both (logit gap 1.0), so
    # only the mass term separates them.
    t_ids = torch.tensor([[[0, 1]]])
    t_logits = torch.tensor([[[3.0, 2.0]]])
    valid = torch.ones(1, 1)
    s_on = torch.tensor([[[5.0, 4.0, -5.0, -5.0]]])
    s_off = torch.tensor([[[5.0, 4.0, -5.0, 9.0]]])       # mass leaked to id 3
    kw = dict(temperature=2.0, mass_weight=1.0, ce_mix=0.0,
              return_components=True)
    _, c_on = decoupled_topk_kd_loss(s_on, t_ids, t_logits, valid, **kw)
    _, c_off = decoupled_topk_kd_loss(s_off, t_ids, t_logits, valid, **kw)
    assert c_on["student_topk_mass"] > 0.99
    assert c_off["student_topk_mass"] < 0.10
    assert c_off["mass"] > c_on["mass"] + 1.0
    # And the LEGACY objective is blind to the leak (same renormalised
    # within-set distribution) — the exact support-mismatch failure the
    # decoupled objective fixes.
    leg_on = legacy_topk_kd_loss(s_on, t_ids, t_logits, valid, 2.0)
    leg_off = legacy_topk_kd_loss(s_off, t_ids, t_logits, valid, 2.0)
    torch.testing.assert_close(leg_on, leg_off, rtol=1e-6, atol=1e-6)


def test_mass_term_gradient_pushes_offset_logits_down():
    t_ids = torch.tensor([[[0, 1]]])
    t_logits = torch.tensor([[[3.0, 2.0]]])
    valid = torch.ones(1, 1)
    s = torch.tensor([[[1.0, 0.5, 0.0, 2.0]]], requires_grad=True)
    loss = decoupled_topk_kd_loss(s, t_ids, t_logits, valid,
                                  temperature=2.0, mass_weight=1.0, ce_mix=0.0)
    loss.backward()
    g = s.grad[0, 0]
    # Increasing an off-set logit (ids 2, 3) increases the loss.
    assert g[2] > 0 and g[3] > 0
    # And the total gradient into the on-set logits is negative (mass pulled in).
    assert (g[0] + g[1]) < 0


# ---------------------------------------------------------------------------
# CE mix.

def test_ce_mix_one_is_plain_masked_ce():
    student, t_ids, t_logits, valid = _rand_case(B=2, T=4, V=16, k=4, seed=5)
    y = torch.randint(0, 16, (2, 4))
    got = decoupled_topk_kd_loss(student, t_ids, t_logits, valid, targets=y,
                                 temperature=2.0, mass_weight=1.0, ce_mix=1.0)
    expected = F.cross_entropy(student.reshape(-1, 16), y.reshape(-1))
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)


def test_ce_mix_half_is_exact_convex_mix():
    student, t_ids, t_logits, valid = _rand_case(seed=6)
    y = torch.randint(0, 16, (2, 3))
    kd = decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                                temperature=2.0, mass_weight=0.7, ce_mix=0.0)
    ce = decoupled_topk_kd_loss(student, t_ids, t_logits, valid, targets=y,
                                temperature=2.0, mass_weight=0.7, ce_mix=1.0)
    both = decoupled_topk_kd_loss(student, t_ids, t_logits, valid, targets=y,
                                  temperature=2.0, mass_weight=0.7, ce_mix=0.5)
    torch.testing.assert_close(both, 0.5 * kd + 0.5 * ce,
                               rtol=1e-6, atol=1e-6)


def test_ce_mix_zero_ignores_targets_and_topk_miss_is_finite():
    student, t_ids, t_logits, valid = _rand_case(seed=7)
    a = decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                               targets=None, ce_mix=0.0)
    y = torch.randint(0, 16, (2, 3))
    b = decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                               targets=y, ce_mix=0.0)
    assert torch.equal(a, b)
    assert torch.isfinite(a)


def test_ce_mix_out_of_range_raises():
    student, t_ids, t_logits, valid = _rand_case(seed=8)
    with pytest.raises(ValueError, match="ce_mix"):
        decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                               targets=torch.zeros(2, 3, dtype=torch.long),
                               ce_mix=1.5)
    with pytest.raises(ValueError, match="requires targets"):
        decoupled_topk_kd_loss(student, t_ids, t_logits, valid, ce_mix=0.5)


# ---------------------------------------------------------------------------
# Valid mask.

def test_valid_mask_excludes_positions_and_tolerates_ignore_targets():
    student, t_ids, t_logits, valid = _rand_case(B=1, T=4, V=16, k=4, seed=9)
    mask = valid.clone()
    mask[0, 1] = 0.0
    y = torch.randint(0, 16, (1, 4))
    y[0, 1] = -100                       # ignore label only at masked position
    base = decoupled_topk_kd_loss(student, t_ids, t_logits, mask, targets=y,
                                  temperature=2.0, ce_mix=0.3)
    # Wildly perturb the masked position — loss must not move.
    student2 = student.clone()
    student2[0, 1] += 100.0
    pert = decoupled_topk_kd_loss(student2, t_ids, t_logits, mask, targets=y,
                                  temperature=2.0, ce_mix=0.3)
    torch.testing.assert_close(base, pert, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(base)


def test_components_compose_to_total():
    student, t_ids, t_logits, valid = _rand_case(seed=10)
    y = torch.randint(0, 16, (2, 3))
    lam, mw = 0.3, 0.5
    total, c = decoupled_topk_kd_loss(student, t_ids, t_logits, valid,
                                      targets=y, temperature=3.0,
                                      mass_weight=mw, ce_mix=lam,
                                      return_components=True)
    recomposed = (1 - lam) * (c["kl_within"] + mw * c["mass"]) + lam * c["ce"]
    torch.testing.assert_close(total, torch.tensor(recomposed),
                               rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Trainer dispatch (flags + _nonthink_forward_loss end-to-end).

def _build_store(tmp_path, x, teacher_full_logits, k, vocab):
    B, T = x.shape
    topv, topi = torch.topk(teacher_full_logits, k, dim=-1)
    w = LogitStoreWriter(str(tmp_path), k=k, vocab_size=vocab,
                         teacher_model="fake/t", tokenizer_name="fake/tok")
    w.append(topi.reshape(-1, k), topv.reshape(-1, k), x.reshape(-1))
    w.close()
    return LogitStoreReader(str(tmp_path))


class _FakeStudent(torch.nn.Module):
    def __init__(self, B, T, V):
        super().__init__()
        self.register_buffer("fixed", torch.randn(B, T, V))
        self.scale = torch.nn.Parameter(torch.ones(()))
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, x, doc_ids=None, return_aux=False, **kw):
        return self.scale * self.fixed


def _args(**kw):
    base = dict(aux_brackets=False, aux_max_depth=4, output_gate=False,
                z_loss=0.0, distill_weight=1.0, distill_temp=2.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_flags_default_legacy():
    args = build_parser().parse_args([])
    assert args.kd_objective == "legacy"
    assert args.kd_temperature == 2.0
    assert args.kd_mass_weight == 1.0
    assert args.kd_ce_mix == 0.0


def test_dispatch_legacy_default_byte_identity(tmp_path):
    """Args WITHOUT any kd_objective attr (every pre-existing caller) and
    args with the explicit default both route to the untouched legacy term."""
    B, T, Vs, k = 2, 4, 32, 6
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    teacher_full = torch.randn(B, T, Vs)

    def run(args):
        store = _build_store(tmp_path / f"s{run.i}", x, teacher_full, k, Vs)
        run.i += 1
        return _nonthink_forward_loss(student, x, y, args, 0, None,
                                      kd_logit_store=store)[6].detach()
    run.i = 0
    kd_noattr = run(_args())
    kd_legacy = run(_args(kd_objective="legacy"))
    s_logits = student(x).detach()
    topv, topi = torch.topk(teacher_full, k, dim=-1)
    # The store persists teacher logits as fp16 — mirror that in the reference.
    expected = _kd_loss_term_topk(s_logits, topi, topv.to(torch.float16),
                                  torch.ones(B, T), 2.0)
    assert torch.equal(kd_noattr, kd_legacy)
    torch.testing.assert_close(kd_noattr, expected, rtol=1e-5, atol=1e-6)


def test_dispatch_decoupled_end_to_end(tmp_path):
    B, T, Vs, k = 2, 4, 32, 6
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    teacher_full = torch.randn(B, T, Vs)
    store = _build_store(tmp_path, x, teacher_full, k, Vs)
    args = _args(kd_objective="decoupled", kd_temperature=3.0,
                 kd_mass_weight=0.5, kd_ce_mix=0.1)
    kd = _nonthink_forward_loss(student, x, y, args, 0, None,
                                kd_logit_store=store)[6]
    topv, topi = torch.topk(teacher_full, k, dim=-1)
    # The store persists teacher logits as fp16 — mirror that in the reference.
    expected = decoupled_topk_kd_loss(
        student(x).detach(), topi, topv.to(torch.float16),
        torch.ones(B, T), targets=y,
        temperature=3.0, mass_weight=0.5, ce_mix=0.1)
    torch.testing.assert_close(kd.detach(), expected, rtol=1e-5, atol=1e-6)
    assert kd.requires_grad
    kd.backward()
    assert torch.isfinite(student.scale.grad)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
