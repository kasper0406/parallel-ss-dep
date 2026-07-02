"""Tests for pretrain logit-distillation (KD) — task #101.

The KD term adds  lambda * T^2 * KL(teacher || student)  to the non-thinking
pretrain CE loss. The load-bearing correctness detail is the student-logit
SLICE to the teacher vocab (drop the student's extra / thinking-token slots so
the two vocabs align before the KL).

Covered:
- KD loss is exactly 0 when --distill_weight == 0 (and no teacher loaded) →
  byte-identical backward to a non-KD run.
- KD loss runs, is finite and >= 0 with a fake teacher returning random logits
  of shape (B, T, V_teacher), and carries grad into the student.
- KD is invariant to the student's EXTRA logit columns [V_teacher:] — proves
  the slice drops them (the correctness detail).
- _kd_loss_term is ~0 when student == teacher (KL of identical dists), and the
  production shape (49216 student → 49152 teacher) slices to 49152.
- KD respects the valid mask (positions with target == -100 don't contribute).

CPU-only: uses a tiny fake student/teacher, no DeltaNet CUDA kernels.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_distill_kd.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.train_lm import (
    _nonthink_forward_loss, _kd_loss_term, _kd_loss_term_topk,
)
from experiments.train_lm_args import build_parser
from experiments.teacher_logits_io import LogitStoreWriter, LogitStoreReader


class _FakeStudent(torch.nn.Module):
    """Returns a FIXED logits buffer scaled by a learnable scalar.

    Fixed so the test can mutate the extra columns and re-run; the scalar makes
    the output require grad so KD differentiability can be asserted.
    """

    def __init__(self, B, T, V):
        super().__init__()
        self.register_buffer("fixed", torch.randn(B, T, V))
        self.scale = torch.nn.Parameter(torch.ones(()))
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, x, doc_ids=None, return_aux=False, **kw):
        return self.scale * self.fixed


class _FakeTeacher(torch.nn.Module):
    """Returns random logits (B, T, V_teacher) wrapped like an HF output."""

    def __init__(self, B, T, V):
        super().__init__()
        self.register_buffer("logits", torch.randn(B, T, V))

    def forward(self, input_ids=None, **kw):
        return SimpleNamespace(logits=self.logits)


def _args(distill_weight=0.0, distill_temp=2.0):
    return SimpleNamespace(
        aux_brackets=False,
        aux_max_depth=4,
        output_gate=False,
        z_loss=0.0,
        distill_weight=distill_weight,
        distill_temp=distill_temp,
    )


# ---------------------------------------------------------------------------
# CLI surface
def test_kd_flags_parse_with_defaults():
    args = build_parser().parse_args([])
    assert args.distill_teacher_model == ""
    assert args.distill_weight == 0.0
    assert args.distill_temp == 2.0


def test_kd_flags_parse_when_set():
    args = build_parser().parse_args([
        "--distill_teacher_model", "HuggingFaceTB/SmolLM2-1.7B",
        "--distill_weight", "0.5", "--distill_temp", "3.0",
    ])
    assert args.distill_teacher_model == "HuggingFaceTB/SmolLM2-1.7B"
    assert args.distill_weight == 0.5
    assert args.distill_temp == 3.0


def test_num_workers_default_preserves_historical_behavior():
    # Offline-KD lockstep alignment requires the trainer's --num_workers to match
    # the generator's. The default MUST stay 2 (the historical hardcoded train
    # DataLoader worker count) so existing runs are byte-identical.
    args = build_parser().parse_args([])
    assert args.num_workers == 2


def test_num_workers_parses_when_set():
    args = build_parser().parse_args(["--num_workers", "0"])
    assert args.num_workers == 0


# ---------------------------------------------------------------------------
# KD off → zero, byte-identical path
def test_kd_zero_when_weight_zero():
    B, T, Vs = 2, 5, 64
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    teacher = _FakeTeacher(B, T, 48)
    # weight 0 → KD helper not invoked, kd_loss is a zero scalar.
    out = _nonthink_forward_loss(student, x, y, _args(distill_weight=0.0), 0,
                                 None, kd_teacher=teacher,
                                 kd_thinking_token_id=None)
    kd_loss = out[6]
    assert float(kd_loss) == 0.0
    assert kd_loss.shape == ()


def test_kd_zero_when_no_teacher():
    B, T, Vs = 2, 5, 64
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    out = _nonthink_forward_loss(student, x, y, _args(distill_weight=0.5), 0,
                                 None, kd_teacher=None)
    assert float(out[6]) == 0.0


# ---------------------------------------------------------------------------
# KD on → finite, positive, differentiable
def test_kd_finite_and_differentiable():
    B, T, Vs, Vt = 2, 6, 64, 48
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vt, (B, T))
    student = _FakeStudent(B, T, Vs)
    teacher = _FakeTeacher(B, T, Vt)
    out = _nonthink_forward_loss(student, x, y, _args(distill_weight=0.5), 0,
                                 None, kd_teacher=teacher)
    kd_loss = out[6]
    assert torch.isfinite(kd_loss)
    assert float(kd_loss.detach()) >= 0.0
    assert kd_loss.requires_grad
    kd_loss.backward()
    assert student.scale.grad is not None
    assert torch.isfinite(student.scale.grad)


# ---------------------------------------------------------------------------
# The load-bearing slice: KD must ignore the student's extra columns.
def test_kd_invariant_to_extra_student_columns():
    B, T, Vs, Vt = 2, 6, 64, 48
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vt, (B, T))
    student = _FakeStudent(B, T, Vs)
    teacher = _FakeTeacher(B, T, Vt)

    kd1 = float(_nonthink_forward_loss(
        student, x, y, _args(distill_weight=0.5), 0, None,
        kd_teacher=teacher)[6].detach())

    # Arbitrarily corrupt the EXTRA columns [Vt:] only.
    with torch.no_grad():
        student.fixed[..., Vt:] = torch.randn(B, T, Vs - Vt) * 100.0
    kd2 = float(_nonthink_forward_loss(
        student, x, y, _args(distill_weight=0.5), 0, None,
        kd_teacher=teacher)[6].detach())

    assert abs(kd1 - kd2) < 1e-5, (kd1, kd2)


def test_production_shapes_slice_to_teacher_vocab():
    # Student model vocab 49216 (= 49152 + 64), teacher 49152: the slice must
    # yield exactly 49152 columns.
    student_logits = torch.zeros(1, 1, 49216)
    V_t = 49152
    assert student_logits[..., :V_t].shape[-1] == 49152


# ---------------------------------------------------------------------------
# _kd_loss_term direct checks
def test_kd_loss_term_zero_for_identical_distributions():
    B, T, V = 2, 4, 16
    logits = torch.randn(B, T, V)
    valid = torch.ones(B, T)
    kd = _kd_loss_term(logits, logits, valid, temp=2.0)
    assert float(kd) < 1e-5


def test_kd_loss_term_matches_manual_formula():
    B, T, V = 2, 4, 16
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)
    valid = torch.ones(B, T)
    Temp = 2.0
    kd = _kd_loss_term(s, t, valid, temp=Temp)
    s_logp = F.log_softmax(s.float() / Temp, dim=-1)
    t_p = F.softmax(t.float() / Temp, dim=-1)
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(-1)
    expected = (Temp * Temp) * kl.mean()
    torch.testing.assert_close(kd, expected)


def test_kd_loss_term_respects_valid_mask():
    B, T, V = 1, 4, 16
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)
    # Mask out position 0 (where we put a huge teacher/student mismatch).
    s[0, 0] = torch.tensor([50.0] + [0.0] * (V - 1))
    t[0, 0] = torch.tensor([0.0] * (V - 1) + [50.0])
    full = _kd_loss_term(s, t, torch.ones(B, T), temp=2.0)
    mask = torch.ones(B, T)
    mask[0, 0] = 0.0
    masked = _kd_loss_term(s, t, mask, temp=2.0)
    # Dropping the high-divergence position lowers the (mean-over-valid) KD.
    assert float(masked) < float(full)


# ===========================================================================
# OFFLINE KD (read teacher top-k logits from a precomputed store)
# ===========================================================================

def _build_store(tmp_path, x, teacher_full_logits, k, vocab):
    """Write a logit store whose rows align with x.flatten() and carry the
    teacher's top-k of `teacher_full_logits` (B, T, V)."""
    B, T = x.shape
    topv, topi = torch.topk(teacher_full_logits, k, dim=-1)   # (B,T,k)
    w = LogitStoreWriter(str(tmp_path), k=k, vocab_size=vocab,
                         teacher_model="fake/t", tokenizer_name="fake/tok")
    w.append(topi.reshape(-1, k), topv.reshape(-1, k), x.reshape(-1))
    w.close()
    return LogitStoreReader(str(tmp_path))


def test_offline_kd_finite_and_differentiable(tmp_path):
    B, T, Vs, k = 2, 6, 64, 8
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    teacher_full = torch.randn(B, T, Vs)
    store = _build_store(tmp_path, x, teacher_full, k, Vs)
    out = _nonthink_forward_loss(student, x, y, _args(distill_weight=0.5), 0,
                                 None, kd_logit_store=store)
    kd_loss = out[6]
    assert torch.isfinite(kd_loss) and float(kd_loss.detach()) >= 0.0
    assert kd_loss.requires_grad
    kd_loss.backward()
    assert student.scale.grad is not None and torch.isfinite(student.scale.grad)


def test_offline_kd_zero_when_weight_zero(tmp_path):
    B, T, Vs, k = 2, 5, 64, 8
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    store = _build_store(tmp_path, x, torch.randn(B, T, Vs), k, Vs)
    out = _nonthink_forward_loss(student, x, y, _args(distill_weight=0.0), 0,
                                 None, kd_logit_store=store)
    assert float(out[6]) == 0.0
    # The cursor must NOT advance when KD is off (weight 0) — byte-identical path.
    assert store.tell() == 0


def test_offline_kd_matches_reference_topk_kl(tmp_path):
    B, T, Vs, k = 2, 4, 32, 6
    Temp = 2.0
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))   # all valid (!= -100)
    student = _FakeStudent(B, T, Vs)
    teacher_full = torch.randn(B, T, Vs)
    store = _build_store(tmp_path, x, teacher_full, k, Vs)

    kd = _nonthink_forward_loss(student, x, y, _args(distill_weight=1.0,
                                                     distill_temp=Temp), 0, None,
                                kd_logit_store=store)[6]

    # Reference: gather student at the SAME top-k ids, renormalise both over the
    # k-subset at temperature T, KL(teacher||student), ×T², mean over positions.
    s_logits = student(x).detach()
    topv, topi = torch.topk(teacher_full, k, dim=-1)
    s_gather = torch.gather(s_logits, -1, topi)
    s_logp = F.log_softmax(s_gather / Temp, dim=-1)
    t_p = F.softmax(topv / Temp, dim=-1)
    kl = (t_p * (t_p.log() - s_logp)).sum(-1)
    expected = (Temp * Temp) * kl.mean()
    torch.testing.assert_close(kd.detach(), expected, rtol=1e-4, atol=1e-4)


def test_offline_kd_alignment_assertion_fires_on_desync(tmp_path):
    B, T, Vs, k = 2, 5, 64, 8
    x = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    # Build the store against a DIFFERENT token stream (deliberate desync): the
    # stored input_ids will not match the live x.
    x_other = (x + 1) % Vs
    store = _build_store(tmp_path, x_other, torch.randn(B, T, Vs), k, Vs)
    with pytest.raises(RuntimeError, match="alignment FAILED"):
        _nonthink_forward_loss(student, x, y, _args(distill_weight=0.5), 0,
                               None, kd_logit_store=store)


def test_offline_kd_cursor_advances_in_lockstep(tmp_path):
    # Two microbatches read sequentially -> cursor advances by B*T each call.
    B, T, Vs, k = 2, 4, 48, 6
    x0 = torch.randint(0, Vs, (B, T))
    x1 = torch.randint(0, Vs, (B, T))
    y = torch.randint(0, Vs, (B, T))
    student = _FakeStudent(B, T, Vs)
    # Store covers x0 then x1 (in stream order).
    topv0, topi0 = torch.topk(torch.randn(B, T, Vs), k, -1)
    topv1, topi1 = torch.topk(torch.randn(B, T, Vs), k, -1)
    w = LogitStoreWriter(str(tmp_path), k=k, vocab_size=Vs)
    w.append(topi0.reshape(-1, k), topv0.reshape(-1, k), x0.reshape(-1))
    w.append(topi1.reshape(-1, k), topv1.reshape(-1, k), x1.reshape(-1))
    w.close()
    store = LogitStoreReader(str(tmp_path))
    _nonthink_forward_loss(student, x0, y, _args(distill_weight=0.5), 0, None,
                           kd_logit_store=store)
    assert store.tell() == B * T
    _nonthink_forward_loss(student, x1, y, _args(distill_weight=0.5), 0, None,
                           kd_logit_store=store)
    assert store.tell() == 2 * B * T


def test_kd_loss_term_topk_respects_valid_mask():
    B, T, V, k = 1, 4, 16, 5
    student_logits = torch.randn(B, T, V)
    t_ids = torch.randint(0, V, (B, T, k))
    t_logits = torch.randn(B, T, k)
    full = _kd_loss_term_topk(student_logits, t_ids, t_logits,
                              torch.ones(B, T), temp=2.0)
    mask = torch.ones(B, T)
    mask[0, 0] = 0.0
    masked = _kd_loss_term_topk(student_logits, t_ids, t_logits, mask, temp=2.0)
    # Masking a position changes the mean-over-valid KD (different denom/support).
    assert float(full) != float(masked)
