"""Top-K knowledge-distillation objectives over the STORED teacher format.

The offline-KD store (`teacher_logits_io.py`) keeps, per token position, the
teacher's top-K token ids + RAW top-K logits (fp16) — no full distribution, no
log-partition. Two objectives over that format live here:

* ``legacy_topk_kd_loss`` — the original truncated KD used by
  ``train_lm.py::_kd_loss_term_topk`` (kept there as the default code path;
  re-implemented here byte-for-byte so tests can guard against drift): both
  teacher top-K logits and the gathered student logits are softmax-renormalised
  over the K-subset at temperature T, KL(teacher‖student) ×T² (Hinton), mean
  over valid positions. Known weakness (the LFM2 report's motivation): the
  student's renormalised K-distribution says nothing about how much of its
  FULL-vocab mass is on the teacher's support — a student putting 1% of its
  total mass on the top-K set in exactly the right proportions scores a
  perfect 0, and temperature scaling interacts badly with that support
  mismatch (tempering flattens the renormalised target while the out-of-set
  mass is invisible to the loss).

* ``decoupled_topk_kd_loss`` — the LFM2-style tempered, DECOUPLED objective
  (LFM2 tech report, arXiv:2511.23404: KD from LFM1-7B for the whole 10T-token
  pretrain used exactly this decomposition). The full-vocab KL is split into

      (a) an UNTEMPERED binary term matching the student's TOTAL probability
          mass on the teacher's stored top-K set, and
      (b) a temperature-scaled KL WITHIN the top-K set (identical math to the
          legacy term),

  optionally mixed with hard-label CE. The binary term restores the missing
  support-matching gradient at T=1 sharpness while (b) keeps the tempered
  dark-knowledge signal — solving the truncation/temperature support-mismatch
  instability of naive truncated+tempered KD.

  STORED-FORMAT APPROXIMATION (documented, load-bearing): the store keeps RAW
  top-K logits without the teacher's log-partition, so the teacher's true
  in-set mass m_T = Σ_{i∈topK} softmax_full(teacher)_i is unrecoverable. The
  in-format approximation logZ ≈ logsumexp(topK logits) gives m_T = 1, under
  which the binary KL  m_T·log(m_T/m_S) + (1−m_T)·log((1−m_T)/(1−m_S))
  reduces to  −log(m_S)  — "pull the student's total mass on the teacher's
  top-K set toward (the teacher's, ≈) 1". For a peaked code teacher at K=24
  the true m_T is ≈1, so the approximation error is a small constant bias in
  the target, not a direction error.

Composition (``decoupled_topk_kd_loss``)::

    kd  = T² · KL_within(teacher_topK ‖ student_topK)   # tempered, per-position
        + mass_weight · (−log m_S)                       # UNTEMPERED
    loss = (1 − ce_mix) · kd + ce_mix · CE(student, y)   # hard-label mix

  mean-reduced over ``valid_mask`` positions, fp32 throughout. With
  ``mass_weight=0, ce_mix=0`` this is numerically the legacy objective.

Trainer wiring: ``train_lm.py`` dispatches on ``--kd_objective``
(``legacy`` default → the untouched ``_kd_loss_term_topk`` path,
byte-identical; ``decoupled`` → this module) with ``--kd_temperature``,
``--kd_mass_weight``, ``--kd_ce_mix``. The decoupled objective is implemented
for the OFFLINE top-K store path only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def legacy_topk_kd_loss(student_logits: torch.Tensor,
                        teacher_ids: torch.Tensor,
                        teacher_topk_logits: torch.Tensor,
                        valid_mask: torch.Tensor,
                        temp: float) -> torch.Tensor:
    """Byte-for-byte re-implementation of ``train_lm._kd_loss_term_topk``.

    NOT called by the trainer (the default path stays in train_lm.py so
    ``--kd_objective legacy`` is literally the pre-change code); exists so
    ``test_kd_objectives.py`` can pin the two against each other and catch
    drift in either copy.
    """
    Tmp = max(float(temp), 1e-6)
    s_gather = torch.gather(student_logits.float(), -1,
                            teacher_ids.long())                  # (B,T,k)
    s_logp = F.log_softmax(s_gather / Tmp, dim=-1)               # renorm over k
    t_p = F.softmax(teacher_topk_logits.float() / Tmp, dim=-1)   # renorm over k
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(dim=-1)     # (B, T)
    denom = valid_mask.sum().clamp(min=1.0)
    return (Tmp * Tmp) * (kl * valid_mask).sum() / denom


def decoupled_topk_kd_loss(student_logits: torch.Tensor,
                           teacher_ids: torch.Tensor,
                           teacher_topk_logits: torch.Tensor,
                           valid_mask: torch.Tensor,
                           *,
                           targets: torch.Tensor | None = None,
                           temperature: float = 2.0,
                           mass_weight: float = 1.0,
                           ce_mix: float = 0.0,
                           eps: float = 1e-12,
                           return_components: bool = False):
    """LFM2-style tempered, decoupled top-K KD (see module docstring).

    Args:
        student_logits:      (B, T, V) student FULL next-token logits.
        teacher_ids:         (B, T, k) teacher's top-k token ids (distinct
                             per position, by construction of topk).
        teacher_topk_logits: (B, T, k) teacher's RAW top-k logits.
        valid_mask:          (B, T) float in {0, 1}.
        targets:             (B, T) hard labels; required when ce_mix > 0.
                             May contain -100 / out-of-support ids ONLY at
                             positions where valid_mask == 0 (they are clamped
                             for the gather and masked out of the mean).
        temperature:         T for the WITHIN-top-K KL (×T², Hinton). The
                             mass term is deliberately UNTEMPERED.
        mass_weight:         weight on the −log(m_S) mass term.
        ce_mix:              λ ∈ [0, 1]; loss = (1−λ)·kd + λ·CE.
        return_components:   also return a dict of scalar components
                             (each mean-reduced over valid positions).

    Returns a scalar (or ``(scalar, components)``), fp32.
    """
    if not (0.0 <= float(ce_mix) <= 1.0):
        raise ValueError(f"ce_mix must be in [0, 1], got {ce_mix}")
    s = student_logits.float()
    ids = teacher_ids.long()
    valid = valid_mask.float()
    denom = valid.sum().clamp(min=1.0)
    Tmp = max(float(temperature), 1e-6)

    # (a) UNTEMPERED binary mass term: student's total (full-vocab softmax)
    # probability on the teacher's stored top-K set, pulled toward the
    # teacher's in-set mass (≈1 under the stored-format approximation —
    # module docstring). Binary KL with m_T=1 reduces to −log(m_S).
    s_logp_full = F.log_softmax(s, dim=-1)                       # (B,T,V)
    m_s = torch.gather(s_logp_full, -1, ids).exp().sum(dim=-1)   # (B,T)
    m_s = m_s.clamp(min=eps, max=1.0)
    mass = -m_s.log()                                            # (B,T) ≥ 0

    # (b) tempered KL WITHIN the top-K set — identical math to the legacy
    # objective (renormalise both over the K support at temperature T, ×T²).
    s_gather = torch.gather(s, -1, ids)                          # (B,T,k)
    s_logp_k = F.log_softmax(s_gather / Tmp, dim=-1)
    t_p_k = F.softmax(teacher_topk_logits.float() / Tmp, dim=-1)
    kl_within = (Tmp * Tmp) * F.kl_div(
        s_logp_k, t_p_k, reduction="none").sum(dim=-1)           # (B,T)

    kd = kl_within + float(mass_weight) * mass                   # (B,T)

    # (c) hard-label CE mix. -100 / OOB ids are only legal at masked
    # positions; clamp for the gather, the mask zeroes their contribution.
    ce_mix = float(ce_mix)
    if ce_mix > 0.0:
        if targets is None:
            raise ValueError("ce_mix > 0 requires targets")
        V = s.size(-1)
        y_safe = targets.long().clamp(min=0, max=V - 1)
        ce = -torch.gather(s_logp_full, -1,
                           y_safe.unsqueeze(-1)).squeeze(-1)     # (B,T)
        loss_pos = (1.0 - ce_mix) * kd + ce_mix * ce
    else:
        ce = None
        loss_pos = kd

    loss = (loss_pos * valid).sum() / denom
    if not return_components:
        return loss
    comps = {
        "kl_within": float((kl_within * valid).sum() / denom),
        "mass": float((mass * valid).sum() / denom),
        "ce": float((ce * valid).sum() / denom) if ce is not None else 0.0,
        "student_topk_mass": float((m_s * valid).sum() / denom),
    }
    return loss, comps
