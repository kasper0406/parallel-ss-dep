"""Trunk multi-horizon gist loss — shared by sft_code.py and train_lm.py.

The trunk's "high-level direction" objective: at position t, a per-horizon
head predicts the GIST of the upcoming window — the mean-pooled hidden
state over h[t+1 : t+1+K], stop-grad'd. Because the trunk is causal each
h[t] is a running contextualised summary, so the windowed mean is a
genuine "where this is going" vector. Multi-horizon K gives local tactic
+ mid plan + global direction.

History (see GEMINI.md): v5 supervised the WM read to predict a single
future input embedding (context-free lexical); v6 supervised the WM read
to predict this gist (a blurry target routed through the recall path —
broke precise recall, 99%→61%). v7 moved the gist target to the TRUNK and
left WM free to learn precise retrieval. This module is that v7 mechanism,
factored out so the pretrain trainer can bake it in from step 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_horizons(s: str) -> list[int]:
    """Parse a comma-separated horizon string ('16,64,256') into a sorted
    unique list of positive ints."""
    hs = sorted({int(k) for k in s.split(",") if k.strip()})
    if not hs or any(k <= 0 for k in hs):
        raise ValueError(f"invalid gist horizons: {s!r}")
    return hs


def windowed_future_gist(h: torch.Tensor, K: int):
    """Mean-pooled hidden state over the K positions FOLLOWING each source
    position — the gist-prediction target.

    Given hidden states h of shape (B, T, d), returns (gist, valid_len):
      gist[:, t] = mean(h[:, t+1 : t+1+K])   for source t in [0, T-1-K]
      valid_len  = T - K  (number of source positions with a full window)

    Returns (None, 0) when K >= T (no full window). Computed in fp32 via a
    cumulative sum so it is O(T) not O(T*K). NOT detached — the caller
    detaches it (the trunk hidden states stay supervised only by the main
    loss; the gist is a stop-grad target)."""
    T = h.shape[1]
    if K >= T:
        return None, 0
    cumh = torch.cumsum(h.float(), dim=1)            # (B, T, d)
    # sum h[t+1 .. t+K] = cumh[t+K] - cumh[t]
    win_sum = cumh[:, K:] - cumh[:, :-K]             # (B, T-K, d)
    return win_sum / float(K), T - K


def build_gist_heads(d_model: int, horizons: list[int]) -> nn.ModuleDict:
    """One bias-free Linear(d_model, d_model) prediction head per horizon,
    keyed by str(K). Small init so the aux loss starts gentle."""
    heads = nn.ModuleDict({
        str(k): nn.Linear(d_model, d_model, bias=False) for k in horizons
    })
    for head in heads.values():
        nn.init.normal_(head.weight, std=0.02 / math.sqrt(2))
    return heads


def trunk_gist_loss(h: torch.Tensor, heads: nn.ModuleDict,
                    horizons: list[int]) -> torch.Tensor:
    """Multi-horizon trunk gist loss: for each horizon K, predict
    windowed_future_gist(h, K) from h[t] via head_K, score by 1 - cosine,
    average over positions and over the horizons that produced a valid
    window. The gist target is stop-grad'd; gradient flows to the trunk
    only through the prediction path."""
    acc = h.new_zeros(())
    n_used = 0
    for K in horizons:
        gist, vlen = windowed_future_gist(h, K)
        if gist is None:
            continue
        pred = heads[str(K)](h[:, :vlen].contiguous())
        cos = F.cosine_similarity(pred.float(), gist.detach(), dim=-1)
        acc = acc + (1.0 - cos).mean()
        n_used += 1
    if n_used == 0:
        return h.new_zeros(())
    return acc / n_used


# --------------------------------------------------------------------
# Phase 1a — gist-at-think supervision (THINKING_PLAN.md Phase 1a).
#
# A small head projects the student's hidden state at a think position
# into the teacher's hidden-state space. The teacher is the same model
# (no_grad) run over the full (prompt + CoT prose + code) sequence. Each
# student think position is supervised against ONE teacher CoT-position
# hidden state (the last token of the K-chunk it represents). The
# architectural claim being tested: WM + FiLM + retrieval-as-input give
# the student enough capacity per think to fit K tokens of compressed
# reasoning.
# --------------------------------------------------------------------
def build_think_gist_head(d_model: int) -> nn.Linear:
    """Single bias-free Linear(d_model, d_model) projecting student think
    hidden states into the teacher's hidden-state space. Small init so
    the aux loss starts gentle (matches build_gist_heads convention)."""
    head = nn.Linear(d_model, d_model, bias=False)
    nn.init.normal_(head.weight, std=0.02 / math.sqrt(2))
    return head


def think_gist_loss(
    student_h: torch.Tensor,
    teacher_h: torch.Tensor,
    think_positions: list[list[int]],
    teacher_cot_positions: list[list[int]],
    head: nn.Linear,
    *,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """Gist supervision at student think positions.

    For each (b, think_pos, cot_pos) pair, compute
      pred  = head(student_h[b, think_pos])
      target = teacher_h[b, cot_pos].detach()
      loss  = 1 - cosine(pred, target)   (or MSE)
    Mean over all pairs; returns 0 if no pairs were provided.

    Shapes:
      student_h: (B, T_student, d_model)
      teacher_h: (B, T_teacher, d_model) — already gathered into the same
        batch dim (teacher and student share batch & rotation).
      think_positions[b]: list of positions in student_h for batch b.
      teacher_cot_positions[b]: parallel list of positions in teacher_h.
        len(think_positions[b]) MUST equal len(teacher_cot_positions[b]).
    """
    if loss_type not in ("cosine", "mse"):
        raise ValueError(f"unknown loss_type {loss_type!r}")
    B = student_h.shape[0]
    if len(think_positions) != B or len(teacher_cot_positions) != B:
        raise ValueError(
            f"think_positions/teacher_cot_positions batch mismatch: "
            f"{len(think_positions)}, {len(teacher_cot_positions)}, B={B}")

    student_vecs = []
    teacher_vecs = []
    for b in range(B):
        sp = think_positions[b]
        tp = teacher_cot_positions[b]
        if len(sp) != len(tp):
            raise ValueError(
                f"batch {b}: think_positions len {len(sp)} != "
                f"teacher_cot_positions len {len(tp)}")
        if len(sp) == 0:
            continue
        sp_t = torch.as_tensor(sp, dtype=torch.long, device=student_h.device)
        tp_t = torch.as_tensor(tp, dtype=torch.long, device=teacher_h.device)
        student_vecs.append(student_h[b].index_select(0, sp_t))
        teacher_vecs.append(teacher_h[b].index_select(0, tp_t))

    if not student_vecs:
        return student_h.new_zeros(())

    s_cat = torch.cat(student_vecs, dim=0)        # (N_pairs, d)
    t_cat = torch.cat(teacher_vecs, dim=0).detach()
    pred = head(s_cat)
    if loss_type == "cosine":
        cos = F.cosine_similarity(pred.float(), t_cat.float(), dim=-1)
        return (1.0 - cos).mean()
    # MSE in fp32 (numerical stability).
    return F.mse_loss(pred.float(), t_cat.float())
