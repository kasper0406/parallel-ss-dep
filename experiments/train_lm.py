"""
Minimal language modelling training driver — practical demonstration.

The point: show that the hybrid scaffold trains stably on real text at
~135M-param scale and reaches competitive perplexity vs pure DeltaNet.

Uses TinyStories (HuggingFaceH4/TinyStories, ~500MB of children's
stories) tokenised with SmolLM2's tokeniser (49152 vocab). This is the
smallest realistic LM benchmark; if hybrid can't match DeltaNet here,
distillation is futile.

Usage:
    python experiments/train_lm.py --arch deltanet --steps 5000
    python experiments/train_lm.py --arch hybrid   --steps 5000
"""
from __future__ import annotations

import argparse
import contextlib
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from experiments.layers import (
    DeltaNetAttention,
    SoftmaxAttention, Mamba2Attention,
)
from experiments.model import TinyLM
from experiments.aux_brackets import compute_bracket_deltas, bracket_depth
from experiments.gist_loss import build_gist_heads, parse_horizons
from experiments.thinking import (
    ThinkContinuation,
    ThinkContinuationQueue,
    ThinkReplay,
    ThinkReplayQueue,
    build_continuation_batch,
    build_replay_batch,
    choose_explore_actions,
    choose_think_actions,
    cross_entropy_masking_token,
    latent_cotrain_loss,
    mask_token_logit,
)


from experiments.gate_calibration import compute_gate_calibration_loss
# LatentReasoningCotrain is imported lazily inside main() — its dependency chain
# (latent_sft → eval_bracket_structure → train_lm) would otherwise be circular.
from experiments.build_arch import build_arch, parse_layers_arg, _NAME_TO_CLS  # noqa: F401


# ---------------------------------------------------------------------------
# Per-layer learning diagnostics. Logged every --log_every steps to answer a
# specific question: are early layers gradient-starved (the vanishing-gradient
# signature) or is the gradient healthy and "slow early layers" just a misread
# of the logit-lens? Cheap — grad norms are free post-backward; the
# update-ratio clones one matrix per block on log steps only.

def _block_repr_weight(blk):
    """First 2D weight in a block — the representative matrix tracked for the
    update-to-weight ratio."""
    return next((p for p in blk.parameters() if p.ndim == 2), None)


def _block_grad_norms(model) -> list[float]:
    """Per-block total gradient L2 norm. Call after backward(), before clip."""
    out = []
    for blk in model.blocks:
        sq = 0.0
        for p in blk.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
        out.append(sq ** 0.5)
    return out


def _block_weight_snapshot(model) -> list:
    """Clone each block's representative weight (call pre-step)."""
    snap = []
    for blk in model.blocks:
        w = _block_repr_weight(blk)
        snap.append(None if w is None else w.detach().clone())
    return snap


def _block_update_ratios(model, snapshot) -> list[float]:
    """‖ΔW‖/‖W‖ for each block's representative weight (call post-step)."""
    out = []
    for blk, w_before in zip(model.blocks, snapshot):
        w_after = _block_repr_weight(blk)
        if w_before is None or w_after is None:
            out.append(float("nan"))
            continue
        delta = float((w_after.detach() - w_before).float().norm())
        denom = max(float(w_before.float().norm()), 1e-9)
        out.append(delta / denom)
    return out


def _nonthink_forward_loss(model, x, y, args, step, bracket_deltas,
                           doc_ids=None, gist_horizons=None, fwd_model=None,
                           mem_read_mask=None, kd_teacher=None,
                           kd_thinking_token_id=None, kd_logit_store=None):
    """Forward + LM loss for the non-thinking-token (pretrain) path.

    Returns (logits, ce_per_token, lm_loss, aux_loss, gate_aux_loss,
    gist_loss, kd_loss). Factored out of the step loop so gradient
    accumulation can run it once per microbatch. Mirrors the inline
    forward + gate/plain-loss branches exactly.

    KD has two mutually-exclusive sources:
    * OFFLINE (`kd_logit_store` set): the teacher's TOP-K logits are read from a
      `LogitStoreReader` cursor that advances in LOCKSTEP with the data iterator
      (one `next_block(x.numel())` per microbatch). The store's `input_ids` are
      asserted equal to the live `x` (the load-bearing alignment safety check),
      then the student is gathered at the teacher's top-k ids and a top-k KL is
      added. NO teacher in the loop.
    * LIVE (`kd_teacher` set): see below.

    When `kd_teacher` is set AND --distill_weight > 0, an extra frozen-teacher
    forward (no_grad) over the SAME `x` produces teacher logits; the student
    logits are sliced to the teacher vocab and a temperature-scaled KL is
    returned as `kd_loss` (added to the total loss at the call site, scaled by
    --distill_weight). kd_loss is a zero scalar when KD is off → byte-identical
    backward to the pre-KD path.

    When `gist_horizons` is set and --gist_loss_weight > 0, the model
    computes the multi-horizon trunk gist loss INSIDE its forward and
    returns it as a scalar (model._gist_loss_enabled gates it). Not
    supported together with --aux_brackets.

    When --gate_entropy_aux_weight > 0, the gate logit gets an auxiliary
    BCE target derived from the SAME forward's per-position next-token
    entropy (detached): target_t = exp(-H_t/T). High entropy ⇒ low target
    ⇒ gate trained to close (think). Costs nothing extra — no second
    forward, just turns the gate into a free predictive-uncertainty head.
    """
    # `fwd_model` is the (possibly DDP-wrapped) callable used for the
    # loss-bearing forward so DDP's grad-sync hooks fire; `model` stays the raw
    # module for attribute reads (_last_gate, pkm_layer, …) which DDP doesn't
    # proxy. Default fwd_model=model preserves the single-GPU path exactly.
    fwd_model = fwd_model if fwd_model is not None else model
    # v14: thread mem_read_mask ONLY when present, so the forward call is
    # byte-identical to the pre-v14 `fwd_model(x, doc_ids=doc_ids)` when off
    # (also keeps mock LMs that don't accept the kwarg working).
    _fwd_kw = {"doc_ids": doc_ids}
    if mem_read_mask is not None:
        _fwd_kw["mem_read_mask"] = mem_read_mask
    want_gist = (gist_horizons is not None
                 and getattr(args, "gist_loss_weight", 0.0) > 0.0)
    if args.aux_brackets:
        if want_gist:
            raise SystemExit("--gist_loss_weight is not supported "
                             "together with --aux_brackets.")
        logits, aux_logits = fwd_model(x, return_aux=True, **_fwd_kw)
        depth = bracket_depth(x, bracket_deltas).clamp(0, args.aux_max_depth)
        aux_loss = F.cross_entropy(
            aux_logits.reshape(-1, args.aux_max_depth + 1),
            depth.reshape(-1),
        )
        gist_loss = logits.new_zeros(())
    elif want_gist:
        # The model computes the gist loss inside the (compiled) forward
        # and returns it as a scalar — see TinyLM._finalize. This keeps
        # the hidden state from ever crossing the compile boundary.
        logits, gist_loss = fwd_model(x, **_fwd_kw)
        aux_loss = logits.new_zeros(())
    else:
        logits = fwd_model(x, **_fwd_kw)
        aux_loss = logits.new_zeros(())
        gist_loss = logits.new_zeros(())
    V = logits.shape[-1]
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, V), y.reshape(-1), reduction="none",
    ).reshape(y.shape)                                                   # (B, T)
    if args.output_gate:
        g = model._last_gate                                             # (B, T)
        if args.gate_warmup_steps > 0:
            progress = min(1.0, step / args.gate_warmup_steps)
            gate_floor = (1.0 - progress) * 1.0 + progress * args.gate_floor_min
        else:
            gate_floor = args.gate_floor_min
        g_eff = g.clamp(min=gate_floor) if gate_floor > 0.0 else g
        # Ponder cost. The CE term uses the floor-clamped g_eff (so real-token
        # loss keeps weight >= floor — the gate_floor_min anti-collapse fix).
        # The THINK-cost term, however, must penalise the RAW gate g, not g_eff:
        # clamp() zeros the gradient to g whenever g < floor, so with the cost
        # on g_eff the *raw* gate (the one generation uses, with no clamp) is
        # never penalised for thinking -> it over-thinks at deploy (think_frac
        # ~0.6 observed). Costing the raw g restores that gradient so the gate
        # learns to emit unless CE > gate_lambda. Default off (backwards-compat).
        ponder_gate = g if getattr(args, "gate_ponder_raw", False) else g_eff
        gate_terms = g_eff * ce_per_token + (1.0 - ponder_gate) * args.gate_lambda
        valid = (y != -100).float()
        denom = valid.sum().clamp(min=1.0)
        lm_loss = (gate_terms * valid).sum() / denom
    else:
        valid = (y != -100).float()
        denom = valid.sum().clamp(min=1.0)
        lm_loss = (ce_per_token * valid).sum() / denom
    # Entropy-grounded gate target (CE-reduction self-reward, cheap form).
    gate_aux_loss = torch.zeros((), device=logits.device)
    if args.output_gate and args.gate_entropy_aux_weight > 0.0:
        gate_logits = model._last_gate_logits                            # (B, T)
        # logsumexp + p·logp = stable entropy. We detach the source logits
        # because the target is a SELF-supervised signal — gradient must
        # flow into the gate head, not into the LM head.
        lse = torch.logsumexp(logits.detach(), dim=-1)                   # (B, T)
        # H_t = lse - sum(p * raw_logit) = lse - mean over support
        # using p = softmax(logits): H = lse - sum(p*logits)
        p = (logits.detach() - lse.unsqueeze(-1)).exp()                  # (B, T, V)
        H = lse - (p * logits.detach()).sum(dim=-1)                      # (B, T) ≥ 0
        T = max(args.gate_entropy_aux_temperature, 1e-6)
        target = torch.exp(-H / T).clamp(0.0, 1.0)                       # (B, T)
        c = args.gate_entropy_aux_target_clamp
        if c > 0.0:
            target = target.clamp(c, 1.0 - c)
        # BCE-with-logits over valid (non-ignored) positions only.
        valid = (y != -100).float()
        bce = F.binary_cross_entropy_with_logits(
            gate_logits, target, reduction="none",
        )
        denom = valid.sum().clamp(min=1.0)
        gate_aux_loss = (bce * valid).sum() / denom
    # Logit-distillation (KD) from a frozen teacher. The teacher shares the
    # SmolLM2 tokenizer so ids in `x` align; we slice the student logits to the
    # teacher vocab BEFORE the KL so the extra/thinking slots are dropped (the
    # load-bearing correctness detail). KD only on real (non-ignored) positions
    # -- and, when `doc_ids` is available, only on positions whose TARGET is in
    # the same document (see `_kd_valid_mask` / `_same_doc_target_mask` above).
    kd_loss = torch.zeros((), device=logits.device)
    if (kd_logit_store is not None
            and getattr(args, "distill_weight", 0.0) > 0.0):
        # OFFLINE KD: read the teacher's top-k for the NEXT x.numel() tokens of
        # the precomputed stream. The cursor advances exactly once per call, so
        # it stays locked to the data iterator (which is also advanced once per
        # microbatch by the caller).
        B, T = x.shape
        n = B * T
        t_ids, t_logits, t_input_ids = kd_logit_store.next_block(n)
        dev = logits.device
        t_ids = t_ids.to(dev).view(B, T, -1)                 # (B,T,k) int64
        t_logits = t_logits.to(dev).view(B, T, -1)           # (B,T,k) fp16
        t_input_ids = t_input_ids.to(dev).view(B, T)         # (B,T) int64
        # ----- ALIGNMENT SAFETY (load-bearing) -----
        # The store row order MUST equal the trainer's flattened token order.
        # If it doesn't, KD would distil onto the WRONG positions — a silent,
        # catastrophic corruption. Assert the stored input_ids match the live
        # tokens for this block; fail loudly with the first mismatch otherwise.
        if not torch.equal(t_input_ids, x.to(torch.int64)):
            mism = (t_input_ids != x.to(torch.int64))
            idx = int(mism.flatten().nonzero(as_tuple=False)[0].item())
            bi, ti = idx // T, idx % T
            raise RuntimeError(
                "Offline-KD alignment FAILED: the teacher logit store is out of "
                "sync with the training token stream (the #1 silent-corruption "
                f"mode). First mismatch at flat index {idx} (batch {bi}, pos "
                f"{ti}): store input_id={int(t_input_ids[bi, ti])} vs live "
                f"x={int(x[bi, ti])}. The store and trainer must use the SAME "
                "data_mix / seed / T / batch / num_workers / think_burst_prob. "
                f"Store: {getattr(kd_logit_store, 'tokenizer_name', '?')} / "
                f"{getattr(kd_logit_store, 'teacher_model', '?')}.")
        valid_kd = _kd_valid_mask(y, doc_ids, kd_thinking_token_id)
        kd_loss = _kd_loss_term_topk(logits, t_ids, t_logits,
                                     valid_kd.float(), args.distill_temp)
    elif kd_teacher is not None and getattr(args, "distill_weight", 0.0) > 0.0:
        # The data stream may insert thinking_token_id (== base vocab size, i.e.
        # one PAST the teacher's last valid id) into the INPUTS when
        # --think_burst_prob>0. Clamp any such id down to a valid teacher id so
        # the teacher's embedding lookup never indexes OOB (a CUDA assert). These
        # positions are excluded from the KD loss by valid_kd below anyway, and
        # for --think_burst_prob 0 (the A/B) no id exceeds the cap → x unchanged.
        x_teacher = x
        if kd_thinking_token_id is not None:
            x_teacher = x.clamp(max=int(kd_thinking_token_id) - 1)
        # Cross-document isolation (2026-07-01 fix): when the data stream
        # carries doc_ids, forward the teacher PER-DOCUMENT (see
        # `_kd_teacher_forward_doc_isolated`) instead of over the whole packed
        # block, so the teacher's soft targets don't see context the student's
        # cu_seqlens-reset DeltaNet state never had access to. `doc_ids is
        # None` (non-data_mix streams) keeps the original single full-block
        # forward — byte-identical to pre-fix behaviour.
        if doc_ids is not None:
            teacher_logits = _kd_teacher_forward_doc_isolated(
                kd_teacher, x_teacher, doc_ids)                  # (B,T,Vt)
        else:
            with torch.no_grad():
                t_out = kd_teacher(input_ids=x_teacher)
                teacher_logits = getattr(t_out, "logits", t_out)  # (B,T,Vt)
        V_t = teacher_logits.size(-1)
        s = logits[..., :V_t]                                            # slice
        valid_kd = _kd_valid_mask(y, doc_ids, kd_thinking_token_id)
        kd_loss = _kd_loss_term(s, teacher_logits, valid_kd.float(),
                                args.distill_temp)
    return (logits, ce_per_token, lm_loss, aux_loss, gate_aux_loss,
            gist_loss, kd_loss)


def _z_loss_term(logits, weight):
    """z-loss regulariser: weight * mean(logsumexp(logits)^2)."""
    if weight <= 0.0:
        return logits.new_zeros(())
    return weight * (torch.logsumexp(logits, dim=-1) ** 2).mean()


def _kd_loss_term(student_logits, teacher_logits, valid_mask, temp):
    """Hinton logit-distillation: T^2 * KL(teacher || student) over valid
    (non-ignored) positions, mean-reduced.

    `student_logits` must ALREADY be sliced to the teacher vocab (same last
    dim as `teacher_logits`) — the load-bearing correctness detail that drops
    the student's extra / thinking-token slots so the two vocabs align.

    teacher_logits is the TARGET distribution. We use
    F.kl_div(log_softmax(s/T), softmax(teacher/T), reduction='none').sum(-1)
    which computes per-position KL(teacher || student); ×T² (Hinton) and mean
    over valid positions. fp32 throughout for numerical stability.

    valid_mask: (B, T) float in {0, 1}. Returns a scalar.
    """
    T = max(float(temp), 1e-6)
    s_logp = F.log_softmax(student_logits.float() / T, dim=-1)        # (B,T,Vt)
    t_p = F.softmax(teacher_logits.float() / T, dim=-1)               # (B,T,Vt)
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(dim=-1)         # (B, T)
    denom = valid_mask.sum().clamp(min=1.0)
    return (T * T) * (kl * valid_mask).sum() / denom


def _kd_loss_term_topk(student_logits, teacher_ids, teacher_topk_logits,
                       valid_mask, temp):
    """OFFLINE-KD analog of `_kd_loss_term`: the teacher distribution is given
    only as TOP-K (ids + raw logits) from disk, so the KL is computed over the
    teacher's top-k SUPPORT instead of the full vocab.

    - `student_logits`   (B, T, V): the student's FULL next-token logits.
    - `teacher_ids`      (B, T, k): the teacher's top-k token ids (int).
    - `teacher_topk_logits` (B, T, k): the teacher's RAW top-k logits.
    - `valid_mask`       (B, T) float in {0,1}.

    The student is GATHERED at the teacher's top-k ids, then both the teacher
    top-k logits and the gathered-student logits are softmax-renormalised over
    the k-subset (so they are proper distributions over the same support) at
    temperature T; the per-position KL(teacher || student) is summed over k,
    ×T² (Hinton), mean over valid positions. fp32 throughout.
    """
    Tmp = max(float(temp), 1e-6)
    s_gather = torch.gather(student_logits.float(), -1,
                            teacher_ids.long())                  # (B,T,k)
    s_logp = F.log_softmax(s_gather / Tmp, dim=-1)               # renorm over k
    t_p = F.softmax(teacher_topk_logits.float() / Tmp, dim=-1)   # renorm over k
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(dim=-1)     # (B, T)
    denom = valid_mask.sum().clamp(min=1.0)
    return (Tmp * Tmp) * (kl * valid_mask).sum() / denom


# ---------------------------------------------------------------------------
# KD doc-isolation (audited defect fix, 2026-07-01): the student forward is
# cross-document-isolated (doc_ids -> cu_seqlens resets the DeltaNet
# recurrent state at every in-block document boundary, see the "Cross-
# document state isolation" AGENTS.md section), but a naive teacher forward
# over the whole packed T=2048 block is NOT isolated — a full-attention HF
# model conditions every position on the entire block, so its prediction for
# the first tokens of doc N+1 sees doc N's content, which the student never
# gets to see. That mismatch makes the "soft target" systematically easier
# than what the student is actually being asked to match. The two helpers
# below (a) reproduce the student's per-document isolation in the teacher
# forward, and (b) mask out the one position per document boundary where even
# a correctly-isolated teacher forward has nothing meaningful to predict (the
# LAST token of a document predicts the FIRST token of the next, unrelated,
# document — "meaningless" regardless of how the teacher forward is computed).

def _same_doc_target_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Boolean (B, T) mask: True where the TARGET position is in the SAME
    document as the source position `t` — i.e. where a KD signal at `t` is
    meaningful. False exactly at each document's LAST token (its target is
    the first token of the next document).

    `doc_ids` is aligned with the INPUT stream `x` (data_mix.py:
    `doc_ids[t]` is the id of `x[t]`), and the target `y[t]` is the token
    that followed `x[t]` in the original packed chunk — i.e. the token whose
    doc id is `doc_ids[t + 1]` (documents are packed contiguously with
    monotonically non-decreasing ids per row, see MixedSourceStream.__iter__).
    So `doc_ids[t] == doc_ids[t + 1]` is exactly "same doc as target".

    The chunk's very last input position (t = T-1) predicts a token whose own
    doc id was never carried through the (x, y, doc_ids) tuple returned by
    the data stream (the source `chunk_docids` has T+1 entries; `doc_ids`
    only keeps the first T, aligned with `x` — see data_mix.py's
    `chunk_docids[:-1]`). We default that one column to True (same-doc): it's
    1 of T positions per row, and defaulting it to "meaningless" would drop a
    real KD signal far more often than it right about a boundary — no
    downstream correctness invariant depends on this one column the way the
    interior-boundary mask does.
    """
    same_interior = doc_ids[:, :-1] == doc_ids[:, 1:]           # (B, T-1)
    last_col = torch.ones_like(doc_ids[:, :1], dtype=torch.bool)
    return torch.cat([same_interior, last_col], dim=1)          # (B, T)


def _kd_valid_mask(y: torch.Tensor, doc_ids: torch.Tensor | None,
                   kd_thinking_token_id: int | None) -> torch.Tensor:
    """Shared valid-position mask for BOTH the live-teacher and offline-store
    KD branches: real targets (not -100), not a thinking-token target, and
    (when doc_ids is available) not a target that crosses a document
    boundary. `doc_ids is None` (non-data_mix streams / eval / MQAR) skips
    the same-doc term entirely -> byte-identical to the pre-doc-isolation
    mask.

    NOTE for the offline logit-store path: the store's teacher logits were
    (or, as of the parallel doc-isolation fix to the offline generators, will
    be) themselves computed doc-isolated at generation time — but the SAME
    "last token of a doc predicts the first token of an unrelated doc" issue
    applies regardless of how the teacher forward was computed, so this mask
    is applied there too. This function is the single shared convention both
    the live and offline consumers must use.
    """
    valid_kd = (y != -100)
    if kd_thinking_token_id is not None:
        valid_kd = valid_kd & (y != int(kd_thinking_token_id))
    if doc_ids is not None:
        valid_kd = valid_kd & _same_doc_target_mask(doc_ids)
    return valid_kd


def _doc_segments(doc_ids: torch.Tensor) -> list[list[tuple[int, int]]]:
    """Per row, the list of (start, length) contiguous same-document runs
    covering [0, T) in order. `doc_ids` (B, T) is assumed non-decreasing
    along T within a row (data_mix.py packs documents contiguously with
    monotonically increasing per-row ids). CPU python loop — negligible next
    to a teacher forward pass (T*B simple int comparisons, no GPU sync since
    everything is pulled to CPU/list up front).
    """
    B, T = doc_ids.shape
    rows = doc_ids.detach().to("cpu").tolist()
    out = []
    for row in rows:
        segs = []
        start = 0
        for t in range(1, T):
            if row[t] != row[t - 1]:
                segs.append((start, t - start))
                start = t
        segs.append((start, T - start))
        out.append(segs)
    return out


def _kd_teacher_forward_doc_isolated(kd_teacher, x_teacher: torch.Tensor,
                                     doc_ids: torch.Tensor) -> torch.Tensor:
    """Forward `kd_teacher` with cross-document state isolation, matching the
    student's cu_seqlens doc reset (see the module-level comment above).

    Splits each row into its contiguous per-document segments and forwards
    every segment AS ITS OWN SEQUENCE starting at position 0 — i.e. exactly
    the forward a fresh, single-document context would produce (this is what
    makes the fix directly verifiable: it doesn't approximate "what a
    doc-isolated forward would look like" via an attention-mask / position-id
    trick, it just literally IS that forward). All segments in the microbatch
    are batched together in one teacher call, right-padded to the batch's
    longest segment: a causal decoder's logits at any REAL (non-pad) position
    never depend on trailing pad tokens (upper-triangular causal attention
    only looks backward, and RMSNorm/MLP are per-position), so right-padding
    is lossless here and needs no attention_mask/position_ids plumbing —
    verified empirically in test_kd_doc_isolation.py against a real
    standalone per-segment forward.

    Known cost/limitation (not fixed here, correctness-first): the padded
    segment-batch has as many rows as there are TOTAL documents in the
    microbatch, which can exceed the student batch size when documents are
    short; for the pretrain data_mix corpora (documents are typically
    hundreds+ of tokens, not single-digit) this stays a small multiple. If a
    future data mix skews toward very short documents, chunk this into
    several teacher calls to bound peak memory.
    """
    B, T = x_teacher.shape
    segs_per_row = _doc_segments(doc_ids)
    segs = [(b, s, l) for b, row_segs in enumerate(segs_per_row)
            for (s, l) in row_segs]
    L_max = max(l for _, _, l in segs)
    n = len(segs)
    seg_x = x_teacher.new_zeros((n, L_max))
    for i, (b, s, l) in enumerate(segs):
        seg_x[i, :l] = x_teacher[b, s:s + l]
    with torch.no_grad():
        t_out = kd_teacher(input_ids=seg_x,
                           attention_mask=torch.ones_like(seg_x))
        seg_logits = getattr(t_out, "logits", t_out)             # (n,Lmax,Vt)
    Vt = seg_logits.size(-1)
    teacher_logits = seg_logits.new_zeros((B, T, Vt))
    for i, (b, s, l) in enumerate(segs):
        teacher_logits[b, s:s + l] = seg_logits[i, :l]
    return teacher_logits


def _latent_reasoning_ramp(step: int, start_step: int, warmup_steps: int) -> float:
    """Linear 0->1 weight ramp for the latent-reasoning co-train.

    Returns 0.0 before `start_step`, then ramps linearly to 1.0 over
    `warmup_steps` after the start, clamped to [0, 1]. `warmup_steps <= 0`
    means full weight immediately at/after the start (byte-identical to the
    pre-ramp path). Keeps the aux gradient negligible while PKM bootstraps —
    the v12-destabilization safety knob.
    """
    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, (step - start_step) / warmup_steps))


def _latent_reasoning_aux_every_gate(step: int, start_step: int, every: int
                                     ) -> tuple[bool, float]:
    """Whether the latent-reasoning aux fires THIS step, and the weight
    multiplier to apply when it does (`--latent_reasoning_aux_every`, K).

    Fires on step indices `start_step, start_step + K, start_step + 2K, ...`
    at weight multiplier K, so the EXPECTED per-step gradient contribution —
    averaged over any K-step window — is identical to firing every step at
    multiplier 1. This is pure variance-for-wall-clock trade (the aux's
    growing-thread forwards, even batched into one, are still the dominant
    per-step cost at ~25% GPU utilization in the sequential path this
    replaces), not a change to what's being optimised. `every <= 1` fires
    every step at multiplier 1 — byte-identical to the pre-aux_every
    behaviour. Steps before `start_step` never fire (multiplier is then
    irrelevant to the caller, returned as 1.0 for a defined value).
    """
    if step < start_step:
        return False, 1.0
    k = max(1, int(every))
    return ((step - start_step) % k == 0), float(k)


# _pkm_diversity_loss REMOVED 2026-06-18 (--pkm_diversity_weight dropped):
# it was an inert NO-OP — it operated on the STASHED *detached* slot indices +
# weights, so it had zero autograd path to any parameter (measured 0% gradient
# share on the v17 ckpt). PKM slot diversity is actually maintained by ε-greedy
# random slot replacement + LayerNorm score-norm + the 100× value-LR — not this
# loss. See memory project_pkm_diversity_inert / LOSS_BALANCE_REPORT.md.


def _ctx_addr_aux_loss(model, input_ids, mem_read_mask, doc_ids=None):
    """Attention-supervision aux for the LEARNED ctx_namekey WM addresser.

    Trains the ctx_namekey read attention to place mass on the SAME binding slot
    the deterministic lexical discrete-code identifies as correct (the teacher),
    at recall answer-span (mem_read_mask) positions. The discrete code is
    parameter-free + vectorized; the learned ctx addresser is the student
    (general at inference, no hash). Mirrors wm_recall_cotrain.py's addr_aux
    (cross-entropy on the read attention onto the queried binding), with the
    binding TARGET derived from the lexical code rather than per-example metadata
    (which the pretrain recall stream does not carry — it only carries the
    answer-span mem_read_mask the discrete path also uses).

    Gradient flows ONLY through the (grad-keeping) read attention; the code /
    vstart / masks are integer/bool teacher signals with no grad. Returns
    (loss, n_positions, mean_p_bind) or None when it cannot fire (ctx_namekey
    off, no mask, no stashed attention graph, or no recall position with a
    causally-valid matching binding in the buffer).
    """
    mem = getattr(model, "memory", None)
    if (mem is None or not getattr(mem, "ctx_namekey", False)
            or mem_read_mask is None):
        return None
    attn = getattr(mem, "_last_read_attn_grad", None)   # (B, T, K) w/ grad
    top_idx = getattr(mem, "_last_top_idx_buf", None)   # (B, K) src positions
    if attn is None or top_idx is None:
        return None
    recall = mem_read_mask.bool()
    if not bool(recall.any()):
        return None
    B, T, K = attn.shape
    device = attn.device
    # Deterministic lexical-code TEACHER (parameter-free, vectorized, on-device)
    # → the correct binding for each position. Construction mirrors the WM
    # discrete branch's `match` tensor exactly (model.py).
    code, vstart = mem._identifier_code_lexical(input_ids)        # (B,T),(B,T)
    Kc = int(getattr(mem, "discrete_key_K", 1 << 24))
    idx = (code + 1).clamp(0, Kc - 1)                             # (B,T) >=0
    zeros = torch.zeros_like(idx)
    key_idx = torch.where(vstart, idx, zeros)                     # write side
    qry_idx = torch.where(code >= 0, idx, zeros)                  # read side
    buf_key_idx = torch.gather(key_idx, 1, top_idx)              # (B, K)
    match = ((qry_idx.unsqueeze(-1) == buf_key_idx.unsqueeze(1))
             & (qry_idx.unsqueeze(-1) > 0))                      # (B, T, K) bool
    # Causal + same-document validity: only count target slots the attention
    # could actually attend (the softmax already masked these to ~0, so this
    # keeps the teacher consistent with the student's support).
    pos = torch.arange(T, device=device).view(1, T, 1)
    src = top_idx.unsqueeze(1)                                    # (B, 1, K)
    valid_slot = src < pos                                        # causal
    if doc_ids is not None:
        buf_doc = torch.gather(doc_ids, 1, top_idx)              # (B, K)
        valid_slot = valid_slot & (buf_doc.unsqueeze(1)
                                   == doc_ids.unsqueeze(-1))
    target = match & valid_slot                                   # (B, T, K)
    has_target = target.any(dim=-1) & recall                     # (B, T)
    if not bool(has_target.any()):
        return None
    p_bind = (attn * target.to(attn.dtype)).sum(dim=-1).clamp_min(1e-9)  # (B,T)
    sel = p_bind[has_target]
    addr = -(torch.log(sel)).mean()
    return addr, int(has_target.sum()), float(sel.mean().detach())


# --- Engagement kill-gates (SESSION_FINDINGS.md 2026-07-02 "recipe rules") --
# The 2026-07-02 forensics pass found that most "features are orthogonal to
# code" verdicts rested on runs where the mechanisms were numerically INERT
# the whole time (PKM alpha stuck at ~init, WM copy-gate never fired, the
# latent aux never even entered its construction branch) — a wasted launch
# masquerading as a negative result. Two independent guards close this hole:
#   1. curricula-must-fit (`validate_curricula_fit`): a warmup/curriculum
#      that can't finish inside the run either wastes the whole run on
#      cold-start (latent) or never reaches a verdict (everything else).
#      Checked ONCE at startup, pure-args, before any GPU/model work.
#   2. engagement kill-gate (`engagement_report` + the per-mechanism
#      evaluators below): at --engagement_check_step, verify each ENABLED
#      mechanism has actually MOVED (not just that its module exists — the
#      separate STARTUP CONSTRUCTION ASSERTS in main() catch the "the
#      construction branch was silently not entered" half of the failure).


def _curriculum_end_frac(steps: int, start: int, warmup: int) -> float:
    """Fraction of the run consumed before a (start, warmup) curriculum
    finishes ramping. >1.0 means the curriculum never completes inside the
    run; steps<=0 returns 0.0 (nothing to validate)."""
    if steps <= 0:
        return 0.0
    return (max(0, int(start)) + max(0, int(warmup))) / float(steps)


def validate_curricula_fit(args) -> None:
    """Curricula-must-fit rule (recipe rule #2). Pure-args; call once at
    startup before any GPU/model work. Two severities:
      - latent_reasoning is HARD-ERRORed past 40% of the run: its cold-start
        destabilizes pretrain when it engages late/abruptly (gnorm spike,
        Δlogp<0 on engagement — see the memory note
        project_cold_latent_cotrain_destabilizes_pretrain, 2026-06-14).
      - everything else (PKM epsilon/floor warmup, ctx_addr aux warmup,
        feedback_self_k warmup, the output-gate floor warmup) WARNS loudly
        past 40% of --steps and ABORTS past 100% (a curriculum that
        literally never finishes before the run ends is a config bug, not a
        slow ramp).
    """
    steps = int(getattr(args, "steps", 0))
    if steps <= 0:
        return

    if float(getattr(args, "latent_reasoning_weight", 0.0)) > 0.0:
        frac = _curriculum_end_frac(
            steps,
            int(getattr(args, "latent_reasoning_start_step", 0)),
            int(getattr(args, "latent_reasoning_weight_warmup_steps", 0)))
        if frac > 0.4:
            raise SystemExit(
                "[curricula-fit] ABORT: --latent_reasoning_start_step + "
                f"--latent_reasoning_weight_warmup_steps ends at "
                f"{frac * 100:.0f}% of --steps={steps} (> 40%). Cold-start "
                "latent co-training destabilizes pretrain when it engages "
                "late/abruptly (gnorm spike, Δlogp<0 on engagement — "
                "SESSION_FINDINGS.md 2026-06-14). Shorten the warmup, start "
                "it earlier, or lengthen the run.")

    # (cli_label, enabled, start_step, warmup_steps)
    checks: list[tuple[str, bool, int, int]] = []
    if getattr(args, "use_pkm", False):
        checks.append((
            "--pkm_epsilon_warmup_steps",
            float(getattr(args, "pkm_epsilon_start", 0.0)) > 0.0,
            0, int(getattr(args, "pkm_epsilon_warmup_steps", 0))))
        checks.append((
            "--pkm_alpha_floor_warmup_steps",
            float(getattr(args, "pkm_alpha_floor_start", 0.0)) > 0.0,
            0, int(getattr(args, "pkm_alpha_floor_warmup_steps", 0))))
    if float(getattr(args, "ctx_addr_aux_weight", 0.0)) > 0.0:
        checks.append((
            "--ctx_addr_aux_start_step + --ctx_addr_aux_warmup_steps", True,
            int(getattr(args, "ctx_addr_aux_start_step", 0)),
            int(getattr(args, "ctx_addr_aux_warmup_steps", 0))))
    if int(getattr(args, "feedback_self_k_warmup_steps", 0)) > 0:
        checks.append((
            "--feedback_self_k_warmup_steps", True,
            0, int(getattr(args, "feedback_self_k_warmup_steps", 0))))
    if bool(getattr(args, "output_gate", False)):
        checks.append((
            "--gate_warmup_steps", True,
            0, int(getattr(args, "gate_warmup_steps", 0))))

    aborts, warns = [], []
    for label, enabled, start, warmup in checks:
        if not enabled:
            continue
        frac = _curriculum_end_frac(steps, start, warmup)
        if frac > 1.0:
            aborts.append(f"{label} ends at {frac * 100:.0f}% of "
                          f"--steps={steps} (> 100% — never finishes)")
        elif frac > 0.4:
            warns.append(f"{label} ends at {frac * 100:.0f}% of "
                         f"--steps={steps} (> 40%)")
    for w in warns:
        print(f"[curricula-fit] WARNING: {w}. A curriculum attached this "
              "late may still be bootstrapping when the run ends.",
              flush=True)
    if aborts:
        raise SystemExit(
            f"[curricula-fit] ABORT: the following curricula never finish "
            f"inside --steps={steps}:\n  - " + "\n  - ".join(aborts) +
            "\nShorten the warmup(s), start them earlier, or lengthen the "
            "run.")


def _max_abs_alpha(alphas: list) -> float:
    """Extract max |alpha| from `TinyLM.feedback_alphas()`'s format-
    polymorphic output (sparse/xattn (target,source,alpha) tuples,
    dense-single flat floats, dense-multi per-layer lists — see that
    method's docstring). Returns 0.0 for an empty list (feedback off)."""
    vals: list[float] = []
    for a in alphas:
        if isinstance(a, tuple):
            vals.append(abs(float(a[-1])))
        elif isinstance(a, list):
            vals.extend(abs(float(x)) for x in a)
        else:
            vals.append(abs(float(a)))
    return max(vals) if vals else 0.0


def _pkm_engaged(alpha_l: float, row_ratio: float, *,
                  alpha_min: float, row_min: float) -> tuple[bool, str]:
    """PKM engagement verdict, reusing the pkm(αL=...,row=...) live
    diagnostic values (never recomputed). `alpha_l` may be nan when
    `pkm.use_output_gate` is off — nan comparisons are False, so that just
    falls through to the row-ratio check."""
    ok = (abs(alpha_l) >= alpha_min) or (row_ratio >= row_min)
    detail = (f"|alphaL|={abs(alpha_l):.4f} (min {alpha_min}), "
              f"row_ratio={row_ratio:.3f} (min {row_min})")
    return ok, detail


def _film_engaged(alphas: list, *, alpha_min: float) -> tuple[bool, str]:
    max_a = _max_abs_alpha(alphas)
    ok = max_a >= alpha_min
    detail = f"max|alpha| over pairs/layers = {max_a:.4g} (min {alpha_min})"
    return ok, detail


def _wm_copy_engaged(bias_now: float, bias_init: float, fire_count: int, *,
                      bias_delta_min: float) -> tuple[bool, str]:
    delta = abs(bias_now - bias_init)
    ok = (delta >= bias_delta_min) or (fire_count > 0)
    detail = (f"gate_bias={bias_now:.4f} (init {bias_init:.4f}, "
              f"|Δ|={delta:.4f}, min {bias_delta_min}), "
              f"cumulative_match_fires={fire_count}")
    return ok, detail


def _latent_engaged(constructed: bool, fire_count: int, all_finite: bool, *,
                     step: int, start_step: int, end_step: int
                     ) -> tuple[bool, str]:
    """`end_step` = start_step + weight_warmup_steps (full-weight point). If
    the check runs before the curriculum has even reached full weight, we
    can only verify `constructed` (the real judgement is deferred) — return
    a NOTE-tagged pass rather than a false failure."""
    if end_step > step:
        return True, (
            f"NOTE: constructed={constructed}; engagement scheduled to ramp "
            f"{start_step}->{end_step}, step {step} is still inside that "
            "window — not evaluated as a failure yet.")
    ok = constructed and fire_count > 0 and all_finite
    detail = (f"constructed={constructed}, fires_since_start={fire_count}, "
              f"all_finite={all_finite}")
    return ok, detail


def engagement_report(step: int, results: list[tuple[str, bool, str]], *,
                       action: str, is_main: bool = True) -> None:
    """Print the loud multi-line engagement report and abort/warn.

    `results` is (mechanism_name, engaged, detail) for every mechanism that
    was actually CHECKED — a mechanism whose flag is off is simply absent
    (an off feature made no claim the run needs to keep).

    The abort/warn DECISION is computed identically regardless of `is_main`
    (every DDP rank has in-sync model state, so every rank reaches the same
    verdict) — only the printing is gated on `is_main`, so 'abort' kills the
    whole distributed job together instead of hanging one dead rank while
    the others wait at the next collective op.
    """
    inert = [r for r in results if not r[1]]
    if is_main:
        print("=" * 78, flush=True)
        if results:
            print(f"[engagement-check] step {step}: "
                  f"{len(results) - len(inert)}/{len(results)} mechanism(s) "
                  "engaged", flush=True)
        else:
            print(f"[engagement-check] step {step}: no mechanisms enabled "
                  "to check", flush=True)
        for name, engaged, detail in results:
            status = "ENGAGED" if engaged else "INERT"
            print(f"  [{status}] {name}: {detail}", flush=True)
        print("=" * 78, flush=True)
    if not inert:
        return
    names = ", ".join(r[0] for r in inert)
    msg = (f"[engagement-check] step {step}: {len(inert)} mechanism(s) "
           f"still INERT: {names}. A feature run that ends inert is a "
           "wasted launch, not a negative result (SESSION_FINDINGS.md "
           "2026-07-02).")
    if action == "abort":
        raise SystemExit(msg)
    if is_main:
        print(f"[engagement-check] WARNING (action=warn, continuing past "
              f"inert mechanisms): {msg}", flush=True)


# --- Engagement kill-gate, STARTUP construction asserts (recipe rule #1) ---
# Extracted into small testable functions (rather than left inline in
# main()) so a synthetic model/args pair missing an expected attribute can
# be exercised directly in tests, without needing a real TinyLM/GPU. Each
# is a no-op when its flag is off, and asserts when the flag is on but the
# corresponding module was never attached — the exact "construction branch
# silently not entered" failure mode from the Phase-1 arm-B autopsy.
def assert_pkm_constructed(args, model) -> None:
    if not getattr(args, "use_pkm", False):
        return
    assert getattr(model, "pkm_layer", None) is not None, (
        "--use_pkm is set but model.pkm_layer was never constructed — "
        "the PKM branch in TinyLM.__init__ was silently skipped.")


def assert_memory_constructed(args, model) -> None:
    if not getattr(args, "use_memory", False):
        return
    assert getattr(model, "memory", None) is not None, (
        "--use_memory is set but model.memory was never constructed — "
        "the WorkingMemory branch in TinyLM.__init__ was silently skipped.")


def assert_sparse_feedback_constructed(args, model) -> None:
    if not (getattr(args, "feedback", "none") == "film"
            and getattr(args, "feedback_pairs", "")):
        return
    assert (getattr(model, "sparse_feedback", None) is not None
            and len(model.sparse_feedback) > 0), (
        "--feedback film --feedback_pairs '...' is set but "
        "model.sparse_feedback was never constructed (or is empty) — the "
        "sparse-FiLM branch in TinyLM.__init__ was silently skipped.")


def assert_latent_reasoner_constructed(args, latent_reasoner) -> None:
    if not (float(getattr(args, "latent_reasoning_weight", 0.0)) > 0.0):
        return
    assert latent_reasoner is not None, (
        "--latent_reasoning_weight > 0 but _latent_reasoner was never "
        "constructed — the construction branch above was silently skipped.")


class TokenisedStream(IterableDataset):
    """Streaming IterableDataset of fixed-length tokenised chunks."""

    def __init__(self, dataset, tokenizer, block_size, text_field="text",
                 shuffle_buffer=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        buf: list[int] = []
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.bos_token_id
        if eos is None:
            eos = 0
        for example in self.dataset:
            text = example[self.text_field]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= self.block_size + 1:
                chunk = buf[: self.block_size + 1]
                buf = buf[self.block_size :]
                inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield inputs, targets


def main():
    from experiments.train_lm_args import build_parser
    p = build_parser()
    args = p.parse_args()

    # --- DDP (batch-parallel, grads-only all-reduce; no NVLink needed) -------
    # Activated when launched under torchrun (WORLD_SIZE>1). Each rank holds a
    # full model copy on its own GPU and processes a disjoint data shard; only
    # gradients cross the PCIe bus (NCCL all-reduce). world_size==1 (plain
    # `python ...`) is byte-identical to the legacy single-GPU path.
    import os as _os
    ddp_world_size = int(_os.environ.get("WORLD_SIZE", "1"))
    ddp_rank = int(_os.environ.get("RANK", "0"))
    ddp_local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    is_ddp = ddp_world_size > 1
    is_main = ddp_rank == 0
    if is_ddp:
        if args.enable_thinking_token:
            raise SystemExit(
                "DDP (WORLD_SIZE>1) is wired for the non-thinking-token "
                "(pretrain) path only; the thinking-token queue path is not "
                "DDP-safe (per-rank dynamic queues desync). Run it single-GPU."
            )
        torch.cuda.set_device(ddp_local_rank)
        torch.distributed.init_process_group(backend="nccl")
        print(f"[ddp] rank {ddp_rank}/{ddp_world_size} "
              f"local_rank={ddp_local_rank} device=cuda:{ddp_local_rank}",
              flush=True)

    def _mainprint(*a, **k):
        if is_main:
            print(*a, **k)

    if args.enable_thinking_token and args.output_gate:
        raise SystemExit(
            "--enable_thinking_token and --output_gate are mutually exclusive. "
            "Use --think_decision gate to train the discrete THINKING path "
            "with a binary gate head."
        )
    if args.enable_thinking_token and args.think_lambda < 0:
        raise SystemExit("--think_lambda must be non-negative.")
    if (args.enable_thinking_token and args.think_lambda_start is not None
            and args.think_lambda_start < 0):
        raise SystemExit("--think_lambda_start must be non-negative.")
    if args.enable_thinking_token and args.think_curriculum_steps < 0:
        raise SystemExit("--think_curriculum_steps must be non-negative.")
    if args.enable_thinking_token and args.think_queue_max <= 0:
        raise SystemExit("--think_queue_max must be positive.")
    if args.enable_thinking_token and not (0.0 <= args.think_explore_prob <= 1.0):
        raise SystemExit("--think_explore_prob must be in [0, 1].")
    if args.enable_thinking_token and not (0.0 <= args.think_explore_start_prob <= 1.0):
        raise SystemExit("--think_explore_start_prob must be in [0, 1].")
    if args.enable_thinking_token and args.think_min_fresh_rows < 0:
        raise SystemExit("--think_min_fresh_rows must be non-negative.")
    if args.enable_thinking_token and not (0.0 < args.think_gate_threshold < 1.0):
        raise SystemExit("--think_gate_threshold must be in (0, 1).")
    if (args.enable_thinking_token and args.think_gate_threshold_start is not None
            and not (0.0 < args.think_gate_threshold_start < 1.0)):
        raise SystemExit("--think_gate_threshold_start must be in (0, 1).")
    if args.enable_thinking_token and args.think_max_new_per_step < 0:
        raise SystemExit("--think_max_new_per_step must be non-negative.")
    if args.enable_thinking_token and args.think_safety_max_depth < 0:
        raise SystemExit("--think_safety_max_depth must be non-negative.")
    if (args.enable_thinking_token and args.think_safety_max_depth_start is not None
            and args.think_safety_max_depth_start < 0):
        raise SystemExit("--think_safety_max_depth_start must be non-negative.")
    if args.enable_thinking_token and args.think_replay_weight < 0:
        raise SystemExit("--think_replay_weight must be non-negative.")
    if args.enable_thinking_token and args.think_aux_loss_scale < 0:
        raise SystemExit("--think_aux_loss_scale must be non-negative.")
    if args.grad_accum < 1:
        raise SystemExit("--grad_accum must be >= 1.")
    if args.grad_accum > 1 and args.enable_thinking_token:
        raise SystemExit(
            "--grad_accum > 1 is only supported on the non-thinking-token "
            "(pretrain) path; the thinking-token path has its own "
            "--think_queue_accum_steps."
        )
    if args.enable_thinking_token and args.think_queue_accum_steps < 0:
        raise SystemExit("--think_queue_accum_steps must be non-negative.")
    if args.enable_thinking_token and args.think_queue_accum_max_steps < 0:
        raise SystemExit("--think_queue_accum_max_steps must be non-negative.")
    if (args.enable_thinking_token and args.think_queue_accum_max_steps > 0
            and args.think_queue_accum_max_steps < args.think_queue_accum_steps):
        raise SystemExit(
            "--think_queue_accum_max_steps must be >= --think_queue_accum_steps."
        )
    if args.enable_thinking_token and args.think_queue_drain_target < -1:
        raise SystemExit("--think_queue_drain_target must be >= -1.")
    if args.enable_thinking_token and args.think_backpressure_target < -1:
        raise SystemExit("--think_backpressure_target must be >= -1.")
    if args.enable_thinking_token and args.think_backpressure_max < 0:
        raise SystemExit("--think_backpressure_max must be non-negative.")
    if args.enable_thinking_token and args.think_backpressure_lambda < 0:
        raise SystemExit("--think_backpressure_lambda must be non-negative.")
    if args.enable_thinking_token and args.think_backpressure_threshold < 0:
        raise SystemExit("--think_backpressure_threshold must be non-negative.")
    if args.enable_thinking_token and args.think_backpressure_explore < 0:
        raise SystemExit("--think_backpressure_explore must be non-negative.")
    if args.enable_thinking_token and args.think_gate_emit_weight < 0:
        raise SystemExit("--think_gate_emit_weight must be non-negative.")
    if args.enable_thinking_token and args.aux_brackets:
        raise SystemExit(
            "--enable_thinking_token currently supports the standard LM loss "
            "path only; disable aux losses for thinking experiments."
        )
    # Logit-KD is implemented only on the non-thinking-queue pretrain path
    # (the teacher forward + KL live in _nonthink_forward_loss). The think-queue
    # path has a different loss structure; rather than silently no-op KD there,
    # fail clearly.
    _kd_live = bool(getattr(args, "distill_teacher_model", ""))
    _kd_offline = bool(getattr(args, "distill_logits_dir", ""))
    # LIVE and OFFLINE KD are mutually exclusive — they are two ways to obtain
    # the same teacher distribution; running both is a config error.
    if _kd_live and _kd_offline:
        raise SystemExit(
            "--distill_teacher_model (live teacher) and --distill_logits_dir "
            "(offline precomputed top-k store) are mutually exclusive. Pick one."
        )
    _kd_on = ((_kd_live or _kd_offline)
              and float(getattr(args, "distill_weight", 0.0)) > 0.0)
    if _kd_on and args.enable_thinking_token:
        raise SystemExit(
            "--distill_teacher_model/--distill_logits_dir/--distill_weight "
            "(logit-KD) is supported only on the non-thinking pretrain path; it "
            "is mutually exclusive with --enable_thinking_token."
        )
    # OFFLINE KD is single-GPU only. The teacher store was generated against ONE
    # deterministic stream (base_seed=args.seed), but each DDP rank streams a
    # DISJOINT shard (base_seed = args.seed + ddp_rank*100003) while all ranks
    # open the same single-seed store at cursor 0 — so rank>=1's live tokens can
    # never match the store and the alignment assertion would abort mid-run.
    # Fail at startup with the fix instead of wasting a launch.
    if _kd_offline and is_ddp:
        raise SystemExit(
            "Offline KD (--distill_logits_dir) is single-GPU only: each DDP "
            f"rank streams a disjoint data shard (base_seed+rank*100003) but the "
            f"store is a single deterministic stream, so rank>=1 would desync "
            f"and abort. You launched under torchrun with WORLD_SIZE="
            f"{ddp_world_size}. Run offline KD without torchrun (single GPU), or "
            "generate one per-rank store per rank's seed and load the matching "
            "one — the latter is not yet implemented."
        )
    # OFFLINE KD requires num_workers in {0,1}. The teacher store was generated
    # against the num_workers=0 flat token order (a single deterministic stream).
    # Only worker counts 0 or 1 reproduce that order (worker 0 uses
    # base_seed+17*0 == base_seed); num_workers>=2 round-robins DISJOINT
    # per-worker sub-streams, so microbatch 2 (worker 1) desyncs and the
    # alignment assert aborts 1-2 steps in — wasting a multi-day launch. The
    # trainer default is num_workers=2, so fail loudly at startup instead.
    if _kd_offline and int(getattr(args, "num_workers", 2)) not in (0, 1):
        raise SystemExit(
            "Offline KD (--distill_logits_dir) requires --num_workers 0 (or 1): "
            "the store was generated at num_workers=0 and only 0/1 reproduce that "
            f"flat token order. You passed --num_workers {args.num_workers}, which "
            "would desync the alignment assert within a couple of steps. Re-run "
            "with --num_workers 0."
        )
    # Offline KD requires NO think-burst insertion: the teacher store was built
    # with bursts OFF (the teacher has no think token), so any burst the trainer
    # inserts would shift the token stream and break lockstep alignment. The
    # trainer default is --think_burst_prob 0.5, so force it to 0 here (with a
    # loud warning) rather than relying on the user to remember the override.
    if _kd_offline and float(getattr(args, "think_burst_prob", 0.0)) != 0.0:
        print(
            f"[offline-KD] WARNING: --think_burst_prob was "
            f"{args.think_burst_prob} but offline KD requires think-burst "
            "insertion OFF (the teacher store has no think tokens; bursts shift "
            "the token stream and break store↔trainer alignment). Forcing "
            "--think_burst_prob 0.0.",
            flush=True,
        )
        args.think_burst_prob = 0.0
    # LIVE KD has the same think-burst problem in a subtler form: bursts insert
    # think ids into the INPUTS, which the teacher forward clamps to a stand-in
    # real token (think_id-1). valid_kd masks only the burst positions
    # themselves, NOT the post-burst positions whose teacher logits were
    # conditioned on the corrupted context — so a default-flag launch
    # (--think_burst_prob 0.5) silently distills against wrong targets.
    # Mirror the offline-KD guard and force bursts off.
    if _kd_live and float(getattr(args, "think_burst_prob", 0.0)) != 0.0:
        print(
            f"[live-KD] WARNING: --think_burst_prob was "
            f"{args.think_burst_prob} but live KD requires think-burst "
            "insertion OFF (the teacher sees a stand-in token at burst "
            "positions, corrupting its context for every position after the "
            "burst). Forcing --think_burst_prob 0.0.",
            flush=True,
        )
        args.think_burst_prob = 0.0

    # Engagement kill-gate, part 1: curricula-must-fit (recipe rule #2). Pure
    # args, no GPU/model work yet — fail fast before wasting a launch.
    validate_curricula_fit(args)

    torch.manual_seed(args.seed)
    arch_label = args.arch if args.arch else f"layers={args.layers}"
    print(f"GPU: {torch.cuda.get_device_name(0)}  arch={arch_label}")

    # 1. Tokeniser + dataset.
    from transformers import AutoTokenizer
    from datasets import load_dataset
    print(f"Loading tokeniser {args.tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    thinking_token_id = None
    added_thinking_tokens = 0
    if args.enable_thinking_token:
        added_thinking_tokens = tok.add_special_tokens(
            {"additional_special_tokens": [args.thinking_token]}
        )
        thinking_token_id = tok.convert_tokens_to_ids(args.thinking_token)
        if thinking_token_id is None or thinking_token_id < 0:
            raise SystemExit(
                f"failed to add/resolve thinking token {args.thinking_token!r}"
            )
    if args.data_mix:
        # Mixed-corpus pretrain. Reserve the FIRST id ABOVE every real token
        # (base vocab AND any added special tokens) for the think token; round
        # model vocab up to a multiple-of-64 so embedding / lm_head dims are
        # GPU-friendly. Using max(vocab_size, len(tok)) is byte-identical for
        # SmolLM2 (len == vocab_size == 49152) but is REQUIRED for tokenizers
        # whose vocab_size slot is a real token — e.g. Qwen, where
        # tok.vocab_size==151643 is "<|endoftext|>" (also the pad token); using
        # it as the think id would alias a real token and make
        # state_readonly_at_think / WM / the gate fire on every EOS/pad position.
        if int(getattr(args, "keep_base_vocab", 0)) > 0:
            # INHERITED-BASE continuation: keep the (tied) embedding at its
            # original size so --load_ckpt restores embed/lm_head with ZERO
            # shape-mismatch. The data_mix default below would round vocab up to
            # round64(vocab+1) (e.g. 49152 -> 49216) to reserve an OUT-OF-RANGE
            # think slot, which would size-mismatch the inherited 49152 rows.
            # With --think_burst_prob 0 (+ --mem_always_read for WM) the discrete
            # think token is never emitted into the stream, so alias it to an
            # IN-RANGE id (EOS) and keep model_vocab == base vocab.
            if float(getattr(args, "think_burst_prob", 0.0)) != 0.0:
                raise SystemExit(
                    "--keep_base_vocab requires --think_burst_prob 0 (an in-range "
                    "think id + bursts would corrupt real tokens).")
            model_vocab_size = int(args.keep_base_vocab)
            thinking_token_id = (tok.eos_token_id
                                 if tok.eos_token_id is not None
                                 else model_vocab_size - 1)
            if int(thinking_token_id) >= model_vocab_size:
                thinking_token_id = model_vocab_size - 1
            thinking_token_id = int(thinking_token_id)
            # The EOS alias makes thinking_token_id == pad_token_id (SmolLM2
            # eos = 0 = the pad fallback), which SILENTLY disables the aux
            # losses guarded by `thinking_token_id != pad_token_id`
            # (gate_calibration, latent_cotrain). A run combining
            # --keep_base_vocab with those losses would train with the loss
            # never engaging and no warning — fail loudly instead.
            # (latent_reasoning_weight is NOT pad-guarded and runs fine under
            # the alias — launch_phase1_ab_B.sh legitimately combines them.)
            _think_losses = {
                "--gate_calibration_weight":
                    float(getattr(args, "gate_calibration_weight", 0.0)),
                "--latent_cotrain_weight":
                    float(getattr(args, "latent_cotrain_weight", 0.0)),
            }
            _on = [k for k, v in _think_losses.items() if v != 0.0]
            if _on:
                raise SystemExit(
                    f"--keep_base_vocab aliases thinking_token_id to EOS, "
                    f"which equals pad_token_id — the think-conditioned aux "
                    f"losses you enabled ({', '.join(_on)}) would be silently "
                    "skipped by their `thinking_token_id != pad_token_id` "
                    "guards. Use the reserved out-of-range think slot (drop "
                    "--keep_base_vocab) for these losses.")
        else:
            thinking_token_id = int(max(tok.vocab_size, len(tok)))
            model_vocab_size = ((thinking_token_id + 1 + 63) // 64) * 64
    else:
        model_vocab_size = len(tok)
    print(f"  vocab size: base={tok.vocab_size}, model={model_vocab_size}"
          f"{f', thinking_id={thinking_token_id}' if thinking_token_id is not None else ''}")

    if args.data_mix:
        print(f"Loading data mix from {args.data_mix} (streaming) ...")
        from experiments.data_mix import (
            MixedSourceStream, load_sources_from_yaml,
        )
        sources = load_sources_from_yaml(args.data_mix)
        print(f"  {len(sources)} sources:")
        for s in sources:
            print(f"    - {s.name:30s} weight={s.weight:.3f}  id={s.dataset_id}")
        train_ds = MixedSourceStream(
            sources=sources, tokenizer=tok, block_size=args.T,
            thinking_token_id=thinking_token_id,
            think_burst_prob=args.think_burst_prob,
            think_max_bursts=args.think_max_bursts,
            think_max_burst_depth=args.think_max_burst_depth,
            # Per-rank seed offset → each DDP rank streams a disjoint shard.
            base_seed=args.seed + ddp_rank * 100_003,
            mask_eos_in_targets=bool(args.mask_eos_in_targets),
            emit_doc_ids=True,
            # v14: emit the per-position recall read-mask as a 4th tuple element
            # (default off → 3-tuple, byte-identical to v12).
            emit_read_mask=bool(getattr(args, "emit_read_mask", False)),
        )
        # Val: same sources, different seed, burst injection off so val PPL
        # reflects the clean data distribution.
        val_ds = MixedSourceStream(
            sources=sources, tokenizer=tok, block_size=args.T,
            thinking_token_id=thinking_token_id,
            think_burst_prob=0.0,
            base_seed=args.seed + 999_983,
            mask_eos_in_targets=bool(args.mask_eos_in_targets),
            emit_doc_ids=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=1)
    else:
        print(f"Loading dataset {args.dataset} (streaming) ...")
        ds_kwargs = dict(streaming=True)
        if args.dataset_config:
            ds_kwargs["name"] = args.dataset_config
        try:
            train_stream = load_dataset(args.dataset, split="train", **ds_kwargs)
        except ValueError:
            # Some datasets only have "train".
            train_stream = load_dataset(args.dataset, **ds_kwargs)["train"]
        try:
            val_stream = load_dataset(args.dataset, split="validation",
                                       **ds_kwargs)
        except (ValueError, KeyError):
            # No validation split — split off a slice of train as held-out.
            try:
                val_stream = load_dataset(args.dataset, split="test", **ds_kwargs)
            except (ValueError, KeyError):
                print("  no val/test split — using shuffled train tail as validation")
                val_stream = load_dataset(args.dataset, split="train",
                                          **ds_kwargs).shuffle(seed=42).skip(10_000)

        train_ds = TokenisedStream(train_stream, tok, args.T,
                                   text_field=args.text_field)
        val_ds = TokenisedStream(val_stream, tok, args.T,
                                 text_field=args.text_field)
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                   num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=1)

    # 2. Model — see experiments/model_builder.py.
    from experiments.model_builder import build_model_from_args
    model, _build_info = build_model_from_args(
        args, vocab_size=model_vocab_size,
        thinking_token_id=thinking_token_id,
    )
    fb_pairs = _build_info.fb_pairs
    fb_xattn_pairs = _build_info.fb_xattn_pairs
    n_layers_actual = _build_info.n_layers
    aux_dim = _build_info.aux_dim
    # Engagement kill-gate, part 2a: STARTUP construction asserts (recipe
    # rule #1). Independent of --engagement_check_step — always on when the
    # corresponding feature flag is set. Catches the "construction branch
    # silently not entered" failure mode BEFORE a multi-hour launch, not at
    # some step N deep into it. (The latent-reasoning analogue of this assert
    # lives further below, right where `_latent_reasoner` is constructed —
    # that object needs the tokenizer/thinking_token_id, which aren't ready
    # yet here.)
    assert_pkm_constructed(args, model)
    assert_memory_constructed(args, model)
    assert_sparse_feedback_constructed(args, model)
    # ctx_namekey addressing aux needs the WM read-attention graph stashed (the
    # copy head also turns this on in TinyLM.__init__; set it explicitly so the
    # addr-aux works even if --use_copy_head is absent). No-op when WM is off.
    if (float(getattr(args, "ctx_addr_aux_weight", 0.0)) > 0.0
            and getattr(model, "memory", None) is not None):
        model.memory._stash_read_attn_grad = True
    # ---- Speed knobs (must run AFTER model is built but BEFORE the train
    # loop touches it). See experiments/speed_knobs.py.
    # The latent co-train aux losses run extra eager forwards at short/odd
    # shapes; under torch.compile (strict mode, no silent fallback) that
    # reproduces the documented Inductor symbolic-shape assertion (2026-05-27
    # gate-calibration smoke, bug #1) as a hard crash mid-run. Auto-disable
    # compile rather than relying on every launcher remembering --no-compile.
    if bool(args.compile) and (getattr(args, "latent_cotrain_weight", 0.0) > 0.0
                               or getattr(args, "latent_reasoning_weight", 0.0) > 0.0):
        print("[compile] AUTO-DISABLED: --latent_cotrain_weight/"
              "--latent_reasoning_weight run variable-shape extra forwards "
              "that crash Inductor under strict compile. Pass --no-compile "
              "to silence this message.")
        args.compile = False
    from experiments.speed_knobs import apply_speed_knobs
    apply_speed_knobs(model, bf16=bool(args.bf16), tf32=bool(args.tf32),
                      compile_model=bool(args.compile),
                      compile_mode=args.compile_mode)
    if fb_xattn_pairs:
        n_total_pairs = sum(len(srcs) for _, srcs in fb_xattn_pairs)
        feedback_desc = (f"xattn[{args.feedback_xattn_form}]"
                         f"(targets={len(fb_xattn_pairs)},"
                         f"src-edges={n_total_pairs},"
                         f"heads={args.feedback_xattn_heads})")
    else:
        feedback_desc = f"{args.feedback}"
    print(f"  params: {model.num_params() / 1e6:.1f}M  aux_dim={aux_dim}  "
          f"feedback={feedback_desc}")

    # ---- Logit-KD teacher (frozen, eval, no grad). Loaded ONCE here; passed
    # into _nonthink_forward_loss. Default OFF (empty id / weight 0) → kd_teacher
    # stays None and the loss path is byte-identical to a non-KD run.
    kd_teacher = None
    if (bool(getattr(args, "distill_teacher_model", ""))
            and float(getattr(args, "distill_weight", 0.0)) > 0.0):
        from transformers import AutoModelForCausalLM
        print(f"Loading KD teacher {args.distill_teacher_model} (bf16) ...")
        kd_teacher = AutoModelForCausalLM.from_pretrained(
            args.distill_teacher_model, torch_dtype=torch.bfloat16)
        kd_teacher.to("cuda").eval()
        for _p in kd_teacher.parameters():
            _p.requires_grad_(False)
        _tp = sum(p.numel() for p in kd_teacher.parameters()) / 1e6
        print(f"  KD teacher: {_tp:.0f}M params, vocab="
              f"{kd_teacher.get_output_embeddings().weight.shape[0]}, "
              f"weight={args.distill_weight}, temp={args.distill_temp}")
        # 2026-07-01 doc-isolation fix: when the batch carries doc_ids (any
        # data_mix source), the live teacher forward now runs PER-DOCUMENT
        # (`_kd_teacher_forward_doc_isolated`) so it never conditions a
        # document's predictions on a preceding, unrelated document — matching
        # the student's cu_seqlens state reset. Targets that straddle a
        # document boundary are additionally excluded from the KD loss
        # (`_kd_valid_mask`), for both this live path and the offline
        # logit-store path. Streams without doc_ids are unaffected.
        print("  KD live-teacher forward: CROSS-DOCUMENT ISOLATION active "
              "(per-doc segment forwards + same-doc target masking whenever "
              "the data stream emits doc_ids).")

    # ---- OFFLINE logit-KD store (precomputed teacher top-k on disk). Opened
    # ONCE; its cursor advances in lockstep with the data iterator inside
    # _nonthink_forward_loss. Default OFF (empty dir / weight 0) → stays None and
    # the loss path is byte-identical to a non-KD run.
    kd_logit_store = None
    if (bool(getattr(args, "distill_logits_dir", ""))
            and float(getattr(args, "distill_weight", 0.0)) > 0.0):
        from experiments.teacher_logits_io import LogitStoreReader
        kd_logit_store = LogitStoreReader(args.distill_logits_dir)
        print(f"Offline KD: opened logit store {args.distill_logits_dir}")
        print(f"  teacher={kd_logit_store.teacher_model}  "
              f"tokenizer={kd_logit_store.tokenizer_name}  "
              f"k={kd_logit_store.k}  vocab={kd_logit_store.vocab_size}  "
              f"tokens={len(kd_logit_store):,}  shards={len(kd_logit_store.shards)}")
        print(f"  weight={args.distill_weight}, temp={args.distill_temp}")
        # The store's teacher vocab must be <= the student vocab so the gather at
        # the teacher's top-k ids is always in-range.
        if kd_logit_store.vocab_size > model_vocab_size:
            raise SystemExit(
                f"offline-KD store vocab {kd_logit_store.vocab_size} exceeds the "
                f"student model vocab {model_vocab_size} — top-k ids would be "
                "out of range for the gather.")

    if args.aux_brackets:
        print("Computing bracket-deltas table for tokenizer ...")
        bracket_deltas = compute_bracket_deltas(tok)
        print(f"  table shape: {bracket_deltas.shape}, "
              f"non-zero count: {(bracket_deltas != 0).sum().item()}")
    else:
        bracket_deltas = None

    # Trunk multi-horizon gist heads (v7, see experiments/gist_loss.py).
    # Attached as a model submodule so they ride the existing optimizer,
    # state_dict and resume paths with no special handling. Must be
    # attached BEFORE build_optimizer so the optimizer picks them up.
    gist_horizons = None
    if args.gist_loss_weight > 0.0:
        gist_horizons = parse_horizons(args.gist_horizons)
        model.gist_heads = build_gist_heads(args.d_model,
                                            gist_horizons).cuda()
        # The model computes the gist loss inside its own forward (see
        # TinyLM._finalize) — it reads these two attributes.
        model._gist_horizons = gist_horizons
        model._gist_loss_enabled = True
        # build_model_from_args loaded --load_ckpt BEFORE gist_heads
        # existed, so its gist_heads.* keys (if any) were dropped as
        # "unexpected". Re-load just those now. The v7.1 pretrain ckpt
        # has none → fresh heads, which is correct for Phase C.
        if args.load_ckpt is not None:
            _ck = torch.load(args.load_ckpt, map_location="cuda",
                             weights_only=False)
            _sd = (_ck["state_dict"] if isinstance(_ck, dict)
                   and "state_dict" in _ck else _ck)
            _gh = {k[len("gist_heads."):]: v for k, v in _sd.items()
                   if k.startswith("gist_heads.")}
            if _gh:
                model.gist_heads.load_state_dict(_gh)
                print(f"  gist_heads: restored from {args.load_ckpt!r}")
            else:
                print("  gist_heads: fresh (absent from loaded ckpt)")
        print(f"  trunk gist loss ON: horizons={gist_horizons} "
              f"weight={args.gist_loss_weight}")

    # DDP wrap: AFTER compile (apply_speed_knobs replaced model.forward) and
    # gist-head attach, BEFORE the optimizer (which reads model.parameters() —
    # DDP shares the same parameter tensors, so the optimizer is built on the
    # raw model and `ddp_model` is used ONLY for the loss-bearing forward).
    #   * find_unused_parameters=True: the FiLM K-warmup bypass and the gate /
    #     PKM / gist heads make the set of grad-receiving params vary per step;
    #     DDP errors without this.
    #   * broadcast_buffers=False: no BN running stats to sync (PKM uses LN).
    ddp_model = model
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(model, device_ids=[ddp_local_rank],
                        output_device=ddp_local_rank,
                        find_unused_parameters=True,
                        broadcast_buffers=False,
                        gradient_as_bucket_view=True)
        # NOTE (2026-06-18): do NOT use _set_static_graph() here.
        #   (1) static_graph records the graph on the FIRST backward, but with
        #       grad_accum>1 the first (n_micro-1) microbatches run under
        #       ddp_model.no_sync() — so the reducer is never prepared on that
        #       first backward → `expect_autograd_hooks_ INTERNAL ASSERT FAILED`
        #       crash at step 1 (the v16 from-scratch DDP failure, 2026-06-18).
        #   (2) the original justification (reentrant activation-checkpoint
        #       double-fires each grad hook) no longer holds: _ckpt_run_block
        #       uses use_reentrant=False (model.py), which fires each hook once.
        # find_unused_parameters=True handles the genuinely varying param-usage
        # set (PKM selects a subset of value rows per step; the FiLM K-warmup
        # bypass) and is compatible with no_sync gradient accumulation.
        # bf16 gradient compression: this rig has NO P2P (GPU0<->GPU1 is PHB /
        # chipset-not-supported), so all-reduce is host-staged at ~4 GB/s.
        # Compressing grads fp32->bf16 halves the per-step all-reduce bytes
        # (~570ms -> ~300ms for the 600M-param grad), recovering most of the
        # 2x. Lossless enough for grad sync (bf16 has fp32 exponent range).
        if not args.ddp_no_bf16_compress:
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks \
                import bf16_compress_hook
            ddp_model.register_comm_hook(state=None, hook=bf16_compress_hook)
            print("[ddp] bf16_compress_hook registered (PCIe-bound link)",
                  flush=True)
        print(f"[ddp] wrapped model on cuda:{ddp_local_rank}", flush=True)

    # Optimizer construction — see experiments/optim_utils.py.
    from experiments.optim_utils import build_optimizer
    opts, scheds = build_optimizer(
        model, optimizer=args.optimizer, lr=args.lr, lr_muon=args.lr_muon,
        alpha_wd=args.alpha_wd, steps=args.steps, wd=args.wd,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup_steps,
        decay_frac=args.lr_decay_frac,
        bf16_optim_state=args.bf16_optim_state,
        pkm_value_lr_mult=float(getattr(args, "pkm_value_lr_mult", 1.0)),
        matrix_optimizer=getattr(args, "matrix_optimizer", "muon"),
        embed_lr_mult=float(getattr(args, "embed_lr_mult", 1.0)),
        embed_optimizer=getattr(args, "embed_optimizer", "adam"),
        feature_lr_mult=float(getattr(args, "feature_lr_mult", 1.0)),
    )
    # Backwards-compat aliases used elsewhere in the loop.
    opt = opts[0]
    scheduler = scheds[0]

    # Resume support: fast-forward LR scheduler to match --start_step so the
    # cosine schedule continues from where the ckpt left off. Optimizer
    # momenta are still fresh (we don't save them in mid-eval ckpts) — a
    # brief loss-spike transient is expected on resume.
    if args.start_step > 0:
        for _ in range(args.start_step):
            for s in scheds:
                s.step()
        print(f"Fast-forwarded LR scheduler by {args.start_step} steps; "
              f"resumed lr={scheduler.get_last_lr()[0]:.2e}")

    # 3. Train loop.
    print(f"\n{'step':>6}  {'tok/s':>8}  {'tloss':>8}  {'lr':>9}")
    t0 = time.perf_counter()
    train_iter = iter(train_loader)
    last_log = t0
    last_log_step = 0
    losses = []
    # Mid-training eval state.
    mid_eval_controller = None
    tokens_seen = 0
    next_eval_at = 0
    tokens_at_last_probe = 0
    next_feature_probe_at = (int(args.feature_probe_every_tokens)
                             if getattr(args, "feature_probe_every_tokens", 0) > 0
                             else 0)
    if next_feature_probe_at:
        print(f"\nPer-feature usefulness probe enabled every "
              f"{args.feature_probe_every_tokens:,} tokens "
              f"(ablation-delta CE for WM/PKM, FiLM α, gate fire-rate).")
    if args.probe_humaneval_every_tokens > 0:
        if not pathlib.Path(args.probe_humaneval_path).exists():
            print(f"  [probe] {args.probe_humaneval_path} not found; "
                  f"run experiments/build_probe_dataset.py first. "
                  f"Disabling probe.")
            args.probe_humaneval_every_tokens = 0
    if args.mid_eval_every_tokens > 0:
        from experiments.eval_callback import EvalStopController, run_eval
        mid_eval_controller = EvalStopController(
            stop_threshold=args.auto_stop_threshold,
            k_consecutive_flat=args.auto_stop_k,
        )
        next_eval_at = int(args.mid_eval_every_tokens)
        print(f"\nMid-training eval enabled: HumanEval @ "
              f"{args.mid_eval_n_problems} problems every "
              f"{args.mid_eval_every_tokens:,} tokens. "
              f"auto_stop={args.auto_stop} (Δ<{args.auto_stop_threshold:.3f} "
              f"for {args.auto_stop_k} consecutive intervals).")
    losses_gate_window: list[tuple] = []  # (mean_g, emit_frac, raw_ce) per step
    think_queue = (ThinkContinuationQueue(args.think_queue_max)
                   if args.enable_thinking_token else None)
    think_replay_queue = (ThinkReplayQueue(args.think_queue_max)
                          if args.enable_thinking_token else None)
    think_stats_window: list[dict[str, float]] = []
    think_closed_traj_window: list[float] = []
    think_queue_batch = args.think_queue_batch or 1
    think_replay_batch = args.think_replay_batch or think_queue_batch
    if args.enable_thinking_token:
        print(f"  THINKING queue: max={args.think_queue_max:,} CPU records, "
              f"packed_cont={think_queue_batch}, packed_replay={think_replay_batch}, "
              f"accum_steps={args.think_queue_accum_steps}, "
              f"accum_max={args.think_queue_accum_max_steps or args.think_queue_accum_steps}, "
              f"drain_target={args.think_queue_drain_target}, "
              f"decision={args.think_decision}, "
              f"priority={'on' if args.think_prioritize_queue else 'off'}")
    pad_token_id = tok.eos_token_id
    if pad_token_id is None:
        pad_token_id = tok.bos_token_id if tok.bos_token_id is not None else 0

    # Depth-matched latent-reasoning co-train (2026-06-05 fix). Loads the
    # depth-bound pointer-chase corpus once; emits one answer-span latent loss
    # per optimizer step (last microbatch) at R=depth with a curriculum.
    _latent_reasoner = None
    if getattr(args, "latent_reasoning_weight", 0.0) > 0.0:
        from experiments.latent_reasoning_cotrain import LatentReasoningCotrain
        if thinking_token_id is None:
            raise SystemExit("--latent_reasoning_weight needs a thinking token "
                             "(set --thinking_token or use --data_mix).")
        _rr_rungs = [int(x) for x in args.latent_reasoning_rungs.split(",")
                     if x.strip()]
        _latent_reasoner = LatentReasoningCotrain(
            train_prefix=args.latent_reasoning_train_prefix,
            rungs=_rr_rungs, tok=tok, thinking_id=int(thinking_token_id),
            eos_id=int(tok.eos_token_id if tok.eos_token_id is not None
                       else pad_token_id),
            device="cuda", max_len=int(args.latent_reasoning_max_len),
            no_ramp=bool(args.latent_reasoning_no_ramp),
            gate_weight=float(getattr(args, "latent_reasoning_gate_weight", 0.0)),
            seed=int(args.seed))
        _lr_aux_every = int(getattr(args, "latent_reasoning_aux_every", 1))
        if _lr_aux_every > 8:
            print(f"[latent-reasoning] WARNING: --latent_reasoning_aux_every="
                  f"{_lr_aux_every} > 8 — firing this rarely (at "
                  f"{_lr_aux_every}x weight when it does) trades a much "
                  "spikier/higher-variance gradient for the wall-clock win; "
                  "keep it <= 8 unless you've checked stability.")
        print(f"Latent-reasoning co-train ON: weight={args.latent_reasoning_weight} "
              f"rungs={_latent_reasoner.rungs} "
              f"n/step={args.latent_reasoning_n} "
              f"aux_every={_lr_aux_every} "
              f"batch_examples={_latent_reasoner.batch_examples} "
              f"(examples/rung: "
              f"{ {n: len(_latent_reasoner.data[n]) for n in _latent_reasoner.rungs} })")
    # Engagement kill-gate, part 2b: STARTUP construction assert (recipe rule
    # #1). If --latent_reasoning_weight>0 said "on", the reasoner object MUST
    # exist by here — this is exactly the Phase-1 arm-B failure mode (the
    # construction branch above silently not entered: two dead attempts died
    # at step 620 with no banner ever printed, and the run-of-record trained
    # this aux at literally zero the whole run — SESSION_FINDINGS.md
    # 2026-07-02). Trivial today by construction; the point is to keep
    # failing loudly if a future edit adds a path around it.
    assert_latent_reasoner_constructed(args, _latent_reasoner)

    # Engagement kill-gate, part 3: per-mechanism counters for the step-N
    # evaluation. Cheap (a couple of int/bool locals); zero runtime cost when
    # --engagement_check_step is 0 (the counters are still updated at their
    # real computation sites below, but nothing ever reads them).
    _copy_fire_count = 0                # cumulative WM copy-gate match-fires
    _latent_reason_fire_count = 0       # times the reason() loss actually ran
    _latent_reason_all_finite = True    # AND of finite-ness across all fires

    # TensorBoard writer — no-op context when --tb_dir is not set. Under DDP
    # only rank 0 writes (multiple ranks → the same event file would corrupt).
    if args.tb_dir and is_main:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.tb_dir)
        print(f"TensorBoard logging → {args.tb_dir}")
    else:
        tb = None

    def thinking_schedule(step: int) -> dict[str, float]:
        if step <= args.think_warmup_steps:
            curriculum = 0.0
        elif args.think_curriculum_steps > 0:
            curriculum = min(
                1.0,
                (step - args.think_warmup_steps)
                / float(args.think_curriculum_steps),
            )
        else:
            curriculum = 1.0
        explore_prob = (
            args.think_explore_start_prob
            + curriculum
            * (args.think_explore_prob - args.think_explore_start_prob)
        )
        lambda_start = (
            args.think_lambda if args.think_lambda_start is None
            else args.think_lambda_start
        )
        lambda_eff = lambda_start + curriculum * (args.think_lambda - lambda_start)
        gate_threshold_start = (
            args.think_gate_threshold
            if args.think_gate_threshold_start is None
            else args.think_gate_threshold_start
        )
        gate_threshold = (
            gate_threshold_start
            + curriculum * (args.think_gate_threshold - gate_threshold_start)
        )
        depth_start = (
            args.think_safety_max_depth
            if args.think_safety_max_depth_start is None
            else args.think_safety_max_depth_start
        )
        if args.think_safety_max_depth <= 0 and depth_start <= 0:
            safety_max_depth = 0
        else:
            depth_eff_float = (
                depth_start
                + curriculum * (args.think_safety_max_depth - depth_start)
            )
            safety_max_depth = max(1, int(round(depth_eff_float)))
        return {
            "curriculum": float(curriculum),
            "explore_prob": float(explore_prob),
            "lambda": float(lambda_eff),
            "gate_threshold": float(gate_threshold),
            "safety_max_depth": float(safety_max_depth),
            "queue_pressure": 0.0,
            "backpressure_target": 0.0,
        }

    def apply_queue_backpressure(schedule: dict[str, float]) -> dict[str, float]:
        if think_queue is None or think_replay_queue is None:
            return schedule
        target = args.think_backpressure_target
        if target < 0:
            target = args.think_queue_drain_target
        uses_backpressure = (
            target >= 0
            and (
                args.think_backpressure_lambda > 0.0
                or args.think_backpressure_threshold > 0.0
                or args.think_backpressure_explore > 0.0
            )
        )
        if not uses_backpressure:
            schedule["backpressure_target"] = float(max(0, target))
            return schedule
        target = max(1, target)
        backlog = max(len(think_queue), len(think_replay_queue))
        pressure = max(0.0, (backlog - target) / float(target))
        if args.think_backpressure_max > 0.0:
            pressure = min(pressure, args.think_backpressure_max)
        schedule = dict(schedule)
        schedule["queue_pressure"] = float(pressure)
        schedule["backpressure_target"] = float(target)
        if pressure <= 0.0:
            return schedule
        schedule["lambda"] = float(
            schedule["lambda"] + args.think_backpressure_lambda * pressure
        )
        if args.think_backpressure_threshold > 0.0:
            divisor = 1.0 + args.think_backpressure_threshold * pressure
            schedule["gate_threshold"] = float(
                max(1e-4, min(0.9999, schedule["gate_threshold"] / divisor))
            )
        if args.think_backpressure_explore > 0.0:
            divisor = 1.0 + args.think_backpressure_explore * pressure
            schedule["explore_prob"] = float(schedule["explore_prob"] / divisor)
        return schedule

    def new_thinking_stats(schedule: dict[str, float]) -> dict[str, float]:
        return {
            "cont_items": 0.0,
            "cont_think": 0.0,
            "cont_explore": 0.0,
            "forced_emit": 0.0,
            "closed": 0.0,
            "replay_items": 0.0,
            "replay_think": 0.0,
            **schedule,
        }

    def merge_thinking_stats(dst: dict[str, float], src: dict[str, float]) -> None:
        summed = {
            "cont_items", "cont_think", "cont_explore", "forced_emit",
            "closed", "replay_items", "replay_think",
        }
        for key, value in src.items():
            if key in summed:
                dst[key] = dst.get(key, 0.0) + float(value)
            else:
                dst[key] = float(value)

    def process_thinking_aux_batch(
        step: int,
        cont_items: list[ThinkContinuation],
        replay_items: list[ThinkReplay],
        schedule: dict[str, float],
    ) -> tuple[torch.Tensor, float, dict[str, float]]:
        if not cont_items and not replay_items:
            return torch.zeros((), device="cuda"), 0.0, new_thinking_stats(schedule)
        assert think_queue is not None and think_replay_queue is not None
        rows = []
        cont_targets = cont_last = None
        replay_targets = replay_last = replay_is_think = None
        if cont_items:
            cont_x, cont_targets, cont_last = build_continuation_batch(
                cont_items, block_size=args.T, pad_token_id=pad_token_id,
                device="cuda",
            )
            rows.append(cont_x)
        if replay_items:
            replay_x, replay_targets, replay_last, replay_is_think = build_replay_batch(
                replay_items, block_size=args.T, pad_token_id=pad_token_id,
                thinking_token_id=int(thinking_token_id), device="cuda",
            )
            rows.append(replay_x)
        aux_x = torch.cat(rows, dim=0)
        if args.think_checkpointing:
            from torch.utils.checkpoint import checkpoint
            # model is a nn.Module, which checkpoint can wrap.
            # We use use_reentrant=False for modern compatibility.
            aux_logits = checkpoint(model, aux_x, use_reentrant=False)
        else:
            aux_logits = model(aux_x)
        loss_terms: list[torch.Tensor] = []
        stats = new_thinking_stats(schedule)
        lambda_eff = float(schedule["lambda"])
        gate_threshold = float(schedule["gate_threshold"])
        safety_max_depth = int(schedule["safety_max_depth"])
        explore_prob = float(schedule["explore_prob"])

        if cont_items:
            assert cont_targets is not None and cont_last is not None
            row = torch.arange(len(cont_items), device=aux_logits.device)
            cont_logits = aux_logits[row, cont_last]
            forced = torch.zeros(len(cont_items), dtype=torch.bool,
                                 device=aux_logits.device)
            if safety_max_depth > 0:
                forced |= torch.tensor(
                    [item.depth >= safety_max_depth for item in cont_items],
                    dtype=torch.bool, device=aux_logits.device,
                )
            if args.think_queue_ttl > 0:
                forced |= torch.tensor(
                    [step - item.origin_step >= args.think_queue_ttl
                     for item in cont_items],
                    dtype=torch.bool, device=aux_logits.device,
                )
            if args.think_decision == "gate":
                cont_gate = model._last_gate[row, cont_last].detach()
                cont_think = (cont_gate < gate_threshold) & ~forced
                cont_gate_logits = model._last_gate_logits[row, cont_last]
                cont_think_nll = F.binary_cross_entropy_with_logits(
                    cont_gate_logits,
                    torch.zeros_like(cont_gate_logits),
                    reduction="none",
                )
            else:
                cont_think = choose_think_actions(
                    cont_logits.detach(), int(thinking_token_id),
                    args.think_policy, args.think_threshold,
                    args.think_temperature, allow_think=~forced,
                )
                think_targets = torch.full_like(cont_targets, int(thinking_token_id))
                cont_think_nll = F.cross_entropy(
                    cont_logits, think_targets, reduction="none",
                )
            cont_explore = torch.zeros_like(cont_think)
            if explore_prob > 0.0:
                cont_explore = (
                    torch.rand_like(cont_think.float()) < explore_prob
                ) & ~forced
                cont_think = cont_think | cont_explore
            cont_answer_nll = cross_entropy_masking_token(
                cont_logits, cont_targets, int(thinking_token_id),
                reduction="none",
            )
            closed_traj: list[float] = []
            for i, item in enumerate(cont_items):
                if bool(cont_think[i].item()):
                    next_ctx = (item.context_ids + [int(thinking_token_id)])[-args.T:]
                    think_queue.enqueue(ThinkContinuation(
                        context_ids=next_ctx,
                        target_id=item.target_id,
                        depth=item.depth + 1,
                        accum_nll=(
                            item.accum_nll
                            + float(cont_think_nll[i].detach().item())
                        ),
                        accum_cost=item.accum_cost + lambda_eff,
                        origin_step=item.origin_step,
                        decision_context_ids=(
                            item.decision_context_ids or item.context_ids
                        ),
                        immediate_nll=item.immediate_nll,
                    ))
                else:
                    answer_after_think = float(cont_answer_nll[i].detach().item())
                    traj = item.accum_nll + item.accum_cost + answer_after_think
                    closed_traj.append(traj)
                    comparable_traj = item.accum_cost + answer_after_think
                    beneficial = (
                        comparable_traj + args.think_advantage_margin
                        < float(item.immediate_nll)
                    )
                    think_replay_queue.enqueue(ThinkReplay(
                        context_ids=item.decision_context_ids or item.context_ids,
                        target_id=item.target_id,
                        target_is_thinking=beneficial,
                    ))
                    loss_terms.append(cont_answer_nll[i])
            if closed_traj:
                think_closed_traj_window.extend(closed_traj)
            stats.update({
                "cont_items": float(len(cont_items)),
                "cont_think": float(cont_think.float().sum().item()),
                "cont_explore": float(cont_explore.float().sum().item()),
                "forced_emit": float(forced.float().sum().item()),
                "closed": float(len(closed_traj)),
            })

        if replay_items:
            assert replay_targets is not None
            assert replay_last is not None and replay_is_think is not None
            start = len(cont_items)
            row = torch.arange(start, start + len(replay_items),
                               device=aux_logits.device)
            replay_logits = aux_logits[row, replay_last]
            if args.think_decision == "gate":
                replay_gate_logits = model._last_gate_logits[row, replay_last]
                emit_targets = (~replay_is_think).float()
                gate_ce = F.binary_cross_entropy_with_logits(
                    replay_gate_logits, emit_targets, reduction="none",
                )
                replay_answer_ce = cross_entropy_masking_token(
                    replay_logits, replay_targets, int(thinking_token_id),
                    reduction="none",
                )
                replay_ce = torch.where(
                    replay_is_think,
                    gate_ce + lambda_eff,
                    gate_ce + replay_answer_ce,
                )
            else:
                replay_ce = F.cross_entropy(
                    replay_logits, replay_targets, reduction="none",
                )
                replay_ce = replay_ce + lambda_eff * replay_is_think.float()
            loss_terms.append(args.think_replay_weight * replay_ce)
            stats["replay_items"] = float(len(replay_items))
            stats["replay_think"] = float(replay_is_think.float().sum().item())

        if not loss_terms:
            return torch.zeros((), device=aux_logits.device), 0.0, stats
        loss_sum = torch.stack([term.sum() for term in loss_terms]).sum()
        count = float(sum(term.numel() for term in loss_terms))
        return loss_sum, count, stats

    # Freeze-trunk feature pre-warm (--freeze_trunk_steps N). A param is a
    # "feature" iff its name contains any of these substrings; only features
    # train during the window so a freshly attached FiLM/PKM/WM/gate/gist stack
    # can adapt to an inherited (competent) trunk before the trunk is unfrozen.
    _FEATURE_PARAM_SUBSTRINGS = (
        "sparse_feedback", "feedback", "pkm_layer", "memory.",
        "gate_head", "gist", "latent_feedback_adapter",
        "retrieval_input_alpha", "future_gist", "ctx_",
        # WM copy/pointer readout lives at top-level copy_head.* (not under
        # memory.) — include it so the WM READOUT pre-warms too, not just the
        # addresser (review caught this false-negative on 2026-06-22).
        "copy_head", "pointer",
    )

    def _is_feature_param(name: str) -> bool:
        return any(s in name for s in _FEATURE_PARAM_SUBSTRINGS)

    _trunk_frozen = False  # tracks whether the trunk freeze is currently active
    # Snapshot each param's ORIGINAL requires_grad so unfreeze restores the
    # pre-freeze state rather than blanket-True (which would clobber intentional
    # freezes like --mem_freeze_read_alpha that pin WM read-α at 0). During the
    # freeze window we ONLY force non-feature (trunk) params off; feature params
    # keep their original requires_grad (so a pinned WM read-α stays pinned).
    _orig_requires_grad = {nm: p.requires_grad for nm, p in model.named_parameters()}

    for step in range(args.start_step + 1, args.steps + 1):
        # Freeze-trunk window: for steps 1..N freeze every non-feature param;
        # at the first step > N restore each param's ORIGINAL requires_grad once.
        # Optimizer step is safe — Muon/AdamW skip params with grad=None.
        if args.freeze_trunk_steps > 0:
            if step <= args.freeze_trunk_steps:
                if not _trunk_frozen:
                    _n_froz = _n_feat = 0
                    for _nm, _p in model.named_parameters():
                        if _is_feature_param(_nm):
                            _n_feat += 1  # leave feature params' requires_grad as-is
                        else:
                            _p.requires_grad_(False)
                            _n_froz += 1
                    _trunk_frozen = True
                    print(f"[step {step}] freeze_trunk ENGAGED: froze {_n_froz} "
                          f"trunk params (requires_grad=False); {_n_feat} feature "
                          f"params keep their original requires_grad until step "
                          f"{args.freeze_trunk_steps}", flush=True)
            elif _trunk_frozen:
                for _nm, _p in model.named_parameters():
                    _p.requires_grad_(_orig_requires_grad.get(_nm, True))
                _trunk_frozen = False
                print(f"[step {step}] freeze_trunk RELEASED: restored original "
                      f"requires_grad on all params; trunk now training", flush=True)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        # Streams may yield (x, y), (x, y, doc_ids), or (x, y, doc_ids,
        # mem_read_mask) [v14]; doc_ids drives cross-document state isolation;
        # mem_read_mask (when present) drives the WM read at recall answer spans.
        x, y, *_rest = batch
        x, y = x.to("cuda"), y.to("cuda")
        doc_ids = _rest[0].to("cuda") if _rest else None
        mem_read_mask = _rest[1].to("cuda") if len(_rest) > 1 else None
        # K-self-feed curriculum: bypass FiLM (1-pass forward) until the
        # warmup boundary, then run the configured --feedback_self_k.
        if args.feedback_self_k_warmup_steps > 0:
            bypass = step <= args.feedback_self_k_warmup_steps
            if bypass != model._film_bypass:
                model._film_bypass = bypass
                if not bypass:
                    print(f"[step {step}] FiLM K-self-feed curriculum: "
                          f"warmup over, enabling feedback_self_k="
                          f"{args.feedback_self_k}")
        # PKM ε-greedy curriculum: linear anneal from --pkm_epsilon_start
        # to 0 over --pkm_epsilon_warmup_steps. Forces every slot to get
        # gradient early; the learned router takes over after warmup.
        if (getattr(args, "use_pkm", False)
                and getattr(args, "pkm_epsilon_start", 0.0) > 0.0):
            warm = max(1, int(getattr(args, "pkm_epsilon_warmup_steps", 0)))
            progress = min(1.0, step / warm) if warm > 0 else 1.0
            eps = float(args.pkm_epsilon_start) * (1.0 - progress)
            model.pkm_layer.random_slot_epsilon = eps
        # PKM α-floor curriculum: linear anneal of the additive
        # sign-preserving floor on the output gate. Forces a minimum PKM
        # contribution during the value-table-bootstrap window so values
        # get meaningful gradient before α can shrink. Synced with ε.
        if (getattr(args, "use_pkm", False)
                and getattr(args, "pkm_alpha_floor_start", 0.0) > 0.0):
            warm = max(1, int(getattr(args, "pkm_alpha_floor_warmup_steps", 0)))
            progress = min(1.0, step / warm) if warm > 0 else 1.0
            floor = float(args.pkm_alpha_floor_start) * (1.0 - progress)
            model.pkm_layer.alpha_floor = floor
        for o in opts:
            o.zero_grad(set_to_none=True)
        pre_think_stats: dict[str, float] | None = None
        if args.enable_thinking_token:
            base_schedule = thinking_schedule(step)
            schedule = apply_queue_backpressure(base_schedule)
            pre_think_stats = new_thinking_stats(schedule)
            if args.think_queue_accum_steps > 0:
                assert think_queue is not None and think_replay_queue is not None
                fresh_token_budget = max(1, x.numel())
                accum_max_steps = (
                    args.think_queue_accum_max_steps
                    or args.think_queue_accum_steps
                )
                accum_step = 0
                while accum_step < accum_max_steps:
                    must_do_minimum = accum_step < args.think_queue_accum_steps
                    should_drain = (
                        args.think_queue_drain_target >= 0
                        and (
                            len(think_queue) > args.think_queue_drain_target
                            or len(think_replay_queue) > args.think_queue_drain_target
                        )
                    )
                    if not must_do_minimum and not should_drain:
                        break
                    cont_n = min(think_queue_batch, len(think_queue))
                    replay_n = min(think_replay_batch, len(think_replay_queue))
                    if cont_n == 0 and replay_n == 0:
                        break
                    cont_items = think_queue.pop_batch(cont_n)
                    replay_items = think_replay_queue.pop_batch(replay_n)
                    aux_schedule = apply_queue_backpressure(base_schedule)
                    aux_loss_sum, aux_count, aux_stats = process_thinking_aux_batch(
                        step, cont_items, replay_items, aux_schedule,
                    )
                    merge_thinking_stats(pre_think_stats, aux_stats)
                    if aux_count > 0:
                        aux_denom = (
                            aux_count
                            if args.think_aux_normalize == "aux_items"
                            else fresh_token_budget
                        )
                        (args.think_aux_loss_scale
                         * aux_loss_sum / max(1.0, aux_denom)).backward()
                    accum_step += 1
        packed_cont_items: list[ThinkContinuation] = []
        packed_replay_items: list[ThinkReplay] = []
        packed_cont_last = packed_cont_targets = None
        packed_replay_last = packed_replay_targets = packed_replay_is_think = None
        fresh_offset = 0
        fresh_n = x.shape[0]
        if args.enable_thinking_token:
            assert think_queue is not None and think_replay_queue is not None
            if args.think_queue_accum_steps > 0:
                n_cont = 0
                n_replay = 0
            elif args.think_prioritize_queue:
                queue_capacity = max(0, x.shape[0] - args.think_min_fresh_rows)
                n_cont = min(len(think_queue), queue_capacity)
                n_replay = min(len(think_replay_queue), queue_capacity - n_cont)
            else:
                max_aux_rows = max(0, x.shape[0] - args.think_min_fresh_rows)
                n_cont = min(think_queue_batch, len(think_queue), max_aux_rows)
                n_replay = min(think_replay_batch, len(think_replay_queue),
                               max_aux_rows - n_cont)
            packed_cont_items = think_queue.pop_batch(n_cont)
            packed_replay_items = think_replay_queue.pop_batch(n_replay)
            fresh_n = x.shape[0] - n_cont - n_replay
            packed_rows = []
            packed_targets = []
            if packed_cont_items:
                cont_x, packed_cont_targets, packed_cont_last = build_continuation_batch(
                    packed_cont_items, block_size=args.T,
                    pad_token_id=pad_token_id, device=x.device,
                )
                packed_rows.append(cont_x)
                packed_targets.append(torch.zeros_like(cont_x))
            if packed_replay_items:
                replay_x, packed_replay_targets, packed_replay_last, packed_replay_is_think = build_replay_batch(
                    packed_replay_items, block_size=args.T,
                    pad_token_id=pad_token_id,
                    thinking_token_id=int(thinking_token_id),
                    device=x.device,
                )
                packed_rows.append(replay_x)
                packed_targets.append(torch.zeros_like(replay_x))
            if fresh_n > 0:
                packed_rows.append(x[:fresh_n])
                packed_targets.append(y[:fresh_n])
            if packed_rows:
                x = torch.cat(packed_rows, dim=0)
                y = torch.cat(packed_targets, dim=0)
            fresh_offset = n_cont + n_replay
        # Gated loss (Phase 23): L = mean(g_t * CE_t + (1-g_t) * λ).
        # g_t is stored in model._last_gate by the forward pass (side effect).
        # When output_gate is off, fall back to standard mean CE.
        if args.enable_thinking_token:
            if args.aux_brackets:
                logits, aux_logits = model(x, return_aux=True)
            else:
                logits = model(x)
                aux_logits = None
            if args.aux_brackets:
                depth = bracket_depth(x, bracket_deltas).clamp(0, args.aux_max_depth)
                aux_loss = F.cross_entropy(
                    aux_logits.reshape(-1, args.aux_max_depth + 1),
                    depth.reshape(-1),
                )
            else:
                aux_loss = torch.zeros((), device="cuda")
            V = logits.shape[-1]
            ce_per_token = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1),
                reduction="none",
            ).reshape(y.shape)                                               # (B, T)
            schedule = apply_queue_backpressure(thinking_schedule(step))
            think_curriculum = schedule["curriculum"]
            think_explore_prob_eff = schedule["explore_prob"]
            think_lambda_eff = schedule["lambda"]
            think_gate_threshold_eff = schedule["gate_threshold"]
            think_safety_max_depth_eff = int(schedule["safety_max_depth"])
            cont_loss_terms: list[torch.Tensor] = []
            replay_loss_terms: list[torch.Tensor] = []
            cont_stats = new_thinking_stats(schedule)
            cont_stats["cont_items"] = float(len(packed_cont_items))
            cont_stats["replay_items"] = float(len(packed_replay_items))
            if packed_cont_items:
                row = torch.arange(len(packed_cont_items), device=logits.device)
                cont_logits = logits[row, packed_cont_last]
                forced = torch.zeros(len(packed_cont_items), dtype=torch.bool,
                                     device=logits.device)
                if think_safety_max_depth_eff > 0:
                    forced |= torch.tensor(
                        [item.depth >= think_safety_max_depth_eff
                         for item in packed_cont_items],
                        dtype=torch.bool, device=logits.device,
                    )
                if args.think_queue_ttl > 0:
                    forced |= torch.tensor(
                        [step - item.origin_step >= args.think_queue_ttl
                         for item in packed_cont_items],
                        dtype=torch.bool, device=logits.device,
                )
                if args.think_decision == "gate":
                    cont_gate = model._last_gate[row, packed_cont_last].detach()
                    cont_think = (cont_gate < think_gate_threshold_eff) & ~forced
                else:
                    cont_think = choose_think_actions(
                        cont_logits.detach(), int(thinking_token_id),
                        args.think_policy, args.think_threshold,
                        args.think_temperature, allow_think=~forced,
                    )
                cont_explore = torch.zeros_like(cont_think)
                if think_explore_prob_eff > 0.0:
                    cont_explore = (
                        torch.rand_like(cont_think.float()) < think_explore_prob_eff
                    ) & ~forced
                    cont_think = cont_think | cont_explore
                think_targets = torch.full_like(packed_cont_targets,
                                                int(thinking_token_id))
                if args.think_decision == "gate":
                    cont_gate_logits = model._last_gate_logits[row, packed_cont_last]
                    cont_think_nll = F.binary_cross_entropy_with_logits(
                        cont_gate_logits,
                        torch.zeros_like(cont_gate_logits),
                        reduction="none",
                    )
                else:
                    cont_think_nll = F.cross_entropy(
                        cont_logits, think_targets, reduction="none"
                    )
                cont_answer_nll = cross_entropy_masking_token(
                    cont_logits, packed_cont_targets, int(thinking_token_id),
                    reduction="none",
                )
                closed_traj: list[float] = []
                for i, item in enumerate(packed_cont_items):
                    if bool(cont_think[i].item()):
                        next_ctx = (item.context_ids + [int(thinking_token_id)])[-args.T:]
                        think_queue.enqueue(ThinkContinuation(
                            context_ids=next_ctx,
                            target_id=item.target_id,
                            depth=item.depth + 1,
                            accum_nll=(item.accum_nll
                                       + float(cont_think_nll[i].detach().item())),
                            accum_cost=item.accum_cost + float(think_lambda_eff),
                            origin_step=item.origin_step,
                            decision_context_ids=(item.decision_context_ids
                                                  or item.context_ids),
                            immediate_nll=item.immediate_nll,
                        ))
                    else:
                        answer_after_think = float(cont_answer_nll[i].detach().item())
                        traj = item.accum_nll + item.accum_cost + answer_after_think
                        closed_traj.append(traj)
                        comparable_traj = item.accum_cost + answer_after_think
                        beneficial = (
                            comparable_traj + args.think_advantage_margin
                            < float(item.immediate_nll)
                        )
                        think_replay_queue.enqueue(ThinkReplay(
                            context_ids=(item.decision_context_ids
                                         or item.context_ids),
                            target_id=item.target_id,
                            target_is_thinking=beneficial,
                        ))
                        cont_loss_terms.append(cont_answer_nll[i])
                if closed_traj:
                    think_closed_traj_window.extend(closed_traj)
                cont_stats.update({
                    "cont_think": float(cont_think.float().sum().item()),
                    "cont_explore": float(cont_explore.float().sum().item()),
                    "forced_emit": float(forced.float().sum().item()),
                    "closed": float(len(closed_traj)),
                })
            if packed_replay_items:
                start = len(packed_cont_items)
                row = torch.arange(start, start + len(packed_replay_items),
                                   device=logits.device)
                replay_logits = logits[row, packed_replay_last]
                if args.think_decision == "gate":
                    replay_gate_logits = model._last_gate_logits[row, packed_replay_last]
                    emit_targets = (~packed_replay_is_think).float()
                    gate_ce = F.binary_cross_entropy_with_logits(
                        replay_gate_logits, emit_targets, reduction="none"
                    )
                    replay_answer_ce = cross_entropy_masking_token(
                        replay_logits, packed_replay_targets,
                        int(thinking_token_id), reduction="none",
                    )
                    replay_ce = torch.where(
                        packed_replay_is_think,
                        gate_ce + think_lambda_eff,
                        gate_ce + replay_answer_ce,
                    )
                else:
                    replay_ce = F.cross_entropy(
                        replay_logits, packed_replay_targets, reduction="none"
                    )
                    replay_ce = (
                        replay_ce
                        + think_lambda_eff * packed_replay_is_think.float()
                    )
                replay_loss_terms.append(args.think_replay_weight * replay_ce)
                cont_stats["replay_think"] = float(
                    packed_replay_is_think.float().sum().item()
                )
            allow_think = step > args.think_warmup_steps
            fresh_logits = logits[fresh_offset:]
            fresh_y = y[fresh_offset:]
            if fresh_n > 0:
                if args.think_decision == "gate":
                    fresh_gate = model._last_gate[fresh_offset:].detach()
                    think_mask = fresh_gate < think_gate_threshold_eff
                    if not allow_think:
                        think_mask = torch.zeros_like(think_mask)
                else:
                    think_mask = choose_think_actions(
                        fresh_logits.detach(), int(thinking_token_id),
                        args.think_policy, args.think_threshold,
                        args.think_temperature, allow_think=allow_think,
                    )
            else:
                think_mask = torch.zeros((0, args.T), dtype=torch.bool,
                                         device=logits.device)
            explore_mask = torch.zeros_like(think_mask)
            free_slots = think_queue.max_len - len(think_queue)
            if fresh_n > 0:
                answer_ce_all = cross_entropy_masking_token(
                    fresh_logits.reshape(-1, V), fresh_y.reshape(-1),
                    int(thinking_token_id), reduction="none",
                ).reshape(fresh_y.shape)
                if allow_think and think_explore_prob_eff > 0.0:
                    explore_mask = choose_explore_actions(
                        answer_ce_all.detach(),
                        probability=think_explore_prob_eff,
                        mode=args.think_explore_mode,
                        top_frac=args.think_explore_top_frac,
                        min_score=args.think_explore_min_ce,
                    )
                    think_mask = think_mask | explore_mask
                fresh_loss_sum = answer_ce_all.sum()
            else:
                answer_ce_all = torch.zeros((0, args.T), device=logits.device)
                fresh_loss_sum = torch.zeros((), device=logits.device)
            if free_slots <= 0:
                think_mask = torch.zeros_like(think_mask)
                explore_mask = torch.zeros_like(explore_mask)
            else:
                think_flat = think_mask.reshape(-1)
                max_new = free_slots
                if args.think_max_new_per_step > 0:
                    max_new = min(max_new, int(args.think_max_new_per_step))
                n_think = int(think_flat.sum().item())
                if n_think > max_new:
                    idx = think_flat.nonzero(as_tuple=False).flatten()
                    if answer_ce_all.numel():
                        scores = answer_ce_all.detach().reshape(-1)[idx]
                        keep_idx = idx[torch.topk(scores, k=max_new).indices]
                    else:
                        keep_idx = idx[:max_new]
                    limited = torch.zeros_like(think_flat)
                    limited[keep_idx] = True
                    think_mask = limited.reshape_as(think_mask)
                    explore_mask = explore_mask & think_mask
            extra_loss_sum = torch.zeros((), device=logits.device)
            extra_count = 0
            if cont_loss_terms:
                extra_loss_sum = extra_loss_sum + torch.stack(cont_loss_terms).sum()
                extra_count += len(cont_loss_terms)
            if replay_loss_terms:
                extra_loss_sum = extra_loss_sum + torch.cat(replay_loss_terms).sum()
                extra_count += int(sum(t.numel() for t in replay_loss_terms))
            gate_emit_items = 0.0
            if (args.think_decision == "gate" and args.think_gate_emit_weight > 0
                    and fresh_n > 0 and think_mask.numel() > 0):
                emit_mask = ~think_mask.detach()
                gate_logits_fresh = model._last_gate_logits[fresh_offset:]
                if emit_mask.any():
                    emit_ce = F.binary_cross_entropy_with_logits(
                        gate_logits_fresh[emit_mask],
                        torch.ones_like(gate_logits_fresh[emit_mask]),
                        reduction="sum",
                    )
                    extra_loss_sum = (
                        extra_loss_sum
                        + args.think_gate_emit_weight * emit_ce
                    )
                    weighted_emit_count = (
                        args.think_gate_emit_weight
                        * float(emit_mask.float().sum().item())
                    )
                    extra_count += weighted_emit_count
                    gate_emit_items = float(emit_mask.float().sum().item())
            denom = fresh_y.numel() + extra_count
            lm_loss = (fresh_loss_sum + extra_loss_sum) / max(1, denom)
            think_coords = think_mask.detach().nonzero(as_tuple=False).cpu().tolist()
            x_cpu = x[fresh_offset:].detach().cpu()
            y_cpu = fresh_y.detach().cpu()
            answer_ce_cpu = answer_ce_all.detach().cpu()
            fresh_logits_cpu = fresh_logits.detach().cpu()
            fresh_gate_logits_cpu = (
                model._last_gate_logits[fresh_offset:].detach().cpu()
                if args.think_decision == "gate" else None
            )
            for b, t_idx in think_coords:
                ctx_ids = x_cpu[b, :t_idx + 1].tolist() + [int(thinking_token_id)]
                decision_ctx = x_cpu[b, :t_idx + 1].tolist()
                if args.think_decision == "gate":
                    assert fresh_gate_logits_cpu is not None
                    think_nll = F.binary_cross_entropy_with_logits(
                        fresh_gate_logits_cpu[b, t_idx].unsqueeze(0),
                        torch.zeros(1),
                        reduction="none",
                    )[0].item()
                else:
                    think_nll = F.cross_entropy(
                        fresh_logits_cpu[b, t_idx].unsqueeze(0),
                        torch.tensor([int(thinking_token_id)]),
                        reduction="none",
                    )[0].item()
                think_queue.enqueue(ThinkContinuation(
                    context_ids=ctx_ids[-args.T:],
                    target_id=int(y_cpu[b, t_idx].item()),
                    depth=1,
                    accum_nll=float(think_nll),
                    accum_cost=float(think_lambda_eff),
                    origin_step=step,
                    decision_context_ids=decision_ctx[-args.T:],
                    immediate_nll=float(answer_ce_cpu[b, t_idx].item()),
                ))
            think_stats = {
                "normal_items": float(fresh_y.numel()),
                "normal_think": float(think_mask.float().sum().item()),
                "normal_explore": float(explore_mask.float().sum().item()),
                "answer_ce": (float(answer_ce_all.mean().detach().item())
                              if answer_ce_all.numel() else 0.0),
                "queue_len": float(len(think_queue)),
                "queue_mean_depth": float(think_queue.mean_depth()),
                "queue_max_depth": float(think_queue.max_depth()),
                "queue_dropped": float(think_queue.dropped),
                "replay_queue_len": float(len(think_replay_queue)),
                "gate_emit_items": gate_emit_items,
                **cont_stats,
            }
            if pre_think_stats is not None:
                merge_thinking_stats(pre_think_stats, think_stats)
                think_stats = pre_think_stats
            think_stats_window.append(think_stats)
            loss = lm_loss + args.aux_weight * aux_loss
            loss = loss + _z_loss_term(logits, args.z_loss)
            loss.backward()
        else:
            # Non-thinking (pretrain) path: gradient accumulation over
            # --grad_accum microbatches, one optimizer step per `step`.
            # The gate/plain-loss branches live in _nonthink_forward_loss.
            n_micro = max(1, args.grad_accum)
            _latent_cotrain_diag = None
            _latent_reasoning_diag = None
            _gate_calib_diag = None
            _ctx_addr_diag = None
            for micro in range(n_micro):
                if micro > 0:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    x, y, *_rest = batch
                    x, y = x.to("cuda"), y.to("cuda")
                    doc_ids = _rest[0].to("cuda") if _rest else None
                    mem_read_mask = _rest[1].to("cuda") if len(_rest) > 1 else None
                # DDP: only all-reduce grads on the LAST microbatch; no_sync()
                # suppresses the reduce on the intermediates (correctness +
                # avoids n_micro× redundant comms). nullcontext when single-GPU.
                _is_last_micro = (micro == n_micro - 1)
                _sync_ctx = (ddp_model.no_sync() if (is_ddp and not _is_last_micro)
                             else contextlib.nullcontext())
                with _sync_ctx:
                    logits, ce_per_token, lm_loss, aux_loss, gate_aux_loss, \
                        gist_loss, kd_loss = _nonthink_forward_loss(
                            model, x, y, args, step, bracket_deltas,
                            doc_ids=doc_ids, gist_horizons=gist_horizons,
                            fwd_model=ddp_model, mem_read_mask=mem_read_mask,
                            kd_teacher=kd_teacher,
                            kd_thinking_token_id=thinking_token_id,
                            kd_logit_store=kd_logit_store)
                    # Snapshot the MAIN-forward gate NOW, before any aux loss
                    # (latent_cotrain / gate_calibration) runs an extra forward
                    # that clobbers model._last_gate(_logits) — the documented
                    # 2026-05-27 footgun. _last_gate is detached for diag;
                    # _last_gate_logits is the GRAD-CARRYING tensor the
                    # gate-calibration BCE flows into, so keep it un-detached.
                    main_gate = getattr(model, "_last_gate", None)
                    main_gate_logits = getattr(model, "_last_gate_logits", None)
                    # Snapshot the WM injection from the MAIN forward too: the
                    # aux extra-forwards (latent_cotrain / gate_calibration) call
                    # model() and clobber memory._last_injection with their own
                    # (N, Lmax, d) shape, which then mismatches x at the
                    # wm(inj=) diagnostic below (IndexError).
                    main_wm_inj = getattr(getattr(model, "memory", None),
                                          "_last_injection", None)
                    # v15 discrete-key copy-head activity, snapshotted from the
                    # MAIN forward (same clobber hazard as the WM injection: aux
                    # extra-forwards overwrite _last_copy_gate_eff / the WM's
                    # _last_match_exists). copy_g_eff is (N_recall, 1) at the
                    # mem_read_mask answer positions; match_exists is (B, T) over
                    # the whole batch. Both None unless use_copy_head + discrete.
                    main_copy_g_eff = getattr(model, "_last_copy_gate_eff", None)
                    main_match_exists = getattr(
                        getattr(model, "memory", None), "_last_match_exists", None)
                    main_read_mask = mem_read_mask
                    # Same hazard for PKM: the aux extra-forwards (latent_cotrain /
                    # latent_reasoning / gate_calibration) run through PKM and
                    # clobber _last_slot_idx/_last_weights — which BOTH the
                    # _pkm_diversity_loss AND the pkm(slots/H,top) health log read.
                    # Snapshot the MAIN forward's so the PKM-bootstrap monitor (the
                    # one that caught v12's top 0.008→0.17) reflects the real corpus
                    # batch, not the tiny batch-1 pointer-chase latent forward that
                    # v13 newly enables. Restored just before those consumers below.
                    _pkm_mon = getattr(model, "pkm_layer", None)
                    main_pkm_slot_idx = (getattr(_pkm_mon, "_last_slot_idx", None)
                                         if _pkm_mon is not None else None)
                    main_pkm_weights = (getattr(_pkm_mon, "_last_weights", None)
                                        if _pkm_mon is not None else None)
                    loss = lm_loss + args.aux_weight * aux_loss
                    loss = loss + _z_loss_term(logits, args.z_loss)
                    if args.output_gate and args.gate_entropy_aux_weight > 0.0:
                        loss = loss + args.gate_entropy_aux_weight * gate_aux_loss
                    if args.gist_loss_weight > 0.0:
                        loss = loss + args.gist_loss_weight * gist_loss
                    # Logit-KD: add lambda * KL(teacher || student). kd_loss is a
                    # zero scalar when KD is off (no live teacher AND no offline
                    # store / weight 0) → no change to the loss or backward graph.
                    if ((kd_teacher is not None or kd_logit_store is not None)
                            and args.distill_weight > 0.0):
                        loss = loss + args.distill_weight * kd_loss
                    # CTX-NAMEKEY addressing aux (learned WM addresser): train the
                    # ctx read attention to land on the binding the deterministic
                    # lexical code identifies as correct, ONLY at recall answer-
                    # span (mem_read_mask) positions. RAMPED weight (0 before
                    # start_step, linear to target over warmup) keeps it negligible
                    # while the trunk/PKM settle. Computed HERE — before the
                    # latent/gate_calib extra forwards clobber the WM's
                    # _last_read_attn_grad / _last_top_idx_buf. Fires every
                    # microbatch (NO n_micro scale — like lm_loss, averaged at the
                    # /n_micro backward). Default weight 0 = OFF (no aux added).
                    _ctx_addr_w = float(getattr(args, "ctx_addr_aux_weight", 0.0))
                    if (_ctx_addr_w > 0.0
                            and getattr(model, "use_memory", False)
                            and getattr(getattr(model, "memory", None),
                                        "ctx_namekey", False)):
                        _ctx_ramp = _latent_reasoning_ramp(
                            step,
                            int(getattr(args, "ctx_addr_aux_start_step", 0)),
                            int(getattr(args, "ctx_addr_aux_warmup_steps", 0)))
                        if _ctx_ramp > 0.0:
                            _ca = _ctx_addr_aux_loss(
                                model, x, mem_read_mask, doc_ids=doc_ids)
                            if _ca is not None:
                                _ca_loss, _ca_n, _ca_p = _ca
                                loss = loss + (_ctx_addr_w * _ctx_ramp) * _ca_loss
                                _ctx_addr_diag = (float(_ca_loss.detach()),
                                                  _ca_n, _ca_p, _ctx_ramp)
                    # v9: latent-thinking co-training — grad CE on the
                    # post-R-latent-think prediction so the trunk learns to do
                    # useful computation during thinking. Logs mean Δlogp (the
                    # "is thinking becoming useful" signal: climbs from ≈-7
                    # toward 0/positive). Requires --no-compile (extra forwards).
                    # Fire ONCE per optimizer step (last microbatch only): the
                    # latent loss runs ~5 extra eager forwards, so doing it on
                    # every microbatch was the main throughput killer. A single
                    # 24-position sample per step is plenty of signal. Scale by
                    # n_micro so the per-step gradient still matches
                    # --latent_cotrain_weight after the (loss / n_micro).backward().
                    if (getattr(args, "latent_cotrain_weight", 0.0) > 0.0
                            and step >= getattr(args, "latent_cotrain_start_step", 0)
                            and _is_last_micro
                            and thinking_token_id is not None
                            and int(thinking_token_id) != int(pad_token_id)):
                        _lc = latent_cotrain_loss(
                            model, x, y, R=args.latent_cotrain_R,
                            thinking_token_id=int(thinking_token_id),
                            sample_frac=args.latent_cotrain_sample_frac,
                            max_positions=args.latent_cotrain_max_positions,
                            max_prefix_len=128,
                            selective=bool(getattr(
                                args, "latent_cotrain_selective", False)),
                            pad_id=int(pad_token_id))
                        if _lc is not None:
                            _lc_loss, _lc_delta, _lc_n = _lc
                            loss = loss + (n_micro * args.latent_cotrain_weight) \
                                * _lc_loss
                            _latent_cotrain_diag = (_lc_delta, _lc_n)
                    # Depth-matched latent-REASONING co-train (the 2026-06-05 fix):
                    # answer-span CE on pointer-chase at R=depth, clean latent
                    # thread (WM off + film bypass inside the helper). Fires once
                    # per optimizer step; scaled by n_micro so the per-step
                    # gradient matches --latent_reasoning_weight after /n_micro.
                    _lr_start = int(getattr(
                        args, "latent_reasoning_start_step", 0))
                    # --latent_reasoning_aux_every K (2026-07-04, default 1 =
                    # every step, byte-identical): fire only every K-th
                    # optimizer step, at K x the weight — same expected
                    # gradient (see _latent_reasoning_aux_every_gate), fewer
                    # stalls from the aux's growing-thread forwards.
                    _lr_fire, _lr_every_mult = _latent_reasoning_aux_every_gate(
                        step, _lr_start,
                        int(getattr(args, "latent_reasoning_aux_every", 1)))
                    if (_latent_reasoner is not None
                            and _is_last_micro
                            and _lr_fire):
                        # Linear 0->1 weight ramp over the warmup window after
                        # the start step — keeps the aux gradient negligible
                        # while PKM bootstraps (the v12-destabilization fix).
                        _lr_ramp = _latent_reasoning_ramp(
                            step, _lr_start,
                            int(getattr(
                                args,
                                "latent_reasoning_weight_warmup_steps", 0)))
                        # Curriculum measured relative to the start step so the
                        # easy-first depth ramp aligns to the engaged window.
                        _lr_loss, _lr_rung = _latent_reasoner.step(
                            model, step - _lr_start, args.steps - _lr_start,
                            int(args.latent_reasoning_n))
                        loss = loss + (n_micro * args.latent_reasoning_weight
                                       * _lr_ramp * _lr_every_mult) * _lr_loss
                        _latent_reasoning_diag = (float(_lr_loss.detach()),
                                                  int(_lr_rung), float(_lr_ramp))
                        # Engagement kill-gate counters (recipe rule #1): the
                        # reason loss ACTUALLY ran this step — count it and
                        # track whether it stayed finite (a NaN/inf reason
                        # loss would silently corrupt the trunk gradient).
                        # NOTE: with aux_every=K this increments only 1 step in
                        # K, so fire_count by --engagement_check_step is ~1/K
                        # of the every-step rate — the engagement check only
                        # requires fire_count > 0 past the ramp window, so this
                        # is a non-issue unless K approaches the distance from
                        # ramp-end to the check step (warned about separately
                        # at K > 8 startup).
                        _latent_reason_fire_count += 1
                        _latent_reason_all_finite = (
                            _latent_reason_all_finite
                            and math.isfinite(_latent_reasoning_diag[0]))
                    # Gate-calibration: train the OUTPUT GATE (not the trunk) to
                    # fire think exactly where a latent think raises
                    # logp(true_next). The BCE flows ONLY into the grad-carrying
                    # gate-logit snapshot taken BEFORE the latent extra forward
                    # clobbered model._last_gate_logits. Fire once/step (last
                    # microbatch) and scale by n_micro to keep the per-step
                    # gradient equal to --gate_calibration_weight after the
                    # (loss / n_micro).backward(). Default weight 0 = OFF.
                    if (getattr(args, "gate_calibration_weight", 0.0) > 0.0
                            and step >= getattr(args, "gate_calibration_start_step", 0)
                            and args.output_gate
                            and _is_last_micro
                            and main_gate_logits is not None
                            and thinking_token_id is not None
                            and int(thinking_token_id) != int(pad_token_id)):
                        _gc = compute_gate_calibration_loss(
                            model, x, y, main_gate_logits,
                            thinking_token_id=int(thinking_token_id),
                            latent_R=int(args.gate_calibration_R),
                            sample_frac=float(args.gate_calibration_sample_frac),
                            max_positions=int(args.gate_calibration_max_positions),
                            sigma_low=float(args.gate_calibration_sigma_low),
                            sigma_high=float(args.gate_calibration_sigma_high),
                            # EOS targets are already -100 under
                            # --mask_eos_in_targets and filtered by the
                            # helper's targets!=-100 check, so eos_id=None.
                            eos_id=None)
                        if _gc is not None:
                            loss = loss + (n_micro * args.gate_calibration_weight) \
                                * _gc.loss
                            _gate_calib_diag = (_gc.target_frac_pos,
                                                _gc.mean_sigma, _gc.mean_delta,
                                                _gc.n_positions)
                    # PKM diversity-bonus: -H(slot-selection distribution) per
                    # head, averaged across batch and heads. We MAXIMISE entropy
                    # so the auxiliary loss is NEGATIVE entropy. This is the
                    # direct fix for v5-pkm's "4 % of slots cover 95 % of mass"
                    # failure mode. The slot indices are detached upstream so the
                    # router itself isn't trained to produce high entropy — we
                    # only nudge the *distribution* (via the value-table grad
                    # this implies for diverse retrievals).
                    # Restore the MAIN-forward PKM slot stats (clobbered by any
                    # aux extra-forward above) so the post-loop pkm(...) health
                    # log reads the real corpus batch.
                    if _pkm_mon is not None and main_pkm_slot_idx is not None:
                        _pkm_mon._last_slot_idx = main_pkm_slot_idx
                        _pkm_mon._last_weights = main_pkm_weights
                    (loss / n_micro).backward()
        _engagement_check_now = (int(getattr(args, "engagement_check_step", 0)) > 0
                                 and step == int(args.engagement_check_step))
        # Force the log-diagnostics block to also run at the engagement-check
        # step (even off the --log_every cadence) so the per-mechanism
        # engagement evaluation below can REUSE the diagnostics it computes
        # (pkm(αL=...,row=...), FiLM α, copy-gate stats) instead of
        # recomputing them — see "Engagement kill-gate" below.
        _log_this_step = (step % args.log_every == 0 or step == args.steps
                          or _engagement_check_now)
        # Per-layer diagnostics: grad norms must be read before clip; the
        # weight snapshot must be taken before opt.step().
        _blk_gnorms = _block_grad_norms(model) if _log_this_step else None
        _blk_wsnap = _block_weight_snapshot(model) if _log_this_step else None
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        _blk_uratios = (_block_update_ratios(model, _blk_wsnap)
                        if _log_this_step else None)
        losses.append(lm_loss.item())  # track LM loss alone for comparison
        if args.output_gate:
            g_detached = (main_gate.detach() if main_gate is not None
                          else model._last_gate.detach())
            ce_detached = ce_per_token.detach()
            emit_mask = g_detached > 0.5
            # CE averaged only over emitted positions — the fair comparison metric.
            emit_ce = (ce_detached[emit_mask].mean().item()
                       if emit_mask.any() else float("nan"))
            losses_gate_window.append((
                float(g_detached.mean().item()),
                float(emit_mask.float().mean().item()),
                float(ce_detached.mean().item()),
                emit_ce,
            ))

        if _log_this_step:
            now = time.perf_counter()
            # ×ddp_world_size so this is GLOBAL throughput (all ranks), directly
            # comparable to the single-GPU baseline. Per-rank does batch*T*ga.
            tok_per_sec = ((step - last_log_step) * args.batch * args.T
                           * args.grad_accum * ddp_world_size / (now - last_log))
            tloss_avg = sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:]))
            line = (f"{step:>6d}  {tok_per_sec:>8.0f}  "
                    f"{tloss_avg:>8.4f}  "
                    f"{scheduler.get_last_lr()[0]:>9.2e}")
            if args.gist_loss_weight > 0.0:
                line += f"  gist={gist_loss.item():.4f}"
            if ((kd_teacher is not None or kd_logit_store is not None)
                    and args.distill_weight > 0.0):
                line += f"  kd={kd_loss.item():.4f}"
            if getattr(args, "latent_cotrain_weight", 0.0) > 0.0 and \
                    _latent_cotrain_diag is not None:
                _d, _n = _latent_cotrain_diag
                line += f"  latent(Δlogp={_d:+.3f},n={_n})"
            if _latent_reasoner is not None and \
                    _latent_reasoning_diag is not None:
                _rl, _rr, _rmp = _latent_reasoning_diag
                line += f"  reason(loss={_rl:.3f},R={_rr},ramp={_rmp:.2f})"
            if getattr(args, "gate_calibration_weight", 0.0) > 0.0 and \
                    _gate_calib_diag is not None:
                _t1, _sg, _gd, _gn = _gate_calib_diag
                # tgt1 = fraction where latent think helped (BCE target=1);
                # σ = mean gate sigmoid at scored positions; Δlogp = mean
                # latent-think benefit. tgt1>σ ⇒ gate UNDER-fires (miscal).
                line += (f"  gc(tgt1={_t1:.2f},σ={_sg:.2f},"
                         f"Δlogp={_gd:+.2f},n={_gn})")
            if getattr(args, "ctx_addr_aux_weight", 0.0) > 0.0 and \
                    _ctx_addr_diag is not None:
                # loss = attention-CE onto the correct binding (falls as the ctx
                # addresser learns); p = mean attention mass on the binding;
                # n = supervised recall positions; ramp = current weight ramp.
                _cal, _can, _cap, _car = _ctx_addr_diag
                line += (f"  addr(loss={_cal:.3f},p={_cap:.3f},"
                         f"n={_can},ramp={_car:.2f})")
            if args.feedback != "none" or fb_xattn_pairs:
                alphas = model.feedback_alphas()
                if not alphas:
                    pass
                elif isinstance(alphas[0], tuple):
                    # Sparse-FiLM:    (target, source_int, alpha)
                    # Cross-attn:     (target, sources_tuple, alpha)
                    if fb_xattn_pairs:
                        # Show first 4 (target<-K:α) entries to keep line short.
                        head_n = 4
                        items = [f"{t}<-{len(srcs)}:{a:+.3f}"
                                 for t, srcs, a in alphas[:head_n]]
                        suffix = f",… ({len(alphas) - head_n} more)" if len(alphas) > head_n else ""
                        line += "  α=[" + ",".join(items) + suffix + "]"
                    else:
                        line += "  α=[" + ",".join(
                            f"{t}<-{s}:{a:+.3f}" for t, s, a in alphas
                        ) + "]"
                elif isinstance(alphas[0], list):
                    # Dense multi-scale: max |α| per layer.
                    summary = [max(abs(a) for a in row) for row in alphas]
                    line += f"  max|α|=[{','.join(f'{a:.3f}' for a in summary)}]"
                else:
                    line += f"  α=[{','.join(f'{a:+.3f}' for a in alphas)}]"
                # Effective FiLM modulation per pair: RMS(out−x)/RMS(x) from
                # the last loss-bearing forward (stashed detached by
                # FeedbackProjection._stash_mod_rms; .item() only here at
                # log cadence). α alone is NOT a utilization measure — the
                # contribution is α·‖W·h‖ (2026-07-02 forensics) — THIS is
                # the live number that tracks whether FiLM is doing work.
                _sf = getattr(model, "sparse_feedback", None)
                if _sf is not None:
                    _effs = []
                    for _k in sorted(_sf.keys(), key=int):
                        _m = getattr(_sf[_k], "_last_mod_rms", None)
                        if _m is not None:
                            _effs.append(f"{_k}:{float(_m):.3f}")
                    if _effs:
                        line += "  filmEff=[" + ",".join(_effs) + "]"
            if args.output_gate and losses_gate_window:
                recent = losses_gate_window[-args.log_every:]
                mean_g = sum(r[0] for r in recent) / len(recent)
                emit_frac = sum(r[1] for r in recent) / len(recent)
                raw_ce = sum(r[2] for r in recent) / len(recent)
                valid_emit = [r[3] for r in recent if r[3] == r[3]]  # drop NaN
                emit_ce = sum(valid_emit) / len(valid_emit) if valid_emit else float("nan")
                if args.gate_warmup_steps > 0:
                    progress = min(1.0, step / args.gate_warmup_steps)
                    floor_now = (1.0 - progress) * 1.0 + progress * args.gate_floor_min
                    gf = f",floor={floor_now:.2f}"
                else:
                    gf = (f",floor={args.gate_floor_min:.2f}"
                          if args.gate_floor_min > 0 else "")
                ece_str = f"{emit_ce:.4f}" if emit_ce == emit_ce else "nan"
                line += (f"  gate(g={mean_g:.3f},emit={emit_frac:.2f},"
                         f"emit_ce={ece_str},ce={raw_ce:.4f}{gf})")
            if args.enable_thinking_token and think_stats_window:
                recent = think_stats_window[-args.log_every:]
                normal_items = sum(r.get("normal_items", 0.0) for r in recent)
                normal_think = sum(r.get("normal_think", 0.0) for r in recent)
                normal_explore = sum(r.get("normal_explore", 0.0) for r in recent)
                cont_items = sum(r.get("cont_items", 0.0) for r in recent)
                cont_think = sum(r.get("cont_think", 0.0) for r in recent)
                cont_explore = sum(r.get("cont_explore", 0.0) for r in recent)
                replay_items = sum(r.get("replay_items", 0.0) for r in recent)
                replay_think = sum(r.get("replay_think", 0.0) for r in recent)
                gate_emit_items = sum(r.get("gate_emit_items", 0.0) for r in recent)
                forced_emit = sum(r.get("forced_emit", 0.0) for r in recent)
                answer_ce = sum(r.get("answer_ce", 0.0) for r in recent) / len(recent)
                think_rate = ((normal_think + cont_think)
                              / max(1.0, normal_items + cont_items))
                explore_rate = ((normal_explore + cont_explore)
                                / max(1.0, normal_items + cont_items))
                forced_rate = forced_emit / max(1.0, cont_items)
                queue_len = recent[-1].get("queue_len", 0.0)
                q_mean_depth = recent[-1].get("queue_mean_depth", 0.0)
                q_max_depth = recent[-1].get("queue_max_depth", 0.0)
                q_dropped = recent[-1].get("queue_dropped", 0.0)
                replay_rate = replay_think / max(1.0, replay_items)
                curr = recent[-1].get("curriculum", 1.0)
                explore_prob = recent[-1].get("explore_prob", args.think_explore_prob)
                lambda_eff = recent[-1].get("lambda", args.think_lambda)
                gate_thr = recent[-1].get("gate_threshold", args.think_gate_threshold)
                q_pressure = recent[-1].get("queue_pressure", 0.0)
                depth_cap = recent[-1].get("safety_max_depth", args.think_safety_max_depth)
                traj_recent = think_closed_traj_window[-args.log_every:]
                traj_nll = (sum(traj_recent) / len(traj_recent)
                            if traj_recent else float("nan"))
                traj_str = f"{traj_nll:.4f}" if traj_nll == traj_nll else "nan"
                line += (f"  think(rate={think_rate:.3f},ans_ce={answer_ce:.4f},"
                          f"traj={traj_str},q={queue_len:.0f},"
                          f"depth={q_mean_depth:.2f}/{q_max_depth:.0f},"
                          f"work={cont_items:.0f}/{replay_items:.0f},"
                          f"forced={forced_rate:.3f},explore={explore_rate:.3f},"
                          f"replay_think={replay_rate:.3f},"
                          f"gate_emit={gate_emit_items:.0f},"
                          f"curr={curr:.2f},eps={explore_prob:.3f},"
                          f"lam={lambda_eff:.3f},thr={gate_thr:.3f},"
                          f"bp={q_pressure:.2f},"
                          f"cap={depth_cap:.0f},"
                          f"drop={q_dropped:.0f})")
            if _blk_gnorms is not None:
                gn, ur = _blk_gnorms, _blk_uratios
                n = len(gn)
                mid = n // 2
                line += (f"  gnorm(L0={gn[0]:.2e},L{mid}={gn[mid]:.2e},"
                         f"L{n-1}={gn[n-1]:.2e},last/first="
                         f"{gn[-1] / max(gn[0], 1e-12):.1f})")
                line += (f"  uratio(L0={ur[0]:.1e},L{mid}={ur[mid]:.1e},"
                         f"L{n-1}={ur[n-1]:.1e})")
            # PKM live diagnostics (v7.1): is the table actually waking up?
            #   αL      = learned scalar gate (init 0)
            #   αeff    = α + sign(α)·alpha_floor (the magnitude that
            #             actually scales PKM output in the forward)
            #   row     = mean row-norm of value table / expected init norm.
            #             >1 means rows have GROWN from init; <1 means they
            #             shrunk. =1 exactly means the table is frozen.
            #             (Replaces the misleading `v_std` diagnostic which
            #             is invariant under updates that preserve overall
            #             Gaussian distribution — frozen and learning-but-
            #             centred values both gave std≈1.)
            #   slots/H = unique slots hit this microbatch (out of n_keys²)
            #   top     = mass on the single hottest slot (lower=more diverse)
            #   ε       = current ε-greedy exploration rate
            #   φ       = current α-floor (decaying from start to 0)
            if (getattr(args, "use_pkm", False)
                    and hasattr(model, "pkm_layer")):
                pkm = model.pkm_layer
                with torch.no_grad():
                    aL = float(pkm.out_alpha.detach()) if pkm.use_output_gate else float("nan")
                    floor = float(getattr(pkm, "alpha_floor", 0.0))
                    sign = 1.0 if aL >= 0.0 or abs(aL) < 1e-3 else -1.0
                    aEff = aL + sign * floor if pkm.use_output_gate else float("nan")
                    # Row-norm drift: mean over rows of ||v_row|| / expected_init.
                    rn_mean = float(torch.stack([
                        emb.weight.float().norm(dim=-1).mean()
                        for emb in pkm.values
                    ]).mean())
                    init_norm = float(pkm._expected_init_row_norm)
                    rn_ratio = rn_mean / max(init_norm, 1e-9)
                    eps = float(getattr(pkm, "random_slot_epsilon", 0.0))
                    n_slots = pkm.n_keys * pkm.n_keys
                    if pkm._last_slot_idx is not None:
                        idx = pkm._last_slot_idx       # (B, T, H, top_k)
                        w = pkm._last_weights          # (B, T, H, top_k)
                        H_ = pkm.n_heads
                        slot_mass = torch.zeros(H_, n_slots,
                                                 device=idx.device, dtype=torch.float32)
                        for h_ in range(H_):
                            slot_mass[h_].scatter_add_(
                                0, idx[:, :, h_, :].reshape(-1),
                                w[:, :, h_, :].reshape(-1).float())
                        slot_mass = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                        unique_hit = int((slot_mass > 0).sum(dim=-1).float().mean())
                        top_share = float(slot_mass.max(dim=-1).values.mean())
                    else:
                        unique_hit, top_share = 0, float("nan")
                line += (f"  pkm(αL={aL:+.3f},αeff={aEff:+.3f},"
                         f"row={rn_ratio:.3f},"
                         f"slots/H={unique_hit}/{n_slots},top={top_share:.3f},"
                         f"ε={eps:.2f},φ={floor:.2f})")
            # WorkingMemory (dynamic RAG) liveness. WM is read/injected ONLY at
            # think positions; the supervised gradient comes from the LAST think
            # in each burst predicting the next REAL token (that target is NOT
            # masked). inj = mean ‖injection‖ at think positions (0 ⇒ WM inert /
            # no think tokens in batch), think% = fraction of think positions in
            # the batch, Wproj = ‖W_proj.weight‖ (drifts from init as WM learns).
            if (getattr(args, "use_memory", False)
                    and hasattr(model, "memory")
                    and getattr(model.memory, "_last_injection", None) is not None):
                with torch.no_grad():
                    inj = main_wm_inj                                    # (B,T,d) main fwd
                    tmask = (x == int(thinking_token_id))                # (B,T)
                    n_think = int(tmask.sum())
                    if (inj is not None and inj.shape[:2] == tmask.shape
                            and n_think > 0):
                        inj_norm = float(
                            inj[tmask].float().norm(dim=-1).mean())
                    else:
                        inj_norm = 0.0
                    think_frac = float(tmask.float().mean())
                    wproj = float(model.memory.W_proj.weight.float().norm())
                line += (f"  wm(inj={inj_norm:.3f},think%={think_frac*100:.1f},"
                         f"Wproj={wproj:.2f})")
            # v15 discrete-key copy-head diagnostic. Proves the WM is ACTIVE:
            # copy_g = mean effective copy gate at recall (mem_read_mask)
            # positions; m%R / m%G = discrete-address match rate at recall vs
            # general positions (recall ≫ general ⇒ the discrete addressing is
            # firing where the answer-token CE wants it and staying quiet on
            # general text — the "don't copy" the 84% general mix teaches).
            if (getattr(args, "use_copy_head", False)
                    and main_copy_g_eff is not None):
                with torch.no_grad():
                    cg = float(main_copy_g_eff.float().mean())
                    m_recall = m_gen = 0.0
                    if (main_match_exists is not None
                            and main_read_mask is not None
                            and main_match_exists.shape == main_read_mask.shape):
                        rm = main_read_mask.bool()
                        me = main_match_exists.bool()
                        if bool(rm.any()):
                            m_recall = float(me[rm].float().mean())
                            # Engagement kill-gate counter (recipe rule #1):
                            # cumulative count of recall positions where a
                            # matched binding existed, i.e. where the copy
                            # gate's g_eff = g·1{match} COULD have fired
                            # (`_apply_copy_head` in model.py). Hooked here
                            # (the existing per-log-step copy diagnostic) per
                            # the recipe rather than every microbatch.
                            _copy_fire_count += int((me & rm).sum().item())
                        if bool((~rm).any()):
                            m_gen = float(me[~rm].float().mean())
                line += (f"  copy(g={cg:.4f},m%R={m_recall*100:.1f},"
                         f"m%G={m_gen*100:.2f})")
            # Engagement kill-gate, part 4: the step-N per-mechanism
            # evaluation (recipe rule #1). Runs at most ONCE (the loop only
            # reaches step==engagement_check_step a single time). Reuses the
            # diagnostics computed above (aL/rn_ratio for PKM, `alphas` for
            # FiLM, the copy-gate bias + _copy_fire_count for WM) instead of
            # recomputing them; only the latent-reasoning counters are its
            # own state (tracked at their real computation sites, not here).
            if _engagement_check_now:
                _eng_results: list[tuple[str, bool, str]] = []
                if getattr(args, "use_pkm", False) and hasattr(model, "pkm_layer"):
                    _ok, _detail = _pkm_engaged(
                        aL, rn_ratio,
                        alpha_min=float(getattr(args, "engage_pkm_alpha_min", 0.02)),
                        row_min=float(getattr(args, "engage_pkm_row_min", 1.02)))
                    _eng_results.append(("PKM", _ok, _detail))
                if args.feedback == "film":
                    _ok, _detail = _film_engaged(
                        alphas, alpha_min=float(
                            getattr(args, "engage_film_alpha_min", 1e-3)))
                    _eng_results.append(("FiLM", _ok, _detail))
                if getattr(args, "use_copy_head", False):
                    _bias_now = float(
                        model.copy_head.gate.bias.detach().mean())
                    _bias_init = float(getattr(args, "copy_gate_bias_init", -6.0))
                    _ok, _detail = _wm_copy_engaged(
                        _bias_now, _bias_init, _copy_fire_count,
                        bias_delta_min=float(
                            getattr(args, "engage_copy_bias_delta_min", 0.05)))
                    _eng_results.append(("WM-copy", _ok, _detail))
                if float(getattr(args, "latent_reasoning_weight", 0.0)) > 0.0:
                    _lre_start = int(getattr(
                        args, "latent_reasoning_start_step", 0))
                    _lre_end = _lre_start + int(getattr(
                        args, "latent_reasoning_weight_warmup_steps", 0))
                    _ok, _detail = _latent_engaged(
                        _latent_reasoner is not None,
                        _latent_reason_fire_count, _latent_reason_all_finite,
                        step=step, start_step=_lre_start, end_step=_lre_end)
                    _eng_results.append(("latent-reasoning", _ok, _detail))
                engagement_report(
                    step, _eng_results,
                    action=str(getattr(args, "engagement_check_action", "abort")),
                    is_main=is_main)
            if is_main:
                print(line)
            if tb is not None:
                tb.add_scalar("train/loss", tloss_avg, step)
                tb.add_scalar("train/ppl", float(torch.tensor(tloss_avg).exp()), step)
                tb.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                tb.add_scalar("train/tok_per_sec", tok_per_sec, step)
                if ((kd_teacher is not None or kd_logit_store is not None)
                        and args.distill_weight > 0.0):
                    tb.add_scalar("train/kd_loss", float(kd_loss.item()), step)
                if _blk_gnorms is not None:
                    for L, (g, u) in enumerate(zip(_blk_gnorms, _blk_uratios)):
                        tb.add_scalar(f"layer_grad_norm/L{L:02d}", g, step)
                        tb.add_scalar(f"layer_update_ratio/L{L:02d}", u, step)
                    tb.add_scalar("layer_grad_norm/last_over_first",
                                  _blk_gnorms[-1] / max(_blk_gnorms[0], 1e-12),
                                  step)
                if (getattr(args, "use_pkm", False)
                        and hasattr(model, "pkm_layer")):
                    pkm = model.pkm_layer
                    with torch.no_grad():
                        if pkm.use_output_gate:
                            tb.add_scalar("pkm/alpha_learned",
                                          float(pkm.out_alpha.detach()), step)
                        floor = float(getattr(pkm, "alpha_floor", 0.0))
                        tb.add_scalar("pkm/alpha_floor", floor, step)
                        rn_mean = float(torch.stack([
                            emb.weight.float().norm(dim=-1).mean()
                            for emb in pkm.values
                        ]).mean())
                        init_norm = float(pkm._expected_init_row_norm)
                        tb.add_scalar("pkm/row_norm_mean", rn_mean, step)
                        tb.add_scalar("pkm/row_norm_ratio_vs_init",
                                      rn_mean / max(init_norm, 1e-9), step)
                        tb.add_scalar("pkm/epsilon",
                                      float(getattr(pkm, "random_slot_epsilon", 0.0)),
                                      step)
                        if pkm._last_slot_idx is not None:
                            n_slots = pkm.n_keys * pkm.n_keys
                            idx = pkm._last_slot_idx
                            w_ = pkm._last_weights
                            H_ = pkm.n_heads
                            slot_mass = torch.zeros(H_, n_slots,
                                                     device=idx.device,
                                                     dtype=torch.float32)
                            for h_ in range(H_):
                                slot_mass[h_].scatter_add_(
                                    0, idx[:, :, h_, :].reshape(-1),
                                    w_[:, :, h_, :].reshape(-1).float())
                            slot_mass = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                            uniq = (slot_mass > 0).sum(dim=-1).float().mean()
                            top = slot_mass.max(dim=-1).values.mean()
                            ent = -(slot_mass * slot_mass.clamp_min(1e-12).log()).sum(dim=-1).mean()
                            tb.add_scalar("pkm/unique_slots_per_head", float(uniq), step)
                            tb.add_scalar("pkm/top_slot_share", float(top), step)
                            tb.add_scalar("pkm/slot_entropy", float(ent), step)
                if args.output_gate and losses_gate_window:
                    tb.add_scalar("gate/think_frac", 1.0 - emit_frac, step)
                    tb.add_scalar("gate/mean_gate", mean_g, step)
                    tb.add_scalar("gate/raw_ce", raw_ce, step)
                    if emit_ce == emit_ce:  # not NaN
                        tb.add_scalar("gate/emit_ce", emit_ce, step)
                    if args.gate_warmup_steps > 0 or args.gate_floor_min > 0:
                        if args.gate_warmup_steps > 0:
                            progress = min(1.0, step / args.gate_warmup_steps)
                            gate_floor = (1.0 - progress) + progress * args.gate_floor_min
                        else:
                            gate_floor = args.gate_floor_min
                        tb.add_scalar("gate/floor", gate_floor, step)
                if (getattr(args, "gate_calibration_weight", 0.0) > 0.0
                        and _gate_calib_diag is not None):
                    _t1, _sg, _gd, _gn = _gate_calib_diag
                    tb.add_scalar("gate_calib/target_frac_pos", _t1, step)
                    tb.add_scalar("gate_calib/mean_sigma", _sg, step)
                    tb.add_scalar("gate_calib/mean_delta_logp", _gd, step)
                if (getattr(args, "latent_cotrain_weight", 0.0) > 0.0
                        and _latent_cotrain_diag is not None):
                    _d, _n = _latent_cotrain_diag
                    tb.add_scalar("latent_cotrain/delta_logp", _d, step)
                if (_latent_reasoner is not None
                        and _latent_reasoning_diag is not None):
                    _rl, _rr, _rmp = _latent_reasoning_diag
                    tb.add_scalar("latent_reasoning/loss", _rl, step)
                    tb.add_scalar("latent_reasoning/rung", _rr, step)
                    tb.add_scalar("latent_reasoning/weight_ramp", _rmp, step)
                if args.enable_thinking_token and think_stats_window:
                    tb.add_scalar("think/rate", think_rate, step)
                    tb.add_scalar("think/explore_rate", explore_rate, step)
                    tb.add_scalar("think/explore_prob", explore_prob, step)
                    tb.add_scalar("think/curriculum", curr, step)
                    tb.add_scalar("think/lambda", lambda_eff, step)
                    tb.add_scalar("think/gate_threshold", gate_thr, step)
                    tb.add_scalar("think/queue_pressure", q_pressure, step)
                    tb.add_scalar("think/safety_max_depth", depth_cap, step)
                    tb.add_scalar("think/replay_think_rate", replay_rate, step)
                    tb.add_scalar("think/answer_ce", answer_ce, step)
                    if traj_nll == traj_nll:
                        tb.add_scalar("think/trajectory_nll", traj_nll, step)
                    tb.add_scalar("think/queue_len", queue_len, step)
                    tb.add_scalar("think/queue_mean_depth", q_mean_depth, step)
                    tb.add_scalar("think/queue_max_depth", q_max_depth, step)
                    tb.add_scalar("think/forced_emit_rate", forced_rate, step)
                    tb.add_scalar("think/queue_dropped", q_dropped, step)
                    tb.add_scalar("think/cont_items", cont_items, step)
                    tb.add_scalar("think/replay_items", replay_items, step)
                # FiLM α values.
                if args.feedback != "none" or fb_xattn_pairs:
                    for entry in model.feedback_alphas():
                        if isinstance(entry, tuple) and len(entry) == 3:
                            t_idx, s_idx, alpha = entry
                            label = (f"alpha/t{t_idx}"
                                     if isinstance(s_idx, tuple)
                                     else f"alpha/t{t_idx}_s{s_idx}")
                            tb.add_scalar(label, alpha, step)
            last_log = now
            last_log_step = step

        # DDP: rank 0 owns validation (collective-free; other ranks just wait
        # at the next backward all-reduce until rank 0 rejoins).
        if (step % args.val_every == 0 or step == args.steps) and is_main:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vx, vy, *_vrest = vbatch
                    vx, vy = vx.to("cuda"), vy.to("cuda")
                    vdoc_ids = _vrest[0].to("cuda") if _vrest else None
                    vlogits = model(vx, doc_ids=vdoc_ids)
                    if args.enable_thinking_token:
                        vloss = cross_entropy_masking_token(
                            vlogits.reshape(-1, vlogits.shape[-1]), vy.reshape(-1),
                            int(thinking_token_id),
                            reduction="mean",
                        )
                    else:
                        vloss = F.cross_entropy(
                            vlogits.reshape(-1, vlogits.shape[-1]), vy.reshape(-1),
                        )
                    val_loss += vloss.item() * vx.numel()
                    n_val += vx.numel()
                    if n_val >= 64 * args.T:                # cap val
                        break
            val_loss /= n_val
            ppl = float(torch.tensor(val_loss).exp())
            print(f"        VAL  loss={val_loss:.4f}  ppl={ppl:.2f}")
            if tb is not None:
                tb.add_scalar("val/loss", val_loss, step)
                tb.add_scalar("val/ppl", ppl, step)
            model.train()
            torch.cuda.empty_cache()

        # Per-feature usefulness probe: ablate each mechanism on a held-out
        # batch and log the CE rise (load-bearing iff Δce > 0). Runs no_grad
        # and restores every poked attribute, so it can't perturb training.
        if (next_feature_probe_at and is_main
                and tokens_seen >= next_feature_probe_at):
            from experiments.feature_probe import (
                run_feature_probe, format_feature_probe)
            try:
                _fp_batch = next(iter(val_loader))
                _fpx, _fpy, *_fprest = _fp_batch
                _fpx, _fpy = _fpx.to("cuda"), _fpy.to("cuda")
                _fpdoc = _fprest[0].to("cuda") if _fprest else None
                _fp_metrics = run_feature_probe(
                    model, _fpx, _fpy, doc_ids=_fpdoc,
                    thinking_token_id=thinking_token_id)
                print("        " + format_feature_probe(_fp_metrics))
                if tb is not None:
                    for _k, _v in _fp_metrics.items():
                        tb.add_scalar(f"probe/{_k}", float(_v), step)
            except Exception as _e:  # never let the probe kill a 20h run
                print(f"        [feature-probe] skipped ({_e})")

            # WM load-bearing signal — the natural-text probe batch above has
            # zero think tokens, so WM's ablation-delta there is ≈0 BY DESIGN
            # (WM reads only at think positions). The real WM signal comes from
            # held-out long-context recall WITH think tokens: recall_full vs
            # recall with the WM read mean-ablated. delta > 0 ⇒ WM load-bearing.
            _wm_recall_path = getattr(
                args, "feature_probe_wm_recall_path", "") or ""
            if (_wm_recall_path and getattr(model, "memory", None) is not None
                    and thinking_token_id is not None):
                try:
                    from experiments.eval_longctx_recall import (
                        eval_longctx_recall)
                    _wm_n = int(getattr(args, "feature_probe_wm_recall_n", 64))
                    _full = eval_longctx_recall(
                        model, tok, _wm_recall_path, n=_wm_n,
                        wm_ablate="none")
                    _abl = eval_longctx_recall(
                        model, tok, _wm_recall_path, n=_wm_n,
                        wm_ablate="mean")
                    _wm_recall = _full["recall"]
                    _wm_recall_delta = _wm_recall - _abl["recall"]
                    print(f"        [wm-recall] recall={_wm_recall:.3f} "
                          f"Δ(full-ablated)={_wm_recall_delta:+.3f} "
                          f"think_frac={_full['think_frac']:.3f} "
                          f"(n={int(_full['n_total'])})")
                    if tb is not None:
                        tb.add_scalar("probe/wm_recall", _wm_recall, step)
                        tb.add_scalar("probe/wm_recall_delta",
                                      _wm_recall_delta, step)
                        tb.add_scalar("probe/wm_recall_think_frac",
                                      _full["think_frac"], step)
                except Exception as _e:  # never let the probe kill a 20h run
                    print(f"        [wm-recall] skipped ({_e})")
                model.train()

            # v14 live WM-usefulness probe on REALISTIC code/agentic recall.
            # Teacher-forced exact recall ON (use_memory) vs full_off (WM
            # disabled): Δ>0 ⇒ WM is load-bearing. Also logs the learned WM
            # read_alpha and the copy gate (if present) — the two scalars whose
            # growth tells us the addressing/readout is engaging. Default off
            # (empty path) so non-v14 runs are unaffected.
            _code_recall_path = getattr(
                args, "feature_probe_code_recall_path", "") or ""
            if (_code_recall_path and getattr(model, "memory", None) is not None):
                try:
                    from experiments.eval_code_recall import (
                        eval_teacher_forced as _eval_cr, _load as _load_cr)
                    _cr_n = int(getattr(args, "feature_probe_code_recall_n", 200))
                    _cr_recs = _load_cr(_code_recall_path, n=_cr_n)
                    _cr_on = _eval_cr(model, tok, _cr_recs, wm_arm="on",
                                      max_T=int(args.T))
                    _cr_off = _eval_cr(model, tok, _cr_recs, wm_arm="full_off",
                                       max_T=int(args.T))
                    _cr_delta = _cr_on["recall"] - _cr_off["recall"]
                    _ra = float(getattr(model.memory, "read_alpha",
                                        torch.zeros(())).item())
                    _cg = getattr(model, "_last_copy_gate", None)
                    _cg_s = (f" copy_g={float(_cg.mean().item()):.3f}"
                             if _cg is not None else "")
                    print(f"        [code-recall] recall_on={_cr_on['recall']:.3f}"
                          f" off={_cr_off['recall']:.3f} Δ={_cr_delta:+.3f}"
                          f" first_on={_cr_on['first_recall']:.3f}"
                          f" read_alpha={_ra:+.3f}{_cg_s}"
                          f" (n={int(_cr_on['n_total'])})")
                    if tb is not None:
                        tb.add_scalar("probe/code_recall_on",
                                      _cr_on["recall"], step)
                        tb.add_scalar("probe/code_recall_delta", _cr_delta, step)
                        tb.add_scalar("probe/code_recall_read_alpha", _ra, step)
                except Exception as _e:  # never let the probe kill a 20h run
                    print(f"        [code-recall] skipped ({_e})")
                model.train()
            torch.cuda.empty_cache()
            while next_feature_probe_at <= tokens_seen:
                next_feature_probe_at += int(args.feature_probe_every_tokens)

        if (args.probe_humaneval_every_tokens > 0 and is_main
                and tokens_seen - tokens_at_last_probe
                    >= args.probe_humaneval_every_tokens):
            from experiments.probe_humaneval import run_humaneval_probe
            n_probe = (args.probe_humaneval_n_problems
                       if args.probe_humaneval_n_problems > 0 else None)
            try:
                res = run_humaneval_probe(
                    model, tok,
                    probe_path=args.probe_humaneval_path,
                    max_gen=args.probe_humaneval_max_gen,
                    n_problems=n_probe,
                    use_thinking=bool(args.output_gate
                                       and thinking_token_id is not None),
                    thinking_token_id=thinking_token_id,
                    gate_floor=float(args.gate_floor_min),
                    min_emit_before_eos=30,
                )
                print(f"        PROBE  pass@1={res['pass_rate']*100:.1f}% "
                      f"({res['n_passed']}/{res['n_total']})  "
                      f"emit={res['mean_emit_tokens']:.0f}tok  "
                      f"t={res['elapsed_s']:.1f}s  "
                      f"@tok={tokens_seen/1e6:.1f}M")
                if tb is not None:
                    tb.add_scalar("probe/pass_rate", res["pass_rate"], step)
                    tb.add_scalar("probe/n_passed", res["n_passed"], step)
            except Exception as e:
                print(f"        PROBE  ERROR: {e}")
            tokens_at_last_probe = tokens_seen
            torch.cuda.empty_cache()

        # ---- Mid-training HumanEval hook (auto_stop). ----
        # tokens_seen is the rough count of tokens consumed by the train
        # loop so far. At every `mid_eval_every_tokens` boundary, save a
        # ckpt, shell out to eval_humaneval.py, log pass-rate to TB,
        # append to controller, optionally stop.
        # Under DDP every rank consumes batch*T*grad_accum tokens per step, so
        # the global token count scales with world size.
        tokens_seen = step * args.batch * args.T * args.grad_accum * ddp_world_size
        if (mid_eval_controller is not None and is_main
                and tokens_seen >= next_eval_at):
            # Snapshot to a numbered ckpt — kept for the curve plot.
            stem = pathlib.Path(args.save_ckpt or "checkpoints/pretrain.pt").stem
            mid_path = pathlib.Path("checkpoints") / (
                f"{stem}_step{step}_tok{tokens_seen}.pt")
            mid_path.parent.mkdir(parents=True, exist_ok=True)
            _save_cfg = dict(
                vocab_size=model_vocab_size,
                tokenizer_base_vocab_size=tok.vocab_size,
                d_model=args.d_model, n_heads=args.n_heads,
                d_head=args.d_head, n_layers=n_layers_actual,
                max_T=args.max_T, feedback_mode=args.feedback,
                feedback_pairs=fb_pairs,
                feedback_self_k=args.feedback_self_k,
                feedback_alpha_mode=args.feedback_alpha_mode,
                feedback_alpha_init=float(
                    getattr(args, "feedback_alpha_init", 0.0)),
                # 2026-07-02 design review FIX 2/3 — persist the ACTUAL mode
                # this run used (not a hardcoded default) so reload is always
                # faithful: cfg.get(key, "none") on a ckpt saved before this
                # key existed correctly falls back to "none" (legacy).
                feedback_src_norm=str(getattr(args, "feedback_src_norm", "none")),
                arch=args.arch, layers_spec=args.layers,
                tokenizer=args.tokenizer,
                thinking_token_id=thinking_token_id,
                use_memory=bool(args.use_memory),
                mem_size=int(args.mem_size) if args.use_memory else 0,
                mem_dim=(int(args.mem_dim) if args.mem_dim > 0
                          else int(args.d_model)) if args.use_memory else 0,
                mem_inj_norm=str(getattr(args, "mem_inj_norm", "none")),
                # WM addressing/readout cfg (no/with state-dict footprint) so a
                # mid-eval ckpt reloads with the same WM as the final ckpt.
                mem_decoupled_kv=bool(getattr(args, "mem_decoupled_kv", False)),
                mem_key_from_embedding=bool(
                    getattr(args, "mem_key_from_embedding", False)),
                mem_key_window=int(getattr(args, "mem_key_window", 4)),
                use_copy_head=bool(getattr(args, "use_copy_head", False)),
                mem_discrete_key=bool(getattr(args, "mem_discrete_key", False)),
                mem_discrete_key_lexical=(
                    not bool(getattr(args, "mem_discrete_key_vstart", False))),
                mem_always_read=bool(getattr(args, "mem_always_read", False)),
                mem_copy_require_match=bool(
                    getattr(args, "mem_copy_require_match", True)),
                mem_discrete_key_match_window=int(
                    getattr(args, "mem_discrete_key_match_window", 32)),
                # SOFT NAME-SPAN addressing cfg (round-trip parity with the
                # discrete/ctx paths; eval_bracket reads these keys).
                mem_soft_namekey=bool(getattr(args, "mem_soft_namekey", False)),
                mem_soft_namekey_dim=int(
                    getattr(args, "mem_soft_namekey_dim", 64)),
                mem_soft_namekey_match_threshold=float(
                    getattr(args, "mem_soft_namekey_match_threshold", 0.5)),
                # CONTEXTUAL NAME-SPAN addressing (learned, no static hash). No
                # state-dict footprint beyond the ctxkey_* encoders → MUST be in
                # cfg so eval_bracket_structure.build_model_from_ckpt rebuilds the
                # same addresser on reload (matches the discrete-key handling).
                mem_ctx_namekey=bool(getattr(args, "mem_ctx_namekey", False)),
                mem_ctx_namekey_dim=int(getattr(args, "ctx_namekey_dim", 192)),
                mem_ctx_namekey_match_threshold=float(
                    getattr(args, "ctx_namekey_match_threshold", 0.5)),
                copy_head_gate_bias_init=float(
                    getattr(args, "copy_gate_bias_init", -6.0)),
                use_pkm=bool(getattr(args, "use_pkm", False)),
                pkm_after_layer=int(getattr(args, "pkm_after_layer", 14)),
                pkm_n_keys=int(getattr(args, "pkm_n_keys", 256)),
                pkm_n_heads=int(getattr(args, "pkm_n_heads", 4)),
                pkm_k_dim=int(getattr(args, "pkm_k_dim", 128)),
                pkm_top_k=int(getattr(args, "pkm_top_k", 32)),
                pkm_value_bf16=bool(getattr(args, "pkm_value_bf16", True)),
                # v7 PKM-bootstrap-fix package.
                pkm_score_norm=str(getattr(args, "pkm_score_norm", "layer")),
                pkm_value_init_std=float(getattr(args, "pkm_value_init_std", 1.0)),
                pkm_use_output_gate=bool(getattr(args, "pkm_use_output_gate", True)),
                output_gate=bool(args.output_gate
                                  or (args.enable_thinking_token
                                      and args.think_decision == "gate")),
                state_readonly_at_think=bool(
                    getattr(args, "state_readonly_at_think", False)),
                use_latent_feedback_adapter=bool(
                    getattr(args, "use_latent_feedback_adapter", False)),
                retrieval_input_additive=False,
                # Audited footgun fix (2026-07-01): d_ff and tie_embeddings were
                # never persisted, so a --tie_embeddings ckpt silently
                # reconstructed UNTIED on reload (eval_bracket_structure.py
                # never passed the kwarg) and d_ff relied entirely on
                # state-dict shape inference. d_ff already has a shape-infer
                # fallback in eval_bracket_structure.build_model_from_ckpt for
                # OLD ckpts that lack this key; the cfg value is now the
                # PRIMARY source going forward. 0 means "not overridden" (the
                # --d_ff CLI default), matching TinyLM's own 4*d_model default.
                d_ff=int(getattr(args, "d_ff", 0)),
                tie_embeddings=bool(getattr(args, "tie_embeddings", False)),
            )
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "config": _save_cfg}, str(mid_path))
            print(f"\n[mid-eval] saved ckpt at step={step} tokens={tokens_seen:,}"
                  f" → {mid_path}")

            # Decide whether to run the HumanEval subprocess. Two skip paths,
            # both leave the ckpt on disk and advance the counter (resume
            # artifact > HumanEval signal during pretrain):
            #   1. --mid_eval_save_only: explicit user opt-out.
            #   2. Auto-skip: trainer is using nearly all of GPU memory; the
            #      eval subprocess would OOM trying to load its own model copy
            #      on the same device (observed in v4 at step 1526).
            skip_eval = False
            skip_reason = ""
            if args.mid_eval_save_only:
                skip_eval = True
                skip_reason = "--mid_eval_save_only"
            elif args.mid_eval_min_free_gib > 0:
                free_b, _ = torch.cuda.mem_get_info()
                free_gib = free_b / (1024 ** 3)
                if free_gib < args.mid_eval_min_free_gib:
                    skip_eval = True
                    skip_reason = (
                        f"free GPU memory {free_gib:.2f} GiB < "
                        f"{args.mid_eval_min_free_gib:.2f} GiB — eval "
                        f"subprocess would OOM")

            if skip_eval:
                print(f"[mid-eval] SKIPPED HumanEval ({skip_reason}). "
                      f"Ckpt is on disk; advancing counter.")
                from experiments.eval_callback import EvalResult
                res = EvalResult(
                    humaneval_pass_rate=float("nan"),
                    mbpp_pass_rate=None,
                    tokens_seen=tokens_seen, step=step,
                    ckpt_path=str(mid_path),
                    raw_log_tail=f"<skipped: {skip_reason}>",
                )
            else:
                print(f"[mid-eval] running HumanEval (max_problems="
                      f"{args.mid_eval_n_problems}) ...")
                model.eval()
                res = run_eval(
                    str(mid_path), tokens_seen=tokens_seen, step=step,
                    n_problems=args.mid_eval_n_problems,
                    max_gen=args.mid_eval_max_gen,
                    use_thinking=bool(args.use_memory),
                    emit_threshold=0.5,
                    min_emit_before_eos=int(args.mid_eval_min_emit_before_eos),
                    gate_floor=float(args.gate_floor_min),
                )
                model.train()
            mid_eval_controller.append(res)
            print(f"[mid-eval] {mid_eval_controller.summary_line()}")
            if not skip_eval and res.humaneval_pass_rate != res.humaneval_pass_rate:  # NaN
                print("[mid-eval] WARNING: humaneval=NaN — eval subprocess "
                      "did not emit a parseable `pass@k =` line. Last 2 kB "
                      "of its stdout/stderr:")
                print(res.raw_log_tail)
            if tb is not None:
                tb.add_scalar("eval/humaneval", res.humaneval_pass_rate, step)
                tb.add_scalar("eval/humaneval_vs_tokens",
                              res.humaneval_pass_rate, tokens_seen // 1_000_000)
            # Advance the next-eval threshold to the next interval; if we
            # blew past several intervals (e.g. small batch * long step),
            # snap forward by multiples.
            while next_eval_at <= tokens_seen:
                next_eval_at += int(args.mid_eval_every_tokens)
            if args.auto_stop and mid_eval_controller.should_stop():
                print(f"[mid-eval] AUTO STOP — pass-rate plateaued. "
                      f"Last {args.auto_stop_k} intervals each gained "
                      f"<{args.auto_stop_threshold:.3f}. Stopping at "
                      f"step={step} tokens={tokens_seen:,}.")
                break

    secs = time.perf_counter() - t0
    print(f"\nDone in {secs:.0f}s ({secs/args.steps*1000:.0f} ms/step avg).")
    if tb is not None:
        tb.close()

    if args.save_ckpt and is_main:
        ckpt = {
            "state_dict": model.state_dict(),
            "step": locals().get("step", 0),
            "config": {
                "vocab_size": model_vocab_size,
                "tokenizer_base_vocab_size": tok.vocab_size,
                "use_memory": bool(args.use_memory),
                "mem_size": (int(args.mem_size) if args.use_memory else 0),
                "mem_decoupled_kv": bool(getattr(args, "mem_decoupled_kv", False)),
                # v14 WM-recall plumbing (no/with state-dict footprint as noted
                # in eval_bracket_structure.build_model_from_ckpt). Saved so a
                # reloaded ckpt reconstructs the same addressing/readout.
                "mem_key_from_embedding": bool(
                    getattr(args, "mem_key_from_embedding", False)),
                "mem_key_window": int(getattr(args, "mem_key_window", 4)),
                "use_copy_head": bool(getattr(args, "use_copy_head", False)),
                # v15 discrete-key WM (no state-dict footprint → MUST be in cfg so
                # eval_bracket_structure.build_model_from_ckpt re-activates the
                # discrete addressing + copy gating on reload, else the trained
                # copy head would address on the legacy cosine scores).
                "mem_discrete_key": bool(getattr(args, "mem_discrete_key", False)),
                "mem_discrete_key_lexical": (
                    not bool(getattr(args, "mem_discrete_key_vstart", False))),
                "mem_always_read": bool(getattr(args, "mem_always_read", False)),
                "mem_copy_require_match": bool(
                    getattr(args, "mem_copy_require_match", True)),
                "mem_discrete_key_match_window": int(
                    getattr(args, "mem_discrete_key_match_window", 32)),
                # SOFT NAME-SPAN addressing cfg (round-trip parity with the
                # discrete/ctx paths; eval_bracket reads these keys).
                "mem_soft_namekey": bool(
                    getattr(args, "mem_soft_namekey", False)),
                "mem_soft_namekey_dim": int(
                    getattr(args, "mem_soft_namekey_dim", 64)),
                "mem_soft_namekey_match_threshold": float(
                    getattr(args, "mem_soft_namekey_match_threshold", 0.5)),
                # CONTEXTUAL NAME-SPAN addressing (learned, no static hash) — cfg
                # so the reload path (eval_bracket_structure) rebuilds the same
                # ctx addresser. ctxkey_* encoders ARE in the state-dict; these
                # scalars are not, so they must travel in cfg.
                "mem_ctx_namekey": bool(getattr(args, "mem_ctx_namekey", False)),
                "mem_ctx_namekey_dim": int(getattr(args, "ctx_namekey_dim", 192)),
                "mem_ctx_namekey_match_threshold": float(
                    getattr(args, "ctx_namekey_match_threshold", 0.5)),
                "copy_head_gate_bias_init": float(
                    getattr(args, "copy_gate_bias_init", -6.0)),
                "mem_dim": ((int(args.mem_dim) if args.mem_dim > 0
                             else int(args.d_model)) if args.use_memory else 0),
                # 2026-07-02 design review FIX 3 — see the matching comment at
                # the mid-eval save site above.
                "mem_inj_norm": str(getattr(args, "mem_inj_norm", "none")),
                "use_pkm": bool(getattr(args, "use_pkm", False)),
                "pkm_after_layer": int(getattr(args, "pkm_after_layer", 14)),
                "pkm_n_keys": int(getattr(args, "pkm_n_keys", 256)),
                "pkm_n_heads": int(getattr(args, "pkm_n_heads", 4)),
                "pkm_k_dim": int(getattr(args, "pkm_k_dim", 128)),
                "pkm_top_k": int(getattr(args, "pkm_top_k", 32)),
                "pkm_value_bf16": bool(getattr(args, "pkm_value_bf16", True)),
                "pkm_score_norm": str(getattr(args, "pkm_score_norm", "layer")),
                "pkm_value_init_std": float(getattr(args, "pkm_value_init_std", 1.0)),
                "pkm_use_output_gate": bool(getattr(args, "pkm_use_output_gate", True)),
                "data_mix": args.data_mix,
                "think_burst_prob": args.think_burst_prob,
                "think_max_bursts": args.think_max_bursts,
                "think_max_burst_depth": args.think_max_burst_depth,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "d_head": args.d_head, "n_layers": n_layers_actual,
                "max_T": args.T, "feedback_mode": args.feedback,
                "feedback_pairs": fb_pairs,
                "feedback_xattn_pairs": fb_xattn_pairs,
                "feedback_xattn_heads": args.feedback_xattn_heads,
                "feedback_xattn_form": args.feedback_xattn_form,
                "feedback_self_k": args.feedback_self_k,
                "feedback_alpha_mode": args.feedback_alpha_mode,
                "feedback_alpha_init": float(
                    getattr(args, "feedback_alpha_init", 0.0)),
                # 2026-07-02 design review FIX 2 — persist the ACTUAL mode
                # this run used; cfg.get(key, "none") on a ckpt saved before
                # this key existed correctly falls back to "none" (legacy).
                "feedback_src_norm": str(
                    getattr(args, "feedback_src_norm", "none")),
                "arch": args.arch, "layers_spec": args.layers,
                "tokenizer": args.tokenizer,
                "enable_thinking_token": bool(args.enable_thinking_token),
                "thinking_token": args.thinking_token,
                "thinking_token_id": thinking_token_id,
                "think_lambda": args.think_lambda,
                "think_lambda_start": args.think_lambda_start,
                "think_curriculum_steps": args.think_curriculum_steps,
                "think_policy": args.think_policy,
                "think_decision": args.think_decision,
                "think_gate_threshold": args.think_gate_threshold,
                "think_gate_threshold_start": args.think_gate_threshold_start,
                "think_explore_prob": args.think_explore_prob,
                "think_explore_start_prob": args.think_explore_start_prob,
                "think_safety_max_depth": args.think_safety_max_depth,
                "think_safety_max_depth_start": args.think_safety_max_depth_start,
                "think_aux_normalize": args.think_aux_normalize,
                "think_aux_loss_scale": args.think_aux_loss_scale,
                "think_queue_accum_steps": args.think_queue_accum_steps,
                "think_queue_accum_max_steps": args.think_queue_accum_max_steps,
                "think_queue_drain_target": args.think_queue_drain_target,
                "think_backpressure_target": args.think_backpressure_target,
                "think_backpressure_max": args.think_backpressure_max,
                "think_backpressure_lambda": args.think_backpressure_lambda,
                "think_backpressure_threshold": args.think_backpressure_threshold,
                "think_backpressure_explore": args.think_backpressure_explore,
                "state_readonly_at_think": bool(
                    getattr(args, "state_readonly_at_think", False)),
                "use_latent_feedback_adapter": bool(
                    getattr(args, "use_latent_feedback_adapter", False)),
                "retrieval_input_additive": False,
                # Audited footgun fix (2026-07-01) — see the matching comment
                # at the mid-eval save site above for the full rationale.
                "d_ff": int(getattr(args, "d_ff", 0)),
                "tie_embeddings": bool(getattr(args, "tie_embeddings", False)),
            },
        }
        pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save_ckpt)
        print(f"Checkpoint saved to {args.save_ckpt}")

    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
