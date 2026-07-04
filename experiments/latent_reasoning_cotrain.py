"""Depth-matched latent-reasoning co-train for real pretrain (2026-06-05).

The validated fix for "latent thinking is inert in real pretrain". The shipped
`thinking.latent_cotrain_loss` supervises R latent steps to predict a RANDOM
natural-text next token; that signal is broken (natural next-token rarely needs
sequential depth) so the trunk learns "thinking isn't needed" (probe: mean
Δlogp -0.69, only 28% of positions benefit, gate AUC 0.43). Instead, draw from a
DEPTH-BOUND, NON-PARALLELIZABLE reasoning corpus (pointer-chase f^K) and supervise
the ANSWER with R = the problem's depth + a depth curriculum — the recipe
validated standalone in `latent_arith_real.py` (fair lift +0.40-0.65 over a
fully-trained no-think baseline; autonomous gate allocates exactly n hops,
0.94-1.00). This co-trains it at low weight ALONGSIDE the general pretrain mix so
the trunk keeps general/code ability (post-pretrain bolt-ons go inert / forget).

Clean latent thread: WorkingMemory is toggled OFF and FiLM bypassed ONLY during
the latent forwards (WM injects at think positions and contaminates the fed-back
hidden — the documented blocker; the validated run used --no_memory + _film_bypass).
The main general-data forward is untouched.

BATCHED growing-thread (2026-07-04, the "aux 2x cheaper" fix). The original
mechanism ran its n examples/step SEQUENTIALLY as n independent B=1 growing
threads — measured at ~25% GPU utilization (kernel-launch/latency-bound, not
compute-bound), ~5.6s of every 10.4s step in the running N1 config. Since all n
examples of a `step()` call already share the SAME rung R (`_pick_rung` is
called once per step, not once per example), they can run as ONE batched
growing thread of batch size n instead — see `_answer_span_latent_loss_batched`
for the full design (LEFT-pad + doc_ids state-isolation is the load-bearing
part: pad tokens WRITE to a linear-RNN's recurrent state, unlike causal-
attention padding, so naive left-padding would silently corrupt the real
prompt's starting state). `LatentReasoningCotrain(batch_examples=True)`
(default) uses the batched path; `batch_examples=False` is the escape hatch
that reproduces the exact old sequential behaviour (same example sampling
order either way — only how they're processed differs), kept for A/B
comparison against the running N1 if ever needed. Equivalence (loss + grads,
batched == mean-of-sequential) pinned in test_latent_reasoning_batched.py.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from experiments.thinking import clean_latent_thread


def _run_latent_forward(model, ids, embeds, return_hidden, doc_ids=None):
    """Free function (not a closure) so torch.utils.checkpoint's
    non-reentrant recompute can re-invoke it cleanly — mirrors
    experiments/model.py::_run_block, which notes closures over `self`
    cause occasional issues with the non-reentrant checkpoint path.

    `doc_ids` (n, T) | None: threaded straight to `TinyLM.forward`'s existing
    cross-document-isolation kwarg (model.py::_build_cu_seqlens). None (the
    default, and always the value used by the single-example
    `_answer_span_latent_loss` path) is byte-identical to the pre-batching
    behaviour. The batched growing-thread (`_answer_span_latent_loss_batched`)
    passes a real mask marking each row's left-pad prefix as a separate
    "document" from its real content, so the recurrent state resets to zero
    exactly at the pad/real boundary — see that function's docstring.

    Re-asserts `clean_latent_thread` on EVERY invocation (not just once for
    the whole `step()` call) — this is load-bearing, not cosmetic. Found
    empirically (2026-07-02) verifying the checkpoint fix on the real 32L
    config: `torch.utils.checkpoint`'s non-reentrant recompute can fire
    later, from an UN-NESTED `.backward()` call the caller (train_lm.py)
    issues after combining this loss with the rest of the step's loss terms
    — by which time `LatentReasoningCotrain.step()`'s own
    `with clean_latent_thread(...):` block (scoped only around the
    FORWARD) has already exited, restoring `model.use_memory` /
    `model._film_bypass` / `model.activation_checkpointing` to the real
    pretrain-config values. Without re-asserting here, the recompute
    silently runs a DIFFERENT graph than the original forward (WM
    re-enabled, K=3 self-feed engaged, per-block checkpointing nested
    inside this outer one) — caught as
    `torch.utils.checkpoint.CheckpointError: a different number of tensors
    was saved during the original forward and recomputation` (1600 vs 218)
    on the first real-config verification run. Wrapping HERE makes both the
    original call and any later recompute call see the identical toggled
    state, independent of what the outer `with` block has since restored.
    """
    with clean_latent_thread(model, film_bypass=True, no_activation_ckpt=True):
        return model(ids, inputs_embeds=embeds, return_hidden=return_hidden,
                     doc_ids=doc_ids)


def _latent_model_call(model, ids, embeds, return_hidden, use_checkpoint,
                       doc_ids=None):
    """One latent-thread forward, optionally activation-checkpointed.

    ROOT CAUSE (2026-07-02 OOM postmortem): `clean_latent_thread(...,
    no_activation_ckpt=True)` forces `model.activation_checkpointing =
    False` for the whole latent thread — that flag guards the model's
    PER-BLOCK checkpoint (`model.py::_ckpt_run_block`), which intermittently
    hits a Blackwell "unspecified launch failure" when it recomputes FLA
    kernels at the latent thread's short/odd sequence lengths (documented in
    `thinking.clean_latent_thread`'s docstring — do NOT flip that flag back
    on for this path). The consequence: every latent-thread forward keeps
    its FULL ~32-layer activation trace alive until the caller's single
    combined `(loss / n_micro).backward()`. With `--latent_reasoning_n 4`
    and rungs up to 8, one `step()` call does up to (R+1)*n_examples = 36
    such forwards, ALL summed into one graph before that one backward —
    ~16 GB by itself on the 32L x 960d config (see the
    project_phase1_ab_features_nettax memory note; this is what killed both
    Arm-B attempts at step 620).

    Fix: checkpoint EACH latent forward at the OUTER (whole-model) grain
    instead — one `torch.utils.checkpoint.checkpoint(..., use_reentrant=
    False)` boundary per `model(...)` call, not one per Block. This is a
    coarser, DIFFERENT checkpoint boundary than the per-block one the
    Blackwell bug was hit on (1 recompute per model call vs. 1 recompute per
    block per model call — far fewer, less nested checkpoint/recompute
    transitions), while still discarding the full internal activation trace
    between calls, cutting retained memory per call from O(layers) to O(1)
    boundary tensor. `model.activation_checkpointing` itself is left
    strictly OFF throughout (untouched) — this is a second, independent
    checkpoint layer wrapped from the outside, not a reversion of the
    Blackwell workaround. `use_reentrant=False` composes safely with the
    outer grad-accum microbatch loop (no reentrant-autograd nesting hazard,
    and the documented DDP+latent incompatibility is a separate, orthogonal
    static_graph issue — this path is single-GPU only regardless).
    """
    if use_checkpoint and torch.is_grad_enabled():
        return _torch_checkpoint(_run_latent_forward, model, ids, embeds,
                                 return_hidden, doc_ids, use_reentrant=False)
    return _run_latent_forward(model, ids, embeds, return_hidden, doc_ids)


def _answer_span_latent_loss(model, comment_ids, sol_ids, eos_id, R,
                             thinking_id, device, gate_weight=0.0,
                             checkpoint_latent=True, inter_ids=None,
                             perhop_weight=0.0, return_components=False):
    """Answer-span CE (+ optional per-hop CE) after an R-step latent burst.

    Self-contained twin of latent_sft.latent_sft_loss, but robust to the
    pretrain forward's return arity: with the gist loss active, training-mode
    `model(..., return_hidden=True)` returns (logits, hidden, gist) — a 3-tuple.
    The convention is hidden-at-index-1 (gist appended last), so we INDEX rather
    than tuple-unpack (the latent_sft version assumed exactly 2 → crashed here).

    ``gate_weight`` > 0 ALSO trains the output gate to invoke + halt thinking on
    its own (the latent_arith_real.autonomous_halt_loss recipe): P(emit) is
    supervised at the R+1 decision positions P-1..P+R-1 → THINK (0) for the
    first R, EMIT (1) at the last. Without this the bake installs the reasoning
    CAPABILITY but the gate never fires it (avg_steps≈0.77 vs target n).

    ``checkpoint_latent`` (default True) wraps every latent-thread forward in
    an outer activation checkpoint — see `_latent_model_call`'s docstring for
    the OOM this fixes. Exact (not approximate): checkpointing recomputes the
    identical forward during backward, so the loss VALUE and gradients are
    unchanged from the unchecked path (bar bit-level nondeterminism from
    re-running the same kernels twice, which `use_reentrant=False` handles
    deterministically via RNG-state save/restore for anything stochastic,
    e.g. PKM's epsilon-greedy slot replacement).

    ``inter_ids`` / ``perhop_weight`` (2026-07-04, the exec-trace program):
    DENSE per-step supervision that mirrors
    ``latent_arith_real.latent_perhop_loss`` EXACTLY. Think slot j (1-indexed)
    sits at absolute position ``P + j - 1``; the logits AT that position
    (SAME output head as the answer span — ``model.forward``'s
    ``out_norm -> lm_head``, read as ``logits[:, P+j-1, :]`` with NO causal
    shift) decode the j-th intermediate ``f^j(s) = inter_ids[j-1]``. The last
    slot (j=R) predicts the answer, exactly the position the answer-span shift
    decodes ``sol[0]`` from — the two conventions agree. Only ``j <=
    min(R, len(inter_ids))`` (rungs whose R exceeds the available
    intermediates supervise what exists). The trailing answer span is causally
    masked out of these positions, so ``logits[:, P:P+n_hops, :]`` here equal
    ``latent_perhop_loss``'s own forward over ``[comment, think_1..think_R]``
    (pinned in test_latent_reasoning_perhop.py). ``perhop_weight == 0.0`` (with
    ``return_components=False``) is byte-identical to the answer-only path
    (N0/N1 ran at the equivalent of 0.0). Total = answer_ce (+ gate)
    ``+ perhop_weight * perhop_ce``. ``return_components`` returns
    ``(total, answer_ce, perhop_ce)`` (both grad-connected) for separate
    logging / gradient-isolation tests.
    """
    base_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long,
                           device=device)
    P = len(comment_ids)
    for _ in range(int(R)):
        out = _latent_model_call(model, cur_ids, cur_emb, True,
                                 checkpoint_latent)
        h = out[1]                                   # hidden = index 1 always
        z = h[:, -1:, :].to(cur_emb.dtype)
        z = model.apply_latent_feedback_adapter(z).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    sol_t = torch.tensor([list(sol_ids) + [int(eos_id)]], dtype=torch.long,
                         device=device)
    full_ids = torch.cat([cur_ids, sol_t], dim=1)
    full_emb = torch.cat([cur_emb, model.embed(sol_t)], dim=1)
    out = _latent_model_call(model, full_ids, full_emb, False,
                             checkpoint_latent)
    logits = out[0] if isinstance(out, tuple) else out
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:].clone()
    start = P + int(R) - 1                            # supervise sol[0]..eos only
    shift_labels[:, :start] = -100
    ans_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), ignore_index=-100)
    loss = ans_ce
    # Per-hop supervision: decode think slot j (position P+j-1) -> inter[j-1].
    # UNSHIFTED logits at the think-slot positions, exactly like
    # latent_arith_real.latent_perhop_loss. Only computed when it actually
    # affects the loss (perhop_weight != 0) OR the caller wants the component
    # for logging (return_components) — so the perhop_weight=0 /
    # return_components=False path is byte-identical (no extra graph nodes).
    n_hops = min(int(R), len(inter_ids)) if inter_ids is not None else 0
    if n_hops > 0 and (perhop_weight != 0.0 or return_components):
        slot_logits = logits[:, P:P + n_hops, :]          # (1, n_hops, V)
        itgt = torch.tensor([list(inter_ids[:n_hops])], dtype=torch.long,
                            device=device)
        perhop_ce = F.cross_entropy(
            slot_logits.reshape(-1, slot_logits.shape[-1]), itgt.reshape(-1))
    else:
        perhop_ce = torch.zeros((), device=logits.device, dtype=loss.dtype)
    if perhop_weight != 0.0 and n_hops > 0:
        loss = loss + float(perhop_weight) * perhop_ce
    if gate_weight > 0.0 and getattr(model, "_last_gate_logits", None) is not None:
        gate_logits = model._last_gate_logits             # (1, T) pre-sigmoid emit
        dec = list(range(P - 1, P + int(R)))              # R+1 decision positions
        gl = gate_logits[0, dec]
        tgt = torch.zeros(len(dec), device=device, dtype=gl.dtype)
        tgt[-1] = 1.0                                      # EMIT at the last
        gate_loss = F.binary_cross_entropy_with_logits(gl, tgt)
        loss = loss + float(gate_weight) * gate_loss
    if return_components:
        return loss, ans_ce, perhop_ce
    return loss


def _left_pad_prompts(prompts: list, pad_id: int, device
                      ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Left-pad a batch of variable-length prompt token-id lists to a common
    length, and build the (n, P_max) `doc_ids` marking the pad prefix (0) vs.
    the real content (1) — the DeltaNet state-isolation boundary the batched
    growing-thread relies on (see `_answer_span_latent_loss_batched`).

    LEFT-pad (not right-pad) is the key choice: it keeps every row's real
    content ending at the same absolute position (P_max - 1) throughout the
    whole growing thread (padding is only ever at the very front, and every
    later append is a single shared column for every row), so the thread's
    `h[:, -1:, :]` read is correct for every row at every step with no
    per-row gather — exactly the single-example code's indexing, unchanged.
    """
    n = len(prompts)
    P_max = max(len(p) for p in prompts)
    ids = torch.full((n, P_max), int(pad_id), dtype=torch.long, device=device)
    doc_ids = torch.zeros((n, P_max), dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        pad_len = P_max - len(p)
        if p:
            ids[i, pad_len:] = torch.tensor(p, dtype=torch.long, device=device)
        doc_ids[i, pad_len:] = 1
    return ids, doc_ids, P_max


def _right_pad_solutions(sols: list, eos_id: int, pad_id: int, device
                         ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Right-pad `sol_ids + [eos_id]` per example to a common length.

    Returns `(ids (n, S_max), lens (n,), S_max)` — `lens` is each row's REAL
    length (including eos), used to mask the loss beyond that row's own
    content. Right-padding here (unlike the prompt) needs no state isolation:
    this trailing region only ever follows the positions being supervised, so
    causal order alone keeps pad content from influencing any loss-bearing
    position — the per-row loss mask (built from `lens`) is enough.
    """
    n = len(sols)
    with_eos = [list(s) + [int(eos_id)] for s in sols]
    lens = torch.tensor([len(s) for s in with_eos], dtype=torch.long,
                        device=device)
    S_max = int(lens.max().item())
    ids = torch.full((n, S_max), int(pad_id), dtype=torch.long, device=device)
    for i, s in enumerate(with_eos):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return ids, lens, S_max


def _answer_span_latent_loss_batched(model, examples, eos_id, R, thinking_id,
                                     device, gate_weight=0.0,
                                     checkpoint_latent=True, pad_id=0,
                                     perhop_weight=0.0, return_components=False):
    """Batched twin of `_answer_span_latent_loss`: ONE growing latent thread
    of batch size n = len(examples) instead of n sequential B=1 threads, for
    the shared rung R every example in a `step()` call already uses. Returns
    the SAME quantity as
    ``mean(_answer_span_latent_loss(model, *ex, ..., R) for ex in examples)``
    — equivalence (loss + gradients) pinned in
    test_latent_reasoning_batched.py — at roughly 1/n the wall-clock of n
    sequential B=1 forwards (the fix for the ~25%-GPU-utilization stall the
    sequential growing-thread caused: n=4 B=1 forwards are latency/kernel-
    launch-bound, not compute-bound, so batching amortizes that overhead).

    PADDING DESIGN (the load-bearing correctness point): a pad token WRITES
    to a linear-RNN's (DeltaNet's) recurrent state before the real prompt
    starts — unlike causal-attention padding, which is free as long as it's
    masked. Two things make this batched thread exact:
      1. LEFT-pad each prompt (`_left_pad_prompts`) so every row's real
         content ends at the same absolute position — the thread's
         `h[:, -1:, :]` read (every latent step) stays correct with no
         per-row gather.
      2. Mark the left-pad prefix as a separate "document" (doc_id 0) from
         the real content (doc_id 1) using the EXISTING cross-document
         isolation machinery (`doc_ids` -> model.py::_build_cu_seqlens ->
         FLA's packed `cu_seqlens` kernels, validated 2026-05-14 for packing
         multiple documents per pretrain row). This HARD-resets the
         recurrent state to zero exactly at the pad/real boundary,
         independent of whether the model was built with
         `--state_readonly_at_think` — the currently-running N1 process was
         NOT (checked directly against its live cmdline), so leaning on that
         flag's β=0 forcing instead would have been silently inert for it.
         `doc_ids` isolation has no such prerequisite: it works purely via
         the packed-sequence kernel path, always.
    The trailing (right-padded) solution region needs no isolation — see
    `_right_pad_solutions`.

    The per-example CE is computed with a PER-ROW mean (not a single
    batch-flattened mean) before averaging over examples — solution lengths
    differ per example, and a flattened `reduction="mean"` would silently
    token-weight the average (longer solutions counting more), whereas
    `LatentReasoningCotrain.step()`'s sequential path weights every example
    equally (`total / n_examples`, each term itself a per-example mean CE).

    MEASURED TRADE-OFF (`experiments/bench_latent_reasoning_batched.py`, real
    32L x 960d config matching the running N1 process): 3.2-3.5x wall-clock
    speedup (exceeds the ~2x target) at the cost of a real, not-negligible
    memory increase — peak allocated goes from ~6.0-6.4 GiB (sequential) to
    ~11.4-11.9 GiB (batched) for the aux ALONE at n=4, R=8 (both the absolute
    worst case, all prompts near --latent_reasoning_max_len 512, and the
    typical real-corpus rung-8 length spread, ~400-460 tok). This exceeds the
    informal "~12GB combined" prior target on paper, but still leaves >15 GiB
    of headroom on the 32GB card at the currently-observed ~7.5GB combined
    N1 usage — no OOM risk on this hardware. If memory margin ever gets
    tight (e.g. a wider trunk, or running concurrently with something else),
    the mitigation is lowering `--latent_reasoning_n` (this function's cost
    scales with n, same as the sequential path's did, just with a much
    better constant) — sub-batching within this function was NOT
    implemented (out of the requested scope; flag as a follow-up if needed).
    """
    n = len(examples)
    # Arity-tolerant: examples may be (comment, sol) 2-tuples (answer-only /
    # existing tests) or (comment, sol, inter_ids) 3-tuples (per-hop). The
    # per-hop targets are None/[] for 2-tuples => the perhop term is inert.
    prompts = [list(ex[0]) for ex in examples]
    sols = [list(ex[1]) for ex in examples]
    inters = [(list(ex[2]) if len(ex) > 2 and ex[2] is not None else [])
              for ex in examples]
    base_ids, doc_prompt, P_max = _left_pad_prompts(prompts, int(pad_id), device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    cur_doc = doc_prompt
    think_tok = torch.full((n, 1), int(thinking_id), dtype=torch.long,
                           device=device)
    think_doc = torch.ones((n, 1), dtype=torch.long, device=device)
    for _ in range(int(R)):
        out = _latent_model_call(model, cur_ids, cur_emb, True,
                                 checkpoint_latent, doc_ids=cur_doc)
        h = out[1]                                   # hidden = index 1 always
        z = h[:, -1:, :].to(cur_emb.dtype)
        z = model.apply_latent_feedback_adapter(z).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
        cur_doc = torch.cat([cur_doc, think_doc], dim=1)
    sol_t, sol_lens, S_max = _right_pad_solutions(sols, eos_id, int(pad_id),
                                                  device)
    full_ids = torch.cat([cur_ids, sol_t], dim=1)
    full_emb = torch.cat([cur_emb, model.embed(sol_t)], dim=1)
    full_doc = torch.cat(
        [cur_doc, torch.ones((n, S_max), dtype=torch.long, device=device)],
        dim=1)
    out = _latent_model_call(model, full_ids, full_emb, False,
                             checkpoint_latent, doc_ids=full_doc)
    logits = out[0] if isinstance(out, tuple) else out
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:].clone()
    start = P_max + int(R) - 1                        # uniform across rows
    shift_labels[:, :start] = -100
    col = torch.arange(shift_labels.shape[1], device=device).unsqueeze(0)
    end = (start + sol_lens).unsqueeze(1)              # (n, 1) per-row end
    shift_labels = torch.where(col >= end,
                               torch.full_like(shift_labels, -100),
                               shift_labels)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(n, -1)
    valid = (shift_labels != -100)
    ans_per_row = ce.sum(dim=1) / valid.sum(dim=1).clamp_min(1)   # (n,)
    per_example = ans_per_row
    # Per-hop supervision (same convention as the single-example path): think
    # slot j (1-indexed) is at the UNIFORM (left-pad => shared) absolute
    # position P_max+j-1; its UNSHIFTED logits decode inter j = f^j(s). Targets
    # are per-row (n, R) with -100 for rows/steps lacking an intermediate
    # (rungs whose R exceeds a row's available intermediates use what exists).
    # Per-row-mean over that row's valid steps, then example-mean — matching
    # the answer term's per-example semantics so batched == mean-of-sequential.
    perhop_mean = torch.zeros((), device=logits.device, dtype=per_example.dtype)
    n_hops = [min(int(R), len(it)) for it in inters]
    if (perhop_weight != 0.0 or return_components) and any(h > 0 for h in n_hops):
        itgt = torch.full((n, int(R)), -100, dtype=torch.long, device=device)
        for i, it in enumerate(inters):
            h = n_hops[i]
            if h > 0:
                itgt[i, :h] = torch.tensor(it[:h], dtype=torch.long,
                                           device=device)
        slot_logits = logits[:, P_max:P_max + int(R), :]          # (n, R, V)
        ph = F.cross_entropy(
            slot_logits.reshape(-1, slot_logits.shape[-1]), itgt.reshape(-1),
            ignore_index=-100, reduction="none").reshape(n, int(R))
        ph_valid = (itgt != -100)
        per_row_perhop = ph.sum(dim=1) / ph_valid.sum(dim=1).clamp_min(1)  # (n,)
        perhop_mean = per_row_perhop.mean()
        if perhop_weight != 0.0:
            per_example = per_example + float(perhop_weight) * per_row_perhop
    if gate_weight > 0.0 and getattr(model, "_last_gate_logits", None) is not None:
        gate_logits = model._last_gate_logits              # (n, T) pre-sigmoid
        dec = list(range(P_max - 1, P_max + int(R)))        # R+1 decisions
        gl = gate_logits[:, dec]
        tgt = torch.zeros_like(gl)
        tgt[:, -1] = 1.0                                    # EMIT at the last
        gate_ce = F.binary_cross_entropy_with_logits(gl, tgt, reduction="none")
        per_example = per_example + float(gate_weight) * gate_ce.mean(dim=1)
    total = per_example.mean()
    if return_components:
        return total, ans_per_row.mean(), perhop_mean
    return total


def _load_rung(prefix: str, n: int, tok, max_len: int,
               require_single_token_inter: bool = False) -> list[tuple]:
    """Pointer-chase / exec-trace records → (comment_ids, answer_ids,
    inter_ids). Mirrors latent_arith_real._load_rung (answer rendered as
    `def solve(): return <ans>`), additionally tokenising each element of the
    record's `intermediates` field into a single-token id (the j-th
    intermediate f^j(s)) for the per-hop supervision.

    `require_single_token_inter` (set True by LatentReasoningCotrain whenever
    `perhop_weight != 0`) enforces the v1 exec-trace data contract: every
    intermediate must be a single token AND at least one must be present.
    Examples violating it are SKIPPED and counted, and a rung with < 50 %
    usable examples raises loudly (bad data — regenerate with single-token
    values, or run answer-only via `--latent_reasoning_perhop_weight 0`).

    `require_single_token_inter=False` (the answer-only / escape-hatch default,
    and every pre-per-hop caller) keeps the OLD example set byte-for-byte:
    every length-passing record is returned, inter_ids best-effort (`[]` when
    absent or multi-token, and never consulted since perhop_weight is 0). The
    (comment, answer) content, count, and order are identical to the old
    2-tuple loader — only the tuple arity grew, which the sampling (index into
    self.data) and loss helpers (arity-tolerant) are indifferent to."""
    out = []
    path = f"{prefix}_n{n}.jsonl"
    total = 0                       # length-passing records considered
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        pfx = r["prompt"] + "\ndef solve():\n    return "
        c = tok.encode(pfx, add_special_tokens=False)
        s = tok.encode(str(r["answer"]), add_special_tokens=False)
        if len(c) + len(s) + n + 2 > max_len:
            continue
        total += 1
        inter_ids = []
        multitoken = False
        for v in r.get("intermediates", []):
            enc = tok.encode(str(v), add_special_tokens=False)
            if len(enc) != 1:
                multitoken = True
                break
            inter_ids.append(int(enc[0]))
        if require_single_token_inter:
            if multitoken or len(inter_ids) == 0:
                continue            # skip-and-count (via total - len(out))
            out.append((c, s, inter_ids))
        else:
            out.append((c, s, [] if multitoken else inter_ids))
    if require_single_token_inter and total > 0:
        usable = len(out)
        if usable < 0.5 * total:
            raise ValueError(
                f"latent-reasoning per-hop: rung n={n} at {path} has only "
                f"{usable}/{total} ({100.0 * usable / total:.0f}%) examples "
                f"with usable single-token intermediates (need >= 50%). "
                f"Multi-token or missing intermediates violate the v1 "
                f"exec-trace data contract — regenerate with single-token "
                f"(small-int) values, or set --latent_reasoning_perhop_weight "
                f"0 for the answer-only escape hatch.")
    return out


class LatentReasoningCotrain:
    """Holds the depth-bound reasoning corpus and emits one answer-span latent
    loss per call, with a depth curriculum (ramp 1->max over 60% of steps, then
    uniform consolidation). Does NOT call backward — the caller adds the returned
    loss to the total and backprops with the rest of the step."""

    def __init__(self, train_prefix: str, rungs, tok, thinking_id: int,
                 eos_id: int, device, max_len: int = 256, no_ramp: bool = False,
                 gate_weight: float = 0.0, seed: int = 0,
                 checkpoint_latent: bool = True, batch_examples: bool = True,
                 pad_id: int = 0, perhop_weight: float = 0.0):
        self.device = device
        self.thinking_id = int(thinking_id)
        self.eos_id = int(eos_id)
        self.no_ramp = bool(no_ramp)
        self.gate_weight = float(gate_weight)
        # Per-hop supervision weight (2026-07-04, the exec-trace program).
        # DENSE per-step CE that decodes latent step j -> intermediate j =
        # f^j(s), the mechanism the "per-step credit assignment fixes latent"
        # hypothesis rests on. Total aux = answer_ce + perhop_weight *
        # mean_perhop_ce. The CLASS default is 0.0 (answer-only = the exact
        # N0/N1 behaviour + backwards-compat for every existing caller/test);
        # the PRODUCT default is 1.0 via train_lm.py's
        # --latent_reasoning_perhop_weight ("per-hop IS the program"). Non-zero
        # weight also switches _load_rung into single-token-contract-enforcing
        # mode (skip+count violators, error < 50% usable).
        self.perhop_weight = float(perhop_weight)
        # Last-step component diagnostics (set by step(), read by train_lm.py
        # for the reason(ans=..,hop=..) log line + TB scalars).
        self.last_ans = 0.0
        self.last_perhop = 0.0
        # OOM fix (2026-07-02, default ON): activation-checkpoint every
        # latent-thread forward — see _latent_model_call's docstring. Escape
        # hatch for anyone who wants the old (unchecked, ~2x cheaper compute,
        # ~16 GB heavier) behaviour back.
        self.checkpoint_latent = bool(checkpoint_latent)
        # Batched growing-thread (2026-07-04, default ON): process the step's
        # n_examples as ONE batch-n thread instead of n sequential B=1
        # threads — see _answer_span_latent_loss_batched's docstring for the
        # full design (left-pad + doc_ids state isolation). `pad_id` is a
        # purely-internal filler id for the batched path's left/right pad
        # positions — its exact value never matters for correctness (the
        # doc_ids isolation discards the pad-prefix's effect on state
        # entirely, and pad-id != thinking_id is all that's required to keep
        # it from being mistaken for a real think token by other think_mask
        # consumers) — 0 is safe regardless of the tokenizer's real pad
        # convention. `batch_examples=False` is the escape hatch reproducing
        # the exact old sequential behaviour (same example sampling order
        # either way — see `step()`).
        self.batch_examples = bool(batch_examples)
        self.pad_id = int(pad_id)
        self.data: dict[int, list[tuple]] = {}
        _require_inter = self.perhop_weight != 0.0
        for n in rungs:
            recs = _load_rung(train_prefix, int(n), tok, max_len,
                              require_single_token_inter=_require_inter)
            if recs:
                self.data[int(n)] = recs
        self.rungs = sorted(self.data.keys())
        if not self.rungs:
            raise ValueError(
                f"no latent-reasoning data at {train_prefix}_n*.jsonl for "
                f"rungs {list(rungs)}")
        self.g = torch.Generator().manual_seed(int(seed))
        self._ptr = {n: 0 for n in self.rungs}
        self._perm = {n: torch.randperm(len(self.data[n]), generator=self.g).tolist()
                      for n in self.rungs}

    def _next(self, n: int):
        if self._ptr[n] >= len(self._perm[n]):
            self._perm[n] = torch.randperm(len(self.data[n]),
                                           generator=self.g).tolist()
            self._ptr[n] = 0
        ex = self.data[n][self._perm[n][self._ptr[n]]]
        self._ptr[n] += 1
        return ex

    def _pick_rung(self, step: int, total_steps: int) -> int:
        if self.no_ramp:
            choices = self.rungs
        else:
            ramp_end = max(1.0, 0.6 * total_steps)
            if step < ramp_end:
                frac = step / ramp_end
                frontier = max(1, int(round(1 + frac * (len(self.rungs) - 1))))
                choices = self.rungs[:frontier]
            else:
                choices = self.rungs
        return choices[int(torch.randint(0, len(choices), (1,),
                                         generator=self.g).item())]

    def step(self, model, step: int, total_steps: int, n_examples: int):
        """Return (loss, rung): mean answer-span CE over n_examples at depth=rung
        (R=rung latent steps).

        The clean latent thread (WM off, FiLM bypass, per-block
        activation-checkpointing off) is asserted PER model() CALL inside
        `_run_latent_forward`, not once here for the whole step — required
        for `checkpoint_latent=True` correctness (see that function's
        docstring): an outer `with clean_latent_thread(...):` scoped to
        just this method would have already exited (restoring the real
        pretrain-config toggles) by the time the caller's combined
        `.backward()` triggers checkpoint's recompute, making forward and
        recompute silently diverge. Per-call re-assertion is exactly
        equivalent when `checkpoint_latent=False` too (each of the (R+1)*
        n_examples model() calls just re-applies/restores the same
        no-op-when-already-clean toggles), so this is not a behaviour
        change for the unchecked path.

        `batch_examples` (default True, set in `__init__`) runs the
        n_examples as ONE batched growing thread
        (`_answer_span_latent_loss_batched`) instead of n sequential B=1
        threads — the ~2x step-time fix. Example SAMPLING is identical
        either way (drawn here, before branching) so the escape hatch
        (`batch_examples=False`) is a true A/B: only how the same examples
        are processed differs, not which examples are used.
        """
        rung = self._pick_rung(step, total_steps)
        n = max(1, int(n_examples))
        examples = [self._next(rung) for _ in range(n)]
        # return_components=True so we can log the answer / per-hop CE
        # separately (reason(ans=..,hop=..)); it forces the per-hop CE to be
        # COMPUTED for logging even at perhop_weight=0, but never ADDED to the
        # loss there — so the perhop_weight=0 loss/gradient stays byte-
        # identical to the answer-only path (the escape hatch).
        if self.batch_examples:
            loss, ans, hop = _answer_span_latent_loss_batched(
                model, examples, self.eos_id, rung, self.thinking_id,
                self.device, gate_weight=self.gate_weight,
                checkpoint_latent=self.checkpoint_latent, pad_id=self.pad_id,
                perhop_weight=self.perhop_weight, return_components=True)
            self.last_ans = float(ans.detach())
            self.last_perhop = float(hop.detach())
            return loss, rung
        total = None
        ans_sum = 0.0
        hop_sum = 0.0
        for ex in examples:
            c, s = ex[0], ex[1]
            inter = ex[2] if len(ex) > 2 else None
            l, a, h = _answer_span_latent_loss(
                model, c, s, self.eos_id, rung, self.thinking_id, self.device,
                gate_weight=self.gate_weight,
                checkpoint_latent=self.checkpoint_latent,
                inter_ids=inter, perhop_weight=self.perhop_weight,
                return_components=True)
            total = l if total is None else total + l
            ans_sum += float(a.detach())
            hop_sum += float(h.detach())
        loss = total / n
        self.last_ans = ans_sum / n
        self.last_perhop = hop_sum / n
        return loss, rung
