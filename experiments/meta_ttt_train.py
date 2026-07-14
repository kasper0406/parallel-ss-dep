"""Meta-TTT episode training (repo-adaptive coder, Phase P1 — 2026-07-13).

Meta-trains DeltaNet's recurrent state into a *deliberate test-time learner*
over full-repository ingestion: the model reads a whole repo (4-32k tokens of
cross-file context) at O(1)/8.2 MiB cost, then is supervised ONLY on the
cross-file usage task span. The bet (META_TTT_PLAN_2026_07_13.md): the state
dynamics can be *shaped* so that "having read your repo" measurably helps the
task, and the lift GROWS with ingested-repo size (the curve-bending signature).

This module is the P1 TRAINING wiring — an aux path mirroring
`latent_reasoning_cotrain.LatentReasoningCotrain`: per aux step it samples one
episode (bucket-balanced), ingests the repo context, and returns an answer-span
CE loss + diagnostics. The caller (`train_lm.py`) adds it to the step's loss and
backprops with the rest; the standard `--data_mix` stream is the retention
anchor (the Stage-A/B lesson: co-train, never fine-tune the base away).

-------------------------------------------------------------------------------
Two design constraints drive everything (see the module tests + AGENTS.md):

1. T MISMATCH + STATE MUST FLOW ACROSS THE EPISODE.
   train_lm trains at T=2048; episodes are 4-32k. The recurrent state must FLOW
   across the whole episode — no doc-boundary reset inside one episode (that IS
   the meta-TTT signal). The standard chunked loader + cu_seqlens machinery
   resets state at every row/doc boundary, so it cannot express "one 30k-token
   sequence". Instead we run the episode through the model in T-sized chunks
   SEQUENTIALLY, carrying the DeltaNet recurrent state via the FLA cache.

   WHAT FLA SUPPORTS (the load-bearing fact): `fla/layers/delta_net.py`
   DeltaNet.forward reads its initial recurrent+conv state from
   `past_key_values` (`get_layer_cache`) and, in `mode="chunk"` (q_len>64),
   passes it as `chunk_delta_rule(..., initial_state=recurrent_state,
   output_final_state=use_cache)` — see fla/ops/delta_rule/chunk.py:220,261.
   `fla/models/utils.py::Cache.update` REPLACES recurrent_state/conv_state per
   call (not concat), so feeding sequential T>1 chunks through ONE cache carries
   the full state EXACTLY (== a single full forward, up to chunk-kernel
   numerics). We reuse `TinyLM._step_block(blk, x, past=cache, layer_idx=L)`,
   which is exactly the prefill code-path (already logit-equivalence-tested in
   test_incremental_decode.py) — here across MULTIPLE T>1 chunks.

   GRADIENT — truncated BPTT with a REAL gradient window (the critical point):
   full BPTT over 32k is memory-prohibitive, but running the WHOLE context under
   no_grad gives the state-producing weights NO gradient through ingestion —
   which defeats meta-training (the point is to shape HOW the state is built).
   Compromise (`--meta_ttt_grad_chunks N`, default 2 = 4k tokens): the LAST N
   context chunks are grad-enabled; earlier chunks run under no_grad (state
   carried, DETACHED at the boundary). Gradient then flows into the trunk's
   q/k/v AND the DeltaNet write-gate `b_proj`/β from the grad window's ingestion
   of the near-task context — i.e. "produce a state that helps the task".
   N can grow with available memory (the grad-window forward is O(N*chunk) in
   activation memory at batch 1).

   NB on activation checkpointing: the intent ("activation-checkpointed grad
   chunks") is NOT wired via `torch.utils.checkpoint` here, on purpose. The FLA
   cache is READ then WRITTEN in the same layer forward; checkpoint's backward
   recompute would re-read the ALREADY-updated cache slot and silently diverge
   (a correctness hazard we cannot validate without a GPU). The grad window is
   instead run as ONE forward from the detached boundary state (no intra-window
   cache handoff), and its activation memory is bounded directly by
   grad_chunks*chunk_size. The FLA chunk kernel is itself state-bounded (its own
   autograd Function recomputes internals in backward), so per-chunk memory is
   dominated by the MLP/norm activations that grad_chunks controls. If a future
   run needs a bigger grad window than fits, lower chunk_size or grad_chunks, or
   wire an explicit-initial-state checkpoint path (documented follow-up).

2. LOSS MASKING — EXACTLY the eval's token positions.
   CE is computed ONLY on the `task_line` tokens (and optionally the last M
   tokens of `task_prefix`, `--meta_ttt_prefix_supervise_m`, default 0), at the
   SAME token positions `eval_repo_adaptive.py` scores — by IMPORTING its
   tokenization helpers (`context_ids`, `task_line_span_tokens`) and
   `gen_repo_episodes.perline_ids`, never re-deriving them. With M=0 the trained
   CE is exactly the eval's `real`-arm line CE, so train/eval measure the same
   quantity.
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import contextlib
import json
import os

import torch
import torch.nn.functional as F

from experiments.eval_repo_adaptive import context_ids, task_line_span_tokens
from experiments.gen_repo_episodes import perline_ids

BUCKET_ORDER = ("4-8k", "8-16k", "16-32k")


# --------------------------------------------------------------------------- #
# Chunk boundary math (pure, unit-tested).
# --------------------------------------------------------------------------- #

def chunk_bounds(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split ``range(total)`` into contiguous [start, end) chunks of at most
    `chunk_size`, with EVERY chunk of length >= 2 (a size-1 tail is absorbed
    into the previous chunk). Length-1 chunks would route `_step_block` down the
    T==1 `forward_step` (fused_recurrent) path — correct, but we keep every
    ingest chunk on the chunk-kernel path for uniformity, so we never emit one.
    `total == 1` returns a single [0,1) chunk (nothing to absorb into)."""
    if total <= 0:
        return []
    cs = max(2, int(chunk_size))
    bounds: list[tuple[int, int]] = []
    pos = 0
    while pos < total:
        end = min(pos + cs, total)
        # If a length-1 remainder would follow, absorb it into this chunk.
        if 0 < total - end < 2:
            end = total
        bounds.append((pos, end))
        pos = end
    return bounds


def grad_boundary(total_len: int, grad_window_len: int,
                  earliest_predictor: int) -> int:
    """First index of the gradient window. The window is the LAST
    `grad_window_len` tokens, pulled back if necessary so it always covers the
    earliest supervised PREDICTOR position (so every CE target has a
    grad-connected predictor). Clamped to [0, total_len)."""
    boundary = min(total_len - grad_window_len, earliest_predictor)
    return max(0, min(boundary, total_len - 1))


# --------------------------------------------------------------------------- #
# Episode token assembly (reuses the eval's tokenization, verbatim).
# --------------------------------------------------------------------------- #

def assemble_episode_tokens(episode: dict, tok) -> dict:
    """Tokenise one episode into the SAME token sequence the eval's `real` arm
    scores: [context files except task file] + [task_prefix] + [task_line].
    Returns a dict of int lists / ints (no tensors — cheap to cache).

    Keys:
      full_ids       : list[int]  the whole sequence
      p_line         : int        absolute index where task_line starts
      line_len       : int        len(task_line_ids)
      span_local     : list[int]  identifier-span indices WITHIN task_line
      n_ctx_tokens   : int        episode-reported context length (for buckets)
      bucket         : str
    """
    prefix_ctx = context_ids(episode, tok)                 # context, no task file
    task_prefix_ids = perline_ids(episode["task_prefix"], tok)
    line_ids, span_local = task_line_span_tokens(episode, tok)
    full_ids = list(prefix_ctx) + list(task_prefix_ids) + list(line_ids)
    p_line = len(prefix_ctx) + len(task_prefix_ids)
    return {
        "full_ids": full_ids,
        "p_line": p_line,
        "line_len": len(line_ids),
        "span_local": span_local,
        "n_ctx_tokens": int(episode.get("n_ctx_tokens", len(prefix_ctx))),
        "bucket": episode.get("bucket"),
        "episode_id": episode.get("episode_id"),
    }


def supervised_positions(p_line: int, line_len: int, prefix_supervise_m: int
                         ) -> tuple[list[int], list[int]]:
    """(target_positions, predictor_positions) for the supervised span.

    Targets are absolute positions [p_line - M, p_line + line_len); predictor
    for target t is t-1 (teacher forcing). With M=0 this is exactly the eval's
    `real` arm (task_line predicted from [P-1, P+L-1))."""
    m = max(0, int(prefix_supervise_m))
    start = max(1, p_line - m)                              # need a predictor (t-1>=0)
    tgt = list(range(start, p_line + line_len))
    pred = [t - 1 for t in tgt]
    return tgt, pred


# --------------------------------------------------------------------------- #
# Chunked ingest with recurrent-state carry.
# --------------------------------------------------------------------------- #

def _embed_chunk(model, ids_chunk: torch.Tensor, pos_offset: int) -> torch.Tensor:
    """Input embeddings for one chunk, mirroring TinyLM.forward's embed + abs
    pos-embed (+ optional think-index). `pos_offset` is the chunk's absolute
    start position — REQUIRED for pos-embed correctness across chunks. Our
    production base is a linear-RNN with max_T=0 (no pos-embed, position-free),
    so the pos branch is inert there; it is kept correct for max_T>0 models
    whose episodes fit within max_T."""
    x = model.embed(ids_chunk)
    if getattr(model, "max_T", 0) > 0:
        T = ids_chunk.shape[1]
        pos = torch.arange(pos_offset, pos_offset + T,
                           device=ids_chunk.device).clamp_max(model.max_T - 1)
        x = x + model.pos_embed(pos)
    # Think-index embedding (only if the model has thinking tokens; our meta-TTT
    # bases don't, so this is inert — kept for generality with think-token bases).
    if (getattr(model, "think_index_emb_size", 0) > 0
            and getattr(model, "thinking_token_id", None) is not None):
        x = x + model._compute_think_index_emb(ids_chunk)
    return x


def _block_stack(model, x: torch.Tensor, cache) -> torch.Tensor:
    """Run the plain block stack over `x`, threading the FLA `cache` through
    each attention layer (state carry). Mirrors TinyLM.prefill's plain-bypass
    block loop (`_step_block` + `_maybe_pkm`). think_mask=None (no think tokens
    in episodes); feedback is off on the meta-TTT bases (`--feedback none`)."""
    h = x
    for L, blk in enumerate(model.blocks):
        h = model._step_block(blk, h, past=cache, layer_idx=L, think_mask=None)
        h = model._maybe_pkm(h, L)
    return h


def _finalize_logits_at(model, hidden: torch.Tensor, ids_window: torch.Tensor,
                        local_pred: list[int]) -> torch.Tensor:
    """Mirror TinyLM._finalize's tail (refinement head -> out_norm -> memory ->
    lm_head) but run lm_head ONLY at the `local_pred` positions, so the full
    (T, V) logits tensor is never materialised (V ~= 49k >> the ~tens of
    supervised positions). Memory/refinement are no-ops on the plain base."""
    h = model._apply_refinement_head(hidden)
    h = model.out_norm(h)
    h = model._apply_memory(h, ids_window, read_mask=None, doc_ids=None)
    sel = h[0, local_pred, :]                              # (n_pred, d)
    return model.lm_head(sel)                             # (n_pred, V)


def ingest_and_task_logits(model, full_ids: list[int], boundary: int,
                           chunk_size: int, pred_pos: list[int],
                           device) -> torch.Tensor:
    """Chunked ingest with state carry; return logits at the (absolute)
    `pred_pos` predictor positions.

    - Chunks of `full_ids[:boundary]` run under `torch.no_grad()` (state carried
      via one shared FLA cache, DETACHED at the boundary).
    - `full_ids[boundary:]` (the grad window, covering all `pred_pos`) runs as
      ONE grad-enabled forward continuing the SAME cache — so gradient flows into
      the trunk (incl. β/write-gate) from the near-task ingestion, but stops at
      the detached boundary state (truncated BPTT).
    """
    from fla.models.utils import Cache as FLACache

    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    cache = FLACache(seen_tokens=0)

    # Match the main training step's precision: bf16 autocast on CUDA (the FLA
    # kernel needs bf16 anyway, and this keeps the aux's embed/MLP/lm_head at the
    # same precision + memory as the pretrain step and the eval's arm_ce).
    # On CPU (tests) use fp32 so chunked==full is exact.
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if dev.type == "cuda" else contextlib.nullcontext())

    # ---- no_grad prefix: carry state across chunks -------------------------
    if boundary > 0:
        with torch.no_grad(), autocast:
            for a, b in chunk_bounds(boundary, chunk_size):
                x = _embed_chunk(model, ids_t[:, a:b], pos_offset=a)
                _block_stack(model, x, cache)              # writes state into cache

    # ---- grad window: one forward continuing the carried (detached) state --
    autocast2 = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                 if dev.type == "cuda" else contextlib.nullcontext())
    with autocast2:
        win = ids_t[:, boundary:]
        x = _embed_chunk(model, win, pos_offset=boundary)
        hidden = _block_stack(model, x, cache)             # (1, win_len, d), grad-on
        local_pred = [p - boundary for p in pred_pos]
        logits = _finalize_logits_at(model, hidden, win, local_pred)
    return logits


# --------------------------------------------------------------------------- #
# The trainer.
# --------------------------------------------------------------------------- #

def _resolve_path(train_prefix: str) -> str:
    if os.path.exists(train_prefix):
        return train_prefix
    if os.path.exists(train_prefix + ".jsonl"):
        return train_prefix + ".jsonl"
    raise FileNotFoundError(
        f"meta-TTT episodes not found at '{train_prefix}' or "
        f"'{train_prefix}.jsonl' (expected the gen_repo_episodes train split).")


class MetaTTTEpisodeTrainer:
    """Holds the repo-episode corpus; emits one answer-span CE loss per call.

    Bucket-balanced sampling: a bucket is drawn uniformly from the non-empty
    buckets, then an episode within it (a shuffled cycling permutation, like
    LatentReasoningCotrain) — so the n_ctx distribution the state dynamics see
    is balanced across 4-8k / 8-16k / 16-32k rather than dominated by the
    heaviest bucket (needed for the curve-bending eval).

    Does NOT call backward — the caller adds the returned loss to the total and
    backprops with the rest of the step.
    """

    def __init__(self, train_prefix: str, tok, device, chunk_size: int = 2048,
                 grad_chunks: int = 2, prefix_supervise_m: int = 0,
                 max_ctx_tokens: int = 32000, seed: int = 0,
                 buckets: tuple = BUCKET_ORDER):
        self.tok = tok
        self.device = device
        self.chunk_size = max(2, int(chunk_size))
        self.grad_chunks = max(1, int(grad_chunks))
        self.prefix_supervise_m = max(0, int(prefix_supervise_m))
        self.max_ctx_tokens = int(max_ctx_tokens)
        path = _resolve_path(train_prefix)
        self.path = path
        episodes = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        # Bucket-index; drop episodes whose context exceeds the token budget
        # (mirrors the eval's --max_ctx_tokens skip so train/eval see the same
        # supported set).
        self.by_bucket: dict[str, list[dict]] = {}
        n_dropped = 0
        for ep in episodes:
            if int(ep.get("n_ctx_tokens", 0)) > self.max_ctx_tokens:
                n_dropped += 1
                continue
            b = ep.get("bucket")
            self.by_bucket.setdefault(b, []).append(ep)
        # Deterministic bucket order (only the non-empty ones present in data).
        self.buckets = [b for b in buckets if self.by_bucket.get(b)]
        # Any extra buckets present in data but not in the requested order.
        for b in self.by_bucket:
            if b not in self.buckets and self.by_bucket[b]:
                self.buckets.append(b)
        if not self.buckets:
            raise ValueError(
                f"no meta-TTT episodes within max_ctx_tokens={max_ctx_tokens} "
                f"at {path}")
        self.n_dropped = n_dropped
        self.n_episodes = sum(len(v) for v in self.by_bucket.values())
        # RNG + per-bucket cycling permutations.
        self.g = torch.Generator().manual_seed(int(seed))
        self._perm = {b: torch.randperm(len(self.by_bucket[b]),
                                        generator=self.g).tolist()
                      for b in self.buckets}
        self._ptr = {b: 0 for b in self.buckets}
        # Tokenized-episode cache (lazy; ~500 episodes total).
        self._tok_cache: dict = {}
        # Last-step diagnostics (read by train_lm.py for the log line).
        self.last_bucket = None
        self.last_n_ctx = 0
        self.last_task_ce = 0.0
        self.last_span_ce = 0.0
        self.last_grad_tokens = 0

    # -- sampling -----------------------------------------------------------
    def _next_in_bucket(self, b: str) -> dict:
        if self._ptr[b] >= len(self._perm[b]):
            self._perm[b] = torch.randperm(len(self.by_bucket[b]),
                                           generator=self.g).tolist()
            self._ptr[b] = 0
        ep = self.by_bucket[b][self._perm[b][self._ptr[b]]]
        self._ptr[b] += 1
        return ep

    def _sample_episode(self) -> dict:
        bi = int(torch.randint(0, len(self.buckets), (1,),
                               generator=self.g).item())
        b = self.buckets[bi]
        return self._next_in_bucket(b)

    def _tokenize(self, ep: dict) -> dict:
        key = ep.get("episode_id") or id(ep)
        cached = self._tok_cache.get(key)
        if cached is None:
            cached = assemble_episode_tokens(ep, self.tok)
            self._tok_cache[key] = cached
        return cached

    # -- one aux step -------------------------------------------------------
    def step(self, model):
        """Return (loss, diag). `loss` is the answer-span CE (line + optional M
        prefix tokens); `diag` = {ce, span_ce, ctx, bkt, grad_tokens, n_pred}."""
        ep = self._sample_episode()
        a = self._tokenize(ep)
        full_ids = a["full_ids"]
        total = len(full_ids)
        p_line = a["p_line"]
        line_len = a["line_len"]
        if line_len == 0 or p_line < 1 or total < 2:
            # Degenerate episode (no task line / no predictor). Return a zero
            # loss that still touches params so the grad graph is well-formed.
            zero = 0.0 * sum(p.sum() for p in model.parameters())
            self.last_bucket = a["bucket"]
            self.last_n_ctx = a["n_ctx_tokens"]
            self.last_task_ce = 0.0
            self.last_span_ce = 0.0
            self.last_grad_tokens = 0
            return zero, {"ce": 0.0, "span_ce": None, "ctx": a["n_ctx_tokens"],
                          "bkt": a["bucket"], "grad_tokens": 0, "n_pred": 0}

        tgt_pos, pred_pos = supervised_positions(
            p_line, line_len, self.prefix_supervise_m)
        earliest_pred = pred_pos[0]
        grad_window_len = self.grad_chunks * self.chunk_size
        boundary = grad_boundary(total, grad_window_len, earliest_pred)

        logits = ingest_and_task_logits(
            model, full_ids, boundary, self.chunk_size, pred_pos, self.device)
        targets = torch.tensor([full_ids[t] for t in tgt_pos],
                               dtype=torch.long, device=self.device)
        ce_tok = F.cross_entropy(logits.float(), targets, reduction="none")
        loss = ce_tok.mean()

        # Line CE (the eval-matched metric) = the last `line_len` supervised
        # positions (targets end with the task line; the first M are prefix).
        line_ce = float(ce_tok[-line_len:].mean().detach())
        # Span CE over the identifier tokens (diagnostic; matches eval's span).
        span_local = a["span_local"]
        span_ce = None
        if span_local:
            # span target positions are p_line + span_local; their index within
            # `tgt_pos` is (p_line + s) - tgt_pos[0].
            base = tgt_pos[0]
            span_idx_in_tgt = [(p_line + s) - base for s in span_local
                               if 0 <= (p_line + s) - base < len(tgt_pos)]
            if span_idx_in_tgt:
                span_ce = float(ce_tok[torch.tensor(
                    span_idx_in_tgt, device=self.device)].mean().detach())

        self.last_bucket = a["bucket"]
        self.last_n_ctx = a["n_ctx_tokens"]
        self.last_task_ce = line_ce
        self.last_span_ce = span_ce if span_ce is not None else 0.0
        self.last_grad_tokens = total - boundary
        return loss, {"ce": line_ce, "span_ce": span_ce,
                      "ctx": a["n_ctx_tokens"], "bkt": a["bucket"],
                      "grad_tokens": total - boundary, "n_pred": len(tgt_pos)}
