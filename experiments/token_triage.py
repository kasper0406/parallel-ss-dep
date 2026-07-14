"""Rho-1-style selective-token-loss (triage) — per-token loss routing driven by
excess loss (student CE − reference CE).

Design references: `ideas_2026_07_13/03_training_data.md` idea 1 (teacher-scored
token triage, RANK 1) and `ideas_2026_07_13/01_optimization.md` idea 2 (RHO-1
selective LM). Rho-1 = "Not All Tokens Are What You Need" (arXiv 2404.07965):
score every pretrain token by excess loss and only train CE on the top ~keep_frac
of tokens; already-learned / unlearnable-noise tokens get zero loss.

THREE-WAY ROUTE (this file), fused with the KD logits we ALREADY store:
  * KD    — teacher-confident & student-wrong  (high excess loss). The informative
            tokens. These carry full CE (`w_ce_kept`) AND a sparse top-k KD term
            (`w_kd`, applied on the trainer side over `kd_mask`).
  * DROP  — both-wrong / high-teacher-entropy   (unlearnable — license headers,
            random hex, unique identifiers). Zero CE weight → no gradient.
  * EASY  — both-right                          (already learned). Low-weight CE
            (`w_ce_easy`) so the budget stays on the KD tokens without fully
            forgetting the easy ones.

REFERENCE MODES (both supported — see `compute_triage_mask`):
  (a) Stored teacher TOP-K (`teacher_ids`, `teacher_vals`): the free byproduct of
      the offline KD bridge (`teacher_logits_io.LogitStoreReader`). The reference
      CE is `−log p_teacher(target)`; when the target is in the stored top-k it is
      read directly, otherwise a documented fallback (residual-mass split for a
      log-prob store, or worst-top-k floor) estimates it.
      **What the production store actually holds** (`gen_teacher_logits_vllm.py`,
      Qwen/Qwen2.5-Coder-7B-Instruct-AWQ, top_k=16): the `logits` field is vLLM
      `prompt_logprobs` — i.e. LOG-SOFTMAX log-probabilities, not raw logits — so
      `stored_is_logprob=True` (the default) is EXACT for in-top-k targets. The HF
      reference generator (`gen_teacher_logits.py`) instead stores RAW logits;
      pass `stored_is_logprob=False` for such a store (ref CE is then a top-k-only
      softmax approximation).
  (b) A precomputed per-token reference-CE tensor (`ref_ce`): for a future
      SmolLM2-scored cache (idea 2). No teacher-entropy signal is available in
      this mode, so the entropy-drop route is inactive.

Everything here is a PURE tensor function — no I/O, no globals, fully unit-testable
(`experiments/test_token_triage.py`). The trainer wiring lives in
`train_lm.py` (`--token_triage`, default off → byte-identical).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch


# Route ids. IGNORE covers positions excluded from the loss entirely (target
# == -100 / think-token / cross-doc); it is never counted in the stats.
ROUTE_IGNORE = -1
ROUTE_DROP = 0
ROUTE_KD = 1
ROUTE_EASY = 2
ROUTE_NAMES = {ROUTE_DROP: "drop", ROUTE_KD: "kd", ROUTE_EASY: "easy"}

_EPS = 1e-12


@dataclass
class TriageConfig:
    """Thresholds + route weights for `compute_triage_mask`.

    Selection is by excess-loss quantile: the top `keep_frac` of valid tokens
    (highest student−reference excess) route to KD, unless overridden by an
    explicit `excess_cutoff`. Among the remaining (low-excess) tokens, those with
    student CE >= `hard_ce_cutoff` route to DROP (both-wrong / unlearnable), the
    rest to EASY. Independently, any token whose teacher top-k entropy exceeds
    `entropy_cutoff` is forced to DROP (the "high teacher entropy → drop" rule)
    — this override wins even over KD.
    """
    # --- selection ---
    keep_frac: float = 0.6
    excess_cutoff: Optional[float] = None       # explicit override of keep_frac
    entropy_cutoff: Optional[float] = None       # nats; > this ⇒ DROP. None = off
    hard_ce_cutoff: Optional[float] = None       # non-kept CE >= this ⇒ DROP. None = off
    # --- route weights ---
    w_kd: float = 1.0                            # KD-loss scale on kd_mask tokens
    w_ce_kept: float = 1.0                       # CE weight on KD-route tokens
    w_ce_easy: float = 0.1                       # CE weight on EASY-route tokens
    w_drop: float = 0.0                          # CE weight on DROP-route tokens
    # --- reference decoding ---
    stored_is_logprob: bool = True               # top-k stored as log-softmax (vLLM)
    residual_fallback: bool = True               # target∉top-k ⇒ residual-mass ref CE
    vocab_size: Optional[int] = None             # full vocab, for the residual split

    def validate(self) -> None:
        if not (0.0 <= self.keep_frac <= 1.0):
            raise ValueError(f"keep_frac must be in [0, 1], got {self.keep_frac}")


@dataclass
class TriageResult:
    """Per-token routing output + accounting.

    ce_weights : (B, T) float — multiply the per-token CE by this in the LM loss.
                 DROP→w_drop (0), KD→w_ce_kept, EASY→w_ce_easy, IGNORE→0.
    kd_mask    : (B, T) float in {0, 1} — 1 where a KD term should apply (the
                 KD-route tokens ∩ valid). AND this with the trainer's own
                 `valid_kd` mask.
    route      : (B, T) int — ROUTE_* per token (IGNORE at excluded positions).
    ref_ce     : (B, T) float — the reference (teacher) CE used for excess loss.
    excess     : (B, T) float — student_ce − ref_ce (−inf at excluded positions).
    stats      : dict — global fractions/counts over valid tokens.
    per_source : dict[int, dict] — per-source fractions/counts (empty if no
                 source_ids given).
    """
    ce_weights: torch.Tensor
    kd_mask: torch.Tensor
    route: torch.Tensor
    ref_ce: torch.Tensor
    excess: torch.Tensor
    stats: dict = field(default_factory=dict)
    per_source: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference-CE + entropy from a stored teacher top-k
# ---------------------------------------------------------------------------
def ref_ce_from_topk(teacher_ids: torch.Tensor,
                     teacher_vals: torch.Tensor,
                     targets: torch.Tensor,
                     cfg: TriageConfig) -> torch.Tensor:
    """Reference CE `−log p_teacher(target)` from a stored top-k, shape (B, T).

    `teacher_ids` (B, T, k) int, `teacher_vals` (B, T, k) float. When
    `cfg.stored_is_logprob` the values are log-softmax log-probs (vLLM store) so
    the in-top-k CE is EXACT; otherwise they are raw logits and we softmax over
    the top-k SUPPORT only (a documented approximation that ignores tail mass).

    target-not-in-top-k fallback:
      * log-prob store + residual_fallback: the tail mass `1 − Σ exp(logp_topk)`
        is split uniformly over the `V − k` unseen ids → `ref_ce = −log(tail/(V−k))`
        (large, so the token reads as "teacher also fails" → not selected). Needs
        `cfg.vocab_size`; if unset, falls back to the worst-top-k floor below.
      * otherwise (raw-logit store, or no vocab_size): the worst (smallest) top-k
        (log-)prob is used as a floor — `target` is assumed at least as unlikely
        as the least-likely stored id.
    """
    vals = teacher_vals.float()
    k = vals.shape[-1]
    if cfg.stored_is_logprob:
        logp = vals                                    # already log-softmax
    else:
        logp = torch.log_softmax(vals, dim=-1)          # over top-k support only
    match = (teacher_ids == targets.unsqueeze(-1))      # (B, T, k)
    in_topk = match.any(dim=-1)                         # (B, T)
    # In-top-k CE: take the MAX log-prob over matching slots (NOT the sum). The
    # vLLM store pads short rows by DUPLICATING the last id with logprob -30
    # (gen_teacher_logits_vllm.py); if the target equals that id, `match` fires
    # on the real slot AND every -30 pad slot, so a sum would inflate ref_ce by
    # ~30·n_pad. The real top-k slot's log-prob always dominates the -30 pads, so
    # the max recovers it. No-match rows give -inf here but are discarded by the
    # `torch.where(in_topk, ...)` below.
    neg_inf = logp.new_full((), float("-inf"))
    matched_logp = torch.where(match, logp, neg_inf).max(dim=-1).values  # (B, T)
    ref_ce_in = -matched_logp

    use_residual = (cfg.stored_is_logprob and cfg.residual_fallback
                    and cfg.vocab_size is not None)
    if use_residual:
        p_topk = logp.exp()                             # true probs (sum ≤ 1)
        tail = (1.0 - p_topk.sum(dim=-1)).clamp(min=_EPS)
        n_tail = max(int(cfg.vocab_size) - k, 1)
        ref_ce_out = -(tail / n_tail).clamp(min=_EPS).log()
    else:
        # Worst-top-k floor: −log of the smallest stored (log-)prob.
        worst_logp = logp.min(dim=-1).values
        if cfg.stored_is_logprob:
            ref_ce_out = -worst_logp
        else:
            # Raw-logit softmax-over-top-k: renormalised worst prob.
            ref_ce_out = -worst_logp
    return torch.where(in_topk, ref_ce_in, ref_ce_out)


def teacher_entropy_from_topk(teacher_vals: torch.Tensor,
                              cfg: TriageConfig) -> torch.Tensor:
    """Teacher predictive entropy (nats), shape (B, T), estimated from the stored
    top-k. A high value flags an inherently unpredictable target (the DROP route).

    For a log-prob store the top-k true probs are used and the leftover tail mass
    is handled by a MAX-ENTROPY assumption — spread uniformly over the `V − k`
    unseen ids (needs `cfg.vocab_size`; contributes `tail·(log(V−k) − log tail)`),
    which for a truly uniform teacher recovers the exact `log V`. Without a
    `vocab_size` the tail is folded in as a single lumped bucket (a monotone
    under-estimate). For a raw-logit store the entropy is over the softmax of the
    top-k support only (bounded by log k).
    """
    vals = teacher_vals.float()
    if cfg.stored_is_logprob:
        k = vals.shape[-1]
        p = vals.exp()                                  # true probs
        H = -(p * (p.clamp(min=_EPS)).log()).sum(dim=-1)
        tail = (1.0 - p.sum(dim=-1)).clamp(min=0.0)
        if cfg.vocab_size is not None and int(cfg.vocab_size) > k:
            # Max-entropy tail: `tail` mass spread over (V − k) unseen ids, each
            # at prob tail/(V−k). Entropy contribution = −tail·log(tail/(V−k)).
            n_tail = int(cfg.vocab_size) - k
            per = (tail / n_tail).clamp(min=_EPS)
            H = H - tail * per.log()
        else:
            H = H - tail * (tail.clamp(min=_EPS)).log()
        return H
    p = torch.softmax(vals, dim=-1)
    return -(p * (p.clamp(min=_EPS)).log()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------
def compute_triage_mask(student_ce_per_tok: torch.Tensor,
                        reference,
                        targets: torch.Tensor,
                        cfg: TriageConfig,
                        *,
                        teacher_ids: Optional[torch.Tensor] = None,
                        teacher_vals: Optional[torch.Tensor] = None,
                        ref_ce: Optional[torch.Tensor] = None,
                        valid_mask: Optional[torch.Tensor] = None,
                        source_ids: Optional[torch.Tensor] = None,
                        ) -> TriageResult:
    """Route every token to KD / DROP / EASY by excess loss; return per-token
    weights + accounting. Pure tensor function.

    student_ce_per_tok : (B, T) float — the student's per-token CE (DETACH before
                         calling; triage weights must not backprop through the
                         selection).
    reference          : the 2nd positional slot from the design signature. Accepts
                         EITHER a (B, T) `ref_ce` tensor (mode b) OR a
                         `(teacher_ids, teacher_vals)` tuple (mode a). May be None
                         if the corresponding keyword args are given instead.
    targets            : (B, T) long — the true next-token ids (−100 = ignore).
    cfg                : TriageConfig.
    valid_mask         : (B, T) bool — positions eligible for a loss. Defaults to
                         `targets != -100`. (The trainer additionally folds in
                         think-token / cross-doc masks before use.)
    source_ids         : (B, T) int|None — per-token source index for the
                         per-source audit (the data-hygiene mandate). None = skip.
    """
    cfg.validate()
    # --- unpack the flexible `reference` positional ---
    if reference is not None:
        if isinstance(reference, (tuple, list)):
            teacher_ids, teacher_vals = reference
        elif torch.is_tensor(reference):
            ref_ce = reference
        else:
            raise TypeError(
                "reference must be a (teacher_ids, teacher_vals) tuple, a ref_ce "
                f"tensor, or None; got {type(reference)}")

    student_ce = student_ce_per_tok.float()
    dev = student_ce.device
    if valid_mask is None:
        valid = targets != -100
    else:
        valid = valid_mask.bool()

    # --- reference CE + optional teacher entropy ---
    teacher_entropy = None
    if ref_ce is not None:
        if teacher_ids is not None or teacher_vals is not None:
            raise ValueError("pass EITHER ref_ce (mode b) OR teacher top-k "
                             "(mode a), not both")
        ref = ref_ce.float().to(dev)
    else:
        if teacher_ids is None or teacher_vals is None:
            raise ValueError("mode a needs both teacher_ids and teacher_vals")
        teacher_ids = teacher_ids.to(dev)
        teacher_vals = teacher_vals.to(dev)
        ref = ref_ce_from_topk(teacher_ids, teacher_vals, targets, cfg)
        if cfg.entropy_cutoff is not None:
            teacher_entropy = teacher_entropy_from_topk(teacher_vals, cfg)

    excess = student_ce - ref                           # (B, T)
    # Exclude invalid positions from the quantile + never keep them.
    neg_inf = torch.finfo(excess.dtype).min
    excess_valid_masked = torch.where(valid, excess, excess.new_full((), neg_inf))

    # --- excess-loss selection ---
    if cfg.excess_cutoff is not None:
        cutoff = float(cfg.excess_cutoff)
        kept = (excess >= cutoff) & valid
    else:
        vsum = int(valid.sum().item())
        if vsum == 0:
            cutoff = float("inf")
            kept = torch.zeros_like(valid)
        else:
            q = 1.0 - float(cfg.keep_frac)
            # Quantile over VALID excess only.
            ev = excess[valid].float()
            cutoff = float(torch.quantile(ev, min(max(q, 0.0), 1.0)).item())
            kept = (excess_valid_masked >= cutoff) & valid

    # --- route assignment (priority: entropy-DROP > KD > hard-CE-DROP > EASY) ---
    route = torch.full_like(targets, ROUTE_IGNORE, dtype=torch.long)
    route = torch.where(valid, torch.full_like(route, ROUTE_EASY), route)
    route = torch.where(kept, torch.full_like(route, ROUTE_KD), route)
    if cfg.hard_ce_cutoff is not None:
        both_wrong = valid & (~kept) & (student_ce >= float(cfg.hard_ce_cutoff))
        route = torch.where(both_wrong, torch.full_like(route, ROUTE_DROP), route)
    if teacher_entropy is not None and cfg.entropy_cutoff is not None:
        unlearnable = valid & (teacher_entropy > float(cfg.entropy_cutoff))
        route = torch.where(unlearnable, torch.full_like(route, ROUTE_DROP), route)

    # --- per-token CE weights + KD mask ---
    ce_weights = torch.zeros_like(student_ce)
    ce_weights = torch.where(route == ROUTE_EASY,
                             ce_weights.new_full((), cfg.w_ce_easy), ce_weights)
    ce_weights = torch.where(route == ROUTE_KD,
                             ce_weights.new_full((), cfg.w_ce_kept), ce_weights)
    ce_weights = torch.where(route == ROUTE_DROP,
                             ce_weights.new_full((), cfg.w_drop), ce_weights)
    kd_mask = (route == ROUTE_KD).float()

    excess_out = torch.where(valid, excess, excess.new_full((), neg_inf))

    stats = _route_stats(route, valid)
    stats["cutoff"] = cutoff
    stats["ref_ce_mean"] = (float(ref[valid].mean().item())
                            if valid.any() else 0.0)
    stats["excess_mean"] = (float(excess[valid].mean().item())
                            if valid.any() else 0.0)
    per_source = ({} if source_ids is None
                  else _per_source_stats(route, valid, source_ids))

    return TriageResult(ce_weights=ce_weights, kd_mask=kd_mask, route=route,
                        ref_ce=ref, excess=excess_out, stats=stats,
                        per_source=per_source)


def _route_stats(route: torch.Tensor, valid: torch.Tensor) -> dict:
    """Global route fractions/counts over valid tokens."""
    n = int(valid.sum().item())
    out = {"n_valid": n}
    for rid, name in ROUTE_NAMES.items():
        c = int(((route == rid) & valid).sum().item())
        out[f"n_{name}"] = c
        out[f"frac_{name}"] = (c / n) if n else 0.0
    # "keep" == KD-route fraction (the tokens carrying full CE + KD).
    out["frac_keep"] = out["frac_kd"]
    return out


def _per_source_stats(route: torch.Tensor, valid: torch.Tensor,
                      source_ids: torch.Tensor) -> dict:
    """Per-source route fractions/counts. `source_ids` (B, T) int; each valid
    token is attributed to its source. This is the auditable-per-source drop
    accounting the data-hygiene mandate (min_content_len lesson) requires."""
    sids = source_ids.to(route.device)
    out: dict = {}
    uniq = torch.unique(sids[valid]) if valid.any() else sids.new_empty(0)
    for s in uniq.tolist():
        m = valid & (sids == s)
        n = int(m.sum().item())
        rec = {"n_valid": n}
        for rid, name in ROUTE_NAMES.items():
            c = int(((route == rid) & m).sum().item())
            rec[f"n_{name}"] = c
            rec[f"frac_{name}"] = (c / n) if n else 0.0
        rec["frac_keep"] = rec["frac_kd"]
        out[int(s)] = rec
    return out


def format_triage_log(stats: dict) -> str:
    """Compact one-line summary for the trainer log:
    `triage(keep=..%, kd=..%, drop=..%, easy=..%)`. `keep` == `kd` (the
    full-CE+KD route); both are shown since the design doc names both."""
    def pct(key):
        return 100.0 * stats.get(key, 0.0)
    return (f"triage(keep={pct('frac_keep'):.1f}%, kd={pct('frac_kd'):.1f}%, "
            f"drop={pct('frac_drop'):.1f}%, easy={pct('frac_easy'):.1f}%)")


# ---------------------------------------------------------------------------
# Reference-CE store (mode b): a precomputed per-token ref CE cache.
# ---------------------------------------------------------------------------
# Mirrors `teacher_logits_io.LogitStoreReader`'s cursor contract so the trainer
# reads it in LOCKSTEP with the data iterator (one `next_block(n)` per
# microbatch) exactly like the KD store. The GPU cache-BUILD (score the mix with
# e.g. SmolLM2-360M and dump ref CE) is a documented follow-up; this reader +
# the round-trip writer make the `--triage_ref_ce_dir` wiring real and testable.
#
# Storage (safetensors shards + manifest.json):
#     ref_ce     fp16  [n]   the reference model's −log p(target) at each token
#     input_ids  uint32[n]   the actual token (for the alignment assertion)
class RefCEStoreWriter:
    """Append-and-shard writer for a per-token reference-CE cache."""

    def __init__(self, out_dir: str, ref_model: str = "",
                 tokenizer_name: str = "",
                 shard_max_tokens: int = 8_000_000):
        from safetensors.torch import save_file  # local import: optional dep
        self._save_file = save_file
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ref_model = str(ref_model)
        self.tokenizer_name = str(tokenizer_name)
        self.shard_max_tokens = int(shard_max_tokens)
        self._buf_ce: list[torch.Tensor] = []
        self._buf_ids: list[torch.Tensor] = []
        self._buf = 0
        self._total = 0
        self._shards: list[dict] = []
        self._idx = 0
        self._closed = False

    def append(self, ref_ce, input_ids) -> None:
        ce = torch.as_tensor(ref_ce).detach().cpu().float().flatten()
        ids = torch.as_tensor(input_ids).detach().cpu().flatten()
        if ce.shape != ids.shape:
            raise ValueError(f"ref_ce {tuple(ce.shape)} != input_ids "
                             f"{tuple(ids.shape)}")
        self._buf_ce.append(ce.to(torch.float16))
        self._buf_ids.append(ids.to(torch.int64).to(torch.uint32))
        self._buf += ce.shape[0]
        self._total += ce.shape[0]
        while self._buf >= self.shard_max_tokens:
            self._flush(self.shard_max_tokens)

    def _flush(self, n_take: int) -> None:
        ce = torch.cat(self._buf_ce)
        ids = torch.cat(self._buf_ids)
        name = f"refce_{self._idx:05d}.safetensors"
        start = self._shards[-1]["end"] if self._shards else 0
        self._save_file({"ref_ce": ce[:n_take].contiguous(),
                         "input_ids": ids[:n_take].contiguous()},
                        str(self.dir / name))
        self._shards.append({"name": name, "start": start,
                             "end": start + n_take, "n": n_take})
        self._idx += 1
        rem_ce, rem_ids = ce[n_take:].contiguous(), ids[n_take:].contiguous()
        self._buf_ce = [rem_ce] if rem_ce.numel() else []
        self._buf_ids = [rem_ids] if rem_ids.numel() else []
        self._buf = rem_ce.shape[0]

    def close(self) -> dict:
        if self._buf > 0:
            self._flush(self._buf)
        manifest = {"total_tokens": self._total, "ref_model": self.ref_model,
                    "tokenizer_name": self.tokenizer_name, "shards": self._shards}
        with open(self.dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        self._closed = True
        return manifest

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._closed:
            self.close()


class RefCEStoreReader:
    """Sequential-cursor reader over a per-token reference-CE cache. Same
    `next_block(n)` contract as `teacher_logits_io.LogitStoreReader`."""

    def __init__(self, directory: str):
        from safetensors.torch import safe_open
        self._safe_open = safe_open
        self.dir = pathlib.Path(directory)
        with open(self.dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.total_tokens = int(self.manifest["total_tokens"])
        self.ref_model = self.manifest.get("ref_model", "")
        self.tokenizer_name = self.manifest.get("tokenizer_name", "")
        self.shards = list(self.manifest["shards"])
        expect = 0
        for sh in self.shards:
            if int(sh["start"]) != expect:
                raise ValueError(f"gap/overlap at shard {sh['name']}")
            expect = int(sh["end"])
        if expect != self.total_tokens:
            raise ValueError("shards do not tile total_tokens")
        self._cursor = 0
        self._handles: dict = {}

    def __len__(self) -> int:
        return self.total_tokens

    def tell(self) -> int:
        return self._cursor

    def next_block(self, n: int):
        if self._cursor + n > self.total_tokens:
            raise IndexError(
                f"next_block({n}) past end: cursor={self._cursor}, "
                f"total={self.total_tokens} — ref-CE cache too small; "
                "regenerate covering more tokens.")
        out = self.get_range(self._cursor, self._cursor + n)
        self._cursor += n
        return out

    def get_range(self, start: int, end: int):
        ce_parts, id_parts = [], []
        for sh in self.shards:
            s0, s1 = int(sh["start"]), int(sh["end"])
            lo, hi = max(start, s0), min(end, s1)
            if lo >= hi:
                continue
            a, b = lo - s0, hi - s0
            h = self._handle(sh["name"])
            ce_parts.append(h.get_slice("ref_ce")[a:b])
            id_parts.append(h.get_slice("input_ids")[a:b].to(torch.int64))
        if not ce_parts:
            return (torch.empty(0, dtype=torch.float16),
                    torch.empty(0, dtype=torch.int64))
        ce = torch.cat(ce_parts) if len(ce_parts) > 1 else ce_parts[0]
        ids = torch.cat(id_parts) if len(id_parts) > 1 else id_parts[0]
        return ce, ids

    def _handle(self, name: str):
        h = self._handles.get(name)
        if h is None:
            h = self._safe_open(str(self.dir / name), framework="pt")
            self._handles[name] = h
        return h


# ---------------------------------------------------------------------------
# Trainer helper: build a TriageConfig from argparse args
# ---------------------------------------------------------------------------
def triage_config_from_args(args, vocab_size: Optional[int] = None) -> TriageConfig:
    """Construct a TriageConfig from the `--triage_*` CLI flags. Sentinels:
    negative entropy / hard-CE cutoffs mean OFF; a negative excess override is
    ignored (keep_frac drives selection)."""
    ent = float(getattr(args, "triage_entropy_cutoff", -1.0))
    hard = float(getattr(args, "triage_hard_ce_cutoff", -1.0))
    return TriageConfig(
        keep_frac=float(getattr(args, "triage_keep_frac", 0.6)),
        entropy_cutoff=(ent if ent >= 0.0 else None),
        hard_ce_cutoff=(hard if hard >= 0.0 else None),
        w_kd=1.0,  # scaled on the trainer side by --triage_kd_weight/--distill_weight
        w_ce_easy=float(getattr(args, "triage_easy_weight", 0.1)),
        stored_is_logprob=not bool(getattr(args, "triage_store_raw_logits", False)),
        vocab_size=vocab_size,
    )
