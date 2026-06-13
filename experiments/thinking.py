from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@contextmanager
def clean_latent_thread(model, *, film_bypass: bool | None = None,
                        no_activation_ckpt: bool | None = None):
    """Run the latent-thinking thread on a CLEAN model config, restore after.

    Canonical home of the WM-contamination fix (2026-06-05): every latent
    training path feeds back the clean ``out_norm(h)``; leaving WorkingMemory
    on at a latent callsite feeds back ``out_norm(h) + α·W_proj(WM_read)`` — an
    OOD signal the latent adapter was built to avoid → corrupted feedback →
    run-on output. Every ``generate_latent_think`` / latent-loss callsite must
    run inside this context (or set the toggles permanently, as whole-process
    trainers like latent_rl.py do).

    Optional toggles (default: leave untouched, preserving caller semantics):
      film_bypass=True        — single-forward FiLM for short latent seqs
                                (validated speed win, parity with training).
      no_activation_ckpt=True — checkpointing's backward RECOMPUTES the FLA
                                kernel at the latent path's short/odd lengths,
                                intermittently hitting a Blackwell
                                "unspecified launch failure"; latent seqs are
                                tiny so retaining activations costs ~nothing.
    """
    saved_mem = getattr(model, "use_memory", False)
    saved_bypass = getattr(model, "_film_bypass", False)
    saved_ckpt = getattr(model, "activation_checkpointing", False)
    model.use_memory = False
    if film_bypass is not None:
        model._film_bypass = bool(film_bypass)
    if no_activation_ckpt is not None:
        model.activation_checkpointing = not bool(no_activation_ckpt)
    try:
        yield model
    finally:
        model.use_memory = saved_mem
        if film_bypass is not None:
            model._film_bypass = saved_bypass
        if no_activation_ckpt is not None:
            model.activation_checkpointing = saved_ckpt


def latent_think_step_input(model, z_premem, wm_inj=None):
    """THE single think-step input builder, shared by the co-train grad twin,
    the no-grad measurement primitive, and the inference generator — so the
    train/eval latent formula can never silently diverge (the recurring killer).

    next-input = adapter(z_premem) [+ mem_alpha · wm_inj]

    `z_premem` is the CLEAN pre-memory hidden the adapter was trained on. The WM
    term is added AFTER the adapter and α-gated from ~0 (mem_alpha), so it shapes
    the next think input without contaminating the adapter's input manifold (the
    2026-06-05 corruption fix). When the model has no `mem_alpha` (cooperation
    off) or `wm_inj is None`, this is exactly the legacy `adapter(z)` path —
    byte-identical to pre-cooperation behaviour.
    """
    zi = model.apply_latent_feedback_adapter(z_premem)
    ma = getattr(model, "mem_alpha", None)
    if ma is not None and wm_inj is not None:
        zi = zi + ma.to(zi.dtype) * wm_inj.to(zi.dtype)
    return zi


def latent_wm_injection(model, *, grad: bool):
    """The WM retrieval to feed into the next think step, or None when WM/
    cooperation is off. Reads the pre-read-mask injection stash from the most
    recent forward (grad-carrying for training, detached for inference)."""
    if getattr(model, "mem_alpha", None) is None or not getattr(model, "use_memory", False):
        return None
    mem = getattr(model, "memory", None)
    if mem is None:
        return None
    inj = getattr(mem, "_last_injection_grad" if grad else "_last_injection", None)
    return inj[:, -1:, :] if inj is not None else None


def load_latent_model(ckpt_path: str, device: str = "cuda", *,
                      train: bool = False, fresh_adapter: bool = True,
                      wm_on: bool = False):
    """Canonical model-bootstrap for latent-thinking scripts.

    One place for the boilerplate that had drifted across ~8 scripts (missing
    force flags, hardcoded tokenizers): build with the latent force flags, move
    to device, disable WM + gist for the latent thread, resolve the
    thinking-token id and tokenizer from cfg.

    `wm_on=True` keeps WorkingMemory enabled (for the WM×latent COOPERATION path,
    where the latent step feeds back `adapter(h_premem)+mem_alpha·WM_inj`); also
    sets the pre-memory thread so the WM injection can't contaminate the adapter
    input. Default False = the clean WM-off latent thread (parity with the
    standard latent generator).

    Returns ``(model, cfg, thinking_id, tok, eos_id)``.
    """
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    model, cfg = build_model_from_ckpt(
        ckpt_path, force_state_readonly=True,
        force_use_latent_feedback_adapter=True if fresh_adapter else None,
        force_cooperative_latent_wm=True if wm_on else None)
    model = model.to(device)
    model.train() if train else model.eval()
    model._gist_loss_enabled = False
    if wm_on:
        model._latent_feedback_premem = True   # carry clean pre-mem hidden
        # leave use_memory as-is (on) so WM injects at think slots
    elif getattr(model, "use_memory", False):
        model.use_memory = False        # latent thread runs WM-off (parity)
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    return model, cfg, thinking_id, tok, tok.eos_token_id


@dataclass
class ThinkContinuation:
    context_ids: list[int]
    target_id: int
    depth: int = 0
    accum_nll: float = 0.0
    accum_cost: float = 0.0
    origin_step: int = 0
    decision_context_ids: list[int] | None = None
    immediate_nll: float = 0.0


@dataclass
class ThinkReplay:
    context_ids: list[int]
    target_id: int
    target_is_thinking: bool


@dataclass
class ThoughtTrajectory:
    """A single sampled thinking trajectory for RL."""
    initial_context: list[int]
    target_id: int
    actions: list[int]  # 1 for Think, 0 for Emit
    action_logprobs: list[float]
    final_logits: torch.Tensor | None = None
    depth: int = 0
    # Counterfactual baseline: the logits the model would have emitted at
    # depth=0 (immediate emit, no thinking). Captured once at the start of
    # every rollout so reward shaping can ask "did thinking pay off vs.
    # not thinking at all?" without re-running the forward.
    immediate_logits: torch.Tensor | None = None


# fla's short_conv routes seq_len==1 through a decode/step path that requires
# a cache tensor; the rollout always passes None for cache, which crashes
# Triton's autotuner. Keep every rollout input ≥ MIN_ROLLOUT_LEN.
MIN_ROLLOUT_LEN = 2


@torch.no_grad()
def generate_thought_trajectories(
    model: torch.nn.Module,
    initial_contexts: list[list[int]],
    target_ids: list[int],
    n_group: int,
    max_depth: int,
    thinking_token_id: int,
    block_size: int,
    temperature: float = 1.0,
    device: str = "cuda",
    pad_token_id: int = 0,
) -> list[list[ThoughtTrajectory]]:
    """Sample n_group trajectories per (context, target) pair.

    The model decides at each step whether to emit (terminate) or think
    (append another thinking token and retry). The working memory inside
    `TinyLM.forward` reads from past hidden states of the same sequence at
    every think position — there is no external RAG store in this rollout.
    """
    B = len(initial_contexts)
    N = n_group

    # Initialize trajectories: [B][N]
    trajectories: list[list[ThoughtTrajectory]] = [
        [ThoughtTrajectory(initial_context=ctx, target_id=tid, actions=[], action_logprobs=[])
         for _ in range(N)]
        for ctx, tid in zip(initial_contexts, target_ids)
    ]

    # Active trajectories: flat list of (B*N)
    active_trajs: list[ThoughtTrajectory] = [t for group in trajectories for t in group]
    finished_mask = torch.zeros(B * N, dtype=torch.bool, device=device)

    # Cap context so appending the worst-case (max_depth) thinking tokens never
    # overflows block_size — keeps the chunked-conv path engaged consistently.
    ctx_budget = max(MIN_ROLLOUT_LEN, block_size - max_depth)

    for d in range(max_depth + 1):
        if finished_mask.all():
            break

        ctx_list = []
        for t in active_trajs:
            ctx = (t.initial_context + ([thinking_token_id] * t.depth))[-ctx_budget:]
            ctx_list.append(ctx)

        # Pad contexts for batching. Always pad to at least MIN_ROLLOUT_LEN so
        # fla's short_conv does not route seq_len=1 inputs through the
        # decode/step path (which expects a cache tensor and crashes).
        max_len = max(MIN_ROLLOUT_LEN, max(len(c) for c in ctx_list))
        input_ids = torch.full(
            (B * N, max_len), int(pad_token_id), dtype=torch.long, device=device
        )
        last_positions = torch.empty(B * N, dtype=torch.long, device=device)
        for i, c in enumerate(ctx_list):
            input_ids[i, -len(c):] = torch.tensor(c, dtype=torch.long, device=device)
            last_positions[i] = max_len - 1  # left-padded: real tokens end at -1

        # Single forward pass.
        logits, gate = model(input_ids, return_gate=True)

        # Gather at the last real position per row.
        row_idx = torch.arange(B * N, device=device)
        last_logits = logits[row_idx, last_positions]
        last_gate = gate[row_idx, last_positions]

        # Action probability: gate = prob(Emit), so prob(Think) = 1 - gate
        probs = torch.stack([1 - last_gate, last_gate], dim=-1)  # [BN, 2]
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()  # 0: Think, 1: Emit
        log_probs = dist.log_prob(actions)

        for i, t in enumerate(active_trajs):
            if finished_mask[i]:
                continue

            # Capture the depth-0 (immediate-emit) baseline before any
            # think tokens are appended — used by counterfactual reward
            # shaping in `compute_grpo_advantages`.
            if d == 0:
                t.immediate_logits = mask_token_logit(
                    last_logits[i], thinking_token_id
                ).cpu()

            action = int(actions[i].item())
            t.actions.append(1 - action)  # Store 1 for Think, 0 for Emit
            t.action_logprobs.append(float(log_probs[i].item()))

            if action == 1 or d == max_depth:  # Emit or reached max depth
                finished_mask[i] = True
                t.final_logits = mask_token_logit(last_logits[i], thinking_token_id).cpu()
                t.depth = d
            else:
                t.depth += 1

    return trajectories


def compute_grpo_advantages(
    trajectory_groups: list[list[ThoughtTrajectory]],
    ponder_cost: float,
    *,
    ponder_shape: str = "linear",
    counterfactual: bool = False,
    separate_ponder_norm: bool = False,
) -> torch.Tensor:
    """Compute GRPO advantages with configurable ponder-cost shaping.

    Reward shapes (selected by the kwargs):

    - **Default (backward-compat)** — `linear` shape, no counterfactual,
      ponder bundled into the GRPO normalization:
      ``reward = -CE(d) - cost * d``, then z-score within group.
      This is the original formula and stays default.

    - ``ponder_shape='quadratic'`` — uses ``cost * d^2`` instead of
      ``cost * d``. Marginal cost grows with depth, matching the
      diminishing value of deeper thinking.

    - ``counterfactual=True`` — clamps the task component at the
      depth-0 baseline so thinking can never make the task reward
      worse than not thinking, but *always* charges the depth cost:
      ``reward = max(-CE(d), -CE(0)) - cost * f(d)``.
      Encourages exploration of thinking (no task-side punishment for
      trying) while still pushing toward *minimum* depth via the cost
      term — matches "if thinking solves the task, it pays off; if
      not, it costs but doesn't catastrophise".

    - ``separate_ponder_norm=True`` (only meaningful when
      ``counterfactual=False``): z-score task reward within group,
      then subtract the absolute ponder term *after* normalization.
      Prevents the group-z-score from squashing the (typically much
      smaller) ponder magnitude into noise.

    Returns a `(B, N)` advantage tensor.
    """
    if ponder_shape not in ("linear", "quadratic"):
        raise ValueError(f"unknown ponder_shape: {ponder_shape!r}")
    B = len(trajectory_groups)
    N = len(trajectory_groups[0])

    # Compute per-trajectory task rewards at chosen depth and at depth 0,
    # plus the depth itself, in a single pass.
    task_rewards_d = torch.zeros((B, N))
    task_rewards_0 = torch.zeros((B, N))
    depths = torch.zeros((B, N))
    for i, group in enumerate(trajectory_groups):
        for j, t in enumerate(group):
            target = torch.tensor(t.target_id)
            ce_d = F.cross_entropy(t.final_logits.unsqueeze(0),
                                    target.unsqueeze(0))
            task_rewards_d[i, j] = -float(ce_d.item())
            if t.immediate_logits is not None:
                ce_0 = F.cross_entropy(t.immediate_logits.unsqueeze(0),
                                        target.unsqueeze(0))
                task_rewards_0[i, j] = -float(ce_0.item())
            else:
                # Fallback when an older rollout didn't capture the baseline.
                task_rewards_0[i, j] = task_rewards_d[i, j]
            depths[i, j] = float(t.depth)

    # Depth cost: shape-dependent.
    if ponder_shape == "linear":
        depth_cost = ponder_cost * depths
    else:  # quadratic
        depth_cost = ponder_cost * (depths ** 2)

    # Use the biased (population) std estimator so single-element groups
    # don't blow up to NaN — `unbiased=True` with N=1 has 0 degrees of
    # freedom and returns NaN. The 1e-8 epsilon then prevents division by
    # zero. This matches standard GRPO / PPO advantage-normalisation
    # convention.

    if counterfactual:
        # Clamp the task component at the depth-0 baseline ("thinking
        # never makes the task reward worse than not thinking") and
        # then always charge the depth cost. Encourages exploration of
        # thinking while still pushing the policy toward minimum depth.
        task_component = torch.maximum(task_rewards_d, task_rewards_0)
        rewards = task_component - depth_cost
        means = rewards.mean(dim=1, keepdim=True)
        stds = rewards.std(dim=1, unbiased=False, keepdim=True) + 1e-8
        return (rewards - means) / stds

    if separate_ponder_norm:
        # Z-score task only; ponder stays in absolute units so the
        # group-relative comparison doesn't squash its (typically small)
        # magnitude into noise.
        task_means = task_rewards_d.mean(dim=1, keepdim=True)
        task_stds = task_rewards_d.std(dim=1, unbiased=False,
                                         keepdim=True) + 1e-8
        return (task_rewards_d - task_means) / task_stds - depth_cost

    # Original behavior (backward-compatible): ponder bundled into reward,
    # full reward z-scored within group.
    rewards = task_rewards_d - depth_cost
    means = rewards.mean(dim=1, keepdim=True)
    stds = rewards.std(dim=1, unbiased=False, keepdim=True) + 1e-8
    return (rewards - means) / stds


class ThinkContinuationQueue:
    def __init__(self, max_len: int):
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.max_len = int(max_len)
        self._items: deque[ThinkContinuation] = deque()
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, item: ThinkContinuation) -> None:
        if len(self._items) >= self.max_len:
            raise OverflowError(
                "thinking continuation queue is full; refusing to drop an "
                "unresolved target obligation"
            )
        self._items.append(item)

    def extend(self, items: Iterable[ThinkContinuation]) -> None:
        for item in items:
            self.enqueue(item)

    def pop_batch(self, n: int) -> list[ThinkContinuation]:
        out: list[ThinkContinuation] = []
        for _ in range(min(int(n), len(self._items))):
            out.append(self._items.popleft())
        return out

    def mean_depth(self) -> float:
        if not self._items:
            return 0.0
        return sum(item.depth for item in self._items) / len(self._items)

    def max_depth(self) -> int:
        if not self._items:
            return 0
        return max(item.depth for item in self._items)


class ThinkReplayQueue:
    def __init__(self, max_len: int):
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.max_len = int(max_len)
        self._items: deque[ThinkReplay] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, item: ThinkReplay) -> None:
        if len(self._items) >= self.max_len:
            self._items.popleft()
        self._items.append(item)

    def pop_batch(self, n: int) -> list[ThinkReplay]:
        out: list[ThinkReplay] = []
        for _ in range(min(int(n), len(self._items))):
            out.append(self._items.popleft())
        return out


def mask_token_logit(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    masked = logits.clone()
    masked[..., int(token_id)] = -torch.inf
    return masked


def cross_entropy_masking_token(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_id: int,
    reduction: str = "none",
) -> torch.Tensor:
    return F.cross_entropy(
        mask_token_logit(logits, token_id),
        targets,
        reduction=reduction,
    )


def choose_think_actions(
    logits: torch.Tensor,
    thinking_token_id: int,
    policy: str,
    threshold: float,
    temperature: float,
    allow_think: torch.Tensor | bool = True,
) -> torch.Tensor:
    """Return a boolean mask of examples whose on-policy action is THINKING."""
    if policy == "greedy":
        think = logits.argmax(dim=-1) == thinking_token_id
    elif policy == "threshold":
        probs = torch.softmax(logits, dim=-1)
        think = probs[..., thinking_token_id] >= threshold
    elif policy == "sample":
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        sample = torch.distributions.Categorical(logits=logits / temperature).sample()
        think = sample == thinking_token_id
    else:
        raise ValueError(f"unknown thinking policy: {policy!r}")

    if isinstance(allow_think, bool):
        return think if allow_think else torch.zeros_like(think, dtype=torch.bool)
    return think & allow_think.to(device=think.device, dtype=torch.bool)


def choose_explore_actions(
    scores: torch.Tensor,
    probability: float,
    mode: str = "uniform",
    top_frac: float = 1.0,
    min_score: float = 0.0,
    allow: torch.Tensor | bool = True,
) -> torch.Tensor:
    """Sample exploratory THINKING actions, optionally biased to high-score sites."""
    if probability <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if not (0.0 < top_frac <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")

    if isinstance(allow, bool):
        eligible = torch.ones_like(scores, dtype=torch.bool) if allow else torch.zeros_like(scores, dtype=torch.bool)
    else:
        eligible = allow.to(device=scores.device, dtype=torch.bool).clone()
    eligible &= scores >= min_score

    if mode == "uniform":
        candidate = eligible
    elif mode == "high_ce":
        candidate = torch.zeros_like(eligible)
        idx = eligible.reshape(-1).nonzero(as_tuple=False).flatten()
        if idx.numel() > 0:
            k = max(1, int(idx.numel() * top_frac))
            flat_scores = scores.reshape(-1)
            top_idx = idx[torch.topk(flat_scores[idx], k=k).indices]
            candidate.reshape(-1)[top_idx] = True
    else:
        raise ValueError(f"unknown exploration mode: {mode!r}")

    return (torch.rand_like(scores.float()) < probability) & candidate


def build_continuation_batch(
    items: list[ThinkContinuation],
    block_size: int,
    pad_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad/crop continuation contexts and return (ids, targets, last_pos)."""
    if not items:
        raise ValueError("cannot build an empty continuation batch")
    ctx_rows: list[list[int]] = []
    last_positions: list[int] = []
    targets: list[int] = []
    for item in items:
        ctx = item.context_ids[-block_size:]
        if not ctx:
            raise ValueError("continuation context_ids must be non-empty")
        last_positions.append(len(ctx) - 1)
        targets.append(int(item.target_id))
        if len(ctx) < block_size:
            ctx = ctx + [int(pad_token_id)] * (block_size - len(ctx))
        ctx_rows.append(ctx)
    return (
        torch.tensor(ctx_rows, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(last_positions, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Canonical latent-thinking primitive.
#
# This is THE single implementation of "run R state-readonly latent think
# steps and read the resulting next-token distribution". Every consumer — the
# gate-calibration loss, the gate-calibration probe, eval generators — MUST
# import this rather than re-deriving an append-and-forward, so the mechanism
# can never silently diverge again. (It diverged once: the gate teacher + probe
# were built on the DISCRETE-token append `[THINK]*K` mechanism, which is the
# validated `mode="token"` baseline that does NOT help, while the validated
# breakthrough is the LATENT mechanism below — feed the trunk's own hidden
# state back as the think-slot input embedding. See latent_think.py and
# THINKING_GATE_SELECTIVITY_2026_05_30.md.)
#
# Mechanism (Coconut-style, state-readonly): append ONE think slot whose input
# embedding is the trunk's own last hidden state, iterate R times. The model is
# expected to be built with `state_readonly_at_think=True` so the DeltaNet
# write-gate β=0 at the think slot — the think READS the recurrent state but
# never WRITES, so long-range bindings are preserved.
# ---------------------------------------------------------------------------


def _logits_hidden(out):
    """Extract (logits, hidden) from a `model(..., return_hidden=True)` return.

    With `return_aux=False, return_gate=False` the forward returns
    `(logits, hidden)` — or `(logits, hidden, gist_scalar)` in training mode
    with the trunk-gist loss enabled (the scalar is appended LAST). So hidden
    is always index 1. A bare-tensor return (no hidden) is a caller error here.
    """
    return out[0], out[1]


@torch.no_grad()
def _latent_think_logits(model, prefixes: torch.Tensor, *, R: int,
                         thinking_token_id: int,
                         wm_off: bool = False) -> torch.Tensor:
    """Logits (N, V) at the think slot after R latent (hidden-feedback) steps.

    ``prefixes`` (N, Lmax) is the (left-padded) context. We append one think
    slot, initialise its latent from the prefix's final hidden state, and feed
    that latent back as the slot's input embedding for R iterations. ``input_ids``
    still carry the think-token id at the slot (so state-readonly / working-
    memory masks fire there), while ``inputs_embeds`` overrides the embedding —
    exactly latent_think.think_forward(mode="latent").

    ``wm_off``: pass an all-zero ``mem_read_mask`` so WorkingMemory never
    injects — isolates the trunk's latent computation from the WM-injection
    confound (a use_memory model injects WM at the think slot by default).
    """
    base_emb = model.embed(prefixes)                             # (N, Lmax, d)
    N = prefixes.shape[0]
    think_col = torch.full((N, 1), int(thinking_token_id),
                           dtype=prefixes.dtype, device=prefixes.device)
    ids = torch.cat([prefixes, think_col], dim=1)                # (N, Lmax+1)
    mem_read_mask = (torch.zeros(ids.shape, dtype=torch.float32,
                                 device=prefixes.device)
                     if wm_off else None)
    # Memory-frugal (2026-06-04): the old path called model(..., return_hidden=
    # True) which materialises the FULL (N, Lmax+1, V) logits EVERY iteration,
    # but only the think slot (last position) is ever used. V≈49k ≫ d, so that
    # tensor dominates — it is what made gate_calibration OOM at T=2048.
    # skip_lm_head returns the post-out_norm+memory hidden WITHOUT the lm_head
    # matmul; we apply lm_head to the LAST position only, at the end. Result is
    # identical (lm_head(h)[:, -1] == lm_head(h[:, -1])). _last_premem_hidden
    # preserves the premem-feedback semantics of the old return_hidden path.
    premem = bool(getattr(model, "_latent_feedback_premem", False))
    h0 = model(prefixes, skip_lm_head=True)                      # (N, Lmax, d)
    z = (model._last_premem_hidden if premem else h0)[:, -1:, :]  # (N, 1, d)
    h_last = None
    for _ in range(max(1, int(R))):
        # Map the fed-back out_norm hidden into the input-embedding manifold
        # via the learned adapter (identity when the model has none / it is
        # untrained — so this is byte-identical to the no-adapter path then).
        zi = model.apply_latent_feedback_adapter(z)
        ie = torch.cat([base_emb, zi.to(base_emb.dtype)], dim=1)  # (N, Lmax+1, d)
        h_last = model(ids, inputs_embeds=ie, skip_lm_head=True,
                       mem_read_mask=mem_read_mask)              # (N, Lmax+1, d)
        z = (model._last_premem_hidden if premem else h_last)[:, -1:, :]
    # lm_head at the think slot ONLY → (N, V), not (N, Lmax+1, V).
    _dev = h_last.device.type
    with torch.autocast(device_type=_dev, dtype=torch.bfloat16,
                        enabled=(_dev == "cuda")):
        last_logits = model.lm_head(h_last[:, -1, :])           # (N, V)
    return last_logits


def latent_think_logp(model, prefixes: torch.Tensor, true_next: torch.Tensor,
                      *, R: int, thinking_token_id: int,
                      pad_id: int = 0, wm_off: bool = False) -> torch.Tensor:
    """log p(true_next | prefix + R latent thinks) at the think slot. (N,)

    The canonical measurement of "does latent thinking help here": compare this
    against the no-think baseline log p(true_next | prefix). The return is a
    pure target (computed under no_grad) — callers detach it and let gradient
    flow through the gate logits, not through this forward.

    ``wm_off`` suppresses WorkingMemory injection (diagnostic isolation; see
    ``_latent_think_logits``).
    """
    assert int(thinking_token_id) != int(pad_id), (
        f"thinking_token_id ({thinking_token_id}) must differ from pad_id "
        f"({pad_id}) — pad-as-think corrupts the state-readonly mask")
    logits = _latent_think_logits(
        model, prefixes, R=R, thinking_token_id=thinking_token_id,
        wm_off=wm_off).float()
    lp = F.log_softmax(logits, dim=-1)                            # (N, V)
    return lp.gather(1, true_next.view(-1, 1)).squeeze(1)         # (N,)


# torch.compile-safety: the variable-length extra forward tripped an Inductor
# symbolic-shape assertion in prior work. Disabling dynamo around the primitive
# keeps a compiled training run from crashing; the @no_grad already means there
# is nothing to compile through for the backward.
try:  # pragma: no cover - depends on torch build
    latent_think_logp = torch._dynamo.disable(latent_think_logp)  # type: ignore
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Latent thinking CO-TRAINING loss (v9 — make thinking USEFUL from day 1).
#
# The measurement primitive above is @no_grad. CO-TRAINING needs gradient to
# flow through the R latent steps INTO the trunk, so the trunk learns to do
# useful sequential computation during thinking. On v8 (no latent co-training)
# the probe showed latent thinking HURTS (Δlogp ≈ -7). This loss trains the
# trunk so the post-R-latent-think prediction of the TRUE next token is good —
# the direct analog of the validated latent_think.py final-answer supervision,
# applied to real pretrain text. Co-trained alongside the normal no-think LM
# loss, the gate can then learn WHERE thinking helps.
# ---------------------------------------------------------------------------


def _latent_think_logits_grad(model, prefixes: torch.Tensor, *, R: int,
                              thinking_token_id: int) -> torch.Tensor:
    """Grad-enabled twin of `_latent_think_logits` — gradient flows through the
    R latent steps into the trunk. Returns logits (N, V) at the think slot."""
    base_emb = model.embed(prefixes)
    N = prefixes.shape[0]
    think_col = torch.full((N, 1), int(thinking_token_id),
                           dtype=prefixes.dtype, device=prefixes.device)
    ids = torch.cat([prefixes, think_col], dim=1)
    _l0, h0 = _logits_hidden(model(prefixes, return_hidden=True))
    z = h0[:, -1:, :]
    last_logits = None
    for _ in range(max(1, int(R))):
        # Shared think-step builder: adapter(z) [+ mem_alpha·WM_inj] — gradient
        # flows through the adapter, the R latent steps into the trunk, AND (in
        # WM×latent cooperation) the WM read path + mem_alpha. wm_inj is None
        # unless cooperation is on (use_memory + mem_alpha) → byte-identical to
        # the legacy adapter-only path otherwise. Uses the SAME helper as the
        # inference generator so train/eval can't diverge.
        wm_inj = latent_wm_injection(model, grad=True)
        zi = latent_think_step_input(model, z, wm_inj)
        ie = torch.cat([base_emb, zi.to(base_emb.dtype)], dim=1)
        logits, h = _logits_hidden(model(ids, inputs_embeds=ie,
                                         return_hidden=True))
        z = h[:, -1:, :]
        last_logits = logits[:, -1, :]
    return last_logits


def _latent_clean_mask(input_ids, targets, *, thinking_token_id, pad_id, eos_id):
    valid = targets != -100
    valid = valid & (targets != int(thinking_token_id)) & (targets != int(pad_id))
    valid = valid & (input_ids != int(thinking_token_id)) & (input_ids != int(pad_id))
    if eos_id is not None:
        valid = valid & (targets != int(eos_id))
    valid[:, input_ids.shape[1] - 1] = False  # need a t+1 target
    return valid


def _selective_position_weights(model, input_ids, bsel, tsel):
    """Per-(valid-position) sampling weight in [eps, ∞) biased toward where
    thinking SHOULD help: high gate σ (P(think)=1-σ... see note) OR high
    no-think predictive entropy. Computed under no_grad from ONE clean forward.

    Note on the gate convention: `model._last_gate` is σ = P(EMIT). Thinking is
    wanted where the model is UNCERTAIN, i.e. where σ is LOW (the gate would
    route to think) OR the next-token entropy is HIGH. We weight by
    `w = (1 - σ) + entropy_norm` so both "gate wants to think" and "model is
    uncertain" pull the sample toward useful positions. Returns a (P,) tensor.
    """
    with torch.no_grad():
        out = model(input_ids, return_hidden=False, return_gate=True)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.float()
        # Predictive entropy of the no-think next-token distribution.
        logp = F.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1)                  # (B, T)
        # Normalise entropy to ~[0,1] by its own batch max (robust, scale-free).
        ent = ent / ent.amax().clamp_min(1e-6)
        gate = getattr(model, "_last_gate", None)               # σ=P(emit), (B,T)
        if gate is not None:
            gate = gate.float()
            want_think = (1.0 - gate.clamp(0.0, 1.0))           # high where σ low
        else:
            want_think = torch.zeros_like(ent)
        score = want_think + ent                                # (B, T)
        w = score[bsel, tsel]                                   # (P,)
        # Floor so every valid position keeps a non-zero probability (the
        # selective bias is a tilt, not a hard filter — avoids starving the
        # trunk of "thinking didn't help here" negatives entirely).
        return w.clamp_min(1e-3)


def latent_cotrain_loss(model, input_ids: torch.Tensor, targets: torch.Tensor,
                        *, R: int, thinking_token_id: int,
                        sample_frac: float = 0.05, max_positions: int = 32,
                        max_prefix_len: int = 256,
                        pad_id: int = 0, eos_id: int | None = None,
                        selective: bool = False,
                        generator: torch.Generator | None = None):
    """Grad CE on the post-R-latent-think prediction at sampled positions.

    Returns (loss, mean_delta_logp, n_positions) where ``mean_delta_logp`` is
    the (detached) diagnostic Δlogp = logp_after_R_thinks(true) −
    logp_no_think(true): the v9 validation signal — it should climb from v8's
    ≈ -7 toward 0/positive as the trunk learns to use thinking. ``None`` if no
    positions sampled.

    ``selective``: when True, sample the ``max_positions`` positions WEIGHTED
    toward where thinking should help (high gate-think / high no-think entropy)
    via `_selective_position_weights`, instead of uniform-random. The
    fixed-shape contract (exactly ``max_positions`` rows, full ``max_prefix_len``
    window) is preserved so the compile path still specializes once.
    """
    assert int(thinking_token_id) != int(pad_id), "think id must differ from pad"
    B, T = input_ids.shape
    valid = _latent_clean_mask(input_ids, targets,
                               thinking_token_id=thinking_token_id,
                               pad_id=pad_id, eos_id=eos_id)
    bsel, tsel = torch.nonzero(valid, as_tuple=True)
    P = bsel.numel()
    # FIXED-SHAPE for torch.compile: the latent extra forward must have a
    # CONSTANT shape (max_positions, max_prefix_len+1) so the compiled
    # model.forward specializes ONCE instead of recompiling/Inductor-asserting
    # on every new (n, L). So: require >= max_positions valid positions, sample
    # EXACTLY that many, and always use the full max_prefix_len window. Skip
    # (None) when a microbatch has too few valid positions — it's a sampled
    # aux loss, dropping the occasional microbatch costs nothing. (sample_frac
    # is unused in fixed-shape mode; kept for signature compatibility.)
    del sample_frac
    if P < max_positions:
        return None
    gdev = generator.device if generator is not None else bsel.device

    # Snapshot the model's per-step diagnostic stashes (_last_gate*,
    # memory._last_*) NOW — every extra forward below (the selective-weight
    # forward AND the grad/no-think latent forwards) clobbers them with a
    # wrong-shape (max_positions, …) tensor that the main training loop would
    # otherwise index with the MAIN-batch shape. Restored in the `finally`
    # (the documented 2026-05-27 footgun, generalized).
    mem = getattr(model, "memory", None)
    _saved = {a: getattr(model, a, None)
              for a in ("_last_gate", "_last_gate_logits")}
    _saved_mem = ({a: getattr(mem, a, None)
                   for a in ("_last_injection", "_last_write_gate")}
                  if mem is not None else {})
    try:
        if selective:
            # Weighted sampling (without replacement) toward useful positions.
            w = _selective_position_weights(model, input_ids, bsel, tsel)
            perm = torch.multinomial(w.to(gdev), max_positions,
                                     replacement=False,
                                     generator=generator).to(bsel.device)
        else:
            perm = torch.randperm(P, generator=generator,
                                  device=gdev)[:max_positions].to(bsel.device)
        bsel, tsel = bsel[perm], tsel[perm]
        true_next = targets[bsel, tsel]                          # (max_positions,)
        Lmax = int(max_prefix_len)
        rows = []
        for j in range(bsel.numel()):
            b, t = int(bsel[j].item()), int(tsel[j].item())
            pref = input_ids[b, : t + 1][-Lmax:]
            if pref.numel() < Lmax:
                pad = torch.full((Lmax - pref.numel(),), pad_id,
                                 dtype=pref.dtype, device=input_ids.device)
                pref = torch.cat([pad, pref], dim=0)             # LEFT pad
            rows.append(pref)
        prefixes = torch.stack(rows, dim=0)

        logits = _latent_think_logits_grad(model, prefixes, R=R,
                                           thinking_token_id=thinking_token_id)
        loss = F.cross_entropy(logits.float(), true_next)
        with torch.no_grad():
            lpR = F.log_softmax(logits.float(), dim=-1).gather(
                1, true_next.view(-1, 1)).squeeze(1)
            base = _logits_hidden(model(prefixes, return_hidden=True))[0]
            lp0 = F.log_softmax(base[:, -1, :].float(), dim=-1).gather(
                1, true_next.view(-1, 1)).squeeze(1)
            mean_delta = float((lpR - lp0).mean().item())
    finally:
        for a, v in _saved.items():
            setattr(model, a, v)
        for a, v in _saved_mem.items():
            setattr(mem, a, v)
    return loss, mean_delta, int(bsel.numel())


# Keep the variable-shape, return_hidden latent extra-forward OUT of the
# compiled graph: handing `hidden` out of a compiled forward trips
# AOTAutograd's output-alias replay (the gist-loss-inside-forward workaround
# exists for the same reason). dynamo.disable runs this eager while the main
# training forward stays compiled (the bulk of the speedup). The fixed-shape
# sampling above is the additional safety net against recompile storms.
try:  # pragma: no cover - depends on torch build
    latent_cotrain_loss = torch._dynamo.disable(latent_cotrain_loss)  # type: ignore
except Exception:  # pragma: no cover
    pass


def build_replay_batch(
    items: list[ThinkReplay],
    block_size: int,
    pad_token_id: int,
    thinking_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("cannot build an empty replay batch")
    ctx_rows: list[list[int]] = []
    last_positions: list[int] = []
    targets: list[int] = []
    target_is_thinking: list[bool] = []
    for item in items:
        ctx = item.context_ids[-block_size:]
        if not ctx:
            raise ValueError("replay context_ids must be non-empty")
        last_positions.append(len(ctx) - 1)
        targets.append(int(thinking_token_id if item.target_is_thinking else item.target_id))
        target_is_thinking.append(bool(item.target_is_thinking))
        if len(ctx) < block_size:
            ctx = ctx + [int(pad_token_id)] * (block_size - len(ctx))
        ctx_rows.append(ctx)
    return (
        torch.tensor(ctx_rows, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(last_positions, dtype=torch.long, device=device),
        torch.tensor(target_is_thinking, dtype=torch.bool, device=device),
    )
