from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


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
