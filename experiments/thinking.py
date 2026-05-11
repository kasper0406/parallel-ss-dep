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
    rag_history: list[int] | None = None # Indices of retrieved facts


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
    rag_keys: torch.Tensor | None = None,
) -> list[list[ThoughtTrajectory]]:
    """Sample n_group trajectories per (context, target) pair."""
    B = len(initial_contexts)
    N = n_group
    
    # Initialize trajectories: [B][N]
    trajectories: list[list[ThoughtTrajectory]] = [
        [ThoughtTrajectory(initial_context=ctx, target_id=tid, actions=[], action_logprobs=[], rag_history=[])
         for _ in range(N)]
        for ctx, tid in zip(initial_contexts, target_ids)
    ]
    
    # Active trajectories: flat list of (B*N)
    active_trajs: list[ThoughtTrajectory] = [t for group in trajectories for t in group]
    finished_mask = torch.zeros(B * N, dtype=torch.bool, device=device)
    
    # Track the embedding to inject in the NEXT pass for each trajectory
    # Shape: (BN, d_model)
    d_model = model.d_model if hasattr(model, "d_model") else 576
    next_injection = torch.zeros((B * N, d_model), device=device)
    
    for d in range(max_depth + 1):
        if finished_mask.all():
            break
            
        ctx_list = []
        for i, t in enumerate(active_trajs):
            ctx = (t.initial_context + ([thinking_token_id] * t.depth))[-block_size:]
            ctx_list.append(ctx)
            
        # Pad contexts for batching
        max_len = max(len(c) for c in ctx_list)
        input_ids = torch.full((B * N, max_len), 0, dtype=torch.long, device=device)
        for i, c in enumerate(ctx_list):
            input_ids[i, -len(c):] = torch.tensor(c, dtype=torch.long, device=device)
        
        # Build rag_hidden: inject at the last position only
        rag_hidden = torch.zeros((B * N, max_len, d_model), device=device)
        rag_hidden[:, -1, :] = next_injection
        
        # Forward pass
        logits, gate = model(input_ids, return_gate=True, rag_hidden=rag_hidden)
        
        # We only care about the last position
        last_logits = logits[:, -1]
        last_gate = gate[:, -1]
        
        # Perform retrieval for the NEXT step
        if rag_keys is not None:
            # Query is the hidden state at the last position
            _, hidden = model(input_ids, return_hidden=True, rag_hidden=rag_hidden)
            query = hidden[:, -1, :]
            query = F.normalize(query, p=2, dim=1)
            
            # Simple MatMul search
            scores = torch.matmul(query, rag_keys.T) # (BN, N_chunks)
            best_idx = scores.argmax(dim=-1)
            next_injection = rag_keys[best_idx]
            
            # Record history
            for i, t in enumerate(active_trajs):
                if not finished_mask[i]:
                    t.rag_history.append(int(best_idx[i].item()))
        
        # Action probability: gate = prob(Emit)
        # So prob(Think) = 1 - gate
        probs = torch.stack([1 - last_gate, last_gate], dim=-1) # [BN, 2]
        
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample() # 0: Think, 1: Emit
        log_probs = dist.log_prob(actions)
        
        for i, t in enumerate(active_trajs):
            if finished_mask[i]:
                continue
            
            action = int(actions[i].item())
            t.actions.append(1 - action) # Store 1 for Think, 0 for Emit
            t.action_logprobs.append(float(log_probs[i].item()))
            
            if action == 1 or d == max_depth: # Emit or reached max depth
                finished_mask[i] = True
                # Mask thinking token from logits before storing
                t.final_logits = mask_token_logit(last_logits[i], thinking_token_id).cpu()
                t.depth = d
            else:
                t.depth += 1
                
    return trajectories


def compute_grpo_advantages(
    trajectory_groups: list[list[ThoughtTrajectory]],
    ponder_cost: float,
) -> torch.Tensor:
    """Calculate GRPO advantages normalized within each group.
    
    Returns a tensor of shape [B, N].
    """
    B = len(trajectory_groups)
    N = len(trajectory_groups[0])
    rewards = torch.zeros((B, N))
    
    for i, group in enumerate(trajectory_groups):
        for j, t in enumerate(group):
            # Task Reward: -CE(logits, target)
            # final_logits shape: [V]
            target = torch.tensor(t.target_id)
            logits = t.final_logits.unsqueeze(0) # [1, V]
            ce = F.cross_entropy(logits, target.unsqueeze(0))
            task_reward = -ce.item()
            
            # Ponder Penalty
            ponder_reward = -ponder_cost * t.depth
            
            rewards[i, j] = task_reward + ponder_reward
            
    # Normalize within each group
    group_means = rewards.mean(dim=1, keepdim=True)
    group_stds = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = (rewards - group_means) / group_stds
    
    return advantages


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
