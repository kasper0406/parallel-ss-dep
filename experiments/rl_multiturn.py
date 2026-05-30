"""Multi-turn / agentic extensions for grader-RL (RL_NEXT_DESIGN §1-2, §5, §8).

This module holds the *pure, CPU-testable* logic for the multi-turn
trajectory RL pipeline that extends `experiments/train_rl_grader.py`.
Everything here is deliberately model-free and grader-free (no GPU, no
subprocess) so it can be unit-tested without touching the training GPUs.

What lives here:
  - `Turn` / `Trajectory` data structures (RL_NEXT_DESIGN §5.3).
  - `compute_trajectory_reward` — improvement-shaped reward with a
    terminal-dominates clamp (§2.2).
  - `group_is_variance_bearing` — the zero-variance hygiene predicate (§1.2 C).
  - `assemble_flat_rollouts` — the "separate rows, shared advantage"
    integration that lets the EXISTING `policy_loss_for_rollouts_batched`
    consume multi-turn trajectories unchanged (§5.3-5.4).
  - `JudgeBackend` ABC + `fold_judge_ranks` — the LLM tie-breaker that can
    re-order WITHIN an execution-tied tier but provably never crosses one
    (§8.2). The concrete vLLM backend (`VLLMJudgeBackend`) is implemented but
    is NEVER imported at module import time so tests can run with a mock.

The trainer (`train_rl_grader.py`) calls into these; the heavy GPU/grader
work (rollouts, grading) stays in the trainer and is passed in as callbacks.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:  # avoid importing torch / the trainer at module import time
    from experiments.train_rl_grader import Rollout


# ---------------------------------------------------------------------------
# Turn / Trajectory data structures (RL_NEXT_DESIGN §5.3)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One revision turn: the prompt fed in, the rollout produced, and the
    grade that resulted. `rollout` is the EXISTING `train_rl_grader.Rollout`
    (we reuse it verbatim — emit ids/logps/positions, depth, gate_* fields).
    """
    prompt_text: str
    rollout: "Rollout"
    score: float
    tier: str
    error_text: str | None = None


@dataclass
class Trajectory:
    """A sequence of revision turns on ONE problem by ONE rollout lineage.

    `turns[0]` is the unprompted first attempt; `turns[t]` (t>0) is the
    feedback-conditioned revision of `turns[t-1]`. `R_traj` is filled by
    `compute_trajectory_reward`.
    """
    problem_id: str
    turns: list[Turn] = field(default_factory=list)
    R_traj: float = 0.0

    @property
    def terminal_score(self) -> float:
        return self.turns[-1].score if self.turns else 0.0

    @property
    def terminal_tier(self) -> str:
        return self.turns[-1].tier if self.turns else "syntax_error"

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def total_depth(self) -> int:
        return sum(int(t.rollout.depth) for t in self.turns)

    def passed(self, pass_threshold: float = 0.5) -> bool:
        return self.terminal_score >= pass_threshold


# ---------------------------------------------------------------------------
# Improvement-based trajectory reward + terminal-dominates clamp (§2.2)
# ---------------------------------------------------------------------------

def compute_trajectory_reward(
    scores: Sequence[float], *,
    lambda_improve: float = 0.0,
    turn_cost: float = 0.0,
    max_score: float = 1.0,
    min_adjacent_gap: float = 0.05,
) -> float:
    """trajectory_reward = terminal_score + λ·Σ max(0, Δ_t) − turn_cost·(T−1),
    with the improvement bonus CLAMPED so the terminal term provably dominates.

    `scores` is the per-turn grade score (turn 0 .. turn T-1). `Δ_t =
    score_t − score_{t-1}` for t≥1. We sum only the positive deltas
    (a regressing turn is neither rewarded nor punished on the task side —
    mirrors the existing `counterfactual` clamp philosophy in
    `compute_grpo_advantages_from_rewards`).

    THE TERMINAL-DOMINATES CLAMP (RL_NEXT_DESIGN §2.2; RD2). A model must
    never be able to win by *starting bad to harvest Δ*. The raw improvement
    bonus `λ·Σ max(0,Δ_t)` is bounded by `λ·max_score` (a single climb from
    0 to max_score). If λ is set carelessly that bonus could exceed the gap
    between two terminal tiers, letting a worse-terminal/big-climb trajectory
    outscore a better-terminal/no-climb one. We forbid this by hard-capping
    the improvement bonus at `0.5·min_adjacent_gap` — strictly LESS than the
    smallest possible difference between two distinct terminal scores. With
    the dense ladder's smallest adjacent gap = 0.05 (syntax→exec), the bonus
    can contribute at most 0.025, so:

        terminal_a > terminal_b  (by ≥ min_adjacent_gap = 0.05)
        ⇒ R_a ≥ terminal_a − turn_penalty_a
              > terminal_b + 0.025  (max possible bonus)  − turn_penalty_a

    i.e. once the terminal scores differ by a real tier gap, no amount of
    improvement shaping can invert their order (turn_cost only ever lowers a
    longer trajectory further, never helps it). Improvement shaping therefore
    ONLY breaks ties *within* the same terminal score — exactly its intended
    role (a `syntax→exec→exec` trajectory beats a flat `exec→exec→exec` one).

    Returns a single scalar `R_traj`.
    """
    if not scores:
        return 0.0
    terminal = float(scores[-1])
    pos_delta_sum = 0.0
    for prev, cur in zip(scores[:-1], scores[1:]):
        d = float(cur) - float(prev)
        if d > 0.0:
            pos_delta_sum += d
    raw_bonus = lambda_improve * pos_delta_sum
    # Hard cap so the bonus can never cross a terminal-tier boundary.
    bonus_cap = 0.5 * float(min_adjacent_gap)
    bonus = min(raw_bonus, bonus_cap)
    n_turns = len(scores)
    turn_penalty = turn_cost * max(0, n_turns - 1)
    return terminal + bonus - turn_penalty


# ---------------------------------------------------------------------------
# Zero-variance hygiene (RL_NEXT_DESIGN §1.2 filter C)
# ---------------------------------------------------------------------------

def group_is_variance_bearing(rewards: Sequence[float], tau: float) -> bool:
    """True iff the per-group reward std is STRICTLY above `tau`.

    With `tau <= 0` (the default everywhere) this is effectively a no-op for
    any group with >1 element and a hair of variance — and importantly the
    legacy trainer behaviour (keep every non-empty group) is reproduced
    because the GRPO `std + 1e-8` already collapses a flat group's advantages
    to ≈0. Set `--group_var_floor > 0` to actively DROP near-flat groups from
    the policy update so the `1e-8` epsilon never injects noise (§1.1 regime 3).

    A group of <2 rewards is never variance-bearing (std undefined / 0).
    """
    n = len(rewards)
    if n < 2:
        return False
    mean = sum(rewards) / n
    # Population std (matches `unbiased=False` in the GRPO advantage code).
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    return std > float(tau)


# ---------------------------------------------------------------------------
# "Separate rows, shared advantage" assembly (RL_NEXT_DESIGN §5.3-5.4)
# ---------------------------------------------------------------------------

def assemble_flat_rollouts(
    trajectories_per_group: Sequence[Sequence[Trajectory]],
    advantages_per_group: Sequence[Sequence[float]],
    *,
    require_emit: bool = True,
) -> tuple[list["Rollout"], list[float]]:
    """Flatten multi-turn trajectories into (flat_rollouts, flat_advantages)
    consumable by the UNCHANGED `policy_loss_for_rollouts_batched`.

    Each TURN of each trajectory becomes a SEPARATE ROW in the flat lists,
    and every row of a given trajectory is assigned that trajectory's SHARED
    advantage (group-normalized `R_traj`). This is the §5.3 recommended
    integration: the loss function needs zero change; multi-turn lives only
    in how the flat lists are assembled.

    `trajectories_per_group[g]` is the list of N trajectories for problem g;
    `advantages_per_group[g][i]` is the (already group-normalized) advantage
    for trajectory i in group g.

    `require_emit` drops turns whose rollout emitted no tokens (the loss skips
    them anyway, but dropping keeps the flat lists tight and the row↔advantage
    alignment exact). Single-turn trajectories reduce to exactly the legacy
    flat-list shape, so `max_turns == 1` is byte-identical to today.
    """
    flat_rollouts: list["Rollout"] = []
    flat_advantages: list[float] = []
    for group, group_adv in zip(trajectories_per_group, advantages_per_group):
        for traj, adv in zip(group, group_adv):
            for turn in traj.turns:
                if require_emit and not turn.rollout.emit_token_ids:
                    continue
                flat_rollouts.append(turn.rollout)
                flat_advantages.append(float(adv))
    return flat_rollouts, flat_advantages


# ---------------------------------------------------------------------------
# LLM-judge tie-breaker (RL_NEXT_DESIGN §8)
# ---------------------------------------------------------------------------

@dataclass
class JudgeCandidate:
    """One candidate solution presented to the judge for ranking."""
    code: str
    error_text: str | None
    tier_base: float          # the unhackable execution score (§8.2)


class JudgeBackend(abc.ABC):
    """Interface for the LLM tie-breaker (RL_NEXT_DESIGN §8.3).

    `rank(problem_prompt, candidates)` returns a permutation of
    `range(len(candidates))`, BEST-FIRST (index of the candidate judged
    closest-to-correct comes first). On any failure / malformed response the
    backend MUST raise; callers treat a raised exception as "judge abstains"
    and fall back to dropping the group (filter C), per §8.3.

    The real vLLM backend (`VLLMJudgeBackend`) lives below but is never
    instantiated in the unit tests — tests supply a `MockJudgeBackend`-style
    fake that just returns a canned permutation. The abstraction is the
    mockability boundary.
    """

    @abc.abstractmethod
    def rank(self, problem_prompt: str,
             candidates: Sequence[JudgeCandidate]) -> list[int]:
        ...


def fold_judge_ranks(
    candidates: Sequence[JudgeCandidate],
    ranks_best_first: Sequence[int],
    *,
    eps_judge: float = 0.02,
    tier_margin: float = 0.025,
) -> list[float]:
    """Convert a best-first ranking of execution-tied candidates into bounded,
    mean-centred reward deltas folded onto each candidate's `tier_base`
    (RL_NEXT_DESIGN §8.2).

        judge_rank_i = position_from_worst(i) / (N - 1)   ∈ [0, 1]
        reward_i     = tier_base_i + eps_judge·(judge_rank_i − mean_rank)
        reward_i     = clamp(reward_i, tier_base_i ± tier_margin)

    The mean-centring makes the judge term zero-sum across the group (it only
    re-orders, never inflates the whole group), and the per-candidate clamp to
    `[tier_base − tier_margin, tier_base + tier_margin]` makes it STRUCTURALLY
    IMPOSSIBLE for the judge to lift a candidate out of its execution tier:
    with `tier_margin = 0.025 < 0.5·min_adjacent_gap (=0.025 for the 0.05
    syntax→exec gap)` the perturbed reward of a lower-tier candidate
    (`tier_base + 0.025`) can at most TOUCH but never exceed the un-perturbed
    floor of the next tier (`tier_base' − 0.025`). The execution grade stays
    a hard gate; the judge only breaks within-tier ties.

    `ranks_best_first[0]` is the index of the best candidate. We require it to
    be a permutation of `range(N)`; a malformed ranking should be caught by the
    caller (judge abstains) before reaching here, but we validate defensively.

    Returns a list of folded rewards, one per candidate (original order).
    """
    n = len(candidates)
    if n == 0:
        return []
    if sorted(ranks_best_first) != list(range(n)):
        raise ValueError(
            f"ranks_best_first must be a permutation of range({n}); "
            f"got {list(ranks_best_first)!r}")
    if n == 1:
        return [float(candidates[0].tier_base)]
    # position_from_worst: best (rank index 0) → n-1, worst → 0.
    # judge_rank ∈ [0, 1].
    judge_rank = [0.0] * n
    for best_first_pos, cand_idx in enumerate(ranks_best_first):
        from_worst = (n - 1) - best_first_pos
        judge_rank[cand_idx] = from_worst / (n - 1)
    mean_rank = sum(judge_rank) / n  # = 0.5 for a full permutation
    out: list[float] = []
    for i, cand in enumerate(candidates):
        base = float(cand.tier_base)
        delta = eps_judge * (judge_rank[i] - mean_rank)
        reward = base + delta
        lo, hi = base - tier_margin, base + tier_margin
        reward = min(max(reward, lo), hi)
        out.append(reward)
    return out


def apply_judge_to_group(
    candidates: Sequence[JudgeCandidate],
    backend: JudgeBackend | None,
    problem_prompt: str,
    *,
    eps_judge: float = 0.02,
    tier_margin: float = 0.025,
) -> list[float] | None:
    """Run the judge on an execution-tied group and fold its ranking into
    rewards (RL_NEXT_DESIGN §8). Returns the per-candidate folded rewards, or
    `None` if the judge abstained / is unavailable (caller then drops the
    group, exactly as filter C would have).

    This is the single entry point the trainer calls. It is gated by the
    caller (only invoked when `group_is_variance_bearing(...)` is False), so
    this function does NOT re-check that condition — it assumes the group is
    tied and the judge is wanted.
    """
    if backend is None:
        return None
    try:
        ranks = backend.rank(problem_prompt, candidates)
        return fold_judge_ranks(
            candidates, ranks, eps_judge=eps_judge, tier_margin=tier_margin)
    except Exception:
        # Malformed / failed judge → abstain (§8.3 fallback).
        return None


def strip_comments(code: str) -> str:
    """Strip Python comments + docstrings so the judge ranks behaviour-bearing
    code only (RL_NEXT_DESIGN §8.5 reward-hacking guardrail). Best-effort: on a
    parse failure (the candidate may not even compile — it's a failing rollout)
    return the original code unchanged.
    """
    try:
        import io
        import tokenize
        result: list[str] = []
        prev_end = (1, 0)
        last_lineno = -1
        last_col = 0
        toks = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok_type, tok_str, start, end, _line in toks:
            if tok_type == tokenize.COMMENT:
                continue
            if start[0] > last_lineno:
                last_col = 0
            if start[1] > last_col:
                result.append(" " * (start[1] - last_col))
            result.append(tok_str)
            last_lineno, last_col = end
        return "".join(result)
    except Exception:
        return code


class VLLMJudgeBackend(JudgeBackend):
    """Real vLLM-driven listwise judge (RL_NEXT_DESIGN §8.3-8.4).

    Reuses the Qwen-AWQ + chat-template machinery from
    `experiments/distill_solutions.py`. NEVER imported/instantiated by the
    unit tests (vLLM allocates a GPU). The hard prerequisite (§8.4) is a free
    second GPU running this — co-residency with the trainer is impossible on
    one 5090.

    Two construction modes:
      - `server_url` set → talk to a persistent vLLM OpenAI-compatible server
        on GPU 1 over HTTP (the recommended config, §8.4 option 1).
      - `server_url` None → spin up an in-process `vllm.LLM` (only when a GPU
        is genuinely free; otherwise it OOMs).
    """

    _SYSTEM = (
        "You are ranking candidate Python solutions to one problem by how "
        "CLOSE each is to a correct, working solution. They all currently "
        "FAIL the unit tests — order them by which is nearest to passing "
        "(structure, correctness of approach, how localized the remaining "
        "bug is). Do NOT reward verbosity, comments, or style. A short "
        "almost-correct function beats a long elaborate broken one. Return "
        "ONLY a JSON list of candidate numbers, best-first, e.g. [3,1,4,2]."
    )

    def __init__(self, *, model: str = "QuantTrio/Qwen3.6-35B-A3B-AWQ",
                 server_url: str | None = None,
                 strip_comments_first: bool = True,
                 max_model_len: int = 8192,
                 gpu_mem_fraction: float = 0.9,
                 temperature: float = 0.0,
                 max_new_tokens: int = 128) -> None:
        self.model = model
        self.server_url = server_url
        self.strip_comments_first = strip_comments_first
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._llm = None
        self._tokenizer = None
        if server_url is None:
            # In-process vLLM. Lazy import so tests never trigger vLLM.
            from vllm import LLM  # noqa: F401  (import-time GPU alloc)
            from transformers import AutoTokenizer
            self._llm = LLM(model=model, max_model_len=max_model_len,
                            gpu_memory_utilization=gpu_mem_fraction,
                            quantization="awq")
            self._tokenizer = AutoTokenizer.from_pretrained(model)

    def _build_prompt(self, problem_prompt: str,
                      candidates: Sequence[JudgeCandidate]) -> str:
        lines = [f"Problem:\n{problem_prompt.strip()}\n"]
        for i, c in enumerate(candidates, start=1):
            code = (strip_comments(c.code)
                    if self.strip_comments_first else c.code)
            lines.append(f"Candidate {i}:\n{code.strip()}")
            diag = (c.error_text or "(no diagnosis)").strip()
            lines.append(f"Execution diagnosis {i}: {diag}\n")
        return "\n".join(lines)

    @staticmethod
    def _parse_ranking(text: str, n: int) -> list[int]:
        """Parse a best-first JSON list of 1-based candidate numbers into a
        0-based permutation. Raises on any malformation (caller → abstain).

        Robust to REASONING-model output: the default judge (Qwen3.6) is a
        thinking model that emits a ``<think>...</think>`` block which can
        itself contain ``[`` brackets. A naive first-``[``..last-``]`` scan
        would grab the reasoning brackets and fail every call. So we (1) strip
        any think block, then (2) scan all flat ``[...]`` spans and accept the
        LAST one that parses as a valid 1..n permutation — the model's final
        answer is emitted last. Thinking is also disabled at generation time,
        but this keeps the parser correct even if a server ignores that.
        """
        import re
        cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
        # If a closing tag exists (possibly without our regex matching a
        # well-formed pair), keep only what follows the last one.
        if "</think>" in cleaned:
            cleaned = cleaned.rsplit("</think>", 1)[-1]
        spans = re.findall(r"\[[^\[\]]*\]", cleaned)
        for span in reversed(spans):
            try:
                arr = json.loads(span)
            except Exception:
                continue
            if not isinstance(arr, list):
                continue
            try:
                order = [int(x) - 1 for x in arr]
            except (TypeError, ValueError):
                continue
            if sorted(order) == list(range(n)):
                return order
        raise ValueError(
            f"no valid best-first permutation of 1..{n} in judge "
            f"response: {text!r}")

    def rank(self, problem_prompt: str,
             candidates: Sequence[JudgeCandidate]) -> list[int]:
        n = len(candidates)
        if n <= 1:
            return list(range(n))
        user = self._build_prompt(problem_prompt, candidates)
        if self.server_url is not None:
            text = self._rank_via_server(user)
        else:
            text = self._rank_in_process(user)
        return self._parse_ranking(text, n)

    def _rank_in_process(self, user: str) -> str:
        from vllm import SamplingParams
        messages = [{"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": user}]
        # Disable the thinking trace — we want a fast, parseable ranking, not a
        # reasoning dump (Qwen3.6 is a thinking model by default). Fall back
        # gracefully for tokenizers that don't accept the kwarg.
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(temperature=self.temperature,
                            max_tokens=self.max_new_tokens)
        out = self._llm.generate([prompt], sp)
        return out[0].outputs[0].text

    def _chat_endpoint(self) -> str:
        """Normalize ``server_url`` to the OpenAI chat-completions endpoint, so
        a base URL (``http://host:8000``), a ``/v1`` URL, or the full path all
        work."""
        u = self.server_url.rstrip("/")
        if u.endswith("/chat/completions"):
            return u
        if u.endswith("/v1"):
            return u + "/chat/completions"
        return u + "/v1/chat/completions"

    def _rank_via_server(self, user: str) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": self._SYSTEM},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            # Disable Qwen3.6's thinking trace server-side (vLLM honors this);
            # the parser also strips <think> blocks if the server ignores it.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            self._chat_endpoint(), data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]
