"""CPU-only tests for the multi-turn agentic RL extensions (RL_NEXT_DESIGN).

ALL tests here use a fake model (None) + a mock grader (a dict-driven callable)
and never touch a GPU. Covers:
  - single-turn (max_turns=1) byte-identity: advantages match the pre-change
    single-shot computation;
  - multi-turn assembly: K turns → K rows with the correct shared advantage;
  - improvement reward math + the terminal-dominates clamp ("start-bad-to-
    harvest-Δ" can't outscore a clean pass);
  - variance-bearing predicate + group_var_floor filtering;
  - LLM-judge folding: a mock judge re-orders WITHIN a tied tier but provably
    never crosses a tier boundary.

Run ONLY this file (the full suite has CUDA tests that would OOM a co-resident
training run):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \\
        experiments/test_rl_multiturn.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from experiments.code_grader import Problem, _TIER_BASE_SCORE, _compute_score
from experiments.rl_multiturn import (
    JudgeBackend,
    JudgeCandidate,
    Trajectory,
    Turn,
    apply_judge_to_group,
    assemble_flat_rollouts,
    compute_trajectory_reward,
    fold_judge_ranks,
    group_is_variance_bearing,
    strip_comments,
)
from experiments.train_rl_grader import (
    Rollout,
    compute_grpo_advantages_from_rewards,
    run_trajectories_for_group,
)


# ---------------------------------------------------------------------------
# Fakes / mocks (no GPU, no model, no subprocess)
# ---------------------------------------------------------------------------

def _make_rollout(emit_ids=(1, 2, 3), depth=0) -> Rollout:
    """A minimal Rollout with the fields the assembly/loss read."""
    emit_ids = list(emit_ids)
    return Rollout(
        prompt_len=2,
        emit_token_ids=emit_ids,
        emit_log_probs=[-0.1] * len(emit_ids),
        emit_positions=list(range(2, 2 + len(emit_ids))),
        full_ids=[5, 6] + emit_ids,
        depth=depth,
        text="def f(): pass",
    )


class _MockGrade:
    def __init__(self, score, tier, error_text=None, n_passed=0, n_tests=1):
        self.score = score
        self.tier = tier
        self.error_text = error_text
        self.n_passed = n_passed
        self.n_tests = n_tests
        self.passed = (tier == "pass")


class _ScriptedGrader:
    """Returns a scripted sequence of grades per (turn-index) call.

    `script` is a list of _MockGrade, consumed in call order. Used to drive a
    deterministic trajectory: turn 0 grade, turn 1 grade, ...
    """
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, problem, code):
        self.calls.append(code)
        return self.script.pop(0)


def _problem(pid="t/1"):
    return Problem(task_id=pid, prompt="# add a and b\n",
                   tests="def check(candidate):\n    assert candidate(1,2)==3\n",
                   entry_point="f", prompt_is_code=False)


# ---------------------------------------------------------------------------
# 1. Single-turn (max_turns=1) byte-identity
# ---------------------------------------------------------------------------

def test_single_turn_advantages_match_legacy_computation():
    """max_turns=1 trajectory reward == terminal score, and the assembled
    advantages match the legacy single-shot GRPO computation exactly."""
    scores = [0.0, 0.2, 1.0, 0.05]
    # Trajectory reward with default (pure-terminal) settings == score.
    for s in scores:
        assert compute_trajectory_reward([s]) == s

    # Build a 1-turn trajectory per score, group-normalize, and compare to the
    # legacy path fed the same scalar rewards.
    trajs = []
    for s in scores:
        t = Trajectory(problem_id="t/1")
        t.turns.append(Turn(prompt_text="p", rollout=_make_rollout(),
                            score=s, tier="x"))
        t.R_traj = compute_trajectory_reward([s])
        trajs.append(t)
    R = torch.tensor([[t.R_traj for t in trajs]])
    D = torch.tensor([[0.0] * len(trajs)])
    adv = compute_grpo_advantages_from_rewards(
        R, D, ponder_cost=0.0, counterfactual=True, ponder_warmup_scale=1.0)[0]

    legacy = compute_grpo_advantages_from_rewards(
        torch.tensor([scores]), D, ponder_cost=0.0,
        counterfactual=True, ponder_warmup_scale=1.0)[0]
    assert torch.allclose(adv, legacy, atol=1e-7)

    # And assembly produces exactly one row per trajectory with that advantage.
    flat_r, flat_a = assemble_flat_rollouts([trajs], [adv.tolist()])
    assert len(flat_r) == len(scores)
    assert flat_a == pytest.approx(adv.tolist())


def test_run_trajectories_single_turn_is_one_turn():
    grader = _ScriptedGrader([_MockGrade(0.2, "partial")] * 4)
    rollouts = [_make_rollout() for _ in range(4)]
    seq = iter(rollouts)

    def rollout_fn(prompt, n):
        return [next(seq) for _ in range(n)]

    trajs = run_trajectories_for_group(
        _problem(), "# p\n",
        rollout_fn=rollout_fn, grade_fn=grader,
        extract_fn=lambda r: r.text,
        n_rollouts=4, max_turns=1, carry_history=False)
    assert len(trajs) == 4
    assert all(t.n_turns == 1 for t in trajs)
    # Exactly 4 grade calls (no repair).
    assert len(grader.calls) == 4


# ---------------------------------------------------------------------------
# 2. Multi-turn assembly: K turns → K rows w/ shared advantage
# ---------------------------------------------------------------------------

def test_multiturn_assembly_shared_advantage_per_turn():
    # One group, 2 trajectories. Traj A has 3 turns, traj B has 1 turn.
    tA = Trajectory(problem_id="t/1")
    for s in (0.0, 0.05, 0.2):
        tA.turns.append(Turn("p", _make_rollout(), s, "x"))
    tB = Trajectory(problem_id="t/1")
    tB.turns.append(Turn("p", _make_rollout(), 1.0, "pass"))
    # Hand-set advantages: A → -1.0, B → +1.0.
    flat_r, flat_a = assemble_flat_rollouts([[tA, tB]], [[-1.0, 1.0]])
    # 3 rows from A + 1 row from B = 4 rows.
    assert len(flat_r) == 4
    # First 3 rows share A's advantage; last row is B's.
    assert flat_a == [-1.0, -1.0, -1.0, 1.0]


def test_multiturn_run_stops_on_pass_and_revises_failures():
    # Turn-0: rollout0 passes (stop), rollout1 fails → repair on turn 1.
    grader = _ScriptedGrader([
        _MockGrade(1.0, "pass"),                 # r0 turn0: solved → stop
        _MockGrade(0.0, "syntax_error", "SyntaxError: bad"),  # r1 turn0
        _MockGrade(0.2, "partial"),              # r1 turn1: improved (not pass)
        _MockGrade(1.0, "pass"),                 # r1 turn2: solved → stop
    ])
    pool = [_make_rollout() for _ in range(8)]
    seq = iter(pool)

    def rollout_fn(prompt, n):
        return [next(seq) for _ in range(n)]

    trajs = run_trajectories_for_group(
        _problem(), "# p\n",
        rollout_fn=rollout_fn, grade_fn=grader,
        extract_fn=lambda r: r.text,
        n_rollouts=2, max_turns=3, carry_history=False)
    assert len(trajs) == 2
    # r0 solved on turn 0 → only 1 turn (early-stop).
    assert trajs[0].n_turns == 1 and trajs[0].passed()
    # r1 failed (syntax) → revised to partial (turn1) → solved (turn2) = 3 turns.
    assert trajs[1].n_turns == 3
    assert trajs[1].turns[0].tier == "syntax_error"
    assert trajs[1].passed()


# ---------------------------------------------------------------------------
# 3. Improvement reward + terminal-dominates clamp
# ---------------------------------------------------------------------------

def test_improvement_bonus_breaks_within_tier_ties():
    # Two trajectories both terminate at exec_error (0.05). One climbed
    # syntax(0.0)→exec(0.05); the other was flat exec→exec. The climber
    # should score higher (positive Δ bonus).
    climber = compute_trajectory_reward([0.0, 0.05], lambda_improve=0.3)
    flat = compute_trajectory_reward([0.05, 0.05], lambda_improve=0.3)
    assert climber > flat
    # But the bonus is tiny (≤ 0.025 cap), so both stay within the tier band.
    assert climber - 0.05 <= 0.025 + 1e-9


def test_terminal_dominates_clamp_start_bad_cannot_outscore_clean_pass():
    """A 'start bad to harvest Δ' trajectory (syntax→partial→pass, big climb)
    must NOT outscore a clean immediate pass — the terminal must dominate."""
    # Aggressive λ to try to break it.
    harvester = compute_trajectory_reward(
        [0.0, 0.2, 1.0], lambda_improve=100.0, turn_cost=0.0)
    clean_pass = compute_trajectory_reward([1.0], lambda_improve=100.0)
    # Both terminate at 1.0; the harvester's bonus is clamped so it can at most
    # TIE within the same terminal tier, never exceed by a tier gap.
    assert harvester <= clean_pass + 0.025 + 1e-9
    # And a worse terminal (partial 0.55) with a giant climb still loses to a
    # clean pass by ~a full tier, no matter how large λ is.
    worse_terminal = compute_trajectory_reward(
        [0.0, 0.05, 0.55], lambda_improve=1000.0)
    assert worse_terminal < clean_pass
    # The gap is at least the real tier gap minus the bonus cap (the bonus can
    # close the gap by at most 0.025, never invert it).
    assert clean_pass - worse_terminal >= (1.0 - 0.55) - 0.025 - 1e-9


def test_turn_cost_penalizes_longer_trajectories():
    short = compute_trajectory_reward([0.2], turn_cost=0.02)
    long_ = compute_trajectory_reward([0.2, 0.2, 0.2], turn_cost=0.02)
    assert short > long_
    assert math.isclose(short - long_, 0.04, abs_tol=1e-9)


def test_regressing_turn_not_punished_on_task_side():
    # A turn that regresses (0.2 → 0.0) contributes no positive Δ, but is not
    # punished (max(0, Δ) clamp). Reward == terminal (minus turn cost).
    r = compute_trajectory_reward([0.2, 0.0], lambda_improve=0.3, turn_cost=0.0)
    assert r == 0.0  # terminal score, no negative penalty added


# ---------------------------------------------------------------------------
# 4. Variance-bearing predicate + group_var_floor
# ---------------------------------------------------------------------------

def test_variance_bearing_predicate():
    # Flat group dropped; varied group kept.
    assert not group_is_variance_bearing([0.0, 0.0, 0.0, 0.0], tau=1e-3)
    assert not group_is_variance_bearing([0.05, 0.05], tau=1e-3)
    assert group_is_variance_bearing([0.0, 1.0], tau=1e-3)
    assert group_is_variance_bearing([0.0, 0.05, 0.2, 1.0], tau=1e-3)
    # Single-element / empty groups are never variance-bearing.
    assert not group_is_variance_bearing([0.5], tau=0.0)
    assert not group_is_variance_bearing([], tau=0.0)


def test_variance_floor_just_below_real_spread_keeps_group():
    # A tiny but real spread (std ≈ 0.025) is kept at floor 1e-3, dropped at
    # floor 0.1.
    g = [0.2, 0.25]
    assert group_is_variance_bearing(g, tau=1e-3)
    assert not group_is_variance_bearing(g, tau=0.1)


# ---------------------------------------------------------------------------
# 5. LLM-judge folding: re-order within tier, never cross a tier
# ---------------------------------------------------------------------------

class _MockJudge(JudgeBackend):
    """A mock judge that returns a fixed best-first ranking, ignoring inputs.

    The real VLLMJudgeBackend is NEVER imported/instantiated in these tests —
    this fake exercises the folding contract only.
    """
    def __init__(self, ranking):
        self.ranking = list(ranking)
        self.calls = 0

    def rank(self, problem_prompt, candidates):
        self.calls += 1
        assert len(self.ranking) == len(candidates)
        return list(self.ranking)


def test_judge_folding_reorders_within_a_tied_tier():
    # All 4 candidates are syntax_error (tier_base 0.0). The judge ranks
    # candidate 2 best, candidate 0 worst.
    base = _TIER_BASE_SCORE["syntax_error"]
    cands = [JudgeCandidate(code=f"c{i}", error_text="SyntaxError",
                            tier_base=base) for i in range(4)]
    ranks = [2, 1, 3, 0]  # best-first
    folded = fold_judge_ranks(cands, ranks, eps_judge=0.02, tier_margin=0.025)
    # The best-ranked candidate (idx 2) gets the highest folded reward; the
    # worst (idx 0) the lowest.
    assert folded[2] == max(folded)
    assert folded[0] == min(folded)
    # Mean-centred → sum ≈ 4·base (zero-sum perturbation).
    assert math.isclose(sum(folded), 4 * base, abs_tol=1e-9)
    # The group regained variance.
    assert group_is_variance_bearing(folded, tau=1e-6)


def test_judge_cannot_cross_a_tier_boundary_adversarial():
    """Adversarial: a mixed-tier group with the judge ranking trying to flip
    the order. The folded reward must NEVER lift a lower-tier candidate above
    a higher-tier one — sorted order by tier_base is preserved."""
    # Candidates across the binding-constraint tiers: syntax(0.0) and exec(0.05),
    # the smallest adjacent gap.
    tiers = ["syntax_error", "exec_error", "syntax_error", "exec_error"]
    cands = [JudgeCandidate(code=f"c{i}", error_text="e",
                            tier_base=_TIER_BASE_SCORE[t])
             for i, t in enumerate(tiers)]
    # Adversarial judge: rank the WORST-tier candidates best to try to flip.
    # idx0 (syntax) best, idx1 (exec) worst.
    ranks = [0, 2, 3, 1]
    folded = fold_judge_ranks(cands, ranks, eps_judge=0.02, tier_margin=0.025)
    # Every exec candidate must still be >= every syntax candidate.
    syntax_rewards = [folded[i] for i, t in enumerate(tiers)
                      if t == "syntax_error"]
    exec_rewards = [folded[i] for i, t in enumerate(tiers)
                    if t == "exec_error"]
    assert min(exec_rewards) >= max(syntax_rewards)
    # And each folded reward stays inside its own tier band.
    for i, t in enumerate(tiers):
        base = _TIER_BASE_SCORE[t]
        assert base - 0.025 - 1e-9 <= folded[i] <= base + 0.025 + 1e-9


def test_judge_cannot_cross_tier_property_random():
    """Property test: random tiers + random (adversarial) judge ranks →
    sorted order by tier_base is always preserved within the bound."""
    import random
    rng = random.Random(1234)
    tier_names = list(_TIER_BASE_SCORE.keys())
    for _ in range(200):
        n = rng.randint(2, 6)
        chosen = [rng.choice(tier_names) for _ in range(n)]
        cands = [JudgeCandidate(code="x", error_text=None,
                                tier_base=_compute_score(t, 4, rng.randint(0, 4)))
                 for t in chosen]
        ranks = list(range(n))
        rng.shuffle(ranks)
        folded = fold_judge_ranks(cands, ranks, eps_judge=0.02,
                                  tier_margin=0.025)
        # Pair-wise: if tier_base_i differs from tier_base_j by >= the smallest
        # adjacent gap (0.05), the folded order must agree with tier_base order.
        for i in range(n):
            for j in range(n):
                if cands[i].tier_base - cands[j].tier_base >= 0.05 - 1e-9:
                    assert folded[i] >= folded[j], (
                        f"judge crossed a tier: base_i={cands[i].tier_base} "
                        f"base_j={cands[j].tier_base} "
                        f"folded_i={folded[i]} folded_j={folded[j]}")


def test_apply_judge_to_group_abstains_on_malformed_ranking():
    class _BadJudge(JudgeBackend):
        def rank(self, problem_prompt, candidates):
            return [0, 0, 1, 2]  # not a permutation → fold raises
    cands = [JudgeCandidate("c", None, 0.0) for _ in range(4)]
    out = apply_judge_to_group(cands, _BadJudge(), "p")
    assert out is None  # abstain → caller drops the group (filter C)


def test_apply_judge_to_group_none_backend_returns_none():
    cands = [JudgeCandidate("c", None, 0.0) for _ in range(4)]
    assert apply_judge_to_group(cands, None, "p") is None


def test_judge_fires_only_conceptually_on_tied_group():
    # The gating is the caller's job (only call apply_judge_to_group when the
    # group is NOT variance-bearing). Here we assert the predicate that gates
    # it: a variance-bearing group returns True (judge would be skipped).
    varied = [0.0, 0.05, 1.0]
    tied = [0.0, 0.0, 0.0]
    assert group_is_variance_bearing(varied, tau=1e-3) is True
    assert group_is_variance_bearing(tied, tau=1e-3) is False


# ---------------------------------------------------------------------------
# Misc: comment stripping guardrail (§8.5)
# ---------------------------------------------------------------------------

def test_strip_comments_removes_comments_keeps_code():
    code = "def f(x):  # add one\n    # a comment\n    return x + 1\n"
    out = strip_comments(code)
    assert "# add one" not in out
    assert "# a comment" not in out
    assert "return x + 1" in out


def test_strip_comments_survives_unparseable_code():
    # A failing rollout may be syntactically broken — strip must not raise.
    broken = "def f(x)\n    return x +"
    out = strip_comments(broken)
    assert isinstance(out, str)
