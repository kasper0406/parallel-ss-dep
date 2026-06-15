"""Verifier-arbitrated best-of-{no-think, think} — the Pareto-safe thinking
mechanism (2026-06-15).

WHY this exists
---------------
The project invariant is "thinking must never be worse than no-think". The
2026-06-15 probe (`/tmp/pareto_probe.py`, summarised in HANDOFF) measured, on
the v13 base, three candidate arbiters for WHEN to adopt a think result:

  case (v13, R=4)        none-acc  always-think  confidence-veto  verifier-best-of
  recall  (dist>=256)      0.000      +0.188        +0.000           +0.188
  code    (next-token)     0.656      -0.521        -0.042           +0.005
  ptr n=2 (depth)          0.333      +0.125        +0.146           +0.188
  ptr n=3 (depth)          0.208      +0.083        +0.104           +0.167

Findings that motivate this module:
  * ALWAYS-THINK is catastrophic on code (-0.52) — the raw downside.
  * The CONFIDENCE-VETO (adopt think iff it is more confident than no-think) is
    NOT strictly-safe: still net-negative on code (-0.042), AND it discards the
    recall upside (+0.00 vs +0.188) because no-think there is CONFIDENTLY WRONG.
    Confidence is not correctness, so no label-free per-token arbiter is
    provably >= no-think in this stack.
  * The only arbiter that is strictly >= no-think on EVERY case (worst +0.005,
    the real code hurt-case) AND keeps the upside is a RELIABLE arbiter — at
    deploy time, a VERIFIER. In the project's target domain (code) the verifier
    EXISTS: `experiments.code_grader.grade`.

GUARANTEE (by construction)
---------------------------
`best_of_think` returns the candidate with the MAX verifier score among
{no-think, think}, tie-broken toward no-think. Therefore the returned score is
`max(score_nothink, score_think) >= score_nothink` for EVERY input — true
per-input Pareto-safety, not an in-expectation claim.

This is the SEQUENCE-LEVEL analog of the probe's `oracle` arm: you do not need
a per-token label, only the sequence-level pass/fail the grader already
provides.

STRUCTURAL FLOOR underneath (independent, also load-bearing)
-----------------------------------------------------------
Even before the arbiter, the think branch is structurally bounded:
  * the no-think forward path is BYTE-IDENTICAL to base — the latent adapter is
    only ever applied at think slots (probe CHECK1: max|Δlogits|=0 when the
    adapter α is perturbed on a no-think forward). If the gate never fires, the
    model is exactly the base model.
  * `state_readonly_at_think` (DeltaNet β=0 at think positions) means a think
    cannot WRITE the recurrent state, so a think can only ever change the ONE
    emit it precedes — it can never corrupt downstream long-range recall.
So even a bad think branch has a bounded blast radius; the verifier arbiter then
removes that bounded downside at the sequence level.

COST
----
~2x generation in the worst case (one no-think + one think gen). With
`skip_think_if_passed=True` the think branch runs ONLY when no-think already
fails the verifier, so amortised extra cost = (no-think fail rate) x (think
gen). The think branch is also the slow one (latent steps), so spend it only on
the hard inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class BestOfResult:
    text: str
    score: float
    used_think: bool          # True iff the THINK candidate was adopted
    think_evaluated: bool     # True iff the think branch was generated+graded
    score_nothink: float
    score_think: float | None  # None when the think branch was skipped


def arbitrate(scores: list[float], prefer: int = 0) -> int:
    """Index of the max score; ties resolved toward `prefer` (the cheap / safe
    default = no-think at index 0). Strictly: the returned index i satisfies
    scores[i] == max(scores), and scores[i] >= scores[prefer]."""
    if not scores:
        raise ValueError("arbitrate needs at least one score")
    best = prefer if 0 <= prefer < len(scores) else 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


def best_of_think(
    gen_nothink: Callable[[], str],
    gen_think: Callable[[], str],
    grade: Callable[[str], float],
    *,
    max_score: float = 1.0,
    skip_think_if_passed: bool = True,
) -> BestOfResult:
    """Verifier-arbitrated best-of-{no-think, think}. Strictly >= no-think.

    Parameters
    ----------
    gen_nothink, gen_think : zero-arg callables that return a candidate string.
        `gen_nothink` MUST be the base no-think decode (e.g.
        `eval_humaneval.generate` with thinking disabled, or
        `generate_latent_think(..., emit_threshold=1.0)` so the gate never
        fires). `gen_think` is the thinking decode (`generate_latent_think`
        with the gate / a forced think budget).
    grade : maps a candidate string to a scalar score in [0, max_score]
        (higher = better). For code use `code_grader.grade(...).score`.
    max_score : the score that means "fully correct" — when no-think already
        reaches it and `skip_think_if_passed`, the think branch is skipped
        (it cannot improve on a perfect score).
    skip_think_if_passed : cost lever (default on). See module docstring.

    Returns a `BestOfResult` whose `.score == max(score_nothink, score_think)`.
    """
    c0 = gen_nothink()
    s0 = float(grade(c0))
    if skip_think_if_passed and s0 >= max_score:
        return BestOfResult(text=c0, score=s0, used_think=False,
                            think_evaluated=False, score_nothink=s0,
                            score_think=None)
    cT = gen_think()
    sT = float(grade(cT))
    idx = arbitrate([s0, sT], prefer=0)   # ties -> no-think
    if idx == 1:
        return BestOfResult(text=cT, score=sT, used_think=True,
                            think_evaluated=True, score_nothink=s0,
                            score_think=sT)
    return BestOfResult(text=c0, score=s0, used_think=False,
                        think_evaluated=True, score_nothink=s0, score_think=sT)
