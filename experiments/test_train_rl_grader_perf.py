"""Perf + correctness tests for the RL-grader speedups (2026-05-24).

Fix 1: `grade_in_parallel` wraps `code_grader.grade` in a
ThreadPoolExecutor. Per-thread `grade()` spawns a subprocess, so the
GIL is irrelevant — threads give real CPU parallelism for the exec
phase. Tests:
  * Result-order correctness: parallel results match sequential results
    exactly when given identical inputs.
  * Speedup: with a mock 100 ms `grade()`, parallel-N is at least
    Nx faster than sequential-N for N rollouts (bounded by worker count).

Run with:
    PYTHONPATH=. .venv/bin/python -m pytest \
        experiments/test_train_rl_grader_perf.py -v -s
"""
from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from experiments.train_rl_grader import grade_in_parallel


@dataclass
class _FakeGradingResult:
    score: float
    tier: str
    n_passed: int
    n_tests: int
    error_text: str | None


@dataclass
class _FakeProblem:
    task_id: str


def _slow_grade(problem, completion, timeout_s=5):
    time.sleep(0.1)
    # Encode `problem.task_id` into the score so we can verify the
    # parallel scheduler hands the right result back to the right input.
    pid = int(problem.task_id.split("_")[-1])
    return _FakeGradingResult(
        score=float(pid),
        tier="pass" if pid % 2 == 0 else "exec_error",
        n_passed=pid,
        n_tests=10,
        error_text=None if pid % 2 == 0 else f"err_{pid}",
    )


def test_grade_in_parallel_preserves_order():
    """The parallel grader must return results in the same order as the
    input jobs, regardless of which subprocess finishes first."""
    with mock.patch("experiments.train_rl_grader.grade", side_effect=_slow_grade):
        jobs = [(_FakeProblem(f"p_{i}"), f"code_{i}") for i in range(8)]
        # Sequential reference.
        seq = [_slow_grade(p, c) for p, c in jobs]
        par = grade_in_parallel(jobs, timeout_s=5, max_workers=8)
        assert len(seq) == len(par)
        for s, p in zip(seq, par):
            assert s.score == p.score
            assert s.tier == p.tier
            assert s.n_passed == p.n_passed
            assert s.error_text == p.error_text


def test_grade_in_parallel_is_faster_than_sequential():
    """With a 100 ms-per-call mock grader, parallel-8 should be
    substantially faster than sequential-8 (>4x is the load-bearing claim;
    8x is the theoretical best). Loose threshold for CI jitter."""
    n = 8
    with mock.patch("experiments.train_rl_grader.grade", side_effect=_slow_grade):
        jobs = [(_FakeProblem(f"p_{i}"), f"code_{i}") for i in range(n)]
        t0 = time.perf_counter()
        seq = grade_in_parallel(jobs, timeout_s=5, max_workers=1)
        t_seq = time.perf_counter() - t0
        t0 = time.perf_counter()
        par = grade_in_parallel(jobs, timeout_s=5, max_workers=8)
        t_par = time.perf_counter() - t0
    speedup = t_seq / max(t_par, 1e-6)
    print(f"\n  sequential: {t_seq*1000:.0f} ms  "
          f"parallel(8): {t_par*1000:.0f} ms  "
          f"speedup: {speedup:.1f}x")
    assert speedup >= 4.0, (
        f"expected >=4x speedup; got {speedup:.1f}x "
        f"(seq={t_seq*1000:.0f} ms par={t_par*1000:.0f} ms)")
    # Results must STILL match.
    for s, p in zip(seq, par):
        assert s.score == p.score
        assert s.tier == p.tier


def test_grade_in_parallel_handles_empty():
    """Empty input list must not crash and must return []."""
    assert grade_in_parallel([], timeout_s=5, max_workers=4) == []


def test_grade_in_parallel_handles_single_job():
    """One-job batches should short-circuit without spinning up the pool."""
    with mock.patch("experiments.train_rl_grader.grade", side_effect=_slow_grade):
        out = grade_in_parallel(
            [(_FakeProblem("p_3"), "code")], timeout_s=5, max_workers=8)
    assert len(out) == 1
    assert out[0].score == 3.0


def test_grade_in_parallel_serial_mode():
    """max_workers=1 must fall through to the sequential path (same code
    that already shipped) and produce identical results."""
    with mock.patch("experiments.train_rl_grader.grade", side_effect=_slow_grade):
        jobs = [(_FakeProblem(f"p_{i}"), f"code_{i}") for i in range(4)]
        seq = grade_in_parallel(jobs, timeout_s=5, max_workers=1)
        par = grade_in_parallel(jobs, timeout_s=5, max_workers=4)
    for s, p in zip(seq, par):
        assert s.score == p.score and s.tier == p.tier
