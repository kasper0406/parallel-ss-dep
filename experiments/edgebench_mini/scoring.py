"""EdgeBench-mini trajectory scoring.

Turns per-episode trajectories (produced by `harness.run_episode(...).to_dict()`)
into the ByteDance-EdgeBench-style metric family, sized for small-model dev use:

  score@budget curve   best milestone fraction reachable within a cost budget,
                       in generated-token AND tool-call units (two cost axes).
  AUC                  normalized area under score@budget (default over
                       LOG-budget — the interaction-time weighting EdgeBench's
                       log-sigmoid law encodes), a single scalar in [0,1].
  cost-normalized      best score per 1k generated tokens — where an O(1)-decode
                       model is meant to *win* rather than trail on a point score.
  bootstrap CIs        over TASKS, so two checkpoints can be ordered with a
                       stated uncertainty (the whole point: greedy HumanEval-164
                       could not separate a base < SFT < RL ladder).

Everything here is a PURE function of trajectory dicts + Python/numpy — no
torch, no subprocess, no filesystem — so the math is unit-testable against
hand-computed references without a GPU.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

# Default cost grids (log-spaced) for the score@budget curves / AUC.
DEFAULT_TOKEN_BUDGETS: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)
DEFAULT_CALL_BUDGETS: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 18, 24)

TOKENS_KEY = "gen_tokens_cumulative"
CALLS_KEY = "tool_calls_cumulative"


# --------------------------------------------------------------------------- #
# Per-trajectory primitives
# --------------------------------------------------------------------------- #

def best_score_under_budget(points: Sequence[dict], budget: float,
                            cost_key: str) -> float:
    """Best milestone fraction reached at any point whose cumulative cost is
    <= `budget`. 0.0 if no point fits (budget below the first action's cost)."""
    best = 0.0
    seen = False
    for p in points:
        if p[cost_key] <= budget:
            seen = True
            if p["milestone_frac"] > best:
                best = p["milestone_frac"]
    return best if seen else 0.0


def score_curve(points: Sequence[dict], budgets: Sequence[float],
                cost_key: str) -> list[float]:
    return [best_score_under_budget(points, b, cost_key) for b in budgets]


def auc_normalized(points: Sequence[dict], budgets: Sequence[float],
                   cost_key: str, log_x: bool = True) -> float:
    """Trapezoidal area under the score@budget curve vs budget, normalized by
    the budget span so the result is in [0, 1]. With `log_x` the integral is
    over log(budget) (the interaction-time weighting). Requires >= 2 budgets."""
    b = list(budgets)
    if len(b) < 2:
        raise ValueError("auc_normalized needs at least 2 budgets")
    xs = [math.log(v) for v in b] if log_x else [float(v) for v in b]
    ys = score_curve(points, b, cost_key)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    span = xs[-1] - xs[0]
    if span <= 0:
        return float(ys[0])
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return area / span


def task_summary(traj: dict,
                 token_budgets: Sequence[int] = DEFAULT_TOKEN_BUDGETS,
                 call_budgets: Sequence[int] = DEFAULT_CALL_BUDGETS) -> dict:
    """Scalar metrics for one task trajectory dict (from EpisodeResult.to_dict)."""
    points = traj["points"]
    fracs = [p["milestone_frac"] for p in points]
    best = max(fracs) if fracs else 0.0
    final = fracs[-1] if fracs else 0.0
    gen_total = points[-1][TOKENS_KEY] if points else 0
    calls_total = points[-1][CALLS_KEY] if points else 0
    score_per_1k = (best / (gen_total / 1000.0)) if gen_total > 0 else 0.0
    return {
        "task_id": traj.get("task_id"),
        "bucket": traj.get("bucket"),
        "n_milestones": traj.get("n_milestones"),
        "best_score": best,
        "final_score": final,
        "gen_tokens_total": gen_total,
        "tool_calls_total": calls_total,
        "auc_tokens": auc_normalized(points, token_budgets, TOKENS_KEY),
        "auc_calls": auc_normalized(points, call_budgets, CALLS_KEY),
        "score_per_1k_tokens": score_per_1k,
        "finished_reason": traj.get("finished_reason"),
    }


# --------------------------------------------------------------------------- #
# Bootstrap CIs over tasks
# --------------------------------------------------------------------------- #

def _stat(values: Sequence[float], statistic: str | Callable) -> float:
    if callable(statistic):
        return float(statistic(values))
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    raise ValueError(f"unknown statistic {statistic!r}")


def bootstrap_ci(values: Sequence[float], statistic: str | Callable = "mean",
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI of `statistic` over `values` (the per-task
    metric). Deterministic given `seed`. Degenerate inputs are handled:
    empty -> all None; length-1 or constant -> a zero-width CI at the value."""
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return {"point": None, "lo": None, "hi": None, "n": 0}
    point = _stat(vals, statistic)
    if vals.size == 1:
        return {"point": point, "lo": point, "hi": point, "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = np.array([_stat(vals[row], statistic) for row in idx])
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {"point": point, "lo": lo, "hi": hi, "n": int(vals.size)}


def bootstrap_diff_ci(values_a: Sequence[float], values_b: Sequence[float],
                      statistic: str | Callable = "mean", n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 0) -> dict:
    """PAIRED bootstrap CI of stat(a) - stat(b): resample task INDICES once per
    replicate and apply to both arms (so it accounts for per-task correlation —
    a stronger separation test than comparing two independent CIs). `a` and `b`
    must be aligned per task and equal length. `excludes_zero` True => the two
    checkpoints are separated at this alpha."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.size != b.size or a.size == 0:
        raise ValueError("paired diff needs equal-length, non-empty arms")
    point = _stat(a, statistic) - _stat(b, statistic)
    if a.size == 1:
        return {"point": point, "lo": point, "hi": point,
                "excludes_zero": bool(point != 0.0), "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.array([_stat(a[row], statistic) - _stat(b[row], statistic)
                     for row in idx])
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {"point": point, "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0), "n": int(a.size)}


def suite_summary(task_summaries: Sequence[dict],
                  metric_keys: Sequence[str] = (
                      "best_score", "final_score", "auc_tokens", "auc_calls",
                      "score_per_1k_tokens", "gen_tokens_total"),
                  n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                  by_bucket: bool = True) -> dict:
    """Aggregate per-task summaries into suite-level means + bootstrap CIs,
    optionally broken down by difficulty bucket."""
    out: dict = {"n_tasks": len(task_summaries), "metrics": {}}
    for key in metric_keys:
        vals = [s[key] for s in task_summaries if s.get(key) is not None]
        out["metrics"][key] = bootstrap_ci(vals, "mean", n_boot, alpha, seed)
    if by_bucket:
        out["by_bucket"] = {}
        for bucket in sorted({s.get("bucket") for s in task_summaries}):
            subset = [s for s in task_summaries if s.get("bucket") == bucket]
            out["by_bucket"][bucket] = {
                "n": len(subset),
                "best_score": bootstrap_ci([s["best_score"] for s in subset],
                                           "mean", n_boot, alpha, seed),
                "auc_tokens": bootstrap_ci([s["auc_tokens"] for s in subset],
                                           "mean", n_boot, alpha, seed),
            }
    return out


# --------------------------------------------------------------------------- #
# Log-sigmoid interaction-time law (optional summary; AUC is the robust headline)
# --------------------------------------------------------------------------- #

def fit_log_sigmoid(budgets: Sequence[float], scores: Sequence[float]) -> dict:
    """Fit s(b) ~= L * sigmoid(k * (log b - x0)) to a score@budget curve (the
    EdgeBench log-sigmoid interaction-time law) via a deterministic coarse grid
    over (k, x0) with L = max(scores). No scipy. Returns params + rmse, or a
    degenerate fit if the curve is flat. This is a descriptive summary only."""
    b = np.asarray(list(budgets), dtype=float)
    y = np.asarray(list(scores), dtype=float)
    if b.size < 2 or float(np.ptp(y)) < 1e-9:
        return {"L": float(np.max(y)) if y.size else 0.0, "k": 0.0,
                "x0": float(np.mean(np.log(b))) if b.size else 0.0,
                "rmse": 0.0, "degenerate": True}
    logb = np.log(b)
    L = float(max(np.max(y), 1e-6))
    best = None
    for k in np.linspace(0.5, 8.0, 40):
        for x0 in np.linspace(logb.min(), logb.max(), 40):
            pred = L / (1.0 + np.exp(-k * (logb - x0)))
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            if best is None or rmse < best["rmse"]:
                best = {"L": L, "k": float(k), "x0": float(x0), "rmse": rmse,
                        "degenerate": False}
    return best


# --------------------------------------------------------------------------- #
# Checkpoint comparison — the headline artifact
# --------------------------------------------------------------------------- #

def compare_checkpoints(named_summaries: dict[str, Sequence[dict]],
                        headline_metric: str = "auc_tokens",
                        n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> dict:
    """Compare N checkpoints on `headline_metric`.

    `named_summaries`: {ckpt_name: [task_summary, ...]} — the per-task summaries
    (from `task_summary`) for each checkpoint, ONE PER TASK, in the SAME task
    order across checkpoints (so the paired diff bootstrap is valid).

    Returns per-checkpoint CIs, a descending ranking, and BOTH separation
    signals for every ordered pair: `ci_disjoint` (the requested
    non-overlapping-CI test) and `paired_diff_excludes_zero` (a stronger,
    per-task-correlation-aware test)."""
    names = list(named_summaries)
    per_ckpt = {}
    values = {}
    for name in names:
        vals = [s[headline_metric] for s in named_summaries[name]]
        values[name] = vals
        per_ckpt[name] = bootstrap_ci(vals, "mean", n_boot, alpha, seed)

    ranking = sorted(names, key=lambda n: per_ckpt[n]["point"], reverse=True)

    pairs = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            hi_name, lo_name = ranking[i], ranking[j]
            ci_hi, ci_lo = per_ckpt[hi_name], per_ckpt[lo_name]
            ci_disjoint = bool(ci_hi["lo"] > ci_lo["hi"])
            diff = None
            if len(values[hi_name]) == len(values[lo_name]):
                diff = bootstrap_diff_ci(values[hi_name], values[lo_name],
                                         "mean", n_boot, alpha, seed)
            pairs.append({
                "higher": hi_name, "lower": lo_name,
                "delta": per_ckpt[hi_name]["point"] - per_ckpt[lo_name]["point"],
                "ci_disjoint": ci_disjoint,
                "paired_diff_excludes_zero": (diff["excludes_zero"]
                                              if diff else None),
                "paired_diff_ci": diff,
            })
    return {
        "headline_metric": headline_metric, "alpha": alpha,
        "per_ckpt": per_ckpt, "ranking": ranking, "pairs": pairs,
    }


def check_monotonic_separation(comparison: dict,
                               expected_order: Sequence[str]) -> dict:
    """Given an EXPECTED best->worst ordering (e.g. ["rl", "sft", "base"]), test
    the acceptance gate: does the observed ranking match, AND is every ADJACENT
    expected pair separated by non-overlapping CIs? Returns the verdict used by
    `validate_discrimination.py`."""
    ranking = comparison["ranking"]
    order_matches = list(ranking) == list(expected_order)
    pair_lookup = {(p["higher"], p["lower"]): p for p in comparison["pairs"]}
    adjacent = []
    all_sep = True
    for a, b in zip(expected_order, list(expected_order)[1:]):
        p = pair_lookup.get((a, b))
        sep = bool(p and p["ci_disjoint"])
        diff_sep = bool(p and p.get("paired_diff_excludes_zero"))
        adjacent.append({"higher": a, "lower": b, "ci_disjoint": sep,
                         "paired_diff_excludes_zero": diff_sep})
        all_sep = all_sep and sep
    return {
        "expected_order": list(expected_order),
        "observed_ranking": list(ranking),
        "order_matches": order_matches,
        "adjacent_pairs": adjacent,
        "all_adjacent_ci_disjoint": all_sep,
        "PASS": bool(order_matches and all_sep),
    }


def render_comparison_table(comparison: dict) -> str:
    """Human-readable table for a checkpoint comparison (printed by the
    validation/eval drivers)."""
    metric = comparison["headline_metric"]
    lines = [f"=== checkpoint comparison on {metric} "
             f"(mean +/- {int((1 - comparison['alpha']) * 100)}% CI) ==="]
    hdr = f"{'rank':>4}  {'ckpt':<28} {'point':>8}  {'CI_lo':>8}  {'CI_hi':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r, name in enumerate(comparison["ranking"], 1):
        ci = comparison["per_ckpt"][name]
        lines.append(f"{r:>4}  {name:<28} {ci['point']:>8.4f}  "
                     f"{ci['lo']:>8.4f}  {ci['hi']:>8.4f}")
    lines.append("")
    lines.append("pairwise separation (higher vs lower):")
    for p in comparison["pairs"]:
        mark = "SEP " if p["ci_disjoint"] else "over"
        dz = p["paired_diff_excludes_zero"]
        dz_s = "yes" if dz else ("no" if dz is not None else "n/a")
        lines.append(f"  [{mark}] {p['higher']:<20} > {p['lower']:<20} "
                     f"delta={p['delta']:+.4f}  diff!=0:{dz_s}")
    return "\n".join(lines)
