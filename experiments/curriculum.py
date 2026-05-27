"""Difficulty-weighted curriculum for GRPO problem sampling.

Per-problem pass-rate exponential moving average. Three sampling modes:

1. Variance-weighted (default): weight = 4·p·(1-p) + eps. Peaks at p=0.5
   (max GRPO advantage variance), down-weights saturated p=0 and p=1
   problems equally. Time-invariant.

2. Progressive (opt-in, open-loop): weight = exp(-(p - target(step))²
   / 2σ²) + eps. target_p decays linearly from target_start (0.7,
   "easy") to target_end (0.2, "hard") over total_steps. Fixed schedule,
   no feedback from model performance.

3. Adaptive (opt-in, closed-loop): same Gaussian weighting but
   target_p = max(target_floor, 1.0 - mean_p_over_seen), where mean_p
   is the running EMA average. When model is weak (mean_p low), target
   is high (sample easy problems where success is possible). As model
   strengthens (mean_p rises), target lowers toward target_floor (keeping
   us in the productive variance zone). Parameter-free except for the
   floor. Tracks the capability frontier automatically.

The EMA is a stable rolling estimate of the model's unprompted pass rate
on each problem; iterative repair rollouts MUST NOT update it (they
would inflate the estimate).
"""
from __future__ import annotations

import math


class ProblemDifficultyEMA:
    def __init__(
        self,
        problem_ids: list[str],
        alpha: float = 0.1,
        init_pass_rate: float = 0.25,
        eps: float = 0.05,
        progressive: bool = False,
        target_start: float = 0.7,
        target_end: float = 0.2,
        target_sigma: float = 0.15,
        total_steps: int | None = None,
        adaptive: bool = False,
        adaptive_floor: float = 0.3,
    ):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.init_pass_rate = float(init_pass_rate)
        self.progressive = bool(progressive)
        self.adaptive = bool(adaptive)
        self.target_start = float(target_start)
        self.target_end = float(target_end)
        self.target_sigma = float(target_sigma)
        self.adaptive_floor = float(adaptive_floor)
        self.total_steps = (None if total_steps is None
                            else max(1, int(total_steps)))
        if self.progressive and self.total_steps is None:
            raise ValueError("progressive=True requires total_steps")
        if self.progressive and self.adaptive:
            raise ValueError("progressive and adaptive are mutually exclusive")
        self.ema: dict[str, float] = {
            str(pid): float(init_pass_rate) for pid in problem_ids
        }
        self.seen: set[str] = set()

    def _ensure(self, problem_id: str) -> None:
        pid = str(problem_id)
        if pid not in self.ema:
            self.ema[pid] = self.init_pass_rate

    def update(self, problem_id: str, rewards: list[float]) -> None:
        if not rewards:
            return
        pid = str(problem_id)
        self._ensure(pid)
        n_pass = sum(1 for r in rewards if float(r) >= 0.5)
        pass_rate = n_pass / len(rewards)
        self.ema[pid] = (1.0 - self.alpha) * self.ema[pid] + self.alpha * pass_rate
        self.seen.add(pid)

    def _mean_p_seen(self) -> float:
        if not self.seen:
            return self.init_pass_rate
        vals = [self.ema[pid] for pid in self.seen]
        return sum(vals) / len(vals)

    def target_at(self, step: int) -> float:
        if self.adaptive:
            return max(self.adaptive_floor, 1.0 - self._mean_p_seen())
        if not self.progressive:
            return float("nan")
        frac = min(1.0, max(0.0, step / self.total_steps))
        return self.target_start + (self.target_end - self.target_start) * frac

    def sampling_weight(self, problem_id: str, step: int | None = None) -> float:
        pid = str(problem_id)
        p = self.ema.get(pid, self.init_pass_rate)
        if self.adaptive or (self.progressive and step is not None):
            target = self.target_at(step or 0)
            return math.exp(-((p - target) ** 2) /
                            (2.0 * self.target_sigma ** 2)) + self.eps
        return 4.0 * p * (1.0 - p) + self.eps

    def sampling_weights(self, problem_ids: list[str],
                         step: int | None = None) -> list[float]:
        return [self.sampling_weight(pid, step) for pid in problem_ids]

    def stats(self, step: int | None = None) -> dict:
        n_seen = len(self.seen)
        out = {"n_seen": n_seen,
               "mean_p": float("nan"),
               "pct_in_band": float("nan")}
        if n_seen > 0:
            seen_vals = [self.ema[pid] for pid in self.seen]
            out["mean_p"] = sum(seen_vals) / len(seen_vals)
            out["pct_in_band"] = sum(
                1 for p in seen_vals if 0.1 <= p <= 0.9) / n_seen
        if self.adaptive or (self.progressive and step is not None):
            out["target_p"] = self.target_at(step or 0)
        return out

    def state_dict(self) -> dict:
        return {
            "ema": dict(self.ema),
            "alpha": self.alpha,
            "eps": self.eps,
            "init_pass_rate": self.init_pass_rate,
            "seen": sorted(self.seen),
            "progressive": self.progressive,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "target_sigma": self.target_sigma,
            "total_steps": self.total_steps,
            "adaptive": self.adaptive,
            "adaptive_floor": self.adaptive_floor,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema = {str(k): float(v) for k, v in state["ema"].items()}
        self.alpha = float(state["alpha"])
        self.eps = float(state["eps"])
        self.init_pass_rate = float(state["init_pass_rate"])
        self.seen = set(str(s) for s in state.get("seen", []))
        if "progressive" in state:
            self.progressive = bool(state["progressive"])
            self.target_start = float(state["target_start"])
            self.target_end = float(state["target_end"])
            self.target_sigma = float(state["target_sigma"])
            self.total_steps = state["total_steps"]
        if "adaptive" in state:
            self.adaptive = bool(state["adaptive"])
            self.adaptive_floor = float(state["adaptive_floor"])


def merge_rank_updates(
    rank_updates: list[list[tuple[str, list[float]]]],
) -> list[tuple[str, list[float]]]:
    """Merge per-rank (problem_id, rewards) update lists into a single
    list where each unique problem_id appears once with all rewards
    concatenated. Used to give every DDP rank an identical view of the
    step's rewards before applying EMA updates locally.
    """
    merged: dict[str, list[float]] = {}
    for rank in rank_updates:
        for pid, rewards in rank:
            merged.setdefault(str(pid), []).extend(float(r) for r in rewards)
    return [(pid, merged[pid]) for pid in sorted(merged.keys())]
