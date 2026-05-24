"""Difficulty-weighted curriculum for GRPO problem sampling.

Per-problem pass-rate exponential moving average. Used to weight problem
sampling toward the variance-bearing zone (pass-rate ~0.5) where a GRPO
group has the most advantage signal. Saturated problems (p≈0 or p≈1)
produce zero-variance groups and contribute no gradient.

The EMA is a stable rolling estimate of the model's unprompted pass rate
on each problem; iterative repair rollouts MUST NOT update it (they
would inflate the estimate).
"""
from __future__ import annotations


class ProblemDifficultyEMA:
    def __init__(self, problem_ids: list[str], alpha: float = 0.1,
                 init_pass_rate: float = 0.25, eps: float = 0.05):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.init_pass_rate = float(init_pass_rate)
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

    def sampling_weight(self, problem_id: str) -> float:
        pid = str(problem_id)
        p = self.ema.get(pid, self.init_pass_rate)
        return 4.0 * p * (1.0 - p) + self.eps

    def sampling_weights(self, problem_ids: list[str]) -> list[float]:
        return [self.sampling_weight(pid) for pid in problem_ids]

    def stats(self) -> dict:
        n_seen = len(self.seen)
        if n_seen == 0:
            return {"n_seen": 0, "mean_p": float("nan"), "pct_in_band": float("nan")}
        seen_vals = [self.ema[pid] for pid in self.seen]
        mean_p = sum(seen_vals) / len(seen_vals)
        in_band = sum(1 for p in seen_vals if 0.1 <= p <= 0.9)
        return {
            "n_seen": n_seen,
            "mean_p": mean_p,
            "pct_in_band": in_band / n_seen,
        }

    def state_dict(self) -> dict:
        return {
            "ema": dict(self.ema),
            "alpha": self.alpha,
            "eps": self.eps,
            "init_pass_rate": self.init_pass_rate,
            "seen": sorted(self.seen),
        }

    def load_state_dict(self, state: dict) -> None:
        self.ema = {str(k): float(v) for k, v in state["ema"].items()}
        self.alpha = float(state["alpha"])
        self.eps = float(state["eps"])
        self.init_pass_rate = float(state["init_pass_rate"])
        self.seen = set(str(s) for s in state.get("seen", []))


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
