"""
Mid-training eval driver: spawn HumanEval (and optionally MBPP) as a
subprocess against a checkpoint, parse the pass-rate, and feed an
`EvalStopController` that decides whether the pretrain has plateaued.

Used by `experiments/train_lm.py` when `--auto_stop` is set: every
`--mid_eval_every_tokens` tokens, save a temp ckpt, call `run_eval`,
log to TB, append to the controller, and break the train loop if
`controller.should_stop()` returns True.

Subprocess isolation matters because eval loads its own copy of the
model + datasets — running it in-process would fragment GPU memory
right when training is already memory-pressured.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
import subprocess
import sys


_PASS_LINE_RE = re.compile(r"pass@\d+\s*=\s*([\d.]+)")


@dataclasses.dataclass
class EvalResult:
    humaneval_pass_rate: float
    mbpp_pass_rate: float | None
    tokens_seen: int
    step: int
    ckpt_path: str
    raw_log_tail: str  # last few lines of stdout/stderr for debugging


def _parse_pass_rate(stdout: str) -> float | None:
    """Pull the pass rate out of `eval_humaneval.py`'s output. Looks for the
    `pass@k = <rate>` line and returns the float, or None if not found."""
    for line in reversed(stdout.splitlines()):
        m = _PASS_LINE_RE.search(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def run_humaneval(ckpt_path: str, n_problems: int = 50,
                   max_gen: int = 192,
                   use_thinking: bool = False,
                   emit_threshold: float = 0.5,
                   max_think_per_step: int = 8,
                   min_emit_before_eos: int = 0,
                   gate_floor: float = 0.0,
                   timeout_s: int = 1800,
                   python_executable: str | None = None,
                   ) -> tuple[float | None, str]:
    """Shell out to `experiments/eval_humaneval.py` and return
    `(pass_rate, log_tail)`. `pass_rate` is None if parsing failed.

    `timeout_s` is generous (30 min) because greedy HumanEval on a 50-problem
    subset at 217 M takes ~3-8 min usually but a slow load can push it.
    """
    py = python_executable or sys.executable
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cmd = [
        py, str(repo_root / "experiments" / "eval_humaneval.py"),
        "--ckpt", ckpt_path,
        "--max_problems", str(n_problems),
        "--max_gen", str(max_gen),
    ]
    if use_thinking:
        cmd += [
            "--use_thinking",
            "--emit_threshold", str(emit_threshold),
            "--max_think_per_step", str(max_think_per_step),
        ]
    if min_emit_before_eos > 0:
        cmd += ["--min_emit_before_eos", str(min_emit_before_eos)]
    if gate_floor > 0.0:
        cmd += ["--gate_floor", str(gate_floor)]
    env = None  # inherit; PYTHONPATH will already include repo if launcher set it
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as e:
        tail = (e.stdout or "")[-2000:] + (e.stderr or "")[-2000:]
        return None, f"<TIMEOUT after {timeout_s}s>\n{tail}"
    log = (proc.stdout or "") + (proc.stderr or "")
    rate = _parse_pass_rate(log)
    return rate, log[-2000:]


def run_eval(ckpt_path: str, tokens_seen: int, step: int,
              n_problems: int = 50, max_gen: int = 192,
              use_thinking: bool = False,
              emit_threshold: float = 0.5,
              min_emit_before_eos: int = 0,
              gate_floor: float = 0.0,
              ) -> EvalResult:
    """Convenience wrapper: HumanEval only for now (MBPP path can be added
    once `eval_mbpp.py` exists in this repo)."""
    he_rate, log_tail = run_humaneval(
        ckpt_path, n_problems=n_problems, max_gen=max_gen,
        use_thinking=use_thinking, emit_threshold=emit_threshold,
        min_emit_before_eos=min_emit_before_eos,
        gate_floor=gate_floor,
    )
    return EvalResult(
        humaneval_pass_rate=float(he_rate) if he_rate is not None else float("nan"),
        mbpp_pass_rate=None,
        tokens_seen=tokens_seen,
        step=step,
        ckpt_path=ckpt_path,
        raw_log_tail=log_tail,
    )


class EvalStopController:
    """Tracks (tokens, pass_rate) and decides when to stop.

    Stop rule: the previous `k_consecutive_flat` intervals each gained less
    than `stop_threshold` HumanEval pass-rate. We require *consecutive* flat
    intervals to avoid stopping on a single noisy eval.

    We treat the *first* eval as a free anchor (always logged, never
    triggers stop).
    """

    def __init__(self, stop_threshold: float = 0.01,
                 k_consecutive_flat: int = 2):
        if k_consecutive_flat < 1:
            raise ValueError("k_consecutive_flat must be >= 1")
        self.stop_threshold = float(stop_threshold)
        self.k_consecutive_flat = int(k_consecutive_flat)
        self.history: list[EvalResult] = []

    def append(self, result: EvalResult) -> None:
        self.history.append(result)

    def should_stop(self) -> bool:
        # Need at least k+1 evals to have k intervals.
        if len(self.history) < self.k_consecutive_flat + 1:
            return False
        rates = [r.humaneval_pass_rate for r in self.history]
        # NaN-safe: if any of the last k+1 is NaN, don't stop (assume
        # something went wrong, keep training and surface it).
        tail = rates[-(self.k_consecutive_flat + 1):]
        if any(r != r for r in tail):  # NaN check
            return False
        for i in range(1, len(tail)):
            delta = tail[i] - tail[i - 1]
            if delta >= self.stop_threshold:
                return False
        return True

    def latest_delta(self) -> float | None:
        if len(self.history) < 2:
            return None
        return (self.history[-1].humaneval_pass_rate
                - self.history[-2].humaneval_pass_rate)

    def summary_line(self) -> str:
        if not self.history:
            return "<no evals>"
        h = self.history[-1]
        d = self.latest_delta()
        d_str = f"Δ={d:+.3f}" if d is not None else "Δ=NA"
        return (f"tokens={h.tokens_seen:,}  step={h.step}  "
                f"humaneval={h.humaneval_pass_rate:.3f}  {d_str}")
