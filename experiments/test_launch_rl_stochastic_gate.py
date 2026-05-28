"""Arg-parse / config correctness test for launch_rl_stochastic_gate.sh.

Guards the PLAN_FLAW_C primary RL launcher: every --flag it passes must be a
real train_rl_grader.py argument, and the load-bearing values (the validated
v2 stability recipe + the new stochastic-gate policy knobs) must match the
plan. This catches a typo'd flag or a silently-dropped knob before a multi-
hour GPU run.
"""
from __future__ import annotations

import pathlib
import re
import shlex

REPO = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "launch_rl_stochastic_gate.sh"
TRAINER = REPO / "experiments" / "train_rl_grader.py"


def _launcher_flags() -> dict[str, str | bool]:
    """Parse the `--flag value` pairs from the launcher's python invocation.

    Returns flag -> value (True for store_true / boolean flags with no
    following value). Shell variable substitutions like "${STEPS}" are kept
    verbatim — we only check the flag NAMES against the parser for those.
    """
    text = LAUNCHER.read_text()
    # Grab the line-continued python invocation block.
    m = re.search(r"experiments/train_rl_grader\.py(.*?)>\s*runs/", text,
                  re.DOTALL)
    assert m, "could not find the train_rl_grader.py invocation"
    block = m.group(1).replace("\\\n", " ")
    tokens = shlex.split(block)
    flags: dict[str, str | bool] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            name = t
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[name] = tokens[i + 1]
                i += 2
            else:
                flags[name] = True
                i += 1
        else:
            i += 1
    return flags


def _trainer_known_flags() -> set[str]:
    """Scrape all add_argument flag strings from the trainer source."""
    src = TRAINER.read_text()
    return set(re.findall(r"p\.add_argument\(\s*\"(--[a-zA-Z0-9_]+)\"", src))


def test_all_launcher_flags_are_real_trainer_args():
    flags = _launcher_flags()
    known = _trainer_known_flags()
    unknown = sorted(set(flags) - known)
    assert not unknown, f"launcher passes unknown flags: {unknown}"


def test_stochastic_gate_policy_knobs_present():
    flags = _launcher_flags()
    # The whole point of this launcher: gate as a policy variable, exploring
    # only the uncertain band.
    assert flags.get("--stochastic_gate") is True
    assert flags["--gate_sample_range_low"] == "0.1"
    assert flags["--gate_sample_range_high"] == "0.9"
    # Entropy curriculum (anti-collapse, anneal exploration).
    assert flags["--gate_entropy_bonus_start"] == "0.03"
    assert flags["--gate_entropy_bonus_end"] == "0.0"
    assert flags["--gate_entropy_curriculum_steps"] == "200"


def test_v2_stability_recipe_preserved():
    flags = _launcher_flags()
    # The validated monotonic-climb recipe must be carried verbatim.
    assert flags["--kl_coef"] == "0.05"
    assert flags["--lr"] == "2e-6"
    assert flags["--clip_eps"] == "0.1"
    assert flags["--temperature"] == "0.7"
    # v2's hard-won lesson: NO depth pressure on the first terminal-signal test.
    assert flags["--ponder_cost"] == "0.0"


def test_gate_floor_below_emit_threshold_pinned_rule():
    """PINNED footgun: gate_floor >= emit_threshold silently makes the gate
    never-think. The launcher must keep gate_floor < emit_threshold."""
    flags = _launcher_flags()
    assert float(flags["--gate_floor"]) < float(flags["--emit_threshold"])


def test_base_ckpt_is_sft_not_v2_step300():
    """PLAN_FLAW_C: the PRIMARY run starts from the SFT base (non-degenerate
    gate prior), not v2-step300 (gate shaped by never-think RL)."""
    text = LAUNCHER.read_text()
    m = re.search(r"BASE=\$\{BASE:-(\S+)\}", text)
    assert m, "launcher must default BASE"
    assert m.group(1) == "checkpoints/sft_phase_c_combined.pt"
