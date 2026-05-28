"""Tests for the D18 rework of train_rl_grader.py:

  - `--state_readonly_at_think` flag (forces DeltaNet beta=0 at think
    positions via build_model_from_ckpt(force_state_readonly=True), on
    both the policy and the frozen KL reference).
  - `--dataset_jsonl PATH` (loads a synth_reasoning-schema JSONL via
    code_grader.load_synth_reasoning instead of the LOADERS registry).

Both must be backwards-compatible: off / None leaves the trainer
unchanged. The parser is built inline in `main()`, so we reconstruct
the relevant argument definitions and assert the wiring in source —
plus a functional check that load_synth_reasoning round-trips the
arithmetic ladder JSONL and that build_model_from_ckpt exposes the
force_state_readonly override the flag relies on.
"""
import inspect
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _src(rel):
    return (ROOT / rel).read_text()


def test_module_imports():
    import experiments.train_rl_grader as trg  # noqa: F401


def test_new_cli_flags_present_in_parser():
    src = _src("experiments/train_rl_grader.py")
    assert '"--state_readonly_at_think"' in src
    assert '"--dataset_jsonl"' in src


def test_state_readonly_flag_reaches_build_model():
    """The flag must be translated into force_state_readonly=True and
    passed to BOTH build_model_from_ckpt calls (policy + KL reference)."""
    src = _src("experiments/train_rl_grader.py")
    assert "args.state_readonly_at_think" in src
    # the translated sentinel
    assert "force_state_readonly=_force_sr" in src
    # exactly two build_model_from_ckpt calls receive the override
    assert src.count("force_state_readonly=_force_sr") == 2


def test_state_readonly_recorded_in_saved_cfg():
    """When on, cfg must be tagged so the reloaded ckpt auto-enables the
    hook (build_model_from_ckpt reads cfg['state_readonly_at_think'])."""
    src = _src("experiments/train_rl_grader.py")
    assert 'cfg["state_readonly_at_think"] = True' in src


def test_dataset_jsonl_uses_load_synth_reasoning():
    src = _src("experiments/train_rl_grader.py")
    assert "args.dataset_jsonl is not None" in src
    assert "load_synth_reasoning(path=args.dataset_jsonl)" in src
    # backwards-compat: the LOADERS path is still present in the else
    assert "LOADERS[args.dataset]()" in src


def test_build_model_from_ckpt_accepts_force_state_readonly():
    from experiments.eval_bracket_structure import build_model_from_ckpt
    sig = inspect.signature(build_model_from_ckpt)
    assert "force_state_readonly" in sig.parameters


def test_load_synth_reasoning_roundtrips_arith_ladder():
    from experiments.code_grader import load_synth_reasoning
    path = ROOT / "data" / "synth_arith_ladder_n2.jsonl"
    if not path.exists():
        pytest.skip("arith ladder data not present")
    probs = load_synth_reasoning(path=str(path))
    assert len(probs) > 0
    p0 = probs[0]
    assert p0.entry_point == "solve"
    assert p0.prompt_is_code is False
    assert "candidate()" in p0.tests
    assert p0.gold_solution is not None


def test_force_sr_sentinel_is_none_when_flag_off():
    """_force_sr must be None (not False) when the flag is off so the
    auto-detect path in build_model_from_ckpt is preserved (None =
    don't override)."""
    src = _src("experiments/train_rl_grader.py")
    assert "_force_sr = True if args.state_readonly_at_think else None" in src
