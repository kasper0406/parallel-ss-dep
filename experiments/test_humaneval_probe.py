"""
Tests for the in-process HumanEval probe.

Covers:
- Dataset builder produces correct schema + size (uses the on-disk file if
  available, otherwise skips to avoid a network hit).
- Probe runner correctly identifies pass / fail / syntax-error generations
  (the model object is mocked; we monkey-patch ``generate`` and use a real
  tokenizer to round-trip text → ids → text).
- model.training is restored after the probe runs.

Run:  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_humaneval_probe.py -v
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments import probe_humaneval as probe_mod


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

PROBE_PATH = pathlib.Path("data/probe_humaneval_50.jsonl")


class _DummyTokenizer:
    """Round-trips bytes via ord/chr — keeps the test self-contained."""

    eos_token_id = 0

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class _DummyModel(nn.Module):
    """A torch Module so .eval() / .train() / .parameters() all work."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.max_T = 4096

    def forward(self, x):  # not called: we monkey-patch generate
        raise RuntimeError("dummy forward should not run")


@pytest.fixture
def dummy_model():
    return _DummyModel()


@pytest.fixture
def dummy_tok():
    return _DummyTokenizer()


def _make_fake_generate(continuation_text: str):
    """Returns a callable that mimics ``generate`` by appending an
    encoded ``continuation_text`` to whatever prompt_ids it receives."""

    def fake_generate(model, prompt_ids, max_gen=256, temperature=0.0,
                      eos_token_id=None):
        cont = [b for b in continuation_text.encode("utf-8")][:max_gen]
        cont_t = torch.tensor(cont, dtype=prompt_ids.dtype,
                              device=prompt_ids.device).unsqueeze(0)
        return torch.cat([prompt_ids, cont_t], dim=1)
    return fake_generate


# ---------------------------------------------------------------------------
# Tests: dataset builder
# ---------------------------------------------------------------------------

def test_probe_dataset_exists_and_has_50_rows():
    if not PROBE_PATH.exists():
        pytest.skip("probe JSONL missing; run experiments/build_probe_dataset.py")
    rows = [json.loads(l) for l in PROBE_PATH.read_text().splitlines() if l.strip()]
    assert len(rows) == 50
    for r in rows:
        assert set(r.keys()) >= {"task_id", "prompt", "test", "entry_point"}
        assert isinstance(r["prompt"], str) and r["prompt"]
        assert isinstance(r["test"], str) and "def check" in r["test"]
        assert r["task_id"].startswith("HumanEval/")


# ---------------------------------------------------------------------------
# Tests: probe runner — pass / fail / syntax error
# ---------------------------------------------------------------------------

_TINY_PROBE_PAYLOAD = [
    {
        "task_id": "Tiny/0",
        "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1, 2) == 3\n"
            "    assert candidate(0, 0) == 0\n"
        ),
        "entry_point": "add",
    },
]


def _write_tiny_probe(tmp_path: pathlib.Path) -> str:
    p = tmp_path / "tiny_probe.jsonl"
    with open(p, "w") as f:
        for r in _TINY_PROBE_PAYLOAD:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_probe_recognises_passing_solution(tmp_path, monkeypatch,
                                            dummy_model, dummy_tok):
    probe_file = _write_tiny_probe(tmp_path)
    good = "    return a + b\n"
    monkeypatch.setattr(probe_mod, "generate", _make_fake_generate(good))
    res = probe_mod.run_humaneval_probe(
        dummy_model, dummy_tok, probe_path=probe_file, max_gen=64, timeout_s=5,
    )
    assert res["n_total"] == 1
    assert res["n_passed"] == 1
    assert res["pass_rate"] == 1.0
    assert res["mean_emit_tokens"] > 0


def test_probe_recognises_failing_solution(tmp_path, monkeypatch,
                                            dummy_model, dummy_tok):
    probe_file = _write_tiny_probe(tmp_path)
    wrong = "    return a - b\n"
    monkeypatch.setattr(probe_mod, "generate", _make_fake_generate(wrong))
    res = probe_mod.run_humaneval_probe(
        dummy_model, dummy_tok, probe_path=probe_file, max_gen=64, timeout_s=5,
    )
    assert res["n_passed"] == 0
    assert res["pass_rate"] == 0.0


def test_probe_tolerates_syntax_error(tmp_path, monkeypatch,
                                       dummy_model, dummy_tok):
    probe_file = _write_tiny_probe(tmp_path)
    junk = "    return ((((\n"
    monkeypatch.setattr(probe_mod, "generate", _make_fake_generate(junk))
    res = probe_mod.run_humaneval_probe(
        dummy_model, dummy_tok, probe_path=probe_file, max_gen=64, timeout_s=5,
    )
    assert res["n_passed"] == 0  # graded fail, not crash
    assert res["n_total"] == 1


# ---------------------------------------------------------------------------
# Tests: state-restore
# ---------------------------------------------------------------------------

def test_probe_restores_training_mode_true(tmp_path, monkeypatch,
                                            dummy_model, dummy_tok):
    probe_file = _write_tiny_probe(tmp_path)
    monkeypatch.setattr(probe_mod, "generate", _make_fake_generate("    return a + b\n"))
    dummy_model.train()
    assert dummy_model.training is True
    probe_mod.run_humaneval_probe(
        dummy_model, dummy_tok, probe_path=probe_file, max_gen=32,
    )
    assert dummy_model.training is True, "probe must restore train() mode"


def test_probe_restores_training_mode_false(tmp_path, monkeypatch,
                                             dummy_model, dummy_tok):
    probe_file = _write_tiny_probe(tmp_path)
    monkeypatch.setattr(probe_mod, "generate", _make_fake_generate("    return a + b\n"))
    dummy_model.eval()
    assert dummy_model.training is False
    probe_mod.run_humaneval_probe(
        dummy_model, dummy_tok, probe_path=probe_file, max_gen=32,
    )
    assert dummy_model.training is False, "probe must leave eval() mode alone"


def test_probe_empty_file(tmp_path, dummy_model, dummy_tok):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    res = probe_mod.run_humaneval_probe(
        dummy_model, dummy_tok, probe_path=str(empty),
    )
    assert res["n_total"] == 0
    assert res["pass_rate"] == 0.0
