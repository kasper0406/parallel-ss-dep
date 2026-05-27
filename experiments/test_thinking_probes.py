from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments import (
    probe_thinking_counterfactual as probe2,
    probe_thinking_per_position_ce as probe1,
    probe_thinking_rl_correlation as probe3,
)


THINK_ID = 7   # arbitrary token id used as thinking_token_id in stub models


class _DummyTokenizer:
    """Round-trips bytes via ord/chr; THINK_ID is reserved."""

    eos_token_id = 0

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [b for b in text.encode("utf-8") if b not in (0, THINK_ID)]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        clean = [int(i) % 256 for i in ids if int(i) not in (0, THINK_ID)]
        return bytes(clean).decode("utf-8", errors="replace")


class _StubGateModel(nn.Module):
    """A model with a learnable side-effect _last_gate. forward returns
    logits of shape (B, T, V); also sets _last_gate of shape (B, T)."""

    def __init__(self, vocab_size: int = 256,
                 gate_value: float = 0.2,
                 next_token: int = ord("X"),
                 ce_with_think: float = 0.1,
                 ce_without_think: float = 1.5):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.max_T = 4096
        self.thinking_token_id = THINK_ID
        self.gate_value = float(gate_value)
        self.next_token = int(next_token)
        self.ce_with_think = float(ce_with_think)
        self.ce_without_think = float(ce_without_think)
        self._last_gate = None

    def _logits_for(self, ids: torch.Tensor) -> torch.Tensor:
        # Deterministic stub: at every position, predict the FIXED token
        # `self.next_token`. When the LAST input is THINK, boost it
        # strongly (low CE). Otherwise boost it weakly (higher CE).
        # The probe's target_id MUST be `self.next_token` for this to
        # produce a positive delta — tests engineer their prompts so
        # that the byte after the position-of-interest is `next_token`.
        B, T = ids.shape
        logits = torch.full(
            (B, T, self.vocab_size), -10.0, dtype=torch.float32,
            device=ids.device)
        boost_high = 10.0     # confident, near-zero CE
        boost_low = 0.0       # diffuse, higher CE (still favors target)
        for t in range(T):
            is_think_input = (ids[:, t] == THINK_ID)   # (B,)
            for b in range(B):
                lv = boost_high if bool(is_think_input[b].item()) else boost_low
                logits[b, t, self.next_token] = lv
        return logits

    def forward(self, ids: torch.Tensor):
        B, T = ids.shape
        self._last_gate = torch.full((B, T), self.gate_value,
                                     device=ids.device, dtype=torch.float32)
        return self._logits_for(ids)


# ---------------------------------------------------------------------------
# Tests: Probe 1 (per-position CE delta)
# ---------------------------------------------------------------------------

def _write_tiny_probe(tmp_path, rows):
    p = tmp_path / "probe.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_probe1_runs_and_reports_positive_delta(tmp_path):
    # Prompt is all "X" — the stub model predicts "X" with high confidence
    # at THINK positions and weakly at non-think positions, so Δce > 0.
    rows = [
        {"task_id": "Tiny/0",
         "prompt": "XXXXXXXXXX",
         "test": "def check(c):\n    pass\n",
         "entry_point": "f"},
    ]
    probe_file = _write_tiny_probe(tmp_path, rows)
    model = _StubGateModel(gate_value=0.2, next_token=ord("X"))
    tok = _DummyTokenizer()
    res = probe1.run_probe(
        model, tok, probe_path=probe_file,
        n_positions=20, emit_threshold=0.5,
        thinking_token_id=THINK_ID,
    )
    assert res["n_positions_probed"] > 0
    # The stub returns lower CE when the last input is THINK, so Δce > 0.
    assert res["mean_delta_ce"] > 0.0
    assert res["frac_positions_delta_positive"] == 1.0


def test_probe1_no_fires_when_gate_high(tmp_path):
    rows = [{"task_id": "T", "prompt": "abcdefg",
             "test": "def check(c):\n    pass\n", "entry_point": "f"}]
    probe_file = _write_tiny_probe(tmp_path, rows)
    model = _StubGateModel(gate_value=0.9)
    tok = _DummyTokenizer()
    res = probe1.run_probe(
        model, tok, probe_path=probe_file,
        n_positions=20, emit_threshold=0.5,
        thinking_token_id=THINK_ID,
    )
    assert res["n_positions_probed"] == 0
    assert res["n_gate_fires_seen"] == 0


def test_probe1_empty_file_no_error(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    model = _StubGateModel()
    tok = _DummyTokenizer()
    res = probe1.run_probe(
        model, tok, probe_path=str(p), n_positions=10,
        thinking_token_id=THINK_ID,
    )
    assert res["n_positions_probed"] == 0


def test_probe1_restores_training_mode(tmp_path):
    rows = [{"task_id": "T", "prompt": "abcdefg",
             "test": "def check(c):\n    pass\n", "entry_point": "f"}]
    probe_file = _write_tiny_probe(tmp_path, rows)
    model = _StubGateModel()
    tok = _DummyTokenizer()
    model.train()
    probe1.run_probe(model, tok, probe_path=probe_file,
                      thinking_token_id=THINK_ID, n_positions=4)
    assert model.training is True


# ---------------------------------------------------------------------------
# Tests: Probe 2 (counterfactual think-budget=0 vs N)
# ---------------------------------------------------------------------------

def _fake_generate(continuation_text: str):
    """Stand-in for eval_humaneval.generate."""

    def _gen(model, prompt_ids, max_gen=256, temperature=0.0,
             eos_token_id=None, use_thinking=False,
             thinking_token_id=None, total_think_budget=None,
             **kwargs):
        cont = [b for b in continuation_text.encode("utf-8")
                if b not in (0, THINK_ID)][:max_gen]
        cont_t = torch.tensor(cont, dtype=prompt_ids.dtype,
                              device=prompt_ids.device).unsqueeze(0)
        out = torch.cat([prompt_ids, cont_t], dim=1)
        diag = {"think_total": (10 if use_thinking else 0),
                "emit_count": len(cont)}
        return out, diag
    return _gen


def test_probe2_pass_fail_distinction(tmp_path, monkeypatch):
    rows = [
        {"task_id": "Tiny/0",
         "prompt": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
         "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
         "entry_point": "add"},
    ]
    probe_file = _write_tiny_probe(tmp_path, rows)
    monkeypatch.setattr(probe2, "generate",
                        _fake_generate("    return a + b\n"))
    model = _StubGateModel()
    tok = _DummyTokenizer()
    res = probe2.run_probe(
        model, tok, probe_path=probe_file,
        n_problems=None, max_gen=64,
        think_budget=8, thinking_token_id=THINK_ID,
        min_emit_before_eos=0, timeout_s=5,
    )
    assert res["n_total"] == 1
    assert res["no_think"]["n_passed"] == 1
    assert res["with_think"]["n_passed"] == 1
    assert "Tiny/0" in res["both"]


def test_probe2_fail_case(tmp_path, monkeypatch):
    rows = [
        {"task_id": "Tiny/0",
         "prompt": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
         "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
         "entry_point": "add"},
    ]
    probe_file = _write_tiny_probe(tmp_path, rows)
    monkeypatch.setattr(probe2, "generate",
                        _fake_generate("    return a - b\n"))
    model = _StubGateModel()
    tok = _DummyTokenizer()
    res = probe2.run_probe(
        model, tok, probe_path=probe_file,
        n_problems=None, max_gen=64,
        think_budget=8, thinking_token_id=THINK_ID,
        min_emit_before_eos=0, timeout_s=5,
    )
    assert res["no_think"]["n_passed"] == 0
    assert res["with_think"]["n_passed"] == 0
    assert "Tiny/0" in res["neither"]


def test_probe2_think_budget_zero_falls_back_to_no_think(tmp_path, monkeypatch):
    rows = [
        {"task_id": "Tiny/0",
         "prompt": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
         "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
         "entry_point": "add"},
    ]
    probe_file = _write_tiny_probe(tmp_path, rows)
    monkeypatch.setattr(probe2, "generate",
                        _fake_generate("    return a + b\n"))
    model = _StubGateModel()
    tok = _DummyTokenizer()
    res = probe2.run_probe(
        model, tok, probe_path=probe_file,
        think_budget=0, thinking_token_id=THINK_ID,
        min_emit_before_eos=0, max_gen=64, timeout_s=5,
    )
    assert res["no_think"]["n_passed"] == res["with_think"]["n_passed"]


# ---------------------------------------------------------------------------
# Tests: Probe 3 (think-count vs reward Spearman)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _FakeProblem:
    task_id: str = "Fake/0"
    prompt: str = "abc"


@dataclasses.dataclass
class _FakeGrade:
    score: float
    passed: bool = False


def _fake_generate_with_thinks(n_think_seq: list[int]):
    """A generator that returns rollouts with controlled think counts.
    Cycles through n_think_seq across calls."""
    state = {"i": 0}

    def _gen(model, prompt_ids, max_gen=256, temperature=0.0,
             eos_token_id=None, use_thinking=False,
             thinking_token_id=None, **kwargs):
        n_think = n_think_seq[state["i"] % len(n_think_seq)]
        state["i"] += 1
        # Build: prompt + n_think think tokens + 5 emit tokens.
        think_block = [int(thinking_token_id)] * n_think
        emit_block = [ord("a"), ord("b"), ord("c"), ord("d"), ord("e")]
        suffix = think_block + emit_block
        suf_t = torch.tensor(suffix, dtype=prompt_ids.dtype,
                              device=prompt_ids.device).unsqueeze(0)
        out = torch.cat([prompt_ids, suf_t], dim=1)
        diag = {"think_total": n_think, "emit_count": len(emit_block)}
        return out, diag
    return _gen


def test_probe3_correlation_positive_when_thinks_help(monkeypatch):
    # 4 rollouts, increasing n_think and increasing reward → ρ = +1.
    n_thinks = [0, 30, 60, 100]
    rewards = [0.1, 0.3, 0.6, 0.9]
    monkeypatch.setattr(probe3, "generate",
                        _fake_generate_with_thinks(n_thinks))

    rew_iter = iter(rewards)
    def fake_grade(problem, completion, timeout_s=7):
        return _FakeGrade(score=next(rew_iter))

    model = _StubGateModel()
    tok = _DummyTokenizer()
    problems = [_FakeProblem(task_id="P0", prompt="abc")]
    res = probe3.run_probe(
        model, tok, problems, grade_fn=fake_grade,
        n_problems=1, n_rollouts_per_problem=4,
        temperature=0.7, max_gen=16, total_think_budget=120,
        thinking_token_id=THINK_ID,
        min_emit_before_eos=0,
    )
    assert res["n_rollouts"] == 4
    assert res["spearman_think_vs_reward"] == pytest.approx(1.0, abs=1e-6)
    assert res["buckets"]["0"]["n"] == 1
    assert res["buckets"]["1-30"]["n"] == 1
    assert res["buckets"]["31-60"]["n"] == 1
    assert res["buckets"]["91-120+"]["n"] == 1


def test_probe3_zero_correlation_for_uniform_reward(monkeypatch):
    n_thinks = [0, 30, 60, 100]
    monkeypatch.setattr(probe3, "generate",
                        _fake_generate_with_thinks(n_thinks))

    def fake_grade(problem, completion, timeout_s=7):
        return _FakeGrade(score=0.5)

    model = _StubGateModel()
    tok = _DummyTokenizer()
    problems = [_FakeProblem(task_id="P0", prompt="abc")]
    res = probe3.run_probe(
        model, tok, problems, grade_fn=fake_grade,
        n_problems=1, n_rollouts_per_problem=4,
        temperature=0.7, max_gen=16, total_think_budget=120,
        thinking_token_id=THINK_ID,
        min_emit_before_eos=0,
    )
    # All ys identical → denominator zero → returns 0.0 by convention.
    assert res["spearman_think_vs_reward"] == 0.0
    for k in ["0", "1-30", "31-60", "91-120+"]:
        assert res["buckets"][k]["mean_reward"] == 0.5


def test_spearman_helper_basic():
    assert probe3._spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert probe3._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert probe3._spearman([1, 1, 1], [2, 3, 4]) == 0.0


def test_probe3_restores_training_mode(monkeypatch):
    monkeypatch.setattr(probe3, "generate",
                        _fake_generate_with_thinks([5]))

    def fake_grade(problem, completion, timeout_s=7):
        return _FakeGrade(score=0.5)

    model = _StubGateModel()
    tok = _DummyTokenizer()
    model.train()
    problems = [_FakeProblem(task_id="P0", prompt="abc")]
    probe3.run_probe(
        model, tok, problems, grade_fn=fake_grade,
        n_problems=1, n_rollouts_per_problem=1,
        thinking_token_id=THINK_ID, min_emit_before_eos=0,
        max_gen=8,
    )
    assert model.training is True
