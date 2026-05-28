"""Tests for grader-grounded gate calibration (2026-05-28).

The mechanism: roll out with-think + no-think, grade both, derive a BCE
target from Δ(grader score), and BCE the gate logits at the with-think
rollout's fire positions.

What these tests pin (CPU-fast, fully mocked model + grader):
  1. forced Δ>0 (thinking helped) → gate σ RISES at fire positions after a
     step of optimization.
  2. forced Δ<0 (thinking hurt) → gate σ FALLS.
  3. Δ==0 (equal scores) → no loss / no grad / problem skipped.
  4. bounding: max_problems_per_step + max_gate_positions_per_problem caps.
  5. gate logits receive gradient (the smoke-equivalent unit assertion).

Run:
  PYTHONPATH=. .venv/bin/python -m pytest \
      experiments/test_grader_gate_calibration.py -v
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.grader_gate_calibration import (
    compute_grader_gate_calibration_loss,
    _rollout_with_think_record_fires,
)


THINK_ID = 7
EOS_ID = 2
VOCAB = 16
D = 8


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------
class _MockMemory(nn.Module):
    def __init__(self, d):
        super().__init__()
        self._last_injection = None
        self.W_proj = nn.Linear(d, d)  # gives the model a memory-ish param


class _MockLM(nn.Module):
    """Tiny stand-in exposing the surface grader_gate_calibration needs.

    Gate behaviour is scripted: the gate σ for the position is a function
    of `n_think_so_far` in the sequence so we can FORCE a fixed number of
    fire decisions. `fire_steps` consecutive thinks then emit.
    """
    def __init__(self, vocab=VOCAB, d=D, fire_steps=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.gate_head = nn.Linear(d, 1)
        self.memory = _MockMemory(d)
        self.retrieval_input_alpha = nn.Parameter(torch.tensor(0.1))
        self._last_gate = None
        self._last_gate_logits = None
        self.fire_steps = fire_steps

    def forward(self, x, inputs_embeds=None, return_gate=False):
        h = inputs_embeds if inputs_embeds is not None else self.embed(x)
        h = h + h.tanh() * 0.01
        # Stash a fake WM injection so the retrieval-as-input path works.
        self.memory._last_injection = h.detach()
        gl = self.gate_head(h).squeeze(-1)            # (B, T)
        # Script the LAST position's gate to fire while #thinks < fire_steps.
        n_think = int((x[0] == THINK_ID).sum().item())
        last_logit = -5.0 if n_think < self.fire_steps else 5.0
        gl = gl.clone()
        gl[:, -1] = last_logit
        self._last_gate_logits = gl
        self._last_gate = torch.sigmoid(gl)
        logits = self.head(h)
        if return_gate:
            return logits, self._last_gate
        return logits

    def parameters_device(self):
        return next(self.parameters()).device


class _MockTok:
    eos_token_id = EOS_ID

    def encode(self, s, add_special_tokens=False):
        # deterministic small prompt
        return [3, 4, 5]

    def decode(self, ids, skip_special_tokens=True):
        return "GEN:" + ",".join(str(i) for i in ids)


class _MockProblem:
    def __init__(self, task_id="p0"):
        self.task_id = task_id
        self.prompt = "def f():"
        self.prompt_is_code = True


def _mock_build_completion(problem, gen_text, *, extract_code_block):
    return problem, gen_text


class _GR:
    def __init__(self, score):
        self.score = score


def _make_grader(score_with, score_without):
    """grade(prob, comp, timeout_s) — returns score_with for the with-think
    completion (which contains think-derived text) and score_without
    otherwise. We distinguish by completion length: with-think rollouts
    emit the same tokens but the helper grades the *with-think* completion
    first in the loop, so instead we key on a per-call counter."""
    calls = {"n": 0}

    def _grade(prob, comp, timeout_s=7):
        # In compute_*: with-think graded first, then no-think (per problem).
        idx = calls["n"] % 2
        calls["n"] += 1
        return _GR(score_with if idx == 0 else score_without)
    return _grade


def _common_kwargs(model, tok):
    return dict(
        thinking_token_id=THINK_ID,
        prompt_style="humaneval",
        extract_code_block=False,
        retrieval_as_input=True, additive=True,
        max_gen=4, total_think_budget=8, max_think_per_step=4,
        emit_threshold=0.5, gate_floor=0.0, min_emit_before_eos=0,
        max_T=512, timeout_s=2,
        build_completion_fn=_mock_build_completion,
        generate_fn=_mock_generate,
    )


def _mock_generate(model, prompt_t, **kw):
    """Stand-in for generate_with_retrieval_as_input no-think path.
    Forced never-think (emit_threshold=1.0 / budget 0) → just emit tokens."""
    out = prompt_t.clone()
    for _ in range(min(4, kw.get("max_gen", 4))):
        nxt = torch.full((1, 1), 9, dtype=out.dtype, device=out.device)
        out = torch.cat([out, nxt], dim=1)
    return out, {"think_total": 0}


# ---------------------------------------------------------------------------
# 1. forced Δ>0 → gate σ rises at fire positions
# ---------------------------------------------------------------------------
def test_delta_positive_raises_gate_sigma():
    torch.manual_seed(0)
    model = _MockLM(fire_steps=2)
    tok = _MockTok()
    grade_fn = _make_grader(score_with=1.0, score_without=0.0)  # Δ=+1
    problems = [_MockProblem("p0")]

    loss, stats = compute_grader_gate_calibration_loss(
        model, tok, problems, grade_fn=grade_fn,
        max_problems_per_step=1, max_gate_positions_per_problem=64,
        **_common_kwargs(model, tok))
    assert stats.n_used == 1
    assert stats.n_helped == 1
    assert stats.n_gate_positions >= 1
    assert loss.requires_grad

    # Capture σ before/after a step on the recorded fire positions.
    sigma_before = stats.mean_fire_sigma
    opt = torch.optim.SGD(model.parameters(), lr=5.0)
    opt.zero_grad(); loss.backward()
    # gate_head must have a gradient (the load-bearing assertion).
    assert model.gate_head.weight.grad is not None
    assert model.gate_head.weight.grad.abs().sum() > 0
    opt.step()

    # Re-roll and recompute σ at fire positions → should have risen.
    _, stats2 = compute_grader_gate_calibration_loss(
        model, tok, problems, grade_fn=_make_grader(1.0, 0.0),
        max_problems_per_step=1, max_gate_positions_per_problem=64,
        **_common_kwargs(model, tok))
    assert stats2.mean_fire_sigma > sigma_before


# ---------------------------------------------------------------------------
# 2. forced Δ<0 → gate σ falls
# ---------------------------------------------------------------------------
def test_delta_negative_lowers_gate_sigma():
    torch.manual_seed(0)
    model = _MockLM(fire_steps=2)
    tok = _MockTok()
    problems = [_MockProblem("p0")]
    grade_fn = _make_grader(score_with=0.0, score_without=1.0)  # Δ=-1

    loss, stats = compute_grader_gate_calibration_loss(
        model, tok, problems, grade_fn=grade_fn,
        max_problems_per_step=1, max_gate_positions_per_problem=64,
        **_common_kwargs(model, tok))
    assert stats.n_used == 1
    assert stats.n_hurt == 1
    sigma_before = stats.mean_fire_sigma
    opt = torch.optim.SGD(model.parameters(), lr=5.0)
    opt.zero_grad(); loss.backward(); opt.step()

    _, stats2 = compute_grader_gate_calibration_loss(
        model, tok, problems, grade_fn=_make_grader(0.0, 1.0),
        max_problems_per_step=1, max_gate_positions_per_problem=64,
        **_common_kwargs(model, tok))
    assert stats2.mean_fire_sigma < sigma_before


# ---------------------------------------------------------------------------
# 3. Δ==0 → no loss, no grad, skipped
# ---------------------------------------------------------------------------
def test_delta_zero_no_signal():
    torch.manual_seed(0)
    model = _MockLM(fire_steps=2)
    tok = _MockTok()
    problems = [_MockProblem("p0")]
    grade_fn = _make_grader(score_with=0.5, score_without=0.5)  # Δ=0

    loss, stats = compute_grader_gate_calibration_loss(
        model, tok, problems, grade_fn=grade_fn,
        max_problems_per_step=1, max_gate_positions_per_problem=64,
        **_common_kwargs(model, tok))
    assert stats.n_used == 0
    assert stats.n_skipped_equal == 1
    assert loss.item() == 0.0
    assert not loss.requires_grad


# ---------------------------------------------------------------------------
# 4. bounding caps
# ---------------------------------------------------------------------------
def test_max_problems_cap():
    torch.manual_seed(0)
    model = _MockLM(fire_steps=2)
    tok = _MockTok()
    problems = [_MockProblem(f"p{i}") for i in range(5)]
    grade_fn = _make_grader(1.0, 0.0)
    _, stats = compute_grader_gate_calibration_loss(
        model, tok, problems, grade_fn=grade_fn,
        max_problems_per_step=2, max_gate_positions_per_problem=64,
        **{k: v for k, v in _common_kwargs(model, tok).items()})
    assert stats.n_problems == 2  # capped


def test_max_gate_positions_cap():
    torch.manual_seed(0)
    # fire_steps large → many fires; cap to 1.
    model = _MockLM(fire_steps=6)
    tok = _MockTok()
    problems = [_MockProblem("p0")]
    grade_fn = _make_grader(1.0, 0.0)
    kw = _common_kwargs(model, tok)
    kw["max_think_per_step"] = 8
    kw["total_think_budget"] = 16
    _, stats = compute_grader_gate_calibration_loss(
        model, tok, problems, grade_fn=grade_fn,
        max_problems_per_step=1, max_gate_positions_per_problem=1,
        **kw)
    assert stats.n_gate_positions == 1  # capped to 1 per problem


# ---------------------------------------------------------------------------
# 5. rollout records the scripted number of fires
# ---------------------------------------------------------------------------
def test_rollout_records_fires():
    torch.manual_seed(0)
    model = _MockLM(fire_steps=3)
    tok = _MockTok()
    prob = _MockProblem("p0")
    wt = _rollout_with_think_record_fires(
        model, tok, prob, prompt_style="humaneval", max_gen=2,
        total_think_budget=16, max_think_per_step=8,
        emit_threshold=0.5, gate_floor=0.0, min_emit_before_eos=0,
        thinking_token_id=THINK_ID, additive=True, max_T=512)
    # The gate fires until n_think >= fire_steps, so the first emit step
    # records exactly fire_steps fires.
    assert len(wt.fire_positions) >= 3
    # fire positions must be valid indices into full_ids.
    assert all(0 <= p < len(wt.full_ids) for p in wt.fire_positions)
    # The token AFTER each fire position must be the THINK token.
    for p in wt.fire_positions:
        assert wt.full_ids[p + 1] == THINK_ID
