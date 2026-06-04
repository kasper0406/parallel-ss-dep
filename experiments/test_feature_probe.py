"""Tests for the per-feature usefulness probe and the new CLI flag plumbing.

The probe ablates each mechanism (WM / PKM) on a held-out batch and reports
the CE rise — load-bearing iff Δce > 0. These tests assert:
  * the ablation-delta is ~0 when a feature is INERT (α already 0), and
  * the ablation-delta is > 0 when a feature is LOAD-BEARING (it lowers CE),
  * the probe restores every poked attribute (idempotent, no training leak),
  * the new train_lm_args flags parse with backwards-compatible defaults,
  * model_builder threads the WM-DKV kwargs into TinyLM/WorkingMemory.

CPU-only: tiny mock models, no DeltaNet CUDA kernels.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_feature_probe.py -v
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.feature_probe import (
    run_feature_probe, format_feature_probe)
from experiments.train_lm_args import build_parser


# --------------------------------------------------------------------------- #
# Mock model with a controllable WM (read_alpha) and PKM (out_alpha) so we can
# dial each feature between "inert" and "load-bearing".
# --------------------------------------------------------------------------- #
class _FakeWM(nn.Module):
    """A read injection scaled by read_alpha. The injection is a fixed, useful
    correction toward the target logits (so a non-zero α LOWERS CE)."""

    def __init__(self, d, vocab):
        super().__init__()
        self.read_alpha = nn.Parameter(torch.tensor(1.0))
        # `correction` is added to the hidden; with the head below this moves
        # logits toward the true class. Registered as a buffer (not learned).
        self.register_buffer("correction", torch.zeros(d))


class _FakePKM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.use_output_gate = True
        self.out_alpha = nn.Parameter(torch.zeros(1))
        self.alpha_floor = 0.0
        self.register_buffer("correction", torch.zeros(d))


class _MockLMWithFeatures(nn.Module):
    """logits = head(emb(x) + wm.read_alpha*wm.correction
                       + pkm.out_alpha*pkm.correction). The corrections are set
    up so that, when active, they push the hidden toward a direction the head
    maps to the true next token — i.e. the feature LOWERS CE."""

    def __init__(self, vocab=8, d=8, with_wm=True, with_pkm=True,
                 with_film=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.gate = nn.Linear(d, 1)
        self._last_gate = None
        self._last_gate_logits = None
        self.memory = _FakeWM(d, vocab) if with_wm else None
        self.pkm_layer = _FakePKM(d) if with_pkm else None
        self._film_alphas = ([(0, 5, 0.42), (1, 6, -0.13)] if with_film
                             else [])

    def feedback_alphas(self):
        return list(self._film_alphas)

    def forward(self, x, doc_ids=None, return_aux=False):
        h = self.emb(x)
        if self.memory is not None:
            h = h + self.memory.read_alpha * self.memory.correction
        if self.pkm_layer is not None:
            a = self.pkm_layer.out_alpha
            if self.pkm_layer.alpha_floor > 0.0:
                a = a + torch.sign(a) * self.pkm_layer.alpha_floor
            h = h + a * self.pkm_layer.correction
        gl = self.gate(h).squeeze(-1)
        self._last_gate_logits = gl
        self._last_gate = torch.sigmoid(gl)
        return self.head(h)


def _make_loadbearing_model():
    """Build a model where WM and PKM corrections genuinely lower CE."""
    torch.manual_seed(0)
    vocab, d = 8, 8
    m = _MockLMWithFeatures(vocab, d, with_wm=True, with_pkm=True,
                            with_film=True)
    # Make head identity-ish so a hidden in the direction of one-hot(c) yields
    # the largest logit for class c.
    with torch.no_grad():
        m.head.weight.copy_(torch.eye(vocab, d))
        # WM correction points strongly along class 3; PKM along class 5.
        m.memory.correction[3] = 5.0
        m.pkm_layer.correction[5] = 5.0
        m.pkm_layer.out_alpha.fill_(1.0)   # PKM active
    return m, vocab, d


def test_probe_returns_finite_metrics():
    m, vocab, d = _make_loadbearing_model()
    x = torch.randint(0, vocab, (2, 6))
    y = torch.randint(0, vocab, (2, 6))
    metrics = run_feature_probe(m, x, y, thinking_token_id=vocab - 1)
    for k, v in metrics.items():
        assert v == v, f"{k} is NaN"          # finite
        assert abs(v) < 1e6
    assert "base_ce" in metrics
    assert "wm_ablation_delta_ce" in metrics
    assert "pkm_ablation_delta_ce" in metrics
    assert "wm_read_alpha" in metrics
    assert "pkm_alpha" in metrics
    assert "film_alpha_max_abs" in metrics
    assert "gate_fire_rate" in metrics
    # format helper produces a non-empty one-liner.
    s = format_feature_probe(metrics)
    assert s.startswith("[feature-probe]")
    assert "wm(" in s and "pkm(" in s


def test_ablation_delta_zero_when_feature_inert():
    """When read_alpha / out_alpha are already 0, ablating changes nothing."""
    torch.manual_seed(1)
    vocab, d = 8, 8
    m = _MockLMWithFeatures(vocab, d, with_wm=True, with_pkm=True)
    with torch.no_grad():
        m.memory.read_alpha.zero_()       # WM inert
        m.pkm_layer.out_alpha.zero_()     # PKM inert (already 0)
        m.memory.correction[3] = 5.0
        m.pkm_layer.correction[5] = 5.0
    x = torch.randint(0, vocab, (2, 6))
    y = torch.randint(0, vocab, (2, 6))
    metrics = run_feature_probe(m, x, y)
    assert abs(metrics["wm_ablation_delta_ce"]) < 1e-5
    assert abs(metrics["pkm_ablation_delta_ce"]) < 1e-5


def test_ablation_delta_positive_when_loadbearing():
    """When the feature lowers CE, ablating it must RAISE CE (delta > 0)."""
    m, vocab, d = _make_loadbearing_model()
    # Construct targets that the corrections actually help predict: the WM
    # correction points at class 3, PKM at class 5. Use a batch whose targets
    # are those classes so the active feature genuinely lowers CE.
    x = torch.zeros(2, 4, dtype=torch.long)
    y = torch.full((2, 4), 3, dtype=torch.long)   # WM correction → class 3
    m_wm_only, _, _ = _make_loadbearing_model()
    with torch.no_grad():
        m_wm_only.pkm_layer.out_alpha.zero_()     # isolate WM
    metrics = run_feature_probe(m_wm_only, x, y)
    assert metrics["wm_ablation_delta_ce"] > 0.0, metrics

    y_pkm = torch.full((2, 4), 5, dtype=torch.long)  # PKM correction → class 5
    m_pkm_only, _, _ = _make_loadbearing_model()
    with torch.no_grad():
        m_pkm_only.memory.read_alpha.zero_()      # isolate PKM
    metrics_p = run_feature_probe(m_pkm_only, x, y_pkm)
    assert metrics_p["pkm_ablation_delta_ce"] > 0.0, metrics_p


def test_probe_restores_attributes_and_training_mode():
    m, vocab, d = _make_loadbearing_model()
    m.train()
    a0 = float(m.memory.read_alpha.detach())
    p0 = float(m.pkm_layer.out_alpha.detach())
    f0 = float(m.pkm_layer.alpha_floor)
    x = torch.randint(0, vocab, (2, 5))
    y = torch.randint(0, vocab, (2, 5))
    _ = run_feature_probe(m, x, y)
    assert float(m.memory.read_alpha.detach()) == a0
    assert float(m.pkm_layer.out_alpha.detach()) == p0
    assert float(m.pkm_layer.alpha_floor) == f0
    assert m.training is True   # restored to train mode


def test_probe_handles_missing_features():
    """A model with no WM / PKM / FiLM just returns base_ce + gate."""
    torch.manual_seed(2)
    vocab, d = 8, 8
    m = _MockLMWithFeatures(vocab, d, with_wm=False, with_pkm=False)
    m._film_alphas = []
    x = torch.randint(0, vocab, (2, 5))
    y = torch.randint(0, vocab, (2, 5))
    metrics = run_feature_probe(m, x, y)
    assert "base_ce" in metrics
    assert "wm_ablation_delta_ce" not in metrics
    assert "pkm_ablation_delta_ce" not in metrics
    assert "film_alpha_max_abs" not in metrics


# --------------------------------------------------------------------------- #
# WM-recall probe — the WM load-bearing signal must come from a think-bearing
# RECALL batch (the natural-text val batch has zero think tokens). This tests
# the importable eval_longctx_recall() callable on a tiny fake recall set with
# the generators monkeypatched (CPU, no CUDA / DeltaNet kernels).
# --------------------------------------------------------------------------- #
class _FakeTok:
    eos_token_id = 0

    def encode(self, s, add_special_tokens=False):
        # Deterministic: length proportional to prompt so truncation logic runs.
        return [1, 2, 3, 4, 5]

    def decode(self, ids, skip_special_tokens=True):
        # The generator stashes the gold answer into the returned ids; decode
        # turns them back into the "Answer: N" text the extractor parses.
        return "Answer: " + "".join(chr(48 + (i % 10)) for i in ids)


class _RecallMockModel(nn.Module):
    """Minimal model contract for eval_longctx_recall: has thinking_token_id,
    a WM with read_alpha (so the mean-ablation contextmanager has something to
    poke), and max_T. Generation is monkeypatched at module level."""

    def __init__(self):
        super().__init__()
        self.thinking_token_id = 99
        self.max_T = 2048
        self.retrieval_input_additive = False
        self.memory = _FakeWM(4, 8)
        self.train()


def _make_fake_recall_jsonl(tmp_path):
    import json
    p = tmp_path / "recall.jsonl"
    rows = [
        {"task_id": "longctx/d64/0", "problem_prompt":
            "x = 7\n# distractor\nprint(x)", "answer": "7"},
        {"task_id": "longctx/d64/1", "problem_prompt":
            "y = 3\n# distractor\nprint(y)", "answer": "3"},
    ]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_wm_recall_probe_returns_finite_and_think_guard(tmp_path, monkeypatch):
    """The WM-recall callable returns finite numbers, recall in [0,1], and a
    think_frac > 0 guard (the probe must actually exercise the think path)."""
    import experiments.eval_humaneval as eh

    # Monkeypatch the generator to a tiny CPU stub that 'recalls' the binding
    # perfectly and reports a non-zero think count (so think_frac > 0).
    def _fake_gen(model, prompt_t, *, max_gen, temperature, eos_token_id,
                 thinking_token_id, total_think_budget, emit_threshold,
                 gate_floor, additive=False, **kw):
        # Emit the digit of the answer encoded so _FakeTok.decode yields it.
        # We want "Answer: 7" → emit token whose (id % 10) == 7.
        ans_digit = int(model._pending_answer)
        gen_ids = list(prompt_t[0].tolist()) + [ans_digit]
        gen = torch.tensor(gen_ids, dtype=torch.long).unsqueeze(0)
        diag = {"think_total": 4, "emit_count": 6}
        return gen, diag

    monkeypatch.setattr(eh, "generate_with_retrieval_as_input", _fake_gen)
    monkeypatch.setattr(eh, "generate", _fake_gen)

    from experiments.eval_longctx_recall import eval_longctx_recall

    model = _RecallMockModel()
    # The stub needs to know the per-record gold; thread it via a side channel
    # by wrapping the records loop is overkill — instead set it from the model.
    model._pending_answer = "7"  # both fake rows resolve correct vs gold check
    tok = _FakeTok()
    path = _make_fake_recall_jsonl(tmp_path)

    out = eval_longctx_recall(
        model, tok, path, n=2, generator="retrieval_as_input",
        wm_ablate="none", device="cpu", max_gen=8, total_think_budget=4)
    for k, v in out.items():
        assert v == v, f"{k} is NaN"
        assert abs(v) < 1e9
    assert 0.0 <= out["recall"] <= 1.0
    assert out["n_total"] == 2.0
    assert out["think_frac"] > 0.0, "probe must exercise the think path"
    # train mode restored after the no_grad eval.
    assert model.training is True


def test_wm_recall_mean_ablation_zeros_read_alpha(tmp_path, monkeypatch):
    """The 'mean' ablation must zero read_alpha DURING generation and restore
    it afterwards (idempotent — no training-state leak)."""
    import experiments.eval_humaneval as eh

    seen_alpha = {}

    def _fake_gen(model, prompt_t, **kw):
        seen_alpha["during"] = float(model.memory.read_alpha.detach())
        gen = torch.cat([prompt_t, torch.tensor([[7]])], dim=1)
        return gen, {"think_total": 2, "emit_count": 3}

    monkeypatch.setattr(eh, "generate_with_retrieval_as_input", _fake_gen)

    from experiments.eval_longctx_recall import eval_longctx_recall

    model = _RecallMockModel()
    with torch.no_grad():
        model.memory.read_alpha.fill_(0.75)
    before = float(model.memory.read_alpha.detach())
    tok = _FakeTok()
    path = _make_fake_recall_jsonl(tmp_path)

    out = eval_longctx_recall(
        model, tok, path, n=1, generator="retrieval_as_input",
        wm_ablate="mean", device="cpu", max_gen=8, total_think_budget=4)
    assert seen_alpha["during"] == 0.0, "read_alpha must be 0 during ablation"
    assert float(model.memory.read_alpha.detach()) == before, "must restore"
    assert out["wm_ablated"] == 1.0


# --------------------------------------------------------------------------- #
# CLI flag plumbing — defaults must be backwards-compatible (off).
# --------------------------------------------------------------------------- #
def test_new_flags_default_off():
    a = build_parser().parse_args([])
    # WM-DKV
    assert a.mem_decoupled_kv is False
    assert a.mem_read_alpha_init == 1.0
    assert a.mem_read_alpha_floor_start == 0.0
    assert a.mem_read_alpha_floor_warmup_steps == 0
    # gate-calibration
    assert a.gate_calibration_weight == 0.0
    assert a.gate_calibration_R == 4
    assert a.gate_calibration_sample_frac == 0.05
    assert a.gate_calibration_max_positions == 32
    assert a.gate_calibration_sigma_low == 0.0
    assert a.gate_calibration_sigma_high == 1.0
    # feature probe
    assert a.feature_probe_every_tokens == 0


def test_new_flags_parse_values():
    a = build_parser().parse_args([
        "--arch", "deltanet",
        "--mem_decoupled_kv",
        "--mem_read_alpha_init", "0.0",
        "--mem_read_alpha_floor_start", "0.5",
        "--mem_read_alpha_floor_warmup_steps", "3000",
        "--gate_calibration_weight", "0.05",
        "--gate_calibration_R", "3",
        "--feature_probe_every_tokens", "250000000",
    ])
    assert a.mem_decoupled_kv is True
    assert a.mem_read_alpha_init == 0.0
    assert a.mem_read_alpha_floor_start == 0.5
    assert a.mem_read_alpha_floor_warmup_steps == 3000
    assert a.gate_calibration_weight == 0.05
    assert a.gate_calibration_R == 3
    assert a.feature_probe_every_tokens == 250000000


def test_model_builder_threads_wm_dkv_kwargs(monkeypatch):
    """build_model_from_args must forward the WM-DKV kwargs to TinyLM."""
    import experiments.model_builder as mb

    captured = {}

    class _FakeTiny:
        def __init__(self, **kw):
            captured.update(kw)

        def to(self, *_a, **_k):
            return self

    monkeypatch.setattr(mb, "TinyLM", _FakeTiny)
    a = build_parser().parse_args([
        "--arch", "deltanet", "--n_layers", "2",
        "--d_model", "16", "--n_heads", "1", "--d_head", "16",
        "--use_memory", "--mem_size", "32",
        "--mem_decoupled_kv",
        "--mem_read_alpha_init", "0.0",
        "--mem_read_alpha_floor_start", "0.5",
        "--mem_read_alpha_floor_warmup_steps", "2500",
    ])
    mb.build_model_from_args(a, vocab_size=40, thinking_token_id=39)
    assert captured["use_memory"] is True
    assert captured["mem_decoupled_kv"] is True
    assert captured["mem_read_alpha_init"] == 0.0
    assert captured["mem_read_alpha_floor_start"] == 0.5
    assert captured["mem_read_alpha_floor_warmup_steps"] == 2500
