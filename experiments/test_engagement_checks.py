"""Tests for the engagement kill-gate + --feature_lr_mult trainer features
(SESSION_FINDINGS.md 2026-07-02 "recipe rules" — a feature run that ends
inert is a wasted launch, not a negative result).

Covers, CPU-only (no CUDA / real TinyLM needed — pure functions + tiny
nn.Module stand-ins, mirroring test_pretrain_knobs.py /
test_embed_optimizer_wiring.py's style):

  1. `validate_curricula_fit`: warn at >40% of --steps, abort at >100%
     (>40% for latent_reasoning specifically); flags off => no-op.
  2. Per-mechanism engagement evaluators (`_pkm_engaged`, `_film_engaged`,
     `_wm_copy_engaged`, `_latent_engaged`) against synthetic
     committed-vs-inert module states.
  3. `engagement_report`: abort vs warn action, is_main gating.
  4. STARTUP construction asserts (`assert_*_constructed`) fire when a
     flag is on but the corresponding attribute is missing from the model.
  5. `--feature_lr_mult` in `optim_utils.build_optimizer`: default 1.0 is
     param->lr-mapping-identical to the legacy path; !=1.0 carves feature
     params into their own group(s) at lr*mult on both the Muon and AdamW
     sides, non-feature params untouched, PKM values excluded.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_engagement_checks.py -v
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from experiments.train_lm import (
    validate_curricula_fit,
    _curriculum_end_frac,
    _max_abs_alpha,
    _pkm_engaged,
    _film_engaged,
    _wm_copy_engaged,
    _latent_engaged,
    engagement_report,
    assert_pkm_constructed,
    assert_memory_constructed,
    assert_sparse_feedback_constructed,
    assert_latent_reasoner_constructed,
)
from experiments.optim_utils import build_optimizer, _is_feature_lr_param


# ===========================================================================
# 1. Curricula-must-fit validator
# ===========================================================================

def test_curriculum_end_frac():
    assert _curriculum_end_frac(1000, 0, 400) == pytest.approx(0.4)
    assert _curriculum_end_frac(1000, 100, 300) == pytest.approx(0.4)
    assert _curriculum_end_frac(0, 0, 300) == 0.0  # no steps -> 0, not div/0
    assert _curriculum_end_frac(1000, -5, -5) == 0.0  # negatives clamp to 0


def test_no_flags_is_noop():
    # No feature flags set at all -> must not raise or print anything odd.
    validate_curricula_fit(SimpleNamespace(steps=1000))


def test_latent_reasoning_within_40pct_passes():
    validate_curricula_fit(SimpleNamespace(
        steps=1000, latent_reasoning_weight=1.0,
        latent_reasoning_start_step=0,
        latent_reasoning_weight_warmup_steps=300))


def test_latent_reasoning_past_40pct_hard_errors():
    with pytest.raises(SystemExit, match=r"curricula-fit.*40%"):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, latent_reasoning_weight=1.0,
            latent_reasoning_start_step=0,
            latent_reasoning_weight_warmup_steps=500))


def test_latent_reasoning_start_step_counts_toward_the_40pct():
    # start_step=350 + warmup=100 = 450 > 40% of 1000, even though the
    # warmup itself is short.
    with pytest.raises(SystemExit):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, latent_reasoning_weight=1.0,
            latent_reasoning_start_step=350,
            latent_reasoning_weight_warmup_steps=100))


def test_latent_reasoning_weight_zero_is_not_checked():
    # weight==0 means the aux is off -> no curriculum to validate, even
    # with an absurd warmup.
    validate_curricula_fit(SimpleNamespace(
        steps=1000, latent_reasoning_weight=0.0,
        latent_reasoning_start_step=0,
        latent_reasoning_weight_warmup_steps=10_000))


def test_pkm_epsilon_warmup_between_40_and_100pct_warns_not_aborts():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, use_pkm=True,
            pkm_epsilon_start=0.5, pkm_epsilon_warmup_steps=600,
            pkm_alpha_floor_start=0.0, pkm_alpha_floor_warmup_steps=0))
    assert "WARNING" in buf.getvalue()
    assert "pkm_epsilon_warmup_steps" in buf.getvalue()


def test_pkm_epsilon_warmup_past_100pct_aborts():
    with pytest.raises(SystemExit, match=r"never finish"):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, use_pkm=True,
            pkm_epsilon_start=0.5, pkm_epsilon_warmup_steps=1200,
            pkm_alpha_floor_start=0.0, pkm_alpha_floor_warmup_steps=0))


def test_pkm_off_ignores_absurd_epsilon_warmup():
    # use_pkm=False -> the epsilon/floor curricula never actually run, so
    # even a warmup longer than the whole training run is not an error.
    validate_curricula_fit(SimpleNamespace(
        steps=1000, use_pkm=False,
        pkm_epsilon_start=0.5, pkm_epsilon_warmup_steps=10_000,
        pkm_alpha_floor_start=0.0, pkm_alpha_floor_warmup_steps=0))


def test_pkm_epsilon_start_zero_disables_that_specific_check():
    # use_pkm=True but pkm_epsilon_start==0 means the ε-curriculum never
    # actually engages (see the train loop guard) -> not checked, even
    # though pkm_alpha_floor_start>0 (floor) still is.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, use_pkm=True,
            pkm_epsilon_start=0.0, pkm_epsilon_warmup_steps=10_000,
            pkm_alpha_floor_start=0.3, pkm_alpha_floor_warmup_steps=100))
    assert "pkm_epsilon_warmup_steps" not in buf.getvalue()


def test_ctx_addr_aux_warmup_checked_only_when_weight_on():
    with pytest.raises(SystemExit):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, ctx_addr_aux_weight=0.1,
            ctx_addr_aux_start_step=0, ctx_addr_aux_warmup_steps=1500))
    # weight 0 -> no-op even with the same absurd warmup.
    validate_curricula_fit(SimpleNamespace(
        steps=1000, ctx_addr_aux_weight=0.0,
        ctx_addr_aux_start_step=0, ctx_addr_aux_warmup_steps=1500))


def test_feedback_self_k_warmup_checked_unconditionally_when_nonzero():
    with pytest.raises(SystemExit):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, feedback_self_k_warmup_steps=1500))
    validate_curricula_fit(SimpleNamespace(
        steps=1000, feedback_self_k_warmup_steps=0))


def test_gate_warmup_checked_only_when_output_gate_on():
    with pytest.raises(SystemExit):
        validate_curricula_fit(SimpleNamespace(
            steps=1000, output_gate=True, gate_warmup_steps=1500))
    validate_curricula_fit(SimpleNamespace(
        steps=1000, output_gate=False, gate_warmup_steps=1500))


def test_multiple_curricula_one_aborting_one_warning_still_aborts():
    with pytest.raises(SystemExit, match=r"never finish"):
        validate_curricula_fit(SimpleNamespace(
            steps=1000,
            use_pkm=True,
            pkm_epsilon_start=0.5, pkm_epsilon_warmup_steps=600,   # warn (60%)
            pkm_alpha_floor_start=0.5, pkm_alpha_floor_warmup_steps=1500,  # abort
        ))


# ===========================================================================
# 2. Per-mechanism engagement evaluators (synthetic committed vs inert)
# ===========================================================================

def test_max_abs_alpha_polymorphic_formats():
    assert _max_abs_alpha([]) == 0.0
    assert _max_abs_alpha([0.1, -0.5, 0.2]) == pytest.approx(0.5)          # dense-single
    assert _max_abs_alpha([[0.1, -0.2], [0.3, 0.05]]) == pytest.approx(0.3)  # dense-multi
    assert _max_abs_alpha([(2, 28, -0.7), (3, 27, 0.1)]) == pytest.approx(0.7)  # sparse/xattn


def test_pkm_engaged_via_alpha():
    ok, detail = _pkm_engaged(0.05, 1.0, alpha_min=0.02, row_min=1.02)
    assert ok
    assert "alphaL" in detail


def test_pkm_engaged_via_row_ratio():
    ok, _ = _pkm_engaged(0.001, 1.10, alpha_min=0.02, row_min=1.02)
    assert ok


def test_pkm_inert_both_below_threshold():
    ok, _ = _pkm_engaged(0.0005, 1.004, alpha_min=0.02, row_min=1.02)
    assert not ok


def test_pkm_engaged_nan_alpha_falls_through_to_row():
    # pkm.use_output_gate=False -> aL is nan; nan comparisons are False, so
    # this must not crash and must fall through to the row-ratio check.
    ok, _ = _pkm_engaged(float("nan"), 1.10, alpha_min=0.02, row_min=1.02)
    assert ok
    ok, _ = _pkm_engaged(float("nan"), 1.0, alpha_min=0.02, row_min=1.02)
    assert not ok


def test_film_engaged_and_inert():
    ok, _ = _film_engaged([(2, 28, 0.05)], alpha_min=1e-3)
    assert ok
    ok, _ = _film_engaged([(2, 28, 1e-5)], alpha_min=1e-3)
    assert not ok
    ok, _ = _film_engaged([], alpha_min=1e-3)
    assert not ok


def test_wm_copy_engaged_via_bias_delta():
    ok, detail = _wm_copy_engaged(-5.9, -6.0, fire_count=0,
                                  bias_delta_min=0.05)
    assert ok
    assert "cumulative_match_fires=0" in detail


def test_wm_copy_engaged_via_fire_count_even_if_bias_static():
    ok, _ = _wm_copy_engaged(-6.0, -6.0, fire_count=3, bias_delta_min=0.05)
    assert ok


def test_wm_copy_inert():
    ok, _ = _wm_copy_engaged(-5.98, -6.0, fire_count=0, bias_delta_min=0.05)
    assert not ok


def test_latent_engaged_before_curriculum_end_is_note_not_failure():
    ok, detail = _latent_engaged(True, 0, True, step=100, start_step=0,
                                 end_step=500)
    assert ok  # NOTE-tagged pass, not a false failure
    assert detail.startswith("NOTE:")


def test_latent_engaged_after_end_committed():
    ok, _ = _latent_engaged(True, 5, True, step=600, start_step=0,
                            end_step=500)
    assert ok


def test_latent_inert_not_constructed():
    ok, _ = _latent_engaged(False, 5, True, step=600, start_step=0,
                            end_step=500)
    assert not ok


def test_latent_inert_never_fired():
    ok, _ = _latent_engaged(True, 0, True, step=600, start_step=0,
                            end_step=500)
    assert not ok


def test_latent_inert_nonfinite():
    ok, _ = _latent_engaged(True, 5, False, step=600, start_step=0,
                            end_step=500)
    assert not ok


# ===========================================================================
# 3. engagement_report: abort vs warn, is_main gating
# ===========================================================================

def test_engagement_report_all_engaged_no_raise():
    engagement_report(100, [("PKM", True, "ok"), ("FiLM", True, "ok")],
                      action="abort", is_main=False)  # must not raise


def test_engagement_report_abort_raises_with_inert():
    with pytest.raises(SystemExit, match=r"INERT: PKM"):
        engagement_report(100, [("PKM", False, "dead"), ("FiLM", True, "ok")],
                          action="abort", is_main=False)


def test_engagement_report_warn_does_not_raise():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        engagement_report(100, [("PKM", False, "dead")],
                          action="warn", is_main=True)
    assert "WARNING" in buf.getvalue()
    assert "INERT" in buf.getvalue()


def test_engagement_report_is_main_false_suppresses_print_but_still_aborts():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(SystemExit):
            engagement_report(100, [("PKM", False, "dead")],
                              action="abort", is_main=False)
    # Non-main rank prints nothing (avoid DDP log spam) but the SAME
    # SystemExit fires -> the whole job dies together, not just rank 0.
    assert buf.getvalue() == ""


def test_engagement_report_empty_results_no_raise():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        engagement_report(100, [], action="abort", is_main=True)
    assert "no mechanisms enabled" in buf.getvalue()


# ===========================================================================
# 4. STARTUP construction asserts
# ===========================================================================

def test_assert_pkm_constructed_passes_when_off():
    assert_pkm_constructed(SimpleNamespace(use_pkm=False), object())


def test_assert_pkm_constructed_fires_when_missing():
    model = SimpleNamespace()  # no .pkm_layer attribute
    with pytest.raises(AssertionError, match="pkm_layer"):
        assert_pkm_constructed(SimpleNamespace(use_pkm=True), model)


def test_assert_pkm_constructed_passes_when_present():
    model = SimpleNamespace(pkm_layer=object())
    assert_pkm_constructed(SimpleNamespace(use_pkm=True), model)


def test_assert_memory_constructed_fires_when_missing():
    model = SimpleNamespace()
    with pytest.raises(AssertionError, match="memory"):
        assert_memory_constructed(SimpleNamespace(use_memory=True), model)


def test_assert_memory_constructed_passes_when_present():
    model = SimpleNamespace(memory=object())
    assert_memory_constructed(SimpleNamespace(use_memory=True), model)


def test_assert_sparse_feedback_constructed_fires_when_missing():
    model = SimpleNamespace()
    args = SimpleNamespace(feedback="film", feedback_pairs="2,28")
    with pytest.raises(AssertionError, match="sparse_feedback"):
        assert_sparse_feedback_constructed(args, model)


def test_assert_sparse_feedback_constructed_fires_when_empty():
    model = SimpleNamespace(sparse_feedback={})
    args = SimpleNamespace(feedback="film", feedback_pairs="2,28")
    with pytest.raises(AssertionError, match="sparse_feedback"):
        assert_sparse_feedback_constructed(args, model)


def test_assert_sparse_feedback_constructed_passes_when_present():
    model = SimpleNamespace(sparse_feedback={"2": object()})
    args = SimpleNamespace(feedback="film", feedback_pairs="2,28")
    assert_sparse_feedback_constructed(args, model)


def test_assert_sparse_feedback_skipped_without_feedback_pairs():
    # feedback="film" but no --feedback_pairs -> dense (non-sparse) FiLM
    # path; sparse_feedback is legitimately absent there.
    model = SimpleNamespace()
    args = SimpleNamespace(feedback="film", feedback_pairs="")
    assert_sparse_feedback_constructed(args, model)  # must not raise


def test_assert_latent_reasoner_constructed_fires_when_missing():
    args = SimpleNamespace(latent_reasoning_weight=1.0)
    with pytest.raises(AssertionError, match="_latent_reasoner"):
        assert_latent_reasoner_constructed(args, None)


def test_assert_latent_reasoner_constructed_passes_when_present():
    args = SimpleNamespace(latent_reasoning_weight=1.0)
    assert_latent_reasoner_constructed(args, object())


def test_assert_latent_reasoner_skipped_when_weight_zero():
    args = SimpleNamespace(latent_reasoning_weight=0.0)
    assert_latent_reasoner_constructed(args, None)  # must not raise


# ===========================================================================
# 5. --feature_lr_mult in optim_utils.build_optimizer
# ===========================================================================

_OPT_KW = dict(optimizer="muon", lr=1e-3, lr_muon=5e-3, alpha_wd=0.0,
               steps=100, wd=0.01, lr_schedule="wsd", warmup_steps=10,
               verbose=False)


class _TinyWithFeatures(nn.Module):
    """Exercises every --feature_lr_mult routing branch: a plain trunk
    (embed/lm_head/proj/norm, unaffected) plus one param from each feature
    module named in `_is_feature_lr_param` (sparse_feedback 2D + alpha,
    memory 2D, pkm_layer query 2D + out_alpha 1D + EXCLUDED value table,
    latent_feedback_adapter 2D + alpha, gate_head 2D + bias)."""

    def __init__(self, vocab=32, d=16):
        super().__init__()
        # Trunk (non-feature) — must be completely unaffected by
        # feature_lr_mult.
        self.embed = nn.Embedding(vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        # Feature modules.
        self.sparse_feedback = nn.ModuleDict({
            "2": nn.Module(),
        })
        self.sparse_feedback["2"].gamma = nn.Linear(d, d, bias=False)
        self.sparse_feedback["2"].alpha = nn.Parameter(torch.zeros(1))
        self.memory = nn.Module()
        self.memory.W_proj = nn.Linear(d, d, bias=False)
        self.pkm_layer = nn.Module()
        self.pkm_layer.query_proj = nn.Linear(d, d, bias=True)
        self.pkm_layer.out_alpha = nn.Parameter(torch.zeros(1))
        self.pkm_layer.values = nn.ModuleList([nn.Embedding(8, d)])
        self.latent_feedback_adapter = nn.Module()
        self.latent_feedback_adapter.proj = nn.Linear(d, d, bias=True)
        self.latent_feedback_adapter.alpha = nn.Parameter(torch.zeros(1))
        self.gate_head = nn.Linear(d, 1, bias=True)

    def named_parameters(self, *a, **kw):  # noqa: D102 - delegate to super
        return super().named_parameters(*a, **kw)


def _model(seed=0):
    torch.manual_seed(seed)
    return _TinyWithFeatures()


def _name_of(model):
    return {id(p): n for n, p in model.named_parameters()}


def _param_lr_map(model, opts):
    """param name -> effective lr, across every optimizer/group."""
    name_of = _name_of(model)
    out = {}
    for opt in opts:
        for g in opt.param_groups:
            lr = g.get("initial_lr", g["lr"])
            for p in g["params"]:
                out[name_of[id(p)]] = lr
    return out


def test_feature_param_prefixes_matched():
    assert _is_feature_lr_param("sparse_feedback.2.gamma.weight")
    assert _is_feature_lr_param("sparse_feedback.2.alpha")
    assert _is_feature_lr_param("memory.W_proj.weight")
    assert _is_feature_lr_param("pkm_layer.query_proj.weight")
    assert _is_feature_lr_param("pkm_layer.out_alpha")
    assert _is_feature_lr_param("latent_feedback_adapter.proj.weight")
    assert _is_feature_lr_param("gate_head.weight")
    # PKM values are explicitly EXCLUDED (own --pkm_value_lr_mult path).
    assert not _is_feature_lr_param("pkm_layer.values.0.weight")
    # Trunk params are not matched.
    assert not _is_feature_lr_param("embed.weight")
    assert not _is_feature_lr_param("proj.weight")


def test_feature_lr_mult_default_is_group_identical_to_no_kwarg():
    """feature_lr_mult=1.0 (default) must give the EXACT SAME param->lr
    mapping as never passing the kwarg at all."""
    mA, mB = _model(0), _model(0)
    optsA, _ = build_optimizer(mA, **_OPT_KW)
    optsB, _ = build_optimizer(mB, feature_lr_mult=1.0, **_OPT_KW)
    mapA, mapB = _param_lr_map(mA, optsA), _param_lr_map(mB, optsB)
    assert mapA == mapB


def test_feature_lr_mult_default_no_extra_groups():
    m = _model()
    opts, _ = build_optimizer(m, feature_lr_mult=1.0, **_OPT_KW)
    # Still exactly matrix + AdamW — no extra optimizer instances.
    assert len(opts) == 2


def test_feature_lr_mult_scales_feature_params_both_sides():
    m = _model()
    opts, _ = build_optimizer(m, feature_lr_mult=5.0, **_OPT_KW)
    lrmap = _param_lr_map(m, opts)
    base_lr, base_lr_muon = _OPT_KW["lr"], _OPT_KW["lr_muon"]

    # Muon-side: 2D hidden matrices that are NOT diverted to AdamW by an
    # earlier special-case. latent_feedback_adapter is EXCLUDED here even
    # though .proj.weight is 2D — build_optimizer routes the WHOLE adapter
    # (matrix included) to AdamW unconditionally (see
    # `_is_latent_feedback_adapter`'s docstring: zero-init + identity
    # residual makes Newton-Schulz the wrong inductive bias there).
    feature_muon_2d = {"sparse_feedback.2.gamma.weight", "memory.W_proj.weight",
                       "pkm_layer.query_proj.weight", "gate_head.weight"}
    feature_adamw = {"sparse_feedback.2.alpha", "pkm_layer.out_alpha",
                     "latent_feedback_adapter.proj.weight",
                     "latent_feedback_adapter.proj.bias",
                     "latent_feedback_adapter.alpha",
                     "pkm_layer.query_proj.bias", "gate_head.bias"}
    for n in feature_muon_2d:
        assert lrmap[n] == pytest.approx(base_lr_muon * 5.0), (n, lrmap[n])
    for n in feature_adamw:
        assert lrmap[n] == pytest.approx(base_lr * 5.0), (n, lrmap[n])

    # PKM value table: EXCLUDED from feature_lr_mult (stays at base lr,
    # since pkm_value_lr_mult defaults to 1.0 here).
    assert lrmap["pkm_layer.values.0.weight"] == pytest.approx(base_lr)

    # Trunk params completely untouched: proj.weight is a plain 2D matrix
    # (-> Muon, lr_muon); the rest are embedding-like/1D (-> AdamW, lr).
    assert lrmap["proj.weight"] == pytest.approx(base_lr_muon), (
        "proj.weight", lrmap["proj.weight"])
    for n in ("embed.weight", "lm_head.weight", "norm.weight", "norm.bias"):
        assert lrmap[n] == pytest.approx(base_lr), (n, lrmap[n])


def test_feature_lr_mult_only_moves_feature_params_numerically():
    """Step both models with identical grads; non-feature params must be
    bit-identical, feature params must have moved MORE (higher lr)."""
    mA, mB = _model(0), _model(0)
    optsA, _ = build_optimizer(mA, **_OPT_KW)
    optsB, _ = build_optimizer(mB, feature_lr_mult=8.0, **_OPT_KW)
    gA = torch.Generator().manual_seed(11)
    gB = torch.Generator().manual_seed(11)

    def _set_grads(model, gen):
        for _, p in model.named_parameters():
            p.grad = torch.randn(p.shape, generator=gen) * 0.01

    def _step(model, opts, gen):
        _set_grads(model, gen)
        for o in opts:
            o.step()

    for _ in range(3):
        _step(mA, optsA, gA)
        _step(mB, optsB, gB)

    pB = {n: p for n, p in mB.named_parameters()}
    feature_names = {n for n, _ in mA.named_parameters()
                     if _is_feature_lr_param(n) and "values" not in n}
    for n, pA in mA.named_parameters():
        d = (pA - pB[n]).abs().max().item()
        if n in feature_names:
            assert d > 0.0, f"feature param {n} should move differently at 8x lr"
        else:
            assert d == 0.0, f"non-feature {n} must match baseline exactly, got {d}"


def test_feature_lr_mult_works_with_plain_adamw_optimizer():
    m = _model()
    kw = dict(_OPT_KW)
    kw["optimizer"] = "adamw"
    opts, _ = build_optimizer(m, feature_lr_mult=4.0, **kw)
    assert len(opts) == 1
    lrmap = _param_lr_map(m, opts)
    assert lrmap["memory.W_proj.weight"] == pytest.approx(_OPT_KW["lr"] * 4.0)
    assert lrmap["embed.weight"] == pytest.approx(_OPT_KW["lr"])


def test_feature_lr_mult_prints_group_sizes(capsys):
    m = _model()
    build_optimizer(m, feature_lr_mult=3.0, optimizer="muon", lr=1e-3,
                    lr_muon=5e-3, alpha_wd=0.0, steps=100, wd=0.01,
                    lr_schedule="wsd", warmup_steps=10, verbose=True)
    out = capsys.readouterr().out
    assert "feature LR mult" in out


def test_feature_lr_mult_rejects_fused_deltanet_ns():
    m = _model()
    with pytest.raises(ValueError, match="fused_deltanet_ns"):
        build_optimizer(m, feature_lr_mult=2.0,
                        matrix_optimizer="fused_deltanet_ns", **_OPT_KW)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
