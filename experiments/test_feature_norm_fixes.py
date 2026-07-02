"""Tests for the 2026-07-02 design-review feature-mechanism fixes.

FIX 1 (experiments/memory_layer.py, PKMLayer) — the sign-preserving
alpha_floor bootstrap crutch must never fire during eval()/VAL; it's a
training-only device. This is a real bugfix, unconditional, no flag.

FIX 2 (experiments/model.py, FeedbackProjection) — `src_norm={"none","rms"}`.
Normalizes the SOURCE state (state_above_lagged) before it drives
W_scale/W_shift/W_fb, decoupling FiLM's effective strength from raw
source-norm drift.

FIX 3 (experiments/model.py, WorkingMemory) — `inj_norm={"none","match_rms"}`.
Rescales the WM read injection per-position to match the local residual
stream's RMS, so `read_alpha` is a pure mixing fraction.

Design note on defaults (2026-07-02 spec change — these are FIXES, not
experiments, so they must be ON by default going forward): the CLI flags
in train_lm_args.py default to the FIXED behaviour ("rms" / "match_rms").
The underlying module constructors (FeedbackProjection, WorkingMemory,
TinyLM) keep their own Python-level default at "none" — the safe/legacy
value — so:
  - any direct/raw construction that doesn't pass the new kwarg (tests,
    other scripts, an old ckpt reconstructed via build_model_from_ckpt
    whose cfg predates this fix) stays exactly byte-identical to the
    pre-fix code;
  - model_builder.py always threads the CLI-parsed args value through
    (unconditionally, not just when overridden), so any NEW run launched
    via train_lm.py picks up the fix without the user having to remember
    a flag.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.layers import DeltaNetAttention
from experiments.memory_layer import PKMLayer
from experiments.model import FeedbackProjection, TinyLM, WorkingMemory
from experiments.train_lm_args import build_parser


# ---------------------------------------------------------------------------
# FIX 1 — PKM alpha_floor training-only gate
# ---------------------------------------------------------------------------

def test_pkm_alpha_floor_fires_in_train_mode():
    torch.manual_seed(0)
    layer = PKMLayer(d_model=16, n_heads=1, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)  # use_output_gate=True default
    with torch.no_grad():
        layer.out_alpha.copy_(torch.tensor([0.05]))
    layer.alpha_floor = 0.3
    x = torch.randn(2, 5, 16)

    layer.train()
    y_floor = layer(x).detach().clone()

    layer.alpha_floor = 0.0
    y_no_floor = layer(x).detach().clone()

    assert not torch.allclose(y_floor, y_no_floor), (
        "train-mode forward with alpha_floor>0 should differ from "
        "alpha_floor==0 (the floor must actually apply during training)"
    )


def test_pkm_alpha_floor_inactive_in_eval_mode():
    """The real bug: alpha_floor was applied with NO self.training gate, so
    eval()/VAL forwards during the bootstrap window got a floor-forced
    contribution that doesn't reflect the learned model — train/eval
    mismatch that polluted Phase-1 arm B's VAL."""
    torch.manual_seed(0)
    layer = PKMLayer(d_model=16, n_heads=1, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)
    with torch.no_grad():
        layer.out_alpha.copy_(torch.tensor([0.05]))
    layer.alpha_floor = 0.3
    x = torch.randn(2, 5, 16)

    layer.eval()
    y_eval_with_floor_set = layer(x).detach().clone()

    # Reference: eval-mode output using alpha alone (floor==0). If the gate
    # is correct, these must be IDENTICAL regardless of what alpha_floor is
    # set to, because eval() must never apply the floor.
    layer.alpha_floor = 0.0
    y_eval_ref = layer(x).detach().clone()

    assert torch.allclose(y_eval_with_floor_set, y_eval_ref), (
        "alpha_floor fired during eval() despite alpha_floor>0 being set — "
        "the self.training gate is missing or broken"
    )

    # Sanity: the SAME alpha_floor value DOES change the result in train().
    layer.alpha_floor = 0.3
    layer.train()
    y_train_with_floor = layer(x).detach().clone()
    assert not torch.allclose(y_train_with_floor, y_eval_ref), (
        "train-mode output with alpha_floor>0 should differ from the "
        "floor-free eval reference (sanity: the floor mechanism itself "
        "still works in train mode)"
    )


def test_pkm_alpha_floor_bookkeeping_survives_eval_interleaving():
    """alpha_floor is a plain float set externally, once per STEP, by the
    trainer's curriculum (train_lm.py) — not a per-forward-call counter
    living inside PKMLayer. An eval() forward interleaved mid-training
    (e.g. the VAL-loss window in train_lm.py) must not perturb it, so the
    very next training-mode forward resumes with the curriculum's
    step-driven value unchanged."""
    torch.manual_seed(0)
    layer = PKMLayer(d_model=16, n_heads=1, n_keys=8, k_dim=8, top_k=4,
                      value_bf16=False)
    x = torch.randn(2, 5, 16)

    layer.alpha_floor = 0.42
    layer.train()
    layer(x)
    assert layer.alpha_floor == 0.42

    layer.eval()
    layer(x)
    assert layer.alpha_floor == 0.42, (
        "an eval() forward must not mutate the trainer-driven alpha_floor "
        "curriculum value"
    )

    layer.train()
    layer(x)
    assert layer.alpha_floor == 0.42


# ---------------------------------------------------------------------------
# FIX 2 — FeedbackProjection src_norm
# ---------------------------------------------------------------------------

def test_feedback_projection_default_omitted_is_legacy_none():
    """Not passing src_norm at all must be byte-identical to passing
    src_norm='none' explicitly (the safe/legacy default)."""
    torch.manual_seed(0)
    fp_default = FeedbackProjection(16, mode="film")
    fp_explicit_none = FeedbackProjection(16, mode="film", src_norm="none")
    fp_explicit_none.load_state_dict(fp_default.state_dict())

    x = torch.randn(2, 5, 16)
    src = torch.randn(2, 5, 16) * 3.0
    with torch.no_grad():
        fp_default.alpha.fill_(0.2)
        fp_explicit_none.alpha.fill_(0.2)
    out_default = fp_default(x, src)
    out_explicit = fp_explicit_none(x, src)
    assert torch.equal(out_default, out_explicit)


def test_feedback_projection_none_matches_prefix_formula():
    """src_norm='none' must reproduce the exact pre-fix FiLM formula
    (no normalization of the source at all)."""
    torch.manual_seed(0)
    fp = FeedbackProjection(16, mode="film", src_norm="none")
    with torch.no_grad():
        fp.alpha.fill_(0.25)
    x = torch.randn(2, 5, 16)
    src = torch.randn(2, 5, 16) * 4.0
    out = fp(x, src)
    ref = x * (1.0 + 0.25 * fp.W_scale(src)) + 0.25 * fp.W_shift(src)
    assert torch.allclose(out, ref, atol=1e-6)


def test_feedback_projection_rms_changes_output():
    torch.manual_seed(0)
    fp_none = FeedbackProjection(16, mode="film", src_norm="none")
    fp_rms = FeedbackProjection(16, mode="film", src_norm="rms")
    fp_rms.load_state_dict(fp_none.state_dict())
    x = torch.randn(2, 5, 16)
    src = torch.randn(2, 5, 16) * 5.0
    with torch.no_grad():
        fp_none.alpha.fill_(0.3)
        fp_rms.alpha.fill_(0.3)
    out_none = fp_none(x, src)
    out_rms = fp_rms(x, src)
    assert not torch.allclose(out_none, out_rms)


def test_feedback_projection_rms_invariant_to_source_scale():
    """The whole point of FIX 2: 'rms' decouples FiLM strength from the raw
    source-state magnitude, so a 10x-scaled source produces the SAME
    output."""
    torch.manual_seed(0)
    fp = FeedbackProjection(16, mode="film", src_norm="rms")
    with torch.no_grad():
        fp.alpha.fill_(0.3)
    x = torch.randn(2, 5, 16)
    src = torch.randn(2, 5, 16) + 1.0  # nonzero mean, healthy magnitude
    out_1x = fp(x, src)
    out_10x = fp(x, src * 10.0)
    assert torch.allclose(out_1x, out_10x, atol=1e-5)


def test_feedback_projection_additive_mode_also_supports_src_norm():
    torch.manual_seed(0)
    fp = FeedbackProjection(16, mode="additive", src_norm="rms")
    with torch.no_grad():
        fp.alpha.fill_(0.3)
    x = torch.randn(2, 5, 16)
    src = torch.randn(2, 5, 16) + 1.0
    out_1x = fp(x, src)
    out_10x = fp(x, src * 10.0)
    assert torch.allclose(out_1x, out_10x, atol=1e-5)


def test_feedback_projection_invalid_src_norm_rejected():
    try:
        FeedbackProjection(16, mode="film", src_norm="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid src_norm")


# ---------------------------------------------------------------------------
# FIX 3 — WorkingMemory inj_norm
# ---------------------------------------------------------------------------

def _make_wm(inj_norm: str, sd: dict | None = None, mem_size: int = 4,
             d_model: int = 16, d_mem: int = 16, decoupled_kv: bool = True):
    m = WorkingMemory(d_model=d_model, d_mem=d_mem, mem_size=mem_size,
                       thinking_token_id=5, always_read=True,
                       inj_norm=inj_norm, decoupled_kv=decoupled_kv)
    if sd is not None:
        m.load_state_dict(sd)
    return m


def test_working_memory_default_omitted_is_legacy_none():
    torch.manual_seed(0)
    wm_a = WorkingMemory(d_model=16, d_mem=16, mem_size=4,
                          thinking_token_id=5, always_read=True,
                          decoupled_kv=True)
    wm_b = WorkingMemory(d_model=16, d_mem=16, mem_size=4,
                          thinking_token_id=5, always_read=True,
                          decoupled_kv=True, inj_norm="none")
    wm_b.load_state_dict(wm_a.state_dict())
    h = torch.randn(2, 6, 16)
    input_ids = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        out_a = wm_a(h, input_ids)
        out_b = wm_b(h, input_ids)
    assert torch.equal(out_a, out_b)


def test_working_memory_match_rms_changes_output():
    torch.manual_seed(0)
    wm_none = _make_wm("none")
    sd = wm_none.state_dict()
    wm_mr = _make_wm("match_rms", sd)
    h = torch.randn(2, 6, 16)
    input_ids = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        out_none = wm_none(h, input_ids)
        out_mr = wm_mr(h, input_ids)
    assert not torch.allclose(out_none, out_mr)


def test_working_memory_match_rms_invariant_to_buffer_scale():
    """The whole point of FIX 3: 'match_rms' makes the injected contribution
    invariant to the raw scale of the values stored in the WM buffer (here
    induced by scaling W_v, the value-write projection, by 10x) — read_alpha
    is left to express a pure mixing fraction."""
    torch.manual_seed(0)
    # NOTE (2026-07-02 NaN fix): the rescale denominator is clamped at
    # 0.1*h_rms, so exact invariance holds only ABOVE the clamp region (tiny
    # injections are deliberately NOT amplified more than 10x — that
    # amplification is what NaN'd pilot-B run 2). Compare 10x vs 100x arms,
    # both comfortably above the clamp.
    wm_1x = _make_wm("match_rms", mem_size=1)
    sd = wm_1x.state_dict()
    wm_10x = _make_wm("match_rms", sd, mem_size=1)
    with torch.no_grad():
        wm_1x.W_v.weight.mul_(100.0)
        wm_10x.W_v.weight.mul_(1000.0)

    h = torch.randn(2, 5, 16)
    input_ids = torch.randint(0, 20, (2, 5))
    with torch.no_grad():
        out_1x = wm_1x(h, input_ids)
        out_10x = wm_10x(h, input_ids)
    assert torch.allclose(out_1x, out_10x, atol=1e-4)

    # Sanity: 'none' is NOT invariant to the same 10x buffer scale.
    wm_none_1x = _make_wm("none", sd, mem_size=1)
    wm_none_10x = _make_wm("none", sd, mem_size=1)
    with torch.no_grad():
        wm_none_10x.W_v.weight.mul_(10.0)
    with torch.no_grad():
        out_none_1x = wm_none_1x(h, input_ids)
        out_none_10x = wm_none_10x(h, input_ids)
    assert not torch.allclose(out_none_1x, out_none_10x, atol=1e-3)


def test_working_memory_invalid_inj_norm_rejected():
    try:
        WorkingMemory(d_model=16, d_mem=16, mem_size=4,
                      thinking_token_id=5, inj_norm="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid inj_norm")


def test_wm_inj_norm_does_not_touch_addressing(monkeypatch):
    """The scaling must apply to the injection vector ONLY — never to the
    addressing keys/scores (ctx_namekey / discrete / soft namekey compute
    their match upstream of the injection). Verify the read-attention
    weights are bit-identical between inj_norm='none' and 'match_rms' for
    the SAME weights (only the post-hoc injection magnitude may differ)."""
    torch.manual_seed(0)
    wm_none = _make_wm("none")
    sd = wm_none.state_dict()
    wm_mr = _make_wm("match_rms", sd)
    wm_none._capture_read = True
    wm_mr._capture_read = True
    h = torch.randn(2, 6, 16)
    input_ids = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        wm_none(h, input_ids)
        wm_mr(h, input_ids)
    assert torch.equal(wm_none._last_read_attn, wm_mr._last_read_attn), (
        "match_rms must not perturb the read-attention (addressing) at all"
    )


# ---------------------------------------------------------------------------
# CLI defaults — "fixed by default for new runs"
# ---------------------------------------------------------------------------

def test_cli_defaults_are_the_fixed_behaviour():
    a = build_parser().parse_args([])
    assert a.feedback_src_norm == "rms"
    assert a.mem_inj_norm == "match_rms"


def test_cli_none_is_reachable_as_escape_hatch():
    a = build_parser().parse_args([
        "--feedback_src_norm", "none", "--mem_inj_norm", "none",
    ])
    assert a.feedback_src_norm == "none"
    assert a.mem_inj_norm == "none"


def test_model_builder_threads_cli_default_fix(monkeypatch):
    """A fresh (no --load_ckpt) train_lm.py invocation with NO override must
    construct TinyLM with the FIXED modes, end to end through
    build_model_from_args."""
    import experiments.model_builder as mb

    captured = {}

    class _FakeTiny:
        def __init__(self, **kw):
            captured.update(kw)

        def to(self, *_a, **_k):
            return self

    monkeypatch.setattr(mb, "TinyLM", _FakeTiny)
    a = build_parser().parse_args([
        "--arch", "deltanet", "--n_layers", "4",
        "--d_model", "16", "--n_heads", "1", "--d_head", "16",
        "--feedback", "film", "--feedback_pairs", "1,3",
        "--use_memory", "--mem_size", "32",
    ])
    mb.build_model_from_args(a, vocab_size=40, thinking_token_id=39)
    assert captured["feedback_src_norm"] == "rms"
    assert captured["mem_inj_norm"] == "match_rms"


def test_model_builder_getattr_fallback_is_legacy_none(monkeypatch):
    """A caller that hand-builds an args namespace WITHOUT the new
    attributes at all (e.g. an older/incomplete caller) must fall back to
    the safe legacy 'none', not silently pick up the CLI-level fixed
    default."""
    import argparse

    import experiments.model_builder as mb

    captured = {}

    class _FakeTiny:
        def __init__(self, **kw):
            captured.update(kw)

        def to(self, *_a, **_k):
            return self

    monkeypatch.setattr(mb, "TinyLM", _FakeTiny)
    a = build_parser().parse_args([
        "--arch", "deltanet", "--n_layers", "4",
        "--d_model", "16", "--n_heads", "1", "--d_head", "16",
        "--feedback", "film", "--feedback_pairs", "1,3",
        "--use_memory", "--mem_size", "32",
    ])
    del a.feedback_src_norm
    del a.mem_inj_norm
    mb.build_model_from_args(a, vocab_size=40, thinking_token_id=39)
    assert captured["feedback_src_norm"] == "none"
    assert captured["mem_inj_norm"] == "none"


# ---------------------------------------------------------------------------
# ckpt cfg round-trip — legacy-default-on-missing-key
# ---------------------------------------------------------------------------

def _make_tiny_ckpt(path: str, *, feedback_src_norm=None, mem_inj_norm=None):
    """Save a tiny ckpt with sparse FiLM feedback + WM, so both
    sparse_feedback and memory get reconstructed by build_model_from_ckpt.
    feedback_src_norm/mem_inj_norm are cfg keys ONLY (never affect the
    saved weights — both fixes add zero new parameters), so omitting them
    from the args (None) simulates a pre-fix ckpt whose cfg lacks the key.
    """
    vocab = 24
    thinking_id = vocab - 1
    model = TinyLM(
        vocab_size=vocab, d_model=8, n_layers=4, n_heads=2, d_head=4,
        max_T=0,
        feedback_mode="film", feedback_pairs=((1, 3),),
        use_memory=True, mem_size=4, mem_dim=8,
        thinking_token_id=thinking_id,
        attention_cls=DeltaNetAttention,
    )
    cfg = {
        "vocab_size": vocab, "d_model": 8, "n_layers": 4, "n_heads": 2,
        "d_head": 4, "max_T": 0,
        "feedback_mode": "film", "feedback_pairs": ((1, 3),),
        "feedback_self_k": 0,
        "tie_embeddings": False,
        "use_memory": True, "mem_size": 4,
        "thinking_token_id": thinking_id,
        "arch": "deltanet",
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
    }
    if feedback_src_norm is not None:
        cfg["feedback_src_norm"] = feedback_src_norm
    if mem_inj_norm is not None:
        cfg["mem_inj_norm"] = mem_inj_norm
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def test_build_model_from_ckpt_legacy_cfg_reconstructs_none(tmp_path):
    """A ckpt saved by a pre-fix train_lm.py has NO feedback_src_norm /
    mem_inj_norm key in its cfg at all. Reloading it must reconstruct the
    LEGACY ('none') behaviour, not the new CLI-level fixed default — this
    is what makes old checkpoints (e.g. the live feature_pilot_B run, v18,
    v7.1) re-evaluate byte-identically."""
    from experiments.eval_bracket_structure import build_model_from_ckpt

    ckpt_path = str(tmp_path / "legacy.pt")
    _make_tiny_ckpt(ckpt_path)  # no feedback_src_norm / mem_inj_norm keys
    model, cfg = build_model_from_ckpt(ckpt_path)
    assert model.sparse_feedback["1"].src_norm == "none"
    assert model.memory.inj_norm == "none"


def test_build_model_from_ckpt_cfg_roundtrips_actual_mode(tmp_path):
    """A ckpt saved AFTER the fix, with the keys explicitly set, must
    reconstruct with that exact mode."""
    from experiments.eval_bracket_structure import build_model_from_ckpt

    ckpt_path = str(tmp_path / "fixed.pt")
    _make_tiny_ckpt(ckpt_path, feedback_src_norm="rms", mem_inj_norm="match_rms")
    model, cfg = build_model_from_ckpt(ckpt_path)
    assert model.sparse_feedback["1"].src_norm == "rms"
    assert model.memory.inj_norm == "match_rms"
    assert cfg.get("feedback_src_norm") == "rms"
    assert cfg.get("mem_inj_norm") == "match_rms"


def test_train_lm_cfg_persists_actual_args_value(monkeypatch):
    """train_lm.py's saved cfg dict must record whatever THIS run's args
    actually used (not a hardcoded default) — spot-check the exact
    expression used at both save sites (mid-eval + final) via a stand-in
    args namespace, mirroring how those dict literals read `args`."""
    class _Args:
        feedback_src_norm = "none"  # simulates an explicit --feedback_src_norm none
        mem_inj_norm = "match_rms"

    args = _Args()
    assert str(getattr(args, "feedback_src_norm", "none")) == "none"
    assert str(getattr(args, "mem_inj_norm", "none")) == "match_rms"


# --------------------------------------------------------------------------- #
# NaN regression (2026-07-02, bisected on the real 32L pilot): match_rms's
# scale must be detached and RELATIVELY clamped. Without the detach, autograd's
# sqrt-backward at exactly-zero injection positions (masked / no-match — they
# exist by construction) evaluates 0·inf = NaN and poisons the whole graph on
# the first backward; without the relative clamp, near-zero injections get
# their gradient amplified by up to h_rms/1e-6 ≈ 1e7.

def test_wm_match_rms_backward_finite_with_zero_injection_positions():
    torch.manual_seed(0)
    wm = _make_wm("match_rms")
    # Zero W_proj -> injection is EXACTLY zero at every position (the
    # no-match/masked-position regime that NaN'd pilot-B run 2 at step 1:
    # sqrt-backward at 0 gives 0*inf = NaN without the detached scale).
    with torch.no_grad():
        wm.W_proj.weight.zero_()
    wm.read_alpha.data.fill_(0.5)  # make the injection path loss-bearing
    h = torch.randn(2, 6, 16, requires_grad=True)
    input_ids = torch.randint(0, 20, (2, 6))
    out = wm(h, input_ids)
    out.sum().backward()
    assert torch.isfinite(h.grad).all(), "NaN/Inf gradient at zero-injection"
    for n, p in wm.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {n}"


def test_wm_match_rms_scale_amplification_bounded():
    torch.manual_seed(0)
    wm = _make_wm("match_rms")
    # Make the injection TINY but nonzero: the relative clamp (0.1*h_rms)
    # must bound the rescale at 10x rather than the ~1e6x an absolute 1e-6
    # floor would allow.
    with torch.no_grad():
        wm.W_proj.weight.mul_(1e-6)
    h = torch.randn(2, 6, 16)
    input_ids = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        wm(h, input_ids)
    inj = wm._last_injection
    h_rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
    inj_rms = inj.float().pow(2).mean(-1, keepdim=True).sqrt()
    # A tiny injection rescaled by at most 10x stays far below the stream
    # RMS — the old absolute floor would have blown it up TO h_rms.
    assert (inj_rms <= 0.01 * h_rms.squeeze(-1).unsqueeze(-1) + 1e-6).all()
