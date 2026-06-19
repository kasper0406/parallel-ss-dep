"""Tests for the `--embed_lr_mult` / `--embed_optimizer {adam,rownorm}` wiring
in `optim_utils.build_optimizer`, plus the `RowNormEmbed` dualizer math.

Fairness / additivity guarantees pinned here (the whole point of the A/B):
  1. default (embed_lr_mult=1.0, embed_optimizer="adam") is BYTE-IDENTICAL to
     the legacy shared-LR path — embed/lm_head stay in the shared AdamW group,
     no extra optimizer is created.
  2. embed_lr_mult>1 carves embed/lm_head into their own AdamW group at lr*mult;
     every NON-embed param update is byte-identical to the baseline.
  3. embed_optimizer="rownorm" routes embed/lm_head to a SEPARATE RowNormEmbed
     optimizer; non-embed updates byte-identical; embed updates differ.
  4. the embed group excludes PKM value tables (they keep their own lr path).
  5. non-default embed treatment requires --optimizer muon (loud reject).

CPU-only + a tiny module so it runs in CI without a GPU (fake grads, no forward).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.optim_utils import build_optimizer
from experiments.embed_optim import RowNormEmbed


_OPT_KW = dict(optimizer="muon", lr=1e-3, lr_muon=5e-3, alpha_wd=0.0,
               steps=100, wd=0.01, lr_schedule="wsd", warmup_steps=10,
               verbose=False)


class _Tiny(nn.Module):
    """Minimal module exercising every optimizer route: an embedding table, an
    untied lm_head, a 2D hidden matrix (-> Muon), and a 1D norm (-> AdamW)."""

    def __init__(self, vocab=64, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)


def _model(seed=0):
    torch.manual_seed(seed)
    return _Tiny()


def _name_of(model):
    return {id(p): n for n, p in model.named_parameters()}


def _set_grads(model, gen):
    for _, p in model.named_parameters():
        p.grad = torch.randn(p.shape, generator=gen) * 0.01


def _opt_named_groups(model, opt):
    name_of = _name_of(model)
    return [sorted(name_of[id(p)] for p in g["params"]) for g in opt.param_groups]


def _step_all(model, opts, gen):
    _set_grads(model, gen)
    for o in opts:
        o.step()


def _params_by_name(model):
    return {n: p for n, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# 1. default == explicit adam/mult-1.0 == byte-identical, no extra optimizer
# ---------------------------------------------------------------------------
def test_default_is_two_optimizers():
    m = _model()
    opts, scheds = build_optimizer(m, **_OPT_KW)
    assert len(opts) == 2 and len(scheds) == 2  # matrix + AdamW only


def test_default_matches_explicit_adam_mult1_byte_identical():
    mA, mB = _model(0), _model(0)
    optsA, _ = build_optimizer(mA, **_OPT_KW)                       # default
    optsB, _ = build_optimizer(mB, embed_lr_mult=1.0,
                               embed_optimizer="adam", **_OPT_KW)
    gA = torch.Generator().manual_seed(123)
    gB = torch.Generator().manual_seed(123)
    for _ in range(4):
        _step_all(mA, optsA, gA)
        _step_all(mB, optsB, gB)
    pB = _params_by_name(mB)
    max_d = max((pA - pB[n]).abs().max().item()
                for n, pA in mA.named_parameters())
    assert max_d == 0.0, f"default vs explicit adam/mult1 drifted: {max_d}"


# ---------------------------------------------------------------------------
# 2. embed_lr_mult: own AdamW group at lr*mult; non-embed byte-identical
# ---------------------------------------------------------------------------
def test_embed_lr_mult_creates_scaled_group():
    m = _model()
    opts, _ = build_optimizer(m, embed_lr_mult=5.0, **_OPT_KW)
    assert len(opts) == 2  # still matrix + AdamW (no separate optimizer)
    adamw = opts[1]
    name_of = _name_of(m)
    # group base lr is captured in initial_lr (the scheduler rescales "lr").
    scaled = [g for g in adamw.param_groups
              if abs(g.get("initial_lr", g["lr"]) - _OPT_KW["lr"] * 5.0) < 1e-12]
    assert len(scaled) == 1, "expected exactly one embed group at lr*5"
    got = sorted(name_of[id(p)] for p in scaled[0]["params"])
    assert got == ["embed.weight", "lm_head.weight"], got


def test_embed_lr_mult_only_moves_embeds():
    mA, mB = _model(0), _model(0)
    optsA, _ = build_optimizer(mA, **_OPT_KW)                       # baseline
    optsB, _ = build_optimizer(mB, embed_lr_mult=5.0, **_OPT_KW)
    gA = torch.Generator().manual_seed(7)
    gB = torch.Generator().manual_seed(7)
    for _ in range(3):
        _step_all(mA, optsA, gA)
        _step_all(mB, optsB, gB)
    pB = _params_by_name(mB)
    embed_names = {"embed.weight", "lm_head.weight"}
    for n, pA in mA.named_parameters():
        d = (pA - pB[n]).abs().max().item()
        if n in embed_names:
            assert d > 0.0, f"{n} should move more at 5x lr"
        else:
            assert d == 0.0, f"non-embed {n} must match baseline exactly, got {d}"


# ---------------------------------------------------------------------------
# 3. rownorm: separate optimizer; non-embed byte-identical; embed differs
# ---------------------------------------------------------------------------
def test_rownorm_adds_separate_optimizer():
    m = _model()
    opts, scheds = build_optimizer(m, embed_optimizer="rownorm", **_OPT_KW)
    assert len(opts) == 3 and len(scheds) == 3
    assert isinstance(opts[2], RowNormEmbed)
    name_of = _name_of(m)
    rn_names = sorted(name_of[id(p)] for g in opts[2].param_groups
                      for p in g["params"])
    assert rn_names == ["embed.weight", "lm_head.weight"], rn_names
    # AdamW (opts[1]) must NOT contain embed/lm_head anymore
    adamw_names = {n for grp in _opt_named_groups(m, opts[1]) for n in grp}
    assert "embed.weight" not in adamw_names
    assert "lm_head.weight" not in adamw_names


def test_rownorm_only_changes_embeds():
    mA, mB = _model(0), _model(0)
    optsA, _ = build_optimizer(mA, **_OPT_KW)                       # baseline adam
    optsB, _ = build_optimizer(mB, embed_optimizer="rownorm", **_OPT_KW)
    gA = torch.Generator().manual_seed(11)
    gB = torch.Generator().manual_seed(11)
    for _ in range(3):
        _step_all(mA, optsA, gA)
        _step_all(mB, optsB, gB)
    pB = _params_by_name(mB)
    embed_names = {"embed.weight", "lm_head.weight"}
    for n, pA in mA.named_parameters():
        d = (pA - pB[n]).abs().max().item()
        if n in embed_names:
            assert d > 0.0, f"{n} should differ under rownorm"
        else:
            assert d == 0.0, f"non-embed {n} must match baseline exactly, got {d}"


# ---------------------------------------------------------------------------
# 4. RowNormEmbed math: each row's applied step has RMS == lr (wd=0)
# ---------------------------------------------------------------------------
def test_rownorm_per_row_rms_equals_lr():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(5, 8))
    p0 = p.detach().clone()
    g = torch.randn(5, 8) * 3.0  # arbitrary scale; row-norm should cancel it
    p.grad = g.clone()
    lr = 0.1
    opt = RowNormEmbed([p], lr=lr, momentum=0.95, nesterov=True, weight_decay=0.0)
    opt.step()
    delta = (p0 - p.detach())  # = lr * row-normalized update
    row_rms = delta.pow(2).mean(dim=1).sqrt()
    assert torch.allclose(row_rms, torch.full((5,), lr), atol=1e-5), row_rms


def test_rownorm_scale_invariant():
    """Scaling the gradient by a constant must not change the update direction
    or magnitude (the row-RMS normalization removes raw-gradient scale)."""
    torch.manual_seed(1)
    base_g = torch.randn(4, 6)
    deltas = []
    for s in (1.0, 100.0):
        p = nn.Parameter(torch.zeros(4, 6))
        p.grad = base_g * s
        opt = RowNormEmbed([p], lr=0.05, momentum=0.0, nesterov=False)
        opt.step()
        deltas.append(p.detach().clone())
    assert torch.allclose(deltas[0], deltas[1], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. embed group excludes PKM value tables (they keep their own lr path)
# ---------------------------------------------------------------------------
def test_embed_group_excludes_pkm_value_table():
    m = _model()
    pkm = nn.Module()
    pkm.values = nn.ModuleList([nn.Embedding(16, 8)])  # pkm_layer.values.0.weight
    m.add_module("pkm_layer", pkm)
    opts, _ = build_optimizer(m, embed_lr_mult=5.0, **_OPT_KW)
    name_of = _name_of(m)
    scaled = [g for g in opts[1].param_groups
              if abs(g.get("initial_lr", g["lr"]) - _OPT_KW["lr"] * 5.0) < 1e-12][0]
    got = sorted(name_of[id(p)] for p in scaled["params"])
    assert "pkm_layer.values.0.weight" not in got, got
    assert got == ["embed.weight", "lm_head.weight"], got


# ---------------------------------------------------------------------------
# 6. guards
# ---------------------------------------------------------------------------
def test_nondefault_embed_requires_muon():
    for kw in (dict(embed_lr_mult=5.0), dict(embed_optimizer="rownorm")):
        for bad_opt in ("adamw", "soap"):
            m = _model()
            try:
                build_optimizer(m, **{**_OPT_KW, "optimizer": bad_opt, **kw})
            except ValueError as e:
                assert "requires --optimizer muon" in str(e)
            else:
                raise AssertionError(f"expected ValueError for {kw} + {bad_opt}")


def test_unknown_embed_optimizer_raises():
    m = _model()
    try:
        build_optimizer(m, **{**_OPT_KW, "embed_optimizer": "bogus"})
    except ValueError as e:
        assert "unknown embed_optimizer" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown embed_optimizer")
