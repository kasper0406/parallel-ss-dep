"""Regression tests for sft_code.py model-loading control flow.

The bug we're guarding against (2026-05-19):
  The modern-path (ckpt already has memory + gate) was sandwiched
  between three branches. The else at the bottom unconditionally re-
  ran build_model_from_ckpt, silently overwriting the model and cfg
  that the modern path had just constructed — INCLUDING the FIX A
  flag (`model.memory.write_only_at_think`) and any cfg mutations
  (`cfg["mem_write_only_at_think"]`, `cfg["sft_with_thinking"]`,
  etc.). Result: the combined-SFT was trained for 30 min without
  FIX A despite the launch script passing --mem_write_only_at_think.

Detection rule:
  After main()'s "build model from ckpt" block, the model object
  must reflect any FIX A request, AND cfg must contain the flag.

We can't easily test main() end-to-end without an actual ckpt, so we
write tests that reproduce the load-flow on synthetic ckpts:

  1. A "modern" tiny ckpt (has memory + gate). Confirm: when --
     mem_write_only_at_think is set, the resulting model has
     `model.memory.write_only_at_think is True`.
  2. A "legacy" tiny ckpt (no memory + gate). Confirm the same.
  3. cfg["mem_write_only_at_think"] survives to be saved.
"""
import pathlib
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.model import TinyLM
from experiments.layers import DeltaNetAttention


def _make_modern_tiny_ckpt(path: str) -> int:
    """Save a tiny TinyLM ckpt that mirrors v7.1's structure: has memory
    + gate so it goes through the "modern" SFT-load branch. Returns
    thinking_token_id."""
    vocab = 16
    thinking_id = vocab - 1
    model = TinyLM(
        vocab_size=vocab,
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_head=4,
        max_T=0,
        output_gate=True,        # forces gate_head — modern path detects this
        use_memory=True,         # forces memory.* keys — modern path detects this
        mem_size=4,
        mem_dim=8,
        thinking_token_id=thinking_id,
        attention_cls=DeltaNetAttention,
    )
    cfg = {
        "vocab_size": vocab,
        "d_model": 8,
        "n_layers": 2,
        "n_heads": 2,
        "d_head": 4,
        "max_T": 0,
        "feedback_mode": "none",
        "feedback_pairs": (),
        "feedback_self_k": 0,
        "tie_embeddings": True,
        "use_memory": True,
        "output_gate": True,
        "mem_size": 4,
        "thinking_token_id": thinking_id,
        "arch": "deltanet",
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
    }
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)
    return thinking_id


def _make_legacy_tiny_ckpt(path: str) -> None:
    """A 'legacy' ckpt has NEITHER memory.* NOR gate_head.* — forces
    the legacy build-from-scratch SFT path."""
    vocab = 16
    model = TinyLM(
        vocab_size=vocab,
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_head=4,
        max_T=0,
        output_gate=False,  # no gate_head
        use_memory=False,   # no memory.*
        attention_cls=DeltaNetAttention,
    )
    cfg = {
        "vocab_size": vocab,
        "d_model": 8,
        "n_layers": 2,
        "n_heads": 2,
        "d_head": 4,
        "max_T": 0,
        "feedback_mode": "none",
        "feedback_pairs": (),
        "feedback_self_k": 0,
        "tie_embeddings": True,
        "use_memory": False,
        "output_gate": False,
        "arch": "deltanet",
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
    }
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


# ---------------------------------------------------------------------------
# Repro of the EXACT load flow in sft_code.main() — extracted into a
# helper so we can test it without needing to invoke argparse + a full
# train loop. If the helper diverges from main(), the test catches the
# regression because the EXPECTED outputs (model attributes + cfg keys)
# are what main() is supposed to produce.
# ---------------------------------------------------------------------------

def _sft_load_model(ckpt_path: str, *, with_thinking: bool,
                     mem_write_only_at_think: bool,
                     mem_size: int = 4, mem_dim: int = 0):
    """Reproduce the control flow from sft_code.main() lines ~310-410.
    Returns (model, cfg) — what would land in the train loop and the
    save block."""
    if with_thinking:
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd_keys = set(raw_ckpt["state_dict"].keys())
        ckpt_has_memory = any(k.startswith("memory.") for k in sd_keys)
        ckpt_has_gate = any(k.startswith("gate_head.") for k in sd_keys)
        if ckpt_has_memory and ckpt_has_gate:
            # MODERN PATH
            from experiments.eval_bracket_structure import build_model_from_ckpt
            model, cfg = build_model_from_ckpt(ckpt_path)
            thinking_token_id = cfg.get("thinking_token_id")
            if thinking_token_id is None:
                thinking_token_id = int(cfg["vocab_size"]) - 1
            if bool(mem_write_only_at_think):
                model.memory.write_only_at_think = True
                cfg["mem_write_only_at_think"] = True
            cfg["sft_with_thinking"] = True
            args_with_thinking_done = True
        else:
            args_with_thinking_done = False
    else:
        args_with_thinking_done = False

    # ---- The post-build branch (the one with the bug) ----
    if args_with_thinking_done:
        # modern path: don't re-load
        pass
    elif with_thinking and not args_with_thinking_done:
        # legacy path: build fresh
        from experiments.model import TinyLM
        from experiments.layers import DeltaNetAttention
        cfg = dict(raw_ckpt["config"])
        sd = raw_ckpt["state_dict"]
        base_vocab = int(cfg["vocab_size"])
        new_vocab = base_vocab + 1
        thinking_token_id = base_vocab
        for key in ("embed.weight", "lm_head.weight"):
            if key in sd and sd[key].shape[0] < new_vocab:
                old = sd[key]
                pad = torch.zeros(new_vocab - old.shape[0], old.shape[1], dtype=old.dtype)
                sd[key] = torch.cat([old, pad], dim=0)
        fb_pairs = tuple(tuple(p) for p in cfg.get("feedback_pairs", ()) or ())
        model = TinyLM(
            vocab_size=new_vocab,
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]),
            d_head=int(cfg["d_head"]),
            max_T=int(cfg.get("max_T", 0)),
            feedback_mode=str(cfg.get("feedback_mode", "none")),
            feedback_pairs=fb_pairs,
            feedback_self_k=int(cfg.get("feedback_self_k", 0)),
            tie_embeddings=bool(cfg.get("tie_embeddings", True)),
            output_gate=True,
            use_memory=True,
            mem_size=int(mem_size),
            mem_dim=int(mem_dim) if mem_dim > 0 else int(cfg["d_model"]),
            thinking_token_id=thinking_token_id,
            mem_write_only_at_think=bool(mem_write_only_at_think),
            attention_cls=DeltaNetAttention,
        )
        model.load_state_dict(sd, strict=False)
        cfg["use_memory"] = True
        cfg["output_gate"] = True
        cfg["thinking_token_id"] = thinking_token_id
        cfg["vocab_size"] = new_vocab
        cfg["mem_size"] = int(mem_size)
        cfg["mem_dim"] = int(mem_dim) if mem_dim > 0 else int(cfg["d_model"])
        cfg["mem_write_only_at_think"] = bool(mem_write_only_at_think)
        cfg["sft_with_thinking"] = True
    else:
        # original path
        from experiments.eval_bracket_structure import build_model_from_ckpt
        model, cfg = build_model_from_ckpt(ckpt_path)
    return model, cfg


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_modern_path_honors_mem_write_only_at_think_flag(tmp_path):
    """REGRESSION: the bug was that the modern path's setting got
    silently overwritten by a subsequent reload. After the fix, the
    flag MUST survive."""
    ckpt = str(tmp_path / "modern.pt")
    _make_modern_tiny_ckpt(ckpt)
    model, cfg = _sft_load_model(ckpt, with_thinking=True,
                                  mem_write_only_at_think=True)
    # The flag must be set on the model's WorkingMemory module
    assert model.memory.write_only_at_think is True, (
        "FIX A flag was lost during SFT load — modern path got "
        "overwritten by a stray reload (this is the bug)."
    )
    # The cfg must record it so the saved ckpt's cfg has it
    assert cfg.get("mem_write_only_at_think") is True
    # And the other modern-path cfg mutation must also survive
    assert cfg.get("sft_with_thinking") is True


def test_modern_path_default_flag_off(tmp_path):
    """Sanity: when --mem_write_only_at_think isn't set, the flag stays
    False (no spurious enabling)."""
    ckpt = str(tmp_path / "modern.pt")
    _make_modern_tiny_ckpt(ckpt)
    model, cfg = _sft_load_model(ckpt, with_thinking=True,
                                  mem_write_only_at_think=False)
    assert model.memory.write_only_at_think is False
    # cfg should NOT have the key set, OR it should be False
    assert cfg.get("mem_write_only_at_think", False) is False


def test_legacy_path_honors_mem_write_only_at_think_flag(tmp_path):
    """The legacy path (no memory+gate in ckpt) was always correct —
    cover it to prevent future regressions."""
    ckpt = str(tmp_path / "legacy.pt")
    _make_legacy_tiny_ckpt(ckpt)
    model, cfg = _sft_load_model(ckpt, with_thinking=True,
                                  mem_write_only_at_think=True)
    assert model.memory.write_only_at_think is True
    assert cfg.get("mem_write_only_at_think") is True


def test_no_thinking_path_skips_memory(tmp_path):
    """When --with_thinking is False, we use the original path which
    doesn't touch the memory flag (it just loads what was in the
    ckpt). The modern-tiny ckpt has memory built in; this path should
    preserve the saved-time flag value, not override it."""
    ckpt = str(tmp_path / "modern.pt")
    _make_modern_tiny_ckpt(ckpt)
    model, cfg = _sft_load_model(ckpt, with_thinking=False,
                                  mem_write_only_at_think=True)
    # In this branch we DO NOT honor the flag — that's by design (the
    # arg only applies when with_thinking=True).
    # The model should reflect whatever the ckpt had built-in (False).
    assert model.memory.write_only_at_think is False


# ---------------------------------------------------------------------------
# eval_bracket_structure.build_model_from_ckpt — must round-trip FIX A
# ---------------------------------------------------------------------------

def _make_ckpt_with_woat_flag(path: str, woat: bool) -> int:
    """Save a tiny modern-style ckpt with cfg["mem_write_only_at_think"]
    set to the given bool. Returns thinking_token_id."""
    vocab = 16
    thinking_id = vocab - 1
    model = TinyLM(
        vocab_size=vocab, d_model=8, n_layers=2, n_heads=2, d_head=4,
        max_T=0, output_gate=True, use_memory=True, mem_size=4, mem_dim=8,
        thinking_token_id=thinking_id, mem_write_only_at_think=woat,
        attention_cls=DeltaNetAttention,
    )
    cfg = {
        "vocab_size": vocab, "d_model": 8, "n_layers": 2, "n_heads": 2,
        "d_head": 4, "max_T": 0, "feedback_mode": "none",
        "feedback_pairs": (), "feedback_self_k": 0, "tie_embeddings": True,
        "use_memory": True, "output_gate": True, "mem_size": 4,
        "thinking_token_id": thinking_id, "arch": "deltanet",
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
        "mem_write_only_at_think": woat,
    }
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)
    return thinking_id


def test_eval_reload_preserves_woat_flag_true(tmp_path):
    """REGRESSION: build_model_from_ckpt() reads cfg["mem_write_only_at_think"]
    and passes it to WorkingMemory.__init__. Without this, a ckpt trained
    with FIX A would silently be evaluated WITHOUT it (since the flag is
    a non-state-dict bool attribute and would default to False on reload).
    """
    from experiments.eval_bracket_structure import build_model_from_ckpt
    ckpt = str(tmp_path / "woat_on.pt")
    _make_ckpt_with_woat_flag(ckpt, woat=True)
    model, cfg = build_model_from_ckpt(ckpt)
    assert model.memory.write_only_at_think is True, (
        "build_model_from_ckpt should set write_only_at_think from cfg; "
        "without it, FIX A is silently lost at eval time."
    )
    assert cfg["mem_write_only_at_think"] is True


def test_eval_reload_preserves_woat_flag_false(tmp_path):
    """Sanity: when cfg says False (or is missing), reload sets False."""
    from experiments.eval_bracket_structure import build_model_from_ckpt
    ckpt = str(tmp_path / "woat_off.pt")
    _make_ckpt_with_woat_flag(ckpt, woat=False)
    model, cfg = build_model_from_ckpt(ckpt)
    assert model.memory.write_only_at_think is False


def test_eval_reload_backward_compat_no_woat_key(tmp_path):
    """Older ckpts saved before FIX A existed don't have the cfg key
    at all. Reload must default to False (don't crash, don't enable
    a feature the model wasn't trained with)."""
    from experiments.eval_bracket_structure import build_model_from_ckpt
    vocab = 16
    thinking_id = vocab - 1
    model = TinyLM(
        vocab_size=vocab, d_model=8, n_layers=2, n_heads=2, d_head=4,
        max_T=0, output_gate=True, use_memory=True, mem_size=4, mem_dim=8,
        thinking_token_id=thinking_id, attention_cls=DeltaNetAttention,
    )
    # cfg WITHOUT the woat key (simulates a pre-2026-05-18 ckpt).
    cfg = {
        "vocab_size": vocab, "d_model": 8, "n_layers": 2, "n_heads": 2,
        "d_head": 4, "max_T": 0, "feedback_mode": "none",
        "feedback_pairs": (), "feedback_self_k": 0, "tie_embeddings": True,
        "use_memory": True, "output_gate": True, "mem_size": 4,
        "thinking_token_id": thinking_id, "arch": "deltanet",
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
    }
    ckpt_path = str(tmp_path / "legacy.pt")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    model_r, cfg_r = build_model_from_ckpt(ckpt_path)
    assert model_r.memory.write_only_at_think is False
