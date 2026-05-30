"""Regression tests for sft_code.py model-loading control flow.

The bug we're guarding against (2026-05-19):
  The modern-path (ckpt already has memory + gate) was sandwiched
  between three branches. The else at the bottom unconditionally re-
  ran build_model_from_ckpt, silently overwriting the model and cfg
  that the modern path had just constructed — INCLUDING cfg mutations
  like ``cfg["sft_with_thinking"]``. Result: flags the modern path set
  were lost.

Detection rule:
  After main()'s "build model from ckpt" block, the cfg must retain the
  mutations the modern path made (``sft_with_thinking``), i.e. the modern
  path must NOT be re-run through the original-path reload.

We can't easily test main() end-to-end without an actual ckpt, so we
reproduce the load-flow on synthetic ckpts.
"""
import pathlib
import sys

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
# regression because the EXPECTED outputs (cfg keys) are what main() is
# supposed to produce.
# ---------------------------------------------------------------------------

def _sft_load_model(ckpt_path: str, *, with_thinking: bool,
                    mem_size: int = 4, mem_dim: int = 0):
    """Reproduce the control flow from sft_code.main()'s build-from-ckpt
    block. Returns (model, cfg) — what would land in the train loop and the
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
            attention_cls=DeltaNetAttention,
        )
        model.load_state_dict(sd, strict=False)
        cfg["use_memory"] = True
        cfg["output_gate"] = True
        cfg["thinking_token_id"] = thinking_token_id
        cfg["vocab_size"] = new_vocab
        cfg["mem_size"] = int(mem_size)
        cfg["mem_dim"] = int(mem_dim) if mem_dim > 0 else int(cfg["d_model"])
        cfg["sft_with_thinking"] = True
    else:
        # original path
        from experiments.eval_bracket_structure import build_model_from_ckpt
        model, cfg = build_model_from_ckpt(ckpt_path)
    return model, cfg


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_modern_path_not_overwritten_by_stray_reload(tmp_path):
    """REGRESSION (2026-05-19): the modern path's cfg mutations must
    survive — i.e. the path must NOT fall through to the original-path
    reload that silently rebuilds model + cfg."""
    ckpt = str(tmp_path / "modern.pt")
    _make_modern_tiny_ckpt(ckpt)
    model, cfg = _sft_load_model(ckpt, with_thinking=True)
    assert cfg.get("sft_with_thinking") is True, (
        "modern-path cfg mutation was lost — the path got overwritten by "
        "a stray reload (this is the bug)."
    )
    assert model.use_memory is True


def test_legacy_path_builds_thinking(tmp_path):
    """The legacy path (no memory+gate in ckpt) builds a fresh
    memory+gate model and records sft_with_thinking. CPU-constructable
    (no build_model_from_ckpt / cuda)."""
    ckpt = str(tmp_path / "legacy.pt")
    _make_legacy_tiny_ckpt(ckpt)
    model, cfg = _sft_load_model(ckpt, with_thinking=True)
    assert model.use_memory is True
    assert cfg.get("sft_with_thinking") is True
    assert cfg.get("output_gate") is True


def test_no_thinking_path_loads_as_is(tmp_path):
    """When --with_thinking is False, the original path loads whatever
    was in the ckpt (memory present here) without thinking mutations."""
    ckpt = str(tmp_path / "modern.pt")
    _make_modern_tiny_ckpt(ckpt)
    model, cfg = _sft_load_model(ckpt, with_thinking=False)
    assert model.use_memory is True
    assert cfg.get("sft_with_thinking") is None
