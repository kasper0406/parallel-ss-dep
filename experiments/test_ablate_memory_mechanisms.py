"""Tests for the ablation helper in ablate_memory_mechanisms.py.

These are unit-only tests of the state_dict surgery — they don't run
the model. The full pipeline is exercised manually by the headline
experiment.
"""
import pathlib
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.ablate_memory_mechanisms import _write_ablated_ckpt


def _make_ckpt_with_wm_pkm_tensors(tmp_path):
    """Synthesize a state_dict with the keys our ablations target."""
    sd = {
        "memory.W_proj.weight": torch.randn(8, 16),
        "pkm_layer.out_alpha": torch.tensor([0.42]),
        "embed.weight": torch.randn(32, 16),  # unrelated
    }
    cfg = {"vocab_size": 32, "d_model": 16}
    ckpt_path = tmp_path / "src.pt"
    torch.save({"state_dict": sd, "config": cfg}, ckpt_path)
    return str(ckpt_path)


def test_wm_off_zeros_only_W_proj(tmp_path):
    src = _make_ckpt_with_wm_pkm_tensors(tmp_path)
    dst = tmp_path / "wm_off.pt"
    _write_ablated_ckpt(src, str(dst), disable_wm=True, disable_pkm=False)
    sd_dst = torch.load(dst, map_location="cpu", weights_only=False)["state_dict"]
    assert sd_dst["memory.W_proj.weight"].abs().sum().item() == 0
    # PKM alpha unchanged
    assert sd_dst["pkm_layer.out_alpha"].item() == pytest.approx(0.42)
    # Unrelated weights unchanged
    sd_src = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
    assert torch.allclose(sd_dst["embed.weight"], sd_src["embed.weight"])


def test_pkm_off_zeros_only_out_alpha(tmp_path):
    src = _make_ckpt_with_wm_pkm_tensors(tmp_path)
    dst = tmp_path / "pkm_off.pt"
    _write_ablated_ckpt(src, str(dst), disable_wm=False, disable_pkm=True)
    sd_dst = torch.load(dst, map_location="cpu", weights_only=False)["state_dict"]
    assert sd_dst["pkm_layer.out_alpha"].item() == 0.0
    # WM W_proj unchanged
    sd_src = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
    assert torch.allclose(sd_dst["memory.W_proj.weight"],
                          sd_src["memory.W_proj.weight"])


def test_both_off_zeros_both(tmp_path):
    src = _make_ckpt_with_wm_pkm_tensors(tmp_path)
    dst = tmp_path / "both_off.pt"
    _write_ablated_ckpt(src, str(dst), disable_wm=True, disable_pkm=True)
    sd_dst = torch.load(dst, map_location="cpu", weights_only=False)["state_dict"]
    assert sd_dst["memory.W_proj.weight"].abs().sum().item() == 0
    assert sd_dst["pkm_layer.out_alpha"].item() == 0.0


def test_baseline_no_op_preserves_cfg_keys(tmp_path):
    """Sanity: if a ckpt doesn't have memory or pkm keys, ablation is
    a no-op and doesn't crash (used when probing models without these
    components)."""
    sd = {"embed.weight": torch.randn(32, 16)}
    cfg = {"vocab_size": 32, "d_model": 16}
    src = tmp_path / "minimal.pt"
    torch.save({"state_dict": sd, "config": cfg}, src)
    dst = tmp_path / "minimal_off.pt"
    _write_ablated_ckpt(str(src), str(dst), disable_wm=True, disable_pkm=True)
    sd_dst = torch.load(dst, map_location="cpu", weights_only=False)["state_dict"]
    assert torch.allclose(sd_dst["embed.weight"], sd["embed.weight"])
