"""Attention-class registry + arch-string parsers.

Extracted from `train_lm.py` so `model_builder.py` can import them
without depending on the trainer module. The legacy names live here
verbatim — the parsing surface is exactly what the existing
checkpoints' `cfg['arch']` and `cfg['layers_spec']` strings expect.
"""
from __future__ import annotations

from experiments.layers import (
    DeltaNetAttention,
    SoftmaxAttention,
    Mamba2Attention,
)


_NAME_TO_CLS = {
    "deltanet":    DeltaNetAttention,
    "transformer": SoftmaxAttention,
    "mamba2":      Mamba2Attention,
}


def build_arch(name: str, n_layers: int):
    """Map an arch name to a single attention class or a per-layer list."""
    if name in _NAME_TO_CLS:
        return dict(attention_cls=_NAME_TO_CLS[name])
    raise ValueError(f"unknown arch: {name}")


def parse_layers_arg(spec: str) -> list:
    """Parse comma-separated --layers spec into a class list."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [_NAME_TO_CLS[p] for p in parts]
