"""
Modular addition mod p — the sharpest theoretical separator between
SO(n)-based architectures and DeltaNet/DeltaProduct.

Spec:
  - Vocab is {0, 1, …, p-1}. Each token is a digit mod p.
  - Label at position t is the running sum mod p:
        y_t = (x_0 + x_1 + … + x_t) mod p

Why this task:
  - For p = 2 (parity): both SO(n) and DeltaNet+`allow_neg_eigval` solve
    via Z_2.
  - For p > 2: Z_p ⊂ SO(2) (rotation by 2π/p), so SO(n)-based architectures
    (ortho, hybrid) should solve any p naturally. DeltaNet's transition
    `(I − β k kᵀ)` has eigenvalues `1 − β` along k and `1` orthogonal —
    even with `allow_neg_eigval=True`, the achievable eigenvalues are
    only `±1`, so DeltaNet only realises Z_2. DeltaProduct's Householder
    products preserve this (each Householder has eigenvalue ±1).
  - Therefore the prediction: SO(n)-based architectures solve all p,
    DeltaNet/DeltaProduct fail for p > 2.

This is the cleanest empirical separator between the hybrid framing and
the existing DeltaNet/DeltaProduct line.

Reference: Grazzi et al. ICLR'25 (https://arxiv.org/abs/2411.12537),
which proves this for the linear-RNN class but doesn't test SO(n) directly.
"""
from __future__ import annotations

import torch


def make_batch(B: int, T: int, p: int = 3,
               device: torch.device | str = "cuda",
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (input_ids, labels) for modular addition mod p.

    Returns:
        input_ids : (B, T) int64, values in {0, …, p-1}
        labels    : (B, T) int64, running sum mod p of the input prefix
    """
    assert p >= 2, f"need p ≥ 2, got {p}"
    digits = torch.randint(0, p, (B, T), device=device, generator=generator,
                           dtype=torch.int64)
    labels = digits.cumsum(dim=-1) % p
    return digits, labels
