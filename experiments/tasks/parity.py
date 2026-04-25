"""
Parity task — the canonical state-tracking failure mode for linear RNNs.

Spec:
  - Input: uniform random binary sequence  x ∈ {0, 1}^T  (vocab size 2).
  - Label at position t: running parity  y_t = (x_0 ⊕ x_1 ⊕ ... ⊕ x_t).
  - Loss: cross-entropy at every position.

Why this task:
  - Linear RNNs with positive-only diagonal state transitions provably
    cannot learn parity at constant width (Grazzi et al. ICLR'25).
  - Heisenberg's bilinear cross term `Σ_{i<j} a_i ⊗ b_j` generates
    pairwise products of inputs, which is exactly what parity-over-R
    needs (parity = XOR, expressible via second-order moments of ±1
    encodings: `prod (1 - 2x_i)`).
  - So this is the cleanest direct test of whether the bilinear cross
    term is expressively useful.
"""
from __future__ import annotations

import torch


def make_batch(B: int, T: int, device: torch.device | str = "cuda",
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (input_ids, labels) for parity.

    Returns:
        input_ids : (B, T) int64, values in {0, 1}.
        labels    : (B, T) int64, running parity of input_ids prefix.
    """
    bits = torch.randint(0, 2, (B, T), device=device, generator=generator,
                         dtype=torch.int64)
    labels = bits.cumsum(dim=-1) % 2
    return bits, labels
