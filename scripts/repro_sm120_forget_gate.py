"""Minimal reproducer for the fla forget-gate Triton crash on RTX 5090 (sm_120).

Forward succeeds; backward raises:
    RuntimeError: Triton Error [CUDA]: misaligned address
from triton/compiler/compiler.py:_init_handles when the wy-repr backward
kernel `prepare_wy_repr_bwd_kernel` (fla/ops/gated_delta_rule/wy_fast.py)
is loaded.

Flip `USE_FORGET_GATE` below to False to bypass the crash. See
BUG_sm120_forget_gate.md for full context and environment.
"""

import torch
from fla.layers import GatedDeltaProduct

USE_FORGET_GATE = True

torch.manual_seed(0)
device = "cuda"
dtype = torch.bfloat16

B, T, D, H = 1, 64, 256, 4
layer = GatedDeltaProduct(
    hidden_size=D,
    head_dim=D // H,
    num_heads=H,
    num_v_heads=H,
    mode="chunk",
    use_output_gate=True,
    use_short_conv=True,
    use_forget_gate=USE_FORGET_GATE,
    allow_neg_eigval=True,
    num_householder=2,
).to(device=device, dtype=dtype)

x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
y, *_ = layer(x)
print(f"forward ok  use_forget_gate={USE_FORGET_GATE}  y.shape={tuple(y.shape)}")
y.sum().backward()
print("backward ok")
