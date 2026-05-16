"""bf16-optimizer-state: AdamW + Muon variants with bf16 state storage.

Persistent state (`exp_avg`, `exp_avg_sq` for AdamW; `momentum_buffer`
for Muon) is stored in bf16, halving optimizer-state memory. The
arithmetic is performed in fp32 (lift → step → round-to-nearest-even
back to bf16) so precision-critical operations like the second-moment
accumulation `v ← β·v + (1−β)·g²` aren't done in 8-bit-mantissa space.

Memory accounting for the v4 setup (218 M params, 247 Muon-eligible /
161.1 M, 186 AdamW / 56.9 M):
  - stock AdamW state (2× fp32): 455 MB → bf16: 228 MB (saves 227 MB)
  - stock Muon momentum (fp32):  644 MB → bf16: 322 MB (saves 322 MB)
  - total persistent saving:                              ~550 MB

Empirical precision: validated on a small DeltaNet against
torch.optim.AdamW with grad_accum=8 + bf16 autocast; max |Δ_loss| over
200 steps was 0.004 (within batch-noise). See
`experiments/test_bf16_optim.py`.

What this is NOT:
  - It does not save peak GPU memory from gradients (autograd's
    backward still allocates fp32 intermediates regardless of the
    `grad_dtype` hint — measured directly; see
    `experiments/test_bf16_grad_memory.py`). For a peak-memory win on
    grads you need bf16 master weights with stochastic rounding.
  - No 8-bit/block quantization. If 4× memory pressure appears, drop
    in bitsandbytes' `AdamW8bit` instead.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim._muon import (
    _zeropower_via_newtonschulz,
    _adjust_lr,
    DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_NS_STEPS, EPS,
)


__all__ = ["BF16StateAdamW", "BF16StateMuon"]


def _adamw_param_step(p: Tensor, grad: Tensor,
                       exp_avg: Tensor, exp_avg_sq: Tensor,
                       wd_scale: Tensor, neg_step_size: Tensor,
                       bias_c2_sqrt_inv: Tensor,
                       beta1: float, beta2: float, eps: float) -> None:
    """Pure-tensor per-param AdamW update. Suitable for `torch.compile`.

    Per-step scalars (`wd_scale`, `neg_step_size`, `bias_c2_sqrt_inv`)
    are passed as 0-dim tensors so compile does not specialize on
    their values and re-trace every step. `beta1`, `beta2`, `eps` are
    passed as Python floats because they are constant across all steps
    of a given param group (compile-time specialization is fine).

    Scalar conventions (chosen so the body has no Python math):
      wd_scale          = 1 - lr * wd
      neg_step_size     = -lr / (1 - beta1**t)         (note the sign)
      bias_c2_sqrt_inv  = 1 / sqrt(1 - beta2**t)
    """
    # Decoupled weight decay (AdamW form).
    p.mul_(wd_scale)

    # Lift bf16 state + (possibly-bf16) grad to fp32 for the math.
    m = exp_avg.to(torch.float32)
    v = exp_avg_sq.to(torch.float32)
    g = grad if grad.dtype == torch.float32 else grad.to(torch.float32)

    m.lerp_(g, 1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

    # Reciprocal of bias-corrected sqrt(v): premultiply for the scaled-step
    # parameter update p += neg_step_size * m / denom.
    denom = v.sqrt().mul_(bias_c2_sqrt_inv).add_(eps)
    # Combine the two scalars into a single addcdiv by folding step_size in.
    # Equivalent: p += neg_step_size * (m / denom)
    update = m / denom
    p.add_(update * neg_step_size)

    # Round-to-nearest-even back to bf16 storage.
    exp_avg.copy_(m)
    exp_avg_sq.copy_(v)


class BF16StateAdamW(Optimizer):
    """AdamW where exp_avg / exp_avg_sq are stored as bf16.

    Math runs in fp32 (lift state to fp32 each step, cast back at the
    end). Equivalent to stock torch.optim.AdamW for our flag set
    (no maximize, capturable, fused, amsgrad).

    `compile_step=True` wraps the per-param update in `torch.compile`,
    fusing the bf16↔fp32 casts + elementwise math + cast-back into
    fewer kernel launches. First step incurs a compile cost (~1-2 s per
    distinct param shape); subsequent steps run from the cache.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2,
                 compile_step: bool = True):
        if not 0.0 <= lr:
            raise ValueError(f"lr must be >= 0 (got {lr})")
        if not 0.0 <= eps:
            raise ValueError(f"eps must be >= 0 (got {eps})")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"betas must be in [0, 1) (got {betas})")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._param_step = (
            torch.compile(_adamw_param_step, dynamic=False, fullgraph=False)
            if compile_step else _adamw_param_step
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            # Pre-compute per-step scalars on the same device as the first
            # available param tensor, as 0-dim tensors (prevents recompile
            # per step in the compiled hot loop).
            sample_p = next((p for p in group["params"] if p.grad is not None),
                            None)
            if sample_p is None:
                continue
            dev = sample_p.device
            wd_scale_py = 1.0 - lr * wd
            wd_scale = torch.tensor(wd_scale_py, device=dev, dtype=torch.float32)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16,
                                                         memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16,
                                                           memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]

                # Per-param-step scalars (depend on t which is per-param).
                bias_c1 = 1.0 - beta1 ** t
                bias_c2 = 1.0 - beta2 ** t
                neg_step_size = torch.tensor(-lr / bias_c1,
                                             device=dev, dtype=torch.float32)
                bias_c2_sqrt_inv = torch.tensor(1.0 / math.sqrt(bias_c2),
                                                device=dev, dtype=torch.float32)

                self._param_step(
                    p, grad, state["exp_avg"], state["exp_avg_sq"],
                    wd_scale, neg_step_size, bias_c2_sqrt_inv,
                    beta1, beta2, eps,
                )
        return loss


class BF16StateMuon(Optimizer):
    """Muon with bf16 momentum_buffer storage.

    Mirrors torch.optim.Muon's algorithm exactly (Newton-Schulz
    orthogonalization with the same DEFAULT_A/B/C coefficients), but
    stores `momentum_buffer` in bf16 between steps. The lerp_/add ops
    that touch the buffer are done in fp32, then results are cast
    back to bf16 for storage. NS itself already runs in bf16
    upstream (`grad.bfloat16()` inside `_zeropower_via_newtonschulz`),
    so this changes only the *storage* of the momentum buffer.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                 nesterov=True,
                 ns_coefficients=(DEFAULT_A, DEFAULT_B, DEFAULT_C),
                 eps=EPS, ns_steps=DEFAULT_NS_STEPS,
                 adjust_lr_fn=None):
        if not 0.0 <= lr:
            raise ValueError(f"lr must be >= 0 (got {lr})")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum must be >= 0 (got {momentum})")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay must be >= 0 (got {weight_decay})")
        if adjust_lr_fn is not None and adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(f"adjust_lr_fn={adjust_lr_fn!r} not supported")
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                         nesterov=nesterov, ns_coefficients=ns_coefficients,
                         eps=eps, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters; got {tuple(p.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefs = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adj_fn = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if grad.ndim != 2:
                    raise ValueError("Muon param gradient must be 2D")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, dtype=torch.bfloat16,
                        memory_format=torch.preserve_format,
                    )

                # Lift buf and grad to fp32 for the lerp/lerp.
                buf = state["momentum_buffer"].to(torch.float32)
                g = grad if grad.dtype == torch.float32 else grad.to(torch.float32)

                buf.lerp_(g, 1 - momentum)
                update = g.lerp(buf, momentum) if nesterov else buf

                # NS orthogonalization (runs in bf16 internally upstream).
                update = _zeropower_via_newtonschulz(update, ns_coefs, ns_steps, eps)

                adjusted_lr = _adjust_lr(lr, adj_fn, p.shape)
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-adjusted_lr)

                # Persist updated buf in bf16.
                state["momentum_buffer"].copy_(buf)
        return loss
