"""Measure peak GPU memory for fp32 grads vs bf16 grads.

Critical question: does telling autograd to allocate bf16 grads
(via p.grad_dtype = bf16) actually reduce peak memory, or does
backward still allocate fp32 internally?
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


def build_v4_like_model(d_model=576, n_layers=30, n_heads=9, d_head=64):
    """Same dims as the running v4 model, no FiLM/memory for simplicity."""
    return TinyLM(
        vocab_size=49152 + 1,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        attention_cls=DeltaNetAttention,
        feedback_mode="none",
    ).to("cuda")


def measure(grad_dtype: torch.dtype | None, autocast: bool, batch=4, T=512):
    print(f"\n=== grad_dtype={grad_dtype} autocast={autocast} batch={batch} T={T} ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_v4_like_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M")

    if grad_dtype is not None:
        for p in model.parameters():
            try:
                p.grad_dtype = grad_dtype
            except AttributeError:
                pass

    # One forward+backward to materialize all the autograd buffers.
    ids = torch.randint(0, 49152, (batch, T), device="cuda")
    if autocast:
        ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                ids[:, 1:].reshape(-1))
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    grad_dtypes = {p.grad.dtype for p in model.parameters() if p.grad is not None}
    grad_bytes = sum(p.grad.element_size() * p.grad.numel()
                      for p in model.parameters() if p.grad is not None)
    print(f"  peak GPU memory : {peak_mb:.0f} MB")
    print(f"  grad dtypes     : {grad_dtypes}")
    print(f"  total grad bytes: {grad_bytes/1e6:.0f} MB")

    del model, ids, loss, out, logits
    torch.cuda.empty_cache()
    return peak_mb


if __name__ == "__main__":
    measure(grad_dtype=None, autocast=True)
    measure(grad_dtype=torch.bfloat16, autocast=True)
    measure(grad_dtype=None, autocast=False)
    measure(grad_dtype=torch.bfloat16, autocast=False)
