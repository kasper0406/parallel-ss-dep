"""DDP-free data-parallel primitives: manual bucketed gradient all-reduce.

DistributedDataParallel is incompatible with this repo's latent-thinking path
(reentrant adapter forwards break static_graph; static_graph + grad-accum
no_sync is an unfixed PyTorch regression). This module does the one thing DDP
would do for us — average gradients across ranks — but as a plain synchronous
collective at the grad-accum boundary, with NO autograd hooks and NO graph
assumptions, so it composes with any forward (latent R=2..8, PKM, WM, …).

`experiments/bench_two_gpu_allreduce.py` measured the flat all_reduce of the
full 0.80 GB bf16 grad at 93.8 ms = 1.9% of a ~5 s step, so a simple
synchronous bucketed reduce is essentially free (no overlap engineering
needed). Wired into `experiments/train_lm.py` behind `--manual_allreduce`.

All functions are no-ops when world_size <= 1, so the single-process path never
pays for a collective.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# 64 MiB buckets — matches bench_two_gpu_allreduce.py's --bucket_mb default.
DEFAULT_BUCKET_BYTES = 64 * 1024 * 1024


def _bucket_by_bytes(tensors, bucket_bytes):
    """Group tensors (in the given, rank-consistent order) into contiguous runs
    of at most ~bucket_bytes. A new bucket is also started on a dtype change,
    since _flatten_dense_tensors requires a single dtype per bucket."""
    buckets = []
    cur, cur_bytes = [], 0
    for t in tensors:
        tb = t.numel() * t.element_size()
        if cur and (cur[0].dtype != t.dtype or cur_bytes + tb > bucket_bytes):
            buckets.append(cur)
            cur, cur_bytes = [], 0
        cur.append(t)
        cur_bytes += tb
    if cur:
        buckets.append(cur)
    return buckets


def allreduce_gradients(params, world_size, bucket_bytes=DEFAULT_BUCKET_BYTES):
    """In-place average of `.grad` across all ranks, flatten-bucketed.

    Reduces over the FIXED trainable set (`requires_grad=True`), materializing a
    zero grad for any such param NOT touched this step, then packs grads into
    ~bucket_bytes contiguous buckets (the bench's approach), all_reduce(SUM)'s,
    divides by world_size, and copies back.

    The zero-fill is what makes the bucket layout identical across ranks even
    when a feature fires on a data-dependent subset of ranks — the copy-head is
    skipped on batches with no recall positions (model.py::_apply_copy_head),
    and ranks stream DISJOINT data shards, so one rank can have that grad while
    another does not on the same step. This matches DDP + find_unused_parameters,
    which likewise zero-fills unused grads. Params with `requires_grad=False`
    (frozen / pinned) are left untouched so the optimizer keeps skipping them.
    Iteration follows `params` order (identical on every rank — same model
    construction), so bucket boundaries line up across ranks.
    """
    if world_size <= 1:
        return
    grads = []
    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    if not grads:
        return
    inv = 1.0 / float(world_size)
    for bucket in _bucket_by_bytes(grads, bucket_bytes):
        flat = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.mul_(inv)
        for g, synced in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
            g.copy_(synced)


def check_grad_param_handshake(params, world_size, device):
    """One-time assert that every rank's TRAINABLE param set is identical.

    allreduce_gradients reduces over the fixed `requires_grad=True` set (with
    zero-fill), so the invariant that must hold across ranks is that this set
    aligns — NOT that grad PRESENCE aligns (which is legitimately data-dependent
    for the copy-head and handled by the zero-fill). A mismatch here means the
    ranks built the model differently or froze a different subset, which would
    mis-align the bucketed all_reduce. Encodes the per-index requires_grad mask
    and all_reduce(MIN)/MAX — any disagreement makes MIN != MAX at that index.
    """
    if world_size <= 1:
        return
    mask = torch.tensor([1.0 if p.requires_grad else 0.0 for p in params],
                        device=device)
    mn = mask.clone()
    mx = mask.clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    if not torch.equal(mn, mx):
        bad = (mn != mx).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(
            "[manual_allreduce] trainable (requires_grad) param set differs "
            f"across ranks at param indices {bad[:16]}"
            f"{'...' if len(bad) > 16 else ''}. The bucketed all_reduce would "
            "mis-align gradients. Ranks must build/freeze the model identically.")


def broadcast_module(module, world_size, src=0):
    """Broadcast rank-`src`'s params AND buffers to every rank, in place.

    Used to force byte-identical init across ranks (robust vs. relying on a
    shared seed alone — different RNG consumption between build steps could
    otherwise desync). No-op when world_size <= 1.
    """
    if world_size <= 1:
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def drift_check(param, world_size, device, tol=1e-4):
    """Cheap cross-rank divergence probe on a fixed reference param.

    Returns (ok, spread) where spread = (max - min) of the param's L2 norm
    across ranks. Ranks that have silently diverged (a non-deterministic op, a
    missed reduce) show spread > tol. No collective when world_size <= 1.
    """
    if world_size <= 1:
        return True, 0.0
    v = param.detach().float().norm().reshape(1).to(device)
    vmax = v.clone()
    vmin = v.clone()
    dist.all_reduce(vmax, op=dist.ReduceOp.MAX)
    dist.all_reduce(vmin, op=dist.ReduceOp.MIN)
    spread = float((vmax - vmin).item())
    return spread <= tol, spread
