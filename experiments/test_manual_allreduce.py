"""Tests for the DDP-free manual bucketed gradient all-reduce.

The manual-allreduce path (experiments/manual_allreduce.py, wired into
train_lm.py behind --manual_allreduce) is the 2-GPU data-parallel mode that
composes with the latent-thinking forward, which DDP cannot. These tests run
WITHOUT GPUs — gloo backend, CPU tensors, torch.multiprocessing spawn, tiny
toy models — so they are safe to run any time.

Coverage:
  - equivalence: 2-rank manual-allreduce == single-process on the concatenated
    per-rank batches (same final weights, fp32 tolerance),
  - bucketing correctness: bucketed reduce == direct per-tensor reduce (exact),
  - none-grad handling: a param with no grad on both ranks is untouched,
  - flag-off purity: no process group is initialized when the flag is off.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_manual_allreduce.py -v
"""
from __future__ import annotations

import argparse
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from experiments.manual_allreduce import (
    allreduce_gradients,
    broadcast_module,
    check_grad_param_handshake,
    drift_check,
)


# --------------------------------------------------------------------------
# Toy model + fixed dataset (shared by the multiprocess tests)
# --------------------------------------------------------------------------

D_IN, D_HID, N_CLS = 8, 16, 4
N_STEPS = 3
BATCH_PER_RANK = 5
N_MICRO = 2            # grad accumulation per rank per step
LR = 0.1
WORLD = 2


def _make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(D_IN, D_HID),
        nn.Tanh(),
        nn.Linear(D_HID, N_CLS),
    )


def _fixed_dataset():
    """Deterministic per-(step, rank, micro) batches. rank r accumulates over
    its N_MICRO micro-batches; the single-process reference accumulates over ALL
    ranks' micro-batches (the equivalent global batch)."""
    g = torch.Generator().manual_seed(1234)
    data = []
    for _s in range(N_STEPS):
        per_rank = []
        for _r in range(WORLD):
            micros = []
            for _m in range(N_MICRO):
                x = torch.randn(BATCH_PER_RANK, D_IN, generator=g)
                y = torch.randint(0, N_CLS, (BATCH_PER_RANK,), generator=g)
                micros.append((x, y))
            per_rank.append(micros)
        data.append(per_rank)
    return data


def _single_process_reference(data):
    """Ground truth: accumulate every rank's every micro-batch with the global
    divisor (WORLD*N_MICRO), plain SGD, mean CE — what the averaged multi-rank
    grad-accum run must reproduce."""
    model = _make_model(seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(reduction="mean")
    total_micro = WORLD * N_MICRO
    for step in range(N_STEPS):
        opt.zero_grad(set_to_none=True)
        for r in range(WORLD):
            for m in range(N_MICRO):
                x, y = data[step][r][m]
                (lossf(model(x), y) / total_micro).backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


def _dist_worker(rank, world, backend, out_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29578")
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        data = _fixed_dataset()
        # Different per-rank init on purpose — broadcast must make them identical.
        model = _make_model(seed=rank)
        broadcast_module(model, world, src=0)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss(reduction="mean")
        for step in range(N_STEPS):
            opt.zero_grad(set_to_none=True)
            # Grad accumulation, each micro scaled by 1/N_MICRO; the manual
            # all_reduce runs ONCE after the whole accumulation (the trainer's
            # grad-accum boundary), matching the real path.
            for m in range(N_MICRO):
                x, y = data[step][rank][m]
                (lossf(model(x), y) / N_MICRO).backward()
            check_grad_param_handshake(list(model.parameters()), world,
                                       torch.device("cpu"))
            # Force multiple tiny buckets to exercise the bucketing loop.
            allreduce_gradients(list(model.parameters()), world,
                                bucket_bytes=64)
            opt.step()
        ok, spread = drift_check(next(model.parameters()), world,
                                 torch.device("cpu"))
        assert ok, f"rank drift spread={spread}"
        if rank == 0:
            torch.save([p.detach().clone() for p in model.parameters()],
                       out_path)
    finally:
        dist.destroy_process_group()


def _bucket_worker(rank, world, backend, out_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29579")
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        # Deterministic per-rank grad set spanning several dtypes/sizes so the
        # byte-budget bucketing produces >1 bucket.
        torch.manual_seed(100 + rank)
        shapes = [(64,), (128, 4), (7,), (256,), (3, 3)]
        params = [nn.Parameter(torch.zeros(s)) for s in shapes]
        for p in params:
            p.grad = torch.randn_like(p)
        # Reference: direct per-tensor all_reduce(SUM)/world on a clone.
        ref = [g.grad.clone() for g in params]
        for t in ref:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t.div_(world)
        # Bucketed path (tiny buckets -> multiple buckets).
        allreduce_gradients(params, world, bucket_bytes=512)
        if rank == 0:
            got = [p.grad.clone() for p in params]
            exact = all(torch.equal(a, b) for a, b in zip(got, ref))
            torch.save({"exact": exact}, out_path)
    finally:
        dist.destroy_process_group()


def _nonegrad_worker(rank, world, backend, out_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29580")
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        p_grad = nn.Parameter(torch.zeros(16))
        # A FROZEN param (requires_grad=False) never receives a grad and must be
        # left completely untouched (grad stays None, data unchanged).
        p_frozen = nn.Parameter(torch.ones(8), requires_grad=False)
        # Both ranks derive g0,g1 from the same seed so rank 0 can predict the
        # average; each rank uses its own as the local grad.
        torch.manual_seed(0)
        g0 = torch.randn(16)
        g1 = torch.randn(16) * 2
        p_grad.grad = (g0 if rank == 0 else g1).clone()
        params = [p_grad, p_frozen]
        # Handshake agrees: both ranks mark index0 trainable, index1 frozen.
        check_grad_param_handshake(params, world, torch.device("cpu"))
        allreduce_gradients(params, world, bucket_bytes=64)
        if rank == 0:
            expected = (g0 + g1) / world
            ok_val = torch.allclose(p_grad.grad, expected, atol=1e-6)
            ok_none = p_frozen.grad is None
            ok_untouched = torch.equal(p_frozen.data, torch.ones(8))
            torch.save({"ok_val": ok_val, "ok_none": ok_none,
                        "ok_untouched": ok_untouched}, out_path)
    finally:
        dist.destroy_process_group()


def _datadep_worker(rank, world, backend, out_path):
    """The copy-head pattern: a requires_grad=True param gets a grad on only one
    rank (data-dependent), while ranks stream disjoint shards. Zero-fill must
    keep the bucket layout aligned (no deadlock) and average correctly."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    dist.init_process_group(backend, rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        shared0 = torch.randn(32)
        shared1 = torch.randn(32) * 3
        cond0 = torch.randn(12)
        p_shared = nn.Parameter(torch.zeros(32))   # touched on both ranks
        p_cond = nn.Parameter(torch.zeros(12))     # touched only on rank 0
        p_shared.grad = (shared0 if rank == 0 else shared1).clone()
        if rank == 0:
            p_cond.grad = cond0.clone()            # rank 1: stays None
        params = [p_shared, p_cond]
        # Tiny buckets: without zero-fill the ranks would build a DIFFERENT
        # number of buckets (rank 1 lacks p_cond) → different all_reduce count →
        # deadlock. With zero-fill both see 2 tensors → 2 buckets → aligned.
        allreduce_gradients(params, world, bucket_bytes=64)
        if rank == 0:
            ok_shared = torch.allclose(p_shared.grad,
                                       (shared0 + shared1) / world, atol=1e-6)
            ok_cond = torch.allclose(p_cond.grad, cond0 / world, atol=1e-6)
            torch.save({"ok_shared": ok_shared, "ok_cond": ok_cond}, out_path)
    finally:
        dist.destroy_process_group()


def _spawn(fn, out_path):
    mp.spawn(fn, args=(WORLD, "gloo", out_path), nprocs=WORLD, join=True)
    return torch.load(out_path, weights_only=False)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_equivalence_two_rank_vs_single_process(tmp_path):
    """2-rank manual-allreduce with grad accumulation == single-process over the
    equivalent global batch (all ranks' micro-batches)."""
    ref = _single_process_reference(_fixed_dataset())
    out = tmp_path / "dist_weights.pt"
    got = _spawn(_dist_worker, str(out))
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), \
            f"weight mismatch: max|Δ|={(a - b).abs().max().item():.3e}"


def test_bucketing_matches_direct_allreduce(tmp_path):
    """Bucketed reduce == direct per-tensor reduce, exactly."""
    out = tmp_path / "bucket.pt"
    res = _spawn(_bucket_worker, str(out))
    assert res["exact"], "bucketed all_reduce != direct per-tensor all_reduce"


def test_none_grad_param_is_skipped(tmp_path):
    """A frozen (requires_grad=False) param with no grad on both ranks doesn't
    crash and stays completely untouched."""
    out = tmp_path / "none.pt"
    res = _spawn(_nonegrad_worker, str(out))
    assert res["ok_val"], "grad-bearing param was not averaged correctly"
    assert res["ok_none"], "frozen param spuriously got a grad"
    assert res["ok_untouched"], "frozen param data was modified"


def test_data_dependent_grad_presence_no_deadlock(tmp_path):
    """A trainable param with a grad on only one rank (the copy-head pattern)
    is zero-filled: no deadlock, averaged as (rank-0 grad)/world."""
    out = tmp_path / "datadep.pt"
    res = _spawn(_datadep_worker, str(out))
    assert res["ok_shared"], "shared grad averaged incorrectly"
    assert res["ok_cond"], \
        "data-dependent grad not zero-filled/averaged correctly"


def test_flag_off_no_process_group():
    """With --manual_allreduce off, nothing in the module touches distributed."""
    from experiments.train_lm_args import build_parser
    args = build_parser().parse_args([])
    assert args.manual_allreduce is False
    # By construction: every helper is a no-op at world_size<=1, and the trainer
    # only inits a process group when manual_allreduce AND WORLD_SIZE>1. Verify
    # the helpers never touch distributed at world_size=1.
    assert not dist.is_initialized()
    model = _make_model()
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    allreduce_gradients(list(model.parameters()), world_size=1)
    broadcast_module(model, world_size=1)
    check_grad_param_handshake(list(model.parameters()), 1, torch.device("cpu"))
    ok, spread = drift_check(next(model.parameters()), 1, torch.device("cpu"))
    assert ok and spread == 0.0
    assert not dist.is_initialized()


@pytest.mark.parametrize("extra, needle", [
    (["--enable_thinking_token"], "thinking-token"),
    (["--ddp_no_bf16_compress"], "mutually exclusive"),
])
def test_manual_mode_rejects_incompatible_flags(monkeypatch, extra, needle):
    """--manual_allreduce + an incompatible flag must SystemExit early (before
    any model/data build)."""
    import experiments.train_lm as train_lm
    monkeypatch.setattr("sys.argv", ["train_lm.py", "--manual_allreduce"] + extra)
    with pytest.raises(SystemExit) as ei:
        train_lm.main()
    assert needle in str(ei.value)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
