"""Python-level workaround for the TileLang sm_120 (Blackwell consumer) TMA
shared-memory misalignment bug.

Bug
---
TileLang's `SharedMemoryAlignmentPlanner`
(`src/transform/merge_shared_memory_allocations.cc:393`) marks TMA-touched
buffers as needing 1024-byte alignment only when `TargetIsHopper(target)`
returns true (i.e. arch in [90, 100)). RTX 5090 (sm_120) is not Hopper, so
the planner falls back to 16-byte alignment, but the
`cp.async.bulk.tensor.{1..5}d.shared::cta.global.*` PTX instruction enforces
**128-byte** alignment of the destination smem pointer. That mismatch trips
`CUDA_ERROR_MISALIGNED_ADDRESS` on the first TMA load whenever the dynamic
smem arena begins with any non-128-byte buffer (e.g. a small `float` scalar,
which is what `gdn_chunk_bwd_dqkwg`'s `s_dg_last_acc` allocation is).

In `fla 0.5.0`, this kernel is the TileLang implementation of
`fla.ops.common.backends.tilelang.chunk_bwd.chunk_bwd_dqkwg_tilelang`, which
is dispatched from `chunk_gated_delta_rule_bwd` whenever `g is not None`
(forget gate enabled). That's why `GatedDeltaProduct(use_forget_gate=True)`'s
backward crashes on RTX 5090.

Workaround
----------
Force the global `align_bytes` default in `MergeSharedMemoryAllocations` from
16 up to 1024. This over-aligns *every* dynamic-smem buffer (some unrelated
small accumulators get 1024-byte cells too — wasteful but harmless), and
trivially covers the 128-byte requirement for any TMA destination.

To use, import this module before constructing any TileLang-backed module:

    import scripts.sm120_tilelang_workaround  # noqa
    from fla.layers import GatedDeltaProduct
    ...

or set `PYTHONSTARTUP` to point at this file. The patch is a no-op on
Hopper / Ampere / non-Blackwell targets because TileLang's planner already
forces 1024-byte alignment there.

**Important: clear TileLang's JIT cache once.** TileLang caches compiled
kernels under `~/.tilelang/cache/<version>-<arch>/kernels/`. Any kernel
compiled before this workaround was applied is the broken (16-byte aligned)
version, and TileLang will reuse it from cache instead of recompiling under
the patched aligner. After installing this workaround, run:

    rm -rf ~/.tilelang/cache/*/kernels/*

once. Subsequent runs will rebuild affected kernels with the correct
alignment and remain cached.

Proper upstream fix (drafted in BUG_sm120_forget_gate.md): change
`TargetIsHopper(target)` to `TargetHasBulkCopy(target)` in the planner so
that any TMA-capable target (sm_90, sm_100, sm_120) gets the strict
alignment automatically.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply() -> bool:
    """Apply the smem alignment workaround. Returns True on success."""
    try:
        import tilelang.transform as tl_xform
        import tilelang.engine.phase as phase
    except ImportError:
        logger.info("[sm_120 workaround] tilelang not importable; skipping")
        return False

    if getattr(tl_xform.MergeSharedMemoryAllocations, "_sm120_patched", False):
        return True

    _orig = tl_xform.MergeSharedMemoryAllocations

    def patched(enable_aggressive_merge: bool = False, align_bytes: int = 16):
        return _orig(enable_aggressive_merge=enable_aggressive_merge, align_bytes=1024)

    patched._sm120_patched = True  # type: ignore[attr-defined]
    tl_xform.MergeSharedMemoryAllocations = patched
    phase.MergeSharedMemoryAllocations = patched
    logger.info(
        "[sm_120 workaround] forced align_bytes=1024 in tilelang.transform.MergeSharedMemoryAllocations "
        "(see scripts/sm120_tilelang_workaround.py)"
    )
    return True


# Auto-apply on import so callers can do `import sm120_tilelang_workaround` and
# forget about it.
apply()
