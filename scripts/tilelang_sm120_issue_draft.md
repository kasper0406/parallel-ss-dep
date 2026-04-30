# [BUG] sm_120 (RTX 5090 / Blackwell consumer): TMA load triggers `CUDA_ERROR_MISALIGNED_ADDRESS` when smem arena starts with a small buffer

## Summary

`SharedMemoryAlignmentPlanner` in
`src/transform/merge_shared_memory_allocations.cc` only enforces 1024-byte
alignment on TMA-touched dynamic-smem buffers when `TargetIsHopper(target)`
is true (arch in `[90, 100)`). For sm_120 (RTX 5090 and other Blackwell
consumer cards, also sm_100 / sm_103) the planner falls back to 16-byte
alignment. But `cp.async.bulk.tensor.{1..5}d.shared::cta.global.*`
**requires the destination shared-memory pointer to be 128-byte aligned**
(per the [PTX ISA spec](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)).
Whenever the dynamic-smem arena begins with a small (e.g. 4-byte float)
allocation, the TMA destination buffers placed after it land at 16-byte
(but not 128-byte) aligned offsets and the load raises
`CUDA_ERROR_MISALIGNED_ADDRESS`.

Hopper appears to tolerate this misalignment in practice (probably because
the arena layout for typical GEMM kernels happens to land buffers on
128-byte boundaries anyway). sm_120 enforces it strictly, so the same
generated kernel that works on H100 crashes on a 5090.

## Environment

| Component | Version |
|-----------|---------|
| GPU       | NVIDIA GeForce RTX 5090 (sm_120, 32 GB) |
| Driver    | 580.126.20 |
| TileLang  | 0.1.9 (also reproduces on `main`, see "Code pointer" below) |
| PyTorch   | 2.13.0.dev20260427+cu132 (cu132 nightly) |
| Triton    | 3.7.0 (bundled) |
| Python    | 3.12 |
| OS        | Ubuntu 24.04, Linux 6.8.0-110-generic |

## Minimal reproducer (pure TileLang, no fla/Triton)

```python
# repro_sm120_pure_tilelang.py
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
})
def build_kernel(M, N, BM, BN):
    dt = T.bfloat16
    threads = 256

    @T.prim_func
    def kernel(
        a: T.Tensor((M, N), dt),
        b: T.Tensor((M, N), dt),
        c: T.Tensor((M, N), dt),
    ):
        with T.Kernel(M // BM, threads=threads) as i_m:
            # Tiny shared scalar to perturb the arena base offset.
            s_acc = T.alloc_shared((1,), T.float32)
            for _i in T.Parallel(1):
                s_acc[0] = 0.0
            T.sync_threads()

            s_a = T.alloc_shared((BM, BN), dt)
            s_b = T.alloc_shared((BM, BN), dt)

            for i_n in T.Pipelined(N // BN, num_stages=2):
                T.copy(a[i_m * BM:(i_m + 1) * BM, i_n * BN:(i_n + 1) * BN], s_a)
                T.copy(b[i_m * BM:(i_m + 1) * BM, i_n * BN:(i_n + 1) * BN], s_b)
                for _i, _j in T.Parallel(BM, BN):
                    c[i_m * BM + _i, i_n * BN + _j] = s_a[_i, _j] + s_b[_i, _j]

    return kernel


def main():
    M, N, BM, BN = 64, 128, 64, 32
    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    build_kernel(M, N, BM, BN)(a, b, c)
    torch.cuda.synchronize()
    print("ok", c[0, 0].item())


if __name__ == "__main__":
    main()
```

```bash
$ CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python repro_sm120_pure_tilelang.py
RuntimeError: CUDALaunch CUDA_ERROR_MISALIGNED_ADDRESS
 grid=(1,1,1), block=(384,1,1), dyn_smem_bytes=16400
```

Removing the `s_acc` allocation makes the same kernel succeed; the only role
of `s_acc` is to push the next allocation 16 bytes (and thus off the
128-byte boundary) into the arena.

## Observed PTX-level evidence

The generated `.cu` (under `TVMDebugMode`-style `tvm_kernels.cu`) places
`s_a` and `s_b` at byte offsets `16` and `8208` from `buf_dyn_shmem`:

```c
extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
...
// stage 0:
tl::tma_load(a_desc, ..., (&((bf16_t*)buf_dyn_shmem)[8])  /* byte=16   */, ...);
tl::tma_load(b_desc, ..., (&((bf16_t*)buf_dyn_shmem)[4104]) /* byte=8208 */, ...);
```

`16 mod 128 = 16` and `8208 mod 128 = 16`. Both are 16-byte aligned, both
violate the 128-byte requirement.

Manually editing the `.cu` (via an `nvcc` wrapper) to bump the offsets to
`64` (byte 128) and `4160` (byte 8320) makes the same kernel run. So does
forcing `align_bytes=1024` in `MergeSharedMemoryAllocations` from Python
(see "Workaround" below).

## Real-world impact

This kernel is what `fla.ops.common.backends.tilelang.chunk_bwd.chunk_bwd_dqkwg_tilelang`
becomes after TileLang compilation. `fla` 0.5.0 dispatches to it from
`fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_bwd` whenever the
forget gate is enabled (`g is not None`). Hence
`GatedDeltaProduct(use_forget_gate=True).backward()` crashes on any sm_120
GPU with `fla 0.5.0 + tilelang 0.1.9`.

## Code pointer

```c++
// src/transform/merge_shared_memory_allocations.cc:393 (HEAD as of 936ae92)
class SharedMemoryAlignmentPlanner : public StmtExprVisitor {
  ...
  void MarkSharedVarIfNeeded(const VarNode *op) {
    ...
    if (scope == "shared" || scope == "shared.dyn") {
      auto target = Target::Current();
      ICHECK(target.defined()) << "Target is not defined";
      const int alignment = TargetIsHopper(target) ? 1024 : 16;  // <-- BUG
      shmem_alignment_map_[op] = alignment;
    }
  }
```

`TargetIsHopper(target)` returns true only for `arch in [90, 100)`. For
sm_120 it returns false, so `shmem_alignment_map_` is never updated and the
TMA-touched buffer falls back to the global `align_bytes_` default (16).

## Suggested fix

```diff
-      const int alignment = TargetIsHopper(target) ? 1024 : 16;
+      const int alignment = TargetHasBulkCopy(target) ? 1024 : 16;
```

`TargetHasBulkCopy(target)` is already defined in
`src/target/utils.cc:151-156` and returns true for any CUDA target with
arch `>= 90` (so sm_90 / sm_100 / sm_103 / sm_120 / future). It's the
correct predicate because TMA support and TMA's alignment requirements are
the same set of targets.

Verified on the minimal repro and on the real-world fla
`GatedDeltaProduct(use_forget_gate=True)` end-to-end backward + 3 optimizer
steps; gradients match the Triton fallback (`FLA_TILELANG=0`) bit-exactly.

## Regression test

The patch adds
`testing/python/issue/test_tilelang_issue_sm120_tma_smem_alignment.py`, a
host-independent regression test. It compiles a kernel with the
buggy-arena layout (small `T.float32` shared scalar allocated *before*
the TMA-loaded buffers) for explicit `cuda -arch=sm_{90,100,120}`
targets, then walks every emitted `tl::tma_load(..., (&((T*)buf_dyn_shmem)[expr]), ...)`
and asserts that the additive constant in `expr`, scaled by `sizeof(T)`,
is divisible by 128 (the alignment requirement for
`cp.async.bulk.tensor.*`). Variable strides in `expr` are checked the
same way so the assertion holds for every iteration value.

Behaviour before and after the patch:

| arch    | pre-fix | post-fix |
|---------|---------|----------|
| sm_90   | PASS    | PASS     |
| sm_100  | FAIL (additive base = 16 bytes) | PASS |
| sm_120  | FAIL (additive base = 16 bytes) | PASS |

The test is GPU-host independent — it compiles for explicit `cuda -arch=...`
targets and never launches the kernel, so CI hosts without a Blackwell GPU
exercise the regression too.

## Workaround for users without rebuilding TileLang

Force the global `align_bytes` from 16 → 1024 in the Python-side wrapper
of `MergeSharedMemoryAllocations`:

```python
import tilelang.transform as tl_xform
import tilelang.engine.phase as phase

_orig = tl_xform.MergeSharedMemoryAllocations
def _patched(enable_aggressive_merge=False, align_bytes=16):
    return _orig(enable_aggressive_merge=enable_aggressive_merge, align_bytes=1024)
tl_xform.MergeSharedMemoryAllocations = _patched
phase.MergeSharedMemoryAllocations = _patched
```

This over-aligns every smem buffer to 1024 bytes (some unrelated small
accumulators get 1024-byte cells too — wasteful but harmless), and
trivially covers the 128-byte requirement for any TMA destination.
