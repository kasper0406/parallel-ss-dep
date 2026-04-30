"""Pure-TileLang minimal reproducer for the sm_120 TMA misalignment bug.

A standard 2-stage pipelined `bfloat16` GEMM with a tiny `T.float32` shared
scalar allocated *before* the TMA-loaded `s_a` / `s_b` tiles. The scalar
pushes the TMA destinations off the 128-byte boundary in the merged
dynamic-smem arena. Pre-fix on sm_120 / sm_100 the first TMA load raises
`CUDA_ERROR_MISALIGNED_ADDRESS`; on sm_90 (Hopper) the same kernel happens
to land at acceptable offsets and runs.

This is intended to be attached to a TileLang upstream issue: it has no
fla / Triton / PyTorch-compute-kernel dependencies — only `tilelang`,
`torch` (for the input tensors) and a bf16 GEMM that exists in any
TileLang test environment.

Expected on RTX 5090 (sm_120) + tilelang 0.1.9:
    [...] : Fatal: CUDAError: cuModuleUnload(module_[i]) failed with error:
        CUDA_ERROR_MISALIGNED_ADDRESS

Run:
    rm -rf ~/.tilelang/cache/*/kernels/*   # avoid stale cubin
    CUDA_VISIBLE_DEVICES=0 python repro_sm120_pure_tilelang.py
"""

import torch
import tilelang
import tilelang.language as T


def build_gemm_kernel(M, N, K, BM, BN, BK):
    @T.prim_func
    def gemm(
        a: T.Tensor((M, K), "bfloat16"),
        b: T.Tensor((K, N), "bfloat16"),
        c: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (bx, by):
            # Tiny shared scalar allocated FIRST — perturbs the dynamic-smem
            # arena so the TMA-loaded tiles below land at 16-byte (but not
            # 128-byte) aligned offsets when the planner only marks Hopper
            # as needing 1024-byte alignment.
            s_acc_scalar = T.alloc_shared((1,), "float32")
            for _i in T.Parallel(1):
                s_acc_scalar[0] = 0.0
            T.sync_threads()

            s_a = T.alloc_shared((BM, BK), "bfloat16")
            s_b = T.alloc_shared((BK, BN), "bfloat16")
            c_local = T.alloc_fragment((BM, BN), "float32")
            T.clear(c_local)

            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(a[bx * BM:(bx + 1) * BM, k * BK:(k + 1) * BK], s_a)
                T.copy(b[k * BK:(k + 1) * BK, by * BN:(by + 1) * BN], s_b)
                T.gemm(s_a, s_b, c_local)
            T.copy(c_local, c[bx * BM:(bx + 1) * BM, by * BN:(by + 1) * BN])
    return gemm


def main():
    M, N, K = 128, 128, 256
    BM, BN, BK = 64, 64, 32
    tilelang.disable_cache()
    kernel = tilelang.compile(build_gemm_kernel(M, N, K, BM, BN, BK), target="cuda")
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    kernel(a, b, c)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)
    print("ok")


if __name__ == "__main__":
    main()
