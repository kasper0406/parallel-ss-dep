"""Pure-TileLang minimal reproducer for the sm_120 TMA misalignment bug.

This is the smallest reproduction that does not depend on fla/Triton/PyTorch
compute kernels — it's intended to be attached directly to a TileLang
upstream issue. It builds a trivial 2-buffer pipelined kernel; the only
non-trivial piece is the small `T.float32` shared scalar allocated *before*
the two TMA-loaded `s_a` / `s_b` tiles, which forces the smem arena planner
to land the TMA destinations at offsets that are 16-byte (but not 128-byte)
aligned.

Expected on sm_120 + tilelang 0.1.9:
    RuntimeError: CUDALaunch CUDA_ERROR_MISALIGNED_ADDRESS
    grid=(1,1,1), block=(384,1,1), dyn_smem_bytes=16400

Run:
    CUDA_VISIBLE_DEVICES=0 python repro_sm120_pure_tilelang.py
"""

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
            # Tiny shared scalar to force the arena base offset off
            # 128-byte alignment for the buffers that follow.
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
    torch.manual_seed(0)
    M, N, BM, BN = 64, 128, 64, 32
    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    kernel = build_kernel(M, N, BM, BN)
    kernel(a, b, c)
    torch.cuda.synchronize()
    print("ok", c[0, 0].item())


if __name__ == "__main__":
    main()
