"""
SO(n) scan — Triton kernel.

Strategy (mirrors `reference.chunked`):
  - One program per (batch, head) pair.
  - Walk T in BLOCK_T chunks.
  - Within each chunk: sequential left-fold of n×n matrix multiplications.
    For small n (n=4), the matmul is a 64-flop op, so the chunk-local fold
    is fast even sequentially.
  - Across chunks: maintain a running cumulative orthogonal matrix R_run
    in registers. After each chunk-local fold, multiply by R_run on the
    right and store; advance R_run by chunk_total.

The matrix exp is computed *outside* the kernel on the host via PyTorch
(`torch.linalg.matrix_exp`), and passed in as a (B, H, T, n, n) tensor
of orthogonal rotations. The kernel handles only the scan — generic for
any matrix-multiplication scan (SO(n), Möbius/PGL₂, DeltaNet erase, ...).

Triton import is guarded for Mac (no triton wheels).
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def matmul_scan_fwd(
        O_ptr,          # (B, H, T, n, n) — input rotations
        Y_ptr,          # (B, H, T, n, n) — output cumulative product
        T_seq,
        stride_ob, stride_oh, stride_ot, stride_oi, stride_oj,
        stride_yb, stride_yh, stride_yt, stride_yi, stride_yj,
        NUM_HEADS: tl.constexpr,
        N: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """One program = one (batch, head). Walks T in BLOCK_T chunks.

        Combine op: out[t] = O[t] @ out[t-1] (left multiplication —
        the new rotation is applied last, matching `reference.chunked`).
        """
        pid_bh = tl.program_id(0)
        pid_b = pid_bh // NUM_HEADS
        pid_h = pid_bh %  NUM_HEADS

        o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
        y_base = Y_ptr + pid_b * stride_yb + pid_h * stride_yh

        # Running cumulative product, kept in registers as an N×N tile.
        # Initialised to identity.
        offs_i = tl.arange(0, N)
        offs_j = tl.arange(0, N)
        identity = (offs_i[:, None] == offs_j[None, :]).to(tl.float32)
        R_run = identity                                                # (N, N)

        for start in range(0, T_seq, BLOCK_T):
            # Process one token at a time within the chunk; fully sequential
            # within a chunk because each step's matmul depends on the prior.
            for i in range(0, BLOCK_T):
                t = start + i
                if t < T_seq:
                    # Load O[t] (N×N) — N is small so just grid-load.
                    O_t = tl.load(
                        o_base
                        + t * stride_ot
                        + offs_i[:, None] * stride_oi
                        + offs_j[None, :] * stride_oj,
                    )                                                   # (N, N)

                    # Update: R_run = O_t @ R_run.
                    # `tl.dot` requires K ≥ 16; for our small N we do the
                    # n×n matmul manually via broadcast + reduce.
                    O_t_f32 = O_t.to(tl.float32)
                    # R_new[i, j] = Σ_k O_t[i, k] * R_run[k, j].
                    R_run = tl.sum(
                        O_t_f32[:, :, None] * R_run[None, :, :], axis=1,
                    )                                                   # (N, N)

                    # Store y[t] = R_run.
                    tl.store(
                        y_base
                        + t * stride_yt
                        + offs_i[:, None] * stride_yi
                        + offs_j[None, :] * stride_yj,
                        R_run.to(O_t.dtype),
                    )

    def launch(O, block_t: int = 64):
        """Cumulative left-multiplication scan over input rotations.

        O: (B, H, T, n, n) — orthogonal matrices, typically computed as
        `matrix_exp(X)` on the host before launch.
        Returns Y: same shape, with `Y[t] = O[t] @ O[t-1] @ ... @ O[0]`.
        """
        import torch

        assert O.is_cuda, "matmul_scan kernel needs CUDA"
        B, H, T, N, N2 = O.shape
        assert N == N2, f"expected square matrices, got {N}×{N2}"
        Y = torch.empty_like(O)

        grid = (B * H,)
        matmul_scan_fwd[grid](
            O, Y, T,
            *O.stride(), *Y.stride(),
            NUM_HEADS=H, N=N, BLOCK_T=block_t,
            num_warps=1,                                # tiny per-program work
        )
        return Y

else:

    def launch(O, block_t: int = 64):  # type: ignore[no-redef]
        raise RuntimeError(
            "Triton not available on this platform. "
            "Use `kernels/ortho_son/reference.py` on CPU."
        )
