"""
U_4 unipotent scan — Triton kernel.

Mirrors `reference.chunked` 1-to-1: the outer loop over chunks carries
the six-scalar running state `S = (s12, s13, s14, s23, s24, s34)`
across the trailing `d` axis; each chunk emits its inclusive local
U_4 scan prefixed by `S`, and at the end of the chunk `S` is advanced
by one U_4 combine with the chunk total.

Unlike linear attention, there is no GEMM here — all per-step work is
elementwise scalar multiplies and adds across the `BLOCK_D` vector.
`tl.dot` is therefore not useful in this kernel; the arithmetic
intensity is carried by the combine's six lines, and the parallelism
is over `d` independent U_4 scans.

Keeping the combine associative-scan-friendly
---------------------------------------------
The U_4 combine is *non-commutative* but associative (proved in
`StateDep/StateDep/Unipotent.lean`). The Blelloch up/down sweep only
requires associativity: every intermediate node of the reduction tree
is combined in left-to-right order, so the nested multiplies in the
`x14` line (`a12*b24 + a13*b34`) are fine — they're part of the
pair-wise combine, not accumulators that we expect to commute. The one
rule we have to obey is: *whenever we combine `(A, B)` we must put the
earlier element on the left.* That holds by construction throughout
the kernel: chunk-local scans combine earlier-on-the-left, and the
cross-chunk step does `S = S * chunk_total` (state first, new chunk
second). Get that ordering right and the kernel is correct by the
same Lean theorem that licences linear-attention blocking.

The Triton import is guarded so this module can be imported on Mac
(where `triton` is not installed). The kernel is obviously only
usable on a GPU.

Launch grid:  (B * H,)  ← one program per (batch, head) pair.
Each program walks the T axis in chunks of `BLOCK_T`, tiling `d` in
blocks of `BLOCK_D`. The running state is six scalars per lane, kept
in registers across chunks.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # Mac / no GPU
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _u4_combine(
        a12, a13, a14, a23, a24, a34,
        b12, b13, b14, b23, b24, b34,
    ):
        """U_4 monoid combine, six scalar channels, elementwise across BLOCK_D.

        Must be invoked with `a` = earlier element, `b` = later element.
        """
        c12 = a12 + b12
        c23 = a23 + b23
        c34 = a34 + b34
        c13 = a13 + b13 + a12 * b23
        c24 = a24 + b24 + a23 * b34
        c14 = a14 + b14 + a12 * b24 + a13 * b34
        return c12, c13, c14, c23, c24, c34

    @triton.jit
    def u4_scan_fwd(
        X_ptr,          # (B, H, T, 6, D)
        Y_ptr,          # (B, H, T, 6, D)
        T_seq,
        stride_xb, stride_xh, stride_xt, stride_xk, stride_xd,
        stride_yb, stride_yh, stride_yt, stride_yk, stride_yd,
        NUM_HEADS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program = one (batch, head, d-block). Walks T in BLOCK_T chunks."""
        pid = tl.program_id(0)
        num_d = tl.cdiv(D, BLOCK_D)
        # Decode (b, h, d_block) from the flat program id.
        pid_bh = pid // num_d
        pid_d  = pid %  num_d
        pid_b = pid_bh // NUM_HEADS
        pid_h = pid_bh %  NUM_HEADS

        x_base = X_ptr + pid_b * stride_xb + pid_h * stride_xh
        y_base = Y_ptr + pid_b * stride_yb + pid_h * stride_yh

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Running state S, six (BLOCK_D,) vectors, one per channel. Start = 0.
        S12 = tl.zeros([BLOCK_D], dtype=tl.float32)
        S13 = tl.zeros([BLOCK_D], dtype=tl.float32)
        S14 = tl.zeros([BLOCK_D], dtype=tl.float32)
        S23 = tl.zeros([BLOCK_D], dtype=tl.float32)
        S24 = tl.zeros([BLOCK_D], dtype=tl.float32)
        S34 = tl.zeros([BLOCK_D], dtype=tl.float32)

        for start in range(0, T_seq, BLOCK_T):
            # Chunk-local running state — separate from cross-chunk S. We do
            # a *sequential* intra-chunk scan here for clarity; a real
            # kernel would fuse a Blelloch sweep instead, but the two
            # produce the same result because U_4 is associative.
            L12 = tl.zeros([BLOCK_D], dtype=tl.float32)
            L13 = tl.zeros([BLOCK_D], dtype=tl.float32)
            L14 = tl.zeros([BLOCK_D], dtype=tl.float32)
            L23 = tl.zeros([BLOCK_D], dtype=tl.float32)
            L24 = tl.zeros([BLOCK_D], dtype=tl.float32)
            L34 = tl.zeros([BLOCK_D], dtype=tl.float32)

            for i in range(0, BLOCK_T):
                t = start + i
                if t < T_seq:
                    # Load the six channel slices for step t.
                    base_in  = x_base + t * stride_xt + offs_d * stride_xd
                    base_out = y_base + t * stride_yt + offs_d * stride_yd

                    b12 = tl.load(base_in + 0 * stride_xk, mask=mask_d, other=0.0).to(tl.float32)
                    b13 = tl.load(base_in + 1 * stride_xk, mask=mask_d, other=0.0).to(tl.float32)
                    b14 = tl.load(base_in + 2 * stride_xk, mask=mask_d, other=0.0).to(tl.float32)
                    b23 = tl.load(base_in + 3 * stride_xk, mask=mask_d, other=0.0).to(tl.float32)
                    b24 = tl.load(base_in + 4 * stride_xk, mask=mask_d, other=0.0).to(tl.float32)
                    b34 = tl.load(base_in + 5 * stride_xk, mask=mask_d, other=0.0).to(tl.float32)

                    # Intra-chunk step: L <- L * b.
                    L12, L13, L14, L23, L24, L34 = _u4_combine(
                        L12, L13, L14, L23, L24, L34,
                        b12, b13, b14, b23, b24, b34,
                    )

                    # Output for step t: (S * L). S is the closed chunks'
                    # total product so far; L is the local partial up to t.
                    o12, o13, o14, o23, o24, o34 = _u4_combine(
                        S12, S13, S14, S23, S24, S34,
                        L12, L13, L14, L23, L24, L34,
                    )

                    tl.store(base_out + 0 * stride_yk, o12, mask=mask_d)
                    tl.store(base_out + 1 * stride_yk, o13, mask=mask_d)
                    tl.store(base_out + 2 * stride_yk, o14, mask=mask_d)
                    tl.store(base_out + 3 * stride_yk, o23, mask=mask_d)
                    tl.store(base_out + 4 * stride_yk, o24, mask=mask_d)
                    tl.store(base_out + 5 * stride_yk, o34, mask=mask_d)

            # End of chunk: advance S by the chunk-total L.
            S12, S13, S14, S23, S24, S34 = _u4_combine(
                S12, S13, S14, S23, S24, S34,
                L12, L13, L14, L23, L24, L34,
            )

    def launch(x, block_t: int = 64, block_d: int = 64):
        """Python wrapper — GPU-only.

        x : (B, H, T, 6, D) — one U_4 scan per (batch, head); the trailing
        D runs `D` independent U_4 scans in parallel (elementwise).
        Returns y of the same shape.
        """
        import torch

        assert x.is_cuda, "u4 scan kernel needs CUDA"
        B, H, T, six, D = x.shape
        assert six == 6, f"expected 6 channels in axis -2, got {six}"
        # The trilinear x14 channel grows as O(T³) worst-case; BF16's 7-bit
        # mantissa overflows by T~1024. Output stays fp32 regardless of input.
        y = torch.empty(B, H, T, six, D, dtype=torch.float32, device=x.device)

        num_d = (D + block_d - 1) // block_d
        grid = (B * H * num_d,)
        u4_scan_fwd[grid](
            x, y, T,
            *x.stride(), *y.stride(),
            NUM_HEADS=H, D=D, BLOCK_T=block_t, BLOCK_D=block_d,
            num_warps=4,
        )
        return y

else:

    def launch(x, block_t: int = 64, block_d: int = 64):  # type: ignore[no-redef]
        raise RuntimeError(
            "Triton not available on this platform. "
            "This kernel is GPU-only; use reference.chunked on CPU."
        )
