"""
d-dimensional Heisenberg monoid scan — Triton kernel.

Mirrors `reference.chunked` 1-to-1: the outer loop over chunks carries
`(running_A ∈ R^d, running_C ∈ R^{d×d})` in registers, and each chunk
emits the per-token `c_t = running_C + running_A ⊗ b_c[t] + A_excl[t] ⊗ b_c[t]`.

Triton import is guarded so this module can be imported on Mac (where
`triton` is not installed). The kernel is obviously only *usable* on
a GPU, but the source lives here for later validation on the cluster.

Launch grid:  (B * H,)   ← one program per (batch, head)
Each program walks the T axis in chunks of `BLOCK_T` and accumulates
the Heisenberg state in SRAM / registers. `d` is assumed ≤ a block tile
(extend with an outer tiling loop when that fails).

Inputs (one program = one (batch, head)):
    A_ptr : (B, H, T, d)
    B_ptr : (B, H, T, d)
Outputs:
    Oa_ptr : (B, H, T, d)        per-token running_a  (inclusive)
    Ob_ptr : (B, H, T, d)        per-token running_b  (inclusive)
    Oc_ptr : (B, H, T, d, d)     per-token running_c  (strict i<j cross-pair
                                 outer-product statistic — the novel output)

The chunkwise update corresponds to the following math, written in the
Heisenberg monoid's (a, b, c) form where `*` is the monoid op:

    (running_A, _, running_C) *  chunk_aggregate
      = (running_A + ΣA_chunk,
         ...,
         running_C + running_A ⊗ ΣB_chunk + Σ_{s<t within chunk} a_c[s] ⊗ b_c[t])

All inner products land on `tl.dot`-friendly shapes: the intra-chunk
cross term `Σ_{s<t} a_c[s] ⊗ b_c[t]` is computed as `A_exclᵀ @ b_c`,
a `(d × BLOCK_T) · (BLOCK_T × d)` GEMM.
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
    def heisenberg_d_fwd(
        A_ptr,          # (B, H, T, d)
        B_ptr,          # (B, H, T, d)
        Oa_ptr,         # (B, H, T, d)
        Ob_ptr,         # (B, H, T, d)
        Oc_ptr,         # (B, H, T, d, d)  — fp32, see launch()
        T_seq,
        stride_ab, stride_ah, stride_at, stride_ad,
        stride_bb, stride_bh, stride_bt, stride_bd,
        stride_oab, stride_oah, stride_oat, stride_oad,
        stride_obb, stride_obh, stride_obt, stride_obd,
        stride_ocb, stride_och, stride_oct, stride_oci, stride_ocj,
        NUM_HEADS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """One program = one (batch, head) pair; walks T in BLOCK_T chunks."""
        pid_bh = tl.program_id(0)
        pid_b = pid_bh // NUM_HEADS
        pid_h = pid_bh %  NUM_HEADS

        # Per-head base pointers.
        a_base  = A_ptr  + pid_b * stride_ab  + pid_h * stride_ah
        b_base  = B_ptr  + pid_b * stride_bb  + pid_h * stride_bh
        oa_base = Oa_ptr + pid_b * stride_oab + pid_h * stride_oah
        ob_base = Ob_ptr + pid_b * stride_obb + pid_h * stride_obh
        oc_base = Oc_ptr + pid_b * stride_ocb + pid_h * stride_och

        # Running Heisenberg state, carried in registers across chunks.
        #   run_A : Σ_{i<chunk_start} a_i                  — shape (D,)
        #   run_B : Σ_{i<chunk_start} b_i                  — shape (D,)
        #   run_C : Σ_{i<j<chunk_start} a_i ⊗ b_j         — shape (D, D)
        run_A = tl.zeros([D], dtype=tl.float32)
        run_B = tl.zeros([D], dtype=tl.float32)
        run_C = tl.zeros([D, D], dtype=tl.float32)

        # Reusable index grids.
        offs_c = tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, D)
        offs_i = tl.arange(0, D)
        offs_j = tl.arange(0, D)

        for start in range(0, T_seq, BLOCK_T):
            ts = start + offs_c
            mask_t = ts < T_seq

            # Load the chunk tiles. Masked lanes are zeroed so they contribute
            # nothing to any reduction.
            a_tile = tl.load(
                a_base + ts[:, None] * stride_at + offs_d[None, :] * stride_ad,
                mask=mask_t[:, None], other=0.0,
            )                                                   # (BLOCK_T, D)
            b_tile = tl.load(
                b_base + ts[:, None] * stride_bt + offs_d[None, :] * stride_bd,
                mask=mask_t[:, None], other=0.0,
            )                                                   # (BLOCK_T, D)

            # --- Intra-chunk exclusive prefix-sum of `a` (strict s<t).
            # A_excl[t, :] = Σ_{s<t} a_tile[s, :]
            # The cleanest Triton expression is a masked broadcast-and-sum:
            # let P[t, s] = 1 iff s<t. Then A_excl = P @ a_tile.
            idx_row = offs_c[:, None]                            # t
            idx_col = offs_c[None, :]                            # s
            strict_lower = (idx_col < idx_row) & mask_t[:, None] & mask_t[None, :]
            P = tl.where(strict_lower, 1.0, 0.0).to(tl.float32)  # (BLOCK_T, BLOCK_T)
            A_excl = tl.dot(P, a_tile.to(tl.float32))            # (BLOCK_T, D)

            # --- Per-token outputs.
            # Inclusive prefix sums for running_a / running_b outputs:
            # use P_incl[t, s] = 1 iff s<=t.
            incl_mask = (idx_col <= idx_row) & mask_t[:, None] & mask_t[None, :]
            P_incl = tl.where(incl_mask, 1.0, 0.0).to(tl.float32)
            A_incl = tl.dot(P_incl, a_tile.to(tl.float32))       # (BLOCK_T, D)
            B_incl = tl.dot(P_incl, b_tile.to(tl.float32))       # (BLOCK_T, D)

            out_a = run_A[None, :] + A_incl                      # (BLOCK_T, D)
            out_b = run_B[None, :] + B_incl                      # (BLOCK_T, D)

            tl.store(
                oa_base + ts[:, None] * stride_oat + offs_d[None, :] * stride_oad,
                out_a.to(a_tile.dtype),
                mask=mask_t[:, None],
            )
            tl.store(
                ob_base + ts[:, None] * stride_obt + offs_d[None, :] * stride_obd,
                out_b.to(b_tile.dtype),
                mask=mask_t[:, None],
            )

            # Per-token c output. Recall that c_t = Σ_{i<j≤t} a_i ⊗ b_j
            # splits as
            #     c_t = run_C + run_A ⊗ B_incl[l] + C_intra[l]
            # where `C_intra[l] = Σ_{j'≤l} A_excl[j'] ⊗ b_tile[j']` is the
            # running "Heisenberg cumsum" within the chunk. We build this
            # as a BLOCK_T-step rank-1 update in SRAM, emitting each D×D
            # slice as we go. This mirrors `reference.chunked`'s cumsum.
            c_intra = tl.zeros([D, D], dtype=tl.float32)
            b_tile_f32 = b_tile.to(tl.float32)
            for local_t in tl.static_range(0, BLOCK_T):
                t_global = start + local_t
                active = t_global < T_seq
                # Select the local_t-th row of A_excl, b_tile, B_incl.
                row_mask = offs_c == local_t
                a_excl_row = tl.sum(
                    tl.where(row_mask[:, None], A_excl, 0.0), axis=0
                )                                                # (D,)
                b_row = tl.sum(
                    tl.where(row_mask[:, None], b_tile_f32, 0.0), axis=0
                )                                                # (D,)
                B_incl_row = tl.sum(
                    tl.where(row_mask[:, None], B_incl, 0.0), axis=0
                )                                                # (D,)
                # Inclusive rank-1 update: C_intra[l] = C_intra[l-1] + A_excl[l] ⊗ b[l].
                c_intra = c_intra + a_excl_row[:, None] * b_row[None, :]
                # Final c value at this token.
                c_tile = run_C + run_A[:, None] * B_incl_row[None, :] + c_intra
                ptrs = (
                    oc_base
                    + t_global * stride_oct
                    + offs_i[:, None] * stride_oci
                    + offs_j[None, :] * stride_ocj
                )
                tl.store(ptrs, c_tile, mask=active)

            # --- Advance the running state by the full chunk.
            # ΣA_chunk, ΣB_chunk: last inclusive prefix row is the chunk sum
            # over only the *valid* (unmasked) lanes, because masked lanes
            # were zeroed on load.
            sumA = tl.sum(a_tile.to(tl.float32), axis=0)         # (D,)
            sumB = tl.sum(b_tile.to(tl.float32), axis=0)         # (D,)

            # Intra-chunk cross aggregate Σ_{s<t} a[s] ⊗ b[t] = A_exclᵀ @ b.
            cross_chunk = tl.dot(
                tl.trans(A_excl), b_tile.to(tl.float32)
            )                                                     # (D, D)

            run_C = run_C + run_A[:, None] * sumB[None, :] + cross_chunk
            run_A = run_A + sumA
            run_B = run_B + sumB

    @triton.jit
    def heisenberg_d_readout_fwd(
        Q_ptr,          # (B, H, T, d)
        A_ptr,          # (B, H, T, d)
        B_ptr,          # (B, H, T, d)
        O_ptr,          # (B, H, T, d) — q_t · c_t readout, fp output dtype
        T_seq,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_ab, stride_ah, stride_at, stride_ad,
        stride_bb, stride_bh, stride_bt, stride_bd,
        stride_ob, stride_oh, stride_ot, stride_od,
        NUM_HEADS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Fused-readout Heisenberg scan.

        Per token computes  o_t = q_t · c_t  where  c_t = Σ_{i<j≤t} a_i ⊗ b_j.
        c_t is never materialized — only the D-vector readout is emitted.

        Within a chunk starting at `s` with local index `l`:
          c_t = run_C + run_A ⊗ B_incl[l] + Σ_{l₁<l₂≤l} a[l₁] ⊗ b[l₂]
          o_t = q_l · run_C
              + (q_l · run_A) * B_incl[l]
              + Σ_{l₂≤l} (q_l · A_excl[l₂]) * b[l₂]

        The three terms become three GEMMs per chunk — same structure as the
        baseline `linear_attn` kernel.
        """
        pid_bh = tl.program_id(0)
        pid_b = pid_bh // NUM_HEADS
        pid_h = pid_bh %  NUM_HEADS

        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        a_base = A_ptr + pid_b * stride_ab + pid_h * stride_ah
        b_base = B_ptr + pid_b * stride_bb + pid_h * stride_bh
        o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh

        run_A = tl.zeros([D], dtype=tl.float32)
        run_C = tl.zeros([D, D], dtype=tl.float32)

        offs_c = tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, D)

        for start in range(0, T_seq, BLOCK_T):
            ts = start + offs_c
            mask_t = ts < T_seq

            q_tile = tl.load(
                q_base + ts[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                mask=mask_t[:, None], other=0.0,
            )                                                   # (BLOCK_T, D)
            a_tile = tl.load(
                a_base + ts[:, None] * stride_at + offs_d[None, :] * stride_ad,
                mask=mask_t[:, None], other=0.0,
            )
            b_tile = tl.load(
                b_base + ts[:, None] * stride_bt + offs_d[None, :] * stride_bd,
                mask=mask_t[:, None], other=0.0,
            )

            idx_row = offs_c[:, None]
            idx_col = offs_c[None, :]
            strict_lower = (idx_col < idx_row) & mask_t[:, None] & mask_t[None, :]
            P = tl.where(strict_lower, 1.0, 0.0).to(tl.float32)
            A_excl = tl.dot(P, a_tile.to(tl.float32))           # (BLOCK_T, D)

            incl_mask = (idx_col <= idx_row) & mask_t[:, None] & mask_t[None, :]
            P_incl = tl.where(incl_mask, 1.0, 0.0).to(tl.float32)
            B_incl = tl.dot(P_incl, b_tile.to(tl.float32))      # (BLOCK_T, D)

            # Readout decomposition:
            # 1. q_tile @ run_C  →  (BLOCK_T, D)
            o_inter1 = tl.dot(q_tile.to(tl.float32), run_C)

            # 2. (q_tile · run_A) * B_incl  →  (BLOCK_T, D)
            q_dot_runA = tl.sum(q_tile.to(tl.float32) * run_A[None, :], axis=1)
            o_inter2 = q_dot_runA[:, None] * B_incl

            # 3. Causal intra-chunk linear-attn pattern:
            #    Σ_{l₂≤l} (q_l · A_excl[l₂]) * b[l₂]
            attn = tl.dot(q_tile.to(tl.float32), tl.trans(A_excl))  # (BLOCK_T, BLOCK_T)
            causal = (idx_col <= idx_row) & mask_t[:, None] & mask_t[None, :]
            attn = tl.where(causal, attn, 0.0)
            o_intra = tl.dot(attn, b_tile.to(tl.float32))       # (BLOCK_T, D)

            o_tile = o_inter1 + o_inter2 + o_intra
            tl.store(
                o_base + ts[:, None] * stride_ot + offs_d[None, :] * stride_od,
                o_tile.to(q_tile.dtype),
                mask=mask_t[:, None],
            )

            # End-of-chunk state advance — same as `heisenberg_d_fwd`.
            sumA = tl.sum(a_tile.to(tl.float32), axis=0)
            sumB = tl.sum(b_tile.to(tl.float32), axis=0)
            cross_chunk = tl.dot(tl.trans(A_excl), b_tile.to(tl.float32))
            run_C = run_C + run_A[:, None] * sumB[None, :] + cross_chunk
            run_A = run_A + sumA

    def launch_with_readout(q, a, b, block_t: int = 64):
        """Fused readout: returns o (B,H,T,D) = q_t · c_t per token.

        Apples-to-apples comparison point with `linear_attn` — emits a single
        D-vector per token, never materializes the d×d state.
        """
        import torch

        assert q.is_cuda and a.is_cuda and b.is_cuda
        assert q.shape == a.shape == b.shape, \
            f"q/a/b shape mismatch: {q.shape}, {a.shape}, {b.shape}"
        B, H, T, D = q.shape
        o = torch.empty(B, H, T, D, dtype=q.dtype, device=q.device)

        grid = (B * H,)
        heisenberg_d_readout_fwd[grid](
            q, a, b, o, T,
            *q.stride(), *a.stride(), *b.stride(), *o.stride(),
            NUM_HEADS=H, D=D, BLOCK_T=block_t,
            num_warps=4,
        )
        return o

    def launch(a, b, block_t: int = 64):
        """Python wrapper — GPU-only. Imports torch lazily.

        a, b: (B, H, T, d). Returns (out_a, out_b, out_c) with shapes
        (B, H, T, d), (B, H, T, d), (B, H, T, d, d).
        """
        import torch

        assert a.is_cuda and b.is_cuda, "heisenberg_d kernel needs CUDA"
        assert a.shape == b.shape, f"a/b shape mismatch: {a.shape} vs {b.shape}"
        B, H, T, D = a.shape
        out_a = torch.empty(B, H, T, D, dtype=a.dtype, device=a.device)
        out_b = torch.empty(B, H, T, D, dtype=a.dtype, device=a.device)
        # The d×d state grows as Σ_{i<j} aᵢbⱼᵀ ~ O(T) random-walk, O(T²) worst
        # case — BF16's 7-bit mantissa cannot represent it; force fp32 output.
        out_c = torch.empty(B, H, T, D, D, dtype=torch.float32, device=a.device)

        grid = (B * H,)
        heisenberg_d_fwd[grid](
            a, b, out_a, out_b, out_c, T,
            *a.stride(), *b.stride(),
            *out_a.stride(), *out_b.stride(), *out_c.stride(),
            NUM_HEADS=H, D=D, BLOCK_T=block_t,
            num_warps=4,
        )
        return out_a, out_b, out_c

else:

    def launch(a, b, block_t: int = 64):  # type: ignore[no-redef]
        raise RuntimeError(
            "Triton not available on this platform. "
            "This kernel is GPU-only; use reference.chunked on CPU."
        )

    def launch_with_readout(q, a, b, block_t: int = 64):  # type: ignore[no-redef]
        raise RuntimeError(
            "Triton not available on this platform. "
            "This kernel is GPU-only."
        )
