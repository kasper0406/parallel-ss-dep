"""
Linear attention — Triton kernel.

Mirrors `reference.chunked` 1-to-1: the outer loop over chunks carries
the running state `S ∈ R^{d_k × d_v}`; each chunk emits `(inter + intra)`.

The Triton import is guarded so this module can be imported on Mac
(where `triton` is not installed) — the kernel is obviously only *usable*
on a GPU, but the source lives here so it can be validated later.

Launch grid:  (num_heads, num_batches)   ← one program per head-batch
Each program walks the T axis in chunks of `BLOCK_T` and accumulates
`S` in registers / SRAM. `d_k` and `d_v` are assumed ≤ a block tile
(extend with an outer tiling loop when that fails).
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
    def linear_attn_fwd(
        Q_ptr,          # (B, H, T, d_k)
        K_ptr,          # (B, H, T, d_k)
        V_ptr,          # (B, H, T, d_v)
        O_ptr,          # (B, H, T, d_v)
        T_seq,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_ob, stride_oh, stride_ot, stride_od,
        D_K: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """One program = one (batch, head) pair; walks T in BLOCK_T chunks."""
        pid_bh = tl.program_id(0)
        pid_b = pid_bh // tl.num_programs(1)
        pid_h = pid_bh %  tl.num_programs(1)

        # Compute per-head base pointers.
        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh

        # Running state, kept in registers across chunks.
        S = tl.zeros([D_K, D_V], dtype=tl.float32)

        # Reusable index grids for a chunk tile.
        offs_c = tl.arange(0, BLOCK_T)
        offs_dk = tl.arange(0, D_K)
        offs_dv = tl.arange(0, D_V)

        for start in range(0, T_seq, BLOCK_T):
            ts = start + offs_c
            mask_t = ts < T_seq

            # Load Q, K, V tiles for this chunk.
            q = tl.load(
                q_base + ts[:, None] * stride_qt + offs_dk[None, :] * stride_qd,
                mask=mask_t[:, None], other=0.0,
            )
            k = tl.load(
                k_base + ts[:, None] * stride_kt + offs_dk[None, :] * stride_kd,
                mask=mask_t[:, None], other=0.0,
            )
            v = tl.load(
                v_base + ts[:, None] * stride_vt + offs_dv[None, :] * stride_vd,
                mask=mask_t[:, None], other=0.0,
            )

            # Inter-chunk: q @ S.
            inter = tl.dot(q, S)                          # (BLOCK_T, D_V)

            # Intra-chunk: causal dense attention over the BLOCK_T window.
            attn = tl.dot(q, tl.trans(k))                 # (BLOCK_T, BLOCK_T)
            causal = offs_c[None, :] <= offs_c[:, None]
            attn = tl.where(causal, attn, 0.0)
            intra = tl.dot(attn.to(v.dtype), v)

            # Store outputs.
            tl.store(
                o_base + ts[:, None] * stride_ot + offs_dv[None, :] * stride_od,
                (inter + intra).to(v.dtype),
                mask=mask_t[:, None],
            )

            # Advance the running state: S += kᵀ @ v.
            S += tl.dot(tl.trans(k), v).to(tl.float32)

    def launch(q, k, v, block_t: int = 64):
        """Python wrapper — GPU-only. Imports torch lazily."""
        import torch

        assert q.is_cuda and k.is_cuda and v.is_cuda, "linear_attn kernel needs CUDA"
        B, H, T, D_K = q.shape
        _, _, _, D_V = v.shape
        o = torch.empty(B, H, T, D_V, dtype=q.dtype, device=q.device)

        grid = (B * H,)
        linear_attn_fwd[grid](
            q, k, v, o, T,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            D_K=D_K, D_V=D_V, BLOCK_T=block_t,
            num_warps=4,
        )
        return o

else:

    def launch(q, k, v, block_t: int = 64):  # type: ignore[no-redef]
        raise RuntimeError(
            "Triton not available on this platform. "
            "This kernel is GPU-only; use reference.chunked on CPU."
        )
