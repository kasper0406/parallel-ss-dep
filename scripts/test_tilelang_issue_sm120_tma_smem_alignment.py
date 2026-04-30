"""Regression test for the sm_120 (Blackwell consumer) TMA shared-memory
alignment bug.

Pre-fix `SharedMemoryAlignmentPlanner` (`merge_shared_memory_allocations.cc`)
gated the 1024-byte alignment of TMA-touched smem buffers on
`TargetIsHopper(target)` (arch in `[90, 100)`). For sm_100 / sm_120 (also
TMA-capable: see `cp.async.bulk.tensor.{1..5}d.shared::cta.global.*` PTX
support gated by `TargetHasBulkCopy(target)` in `src/target/utils.cc`)
the planner fell back to the global default (16 bytes) and TMA destinations
landed at 16-byte-aligned offsets. `cp.async.bulk.tensor.*` requires the
destination smem pointer to be 128-byte aligned, so on sm_100/sm_120 this
caused `CUDA_ERROR_MISALIGNED_ADDRESS` whenever the dynamic-smem arena
started with a small (e.g. 4-byte float) buffer that pushed subsequent TMA
buffers off the 128-byte boundary.

Post-fix the predicate is `TargetHasBulkCopy(target)`, which is true for
arch >= 90 (sm_90, sm_100, sm_103, sm_120 and any future TMA-capable
target) — the correct set, since TMA support and TMA's alignment
requirements are coextensive.

The test has two parts:

1. **Codegen IR check** (host-independent). Compiles the buggy-arena
   kernel for explicit `cuda -arch=sm_{90,100,120}` targets and asserts
   every emitted `tl::tma_load(..., (&((T*)buf_dyn_shmem)[expr]), ...)`
   destination offset is 128-byte aligned. Runs on any host.

2. **Runtime check** on the host GPU. Compiles + executes the same kernel
   for the host arch and asserts it doesn't raise
   `CUDA_ERROR_MISALIGNED_ADDRESS`. Pre-fix on sm_120 this crashes; on
   sm_90 (Hopper) the kernel happens to land buffers correctly even
   pre-fix. Post-fix it passes on every TMA-capable arch.
"""
import ast
import re

import torch

import tilelang
import tilelang.testing
from tilelang import language as T


def _make_buggy_arena_kernel(M, N, K, BM, BN, BK):
    """Standard 2-stage pipelined bf16 GEMM with a tiny T.float32 shared
    scalar allocated *before* the TMA-loaded buffers. The scalar pushes
    the TMA destinations off the 128-byte boundary in the merged
    dynamic-smem arena if the planner doesn't enforce 1024-byte
    per-buffer alignment for TMA-capable targets.

    Modelled on the structure of the fla
    `chunk_bwd_dqkwg_tilelang` kernel (small dg-last accumulator allocated
    before the larger TMA-loaded `s_v` / `s_do` / `s_h` / `s_dh` tiles).
    """

    @T.prim_func
    def gemm(
        a: T.Tensor((M, K), "bfloat16"),
        b: T.Tensor((K, N), "bfloat16"),
        c: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (bx, by):
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


# --------------------------------------------------------------------------
# Codegen IR check (host-independent)
# --------------------------------------------------------------------------

_TMA_DST_RE = re.compile(
    r"tma_load\([^;]*?"               # tma_load call
    r"\(\s*&\s*\(*\s*"                # (&  optional cast wrappers
    r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*\)\s*"  # element type cast e.g. (bfloat16_t*)
    r"buf_dyn_shmem\s*\)*\s*"
    r"\[\s*([^]]+?)\s*\]"             # element-index expression
)

_DTYPE_BYTES = {
    "bfloat16_t": 2, "half_t": 2, "uint16_t": 2, "int16_t": 2,
    "float": 4, "uint32_t": 4, "int32_t": 4,
    "double": 8, "uint64_t": 8, "int64_t": 8,
    "uchar": 1, "uint8_t": 1, "int8_t": 1,
}


def _parse_offset_constants(expr: str):
    """Return `(additive_base, [variable_coefficients])` extracted from a
    TileLang-emitted offset expression. Both — multiplied by sizeof(elt) —
    must be a multiple of 128 for the destination pointer to be 128-byte
    aligned for every runtime variable value.

    C-style bitwise operators (`&`, `|`, `^`) are remapped to `+` so
    Python's `ast` can parse the expression; we only care about the
    structural placement of integer literals, not the operator's
    semantics.
    """
    py_src = expr.replace("&", "+").replace("|", "+").replace("^", "+")
    tree = ast.parse(py_src, mode="eval").body

    base = 0
    strides: list[int] = []

    def _has_name(node) -> bool:
        return any(isinstance(n, ast.Name) for n in ast.walk(node))

    def _classify_term(term):
        nonlocal base
        if isinstance(term, ast.Constant) and isinstance(term.value, int):
            base += term.value
            return
        if isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult):
            left, right = term.left, term.right
            const = var_side = None
            if isinstance(left, ast.Constant) and isinstance(left.value, int):
                const, var_side = left.value, right
            elif isinstance(right, ast.Constant) and isinstance(right.value, int):
                const, var_side = right.value, left
            if const is not None and _has_name(var_side):
                strides.append(const)
                return
        if isinstance(term, ast.BinOp):
            _walk_addends(term.left)
            _walk_addends(term.right)

    def _walk_addends(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            _walk_addends(node.left)
            _walk_addends(node.right)
        else:
            _classify_term(node)

    _walk_addends(tree)
    return base, strides


def _tma_dst_byte_terms(src: str):
    out = []
    for m in _TMA_DST_RE.finditer(src):
        elt_type = m.group(1)
        idx_expr = m.group(2)
        bytes_per = _DTYPE_BYTES.get(elt_type)
        assert bytes_per is not None, (
            f"unrecognized element type '{elt_type}' in tma_load destination"
        )
        base, strides = _parse_offset_constants(idx_expr)
        out.append((base * bytes_per, [s * bytes_per for s in strides], idx_expr, elt_type))
    return out


def _compile_for(arch: str):
    M, N, K = 128, 128, 256
    BM, BN, BK = 64, 64, 32
    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(
            _make_buggy_arena_kernel(M, N, K, BM, BN, BK),
            target=f"cuda -arch={arch}",
            execution_backend="tvm_ffi",
        )
    finally:
        tilelang.enable_cache()
    return kernel.get_kernel_source()


def _assert_tma_destinations_128_aligned(arch: str, src: str):
    terms = _tma_dst_byte_terms(src)
    assert terms, (
        f"[{arch}] expected at least one tl::tma_load(...buf_dyn_shmem...) "
        "in generated source — the kernel layout should produce TMA loads"
    )
    for byte_base, byte_strides, idx_expr, elt_type in terms:
        assert byte_base % 128 == 0, (
            f"[{arch}] TMA load destination has non-128-aligned additive "
            f"base: byte_base={byte_base} "
            f"(idx_expr=({idx_expr}) elt_type={elt_type}). "
            f"cp.async.bulk.tensor.{{1..5}}d.shared::cta.global.* requires "
            f"the destination pointer to be 128-byte aligned at every "
            f"iteration. See merge_shared_memory_allocations.cc:393."
        )
        for stride in byte_strides:
            assert stride % 128 == 0, (
                f"[{arch}] TMA load destination has non-128-aligned "
                f"variable stride: stride={stride} "
                f"(idx_expr=({idx_expr}) elt_type={elt_type}). "
                f"See merge_shared_memory_allocations.cc:393."
            )


@tilelang.testing.requires_cuda
def test_tma_smem_alignment_codegen_sm90_hopper():
    """sm_90 (Hopper) — covered by the original `TargetIsHopper` predicate,
    pinned here as a regression test against re-narrowing."""
    src = _compile_for("sm_90")
    _assert_tma_destinations_128_aligned("sm_90", src)


@tilelang.testing.requires_cuda
def test_tma_smem_alignment_codegen_sm100_blackwell_dc():
    """sm_100 (Blackwell DC) — TMA-capable. Pre-fix this fell back to
    16-byte alignment because `TargetIsHopper` is true only for arch
    `[90, 100)`."""
    src = _compile_for("sm_100")
    _assert_tma_destinations_128_aligned("sm_100", src)


@tilelang.testing.requires_cuda
def test_tma_smem_alignment_codegen_sm120_blackwell_consumer():
    """sm_120 (Blackwell consumer / RTX 5090) — TMA-capable. Pre-fix this
    fell back to 16-byte alignment and triggered
    `CUDA_ERROR_MISALIGNED_ADDRESS` at runtime on real Blackwell consumer
    silicon."""
    src = _compile_for("sm_120")
    _assert_tma_destinations_128_aligned("sm_120", src)


# --------------------------------------------------------------------------
# Runtime check on host GPU
# --------------------------------------------------------------------------


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_smem_alignment_runtime_hostarch():
    """Compile + execute the buggy-arena GEMM on the host's TMA-capable
    GPU. Pre-fix on sm_100 / sm_120 this raises
    `CUDA_ERROR_MISALIGNED_ADDRESS` on the first TMA load. On sm_90
    (Hopper) the same kernel happens to land buffers at acceptable
    offsets and runs even pre-fix. Post-fix it runs and produces correct
    outputs on every TMA-capable host."""
    M, N, K = 128, 128, 256
    BM, BN, BK = 64, 64, 32
    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(
            _make_buggy_arena_kernel(M, N, K, BM, BN, BK),
            target="cuda",
        )
    finally:
        tilelang.enable_cache()
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    kernel(a, b, c)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
