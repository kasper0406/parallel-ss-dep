# Triton prototypes

GPU kernels for the most promising cells from our Lean formalisation.
Each kernel comes in three files:

```
<cell>/
  reference.py   PyTorch reference — RUNS ON MAC (CPU/MPS)
  kernel.py      Triton source     — validated on GPU cluster
  test.py        Correctness check — RUNS ON MAC (naive loop vs reference)
```

## Why three files?

- Triton has no macOS wheels (no official support, no `triton-cpu` on pip).
- So locally we verify **algorithm correctness** via the PyTorch reference
  and a naive-loop ground truth.
- On the GPU cluster we compile & benchmark `kernel.py` and compare its
  output against the reference.

## Running locally

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
python kernels/linear_attn/test.py
python kernels/heisenberg_d/test.py
python kernels/unipotent_u4/test.py
```

Each `test.py` prints `OK` per structure or raises `AssertionError` with
the max elementwise diff on failure.

## Running on GPU (later)

```bash
pip install triton  # NVIDIA wheel, Linux only
python kernels/<cell>/kernel.py  # compile-only smoke
python kernels/bench.py          # end-to-end perf (TODO)
```

## Cells prototyped

| Cell | Lean source | Combine | Purpose |
|---|---|---|---|
| `linear_attn` | `Affine.lean` + fast-weight | O(d²) | **baseline** for all benchmarks |
| `heisenberg_d` | `HeisenbergD.lean` | O(d²) | novel: causal-pair outer product `Σ_{i<j} aᵢbⱼᵀ` |
| `unipotent_u4` | `Unipotent.lean` | O(1) × 6 | novel: trilinear ordered triple `Σ_{i<j<k}` |

All three use a **Blelloch-style associative scan**; the only thing that
changes across cells is the monoid combine function — same scaffolding.

## Design principle

Each kernel's combine op is a direct transcription of the `Mul` instance
we proved in Lean. The proofs give us correctness of the re-association
strategy; the kernel gets to pick any scan schedule (sequential, Hillis-
Steele, Blelloch) because associativity guarantees they all agree.

For the Mac ⟶ Blackwell path: the kernel template is written against
Triton 3.x semantics (block pointers, `tl.load`/`tl.store`, `tl.dot`),
and assumes the eventual target supports tensor cores. No Blackwell-
specific intrinsics yet — those come in a later tuning pass.
