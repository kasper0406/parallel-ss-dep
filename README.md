# state-dep-parallel

Formalising associativity for **state-dependent, parallelizable RNN cells**
in Lean, then building Triton kernels for the most promising candidates.

The guiding question: *which algebraic structures let a step operator
depend on input / prior state while still collapsing into an associative
prefix-sum scan?* Everything in modern linear-attention land is an
instance of this pattern (Mamba, GLA, DeltaNet, Gated DeltaNet, RWKV, …).
We identify the pattern and search the unexplored corners.

- Research plan: [`PLAN.md`](PLAN.md)
- Lean project: [`StateDep/`](StateDep/)

## Reframing

Every known state-dependent parallelizable RNN cell corresponds to a
**finite-dimensional associative algebra** `A` over ℝ (or an ordered
semiring), where:

- state ∈ A,
- each step is left-multiplication by an input-dependent element of A,
- composition across steps is A's multiplication table.

The abstract parallel-scan correctness theorem (`StateDep/Scan.lean`)
says: for any monoid, any binary-tree re-association of a fold equals
the sequential fold. So *the only* algebraic content a kernel needs is a
`Monoid` instance on the parameter type. The design problem reduces to
finding an `A` that is:

1. low-dimensional (memory bandwidth),
2. cheap to multiply (compute),
3. expressively nonlinear in a useful way.

## What's formalised

All proofs are mechanised in Lean 4 + mathlib; no `sorry`s. Full project
builds from a clean checkout with `lake exe cache get && lake build`.

| # | Module | Structure | Combine | State | Cross-term |
|---|---|---|---|---|---|
| 1 | `Scan.lean` | abstract parallel-scan theorem over any monoid | — | — | — |
| 2 | `Heisenberg.lean` | scalar Heisenberg `H_3` | O(1) | 3 | bilinear |
| 3 | `HeisenbergD.lean` | multi-dim Heisenberg, vector `a, b`, matrix `c` | O(d²) | 2d + d² | bilinear outer product |
| 4 | `Unipotent.lean` | `U_4` as a 6-parameter group | O(1) | 6 | **trilinear ordered triple** |
| 5 | `Rotor.lean` | quaternion rotor cell (non-commutative) | O(1) | 4 | non-commutative group |
| 6 | `Jet.lean` | first-order jets / dual numbers | O(1) | 2 | Leibniz |
| 7 | `Affine.lean` | scalar affine `(a, b)` — SSM backbone | O(1) | 2 | linear |
| 8 | `Tropical.lean` | `Tropical R` + direct `Viterbi (A, b)` cell | O(1) | 2 | (min, +) |
| 9 | `Delta.lean` | Sherman-Morrison composition of rank-1 perts | O(d²) | — | rank-1 → rank-2 |

The only substantive matrix proof is `Delta`'s
`(I − β₁k₁k₁ᵀ)(I − β₂k₂k₂ᵀ) = I − β₁k₁k₁ᵀ − β₂k₂k₂ᵀ + β₁β₂(k₁·k₂)·k₁k₂ᵀ`,
which rides on mathlib's `vecMulVec_mul_vecMulVec`. Everything else
reduces to `ext` into component goals then `ring` / `abel`.

## The interesting observation from the proofs

The hard line in each associativity proof is *always* the single bilinear
cross-term between consecutive steps. Every other coordinate is closed by
`simp` or `rfl`. So "expressivity" in this whole family of architectures
lives in one bilinear form per step, wrapped in an associative envelope —
and the wrapper is the multiplication table of the chosen finite-dim
algebra.

## Novelty table (after literature search)

Verdict: 🟢 open / 🟡 adjacent / 🔴 covered.

| Structure | Combine | State | Closest prior work | Verdict |
|---|---|---|---|---|
| **Multi-d Heisenberg** `Σ_{i<j} aᵢ ⊗ bⱼ` | O(d²) | 2d + d² | [HLA](https://arxiv.org/abs/2510.27258) uses single-index sums `S_tᴷ = Σ kᵢkᵢᵀ` mixed by matrix product — a **different** statistic. [Fast-weight (Schlag 2021)](https://arxiv.org/abs/2102.11174) and [DeltaNet](https://arxiv.org/abs/2406.06484) accumulate `Σ kᵢvᵢᵀ` (same index). Nothing I could find uses the ordered-pair sum directly. | 🟢 |
| **Unipotent U_n** ordered `Σ_{i<j<k}` up to order `n−1` | O(n²) | `n(n−1)/2` | [Bilinear RNN (Csordas 2025)](https://arxiv.org/abs/2505.21749) is bilinear *transition matrix*, different. [Sparse-matrix framework 2506.01966](https://arxiv.org/abs/2506.01966) uses upper triangular *weights*, also different. | 🟢 |
| Scalar Heisenberg `H_3` | O(1) | 3 | — | 🟢 (but too small alone) |
| Quaternion rotor | O(1) | 4 | Well-covered: [QRNN (Parcollet 2018)](https://arxiv.org/abs/1806.04418), [GATr (Brehmer 2023)](https://arxiv.org/abs/2305.18415), [Clifford Group Equivariant](https://proceedings.neurips.cc/paper_files/paper/2023/file/c6e0125e14ea3d1a3de3c33fd2d49fc4-Paper-Conference.pdf) | 🔴 |
| Clifford / geometric algebra | O(2^d) | 2^d | [CliffordLayers](https://microsoft.github.io/cliffordlayers/), [L-GATr NeurIPS'24](https://neurips.cc/virtual/2024/poster/94796), [Geometric Clifford Algebra Networks](https://brandstetter-johannes.github.io/publication/ruhe-2023-cgans/) | 🟡 as a scan primitive |
| Jet / dual number | O(1) | 2 | Standard [dual-number AD](https://arxiv.org/html/2501.04159v1); RTRL as forward-mode AD | 🟡 (likely uninteresting — too little expressivity) |
| Tropical / Viterbi | O(1) / O(d²) | d to d² | [Tropical NN (Zhang 2018)](http://proceedings.mlr.press/v80/zhang18i/zhang18i.pdf), [Min-Max-Plus NN (2021)](https://arxiv.org/abs/2102.06358), [ViterbiNet](https://arxiv.org/pdf/1905.10750), [UltraLIF](https://arxiv.org/html/2602.11206) | 🔴 / 🟡 |
| Affine / SSM | O(d²) | d + d² | Mamba, S4, RWKV, GLA — see [survey](https://arxiv.org/pdf/2503.18970) | 🔴 |
| Delta / Sherman-Morrison | O(d²) w/ WY | d² chunk-bounded | [DeltaNet (Yang 2024)](https://arxiv.org/abs/2406.06484), [Gated DeltaNet ICLR'25](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf), [Kimi Linear](https://www.emergentmind.com/topics/kimi-linear) | 🔴 |

### The novelty claim, pinned down

HLA's masked second-order state (verified against the arXiv HTML) is

```
S_tᴷ   = Σ_{i≤t} kᵢ kᵢᵀ        (single-index key Gram)
C_t^QV = Σ_{i≤t} qᵢ vᵢᵀ         (single-index query-value outer)
o_t    = q_tᵀ · S_tᴷ · C_t^QV
```

— single-index sums multiplied together. The "higher-orderness" is the
matrix product of two same-step statistics, not an ordered-pair sum.

Our multi-d Heisenberg state is explicitly

```
c_t = Σ_{i<j≤t} aᵢ bⱼᵀ          (ordered-pair sum, i strictly before j)
```

These are *different tensors*. Expanding HLA's product and comparing
indices, HLA's statistic is
`Σ_{i,j}(kᵢ · qⱼ) kᵢ vⱼᵀ` — symmetric over `(i,j)` pairs,
whereas Heisenberg's is strictly triangular `i<j`. Unipotent `U_n` gives
the analogous triangular triple/higher-order sums.

## Adjacent "find a non-obvious associative structure" work

Papers that share the methodology (find a monoid where the combine law
is not obvious) but target different algebras:

- [**Log-Linear Attention** (Guo 2025, ICLR'26)](https://arxiv.org/abs/2506.04761) — logarithmically growing state via hierarchical scans. Orthogonal: we could stack a log-linear variant on top of any of our cells.
- [**Kalman Linear Attention** (KLA 2026)](https://arxiv.org/abs/2602.10743) — Kalman precision recursion is a Möbius / fractional-linear map, composes via 2×2 matrix multiplication. Different monoid (GL₂) but exact same discovery pattern as ours.
- [**Matrix Is All You Need** (2506.01966)](https://arxiv.org/abs/2506.01966) — unifies convolution/recurrence/attention under sparse-matrix factorisations (overlapping but different framing).

Tangential but useful context:
- [Higher-Order RNN (Soltani 2016)](https://arxiv.org/pdf/1605.00064), [Multiplicative Integration (Wu 2016)](https://www.researchgate.net/publication/304226007_On_Multiplicative_Integration_with_Recurrent_Neural_Networks), [2nd-order RNN (Sutskever 2011)](https://icml.cc/2011/papers/524_icmlpaper.pdf) — classical bilinear-RNN lineage.
- [Comba (2506.02475)](https://deep-paper.org/en/paper/2506.02475/) — closed-loop bilinear RNN, evidence the bilinear family scales.
- [HGRN (Qin 2023)](https://arxiv.org/abs/2311.04823), [Awesome Linear Attention Survey](https://github.com/btzyd/Awesome-Linear-Attention-Survey), [SSM survey (2025)](https://arxiv.org/pdf/2503.18970), [Efficient Attention survey (2025)](https://arxiv.org/html/2507.19595v3).

## Planned next steps

1. **Triton kernel prototype for multi-d Heisenberg.** Blelloch scan; only the `c`-coordinate does a nontrivial O(d²) outer product per combine. Benchmark vs Gated DeltaNet on matched parameter count.
2. **Lean: chunkwise-scan correctness for `Delta`** — state the explicit growing-rank invariant the kernel preserves.
3. **Lean: `U_n` for arbitrary `n`** — show closure + associativity by induction on the entry index gap `j − i`, making the "correlation order" knob explicit.

## Build

Requires Lean 4.30.0-rc2 via `elan`, `lake`, and network access for the
first `cache get` to download mathlib.

```bash
cd StateDep
source $HOME/.elan/env
lake exe cache get   # pulls prebuilt mathlib oleans
lake build           # ~1s on a warm cache
```

No unit tests yet; `lake build` succeeding is the correctness check —
each module ends with a `Tree.eval_eq_prod` corollary that type-checks
the specific monoid against the abstract scan theorem.
