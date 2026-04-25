# state-dep-parallel

Formalising associativity for **state-dependent, parallelizable RNN cells**
in Lean, then building Triton kernels for the most promising candidates.

The guiding question: *which algebraic structures let a step operator
depend on input / prior state while still collapsing into an associative
prefix-sum scan?* Everything in modern linear-attention land is an
instance of this pattern (Mamba, GLA, DeltaNet, Gated DeltaNet, RWKV, …).
We identify the pattern and search the unexplored corners.

- Research plan: [`PLAN.md`](PLAN.md)
- Brainstorm of 25 candidate monoids: [`IDEAS.md`](IDEAS.md)
- Follow-up literature search: [`LITERATURE.md`](LITERATURE.md)
- GPU / evaluation strategy: [`EVAL_PLAN.md`](EVAL_PLAN.md)
- **Empirical results (2026-04-25): [`RESULTS.md`](RESULTS.md)**
- Lean project: [`StateDep/`](StateDep/)
- Kernel prototypes (PyTorch reference + Triton source + Mac tests): [`kernels/`](kernels/)
- Layer modules + training loop: [`experiments/`](experiments/)

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

Thirteen monoids / groups proved in Lean 4 + mathlib, 1845 lines, no
`sorry`s, builds clean from a warm mathlib cache in ~2 seconds.

| # | Module | Structure | Combine | State | Cross-term |
|---|---|---|---|---|---|
| 1 | `Scan.lean` | abstract parallel-scan theorem over any monoid | — | — | — |
| 2 | `Affine.lean` | scalar affine `(a, b)` — SSM backbone | O(1) | 2 | linear |
| 3 | `Jet.lean` | first-order jets / dual numbers | O(1) | 2 | Leibniz |
| 4 | `Tropical.lean` | `Tropical R` + direct `Viterbi (A, b)` cell | O(1) | 2 | (min, +) |
| 5 | `Heisenberg.lean` | scalar Heisenberg `H_3` | O(1) | 3 | bilinear |
| 6 | `GF2Heisenberg.lean` | Heisenberg over `ZMod 2`; parity via popcount-choose-2 | O(1) | 3 bits | bilinear over GF(2) |
| 7 | `Rotor.lean` | quaternion rotor cell (non-commutative) | O(1) | 4 | non-commutative group |
| 8 | `Dyck.lean` | parenthesis-balance `(c, d) ∈ ℕ²`, `min`-cancel | O(1) | 2 | nested `min` |
| 9 | `Unipotent.lean` | `U_4` as a 6-parameter group | O(1) | 6 | **trilinear ordered triple** |
| 10 | `HeisenbergD.lean` | multi-dim Heisenberg, vector `a, b`, matrix `c` | O(d²) | 2d + d² | bilinear outer product |
| 11 | `Delta.lean` | Sherman-Morrison composition of rank-1 perts | O(d²) | — | rank-1 → rank-2 |
| 12 | `Magnus.lean` | BCH-truncated-at-2 over matrix pairs `(A, B)` | O(d³) | 2d² | commutator `[A, A']` |
| 13 | `Signature.lean` | level-3 truncated tensor-algebra signature `(s, v, M, T)` | O(d³) | 1 + d + d² + d³ | graded tensor product |

Several deserve special note:

- `Scan.lean` is the one piece of infrastructure everyone else rides on: any `Monoid` instance inherits parallel-scan correctness for free.
- `Delta.lean`'s Sherman-Morrison identity `(I − β₁k₁k₁ᵀ)(I − β₂k₂k₂ᵀ) = I − β₁k₁k₁ᵀ − β₂k₂k₂ᵀ + β₁β₂(k₁·k₂)·k₁k₂ᵀ` is the only substantive matrix proof; everything else reduces to `ext` + `ring` / `abel`.
- `GF2Heisenberg.lean` proves `((parity_inputs xs).prod).c = (popcount xs).choose 2 (mod 2)` by induction + Pascal — a direct demonstration that a 3-bit monoid state solves a problem linear RNNs provably cannot at constant width.
- `Dyck.lean` is the only strict-monoid-only cell (no inverses); associativity is a four-way `split_ifs <;> omega`.
- `Signature.lean` formalises the grade-3 tensor algebra and is the universal parent of the bilinear and trilinear cells — multi-d Heisenberg is its grade-2 sliver, `U_n`'s trilinear term is a triangular projection of its grade-3 slot.

## The interesting observation from the proofs

The hard line in each associativity proof is *always* the single bilinear
cross-term between consecutive steps. Every other coordinate is closed by
`simp` or `rfl`. So "expressivity" in this whole family of architectures
lives in one bilinear form per step, wrapped in an associative envelope —
and the wrapper is the multiplication table of the chosen finite-dim
algebra.

## Triton kernels

Three PyTorch reference implementations + Triton kernel sources +
Mac-runnable correctness tests (full details in [`kernels/README.md`](kernels/README.md)):

| Kernel | Reference passes at | Source | Purpose |
|---|---|---|---|
| `kernels/linear_attn` | max \|Δ\| ~1e-13 (fp64) | `Affine.lean` + fast-weight | baseline for all benchmarks |
| `kernels/heisenberg_d` | max \|Δ\| ~1e-12 (fp64) | `HeisenbergD.lean` | novel: causal-pair outer product `Σ_{i<j} aᵢbⱼᵀ` |
| `kernels/unipotent_u4` | max \|Δ\| ~1e-11 (fp64), explicit trilinear check passes | `Unipotent.lean` | novel: trilinear ordered triple `Σ_{i<j<k}` |

Each kernel ships as `reference.py` (PyTorch, runs on Mac) +
`kernel.py` (Triton, GPU-only, import-guarded) + `test.py`
(naive-vs-chunked numerical check, runs on Mac). `kernels/shared/scan.py`
has a sequential / Blelloch / Hillis-Steele scan reference used by the
combined `shared/test_scan.py` (the Blelloch down-sweep's operand order
is load-bearing for non-commutative monoids; the test exercises both
commutative and non-commutative cases).

## First-pass empirical results (2026-04-25, 2× RTX 5090)

Full writeup in [`RESULTS.md`](RESULTS.md); strategy doc in
[`NEXT_DIRECTIONS.md`](NEXT_DIRECTIONS.md). Headline:

- **Triton kernels run on sm_120 / Blackwell consumer** — no
  pytorch#176426 segfault; BF16-input + FP32-accum numerics within
  `EVAL_PLAN.md` §2.3 tolerances after fixing three real kernel bugs.
- **Fused-readout `heisenberg_d` runs at 1.8-4.5× `linear_attn`**.
- **Parity kill-gate cleared at T=64** with a +32.6 pp end-token margin
  for our bilinear Heisenberg cell vs plain linear-attn (98 % vs 50 %).
- **At T=128 our Heisenberg cell hits the wall predicted by Grazzi et al.
  ICLR'25** ([2411.12537](https://arxiv.org/abs/2411.12537)) — its
  transition spectrum ⊂ {1}, so it is TC⁰-stuck and cannot solve parity
  at unbounded T. Same wall hits plain linear-attn and no-posenc softmax.
  DeltaNet/GDN/Mamba2 pass T=128 via engineered tricks
  (`use_short_conv=True`).
- 🎯 **The Grazzi-clean fix: SO(n) orthogonal scan.** Per-channel state
  is an `n×n` orthogonal matrix with input-dependent transitions
  `O_t = exp(X_t)` (skew-symmetric `X_t`). Spectrum lives on the unit
  circle, including `−1`. NC¹-complete via Barrington. **Solves parity
  at T=64, T=128, AND T=256 to 100 % accuracy by training step 1 000**
  — faster than every fla baseline, and beats DeltaNet at T=256 where
  DeltaNet stalls at 56 % q4 within our 5 000-step budget. **Also solves
  T=512** (converges by step 2 000); T=1024 testing in progress.
  PyTorch impl in `experiments/layers.py`; Triton kernel + tests in
  `kernels/ortho_son/`.
- ⚠️ **Concurrent prior art**: [AUSSM (Karuvally et al., July 2025,
  arXiv:2507.05238)](https://arxiv.org/abs/2507.05238) independently
  proposed the same core skew-symmetric input-dependent matrix-exp
  transition. Our differentiation: cumulative matrix state (vs acting on
  vectors), multi-layer convergence-speed comparison, sm_120 Triton
  kernel.
- ❌ **Tested follow-up `RotConjAttention` (semidirect `SO(n) ⋉ ℝ^{n×n}`):**
  the conjugated KV memory `c_t = O_t c_{t-1} O_tᵀ + k⊗v` solves parity
  (matches ortho, slower) but **fails induction-heads recall (3.3 % acc,
  chance is 3.1 %)** alongside pure ortho (2.6 %) and linear-attention
  (8.2 %). Only DeltaNet (100 %) solves recall. **Diagnosis:** recall is
  enabled by the **delta-rule erase `(I − β k kᵀ)`**, not by state
  structure or unbounded memory.
- ❌ **Tested second follow-up `RotDeltaAttention` (rotation + delta-rule
  erase, two-sided action `(I − β k kᵀ)(O c Oᵀ) + β k vᵀ`):** verified
  novel by literature search (two-sided action distinct from
  DeltaProduct's left-only). Implemented; **fails induction (3.9 %),
  including with tanh-bounded skew (5.7 %)**. Mechanistic diagnosis: the
  rotation conjugation `O c Oᵀ` rotates stored keys away from their
  original frame, breaking delta-rule recall. **The two mechanisms —
  rotation (Grazzi-clean) and delta-rule recall — are fundamentally
  incompatible in a single state.** Full diagnostic in `RESULTS.md`
  Phase 7.
- ✅ **Hybrid layer stack `[ortho, deltanet, ortho, deltanet]` solves
  both walls.** Induction recall: 100 % by step 1200. Parity at T=128:
  100 % by step 1200 (stable through step 3000 after one training
  transient). The architectural answer to the "both walls in one model"
  question is **specialist layers, not specialist cells** — each layer
  type handles the wall it can, the residual stream carries information
  between them. Mechanistically distinct from existing fla hybrids
  (which are engineering choices); our framing makes the *reason*
  explicit (the two walls are mechanically distinct and can't share a
  state). Full writeup in `RESULTS.md` Phase 8.

## Novelty table (after `LITERATURE.md` follow-up search)

Verdict: 🟢 open / 🟡 adjacent / 🔴 covered. Full per-candidate
adjacent-work write-ups in [`LITERATURE.md`](LITERATURE.md).

| Structure | Combine | State | Verdict | Note |
|---|---|---|---|---|
| **Truncated signature `T^≤3`** (`Signature.lean`) | O(d³) | 1+d+d²+d³ | 🟢 | verified via Log-NCDE, SLiCEs, SigGate — all use signatures as *input features* or *ODE forcing*, never as **scan state**. Chen product as scan monoid remains open. |
| **Multi-d Heisenberg** `Σ_{i<j} aᵢ⊗bⱼ` (`HeisenbergD.lean`) | O(d²) | 2d+d² | 🟢 | [HLA (Zhang 2025)](https://arxiv.org/abs/2510.27258) uses *symmetric* `(Σ kᵢkᵢᵀ)(Σ qⱼvⱼᵀ)`; DeltaNet / fast-weights use same-index `Σ kᵢvᵢᵀ`. The strictly-triangular ordered-pair sum is unused. |
| **Unipotent U_n** `Σ_{i<j<k}` (`Unipotent.lean`) | O(n²) | n(n−1)/2 | 🟢 | closest: [Bilinear RNN (Csordas 2025)](https://arxiv.org/abs/2505.21749), but that's a bilinear transition matrix, different object. |
| **BCH / Magnus K=2** (`Magnus.lean`) | O(d³) | 2d² | 🟢 | closest: [Log-NCDE (Walker 2024)](https://arxiv.org/abs/2402.18512) uses Magnus-style log-signatures as ODE inputs, not as scan state. Truncated BCH as a monoid in ML is unused. |
| **Dyck / balance** (`Dyck.lean`) | O(1) | 2 | 🟢 | Merrill-Sabharwal [*Illusion of State*](https://arxiv.org/abs/2404.08819) (ICML'24), Strobl et al. TACL'24, Hewitt et al. EMNLP'20 all document Dyck as a named linear-RNN / Transformer failure mode; no one proposes this monoid as the fix. |
| **GF(2)-Heisenberg / parity** (`GF2Heisenberg.lean`) | O(1) | 3 bits | 🟡 | softened. Single-index parity *is* now solved in parallel scans by [Grazzi et al. ICLR'25](https://arxiv.org/abs/2411.12537), [DeltaProduct NeurIPS'25](https://arxiv.org/abs/2502.10297), [PD-SSM NeurIPS'25](https://arxiv.org/abs/2509.22284). Our interest narrows to the **ordered bit-pair** `Σ_{i<j} xᵢxⱼ` form (vs single-index parity). |
| Scalar Heisenberg `H_3` | O(1) | 3 | 🟢 | gateway to multi-d; too small alone |
| Quaternion rotor | O(1) | 4 | 🔴 | [QRNN (Parcollet 2018)](https://arxiv.org/abs/1806.04418), [GATr (Brehmer 2023)](https://arxiv.org/abs/2305.18415) — well-covered |
| Clifford / geometric algebra | O(2^d) | 2^d | 🟡 | as a *scan primitive* specifically, less explored |
| Jet / dual number | O(1) | 2 | 🟡 | likely uninteresting — too little expressivity |
| Tropical / Viterbi | O(1) / O(d²) | d to d² | 🟡 | [Tropical Attention NeurIPS'25](https://arxiv.org/abs/2505.17190) crowds the tropical-scan niche |
| Affine / SSM | O(d²) | d+d² | 🔴 | Mamba / S4 / RWKV / GLA |
| Delta / Sherman-Morrison | O(d²) w/ WY | d² chunk-bounded | 🔴 | DeltaNet, Gated DeltaNet, Kimi Linear |

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
the analogous triangular triple/higher-order sums, and `Signature.lean`
generalises to the universal graded object of which both are projections.

## Adjacent "find a non-obvious associative structure" work

Papers that share the methodology (find a monoid where the combine law
is not obvious) but target different algebras:

- [**Log-Linear Attention** (Guo 2025, ICLR'26)](https://arxiv.org/abs/2506.04761) — logarithmically growing state via hierarchical scans. Orthogonal: we could stack a log-linear variant on top of any of our cells.
- [**Kalman Linear Attention** (KLA 2026)](https://arxiv.org/abs/2602.10743) — Kalman precision recursion is a Möbius / fractional-linear map, composes via 2×2 matrix multiplication. Different monoid (PGL₂) but exact same discovery pattern as ours.
- [**DeltaProduct** (NeurIPS 2025)](https://arxiv.org/abs/2502.10297), [**PD-SSM** (NeurIPS 2025 spotlight)](https://arxiv.org/abs/2509.22284) — recent entrants to the "state-tracking via richer monoids" area; both solve single-index parity via scan.
- [**Matrix Is All You Need** (2506.01966)](https://arxiv.org/abs/2506.01966) — unifies convolution/recurrence/attention under sparse-matrix factorisations.

## Status of planned next steps

The plan from [`EVAL_PLAN.md`](EVAL_PLAN.md), revised in light of the
2026-04-25 results in [`RESULTS.md`](RESULTS.md):

1. ✅ **Port kernels to 2× RTX 5090 dev rig.** Done. No sm_120 segfault.
   Three real bugs fixed (BF16 dtype, state-output dtype, multi-head grid
   indexing). Numerics within `EVAL_PLAN.md` §2.3 tolerances. Fused-readout
   `heisenberg_d` variant added — 1.8-4.5× cost vs `linear_attn`.
2. ✅ **Parity kill-gate at small scale.** Cleared at T=64 with +32.6 pp
   end-token margin vs `linear_attn`. Failed at T=128 (so did
   `linear_attn` and softmax-no-posenc); DeltaNet, Gated DeltaNet, and
   Mamba2 all pass T=128 thanks to their `use_short_conv=True` default.
3. **`use_short_conv=False` ablation on DeltaNet/GDN at T=128.** Open.
   Tells us whether the conv or the delta rule is doing the parity work
   in fla. The most informative single experiment we still owe.
4. **Stack a kernel-4 1D causal conv onto Heisenberg.** Open. If the
   conv does the heavy lifting in fla, the same trick should put us back
   in the SOTA conversation at T=128.
5. **MQAR (Zoology) at small scale.** Open — different separator from
   parity (retrieval, not state-tracking). Per `EVAL_PLAN.md` §3.6, this
   completes the kill-gate suite.
6. **`unipotent_u4` (trilinear cell) at T=128.** Open. Tests whether
   higher-grade tensor monoids extend the horizon past Heisenberg's bilinear.
7. **Add scalar RetNet-style decay to `HeisenbergD.lean`.** Open.
   ~30-line Lean extension preserving the monoid, needed for bounded state
   norm in long training.
8. **SmolLM2-135M distillation** (`EVAL_PLAN.md` §3.3). Blocked on
   (a) deciding whether to first stack short_conv onto Heisenberg
   per (4) and (b) writing a custom autograd Function around the Triton
   kernel (current PyTorch-cumsum reference materialises the d²-state,
   which won't fit at 135M).
9. **`Signature.lean` bench kernel** and **chunkwise-scan correctness for
   `Delta` / general `U_n`**. Lower priority than the measurement loop.

The 25-entry brainstorm of further candidate monoids is in
[`IDEAS.md`](IDEAS.md).

## Build

Lean library — requires `elan`, `lake`, and network access for the first
`lake exe cache get`:

```bash
cd StateDep
source $HOME/.elan/env
lake exe cache get   # pulls prebuilt mathlib oleans
lake build           # ~2s on a warm cache
```

Each module ends with a `Tree.eval_eq_prod` corollary that type-checks
the specific monoid against the abstract scan theorem — the build
succeeding *is* the correctness check.

Kernel tests — from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
python kernels/linear_attn/test.py
python kernels/heisenberg_d/test.py
python kernels/unipotent_u4/test.py
python kernels/shared/test_scan.py
```

All four print `OK`. No GPU required.
