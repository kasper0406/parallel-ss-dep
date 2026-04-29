# Project history (pre-Phase-14 work)

This document preserves the architectural exploration that preceded the
**sparse cross-layer feedback** finding. The README leads with the
current architectural result; this file records the earlier work for
context and reproducibility.

## Earlier headline findings (Phases 1-13)

1. **Two architecturally distinct walls** in the parallel-scan-friendly
   design space, each with a published characterisation:
   - **Grazzi's TC⁰ wall** ([Grazzi et al. ICLR'25][grazzi]) — a linear
     RNN with transition spectrum in `[0, 1]^d` cannot express parity
     at unbounded T. Escape requires negative or unit-circle eigenvalues.
   - **The recall wall** ([Zoology / MQAR][zoology]) — recall requires
     a **rank-1 erase `(I − β k kᵀ)` applied per token in a fixed
     frame**. Architectures without it fail induction-heads and MQAR
     even at small scale.
2. **The two walls are mechanically incompatible inside a single state.**
   The rotation that escapes wall 1 (input-dependent two-sided action
   `c → O c Oᵀ`) is the same conjugation that destroys the fixed-frame
   structure recall depends on. We tested four single-cell architectures
   that try to escape both — all fail recall (chance) regardless of
   parameterisation, including a tanh-bounded variant that should
   minimise frame disruption.
3. **The minimum architectural escape is a heterogeneous layer stack.**
   `[ortho, deltanet, ortho, deltanet]` — alternating SO(n) scan layers
   (Grazzi-clean parity) and DeltaNet layers (fixed-frame recall) —
   solves both walls cleanly:
   - Parity at T=512: **100 %** (converges step 1500)
   - Induction-heads recall: **100 %** (converges step 1200)
   - All at 0.94 M params, 4 layers total, 3000 AdamW steps.
3a. **Hybrid empirically dominates `DeltaNet+allow_neg_eigval=True`
    (the strongest published single-cell baseline) on modular addition
    mod-p**, especially at long T:
    - T=128 mod-3 / mod-5: hybrid solves at step 1500-2000, deltanet_negeig
      at step 4000 (2-3× faster).
    - T=512 mod-3: hybrid 100 %, deltanet_negeig 93 %.
    - **T=512 mod-5: hybrid 100 %, deltanet_negeig catastrophically
      diverges to 9.1 % (below random 20 %).**
4. **Honest task-dependent picture** (`RESULTS.md` Phases 10-12):
   - Hybrid wins **continuous-angle modular arithmetic** (Z_p ⊂ SO(2)).
   - DeltaNet+`allow_neg_eigval=True` wins **discrete-reflection
     non-solvable group state-tracking** (S₅ word problem at T=128:
     0.98 pos_recall vs hybrid's 0.71). Householder reflectors are a
     more direct fit for transpositions than rotations.
   - Pure DeltaNet (with `use_short_conv=True`) wins **real-text LM
     out of the box** (TinyStories, 135M, 5000 steps: PPL 5.65 vs
     hybrid v1's 9.79).
5. **Hybrid v2 with engineering tricks closes the LM gap to 1.19×**
   (`RESULTS.md` Phase 12). Adding `use_short_conv=True`, SiLU input
   activation, and L2-normalised rotation target to the ortho layer
   brought hybrid PPL from 9.79 → **6.73** (DeltaNet 5.65), shrinking
   the gap from 73 % worse to 19 % worse.
6. **Hybrid 25/75 ratio (deltanet-heavy) is the right recipe for
   code-LM** (`RESULTS.md` Phase 13). On Python code at 135M for
   5000 steps:
   - 0 % ortho (pure DeltaNet): PPL 51.00
   - **25 % ortho (1 per 4 layers): PPL 54.73 (1.07×)** ⭐
   - 50 % ortho (alternating): PPL 62.13 (1.22×)
   - 75 % ortho: PPL 78.74 (1.54×)
7. **Dyck-2 bracket depth-tracking does *not* separate the architectures**
   — both deltanet+conv and hybrid solve to 100 % even at T=512.
8. Hybrid Triton kernel + custom autograd brings hybrid to 1.27×
   DeltaNet wall-clock at the 25/75 ratio.

## How this work relates to AUSSM (and what was actually new)

Mid-session we discovered **[AUSSM (Karuvally et al., July 2025,
arXiv:2507.05238)][aussm]** independently proposed the same skew-
symmetric input-dependent matrix-exp construction we built as
`OrthogonalScanAttention`. They shipped it ~3 months before us. Honest
accounting:

|  | AUSSM | This project |
|---|---|---|
| **Cell algebra** (skew → exp → unit-circle eigenvalues) | ✓ | concurrent |
| **State per channel** | scalar / vector | **n×n matrix**, non-commuting transitions |
| **Parity / mod-p / solvable-group automata** | ✓ | ✓ |
| **MQAR / associative recall benchmarks** | not evaluated | addressed via DeltaNet layer in hybrid |
| **Combined with DeltaNet rank-1 erase** | suggested | **hybrid layer stack tested at LM scale** |
| **Mechanistic incompatibility theorem** | not stated | **explicit** (rotation ⨯ delta-rule cannot share a state) |
| **Triton kernel for sm_120** | absent | **ships** |

The AUSSM-equivalent piece is `OrthogonalScanAttention`. That cell is
not from-scratch novelty; AUSSM is concurrent prior art for the
construction. The new contributions of the early phases were:

1. The architectural decomposition (rotation specialist + delta-rule
   specialist + their incompatibility argument).
2. The mod-p result: hybrid solves T=512 mod-5 at 100 % where
   `DeltaNet+allow_neg_eigval=True` catastrophically fails to 9 %.
3. The 135M-scale hybrid v2 demonstration on TinyStories.
4. The Triton sm_120 kernel for the matmul-scan with custom autograd
   (4.2× speedup over PyTorch loop).
5. The Lean library formalising 13 monoids whose associativity gives
   parallel-scan correctness.

## The reframing (algebraic view of state-dependent RNNs)

Every known state-dependent parallelizable RNN cell corresponds to a
**finite-dimensional associative algebra** `A` over ℝ (or an ordered
semiring), where:
- state ∈ A,
- each step is left-multiplication by an input-dependent element of A,
- composition across steps is A's multiplication table.

The abstract parallel-scan correctness theorem
([`StateDep/Scan.lean`](StateDep/Scan.lean)) says: for any monoid, any
binary-tree re-association of a fold equals the sequential fold. So
the only algebraic content a kernel needs is a `Monoid` instance on
the parameter type. Design problem reduces to finding an `A` that is
(1) low-dimensional (memory bandwidth), (2) cheap to multiply
(compute), (3) expressively nonlinear in a useful way.

## The wall framing (visual)

```
                                       ┌─ recall wall ──┐
                                       │  (Zoology /    │
                                       │   MQAR — needs │
                                       │   fixed-frame  │
                                       │   rank-1       │
                                       │   erase)       │
                                       └────────────────┘
                                              │
                            ┌─────────────────┴─────────────────┐
                            │                                   │
                ┌─ Grazzi   │                                   │ ─┐
                │  TC⁰ wall │       SINGLE CELL                  │  │ HYBRID
                │  (parity  │       cannot escape both          │  │ STACK
                │  needs    │       (rotation breaks frame)     │  │ escapes
                │  spectrum │                                   │  │ both
                │  outside  │                                   │  │
                │  [0, 1])  │                                   │  │
                └───────────┘                                   └──┘
```

## Solutions tested (cell level)

Eight architectures tested at matched ~1 M params, 4 layers, 3-5 k AdamW
steps, on parity (state-tracking) and induction-heads (recall).

### 1. Heisenberg cross-pair `c_t = Σ_{i<j} aᵢ ⊗ bⱼ`  *(our original novelty)*

- Lean: [`StateDep/HeisenbergD.lean`](StateDep/HeisenbergD.lean)
- Triton: [`kernels/heisenberg_d/`](kernels/heisenberg_d/) — 1.8-4.5× `linear_attn` cost.
- Result: solves T=64 parity, **fails T=128 (TC⁰ stuck)**.
- Literature: distinct from HLA's symmetric `(Σ kᵢkᵢᵀ)(Σ qⱼvⱼᵀ)` and from
  DeltaNet's single-index `Σ kᵢvᵢᵀ`. The strictly-triangular ordered-pair
  sum as scan state is novel; cell is TC⁰-stuck so it doesn't compete
  with Grazzi-clean architectures.

### 2. SO(n) scan (`ortho`) `O_t = exp(skew(W_skew · x_t))`, `R_t = O_t R_{t-1}`

- PyTorch: [`experiments/layers.py:OrthogonalScanAttention`](experiments/layers.py).
- Triton: [`kernels/ortho_son/`](kernels/ortho_son/).
- Result: parity at T=64-512 all 100 %, **fails induction (chance)**.
- Literature: AUSSM is concurrent prior art for the construction.
  Our distinct angle: matrix-state rather than acting-on-vector,
  sm_120 Triton kernel.

### 3. Semidirect `SO(n) ⋉ ℝ^{n×n}` (`rotconj`)

State `(R_t, c_t)`, `c_t = O_t c_{t-1} O_tᵀ + k ⊗ v`.
- Result: solves parity (slower than ortho); **fails induction (chance)**.
- Diagnosis: additive c-update saturates with noise; rotation
  conjugation alone doesn't enable recall.

### 4. Rotation + delta-rule (`rotdelta`)

`c_t = (I − β k kᵀ)(O_t c_{t-1} O_tᵀ) + β k vᵀ`.
- Result: parity (slower); **fails induction at chance** in two
  variants (unbounded skew + tanh-bounded skew ≤ 0.5 rad).
- Conclusion: this combination *cannot work* mechanistically because
  rotation conjugation rotates stored keys away from their original
  frame; the inner-product alignment delta-rule recall depends on is
  broken regardless of rotation magnitude. **Mechanistic
  incompatibility**.

### 5. Hybrid layer stack `[ortho, deltanet, ortho, deltanet]`

- PyTorch: [`experiments/train_hybrid.py`](experiments/train_hybrid.py)
  + `attention_cls_per_layer` arg.
- Result: parity at T=128 → 100 % (step 1200), parity at T=512 →
  100 % (step 1500), induction → 100 % (step 1200). Both walls escaped.
- Closest published precedent: **Olmo-Hybrid** (Merrill, Li et al.
  2026, arXiv:2604.03444) — transformer + Gated DeltaNet tied to a
  TC⁰ → NC¹ argument, but their state-tracking layer is DeltaNet's
  internal Householder. Other hybrid stacks (Qwen3-Next, Hymba, Samba,
  Jamba, Zamba, MAD) frame the layer split via empirical capability
  decomposition, not via the two specific walls we identified.

## Where this places the early work

- The cell-level novelty (`heisenberg_d`, `rotconj`, `rotdelta`) is real
  but those scan structures don't out-perform DeltaProduct or
  DeltaNet+`allow_neg_eigval=True` at single-cell expressivity.
  **Single-cell unification of parity + recall is essentially solved
  by DeltaProduct** ([Yang et al. 2025][deltaproduct]); we add nothing
  to that picture beyond extra cells in the same equivalence class.
- The cleanest contribution of this phase is the **two-walls +
  incompatibility framing** and the **minimum hybrid stack**.
  Olmo-Hybrid argues "transformer + DeltaNet ⇒ TC⁰ ∪ NC¹" but doesn't
  decompose into rotation + erase as separate primitives, and doesn't
  state the impossibility direction. Our `[ortho, deltanet]` is the
  smallest concrete witness.

## Lean library (13 monoids)

Thirteen monoids / groups proved associative in Lean 4 + mathlib,
~1845 lines, no `sorry`s, builds clean from a warm cache in ~2s:

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
| 9 | `Unipotent.lean` | `U_4` as a 6-parameter group | O(1) | 6 | trilinear ordered triple |
| 10 | `HeisenbergD.lean` | multi-dim Heisenberg, vector `a, b`, matrix `c` | O(d²) | 2d + d² | bilinear outer product |
| 11 | `Delta.lean` | Sherman-Morrison composition of rank-1 perts | O(d²) | — | rank-1 → rank-2 |
| 12 | `Magnus.lean` | BCH-truncated-at-2 over matrix pairs `(A, B)` | O(d³) | 2d² | commutator `[A, A']` |
| 13 | `Signature.lean` | level-3 truncated tensor-algebra signature `(s, v, M, T)` | O(d³) | 1 + d + d² + d³ | graded tensor product |

## Triton kernels for sm_120 (Blackwell consumer / RTX 5090)

Four cells with PyTorch reference + Triton source + Mac-runnable
correctness tests:

| Kernel | Purpose | Status on sm_120 |
|---|---|---|
| [`kernels/linear_attn/`](kernels/linear_attn/) | baseline linear attention | ✓ correctness, ~893 Mtok/s peak |
| [`kernels/heisenberg_d/`](kernels/heisenberg_d/) | bilinear cross-pair, with fused-readout variant | ✓ correctness, 1.8-4.5× linear_attn |
| [`kernels/unipotent_u4/`](kernels/unipotent_u4/) | trilinear ordered triple | ✓ correctness, 5-11× linear_attn |
| [`kernels/ortho_son/`](kernels/ortho_son/) | matrix-multiplication scan (SO(n)) | ✓ correctness on FP32 |

No pytorch#176426 (sm_120 multi-`tl.load`) segfault on any kernel.

Three kernel bugs found and fixed:
1. `linear_attn` BF16-input dtype mismatch in `tl.dot(q, S)`;
2. `heisenberg_d` / `unipotent_u4` writing state outputs in BF16
   (overflow at T=2048);
3. `linear_attn` / `unipotent_u4` deriving head-index from
   `tl.num_programs(1)` on a 1-D grid.

## Possible future work (still relevant)

- **S₅ word problem at long T** — tests *non-solvable* state-tracking.
  SO(n) for n ≥ 3 contains A₅ in principle; SGD-findability is open.
  DeltaProduct has the cleanest published numbers; a direct comparison
  would slot in.
- **Lean: incompatibility theorem.** Formalise *"two-sided rotation
  conjugation cannot host both Grazzi-clean spectrum and a fixed-frame
  recall basis simultaneously"* in Lean. New module
  `IncompatibilityTheorem.lean`.

[grazzi]: https://arxiv.org/abs/2411.12537
[zoology]: https://arxiv.org/abs/2312.04927
[aussm]: https://arxiv.org/abs/2507.05238
[deltaproduct]: https://arxiv.org/abs/2502.10297
