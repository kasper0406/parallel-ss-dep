# state-dep-parallel

Mapping the design space of **state-dependent, parallelizable RNN cells**
— Lean formalisation, Triton kernels for Blackwell sm_120, and a
clean diagnostic decomposition of why the dominant architectures
(DeltaNet, Mamba2, AUSSM, …) succeed where they do.

- [`RESULTS.md`](RESULTS.md) — full empirical writeup
- [`NEXT_DIRECTIONS.md`](NEXT_DIRECTIONS.md) — strategy doc with literature search
- [`EVAL_PLAN.md`](EVAL_PLAN.md) — GPU + evaluation plan
- [`PLAN.md`](PLAN.md) — original research plan
- [`IDEAS.md`](IDEAS.md), [`LITERATURE.md`](LITERATURE.md) — earlier brainstorm + lit search
- [`StateDep/`](StateDep/) — Lean project
- [`kernels/`](kernels/) — Triton kernels
- [`experiments/`](experiments/) — training drivers and architecture modules

## Headline findings

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
   the gap from 73 % worse to 19 % worse. Critically: the mod-p
   advantage is *preserved and strengthened* (mod-3 T=128 converges
   at step 1500 with v2 vs step 2000 with v1). This is the practical
   demonstration: hybrid v2 is competitive on real-text LM AND wins
   long-T modular arithmetic where DeltaNet+negeig catastrophically
   fails (mod-5 T=512: hybrid 100 % vs DeltaNet+negeig 9 %).
6. Hybrid Triton kernel + custom autograd brings hybrid to 1.74×
   DeltaNet wall-clock at 135M scale (was 14.5× before). 135M
   distillation is feasible; further Triton backward optimisation
   could close the rest.
4. The hybrid finding gives the empirical rank-ordering on parity:
   linear / heisenberg (✗ TC⁰) → DeltaNet default (T=128 only) → ortho
   / hybrid (T=512). And on recall: linear / ortho / rotconj / rotdelta
   (✗ chance) → DeltaNet / hybrid (✓ 100 %).
5. **Triton kernels for sm_120 (Blackwell consumer / RTX 5090) ship
   for** `linear_attn`, `heisenberg_d` (with fused-readout variant),
   `unipotent_u4`, and `ortho_son` (matrix-multiplication scan).
   Three kernel bugs found and fixed; no pytorch#176426 segfault.

## How this work relates to AUSSM (and what's actually new)

We discovered mid-session that
**[AUSSM (Karuvally et al., July 2025, arXiv:2507.05238)][aussm]**
independently proposed the same skew-symmetric input-dependent matrix-
exp construction we built as `OrthogonalScanAttention`. They shipped
it ~3 months before us. Honest accounting of what overlaps and what
doesn't:

|  | AUSSM | This project |
|---|---|---|
| **Cell algebra** (skew → exp → unit-circle eigenvalues) | ✓ | concurrent |
| **State per channel** | scalar / vector (diagonal SSM, S6 drop-in) | **n×n matrix**, non-commuting transitions across t |
| **Parity / mod-p / solvable-group automata** | ✓ shown | ✓ shown |
| **MQAR / associative recall benchmarks** | not evaluated, gap unacknowledged | ✗ ortho fails by construction; addressed via DeltaNet layer in hybrid |
| **Combined with DeltaNet rank-1 erase** | suggested as future work | **hybrid layer stack tested at LM scale** |
| **Mechanistic incompatibility theorem** | not stated | **explicit**: rotation ⨯ delta-rule cannot share a state (Phases 7, 12) |
| **Per-layer specialisation framing** (Krohn-Rhodes / Barrington) | absent | central organising principle |
| **Engineering tricks on rotation cell** | polar diagonalisation | short_conv + silu + L2 v-norm (hybrid v2) |
| **Diagnostic per-task scorecard** | small-scale only, "modest scale" admitted | mod-3 / mod-5 / parity / induction / S₅ / LM-PPL on TinyStories + Python |
| **Triton kernel for sm_120 (Blackwell consumer)** | absent | **ships**, with custom-autograd Triton-backed scan |

**The AUSSM-equivalent piece is `OrthogonalScanAttention`.** That cell
should not be claimed as a from-scratch novelty; we acknowledge AUSSM
as concurrent prior art for the construction.

**The new contributions of this project are everything below the cell
level and everything above it:**

1. The architectural decomposition (rotation specialist + delta-rule
   specialist + their incompatibility argument) and the empirical
   scoreboard demonstrating that the hybrid escapes both Grazzi's TC⁰
   wall (Phase 4-7) and the Zoology recall wall (Phase 8) at the
   network level.
2. The mod-p result: hybrid solves T=512 mod-5 at 100 % where
   `DeltaNet+allow_neg_eigval=True` (the strongest single-cell baseline
   per Grazzi et al. ICLR'25) catastrophically fails to 9 %
   (Phase 9). AUSSM does not test this regime.
3. The 135M-scale practical demonstration: hybrid v2 within 1.19× of
   DeltaNet PPL on TinyStories while preserving the mod-p win
   (Phase 11-12). AUSSM admits its evaluation is small-scale only.
4. The Triton sm_120 kernel for the matmul-scan with custom autograd
   (4.2× speedup over PyTorch loop). AUSSM uses Mamba's selective-scan
   kernel.
5. The Lean library formalising 13 monoids whose associativity gives
   parallel-scan correctness for free (separate from AUSSM's framing
   entirely).

## Approach

### The reframing

Every known state-dependent parallelizable RNN cell corresponds to a
**finite-dimensional associative algebra** `A` over ℝ (or an ordered
semiring), where:

- state ∈ A,
- each step is left-multiplication by an input-dependent element of A,
- composition across steps is A's multiplication table.

The abstract parallel-scan correctness theorem ([`StateDep/Scan.lean`](StateDep/Scan.lean))
says: for any monoid, any binary-tree re-association of a fold equals
the sequential fold. So *the only* algebraic content a kernel needs is
a `Monoid` instance on the parameter type. The design problem reduces
to finding an `A` that is:

1. low-dimensional (memory bandwidth),
2. cheap to multiply (compute),
3. expressively nonlinear in a useful way.

### The wall framing

After eight rounds of architecture iteration, the empirical surface of
this design space looks like:

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

Each axis has a clean published characterisation; the *incompatibility*
across them in a single state is the new observation. The minimum
escape is alternating specialist layers.

## Solutions tested, with results and literature placement

Eight architectures tested at matched ~1 M params, 4 layers, 3-5 k AdamW
steps, on parity (state-tracking) and induction-heads (recall).

### 1. **Heisenberg cross-pair** `c_t = Σ_{i<j} aᵢ ⊗ bⱼ`  *(our original novelty)*

- Lean: [`StateDep/HeisenbergD.lean`](StateDep/HeisenbergD.lean), bilinear ordered-pair sum, proved associative.
- Triton: [`kernels/heisenberg_d/`](kernels/heisenberg_d/) with fused readout — 1.8-4.5× `linear_attn` cost.
- Result: solves T=64 parity (+32.6 pp end-tok over linear-attn), **fails T=128 (TC⁰ stuck)**.
- Literature: HLA (Zhang 2025, [arXiv:2510.27258](https://arxiv.org/abs/2510.27258)) uses *symmetric* `(Σ kᵢkᵢᵀ)(Σ qⱼvⱼᵀ)`; DeltaNet uses single-index `Σ kᵢvᵢᵀ`. The strictly-triangular ordered-pair sum as scan state is novel; the cell as defined is TC⁰-stuck, so it doesn't compete with Grazzi-clean architectures.

### 2. **SO(n) scan (`ortho`)** `O_t = exp(skew(W_skew · x_t))`, `R_t = O_t R_{t-1}`

- PyTorch impl: [`experiments/layers.py:OrthogonalScanAttention`](experiments/layers.py).
- Triton: [`kernels/ortho_son/`](kernels/ortho_son/) — matrix-multiplication parallel scan.
- Result: solves parity at **T=64, 128, 256, 512 all 100 %** (convergence 1000-2000 steps); **fails induction (chance)**.
- Literature: **AUSSM** ([Karuvally et al., July 2025, arXiv:2507.05238][aussm]) independently proposed the same skew-symmetric input-dependent matrix-exp construction; their state is per-channel diagonal, ours is matrix; otherwise concurrent. **DeltaProduct** ([Yang et al. NeurIPS 2025, arXiv:2502.10297][deltaproduct]) achieves O(n) via products of K Householders — equivalent image at K=n. Our distinct angle: matrix-state rather than acting-on-vector, sm_120 Triton kernel.

### 3. **Semidirect `SO(n) ⋉ ℝ^{n×n}` (`rotconj`)** state `(R_t, c_t)`, `c_t = O_t c_{t-1} O_tᵀ + k ⊗ v`

- PyTorch impl: [`experiments/layers.py:RotConjAttention`](experiments/layers.py).
- Result: solves parity at T=64-256 (slower than ortho); **fails induction (chance)**.
- Literature: novel — no published architecture uses semidirect product `G ⋉ V` with `G` a Lie group acting on matrix-valued `V` by conjugation as the parallel-scan monoid. **Diagnosis after running**: additive c-update saturates with noise from distractor positions; rotation conjugation alone does not enable recall.

### 4. **Rotation + delta-rule (`rotdelta`)** `c_t = (I − β k kᵀ)(O_t c_{t-1} O_tᵀ) + β k vᵀ`

- PyTorch impl: [`experiments/layers.py:RotDeltaAttention`](experiments/layers.py).
- Result: solves parity (slower); **fails induction at chance** in two variants (unbounded skew + tanh-bounded skew ≤ 0.5 rad).
- Literature: novel by combination — an agent verified the two-sided action `c → A c B` is distinct from DeltaProduct's left-only `c → A c`. Triple-monoid `(A, B, d) · (A', B', d') = (A'A, BB', A'dB' + d')` is associative, chunkwise-WY-able. **The empirical result IS the contribution**: this combination *cannot work* mechanistically because rotation conjugation rotates stored keys away from their original frame; the inner-product alignment delta-rule recall depends on is broken regardless of rotation magnitude. **Mechanistic incompatibility**.

### 5. **Hybrid layer stack `[ortho, deltanet, ortho, deltanet]`** *(the answer)*

- PyTorch impl: [`experiments/train_hybrid.py`](experiments/train_hybrid.py) + `attention_cls_per_layer` arg in [`experiments/model.py`](experiments/model.py).
- Result: parity at T=128 → **100 %** (step 1200), parity at T=512 → **100 %** (step 1500), induction → **100 %** (step 1200). Both walls escaped.
- Literature: the closest published precedent is **Olmo-Hybrid** ([Merrill, Li et al. 2026, arXiv:2604.03444](https://arxiv.org/abs/2604.03444)) which combines transformer + Gated DeltaNet and ties it to a TC⁰ → NC¹ argument — but their state-tracking layer is *DeltaNet's internal Householder*; nobody else uses an explicit SO(n) / orthogonal / unitary layer alongside DeltaNet. Other hybrid-layer-stack papers (**Qwen3-Next** 75/25 GDN+softmax, **Hymba**, **Samba**, **Jamba**, **Zamba**, **MAD**) frame the layer split via empirical capability decomposition (recall vs context summarisation), not via the two specific walls we identify.

### Where this places us

- The cell-level novelty (`heisenberg_d`, `rotconj`, `rotdelta`) is real — those scan structures are not in the published literature — but the cells we tested don't out-perform DeltaProduct or DeltaNet+`allow_neg_eigval=True` at single-cell expressivity. **Single-cell unification of parity + recall is essentially solved by DeltaProduct** ([Yang et al. 2025][deltaproduct]); we add nothing to that picture beyond extra cells in the same equivalence class.
- The cleanest novel contribution is the **two-walls + incompatibility framing** and the corresponding **minimum hybrid stack**. Olmo-Hybrid argues "transformer + DeltaNet ⇒ TC⁰ ∪ NC¹" but doesn't decompose into rotation + erase as separate primitives, and doesn't state the impossibility direction. Our `[ortho, deltanet]` alternating stack is the smallest concrete witness we know of.

## What's formalised in Lean

Thirteen monoids / groups proved associative in Lean 4 + mathlib, ~1845
lines, no `sorry`s, builds clean from a warm cache in ~2 seconds. Full
table:

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

## Triton kernels

Four cells with PyTorch reference + Triton source + Mac-runnable
correctness tests:

| Kernel | Purpose | Status on sm_120 (RTX 5090) |
|---|---|---|
| [`kernels/linear_attn/`](kernels/linear_attn/) | baseline linear attention | ✓ correctness, ~893 Mtok/s peak |
| [`kernels/heisenberg_d/`](kernels/heisenberg_d/) | bilinear cross-pair, with fused-readout variant | ✓ correctness, 1.8-4.5× linear_attn |
| [`kernels/unipotent_u4/`](kernels/unipotent_u4/) | trilinear ordered triple | ✓ correctness, 5-11× linear_attn |
| [`kernels/ortho_son/`](kernels/ortho_son/) | matrix-multiplication scan (SO(n)) | ✓ correctness on FP32 |

No pytorch#176426 (sm_120 multi-`tl.load`) segfault on any kernel. BF16
in / FP32 accum within `EVAL_PLAN.md` §2.3 tolerances.

Three kernel bugs found and fixed during the port:
1. `linear_attn` BF16-input dtype mismatch in `tl.dot(q, S)`;
2. `heisenberg_d` / `unipotent_u4` writing state outputs in BF16 (overflow at T=2048);
3. `linear_attn` / `unipotent_u4` deriving head-index from `tl.num_programs(1)` on a 1-D grid (heads aliased onto head 0; out-of-bounds writes for `B*H > B`).

## Build

Lean library — requires `elan`, `lake`, network access:
```bash
cd StateDep
source $HOME/.elan/env
lake exe cache get
lake build
```

Python environment (uv + cu132 nightly torch + triton + flash-linear-attention):
```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install numpy flash-linear-attention
```

Smoke + bench + experiments:
```bash
python kernels/smoke_gpu.py                 # all kernels correctness
python kernels/bench_gpu.py                 # throughput grid
python experiments/train.py --T 64 128 256 --steps 5000 --batch 256 \
    --arches linear,heisenberg,deltanet,mamba2,ortho,rotconj,rotdelta
python experiments/train_induction.py --arches linear,ortho,rotconj,rotdelta,deltanet
python experiments/train_hybrid.py --task induction --layers ortho,deltanet,ortho,deltanet
python experiments/train_hybrid.py --task parity    --layers ortho,deltanet,ortho,deltanet --T 512
```

## What's next

The mechanistic-decomposition story leaves four concrete tasks:

1. **Modular counting (mod 3, 5, 7) experiment.** The single sharpest
   prediction of our framing: `Z_p ⊂ SO(2)` for any p (rotation by 2π/p),
   so SO(n)-scan and our hybrid solve any modular counting; DeltaNet
   even with `allow_neg_eigval=True` only reaches `Z_2`. Likely the
   cleanest separator from DeltaProduct as well (its Householder
   eigenvalues are `±1` only).
2. **S₅ word problem.** Tests *non-solvable* state-tracking. SO(n) for
   n≥3 contains A₅ in principle; the question is whether SGD finds the
   non-abelian rotations. DeltaProduct has the cleanest published
   numbers here (NeurIPS 2025); we'd compare directly.
3. **Lean: incompatibility theorem.** Formalise *"two-sided rotation
   conjugation cannot host both Grazzi-clean spectrum and a fixed-frame
   recall basis simultaneously"* in Lean. New module
   `IncompatibilityTheorem.lean`.
4. **SmolLM2-135M distillation** with hybrid layer stack (per
   [`EVAL_PLAN.md`](EVAL_PLAN.md) §3.3). Stretches the result beyond
   synthetics.

[grazzi]: https://arxiv.org/abs/2411.12537
[zoology]: https://arxiv.org/abs/2312.04927
[aussm]: https://arxiv.org/abs/2507.05238
[deltaproduct]: https://arxiv.org/abs/2502.10297
