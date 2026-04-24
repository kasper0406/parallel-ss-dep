# state-dep-parallel — research plan

## Goal

Find algebraic structures that let state-transition RNN cells be
**state-dependent** (the step operator depends on input / prior state)
while remaining **parallelizable via a prefix-sum scan**. Formalise the
associativity / closure properties in Lean, then implement a Triton
kernel for the most promising candidate and measure it against existing
baselines (linear attention, Gated DeltaNet).

## Central reframing (from the initial formalisation pass)

Every existing state-dependent parallelizable RNN cell corresponds to a
**finite-dimensional associative algebra** A over ℝ (or an ordered
semiring), where:

- the state is an element of A,
- each step is left-multiplication by an input-dependent element of A,
- composition across steps is A's multiplication.

The design problem reduces to finding an A that is:

1. low-dimensional (memory bandwidth),
2. cheap to multiply (compute),
3. expressively nonlinear in a useful way.

## What we already proved in Lean (`StateDep/`)

- **`Scan.lean`** — the abstract parallel-scan correctness theorem.
  Any binary tree over a monoid evaluates to the left-fold. This is the
  *only* algebraic content a parallel-scan kernel needs; every specific
  cell inherits parallel correctness the moment we exhibit a `Monoid`
  instance.
- **`Heisenberg.lean`** — scalar Heisenberg group `(a, b, c) ∈ R³`
  with `(a,b,c)*(a',b',c') = (a+a', b+b', c+c' + a·b')`. Full group
  instance; nonabelian; c-coordinate accumulates the causal cross-step
  bilinear `Σ_{i<j} a_i · b_j`.
- **`Tropical.lean`** — `Tropical R` monoid inherited from mathlib, plus
  a concrete `Viterbi` cell `(A, b)` with combine
  `(A₁, b₁) * (A₂, b₂) = (A₁+A₂, min(A₂+b₁, b₂))` proved associative
  directly (no `Tropical` wrapper needed in the kernel).
- **`Jet.lean`** — first-order jets / dual numbers `(val, tan) ∈ R²`
  with Leibniz multiplication. `Monoid` instance.
- **`Affine.lean`** — scalar affine `(a, b) ∈ R²` with
  `(a₁,b₁)*(a₂,b₂) = (a₂·a₁, a₂·b₁ + b₂)`. The backbone of Mamba / S4 /
  RWKV composition.
- **`Delta.lean`** — Sherman-Morrison composition:
  `(I - β₁·k₁k₁ᵀ)(I - β₂·k₂k₂ᵀ) = I - β₁·k₁k₁ᵀ - β₂·k₂k₂ᵀ + β₁β₂·(k₁·k₂)·k₁k₂ᵀ`
  proved via `vecMulVec_mul_vecMulVec` + `abel`. Rank-1 perturbations of
  I are *not* closed under composition; composition adds one rank per
  step. This is the algebraic fact that forces DeltaNet's chunkwise
  form.

## What we learned

1. The hard line in each proof is always the **single bilinear
   cross-term** between consecutive steps. Everything else is `simp`-
   trivial. So expressivity lives in one bilinear form per step,
   wrapped in an associative envelope.
2. "State-dependent parallelizable" ≡ "finite-dim associative algebra".
3. The kernel's combine op is literally the algebra multiplication
   table — the bridge from Lean to Triton is mechanical.

## Next targets (working in parallel)

The three candidates we judged most interesting, to be formalised in
parallel. Each is a separate Lean file under `StateDep/`; none should
touch the root `StateDep.lean` (we'll wire the imports in at the end).

### (1) Multi-dimensional Heisenberg — `StateDep/HeisenbergD.lean`

Lift the scalar Heisenberg group to `a, b : Fin d → R` and
`c : Matrix (Fin d) (Fin d) R`, composition
  `(a₁,b₁,c₁)*(a₂,b₂,c₂) = (a₁+a₂, b₁+b₂, c₁+c₂ + vecMulVec a₁ b₂)`.
This is the natural vector-valued version of Heisenberg. The c-coord
after n steps is `Σ_{i<j} a_i b_jᵀ` — a causal, order-aware outer
product that no existing architecture I know of uses as a primitive.

Deliverables: `Group` (or at least `Monoid`) instance, a sanity-check
example giving the explicit three-fold composition, and the
`Tree.eval_eq_prod` corollary.

### (2) Quaternion / Clifford rotor cell — `StateDep/Rotor.lean`

State is a unit quaternion (Cl(0,2)-even), update is
  `h ← q_t · h`.
Use mathlib's `Quaternion` to get the monoid / group structure for
free, then prove what we need about the sequence action (a scan
accumulates the full rotation as a single quaternion product). Stretch:
parameterise `q_t` as `exp(axis, angle)` and show parallel scan
corresponds to axis-angle composition.

### (3) Higher unipotent U_n — `StateDep/Unipotent.lean`

The group of n×n upper-triangular matrices with 1s on the diagonal.
For n=3 this is Heisenberg. For n≥4 the cross-term degree grows —
U_4 accumulates a trilinear term `Σ_{i<j<k} a_i b_j c_k`. The
"correlation order" knob. Prove closure (the product of two unipotent
matrices is unipotent) and inherit `Monoid` / `Group` structure from
mathlib's `Matrix`. Focus on U_4 concretely, with gestures toward
general n.

## After (3) are done

- Wire imports into `StateDep.lean`, single `lake build` to confirm
  everything still compiles.
- Reflect on what was easy vs. hard in each, and pick **one** to carry
  into a Triton kernel prototype.
- Target: a parallel-scan kernel matching `flash-linear-attention`'s
  layout, benchmarked against Gated DeltaNet on matched parameters.

## Non-goals / deferred

- Numerical-stability bounds in floating-point (interesting but a large
  project in its own right; revisit once the kernel exists).
- Training-loss measurements (requires a full training run; out of
  scope for this formalisation pass).
