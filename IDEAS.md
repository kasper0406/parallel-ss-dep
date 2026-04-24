# IDEAS — Bold candidate algebraic structures for state-dependent parallel RNN cells

Brainstorm companion to `README.md` / `PLAN.md`. Lean's scan theorem only
needs a `Monoid` instance; the game below is to find monoids that are
*low-dim, cheap, and capture an ordered statistic that linear attention
and DeltaNet miss*.

---

## 1. Criteria restated ("wild but useful")

- **Not a renamed linear attention.** If the cross-term between two
  consecutive steps is "outer product, contract later", we've already
  seen it. We want genuinely new cross-terms (ordered-k tuples, parity,
  max-lag, distribution moments, group-theoretic invariants, …).
- **Low-state, cheap-combine.** O(d) to O(d²) memory, O(d²) or better
  combine. Anything asymptotically heavier has to earn it with
  dramatically better per-parameter expressivity.
- **Gradient-friendly.** Combine must be differentiable a.e. with
  non-vanishing gradient through a training run, and admit a sane init
  (identity element reachable by a small perturbation of the params).
  Tropical / Viterbi-style (subgradient only) is acceptable but flagged.
- **Expressivity tied to a task.** We can name a concrete sequence
  statistic (parity, nesting depth, k-gram order, Levenshtein-ish
  distance, long-range variance, …) the cell captures that DeltaNet
  / HLA / Mamba demonstrably cannot with matched state.
- **Lean is the safety net, not the bottleneck.** If the monoid lives
  inside a mathlib-known algebra (matrices, Lie groups, Hopf algebras,
  group algebras, tropical semirings, polynomials, free monoids, …),
  associativity is inherited and the Lean work is a ~50-line file.

---

## 2. Structure catalogue

### 2.1 — Truncated tensor-algebra signature `T^{≤K}(ℝ^d)`

- **Math.** State is an element of the truncated tensor algebra
  `T^{≤K} = ℝ ⊕ ℝ^d ⊕ (ℝ^d)^{⊗2} ⊕ … ⊕ (ℝ^d)^{⊗K}`, multiplied
  tensor-wise with truncation. Each step maps an input `x_t ∈ ℝ^d` to
  `exp^⊗(x_t) = (1, x_t, x_t^{⊗2}/2!, …)` and combines via tensor
  multiplication.
- **State dim & combine cost.** `Σ_{k=0..K} d^k`. Combine is `O(d^{2K})`
  naively but `O(d^K · K)` with the graded structure; for d=16, K=3
  that's ≈ 12k state and ≈ 12k mul — tractable.
- **What the scan accumulates.** The *path signature* of
  `(x_1, …, x_t)` truncated to level K: all iterated integrals
  `∫∫ … ∫ dx_{i_1} … dx_{i_k}` up to order K. Level 2 with a specific
  antisymmetrisation is exactly multi-d Heisenberg.
- **Expressivity / NN-fit.** Universal feature map on paths (Chen,
  Lyons): any continuous path functional is a linear functional of the
  signature. Invariant to reparametrisation — captures *shape*, not
  timing. Level-3 is strictly more than Heisenberg + Unipotent U_4
  because it carries ordered triples in all `d^3` slots, not just the
  `i<j<k` fibre.
- **Known in literature?** Path signatures are standard in rough-path
  ML (Kidger, Lyons, Salvi; CDE nets; log-signature nets;
  <https://arxiv.org/abs/2506.17634>). *Signature as the monoid for a
  parallel scan inside an LM* is — as far as I can find — unused. The
  existing signature-NN work treats the signature as an input feature,
  not as the hidden state of a causal scan.
- **Lean strategy.** Tensor algebra on a finite `Fin d` index set is a
  graded commutative-by-sign... no, *non-commutative* algebra. Mathlib
  has `TensorAlgebra` but working with the truncated quotient is
  awkward; cleaner to define `Sig K d := Π k≤K, (Fin d → … → ℝ)` and
  prove the grade-concatenation multiplication associative by induction
  on K. One non-trivial but very clean proof.
- **Wildness.** 3. Standard in one field, novel as a scan monoid.

### 2.2 — Shuffle / quasi-shuffle Hopf algebra

- **Math.** Same underlying graded vector space as the signature, but
  product is **shuffle**: `u ⧢ v = Σ_{σ ∈ Sh(|u|,|v|)} σ(uv)`. The
  antipode and coproduct give a commutative Hopf algebra.
  Quasi-shuffle (Hoffman) adds a "stuffle" term: interleavings plus
  contractions, closer to discrete time series.
- **State dim & combine cost.** Same dims as signature. Shuffle of two
  grade-≤K words costs `O(d^K · 2^K)` — ugly but OK for small K.
- **What the scan accumulates.** Commutative record of what happened,
  as a sum of all orderings. Loses strict temporal ordering (that's the
  point of commutative).
- **Expressivity / NN-fit.** Shuffle is **commutative** — useless for
  ordered-stat tasks but perfect for *set-attention-style* scans where
  we want order-independent aggregation of a path. Quasi-shuffle
  recovers order via the stuffle correction. A mixed signature ⊗
  shuffle layer would separate the ordered and unordered components of
  the state into explicit eigenspaces.
- **Known in literature?** Classical in combinatorial Hopf algebra
  (Reutenauer, *Free Lie Algebras*; Hoffman MZV work). Unused in NN as
  a scan monoid. Closest ML touch: Toth-Oberhauser kernel work uses
  shuffle identities but not as the recurrence.
- **Lean strategy.** Can be defined on `FreeMonoid (Fin d)` extended by
  multiset formal sums; associativity of shuffle is a classical but
  fiddly induction. Mathlib lacks it; ~200-line proof.
- **Wildness.** 4.

### 2.3 — Truncated free-nilpotent Lie group via BCH

- **Math.** Free Lie algebra `L(ℝ^d)` truncated at bracket-depth K, Lie
  product is the truncated BCH series
  `log(exp(X)·exp(Y)) = X + Y + ½[X,Y] + 1/12 ([X,[X,Y]] + [Y,[Y,X]]) + …`.
  Elements compose via `X * Y := BCH_K(X, Y)`. This is exactly the
  *universal* nilpotent structure of step K; every other K-nilpotent
  cell is a quotient.
- **State dim & combine cost.** Free nilpotent of depth K on d
  generators has dimension `Σ_{k=1..K} (1/k) Σ_{d|k} μ(d) n^{k/d}`
  (Witt formula). For d=16, K=3 this is 16+120+680 ≈ 800 scalars.
  Combine is polynomial of degree K in the state — O(state²) or better.
- **What the scan accumulates.** All Lie-bracket-invariant ordered
  statistics of the input path up to order K. Crucially *closed under
  composition*: no truncation error beyond depth K.
- **Expressivity / NN-fit.** This is the thing Heisenberg and U_4 are
  pale shadows of. Captures k-fold iterated commutators — bits of the
  input sequence invisible to any commutative statistic and missed by
  HLA entirely. Smooth gradients because the group is a connected Lie
  group.
- **Known in literature?** Free nilpotent groups are textbook (Reutenauer,
  Bourbaki Lie III). Truncated BCH is used numerically in geometric
  integrators (Munthe-Kaas, Iserles) and rough-path numerics. **Not** a
  scan monoid in any ML paper I can find — the closest is Lie-group RNN
  work (Huang et al. LieNet 2017) but that composes in ambient SO(3),
  not in a free-nilpotent envelope.
- **Lean strategy.** Define the free Lie algebra on `Fin d`, mod out by
  brackets of depth > K, prove Jacobi gives BCH up to K. Mathlib has
  `FreeLieAlgebra` and `LowerCentralSeries`; the BCH closed form at
  depth 3 is a finite computation. Probably a 300-line file at K=3.
  Significant but tractable.
- **Wildness.** 4.

### 2.4 — Polynomial-ring scan with FFT combine

- **Math.** State is a polynomial `p_t(x) ∈ ℝ[x]/(x^K)`. Each token
  emits a polynomial `q_t(x)` (e.g. `1 + α_t x + β_t x^2 + …`) and
  combine is multiplication mod `x^K`.
- **State dim & combine cost.** K scalars. Combine is `O(K log K)` via
  FFT, `O(K²)` naively, `O(K)` if the per-step polynomial is sparse
  (degree ≤ 2).
- **What the scan accumulates.** The generating function of the emitted
  coefficients: coefficient of `x^k` in the product is
  `Σ_{k_1 + … + k_t = k} q_1[k_1] · q_2[k_2] · … · q_t[k_t]` — a
  *convolution over time*. With `q_t = 1 + α_t x`, coefficient of `x^k`
  is `e_k(α_1, …, α_t)` (elementary symmetric polynomial).
- **Expressivity / NN-fit.** Elementary symmetric polynomials are a
  classical test of symmetric-function learnability. Capturing `e_k`
  over a 1M token window with K ≤ 32 state is dramatically more
  memory-efficient than any existing cell. Useful init: `q_t(x) = 1`
  means "skip". Differentiable everywhere.
- **Known in literature?** Polynomial scan is implicit in the
  generating-function-for-counting literature (e.g. GF-as-state for CFL
  parsers, Valiant's algebraic-path framework). FFT-based sequence
  convolution is standard (FlashFFTConv, Hyena), but those use a *fixed*
  kernel across time; here the kernel is state-dependent. Not an exact
  match anywhere I can find.
- **Lean strategy.** Trivial. Mathlib has `Polynomial ℝ` as a
  commutative ring; modular quotient by `X^K` is
  `Polynomial.modByMonic (X^K)` or cleaner as a `Fin K → ℝ` with
  truncated multiplication. ~80-line file.
- **Wildness.** 3 (simple to write down, but the state-dependence is
  new).

### 2.5 — Generating-function over a non-commutative variable ("shuffled GF")

- **Math.** Same as 2.4 but over `ℝ⟨x_1, …, x_m⟩ / I_K`, a truncated
  free associative algebra. Each step emits a non-commutative
  polynomial; combine is truncated multiplication.
- **State dim & combine cost.** `Σ m^k` for k ≤ K. With m=2, K=6: 127.
  Combine is `O(m^K)` naively, faster with sparsity.
- **What the scan accumulates.** A truncated generating function over
  *words* in the alphabet — coefficient of `x_{i_1} x_{i_2} … x_{i_k}`
  is a signed sum over ordered k-tuples of positions, each tagged by
  the letters the tokens chose.
- **Expressivity / NN-fit.** Captures counts-of-bigrams, trigrams,
  k-grams but *position-aware* — distinct from a bag-of-n-grams. Useful
  for parity (one x_1, x_2 per pair gives Σ(−1)^{words}), Dyck-language
  nesting (commutator in x_[, x_] tracks imbalance), and
  cross-channel-ordered statistics.
- **Known in literature?** Free associative algebras are the natural
  home of non-commutative symbolic dynamics (Reutenauer). Unused in NN
  scans.
- **Lean strategy.** `FreeMonoid (Fin m)` as the word monoid; state is
  `FreeMonoid (Fin m) →₀ ℝ` truncated by word length. Associativity
  inherited from `MonoidAlgebra`. ~120 lines.
- **Wildness.** 4.

### 2.6 — Count-min / count-sketch as an associative monoid

- **Math.** State is a `w × d` matrix of counters. Each token maps
  `key_t, val_t → M_t` where `M_t[i, h_i(key_t)] = val_t`, and combine
  is **matrix addition** (it's a linear sketch).
- **State dim & combine cost.** `w d` scalars (w ≈ 4, d ≈ 256). Combine
  is O(wd) add. Trivially parallel.
- **What the scan accumulates.** A sketched histogram over
  (hashed-)keys: `read(k) = min_i sketch[i, h_i(k)]` gives `Σ_{t'≤t} [
  key_{t'}=k] v_{t'} ± ε·Σ|v|` w.h.p. Same associativity as linear
  attention but the *decoding* is non-linear (the min).
- **Expressivity / NN-fit.** Concentration gives a provable associative
  recall, unlike linear attention's noisy lookup. Useful for exact-ish
  key-value recall in small state. Gradient through `min` is
  subgradient but stable. Init = zero counter.
- **Known in literature?** Classical sketches (Cormode-Muthukrishnan
  2005) are associative but the neural-scan framing is not standard.
  "Scatterbrain / Nyström" approximate attention lines up partly.
  Neural sketches (Mitzenmacher) treat the sketch as a learned filter,
  not as the state of a scan.
- **Lean strategy.** State = matrix, combine = addition → `AddCommMonoid`
  is inherited from mathlib's `Matrix`. The interesting theorem is the
  *accuracy guarantee*, not associativity (which is `rfl`). ~30 lines
  for monoid; a concentration bound is a real measure-theoretic proof.
- **Wildness.** 2 for the algebra, 4 for the "this is a scan primitive"
  reframing.

### 2.7 — Möbius / fractional-linear (PGL_2) recurrence

- **Math.** State is `[[a,b],[c,d]] ∈ GL_2(ℝ)` acting by Möbius
  transformation `z ↦ (az+b)/(cz+d)`; combine is matrix multiplication.
- **State dim & combine cost.** 4 scalars per head (or 3 if normalized).
  Combine is O(1) per head — a `2 × 2 × 2 × 2` mul.
- **What the scan accumulates.** A nested Möbius map of the initial
  scalar — equivalent to continued-fraction composition; equivalent to
  the Kalman-filter precision update in 1D.
- **Expressivity / NN-fit.** Möbius maps can implement gating, damping,
  and hyperbolic translation with 4 params. Per-head cost is *cheaper*
  than Mamba's affine cell but has a richer group (PGL_2 not ℝ ⋊ ℝ).
  Multi-head parallel Möbius scans with inter-head mixing look like a
  clean alternative SSM.
- **Known in literature?** Kalman Linear Attention (KLA, 2602.10743)
  uses exactly this monoid for 1D scalar filtering. Multi-head, richer,
  or hyperbolic-geometry interpretations are less explored.
- **Lean strategy.** `Matrix (Fin 2) (Fin 2) ℝ` with matrix product is
  a monoid out of the box. Closure under GL_2 (det ≠ 0) is a sub-monoid
  proof — trivial. 40 lines.
- **Wildness.** 2.

### 2.8 — Lorentz / hyperbolic boost group SO(1,1) × translations

- **Math.** State in ℝ^{1,1} isometries: `(β, v)` with `β ∈ (−1,1)`
  rapidity and `v ∈ ℝ²` translation. Combine is relativistic velocity
  addition for β and boosted translation for v.
- **State dim & combine cost.** 3 scalars. Combine O(1).
- **What the scan accumulates.** Hyperbolic-geometry nested boost —
  elegant bounded-state analogue of affine scan where `|β|<1` is
  enforced algebraically (not by a sigmoid). Self-normalising dynamics.
- **Expressivity / NN-fit.** Provides a built-in, scale-free gate.
  Training stability from the bounded rapidity. Suits tasks where
  "time-dilation"-ish effects are natural (irregular sampling, tempo).
- **Known in literature?** Hyperbolic neural nets (Nickel-Kiela) use
  hyperbolic *embeddings*, not Lorentz as a scan monoid. Lorentz
  equivariant NNs exist (LorentzNet) but for static inputs. Scan-monoid
  framing seems new.
- **Lean strategy.** SO(1,1) is isomorphic to ℝ under rapidity addition.
  Combined with ℝ² translations we get a 3-param solvable Lie group.
  Easy: 60 lines on top of `Real`.
- **Wildness.** 3.

### 2.9 — p-adic digit-wise semigroup with carry

- **Math.** State is a p-adic integer truncated to K digits,
  `s ∈ ℤ/p^K`. Combine is p-adic addition *with carry* (= ℤ/p^K
  addition) — or more interestingly, p-adic multiplication. Each token
  contributes a digit pattern.
- **State dim & combine cost.** K scalars (digits, not floats). Combine
  is O(K) with carry chain, or O(K log K) via fancy tricks.
- **What the scan accumulates.** A scale-aware hierarchical
  fingerprint: low digits track short-range structure, high digits
  track long-range. Carry couples the scales.
- **Expressivity / NN-fit.** Natural for hierarchical sequences (code,
  XML, music). Carry gives bounded-degree cross-term between digit
  positions — a "tree-shaped" non-locality that neither RWKV nor Mamba
  captures. Differentiable straight-through via learned soft-digit
  encodings.
- **Known in literature?** p-adic cellular nets (Zúñiga-Galindo) but
  non-sequential. No prior scan framing.
- **Lean strategy.** `ZMod (p^K)` is a commutative ring in mathlib —
  monoid under + or ×, inherited. ~20 lines.
- **Wildness.** 4. Totally unusual in ML.

### 2.10 — Braid group B_n normal form as a monoid

- **Math.** State is an element of B_n in left-Garside normal form
  `Δ^k · s_{i_1} · … · s_{i_m}` where Δ is the half-twist. Combine is
  braid multiplication followed by renormalisation. Each token picks a
  generator `σ_i`.
- **State dim & combine cost.** O(n) for the Garside word; combine is
  O(n log n) using Dehornoy's handle reduction, or O(n²) with a simple
  normal-form rewriter. For n=8 that's 64 per combine.
- **What the scan accumulates.** A topologically non-trivial record of
  crossings. Maps onto the symmetric group S_n (a *coarse* statistic)
  and onto the Burau/LKB representations (finer statistics usable as
  real-valued outputs).
- **Expressivity / NN-fit.** Non-commutative, rich representation
  theory. Naturally encodes "which strand ended up where" — useful for
  pointer-network / permutation-learning tasks. The Burau rep at a
  complex parameter gives smooth scalar readouts.
- **Known in literature?** BraidNet (2104.10010) and braid-arrangement
  NN (2502.09324) use braids for *static* architectures. Braid group
  as an RNN scan is new.
- **Lean strategy.** Braid groups via Artin presentation exist in
  mathlib (`BraidGroup`). Normal-form computation is not in mathlib.
  Probably 400+ lines. For a first pass, skip normal form and work
  abstractly via the presentation monoid.
- **Wildness.** 5.

### 2.11 — Symmetric-group algebra ℝ[S_n] with learned permutation emit

- **Math.** State is in ℝ[S_n] (n! dim in the worst case; cap n ≈ 5-6
  for n! ≤ 720), combine is group-algebra convolution. Each token
  emits a convex combination over n! transpositions or a sparse
  element.
- **State dim & combine cost.** n! scalars; combine O((n!)²) —
  expensive fast. Cheaper version: state is an element of ℝ[S_n]
  supported on the adjacent-transpositions, costs O(n²).
- **What the scan accumulates.** A *distribution over permutations* of
  the input positions — a full soft-sorting state. Readout: sample /
  expectation of the permutation applied to values.
- **Expressivity / NN-fit.** Directly encodes "soft pointer into a
  window" and "soft sort". Useful for algorithmic tasks
  (array-reversal, dynamic programming). Gradient via soft ops is
  clean.
- **Known in literature?** Sinkhorn-style differentiable sorts (Mena
  2018) target the Birkhoff polytope; ours targets S_n algebraically.
  Close but not identical.
- **Lean strategy.** `MonoidAlgebra ℝ (Equiv.Perm (Fin n))` gives the
  whole thing in mathlib. ~40 lines.
- **Wildness.** 3.

### 2.12 — Tropical × linear (max-plus Heisenberg)

- **Math.** Tensor product of `Tropical ℝ` with the standard Heisenberg
  group: state `(a, b, c) ∈ T^3` where `T = (ℝ ∪ {−∞}, max, +)` and
  combine is `(a₁, b₁, c₁) * (a₂, b₂, c₂) = (max(a₁,a₂), max(b₁,b₂),
  max(c₁, c₂, a₁ + b₂))`. (Note `a_1 + b_2` is the tropical analogue
  of `a_1 · b_2`.)
- **State dim & combine cost.** 3 scalars, O(1) combine.
- **What the scan accumulates.** `c` ends up as `max_{i ≤ j} (a_i +
  b_j)` — the *best ordered pair* statistic. Combined with the
  max-plus semiring this is like Viterbi for an ordered pair of events.
- **Expressivity / NN-fit.** Tracks "best matching pair" over an
  unbounded window with O(1) state. Can encode trigger-response
  patterns, best-bid-ask timing, latest-peak-after-an-earlier-peak.
  Subgradients only.
- **Known in literature?** Tropical Heisenberg appears in tropical
  geometry (Mikhalkin) but not as an NN primitive. Tropical NNs exist
  (Zhang 2018) but don't use the ordered-pair structure.
- **Lean strategy.** Define `TropHeisenberg` with the combine above;
  prove associativity by case analysis on the max arguments. ~120 lines
  — the cross-term `max(c, a+b)` distributes over associativity after
  some casework.
- **Wildness.** 4.

### 2.13 — "Edit distance" monoid via pairwise alignment composition

- **Math.** State is a small Levenshtein automaton: a DP tableau tail
  of size `O(k)` (last k diagonals) for edit-distance-at-most-k. Each
  token advances the tableau. Combine of two tableaux is associative
  if we accept a `O(k²)` combine (concatenation of partial alignments).
- **State dim & combine cost.** O(k) to O(k²); combine O(k²).
- **What the scan accumulates.** Online edit distance from a reference
  pattern — equivalently, a tropical weighted automaton's run.
- **Expressivity / NN-fit.** Pattern-match state for sequence
  classification, fuzzy-key lookup, DNA-style tasks. Tropical but with
  richer cross-term than flat Viterbi.
- **Known in literature?** Weighted automata in min-plus are well
  studied (Mohri); "finite-tailed alignment as an associative combine"
  I don't see.
- **Lean strategy.** Realize as a tropical matrix semigroup on a
  small explicit state graph. Reuses `Tropical.lean` + matrix mul.
  ~100 lines.
- **Wildness.** 3.

### 2.14 — Group algebra of a cyclic group ℝ[ℤ/n] — pitched as "phase memory"

- **Math.** State in `ℝ[ℤ/n] ≅ ℝ^n` under circular convolution. Each
  token emits a sparse element (e.g. a Dirac at a learned phase).
  Combine is length-n cyclic convolution — FFT diagonalises it.
- **State dim & combine cost.** n scalars; O(n log n) combine, O(n) in
  Fourier basis (pointwise multiply of spectra).
- **What the scan accumulates.** A "phase histogram" of the stream —
  peak at phase `φ` if many tokens emitted that phase. Nonlinear
  readout (argmax, norm) yields clock / period extraction.
- **Expressivity / NN-fit.** Native periodicity detector. Useful for
  music, code indentation, any cyclic structure. Clean init: 1 at
  phase 0.
- **Known in literature?** Holographic Reduced Representations (Plate
  1995), vector-symbolic arch (Kanerva, Eliasmith) use circular
  convolution as binding. VSA as a causal scan monoid, with learned
  emits, is a narrow but existing niche (Kleyko surveys).
- **Lean strategy.** `MonoidAlgebra ℝ (ZMod n)` gives a commutative
  ring for free. 20 lines.
- **Wildness.** 2.

### 2.15 — Dihedral group algebra ℝ[D_n] ("phase + flip memory")

- **Math.** State in ℝ[D_n] = n rotations × 2 reflections; combine is
  group-algebra convolution. Non-abelian for n ≥ 3.
- **State dim & combine cost.** 2n scalars; combine O((2n)²). At n=16:
  32 scalars, 1024 mul — fine.
- **What the scan accumulates.** Orientation-and-phase memory.
  Non-abelian, so the order of emits matters.
- **Expressivity / NN-fit.** Cheap non-commutative scan. Exactly as
  rich as the regular rep of D_n — plus an extra reflection bit
  compared to ℝ[ℤ/n].
- **Known in literature?** Dihedral convolutions in equivariant NNs.
  As an RNN scan monoid, unused.
- **Lean strategy.** `MonoidAlgebra ℝ (DihedralGroup n)` — mathlib has
  `DihedralGroup`. 40 lines.
- **Wildness.** 3.

### 2.16 — Non-abelian 2-step nilpotent N(d, 2) over ℤ/p (digit Heisenberg)

- **Math.** Heisenberg group over the ring `ℤ/p` rather than ℝ.
  `(a, b, c) ∈ (ℤ/p)^{2d+d²}` with the usual Heisenberg product mod p.
- **State dim & combine cost.** Same as multi-d Heisenberg but with
  log p bits per scalar instead of 32. For p=256, state fits in bytes.
- **What the scan accumulates.** Same ordered-pair sum as Heisenberg,
  modulo p. Surprisingly, modular reductions of bilinear forms are
  empirically informative (used in cryptographic hashing and in AKS
  primality).
- **Expressivity / NN-fit.** Extreme memory compression. Straight-
  through gradient with learned soft-mod. Suited for long-context
  tasks where exact bits per step are scarce.
- **Known in literature?** Modular hashing is classical; modular
  Heisenberg as an NN primitive — nothing.
- **Lean strategy.** Same proof as `HeisenbergD` but over `ZMod p`.
  Already essentially inherited from mathlib's ring machinery. 50 lines.
- **Wildness.** 4.

### 2.17 — Monoid of stochastic matrices on k states (discrete belief scan)

- **Math.** State is a `k × k` column-stochastic matrix `P_t` (the
  current belief-transition operator). Combine is matrix multiplication.
  Each token emits a perturbation to a base matrix (e.g. `exp(ε · A_t)`).
- **State dim & combine cost.** k² scalars; combine O(k³). k = 8 is
  cheap (512 mul).
- **What the scan accumulates.** A small HMM transition operator.
  Left-multiplying a belief vector gives the current posterior.
- **Expressivity / NN-fit.** Explicit probabilistic interpretation —
  perfect for tasks with a latent state of small cardinality (POS
  tags, DNA codons, phonemes). Softmax-parametrised emits keep it
  stochastic. Good init: all P_t = I.
- **Known in literature?** Structured HMM + NN hybrids (Tran 2016,
  neural HMMs) exist; but the parallel-scan-monoid framing is cleaner
  and new.
- **Lean strategy.** Stochastic matrices are closed under product —
  routine in mathlib (`Matrix.mul` + `Finset.sum_stoch`). 80 lines.
- **Wildness.** 2.

### 2.18 — "Nimber / XOR" monoid (GF(2)^k under XOR + a bilinear bit)

- **Math.** State `(a, c) ∈ GF(2)^k × GF(2)^{k(k-1)/2}` with
  `(a_1,c_1) * (a_2,c_2) = (a_1 ⊕ a_2, c_1 ⊕ c_2 ⊕ (a_1 ∧ a_2^T))`
  upper-triangular, i.e. the GF(2) Heisenberg group.
- **State dim & combine cost.** k + k²/2 bits; combine is O(k²) bit
  ops — tiny.
- **What the scan accumulates.** Parity of XOR and parity of ordered
  bit-pairs — think "parity of number of 01 patterns". Exactly the
  statistics needed for tasks like parity / Dyck-of-depth-2 / modular
  counting, which Transformers and Mamba famously fail on.
- **Expressivity / NN-fit.** Provably captures a regular language
  hierarchy below AC0 that standard attention cannot. Relaxed to ℝ via
  soft-tanh gives a trainable version.
- **Known in literature?** Nimbers (`Nim.lean` in mathlib) define a
  different (Conway) product; GF(2)-Heisenberg is the "ε-nilpotent"
  (non-Conway) variant. Regularity-hierarchy analysis of Transformers
  (Merrill, Hahn) proves attention can't do parity — a cell that does
  parity *by construction* is valuable.
- **Lean strategy.** `ZMod 2` scalars, rest identical to `HeisenbergD`.
  60 lines.
- **Wildness.** 4.

### 2.19 — Learned monoid via a learnable Cayley table with associativity
    projection

- **Math.** Fix a small finite set `S` (|S| = n, say 16). Parametrize a
  combine table `T : S × S → S` as a softmax over `S` for each `(a,b)`.
  Add a soft associativity penalty
  `Σ_{a,b,c} ‖T(T(a,b),c) − T(a,T(b,c))‖²`. Inside inference / Lean
  proofs, use the *projected* table that is associative by construction
  — e.g. via a learned embedding into a known associative algebra and
  a discretisation step.
- **State dim & combine cost.** log₂ n bits per step, O(n²) table.
  Combine is a table lookup.
- **What the scan accumulates.** Whatever the learned monoid happens
  to represent — could interpolate between Z/n, S_n, and more exotic
  ones depending on training signal.
- **Expressivity / NN-fit.** The ultimate "let the data pick the
  algebra" move. Cleanest safe version: parametrise by a real Lie
  algebra element `L(θ)` and define combine via `exp(L(θ))` — guarantees
  associativity without a penalty.
- **Known in literature?** Discovering group structure (Grover, Power)
  as a modular-arithmetic grokking phenomenon; learning group ops from
  data (Pearce-Crump, Liao 2024). Not framed as an RNN scan monoid.
- **Lean strategy.** Abstract: assume an associative table, derive
  scan correctness (already done). Concrete: prove that the
  exp-of-learned-Lie-element version is associative (mathlib has
  `LieAlgebra.exp`). 100 lines.
- **Wildness.** 5.

### 2.20 — Hopf-algebra-enriched Heisenberg (ordered moments + antipode)

- **Math.** Take the quasi-shuffle Hopf algebra on ℝ^d and truncate at
  K=2 with an explicit antipode that allows negating ordered pairs.
  State is `(a, b, c, c̄) ∈ ℝ^{2d + 2d²}` where `c = Σ_{i<j} a_i b_j`
  and `c̄ = Σ_{i<j} (−a_i)(−b_j) = c` (trivial in grade 2). More
  interesting at grade 3: antipode relates `Σ_{i<j<k}` to `−Σ_{k<j<i}`,
  i.e. reversed-time statistics.
- **State dim & combine cost.** 2x multi-d Heisenberg. O(d²) combine.
- **What the scan accumulates.** Forward + time-reversed bilinear /
  trilinear statistics simultaneously. Think "what pairs happened
  before vs after this event" as native state.
- **Expressivity / NN-fit.** Captures both future-prediction and
  past-explanation statistics in one scan. For causal LMs the
  time-reversed half is a readout rather than a predictor.
- **Known in literature?** Hopf algebras in rough paths
  (Connes-Kreimer, Hairer regularity structures). Not seen in ML.
- **Lean strategy.** Extension of `HeisenbergD.lean` with a second
  copy; antipode properties are just sign flips in grade 2. 100 lines.
- **Wildness.** 4.

### 2.21 — Locally-finite Inverse-semigroup scan (partial-function composition)

- **Math.** An inverse semigroup where each element is a partial
  injection on a finite set `X` (|X|=k). Composition is "defined where
  both are defined". Unlike groups, tokens can *un-define* the state
  (useful: end-of-scope, token dropout, MoE routing).
- **State dim & combine cost.** O(k) to describe a partial injection;
  combine O(k).
- **What the scan accumulates.** A partial correspondence from *some*
  previous positions to *some* current ones — in effect, a learned
  linking structure (parser stack, bracket matcher, alias table).
- **Expressivity / NN-fit.** Captures parenthesisation / scoping
  primitives that regular groups can't (because groups can't forget).
  Straight-through softmax gives a relaxed inverse semigroup.
- **Known in literature?** Inverse semigroups in formal-language
  theory (tile semigroups). Nothing in NN scans.
- **Lean strategy.** Inverse semigroups are in mathlib
  (`InverseSemigroup` missing but `Semigroup` + partial maps work).
  Medium effort, 150 lines.
- **Wildness.** 5.

### 2.22 — Dyck-word / parenthesis-balance monoid

- **Math.** State is a pair `(l, r) ∈ ℕ × ℕ` of "unmatched ( on left"
  and "unmatched ) on right" of a substring, with combine
  `(l_1,r_1) * (l_2,r_2) = (l_1 + max(l_2 − r_1, 0), r_2 + max(r_1 −
  l_2, 0))`. Associative by a classical (and short) check. Relaxed to
  ℝ² for training.
- **State dim & combine cost.** 2 scalars. O(1) combine.
- **What the scan accumulates.** The Dyck-reduced substring balance.
  Joined via a tropical-like "cancel as much as possible" rule.
- **Expressivity / NN-fit.** Provably tracks matched-parenthesis
  depth *and* supports divide-and-conquer parse — a hard task for
  vanilla Mamba / attention. Soft-relaxation to ℝ² has well-defined
  gradients.
- **Known in literature?** Classical in concurrent-computation folklore
  (bracket monoid / free group reduction). Not used as an RNN scan in
  ML that I can find.
- **Lean strategy.** Define the state and combine, prove associativity
  by case analysis on sign of `l_2 − r_1`. ~60 lines, no mathlib
  hooks.
- **Wildness.** 4. Simple but genuinely new primitive for NNs.

### 2.23 — Reservoir-echo-state monoid (random orthogonal SSM with fixed spectrum)

- **Math.** Fix a frozen orthogonal matrix `U` with uniform spectrum on
  the unit circle (random reservoir). State is a vector; combine is
  `(U^{τ_1}, v_1) · (U^{τ_2}, v_2) = (U^{τ_1 + τ_2}, U^{τ_2} v_1 +
  v_2)`. I.e., the affine semigroup over the (abelian) subgroup
  `{U^t}`.
- **State dim & combine cost.** d + 1 scalars (vector state + scalar
  "phase advance" τ). Combine O(d²) with matrix-vector, or O(d log d)
  if U is diagonalised / circulant.
- **What the scan accumulates.** A frozen-random linear filter bank
  with learnable inputs — dynamical-systems tricks of echo-state
  networks, turned into a monoid.
- **Expressivity / NN-fit.** "Reservoir computing as scan primitive".
  Inherits ESN's long-memory guarantees (edge of chaos) for free; only
  the linear read-in/out is learned.
- **Known in literature?** LRU (Orvieto 2023) and HiPPO / S4 are
  *close*: SSMs with spectral constraints. Explicit "frozen random
  orthogonal U as a subgroup scan" is less explicit; frames ESN
  cleanly.
- **Lean strategy.** Subgroup of SO(d) generated by U is abelian
  (≅ ℤ), so we just need (ℤ × ℝ^d) with a semi-direct product structure.
  Lean-easy, 80 lines.
- **Wildness.** 2.

### 2.24 — Quaternion ⊗ Tropical (rotation-max)

- **Math.** State `(q, v) ∈ H^× × (T^d)` where q is a unit quaternion
  and v tracks per-feature tropical max. Combine multiplies quaternions
  and applies the rotation to v before max-combining.
- **State dim & combine cost.** 4 + d scalars; O(d) combine.
- **What the scan accumulates.** Max-aggregated features in a rotated
  (time-aware-orientation) frame. The orientation evolves continuously
  while the feature side picks peaks.
- **Expressivity / NN-fit.** Combines two primitives we already proved.
  Subgradient on the max side, smooth on the quaternion side.
- **Known in literature?** Novel as far as I can tell — tensor products
  of known scan monoids don't appear in the NN literature.
- **Lean strategy.** `ProductMonoid` of `Rotor` and `Tropical^d`; the
  non-trivial content is the equivariant action of q on v, for which
  we just postcompose the outputs. 80 lines.
- **Wildness.** 4.

### 2.25 — Semi-numerical "Lie-exponential of learnable matrix" monoid
    (truncated matrix exp)

- **Math.** Each step emits a small matrix `A_t ∈ ℝ^{d×d}`; state is
  `exp(Σ_{i<t} A_i + corrections)` under the exact BCH at finite order.
  Practically: represent state as `(A, B) ∈ ℝ^{d×d} × ℝ^{d×d}` where A
  is the "linear" term and B is the "nested-commutator" correction,
  combined by BCH truncated at level 2:
  `(A_1, B_1) * (A_2, B_2) = (A_1 + A_2, B_1 + B_2 + [A_1, A_2]/2)`.
- **State dim & combine cost.** 2 d² scalars. Combine is O(d²) +
  one matrix commutator = O(d^{ω}).
- **What the scan accumulates.** A 2nd-order-accurate matrix
  exponential of the sum of emitted Lie-algebra elements — a
  *rotation-aware* generalisation of affine SSM.
- **Expressivity / NN-fit.** Exactly as expressive as a non-abelian SSM
  with commutator corrections. Fits naturally on top of Mamba-style
  cells to model non-commutative gating.
- **Known in literature?** Magnus / Munthe-Kaas integrators in
  numerical ODE — a clean fit, but not deployed in NN scans yet.
- **Lean strategy.** The truncated BCH at level 2 is a short explicit
  formula; associativity is a computation in the free Lie algebra mod
  depth 2. ~150 lines. Strictly a special case of 2.3 but much more
  lightweight.
- **Wildness.** 3.

---

## 3. Top-3 picks for Lean next

### Pick A — Truncated tensor-algebra signature (§2.1)

The deepest payoff: signatures are the universal ordered-path feature
map, and the existing Heisenberg / Unipotent cells are the grade-2 /
triangular-cone slivers of this construction. A K=3 implementation on
d=8 fits in ~600 state slots (8+64+512) — a parameter budget
Gated-DeltaNet is already using. Lean cost is moderate (clean graded
associativity, ~200 lines using mathlib `TensorProduct`). The scientific
payoff is large: once we have it, we can *strictly compare* the
expressivity of Heisenberg / Unipotent against a known universal object
and prove they are its projections.

### Pick B — Dyck / parenthesis-balance monoid (§2.22)

Small (2 scalars), O(1) combine, no mathlib dependency, and solves a
named problem (bracket matching / scoping) that is empirically hard for
linear attention. Lean proof is a short case-analysis; could be written
in an afternoon. High "wildness-per-Lean-line": it will force a
conversation about what non-linear associative cells look like, since
the combine law uses `max` essentially. Great teaching example of
"non-linear but still associative".

### Pick C — Truncated BCH / free-nilpotent group (§2.3 at K=3, or §2.25 at K=2)

The principled generalisation of our Heisenberg / Unipotent / Rotor
line. Directly subsumes all three (and subsumes multi-d Heisenberg,
which is K=2). For a first Lean pass, the K=2 Magnus-style
`(A, B) * (A', B') = (A+A', B+B' + [A,A']/2)` is a few pages of proof
and gives us a usable kernel candidate immediately. K=3 is a separate
follow-up. The novelty-to-difficulty ratio at K=2 is excellent and the
resulting structure is genuinely under-explored in NN practice.

---

## 4. Open research questions

- **Minimal ordered-k-statistic monoid.** What is the least-dimensional
  associative structure on state whose scan exactly recovers
  `Σ_{i_1 < … < i_k} f_k(x_{i_1}, …, x_{i_k})` for a given
  multilinear `f_k`? Unipotent U_{k+1} achieves dim `O(k²)`; can we
  beat it for specific `f_k` (e.g. rank-1 tensor)?

- **Lower bounds via regular-language expressivity.** For each monoid
  in the catalogue, what regular (or non-regular) language can its
  scan decide when wrapped in a linear readout + threshold? Parity is
  a known Mamba-failure; §2.18 does it trivially — can we give an
  exact expressivity classification?

- **When is soft relaxation still associative enough?** §2.6, §2.8,
  §2.12, §2.22 use max/min subgradients. The relaxed combine is not
  exactly associative under smoothed-min. Is there a formal
  "ε-associative" category where the scan theorem still holds modulo
  a controlled error that vanishes with training-time anneal?

- **Commutator budget.** In BCH-style cells (§2.3, §2.25), how does
  validation loss scale with truncation depth K, for fixed state
  size? Is there a "commutator law" — the next factor of 2 in perplexity
  reduction costs doubling K, or does it plateau at K=3?

- **Learned monoid convergence.** Does §2.19 train to a known algebra,
  or does it stabilise in the "almost-associative" interior of the
  polytope of combine tables? The grokking-group-structure literature
  says the former for toy tasks; we'd want the empirical picture on
  genuine language modelling.

- **Free objects as ceilings.** The truncated tensor algebra §2.1 is
  the free associative algebra mod degree K on d generators. Is there
  a theorem of the form: every O(d²)-combine, polynomial-state monoid
  on sequences of ℝ^d embeds as a quotient of `T^{≤K}(ℝ^d)` for K =
  O(1)?

- **Hopf-algebraic "antipode" as train-test asymmetry.** §2.20's
  antipode gives time-reversed readouts for free. Could we design a
  *causal-only* training objective whose gradient naturally flows
  through the antipode in backward mode? — i.e. the adjoint of the
  scan is literally the antipode.

- **Lean-first discovery.** Is there a search procedure over small
  Lean tactic-provable monoid structures (e.g., enumerate
  4-dimensional algebras with a short `ring` associativity proof)
  that would turn up non-obvious candidates we missed? An adjacent
  line: use mathlib's classification of finite-dimensional algebras
  to machine-generate candidate cells and filter by combine cost.

---

Sources consulted while brainstorming (representative, not exhaustive):

- [Path Signatures & Rough Paths — primer](https://arxiv.org/pdf/1603.03788)
- [Path Signatures in ML (2025)](https://arxiv.org/abs/2506.17634)
- [BCH & free Lie algebras (MIT 18.745 notes)](https://ocw.mit.edu/courses/18-745-lie-groups-and-lie-algebras-i-fall-2020/mit18_745_f20_lec14.pdf)
- [Shuffle Hopf algebra](https://en.wikipedia.org/wiki/Shuffle_algebra)
- [Count-Min Sketch](https://sites.cs.ucsb.edu/~suri/cs290/CormodeMuthukrishnan-CMSketch-JAlg04.pdf)
- [p-adic neural networks](https://arxiv.org/abs/2107.07980)
- [Tropical geometry / Mikhalkin](https://arxiv.org/abs/math/0601041)
- [Monoid-automaton classical text (Pin)](https://www.irif.fr/~jep/PDF/Automata.pdf)
- [Sprague-Grundy / Nim](https://en.wikipedia.org/wiki/Sprague%E2%80%93Grundy_theorem)
- [Moufang loops](https://en.wikipedia.org/wiki/Moufang_loop)
- [Depth-bounds via braid arrangement (NeurIPS 2025)](https://arxiv.org/abs/2502.09324)
- [Kalman Linear Attention (2026)](https://arxiv.org/abs/2602.10743)
