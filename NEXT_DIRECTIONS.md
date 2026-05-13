# NEXT_DIRECTIONS.md

## Current Focus (2026-05-12): Small Super-Coder

The top-level goal is now a small DeltaNet-backbone code model that competes
with larger Transformer code models on HumanEval/MBPP under tight compute.
The thinking head + memory architecture below is in service of that target.

### What changed since 2026-05-10

- **Bounded working memory** (`experiments/model.py::WorkingMemory`) supersedes
  the old "Continuous RAG" (`rag_projection`) idea — the latter was structurally
  dead (gradient = 0 throughout RL training; runs were bit-identical with vs
  without). The new module is in-sequence, bounded at `mem_size` (default 1024)
  with write-gated soft-attention reads at think / query positions. Cost
  O(T·K·d), no quadratic attention. Validated on MQAR: **+10–11 pp recall** at
  the saturation regime.
- **Cross-task validation** shows the architecture has a *read-event-density*
  threshold: helps on tasks with many reads per sequence (MQAR, dyck) and hurts
  on sparse-decision tasks (induction, single read site).
- **Distillation pipeline** from Qwen3.6-35B-A3B-AWQ is wired and runs end-to-end:
  PPL 81 → 41 on 10 M tokens. Subsequent SFT on MBPP+CodeAlpaca drops loss further
  but HumanEval pass@1 remains 0/50 — bottleneck is data scale (we're at ~10 M
  distill + 10 k SFT tokens vs typical small-coder budgets of 100 M-1 B+).
- **Code grader** (`experiments/code_grader.py`) wired against HumanEval and
  MBPP, gold-check verified. Drop-in reward function for the eventual RL push.

### Active next steps
1. Scale the distillation extract (more Qwen3.6 tokens, broader prompt mix).
2. Generate or curate ≥100 k HumanEval-format (signature + docstring → body)
   SFT pairs — the current SFT format mismatch is a known issue.
3. Once HumanEval pass@1 is nonzero, the existing GRPO infra +
   `code_grader.grade` is the natural follow-up.

### Ablations to revisit at training scale (post-pretrain-v1)

Observed in `pretrain_mix_v1` (217 M / mixed-corpus / Muon / 60 k+ steps):
the FiLM (2, 28) scalar α grew **monotonically** from 0 → +0.75 over 60 k
steps and appears to be saturating in the last 5 k (range 0.743-0.756).

Trajectory:

| step | α |
|---:|---:|
| 100 | +0.003 |
| 1 k | +0.026 |
| 5 k | +0.127 |
| 10 k | +0.271 |
| 20 k | +0.478 |
| 40 k | +0.677 |
| 60 k | +0.748 |

Compared to the original FiLM validation runs (α ≈ +0.16 to ±0.20 at
5-15 k steps), this run pushes the scalar 3-5× higher. **Saturation
suggests the *scalar-α* form is hitting its capacity ceiling** — the
model is asking for more cross-layer signal than a single learnable
scalar can deliver.

Things worth retrying at full-pretrain scale, motivated by this
observation:

- **Cross-attention feedback** (`--feedback_xattn_pairs`) — tested at
  short scale (5 k steps, codeparrot only) and didn't beat scalar-α
  FiLM there. With scalar-α now visibly saturated, the higher-capacity
  cross-attention form has a real shot at paying back. Try a 1-2 B
  token rerun with e.g. `--feedback_xattn "2:14,21,28"` (layer 2's
  input attends over layers 14, 21, 28).
- **Per-channel α** (`--feedback_per_channel_alpha`) — scalar α
  forces every channel to share one feedback strength. If different
  channels have different needs, per-channel α should unlock more
  capacity at much lower compute cost than full cross-attention.
  Cheap experiment.
- **Multi-scale feedback distances** (`--feedback_distances 1,2,4`) —
  the current run uses single-step lag. Multi-scale was validated
  at short scale; revisit whether it compounds at long training.
- **K=4 or K=5 self-feed** (currently K=3) — if the self-feed
  amplifies the FiLM effect and K=3 is on a plateau, more iterations
  might extend it. Cheap diagnostic.

Decision: queue these for after pretrain-v1 finishes. Run the
`--freeze_alpha` control first to attribute how much of the actual
val-PPL drop is FiLM-driven; then run the highest-capacity variant
(cross-attention) at matched compute to see if the headline metric
moves.

### Historical context (original 2026-05-10 framing below)

While the sparse far-distance feedback (Finding 9) remains a core architectural
win, the project has pivoted to the **Thinking Head** as the primary path to
frontier scaling and **Continuous RAG**.

### 1. The Thinking Head (Active)
We have implemented a discrete gated path that enables Linear RNNs (DeltaNet) to
perform **Adaptive Computation Time (ACT)** without increasing context length or
KV-cache size.

- **Current Status:** Sweep on auxiliary loss normalization/scaling is active.
- **Key Insight:** Moving from `fresh_tokens` to `aux_items` normalization
  stabilizes the curriculum by decoupling the learning signal from the frequency
  of thinking steps.
- **Verification:** Monitor `runs/think_sweep_*.log`.
- **Scaling Path:** Transition to Reinforcement Learning (**GRPO**) to support
  deep thinking chains (depth 50+) and Continuous RAG. See
  [`RL_RAG_ROADMAP.md`](RL_RAG_ROADMAP.md).

### 2. The Roadmap to Continuous RAG
The thinking mechanism provides the necessary temporal hooks for **External Retrieval**.

- **Mechanism:** Thinking steps will be extended to trigger vector-database
  lookups. The retrieved facts will be projected directly into the DeltaNet
  recurrent state matrix.
- **Impact:** Eliminates the need for massive prompt prefixing in RAG; the model
  loads facts into memory dynamically.

### 3. Scaling & Distillation (Next)
Once the thinking curriculum is stabilized at the 217M scale, we will return to
the distillation track:
- Distill from **Qwen2.5-Coder** (or Qwen3.6) using the Thinking-DeltaNet backbone.
- Evaluate on **HumanEval** and **MBPP** to measure the reasoning lift from the
  extra "thought" passes.

---

## Previous focus (2026-04-27): scaling sparse far-distance feedback

After the architectural exploration logged in
[`SESSION_FINDINGS.md`](SESSION_FINDINGS.md) and
[`RESULTS.md`](RESULTS.md) Phase 14, the project's most defensible
architectural finding is **sparse far-distance cross-layer top-down
feedback**. A single `(target=2, source=28)` connection in a 30-layer
DeltaNet stack gives:

- **−3.5 % PPL vs DN @30L** at +0.3 % params on Python code (T=512)
- **Stable −3.6 to −5.1 %** advantage from training T=512 to T=8192
  (16× extrapolation)
- Survives both param-matched (depth-reduced) and width-matched
  (d_model-reduced) controlled comparisons
- Mechanistically explained by avoiding the compounding pass-2-vs-
  pass-1 divergence that defeats dense feedback variants (probe data
  in [`experiments/probe_feedback.py`](experiments/probe_feedback.py))

### Pre-flight verification (2026-04-27, complete except #4)

1. **3-seed reproducibility at @30L** — ✅ **mean 49.40 ± 0.31 vs DN
   51.00 → −3.14 %**. Real, reproducible.
2. **TinyStories test** — ✅ −1.6 % on natural text. Win is not
   code-specific (smaller delta consistent with simpler corpus).
3. **Depth ablation @15L and @8L** — ✅ sparse beats DN at every
   depth tested (−2.6 %, −0.8 %, −3.1 % at @8L/@15L/@30L). Dense
   single-step has a small @15L sweet spot (loses to sparse at @8L
   and @30L); sparse is the safer default.
4. **Sparse-pair design ablation** — ✅ confirms `(2, 28)` is
   the load-bearing configuration. Reverse direction `(28, 2)` ties
   DN baseline (no win); target shifted to layer 5 `(5, 28)` cuts
   the win to −1.3 %; mid `(2, 14)` and close `(2, 5)` sources
   give graded smaller wins (−1.3 % / −1.0 %). Multi-pair patterns
   `(2, 28)+(10, 20)` and `(2, 28)+(5, 28)` are neutral or worse
   than `(2, 28)` alone. Critically, **only `(2, 28)` learns a
   negative α** (subtractive predictive-coding filter); every other
   pairing converges to positive α with larger magnitude but
   smaller PPL benefit. Full table in `RESULTS.md` Phase 14.

### Next: distillation from Qwen3.6-35B-A3B (chosen path)

Decision (2026-04-28): scaling track is **distillation from
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)** in
INT4 quant ([QuantTrio AWQ](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)).

**Why this teacher specifically:**

- **Architectural alignment.** Qwen3.6's 40 layers are 75 % Gated
  DeltaNet (linear attention) + 25 % Gated Attention. Our student
  is pure DeltaNet with sparse feedback. Most teacher hidden states
  come from the same algebraic family, so layer-wise distillation
  is meaningful (not the cross-architecture mismatch HALO/LoLCATs
  was originally fighting).
- **Frontier coder.** SWE-bench Verified 73.4 %, AIME 2026 92.7 %,
  HumanEval frontier-level. Apache 2.0, released 2026-04-16.
- **Hardware fit.** 35 B total / 3 B active MoE in INT4 ≈ 17 GB —
  fits on one 5090; student trains on the other.
- **Speculative decoding** with the multi-token-prediction head it
  was trained with should give ~2-3× decode throughput.

**Architectural choices we adopt from Qwen3.6 (none of attention):**

1. **Gated DeltaNet** instead of plain DeltaNet (use_gate=True; same
   library: `fla.layers.GatedDeltaNet`). Cheap test underway at
   `gated_deltanet --feedback_pairs "2,28"` @30L.
2. **MoE feedforward — deferred.** Worth revisiting at the 1 B+
   stage, not at 217 M.
3. **No attention layers** (per user direction). We stay pure RNN.
4. **Keep our sparse (2, 28) feedback** as the main architectural
   delta vs the teacher.

**Concrete plan:**

- **Phase 5a — quick architectural sanity** (~30 min): does Gated
  DeltaNet at @30L with sparse (2, 28) beat plain-DN sparse at 49.40
  PPL? Pick whichever wins for the student.
- **Phase 5b — teacher inference** (~1-2 days): set up INT4 inference
  for Qwen3.6 on one 5090 with speculative decoding. vLLM if it
  builds for sm_120, transformers + autoawq if not. Validate teacher
  generates sensible code at >50 tok/s.
- **Phase 5c — mini distillation** (~2-3 days): 217 M (Gated)DN +
  sparse student, KL on output logits + optional last-layer
  hidden-state alignment, 100 M tokens of code. Compare vs
  same-arch trained from scratch (no teacher). The ablation that
  matters: does sparse feedback help distillation more than plain
  DN does?
- **Phase 5d — full scale** (~2 weeks): if 5c wins, push student to
  350-500 M params, 5-10 B tokens. Eval on HumanEval, MBPP, and a
  small SWE-bench-Lite slice.

**Tokenizer note:** student moves to Qwen tokenizer for
distillation (same vocab as teacher → clean logit alignment). Loses
SmolLM2-135M baseline comparability, but that comparability was
never load-bearing for the architectural finding.

### What this supersedes

The Grazzi-clean-cell exploration (below) was the project's original
2026-04 strategy. Several of those candidates were tested at small
scale and either tied with DeltaNet or required non-trivial
infrastructure not worth the engineering vs picking a known-strong
cell + the new sparse-feedback inductive bias on top. The architectural
emphasis has shifted from "design a better cell" to "design better
information flow between cells," which the sparse-feedback finding
empirically validates.

The cell-level work (Heisenberg/RotConj/RotDelta/SO(n)/multi-pass)
remains the documented architectural-decomposition story
(`README.md` headline findings 1-8). Sparse far-distance feedback
(headline finding 9) is the current actionable scaling target.

---

## Historical strategy (kept for context — superseded by the sparse-feedback finding above)

The earlier framing was: break past the T=128 parity wall, gated by
**two constraints that interact**:

1. **Grazzi et al. ICLR'25** ([2411.12537](https://arxiv.org/abs/2411.12537)) —
   any linear RNN with transition spectrum in `[0, 1]^d` is stuck in TC⁰
   and *cannot* express parity at unbounded T. Negative eigenvalues are
   the fix.
2. **Novelty over the modern (2024-2026) parallel-scan literature** —
   the project's value proposition is novel algebraic structures, not
   re-implementations of existing tricks.

A candidate must clear *both* gates to be worth pursuing.

## Candidates eliminated by Grazzi (TC⁰ stuck)

These were brainstormed in an earlier round; all have transition spectrum
in `[0, 1]^d` and so cannot solve parity at unbounded T:

| Candidate | Transition `A_t` | Why eliminated |
|---|---|---|
| Sign-augmented Heisenberg (signs on inputs) | `I` | Signs on inputs don't enter `A_t`. Strictly TC⁰. |
| Selective Heisenberg with positive λ_t | `λ_t·I`, `λ ∈ (0,1)` | Positive decay. Strictly TC⁰. |
| Conv-augmented Heisenberg | `I` (conv pre-mixes inputs) | Extends practical T but not class. |
| Signature K=3 | `I` on each grade | Higher grade ≠ negative eigenvalue. |
| Plain complex Heisenberg (additive) | `I` over `C` | Additive recurrence, transition still identity. |

## Candidates eliminated by novelty (covered by 2024-2026 work)

| Candidate | Subsumed by | Citation |
|---|---|---|
| PGL₂ / Möbius scan | DeltaProduct (K Householder products generate O(d) and beyond) | [Yang et al. NeurIPS 2025](https://arxiv.org/abs/2502.10297) |
| Phase-modulated complex Heisenberg | Complex DeltaNet / S4D-Lin variants | [Gu et al. 2022](https://arxiv.org/abs/2206.11893) |

## The three Grazzi-clean + novel candidates

### A. Negative-eigenvalue Heisenberg ⭐ *most direct fix, modest novelty*

Combine our bilinear cross-pair with DeltaNet's rank-1 erase:

```
a_run_{t+1} = a_run_t + a_t                                    (additive)
P_{t+1}     = P_t · (I − β_t k_t k_tᵀ)                          (rank-1 erase)
c_{t+1}     = (I − β_t k_t k_tᵀ) c_t + a_run_t ⊗ b_t            (one-sided variant)
```

`β_t ∈ (0, 2)` lets the eigenvalue along `k_t` reach `−1`. Strictly more
expressive than DeltaNet (adds bilinear cross-pair on top of erase);
escapes Grazzi's wall.

**Novelty.** Combination novelty: the Heisenberg cross-pair-as-scan-state
is itself novel (`LITERATURE.md`); combining it with DeltaNet's erase has
not been done. Not as flashy as B / C but strictly novel + strictly
NC¹-accessible.

**Lean status.** New monoid `HeisenbergDelta.lean`. Associativity of the
one-sided variant is the load-bearing proof — likely tractable via WY
representation following DeltaNet's chunkwise-scan strategy.

### B. Sₙ permutation-group scan ⭐ *most novel, NC¹-complete*

Per-channel state is a permutation `π ∈ Sₙ`. Transition multiplies by
an input-dependent permutation parameterised as a soft mixture of
adjacent transpositions (Coxeter generators):

```
π_{t+1} = π_t · (Σᵢ wᵢ(x_t) · sᵢ)         where sᵢ = (i, i+1) ∈ Sₙ
```

`w(x_t) = softmax(W_w x_t) ∈ Δ^{n-1}` is a learned soft attention over
the `n−1` adjacent transpositions.

**Why it escapes Grazzi.** A transposition has eigenvalues `{1, …, 1, −1}`.
Sₙ for `n ≥ 5` is non-solvable; per Barrington's theorem, every NC¹ Boolean
circuit reduces to a width-5 group word problem in S₅. So Sₙ-scan is
*NC¹-complete* — provably the strongest possible parallel-scan primitive.

**Combine with our cross-pair.** Let `c = Σ_{i<j} a_i ⊗ b_j` live in the
*regular representation* of Sₙ (an `n!`-dimensional vector space the group
acts on by left multiplication). The cross-pair then captures pair-
occurrence statistics within a non-solvable group's group algebra. This
is a **distinct mechanism** from DeltaProduct's orthogonal-group products.

**Practical scaling.** For `n=8`: state is 8 integers / channel (or a
soft 8×8 doubly-stochastic matrix). Regular rep is 8!=40k-dim — too big.
Use the *standard rep* (n-dim, faithful for Sₙ) or the *adjoint rep*
((n−1)-dim) instead. Trade off expressivity for state size.

**Novelty.** Per the literature search: **zero prior art** as recurrent
scan state in 2024-2026 architectures. Adjacent: Mena et al.'s Sinkhorn
scans and learnable permutations (used as attention, not scan state).
This is the cleanest novelty claim in the portfolio.

**Lean status.** New monoid `PermutationScan.lean`. Associativity is
trivial (Sₙ is a group, multiplication is associative).

### C. Symplectic Sp(2n) scan state *continuous NC¹, novel*

Per-channel state is a `2n×2n` matrix in `Sp(2n, R)`, i.e. preserving a
fixed symplectic form `Ω`. Transition `M_{t+1} = exp(X_t) · M_t` with
`X_t` in the symplectic Lie algebra `sp(2n)`.

**Why it escapes Grazzi.** Sp(2n) is a non-compact simple Lie group (for
n ≥ 1) containing non-commuting generators. Eigenvalues come in pairs
`(λ, 1/λ)` with complex conjugate pairs on the unit circle (including
`±1`). NC¹ accessible.

**Novelty.** SympNets (Greydanus's Hamiltonian NN line, [2003.09779](https://arxiv.org/abs/2003.09779))
parameterise *layers* symplectically but not as a parallel-scan monoid.
Zero hits in 2024-2026 SSM literature for symplectic-state scans.

**Practical scaling.** State per channel: `2n×2n` matrix, but constrained
to Sp(2n) (so 2n²+n parameters). For `n=2`: 4×4 state, 10-dim parameter.
Roughly 2× the cost of an Sp(2)-equivalent SL₂ Möbius cell.

**Lean status.** New monoid `SymplecticScan.lean`. Associativity is
matrix-group-trivial; the `Sp(2n)` constraint requires an invariant proof
or a constrained parameterisation.

## Plan of attack (revised)

1. **Implement A and B as autograd-friendly PyTorch modules** in
   `experiments/layers.py`. Match parameter count to existing 1.05M-param
   scaffold. Defer C unless A/B both fail or under-perform.
2. **Run the parity sweep** at T=64, T=128, T=256 against existing
   baselines (linear, our heisenberg, deltanet, gateddelta, mamba2). The
   theory predicts:
   - **A (neg-eig Heisenberg)**: solves T=128 cleanly, matches DeltaNet
     on parity (both are Grazzi-fixed). Strictly extends DeltaNet by
     adding our cross-pair → may show MQAR/recall advantage.
   - **B (Sₙ scan)**: solves T=128 *and* T=256 *and* the S₅ word problem
     (which DeltaNet alone may struggle with — it's NC¹-complete). Most
     ambitious test.
3. **Triton kernel for the winner.** A is a small extension of our
   existing Heisenberg kernel (one extra MMA per chunk for the erase).
   B requires a new permutation-composition kernel — small per-channel
   matrix multiply with structured sparsity.
4. **Lean formalisation** of the winning monoid's associativity.

## The mathematical structure that breaks TC⁰

This is the *deep* question: **what fundamentally lets a parallel-scan
primitive escape TC⁰?** Once we understand this, we can systematically
design new candidates instead of relying on intuition.

### The structural theorem

**Krohn-Rhodes (1965)** decomposes any finite-state automaton (and thus
any finite monoid that acts on a finite state set) into a wreath product
of two ingredient classes:

  - **Finite simple groups** — the *irreducible* group factors.
  - **"Reset" semigroups** (combinatorial counters; e.g., U₂, T₂).

A monoid is called **solvable** (in the Krohn-Rhodes sense) iff its
decomposition contains *only abelian* simple group factors (Z_p) — no
non-abelian simple groups appear.

**Barrington's theorem (1989)**: a Boolean function `f : {0,1}^n → {0,1}`
is computable by polynomial-size, log-depth circuits (i.e., is in NC¹)
iff it is computable by a width-5 branching program over a *non-solvable*
group (canonically S₅).

**Combining the two**: a parallel-scan primitive (associative monoid `M`
under fold) can recognise a regular language `L` iff `L` is recognised by
the syntactic monoid of `L` — and that monoid embeds in `M`. The
expressivity hierarchy is then:

  - `M` solvable → scan recognises only languages whose syntactic monoid
    is solvable → parity, mod-p counting at *bounded* T but **not**
    unbounded T → strict subclass of TC⁰.
  - `M` contains a non-solvable simple group as a subquotient → scan
    recognises NC¹-complete languages (S₅ word problem).

This is **the** organising principle. Our existing portfolio:

| Cell | Monoid | Solvable? | Class |
|---|---|---|---|
| Linear-attn | `(R^{d×d}, +)` | abelian → solvable | TC⁰ |
| Heisenberg | rank-2 nilpotent group `H_d` | nilpotent → solvable | TC⁰ |
| Mamba2 | `(R^+, ×) ≅ (R, +)` | abelian → solvable | TC⁰ |
| DeltaNet (default) | contractive Householder semigroup | "contracts toward solvable" | TC⁰ practically |
| **DeltaNet `allow_neg_eigval=True`** | full orthogonal group `O(d)` | contains non-solvable A₅ | **NC¹** |
| **DeltaProduct (K Householders)** | products of reflections → `O(d)` | non-solvable | **NC¹** |
| **Sₙ scan (n ≥ 5)** | symmetric group `S_n` | non-solvable | **NC¹-complete** |
| **SO(n) scan (n ≥ 3)** | rotation group `SO(n)` | contains A₅ | **NC¹** |

### Constructive principle for novel candidates

**Recipe:** *to design a new TC⁰-escaping scan, parameterise the
transition `A_t` to take values in a Lie group (or finite group) that
contains a non-solvable simple group as a subquotient.* Add bilinear
cross-pair (Heisenberg-style) on top in a way compatible with the group's
action on the cross-pair's vector space.

This rules out:
  - **Diagonal / scalar transitions** (always solvable).
  - **Triangular / nilpotent transitions** (Heisenberg-style — solvable).
  - **Positive-definite / contractive transitions** (effectively in
    `R^+ ⊂ (R, ×) ≅ (R, +)` — abelian).
  - **Any "decay-only" / "gating" architecture with positive λ_t**.

This admits:
  - **Reflection-based transitions** with negative eigenvalues
    (DeltaNet's true `allow_neg_eigval` mode, DeltaProduct).
  - **Rotation-based transitions** in dim ≥ 3 (SO(n) scan, our quaternion
    Rotor cell with input-dependent rotation).
  - **Permutation-based transitions** (Sₙ scan).
  - **Lie-group transitions** in any non-abelian Lie group with a
    non-solvable finite subgroup (SU(n), Sp(2n), exceptional groups,
    classical Lie groups except for SO(2)).

### Open creative directions

Given the principle, these are the unexplored corners worth searching:

1. **Free-group / braid-group state.** The braid group `B_n` is
   non-solvable for `n ≥ 4` and infinite. As a continuous parameterisation,
   parameterise transitions as braid generators (positive/negative
   crossings between adjacent strands). Connection to topology: the
   scan is computing a braid word, which has rich expressive structure.
2. **Group-algebra cross-pair (the recommended Sₙ + Heisenberg combo).**
   Let `c = Σ_{i<j} a_i ⊗ b_j` live in the *group algebra* `R[G]` of a
   non-solvable group `G`, with the group's action on `R[G]` providing
   the non-solvable transition.
3. **Wreath products.** `Z_2 ≀ S_n` (signed permutations), `S_m ≀ S_n`,
   etc. Wreath products mix abelian and non-abelian structure cleanly.
4. **Complex modular forms / Bianchi groups** `PSL_2(O)` for `O` an
   imaginary-quadratic-integer ring. Used in number theory; never used
   as scan state.
5. **Modular tensor categories.** State lives in objects of a modular
   tensor category; transitions act via the braiding. Connection to
   topological quantum computation.

The first three are concrete and could be implemented next round.

## The active direction: semidirect-product scan `SO(n) ⋉ ℝ^{n×n}`

After validating SO(n) scan empirically (parity at T=64/128/256/512 to
100 %, beating DeltaNet/Mamba2 on convergence speed) and discovering
**concurrent prior art in AUSSM** ([arXiv:2507.05238](https://arxiv.org/abs/2507.05238))
which already proposed essentially the same skew-symmetric input-
dependent matrix-exp construction, we re-strategised. Three parallel
research agents converged on the same next-step: a **semidirect-product
scan combining SO(n) rotation with a rotation-conjugated unbounded
matrix memory**. Most novel + tractable + powerful candidate among the
options surveyed.

### The construction

Per channel, state is a pair `(R_t, c_t)`:
- `R_t ∈ SO(n)` — orthogonal rotation matrix (the "control" slot)
- `c_t ∈ ℝ^{n×n}` — unbounded matrix memory (the "KV" slot)

Per-token input is `(O_t, kv_t)` with `O_t = exp(skew(W_skew · x_t))`
and `kv_t = k_t ⊗ v_t`. Recurrence:

```
R_t = O_t · R_{t-1}                                     (rotation, like ortho)
c_t = O_t · c_{t-1} · O_tᵀ  +  kv_t                     (rotation-conjugated KV)
```

Composition (the monoid combine, **provably associative**):

```
(R_a, c_a) · (R_b, c_b) = (R_b R_a,  R_b c_a R_bᵀ + c_b)
```

Readout: `o_t = q_t · c_t` (linear-attention style, q_t is a learned
projection of the input).

### Why this escapes Grazzi's TC⁰ wall

The transition on `c` is a linear map `c ↦ R c Rᵀ`, which on `vec(c)`
is the Kronecker product `R ⊗ R`. Eigenvalues of `R ⊗ R` are products
`e^{i(θ_a + θ_b)}` of eigenvalues of `R`. **This includes −1** when
`θ_a + θ_b = π` (e.g., two orthogonal-plane π/2 rotations). So the
rotation imports its negative spectrum into the *unbounded* memory slot
itself — Grazzi's condition fails on `c` directly, not just on a
separate rotation channel.

### Why it addresses AUSSM's specific weaknesses

Per the AUSSM paper deep-dive, three explicit gaps in their work:

| AUSSM gap | Their construction | Our fix |
|---|---|---|
| **No MQAR / associative recall.** Pure unitary state has bounded operator norm = 1; can't grow signal with K. AUSSM doesn't even evaluate Zoology benchmarks. | Diagonal/vector unitary state. | Add unbounded `c ∈ ℝ^{n×n}` slot — additive accumulation like linear-attn / DeltaNet. |
| **Cannot reach S₅/A₅** (Sec 6 limitation). | "Requires simultaneous diagonalisability." | Our `R_t ∈ SO(n)` are non-commuting matrix transitions — span A₅ ⊂ SO(3). |
| **Majority/equation fail (0.07-0.10 acc).** | "Pure unitarity destroys aggregation." | `c_t` updates additively → integration / decision natural. |

### Why it's genuinely novel (after thorough check)

Agent 3 ran a deep prior-art sweep:

> *"I cannot find a published scan that uses semidirect product `G ⋉ V`
> (with `G` a Lie group acting on a matrix-valued `V` by conjugation)
> as the parallel-scan monoid."*

Architectures checked and confirmed distinct:
- **DeltaProduct** ([arXiv:2502.10297](https://arxiv.org/abs/2502.10297)):
  rank-1 Householder products *fused* into one matrix update — no
  separate `(R, c)` slot, no conjugation `R c Rᵀ`.
- **Gated DeltaNet** ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)):
  scalar/diagonal gating on KV; not a Lie-group action by conjugation.
- **RWKV-7 "Goose"** ([arXiv:2503.14456](https://arxiv.org/abs/2503.14456)):
  time-varying transitions but diagonal/scalar.
- **PaTH Attention** ([arXiv:2505.16381](https://arxiv.org/abs/2505.16381)):
  Householders for *position encoding*, not as scan state.
- **PD-SSM** ([arXiv:2509.22284](https://arxiv.org/abs/2509.22284)):
  column-one-hot × complex-diagonal, no semidirect framing.
- **AUSSM** ([arXiv:2507.05238](https://arxiv.org/abs/2507.05238)):
  diagonal unitary state, no conjugated memory.
- **HGRN-2, Titans, TTT, Kalman LA**: no rotation × memory hybrid.

Novelty estimate from agent: **7.5/10**. Lower than 9 because the
*individual ingredients* (SO(n) scan, outer-product KV, Householder
products) are 2025-published, but the **explicit semidirect-product
framing** is genuinely new and unifies two of our existing Lean cells
(Heisenberg + SO(n)) under a clean group-theoretic structure.

### Predicted empirical wins

| Task | SO(n) scan | DeltaNet | Mamba2 | AUSSM | Our `(R, c)` |
|---|---|---|---|---|---|
| Parity at T=512 | ✅ slow | ❌ regression | partial | ✅ | ✅ |
| Modular addition mod 3, 5 | ? | ❌ (Z₂ only) | ? | ✅ | ✅ |
| **MQAR / recall** | ❌ (bounded) | ✅ | ✅ | ❌ no eval | ✅ |
| **Majority / aggregation** | ❌ | ✅ | partial | ❌ | ✅ |
| S₅ word problem | partial | partial | ❌ | ❌ | ? non-abelian R |

The key empirical claim we'd test: **`(R, c)` is the first parallel-scan
primitive that solves *both* parity-style state-tracking *and* MQAR-style
recall simultaneously**, addressing the central tradeoff in the field.

### Implementation plan

1. **PyTorch module first.** `RotConjAttention` in
   `experiments/layers.py` — extends `OrthogonalScanAttention` with a
   conjugated outer-product memory. Sequential scan, autograd-friendly.
   ~50 lines.
2. **Empirical sweep, ranked by surprise value:**
   - **MQAR** — the test that matters for the "addresses AUSSM's gap"
     claim. Should beat ortho (which has no recall) and AUSSM.
   - **Modular addition mod 3, 5, 7** — Grazzi only solves p=2; we
     should solve all p.
   - **Parity at T=512+** — confirm we don't lose ortho's strength.
3. **Triton kernel.** Extend `kernels/ortho_son/` with a parallel `c`
   slot. ~60 lines on top of the existing kernel.
4. **Lean proof.** New module `SemidirectScan.lean` — associativity in
   ~30 lines using "conjugation is a group action".
5. **(Optional, ambitious)** S₅ word problem at constant depth — hard
   test of whether SGD finds non-abelian rotations.

## Eliminated (kept for transparency)

- **PGL₂ / Möbius**: covered by DeltaProduct.
- **Phase-modulated complex Heisenberg**: covered by complex DeltaNet
  variants.
- **Sign-aug / selective / conv / signature K=3 Heisenberg**: TC⁰ stuck
  per Grazzi.

These can revisit later if `(R, c)` succeeds and we want complementary
boosts.

## Next iteration: hunting hybrid's failure modes

After Phases 1-12 produced a hybrid that wins on continuous-angle
state-tracking and is competitive on real-text LM, the honest next
move is **systematically finding where hybrid v2 fails** and
identifying the missing primitive.

### Where to look for failures (ordered by expected information value)

1. **Long-T S₅ word problem (T=512)**. At T=128, deltanet_negeig wins
   (98 % pos_recall) and hybrid lags (71 %). At T=512, three possible
   outcomes:
   - (a) deltanet_negeig catastrophically fails like with mod-5 → SGD
     limits expose, hybrid wins by default.
   - (b) hybrid catastrophically fails → confirms SO(n)-rotation
     doesn't scale to long-T non-solvable group state-tracking.
   - (c) both fail → need a new primitive entirely.
   Whichever way it goes, this is the cheapest test to run and
   directly tells us whether SO(n) rotations generalise to long-T
   non-abelian state.

2. **Selective copy** (Mamba paper). Sequence with mostly noise tokens
   and a few signal tokens (marked by a prefix). Output the signal
   tokens in order. **Hybrid likely fails** on this:
   - Ortho rotation is *always-on* — there's no input-dependent
     "skip this token" mechanism. SGD has to learn to make the
     rotation near-identity on noise tokens, which is an
     optimisation problem, not an architectural one.
   - DeltaNet has implicit selectivity via β_t.
   - This task identifies the **selectivity wall** for ortho, which
     points at the missing primitive (Mamba2-style `Δ_t = f(x_t)`
     gating the rotation).

3. **Multi-class mod-p with very large p** (p ∈ {11, 13, 17}). At
   small p the rotation angle 2π/p is easy for SGD; at p=17 it's
   ~21° — much harder to find precisely. Tests the *optimisation
   regime* of SO(n) rather than its expressivity.

4. **Stack-with-types tasks** (deeper Dyck variants). Pure depth
   tracking (current Dyck task) is one thing; matching `}` to a
   specific `{` from N tokens ago is harder — it requires per-depth
   *type memory*. Hybrid's rotation accumulator doesn't natively
   encode "type of thing at depth k". This may be where DeltaNet's
   recall is essential and hybrid's rotation alone insufficient.

5. **Long-context recall (MQAR variants at T=4096+)**. Push
   DeltaNet's recall ability past where it works at scale. If both
   pure DeltaNet and hybrid fail, "selectivity + erase" isn't
   sufficient and we need stronger memory primitives.

### The likely missing primitive: selective rotation

The diagnosis from (2) and (4) above points at **input-dependent
gating of the rotation magnitude**. Mamba2 has this for its scalar
state via `Δ_t = f(x_t)`; ortho currently doesn't. State update
becomes:

```
λ_t = σ(W_λ · x_t)                      ∈ (0, 1)
R_t = exp(λ_t · skew(W_skew · x_t)) · R_{t-1}
```

`λ_t = 0` ⇒ identity rotation (skip token), `λ_t = 1` ⇒ full
rotation. Adds the "selectivity" that ortho currently lacks while
preserving:

- the rotation primitive (Grazzi-clean — eigenvalues still on unit
  circle).
- parallel-scannability (cumulative product, same Triton kernel).
- the Lean associativity proof (semidirect product structure).

This addresses (2) selective copy and probably (4) deep nesting —
both need "skip this token" capability.

Implementation: ~30 lines of `OrthogonalScanAttention` extension. No
kernel change required — `O_t = exp(λ_t · skew_t)` is computed before
the scan; the scan kernel handles whatever per-token rotations it's
given.

### Recommended sequence (after current overnight results land)

1. **S₅ T=512** sweep (fastest test; ~30 min).
2. **Selective copy** sweep (predict failure of ortho, confirm with
   data; ~30 min).
3. **Implement selective rotation** as
   `OrthogonalScanAttention(use_selective_lambda=True)`.
4. **Re-run** the empirical scorecard with the new selective ortho:
   - Mod-p sweep (should preserve win)
   - Selective copy (should now solve)
   - LM PPL on TinyStories (likely improves)
   - LM PPL on Python (the user's stated goal)
5. **Triton backward kernel** for matmul-scan to close the remaining
   ~2× wall-clock gap to DeltaNet.
6. **Distill from a coding teacher** (DeepSeek-Coder, StarCoder) into
   the upgraded hybrid; evaluate on HumanEval / MBPP.
