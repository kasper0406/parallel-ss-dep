# state-dep-parallel — research plan

## Current direction (as of 2026-05-12) — small super-coder

**Top-level goal:** a DeltaNet-backbone code model that competes with
much larger Transformer code models on coding benchmarks under tight
compute (2× RTX 5090). All architectural research below now feeds this
target.

### Validated this round
- **Bounded working memory** (`experiments/model.py::WorkingMemory`):
  write-gated buffer of past hidden states, read at think / query
  positions via soft attention. **+10–11 pp recall** on saturated
  MQAR (T=512 / K=64, 128). Cost O(T·K·d), no quadratic attention.
- **Read-event-density threshold**: induction (1 read/seq) **−28 pp**;
  dyck (per-position) tied but ~2× faster; MQAR (~256+ reads) +10 pp.
  Memory only helps when reads happen frequently.
- **Distillation pipeline end-to-end** with Qwen3.6-35B-A3B-AWQ
  teacher: 10 M tokens → val PPL 81 → 41. SFT on MBPP + CodeAlpaca
  pushes loss to ~2; HumanEval pass@1 stays at 0/50 — at this data
  scale the model is code-shaped but not problem-solving. Bottleneck
  is data, not architecture.

### Ruled out
- Corpus-RAG (the old `rag_projection`) — structurally dead. Replaced
  by `WorkingMemory`.
- Natural-text RL on the 8.2 M-token SFT base (HumanEval 0/20).
- 10 M-token Qwen-distill + 10 k-pair SFT — same scale → same outcome.

### Next round (active)
- **Run the actual RL+memory experiment.** Architecture and pipeline are
  validated independently, but a real RL run with memory enabled and
  extended thinking depth (≥16) on the SFT base has not been measured
  end-to-end. Headline metrics to track: `grpo/avg_depth`,
  `grpo/think_rate`, `mem/proj_norm`, `mem/write_gate_mean`,
  `grpo/depth_ce_d{i}` (does deeper thinking lower CE?), and
  HumanEval/MBPP pass@1 before/after. Memory was only ever trained on
  the synthetic ablations; RL is its first real-data test.
- Scale data ≥10× (deferred until the RL run tells us whether the
  architecture is moving the needle on real text).
- The eval harness (`experiments/code_grader.py`) is ready.

### Update (2026-05-14) — pretrain is now the active blocker, not RL

Diagnosed why pretrain was undertraining badly (residual-stream collapse
from WD 0.1) and fixed it. **The plan above is paused** — RL on an
undertrained base is the cold-start failure mode the docs already warn
about. Sequence now:

1. **v3-long** (active): `--wd 0.01`, 2.13 B tokens. Clean schedule test.
2. **v4 pretrain**: re-weighted mix (the `bigvul`/`cybernative` upward CE
   drift is the one thing still unfixed — mix imbalance) + the staged
   knobs (`--wd 0.01 --lr_schedule wsd --compile
   --feedback_self_k_warmup_steps`).
3. SFT, then GRPO — with the **dense execution-grounded reward** (tier
   ladder + fractional score, `code_grader.py` rebuilt 2026-05-14) and
   the **iterative self-repair loop** (re-add failed tasks with the
   compiler/test `error_text`). See `PHASE_C_RL.md`.

Authoritative current state: `GEMINI.md` "Current state" + the
2026-05-14 entries in `HANDOFF.md` / `SESSION_FINDINGS.md`.

The formalisation work below is still the *backbone* of why we use
DeltaNet; the active research is now applying that backbone to the
super-coder target.

---

## Original goal (formalisation arm)

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

## Current Primary Research Pivot: Reinforcement Learning (GRPO)

Supervised auxiliary losses for the Thinking Head failed in Phase 23 (Maladaptive
Thinking Trap). The focus has now moved to **Reinforcement Learning**:

- **Active Task:** Transition the Thinking Head to **GRPO-based training**.
- **Mechanism:** Use detached trajectories (no-grad) to sample thinking depths
  and reward based on final-token NLL reduction.
- **RAG Integration:** Design the rollout loop to trigger vector DB lookups during
  thinking actions.
- **Immediate Milestone:** Implement `experiments/thinking.py` trajectory
  sampling for RL.
- **Evaluation Phase (Active):** Run RL-trained checkpoints on existing benchmarks
  (HumanEval, Mechanistic Tasks) to verify the reasoning lift.

## After these are done

- **Continuous RAG:** Wire the Thinking Head actions to external vector DBs.
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
