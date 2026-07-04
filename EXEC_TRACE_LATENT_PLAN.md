# Exec-trace latent thinking — "the neural interpreter" (2026-07-04)

**The bet (user-selected over the parked meta-TTT direction — see memory
`project_meta_ttt_repo_adaptive`):** latent thinking's measured wall is
per-step credit assignment under answer-only supervision
(`project_depth_via_iteration`: homogeneous iteration works, heterogeneous
selection collapses; per-hop supervision or teach-then-wean recovers it).
Code is the one domain where dense per-step ground truth is FREE and
INFINITE: execute the program, and every intermediate state is a supervised
target for every latent step. Train the latent iterations to carry the
interpreter's actual state trajectory — simulation with state-level
supervision, not CoT-text imitation. Scaling axis: verifiable trace data.

**Why this configuration is the one our own results endorse:**
- latent BANDWIDTH validated (continuous state carries what tokens can't:
  0.09 token-feedback vs 1.00 latent on pointer-chase);
- per-hop supervision defeats the selection wall (validated on synthetics;
  answer-only collapses);
- prior in-repo signal: synthetic exec-trace +0.78 / state-track +0.5 latent
  wins (2026-06, pre-fair-control — this program re-runs them properly);
- Pareto-safe attach exists (adapter-only think-path; no-think byte-identical).

## Phases

**N0 — sanity rung (launching 2026-07-04, GPU1):** `gen_state_track` data
(K=2..8, per-hop intermediates; the agentic state-tracking family) trained
ADAPTER-ONLY (frozen trunk) on the arm-A base via the existing
`--latent_reasoning_*` machinery. Establishes: the pipeline learns per-hop
decode on the A-base at all. ~2h.

**N1 — REAL trace data (`experiments/gen_exec_traces.py`, agent building):**
sys.settrace over real executions:
- sources: (a) synthetic-but-real Python (loops/conditionals/arith/list-dict
  mutation) with controlled state-event count K≤12; (b) MBPP/code_exercises
  functions on their test inputs (grader sandbox).
- trace = ordered state-change events for a tracked variable set;
  `intermediates` = the K serialized post-states. v1 CONSTRAINT: values
  single-token (small ints) so `latent_perhop_loss` consumes unchanged;
  v2 relaxes to multi-token per-hop decode.
- schema identical to ptr10dict/state_track jsonl (drop-in for
  `--latent_reasoning_train_prefix`).

**N2 — training ladder on the arm-A base:** stage 1 adapter-only frozen
trunk (Pareto-safe learnability test); stage 2 full fine-tune from A ONLY if
stage 1 learns but saturates (accepting trunk risk knowingly, measured
against the A bar). Depth curriculum K 1→8 + consolidation (uniform K), the
validated recipe.

**N3 — pre-registered evals + kill-test:**
- held-out per-hop state accuracy by K; final-output exact match;
- vs a SEPARATELY-TRAINED no-think control (same data/steps, no latent) —
  the fair-baselines mandate;
- length-gen: train K≤6, test K=8-12;
- transfer: CRUXEval-style output prediction (the STRATEGY #4 gate, now with
  the process supervision it was missing).
- **KILL-SIGNAL:** latent(R=K) fails to beat the matched no-think control by
  ≥5pp on held-out output-prediction at K≥4 after stage 2, or per-hop
  accuracy <50% at K=4 → real traces don't inherit the synthetic win; write
  it up as the science result and stop.

## Notes
- Base = `checkpoints/feature_pilot_A.pt` (best trunk; 0.7429 HE-CE) — code
  competence preserved by construction in stage 1.
- The parked sibling bet (meta-TTT repo-adaptive coder) shares this
  substrate; revisit after N3.
