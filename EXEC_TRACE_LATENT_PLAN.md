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

---

## Results (2026-07-04..11) — see SESSION_FINDINGS.md for the full arc

- **N0**: pipeline sanity on state-track passed, but silently trained rungs
  [2,3] only — `--latent_reasoning_max_len` default 256 dropped K≥4 examples.
  LESSON: always pass `--latent_reasoning_max_len 512` for exec traces.
- **N1**: FAIL — but a methodology bug (loss was answer-only; `intermediates`
  never consumed). Not evidence against per-hop. Fixed in fc6a833.
- **N1′** (per-hop wired, full fine-tune, 350M tok): **REAL FAIL on the
  kill-line** (runs/n3_killtest_N1prime.json). All arms ≈ 9–13% at every K,
  lift −2.3..+2.3pp (needed ≥5pp @K≥4), per-hop acc 9–14% ≈ digit prior
  (hop CE plateau ~2.2 ≈ ln 10). The latents learned the VALUE PRIOR, not the
  simulation. Diagnosis (3 stacked): inverted capability gradient (base can't
  do the task in text either), skipped Scratchpad→Coconut staging, and the
  marginal-stats attractor under CE.

## Staged addendum — Stage A (text) → Stage B (latent compression)

**Stage A — text-scratchpad executor: DECISIVE PASS (2026-07-11).**
Format: prompt + `# trace:\n` + `# step j: x = v` lines + `# final: N`, as a
0.15 mix stream over the pilot mix ×0.85, full FT from feature_pilot_A.
Heldout K=2–8: trace step-acc 0.96–0.99, answer-with-trace 0.94–0.97 vs
direct 0.12–0.16 → **+80–84pp causal lift**; base control 0%; **HE-CE 0.7343
(better than arm A's 0.7429)**. Length-gen: partial (step-acc 0.79/0.73/0.60
at K=9/10/12, answers collapse — stops at trained max depth).
Ckpt: `checkpoints/stageA_executor.pt`. Eval: `eval_exec_trace_text.py`.

**Stage B — Coconut-style gradual text→latent replacement (pre-registered
2026-07-13, launching now).**

Mechanism: curriculum stage `s` replaces the FIRST `s` text-trace steps with
`s` latent (hidden-feedback) steps:
`prompt + "# trace:\n" + [s latent slots] + remaining text step lines
(original numbering kept) + "# final: N"`.
Loss = CE on the remaining text + final (teacher-forced) + per-hop CE decoding
latent slot j → intermediates[j−1] (the N1′ machinery, unchanged convention:
slot j at P_max+j−1, unshifted). At s = K the trace is fully latent.
Implementation rides `_answer_span_latent_loss_batched` as-is: R=s, "solution"
= remaining trace text + final line, per-hop targets = intermediates[:s].

Training: full fine-tune from `stageA_executor.pt` (Stage-A lesson: full
plasticity), fresh LatentFeedbackAdapter, background = the Stage-A mix
(×0.85 pilot + 0.15 exec-text stream stays as the anchor keeping the text
executor alive). Curriculum: s_max ramps 0→8 over the first ~55% of steps,
sampling s ∈ {s_max, s_max−1, s_max−2} (30% on the earlier stages); then
consolidation with s ~ uniform{0..min(K,8)} (the validated
ramp-then-consolidate recipe). Rows in one aux step share the rung K (existing
`_pick_rung`), s_row = min(s, K). Single GPU (DDP+latent incompatibility).

Pre-registered lines (eval = latent twin of `eval_exec_trace_text.py`:
inject R=K latent slots after `# trace:\n`, then greedy-decode; parse
`# final: N`):
- **KILL**: full-latent answer_exact fails to beat the same-ckpt DIRECT
  (no-trace) baseline by ≥5pp at K≥4 on heldout → latent cannot absorb what
  text demonstrably carries in this model; write up and close the latent arc.
- **Mechanism gate**: per-hop decode acc at latent slots ≥50% at K=4
  (vs N1′'s 11% = prior-matching).
- **Success bar (the real prize)**: retain ≥50% of the Stage-A text lift at
  K=4–8 (i.e. answer_exact ≥ ~0.55) — K steps of computation without K×~12
  tokens of scratchpad.
- **Code guard**: HE-CE ≤ 0.755 (within ~0.02 of Stage-A's 0.7343).
