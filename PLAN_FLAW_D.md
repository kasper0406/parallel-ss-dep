# PLAN — Flaw D: honest thinking-gate diagnostics

Owner: flaw D of `THINKING_AUDIT_2026_05_28.md`. **Planning only — no impl, no
training.** The bug is a *reasoning/measurement* artifact, not broken code: the
`pr Δlogp=+5..+8` / `gc tgt1=0.6-1.0` training metrics are computed on the SAME
candidate set the loss is actively moving (positions with `σ>min` or
`σ∈[low,high]`), so a positive number is a mechanical fixed point, not evidence
of utility. The repo's own uniform-sample probe (`probe_process_reward.py`)
found mean Δlogp = **-0.165**, only 30.5% positive. We need diagnostics whose
candidate set is NOT defined by the loss, and whose delta is TASK-relevant.

---

## 1. Non-self-fulfilling diagnostic — policy-sampled, task-grounded Δ(reward)

New script `experiments/probe_think_value.py` (a sibling of
`probe_process_reward.py`, but task-grounded). It answers the only honest
question: *on the positions the deployed model would actually think, does
thinking change the graded outcome?*

**Candidate set — by inference policy, NOT by σ-window.** Do not select positions
with `σ ∈ [low,high]`. Instead the candidate set IS whatever the inference
generator chooses. Run `eval_humaneval.generate_with_retrieval_as_input(...)`
(the actual deploy generator, `additive=cfg["retrieval_input_additive"]`,
`use_incremental=True`, same `emit_threshold`/`gate_floor`/`max_think_per_step`
as eval) on a held-out problem set and record, per problem, every position where
the gate fired (`think_total`, `think_steps_used` are already returned in the
diag dict; add a hook to also record the emit positions where it chose NOT to).
This is the policy distribution — it self-selects exactly the positions the model
deploys thinking on, which is the population the loss claims to improve.

**Delta — task reward, NOT next-token Δlogp.** For each problem produce two full
completions through the SAME generator:
  - `think`  : normal policy (gate free to fire).
  - `no_think`: `emit_threshold = 1.0` (or `total_think_budget = 0`) so the gate
    can never fire — identical model, identical decode, thinking budget zero.
Grade both with `code_grader.grade(problem, completion)` and report
`Δreward = score_think − score_no_think` over the held-out set (dense score in
[0,1], so we get variance even pre-solve). Report: mean Δreward, fraction of
problems with Δreward>0, fraction with Δreward<0, and the paired-bucketed split
by `think_total` (does more thinking help or hurt — directly probes flaw E's
"damage scales with think volume").

**Probes to run it on (each is the PROPER probe for one mechanism):**
  - HumanEval/MBPP via `code_grader.load_humaneval()` — terminal coding reward.
  - `eval_longctx_recall.py` held-out set (`data/longctx_recall_heldout.jsonl`) —
    the recall-corruption axis; report Δrecall (think vs no_think) per distance
    bucket. This is where "thinking corrupts recall" must show up.

**Keep a logp control alongside, explicitly labelled biased-vs-honest.** Run
`probe_process_reward.run_probe` (uniform sampling) AND a new "policy-sampled
Δlogp" (same positions as the policy candidate set above) AND the loss's own
σ-window Δlogp. Print all three side by side. The whole point of flaw D is that
these three numbers DIVERGE; showing them together makes the artifact visible.

---

## 2. Replacement training-time logging (kill the `gc(tgt1=...)` confidence line)

`gc(tgt1=...)` / `pr(%pos=...)` are computed over the loss's own moving
candidate set and MUST NOT be read as "thinking helps". Replace/augment with
metrics whose denominator is NOT the candidate set:

- **Report on a FIXED held-out control batch, uniformly sampled, not the train
  candidate set.** Every `--log_every`, additionally evaluate `pr`/`gc` Δlogp on
  a small fixed held-out batch with *uniform* position sampling (reuse
  `_select_candidate_positions` with `apply_min_sigma=-inf`, i.e. all valid
  targets). Log it as `pr_ctrl(Δlogp=..., %pos=...)`. This is the population
  estimate; it is the number CLAUDE.md's probe found negative. Divergence between
  `pr(...)` (candidate-set) and `pr_ctrl(...)` (uniform) is the live alarm.
- **Report gate-policy drift, not target_frac_one.** Log the *whole-batch* gate
  fire-rate `gate_fire = mean(σ>emit_threshold)` and the candidate-window
  occupancy `gc_window_frac = n_candidates / n_valid_positions`. The audit's
  fixed-point failure (σ→1 empties the window) is then visible as
  `gc_window_frac → 0` while `gate_fire → 1`. `tgt1` going to 1 is meaningless;
  `gate_fire` climbing while held-out task metric flat/down is the real signal.
- **Stop printing `tgt1` as a standalone success metric.** If retained at all,
  rename to `gc_candidate_tgt1` and pair it on the SAME log line with
  `pr_ctrl(Δlogp)` so it can never again be read in isolation as "it works".
- **Periodically (every `--mid_eval_every_tokens`) emit one line of the §1
  Δreward** on a small fixed held-out slice (a dozen problems) — the only
  training-time number that is both task-grounded and policy-distributed.

---

## 3. Tripwire — cheap assertion that would have caught the artifact

Add to the `pr`/`gc` logging path (and as a standalone `--think_tripwire` flag):

> Compute Δlogp on the loss's candidate set (`cand`) and on a uniform-sample
> control of the SAME batch (`ctrl`). If `cand_mean_Δlogp > 0` while
> `ctrl_mean_Δlogp < TRIPWIRE_NEG_THRESH` (e.g. −0.05) for K consecutive log
> steps, log a loud `[THINK-TRIPWIRE] self-fulfilling diagnostic: candidate
> Δlogp=+X but uniform Δlogp=−Y — in-distribution metric is selection bias`.

It is ~free: the uniform control is one extra `_select_candidate_positions` call
with `apply_min_sigma=-inf` reusing the same after-forward machinery (cap at
`max_positions` so cost is bounded). The assertion encodes flaw D directly: a
healthy signal has candidate-set and uniform deltas with the SAME sign; the
artifact is exactly the sign divergence. A stricter optional form: alarm if
`gate_fire` rises >X while the §2 held-out Δreward is ≤0 — "more thinking, no
task gain" (couples D's measurement to E's mechanism).

---

## 4. Dependencies / overlaps with A, B, C, E

- **A (mechanism/stage mismatch):** §1 already fixes this for the diagnostic by
  running the ACTUAL `generate_with_retrieval_as_input` generator with the
  ckpt's `retrieval_input_additive` setting — so the probe measures the deployed
  mechanism even while training still uses `retrieval_as_input=False`. The probe
  is the cross-check that would expose A's no-transfer. No code dependency.
- **B (K-at-once vs iterative re-decode, state_readonly on/off):** §1's
  think/no_think completions go through the iterative-re-decode generator, so the
  probe measures the real variable-think-count behaviour, sidestepping the
  K-at-once proxy entirely. The probe is the validator for any B fix.
- **C (logp target is invalid proxy):** §1 is the empirical demonstration of C —
  it replaces Δlogp with Δreward precisely because C argues logp is the wrong
  objective. If §1's Δreward ≤ 0 while §2's `pr_ctrl` Δlogp is also ≤ 0, C is
  confirmed. The honest diagnostics in §1–§3 are the *measurement* that should
  drive the C verdict.
- **E (capacity competition / always-think drift):** §2's `gate_fire` +
  `gc_window_frac` and §3's "more-think-no-gain" tripwire are the live detectors
  for E's predicted end state (σ→1 across the band, VAL-ppl drift). The
  Δreward-by-think-volume bucketing in §1 quantifies E directly.

This plan is purely additive diagnostics; it does not depend on fixing A/B/C/E
and is in fact the instrument by which their fixes (or kills) get judged.

---

## 5. Branch on aux-loss-kept vs RL pivot

**If we KEEP the aux-loss (and C does not kill it):**
- All of §1–§3 ship as-is. The `pr_ctrl`/tripwire are GATING: do not trust any
  aux-loss run whose held-out uniform Δlogp or §1 Δreward is ≤ 0, regardless of
  candidate-set `tgt1`/`%pos`. Promote the §1 Δreward to the run's accept/reject
  criterion; demote `gc(tgt1=...)` to a debug-only field.
- §1's candidate set must additionally fix flaw A first (calibrate the deployed
  retrieval-as-input mechanism in the SAME stage), else the diagnostic will
  correctly read ≤0 and the aux-loss should be abandoned.

**If we PIVOT to RL (`train_rl_grader.py`, the audit's recommended path):**
- §1 IS the native RL reward: Δ(grader reward) over policy-sampled rollouts is
  exactly what `train_rl_grader.grade()`-based GRPO optimizes. The diagnostic
  collapses into the training signal — no proxy. `--stochastic_gate` makes the
  think/no-think decision a policy variable so the group-relative advantage
  scores the gate's own choices; the §1 think/no_think contrast becomes the
  within-group advantage.
- §2/§3 simplify: the self-fulfilling logp metric disappears (no aux candidate
  set). Replace with RL-native logging — `gate_fire`, KL-to-ref (already in v2),
  and reward-vs-think-volume. The tripwire becomes "reward did not improve while
  think volume rose" (the v1-collapse early-warning the audit describes), reusing
  the same per-rollout reward already computed.
- In this branch §2's `pr_ctrl` is retired entirely; keep §1 (Δreward probe) as
  the offline eval and the tripwire as an RL-stability monitor.

**Decision rule shared by both branches:** an honest diagnostic must have its
candidate set chosen by the inference policy and its delta measured by the
grader. Any metric conditioned on `σ ∈ [low,high]` is reported only with its
uniform-control twin, never alone.
