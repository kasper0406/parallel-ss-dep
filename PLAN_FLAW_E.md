# PLAN — Flaw E: capacity competition + always-think drift

Owner of flaw E from `THINKING_AUDIT_2026_05_28.md`. Read-only plan; no impl, no
training. Companion flaws: A (mechanism/stage mismatch), B (K-at-once vs iterative
decode), C (wrong target — next-token logp proxy), D (self-fulfilling diagnostics).

## 0. Restatement / root cause

The gate-calibration + process-reward aux losses drive σ→1 across the entire
uncertain band (audit D: "σ climbed 0.29 → 0.76"). Every extra think is a
state-corrupting DeltaNet recurrence step (CLAUDE.md: "thinking corrupts recall,
damage scales with think volume") AND consumes 287M of shared trunk capacity that
the direct-emit channel needs (audit: VAL ppl +12% smoke). The thinks are
calibrated to sharpen locally-uncertain surface tokens (flaw C), so the spend buys
nothing task-relevant. Net mechanism: over-think → downstream regression
(HumanEval 10→3, synth_reasoning 3→0).

The honest framing (see §4): **flaw E has no independent fix while flaw C's target
is in place.** Any volume penalty added on top of a logp-proxy reward just trades
one wrong operating point for another. The plan therefore has two tiers — (T1)
what to do once the objective is execution-grounded (the real fix), and (T2) a
cheap pretrain-time safety rail that is worth keeping regardless.

## 1. Mechanisms to keep the gate SELECTIVE

Inventory of available levers and where each one belongs:

| lever | where it lives | pretrain | RL | verdict |
|---|---|---|---|---|
| ponder cost (`--grpo_ponder_*`, `train_rl_grader --ponder_cost/_shape/_warmup`) | `thinking.py`, `train_rl_grader.py` | n/a | **primary** | use |
| sparsity prior on σ (mean-σ L1 / KL to low Bernoulli) | new, cheap | secondary rail | redundant | optional |
| entropy regularizer on σ (Bernoulli H) | exists in RL (`--gate_entropy_bonus`) | n/a | anti-collapse only | keep small |
| explicit think budget (per-burst cap, per-seq cap) | partly in RL rollout `force_emit` | hard rail | hard rail | use |
| entropy-grounded gate target (`--gate_entropy_aux_weight`) | `train_lm._nonthink_forward_loss` | **default rail** | n/a | use |

**Pretrain-time (the rail, T2).** Do NOT add the process-reward / gate-calibration
losses (flaw C makes them counterproductive — they are the *source* of the drift).
Replace with the already-validated `--gate_entropy_aux_weight` entropy-grounded
target, which encodes the same "think at uncertain positions" signal at zero extra
forward and without driving σ→1 (it has a calibrated target in [0,1], not a
one-sided "after > before" push). Add a light **sparsity prior**: a penalty
`λ_sparse · mean(σ)` (or KL(σ̄ ‖ p_target) with p_target ≈ 0.1) so the gate's
*average* fire-rate is anchored low and only rises where the entropy target
genuinely justifies it. This is a one-liner in `_nonthink_forward_loss` next to the
existing entropy-aux BCE; default weight 0, opt-in. Rationale: at pretrain time
there is no terminal reward, so the only honest thing the gate can learn is "this
position is uncertain" + "don't fire indiscriminately." Pretrain's job is to
produce a *non-collapsed, low-baseline* gate; RL decides which uncertain positions
are worth the spend.

**RL-time (the real selectivity mechanism, T1).** Move the think decision into
`train_rl_grader.py` as a policy variable via the existing `--stochastic_gate`
(already records `gate_decisions/log_probs/positions` and a per-rollout gate-PPO
surrogate). Apply the existing ponder-cost shaping with the validated Phase-C
recipe: `--ponder_shape quadratic --counterfactual --ponder_warmup_steps 300`.
Quadratic shape penalizes *bursts* super-linearly (a 4-think burst costs 16×, not
4×) — exactly the over-think pattern we must suppress. `--counterfactual` clamps
the task component at the depth-0 baseline so a think can never *worsen* task
reward but always pays its depth cost: this is the precise economic statement of
"only think if it earns more than its capacity+recurrence cost." Keep
`--gate_entropy_bonus` small (0.01) purely as anti-collapse insurance, NOT as a
pressure to think more. Add a **hard per-burst budget** in the rollout
(`force_emit` after N consecutive thinks; N≈4–8) so a runaway burst cannot corrupt
the recurrence catastrophically before the soft cost bites — this is the safety net
that the v1 RL collapse (depth 120→30 swing) showed we need.

**Curriculum note.** The v1→v2 RL history (CLAUDE.md) shows ponder cost is the
collapse trigger if applied cold and too hard. Keep `--ponder_warmup_steps`
non-zero, KL-to-reference on (`--kl_coef 0.05`), and lean on the warmup so depth
decays smoothly rather than snapping. The selectivity must emerge as a slow
equilibrium, not a step-function cut.

## 2. The `state_readonly_at_think` angle

Mechanism (CLAUDE.md): forces the DeltaNet write-gate β→0 at think positions via a
forward-hook on `b_proj`; thinks still READ the recurrent state (local h_t feeds
gate / WM / lm_head) but never WRITE to it. Synthetic probe: ON 0.88 vs OFF 0.41
recall under multi-think bursts.

- **Does a state-readonly think still corrupt recall?** Largely no — that is the
  whole point. The documented corruption is "every think token *steps* the
  recurrence and perturbs the binding the linear-RNN state carried." With β=0 the
  delta-rule write is suppressed, so the state the next emit reads is the same one
  it would have read with zero thinks. This converts an extra think from
  *state-destructive* to *state-neutral* (read-only scratch compute).
- **Should it always be ON to make thinking "safe"?** As a *recall-safety* default
  for the deployed model, yes — and the audit already flags the live bug: pretrain
  sets it ON, SFT/inference do NOT (`grep --state_readonly_at_think
  launch_sft_*.sh` → 0), so the deployed thinks DO corrupt state. **First action:
  make the train and deploy state_readonly setting identical** (overlaps flaw A/B).
  Recommend ON end-to-end as the recall-safe baseline.
- **The trade-off (real).** A think that cannot write cannot *accumulate* reasoning
  across a burst — each think in a burst sees the same frozen recurrent state, so
  multi-step chained reasoning that must persist intermediate results in the
  recurrence is impossible. Reasoning then has to flow through the WM channel and
  the per-position think-index embedding (the additive, non-recurrent paths)
  instead of the DeltaNet state. For SHORT-horizon coding completions (HumanEval
  fits inside the recurrent state, CLAUDE.md) this trade is almost free: we want
  recall-safety, and the WM/think-index channels carry the reasoning. For
  genuinely multi-step chained arithmetic it may cost a little.
- **Recommended posture.** Default state_readonly ON everywhere (safety + the
  forward-path β-mask is the validated code path). This *decouples* flaw E from the
  recall-corruption term: with β=0, over-thinking still costs trunk capacity and
  decode latency, but it no longer corrupts recall — so the remaining penalty
  needed is just the capacity/latency ponder cost of §1, a much milder pressure.
  Treat "make think volume not destroy recall" (state_readonly) and "make think
  volume economically justified" (ponder cost) as the two orthogonal halves of
  flaw E. Open caveat: `forward_step` decode path leaves β unmasked (CLAUDE.md
  known follow-up) — flag for B's owner; the RL rollout uses incremental decode, so
  the safety guarantee currently holds only on the full-sequence path.

## 3. How to measure over-thinking and its cost directly

The audit's core indictment (flaw D) is that every cited "success" was measured on
the loss's own moving candidate set. The fix is policy-distributed, terminal
measurements:

1. **think_rate vs downstream pass — the headline curve.** Sweep the operating
   point (ponder_cost, or a hard budget N) and plot grader pass@1 (HumanEval/MBPP)
   AND long-context recall vs realized think_rate / mean burst depth, measured by
   running the **actual inference generator**. The selective sweet spot is the
   think_rate that maximizes pass without tanking recall. If pass is flat or
   decreasing in think_rate, over-thinking is confirmed and the budget should be
   tightened toward 0. (`eval_humaneval.py` already reports depth; add a think_rate
   column.)
2. **Per-burst state drift (direct recall-corruption probe).** On
   `eval_longctx_recall.py` var_binding tasks, for each generated burst of length d,
   measure ‖h_after_burst − h_before_burst‖ / ‖h_before_burst‖ on the recurrent
   state at the binding position, as a function of d. With state_readonly OFF this
   should grow with d (the corruption); with ON it should be ≈0. This is the
   quantitative confirmation that §2's β-mask actually neutralizes the recurrence
   step. Bucket by distance (64/128/256/384/512) per the existing recall table.
3. **Capacity-competition direct read.** A/B VAL ppl of the same ckpt with the
   aux-think losses ON vs OFF at matched tokens — the +12% smoke number is the
   signal; track it as a guardrail metric on every aux-loss run. A real fix must
   keep this ≤ a small epsilon.
4. **Policy-distributed Δ(grader reward), NOT Δlogp** (the audit's prescribed
   missing control). Sample candidate positions by the *inference gate policy*, run
   the actual retrieval-as-input generator for the think burst, report Δ(pass /
   grader score) with vs without the budget. CLAUDE.md's own uniform-sample probe
   already found Δlogp = −0.165; the grader version is the honest replacement.

## 4. Honest assessment — symptom of C, or independent?

**Mostly a symptom of C, but with a residual independent component.**

- The *direction* of the drift (σ→1, think everywhere) is entirely C's doing: a
  one-sided "after > before next-token logp" reward is ≈ "is this position
  uncertain," which is true almost everywhere in the uncertain band (audit C.2), so
  the gate is pushed up everywhere. Remove that target (use the entropy-grounded
  target + execution-grounded RL per the audit's recommendation) and the *pressure
  to over-think largely disappears at the source.* In that sense E is C's
  downstream symptom and will substantially shrink once the objective is terminal.
- The *residual independent component*: even with a perfect terminal objective, an
  RL gate with no cost on volume has no reason to be *parsimonious* — thinking is
  "free" to the task reward if it doesn't hurt, so the gate can drift toward
  always-think and silently pay the capacity + latency + (if not state_readonly)
  recall cost. That is why the ponder cost / counterfactual / hard budget of §1 are
  still needed *as their own fix*, not just as a patch for C. The repo's v1 RL
  collapse (no/weak ponder governance) is the empirical proof that even an
  execution-grounded objective needs an explicit volume governor.
- **Verdict:** primary fix = remove flaw C's target (this kills ~80% of E); the
  remaining ~20% (parsimony under a correct objective) needs E's own mechanism =
  quadratic counterfactual ponder cost + hard burst budget + state_readonly-ON.
  E is NOT fully independent (don't fix it in isolation on top of the bad target),
  but it is NOT fully reducible to C either (a volume governor is still required).

## 5. Dependencies / overlaps with A, B, C, D

- **C (root cause).** Highest dependency. Do not add any volume penalty on top of
  the logp-proxy target — it would just relocate a wrong operating point. E's plan
  is *contingent* on C's fix (drop logp target → entropy target in pretrain,
  execution-grounded reward in RL). Sequence: fix C first, then tune E's governor.
- **A (mechanism/stage mismatch).** §2's "state_readonly identical train↔deploy" and
  §1's "one stage, one mechanism" directly overlap A's recommendation 3. The ponder
  cost must be applied in the *same* stage and on the *same* think mechanism
  (retrieval-as-input additive-α) that deploys, or the volume governor governs the
  wrong distribution.
- **B (K-at-once vs iterative).** §3's measurements MUST use the iterative
  re-decode inference generator, not the K-at-once forced forward, or think_rate /
  depth are measured on a distribution that never occurs at deploy. The
  `forward_step` β-unmasked gap (§2 caveat) is B's territory and blocks the
  recall-safety guarantee on the RL decode path — coordinate.
- **D (self-fulfilling diagnostics).** §3 *is* the antidote to D: every E metric is
  policy-distributed + terminal, with the explicit VAL-ppl guardrail, so it cannot
  become a fixed point of the loss the way D's candidate-window metrics did.
- **No conflict with the WM / PKM / FiLM stack.** state_readonly + ponder cost are
  orthogonal to those; the only interaction is positive (state_readonly pushes
  reasoning into the WM channel, which is where CLAUDE.md wants it).

## 6. Concrete deliverable sequence (for the implementer, not done here)

1. Make `state_readonly_at_think` ON in SFT + inference (match pretrain). Re-run the
   per-burst state-drift probe (§3.2) to confirm ≈0 drift.
2. Drop process-reward/gate-calibration from pretrain; keep
   `--gate_entropy_aux_weight`; add opt-in `λ_sparse·mean(σ)` sparsity prior.
3. In `train_rl_grader.py`: `--stochastic_gate` + quadratic counterfactual ponder
   cost + `--ponder_warmup_steps` + small `--gate_entropy_bonus` + hard per-burst
   budget; KL-to-ref on.
4. Run the think_rate-vs-pass sweep (§3.1) + VAL-ppl guardrail (§3.3) + Δ(grader)
   policy probe (§3.4). Pick the operating point that maximizes pass at minimal
   think volume.
