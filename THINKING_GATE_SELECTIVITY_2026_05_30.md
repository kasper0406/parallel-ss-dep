# Gate selectivity: "think only where helpful" (2026-05-30)

## The finding (RL rollout smoke on `rl_grader_phase_c_step300`, GPU-1)

At RL-rollout temperature (τ=0.7), the emit/think **gate collapses to thinking
at every position**:

```
max_gen 256, single-turn, B=3×N=4 (12 rollouts):
  reward_mean 0.1708   tier_hist: partial=10, exec_error=1, syntax_error=1
  depth_mean 120.00    think_rate 1.000
```

`depth_mean` is pinned at **exactly `total_think_budget`=120** — the gate never
chooses to emit; it thinks until the budget is exhausted and is then
*force*-emitted. The model still produces gradeable code (10/12 partial), so it
isn't broken — but thinking is **not selective**: it burns ~120 latent-think
forwards per rollout regardless of whether thinking helps. This matches the
documented "gate temperature-fragility" (healthy think_rate≈0.30 at τ=0, bimodal
collapse to ~0 or ~1 at τ>0).

(Side result: **`max_gen` counts EMIT tokens only** — thinking has its own
`total_think_budget` — so 384 emit tokens is generous, 4× the validated 16/164
runs' 96. Generation length is NOT the bottleneck; gate selectivity is.)

## Why it collapses (root cause)

1. **No per-position "does thinking help here" teacher.** In pretrain the gate is
   supervised only by the entropy-aux target `exp(-H/T)` — a proxy that says
   "think where uncertain." But *uncertainty ≠ thinking-helps*: much uncertainty
   is irreducible (a think won't fix it), and some confident positions ARE
   improvable. So the proxy is wrong, and RL's trajectory-level reward is far too
   sparse to teach per-position selectivity.
2. **OOD under sampling.** The gate head was fit on τ=0 hidden states; at τ=0.7
   the hidden distribution drifts off-manifold → σ(gate) ≈ noise → snaps to a
   corner (here, always-think).
3. **Latent thinking is "free" w.r.t. correctness.** State-readonly latent think
   (β=0) *cannot* corrupt the recurrent state — that's its whole point — so a
   think is at worst neutral for the prediction. With no correctness penalty and
   a tiny ponder cost, the model is indifferent and drifts to maximal thinking.

## The fix — three layers; the calibration teacher is the core

"Think only where helpful" requires a **per-position signal of where thinking
helps**. RL alone can't provide it (sparse reward); it has to be supervised.

### Layer 1 (CORE, missing): execution/prediction-grounded gate-calibration loss
For a sampled fraction of positions, run an extra forward that performs **R
latent-think steps** (the validated Coconut/state-readonly mechanism) and measure
```
Δlogp = logp_after_R_latent_thinks(true_token) − logp_no_think(true_token)
target = 1{ Δlogp > margin }      # margin>0 ⇒ only "meaningfully helps"
loss   = BCE_with_logits(gate_logit, target)
```
This is **symmetric**: σ↑ where a think improves the true-token logp, σ↓ where it
doesn't. It is the direct embodiment of "only where helpful" and gives the gate a
**dense per-position target** that RL cannot. This is the removed
`compute_gate_calibration_loss`, re-adapted so the extra forward is R *latent*
think steps (not K discrete `[THINK]` tokens). Co-train it in **SFT** (and ideally
day-1 pretrain — inert features bolted on post-hoc never become load-bearing).
- The 2026-05-27 discrete-token smoke already showed the loss MOVES σ correctly
  (0.29→0.76 at under-thinking positions, Δlogp consistently +3…+8 at high-σ
  positions) — the trunk supports useful thinking, the gate just didn't know
  WHERE. So the mechanism is validated in principle; port it to latent think.
- Cost: ~2× forward on the sampled fraction only; bound via sample_frac ×
  max_positions.
- Known tension: competes with VAL for capacity at 287–600M (~3–5% VAL drift in
  the prior smoke). Bet: downstream evals that USE thinking gain more than VAL
  loses. Sweep the weight (0.025–0.05).

### Layer 2 (PREFERRED FIRST PATH — try before the costlier Layer 1): RL cut-short
`thinking.py::compute_grpo_advantages` with `--ponder_shape quadratic
--counterfactual --ponder_warmup_steps` (+ `--separate_ponder_norm`). Counterfactual
clamps task reward at the depth-0 baseline so a think can never *worsen* task
reward, but the depth cost ALWAYS applies — so RL keeps a think only when its
reward contribution beats the (growing) cost. Directly counters root-cause #3.

**The cheap, RL-native version of "only where helpful" (user idea, 2026-05-30):**
*cut the gate short* and let the reward difference teach selectivity. Cleanest
form = **within-group think-budget diversity**: give the N GRPO rollouts of a
problem DIFFERENT think budgets / cut-points, so the group-relative advantage
directly compares "thought less" vs "thought more" on the same problem. Cutting a
*useless* think → same task reward, lower ponder → higher reward → emit there;
cutting a *useful* think → task-reward drop outweighs ponder savings → keep
thinking there. The counterfactual reveals where thinking helps **for free** (no
extra forward, unlike Layer 1).
- **Caveat 1 — over-correction is a known killer.** Grader-RL v1 COLLAPSED at
  step ~350 when ponder cost bit: depth 120→30 took the output format down,
  reward→0. So cut-short needs the counterfactual clamp + warmup, NOT naive
  cutting.
- **Caveat 2 — coarse credit.** Reward is trajectory-level, so this teaches
  "think less globally" more sharply than fine-grained "here not there." It will
  cut overthinking; whether it reaches *selective* thinking is the open question.
- **So:** try Layer 2 first (cheap, infra mostly exists). If it plateaus at
  "less but not selective," add Layer 1 (the per-position supervised teacher).

### Layer 3 (have it): train the gate under sampling
`--stochastic_gate` (gate = Bernoulli RL action, optimized on the τ>0 hidden
distribution it actually faces) + `--gate_entropy_bonus` (prevents corner
collapse to never/always). Counters root-cause #2.

**Why all three:** L1 gives the correct per-position target (the missing teacher)
but only at the SFT distribution → can re-collapse in RL without L2+L3. L2+L3
(today's RL infra) keep it selective + robust but can't teach fine-grained WHERE
from sparse reward. They are complementary, not alternatives.

## Validation target
- `think_rate` 1.0 → selective **0.1–0.3**; `depth_mean` well below the budget
  ceiling; **reward / pass-rate holds or improves** (thinking concentrated where
  it pays).
- Diagnostic: post-training, σ(gate) should have high AUC against `1{Δlogp>0}`
  (gate fires ⇔ thinking helps). Re-add a `probe_gate_calibration` (latent-think
  version) to measure this.

## Plan (ready for the v8 SFT→RL phase) — cheap RL lever FIRST
1. **v8 RL with the cut-short / Layer-2 lever first** (`train_rl_grader.py`):
   `--ponder_shape quadratic --counterfactual --ponder_warmup_steps N` +
   `--stochastic_gate --gate_entropy_bonus`, `--gate_floor < --emit_threshold`.
   Add within-group think-budget diversity if plain ponder isn't selective
   enough. Watch for the v1-style collapse; keep the counterfactual clamp.
2. **Only if Layer 2 plateaus** at "less but not selective": re-implement
   `compute_gate_calibration_loss` for **latent** thinking (extra forward = R
   latent-think steps via `latent_think`), wire into `sft_code.py` via
   `--gate_calibration_weight`, add tests + the σ-vs-Δlogp diagnostic probe, and
   co-train in a gate-tuning SFT pass.
3. Confirm the validation target (think_rate 0.1–0.3, reward holds) on held-out
   MBPP/HumanEval.

## STANDARDIZATION (2026-05-30, later) — mechanism unified on LATENT

We caught that the gate-calibration LOSS and PROBE were both measuring "does
thinking help" with the DISCRETE-token append (`[THINK]*K`) — the validated
`mode="token"` baseline that does NOT help (0.09 vs 1.00). Calibrating the gate
against discrete thinking is why it learned to suppress thinking. Root cause:
the validated latent mechanism lived only in `latent_think.py` as a standalone
harness and was never wired into the production measurement/loss stack, so each
consumer re-implemented append-and-forward and they silently diverged.

Fix — ONE canonical primitive, every consumer imports it:
- `thinking.latent_think_logp(model, prefixes, true_next, *, R,
  thinking_token_id, pad_id)` — the single implementation of "run R
  state-readonly LATENT think steps, return logp(true_next) at the think slot"
  (refactored from the validated `latent_think.think_forward(mode="latent")`,
  `@no_grad` + `torch._dynamo.disable`).
- `gate_calibration.compute_gate_calibration_loss` now calls it (param renamed
  `K`→`latent_R`); the discrete `_post_think_logp` is deleted. SFT flag renamed
  `--gate_calibration_K`→`--gate_calibration_latent_R`.
- `probe_gate_calibration.py` ported to a latent path (reconciled to import the
  shared primitive).

So Layer-1's "re-implement for latent thinking" (step 2 above) is DONE. Whether
the latent calibration loss should be ENABLED on v8 depends on the probe below.

## PROBE VERDICT (2026-05-30) — thinking is UNPRODUCTIVE on the v8 trunk

`probe_gate_calibration.py --mechanism {discrete,latent}` on
`pretrain_v8_wide_step7630_tok2000158720.pt` (N=4000, WM on unless noted):

| mechanism | % helps (Δ>0) | mean Δlogp | linear-probe AUC | gate σ-vs-y AUC |
|---|---|---|---|---|
| discrete K=4 | 6.6% | -3.78 | 0.647 | 0.376 |
| latent R=1 | 4.1% | -4.84 | 0.661 | 0.403 |
| latent R=2 | 2.2% | -6.98 | 0.633 | 0.386 |
| latent R=4 | 2.0% | -7.61 | 0.602 | 0.355 |
| latent R=8 | 2.0% | -7.70 | 0.600 | 0.362 |
| latent R=4, WM-off | 4.9% | -4.48 | 0.703 | 0.326 |

**Neither mechanism helps the v8 trunk.** Latent thinking HURTS next-token
logp at every R and *monotonically worse with depth* — the exact inverse of the
validated from-scratch result (where acc rises with R≈K). v8 never co-trained
the latent-feedback mechanism, so feeding hidden states back is pure
distribution-shift corruption. WM-off confirms the harm is trunk-intrinsic, not
WM-injection noise. The current gate is mildly ANTI-selective (σ-vs-y AUC 0.36
< 0.5).

**Strategic consequence.** "Selective thinking at beneficial locations"
presupposes beneficial locations exist; on the v8 pretrain trunk they basically
don't (next-token-wise). So gate calibration is NOT the v8 bottleneck — *making
thinking productive at all* is. Two implications:
1. **Do NOT enable the gate-calibration loss on v8 SFT for either mechanism** —
   there is no useful "thinking helps here" target on this trunk. (The 2026-05-27
   smoke that showed Δlogp +3..+8 was on a ckpt being actively co-trained with
   process-reward; that productivity has to be TRAINED IN, it isn't free.)
2. **Latent thinking must be CO-TRAINED from day 1** (v9 / a thinking-SFT pass),
   not bolted onto v8 post-hoc. This requires wiring the latent mechanism + its
   training signal into the pretrain/SFT loss — the prerequisite recorded in
   CLAUDE.md's deprecation-cleanup TODO.
3. **RL is the exception**: grader-RL rewards working CODE, not next-token logp,
   so it can make thinking *instrumentally* useful even where the probe says it
   doesn't help next-token. The prior arc reached 16/164 this way. So the v8 →
   SFT → grader-RL path still stands; what's ruled out is the *next-token
   gate-calibration teacher* on v8.
