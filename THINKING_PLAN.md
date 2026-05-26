# Make EFFICIENT Thinking Work — Discovery-RL v4 (Stability Edition)

Fourth iteration. v3 (discovery RL with stochastic gate) was the right
shape but day-1 evaluation revealed a stability gap: the SFT base
trained with a deterministic gate produces DECISIVE σ values (cluster
near 0 or 1). Sampling at those positions disrupts model behavior at
~5-10% of positions where σ wanders into [0.4, 0.6], degrading output
quality, washing out reward signal, and preventing the gate distribution
from evolving.

## What we know from yesterday

- Stochastic gate **mechanism** works (PPO ratio, log-probs, entropy
  reg all numerically correct; 10/10 tests pass)
- Gate fire rate stayed bounded (0.25-0.37) on 100 steps of mbpp RL
  — entropy reg prevents collapse to 0 or 1
- Gate fire rate did NOT evolve — reward gradient through gate decisions
  isn't moving the gate distribution
- One lucky pass at step 25; reward bounced 0.0-0.09; output mostly
  syntax errors
- Root cause: SFT base has σ clustered near 0 / 1; sampling at the
  uncertain middle disrupts trained behavior at the few positions
  where the gate IS uncertain

## v4 strategy: stability first, exploration second

Three intervention layers stacked from cheapest to most invasive:

### Phase A — Selective stochastic gate (cheap, the load-bearing fix)

Only Bernoulli-sample at positions where the gate is GENUINELY
uncertain. Decisive positions (σ > 0.9 or σ < 0.1) stay deterministic
threshold. Two CLI flags:
- `--gate_sample_range_low` (default 0.1)
- `--gate_sample_range_high` (default 0.9)

This gives:
- Stable output at decisive positions (preserves trained quality)
- Real exploration at uncertain positions (where model is undecided
  and could benefit from gate-direction gradient)
- PPO ratio operates ONLY on sampled decisions (not on the determined
  ones — those didn't have a choice)

Expected effect: stronger reward gradient because output quality stays
high, gate distribution can actually evolve at the uncertain positions.

### Phase B — Entropy curriculum (cheap, additive)

Schedule the entropy bonus to start high (encourage exploration) and
decay over training:
- Step 0:   `--gate_entropy_bonus 0.05` (strong exploration push)
- Step 100: 0.01 (medium)
- Step 200+: 0.001 (let policy converge)

Linear schedule keyed off step number.

### Phase C — Stochastic-aware mini-SFT (only if A+B insufficient)

If selective sampling + entropy curriculum still doesn't move the gate
distribution after 200 steps, we need to teach the base model to be
robust to gate sampling. This is a short SFT phase where:
- Each training example has the gate Bernoulli-sampled during forward
  (with the model's current gate predictions)
- Standard LM loss on solution tokens regardless of think positions
- ~500-1000 steps on existing SFT data

After this, the model is robust to sampled gate at any position; RL
exploration becomes safe.

### Phase D — Continuous thoughts (research bet, reserved)

If A + B + C don't deliver, the binary gate may be too coarse. Pivot
to: gate outputs a continuous thought VECTOR injected into WM at each
position. Much richer action space; bigger build. Only after binary
exploration is exhausted.

## Diagnostics to add

To know WHAT's happening during Phase A:
1. **Gate σ distribution histogram** per step: how many positions are
   decisive (σ > 0.9 or < 0.1) vs uncertain ([0.1, 0.9])
2. **Fraction of positions actually sampled** (vs deterministic)
3. **Per-position correlation**: did the sampled-think positions
   correlate with passing rollouts?

Without these, we won't know if Phase A is firing the right number of
sampling decisions or being too restrictive / permissive.

## What we are NOT doing

- Pre-CoT-imitation SFT (v1 / v2 plans) — degraded the model in every variant
- Latent thought vectors yet — bigger commitment, reserved as Phase D
- Modifying the gate's output magnitude (gate temperature) — would break
  inference compatibility with existing ckpts

## Execution order

1. **Phase A** + diagnostics (build, smoke, launch on mbpp_combined)
2. Watch over 200 steps: gate fire evolve? reward trend up?
3. If yes → ship + try synth_reasoning post-SFT-bootstrap
4. If no → **Phase B** entropy curriculum on top of A
5. If still no → **Phase C** stochastic-aware mini-SFT
6. If still no → **Phase D** continuous thoughts (escalation)

## Decision gates

- Phase A success: gate fire rate evolves > 0.05 from initial AND
  pass_n grows OR partial-credit rate grows OR `--gate_sample_range`
  threshold matters demonstrably
- Phase A failure: 200 steps with no gate evolution AND no reward
  improvement → escalate to B

## Decisions log

See `THINKING_DECISIONS.md` for judgment calls during execution.
