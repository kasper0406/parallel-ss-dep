# THINKING Probe Results — Phase D ckpt

Ran 2026-05-26 against `checkpoints/pretrain_phase_d.pt` (Phase D pretrain,
8.25 B tokens, FIM + synth pyfunc + self-debug mix).

## Headline

**Thinking is currently HARMFUL.** All three probes point the same way:
the mechanism is firing but produces noise — it INCREASES next-token
uncertainty, contributes ZERO problems on the HumanEval probe, and is
NEGATIVELY correlated with rollout reward.

## Probe 1 — Per-position CE delta

For each position where the gate elects to think (gate output below
emit_threshold), measure next-token CE with vs without one think token.

| metric | value |
|---|---|
| positions probed | 200 |
| mean Δce (without − with) | **−0.19 nats** (negative → think HURTS) |
| frac positions where think helps | **36.5%** |
| mean CE without think | 4.45 |
| mean CE with think | **4.64** |

Distribution is centered around zero with a LONG NEGATIVE TAIL: a
non-trivial fraction of positions get +1 to +3 nats WORSE when the
model is forced to think.

**Reading**: the mechanism doesn't reduce next-token uncertainty at the
positions it elects to fire. Think tokens are net-noise injection into
the recurrence.

## Probe 2 — Counterfactual HumanEval (50-problem probe slice)

| condition | n_passed | mean_emit | mean_think |
|---|---|---|---|
| think_budget=0 | 0/50 | 256 | 0 |
| think_budget=120 | 0/50 | 256 | 75.3 |

Both zeros at this pretrain-only stage (consistent with CLAUDE.md
history: pretrain ckpts always 0 on HumanEval). The mean_think=75.3 (of
120 budget) shows the gate IS firing meaningfully on hard
problems — it's just not adding value.

**Reading**: can't discriminate at this stage. Re-probe post-SFT.

## Probe 3 — RL rollout correlation

240 rollouts at T=0.7 across 30 mbpp_combined problems, 8 rollouts/problem.

- Spearman ρ(n_think_tokens, reward): **−0.17** (negative correlation)
- 231/240 rollouts had 0 think tokens (gate seldom fires in code generation)
- 9/240 rollouts with 1-30 thinks: mean reward **0.044** vs no-think 0.181

**Reading**: when the gate DOES fire during rollout, it's anti-correlated
with success. Thinking actively REDUCES the model's ability to solve
problems on this ckpt.

## What this tells us about the plan

The "mechanism works mechanically and we just need better training
signal" assumption is **WRONG**. The mechanism is broken — thinking
injects noise. This re-prioritizes the plan:

1. **Phase 4 (CoT distillation SFT)** is now essential. Without
   demonstrations of thinking-that-actually-helps, the model has
   no signal to learn good thinking from.
2. **Phase 5 (process-aware reward)** becomes MORE important. We need
   explicit gradient that thinking should reduce CE / improve reward.
3. **Phase 2 (state-readonly)** is still good (preserves long-range
   recall by not letting thinks corrupt the recurrence) — and the
   recurrence-corruption finding actually EXPLAINS part of the +0.19
   CE penalty.
4. **Phase 3 (think index emb)** is still useful for breaking the
   homogenization, but won't fix the noise injection.

The next sequence in `run_thinking_pipeline.sh` (CoT distill → SFT-with-
thinking → HumanEval → long-context recall) will tell us whether SFT
on demonstrations can overcome the negative baseline. If it can: Phase
5 is worth building. If it can't: we need a different mechanism
entirely.

## Next probe

Re-run all three probes on `checkpoints/sft_phase_d_cot_thinking.pt`
(produced by step 4 of the pipeline) to see if SFT with thinking
demonstrations flips the sign of any probe.
