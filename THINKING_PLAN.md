# Make EFFICIENT Thinking Work — Discovery-RL Plan (v3)

Top priority. Third iteration of this plan. v1 was "train the gate to
fire correctly via SFT"; v2 was "compress CoT via gist supervision at
think positions". Both shared a hidden assumption — that we should
TEACH the model some predetermined thinking pattern (Qwen's text CoT
in v1, Qwen's CoT compressed by K in v2). The empirical results say
both approaches DEGRADE the model (every CoT-thinking SFT route dropped
HumanEval from 7-8/164 to 0/164).

v3 turns this on its head: the optimal thinking pattern is UNIQUE to
our architecture (FiLM K=3 + WM + retrieval-as-input). We don't know
what it should look like. We must let the model DISCOVER it through
exploration, rewarded by what actually works.

## Root insight

LM loss and SFT can't teach the gate when to fire because:
- They have no notion of "thinking helps" — only of "predict the next
  token correctly"
- Imitating Qwen's CoT teaches our model to use thinking the way Qwen
  uses CoT, which destroys the architectural compression promise

RL with a STOCHASTIC GATE is the natural fit:
- Gate output σ(g_t) becomes a Bernoulli probability over emit-vs-think
- Each rollout SAMPLES different thinking patterns
- PPO attributes reward back to each gate decision via standard ratio
- Compression EMERGES — if thinking doesn't help, policy converges to
  gate=0 (no thinking); if thinking helps, gate=1 at the right
  positions
- The model finds its OWN optimal thinking pattern, not Qwen's

## Phases

### Phase A — Stochastic gate as policy variable

Implementation in `experiments/train_rl_grader.py`:
1. New CLI flag `--stochastic_gate` (default off, backwards-compat)
2. In `rollout_group_batched`: when stochastic_gate is on, sample
   Bernoulli(p=σ(gate)) at each position instead of threshold-deciding
3. Capture per-position `gate_log_prob = log p(emit_decision_made)`
   in the `Rollout` dataclass
4. In `policy_loss_for_rollouts_batched`: extract new-policy gate
   log-probs from the policy forward, compute PPO ratio
5. TOTAL surrogate = mean(emit_surr + gate_surr)
6. Force-emit positions (think budget exhausted etc.) excluded from
   gate gradient — the model didn't have a choice there

### Phase B — Entropy regularization on gate

PPO with binary actions collapses without entropy reg.
- Compute per-position gate entropy `H = -(p log p + (1-p) log(1-p))`
- Subtract `gate_entropy_bonus * mean_entropy` from total loss
- Default: 0.01. Decay schedule TBD based on empirical run behavior.

### Phase C — Train on tasks where thinking IS the difference

Run RL on `synth_reasoning` (the 6-family curriculum we built earlier):
- Multi-step arithmetic chains
- Conditional rule application  
- Counting with offset
- Binary search trace
- Stack machine eval
- Pattern detection (arithmetic / geometric / fibonacci / modular)

These STRUCTURALLY require multi-step reasoning. No-think attempts
should fail. Thinking has clear room to help. The reward signal is
clean: "did the answer match the test?".

mbpp_combined was the wrong RL target — too many problems are
single-shot solvable, the gate never had a reason to fire (231/240
v11c rollouts had 0 thinks).

### Phase D — Token-level reward attribution (if Phase A-C insufficient)

If the global episode reward proves too noisy to teach the gate
properly:
- Per-position counterfactual: for each gate=think decision, re-roll
  the rollout WITHOUT that think; measure Δ pass-rate
- Reward = global pass + α × sum(positive_per_position_deltas)
- Provides finer-grained gradient than monolithic episode reward
- Expensive (extra rollouts) — only enable if Phase A-C don't converge

### Phase E — Stability fixes for known failure modes

Carrying forward from previous plans:
- **State-readonly thinking** (`--state_readonly_at_think`, shipped):
  prevents recurrence corruption from multi-think bursts
- **Think index embedding** (`--think_index_emb_size`, shipped): breaks
  multi-think homogenization
- **KL anchor** (`--kl_coef`): prevents drift from SFT base
- **Adaptive curriculum**: keep on; samples from variance-bearing
  problems

All of these are already-built knobs we keep ON.

## What we are NOT doing anymore

- **CoT imitation SFT** (v1's Design A or v2's gist supervision):
  empirically degrades the model regardless of base / mix
- **Manually-specified think budgets**: budget is a hard upper bound,
  not a target
- **Loss-masked think padding**: provides no signal

## Execution order

1. Build stochastic gate (Phase A) + entropy reg (Phase B) — landed
   as `experiments/test_stochastic_gate.py` + `train_rl_grader.py`
   changes
2. Scale synth_reasoning data (3000 train + 300 held-out) — landed
   as `data/synth_reasoning_{train,heldout}.jsonl`
3. Run first stochastic-gate RL on synth_reasoning — gate_fire_rate
   should evolve from ~0.5 (entropy-regularized init) toward whatever
   pattern wins
4. Eval on synth_reasoning held-out (or HumanEval if applicable) —
   compare to no-think baseline
5. Decision gate: if gate converges to non-trivial pattern AND lifts
   eval pass-rate → efficient thinking discovered. If gate collapses
   to 0 (never think) AND eval matches no-think → thinking doesn't
   help even with exploration; pivot away. If gate stays at 0.5 AND
   eval stable → exploration didn't find a useful pattern; try Phase D
   counterfactual reward.

## Decisions log

See `THINKING_DECISIONS.md` for judgment calls during execution.
