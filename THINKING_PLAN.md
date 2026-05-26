# Make EFFICIENT Thinking Work — Pivoted Plan

Top priority. Replaces the previous version of this file, which was a
plan to "train the gate to fire correctly". Phase 1 probes
(`THINKING_PROBE_RESULTS.md`) and the user's design critique made clear
that plan didn't actually deliver on the architecture's promise.

## The architectural promise (and why the old plan didn't test it)

Our thinking mechanism is supposed to be **more token-efficient than
text CoT**, because:

1. Continuous representations (no word-bottleneck round-trip)
2. WM gives explicit lookback over previous think states
3. FiLM K=3 self-feed → each forward is effectively 3 trunk passes
4. So 1 think token should do the work of K text CoT tokens (K≈5-20)

**The old plan's Design A** trained the model to replace `N_cot` tokens
of Qwen's verbose CoT with `N_think = N_cot` think tokens, all
loss-masked. That's literally "1 think per 1 CoT word" — the opposite
of compression. And the think positions had NO gradient signal, so
the model had no way to learn what to compute during a think.

**Result of pursuing it**: Phase D + Design-A-only SFT scored 0/164 on
HumanEval. Phase 1 probes on the Phase D ckpt also showed thinking
currently INCREASES next-token CE (+0.19 nats) and is anti-correlated
with RL rollout reward (Spearman −0.17). The mechanism is currently
**actively harmful** — it injects noise into the recurrence.

## What the pivot has to deliver

Three properties the new plan must hit:

1. **Think positions get rich gradient signal.** Not just "emit nothing
   here" (loss-masked). The model must learn what to compute during
   a think.
2. **N_think << N_cot.** Compression is the architectural promise; if
   we don't test it, we're not testing the architecture.
3. **Validated against a text-CoT baseline.** If text CoT doesn't help
   our model at all, the issue is scale/capacity, not the thinking
   mechanism, and we should stop investing in it.

## What we have to drop

- **Design A as currently built** (loss-masked think padding). It
  doesn't deliver any of the three properties.
- **"Train the gate to fire correctly"** as the primary objective. Gate
  firing without content-rich thinks is useless; gate-supervision is a
  side-quest, not the main lever.

## Phases (pivoted)

### Phase 0 — Establish the text-CoT baseline (1 day, MUST go first)

Run Phase D + standard Qwen distillation SFT (i.e., target =
`qwen_completion` which IS the CoT prose + code, no think tokens).
The model emits CoT text then code, exactly like a standard LLM.

This answers: **does CoT help our model AT ALL?** If pass@1 ≥ Phase C
SFT's 10/164, CoT helps. If it's still 0-3/164, the issue is model
scale and the thinking mechanism cannot save us. Decision gate for
the rest of the plan.

Infra: existing `launch_sft_phase_d_mixed.sh` (the 54k Qwen distill +
961 CoT-thinking mix already shipped). Run that. The CoT-thinking rows
are minor noise vs the 54k Qwen rows. Expect pass@1 = 8-12/164 if
Qwen CoT helps; 0-3/164 if not.

**Output**: `runs/eval_humaneval_sft_phase_d_mixed.log` + decision in
`THINKING_DECISIONS.md`.

### Phase 1 — Compressed CoT via gist supervision at think positions

The real fix for "think positions have no gradient". At each think
position, ADD a gist loss: predict the hidden state of the CoT teacher
at position `t · K` (K = compression factor, target 5-10).

Architecture:
- Existing `gist_loss_weight` infra (multi-horizon trunk gist) already
  predicts `mean(h[t+1:t+K])` from `h[t]`. Repurpose it at think
  positions to predict the TEACHER's hidden state at the corresponding
  CoT chunk.
- Or simpler: teacher is the SAME model's forward over the full
  CoT prose (frozen ckpt); student forward inserts thinks at every
  K-th position and the gist loss targets the teacher's hidden state
  at position `t · K`.

This gives think positions explicit gradient: their job is to "be" a
compressed version of K CoT tokens. The architectural claim is that
WM + FiLM K=3 + retrieval-as-input give the student enough capacity
per think to fit K tokens of compressed reasoning.

Two variants to try:
- **1a**: hidden-state target (continuous, what gist_loss already does)
- **1b**: next-CoT-chunk-EMBEDDING target (input-level supervision via
  retrieval-as-input — Design C from the old plan)

Start with 1a (lighter-weight; reuses shipped infra).

**Validation**: train SFT with compression K=5; if think-position CE
goes DOWN over training AND HumanEval matches Phase 0 baseline → we
have efficient thinking. If think-position CE stays high → escalate to
1b.

### Phase 2 — State-readonly preserved + per-think index embedding

Already shipped (previous Phase 2 + 3). Keep them ON for all new
training runs:
- `--state_readonly_at_think`: prevents recurrence corruption (+0.465
  on synthetic recall probe)
- `--think_index_emb_size 8`: breaks the multi-think homogenization

These are prerequisites for Phases 1 and 3; they're not the lever.

### Phase 3 — RL with compression-aware reward

After Phase 1 establishes that the model CAN compress CoT into thinks,
RL can reward COMPRESSED productive thinking:
- Reward = grader_pass − α · n_think (linear ponder cost on absolute
  think count)
- AND a counterfactual-CE bonus: when a think position lowered the
  next-emit CE more than the no-think baseline, that think gets +reward
- Bound: total reward stays in [0, 1] so PPO behaves

This is `THINKING_PLAN`'s old Phase 5, now positioned correctly (after
the model has the capacity to think productively).

### Phase 4 — Synthetic reasoning curriculum (shipped, queued for use)

The 6-family reasoning task set is built (`gen_synthetic_reasoning_
tasks.py`, 504 tasks). Use it as RL training data IN PHASE 3 (after
Phase 1) — these tasks structurally require multi-step reasoning, so
they're the right test for "did efficient thinking work?".

## Execution order

1. **Phase 0** (NOW): launch `launch_sft_phase_d_mixed.sh`, wait for
   HumanEval. Decision gate.
2. **If Phase 0 pass@1 ≥ 8**: build & run Phase 1a (gist-at-think
   supervision with K=5). Compare HumanEval against Phase 0.
3. **If Phase 1a HumanEval ≈ Phase 0 with N_think = N_cot/5**: ship.
   This means we have efficient thinking. Proceed to Phase 3 RL.
4. **If Phase 1a HumanEval << Phase 0**: escalate to Phase 1b (CoT
   retrieval-as-input). If that also fails, the architecture cannot
   support compression at this scale; pivot to scale-up.

## What we will NOT do

- Lossy-masked think padding with N_think = N_cot (the old Design A).
  It's been tested — doesn't work.
- More gate-supervision aux losses without process signal (cycled
  through this; never moved the needle).
- Pure thinking-token-count optimization (depth_mean targeting).

## Decisions log

Important judgment calls during execution are logged to
`THINKING_DECISIONS.md` for user review.
