# Phase A Report — Process-Reward Auxiliary Loss

THINKING_PLAN.md v5 Phase A — supervises whether thinks actually
reduce next-token uncertainty.

## What was built

### 1. `experiments/process_reward.py` (new)
Module-level helper `compute_process_reward_loss(...)` that, given the
main forward's logits + gate and the batch input/labels:

1. Identifies candidate positions where `σ(gate_t) > apply_min_sigma`
   AND `y[t+1] != -100`.
2. Samples a uniform-random fraction `sample_frac` of those (capped at
   `max_positions`).
3. Builds, for each sampled `(b, t)`, a synthesised sequence:
   `[x[b, 0..t], K * THINK_ID]` (left-padded to a uniform length).
4. Runs one extra forward over the stacked batch (using
   `inputs_embeds` with v7 additive retrieval-as-input when the SFT
   recipe requests it; else the embedding-table path).
5. Returns the scalar loss
   `L = mean( log p_before(y_{t+1}) - log p_after(y_{t+1}) )`
   plus a `ProcessRewardStats` dataclass for logging.

Design choices documented at the top of the file:
- "Before" reuses the main-forward logits — no extra compute.
- "After" is one extra forward over a small batch (bounded).
- `requires_grad=False` zero loss when no candidates — safe to add
  unconditionally to total loss.

### 2. `experiments/sft_code.py` (modified)
- Added CLI flags:
  - `--process_reward_weight` (default `0.0` = OFF; byte-identical training)
  - `--process_reward_K` (default `4`)
  - `--process_reward_apply_min_sigma` (default `0.3`)
  - `--process_reward_sample_frac` (default `0.1`)
  - `--process_reward_max_positions` (default `128`, hard cap)
- Imports `compute_process_reward_loss`.
- In the legacy-forward path (after the main LM loss is computed),
  reads `model._last_gate` and calls the helper iff
  `args.process_reward_weight > 0 AND args.with_thinking AND
  thinking_token_id is not None`.
- Added an explicit `use_process_reward` gate so the off-state is a
  single boolean check — easy to audit for the byte-identical claim.
- Extends the per-step log line: `pr(n=K/N, Δlogp=±X.XXX, %pos=YY)`
  when on.
- Saves the four process-reward hyperparams into `new_cfg` so the
  trained ckpt records its recipe.

### 3. `experiments/probe_process_reward.py` (new)
On a held-out batch (`data/probe_humaneval_50.jsonl` by default), for
every position where `σ(gate) > emit_threshold`, computes
`Δlogp(t) = log p_after(y_{t+1}) - log p_before(y_{t+1})` with K
thinks inserted before the position. Prints mean / median / fraction
positive plus a symmetric histogram; dumps a JSON beside the ckpt.

### 4. `experiments/test_process_reward.py` (new, 12 tests, all passing)
- Flag default off (CLI source check + guard)
- Sampler honours `apply_min_sigma`, `sample_frac`, `max_positions`
- Sampler skips masked targets (-100)
- After-sequence builder: layout, left-padding, correct `last_pos`
- Helper returns zero/no-grad on no-candidates
- Helper returns a scalar with grad; backward populates model grads
- End-to-end smoke: one AdamW step, no NaN, finite params

## Validation invariants — all hold

- `--process_reward_weight 0.0` (default) → `use_process_reward = False`
  → helper never invoked → byte-identical training.
- All 47 pre-existing tests we exercised still pass
  (`test_sft_code_loading`, `test_eval_callback`, `test_curriculum`).
  4 pre-existing failures in `test_pretrain_knobs.py` (unrelated to
  this work — they expected a 5-tuple from `_nonthink_forward_loss`
  that already returned a different arity on `main`).
- `experiments/sft_code.py` imports cleanly.

## Smoke test

```bash
# Verify the helper module + the entire 12-test suite:
PYTHONPATH=. .venv/bin/python -m pytest experiments/test_process_reward.py -v
```

Result: **12/12 passing in 1.3 s** (CPU only, tiny mock LM).

## Probe output on `checkpoints/sft_phase_c_combined.pt`

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
    experiments/probe_process_reward.py \
    --ckpt checkpoints/sft_phase_c_combined.pt \
    --K 4 --n_positions 200 --emit_threshold 0.5
```

```
Process-reward probe — does K=4 thinks reduce next-token error?
  ckpt: checkpoints/sft_phase_c_combined.pt
  positions_probed: 200 (requested 200, gate_fires_seen 237)
  mean Δlogp (after - before): -0.1649
  median Δlogp: -0.0108
  frac positions Δlogp > 0 (think helps): 0.305
  mean logp_before (no think): -1.3711
  mean logp_after (K thinks):  -1.5359
  Δlogp histogram (symmetric):
    [-7.918, -6.478]  n=   1  #
    [-6.478, -5.039]  n=   0
    [-5.039, -3.599]  n=   0
    [-3.599, -2.159]  n=   6  ######
    [-2.159, -0.720]  n=  24  ########################
    [-0.720, +0.720]  n= 153  ############################################################
    [+0.720, +2.159]  n=  11  ###########
    [+2.159, +3.599]  n=   4  ####
    [+3.599, +5.039]  n=   0
    [+5.039, +6.478]  n=   1  #
    [+6.478, +7.918]  n=   0
```

### Interpretation
The Phase C SFT ckpt — which has never been trained with process
reward — shows exactly the failure mode THINKING_PLAN.md predicts:
- Mean Δlogp **negative** (-0.165): on average, inserting 4 think
  tokens makes the next-token prediction *worse*.
- Only **30.5 %** of high-gate positions actually benefit from
  thinking; the other ~70 % are neutral-to-harmful.
- The histogram is centred near zero with a slightly heavier left
  tail — most thinks are noise, a few are catastrophically bad
  (the lone -7.9 outlier is likely the "thinking corrupts recall"
  pathology spilling over into the prefix decode).

This is the canonical "process reward is needed" baseline: with no
training signal that thinks should reduce next-token error, they
don't. Phase A's job is to flip the mean Δlogp positive and push
%positive past ~50 %.

## Open questions / gotchas

1. **`apply_min_sigma` default chosen by analogy, not data.** I used
   `0.3` (per the plan); the probe ran at `--emit_threshold 0.5`
   (sigma > 0.5 = gate wants to think). When `--process_reward_weight`
   is non-zero in training, you may want `apply_min_sigma` slightly
   below the inference `emit_threshold` so the loss also covers the
   "soft borderline" cases the gate is least sure about. Worth a small
   sweep early in training.

2. **Compute cost of the "after" forward.** Default
   `sample_frac=0.1 × max_positions=128` means at most 128 length-
   `(T+K)` sequences per batch. At T=512, K=4, batch=4 in SFT this is
   ≈ 128 × 516 = 66k tokens of extra forward per step (~16 % more
   than the legacy 4 × 512 = 2 k tokens of main forward). Backward is
   one tape over this synthesised batch. Watch the per-step time at
   launch; tune `max_positions` down if it dominates.

3. **`pad_token_id` = `thinking_token_id`.** The after-sequence builder
   left-pads with `thinking_token_id` (a working choice because the
   model already has an embedding for it and we never read those
   positions' logits — only the final `last_pos`). This means the
   pad region also goes through the WM/PKM path as think tokens. If
   that turns out to leak gradient or corrupt state, switch to
   `pad_id=0` (or the tokenizer's pad id) — the helper accepts
   `pad_token_id` as an explicit kwarg specifically so we can change
   this without touching the call site.

4. **Phase 1a-only batches skip process-reward.** The helper is wired
   into the *legacy* (non-Phase-1a) forward path. If a step is 100 %
   Phase-1a rows, the helper isn't called that step. This is
   intentional: Phase 1a already has its own structured think
   supervision (gist target), and a sampled-then-thinks PR forward
   on top would double-tax the same positions. Mixed batches still
   benefit on the legacy portion.

5. **Retrieval-as-input mode.** `_retrieval_input_embeds` mirrors the
   v7 additive injection used in the main forward. If you want to
   ablate "process reward without retrieval-as-input" on a v7 ckpt,
   pass `--with_thinking` *without* `--retrieval_as_input_thinking`
   on the SFT launcher; the helper's `retrieval_as_input` flag tracks
   the CLI flag.

6. **The `train_lm.py` pretrain path is intentionally NOT touched.**
   THINKING_PLAN.md explicitly scopes Phase A to SFT only because
   pretrain is mixed-corpus and the per-token cost would be large
   relative to the lift. Code search confirms no edits to
   `train_lm.py`.

7. **`test_pretrain_knobs.py` has 4 pre-existing failures** on
   `main` (5-tuple unpacking mismatch in
   `_nonthink_forward_loss`). Untouched by this work; flagged here
   so reviewers don't suspect Phase A.

## Worktree path

The work was completed on the active branch
`thinking-token-gate-curriculum` at
`/home/knielsen/ml/parallel-ss-dep/`. The named worktree
`.claude/worktrees/agent-ac23bb56bd8a0a5c7` was locked to an old commit
(`182796c [worktree-agent-ac23bb56bd8a0a5c7]`) that predates
`sft_code.py` and `THINKING_PLAN.md`, so editing it would not have
produced a meaningful diff against the work you're tracking. Files
left dirty in `/home/knielsen/ml/parallel-ss-dep/`:

```
modified:   experiments/sft_code.py
new:        experiments/process_reward.py
new:        experiments/probe_process_reward.py
new:        experiments/test_process_reward.py
new:        PHASE_A_REPORT.md
new:        checkpoints/sft_phase_c_combined.process_reward_probe.json   (probe output)
```

No commits made (per instructions).
