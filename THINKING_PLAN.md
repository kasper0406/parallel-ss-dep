# Make Thinking Actually Useful — v5 (Mechanism + Supervision)

Fifth iteration. v3/v4 (discovery RL with stochastic gate ± entropy curriculum)
gave us a clean negative diagnosis:

- v4 step-200 HumanEval: **12/164** (vs SFT base 6/164, project best 16/164)
- SFT base at `emit_threshold=0.5`: **think_rate=0.629** (already thinks a lot)
- SFT base at `emit_threshold=0.6`: **9/164, think_rate=0.640** (+3 free)
- thresholds 0.6/0.7/0.8 all give 9/164 — calibration saturates immediately

**The bottleneck is not how often the model thinks** (it already thinks 60+%
of the time). The bottleneck is that **most thinks aren't productive**:
the architecture has nothing specialized to do at think positions, and
nothing in training has ever supervised "did this think actually help?"

## Corrected understanding of train vs eval semantics

At **training time**, the gate is *not* a soft mixture of emit-vs-think
outputs. It is a soft loss weight:
```
loss = g · CE_real_token + (1-g) · λ_ponder
```
Every position outputs a real next-token prediction; no "think path" is
ever computed during training. The gate's only training signal is "how
much should real-token-mistake be penalized vs paying a static ponder
cost." Optionally augmented by entropy-aux loss (target = `exp(-H/T)`).

At **inference time**, the gate gates *compute*: if σ < threshold,
append a THINKING token, run another forward, repeat. The "think path"
only computes at inference.

**Implication**: the model has never seen the consequences of thinking on
its own predictions during training. It learns the gate predicts
uncertainty, not the gate makes good decisions about when to spend
compute. This v5 plan fixes that.

## Five workstreams (orthogonal, parallelizable)

### Phase A — Process reward (the missing supervision)

**The core idea**: thinks must reduce next-token uncertainty.

Add an auxiliary training-time loss that *actually computes* the think
path and measures whether thinking helped. For each emit position t in
the batch with the gate firing (σ(g_t) > τ_consider), compute two
distributions:
- `p_before(t+1)` — model's softmax over next token from a forward
  with zero thinks before position t
- `p_after(t+1)` — model's softmax over next token from a forward with
  K=4 inserted think positions before t

The auxiliary loss is:
```
L_process = mean_t [ max(0, H(p_after) - H(p_before)) ]
                                                     ^
                                            (positive if think INCREASED uncertainty)
```
or symmetric variant:
```
L_process = -mean_t [ H(p_before) - H(p_after) ] · g_t.detach()
```
Critically, this gives the model GRADIENT through the think computation
itself — the trunk weights and WM module learn to make thinks reduce
uncertainty.

**Variants to test:**
- (a) Entropy-reduction reward (above)
- (b) KL-divergence between p_before and p_after (only useful if KL goes
  in the right direction = lower H)
- (c) Direct correctness reward: `-log p_after(true_next_token) + log p_before(true_next_token)`. Cleaner: rewards thinks that put more probability mass on the actual next token.

**Implementation:**
- New CLI flag: `--process_reward_weight` (default 0.0 = off)
- New flag: `--process_reward_K` (default 4 — how many thinks to insert
  for the "after" forward)
- New flag: `--process_reward_apply_min_sigma` (default 0.3 — only apply
  loss where gate σ > this, to avoid wasting compute on positions the
  gate clearly didn't want to think at)
- Inside `train_lm.py::_nonthink_forward_loss` and `sft_code.py`:
  on a fraction of positions (default 10% sampled per batch), compute
  the "after K thinks" forward in addition to the normal forward and
  add `L_process` to total loss
- Cost: ~10% extra compute on selected positions (small batch, K=4 thinks)
- Goal at convergence: think positions provably reduce next-token entropy
  on held-out data

**Why this is load-bearing:**
- Direct supervision of the actual question "is thinking useful?"
- Trains the trunk + WM to make thinks productive (not just the gate)
- Works with any of the architectural fixes below

**Training strategy**: Apply during SFT only, not pretrain — pretrain
is mixed-corpus and the per-token compute overhead would be large.

**Evaluation strategy**:
1. Per-token diagnostic: `experiments/probe_process_reward.py` — on a
   held-out batch, compute H(p_after K thinks) - H(p_before) at every
   position where σ > 0.5. Histogram. Pre-training: ~symmetric around 0
   (think is noise). Post-training: skewed left (think reduces H).
2. HumanEval pass@1 — primary metric.
3. The σ at "useful" positions (large H reduction) vs "useless" — should
   become correlated as training proceeds.

### Phase B — Thinking-specialized parameters

**The core idea**: give the trunk dedicated weights that only fire at
think positions.

Add a small `ThinkAdapter` module — a 2-layer MLP `d_model → d_model →
d_model` (~10-20M params for d_model=896). When the model processes a
think position, the trunk's residual stream goes through:
```
h_post_attn = attention(h_pre)
h_with_adapter = h_post_attn + α_think · think_mask · ThinkAdapter(h_post_attn)
```
where `think_mask` is 1.0 at think positions, 0.0 elsewhere, and
`α_think` is a learned scalar (init 0 — byte-identical at start).

**Why**: right now the trunk runs identical computation at think vs
emit positions. The only difference between them is the input
embedding. Dedicated adapter params give the model the *capacity* to
develop think-time-specialized processing.

**Implementation:**
- New module in `experiments/model.py::ThinkAdapter` or in `Block`
- New flags: `--use_think_adapter`, `--think_adapter_hidden_mult` (default 2)
- Insert call in `Block.forward` after the attention/FLA path, gated
  by think_mask
- Adapter weights → AdamW (not Muon) — small + shape-irregular
- Initialized so that an existing ckpt loads byte-identical

**Evaluation strategy**:
- Per-layer adapter activation magnitude: `‖α · ThinkAdapter(h)‖ / ‖h‖`
  — should grow from 0 over training; if it stays at 0, the model
  judged the adapter useless
- HumanEval pass@1 vs no-adapter baseline (same SFT recipe otherwise)
- Long-context recall (`eval_longctx_recall.py`) — should not regress

### Phase C — Soft-mixture decode (eval-only, the cheap experiment)

**The core idea**: at decode, run both paths and mix by σ.

Currently inference uses a hard threshold: `if σ ≥ τ: emit, else think`.
Replace this with running both branches and mixing logits:
```
emit_branch:   forward through trunk → logits_emit
think_branch:  append think, forward through trunk → logits_think → forward to next position → logits_after_think
final logits = σ · logits_emit + (1-σ) · logits_after_think
sample / argmax final logits
```

This costs 2× decode compute per position but never makes a wrong hard
decision. Useful as a probe of "is the gate's continuous output more
informative than its thresholded version?"

**Implementation:**
- New flag in `eval_humaneval.py`: `--gate_mode {hard, soft}`
- New decode function `generate_soft_mixture(...)` — based on
  `generate_with_retrieval_as_input` but emits 2 branches per step
- Cost: 2× per step but no rollout/training change

**Evaluation strategy**: A/B on existing ckpts (SFT base, v4, RL v2)
with `--gate_mode hard` vs `--gate_mode soft`. If soft strictly wins,
gate's continuous output carries information the threshold throws away.

### Phase D — Bigger thoughts (iterative refinement per think)

**The core idea**: each think gets K extra trunk passes at the SAME
sequence position, instead of K sequential think tokens.

Currently a "burst of 4 thinks" appends 4 THINKING tokens, each running
one forward of the trunk, each stepping the DeltaNet recurrent state
forward 4 times. Linear-RNN state is corrupted 4× even though the
information content was supposed to be "think more about this one
position."

Replace with: when gate fires, do K iterations at the same position,
each iteration:
1. Read from WM with current hidden as query
2. Mix retrieval into current hidden: `h ← h + α_iter · retrieval`
3. Run trunk forward one more time on h
4. Repeat until either K reached or σ(gate) > emit_threshold

Final h becomes the position's hidden state; emit from it. The DeltaNet
recurrent state steps forward exactly once per emit (not per think).

**Why**: addresses the "thinking corrupts recall" finding (CLAUDE.md
long-context section). Also addresses think-position homogeneity (each
iteration sees a different retrieval-mixed input).

**Implementation:**
- New flag: `--bigger_thoughts_iterations` (default 0 = off, original
  per-token thinking)
- Modifies `Block.forward` and the gate-controlled inference loop
- This is the most architecturally invasive — keep BPTT path
  byte-identical when off
- Likely needs its own SFT pass to validate

**Evaluation strategy**:
- Long-context recall: should NOT degrade with thinks (the original
  motivation)
- HumanEval pass@1
- Per-think compute equivalence: K=4 iterations ≈ 4 think tokens in
  raw FLOPs, but only 1 RNN state step

### Phase E — Pure continuous thoughts (Phase D from v4 plan; biggest)

**The core idea**: drop the `[THINKING]` token entirely. When the gate
fires, inject a continuous d_model vector at the SAME position into the
residual stream, no new sequence position consumed.

What we have now (retrieval-as-input): `[THINKING]` token's input
embedding is `think_embed + α·retrieval`. STILL consumes a sequence
position. So we have continuous *input* but discrete *position*.

Pure continuous: gate fires → at the SAME position, add a continuous
vector `Adapter(h)` into the residual stream and re-run the trunk;
emit happens at the same position once gate σ exceeds threshold or K
iterations reached. No sequence position consumed per think.

This subsumes Phase D (Phase D is the case where the "continuous
vector" is a retrieval injection) but more generally lets a learned
adapter compute the injection. Different architectural commitment;
defer until A-D inform.

## Execution order

```
   Day 1-2: build A, B, C in parallel (3 agents)
            │
   Day 3:   review + commit all three
            │
   Day 4:   train recipe(s):
            • SFT v8 with Phase A active (process reward)
            • SFT v9 with Phase A + Phase B (process reward + adapter)
            • Eval all SFT ckpts with Phase C (soft mixture decode)
            │
   Day 5:   HumanEval + long-context recall + process-reward probe
            │
   Decision:
            if A+B+C lifts past 16/164 → ship, plan D for incremental
            if no lift → escalate to D (bigger thoughts)
            if even D doesn't move → E (pure continuous)
```

## Decision gates

- **Phase A success**: process-reward probe shows H reduction at >50%
  of think positions on held-out batch AND HumanEval >= 12/164
- **Phase B success**: adapter α grows from 0 to >0.05 AND HumanEval
  >= 13/164 (one problem over A alone)
- **Phase C success**: soft-mixture decode strictly beats hard threshold
  on existing v4 or v2 ckpts (no retraining needed)
- **Combined A+B+C target**: 17/164 (one problem over current best 16)

## What we are NOT doing in v5

- More entropy-bonus RL — v4 showed this saturates at ~12/164
- More gate-floor / threshold tweaks — calibration is exhausted at
  the SFT base
- CoT distillation of Qwen — toxified the model in earlier attempts

## Files touched per phase

- **A**: `train_lm.py`, `sft_code.py`, `train_lm_args.py`, new
  `experiments/probe_process_reward.py`, new tests
  `experiments/test_process_reward.py`
- **B**: `experiments/model.py` (new `ThinkAdapter` module + `Block`
  integration), `model_builder.py`, `eval_bracket_structure.py`
  (`build_model_from_ckpt` cfg auto-detect), new tests
- **C**: `experiments/eval_humaneval.py` (new `generate_soft_mixture`
  + `--gate_mode` flag), new test
- **D**: `experiments/model.py` (iteration loop in inference),
  `eval_humaneval.py`, sft launcher
- **E**: deferred until A-D land

## Validation invariants (must hold for every phase)

- Default flags (everything off) → byte-identical to current code
- Existing ckpts load without modification (cfg auto-detect)
- All existing tests pass
- New tests added for each new module and CLI flag
- Each new mechanism has a probe script that shows it's actually firing

## Documentation to update after each phase lands

- `CLAUDE.md` — operative flags, what's load-bearing, what's not
- `README.md` — headline results if any phase produces a measurable lift
- `NEXT_DIRECTIONS.md` — strategic queue
- `MILESTONE_ARCH.md` — ablation status
- This file — append results section per phase

## Decisions log

See `THINKING_DECISIONS.md` for judgment calls during execution.

---

## v5 build status (2026-05-26)

All three Phase A/B/C deliverables built, code-reviewed, and integrated.
Eval results pending — phases not yet exercised on a training run.

### Phase A — process reward (BUILT, validated mechanically)

Files: `experiments/process_reward.py`, `experiments/probe_process_reward.py`,
`experiments/test_process_reward.py`, modifications in `experiments/sft_code.py`.

**Probe baseline** on `checkpoints/sft_phase_c_combined.pt` (no Phase A
training yet) — confirms the canonical "thinks aren't productive"
problem the loss is designed to fix:
- mean Δlogp(after K=4 thinks − before) = **−0.165**
- Only **30.5 %** of high-gate positions benefit from K=4 thinks
- Histogram heavily centred near zero with a slight left tail
- Target after Phase A training: mean Δlogp > 0, %positive > 50

### Phase B — ThinkAdapter (BUILT)

Files: `experiments/model.py` (ThinkAdapter class, Block integration,
TinyLM threading, prefill + forward_step `_step_block` adapter
plumbing), `experiments/optim_utils.py`, `experiments/model_builder.py`,
`experiments/train_lm_args.py`, `experiments/eval_bracket_structure.py`
(auto-detect + force_* overrides), `experiments/test_think_adapter.py`,
modifications in `experiments/sft_code.py` (legacy + modern paths,
force_use_think_adapter override).

**Validation invariants**: α=0 byte-identical to no-adapter; adapter
weights route to AdamW (not Muon) verified; state-dict round-trip
preserves output; adapter fires during incremental decode (the v5
review caught `_step_block` was bypassing `Block.forward`; fixed).

### Phase C — soft-mixture decode (BUILT)

Files: `experiments/eval_humaneval.py` (new `generate_soft_mixture`,
`_clone_cache`, `_clone_fla_cache`, `--gate_mode` CLI),
`experiments/test_soft_mixture_decode.py`.

**Validation invariants**: σ=1 → emit-only path; σ=0 → think-only
path; cache clone isolates FLA recurrent state + WM buffer + FiLM
lagged sources + per-row think-run counter; default `--gate_mode hard`
remains byte-identical to prior behaviour.

### Code review findings (all three FIXED)

1. **Phase A pad-as-think bug**: PAD == THINK_ID silently triggered
   `state_readonly_at_think` / `mem_write_only_at_think` on padding,
   corrupting the after-forward's recurrent state. Fixed: caller in
   `sft_code.py` uses `pad_id=0`; `compute_process_reward_loss`
   raises ValueError if pad == think_id; regression test
   `test_pad_eq_think_raises`.
2. **Phase B/C adapter inert at decode**: the state-passing
   incremental-decode path (`_step_block` invoked from `prefill` and
   `forward_step`) bypassed `Block.forward` entirely, so the adapter
   silently did nothing at inference. Fixed: threaded `think_mask`
   through `_step_block` and apply adapter at the end of `_step_block`
   when block has `use_think_adapter`. Regression test
   `test_adapter_fires_during_incremental_decode`.
3. **Phase B SFT plumbing gap**: `--use_think_adapter` wasn't wired
   into `sft_code.py` (could only inherit adapter from a pretrain
   ckpt). Fixed: added CLI flags `--use_think_adapter` /
   `--think_adapter_hidden_mult`, threaded through both modern path
   (`build_model_from_ckpt` `force_use_think_adapter` override) and
   legacy path (TinyLM kwargs).

### Test totals

- New: **30** (13 process-reward + 11 think-adapter + 8 soft-mixture)
- Plus 2 regression tests for the v5-review bugs.
- All 95 tests in the touched-module set pass.

### Next: training recipes

Three SFT runs to exercise the new mechanisms. Each ~30-40 min on a
5090. Run sequentially or in parallel if a second GPU frees up.

1. **SFT v8 — Phase A only** (process reward on the existing trunk):
   ```
   .venv/bin/python -u experiments/sft_code.py \
     --load_ckpt checkpoints/sft_phase_c_combined.pt \
     --save_ckpt checkpoints/sft_v8_process_reward.pt \
     --distilled_jsonl data/sft_cot_thinking_v1.jsonl \
     --distilled_keep_only_passing \
     --with_thinking --retrieval_as_input_thinking \
     --future_emb_loss_weight 0.1 --wm_gist_horizons "16,64,256" \
     --mem_size 1024 --state_readonly_at_think \
     --process_reward_weight 0.1 --process_reward_K 4 \
     --process_reward_apply_min_sigma 0.3 \
     --process_reward_sample_frac 0.1 \
     --process_reward_max_positions 128 \
     --epochs 2 --batch 4 --lr 3e-6 --max_len 1024
   ```
   Eval: HumanEval `--gate_mode hard` and `--gate_mode soft`; process-
   reward probe (expect mean Δlogp > 0 if training worked).
2. **SFT v9 — Phase A + Phase B**: as v8 plus `--use_think_adapter
   --think_adapter_hidden_mult 2`. Tests whether dedicated think-time
   parameters stack with process reward.
3. **Eval-only sweep — Phase C**: A/B `--gate_mode hard` vs
   `--gate_mode soft` on every interesting ckpt (sft base, v4,
   rl_v2_step300, v8, v9).

### Decision gates

- Phase A success: probe `mean Δlogp > 0` AND HumanEval >= 12/164
  (matches v4)
- Phase B success: adapter α grows from 0 to > 0.05 AND HumanEval
  >= 13/164 over Phase A
- Phase C success: soft strictly beats hard on at least one ckpt
- Combined target: 17/164 (one over current best 16)
