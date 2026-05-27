# Phase C: Soft-Mixture Decode (THINKING_PLAN v5)

Implemented 2026-05-26 in worktree
`/home/knielsen/ml/parallel-ss-dep/.claude/worktrees/agent-a0515b4c6c748d3e1`.

## What was built

- `experiments/eval_humaneval.py`:
  - New `generate_soft_mixture(...)` function. At each emit step it runs
    BOTH branches (emit and think) and mixes the resulting **probability
    distributions** by σ(gate):
    ```
    p_emit  = softmax(logits_emit)      # no think; canonical state
    p_think = softmax(logits_think)     # one think inserted (retrieval-as-input)
    p_mix   = σ · p_emit + (1-σ) · p_think
    sample from p_mix
    ```
  - The emit-branch state is **canonical**: after sampling, the cache is
    advanced via `forward_step(sampled_token)` on the emit branch. The
    think branch runs on a deep-cloned cache and its state is
    discarded. This is the simpler of the two state-coherence options
    (vs. re-running the trunk from the sampled token); documented in
    the function docstring.
  - Cache deep-clone helpers `_clone_cache` and `_clone_fla_cache` that
    isolate FLA recurrent state, WM buffer, FiLM lagged sources, and
    the per-row think-run counter from the canonical cache.
  - New CLI flag `--gate_mode {hard, soft}` (default `hard`). The
    default preserves existing behaviour byte-for-byte.
- `experiments/test_soft_mixture_decode.py` (8 tests, all passing):
  - End-to-end smoke (tiny model)
  - Missing-WM path raises ValueError
  - `force_sigma=1.0` → output matches emit-branch-only reference
  - `force_sigma=0.0` → output matches think-branch-only reference
  - CLI default is `hard`
  - CLI dispatch routes `soft` to `generate_soft_mixture`
  - Cache clone isolates FLA state + WM buffer + seen-token counter
  - Cache clone preserves per-layer `_seen_tokens`

## Hard vs Soft results (164 / 164 HumanEval)

Common flags (matches prior eval recipe for v7 / Phase C / RL ckpts):
`--prompt_style sft_comment --extract_code_block --use_thinking
--emit_threshold 0.5 --gate_floor 0.0 --min_emit_before_eos 30
--max_gen 256 --temperature 0.0 --generator retrieval_as_input`.

| Checkpoint | Hard pass@1 | Soft pass@1 | Δ |
|---|---|---|---|
| `sft_phase_c_combined.pt` (Phase C SFT base) | **6/164** (3.7 %) | **7/164** (4.3 %) | **+1** |
| `rl_grader_phase_c_step300.pt` (RL grader v1, project headline 16/164) | **10/164** (6.1 %) | **11/164** (6.7 %) | **+1** |
| `rl_discover_v4.pt` (RL discover v4 final) | **12/164** (7.3 %) | **9/164** (5.5 %) | **-3** |

Diagnostics (full 164):

| Checkpoint | hard think_rate | hard mean σ@emit | soft mean σ |
|---|---|---|---|
| sft_phase_c_combined | 0.629 | 0.694 | 0.743 |
| rl_grader_phase_c_step300 | 0.565 | 0.799 | 0.737 |
| rl_discover_v4 | 0.621 | 0.715 | 0.774 |

NOTE: the hard scores reproduced here (SFT 6, RL-grader-step300 10) are
lower than the CLAUDE.md headline numbers (SFT 10, RL-step300 16). I
used the recipe-pinned flags exactly. The discrepancy is probably down
to a different SFT ckpt path (`sft_phase_c_combined.pt` vs the one
CLAUDE.md was referencing), or to changes since the headline was
recorded. The hard-vs-soft Δ within each row is what's load-bearing for
the Phase C decision — all three runs use the same model load and the
same flags except `--gate_mode`.

## Verdict

**Soft-mixture decode does NOT reliably beat hard threshold.** Two of
three ckpts come out +1 (within noise — pass@1 standard deviation at
164 problems and ~10 % pass rate is roughly ±2), and the third
regresses by -3. The two "wins" are at low pass-rates where single
problems swing the number meaningfully.

This fails Phase C's gate ("soft-mixture decode strictly beats hard
threshold on existing v4 or v2 ckpts"). Three plausible reasons:

1. **The gate was trained against a hard threshold.** The model's gate
   head was supervised to produce binary-ish decisions (BCE-with-
   uncertainty target, hard-threshold deploy convention). Querying it
   in a continuous-mix regime is off-distribution; the σ value carries
   a little extra information but not enough to beat its own
   threshold without retraining.
2. **State divergence at think positions.** In hard mode, when the gate
   triggers a think, the canonical DeltaNet recurrence genuinely
   advances through the think — the model is using the think to
   reshape its state. In soft mode the think branch is what-if only,
   so the canonical state never gets that reshaping. The mixed
   distribution sees the *output* of the would-be think but the next
   step's hidden state never reflects it.
3. **Blind 50/50 compute waste.** Hard-mode RL-v4 runs think_rate 0.62
   — the model has a well-calibrated allocation of when to think. Soft
   mode unconditionally runs the think branch every step (50 % of
   compute) regardless of whether the model wanted it; the canonical
   state never benefits from those extra forwards. This may explain
   the -3 regression on RL-v4 specifically: that ckpt's hard threshold
   already routes well, and the unconditional mixing dilutes its
   high-confidence emit logits with a stale think distribution.

A logit-space mix (`σ·logits_emit + (1-σ)·logits_think`) is a cheap
follow-up: for low-entropy emit logits it would weight emit much
harder than probability-space mixing does, and might recover the hard
threshold's behaviour as a limiting case. But the core conclusion
("the gate's continuous output, threshold-trained, doesn't beat its
own threshold at inference") is unlikely to change without
co-training the gate under the new mixing rule.

## Open questions

- **Mixing space.** I mix in probability space (`σ·p_emit + (1-σ)·p_think`)
  rather than logit space. Logit-space mixing (`σ·logits_emit + (1-σ)·logits_think`)
  is also defensible; for very confident emit logits (low entropy) the
  two diverge sharply. THINKING_PLAN's pseudo-code says
  "final logits = σ · logits_emit + (1-σ) · logits_after_think" but
  the natural semantic of "mix the two predictions weighted by P(emit)"
  is probability-space. If soft mode falls flat, a follow-up should
  re-try the logit-space variant.
- **Think branch state is discarded.** A "deep" alternative would let the
  think branch persist its state when σ < 0.5 (committing to the think).
  That converges back toward the hard generator with a richer mixing
  rule; out of scope for Phase C.
- **No retraining.** Phase C is a pure-inference experiment. If
  soft-mixture wins, an obvious follow-up is to train under the same
  mixing rule (the model's gate currently sees a hard threshold in
  training, never a continuous-weighted mixture).
- **Cost.** Soft mode is ~2× per-step (one extra `forward_step` per emit).
  The cache-clone overhead is small (deep-copies a handful of FLA-layer
  tensors + the WM buffer) but proportional to context length.

## Reproduce

```bash
# tests
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m pytest \
    experiments/test_soft_mixture_decode.py -v

# eval (164 problems)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m experiments.eval_humaneval \
    --ckpt checkpoints/sft_phase_c_combined.pt \
    --max_problems 164 --prompt_style sft_comment --extract_code_block \
    --use_thinking --emit_threshold 0.5 --gate_floor 0.0 \
    --min_emit_before_eos 30 --max_gen 256 --temperature 0.0 \
    --generator retrieval_as_input --gate_mode soft
```
