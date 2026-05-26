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

| Checkpoint | Hard pass@1 | Soft pass@1 | Δ | Notes |
|---|---|---|---|---|
| `sft_phase_c_combined.pt` (SFT base, project: 10/164) | TBD | TBD | TBD | |
| `rl_grader_phase_c_step300.pt` (RL v2, project best: 16/164) | TBD | TBD | TBD | mid-eval Phase C ckpt |
| `rl_discover_v4.pt` (RL v4 final) | TBD | TBD | TBD | |

Soft mean σ at emit (sanity):

| Checkpoint | mean σ (5-problem smoke) | think rate (hard) |
|---|---|---|
| sft_phase_c_combined | 0.646 | 0.664 |
| rl_discover_v4 | 0.761 | TBD |
| rl_grader_phase_c_step300 | TBD | TBD |

## Verdict

(filled in once eval logs land)

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
