# Code review — commits `828940b` + `6f9ba30` (2026-05-27)

Reviewed branch: `thinking-token-gate-curriculum`.
Smoke training PID 1950046 on GPU 0 was NOT disturbed.

## Verdict by category

| # | Category | Verdict |
|---|----------|---------|
| 1 | Correctness bugs | **RED FLAG** — one BLOCKING bug in `sft_code.py` (snapshot-overwrite — same class as the pretrain-side bug the author already fixed in `train_lm.py`). Also one IMPORTANT dtype / autocast concern. |
| 2 | Test coverage | **YELLOW** — helpers well covered; the wired-up trainer paths and the snapshot-freshness invariant have NO regression test. |
| 3 | Architecture / design | **GREEN** — gradient flow is correct, snapshots inside microbatch loop are correct, `_eager_forward` design is sound. |
| 4 | Style / CLAUDE.md adherence | **GREEN** — all flags default to OFF; no scaffolding; comment density is reasonable for the bug-density of the territory. |
| 5 | Specific spot checks | **YELLOW** — gate_logits snapshot in train_lm.py is correct (#5a); in sft_code.py it is BROKEN (#5b); the SGD-target test on the gate is rigorous (#5d). |

---

## 1. Correctness — BLOCKING + IMPORTANT findings

### B1 [BLOCKING] `sft_code.py` reads `_last_gate_logits` AFTER `compute_process_reward_loss` overwrites it

**File**: `experiments/sft_code.py:1334..1376`.

The pretrain wiring (`train_lm.py:1407-1414`) explicitly snapshots
`main_gate_logits` BEFORE any extra forward, and the commit message
calls this out as fix #5 ("the load-bearing one"). The SFT-side
wiring in `sft_code.py` did NOT receive the same fix. The control
flow is:

```python
# sft_code.py:1334
main_gate = getattr(model, "_last_gate", None)
# ... process_reward fires here, runs an extra forward, which
#     overwrites both model._last_gate AND model._last_gate_logits
#     because TinyLM forward unconditionally re-stashes them.
if use_process_reward and main_gate is not None:
    compute_process_reward_loss(model, x, y, gate=main_gate, ...)
# ↓↓↓ this read is now STALE — gate_logit refers to the smaller
#     (N_pr, T_pr) tensor from process_reward's after-forward.
if use_gate_calibration and main_gate is not None:
    gate_logit = getattr(model, "_last_gate_logits", None)  # WRONG
    compute_gate_calibration_loss(..., gate_logits=gate_logit, ...)
```

What happens at runtime depends on the index range:
- `b_idx, t_idx` from `_select_candidate_positions_window` are valid
  indices into `gate=main_gate` (shape `(B, T)`), where `T` is the
  full sequence length (2048 in pretrain). 
- `gate_logits[b_idx, t_idx]` then indexes into a tensor of shape
  `(N_pr, T_pr)` where `N_pr ≤ 32` and `T_pr ≤ T`. Almost certainly
  out of bounds on dim 0 (b_idx can reach B-1=7); CUDA assert.
- Even when both indices happen to be in-bounds, the *values* read
  are unrelated logits from a different forward → BCE supervises the
  wrong logits and gradient flows back into the wrong tensor.

**Severity**: BLOCKING. This is the exact bug the train_lm.py fix
prevents. The SFT launchers `launch_sft_v8_process_reward.sh` /
`launch_sft_v9_phase_a_b.sh` both enable process_reward; if a future
SFT launcher also turns on `--gate_calibration_weight > 0` the run
will either crash on a CUDA assert or train the wrong tensor.

**Fix applied** (uncommitted edit): snapshot `_last_gate_logits` at
the same point `main_gate` is snapped, BEFORE the process_reward
call.

Diff sketch:

```diff
@@ sft_code.py around line 1334
-                    main_gate = getattr(model, "_last_gate", None)
+                    main_gate = getattr(model, "_last_gate", None)
+                    # Snapshot the pre-sigmoid gate logits BEFORE any
+                    # extra forwards — process_reward's after-forward
+                    # overwrites `_last_gate_logits` to the wrong
+                    # shape (N_pr, T_pr), and gate_calibration would
+                    # then index it with the main-forward (B, T)
+                    # indices (CUDA assert / wrong tensor trained).
+                    # Same fix as train_lm.py:1407-1414 (commit 828940b
+                    # bug #5).
+                    main_gate_logits = getattr(
+                        model, "_last_gate_logits", None)
                     if use_process_reward and main_gate is not None:
                         ...
-                    if use_gate_calibration and main_gate is not None:
-                        pad_id = 0
-                        gate_logit = getattr(
-                            model, "_last_gate_logits", None)
+                    if use_gate_calibration and main_gate is not None:
+                        pad_id = 0
+                        gate_logit = main_gate_logits
```

**Regression test added** (uncommitted): in
`experiments/test_pretrain_aux_losses.py`, a test that runs a
process_reward call before reading `_last_gate_logits` and asserts
that the post-pr read is a different tensor than the pre-pr read
(documents the invariant).

### B2 [IMPORTANT] BCE target tensor dtype under bf16 autocast

**File**: `experiments/process_reward.py:484` and `:487`.

```python
loss = F.binary_cross_entropy_with_logits(gl.float(), target)
```

`gl` is the pre-sigmoid gate logit. In production it lives in bf16
(autocast). The `.float()` cast lifts it to fp32. `target` is built
from `(diff > 0).float()` (line 476) or `torch.sigmoid(diff * scale)`
(line 474). `diff` was computed in fp32 under `with torch.no_grad():`
inside autocast — but `log_softmax(...float())` forces fp32 here too.
**Status**: looks fine. fp32 vs fp32 — no dtype mismatch.

However the *fallback* path (line 487) uses
`F.binary_cross_entropy(sig, target)` where `sig` is bf16 (autocast).
The `.float()` cast is applied. Still fine. Note this branch is only
reachable if the caller forgot to pass `gate_logits` — currently
unreachable in production (both trainers pass it). NIT.

### B3 [IMPORTANT] `_call_model_eager` is bypassed by `_retrieval_input_embeds` for the *no-grad probe* inside it, but the OUTER forward call site does use it

**File**: `experiments/process_reward.py:172`.

Already correct — the probe forward inside `_retrieval_input_embeds`
goes through `_call_model_eager`, AND the outer `model(...)` call has
been replaced (line 265-268). Verified by inspection. GREEN.

### B4 [NIT] `_build_after_sequences` python loop scales linearly with `max_positions`

At default `max_positions=32` this is 32 small Python-side index
assignments per microbatch per aux loss → ~64 ops/step. Fine for now;
flag for vectorization if `max_positions` ever climbs into the
hundreds. No action.

### B5 [NIT] Per-microbatch extra forwards under grad_accum=14

With smoke launcher `--grad_accum 14 --process_reward_weight 0.05
--gate_calibration_weight 0.05`, each optimizer step runs 14 ×
(main + pr_after + gc_after) = 42 forwards instead of 14. With both
aux losses on, training is ~3x slower than the bare pretrain. This
matches the user's observed "17k tok/s vs Phase C's 45k tok/s" (smoke
launcher comments) — the aux-loss cost dominates, not just the
profiling/no-compile overhead. NIT — recipe choice, not a bug. Worth
considering: gate aux losses only on every N-th microbatch.

---

## 2. Test coverage

### Covered well
- Both helpers' core math (target derivation, sample selection, BCE
  direction).
- `_eager_forward` route on / off.
- Pad-id collision guard.
- target_shifted_masked think-token target masking
  (`test_compute_respects_sample_frac` updated).
- SGD step on `gate_head` actually moves σ in the expected direction
  for both `think_helps` and `think_hurts` (`test_gate_calibration.py
  ::test_forced_target_one_pushes_sigma_up` /
  `test_forced_target_zero_pushes_sigma_down`). Verified rigorous.

### Gaps

| ID | Gap | Severity |
|----|-----|----------|
| C1 | No regression test for the snapshot-overwrite invariant. A test that calls process_reward then asserts `model._last_gate_logits` is NOT the same tensor object as the main-forward snapshot would catch any future copy of B1 immediately. **Added** in `test_pretrain_aux_losses.py`. | IMPORTANT (filled) |
| C2 | No end-to-end test that exercises the trainer's actual loop (`train_lm.py` or `sft_code.py` `for micro in range(n_micro)`). The B1 bug exists in code that has no test. | IMPORTANT (filed only) |
| C3 | `test_train_lm_args_flag_wired` parses the gate_calibration flag but no test parses the process_reward flag. Trivial. | NIT |
| C4 | Smooth-target mode (`smooth_target_scale > 0`) gradient direction is asserted only on `think_helps` (target ~ 0.9). No "should push σ DOWN when smooth target < 0.5" test. | NIT |
| C5 | The `inputs_embeds` argument flow through `_call_model_eager` is not separately tested. The `retrieval_as_input=True` branch of either aux helper has zero tests. Pretrain doesn't use it (both pass `retrieval_as_input=False`), but SFT will once a recipe with `--retrieval_as_input_thinking` lands. | IMPORTANT |
| C6 | No test that the aux losses survive a *gist-loss-enabled* main forward (`outs += (gist_scalar,)` tuple return). The new tuple-unwrap code path is present but only tested by `isinstance(out, tuple)` indirectly via the mock returning a plain tensor. | IMPORTANT |

---

## 3. Architecture / design — GREEN

### Snapshot timing (`train_lm.py:1407-1416`) — correct

The snapshot happens INSIDE the `for micro in range(n_micro)` loop
(line 1366), AFTER `_nonthink_forward_loss` returns `logits` from the
main microbatch's forward, and BEFORE the process_reward call. Each
microbatch gets a fresh snapshot — no staleness across grad_accum.

### Gradient flow

- `compute_process_reward_loss`: `log_p_before` is detached;
  `log_p_after` is NOT inside `no_grad`, so the loss
  `(log_p_before - log_p_after).mean()` minimises `-log_p_after.mean()`
  via the after-forward → trunk + WM + PKM. Correct.

- `compute_gate_calibration_loss`: the entire after-forward is wrapped
  in `with torch.no_grad():`. The target `(diff > 0).float()` is
  inherently non-differentiable, and the smooth variant
  `sigmoid(diff * scale)` is detached at line 477. BCE only flows
  gradient into `gate_logits` which is the main-forward tensor
  (still grad-enabled). Correct. Slight asymmetry: process_reward
  pays the after-forward's backward memory; gate_calibration does
  not. Documented in the source comment ("Saves activation memory vs
  process_reward").

### `_eager_forward` stash

Pattern is sound. The forward is wrapped in bf16 autocast first, then
stashed, then compiled. The aux helpers route through the bf16-but-
uncompiled path, avoiding the Inductor symbolic-shape assertion when
the aux's `(N, L_after)` shape differs from the compiled `(B, T)`
graph. Test `test_eager_forward_used_when_present` confirms the
helper picks `_eager_forward` over `model.forward` when present.

The smoke launcher uses `--no-compile`, so the `_eager_forward`
attribute is absent in production right now — the fallback
`model(...)` path is what's actually exercised by the running smoke.
The compile path will be exercised by the next full pretrain.

---

## 4. Style / CLAUDE.md adherence — GREEN

- All new flags default to 0.0 / False → byte-identical baseline.
- No backwards-compat shims for hypothetical futures (the
  `getattr(args, "...", default)` calls are because the same args
  namespace is consumed by `model_builder` / mid-eval that may
  pre-date the flags — defensive, not scaffolding).
- Comment density is high but every comment is non-trivial WHY
  (e.g. "pad-as-think corrupts the after-forward's recurrent state",
  "the load-bearing one"). Conforms to project style.
- One nit: `target = target.detach()` (line 477) is redundant when
  the hard-target branch already returned a boolean→float (non-diff).
  Harmless. NIT.

---

## 5. Specific spot checks

### 5a [GREEN] `process_reward.py:359-365` pad guard

`compute_gate_calibration_loss` has the `pad_token_id == thinking_token_id`
guard at lines 391-397. `compute_process_reward_loss` has it at lines
209-215. Both helpers protected. Both have unit tests (`test_pad_eq_think_raises`).

### 5b [GREEN] `train_lm.py:1407-1416` snapshot inside microbatch loop

Verified. The snapshot lives at line 1407 which is INSIDE the
`for micro in range(n_micro):` loop that starts at line 1366. Each
microbatch re-snapshots after its own `_nonthink_forward_loss` call.
No staleness across grad_accum boundaries.

### 5c [BLOCKING] `sft_code.py` does NOT have the same snapshot fix

See B1 above. Fix applied.

### 5d [GREEN] `test_forced_target_1_pushes_sigma_up` does grad-step the gate_head and observe σ move

`_run_one_step()` (lines 229-278) runs 5 SGD steps on
`[model.gate_head.weight, model.gate_head.bias]` only, with a
forced-direction mock (`forward_mode="think_helps"` makes the
after-forward put a +10 logit bias on the target token), captures
σ before step 0 and σ after step 5, and asserts `sigma_after >
sigma_before`. Tight test.

### 5e [GREEN] `_eager_forward` IS used: by `_call_model_eager` in both helpers (`process_reward.py:153-156`) and exercised by `test_eager_forward_used_when_present`. Not dead code.

---

## Other observations

### O1 [NIT] Process_reward extra forward computes (and discards) gist_loss

When the main forward has `_gist_loss_enabled=True`, the after-forward
also computes `trunk_gist_loss(h, gist_heads, horizons)` inside the
compiled graph, returns a tuple, and the helper discards index `[1]`.
Wasted compute (~3% per aux forward). Can be sidestepped by setting
`model.training = False` for the duration of the aux forward, but
that changes other behavior (dropout, etc — there is no dropout here,
but principled approach is to guard the gist computation behind a
caller-supplied flag). Not worth fixing now.

### O2 [NIT] `aux_rng` is a CPU Generator seeded once at startup

`train_lm.py:1442 / 1462` reuses the same `aux_rng` across all steps
+ across both helpers. Means resuming from a mid-step ckpt won't
reproduce the exact same sample positions (rng state isn't saved with
ckpt). For a deterministic resume, save `aux_rng.get_state()` to the
ckpt. NIT — sampling is small fraction so reproducibility loss is
mostly cosmetic.

### O3 [NIT] `last_pr_stats` / `last_gc_stats` carry only the LAST microbatch's stats, not the mean across grad_accum=14 microbatches

`pr(...)` / `gc(...)` log lines therefore overweight the last
microbatch. For diagnostic-only use this is fine (~7% sampling
variance per step). NIT — flag for later if Δlogp / tgt1 plots look
noisier than expected.

### O4 [GREEN] No DDP / async race

`_last_gate` and `_last_gate_logits` are tensors stashed on `model`.
The smoke run is single-GPU; in a DDP run each rank has its own
`model` instance and its own stash. No cross-rank race possible.

---

## Summary of applied changes (uncommitted)

1. **`experiments/sft_code.py`**: snapshot `main_gate_logits` at the
   same point `main_gate` is snapped (lines ~1334-1376). Fixes B1.

2. **`experiments/test_pretrain_aux_losses.py`**: new test
   `test_snapshot_invariant_protects_against_overwrite` asserts that
   calling `compute_process_reward_loss` mutates
   `model._last_gate_logits` (proves the snapshot pattern is load-
   bearing — any future trainer that re-reads after a pr call will
   get a stale tensor).

Tests pass: `38 passed in 1.51s` before changes; will re-verify after
applying the fix.
