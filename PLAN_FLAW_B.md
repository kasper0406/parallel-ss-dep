# PLAN — Flaw B: make train == deploy for the thinking mechanism

Owner: flaw B (THINKING_AUDIT_2026_05_28.md §B). Read-only plan; no impl yet.

Flaw B is three stacked train/deploy divergences in how a think's effect is
computed for the aux losses (`process_reward.py`):

1. **K-at-once vs iterative re-decode.** `_build_after_sequences` jams K thinks
   in at once and reads logits at the last think. Inference
   (`eval_humaneval.generate_with_retrieval_as_input`) inserts thinks ONE AT A
   TIME, re-deciding the gate after each (`if gate_val >= emit_threshold or
   force_emit: break`).
2. **Homogeneous-think retrieval vs per-step retrieval.** `_retrieval_input_embeds`
   does ONE no_grad forward over `[prefix, K×THINK]` where all K positions carry
   the plain `[THINKING]` embedding (the homogeneity pathology retrieval-as-input
   exists to defeat), then shift-by-one injects. Inference recomputes the
   retrieval after EACH augmented think, so each think conditions on the prior
   *augmented* think.
3. **`state_readonly_at_think` ON in pretrain (`train_lm.py:1885`), OFF in SFT**
   (`sft_code.py` does not force it). Pretrain measures state-preserving thinks;
   SFT/deploy runs state-corrupting thinks.

---

## 1. Compute a think's effect via the ACTUAL inference generator

### Target control flow
Replace the K-at-once `_build_after_sequences` + `_retrieval_input_embeds` +
single-forward path with a function `simulate_inference_thinks(model, x, b_idx,
t_idx, ...)` that, per sampled position `(b,t)`, runs the same loop body as
`generate_with_retrieval_as_input` for the THINK phase only:

- Prefill / forward the prefix `x[b, 0..t]` (batched across the N sampled
  positions — see batching below), capture `_last_gate`, `memory._last_injection`.
- Loop: read gate σ at the last position; if `σ >= emit_threshold` or
  `thinks >= max_think_per_step`, stop. Else append a think token whose input is
  `think_emb + α·retrieved` (additive) or `retrieved` (replace), advance one step
  (recompute retrieval), increment count.
- After the loop, read the logits at the final position → `log p_after(y[t+1])`.
- `log p_before` stays the cheap main-forward read (unchanged).

This makes the "after" distribution exactly the deploy distribution: variable
think count, per-step retrieval, gate re-decision.

### Functions to change
- `process_reward.py`: **delete** `_build_after_sequences` (K-at-once) and
  `_retrieval_input_embeds` (homogeneous single-forward) from the loss path.
  Add `simulate_inference_thinks(...)`. Both `compute_process_reward_loss` and
  `compute_gate_calibration_loss` call it instead of building K-think batches.
- **Refactor `eval_humaneval.generate_with_retrieval_as_input`** to extract the
  think-inner-loop into a reusable helper (`_run_think_burst(model, cache,
  pending_logits, additive, alpha, gate_floor, emit_threshold,
  max_think_per_step, ...) -> (cache, pending_logits, n_thinks)`) so train and
  eval call the *same* code — otherwise they will drift again. This is the
  load-bearing change: a single shared think-step function.
- The gradient subtlety: process_reward needs grad through the after path. The
  inference loop uses `forward_step` over a cloned cache. Either (a) keep the
  decode incremental but grad-enabled (forward_step must support grad — it does,
  it is just `Block.forward`/`_step_block`), or (b) for the grad path, after
  determining the variable think count `k*` under no_grad, do ONE grad-enabled
  full forward over `[prefix, k*×THINK]` using per-step retrievals captured
  during the no_grad decode. (b) is cheaper and avoids backprop-through-cache
  fragility; the per-step retrievals are detached inputs (already detached in
  inference: `_last_injection` is detached), so grad flows only through the
  trunk over the assembled sequence, which is exactly what process_reward wants.
  **Recommend (b).** gate_calibration is all no_grad already → just run the
  no_grad decode and read the label.

### Compute cost
Current after-forward: one `(N, ~T+K)` forward, N≤max_positions (256 for
process_reward, 32 for gate_cal). Iterative re-decode replaces that with, per
position, a prefill (`O(t)`) + up-to-`max_think_per_step` single-token steps.
- With state-passing decode (prefill once, then K cheap `forward_step`s), cost
  ≈ one prefix forward + K·O(1) steps per position ≈ **same order as today's
  single forward**, dominated by the prefill — NOT K× worse. The expensive part
  is N independent prefills (today they are batched into one `(N, L)` forward;
  the decode path is also batchable across the N positions if they share a
  common max length via left-padding + a per-row "done" mask).
- The grad path (option b) adds one `(N, t+k*)` forward — comparable to today.

**Mitigation (quantified):** keep `sample_frac` small and `max_positions` low
(process_reward 256→64, gate_cal 32). Batch all N positions into one left-padded
decode with a per-row finished-mask (rows whose gate flipped to emit stop
appending; pad their step). Cap `max_think_per_step` at the deploy value (e.g.
4–8). Net: roughly **1.5–2× the current aux cost**, not K×. Run the aux every
`process_reward_every_n` steps (new flag) rather than every step if it bites.

---

## 2. Resolve `state_readonly_at_think` ON-pretrain / OFF-SFT

**Correct setting: OFF, end-to-end** — match deploy.

Rationale: the deployed model (SFT ckpt, `generate_with_retrieval_as_input`)
runs thinks that DO write to the DeltaNet recurrent state (SFT never sets it;
forward_step does not mask β). The whole premise of flaw B is train==deploy. If
pretrain measures "helpfulness of state-preserving thinks" but deploy runs
state-corrupting thinks, the calibration is for a different operator regardless
of mechanism. So the consistent choice is whatever deploy actually does.

But note the tension with CLAUDE.md's documented finding ("thinking corrupts
long-range recall, damage scales with think volume"; state_readonly was the
proposed *architectural* fix). There are two coherent worlds:
- **World A (recommended now):** state_readonly OFF everywhere. Train, SFT, and
  deploy all run state-writing thinks. Simplest path to train==deploy.
- **World B (architectural, larger):** state_readonly ON everywhere — pretrain,
  SFT, AND inference (`forward_step` must gain the β-mask; currently only the
  full-forward path is wired — CLAUDE.md flags this as an open follow-up). This
  is the principled anti-recall-corruption design but requires implementing the
  decode-path β-mask + a cache sentinel first.

**Decision: World A for the aux-loss fix.** Enforce it by:
- SFT: do NOT pass `--state_readonly_at_think`; assert in `sft_code.py` that if
  a loaded ckpt has `cfg["state_readonly_at_think"]=True` we either (i) warn
  loudly that train≠deploy, or (ii) re-enable it consistently AND require the
  inference generator to use the readonly path. Pick (i) + flip pretrain OFF for
  the next run.
- pretrain: change `launch_pretrain_*_thinking.sh` to drop
  `--state_readonly_at_think` so the calibrated regime matches SFT/deploy.
- The aux loss's simulated thinks then inherit the model's
  `state_readonly_at_think` attribute automatically (think_mask rebuilt inside
  forward) — once the attribute is consistent, the divergence is gone for free.

World B is the right long-term move IF the recall-corruption regression returns,
but it is gated on the `forward_step` β-mask work and belongs with flaw C/RL, not
here.

---

## 3. Honest assessment — worth it for aux-loss, or only for RL redesign?

**Not worth doing for the per-token-logp aux-loss approach.** Flaw C
(THINKING_AUDIT §C, severity FATAL) says the aux target — "a think is good iff it
raises immediate next-token logp" — is structurally the wrong objective: it
rewards sharpening locally-uncertain *surface* tokens (string literals, var
names) uncorrelated with terminal coding success, and is nearly equivalent to
the existing entropy-grounded gate target at a fraction of the cost. Making
train==deploy (flaw B) only ensures we are measuring the *deploy* value of a
think under a *broken* yardstick. A faithful measurement of a wrong target is
still wrong.

**The flaw-B fix IS worth doing as part of the RL redesign** (audit Recommendation
#1/#3). The shared `_run_think_burst` helper and the "simulate the actual
generator for one decision" machinery are exactly what an execution-grounded
reward needs: in `train_rl_grader.py` you already run the real generator and
grade the completed function. Flaw B's deliverable — *one* think mechanism, *one*
state_readonly setting, *one* shared think-step function used by train and eval —
is a prerequisite for any train==deploy signal, RL included. So: **build the
shared think-step helper and unify state_readonly (§1 refactor + §2), but wire it
into the grader-RL terminal reward, NOT the next-token-logp aux loss.** Drop /
shelve `process_reward_weight` + `gate_calibration_weight` per flaw C.

---

## 4. Dependencies / overlaps with A, C, D, E

- **A (mechanism/stage mismatch, MAJOR):** B is the deep version of A. A is "the
  flag isn't passed"; B is "even with the flag, K-at-once ≠ iterative". The §1
  shared-helper refactor *subsumes* A — once train and eval call the same
  think-step code there is no flag to forget. Fix B's refactor first; A's wiring
  becomes a config assertion. **Strong overlap.**
- **C (target is wrong, FATAL):** B is downstream of C. C says don't use the
  logp target at all. If C wins (it should), B's effort redirects into the
  grader-RL loop rather than the aux loss. **B's scope is gated by C's verdict.**
- **D (self-fulfilling diagnostics, MAJOR):** B partially fixes D. D's core
  complaint is candidates sampled by `σ ∈ [low,high]` on the loss's own moving
  set. B's "simulate the actual generator" enables D's recommended control:
  sample by the *inference gate policy* and report Δ(grader reward). Implement
  the §1 simulator and you get D's honest diagnostic for free. **Overlap on the
  diagnostic.**
- **E (capacity competition / always-think drift, MAJOR):** independent of B's
  mechanism work, but B's state_readonly unification (§2) directly bears on the
  "each fire corrupts state" half of E's predicted regression. **Mild overlap.**

Ordering: resolve C's verdict → if keeping any train-time think signal, do B's
shared-helper refactor (subsumes A) + state_readonly unification (§2) → reuse the
simulator for D's honest diagnostic → feed E's depth-cost into the RL loop.

---

## 5. Test plan

Unit (pytest, `experiments/test_*.py`, CUDA tests pinned to a free GPU):
1. **`test_shared_think_step`**: the extracted `_run_think_burst` helper produces
   bit-identical cache + logits whether called from the eval generator or the
   training simulator on the same `(prefix, gate, retrieval)` input (the
   anti-drift guarantee).
2. **`test_simulate_matches_generator`**: for a fixed seed + fixed gate outputs,
   `simulate_inference_thinks` inserts the SAME number of thinks at the SAME
   positions, and yields the same final-position logits, as a literal call to
   `generate_with_retrieval_as_input` truncated to the think phase. Cover both
   `additive=True` (v7) and `additive=False` (v5/v6).
3. **`test_variable_think_count`**: construct gate outputs that flip to emit
   after 2 of a max-4 budget → simulator stops at 2 (not 4); regression against
   the old K-at-once behaviour.
4. **`test_state_readonly_consistency`**: with `model.state_readonly_at_think`
   True vs False, the simulated after-forward's recurrent-state write at think
   positions matches the full-forward path (reuse
   `test_state_readonly_thinking.py` harness); assert SFT default = OFF and that
   a ckpt-cfg mismatch raises/warns.
5. **`test_grad_path`** (process_reward option b): grad flows to trunk params
   through the assembled `[prefix, k*×THINK]` forward, and is zero through the
   detached per-step retrievals.
6. **`test_aux_cost_bound`**: assert N positions decode in ONE batched
   left-padded pass (per-row finished-mask), not N separate forwards.

Integration / diagnostic (not in CI):
7. **Honest control diagnostic** (flaw D): on `sft_phase_c_combined.pt`, sample
   candidate positions by the inference gate policy, run the real generator,
   report Δ(next-token logp) AND Δ(grader reward) on a held-out MBPP slice.
   Expectation from the audit: Δ(grader reward) ≤ 0 on current ckpts — this is
   the go/no-go gate for whether ANY train-time think signal is worth keeping.
8. **Smoke**: `launch_sft_smoke_thinking.sh` runs end-to-end with the unified
   helper, no shape/recompile errors, train==eval think counts logged and equal.
