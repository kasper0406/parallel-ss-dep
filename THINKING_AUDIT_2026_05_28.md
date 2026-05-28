# Thinking-gate aux-loss audit — 2026-05-28

Read-only audit of the process-reward + gate-calibration approach to making the
287M DeltaNet thinking gate productive. Scope: `experiments/process_reward.py`,
the pretrain wiring in `experiments/train_lm.py`, the SFT wiring in
`experiments/sft_code.py`, the inference generator in
`experiments/eval_humaneval.py`, and the two smoke launchers.

Bottom line up front: **the approach is mostly correctly *implemented* (the index
math is right), but it is *mis-designed* and *mis-wired* in ways that fully
explain why in-distribution metrics look great while downstream regresses.** The
mechanism-mismatch the user already found is real and MAJOR, but it is NOT the
whole story — there are at least three independent design-level flaws that each
predict downstream regression on their own.

---

## A. The flaw already found — mechanism mismatch (CONFIRM + severity)

**Status: CONFIRMED. Severity: MAJOR (necessary, not sufficient, to explain the
failure).**

Evidence:
- `train_lm.py:1434` and `train_lm.py:1464` both pass `retrieval_as_input=False`
  to `compute_process_reward_loss` / `compute_gate_calibration_loss`.
- `--retrieval_as_input_thinking` is **not even an argument of `train_lm.py`**
  (`grep retrieval_as_input experiments/train_lm_args.py` → nothing; the only two
  hits in `train_lm.py` are the two hard-coded `False`s). Pretrain has *no*
  retrieval-as-input path. So during pretrain the gate is calibrated for the
  discrete `[THINKING]`-embedding think, and the trunk is taught to be productive
  with that mechanism.
- `launch_sft_smoke_thinking.sh:29` passes `--retrieval_as_input_thinking` but
  does **not** pass `--process_reward_weight` or `--gate_calibration_weight`
  (`grep -c process_reward launch_sft_smoke_thinking.sh` → 0). So during SFT —
  the *only* stage that uses the retrieval-as-input mechanism — the gate is never
  re-calibrated at all. It is fine-tuned by the plain LM loss on the new input
  distribution, drifting freely.

So the calibration that was trained (`σ` for `[THINKING]`-embedding thinks) and
the mechanism deployed (additive-α retrieval-as-input thinks, one-at-a-time with
re-decode) are **two different functions of the hidden state.** The
`σ`-vs-helpfulness mapping learned in pretrain has no reason to transfer.

**Why it's MAJOR not FATAL:** even if you fixed the wiring so SFT calibrated the
right mechanism, flaws B, C, and D below would still make the *target* of the
calibration wrong. The mismatch guarantees no transfer; the design flaws mean
"transfer" wouldn't help anyway.

---

## B. Train/inference mechanism mismatch is *deeper* than the flag — K-at-once vs iterative re-decode

**Severity: MAJOR. Design limitation (not a one-line fix).**

`_build_after_sequences` (process_reward.py:101-138) inserts **all K thinks at
once** (`out[i, pad_n+prefix_len : pad_n+prefix_len+K] = thinking_token_id`,
line 136) and reads logits at the final think (`last_pos = L_after - 1`,
line 137). Inference (`eval_humaneval.py:332-387`) inserts thinks **one at a
time**, and after *each* think the gate re-decides whether to think again
(line 351 `if gate_val >= emit_threshold or force_emit: break`).

Consequences:
1. The aux loss measures "what if the model is *forced* to think exactly K
   times." Inference thinks a *variable*, gate-controlled number of times. The
   quantity being optimized (Δlogp after K forced thinks) is not the quantity
   that occurs at deploy time.
2. In retrieval-as-input mode the difference compounds. `_retrieval_input_embeds`
   (process_reward.py:159-182) does **one** no_grad forward over the
   `[prefix, K×THINK]` sequence to populate `memory._last_injection`, then shifts
   it by one and adds it at think positions. But in that single no_grad forward
   the K think positions all carry the *plain `[THINKING]` embedding* — i.e. the
   homogeneous-input pathology that retrieval-as-input exists to defeat (CLAUDE.md
   "what didn't work: FIX A"). So the retrievals injected into the after-forward
   are computed from a low-rank, homogeneous think manifold. Inference instead
   recomputes the retrieval *after each individual augmented think*, so each
   think's retrieval is conditioned on the previous *augmented* think. The
   aux-loss "after" distribution is therefore systematically different from — and
   more degenerate than — the inference distribution.
3. `state_readonly_at_think` is ON in pretrain (launch_pretrain_smoke_thinking.sh)
   and the after-forward's thinks DO trigger it (think_mask is rebuilt inside
   `TinyLM.forward` from `input_ids == thinking_token_id`, model.py:1932, and
   `after_ids` contains the think id). That is arguably the intended semantics for
   "does thinking help without corrupting state." But SFT does **not** pass
   `--state_readonly_at_think` (grep → 0), so the deployed model's thinks *do*
   write to the recurrent state. Pretrain measured "helpfulness of
   state-preserving thinks"; SFT/inference runs state-corrupting thinks. Third
   independent mechanism divergence.

---

## C. The aux-loss TARGET is an invalid proxy — "improves next-token logp" ≠ "helps the task"

**Severity: FATAL (this is the core design error). Design limitation.**

The premise (process_reward.py:9, compute_gate_calibration_loss:382-385): a think
is "good" iff `log p_after(y[t+1]) > log p_before(y[t+1])` — it improves the
**immediate next-token** prediction. Structural reasons this is the wrong target:

1. **It rewards thinking on locally-hard but globally-irrelevant tokens.**
   Candidate positions are selected purely by `σ > min_sigma` (process_reward) or
   `σ ∈ [min,max]` (gate_calibration) AND a valid target. High-σ / mid-σ positions
   in a code corpus are overwhelmingly high-entropy *surface* tokens: a variable
   name, a string literal, an arbitrary numeric constant, whitespace after a
   docstring. Adding K thinks lets the trunk run more computation and sharpen any
   distribution, including these. The metric says "thinking helped" (Δlogp > 0)
   even though predicting the *next token of a literal* is not the reasoning the
   coding task needs. The reported `%pos = 88-100`, `Δlogp = +3..+8` is exactly
   what you'd expect from "more compute sharpens high-entropy local predictions"
   — it is not evidence of *task-relevant* reasoning.

2. **Next-token logp improvement is almost free for an autoregressive model with
   extra compute.** K extra think positions = K extra layers-worth of sequential
   refinement of the *same* hidden state before emitting. For *any* mildly
   uncertain position this will usually nudge the true token's logp up. The
   target is therefore close to "is this position uncertain?" — which is what the
   entropy-grounded gate target (`--gate_entropy_aux_weight`, already in the repo)
   does directly and far more cheaply. The new loss launders an
   uncertainty-signal through an expensive double forward and calls it
   "helpfulness."

3. **No credit assignment to the task.** HumanEval/MBPP reward is *terminal*
   (does the function pass tests). A think that improves `logp` of the next token
   but pushes the model toward a locally-plausible-but-globally-wrong completion
   is rewarded by this loss and punished by the grader. The only signal that can
   teach productive coding-thinks is execution-grounded (the repo already has it:
   `train_rl_grader.py`). A per-token logp proxy is structurally blind to it.

This is why "in-distribution gains, downstream losses": the loss optimizes the
proxy (sharpen uncertain next-tokens) perfectly and the proxy is anti-correlated
with the actual objective once you spend capacity on it (flaw E).

---

## D. Selection bias makes the in-distribution diagnostics self-fulfilling

**Severity: MAJOR (measurement artifact). Bug-in-reasoning, not code.**

The "thinking works in-distribution" evidence (`Δlogp = +5..+8`, `target_frac_one
= 0.6-1.0`) is measured on **exactly the positions the loss is moving**, with a
selection rule that guarantees a positive-looking number:

- process_reward candidates require `gate > apply_min_sigma` (0.3) AND a real
  target (`_select_candidate_positions`, lines 81-83). `frac_positive` /
  `mean_log_ratio` are then computed over only those positions
  (process_reward.py:286-291). This is conditioning the reported "thinking helps"
  statistic on "the gate already wanted to think" — a biased subsample, not a
  population estimate. The right control is: sample positions *uniformly* (or by
  the inference gate policy) and report Δlogp; that number is the one CLAUDE.md's
  own earlier probe found to be **negative** (`probe_process_reward.py`: mean
  Δlogp = -0.165, only 30.5% positive).
- gate_calibration's `target_frac_one = 0.88` (CLAUDE.md 2026-05-27) is computed
  over positions with `σ ∈ [0.1, 0.9]` (line 333-336) — i.e. positions where the
  gate is *uncertain*. For uncertain positions, "K extra forwards sharpen the
  prediction" is ~always true (flaw C.2), so `target_frac_one → 1` is mechanical,
  not informative. The loss then drives σ → 1 on these positions (CLAUDE.md:
  "σ climbed 0.29 → 0.76"), the candidate window *empties* (σ leaves [0.1,0.9]),
  and the metric stops being able to see the regime it just created. The
  diagnostic is a fixed point of the loss, not an independent measurement.

Net: every in-distribution number cited as success is conditioned on a set the
loss itself defines and moves. There is no held-out / policy-distributed control.

---

## E. Capacity competition + always-think drift — the mechanism that turns the proxy into a regression

**Severity: MAJOR. Design limitation (predicted by the repo's own notes).**

CLAUDE.md 2026-05-27 already records "VAL ppl drifts up ~3-5%": the trunk is being
asked to be good at *both* direct prediction *and* post-K-think prediction, and at
287M they compete. Combined with flaw C (the loss says "thinking helps almost
everywhere") and flaw D (the gate is driven toward σ→1 on the whole uncertain
band), the predicted end state is: **the gate fires far more often, each fire
costs a state-corrupting think (flaw B.3) and trunk capacity (this flaw), and the
extra thinks are calibrated to sharpen irrelevant local tokens (flaw C).** That is
precisely a recipe for "HumanEval 10→3, synth_reasoning 3→0": more thinking, worse
task outcomes, exactly the "thinking corrupts recall" + "over-thinking" failure
the project documented at length (CLAUDE.md "thinking corrupts long-range recall,
and the damage scales with think volume").

---

## F. Implementation correctness — items that are actually FINE (verified)

I traced these to rule them out as the cause; they are correct.

1. **Index alignment is correct.** Convention: labels `y` are aligned to `x`
   (`make_batch`, sft_code.py:1104-1107: `y[i,:len]=labels` at the same positions
   as ids), loss shifts (`shift_logits=logits[:,:-1]`, `shift_labels=y[:,1:]`,
   sft_code.py:1313-1314). So `target_shifted = y[:,1:]` gives
   `target_shifted[t] = y[t+1] = x[t+1]` = the next *real* token after position
   t. "Before" reads `main_logits[b,t]`, which (by the shift) predicts exactly
   `y[t+1]`. CORRECT. CPU trace (B=1, T=6, t=2, K=2): `after_ids = [10,11,12,99,
   99]`, `last_pos = 4`; the logits at the last think predict the token following
   the prefix+thinks = the real next token = `x[3]`, the same target as "before."
   `last_pos` points at the right position. No off-by-one.
2. **base_vocab truncation is symmetric.** Both "before" (`main_logits` sliced to
   `base_vocab_for_loss` by the caller, train_lm.py:1419-1421) and "after"
   (sliced inside the helper, process_reward.py:276-277) use the same softmax
   denominator. The think-id-as-target masking (lines 226-227) correctly avoids
   the OOB gather. Consistent.
3. **The gate snapshot fix is correct under grad_accum > 1.** `main_gate` /
   `main_gate_logits` are read from `model._last_gate*` *inside* the per-microbatch
   loop (train_lm.py:1407-1414, inside `for micro in range(n_micro)`), after that
   microbatch's forward, before its extra forwards. They refresh every microbatch.
   No staleness. (sft_code.py:1334-1345 is the same pattern.)
4. **`_retrieval_input_embeds` shift matches inference.** The shift-by-one
   (`shifted_inj`, process_reward.py:176-177: think at position p reads injection
   from p-1) matches inference, where `retrieved = inj[:,-1:,:]` (the read at the
   just-processed position) becomes the input for the *next* appended think
   (eval_humaneval.py:363,371). The *additive-α* form
   (`base_emb + is_think·α·shifted_inj`, lines 179-181) matches the inference
   additive form (`think_emb + α·retrieved`, line 371). The per-position vs
   one-shot difference is the flaw-B concern, not an indexing bug.
5. **process_reward carries gradient; gate_calibration does not (correctly).**
   process_reward's after-forward is grad-enabled (the trunk is being trained,
   lines 262-268); `log_p_before` is detached (line 254). gate_calibration's
   after-forward is under `no_grad` (line 448) and the target is detached
   (line 477) — correct, since it's a *label* for the gate BCE. Both right.

---

## Ranked list — most likely reasons the approach fails downstream

1. **(C) The optimization target is the wrong objective.** "Improves immediate
   next-token logp" rewards sharpening locally-uncertain surface tokens, which is
   uncorrelated (or anti-correlated, once it costs capacity) with terminal task
   success. This is the root cause; everything else amplifies it.
2. **(A) Mechanism/stage mismatch.** Pretrain calibrates the discrete-think gate;
   SFT deploys retrieval-as-input thinks and never re-calibrates. Even a correct
   target wouldn't transfer.
3. **(E) Capacity competition → always-think drift.** The loss drives σ up across
   the whole uncertain band; more thinks cost trunk capacity and (in SFT/deploy)
   corrupt recurrent state. Direct mechanism for the HumanEval/synth regression.
4. **(D) Self-fulfilling diagnostics.** The "it works in-distribution" evidence is
   measured on the loss's own moving candidate set with no policy-distributed
   control; the repo's own uniform-sample probe already showed Δlogp negative.
5. **(B) K-at-once vs iterative re-decode + state_readonly on-in-pretrain /
   off-in-SFT.** Three further train/deploy divergences stacked on top.

---

## Recommendation

**The approach is fundamentally mis-designed, not merely mis-wired.** Fixing the
mechanism-mismatch (flaw A) alone will *not* produce a downstream gain, because the
target (flaw C) is a next-token-logp proxy that is structurally blind to the
terminal coding objective and competes with it for 287M of capacity. The
in-distribution "success" (flaw D) is a measurement artifact of the candidate
selection. I would not invest more in the per-token-logp proxy.

If you want to salvage *something*, the 2-3 highest-leverage moves, in order:

1. **Drop the per-token-logp target; make the reward terminal/execution-grounded.**
   The repo already has the only signal that can teach productive coding-thinks:
   `train_rl_grader.py` (execution-grounded GRPO, validated +6 HumanEval to the
   16/164 best). "Does thinking help" should be measured by *grader reward of the
   completed function with vs without the think budget*, not next-token logp.
   Move the think-vs-no-think decision into that loop (it already supports a
   stochastic gate as a policy variable, CLAUDE.md `--stochastic_gate`).
2. **If you must keep a pretrain-time gate signal, use the existing
   entropy-grounded target** (`--gate_entropy_aux_weight`, already validated, zero
   extra forwards). It gives the same "think at uncertain positions" signal the
   logp-proxy actually encodes (flaw C.2), without the capacity-competing extra
   forward, without the self-fulfilling diagnostic, and without claiming it's
   "task helpfulness." Then let RL (move 1) decide which uncertain positions are
   worth thinking on.
3. **Make the train and deploy mechanisms identical end to end.** One stage, one
   think mechanism (retrieval-as-input additive-α, iterative re-decode), one
   `state_readonly_at_think` setting, calibrated against the *same* generation
   procedure used at eval. If a position's think label is computed, compute it by
   running the *actual inference generator* for one step, not a K-at-once forced
   forward. Until train==deploy, any in-distribution metric is uninformative.

A useful *diagnostic* to run before any of this (it would have caught the whole
thing): sample candidate positions by the **inference gate policy** (not by
`σ ∈ [low,high]`), run the **actual retrieval-as-input generator** for the think
burst, and report Δ(grader reward), not Δlogp. I expect that number to be ≤ 0 on
the current ckpts — which is the real reason downstream regresses.
