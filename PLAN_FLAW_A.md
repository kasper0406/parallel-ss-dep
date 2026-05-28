# PLAN — Flaw A: train==deploy think-mechanism

Owner: flaw A (mechanism mismatch, MAJOR). Companion to `THINKING_AUDIT_2026_05_28.md`.
Read-only plan; no implementation here.

## 0. What flaw A actually is (after re-tracing the code)

The audit's framing is correct but the locus is sharper than "SFT never re-runs
the aux losses":

1. **Pretrain has no retrieval-as-input path at all.** `train_lm.py:1434` and
   `:1464` hard-code `retrieval_as_input=False`, and `--retrieval_as_input_thinking`
   is **not an argument of `train_lm_args.py`** (only `--enable_thinking_token`
   and `--use_memory` exist). So pretrain *cannot* calibrate the deployed
   mechanism even if we wanted it to. The gate/trunk are calibrated for the
   discrete `[THINKING]`-embedding think.
2. **SFT is already mechanism-aware in code.** `sft_code.py:1365` and `:1405`
   pass `retrieval_as_input=bool(args.retrieval_as_input_thinking)`. The aux-loss
   plumbing, the additive-α `inputs_embeds` main-forward path (`sft_code.py:1283-1306`),
   and the args (`--process_reward_weight`, `--gate_calibration_weight`,
   lines 752-796) are all present. **The only thing missing is that
   `launch_sft_smoke_thinking.sh` does not pass the aux-loss weights** (it passes
   `--retrieval_as_input_thinking` at line 29 but no `--process_reward_weight` /
   `--gate_calibration_weight`).

So flaw A decomposes into: (A1) pretrain calibrates the *wrong* mechanism, and
(A2) SFT calibrates the *right* mechanism but the smoke launcher never turns it
on. A2 is a one-line launcher fix; A1 is a real plumbing job.

## 1. Exact code changes to make train==deploy (if we pursue A)

### A2 — enable SFT-stage calibration of the deployed mechanism (trivial)
- `launch_sft_smoke_thinking.sh`: add `--process_reward_weight 0.05`,
  `--gate_calibration_weight 0.05`, and the matching K / sigma / sample_frac /
  max_positions flags (mirror `launch_pretrain_smoke_thinking.sh:93-103`). With
  `--retrieval_as_input_thinking` already present, `sft_code.py:1365/1405` route
  `retrieval_as_input=True` into the helpers — train mechanism now matches deploy.
- Also add `--state_readonly_at_think` to the SFT launcher (audit B.3): pretrain
  smoke runs it (`launch_pretrain_smoke_thinking.sh:104`) but SFT does not, so the
  deployed model's thinks WRITE to the recurrent state while the calibration was
  measured under state-preserving thinks. `sft_code.py:889-897` already supports
  the override. Make the setting identical across pretrain/SFT/eval.

### A1 — give pretrain the retrieval-as-input path (real work, only if RL plan needs it)
- `train_lm_args.py`: add `--retrieval_as_input_thinking` (store_true) next to the
  thinking flags (~line 211), mirroring `sft_code.py:625`.
- `train_lm.py`: at the two aux callsites, replace the hard-coded
  `retrieval_as_input=False` (`:1434`, `:1464`) with
  `retrieval_as_input=bool(getattr(args, "retrieval_as_input_thinking", False))`.
- `train_lm.py` think-burst main forward: pretrain currently feeds plain
  `[THINKING]` ids and never builds `inputs_embeds`. To truly match deploy, the
  main forward at think positions must use the additive-α injection exactly as
  `sft_code.py:1283-1306`. Factor that block into a shared helper
  (`experiments/thinking_input.py::build_retrieval_input_embeds(model, x,
  thinking_token_id)`) and call it from BOTH trainers and from
  `process_reward._retrieval_input_embeds` (which already duplicates the logic at
  `process_reward.py:159-182`). `model.retrieval_input_alpha` already exists
  unconditionally (`model.py:1637`), so no model-construction change is needed.
- Verify `--use_memory` is on in pretrain whenever `--retrieval_as_input_thinking`
  is set (the injection reads `model.memory._last_injection`); add an assert in
  `train_lm.py` arg-validation.

### B-overlap fix that A's correctness depends on (iterative re-decode label)
The K-at-once forced forward (`process_reward._build_after_sequences`,
`process_reward.py:101-138`) is NOT the deploy mechanism even with
`retrieval_as_input=True`, because deploy re-decodes the retrieval after *each*
think (`eval_humaneval.py:351,363,371`). A truly train==deploy label requires
replacing the K-at-once after-forward with a call to the actual inference
generator for the think burst. **This is flaw B's territory and is the larger
change** — see §3.

## 2. Is fixing A worth doing at all? — NO, not on its own.

**Decision: do A2 (the launcher one-liner + state_readonly parity) only as a
cheap consistency hygiene fix and to unblock a clean diagnostic. Do NOT invest in
A1's full pretrain retrieval-as-input plumbing in service of the per-token-logp
aux loss.**

Reasoning, taking flaw C (FATAL) seriously:
- Flaw C says the calibration *target* (`Δ next-token logp > 0`) is a proxy that
  is uncorrelated/anti-correlated with terminal task reward. Making the mechanism
  match (A) only guarantees that a *wrong* target transfers faithfully. The audit
  is explicit: "the mismatch guarantees no transfer; the design flaws mean transfer
  wouldn't help anyway" (lines 44-47, 245-250).
- Therefore A is **necessary-but-not-sufficient** and only has value *in service
  of a redesigned, execution-grounded approach*, not the current aux loss.

**How A's fix slots into the redesigned (RL) approach** — this is where A pays off:
- The audit's recommended path (lines 254-273) is execution-grounded GRPO via
  `train_rl_grader.py`, with the think-vs-no-think decision as a policy variable
  (`--stochastic_gate`, already in the repo). That loop *rolls out the actual
  inference generator*, so it is train==deploy **by construction** — there is no
  separate K-at-once forced forward to mismatch. The "make train==deploy" goal is
  satisfied for free once the signal moves to RL.
- The one piece of A's plumbing that still matters in the RL world: the rollout
  generator and the trained model must use the **same** think mechanism +
  `state_readonly_at_think` setting that eval uses. `train_rl_grader.py`'s
  `rollout_group_batched` should call `generate_with_retrieval_as_input(...,
  additive=cfg["retrieval_input_additive"])` (it already exists, eval_humaneval.py:244)
  with the same `state_readonly_at_think` the SFT ckpt was trained under. So A's
  deliverable in the RL plan is: **one think mechanism, one state-readonly flag,
  threaded identically through {SFT, RL rollout, eval}** — i.e. the A2 parity work,
  not the A1 pretrain plumbing.
- Concrete recommendation: **drop A1 (pretrain retrieval-as-input + pretrain aux
  losses) entirely.** If a cheap pretrain-time gate prior is still wanted, use the
  already-validated entropy-grounded target (`--gate_entropy_aux_weight`, zero
  extra forwards; audit C.2 / recommendation 2) — it encodes the same "think when
  uncertain" signal the logp proxy launders, with no mechanism to mismatch.

## 3. Dependencies / overlaps with the other flaws

- **B (K-at-once vs iterative):** A and B share the after-forward. A "complete" A1
  fix is blocked by B — making `retrieval_as_input=True` match deploy still leaves
  the K-at-once / single-no_grad-injection divergence (audit B.2). The only way to
  fully close both is to replace the after-forward with the real generator, which
  is the RL move. **A and B are jointly subsumed by going RL.**
- **C (invalid proxy, FATAL):** dominates. C makes A not-worth-doing for the aux
  loss. A only has standalone value once C is resolved by switching to a terminal
  reward. A is downstream of C.
- **D (self-fulfilling metric):** orthogonal to A but interacts: even a perfectly
  mechanism-matched aux loss reports success on its own moving candidate set. The
  diagnostic in §4 (policy-distributed sampling + grader reward) is the D fix and
  is the right way to *measure* whether any A fix helped.
- **E (over-think drift):** A's parity fix (state_readonly_at_think in SFT) is
  partial mitigation for E — making deployed thinks state-preserving reduces the
  "thinking corrupts recall, damage scales with think volume" mechanism. But E's
  root (gate driven σ→1 by the logp proxy) is again a C problem.

**Net ordering:** C → (B,D,E redesign via RL) → A reduces to "thread one mechanism
+ one state-readonly flag through SFT/RL/eval" (the A2 work). A1 is dead.

## 4. Test plan

### Unit / regression (cheap, run now)
- `experiments/test_process_reward.py` already covers the `retrieval_as_input=True`
  branch (test at line 468). Add `test_train_lm_passes_retrieval_as_input_flag`:
  with the new `--retrieval_as_input_thinking` arg, assert `train_lm.py` calls the
  helper with `retrieval_as_input=True` (monkeypatch `compute_process_reward_loss`,
  inspect kwargs). Mirror in `test_sft_code_loading.py` for SFT.
- Add `test_build_retrieval_input_embeds_shared`: the extracted helper produces
  bit-identical `inputs_embeds` to the inline `sft_code.py:1283-1306` block and to
  `process_reward._retrieval_input_embeds` on a fixed tiny model (guards the
  refactor).
- Assert-on-misconfig test: `--retrieval_as_input_thinking` without `--use_memory`
  raises in `train_lm.py` arg validation.

### Mechanism-parity test (the load-bearing one)
- New `experiments/test_train_deploy_parity.py`: build a tiny `TinyLM`
  (`use_memory=True`, `output_gate=True`), pick one position, and assert that the
  log p(next token) produced by (a) the training after-forward with
  `retrieval_as_input=True` and K=1 equals, within tolerance, (b) one step of
  `generate_with_retrieval_as_input(additive=True)`. **Expectation: this test
  FAILS under the current K-at-once path** — that failure is the concrete proof of
  flaw B and the justification for "don't patch A1, go RL." Keep it as a documented
  xfail until the RL generator replaces the after-forward.

### Decisive empirical diagnostic (the audit's recommended pre-work, §recommendation)
- New `experiments/probe_think_grader_reward.py` (read-only on any ckpt): sample
  candidate positions **by the inference gate policy**, not by `σ∈[lo,hi]`; run the
  **actual `generate_with_retrieval_as_input` generator** for the think burst; and
  report Δ(grader reward) using `code_grader.grade`, not Δlogp. Run on
  `sft_smoke_thinking.pt` / `sft_phase_c_combined.pt`.
  - **Go/no-go gate:** if mean Δ(grader reward) ≤ 0 (expected, per audit line 279
    and the existing `probe_process_reward.py` Δlogp=-0.165 result), then **A1 is
    confirmed dead** and we commit to the RL redesign. If it is meaningfully > 0,
    re-open A1.
- Re-run the existing `probe_process_reward.py` with **uniform** position sampling
  (not gate-gated) to reproduce the negative-Δlogp control (audit D) and confirm
  the in-distribution "success" was selection bias.

### Downstream validation (only if a fix ships)
- HumanEval (`eval_humaneval.py --prompt_style sft_comment --extract_code_block`)
  and `eval_longctx_recall.py` before/after, comparing the A2-enabled SFT ckpt vs
  the current `sft_smoke_thinking.pt`. Watch `think_rate` (E regression signal) and
  confirm recall does not drop with state_readonly parity on.
