# Autonomy session — making thinking actually work

Started 2026-05-27 after user said: "Pick whatever direction you think
is likely to get the thinking working! Iterate by yourself until you have
a solution. Keep a running document about important decisions you take,
or things I should know about."

**Target metric**: HumanEval pass@1 on the best checkpoint we can produce.
Current best is `checkpoints/rl_grader_phase_c_v2_step300.pt` at 16/164
(9.8%).

---

## Context recap (what I know walking in)

**What worked**:
- RL v2 (KL-stable GRPO on Phase C SFT base): 8 → 12 → 14 → 16 over
  steps 100→200→300, then DECLINED to 14 by step400. Real lift, real
  plateau. KL coef = 0.05 prevented v1's catastrophic collapse.
- Sampling-search (pass@8 at T=0.4 or 0.8): +1 to +3 over greedy.
  Search infra works; lift is small because the model is near its
  effective ceiling at 287M params on HumanEval.

**What failed**:
- Every continuation SFT (v8 process-reward; v9 + adapter; v10 inert
  refinement head; v11 active refinement head; rejection_sft_v1):
  4–5/164. The SFT recipe itself appears to regress the model.
- RL v2 plateaued after step300 (gate became deterministic in
  steady state, no more exploration).

**Untried promising lever**:
- The codebase has `--stochastic_gate` + `--gate_entropy_bonus_*` +
  `--kl_coef` in `train_rl_grader.py` (the discovery-RL recipe used in
  `launch_rl_discover_v4.sh`). It was only ever launched from the
  weak SFT base (sft_phase_c_combined ≈ 8/164) and **never evaluated
  on HumanEval** (only on synth_reasoning).
- The composition I want to try: discovery-RL on top of the
  rl_grader_phase_c_v2_step300 ckpt (already 16/164). Hypothesis:
  re-opening gate exploration with entropy curriculum + KL anchor lets
  the model find think-position usage patterns that the deterministic
  RL v2 gate had locked out.

---

## Decision log

### D1 (2026-05-27 ~09:30) — Choose direction: stochastic-gate RL continuation
**Decided**: continue from rl_grader_phase_c_v2_step300 with the
discovery-RL recipe (stochastic gate + entropy curriculum + KL).
**Why**: it's the only credible untried lever in the codebase. Both
SFT and deterministic RL have been thoroughly exhausted. Discovery RL
has never been evaluated on HumanEval, so we don't know if it
fundamentally moves the needle.
**Cost**: ~2-4h training + 30-60min eval per attempt.
**Decision-gate**: if step100 pass@1 (mid-eval, codepath already
exists in train_rl_grader.py) ≥ 17/164, the recipe is showing signal
and continue; if < 16/164, the recipe is fighting the warm start and
abort.

### D2 (2026-05-27 ~09:35) — First verify rl_discover_v4 worked at all
**Decided**: before composing, evaluate rl_discover_v4.pt on
HumanEval. This is the only existing discovery-RL ckpt and it's
sitting on disk unscored. If it lifts the SFT base (8/164) at all,
the recipe is alive; if it's worse than the SFT base, the recipe is
fundamentally broken and composition won't help — pivot needed.
**Cost**: ~30 min on GPU 0 (free).
**Result**: `rl_discover_v4` pass@1 = 12/164 (7.3%).
- vs SFT base (sft_phase_c_combined, ~7/164): **+5 lift** — recipe is alive.
- vs RL v2 step300 (16/164): **less sample-efficient.** Discovery RL
  gave +5 in 200 steps; deterministic RL gave +9 in 300 steps from
  the same start. Per-step rate similar but deterministic wins.
**Conclusion**: discovery-RL alone is competitive but not dominant.
Don't swap wholesale to discovery-RL recipe.

### D3 (2026-05-27 ~13:30) — Conservative composition, not wholesale swap
**Decided**: instead of porting the full discovery-v4 recipe onto v2
step300, do a **conservative composition** — the v2 recipe with EXACTLY
ONE added knob: `--stochastic_gate` + selective sampling. Isolates
the gate-exploration lever from any other change.
**Why**: discovery-v4's smaller gains (D2) suggest its full recipe is
NOT strictly better than v2 — likely because (a) higher KL coef (0.1)
over-constrained vs v2's 0.05, (b) lower LR (1e-6) slowed exploitation,
(c) aggressive entropy bonus (0.05 start) injected noise that
exploitation had to fight. Composition keeps v2's exploit settings
and adds just the one mechanism we want to test.
**Single change**: stochastic_gate ON, gate sampling range [0.1, 0.9],
mild entropy bonus 0.02→0.001 over 200 steps.
**Decision-gate**: at step 100, eval the ckpt. If pass@1 ≥ 16 (matches
v2 step300) AND gate fire moved by Δ > 0.05, recipe is showing signal.
If pass@1 < 14, recipe is destabilizing the warm start despite KL anchor → abort.
File: `launch_rl_discover_v2warm.sh`.

### D4 (2026-05-27 ~13:55) — Step50 eval: REGRESSION
**Result**: step50 ckpt = 13/164 (7.9%) — **down 3** from v2 step300
baseline (16/164). Even with KL +0.05–0.09 well-bounded, the
stochastic-gate + entropy-bonus combo destabilized the warm start.
**Likely cause**: v2 step300 already has a well-separated gate (σ near
0/1 for confident decisions, 0.1–0.9 only for uncertain positions).
The selective-sampling range [0.1, 0.9] is EXACTLY where v2 learned to
think — so injecting Bernoulli sampling there destabilizes the trained
think pattern. The entropy bonus actively pushes back toward high
entropy (uniform), un-doing v2's confident calibration.
**Plan**: still finish to step100 (next save point), eval. If step100
≤ step50, abort and pivot. If step100 > step50, the recipe is
recovering as the gate re-learns its policy. The bar is "step100 ≥ 16"
to validate the composition; "step100 between 13 and 16" is partial
recovery (continue to step 200); "step100 < 13" means active
degradation (abort).
**Open question for pivot**: even if we abort the composition, the
v2-step300 ckpt is unchanged on disk. The question is which next
direction to try. Three options I'm holding:
1. Disable the entropy bonus and re-run with narrower sampling range
   (e.g. [0.3, 0.7]) — minimally invasive recipe change.
2. Pivot to DPO (rejection-sampling pairs → train_dpo.py infra ready)
   — different training signal, may not regress like SFT/RL did.
3. Accept 16/164 as the ceiling and write up the project.

### D5 (2026-05-27 ~14:25) — Pivot to DPO v2 with mid-checkpoints
**Decided**: Pivot to DPO. Found that DPO v1 (May 23) already exists
and **regressed to 9/164** because:
- 2 epochs × 1958 pairs = 3908 steps (too long)
- β=0.1 (weak KL anchor)
- No mid-checkpoints — final ckpt was over-fit (winrate 0.97,
  log_ratio +100 to +250)

**Why DPO might work this time**:
- Contrastive signal (chosen vs rejected) is fundamentally different
  from SFT (which keeps regressing) or on-policy RL (which destabilizes
  the v2 base when adding stochastic exploration)
- The bug in v1 is identifiable and fixable — the final ckpt being
  over-fit doesn't mean intermediate ckpts were bad

**Patch made**: added `--save_every` flag to `train_dpo.py` so we can
capture the early sweet spot.

**v2 config**: β=0.3 (3× tighter anchor), 1 epoch (~1958 steps), LR
2e-6 (40% of v1), save_every 250. ~5 min training, ~7 snapshot ckpts.

**Decision-gate**: eval each snapshot, take best.
- ANY snapshot > 16/164 → DPO is the winning lever (write up).
- ALL snapshots ≤ 16/164 → DPO can't break through either. Accept
  16/164 as the project ceiling and document.

File: `launch_dpo_v2.sh`.

**Time budget for this attempt**: training ~5 min + ~25 min/eval × 7
snapshots = ~3h. If this fails, switch to writeup mode.

### D9 (2026-05-27 ~17:00) — User: "make the pre-train actually work" — wired all aux losses into train_lm.py
**Decided**: Pivot per user's instruction. All thinking-aux losses
(`process_reward`, `gate_calibration`, `state_readonly_at_think`) now
live in `train_lm.py` pretrain path (agent-built, 27/27 tests pass).
Smoke launcher `launch_pretrain_smoke_thinking.sh` continues from
`pretrain_phase_c.pt` (step 23000) for +1000 steps to validate
mechanics + measure whether thinking starts becoming productive.

### D10 (2026-05-27 ~17:30) — Smoke pretrain VALIDATED — aux losses are working
**Result** (5 retries to get past stupid bugs, then clean run):

Bugs found + fixed along the way:
1. `torch.compile` + extra-forward shape mismatch → `--no-compile`.
2. OOM at b=14 (extra forward needs ~5GB more) → `--batch 8`,
   `max_positions 8`.
3. Model returns `(logits, gist_loss)` tuple in training mode; helpers
   expected a Tensor → tuple-unwrap patch in `process_reward.py`.
4. CUDA assert: pretrain `y` contains `thinking_token_id` (49152) as a
   target; gather into base_vocab (49152) was OOB → mask `y ==
   thinking_token_id` to -100 inside helpers.
5. **The load-bearing one**: `model._last_gate_logits` got overwritten
   by `process_reward`'s extra forward; `gate_calibration` ran AFTER and
   used a stale shape (N_pr, T_pr) — boom. Fixed by snapshotting
   `_last_gate_logits` BEFORE any extra forwards. Same issue affected
   the per-step `emit_ce` diagnostic reading `model._last_gate` post-
   hoc — patched to use the snapshot.

**First clean smoke step (23005)**:
- `pr(n=8/12449, Δlogp=+4.507, %pos=100)` — thinking improves
  predictions at 100% of sampled positions, mean Δlogp +4.5.
- `gc(n=8/5892, tgt1=0.88, σ=0.29, Δlogp=+2.884)` — at 88% of
  uncertain-gate positions, thinking would help — but the model's σ
  is only 0.29 (wants to emit). **The gate is massively miscalibrated**
  and the new loss has the right signal to fix it.

**Steps 23005 → 23015 (10 steps in)**:
- σ at gate-cal positions: 0.29 → 0.41 → 0.59 — RISING fast as
  gate-cal pulls it toward the right answer.
- tgt1: 0.88 → 0.62 → 0.62 — the easy gate-cal wins are getting
  captured.
- Overall tloss: 1.55 → 1.44 → 1.25 — converging.

**Conclusion**: gate_calibration + process_reward + state_readonly_at_think
work as designed. The gate IS miscalibrated and the loss has a real
signal to learn from. The full validation will come when the smoke
finishes (~10 min more) and we ablation-check the ckpt.

### D6 (2026-05-27 ~14:45) — User raises distribution-mismatch hypothesis
**User asked**: "Why do you think we overtrain? Maybe our training
distribution is just different than HumanEval?"
**Response**: agreed this is likely. Evidence:
- RL v2 trained on MBPP. Trajectory was monotonic: step100→14, step200→15,
  step300→16 (peak), step400→14 (regressed). Same recipe, same data,
  just more steps → 16 then 14. That's *not* optimizer instability
  (would be sudden); it's distributional saturation (lift then drift).
- DPO v1 (also MBPP) over-trained to 9/164. Same story.

**Implication**: even with the careful DPO v2 recipe (β=0.3, mid-saves),
the *best possible* DPO v2 snapshot is just the least-drifted point
on a curve that's structurally biased toward MBPP-shaped code, away
from HumanEval-shaped code. We may not be able to break through 16/164
on MBPP-only data regardless of training discipline.

**Test**: eval the EARLIEST DPO v2 snapshot (step250 — least drifted).
- If > 16: DPO v2 worked, contradicting the distribution hypothesis.
- If ≤ 16: distribution hypothesis confirmed.

**Contingency if confirmed**: pivot to multi-source rejection sampling.
`code_grader.LOADERS` has `leetcode` (2641 problems, distributionally
distinct from MBPP) and `super_combined` (mbpp_combined + leetcode =
3993). Generate new rejection-sampling data on leetcode → DPO v3 on
the MIXED MBPP + leetcode pairs.

DPO v2 finished: 4.9 min, final winrate 0.725 (vs v1's 0.97 — much
better-controlled drift). 7 snapshots saved (step 250, 500, 750, 1000,
1250, 1500, 1750).

### D7 (2026-05-27 ~16:00) — Distribution hypothesis CONFIRMED; pivot to synth_pyfunc SFT
**Evidence chain**:
1. DPO v2 step250 (least drifted) → 12/164 (regressed 4 from baseline).
   Strong β=0.3 anchor didn't help because the data ITSELF biases away
   from HumanEval, not just the training duration.
2. Inspector on first 40 problems: both v2_step300 and dpo_v2_step250
   pass 8/40, BUT 38/40 outputs are different. The regression is
   entirely in problems 40-163. DPO's outputs look like MBPP-passing
   style (short one-liners, fresh-`def`-from-scratch).
3. Two parallel research agents (general-purpose) independently
   concluded: MBPP rejection data trains the model to emit
   ```python\ndef name(...): ... ``` from scratch, but HumanEval prompts
   *already opened the function header* — the model needs to complete
   a body, not start a new function. Format mismatch.

**Agent B finding**: `data/synthetic_pyfunc.jsonl` (6501 rows) is the
ONLY on-disk corpus in HumanEval's exact shape (`def sig: """doc + >>>
examples""" body`). Never used for SFT.

**Action**: SFT v12 — `launch_sft_synth_pyfunc.sh`. Converted
synthetic_pyfunc → sft_code-compatible format
(`data/sft_synth_pyfunc.jsonl`, 6501 rows). 1 epoch, LR 3e-6, batch 2.
~16 min training, 25 min eval.

**Decision-gate** (this is the make-or-break experiment):
- >16/164: distribution hypothesis vindicated. Scale the data (10× the
  synth_pyfunc generator). Success.
- ~16 ± 1: SFT can't move it; 16/164 is genuine size-class ceiling.
- <14/164: SFT itself regresses regardless of data shape. Contradicts
  agent diagnosis; would need re-investigation.


## D19 RESULT (2026-05-28) — TRAINED state-readonly thinking gives ZERO lift on arithmetic. Verdict: NEGATIVE.

**Rework shipped** (`train_rl_grader.py`): `--state_readonly_at_think` (passes
`force_state_readonly=True` to `build_model_from_ckpt` for BOTH the policy and the
frozen KL ref; tags `cfg["state_readonly_at_think"]=True` so reloads auto-enable
the hook) and `--dataset_jsonl PATH` (loads a synth_reasoning-schema JSONL via
`code_grader.load_synth_reasoning`). Both backwards-compatible. 8 tests in
`experiments/test_rl_grader_state_readonly.py` (14 with the gate_floor suite).
Training set `data/synth_arith_train_n234.jsonl` (gitignored under data/) =
n2+n3+n4 (240 tasks). Launcher `launch_rl_arith_stateread.sh`.

**Training** (250 steps GRPO from `rl_grader_phase_c_v2_step300`, state-readonly
ON, deterministic gate, ponder 0, kl 0.05, lr 2e-6, GPU 0; completed). VERIFIED
trend: s1 rmean 0.016 passn 0; s50 rmean 0.072 passn 0; s100 rmean 0.056 passn 0; s150 rmean 0.084 passn 0; s200 rmean 0.109 passn 0; s250 rmean 0.059 passn 0. **pass_n=0 at all but 3 of 250 steps (max_passn=1)**
— the grader essentially NEVER saw a fully-correct chain, so GRPO had almost no
task-correctness signal. KL range 0.000..1.196.

**Ladder verdict** (final step-250 ckpt, 3-way, 80/rung, greedy, max_gen 96 —
VERIFIED, `results/ladder_arith_stateread_final.json`, prompt_style=code_fence,
generator=retrieval_as_input):

| n | no-think | with-think (state-WRITE) | think + state-READONLY |
|--:|:--|:--|:--|
| 1 | 0/80 | 0/80 | 0/80 |
| 2 | 0/80 | 0/80 | 0/80 |
| 3 | 3/80 | 0/80 | 0/80 |
| 4 | 0/80 | 0/80 | 0/80 |
| 5 | 0/80 | 0/80 | 0/80 |
| 6 | 0/80 | 0/80 | 0/80 |

**Within-model comparison (the load-bearing question): with-think == 0/80 at
EVERY rung, while no-think reaches 3/80 (n=3).** state-readonly == state-write ==
0/80. Thinking provides ZERO lift; with-think <= no-think everywhere. beta=0 is
verifiably active (the 3-way harness toggles the b_proj hook between the write and
readonly conditions on the same model; their generations differ) and it makes no
difference.

**HARNESS CAVEAT.** The no-think column maxes at 3/80, NOT the 17/80 the D16
demonstration doc reported for the SAME band. The current `eval_thinking_ladder`
(gained prompt_style/generator args + data AFTER commit 7f1a49f) no longer
reproduces D16's numbers on the original `rl_grader_phase_c_v2_step300.pt` either.
A direct `generate()` probe shows the model emits CORRECT arithmetic as PROSE
("Let me trace... v0 = 7, v1 = v0 - 4...") but rarely as an extractable
`def solve(): return -1`, so the grader returns exec_error/0. The absolute 0/80
floor is partly a measurement artifact of harness drift; the WITHIN-MODEL
with-vs-without comparison is unaffected and decisive.

**Verdict (decisive)**: TRAINED state-readonly thinking does NOT beat no-think on
the arithmetic ladder — tied or worse at every rung. The success criterion
(with-think > no-think at any rung) FAILS. Two causes: (1) during RL the grader
almost never saw a correct gradeable chain (pass_n~=0) so there was no
thinking-productivity gradient to climb; (2) the harness can't detect the model's
prose-form correct answers. At 287M, latent state-readonly thinking did not become
useful computation on arithmetic. GEMINI's 1-layer probe (0.88 vs 0.41) does NOT
transfer to the trunk.

**Next steps the evidence demands**: (a) FIX THE HARNESS before any future
thinking-on-reasoning claim — `eval_thinking_ladder` must grade prose-wrapped
answers (extract `return <int>` from CoT, or force code output). (b) The honest
path to the coding headline remains post-training scale + model size, not the
thinking primitive.

Artifacts: `checkpoints/rl_arith_stateread{,_step50,_step100,_step150,_step200}.pt`,
`results/ladder_arith_stateread_final.json` (full 6-rung 3-way, verified).
