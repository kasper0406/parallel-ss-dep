# PLAN — Flaw C: replace the per-token-logp gate target with an execution-grounded objective

Owner: flaw C of `THINKING_AUDIT_2026_05_28.md` (the FATAL one). Read-only plan; no
code/training here.

## 0. The error in one line

`process_reward.py` / `compute_gate_calibration_loss` reward a think iff it raises
`logp(y[t+1])` — i.e. "sharpen the next surface token", which is near-free extra
compute, biased by self-fulfilling candidate selection, and structurally blind to
whether the emitted *function passes tests*. The fix is to make "did thinking help"
a **terminal grader-reward delta on the actually-generated completion**, decided as
an RL policy action. The repo already has the machinery: `train_rl_grader.py` +
`--stochastic_gate` + `code_grader.grade`.

## 1. The correct objective — precise design

We keep the audit's framing: "does thinking help" must be measured on the *completed
function*, by the verifier, under the *real inference generator*. There are two
viable mechanisms; we use **B as the trainer** and **A as the go/no-go diagnostic**
(section 4), because they answer the same question at different cost/variance.

### Mechanism B (the trainer) — per-gate-decision PPO from terminal reward

This is already 90% built in `train_rl_grader.py`. The gate decision is a Bernoulli
policy action; the *whole completion's* grader score is the return; GRPO credits both
the emitted tokens and the gate decisions with the **same group-relative advantage**.

Exact wiring against existing machinery:

1. Rollouts via `rollout_group_batched(..., stochastic_gate=True)`
   (`train_rl_grader.py:166`). At each step the gate's `p_emit = σ(gate)` drives a
   `torch.bernoulli` draw (line 282); at *uncertain* positions
   (`gate_sample_range_low < p_emit < gate_sample_range_high`, line 280) the draw is
   the action and `(gate_decisions, gate_log_probs, gate_positions)` are recorded
   (lines 358–361). `force_emit` positions (budget exhausted / finished) are excluded
   — no policy choice was made there (line 345). This is exactly right and needs no
   change.
2. Reward = `code_grader.grade(problem, completion).score ∈ [0,1]` (dense tier ladder,
   `code_grader.py:250/87`) on the **decoded emit stream** of each rollout — the same
   text the gate's think decisions actually produced. No `logp` anywhere.
3. Advantage = GRPO group-relative over the `(B, N)` reward tensor
   (`compute_grpo_advantages_from_rewards`, line 553). Because thinking and
   not-thinking rollouts land in the *same group* (same problem), a rollout that
   thought-and-passed gets positive advantage vs a sibling that didn't-think-and-failed
   — that **is** the "did thinking help here, on this task" signal, terminal and
   execution-grounded.
4. Credit assignment to the gate: the gate-PPO surrogate already exists
   (`compute_grpo_loss_batched`, lines 682–711). `new_p_at_dec = σ(gate)` at the
   decision positions, `g_ratio = exp(new_gate_lp − old_gate_lp)`, clipped PPO surrogate
   weighted by the *same* `adv`. High-reward rollout → push P(what the gate chose) up;
   low-reward → push it down. This is the load-bearing line and it is correct.
5. Entropy floor: `--gate_entropy_bonus` (default 0.01, line 982) prevents collapse to
   always-think / never-think. Keep it; optionally use the curriculum
   (`--gate_entropy_bonus_start/_end/_curriculum_steps`) to anneal exploration down.

What this gives that the logp proxy cannot: the gate is rewarded for thinking **only
when thinking changed the trajectory toward a passing function**, and punished when an
extra think burned budget / corrupted recurrent state (the documented "thinking
corrupts recall" failure) into a failing function. The proxy's three pathologies
(surface-token sharpening, free-compute confound, no terminal credit) are all gone
because the objective is the terminal verifier, not a local distribution.

### Mechanism A (diagnostic only) — counterfactual reward delta

For a problem, generate with the *real* `generate_with_retrieval_as_input` generator
TWICE: once with the gate's learned think budget, once forced never-think
(`emit_threshold = 1.0` so the gate can never fire). Report
`Δreward = grade(with_think) − grade(no_think)`. This is the honest population
estimate of "does the current gate policy's thinking help the task." It is too
high-variance / 2× cost to train on directly, but it is the right *measurement* — see
section 4.

## 2. Concrete experiment plan

Two runs, in order. Both reuse the validated v2 stability recipe
(`launch_rl_grader_phase_c_v2.sh`) and only add the stochastic gate as a policy var.

### Base ckpt choice — start from `sft_phase_c_combined.pt` (10/164), NOT v2-step300.

Rationale: `rl_grader_phase_c_v2_step300` (16/164) was trained with a **deterministic,
never-think gate** (`--gate_floor 0.0 --emit_threshold 0.5`, no `--stochastic_gate`;
note the repo's pinned footgun: `gate_floor < emit_threshold` is required or thinking
is silently off). Its gate is already shaped by 300 steps of RL that never exercised
thinking, so it is a poor starting policy for *learning* a thinking gate. The SFT base
is the cleaner substrate: its gate still has the SFT-distilled `think_rate ≈ 0.33` on
code, so the Bernoulli policy starts from a non-degenerate prior. Use v2-step300 only
as a **secondary** run to test "can we recover thinking on top of the best ckpt."

### Run C1 — primary, stochastic-gate RL from SFT base

```
--load_ckpt checkpoints/sft_phase_c_combined.pt
--dataset mbpp_combined --extract_code_block
--stochastic_gate
--gate_sample_range_low 0.1 --gate_sample_range_high 0.9   # only explore uncertain gates
--gate_entropy_bonus 0.02 --gate_entropy_bonus_start 0.03 --gate_entropy_bonus_end 0.0 \
  --gate_entropy_curriculum_steps 200                      # anneal exploration
--gate_floor 0.0 --emit_threshold 0.5                      # gate_floor < emit_threshold (pinned rule)
--kl_coef 0.05                                             # v2 KL-to-SFT-ref stability (frozen ref = load_ckpt)
--lr 2e-6 --clip_eps 0.1 --temperature 0.7                 # v2 values
--ponder_cost 0.0                                          # v2 lesson: depth pressure triggered the collapse
--steps 400 --batch 4 --grpo_n_group 4 --max_gen 384 --save_every 50
--max_think_per_step 4 --total_think_budget 120 --min_emit_before_eos 30
```

Why each deviation from v2: only `--stochastic_gate` + `--gate_sample_range_*` +
entropy curriculum are new. Everything else is the *validated* monotonic-climb recipe.
We deliberately keep `--ponder_cost 0.0` (v2's hard-won lesson — depth pressure caused
v1's catastrophic collapse) so depth cost cannot confound the first test of whether the
*terminal* signal alone shapes the gate. If C1 over-thinks (depth → budget), re-run with
a small warmed-in `--ponder_cost 0.001 --ponder_warmup_steps 100`.

### Run C2 — secondary, same config from `rl_grader_phase_c_v2_step300.pt`

Tests recoverability of thinking on the current best ckpt. Same flags; `kl_coef` ref =
v2-step300 itself. If C2 > 16/164 with non-trivial `gate_fire_rate`, thinking is a
net positive lever on the headline; if it lands at 16/164 with fire-rate → 0, the gate
correctly learned "thinking doesn't help these short problems" — itself a clean,
publishable negative that vindicates dropping the logp proxy.

### Eval per `--save_every 50` ckpt

`eval_humaneval.py --prompt_style sft_comment --extract_code_block` (the mandatory
distilled-ckpt flags) + `eval_longctx_recall.py` (the recall probe where thinking is
supposed to help/hurt). Track HumanEval pass@1 AND `gate_fire_rate` / `depth_mean`
jointly — the success criterion is **HumanEval ≥ 16/164 with a non-degenerate,
task-adaptive fire rate**, not pass@1 alone.

## 3. Pretrain-time gate signal — keep entropy-prior, drop logp aux entirely

User priority is *pretrain-time thinking usefulness*. Assessment:

- **Drop `--process_reward_weight` and `--gate_calibration_weight` at pretrain.** They
  are the flaw-C objective; section 0 applies in full. They also cost a capacity-
  competing extra forward and drift VAL ppl up 3–5% (audit flaw E).
- **Keep `--gate_entropy_aux_weight` (the audit's salvage, already validated 2026-05-17).**
  It supervises the gate logit toward `exp(−H_t/T)` from the *same* forward's entropy —
  zero extra forwards, no self-fulfilling candidate set. Crucially it is *honest about
  what it is*: a "think where the model is locally uncertain" **prior**, not a
  "thinking helps the task" claim. Flaw C.2 of the audit explicitly notes the logp
  proxy was *secretly* just encoding this same uncertainty signal — so use the cheap
  honest version directly. Recommended: `--gate_entropy_aux_weight 0.1
  --gate_entropy_aux_temperature 2.0`.
- **Can entropy-prior + RL refinement give the user what they want? Yes — this is the
  correct division of labor.** Pretrain's entropy aux gives the gate a sane *initial
  policy* (fire on uncertain positions) so RL doesn't start from a degenerate gate.
  Then RL (section 2) refines *which* uncertain positions are worth a think using the
  *terminal* signal. Pretrain cannot itself know task-helpfulness (no verifier in the
  loop, no completed function), so asking pretrain to learn "useful thinking" was the
  category error. Pretrain learns "uncertain ⇒ candidate-to-think"; RL learns "of those,
  these actually help." This is the only factoring consistent with the audit.

## 4. The go/no-go diagnostic — BEFORE committing RL compute

Build on `probe_thinking_counterfactual.py` (it already imports `generate`,
`_run_test_in_subprocess`, `_truncate_at_stop`). New mode (or sibling script
`probe_gate_grader_delta.py`):

1. **Sample positions/problems by the INFERENCE gate policy**, not by `σ ∈ [low,high]`.
   Run the *actual* `generate_with_retrieval_as_input` (the real deploy generator,
   additive-α, iterative re-decode) — never the K-at-once forced forward that flaw B
   showed is a different distribution.
2. For each HumanEval/MBPP problem, generate twice: (a) gate's natural think policy,
   (b) forced never-think (`emit_threshold = 1.0`). Grade both with `code_grader.grade`.
3. Report **Δ(grader reward) = mean(reward_think − reward_nothink)** over the eval set,
   broken down by `depth` bucket. NOT Δlogp.

Go/no-go gate: **only launch C1/C2 if Δ(grader reward) is not significantly negative on
the current SFT ckpt** (i.e. the existing gate's thinking is at worst neutral). The
audit predicts this number is `≤ 0` today — which is exactly the signal that the logp
proxy was optimizing the wrong thing. If Δ ≤ 0, that confirms the diagnosis and C1 is
*still* the right move (RL is what fixes a ≤0 gate); but the diagnostic must be the
*reported metric throughout RL* so we can see thinking become net-positive (Δ crossing
0) rather than trusting pass@1 noise. Run this on `sft_phase_c_combined.pt` and
`rl_grader_phase_c_v2_step300.pt` first — ~1 GPU-hour, gates a multi-hour RL run.

## 5. Dependencies / overlaps with flaws A, B, D, E

Dropping the logp proxy (this plan) moots most of the other flaws:

- **A (pretrain calibrates discrete-think, SFT deploys retrieval-as-input, never
  re-calibrated): MOOT.** There is no calibration loss to mis-stage. The gate is shaped
  end-to-end by RL on the *exact deploy generator* (`generate_with_retrieval_as_input`,
  additive-α, iterative). Train==deploy by construction.
- **B (K-at-once forced forward vs iterative re-decode; `state_readonly_at_think`
  on-in-pretrain/off-in-SFT): MOOT for the gate objective.** The reward comes from real
  iterative rollouts, so there is no K-at-once "after" forward to diverge. The one
  residual: keep the `state_readonly_at_think` setting **identical** between the RL
  rollout/loss path and the eval generator (verify both use the SFT-trained value —
  SFT did not set it, so deploy thinks write to state; RL must match). This is a
  consistency check, not a separate fix.
- **C (this plan): the root cause; replaced wholesale.**
- **D (self-fulfilling in-distribution diagnostics measured on the loss's own candidate
  set): MOOT.** Section 4's diagnostic samples by the *inference policy* and reports
  terminal Δreward on a held-out eval set — the policy-distributed control the audit
  said was missing. No `σ ∈ [low,high]` conditioning anywhere.
- **E (capacity competition + always-think drift): ADDRESSED, not moot.** RL still
  *can* over-think; that is now *bounded by the terminal reward itself* (a think that
  doesn't help the function earns no advantage) plus `--gate_entropy_bonus` (anti-
  collapse) and optional `--ponder_cost` (explicit depth pressure, used cautiously per
  v1's collapse lesson). The KL-to-SFT-ref (`--kl_coef 0.05`) prevents the v1-style
  drift off the output format. So E becomes a *tunable* (entropy/ponder/KL), not a
  structural pathology.

Net: **drop the per-token-logp objective at both pretrain and SFT.** Keep only the
honest, cheap entropy *prior* at pretrain; let execution-grounded GRPO with a
stochastic gate (sections 1–2) decide task-helpfulness; gate the compute on the
terminal-Δreward diagnostic (section 4). Flaws A/B/D collapse to consistency checks; E
becomes a knob.
