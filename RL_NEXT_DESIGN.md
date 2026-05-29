# RL_NEXT_DESIGN — Multi-turn agentic execution-grounded RL

**Status**: design proposal (2026-05-29). No training code written. Grounded in
`experiments/train_rl_grader.py`, `experiments/code_grader.py`,
`experiments/iterative_repair.py`, `experiments/curriculum.py`,
`experiments/thinking.py`, `experiments/latent_think.py`, `experiments/model.py`,
`experiments/gen_rejection_data.py`.

**TL;DR of the project state this builds on.** Current best is **16/164 HumanEval**
via KL-stable grader-GRPO (`train_rl_grader.py` v2, step-300). That trainer
*already* contains four of the pieces a naive reader might propose as "new":

- a **dense reward** (`code_grader.grade` → tiered `score ∈ [0,1]`, not binary),
- a **difficulty curriculum** (`curriculum.ProblemDifficultyEMA`, three modes
  incl. closed-loop adaptive),
- **single-turn iterative repair** (`iterative_repair.build_repair_prompt` +
  `select_repair_targets` + `group_became_variance_bearing`, wired behind
  `--iterative_repair`),
- **KL-to-frozen-reference** (`--kl_coef`, Schulman k3 estimator) and a
  **stochastic gate** policy variable (`--stochastic_gate`).

So this proposal is deliberately **incremental**: it does *not* rewrite the
trainer. It (a) closes the remaining zero-variance gradient holes that the
existing single-turn repair only half-fixes, (b) generalizes the one-shot
repair into a genuine **multi-turn trajectory** with trajectory-level credit
assignment, and (c) wires in the one mechanism that demonstrably performs useful
sequential computation — **latent-space thinking** (`latent_think.py`,
THINKING_LATENT_2026_05_28) — as the inter-turn "reflect on the error" step,
replacing discrete CoT which never helped.

---

## 1. Zero-variance diagnosis & guarantees

### 1.1 Exactly when the GRPO gradient vanishes

The advantage is computed in
`train_rl_grader.compute_grpo_advantages_from_rewards`:

```
full_reward = clamp_min(reward, 0) - ponder_cost·f(depth)      # counterfactual
adv = (full_reward - mean) / (std + 1e-8)
```

and consumed per emit-token in `policy_loss_for_rollouts_batched` as
`surr = -min(ratio·adv, clip(ratio)·adv)`.

The gradient through a *group* (one problem's N rollouts) is exactly zero when
the per-group `std` of `full_reward` is ≈0 (the `+1e-8` makes `adv ≈ 0` for all
N rollouts). This happens in three regimes:

1. **All rollouts land on the same tier with the same n_passed** — e.g. every
   one of N=4 rollouts is `syntax_error` (score 0.0). This is the user's
   reported failure: too-hard problem → uniform failure → no signal. The dense
   ladder in `code_grader._TIER_BASE_SCORE` does *not* help here because the
   variance is *across* rollouts within a problem, and all rollouts collapsed to
   the same rung.
2. **All rollouts pass** (score 1.0) — saturated/too-easy problem. Also zero
   variance, also no gradient. (The `counterfactual=True` clamp does not change
   this — all `full_reward` differ only by the depth term, which the curriculum
   should already be steering away from.)
3. **Degenerate depth-only variance**: if task rewards are identical but depths
   differ, the *only* surviving signal is the ponder cost `f(depth)`. At
   `--ponder_cost 0.001` quadratic this is tiny and z-scoring amplifies pure
   noise — the model gets trained to minimize thinking on a problem it can't
   solve, which is actively harmful.

The existing `curriculum.ProblemDifficultyEMA` attacks regimes 1 & 2 *in
expectation* by up-weighting problems whose EMA pass-rate is near 0.5
(`4·p·(1-p)`, or the adaptive Gaussian). But this is a *sampling-time*
heuristic on a *lagged* EMA: a problem with true p≈0.3 will still produce
all-fail groups a large fraction of the time at N=4 (P(all 4 fail) = 0.7⁴ ≈
0.24). So ~1 in 4 sampled groups is wasted even under a perfect curriculum.
The existing `--iterative_repair` recovers *some* of these (it logs `lift` =
fraction of zero-var groups that became variance-bearing after repair), but it
is one-shot and only fires on `min_failed` failures.

### 1.2 Guarantees to adopt (ranked by leverage)

**ADOPT — (A) Improvement-based reward inside a multi-turn trajectory** (the
headline mechanism, detailed in §2). The decisive structural fix: instead of
scoring each rollout by its *final* score, score a *trajectory* of T turns by
**per-turn improvement** `Δ_t = score(code_t) − score(code_{t-1})`. A problem
where the first attempt is `syntax_error` (0.0) and the repaired attempt is
`exec_error` (0.05) now yields a *positive within-trajectory* signal even though
both attempts "fail". Crucially this breaks the across-rollout tie: two
trajectories that both end at `exec_error` are distinguishable if one got there
by *fixing a syntax error* (informative reasoning) and the other started there.
This converts a too-hard one-shot problem into a sequence of locally-learnable
steps. **This is the primary recommendation.**

**ADOPT — (B) Tie-break reward shaping with a continuous diagnostic feature.**
Even within one tier, add a small continuous term so two identical-tier rollouts
differ. Cheap options that need no extra forward/grade:
- `+ ε · (n_passed / max(1, n_tests))` (already partially in the `partial`
  tier, but `syntax_error`/`exec_error` collapse n_passed=0; extend the feature
  to **negative-log edit-distance to the nearest gold token-prefix** or simply
  **−ε·len(error_text)** as a "how broken" proxy). Recommend the trivial one:
  reward `+= 1e-3 · (1 − min(1, len(error_text)/1000))` so "almost compiles"
  beats "totally garbled". This is a *tie-breaker only* (ε small) — its job is
  to give `std>0`, not to dominate. **Adopt as a safety net under (A).**

**ADOPT — (C) Filter/oversample zero-variance groups before the update.**
After grading, compute per-group `std`. For groups with `std < τ_var` (e.g.
1e-3): (i) **exclude them from the policy loss** (they contribute only noise
through `1e-8`), and (ii) **re-draw replacement problems** to keep the effective
batch full, OR roll the freed compute into more turns on the variance-bearing
groups. The trainer already skips empty groups; extend the skip to
low-variance groups. **This is cheap and strictly removes harmful noise** — do
it regardless of the rest. Implementation: a `_group_is_variance_bearing(rewards,
tau)` predicate gating `per_group_advantages` accumulation.

**ADOPT (already present, keep) — (D) Curriculum.** Keep
`--adaptive_curriculum` (closed-loop) on. It is the right *prior* over which
problems to sample. Do **not** rely on it as the *guarantee* — it is lagged and
probabilistic. (A)+(C) are the guarantee; (D) raises the hit rate.

**DROP / DE-PRIORITIZE — pure problem-difficulty gating to "only sample where
some rollout passes."** Tempting, but at 287M the variance-bearing band is
narrow and shifts every few steps; aggressively gating to it collapses the
training distribution onto a handful of lucky mid-difficulty problems (this is
exactly the documented v7b failure — "drifted into a narrow band of lucky
middle-difficulty problems"). The multi-turn Δ-reward (A) is what lets us keep
sampling genuinely hard problems and *still* get gradient.

**Net guarantee.** With (A) every trajectory on a non-trivial problem produces a
non-zero within-trajectory advantage as long as *any* turn changes the score
(syntax→exec→runtime→partial are all reachable single-edit transitions). With
(C) the residual truly-flat groups are dropped rather than injecting noise. (B)
is the last-resort tie-break. This is a hard guarantee of "no wasted step" only
in the (C) sense (we never *update on* zero signal); (A) makes the wasted
fraction small rather than zero.

---

## 2. Multi-turn / agentic loop design

### 2.1 Turn structure

```
turn 0:  prompt_0 = build_mbpp_prompt(problem)
         code_0   = rollout(prompt_0)                    # latent-think allowed (§4)
         g_0      = grade(problem, code_0)               # tier, score, error_text
         if g_0.tier == "pass": STOP (solved)
turn 1:  prompt_1 = build_repair_prompt(prompt_0, code_0, g_0.error_text)
         code_1   = rollout(prompt_1)                    # may "reflect" via latent think first
         g_1      = grade(...)
         ...
turn t:  stop when g_t.tier == "pass" OR t == max_turns-1
```

`build_repair_prompt` (already in `iterative_repair.py`) is the feedback
injection: it formats `error_text` as `# `-prefixed comment lines appended to
the original prompt + the failed code + `# Fix the code:`. This is reused
verbatim. **Context growth is the cost driver** (§6); cap at `max_turns` and cap
the carried context to the *latest* failed attempt + its error (not the full
history) so prompt length stays bounded ≈ 2× the single-shot prompt.

**`max_turns`: start at 3.** Empirically (and per the StaR/Reflexion
literature at small scale) most recoverable gains are turn-1; turn-2 has sharply
diminishing returns and turn-3+ mostly re-emits. 3 turns × N=4 rollouts is the
compute ceiling we can afford (§6). Make it a CLI flag `--max_turns` (default 1
= byte-identical to today's single-shot; 3 = the new mode).

### 2.2 Reward & credit assignment

Per **trajectory** (one rollout's sequence of turns), define:

```
R_traj = score(code_final)                              # terminal task reward
       + λ_improve · Σ_t max(0, Δ_t)                    # shaped per-turn improvement
       − ponder_cost · f(total_depth across turns)      # existing depth penalty
       − turn_cost · (n_turns − 1)                      # NEW: each extra turn costs
```

with `Δ_t = score(code_t) − score(code_{t-1})`, `λ_improve ≈ 0.3` (so a full
syntax→pass climb over 3 turns is worth ~0.3 on top of the terminal 1.0, never
dominating it), and `turn_cost ≈ 0.02` (a gentle pressure to solve in fewer
turns — the analogue of ponder cost at the turn granularity). Clamping Δ at
`max(0, ·)` means a turn that *regresses* the score isn't punished on the task
side (mirrors the existing `counterfactual` clamp philosophy) but also isn't
rewarded.

**Credit assignment across turns.** Two viable schemes; recommend the simpler:

- **(RECOMMENDED) Trajectory-flat assignment.** Compute one scalar `R_traj` per
  trajectory, group-normalize across the N trajectories of a problem (exactly as
  `compute_grpo_advantages_from_rewards` does today), and apply that single
  advantage to **every emit token in every turn of that trajectory**. This is
  the GRPO-native choice: it reuses `policy_loss_for_rollouts_batched` unchanged
  if we concatenate all turns' emit tokens into one rollout record. Coarse
  credit, but GRPO's group-normalization already throws away fine-grained
  per-token credit, so we lose little and keep the code path identical.
- **(OPTIONAL, phase 3) Per-turn advantage.** Treat each (turn → grade)
  transition as its own reward `Δ_t` and assign it only to the tokens emitted in
  that turn. This is closer to true PPO with intermediate rewards and gives
  sharper credit ("the tokens that fixed the syntax error get the +0.05"), but
  needs a per-turn baseline (the group mean of `Δ_t` at that turn index) and is
  more code. Defer until trajectory-flat is validated.

### 2.3 Why this structurally solves "all rollouts fail"

Under single-shot GRPO, a problem where all N first attempts are `syntax_error`
gives `std=0` → no gradient. Under the multi-turn loop:
- Each rollout gets a *second and third attempt with the compiler's error in
  context*. Different rollouts will fix the syntax error to different degrees
  (some reach `exec_error`, some `partial`), so the **terminal scores diverge**
  → group `std > 0` → gradient. Even if all terminals still tie at
  `exec_error`, the **per-turn Δ differs** (one fixed syntax in turn 1, another
  in turn 2) → `R_traj` differs via the `λ_improve` term.
- The model is trained on the *act of using feedback to improve*, which is the
  transferable skill HumanEval (single-shot at eval) ultimately needs:
  in-context error-correction is a strong prior even for first attempts.

The honest caveat: a problem so hard that **every turn of every rollout stays
syntax_error** still produces zero variance. That residual is caught by filter
(C) — we drop it — and reduced by the curriculum (D). Multi-turn shrinks this
set dramatically (single-edit syntax fixes are the *easiest* thing for a
code model to do in-context) but does not eliminate it.

---

## 3. AlphaZero-style elements — borrow vs drop

**DROP — MCTS over code tokens.** Token-level tree search needs a value network
and thousands of simulations per move; at 287M on 2×5090 with ~30ms/token decode
this is 2–3 orders of magnitude too expensive and the branching factor (vocab
49k) is hopeless. Not viable. Do not attempt.

**DROP — a learned value network / critic.** GRPO's whole appeal here is that
the group mean *is* the baseline — no critic to train. Adding a value head
doubles the parameters that need RL signal and is exactly the kind of
post-pretrain module that this repo has repeatedly found stays inert
(CLAUDE.md: "features added post-pretrain stay inert"). Keep the critic-free
group-baseline.

**BORROW — best-of-N branching with feedback = the multi-turn loop itself (§2).**
This *is* the pragmatic AlphaZero analogue: N parallel rollouts (breadth) ×
T turns of feedback-guided revision (depth), with the grader as a cheap perfect
"environment model". It is shallow tree search where expansion is guided by
real execution feedback rather than a learned value. This is the subset to
adopt.

**BORROW (cheap, phase 3) — value-guided turn selection / branch expansion.**
Instead of revising *every* failed rollout, spend the turn budget on the most
*promising* partial solutions: expand the rollouts with the **highest current
`score`** (closest to passing — `partial` with 2/3 tests beats `syntax_error`),
and **prune** the hopeless ones (stuck at `syntax_error` after turn 1). This is
`select_repair_targets` generalized from "repair the failures" to "expand the
near-misses". Uses the dense score *as* the value estimate — no learned critic.
Concretely: at each turn, rank the N rollouts by `score`, give extra turns to
the top-k, drop the bottom-k. Cheap, principled, defer to phase 3.

**BORROW — self-play-style difficulty escalation = the existing adaptive
curriculum.** `ProblemDifficultyEMA(adaptive=True)` already does the
"automatically track the capability frontier" thing AlphaZero gets from self-
play. Keep it. No new mechanism needed.

**Verdict.** The pragmatic AlphaZero subset is exactly: **(breadth = N GRPO
rollouts) × (depth = T feedback turns) + (dense score as a no-train value for
branch selection) + (adaptive curriculum as self-play difficulty).** Everything
requiring a learned value or token-tree search is dropped.

---

## 4. How thinking + memory participate

### 4.1 Latent thinking is the inter-turn "reflect on the error" step

The repo's hard-won result (THINKING_LATENT_2026_05_28, `latent_think.py`):
**discrete-token CoT never helped on any task (0/80 every arithmetic rung), but
latent-space thinking — feeding the trunk's own continuous hidden state back as
the next input embedding, state-readonly (DeltaNet β=0) — performs real
sequential computation and solved the exact tasks discrete thinking failed.**

Therefore the inter-turn reflection should be **latent, not a CoT prose
explanation**. Concretely, between receiving `error_text` and emitting `code_t`,
run R latent-think iterations at an appended think slot:
- `model.forward(..., inputs_embeds=...)` already supports feeding a continuous
  vector as the next input (used by retrieval-as-input and by
  `latent_think.think_forward`).
- `state_readonly_at_think=True` (already a ckpt-config flag, auto-detected by
  `build_model_from_ckpt`, and already plumbed into `train_rl_grader` via
  `--state_readonly_at_think`) ensures the latent reflection **reads** the
  prompt+error+failed-code recurrent state but never **writes** to it, so it
  can't corrupt the bindings the subsequent code generation needs (the
  documented "thinking corrupts recall" failure mode).

So the per-turn structure is: `[prompt_t | error_t | failed_code_t]` → **R latent
think steps (state-readonly)** → emit `code_t`. The R latent steps are where the
model "thinks about the bug". This is the first time the *working* thinking
mechanism is placed where it has a concrete, gradeable consequence (did the
revision improve the score?), which is precisely the signal latent-think needs
to learn productivity (it learns from final-answer-only + curriculum per the
ablations).

### 4.2 The stochastic gate gets RL signal from turn outcomes

`--stochastic_gate` already makes emit-vs-think a Bernoulli policy action with
its own PPO surrogate sharing the group advantage
(`policy_loss_for_rollouts_batched`, the `all_gate_surrs` block). In the
multi-turn loop the *same* trajectory advantage `R_traj` flows to those gate
decisions. The new, sharper signal: **a trajectory that thought (latent) before
a revision and improved the score gets positive advantage on its think
decisions; one that thought and didn't improve gets negative.** This is the
cleanest reward the gate has ever had — at last "did thinking pay off" is a
*measured execution outcome*, not a proxy logit. Keep `--gate_entropy_bonus` to
prevent collapse, and `gate_floor < emit_threshold` (the
`test_rl_grader_gate_floor.py` invariant).

Note the open finding (CLAUDE.md): the gate is *temperature-fragile* at τ=0.9.
The multi-turn Δ-reward gives it far more, and more directional, gradient than
single-shot, which is the best available robustification short of retraining the
gate. Monitor `gate(fire=, H=)`; if it still collapses bimodally, that is a
phase-gate failure (see §7) and the fallback is the **soft-mixture / continuous
think** head, not more knobs.

### 4.3 WorkingMemory across turns

**Recommendation: do NOT try to carry the WM buffer state across turns in
phase 1–2.** Reasons:
- WM reads at think positions within a *single forward* of the concatenated
  `[prompt | error | code]` context. Because each turn re-forwards the full
  (bounded) context, the relevant early bindings (the problem spec, the prior
  error) are *already re-read* into the recurrent state and WM buffer on each
  turn. We get cross-turn information flow "for free" through the context, not
  through a persisted buffer.
- Persisting the FLA cache + WM buffer across turns while *also* appending new
  feedback text is exactly the kind of stateful-decode complexity that has bitten
  this repo (cross-doc isolation, FiLM lag, the unfinished `forward_step`
  wiring). Not worth it for a second-order gain.
- PKM (read every token) participates automatically and unchanged — it is a
  static side-table, already load-bearing on HumanEval (`pkm_off −5`), and needs
  no per-turn special handling.

Defer cross-turn WM persistence to a later optional phase only if a diagnostic
shows WM reads on turn ≥1 are starved.

---

## 5. Integration path with `train_rl_grader.py`

**Strong preference: extend, do not rewrite.** The trainer already has the
batched rollout, batched grading (`grade_in_parallel`), per-group ragged
advantage computation, the batched PPO loss with KL + stochastic gate, DDP
plumbing, curriculum, and checkpoint/resume. The single-turn `--iterative_repair`
path is *already a one-turn special case of the multi-turn loop* — generalize it
in place.

### 5.1 What is reused unchanged
- `rollout_group_batched(...)` — one batched forward of N rollouts. Called once
  **per turn** instead of once per problem.
- `code_grader.grade` / `grade_in_parallel` — graded after every turn.
- `iterative_repair.build_repair_prompt` — the feedback-injection prompt
  builder, reused verbatim as the turn-(t→t+1) prompt constructor.
- `compute_grpo_advantages_from_rewards` — fed the trajectory reward `R_traj`
  per rollout (a (1,N) tensor per group) exactly as today.
- `policy_loss_for_rollouts_batched` — **unchanged** under trajectory-flat
  credit (§2.2): we hand it one `Rollout` per trajectory whose `emit_positions`
  / `emit_token_ids` / `full_ids` are the **concatenation across turns**.
- `ProblemDifficultyEMA` (updated with **turn-0 first-pass** reward only, never
  later turns — same rule the code already enforces for repair).
- KL, stochastic-gate, curriculum, DDP, ckpt/resume code.

### 5.2 What is extended
- A new `--max_turns` flag (default 1 = exact current behavior). When >1, the
  per-problem block runs the turn loop in §2.1.
- `--lambda_improve`, `--turn_cost`, `--group_var_floor` (filter C),
  `--latent_reflect_steps R` flags.
- `build_repair_prompt` gains an optional `carry_history: bool` (default False =
  only latest attempt) so context stays bounded.
- The curriculum EMA update already uses `first_pass_rewards_per_group`; keep
  that — turn-0 score is the model's true unprompted difficulty.

### 5.3 New data structures

```python
@dataclass
class Turn:
    prompt_ids: list[int]         # the (problem | feedback) prompt for this turn
    rollout: Rollout              # reuse the existing Rollout (emit ids/logps/positions,
                                  #   depth, gate_* fields, latent-think bookkeeping)
    score: float                  # grade score AFTER this turn
    tier: str
    error_text: str | None

@dataclass
class Trajectory:
    problem_id: str
    turns: list[Turn]
    R_traj: float = 0.0           # computed terminal+shaped reward (§2.2)
    # For the policy update we flatten turns into ONE pseudo-Rollout:
    def to_flat_rollout(self) -> Rollout: ...   # concat emit_* and gate_* across turns,
                                                #   re-offsetting emit_positions into the
                                                #   concatenated id stream
```

`to_flat_rollout` is the only genuinely new logic in the hot path: it
concatenates each turn's `full_ids` (each turn re-forwards from its own prompt,
so positions are *per-turn* absolute and must be re-based onto the concatenated
sequence — or, simpler and recommended, keep turns as **separate rows** in the
`policy_loss_for_rollouts_batched` batch and just assign all of them the *same*
trajectory advantage). The "separate rows, shared advantage" option needs **zero
change** to `policy_loss_for_rollouts_batched` — we pass `flat_rollouts =
[all turns of all trajectories]` and `flat_advantages = [R_traj duplicated per
turn]`. This is the cleanest integration and is the recommended implementation.

### 5.4 Loss (unchanged form)

```
adv_g = group_normalize([R_traj for trajectory in group])      # (1, N)
flat_rollouts   = [turn.rollout for traj in group for turn in traj.turns
                   if group_is_variance_bearing(group)]         # filter (C)
flat_advantages = [adv_g[traj_idx] for ... per turn ...]
loss = policy_loss_for_rollouts_batched(flat_rollouts, flat_advantages, ...)
       # already includes PPO clip + KL + stochastic-gate surrogate + entropy
```

So the **loss function is literally the existing one**; multi-turn lives
entirely in how `flat_rollouts`/`flat_advantages` are assembled. This is the
key reason to extend rather than rewrite.

---

## 6. Compute budget (2×RTX 5090, 32 GB ea, no NVLink)

### 6.1 The blow-up

Per training step today (`--batch 3 --grpo_n_group 4 --max_gen 384`):
- **Rollouts**: B·N = 12 sequences, each generated incrementally
  (`forward_step`, ~6ms/tok claimed) up to ~384 emit + ≤120 think tokens.
- **Grades**: 12 subprocess `grade` calls, parallelized over 8 threads.
- **Policy update**: one batched forward over 12 padded sequences (the OOM-prone
  step — `--batch 3` is the documented safe value, lm_head on long sequences is
  the 5.7 GiB tail).

Multi-turn with `max_turns=3` multiplies the rollout + grade count by up to 3×:
- **Rollouts**: up to 3× the forward passes (but turn ≥1 only runs on
  *unsolved* rollouts — early-stopping on pass prunes this; in practice ~1.8–2.2×
  not 3× once any problems solve).
- **Grades**: up to 3× subprocess calls. These are CPU/subprocess-bound, not
  GPU — `grade_in_parallel` already hides them behind the GPU work; bump
  `--grade_workers` to 12–16 and they stay off the critical path.
- **Policy update**: the dominant GPU-memory risk. With "separate rows, shared
  advantage" (§5.3) the policy forward batch grows from B·N to
  B·N·(mean turns) ≈ 12·2 = 24 sequences. **This will OOM** at the current
  `--batch 3` if all rows go into one forward.

### 6.2 Keeping it tractable

1. **Microbatch the policy update over turns.** Accumulate gradients across
   per-turn (or per-rollout) sub-forwards instead of one giant batch. The trainer
   already has `--activation_checkpointing`; add gradient accumulation over the
   flat-rollout list (chunk `flat_rollouts` into groups of ≤ B·N and
   `.backward()` each, scaling the loss by 1/n_chunks). This keeps peak memory
   at the *single-turn* level regardless of turn count.
2. **Early-stop solved rollouts** (don't roll turn t+1 for a rollout that
   already passed). Big real saving — easy problems exit after turn 0.
3. **Bound carried context** (§2.1): only the latest failed attempt + its error,
   not full history → per-turn prompt ≈ 2× single-shot, not t×.
4. **Cap N when turns>1.** Effective breadth×depth is what matters. Recommend
   `N=4, max_turns=3` ≈ same total rollout budget as `N=8` single-shot, which
   the repo already found is the rollout ceiling. Do **not** run `N=8 ×
   max_turns=3`.
5. **Async grading** is already there (`ThreadPoolExecutor` over subprocesses).
   With turns, fire turn-t grades while turn-(t-1) of the *next* problem rolls —
   but this cross-problem pipelining is a phase-3 nicety; the synchronous
   per-turn grade is fine if `--grade_workers ≥ 12` (grades are <5s each, 12 in
   parallel ≈ one grade's wall-time).
6. **Latent reflect steps are cheap** — R≈4 extra forward_step calls per turn at
   T=1, negligible vs the 384-token generation.

**Estimated wall-clock**: ~1.8–2.2× the current step time at `max_turns=3` with
early-stopping, microbatched update, and bumped grade workers. A 200-step v2 run
takes a few hours; multi-turn ≈ 4–5 h. Acceptable. The hard constraint is
**peak GPU memory**, solved by (1) microbatching — non-negotiable for turns>1.

---

## 7. Phased implementation plan + validation gates

Each phase has a **gate** (metric that must hold before proceeding) and the
**failure mode** to watch.

### Phase 0 — Zero-variance hygiene (cheapest, do first)
- Implement filter (C): `--group_var_floor` drops near-flat groups from the
  policy update; log `n_dropped_groups` and `frac_var_bearing`.
- Implement tie-break (B): tiny `error_text`-length term.
- **Gate**: on a 50-step `mbpp_combined` run, `frac_var_bearing` ≥ 0.7 and the
  step-300-equivalent HumanEval is **≥ 16/164** (no regression vs v2). This
  proves the hygiene doesn't hurt.
- **Failure mode**: if dropping low-var groups starves the batch (frac_var_bearing
  too low), the curriculum prior is mis-set — fall back to `--adaptive_curriculum`
  with a lower `--curriculum_adaptive_floor`.

### Phase 1 — Multi-turn loop, trajectory-flat credit, NO latent reflect
- Generalize `--iterative_repair` into the `--max_turns 3` turn loop (§2.1).
- Trajectory reward `R_traj` (§2.2) with `--lambda_improve 0.3 --turn_cost 0.02`.
- "Separate rows, shared advantage" integration (§5.3) — reuses
  `policy_loss_for_rollouts_batched` unchanged.
- Microbatched/grad-accum policy update (§6.2 item 1).
- **Gate**: HumanEval **> 16/164** (beat the single-shot best) within 300 steps,
  AND the multi-turn diagnostic `mean Δ_turn1 > 0` (turn-1 revisions improve
  score on average — proves the model uses feedback). Also: KL stays bounded
  0.05–0.15 (v2's stability band).
- **Failure mode**: the model ignores feedback (`mean Δ_turn1 ≈ 0`). If so, the
  repair prompt format is the suspect — the SFT base may not have seen
  feedback-conditioned data. Mitigation: a short rejection-sampling SFT
  (`gen_rejection_data.py` already exists) on *successful multi-turn
  trajectories* to teach the format, then resume RL. This is the
  cheaper-than-RL bootstrap the repo already advocates.

### Phase 2 — Latent reflection between turns
- Insert R latent-think steps (state-readonly) before each turn-≥1 emit (§4.1),
  with `--latent_reflect_steps 4 --state_readonly_at_think`.
- Let the stochastic gate (`--stochastic_gate`) decide *whether* to reflect, with
  the trajectory advantage as its reward (§4.2).
- **Gate**: HumanEval ≥ Phase-1 number AND a positive ablation —
  `--latent_reflect_steps 0` (reflect off) scores measurably lower on the same
  ckpt's eval. This is the load-bearing test: latent reflection must *cause* the
  lift, not decorate it (the repo's standard for "is it load-bearing").
- **Failure mode**: latent reflect adds compute but no lift, OR the gate
  collapses bimodally at τ (the documented temperature-fragility). If the gate
  collapses, freeze it (deterministic emit_threshold) and reflect a *fixed* R
  steps before every revision — separates "does reflection help" from "can the
  gate learn when to reflect". If reflection itself doesn't help, that's a real
  negative result; stop and bank Phase 1.

### Phase 3 — Value-guided branch selection + per-turn credit (optional)
- Spend turn budget on highest-`score` rollouts, prune hopeless ones (§3).
- Per-turn advantage assignment (§2.2 optional).
- **Gate**: same-or-better HumanEval at *lower* total rollout count (efficiency
  win), or a clear HumanEval lift from sharper credit.
- **Failure mode**: pruning collapses diversity (always expand the same lucky
  rollout) → re-introduce an exploration floor (keep ≥1 random expansion).

### Cross-phase risks (be skeptical)
- **At 287M, multi-turn self-repair may have a low ceiling.** Small models are
  weak in-context bug-fixers; turn-1 Δ could be small. The honest expectation is
  a **+1 to +3 HumanEval** lift, not a transformation. If Phase 1 yields <+1
  after 300 steps, the bottleneck is model scale / SFT data, not the RL
  formulation — pivot back to rejection-sampling SFT scale (the repo's repeated
  conclusion that "the remaining lever is post-training scale + model size").
- **Reward hacking via the tie-break (B)** — keep ε tiny so the model can't farm
  the `error_text`-length term instead of solving.
- **Feedback distribution shift**: the SFT base never saw repair prompts; the
  Phase-1 failure-mode SFT bootstrap is the mitigation, queued and cheap.
- **The latent-think transfer is unproven on real code at this scale** — it
  worked on synthetic pointer-chase and arithmetic-chain text (`latent_arith.py`),
  not yet on HumanEval-style code. Phase 2 is genuinely speculative; gate it hard.

---

## 8. LLM-as-grader (preference tie-breaker for zero-variance groups)

**Scope.** This is the *last* signal in the priority order behind the
unhackable execution grade. It exists to recover gradient from the residual
zero-variance groups that §1's hygiene (filter C) currently just *drops*: the
problems so hard that every one of the N rollouts ties on the same execution
tier (typically all `syntax_error`, score 0.0). On those groups `code_grader`
is correct but *blind* — it cannot tell a near-miss apart from garbage when
nothing compiles. A capable pretrained LLM (the local Qwen 3.6 AWQ already
driving `experiments/distill_solutions.py`) supplies a *preference ordering*
over the tied rollouts so the group regains `std > 0`.

### 8.1 What it is NOT

- **NOT distillation.** We do not copy Qwen's tokens, code, or CoT into the
  student. `distill_solutions.py` (teacher writes the answer, student imitates)
  is a different mechanism and a different phase. Here Qwen never produces a
  target sequence — it only *orders* the student's own rollouts.
- **NOT token-level imitation / a KL-to-Qwen term.** The policy gradient still
  flows from `policy_loss_for_rollouts_batched` over the student's *own* emitted
  tokens; the LLM only sets the scalar advantage those tokens receive.
- **NOT a reward model we train.** No learned RM head (the repo's "post-pretrain
  modules stay inert" lesson, CLAUDE.md). The judge is the frozen off-the-shelf
  Qwen, used at inference only.

### 8.2 Hierarchical folding — execution stays a HARD GATE

The LLM score is folded so it can re-order *only within* an execution-tied tier,
**never across tiers**. Reuse the existing dense ladder
`code_grader._TIER_BASE_SCORE` as the dominant term and add a bounded judge
perturbation:

```
# per rollout i in a group, AFTER grading:
tier_base_i  = code_grader._compute_score(tier_i, n_tests_i, n_passed_i)   # the unhackable part
judge_rank_i = listwise_rank(i) / (N - 1)        # ∈ [0, 1], 0 = worst, 1 = best (§8.3)
reward_i     = tier_base_i + eps_judge * (judge_rank_i - mean_rank)        # mean-centred
```

with `eps_judge` chosen **smaller than the minimum gap between adjacent tiers**.
The smallest adjacent gap in `_TIER_BASE_SCORE` is `exec_error→runtime_error`
(0.05 → 0.20 = 0.15) and `syntax_error→exec_error` (0.0 → 0.05 = **0.05**, the
binding constraint). So `eps_judge ≤ 0.04` guarantees the judge term's full
swing (`eps_judge · 1.0`) can never lift a `syntax_error` rollout
(`tier_base = 0.0`, max reward `0.04`) above an `exec_error` rollout
(`tier_base = 0.05`, min reward `0.05 − 0.04·mean_rank ≥ 0.01` once
mean-centred — and at worst these *touch* but never invert because the better
tier also gets its own positive judge term). To be airtight against the
mean-centring edge case, **set `eps_judge = 0.02`** (well inside the band) and
additionally clamp `reward_i` into `[tier_base_i − tier_margin, tier_base_i +
tier_margin]` with `tier_margin = 0.5 · min_adjacent_gap = 0.025`. The judge
then *provably* re-orders within a tier and is structurally incapable of
crossing one. This bounds the reward-hacking surface to *exactly* the
zero-variance cases — everywhere else execution variance dominates and the
judge term is mean-centred noise of magnitude ≤ 0.02 that group-normalization in
`compute_grpo_advantages_from_rewards` largely washes out.

**Gating condition.** The judge fires on a group iff
`group_is_variance_bearing(execution_rewards, tau=group_var_floor)` is **False**
— i.e. precisely the groups filter (C) would otherwise drop. On
variance-bearing groups the judge is *not even called* (saves compute and avoids
adding a hackable signal where execution already gives a clean one). This is the
single most important property: **the judge has zero influence on any group the
execution grader can already distinguish.**

### 8.3 Listwise ranking, not absolute scores

Ask Qwen to *rank* the N candidates against each other, not to emit absolute
0–1 scores. Rationale (and consistent with the doc's skepticism about small,
noisy deltas): at the small reward magnitudes that matter inside a tied tier,
absolute-score calibration is dominated by the judge's own noise, whereas a
relative ordering of 4 candidates is a far easier, lower-variance ask. Convert
the returned order to advantages by feeding the rank-derived `reward_i` (§8.2)
straight into `compute_grpo_advantages_from_rewards` — no new advantage code
path, the rank just *is* the within-tier reward component.

**Prompt sketch** (one batched `llm.generate` call per tied group, reusing the
`vllm.LLM` + `apply_chat_template` machinery from `distill_solutions.py:154`
`_build_chat_template_prompt`):

```
SYSTEM: You are ranking N candidate Python solutions to one problem by how
CLOSE each is to a correct, working solution. They all currently FAIL the
unit tests — your job is to order them by which is nearest to passing
(structure, correctness of approach, how localized the remaining bug is).
Do NOT reward verbosity, comments, or style. A short almost-correct function
beats a long elaborate broken one.

USER:
Problem:
<problem.prompt>

Candidate 1:
<code_1>
Execution diagnosis 1: <grade_1.error_text>      # the compiler/runtime truth

Candidate 2:
<code_2>
Execution diagnosis 2: <grade_2.error_text>
... (N candidates) ...

Return ONLY a JSON list of candidate numbers, best-first, e.g. [3,1,4,2].
```

Feeding `GradingResult.error_text` (the real SyntaxError / traceback /
failed-assert source) into the judge prompt is what keeps the ranking
*grounded* — the judge ranks "closeness to correct" with the execution truth in
hand, not a free-floating stylistic opinion. Parse the JSON list; on a malformed
response fall back to "all-equal" (judge abstains → group is dropped exactly as
filter C would have, no harm). Recommend `n_samples=1, temperature=0.0` for the
judge so the ranking is deterministic and cacheable.

### 8.4 Self-limiting cost + the GPU dependency

**Frequency decays over training.** The judge fires only on execution-tied
groups. As the difficulty curriculum (`curriculum.ProblemDifficultyEMA`,
adaptive) starts landing passes and the multi-turn Δ-reward (§2) spreads
rollouts across tiers, the fraction of fully-tied groups *falls*, so the
judge's call rate decays as the policy improves — it spends the most compute
exactly when the model is weakest and needs the signal, and tapers off on its
own. Log `frac_judge_fired` per step; expect it to start high (early training,
many all-`syntax_error` groups) and trend down.

**Per-step cost.** One batched `llm.generate` call per tied group. With
`--batch 3 --grpo_n_group 4`, at most 3 judge calls/step, each ranking 4 short
candidates (problem + 4×~200-token solutions + 4 short diagnoses ≈ 2–3k prompt
tokens) → a few hundred output tokens (just the JSON list). Batched across the
≤3 tied groups, that is one `llm.generate` of ≤3 prompts — on the order of
**0.3–1 s** on a dedicated GPU for a 35B-A3B MoE at AWQ, *off the training
critical path if it overlaps the student's grading/rollout work*.

**The hard constraint: it needs a spare GPU.** During RL the *training* GPU is
full (25–30 / 32 GiB; `--batch 3` is the documented OOM-safe value, the policy
forward's lm_head tail is the 5.7 GiB risk). The Qwen 3.6 AWQ teacher needs
~20+ GiB of its own. There is **no NVLink and no room to co-resident** the judge
with the trainer on one card. Options, in order of preference:

1. **Run the judge as a persistent vLLM server on GPU 1** (`CUDA_VISIBLE_DEVICES=1`),
   trainer on GPU 0. The trainer POSTs tied-group ranking requests over HTTP /
   a local queue; vLLM's continuous batching absorbs the bursty per-step load.
   This is the only configuration that does not contend for memory. **Flag this
   as a hard prerequisite: Phase 4 cannot run on a single free GPU while a full
   RL trainer occupies the other.** If both GPUs are needed for DDP training,
   the judge phase is simply not available — it is explicitly an optional,
   resource-gated phase (§8.6).
2. **Cache aggressively.** Key the cache on `(problem_id, sorted tuple of
   candidate-code hashes)`. Within a run the same problem recurs (curriculum
   re-samples it) and—on a too-hard problem—rollouts often repeat near-identical
   broken code, so a content hash cache hits frequently. A persistent on-disk
   cache also lets a *resumed* run skip re-judging.
3. **Cap judge calls/step** (`--judge_max_calls_per_step`) and prefer the
   highest-`group_var`-deficit groups if over budget; the rest fall back to
   filter-C drop.

Latent reflect steps and grading are unchanged; the judge is purely additive
GPU-1 work that the trainer awaits only when a group is fully tied.

### 8.5 Reward-hacking stress test (be skeptical)

The hard execution gate (§8.2) means a *failing* solution can never outrank a
*passing* one, and the judge only operates inside an all-failing tier. But it
is still a learned-prior signal and **can be gamed within that tier**:

- **Verbose / plausible-looking code.** The judge may prefer code that *looks*
  like a complete solution (more lines, more branches) over a terse stub, even
  when both are equally broken. The policy could learn to emit longer
  syntactically-rich garbage to farm rank. **Guardrails**: (i) the explicit
  prompt instruction "do NOT reward verbosity… a short almost-correct function
  beats a long elaborate broken one"; (ii) `eps_judge = 0.02` caps the total
  exploitable reward — a hacked rank is worth at most 0.02, far below any real
  tier climb (≥0.05), so the moment the policy can reach `exec_error` the
  execution gate makes verbosity-farming strictly dominated; (iii) the judge
  *never fires* once the group is variance-bearing, so as soon as a problem
  becomes learnable the hack pays nothing.
- **Comment-stuffing / docstring padding.** Comments don't affect execution but
  could sway a stylistic judge. Same guardrails; additionally we can strip
  comments before sending code to the judge (cheap AST/`tokenize` pass) so the
  judge ranks *behavior-bearing* code only.
- **Matching the judge's stylistic priors** (idiomatic naming, type hints, the
  "Qwen house style"). This is the subtlest: the policy could drift toward
  Qwen-pleasing surface form rather than correctness. **Guardrails**: ranking
  is grounded in `error_text` (execution truth in the prompt, not pure style);
  `eps_judge` cap; and the ranking-only design — the judge expresses *order*,
  not magnitude, so a uniformly stylish-but-broken set still nets near-zero
  centred advantage.
- **Judge–policy collusion drift.** Over a long run the policy could co-adapt to
  whatever idiosyncratic ordering Qwen produces on broken code (Goodhart on a
  fixed judge). **Guardrails**: the self-limiting frequency (§8.4) means the
  judge's cumulative influence shrinks as the model improves; KL-to-frozen-SFT
  (`--kl_coef`) bounds total drift; and a periodic **spot-check** — log a
  sample of (candidates, judge ranking, subsequent turn outcome) and
  occasionally eyeball whether the judge's "closest to correct" actually became
  the rollout that solved it after a repair turn. If the judge's top-ranked
  candidate does *not* correlate with downstream improvement, the judge is
  noise and the phase should be turned off (§8.6 failure mode).

**Net.** The combination — hard execution gate, fires only on tied groups,
`eps_judge ≤ 0.02` with a `tier_margin` clamp, ranking-only, comment-stripping,
KL bound, spot-checks — keeps the hackable surface to "re-order within an
all-failing tier, worth ≤0.02 reward, only while the problem is unsolved." That
is a deliberately tiny, self-extinguishing attack surface.

### 8.6 Where it slots in: Phase 4 (gated, optional, GPU-dependent)

This is **strictly later than Phases 0–2** and only justified once the cheaper,
unhackable mechanisms have proven out. It adds (a) a hackable signal and (b) a
second-GPU dependency — both reasons to defer.

- **Prerequisite (must hold before turning it on):** Phase 0 (zero-variance
  hygiene) and Phase 1 (multi-turn, trajectory-flat credit) are validated, i.e.
  HumanEval ≥ 16/164 with `mean Δ_turn1 > 0` and KL bounded. **And** a second
  GPU is free for a persistent vLLM judge server (§8.4). If either fails, skip
  this phase.
- **The signal it targets:** after Phase 1, measure `frac_var_bearing`. If it is
  already ≥ 0.9 (multi-turn + curriculum recovered almost everything), the judge
  has little to add — **do not bother**. The phase is only worth it if a
  meaningful fraction (say ≥ 10 %) of sampled groups are still fully tied after
  multi-turn.
- **Validation gate to keep it on:** on a 100-step A/B, the judge-enabled run
  must show (i) HumanEval **strictly greater** than the Phase-1 number, AND (ii)
  a positive spot-check correlation (§8.5) — the judge's top-ranked tied
  candidate improves more often than the bottom-ranked one on the *next* turn.
  Both must hold; (ii) without (i) means the judge is right but irrelevant, (i)
  without (ii) means we're hacking it.
- **Failure mode:** HumanEval flat-or-down, or spot-check shows the judge's
  ranking anti-correlates with downstream improvement → **the judge is being
  gamed or is noise; turn it off and bank Phase 1–2.** This is a genuine
  possible negative result, consistent with the doc's stance that the residual
  too-hard groups may simply be beyond a 287M model and the real lever is
  post-training scale.

---

## Appendix — concrete flag surface (proposed additions to `train_rl_grader.py`)

```
--max_turns 3                 # 1 = current single-shot (default, byte-identical)
--lambda_improve 0.3          # weight on Σ max(0, Δ_turn) in R_traj
--turn_cost 0.02              # per-extra-turn penalty
--group_var_floor 1e-3        # drop groups with reward-std below this (filter C)
--tie_break_weight 1e-3       # error_text-length tie-break (shaping B)
--latent_reflect_steps 4      # R latent-think steps before turn-≥1 emit (phase 2)
--carry_history               # (default off) carry full turn history vs latest only
--llm_judge                   # (default off, phase 4) enable LLM tie-breaker on tied groups
--llm_judge_url ...           # vLLM server endpoint (Qwen on GPU 1); required if --llm_judge
--judge_eps 0.02              # within-tier judge perturbation magnitude (≤ 0.04 hard cap)
--judge_tier_margin 0.025     # clamp |reward − tier_base| so judge can't cross a tier
--judge_max_calls_per_step 3  # cost cap; over-budget tied groups fall back to filter-C drop
--judge_strip_comments        # (default on) strip comments before sending code to the judge
```
All default to the values that reproduce today's single-shot trainer, preserving
backwards-compatibility and the existing test suite
(`test_rl_grader_gate_floor.py`, `test_iterative_repair.py`,
`test_rl_grader_state_readonly.py`). New tests to add: trajectory reward math,
`group_is_variance_bearing`, `to_flat_rollout`/separate-rows advantage
duplication, a `max_turns=1` byte-identity regression, and for §8: a
`judge_cannot_cross_tier` property test (random tiers + adversarial judge ranks
→ assert sorted order by `tier_base` is preserved), `judge_fires_only_on_tied_group`
(variance-bearing group → judge not called), and a malformed-judge-response →
group-dropped fallback test.
```

---

## RL Implementation Decision Log

Dated log of non-obvious implementation choices made while building the
multi-turn / zero-variance-hygiene / LLM-judge pipeline. All code is
backwards-compatible: defaults reproduce today's single-shot trainer.

### RD1 (2026-05-29) — Reuse the existing `Rollout`; wrap it in `Turn`/`Trajectory`
**Decision.** Did NOT extend `Rollout` with multi-turn fields. Instead added
`Turn{prompt_text, rollout, score, tier, error_text}` and
`Trajectory{problem_id, turns:[Turn], R_traj}` in a new module
`experiments/rl_multiturn.py`. The existing `Rollout` (emit ids/logps/positions,
depth, gate_* fields) is embedded verbatim in each `Turn`.
**Why.** `Rollout` flows into `policy_loss_for_rollouts_batched` unchanged. If we
had bolted turn bookkeeping onto `Rollout`, every call site (the single-turn path,
the loss, the gate-PPO block) would risk byte-level divergence. Composition keeps
the hot-path object identical and isolates all multi-turn state in the wrapper.
**Rejected.** Subclassing `Rollout` (fragile w.r.t. the dataclass default
ordering and the `gate_*=None` invariants the loss relies on); adding a
`turn_index`/`parent` field to `Rollout` (pollutes the single-turn path).

### RD2 (2026-05-29) — Terminal-dominates clamp = hard-cap the bonus at `0.5·min_adjacent_gap`
**Decision.** `R_traj = terminal + min(λ·Σmax(0,Δ_t), 0.5·min_adjacent_gap)
− turn_cost·(T−1)`. The improvement bonus is hard-capped at `0.025` (half the
smallest adjacent dense-ladder gap, syntax→exec = 0.05), independent of λ.
**Why.** The task spec requires the terminal term to PROVABLY dominate so a model
can never win by "starting bad to harvest Δ". Capping at strictly less than the
smallest gap between two distinct terminal scores guarantees: if terminal_a >
terminal_b by ≥ a real tier gap, no improvement shaping can invert the order
(turn_cost only ever lowers a longer trajectory). Shaping therefore only breaks
ties WITHIN one terminal score. This is structural, not a tuning of λ — a user
can set λ=1000 and the guarantee still holds (test:
`test_terminal_dominates_clamp_start_bad_cannot_outscore_clean_pass`).
**Rejected.** Scaling λ down "by convention" (a careless launcher could break the
guarantee); a soft `tanh` squash (no provable bound); rewarding negative Δ
(punishes exploration — we clamp Δ at `max(0,·)`, mirroring the existing
`counterfactual` philosophy).

### RD3 (2026-05-29) — Integration by assembly: "separate rows, shared advantage"
**Decision.** Each TURN of each trajectory is a SEPARATE ROW fed to the UNCHANGED
`policy_loss_for_rollouts_batched`; every row of a trajectory carries that
trajectory's single group-normalized advantage. `assemble_flat_rollouts` is the
only new hot-path logic. The loss function is not touched.
**Why.** §5.3-5.4 of the design: GRPO's group-normalization already discards
fine-grained per-token credit, so trajectory-flat credit loses little and keeps
the (OOM-tuned, KL-aware, stochastic-gate-aware) batched loss byte-identical.
Per-turn position re-basing onto a concatenated sequence (the `to_flat_rollout`
alternative) is error-prone and unnecessary.
**Rejected.** `to_flat_rollout` concatenating turns into one pseudo-Rollout
(needs emit-position re-offsetting — a latent off-by-one bug surface); a per-turn
advantage with a per-turn-index baseline (deferred to phase 3 per §2.2).

### RD4 (2026-05-29) — `--group_var_floor` defaults to 0.0 (off)
**Decision.** Default 0.0 → no group is ever dropped, and the predicate
`group_is_variance_bearing` is a strict `std > tau`. At tau=0 the legacy behaviour
is exactly reproduced because the GRPO `std + 1e-8` already collapses a flat
group's advantages to ≈0 (dropping it changes nothing numerically).
**Why.** Backwards-compat is the hard requirement; a non-zero default would
silently change every existing launcher's effective batch. The floor is a
strictly-removes-noise opt-in (§1.2 C). Population std (`/n`, not `/(n-1)`)
matches the `unbiased=False` GRPO advantage code so the predicate and the
advantage agree on what "flat" means.
**Rejected.** Default 1e-3 (would alter v2's reproduced numbers); using sample std
(disagrees with the advantage normalization).

### RD5 (2026-05-29) — `JudgeBackend` ABC is the mockability boundary; vLLM never imported in tests
**Decision.** Abstract `JudgeBackend.rank(problem, candidates) -> permutation`.
The real `VLLMJudgeBackend` lives in `rl_multiturn.py` but imports `vllm` only
inside `__init__`/`rank` (never at module import). Tests supply a `_MockJudge`
returning a canned ranking. The trainer constructs `VLLMJudgeBackend` only on the
main rank, only when `--llm_judge`, with a local import.
**Why.** vLLM allocates a GPU at construction; importing it in a unit test would
OOM a co-resident training run (the project's hard constraint). The ABC lets the
folding/gating logic be fully tested with a fake, and keeps the GPU dependency
behind a flag + a free-GPU prerequisite (§8.4).
**Rejected.** A module-level `import vllm` (would force the dependency on every
import path, including CPU tests); a function-pointer instead of an ABC (loses the
typed `rank` contract and the listwise-permutation invariant).

### RD6 (2026-05-29) — Judge folding clamps each reward into its own tier band
**Decision.** `reward_i = clamp(tier_base_i + eps_judge·(rank_i − mean_rank),
tier_base_i ± tier_margin)`, with `eps_judge=0.02`, `tier_margin=0.025`. The
per-candidate clamp to `[tier_base − 0.025, tier_base + 0.025]` is what makes
crossing a tier STRUCTURALLY impossible (not just unlikely): a lower tier's max
folded reward (`base + 0.025`) at most touches the next tier's min
(`base' − 0.025 ≥ base + 0.05 − 0.025 = base + 0.025`).
**Why.** §8.2 demands the execution grade stay a hard gate. The mean-centring
alone bounds the swing but the explicit clamp makes the no-cross property a
local, per-candidate invariant that a property test can assert over random
tiers + adversarial rankings (`test_judge_cannot_cross_tier_property_random`),
rather than a global argument that's easy to break under edge cases.
**Rejected.** Mean-centring without the clamp (correct in expectation but the
property test found the touch-but-never-cross edge is only guaranteed WITH the
clamp); absolute judge scores instead of ranks (judge-noise-dominated at these
magnitudes per §8.3).

### RD7 (2026-05-29) — Orchestration takes injected rollout/grade/extract callbacks
**Decision.** `run_trajectories_for_group(problem, prompt, *, rollout_fn,
grade_fn, extract_fn, ...)` and `_run_multiturn_step(...)` keep the pure turn-loop
logic free of any direct model/grader/GPU reference; the trainer passes thin
closures over `rollout_group_batched` + `code_grader.grade`.
**Why.** This is what makes the multi-turn loop CPU-testable with mocks (no GPU,
no subprocess) — the project mandate. The GPU/grader cost stays in the trainer;
the orchestration (early-stop on pass, repair-prompt construction, per-turn
grading order) is verified deterministically.
**Rejected.** Inlining the turn loop into `main()` (untestable without a model);
making `run_trajectories_for_group` import the grader directly (couples the pure
logic to a subprocess-spawning dependency).
