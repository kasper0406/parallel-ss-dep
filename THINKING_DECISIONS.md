# Thinking-Mechanism Plan — Decisions Log

Running log of judgment calls made during autonomous execution of
`THINKING_PLAN.md`. Each entry: date, decision, why, and the next
review trigger (when to revisit if the assumption breaks).

Plan started: 2026-05-26 (immediately after Phase D pretrain launch).

---

## 2026-05-26 — Start: parallelism and ordering

**Decision**: Build Phase 1 (diagnostics) and Phase 2 (state-read-only
architectural fix) in parallel via background agents, before Phase D
finishes. Both can be tested on the existing Phase C ckpt — no need to
wait for Phase D's final ckpt.

**Why**: Phase 1 results inform whether Phase 4-5 (training signal)
or Phase 2-3 (architecture) is the bigger lever. Don't sit idle.

**Review trigger**: Phase 1 results land → revisit prioritization
between Phase 2 and Phase 4.

---

## 2026-05-26 — Dispatch: 4 agents in parallel

**Decision**: Dispatched Phase 1 (diagnostics), Phase 2 (state-read-only),
Phase 4 (CoT distillation infra), Phase 6 (reasoning tasks) as 4 parallel
background agents in separate worktrees. Phase 3 (per-position think
index embedding) is HELD until Phase 2 lands, because both touch
`model.py` and would conflict. Phase 5 (process-aware reward) is HELD
until Phase 1 results show whether the thinking mechanism even benefits
prediction (informs whether process reward will pay off).

**Why**: Maximize throughput while Phase D pretrain runs on GPU. None of
the dispatched agents need GPU for build+test (Phase 1 builds infra,
Phase 4 is generator skeleton, Phase 6 is CPU-only generator). Phase 2
needs a small GPU smoke for the recall-preservation probe — that may
have to wait for Phase D's GPU pressure to drop.

**Review trigger**: when all 4 agents report, review each + merge
sequentially. Then dispatch Phase 3, and (depending on Phase 1
results) Phase 5.

## 2026-05-26 — Phase 4 CoT-SFT format: Option A

**Decision**: Specified Option A (every CoT token wrapped in
`[THINKING]` markers, masked from loss) for the CoT-to-SFT data
conversion. Rejected Option B (one think token per CoT step with
WM-stored representation) for v1 because it requires new architectural
plumbing.

**Why**: Option A teaches the gate that thinking-bursts are useful
without requiring new arch. If Phase 5 (process-aware reward) confirms
thinking actually helps prediction post-Phase 4, Option B becomes the
next evolution.

**Review trigger**: when Phase 4 + an SFT run on the CoT data reveals
whether the gate learned to fire correctly. If yes → keep Option A;
if no → escalate to Option B.

## 2026-05-26 — All 4 parallel agents merged

**Result**: Phase 1 (3 probes, 11 tests), Phase 2 (state-read-only,
6 tests, +0.465 recall lift on synthetic probe), Phase 4 (CoT distill
infra, 16 tests), Phase 6 (6 reasoning families, 22 tests). All
committed cleanly.

**Phase 2 load-bearing result**: state-readonly thinking shows
acc_on=0.879 vs acc_off=0.414 on the recall-preservation probe.
That's a +0.465 architectural lift — confirms the "think tokens
corrupt the recurrence" diagnosis was correct AND that hooking the
b_proj to zero β at think positions actually fixes it.

**Open follow-up from Phase 2 agent**: forward_step (per-token
decode) path NOT hooked. The FLA fused-recurrent decode path bypasses
the b_proj hook. Decode-time inference is currently safe (decode loops
insert tokens through the same path as training) but the per-token
optimization isn't using state-readonly. Flag for v2 of the fix.

**Review trigger**: when we run state-readonly on a real ckpt (post-
Phase D) + the long-context recall eval, confirm the synthetic-probe
lift transfers to real workload.

## 2026-05-26 — Phase 3 ordering

**Decision**: Now that Phase 2 has landed, dispatching Phase 3 (per-
position think index embedding) since it touches the same files
(model.py).

## 2026-05-26 — Phase 3 merged; defer Phase 5

**Result**: Phase 3 (think index embedding) landed cleanly, 8/8 tests
pass. Vectorized cumsum-reset trick — no Python loop, GPU-friendly.

**Decision**: defer Phase 5 (process-aware reward) until Phase 1
probes run on a real ckpt. Phase 5's counterfactual gate target
needs an extra forward pass per training step — significant compute.
Don't build until we have evidence (from Phase 1) that thinking
actually helps prediction; if Phase 1 says no, we need Phase 2-3
architectural fixes first anyway.

**Next-up**: SFT-with-CoT-thinking wrapper — the integration code
that consumes Phase 4's CoT JSONL. Without this, Phase 4 data is
inert.

## 2026-05-26 — All build phases complete

**Result**: Phase 1, 2, 3, 4, 6 + SFT-CoT wrapper + orchestrator script
all committed. Test counts: Phase 1 (11), Phase 2 (4), Phase 3 (8),
Phase 4 (16), Phase 6 (22), SFT wrapper (8) — **69 new tests, all
passing** (cuda-required tests skip cleanly in CPU-only test runs).

**What's NOT built yet**: Phase 5 (process-aware reward — counterfactual
gate target with per-position next-K CE forward). Held until Phase 1
diagnostics on the post-Phase-D ckpt confirm thinking actually moves
CE. If yes → build Phase 5. If no → architectural work (Phase 2 + 3)
needs to land in the model FIRST via SFT-with-thinking-active before
RL process reward can pay off.

**Orchestration**: `run_thinking_pipeline.sh` runs all 6 post-Phase-D
steps idempotently. Each step is gated on its predecessor's artifact;
re-runs skip completed steps. Designed to fire as one bash invocation
the moment `checkpoints/pretrain_phase_d.pt` lands.

**Next action when Phase D completes**: `bash run_thinking_pipeline.sh`.

## 2026-05-26 — Phase 4 Design A captured in THINKING_PLAN.md

The SFT wrapper agent added the implementation choice to
`THINKING_PLAN.md` Phase 4 section directly. Future readers see the
chosen design + file location alongside the original spec. Good
documentation hygiene.

## 2026-05-26 — CRITICAL: Phase 1 results disprove the "mechanism works" assumption

**Findings on Phase D ckpt** (see `THINKING_PROBE_RESULTS.md` for full
detail):

- **Probe 1 (per-position CE Δ)**: thinking INCREASES next-token CE by
  0.19 nats on average (4.45 → 4.64). Only 36% of positions improve
  with thinking; 64% get WORSE.
- **Probe 2 (counterfactual HumanEval)**: 0/50 either way (pretrain-only,
  not discriminative yet).
- **Probe 3 (RL correlation)**: Spearman ρ = −0.17 between think count
  and reward. The 9 rollouts that thought scored mean 0.04 vs no-think
  mean 0.18.

**Implication**: the thinking mechanism is currently HARMFUL, not
neutral. The model has the architecture but uses it to inject noise
into its own state.

**Re-prioritization**:
1. Phase 4 (CoT distill SFT) is now essential — model has no
   demonstration of useful thinking; without it, no signal.
2. Phase 5 (process-aware reward) becomes MORE important — explicit
   gradient that thinking should reduce CE/improve reward.
3. Phase 2 (state-readonly) partially explains the +0.19 CE penalty
   (recurrence corruption) — likely a contributing factor.

**Pipeline continues**: the orchestrator is already on step 2 (Qwen
CoT distill, multi-hour). Plan unchanged; just confirms the SFT step
is doing exactly the right work.

**Review trigger**: after step 5 (HumanEval on SFT-CoT-thinking ckpt)
+ re-run Phase 1 probes on the SFT'd ckpt. If probes flip positive →
Phase 5 is worth building. If still negative → architectural rethink
needed (current mechanism is fundamentally broken at this scale).

## 2026-05-26 — PIVOT: efficient-thinking plan replaces gate-training plan

**Decision**: rewrote `THINKING_PLAN.md` end-to-end. The previous plan
was "train the gate to fire correctly + give it process reward". The
user pointed out — and Phase 0 result confirmed — that nothing in
that plan actually tested the architecture's CORE PROMISE
(N_think << N_cot via compression of CoT into denser representations).

**Old plan failure mode**: Design A SFT taught the model to spend N
thinks (= N CoT tokens) as loss-masked padding before emitting code.
Think positions had no gradient signal → no way to learn what to
compute there. The gate-firing was decorative; the architectural
"1 think ≈ K CoT tokens" claim was never exercised.

**New plan core**: Phase 1 introduces a gist-loss target at think
positions — predict the CoT teacher's hidden state K steps ahead per
think. THIS gives think positions explicit gradient and EXPLICITLY
trains compression (K=5 means N_think = N_cot / 5).

**Decision gates**:
- Phase 0 (text-CoT baseline): if pass@1 < 5/164, scale issue not
  thinking issue; pivot away from the mechanism.
- Phase 1a (gist supervision K=5): if HumanEval matches Phase 0
  baseline with 5× fewer thinks, efficient thinking works. Ship.
- Phase 1b (retrieval-as-input from CoT, deeper plumbing): fallback
  if 1a doesn't deliver.
- If neither: architecture can't support compression at this scale;
  pivot to scale-up.

**What's still useful from the old plan**:
- Phase 2 (state-readonly): shipped, validated +0.465 recall lift.
  Keep ON for all new runs.
- Phase 3 (think index embedding): shipped. Keep ON.
- Phase 6 (synthetic reasoning curriculum): shipped, 504 tasks. Use
  as RL data in the new Phase 3.

**What's dropped from the old plan**:
- Design A as the primary SFT approach (loss-masked think padding).
  Its only remaining use is as a "gate ramp-up" warmup before
  swapping in gist-at-think.

**Currently running**: the in-flight `run_thinking_pipeline.sh` is on
step 6 (long-context recall) of the OLD plan's pipeline. Let it
finish for the data, but the Phase D + mixed-SFT (Phase 0 of new
plan) is the actual next move — already queued via
`launch_sft_phase_d_mixed.sh`.

## 2026-05-26 — Phase 0 decision-gate FAILED on Phase D; isolation run launched

**Result**: Phase D + mixed SFT (54k Qwen distill + 961 CoT) = **0/164**
on HumanEval. Gate fires at 20% rate (healthy), but pass@1 is zero.

**Comparison**: Phase C + Qwen distill historically scored 10/164.
The pretrain "improvements" in Phase D (FIM augmentation, synth
pyfunc, self-debug fold-in) may have HURT the base model's
code-completion ability — possibilities:
1. FIM training: model learned `<|fim_*|>` patterns; HumanEval
   prompts are OOD.
2. Synth pyfunc: Qwen-generated style may not match HumanEval.
3. Self-debug: model expects errors and produces weird patterns.

**Isolation run launched**: `launch_sft_phase_c_mixed.sh` — same
SFT recipe but resumes from `pretrain_phase_c.pt` (the validated
10/164 baseline) instead of Phase D. Tells us whether the
regression is in Phase D's pretrain or in the SFT recipe itself.

**If Phase C + mixed SFT ≈ 10/164**: Phase D's pretrain regressed
the base; Phase 1a (gist supervision) should run from Phase C base,
not Phase D base.

**If Phase C + mixed SFT also = 0/164**: the SFT recipe itself
broke (CoT-thinking row format or something else). Need to debug.

**Phase 1a (compression gist supervision) build is COMPLETE** — load-
bearing test passed (gist loss 1.05 → 0.0002 in 50 steps). Just
needs a known-working base ckpt to launch from. Awaiting the
isolation result.

## 2026-05-26 — Isolation: SFT RECIPE was the bug, not Phase D pretrain

**Result**: Phase C + my new mixed-SFT recipe also = **0/164**. Same
as Phase D. So Phase D's pretrain is NOT the regression — the SFT
recipe I used is.

**Diff against the proven Phase C SFT recipe (10/164)**:
  | flag                          | proven (10/164) | mine (0/164) |
  | epochs                        | 2               | 1            |
  | lr                            | 3e-5            | 5e-6 (6× lower) |
  | retrieval_as_input_thinking   | ✓               | ✗            |
  | future_emb_loss_weight        | 0.1             | (off)        |
  | wm_gist_horizons              | "16,64,256"     | (off)        |
  | think_max_bursts / depth      | 3 / 8           | defaults     |
  | mem_size                      | 1024            | (off)        |

Massive recipe differences. My version was a low-LR, short, no-WM-gist,
no-retrieval-as-input quickie. The historical recipe is substantial.

**Fix launched**: `launch_sft_phase_d_proper.sh` reruns Phase D SFT
with the proven Phase C recipe. Expected pass@1 ≥ 8/164 if this just
reproduces history; ≥ 10 if the Phase D base + the new data mix
improvements help.

**For Phase 1a (gist supervision)**: update the launcher to use the
proper recipe as the base too. The compression test only makes sense
on top of a working SFT.

## 2026-05-26 — Phase D pretrain regressed the base (confirmed)

**Result**: Phase D + proper recipe = **2/164** vs Phase C + proper
recipe (historical) = 10/164. Same recipe, same data, different
pretrain base → 8-problem regression.

The Phase D pretrain "improvements" (FIM augmentation + synth pyfunc +
self-debug fold-in) HURT the base for HumanEval-style code completion.
Hypotheses (didn't investigate further; rerouting around the regression):
1. FIM sentinel tokens trained but never appear at HumanEval inference.
2. Qwen-synth-pyfunc style mismatch with HumanEval.
3. Self-debug rows train the model to expect/emit errors.

**Decision**: route Phase 1a (gist supervision compression test)
through the Phase C base (validated 10/164 baseline) instead of
Phase D. The Phase 1a compression test only makes sense on top of a
working SFT.

**Launched**: `launch_sft_phase_c_proper.sh` — Phase C + proper
recipe + mixed data (Qwen distill + 961 CoT rows). Expected pass@1
≥ 8/164 if adding CoT rows doesn't hurt. This becomes the new
"Phase 0 baseline" for the compression test.

**v6_all pretrain mix is now deprecated.** Future pretrain runs
should start from Phase C and use mix_v4 (no FIM, no synth pyfunc,
no self-debug fold-in until each can be ablated individually to find
which one regressed).

## 2026-05-26 — Phase 1a result + diagnosis

**Result**: Phase 1a (gist supervision on top of 8/164 historical-repro
base) = **0/164**. Same pattern as Phase 0 — ANY SFT route that
includes the new CoT-thinking dispatch (build_example_with_cot_thinking
OR build_example_with_cot_compression) drops the model to 0/164.

**What we know empirically**:
  | run                                    | HumanEval |
  |----------------------------------------|-----------|
  | historical ckpt (May 22, eval'd today) | 7/164     |
  | historical recipe + historical data    | 8/164     |
  | historical recipe + mixed (Qwen + CoT) | 0/164     |
  | Phase 1a (gist supervision on CoT)     | 0/164     |
  | Phase D base + same mixed              | 2/164     |

  Gist supervision DOES learn (load-bearing test: 1.05 → 0.0002 over
  50 steps). The mechanism works in isolation.

**Hypothesis**: the CoT-only / CoT-mixed-narrow SFT distribution is
too narrow / out-of-distribution for the trained-on-diverse-mix model.
The model needs the synth_memory + longctx_recall anti-forgetting
sources alongside ANY new data added.

**Honest assessment**: today's burn was significant compute (Phase D
pretrain 20h + ~5 SFT runs + ~10 HumanEval evals). We confirmed the
architecture (Phase 2 state-readonly +0.465, Phase 1a gist trainable),
but no run validates the end-to-end efficiency claim because the SFT
data setup keeps breaking the model.

**Pivot proposal (NOT executed without user sign-off)**:
1. Retry Phase 1a with the FULL historical mix + CoT-via-gist rows
   together (distill + synth_memory + longctx_recall + 961 CoT-with-
   gist). Tests whether the issue was MISSING anti-forgetting sources
   or CoT-format toxicity.
2. If that also fails → the new CoT path is structurally broken; need
   to debug the SFT code path or the data format.
3. If that works → ship; we have efficient thinking.

## 2026-05-26 — PIVOT v3: discovery RL instead of CoT imitation

**Decision**: rewrote `THINKING_PLAN.md` again. Both v1 (Design A
CoT-padding) and v2 (gist supervision) shared the hidden assumption
that we should TEACH the model a predetermined thinking pattern from
Qwen. Empirically that destroys the model (0/164 in every variant).

**The user's insight**: the optimal thinking pattern is UNIQUE to our
architecture (FiLM K=3 + WM + retrieval-as-input). We can't copy it
from Qwen because Qwen has a different architecture. We have to let
the model DISCOVER it via exploration.

**v3 mechanism**: stochastic gate.
- Gate output σ(g_t) becomes a Bernoulli probability over emit-vs-think
- Rollouts SAMPLE the decision (not threshold)
- Capture gate log-prob in rollout
- PPO ratio includes gate decisions alongside emit tokens
- Reward attribution flows back through the gate

**Why this might work where SFT didn't**:
- No imitation of Qwen → no architectural mismatch
- Reward signal: did the FINAL answer pass? Reward what works, regardless of pattern
- Compression EMERGES — if gate=0 (never think) wins, model converges there; if pattern X wins, model converges to X
- Entropy regularization (~0.01) keeps exploration alive early

**Training data choice**: NOT mbpp_combined. The v6→v11c data showed
the gate seldom fires there (231/240 rollouts had 0 thinks) because
most MBPP problems are single-shot solvable. Use `synth_reasoning`
(the 6-family curriculum) — multi-step tasks where thinking IS the
difference between success and failure.

**3 parallel agents dispatched**:
1. Stochastic gate + log-prob + PPO ratio + entropy bonus in
   train_rl_grader.py (the big one)
2. Scale synth_reasoning to 3000+ train + 300 held-out
3. (This file + THINKING_PLAN.md rewrite — done in-line)

## 2026-05-26 — First discovery-RL run: synth_reasoning too OOD, pivoting to mbpp

**Result on synth_reasoning_train** (100 steps):
  - reward: 0.08 → 0.02 → 0.004 → 0.000 (full collapse)
  - tier_hist: all syntax_error by step 100
  - gate(fire) stable ~0.25-0.37 (entropy reg working as designed)

**Diagnosis**: synth_reasoning prompts are too OOD from the model's
SFT distribution. The base scored ZERO on these tasks → no reward
gradient → PPO destabilized to junk output.

**Stochastic-gate mechanism IS validated mechanically**:
- Gate fire rate stayed in [0.25, 0.37] across 100 steps (didn't
  collapse to 0 or 1 despite policy noise)
- Gate entropy ~0.30 (model has preferences but not deterministic)
- PPO ratio ~1.000 (numerically stable)

What we CAN'T tell from this run: whether the gate would converge
to a useful pattern, because there was no reward gradient to drive
convergence.

**Pivot launched**: same recipe but `--dataset mbpp_combined`.
Reasoning:
- SFT base has working partial-credit signal on MBPP (8/164 historical)
- v6→v11c's "gate seldom fires on MBPP" finding was on the
  DETERMINISTIC gate; the stochastic gate forces ~30% exploration
  regardless
- If gate fire converges to a different value than its random init,
  reward signal IS reaching the gate decisions

**Open question for next run**: does gate fire EVOLVE (sign that
reward gradient reaches the gate) or stay at its entropy-regularized
value (sign that reward only flows through emit tokens)?

## (Future entries appended below as decisions are made)
