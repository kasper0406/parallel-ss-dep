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

## (Future entries appended below as decisions are made)
