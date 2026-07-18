# State Cartridges on Delta-Rule Models — pre-registered plan (2026-07-19)

## Claim under test

Per-segment DeltaNet states, computed **independently and in parallel** (each
O(1) memory, zero training), can be **mean-merged** into a single 8.2 MiB
state that recovers most of the benefit of full sequential repo ingestion on
held-out repo tasks. If true: instant repo context at constant memory —
"cartridges" without the distillation loop, on the architecture where state
composition is algebraically natural (delta rule = sum of rank-1 updates).

Positioning vs literature (from the ideation-fleet novelty check): Cartridges
(KV-cache self-study distillation) and State Soup (Mamba, synthetic tasks)
exist; **zero-training parallel-composable states on real repository code,
on delta-rule models, is unclaimed.** Our own evidence: 72% mean-merge recall
retention on the synthetic multibind probe (`probe_state_algebra.py`).

## Assets reused (all existing, frozen)

- Model: `checkpoints/production_lean_soup3.pt` (production base, HE-CE 0.6614).
- Eval set: `data/repo_episodes` **frozen** eval split (150 episodes,
  held-out repos, context 8–32k → task), built for meta-TTT P0.
- Harness: `eval_repo_adaptive.py` (task-span CE; real/shuffled/none arms;
  the known incidental-lift anchor: ≈ +0.246 span-CE for real context on an
  untuned base).
- State machinery: FLA `chunk_delta_rule(..., initial_state)` carry
  (`TinyLM._step_block`, validated by the meta-TTT state-carry equivalence
  test); merge ops + conv-state handling from `probe_state_algebra.py`.

## Arms (per episode; eval-only, NO training anywhere)

- **A. sequential** — full repo context prepended (ceiling; = the harness's
  existing "real" arm).
- **B. cartridge** — context split into fixed ~2k-token segments at file
  boundaries where possible; model run over each segment independently from
  zero state; final `recurrent_state`s **mean-merged**; merged state injected
  as `initial_state` for the task. Conv state from the LAST segment (the
  probe's validated choice).
- **C. none** — no context (floor).
- **D. shuffled-cartridge** — arm B with each segment's lines shuffled
  (token-count preserved): does the cartridge carry STRUCTURE or a bag of
  tokens? Mirrors the meta-TTT control design.
- Secondary sweep on B: segments-per-repo ∈ {2, 4, 8} (retention-vs-
  parallelism curve — the paper's cost-quality plot).

## Metrics & decision rule (pre-registered)

Task-span CE per arm, averaged over the 150 frozen episodes (identical
tokenization/loss placement as the harness — pinned equal to eval CE).

- lift(X) = CE(none) − CE(X);  **retention = lift(cartridge) / lift(sequential)**.
- Sanity gate (must hold or the run is void, not a result):
  lift(sequential) ≥ +0.15 span-CE (reproduces the known incidental lift).
- **PASS: retention ≥ 0.60** → cartridges viable; proceed to writeup +
  self-study tier.
- **STRONG PASS: retention ≥ 0.75** → headline claim.
- **KILL: retention < 0.35** → cartridges dead on real code; record and stop
  (the synthetic 72% then reads as a probe-vs-reality gap, itself worth a
  paragraph).
- Structure check: lift(shuffled-cartridge) must be < lift(cartridge) by a
  clear margin for any structural claim; if they tie, the cartridge is a
  token-bag prior (report honestly — still useful, weaker claim).
- Report per-episode variance + bootstrap CI on retention; no single-number
  claims.

## Cost

Eval-only: 150 episodes × ~5 arm-variants of forward passes, both GPUs →
hours, not days. No checkpoint is modified.

## Follow-up (queued, separate registration)

Search-native decoding: state-checkpointed program tree search with the
Stage-A/B neural interpreter as value function on non-executable prefixes —
the high-risk/high-novelty composition. Prereq: exec-trace re-attach on the
production base (absorbs old task #18).
