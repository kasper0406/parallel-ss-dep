# Meta-TTT: the repo-adaptive coder (unparked 2026-07-13)

**Decision (user, 2026-07-13):** with the latent-execution program landing as
solid-but-modest science (see `LITERATURE_LATENT_EXEC_2026_07_13.md` — the
latent-CoT lane is crowded; Coconut/SIM-CoT/CWM own the mechanisms), the
project's main differentiation bet moves to **adaptivity**: meta-train
DeltaNet's recurrent state into a *deliberate test-time learner* over
full-repository ingestion. The claim being chased: **capability that scales
with deployment-time data** — your model, having read your repo at O(1) cost,
beats a bigger model that can't hold it. This is the north-star adaptivity
pillar (NORTH_STAR_2026_06_30.md), promoted from parked
(memory `project_meta_ttt_repo_adaptive`).

## Why this is ours (and why nobody's lane)

- DeltaNet's recurrence IS a learner: delta rule = Widrow-Hoff GD step, β =
  learned per-token LR. Meta-training the state dynamics for downstream task
  benefit is native to this architecture; transformers can't copy it cheaply.
- Ingestion is O(1)/8.2 MiB at ANY length (decode-cost bench, 2026-06-30):
  repo-scale contexts are ergonomically FREE for us, ruinous for full-KV.
- The 2026-07-13 literature sweep found the latent-reasoning lane crowded but
  this lane empty; the agent-economics probe (recall-past-window: graceful
  57%@4k/17%@8k decay vs windowed-transformer hard 0%) showed the raw
  substrate degrades but degrades LEARNABLY — meta-training is the bet that
  the decay curve is trainable.
- Recall co-training already proved state-teachability (0→100% in-window;
  syn64/syn128 strengthening with tokens).

## The experiment (pre-registered kill-test, ~1 week)

**Episodes**: `[repo context (8–32k tok) → task about that repo]`, loss ONLY
on the task span. Task family v1 = **cross-file usage prediction**,
constructible automatically from GitHub snapshots: identifier DEFINED in file
A (early in context), USED in file B (late); task = complete the usage
site(s). This is the dep-distance-stratified probe (our validated feature
dev-signal) scaled across files and meta-trained on.

**Arms per eval task (held-out repos)**:
1. real-repo ingestion → task
2. **shuffled-repo control** — same token count, structure broken (shuffle
   file order + line-shuffle the non-definition files; keep the task-local
   file intact so only the CROSS-FILE knowledge channel is destroyed)
3. no-ingestion (task-local context only)

**Kill line**: after meta-training, if lift(real − shuffled) on held-out
repos does not clearly exceed the same lift measured on the NON-meta-trained
base (incidental state-learning), the state dynamics aren't being meta-shaped
→ park again, write up the negative.

**Success shape** (the phase-shift form): lift(real − shuffled) that GROWS
with ingested-repo size where the base model's is flat/decaying — the
curve-bending signature (EdgeBench log-sigmoid framing,
`reference_edgebench`).

## Phases

- **P0 — episode builder + zero-train baselines (~2–3 days).** Build the
  episode/eval corpus (The Stack / GitHub snapshots; dedup vs pretrain mix);
  measure arms 1–3 on EXISTING ckpts (stageA_executor, feature_pilot_A) with
  NO training — this is the incidental-learning baseline the kill-test needs,
  and it validates the probe has headroom. Includes a T-scaling smoke (T=8192+
  fwd/bwd feasibility with activation checkpointing at batch 1–2; the model
  was trained at T=2048 — episode training needs longer T or doc-isolated
  packing).
- **P1 — meta-train pilot (~1 day GPU).** ~300M tok on episodes mixed with
  the standard code mix (retention anchor, the Stage-A/B lesson), single GPU,
  periodic ckpts, engagement diagnostics (per-arm eval every 50M tok).
- **P2 — kill decision.** Pre-registered lines above, held-out repos only.
- **P3 — if pass**: scale (longer T, more repos), curve-bending eval
  (task-lift vs ingested-tokens trajectory), integrate with the latent
  executor (the agent that reads your repo AND simulates your code), and the
  LoRA-sleep/consolidation tier (`project_ingraining_fastweights_2026_06_29`).

## Sequencing with the other tracks

- **Latent-exec program (winding down to writeup)**: depth-fix arm →
  CRUXEval-O transfer probe (`eval_cruxeval_transfer.py`, building) → write up
  "Latent Execution" + the staging finding. The trained executor remains a
  component of the eventual agent, not the headline.
- **Linearize-Qwen donor (#105 / STRATEGY fork)**: composes, not competes —
  meta-TTT needs a base worth adapting. Run the P0–P2 kill-test on the current
  base FIRST (cheap, decisive); if meta-TTT passes, port the recipe to the
  stronger linearized base as part of P3.
- **Full lean run (task #5)**: superseded in priority by P1 unless meta-TTT
  parks again.

## P1 RESULT (2026-07-14): KILLED as pre-registered — with an instrument caveat

Pilot (launch_meta_ttt_P1.sh, 2,300 steps / ~301M tok from stageA_executor,
500 train episodes): held-out kill-test (results/repo_adaptive_meta_ttt_P1.json)
**lift(shuffled−real) = +0.046 line / +0.126 span — HALF the incidental
baseline (+0.138/+0.246) the kill line required it to clearly exceed. KILL.**
Real-context CE on novel repos DEGRADED (1.79 vs untrained 1.65) while
no-context improved (1.21 vs 1.32): the model got worse at using unseen repo
context.

**Autopsy — the instrument failed before the hypothesis was tested:** training
mttt CE hit 0.000 by step ~700 (500 episodes fully memorized at ~30% of the
run) → 70% of the run delivered zero gradient on the meta objective and pure
overfit consolidation. The N1→N1′ precedent applies: this kills the PILOT, not
cleanly the mechanism. IF the bet is retried, the single-variable fix is a
10–50× episode corpus (scan more of codeparrot; the 300k-row scan was a pilot
slice) + early-stop on train-CE saturation. Otherwise: park per registration.

## P1′ PRE-REGISTRATION (locked 2026-07-14, BEFORE corpus generation)

Single retry of the pilot with the instrument fixed. Locked lines:
1. **Corpus**: ≥5,000 train episodes from repos DISJOINT from the frozen P0
   eval set (`data/repo_episodes/eval*.jsonl` is reused UNCHANGED for
   comparability with the P0 baselines; the deterministic repo-hash split
   guarantees new train repos exclude eval-side repos).
2. **Engagement guard**: if train mttt CE saturates (<0.1 sustained) before
   60% of the run, the run is VOID (instrument still too small) — it does not
   count as a mechanism test either way.
3. **Kill line unchanged**: held-out lift(shuffled−real) must CLEARLY exceed
   the incidental baseline (+0.138 line / +0.246 span on stageA_executor).
4. **No P1′′.** A valid kill parks meta-TTT; adaptivity falls back to the
   state-cartridges/self-study tier (kept alive by the State-Algebra partial-
   additivity result) + LoRA-sleep consolidation.

## P1′ RESULT (2026-07-14): VALID KILL — meta-TTT PARKED

The instrument-fixed retry (6,000 episodes / 2,458 repos, zero eval overlap,
legacy-FIM lineage-matched) PASSED the engagement guard (mttt CE 0.5–1.6
through the whole run, no saturation) and still failed the kill line:
held-out lift(shuffled−real) = **+0.073 line / +0.174 span vs the incidental
bar +0.138/+0.246** (results/repo_adaptive_meta_ttt_P1prime.json). Real
context still hurts vs none (−0.20 line, improved from −0.32 untrained /
−0.58 P1). Bucket profile: structure signal decays 4–8k → 16–32k.

**Conclusion: with this recipe (0.1-weight aux, truncated BPTT with gradient
through only the final ~4k ingestion tokens, 300M tok on a 2048-trained
base), meta-training does not shape the state dynamics toward better
novel-repo use — it mildly dulls the incidental signal while improving the
no-context path. Per the locked registration: NO P1′′; meta-TTT is PARKED.**

If ever revisited (bigger compute era), the known mechanistic suspects, in
order: (1) the truncated gradient — state-producing weights for EARLY context
never receive gradient (grad_chunks=2 was a memory compromise; true
state-shaping likely needs gradient through the full ingestion, i.e. BPTT
memory we don't have); (2) T=2048-trained base dynamics at 8–32k; (3) the
aux-weight/mix balance. Adaptivity pillar falls back to: state cartridges /
self-study (kept alive by the State-Algebra partial-additivity result) and
LoRA-sleep consolidation.
