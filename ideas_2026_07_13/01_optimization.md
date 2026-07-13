# Lens 1: Optimization algorithms (verbatim agent report, 2026-07-13)

# Optimization-lens ideation: more capability per token at 287–402M / 5B tokens

Grounding searches done (papers through 2026-07). Five ideas, ranked by expected value per GPU-hour. All kill-tests assume continuation runs from existing plateau/WSD checkpoints where possible, since WSD makes the decay phase a cheap, re-runnable 15% appendix — that structural fact is exploited by three of the five ideas.

---

## 1. Ship the per-head-NS preconditioner, extended with an activation right-preconditioner (highest EV/hour)

**Mechanism.** The already-validated per-head Newton-Schulz dual on DeltaNet q/k/v/β (+7–13% iso-step, task #97, unshipped) orthogonalizes per-head gradient blocks instead of whole matrices. Extension: compose it with a Newton-Muon-style *activation covariance right-preconditioner* — maintain an EMA of per-layer input activation second moments (one d×d or diagonal-block matrix per projection, updated every ~50 steps) and right-multiply the momentum by its inverse root before the per-head NS. This whitens the input-feature geometry that per-head NS alone can't see, and is exactly where a linear RNN differs from softmax attention: DeltaNet's k/β statistics are heavily non-stationary because they gate a recurrent state.

**Why here.** This is the only optimizer lever with an in-house measured win at our exact scale, and the literature has since converged on the same block structure: Group Muon (arXiv 2605.08933) shows head-group orthogonalization accelerates Muon and analyzes when (whitening gain vs norm cost), and Newton-Muon (arXiv 2604.01472) shows the activation right-preconditioner composes with NS for a ~4% wall-clock gain. Iso-step gains that scale with head count are worth *more* to us than to big labs because we are step-rich and token-poor: any iso-step win is a pure token-efficiency win.

**Kill-test (<1 day).** Wide-10L 306M config, single GPU each arm: 1B-token A/B, Muon vs per-head-NS vs per-head-NS+right-precond, `--bf16` (the pending bf16 verification is part of this). Metric: held-out per-source CE + wall-clock. Ship if ≥3% CE at iso-token and ≤10% wall-clock overhead; kill the right-precond extension separately if it adds <1% over per-head-NS alone.

**Effect size (honest).** +7–13% iso-step is measured but may shrink at production width/duration; net wall-clock-adjusted expectation +4–8% CE-equivalent tokens. Right-precond extension: +0–4% more, genuinely uncertain.

**Novelty.** Closest: Group Muon (2605.08933) and Newton-Muon (2604.01472). Ours differs in being DeltaNet-specific (β write-gate vector included in the block structure; linear-RNN activation statistics) — a small but real delta, and regardless, it's validated in-house.

---

## 2. Selective language modeling (RHO-1-style token masking) with SmolLM2 as the reference model

**Mechanism.** Score the entire 5B-token mix once, offline, with a frozen reference model, caching per-token CE. During training, compute excess loss `CE_ours(t) − CE_ref(t)` (ours comes free from the training forward) and mask the bottom ~40% of tokens out of the loss — tokens that are either already learned (both low) or unlearnable noise (both high) get no gradient; the budget concentrates on learnable-and-not-yet-learned tokens.

**Why here.** This is the single most direct attack on our binding constraint. We have a decisive internal result that we are token-limited, not capacity-limited, and we have the ideal reference model already identified: SmolLM2-360M, which beats our 287M on HumanEval-solution CE (0.614 vs 0.92) precisely because it saw 4T tokens — its per-token loss is a high-quality "irreducible loss" estimate for our data. Scoring 5B tokens with a 360M model on one 5090 is roughly a day, once, cacheable forever. RHO-1 (arXiv 2404.07965) reports 5–10× token-efficiency gains on math and +6.8% average downstream on general pretraining with exactly this recipe; RHO-LOSS (2206.07137) and Irreducible Curriculum (2310.15389) give the principled backing. Caveat we should state upfront: our mix is already curated, so gains will be smaller than RHO-1's web-scale numbers, and a cross-architecture reference (softmax-attn ref for a DeltaNet student) may mis-rank long-range-dependency tokens — which is testable and even interesting.

**Kill-test (~1.5 days incl. scoring).** Score 2B tokens of the mix with SmolLM2-360M (subset is fine). Continuation-pretrain from an existing plateau ckpt: 500M–1B tokens with top-60% token selection vs unmasked control, iso-*trained*-token and iso-*consumed*-token variants (both matter; report both, per the fair-baselines mandate). Metric: code-source held-out CE + HumanEval-solution CE. Kill if <2% code CE gain in the iso-consumed variant.

**Effect size.** 3–10% effective-token multiplier on code sources; potentially more on the noisier web/wiki streams. Could be ~0 if curation already did the work — that's what the kill-test decides.

**Novelty.** Closest: RHO-1 (SLM), RHO-LOSS, Data Selection via Optimal Control (2410.07064). Not novel as a method; the cross-architecture-reference wrinkle and the extreme-token-poverty regime are unreported. Pure adoption value is high.

---

## 3. Decay-phase package: SAM-on-decay + soup-of-decay-branches

**Mechanism.** Two interventions confined to the WSD decay phase (15% of steps, re-runnable from a saved plateau ckpt). (a) SAM applied only during decay: perturb weights by ρ·(normalized grad), compute the gradient there, step with Muon — 2× cost but only on 15% of steps ≈ +15% total. (b) Run the decay branch K=3 times from the same plateau ckpt with different data order/shard, and uniform-average the results (soup). WSD's branch structure makes both nearly free experimentally.

**Why here.** Three converging pieces of evidence. Our own SWA result (plateau-ckpt averaging = free ~3% CE) proves the plateau explores a basin whose *center* generalizes better — decay-branch souping is the natural next rung (average endpoints of independent descents into the valley, not just plateau snapshots). Externally, SAWD / Sharpness-Aware Pretraining does exactly SAM-only-during-decay and reports up to 80% less forgetting after post-training plus quantization robustness, and "SAM selects flatter minima late in training" (2410.10373) shows the last few percent of steps is where SAM's effect concentrates. The forgetting angle is unusually valuable for us: our pipeline always does SFT+RL after pretrain, and the 2605.02105 result (31% less forgetting on an OLMo-1B ckpt from a short SAM phase) targets precisely the "SFT degrades the base / dirty-SFT" failure mode in our history. Also consistent with our weight-teleportation probe postmortem: SAM/soups are the family members that actually work (generalization, not lower loss).

**Kill-test (<0.5 day).** From an existing plateau ckpt, run the ~1500-step decay three ways: plain, SAM-Muon (ρ ∈ {0.01, 0.05}), 3-seed decay-soup. Metrics: held-out per-source CE; then run the standard SFT recipe on each and measure (i) HumanEval-solution CE and (ii) pretrain-CE regression after SFT (forgetting). Adopt SAM-decay if CE is within 0.5% of plain and post-SFT forgetting drops ≥20%; adopt soup if it beats single-decay+SWA by ≥1% CE.

**Effect size.** Raw CE: 0–2% (soup part maybe +1–2% on top of SWA). The real payoff is downstream: better SFT retention, worth several HumanEval-CE points if the forgetting numbers transfer. Honest floor: null on CE, and the forgetting effect might not transfer to Muon-trained DeltaNet.

**Novelty.** SAM-on-decay: SAWD is the closest paper (very recent tiny paper, AdamW-based) — SAM composed with Muon's orthogonalized update, and souping *decay branches* specifically (vs WSD-S's sequential reuse) are both unreported combinations.

---

## 4. Critical-batch-size-aware batch schedule (small early → large late via grad_accum ramp)

**Mechanism.** Our effective batch is fixed at ~229k tok/step (batch 14 × accum 8 × T2048) for the whole run. CBS theory says the loss-preserving batch starts near zero and grows as loss falls — so a fixed 229k almost certainly sits *above* CBS for the first chunk of training, wasting tokens (gradient averaging beyond CBS buys parallelism, not progress). Schedule `grad_accum` 1→2→4→8 keyed to loss milestones (or the simple empirical CBS probe from the Ai2 recipe), holding LR-per-token rules fixed. Implementation is a ~20-line change to the accumulation loop; zero new memory.

**Why here.** Token poverty makes "loss per token consumed" the objective, and this is the one knob that provably trades pure parallelism for token efficiency. Batch-size warmup (2505.23971) reports up to 43% fewer gradient steps *without* sample-efficiency loss going large-late; read in our direction, staying small early avoids the sample-efficiency loss of being over-CBS. "Fast Catch-Up, Late Switching" (2602.14208) shows via functional scaling laws that late-switch schedules dominate constant-batch on data consumption, and the Ai2 CBS study shows CBS scales with tokens seen, not model size — at 5B tokens our CBS is small, so the effect should be visible. One interaction to watch: smaller microbatch counts change Muon/per-head-NS gradient noise structure; also small early batch = more wall-clock steps (fine — we're step-rich).

**Kill-test (~1 day).** 1B-token from-scratch run on the 306M config (or a 150M proxy for speed), grad_accum ramp vs fixed accum=8, iso-token, iso-LR-rule. Metric: VAL CE at 1B tokens. Adopt if ≥1% CE; this is cheap enough that a null is fine.

**Effect size.** 0.5–2% final CE, front-loaded (may partially wash out by end of run). Modest but nearly free and permanent once in the launcher.

**Novelty.** None claimed — pure adoption of 2025/26 results. Nobody has run it on a linear-RNN + Muon stack.

---

## 5. Online gradient-alignment mixture reweighting toward a code dev-gradient (DGA-style)

**Mechanism.** Every N (~100) steps, compute the gradient of a small held-out *code* dev batch (HumanEval-solution-style text). Maintain per-source EMA of cos(g_source, g_dev) using the per-source microbatches the mix already produces, and multiplicatively reweight source sampling probabilities toward positively-aligned sources (with a floor so no source dies — cf. the min_content_len postmortem). This turns the static hand-tuned mix into a closed-loop controller optimizing "gradient progress on the target task per token".

**Why here.** Our per-source CE probes repeatedly showed sources interact non-obviously (bigvul/cybernative drift, recall-mix opportunity cost at ~+0.2 ppl), and the mix weights are hand-set. Under token poverty, a token spent on a source whose gradient opposes the code-dev direction is worse than wasted. DoGE (2310.15393) and DGA (2410.02498) validate exactly this signal at small proxy scale — and we get to skip DoGE's proxy-model stage because we'd run it online on the real model. Main risk: over-steering toward code collapses general CE that code capability secretly depends on; the floor + per-source CE guardrails handle that.

**Kill-test (~1 day).** 1B-token continuation from a plateau ckpt: DGA-reweighted mix vs static mix, iso-token. Metrics: code-source held-out CE + HumanEval-solution CE (primary), all other per-source CEs (guardrail: no source regresses >2%). Adopt if code CE improves ≥2% with guardrail intact.

**Effect size.** 2–5% on code CE plausibly; could be null if the hand-tuned mix is already near-optimal. Extra cost: one dev-batch backward per 100 steps ≈ <1% overhead.

**Novelty.** Closest: DGA, DoGE, GRAPE (2505.20380). Adoption with a target-task (code) twist at a scale where the proxy-model stage collapses into the main run.

---

## Ideas considered and deliberately excluded

- **MARS-M / LiMuon variance reduction on Muon** (2510.21800): improves Muon's rate and beats it empirically on GPT-2-scale — legitimate, but requires a second gradient evaluation per step, and our SOAP result showed wall-clock kills per-step wins at our scale. Worth a note in NEXT_DIRECTIONS, not a top-5 slot.
- **bf16+SR master weights** (2502.20566): validated BF16+stochastic-rounding beating mixed precision (1.54× throughput, 30% memory). Real, but it's a throughput/memory lever, not capability-per-token, and Muon's NS step complicates the SR analysis. Do it when memory blocks a batch-schedule or per-head-NS run, not before.
- **muP transfer**: our scale is essentially fixed; the tuning-run savings don't amortize.

## Single best bet

**Idea 1 — ship the per-head-NS preconditioner.** It is the only candidate with an *in-house, already-measured* +7–13% iso-step win on the exact architecture, the literature has independently converged on the same structure (which raises confidence — nobody has it for DeltaNet's q/k/v/β), and its kill-test is a one-day A/B whose failure mode is merely "wall-clock overhead too high". Under token poverty, a validated iso-step optimizer win is the cleanest possible currency: it converts directly into effective tokens with zero data-pipeline risk. Ideas 2 and 3 are the high-variance follow-ons; 1 is the one I would run tonight.

Sources: Group Muon (arXiv 2605.08933), Newton-Muon (2604.01472), RHO-1 (2404.07965), RHO-LOSS (2206.07137), Irreducible Curriculum (2310.15389), Data Selection via Optimal Control (2410.07064), SAWD/Sharpness-Aware Pretraining (GRaM @ ICLR 2026), SAM Mitigates Catastrophic Forgetting (2605.02105), SAM Selects Flatter Minima Late (2410.10373), WSD River Valley (2410.05192), Critical Batch Size Revisited (2505.23971), Fast Catch-Up Late Switching (2602.14208), Ai2 CBS blog, DoGE (2310.15393), DGA (2410.02498), GRAPE (2505.20380), MARS (2411.10438), MARS-M (2510.21800), Stochastic Rounding for LLM Training (2502.20566).
