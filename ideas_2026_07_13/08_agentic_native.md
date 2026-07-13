# Lens: Agentic-native training (verbatim agent report, 2026-07-13)

Ideation complete. Below are 5 ideas through the agentic-native-training lens, ranked by expected value per GPU-hour, with novelty checks against literature through 2026-07.

---

## Ranked ideas

### 1. Hindsight repair-trace midtraining — the agent loop as a pretrain FORMAT, from on-policy errors

**Mechanism.** Stop treating edit→run→read-error→fix as an RL-time phenomenon and make it a weighted `data_mix` stream. Crucially, source the errors **on-policy**: harvest the model's own failed rollouts (the `gen_rejection_data.py --keep_all` infra already captures them), pair each with the grader's `error_text` and a verified fix (a passing sibling rollout from the same GRPO group, or the reference solution), and serialize as `[task] [attempt] [real traceback/assert diagnostics] [fix] [PASS]` multi-turn documents. Every step is machine-verified by `code_grader.py`; the corpus is infinite and self-refreshing (regenerate from each new checkpoint). Deployment (iterative self-repair loop, currently unused) then exactly matches the training distribution — error-conditioned generation is never OOD.

**Why asymmetrically ours.** (a) The 287M model is token-poor, not capacity-poor (`project_undertrained_not_undercapacity`) — a verified infinite data source is the one lever the memory says moves headlines. (b) Repair trajectories are long (attempt + trace + fix ≈ 3-5× a solution); at O(1) state we can pack sessions of many repair rounds continuously, which transformer trainers truncate. (c) The prior verdict "exec-in-pretraining = SKIP" (`project_exec_in_pretraining_verdict`) was scoped to the *HumanEval single-shot* north star and specifically rejected *synthetic-bug* recipes (consistent with arXiv 2512.02389); the north star is now the agent, and on-policy errors dodge the synthetic-bug distribution mismatch that killed that recipe.

**Kill-test (<2 days).** GPU2 overnight: rollout MBPP-train + code_exercises at τ=0.9, keep (fail, error_text, pass) triples → ~50-150M tokens of trajectories. GPU1: ~1 day continue-pretrain of the wide-10L SWA base with this as a ~15% stream. Gate: repair-conditional pass rate (given attempt+real error, does the fix pass?) vs the same base without the stream, AND single-shot pass@k to confirm no regression. The self-repair eval loop already exists — just switch it on.

**Honest expected effect.** Large gain on repair-conditional generation (this is mostly format-teaching, formats are cheap to learn); roughly flat single-shot HumanEval (OctoPack-style gen-lift needed 16B). That trade is *correct* under the agent thesis: pass@k memory says the greedy→envelope reliability gap (~3→18) is where free points sit, and a trained repair loop is the converter.

**Novelty.** Trajectory corpora exist at scale ([Open-SWE-Traces](https://arxiv.org/abs/2606.16038), SWE-Smith/SWE-Gym lineage, [terminal trajectory generation](https://arxiv.org/pdf/2602.01244)) but are big-teacher SFT distillation. On-policy, per-step-verified repair traces as a *pretrain stream* for a tiny recurrent model — with the deployment loop matched to it — is an uncrowded combination. [FeedbackEval](https://arxiv.org/html/2504.06939v2) shows feedback-driven repair is evaluated, not trained-native.

---

### 2. EdgeBench-mini as the dev signal — optimize trajectories, not point scores

**Mechanism.** Build a ~20-task compressed agentic harness before running ideas 1/3/4: small multi-file toy repos, 3-8 sequential dependent tasks each, budgeted tool calls, grader-verified milestones. Score = area under the score-vs-interaction-time curve (the ByteDance EdgeBench log-sigmoid methodology, already flagged in memory), **cost-normalized** (score per decode-FLOP), reported with bootstrap CIs.

**Why asymmetrically ours.** Cost-normalized trajectory scoring is the only metric family where an O(1)-decode model can *win* rather than trail — it operationalizes the committed north star. And the repo's documented greedy-HumanEval noise trap (`project_humaneval_config_artifact`: a 13-17 band swallowed the entire RL "lift") means every agentic idea will be unfalsifiable without this.

**Kill-test (<2 days, mostly CPU).** Build harness; validate discriminative power: does it order base < SFT < RL checkpoints monotonically with non-overlapping CIs where HumanEval-164 greedy cannot? If it can't separate known-different checkpoints, it's dead.

**Honest expected effect.** Zero direct capability gain; large multiplier on everything else. Ranked #2 on EV/GPU-hour precisely because it costs almost no GPU.

**Novelty.** Adaptation, not invention — [EdgeBench-style trajectory scoring](https://arxiv.org/pdf/2602.17547) exists; a *cost-normalized, small-model-sized* variant used as a training dev-signal does not, to my knowledge.

---

### 3. Simulate-or-shell-out: the latent executor as an internalized tool with a learned router

**Mechanism.** The model just learned to simulate ~6 execution hops latently, zero tokens. Train a router head over two actions at verification points: (a) latent-simulate (R latent steps, ~free) or (b) emit a real RUN call (grader, expensive). The label source is free and infinite: for any snippet, run *both* — `label = 1{latent prediction == real execution}` — and train the router on agreement, exactly the gate-calibration-teacher pattern already validated (`project_gate_selectivity`). At deployment the agent shells out only where its internal world model is unreliable.

**Why asymmetrically ours.** The latent executor exists *because* of the state-readonly latent-think architecture (Coconut-style feedback + β=0); nobody else has a per-decision-cheap internal interpreter in a 300M model. And under the cost-moat thesis, tool calls are the dominant deployment cost — a calibrated 50% shell-out reduction is a direct win on the metric we chose to compete on.

**Kill-test (~1 day).** The N3 exec-trace harness and gate-calibration loss both exist. Generate snippets stratified by hop-depth, train the router on agreement labels, plot the cost-accuracy Pareto vs always-run and never-run. Kill if the router never beats "always run" at any cost weighting.

**Honest expected effect.** Clean Pareto win on the synthetic trace distribution; on real code, the honest prior is the router learns "almost always run" (latent executor is validated only on synthetic ~6-hop programs, and `project_code_thinking_ceiling` says real code is recall-bound). Even that negative is decisive and cheap. Known trap: gate temperature-fragility — train the router under sampled rollouts, not greedy.

**Novelty.** Strongest of the five. [Code World Models](https://rewire.it/blog/code-world-models-teaching-llms-to-simulate-execution/) and SWE-World train *surrogates to replace execution* wholesale; [text world models](https://arxiv.org/pdf/2606.09032) simulate environments in tokens. A per-decision learned router between *latent* (zero-token) simulation and real execution, trained on agreement labels, appears unpublished.

---

### 4. Session-packing horizon curriculum — continuous-state multi-episode training

**Mechanism.** Generate synthetic *sessions*: N mini-tasks packed as one continuous stream where task_i depends on artifacts of task_j<i (reuse a helper defined earlier, respect a naming convention established in task 1, remember a config decision). Deliberately *omit* the cross-doc `cu_seqlens` boundary within a session so the recurrent state must carry the session. Curriculum ramps session length 2k → 16k → 64k tokens.

**Why asymmetrically ours.** This data format is nearly unaffordable for transformers — O(T²) attention or linear KV growth per session — which is exactly why [KLong](https://arxiv.org/abs/2602.17547) *splits* trajectories and [MEM1](https://arxiv.org/abs/2506.15841) learns to *compress* context via RL. We need neither: the compression is architectural, 8.2 MiB flat. Training natively on unsplit long sessions is the format only we can buy. It's also the supervised, gradual version of the parked meta-TTT bet, and directly targets the agent-economics finding that recall degrades 57%@4k → 17%@8k — a *training-distribution* gap, since pretrain never contained cross-task dependencies at those distances.

**Kill-test (<2 days).** Session generator (extend `gen_synthetic_memory_tasks.py` families to cross-task reuse); train two matched continues: session-packed vs doc-isolated same-tokens control; eval cross-task consistency/recall at 2× the trained horizon. Kill if the packed arm doesn't beat the control at ≥8k dependency distance.

**Honest expected effect.** Moderate — the stateful-chunking probe (`project_stateful_chunking_notworth`) warns aggregate-CE effects of state continuity are tiny; the difference here is the eval is *dependency recall*, not CE, where the dep-distance-stratified probe already showed real feature wins. Risk: bounded state interference caps the curve regardless of training (the multibind saturation lesson).

**Novelty.** [Recurrence-complete action models](https://arxiv.org/html/2510.06828) argue for recurrence in long-horizon perception; nobody trains a code agent on deliberately state-continuous multi-task sessions because nobody else's economics permit it.

---

### 5. Repair-step preference factory (execution-outcome DPO, trajectory-level)

**Mechanism.** Systematize what the dense grader gives for free: every GRPO batch yields (state+error → good fix) vs (state+error → bad fix) pairs at the *repair-step* level, not just solution level. Overnight flywheel on GPU2.

**Why ours.** Dense tier ladder + `error_text` make pair mining trivial; nobody needs a judge model.

**Kill-test (~1 day).** `gen_rejection_data.py` + `train_dpo.py` exist; mine ~50k repair-step pairs, DPO on top of idea-1's checkpoint, measure repair success + pass@k envelope.

**Honest expected effect.** Small. Prior DPO runs regressed (9, 12/164) and the pass@k memory says RL-family methods sharpen but don't grow the envelope. Repair-*step* granularity is the only new element. Ranked last; run only as a cheap rider after idea 1. Novelty: low — [dense verifiable rewards](https://arxiv.org/pdf/2601.03525), [solver-verifier self-play](https://arxiv.org/abs/2502.14948v2), [ReVeal](https://arxiv.org/html/2506.11442v1) are adjacent and crowded.

---

## Single best bet

**Idea 1 — hindsight repair-trace midtraining from on-policy errors, gated by the EdgeBench-mini harness (idea 2).** It is the only idea that simultaneously (a) attacks the confirmed binding constraint (token poverty) with a verified infinite data source, (b) makes the unused self-repair deployment loop distribution-matched instead of OOD, (c) exploits the O(1)-state cost structure at both train and deploy time, and (d) has a kill-test that runs on existing infrastructure in under two days with an unambiguous gate (repair-conditional pass rate). The others either instrument it (2), extend it (3, 4), or ride on it (5).

Sources: [Open-SWE-Traces](https://arxiv.org/abs/2606.16038) · [Terminal trajectory generation](https://arxiv.org/pdf/2602.01244) · [KLong](https://arxiv.org/abs/2602.17547) · [MEM1](https://arxiv.org/abs/2506.15841) · [Code World Models overview](https://rewire.it/blog/code-world-models-teaching-llms-to-simulate-execution/) · [Text world models for agents](https://arxiv.org/pdf/2606.09032) · [Solve-and-verify self-play](https://arxiv.org/abs/2502.14948v2) · [ReVeal](https://arxiv.org/html/2506.11442v1) · [Propose-Solve-Verify](https://arxiv.org/abs/2512.18160) · [Dense verifiable rewards for code RL](https://arxiv.org/pdf/2601.03525) · [FeedbackEval](https://arxiv.org/html/2504.06939v2) · [Execution traces for repair](https://arxiv.org/pdf/2505.04441) · [Recurrence-complete action models](https://arxiv.org/html/2510.06828) · [CoSPlay](https://arxiv.org/abs/2605.23491)
