# Lens: Architecture (verbatim agent report, 2026-07-13)

# ARCHITECTURE ideation — 5 ideas for a 287–402M bounded-state code model

All ideas preserve the committed thesis (O(1)-or-near-O(1) decode state, no unbounded KV). Ranked by expected-value-per-GPU-hour at the end; kill-tests sized for 2×5090 and reuse existing harnesses wherever possible.

---

## Idea 1 — Minimal bounded-softmax hybrid, placed where the tax actually lives (G&A-guided SWA layers)

**Mechanism.** Replace 1–2 of the DeltaNet layers (or even just a few *heads* within two layers) with sliding-window softmax attention (W=1024–2048), optionally with ~16 persistent "sink/global" slots. Everything else stays chunk_delta_rule. KV cache is window-bounded: at d_model=960, W=1024, bf16, 2 layers ≈ 7.5 MiB — the *same order as the existing 8.2 MiB recurrent state*, so the cost moat survives essentially intact. Placement is not uniform: the [Gather-and-Aggregate paper (Bick, Xing & Gu, ICML 2025)](https://arxiv.org/abs/2504.18574) shows the recurrent-vs-transformer retrieval gap concentrates in a *handful of heads* — and that replacing a single G&A head in a pretrained SSM with an attention variant measurably boosts retrieval. So: run a G&A-style head attribution on the linearized/donor model first, then spend the softmax budget exactly there.

**Why it fits our evidence.** The +0.145 CE linear-attention tax is our single largest measured quality deficit, and it *grows with donor strength* — meaning it caps the whole linearization strategy. The hybrid ablation harness (`linearize_hybrid_ablation.py`) already exists with the kill-signal defined ("hybrid CE drop >15–20% vs baseline"). This idea sharpens the existing fork with two additions the harness doesn't yet test: (a) *bounded* SWA instead of full attention, so a positive result is directly adoptable rather than thesis-violating; (b) head-level targeting instead of layer-level, which is the cheapest possible softmax budget. Industry has converged on layerwise 3:1 hybrids ([Kimi Linear, 3:1 KDA:MLA](https://arxiv.org/abs/2510.26692); [Qwen3-Next 3:1 Gated-DeltaNet:attention](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)) — but with *unbounded* full-attention layers. Whether window-bounded softmax recovers most of the tax at 400M is exactly the open question we're positioned to answer, and G&A suggests yes (the deficit is aggregation over a mostly-local neighborhood plus a few retrieval events, not global mixing).

**Kill-test (<1.5 days, both GPUs).** Four arms in the existing harness, iso-param, KD-heal from the same donor: (1) pure DeltaNet baseline; (2) hybrid with 2 full-attention layers (upper bound); (3) hybrid with 2 SWA-1024 layers + sink tokens; (4) SWA layers placed by G&A head attribution vs. placed at the default depth (e.g., L5/L8). Metrics: HumanEval-solution CE, the dep-distance-stratified code-CE probe (our first validated feature-sensitive dev signal), recall-past-window at 4k/8k. Decision rule: SWA hybrid recovers ≥70% of the full-attention hybrid's tax reduction → adopt permanently.

**Honest expected effect.** Best-grounded quality lever on the list: plausibly recovers 0.05–0.10 of the 0.145 CE tax and most of the hybrid's retrieval lift, at ~7–15 MiB extra bounded state. It will *not* fix the token-poverty knowledge gap. Risk: at 4k–8k contexts the retrieval events a coding agent needs may exceed W — that's what the recall-past-window arm measures.

**Novelty check.** Layerwise linear/softmax hybrids are standard ([Kimi Linear](https://arxiv.org/abs/2510.26692), Qwen3-Next, Samba). The delta: *bounded-window* softmax + *G&A-attributed placement* at sub-1B scale. Closest to "replace a single G&A head" in [Bick et al.](https://arxiv.org/abs/2504.18574), which was an analysis, not a trained recipe.

---

## Idea 2 — DeltaProduct-2: higher-rank state transitions, possibly only in early layers

**Mechanism.** Swap `chunk_delta_rule` for [DeltaProduct (Siems et al., NeurIPS 2025)](https://arxiv.org/abs/2502.10297) with n_h=2: each token applies a *product of two generalized Householder transforms* to the state — equivalently two gradient steps per token on the associative-recall loss instead of one. Same state size; strictly more expressive transition matrix (can express rotations/permutations, not just single reflections). Code exists at [automl/DeltaProduct](https://github.com/automl/DeltaProduct), built on FLA — drops into our fork.

**Why it fits our evidence.** DeltaProduct is explicitly flagged untested in the repo. The capability it buys — state *tracking* (group composition, length extrapolation) — is precisely what our newest win needs: the exec-trace latent program simulates ~6 hops of program execution in hidden state, and variable-state simulation is a state-tracking task, not a recall task. This dodges the niche-features post-mortem trap: it's not a bolt-on module that can sit inert off the gradient path (the engagement failure mode of PKM/WM/latent adapters); it changes the recurrence substrate itself, so it is exercised by every token from step 0. The heterogeneous-composition collapse also gives it a concrete target: a rank-2 transition can compose two "ops" per step within the state itself.

**Kill-test (<2 days).** Three probes in parallel: (a) the N3 exec-trace kill-test harness (already validated against N0) with DeltaProduct-2 vs DeltaNet, both trained fresh — does the per-forward dependent-hop budget move past ~2? (b) saturating multibind + pointer-chase synthetics; (c) a ~300M, 1–2B-token iso-*param* short pretrain (DeltaProduct adds per-sub-step k/β projections — shrink d_head or heads to compensate, per the fair-baselines mandate) scored on dep-distance-stratified code CE. One arm per GPU.

**Honest expected effect.** Paper shows *modest* LM perplexity gains but strong state-tracking and length-extrapolation improvements; expect the same shape here: small or nil on code CE, potentially decisive on exec-trace hop budget and on latent-iteration quality. ~2× recurrence FLOPs in converted layers (recurrence is not our wall-clock bottleneck at T=2048).

**Novelty check.** Direct application of [DeltaProduct](https://arxiv.org/abs/2502.10297); the novel coupling is *DeltaProduct × latent execution-trace thinking* — using the richer transition to raise the per-forward dependent-computation budget, which nobody has measured.

---

## Idea 3 — Per-step op-selector for latent iteration: routed LoRA-experts over the looped trunk

**Mechanism.** During the R-step latent iteration, insert a small router that reads the current hidden state and selects one of E lightweight "op experts" (rank-16–32 LoRA deltas or FiLM modulations on the shared trunk's MLPs) *per latent step*. The trunk stays shared; only the modulation varies per iteration. This converts "iterate one fixed function R times" (which we proved collapses on heterogeneous composition) into "compose R functions drawn from a learned library" — depth diversity at ~2–5% parameter overhead, zero change to decode state.

**Why it fits our evidence.** This is the repo's own prescription: the depth-via-iteration probe concluded latent-R simulates depth for *homogeneous* iteration but collapses to chance on *heterogeneous* K-op composition, "fix = per-step op-selector, not width/R." The exec-trace latent program (the current win) is the immediate customer: real program execution is heterogeneous (arithmetic step, comparison, branch, index). And per general-latent-training findings, the router gives the "per-step source of task knowledge" that recruit-only signals couldn't.

**Kill-test (<1 day, single GPU).** Rerun the exact heterogeneous-composition synthetic grid with the routed variant: shallow trunk + R latent steps + router over E=4–8 op-LoRAs, K distinct ops in {2,4,8}. Kill-gate: restore ≥0.9 accuracy at K=4–8 where the single trunk sits at chance, with the router's op-choice entropy showing actual specialization (not uniform). If passed, transfer check on the N3 exec-trace harness. Hours per cell; the whole grid is an afternoon.

**Honest expected effect.** High probability of a decisive synthetic result (the mechanism is almost definitionally the missing piece). Transfer to code benchmarks is genuinely uncertain — the code-thinking-ceiling finding (per-token code is recall-not-iteration; latent-on-code null was real) means this upgrades the exec-trace/CRUXEval-style capability, not HumanEval greedy. Score it as science that makes latent thinking *composable*, not a headline mover.

**Novelty check.** Close prior art appeared in 2026: [LoopMoE](https://arxiv.org/html/2606.04438) (looped MoE with iteration-conditioned modulation), [LoopFormer](https://arxiv.org/pdf/2602.11451) (elastic-depth loops via shortcut modulation), and [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) (per-token recursion depth routing, NeurIPS 2025). None do *state-dependent per-step op selection inside a latent (no-token-emitted) iteration on a linear-RNN trunk* — but the delta over LoopMoE is modest; the value is that it's the surgically-correct fix for our measured collapse.

---

## Idea 4 — Multi-timescale state: log-structured memory hierarchy over the delta rule

**Mechanism.** Two tiers, cheapest first. (a) *Decay banks:* Gated DeltaNet-style per-head forgetting, but with head decay rates initialized/pinned to a geometric ladder of timescales (τ ≈ 128 … 32k), so different heads are structurally specialized for different retention horizons instead of all heads learning the same effective window. (b) *Log-state:* adopt [Log-Linear Attention (Guo et al., ICLR 2026)](https://arxiv.org/pdf/2506.04761) — a Fenwick-tree hierarchy of states over the prefix (fine-grained recent buckets, coarse distant summaries), which the paper explicitly instantiates on Gated DeltaNet. State grows O(log T): at 131k context that's ~17 states ≈ 140 MiB vs. the KV cache's 5 GiB — the cost moat is dented, not broken.

**Why it fits our evidence.** The agent-economics probe found our real long-context weakness: recall degrades 57%→17% from 4k→8k — graceful vs. a windowed transformer's collapse, but still decaying, and the current fix on the roadmap is "generic RAG." This attacks the decay *in the substrate*. Critically, it avoids the niche-features failure mode that killed WM/PKM as code-benchmark movers: those were separate read modules gated by think-tokens (structurally inert unless the data forces engagement); a decay bank / log-state is inside the recurrence of every token, so there is no engagement gate to mis-calibrate. Prereq note: `gated_deltanet` previously crashed on sm_120, but the FLA Blackwell-detection bug was root-caused and fixed in our fork (2026-06-05) — re-testing it is itself overdue.

**Kill-test (<2 days).** Tier (a) first: re-validate gated_deltanet on Blackwell (hours), then train ~120–300M arms — DeltaNet, GDN-learned-decay, GDN-decay-ladder — on the recall-past-window probe + saturating multibind, T=8k–16k. Kill-gate: does the 8k recall number move 17%→≥40% at iso-param, and does the decay curve flatten? Tier (b) only if (a) shows timescale specialization matters but caps out (log-linear needs the authors' Triton kernels; budget 2+ days, so it's the follow-up, not the kill-test).

**Honest expected effect.** Likely ppl-neutral at T=2048 (GDN vs DeltaNet differences are small at this scale); the payoff is entirely in the >4k-context regime that the agent thesis lives on. If tier (a) shows nothing, tier (b) is probably also not worth it at our scale.

**Novelty check.** [Log-Linear Attention](https://arxiv.org/pdf/2506.04761) and its follow-up [Adaptive Memory Decay for Log-Linear Attention](https://arxiv.org/html/2605.06946) own the log-state idea; multi-timescale decay priors echo Mamba/S4 initialization lore and [Titans/ATLAS](https://arxiv.org/abs/2505.23735)-style multi-tier memory. Our delta is the deployment setting: bounded-state coding *agent* at 400M where the cost-vs-recall curve is the product.

---

## Idea 5 — Mamba-3-style update upgrades: trapezoidal (two-token) delta updates + MIMO decode

**Mechanism.** Steal two pieces of [Mamba-3 (ICLR 2026)](https://arxiv.org/pdf/2603.15569): (a) *trapezoidal discretization* — the state update blends the current and previous token's contribution (a second-order integrator) instead of the delta rule's single-token outer product; (b) *MIMO decode* — restructure the rank-1 outer-product state update into a small matrix-matrix update, raising arithmetic intensity at decode with no latency cost.

**Why it fits our evidence.** (b) targets a measured, on-thesis deficit: our `forward_step` decode is a constant ~2.3× slower than it should be — an arithmetic-intensity problem, which MIMO addresses by design. (a) is a low-risk expressivity bump validated at 1.5B against Gated DeltaNet itself. Neither adds state; both keep O(1) exactly.

**Kill-test (<2 days, but kernel-heavy).** (b): rewrite the decode step as a rank-r (r=4–8) grouped update and benchmark tokens/s on the existing DECODE_COST_BENCH harness — pure engineering, decisive in a day. (a): implement the three-term update in a small non-fused reference, test on multibind/pointer-chase + 120M short pretrain vs delta rule; port to the fused chunk kernel only if it wins.

**Honest expected effect.** (b): real throughput win (maybe most of the 2.3× gap), zero quality change — it strengthens the moat, not the model. (a): small; Mamba-3's gains over GDN at 1.5B are a fraction of a point on downstream averages, and at 400M with our token budget it may be invisible. This is the "sharpen the thesis" idea, not the "punch above weight" idea.

**Novelty check.** Straight adaptation of [Mamba-3](https://arxiv.org/pdf/2603.15569) mechanisms to the delta rule; MIMO-for-DeltaNet-decode specifically appears unexplored, but it's an efficiency contribution.

---

## Ranking by expected value per GPU-hour

| rank | idea | why |
|---|---|---|
| 1 | **G&A-guided bounded-SWA hybrid** | Attacks the largest measured quality deficit (+0.145 CE tax); harness already built; a positive result is immediately adoptable without breaking the moat; industry priors (3:1 hybrids) say the effect is real and large. |
| 2 | **Per-step op-selector for latent iteration** | Cheapest test on the list (hours, synthetic, existing grid); surgically matches a measured failure; upgrades the one mechanism (exec-trace latent) currently winning. Capped upside on code headlines keeps it at #2. |
| 3 | **DeltaProduct-2** | Code exists, kill-tests exist, substrate-level so it can't be inert; but paper priors say LM gains are modest — the bet is specifically on the exec-trace hop budget. |
| 4 | **Multi-timescale decay banks / log-state** | On-thesis (agentic long-context), engagement-proof by construction; but payoff only in >4k regime and tier-(b) is implementation-heavy. |
| 5 | **Mamba-3 trapezoidal + MIMO decode** | MIMO is near-certain but moat-only; trapezoidal quality gain likely invisible at 400M. |

## Single best bet

**Idea 1.** The project's own evidence chain — tax is real, tax grows with donor strength, tax bottlenecks the linearization pivot, the gap concentrates in a few gather-and-aggregate heads, and the ablation harness is already written — points at one experiment: *find out whether 1–2 window-bounded softmax layers (placed by G&A attribution) recover ≥70% of what full-attention hybrid layers recover.* If yes, the model keeps ~16 MiB total bounded state, the cost moat stands, and the single biggest quality objection to the whole thesis is retired for roughly one day of GPU time. Fold Idea 2 or 3 in afterward as the latent-execution upgrade on top of the hybrid base.

Sources: [DeltaProduct (arXiv 2502.10297)](https://arxiv.org/abs/2502.10297) · [automl/DeltaProduct](https://github.com/automl/DeltaProduct) · [Log-Linear Attention (arXiv 2506.04761)](https://arxiv.org/pdf/2506.04761) · [Adaptive Memory Decay for Log-Linear Attention](https://arxiv.org/html/2605.06946) · [Kimi Linear (arXiv 2510.26692)](https://arxiv.org/abs/2510.26692) · [Qwen3-Next blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) · [Gather-and-Aggregate (arXiv 2504.18574)](https://arxiv.org/abs/2504.18574) · [Mixture-of-Recursions (arXiv 2507.10524)](https://arxiv.org/abs/2507.10524) · [LoopMoE (arXiv 2606.04438)](https://arxiv.org/html/2606.04438) · [LoopFormer (arXiv 2602.11451)](https://arxiv.org/pdf/2602.11451) · [Mamba-3 (arXiv 2603.15569)](https://arxiv.org/pdf/2603.15569) · [Titans (arXiv 2501.00663)](https://arxiv.org/abs/2501.00663) · [ATLAS (arXiv 2505.23735)](https://arxiv.org/abs/2505.23735)
