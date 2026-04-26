# Brainstorm Synthesis — 10 Parallel Agents

**Date:** 2026-04-26
**Trigger:** Hybrid SO(n)+DeltaNet is mechanistically clean but practically a net loss for general LM (1.07-1.22× WORSE PPL + 1.27× wall-clock vs pure DeltaNet on TinyStories + Python code at 135M / 5k steps). User asked for fresh alternatives.

Per-agent briefs in `brainstorm/01-10_*.md`. This file ranks the cross-cutting themes and proposes a concrete roadmap.

---

## Convergent themes (multiple agents independently converged)

### 1. The recall mechanism, not the algebra, is the bottleneck (Agents 07, 09, 03)
- **Agent 09**: 2024-2026 frontier converged on **softmax-attention + linear-RNN hybrids** (Samba, Jamba, Hymba, Granite-4, Zamba2, Qwen3.5-Next). *Never* two-linear-RNN. Industry chose the recipe we didn't.
- **Agent 07**: Linear-RNN students cannot match softmax teachers from-scratch at small scale. Distillation closes 70-95% of the gap with <1% of pre-train tokens.
- **Agent 03**: kNN-cache à la Memorizing Transformer is the only design surveyed with documented PPL wins on GitHub specifically.

**Implication:** the parity/group-theoretic story is real but a sideshow. Code doesn't need parity — it needs content-addressable recall (variable defs, type sigs, function refs).

### 2. "Capacity exists but unused" — training fixes are cheap (Agents 08, 04, 05)
The kill-gate (mod-p, parity) shows the cell has the capacity. PPL gap shows it isn't being used on real data.
- **Agent 08**: identity-init for ortho + log-warmup of rotation magnitude (α: 0→π over 1-2k steps); add structural-probe aux loss (predict bracket-depth from hidden state, weight 0.05 → 0). ~1 day eng, free compute. Anchor: Amos et al. ICLR'24 closed SSM-vs-Transformer LRA gap entirely with denoising pretrain.
- **Agent 05**: 25/75 fails because we pay ortho tax on every token when only ~5% of tokens need parity-tracking. **Soft-MoE over cell outputs** or **Mixture-of-Depths gating ortho on top-k% tokens** fixes this directly. 1-2 days eng.
- **Agent 04**: multi-timescale DeltaNet — split heads into K bands with log-spaced decay priors λ ∈ {0.5..0.9999}. One config knob. Parallel-scan-native.

### 3. Drop-in cell upgrades subsume the hybrid (Agents 01, 02)
The "hybrid two layer types" pattern may be obsoleted by single cells that absorb both walls:
- **Agent 01**: **Gated DeltaProduct** (Siems et al. Feb'25, arXiv 2502.10297) — Householder products with n_h≥2 provably solve group-word problems in 3 layers. Drop-in fla swap. **~1 day. Highest expected value of any cheap experiment.**
- **Agent 02**: content-bearing stack monoid — context-free composition (escapes Merrill-Sabharwal TC⁰ wall on Dyck-2). Directly matches code's bracket/scope/indent. Weekend kernel work.
- Both agents flagged that PD-SSM (NeurIPS'25 spotlight), DeltaProduct, and PaTH already cover the FSA / solvable-group / Householder-product space we explored.

### 4. Inference policy is an orthogonal lever (Agent 10)
Linear-RNN states are *cheap to fork* (one matrix vs T·d KV-cache). Code generation rewards pass@k.
- **S\* state-forking Best-of-N** (Li et al. 2025, EMNLP): +8-18 pp pass@1 on Qwen2.5-Coder. M-cost. Hits HumanEval/MBPP directly. Orthogonal to architecture.
- **Mixture-of-Recursions** (Bae et al., NeurIPS'25): one shared block recursed K times with per-token routing. Reported new Pareto frontier *exactly at our 135M-1.7B scale*. Train-time bet.

---

## What to skip / deprioritize

- **More algebraic structures** beyond Gated DeltaProduct: Agent 02 confirmed the FSA/solvable-group space is now well-covered by published 2025 work. Returns are diminishing.
- **GL_n(GF(p)), exotic Lie groups, free groups**: synthetic wins don't move text PPL. We learned this from our own results.
- **Modern Hopfield, Energy Transformer, Titans-pure**: research risk too high given crowded space.
- **Sequence parallel across the 5090 pair**: no NVLink → premature.
- **Reversible RNN blocks**: architectural risk during hyperparam hunt.
- **Phi-style synthetic data generation**: heroic data-gen effort that distillation provides for free.

---

## Recommended roadmap

Three phases, with explicit decision gates between them. Total Phase 1 cost: ~1 GPU-week of compute, 5-7 days of engineering.

### Phase 1 — Cheap parallel experiments (this week)

Goal: find out *which* of the four "explanations for the PPL gap" is correct. Run all four in parallel on 2× RTX 5090. Each is ≤2 days eng + ≤1 day compute.

| # | Bet | Hypothesis being tested | Eng cost | Compute | Source agents |
|---|-----|------------------------|----------|---------|--------------|
| 1.1 | **Gated DeltaProduct (n_h=2,3,4) replaces ortho+delta hybrid entirely** | "A stronger single cell subsumes the hybrid" | ~1 day (fla wrapper) | ~6h on one 5090 | 01 |
| 1.2 | **Identity-init + warmup α + structural-probe aux loss on existing hybrid** | "Capacity exists but training is broken" | ~1 day | rerun existing hybrid baseline | 08 |
| 1.3 | **Samba-Delta: DeltaNet + SWA(W=256), 1:1 alternating** | "Recall mechanism is what matters; rotation is a wrong turn" | ~0.5 day (FlashAttn SWA + fla) | ~6h | 09 |
| 1.4 | **Soft-MoE over ortho/delta outputs** (per-token convex combo) | "Token-level routing fixes the 25/75 ratio failure" | ~1.5 days | ~6h | 05 |

**Decision gate after Phase 1:**
- If 1.1 (Gated DeltaProduct) wins outright → ship it as the cell. Phase 2 becomes "scale + distill that cell."
- If 1.3 (Samba-Delta) wins outright → pivot to softmax+linear and largely retire the rotation work.
- If 1.2 fixes the ortho hybrid → keep architecture, double down on training innovations.
- If 1.4 fixes the ratio → token-routing is the right framing; investigate MoR next.
- If none clearly wins → the architecture story is dead at this scale; pivot to Phase 2C distillation.

### Phase 2 — Conditional on Phase 1 (next 2 weeks)

**2A — If a cell winner emerged:** scale that single cell to a clean 1.7B run with multi-timescale DeltaNet bands (Agent 04, free) and do a head-to-head vs DeltaNet@1.7B on TinyStories + Python.

**2B — If MoR routing seems live:** train MoR-cell variant (Agent 10) — one Gated DeltaProduct block recursed K times with per-token depth routing. Reported Pareto frontier at our scale.

**2C — Distillation track (run regardless):** **LoLCATs smoke test** (Agent 07) ~3 GPU-days from Qwen2.5-Coder-1.5B. This is the cheapest way to find out whether *any* of our cells can absorb a coding teacher. If yes, Phase 3 is the coding-LLM end-game. If no, we've saved months of dead-end from-scratch training.
- Critical eval: must include autoregressive HumanEval generation, *not just* PPL. Agent 07 flagged a 2026 paper showing 0.2pp PPL match coexisting with 20.8pp HumanEval gap.

### Phase 3 — Coding-LLM end-game (subsequent weeks, conditional)

Stack:
1. Best cell from Phase 1/2A
2. Distilled from Qwen2.5-Coder-1.5B (Phase 2C path) using HALO/HypeNet hybrid distillation (Agent 07): 25% softmax layers selected by HALO metric, 3-stage hidden→KL→long-context, ~5-7 GPU-days, ~2.3B tokens.
3. **S\* state-forking inference** at serve time (Agent 10): exploit our cheap-state advantage for HumanEval@k.
4. On-policy distillation as final polish (Thinking Machines, +2-4 GPU-days; reported 9-30× efficiency over RL).

Total estimated budget to a useful 1.5-3B coding model on 2× RTX 5090: **~3 GPU-weeks**, conditional on Phase 1 finding a competitive cell.

---

## Long-shot directions to keep on the back-burner

- **kNN-cache for variable-def recall** (Agent 03): document-level read-through cache for `def`/`class`/`import` tokens. Unique angle but big eng lift; revisit if Phase 1 cells still struggle on long-range pointer-chasing.
- **Content-bearing stack monoid** (Agent 02): if we want a *novel* publishable contribution rather than tracking the frontier, the context-free direction is the open territory. Weekend kernel work, but eval methodology is unclear.
- **State-checkpointed long-context training** (Agent 06): test whether Heisenberg-vs-DeltaNet parity advantage widens at T=8k-32k. Important *if* we're claiming long-context as a differentiator, but for HumanEval (which is short-context) it's not on the critical path.

---

## What changes immediately

1. **Phase 1 starts now.** Highest priority: experiment 1.1 (Gated DeltaProduct). Single drop-in fla swap, biggest expected information gain.
2. **All Phase 1 evals must include code-PPL and structural-probe accuracy.** PPL alone is misleading per Agent 07's "When Perplexity Lies" finding.
3. **Stop expanding the algebraic-structure search.** PD-SSM / DeltaProduct / PaTH already cover that space.
4. **Add HumanEval to the eval rotation** even at sub-1B scale, as a directional indicator. Even a few percent pass@1 difference is more meaningful than PPL gaps.

---

## Cross-references
- `brainstorm/01_ttt_cells.md` — TTT, Gated DeltaProduct, Titans
- `brainstorm/02_algebraic_structures.md` — content-bearing stack, SL(2,ℝ) Iwasawa
- `brainstorm/03_memory_augmented.md` — kNN cache, Memorizing Transformer
- `brainstorm/04_hierarchical_state.md` — multi-timescale, scope-stack subspace
- `brainstorm/05_mixture_routing.md` — Soft-MoE, MoD, Hymba
- `brainstorm/06_state_management.md` — chunk checkpointing, Video-Ma2mba
- `brainstorm/07_distillation.md` — LoLCATs, HALO, on-policy
- `brainstorm/08_training_innovations.md` — identity-init, aux loss, Amos et al.
- `brainstorm/09_local_attention_hybrids.md` — Samba, Jamba, frontier convergence
- `brainstorm/10_inference_innovations.md` — S\*, MoR, state-forking
