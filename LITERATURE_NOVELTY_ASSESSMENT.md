# Literature Novelty Assessment for Phase 17–22 Findings

**Date:** 2026-04-30
**Scope:** Sober novelty assessment of findings A (sparse cross-layer FiLM), B (K=3 self-feeding training), C (frozen-baseline cosine alignment loss), and D (the integrated combination), in the context of linear-RNN language models trained on Python code at 217M–708M scale. Method: web-search literature review across ~30 targeted queries, plus arxiv abstract / openreview text fetches where available.

This document is intentionally non-defensive: where a result is essentially known, it says so.

---

## Finding A — sparse cross-layer FiLM connection

**Recipe.** Single FiLM modulation `x_t · (1 + α s_t) + α t_t` on layer-2 input, source = lag-1 output of a late layer (layer 28 in 30-layer stack, layer 34 in 36-layer stack), one learnable scalar α (~−0.05 to +0.16), +0.3 % parameters, ~3 % PPL lift across 217 M / 360 M / 708 M.

### Closest 5 prior works

1. **GF-RNN — Gated Feedback Recurrent Neural Networks** (Chung, Gulcehre, Cho, Bengio, ICML 2015). [arXiv 1502.02367](https://arxiv.org/abs/1502.02367).
   - **What's similar.** *The* canonical "top-down feedback in stacked RNNs" paper. Tests on character-level LM and *Python program evaluation* (the project's exact target domain in 2015 vintage). Allows signals from upper recurrent layers to lower layers, gated for each layer pair.
   - **What's different from us.** (i) GF-RNN uses **gated additive** updates on the recurrent state, not multiplicative FiLM modulation on the layer input; (ii) feedback is between **all layer pairs** with a global gating unit per pair, not a single sparse pair; (iii) sequential RNNs (tanh / LSTM / GRU), not parallel-scan-friendly linear-RNN cells; (iv) tested on small character-level / program-eval models, well below modern scale; (v) no negative-α basin observation, no mechanism characterization.
2. **BRIMs — Bidirectional Recurrent Independent Mechanisms** (Mittal, Lamb, Goyal et al., ICML 2020). [arXiv 2006.16981](https://arxiv.org/abs/2006.16981).
   - **What's similar.** The lag-1 trick for parallel-scan friendliness: higher layer at t−1 feeds into lower layer at t, exactly to preserve causality. This is the closest published analogue of our lag-1 protocol.
   - **What's different.** Attention-based modular RIM cells (each layer has a set of mechanisms attending across layers), not single-scalar FiLM. Every adjacent layer pair uses cross-layer attention, not a sparse pair. Small-scale RL-flavored evaluations, not LM at 217M+.
3. **SparX — Sparse Cross-Layer Connection Mechanism for Hierarchical Vision Mamba/Transformer** (Lou, Fu, Yu, AAAI 2025). [arXiv 2409.09649](https://arxiv.org/abs/2409.09649).
   - **What's similar.** Owns the literal phrase "sparse cross-layer connection" and the bi-typed-layers idea (ganglion vs normal layers). Mamba family.
   - **What's different.** Forward DenseNet-style channel-wise feature aggregation in a vision backbone, not late→early feedback in language. Cross-attention with channel-wise routing, not a single FiLM scalar. Different domain (vision classification), different mechanism, different motivation. The project's idea is a *minimum-form*, *single-pair*, *late-to-early* descendant; SparX is a *forward-multi-pair* aggregation cousin.
4. **Feedback Transformer** (Fan et al., 2020). [arXiv 2002.09402](https://arxiv.org/abs/2002.09402).
   - **What's similar.** Exposes upper-layer representations from past timesteps to lower layers' future computation — same high-level idea as GF-RNN but in Transformer blocks.
   - **What's different.** All layers feed into all later layers via a memory aggregation; *not* a sparse single pair. Heavy training cost (5–10×). Different cell class (softmax attention).
5. **TransformerFAM — Feedback Attention is Working Memory** (Hwang et al., 2024). [arXiv 2404.09173](https://arxiv.org/abs/2404.09173).
   - **What's similar.** Feedback attention modulating future computation by past hidden states.
   - **What's different.** Within-block / temporal feedback rather than cross-layer-late-to-early; Transformer-attention-based; primary motivation is long-context memory, not minimal-form architectural lift.

### Honorable mentions
- **Highway Networks** (Srivastava et al., 2015) — gated cross-layer skip, but in *forward* direction within the same timestep.
- **Loop-Residual NN** ([2409.14199](https://arxiv.org/abs/2409.14199)) and **Universal Transformer** ([1807.03819](https://arxiv.org/abs/1807.03819)) — depth-recurrence with parameter sharing rather than a sparse fixed inter-layer connection.
- **PredNet** (Lotter et al., 2017) — predictive coding in vision, the philosophical inspiration for the negative-α basin.

### Novelty assessment

**Verdict:** *incremental extension of GF-RNN (2015) plus BRIMs (2020) lag-1 trick, in a modern linear-RNN context.*

The high-level construct (top-down feedback in stacked RNNs) is decisively not new — GF-RNN owns it. The lag-1 parallel-scan trick is BRIMs. What's empirically new in our work, and survives a careful literature comparison:

- **Minimal form.** A *single sparse layer pair* (2, 28) with a *single scalar α* modulating *FiLM-shaped* output is unpublished. Every prior work uses gated-additive updates (GF-RNN), attention routing (BRIMs, Feedback Transformer), or all-pair / dense feedback (GF-RNN, Feedback Transformer).
- **Mechanism finding (negative-α basin).** The observation that the gradient on α flows through `x · scale` only with multiplicative form, and that non-softmax aggregation is required for the basin to be reachable, is, as far as the search reveals, *not in the GF-RNN / BRIMs / SparX literature*. None of these papers report a sign-of-α observation.
- **Modern linear-RNN context.** Tested on DeltaNet (Yang et al., NeurIPS 2024) and GatedDeltaNet (NVIDIA, ICLR 2025) at scales those papers do not test cross-layer feedback at.

What's *not* new:
- That cross-layer feedback helps stacked recurrent LMs at all (GF-RNN: yes).
- That it can be made parallel-scan friendly via lag-1 (BRIMs: yes).
- That the term "sparse cross-layer" applies (SparX: yes, in vision Mamba).
- That FiLM is a useful conditioning shape in NN (Perez 2018, well-established).

**Confidence:** high. The search was thorough across "top-down feedback," "cross-layer feedback," "feedback memory," "FiLM language model," "GF-RNN extensions," and direct verification of GF-RNN / BRIMs / SparX abstracts.

### Worth publishing? Workshop or short paper, not a flagship venue.

- **As-is, with current evidence:** the architectural lift is small (~3 %) and the prior art is enough that "single sparse FiLM pair beats DeltaNet by 3 %" is not, by itself, a strong NeurIPS / ICML / ICLR contribution. The mechanism characterization (negative-α basin, multiplicative-and-non-softmax conditions) is the most novel piece.
- **Workshop venue (e.g., NeurIPS workshops on efficient sequence modeling, ICLR workshop on linear-attention).** Reasonable as a short paper. The mechanism story is teachable, the depth-scaling pattern (`(2, 28)/30 ≈ (2, 34)/36`) is a clean stylized fact, and the 3-seed reproducibility number σ < 1 % buys credibility.
- **Position-paper / blog framing.** If reframed as "GF-RNN's idea revisited in the linear-RNN era — which minimum form survives at modern scale, and why?" with the negative-α basin as the punchline, this could be a strong blog post that punches above its empirical weight.

---

## Finding B — K=3 iterative fixed-point ("self-feeding") training

**Recipe.** Train with K=3 iterations where iter K's FiLM input is `lag(iter_{K-1}.source.detach())`. Backprop through iter K only. Closes the +7.9 % drift between training-faithful eval and lagged-cached deployment at 708 M.

### Closest 5 prior works

1. **Scheduled Sampling** (Bengio et al., NeurIPS 2015). [arXiv 1506.03099](https://arxiv.org/abs/1506.03099).
   - **What's similar.** Canonical "close the train/inference gap from teacher forcing in autoregressive RNN" technique. Stochastically replaces ground-truth previous tokens with model-generated tokens during training, with a curriculum.
   - **What's different.** *Targets* differ: scheduled sampling addresses the *autoregressive output token gap* (model's own token vs ground truth). K=3 self-feeding addresses an *internal cross-layer FiLM-input gap* (model's own pass-2 source-layer output vs the pass-1 source-layer output). The mechanism is the same pattern (use the model's own output as the train-time signal so the train distribution matches the deploy distribution); the locus is different.
2. **Professor Forcing** (Goyal, Lamb et al., NeurIPS 2016). [arXiv 1610.09038](https://arxiv.org/abs/1610.09038).
   - **What's similar.** Same problem framing as scheduled sampling, addressed via GAN: train the network so its dynamics under teacher forcing match its dynamics under free-running rollout.
   - **What's different.** GAN-based, output-token-level, sequence-RNN context. Mechanism (adversarial discriminator) is unrelated to ours. Same core insight: train-time signal should match inference-time signal.
3. **Self Forcing — Bridging the Train-Test Gap in Autoregressive Video Diffusion** (Huang et al., NeurIPS 2025 spotlight). [arXiv 2506.08009](https://arxiv.org/abs/2506.08009).
   - **What's similar.** Direct contemporary descendant: trains an autoregressive video diffusion model by self-rollout with KV cache so the train conditioning matches inference conditioning. Frames the problem as *bridging the train/test gap* exactly as we do.
   - **What's different.** Video diffusion / KV cache, not linear-RNN cross-layer FiLM. Uses few-step diffusion with stochastic gradient truncation. The pattern of "use the model's own output during training to match what inference will see" is the same — but the actual locus (FiLM cross-layer signal vs frame-conditioning signal) is different.
4. **R-Drop — Regularized Dropout for Neural Networks** (Liang et al., NeurIPS 2021). [arXiv 2106.14448](https://arxiv.org/abs/2106.14448).
   - **What's similar.** Two forward passes per input during training; consistency regularization (KL) between the two outputs to address the train/inference gap caused by dropout. Multiple forward passes for the same input as a training technique.
   - **What's different.** Source of the gap is dropout, not architectural cross-layer feedback. KL-on-output-distributions, not a hidden-state FiLM-input alignment. Same architectural family pattern (multi-pass training) but different motivation.
5. **Deep Equilibrium Models — DEQ** (Bai et al., NeurIPS 2019). [arXiv 1909.01377](https://arxiv.org/abs/1909.01377).
   - **What's similar.** "Iterate to a fixed point at training time" is *the* DEQ slogan. Trains models that compute via a fixed-point iteration, with implicit differentiation through the equilibrium.
   - **What's different.** DEQ uses root-finding / implicit differentiation at the equilibrium. We use explicit K=3 unrolling with backprop through the last iteration only and detach for earlier iterations — closer in spirit to Object Representations as Fixed Points (Chang et al., NeurIPS 2022, [arXiv 2207.00787](https://arxiv.org/abs/2207.00787)) but without implicit differentiation. DEQ's motivation is constant-memory "infinite depth"; ours is closing a deliberate train/deploy asymmetry.

### Honorable mentions
- **Universal Transformer** ([1807.03819](https://arxiv.org/abs/1807.03819)) — depth-recurrence with parameter sharing (different motivation).
- **Looped/recurrent-depth Transformers** ([2604.07822](https://arxiv.org/abs/2604.07822), LoopFormer 2602.11451) — same family, different application.
- **Object Representations as Fixed Points** (Chang et al., NeurIPS 2022, [arXiv 2207.00787](https://arxiv.org/abs/2207.00787)) — implicit-differentiation training of slot attention as fixed-point. Same mechanism family.
- **Self-Refine** ([arXiv 2303.17651](https://arxiv.org/abs/2303.17651)) — *inference-time* iterative refinement, not training-time train/test gap closure.

### Novelty assessment

**Verdict:** *incremental, well-positioned application of the "use-own-outputs-at-train-time" pattern (Scheduled Sampling 2015 → Professor Forcing 2016 → Self Forcing 2025) to a new locus (cross-layer FiLM input).*

The general pattern of "train with K passes where pass K conditions on output of pass K−1, backprop only through the final pass" is, taxonomically, exactly the same family as Scheduled Sampling and Self Forcing. Where K=3 self-feeding diverges:

- **Locus.** None of the prior works address the FiLM-input gap inside a cross-layer feedback architecture. They all address output-token gaps.
- **Detach pattern.** Backprop through iter K only, with detached earlier iters. Standard but unnamed in this exact form for this problem.
- **Empirical contribution.** Closing the +7.9 % drift to ~0 % at 708 M is the clean empirical result.

What's *not* novel:
- Multi-pass training. (R-Drop, Self Forcing, Universal Transformer.)
- Closing train/inference gaps via self-rollout. (Self Forcing.)
- Detaching to control gradient flow. (Standard.)

**Confidence:** medium-high. Self Forcing (2025 NeurIPS) is the closest contemporary; we should add it as a "must cite" — the abstract framing "Bridging the Train-Test Gap" mirrors ours.

### Worth publishing?

- **Standalone:** weak. K=3 self-feeding by itself is one slide of a larger paper; not a 9-page contribution.
- **As part of an integrated paper (Finding D below):** yes, as an enabling technique for the architectural finding to give a 1× decode-cost deployment. Without K=3, the architectural lift comes with a 2× decode cost — the K=3 fix is what makes A *deployable*.
- **As a short methods note ("Closing the train/deploy gap in cross-layer-feedback linear RNNs via K-pass self-feeding"):** plausible at a workshop or as an arxiv preprint of independent record.

---

## Finding C — frozen-vanilla-baseline cosine-distance auxiliary loss

**Recipe.** L_total = L_CE + β · ∑_t L_sem(s_t), where L_sem(s_t) = 1 − cos(W·pool(student_h_t), pool(frozen_baseline_h_t)). Frozen baseline is a same-architecture, same-size, same-corpus, same-compute, separately-trained plain DN. β=1.0, uniform weighting (surprise-weighting was *counterproductive*). Per-statement pooling using AST segmentation. Closes ~−8.7 % at 708 M on top of K=3 self-feeding's −1.5 % for cumulative −10.0 % over plain DN.

### Closest 5 prior works

1. **Born-Again Networks (BAN)** (Furlanello, Lipton, Tschannen, Itti, Anandkumar, ICML 2018). [arXiv 1805.04770](https://arxiv.org/abs/1805.04770).
   - **What's similar.** *Decisively close.* Train a student that is parameterized identically to a teacher, via knowledge distillation, and the student significantly outperforms the teacher. Tested on language modeling: an LSTM model on PTB drops from 71.87 → 68.56 perplexity (4.6 %) when retrained as a BAN-LSTM. Same-size student exceeds teacher in LM.
   - **What's different from us.** BAN uses **KL on output logits** (Hinton-style soft targets, possibly with CWTM/DKPP variants), *not* cosine distance on pooled hidden states. The student–teacher loss is on the prediction distribution, not on intermediate representations. Also: BAN tested LSTM-LM at small scale (PTB, ~25M params), not 217M+ DeltaNet on code; BAN+L (with the label loss combined) was needed for LSTM-LM to work, indicating fragility. The student-exceeds-teacher *effect* is the same; the *loss form* differs.
2. **DistilBERT** (Sanh, Debut, Chaumond, Wolf, EMC^2 2019). [arXiv 1910.01108](https://arxiv.org/abs/1910.01108).
   - **What's similar.** Triple loss: language modeling + KL distillation + **cosine-embedding loss between student and teacher hidden states**. The cosine-on-hidden-states piece is structurally identical to L_sem.
   - **What's different.** Teacher is *larger* (BERT-base, ~110M) than student (~66M); compression-motivated. Teacher and student have different architectures (different number of layers). Hidden states are *per-token*, not pooled per AST statement. No "frozen same-size baseline" framing — it's standard teacher-student compression.
3. **DistilGPT2** (Hugging Face, 2019). [model card](https://huggingface.co/distilbert/distilgpt2).
   - **What's similar.** Applies DistilBERT's recipe (LM loss + KL + cosine embedding) to GPT-2 — same triple loss structure on a generative LM. Cosine embedding loss aligns student–teacher hidden state directions per token.
   - **What's different.** Teacher GPT-2 124M → student 82M (smaller), motivated by compression. Student does *not* exceed teacher (PPL 21.1 vs 16.3). Per-token alignment, no AST-pooled statement granularity.
4. **MOHAWK — Transformers to SSMs** (Bick, Khan, Pohan-Hardy, Roberts, Bisk, NeurIPS 2024). [arXiv 2408.10189](https://arxiv.org/abs/2408.10189).
   - **What's similar.** Multi-stage cross-architecture distillation; one stage is **hidden-state alignment** between Transformer teacher and Mamba student. Same family of ideas applied in linear-RNN / SSM context — closest contemporary in our exact architectural niche.
   - **What's different.** Cross-architecture (Transformer → Mamba), teacher is bigger and pre-trained at scale, motivated by transferring quadratic knowledge to subquadratic students. Per-token hidden-state alignment, not pooled-per-AST-statement. Three-stage pipeline (matrix orientation → hidden states → end-to-end) — much more elaborate than our single auxiliary cosine-pool loss.
5. **Improving Language Model Distillation through Hidden State Matching** (ICLR 2025). [openreview](https://openreview.net/forum?id=IcVSKhVpKu).
   - **What's similar.** Recent work explicitly on LM distillation via hidden state matching. Argues cosine loss restricts dimensionality and proposes CKA as an alternative.
   - **What's different.** Compression motivation — student smaller than teacher. Per-token, not pooled-per-AST. CKA (subspace alignment), not cosine on pooled vectors. Tested on Transformer-LM (BERT-style), not linear-RNN.

### Honorable mentions
- **FitNets — Hints for Thin Deep Nets** (Romero et al., 2014, [arXiv 1412.6550](https://arxiv.org/abs/1412.6550)) — the original "hint distillation" with regression on intermediate features. Teacher is bigger, regression mapping for dim mismatch.
- **MiniLM** (Wang et al., NeurIPS 2020, [arXiv 2002.10957](https://arxiv.org/abs/2002.10957)) — distill self-attention relations; teacher bigger.
- **BAM!** (Clark, Luong, Khandelwal, Manning, Le, ACL 2019, [arXiv 1907.04829](https://arxiv.org/abs/1907.04829)) — Born-Again Multi-Task on top of BERT, with teacher annealing. Same-size student exceeds single-task teacher in NLU.
- **Self-Distillation Amplifies Regularization in Hilbert Space** (Mobahi, Farajtabar, Bartlett, NeurIPS 2020, [arXiv 2002.05715](https://arxiv.org/abs/2002.05715)) — theoretical: self-distillation acts as a basis-restricting regularizer. A few rounds reduce overfitting; further rounds underfit.
- **Does Knowledge Distillation Really Work?** (Stanton et al., NeurIPS 2021, [arXiv 2106.05945](https://arxiv.org/abs/2106.05945)) — empirical study showing student often *cannot* match teacher even when capacity is sufficient; same-size cases included. This paper's null-result framing makes our +10 % effect newsworthy if it survives further scrutiny.
- **BYOL** (Grill et al., NeurIPS 2020) and **SimSiam** (Chen & He, CVPR 2021, [arXiv 2011.10566](https://arxiv.org/abs/2011.10566)) — self-supervised representation learning with stop-gradient targets; minimize negative cosine similarity to a frozen / EMA target. Mechanism family is the same as L_sem (frozen target + cosine), but applied to vision SSL, not LM training.
- **Deep Mutual Learning** (Zhang, Xiang, Hospedales, Lu, CVPR 2018, [arXiv 1706.00384](https://arxiv.org/abs/1706.00384)) — same-architecture peers train each other; no frozen teacher. Closest "co-distillation" cousin.
- **Born-Again Multi-Task Networks** ([BAM, ACL 2019](https://aclanthology.org/P19-1595/)) — Same-size student exceeds same-architecture teacher in BERT NLU multi-task, with teacher annealing. The "same-size student exceeds same-architecture teacher" effect is *demonstrated* in modern NLP.

### Novelty assessment

**Verdict (multi-part):**

**Q1: Is the specific recipe (same-size frozen baseline + cosine pooled-hidden-state loss + LM training) published anywhere?**

**Mostly yes, mechanism-wise; specifically no, locus-wise.** The decomposition:
- "Same-size frozen teacher exceeds vanilla via KD" → BAN (2018) + BAM (2019). Established.
- "Cosine on hidden states as auxiliary distillation loss" → DistilBERT / DistilGPT2 (2019). Established.
- "Frozen-target cosine-loss representation alignment" → BYOL/SimSiam (2020/2021). Established.
- "Hidden-state alignment in Mamba/SSM context" → MOHAWK (2024). Established.

What's specifically *not* in the literature, as best the search can find:
- *Pooled per AST statement* (vs per-token). This is a code-LM-specific granularity choice that may meaningfully differ from token-level cosine because Python statements are coherent semantic units. No prior work I found combines this granularity with cosine alignment.
- *Same-architecture, same-size, same-corpus, same-compute frozen baseline* as the alignment target, in a *linear-RNN* LM context — the BAN-style claim that the student exceeds the teacher, but executed via cosine on pooled hidden states rather than KL on logits.
- *Counterproductive surprise weighting result* — the empirical observation that an oracle-derived per-statement weight makes results worse, while uniform weighting works, is a clean negative result that is itself publishable as a "do not bother" finding for the structural-surprise-as-loss-weight hypothesis.

So: the *mechanism* (frozen-target representation alignment for LM training) is essentially known; the *specific combination* (linear RNN + AST-pooled + same-size-vanilla-baseline target) is novel as a specific recipe. The +10 % cumulative lift at 708 M is the empirical claim.

**Q2: Is there a name we should adopt vs invent?**

**Adopt "self-distillation" or "Born-Again-style alignment."** "Frozen-baseline cosine alignment" is a description; the established taxonomic term for "same-size student exceeds same-architecture frozen teacher" is *self-distillation* (Furlanello 2018, Mobahi 2020, Hugging Face's distillation tutorial section). The specific cosine-on-hidden-states variant is *hint distillation* (FitNets) / *intermediate representation distillation*. Combining: this is "same-size self-distillation via intermediate-representation alignment" — a mouthful but established. The proper precedent term is **Born-Again** (since the student exceeds the teacher); the proper alignment term is **hint distillation** with the modern variant **hidden-state matching** (DistilBERT). I'd recommend framing as **"Born-Again hint distillation for linear RNN LMs, with AST-pooled statement-level cosine alignment."**

**Q3: Is the BAN-style "student exceeds teacher" effect known in the LM literature?**

**Yes, but only fragilely.** BAN-LSTM on PTB shows it (71.87 → 68.56). BAM on BERT NLU shows it. But:
- The effect on PTB-LSTM was 4.6 % PPL improvement, similar magnitude to what we see on PPL pre-K=3-self-feed.
- BAN required the BAN+L variant (label loss preserved) to work for LSTM-LM, indicating fragility.
- Stanton et al. 2021 documents that even when student has capacity to match teacher, optimization difficulties prevent it; the BAN result is somewhat surprising in light of Stanton.
- Modern LLM-scale "same-size self-distillation" is *not* well-established; it's mostly compression-motivated (smaller student) at scale.

**Confidence:** high. The mechanism and goal map cleanly to BAN+DistilBERT+MOHAWK; the specific combination is unpublished. Cosine-on-pooled-hidden-states with same-size frozen baseline as a *primary positive lift* (not just a stabilizer) is, narrowly, novel.

### Worth publishing?

- **As a methods-level contribution at workshop:** yes, even on its own. The +10 % cumulative number at 708 M is large for a same-size auxiliary loss; the AST-pooled granularity is a novel detail; the surprise-weighting null result is a clean negative. The "linear RNN" qualifier matters because BAN/DistilBERT focused on LSTM/Transformer.
- **As a full conference paper:** marginal alone, but possibly with sufficient ablations. Need to demonstrate (a) the same-size frozen-baseline aspect is essential (vs random init partner, vs EMA partner like BYOL); (b) the AST-pooled granularity is essential (vs per-token, vs document-level); (c) the result holds on natural text and at multiple seeds.
- **Strongest framing:** *"Born-Again-style self-distillation for linear-RNN language models: a same-architecture frozen baseline, AST-pooled cosine alignment, gives the student a +10 % lift over the teacher at 708M."* Frame in lineage: BAN (2018) → DistilBERT/DistilGPT2 (2019) → MOHAWK (2024) → ours (2026).

---

## Finding D — the COMBINATION

Sparse cross-layer FiLM (architectural) + K=3 self-feeding (training procedure) + frozen-baseline cosine alignment (auxiliary loss) → **−10.0 % over plain DN at 708 M, deployment-honest, 1× decode cost**, in a modern linear-RNN family on Python code.

### Closest published packages

I found *no* paper that combines all three of (cross-layer FiLM-style feedback × multi-pass self-feeding training × Born-Again hidden-state alignment) in any architecture. The combination is, by my best search, unpublished.

The closest "integrated training-procedure recipe" papers in linear RNN / Mamba space are:
- **MOHAWK** (NeurIPS 2024) — three-stage Transformer→Mamba distillation. Different ingredients (matrix orientation + hidden state + end-to-end), different problem (cross-architecture).
- **Gated Delta Networks** (Yang et al., ICLR 2025, [arXiv 2412.06464](https://arxiv.org/abs/2412.06464)) — combines gating with delta rule. Architecture-only, no training-procedure recipe like ours.
- **Mamba-3** (ICLR 2026, [openreview](https://openreview.net/pdf?id=HwCvaJOiCj)) — architectural improvements to Mamba2. No alignment loss, no cross-layer feedback, no multi-pass training.

### Novelty assessment

**Verdict:** *the combination is novel as a specific recipe; each individual ingredient is incremental over prior art.*

The full integration — architectural sparse-FiLM + training K=3 self-feeding + frozen-baseline cosine alignment — is, as a recipe, unpublished. The −10 % deployment-honest lift at 1× decode cost is, as far as the search reveals, the largest reported lift over a strong DeltaNet baseline at 708 M Muon-trained on code. (Caveat: the literature's strongest comparable point is GatedDeltaNet's claimed lift over DeltaNet, which is at a different scale and on different benchmarks.)

That said:
- **Each ingredient is incremental.** A reviewer can map A→GF-RNN+BRIMs+SparX, B→Self-Forcing/Scheduled-Sampling, C→BAN+DistilBERT+MOHAWK. The combination story is the genuinely new part.
- **The integration is non-trivial.** The K=3 fix exists *because* the architectural FiLM choice introduced a deployment asymmetry. The frozen-baseline alignment exists *because* the architecture+K=3 alone underperformed the cumulative target. Each ingredient solves a problem the prior ingredient creates or fails to solve. That's the kind of recipe-shaped contribution that makes a tidy paper.

### Worth publishing as an integrated paper?

**Yes — an integrated linear-RNN training-procedure paper is the strongest framing for these findings.**

The strongest single-paper title is something like:

> **"Born-Again Sparse FiLM: a recipe for −10% deployment-honest improvement in linear-RNN language models at 1× decode cost"**

or, more in the lineage:

> **"Closing the train/deploy gap in cross-layer-feedback linear RNNs: a Born-Again recipe for DeltaNet"**

The paper's contribution: not architectural (A is incremental over GF-RNN), not procedural-alone (B is incremental over Self Forcing), not aux-loss-alone (C is incremental over BAN/DistilBERT/MOHAWK), but **the deployable combination that extracts a −10 % deployment-honest lift in modern linear RNNs on code, where each ingredient is well-grounded in prior art and each is necessary for the others to work as a deployable system**.

**Venue assessment:**
- **NeurIPS / ICLR main track:** plausible but not a sure thing. Reviewers will (correctly) note that each ingredient is known; the contribution rests on the integration claim and the empirical magnitude. The −10 % number at 708 M would be the headline; the workshop-paper-worthy backup pieces (mechanism characterization, surprise-weighting null, K=3 drift closure) all support it.
- **EMNLP/NAACL/ACL:** code-domain LM is a natural fit; "linear-RNN training procedure" sells better in the NLP track than at NeurIPS where architecture papers dominate.
- **Workshop strong + arxiv preprint:** safest. NeurIPS workshops on efficient sequence modeling, linear attention, or modern architectures would clearly accept this story; preprint can be cited and cleanly extended.
- **Position-paper style ("a recipe" framing):** possible, but the empirical numbers are strong enough to do an empirical paper, not just a position one.

---

## Overall recommendation

**Yes, the project's findings in aggregate merit a paper, but the framing matters.**

### Strongest single-paper framing (ranked)

1. **Integrated recipe paper (Finding D):** *"A Born-Again recipe for sparse-feedback DeltaNet: integrating cross-layer FiLM, K-pass self-feeding, and frozen-baseline alignment for a 10 % deployment-honest improvement at 1× decode cost."* This is the strongest framing because it converts each individually-incremental ingredient into a coherent, integrated, end-to-end recipe with a strong headline number and a clean mechanism story for each piece.

2. **Aux-loss-headlined ("vanilla-baseline alignment for linear RNNs"):** the alignment loss contributes the largest single component (~−8.4 to −8.7 % at both scales). If the integrated framing is rejected, fall back to a Born-Again-style "self-distillation for linear-RNN LMs" paper that uses the rest as supporting context. Lineage: BAN 2018 → DistilBERT 2019 → MOHAWK 2024 → ours 2026.

3. **Procedural ("K=3 self-feeding to close train/inference gap"):** weakest framing alone, because the train/inference gap (created by the FiLM architecture) only matters if the architectural lift survives — and at 708 M, the architectural lift alone (1.5 % at 1× decode, after K=3) is small.

4. **Architectural ("sparse-FiLM"):** small lift (3 %) and substantial prior art (GF-RNN, BRIMs, SparX). Workshop / blog post only.

I'd put the integrated framing forward as the headline.

### Top 3 missing experiments to strengthen the case

1. **Natural-text validation, not just code.** Currently the experiments are all on codeparrot Python. The 217 M TinyStories check (only −1.6 % vs DN) is a yellow flag that the architectural lift may be code-specific. Need:
   - L_sem alignment loss on a natural-text corpus (e.g., FineWeb, OpenWebText, or C4) at 217 M and 360 M.
   - Verify that "same-architecture, same-size, same-corpus frozen baseline as alignment target" gives the +8.4 % lift on natural text.
   - The TinyStories result should be re-run with K=3 + L_sem on the same architecture; if natural-text lift drops to <3 % on natural language, the headline "linear-RNN training procedure" needs a code-domain qualifier.

2. **Multi-seed stability of the L_sem +8.4 % lift.** The 217 M sparse-FiLM finding has a 3-seed σ < 1 %; the L_sem lift is single-seed at both scales. Given Stanton et al. 2021's null-result history with KD, demonstrating 3-seed stability (especially of the +8.7 % at 708 M) is critical to defend against a "you got lucky" reviewer. Recommended: 3 seeds at 217 M and 2 seeds at 708 M.

3. **Ablation against simpler baselines for L_sem:**
   - **EMA-target (BYOL-style)** vs frozen separately-trained baseline. If EMA gives the same lift, the "same-architecture frozen baseline" framing collapses to a self-supervised representation learning trick.
   - **Random-init partner** (no pre-training) as the alignment target. If this works, the "*frozen baseline knows things*" intuition is wrong; the regularizer's source of information is something else.
   - **Different-architecture target** (Mamba2 baseline as alignment for DeltaNet student). Tests whether the same-architecture aspect is essential.
   - **Per-token cosine** vs AST-pooled. The AST-pooled choice is the most novel detail; if per-token gives the same lift, the AST piece can be dropped.

These three would convert the current evidence from "interesting at 708M on code" to "robust mid-scale empirical contribution," which is the threshold for a credible main-track submission.

### Honorable secondary experiments
- **Comparison vs SoTA Mamba2 trained with the same training procedure** (currently compared only to DN/Mamba2/Transformer baselines without L_sem). This tests whether the L_sem lift is DN-specific or a general linear-RNN trick.
- **Distillation from a stronger same-arch teacher** (e.g., a DN trained at 4× the steps), to test whether L_sem's lift is bounded by the teacher's quality (Mobahi 2020 predicts diminishing returns with iterations).
- **Larger scale at fixed compute.** A 1.4 B-param run at 30 M tokens to test whether the +10 % survives a 2× scale-up beyond what's currently in the project.

---

## Per-finding TL;DR

| Finding | Closest prior | Novelty call | Worth publishing? |
|---|---|---|---|
| A — sparse cross-layer FiLM | GF-RNN 2015 + BRIMs 2020 + SparX 2025 | *incremental extension; minimal-form + mechanism is new* | Workshop / short paper alone; component of integrated paper |
| B — K=3 self-feeding | Self Forcing 2025 + Scheduled Sampling 2015 + R-Drop 2021 | *incremental application of "use own outputs at train" pattern to a new locus (FiLM input)* | Methods note; component of integrated paper |
| C — frozen-baseline cosine alignment | BAN 2018 + DistilBERT 2019 + MOHAWK 2024 | *recipe-level novel; mechanism is essentially known; AST-pooled granularity + same-size-vanilla-baseline + linear-RNN context is the new combination* | Stand-alone workshop or main-track with proper ablations |
| D — the combination | None I could find combining all three | *novel as integrated recipe; each ingredient incremental* | **YES — main track, with proper natural-text + multi-seed validation** |

---

## Sources

### Finding A (cross-layer feedback)
- [Chung, Gulcehre, Cho, Bengio. "Gated Feedback Recurrent Neural Networks". ICML 2015.](https://arxiv.org/abs/1502.02367)
- [Mittal, Lamb, Goyal et al. "Learning to Combine Top-Down and Bottom-Up Signals (BRIMs)". ICML 2020.](https://arxiv.org/abs/2006.16981)
- [Lou, Fu, Yu. "SparX: A Sparse Cross-Layer Connection Mechanism for Hierarchical Vision Mamba and Transformer Networks". AAAI 2025.](https://arxiv.org/abs/2409.09649)
- [Fan et al. "Addressing Some Limitations of Transformers with Feedback Memory". 2020.](https://arxiv.org/abs/2002.09402)
- [Hwang et al. "TransformerFAM: Feedback attention is working memory". 2024.](https://arxiv.org/abs/2404.09173)
- [Perez, Strub, de Vries, Dumoulin, Courville. "FiLM: Visual Reasoning with a General Conditioning Layer". AAAI 2018.](https://arxiv.org/abs/1709.07871)
- [Yang et al. "Parallelizing Linear Transformers with the Delta Rule (DeltaNet)". NeurIPS 2024.](https://arxiv.org/abs/2406.06484)
- [Yang et al. "Gated Delta Networks: Improving Mamba2 with Delta Rule". ICLR 2025.](https://arxiv.org/abs/2412.06464)

### Finding B (multi-pass training / train-test gap)
- [Bengio, Vinyals, Jaitly, Shazeer. "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks". NeurIPS 2015.](https://arxiv.org/abs/1506.03099)
- [Goyal, Lamb et al. "Professor Forcing: A New Algorithm for Training Recurrent Networks". NeurIPS 2016.](https://arxiv.org/abs/1610.09038)
- [Huang et al. "Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion". NeurIPS 2025 spotlight.](https://arxiv.org/abs/2506.08009)
- [Liang et al. "R-Drop: Regularized Dropout for Neural Networks". NeurIPS 2021.](https://arxiv.org/abs/2106.14448)
- [Bai, Kolter, Koltun. "Deep Equilibrium Models". NeurIPS 2019.](https://arxiv.org/abs/1909.01377)
- [Dehghani et al. "Universal Transformers". ICLR 2019.](https://arxiv.org/abs/1807.03819)
- [Chang, Griffiths et al. "Object Representations as Fixed Points". NeurIPS 2022.](https://arxiv.org/abs/2207.00787)
- [Madaan et al. "Self-Refine: Iterative Refinement with Self-Feedback". 2023.](https://arxiv.org/abs/2303.17651)

### Finding C (frozen-baseline alignment / Born-Again)
- [Furlanello, Lipton, Tschannen, Itti, Anandkumar. "Born-Again Neural Networks". ICML 2018.](https://arxiv.org/abs/1805.04770)
- [Sanh, Debut, Chaumond, Wolf. "DistilBERT". 2019.](https://arxiv.org/abs/1910.01108)
- [Romero et al. "FitNets: Hints for Thin Deep Nets". 2014.](https://arxiv.org/abs/1412.6550)
- [Wang et al. "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression". NeurIPS 2020.](https://arxiv.org/abs/2002.10957)
- [Bick et al. "Transformers to SSMs (MOHAWK)". NeurIPS 2024.](https://arxiv.org/abs/2408.10189)
- [Mobahi, Farajtabar, Bartlett. "Self-Distillation Amplifies Regularization in Hilbert Space". NeurIPS 2020.](https://arxiv.org/abs/2002.05715)
- [Stanton, Izmailov, Kirichenko, Alemi, Wilson. "Does Knowledge Distillation Really Work?". NeurIPS 2021.](https://arxiv.org/abs/2106.05945)
- [Clark, Luong, Khandelwal, Manning, Le. "BAM! Born-Again Multi-Task Networks for Natural Language Understanding". ACL 2019.](https://aclanthology.org/P19-1595/)
- [Zhang, Xiang, Hospedales, Lu. "Deep Mutual Learning". CVPR 2018.](https://arxiv.org/abs/1706.00384)
- [Grill et al. "Bootstrap Your Own Latent (BYOL)". NeurIPS 2020.](https://arxiv.org/abs/2006.07733)
- [Chen, He. "Exploring Simple Siamese Representation Learning (SimSiam)". CVPR 2021.](https://arxiv.org/abs/2011.10566)
- ["Improving Language Model Distillation through Hidden State Matching". ICLR 2025.](https://openreview.net/forum?id=IcVSKhVpKu)
- [DistilGPT2 model card.](https://huggingface.co/distilbert/distilgpt2)

### Finding D (integrated)
- No published combination found.

### Tertiary (related)
- [Srivastava, Greff, Schmidhuber. "Highway Networks". 2015.](https://arxiv.org/abs/1505.00387)
- [Lotter, Kreiman, Cox. "PredNet". 2017.](https://arxiv.org/abs/1605.08104)
- [Loop-Residual Neural Networks. 2024.](https://arxiv.org/abs/2409.14199)
- [Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network". 2015.](https://arxiv.org/abs/1503.02531)
- [Mamba-3. ICLR 2026.](https://openreview.net/pdf?id=HwCvaJOiCj)
