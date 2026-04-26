# 08 — Curriculum / Pretraining Objectives / Structured Init

**Premise.** The hybrid SO(n)+DeltaNet wins parity by +32.6 pp end-tok at T=64 (kill-gate green) yet loses 1.07–1.22× in PPL on TinyStories+Python at 135M / 5k steps. The cell **has** the capacity; it isn't being **used**. That is the canonical signature of an inductive-bias gap that pretraining objective + init can fix — exactly the regime where the literature is loudest in 2024–2026.

The single strongest empirical anchor: **Amos et al. (ICLR 2024)** show that with a denoising pretrain objective, **Transformers and SSMs become near-indistinguishable on Long Range Arena, even improving PathX-256 by +20 abs pts**. Random init grossly overestimates architectural differences. That is our story.

## Top 5 concrete ideas

### Idea 1: Two-stage synthetic-then-LM curriculum (parity → mod-p → S5 → induction-copy → LM)
- **What it is:** Pretrain ~5–10% of total steps on a mixture of state-tracking synthetics that the ortho cell is uniquely good at (parity, mod-p, S5/A5, bounded Dyck, copy-with-distractor, induction-head task), THEN switch to LM on TinyStories+Python with a short (200-step) warmup interpolation. The synthetic phase is targeted exactly at what the ortho subspace can do but DeltaNet alone can't.
- **Why it might unlock the architecture:** Mod-p / parity wins prove the ortho rotational subspace can encode group state. The bottleneck is that LM gradient never excites that subspace because surface-statistics gradients dominate first and lock the rotation block near identity (or near random noise that's hard to escape). Synthetic phase forces the ortho block into the "rotational basin" before LM training, then LM training only has to refine, not discover, the circuit. This is the Amos et al. story applied to a hybrid cell.
- **Key papers:** Amos et al. ICLR 2024 ("Never Train From Scratch" / pretraining closes SSM-Transformer LRA gap); Grazzi et al. ICLR 2025 ("Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues" — confirms LRNN state-tracking is a learning-dynamics problem fixable by structural+training tweaks); Jobanputra et al. arXiv 2505.21785 ("Born a Transformer — Always a Transformer?" — pretraining selectively enhances some capabilities, not others; targeted pretrain matters).
- **Implementation cost:** **M.** Need a synthetic-task generator (~200 LOC), a mixture sampler with per-task length/difficulty curriculum, and a phase-switch in the training loop. We already have parity/mod-p generators from the kill-gate.
- **Code-LLM relevance:** **High.** Code is ~80% bracket-tracking, scope-tracking, and induction (def-then-use, import-then-call). All four are state-tracking or copy primitives.
- **Risk:** Catastrophic forgetting in stage 2 — synthetic-stage skills get overwritten. Mitigation: replay 5% synthetic in stage 2 (multi-task pretraining is well-studied; cheap insurance).

### Idea 2: Auxiliary "structural probe" loss — predict bracket depth / scope depth from each layer's state
- **What it is:** At every (or every-Nth) layer, attach a small linear probe (1–2 layer MLP, freeze nothing, very low loss weight ~0.01–0.05) that predicts (a) current bracket depth, (b) current indent level, (c) "are we inside a string/comment", (d) for Python: nesting type (paren/bracket/brace). Targets are computed deterministically from input tokens — free supervision.
- **Why it might unlock the architecture:** Forces the ortho block to **actually use** its rotation capacity to maintain a counter-like internal state. The rotation block's natural job IS counting/tracking; this aux loss says "you must keep that information linearly readable." It directly addresses the "capacity not used" diagnosis. PPL gradient alone is too distal.
- **Key papers:** AST-Probe (Sarrof et al., ASE 2022) — ASTs are linearly recoverable from LM hidden states; AST-T5 (Gong et al., ICML 2024) — explicit AST-aware pretrain helps code; StructCoder (TOSEM 2023) — auxiliary AST-Path/Data-Flow heads improve code generation.
- **Implementation cost:** **S.** Tokenizer-level depth annotation is ~50 LOC for Python. Probe heads are trivial. Add to existing training loop.
- **Code-LLM relevance:** **Very high.** Bracket/scope structure is exactly what we want code models to track, and exactly what kills SSMs without negative eigenvalues / state-tracking primitives.
- **Risk:** Aux loss dominates and degrades main LM perplexity (the DeepSeek-V3 warning re: MTP). Mitigation: weight ≤0.05, decay weight to 0 over second half of training, monitor LM loss.

### Idea 3: Identity-bootstrapped ortho init + log-spaced rotation-magnitude warmup
- **What it is:** Initialize the SO(n) block as **exact identity** (zero rotation), and warm up the **maximum allowed rotation magnitude** log-linearly over the first 1k–2k steps from 0 → π. Equivalent: parameterize the rotation generator A as `α(t) · A` with α(t) starting at 0. Cell starts as a pure DeltaNet, smoothly differentiates into the ortho hybrid as training proceeds.
- **Why it might unlock the architecture:** Current init likely puts ortho block at random rotation, which acts as additive noise to the DeltaNet path. Optimizer's first move is to suppress it — and from a suppressed state it's hard to find rotational structure that helps. Identity init guarantees stage 1 is "as good as DeltaNet"; rotation-magnitude warmup lets the optimizer add rotation only when (and where) it actually reduces loss. This is the rotation-block analog of the well-established residual / IRNN identity-init trick (Le et al. 2015; IDInit, Pan et al. ICLR 2025).
- **Key papers:** Le et al. arXiv 1504.00941 ("A Simple Way to Initialize RNNs of ReLUs" — identity init); Pan et al. ICLR 2025 (IDInit); Nowak et al. ICML 2024 ("Exact Orthogonal Initialization" via Givens rotations).
- **Implementation cost:** **S.** ~30 LOC change to init + a scalar schedule.
- **Code-LLM relevance:** **Medium-high** — generic, but specifically addresses our PPL regression.
- **Risk:** If the ortho block "never wakes up" (α never grows because gradient through near-identity rotation is small), we just trained DeltaNet. Mitigation: hard-floor α(t) ≥ schedule, not a learned scalar.

### Idea 4: Multi-task pretraining with state-tracking auxiliary corpus (10% mix throughout)
- **What it is:** Instead of phased curriculum (Idea 1), continuously interleave 10% synthetic state-tracking sequences with 90% LM tokens, every batch, for the entire training run. Synthetic tasks: parity, mod-p arithmetic, balanced parens, S5 word reduction, induction-head retrieval. Treat each as just another sequence.
- **Why it might unlock the architecture:** Avoids the forgetting risk of two-stage. The constant gradient pressure on the rotation subspace keeps it in use. RegMix (Liu et al., ICLR 2025 spotlight) shows data-mixture choices are first-order important and transcend scaling laws — adding ~10% of the right synthetic tasks may shift the loss landscape enough to favor using the ortho block.
- **Key papers:** Liu et al. ICLR 2025 (RegMix); Aksenov et al. arXiv 2603.10055 (NCA pre-pretraining — non-linguistic synthetic data improves LM by up to 6%, accelerates convergence 1.6×).
- **Implementation cost:** **M.** Mixed-batch dataloader + task generators.
- **Code-LLM relevance:** **High.**
- **Risk:** Token budget is finite at 5k steps; 10% mix may starve the LM signal. Less elegant than Idea 1 if forgetting turns out to be small.

### Idea 5: Distill from a frozen Transformer teacher into the hybrid (LoLCATs/RADLADS-style, attention-aligned)
- **What it is:** Take a small pretrained Transformer (Pythia-160M, GPT-Neo-125M, or our own ~135M Transformer trained for the same step budget), freeze it, and distill into our hybrid via two-stage protocol: (i) attention-alignment / hidden-state MSE per layer for 30% of steps, (ii) full logit + LM loss for the rest. The hybrid's ortho block is initialized identity, then rotation grows during stage (i) to match teacher's per-position hidden geometry.
- **Why it might unlock the architecture:** This was the breakthrough recipe of 2024–2025 for linear-attention / Mamba conversion (LoLCATs, RADLADS, MOHAWK, "Mamba in the Llama"). At 135M / 5k steps from scratch, LM signal is too weak to find rotational circuits; teacher signal is dense and supervises each hidden state directly. Especially helpful because attention-alignment loss directly probes whether the hybrid's state tracks what the teacher's attention was tracking — i.e., it forces the ortho block to USE its capacity.
- **Key papers:** Zhang et al. NeurIPS 2024 (LoLCATs); Goldstein et al. arXiv 2505.03005 (RADLADS, 2025); Bick et al. NeurIPS 2024 (MOHAWK); Wang et al. NeurIPS 2024 ("The Mamba in the Llama").
- **Implementation cost:** **L.** Need teacher, alignment loss, layer-correspondence map (transformer L → hybrid L'), two-stage scheduler. ~3–5 days of engineering.
- **Code-LLM relevance:** **Very high** — distillation works particularly well for code (CodeLlama, etc.).
- **Risk:** Teacher quality bound: distilled hybrid won't beat the teacher. At 135M, both are weak. Mainly useful if we want to know "is our cell architecturally bottlenecked or train-time bottlenecked" — distillation isolates the latter. Highest absolute compute cost in the list.

## Recommendation

**Run Ideas 2 and 3 first, in combination, this week. Then escalate to Idea 1.**

**Why 2+3 first:** Together they are S+S cost (~1 day of engineering, no extra compute), they are complementary (Idea 3 makes the rotation block dormant-and-discoverable; Idea 2 supplies the gradient signal that makes it discoverable for the right reason), and they are independently falsifiable. The combined cost is essentially free relative to a 5k-step run. If PPL gap closes — answer found, move on. If not, we have ruled out the cheapest hypothesis and Idea 1 (synthetic-then-LM curriculum) becomes the principled next escalation, with strong literature backing (Amos 2024 is the closest analog to our exact pathology and reports very large gains).

**Skip / deprioritize:** Idea 5 (distillation) is high-cost, high-cost-of-failure-analysis, and the literature is clear it works — we already know how to make a hybrid work via distillation; the open question we care about is whether the hybrid architecture has *intrinsic* advantages, which distillation muddies. Run only if 1–4 fail and we want a "ceiling" measurement.

## Open questions

1. Does the rotation magnitude `α(t)` actually grow above zero with Idea 3 alone, or do we need Idea 2's aux gradient to push it? (Quick test: log per-layer rotation Frobenius norm during training.)
2. What's the right *mixture* of synthetic tasks for Idea 1 — does S5/A5 dominate or is parity sufficient? (Run an ablation at 50M scale.)
3. Does Idea 1's synthetic stage need to use the **same** tokenizer as the LM stage, or can it be raw integer tokens? (Tokenizer mismatch would be a confound; literature is mixed.)
4. Is the rotation block being suppressed by AdamW weight decay? (Easy test: set wd=0 on rotation params and re-run baseline.)
5. Do the kill-gate parity wins survive Idea 3's identity init? If `α(0)=0`, parity capacity at init is exactly DeltaNet's — i.e., zero. We need to confirm parity recovers by step ~500.
