# Lens: Small-model recipe archaeology (verbatim agent report, 2026-07-13)

All research complete. Here is my synthesis.

---

# Reverse-Engineering Small-Model Success: What to Port Under Our Budget

## Part 1 — What the strongest sub-1B(-ish) recipes actually did

**SmolLM2 (1.7B/360M/135M, 11T/4T/2T tokens)** — [paper](https://arxiv.org/html/2502.02737v1). Four-stage curriculum: web-heavy stable phase (6T), math introduced at 6T, StarCoderData→Stack-Edu swap at 8T, then a **decay phase (last ~10%) upsampling the best data hard**: 24% Stack-Edu, 14% math (FineMath4+), LR linearly to 0. Stage-wise MATH went 3.7 → 7.27 → **22.07** — the decay-phase upsample is where the capability jump lives. Context extended 2k→8k only at the very end (40% long docs), "next to no degradation." Crucial for us: **the 135M/360M variants abandoned staging entirely** — at 2–4T tokens they ran a *single stage with the highest-quality data from the start* (Stack-Edu, FineMath, filtered DCLM from token 0), with re-run ablations at target scale. At our 5B budget we are far below even that regime: quality-first from day 0, plus a decay-phase upsample, is the SmolLM-team-consistent extrapolation.

**OpenCoder (1.5B, 2T + 100B anneal)** — [paper](https://arxiv.org/html/2411.04905v1). The most portable code-specific recipe, with ablations. Annealing phase = 100B tokens: 84B original distribution + 12.4B algorithmic corpus + 2.7B synthetic code snippets + 0.9B code textbooks, WSD schedule (peak 3e-4 → 1e-5). **Ablation: removing algorithmic corpus + synthetic data from the anneal materially drops HumanEval/MBPP** ("data quality is more important than quantity in the later stages"). Two-stage SFT (broad CS QA → practical code tasks) **ablation-validated better than joint mixing**. File-level dedup ≫ repo-level dedup. Everything is downloadable: `OpenCoder-LLM/opc-annealing-corpus`, `opc-sft-stage1/stage2` on HF.

**MobileLLM-R1 (140M/360M/950M, 4.2T)** — [paper](https://arxiv.org/html/2509.24945). 2T curated + resampling; influence-score-based mixture optimization (~6,800 GPU-hours of curation — not affordable for us). Mid-training = 2×100B phases where **code+math jump to ~60-80% of the mix** (Nemotron-Code, Nemotron-CC-Math, OpenWebMath). Optional KD from Llama-3.1-8B-Instruct in mid-training. Architecture: still deep-thin (950M = 22L×1536; 360M = 15L×1024), LLaMA3.2 128k tokenizer with tied embeddings *even at 140M*. **The RL ablation is decisive: SFT-trained 950M scores 74.0 GSM8K vs 57.0 for the RL-optimized base, and RL on top of SFT gives minimal gains** — at sub-1B, structured SFT supervision beats RL.

**Qwen3-0.6B (36T)** — [report](https://arxiv.org/html/2505.09388v1). Three stages (general 4k-ctx → STEM/code/reasoning upweight → long-context 32k). Post-training for the small models is **not RL at all: "strong-to-weak distillation" (off-policy teacher-response distillation + logit distillation), which they state outperforms RL in both performance and training efficiency** for small models.

**Qwen2.5-Coder-0.5B (5.5T)** — [report](https://arxiv.org/pdf/2409.12186). Mixture ablation: **70:20:10 code:text:math beat 85:10:5 and 100:0:0** — more code is not monotonically better; general text is load-bearing. Staged file-level FIM → repo-level FIM (with `<|repo_name|>`/`<|file_sep|>` special tokens). Architecture: 24L×896 — deep-thin at exactly our parameter class.

**Gemma-3-270M/1B** — [report](https://arxiv.org/html/2503.19786v1). 6T tokens for the 270M(!), **KD everywhere: trained by distillation from a larger teacher, sampling 256 logits per token weighted by teacher probabilities** — the entire small-Gemma line is KD-native, and post-training is also distillation-based (+BOND/WARM/WARP RL).

**LFM2 (350M–2.6B hybrid conv+GQA, 10T)** — [report](https://arxiv.org/html/2511.23404v1). The closest cousin to our thesis (edge-latency-optimized non-transformer). **KD from LFM1-7B for the entire 10T pretrain**, with a "tempered, decoupled Top-K" objective: KL decomposed into (a) an *untempered* binary term matching total probability mass on the teacher's top-K=32 set + (b) a temperature-scaled KL *within* the top-K — solves the support-mismatch instability of naive truncated+tempered KD. Balanced against hard-label CE. Architecture ablation: under an edge-latency budget, **adding linear-attention/SSM operators did not improve aggregate quality vs gated short conv + 6 GQA blocks out of 16 layers**. FIM applied to 50% of code examples. Notably: 16 layers at 350M–1.2B — the only successful small line that is *not* deep-thin, and it's the one optimized for the same objective (device latency) as us.

**OLMo-2-1B** — [paper](https://arxiv.org/pdf/2501.00656). Mid-training on Dolmino (quality-filtered resample + math/academic + **instruction data (Tülu) inside the mid-training mix**), validated per-source by "microannealing" (small LR-decay probes per candidate dataset). Production trick: **run the ~50B-token anneal 3× on different data orders and soup the checkpoints**.

**Falcon-H1-0.5B** — [report](https://arxiv.org/abs/2507.22448). Hybrid attention+Mamba; "a relatively small fraction of attention heads is sufficient"; claims 2024-7B-class quality at 0.5B.

**RADLADS** — [paper](https://arxiv.org/html/2505.03005v4). Converts Qwen2.5 (7B–72B) to RWKV-variant linear attention with **350–700M tokens (<0.005% of teacher tokens)**, retaining ~90–100% of teacher quality relative-to-chance. Protocol: weight transfer → 100M-token per-layer attention-hidden-state L2 alignment (seq 512) → 250–700M-token KL logit distillation → 100M-token context extension. Ablations: skipping the hidden-state-alignment step plateaus at a much worse minimum; de-novo init much worse; **freezing MLPs significantly hurts**; DCLM beat FineWeb-Edu as conversion data for Qwen teachers; optimizer-step count matters more than batch size; LoRA-rank reduction detrimental; GroupNorm unstable at scale (state-scaling instead). Warning: converting a *reasoning* teacher degraded — the converted linear model "does not improve as much with additional output tokens as teacher."

---

## Part 2 — The highest-EV elements we have NOT adopted and CAN afford

Ranked by expected value per GPU-hour. All fit in ≤2 GPU-weeks; the first three are days.

### 1. A real decay-phase (annealing) data upgrade using *downloaded* high-quality corpora — the single most replicated lever in every report

**Mechanism.** During the final 10–15% of training while LR decays to ~0, swap to a mixture dominated by the highest-quality algorithmic/educational code + math + instruction-formatted data. The low-LR phase imprints high-quality distributions without catastrophic forgetting of the base; every lab found capability jumps concentrated here (SmolLM2 MATH 7→22; OpenCoder ablation-validated HumanEval/MBPP drop without it; OLMo2 Dolmino; MobileLLM-R1 mid-training at 60-80% code+math; Qwen3 stage 2).
**Evidence strength: the strongest of anything surveyed** — independently replicated by 5+ labs, with direct ablations in OpenCoder and SmolLM2.
**Port to DeltaNet.** Zero incompatibility — pure data+schedule. We already run WSD, whose whole point is "decay from any plateau checkpoint"; the decay phase is a slot we are currently filling with *the same mix as the stable phase*, which no strong recipe does. Data is free: `opc-annealing-corpus` (algorithmic corpus + synthetic snippets + code textbooks, ~16B tokens of exactly the ablation-validated material), Stack-Edu, FineMath4+, plus instruction data in the decay mix (OLMo2/Dolmino precedent) which also softens our SFT-format OOD problems.
**Cheapest validation.** OLMo2-style *microannealing*: 3–5 candidate decay mixes × ~200–300M-token decay runs from the same plateau ckpt (~12–20h each on one 5090), scored on our stratified per-source CE + HumanEval-solution CE. Then one production decay of ~0.5–1B tokens.
**Honest expected effect.** At our scale, per-source CE gains of a few percent and low-single-digit HumanEval points on the SFT'd model; it will not close the token gap, but it's the largest data-side move remaining and it compounds with everything else.
**Rider (near-free): OLMo2 decay-soup.** Run the chosen decay 2–3× with different data order and soup. We already proved SWA gives ~3% CE on plateau checkpoints; this is the annealing-specific version, +1–2 days.

### 2. Upgrade the KD objective and check teacher compatibility (LFM2 decoupled top-K; "stronger teacher is not always better")

**Mechanism.** (a) Our offline top-K logit KD has the known truncation/temperature support-mismatch problem; LFM2's fix — untempered binary mass-on-top-K term + tempered within-top-K KL, mixed with hard-label CE — is a drop-in loss change to our existing bridge. (b) [Strong Teacher Not Needed?](https://arxiv.org/abs/2605.23857) shows distillation gains depend on teacher-student *compatibility*; pushing teacher strength can saturate or reverse gains. Gemma3 (256 sampled logits/token) and LFM2 (7B teacher for 350M students, entire 10T) prove KD-native small-model training is the 2025-26 norm — but our 7B-teacher-to-300M-student ratio (23×) is larger than LFM2's (20×) and unexamined.
**Evidence strength: strong on "KD-native works" (three production model lines), weak-to-medium on the specific objective** (LFM2 gives no ablation table for decoupled-top-K; the compatibility paper is systematic but academic-scale).
**Port.** Pure loss-code + teacher choice; nothing DeltaNet-specific.
**Cheapest validation.** Loss change ~1 day of work, then a 3-way A/B on the existing KD-heal setup (current loss vs decoupled-top-K; Qwen2.5-Coder-7B vs 1.5B teacher), ~1–2 days per arm at a few hundred M tokens, scored on HumanEval-solution CE.
**Honest expected effect.** Single-digit-% CE-gap improvements in the heal; the teacher A/B could matter more than the loss (a 1.5B coder teacher may transfer better *and* lets us generate teacher logits ~4× faster, which compounds across all future KD).

### 3. Post-training = staged distillation-SFT, not RL (two-stage SFT + off-policy reasoning distillation)

**Mechanism.** Three independent sources converge: MobileLLM-R1's ablation (SFT 74.0 vs RL-on-base 57.0; RL-on-SFT ≈ nothing), Qwen3's explicit replacement of RL with strong-to-weak distillation for its smallest models, and OpenCoder's ablation that **two-stage SFT (broad → code-specific) beats joint mixing**. This also matches our own pass@k finding (RL = greedy sharpening only, envelope is SFT-set).
**Port.** No incompatibility. Data downloadable: `opc-sft-stage1` (4M broad CS QA) → `opc-sft-stage2` (367K practical code), optionally + OpenCodeReasoning-2. Off-policy teacher-response + logit distillation on SFT data = our existing KD bridge pointed at instruct data.
**Cheapest validation.** Our SFT runs are ~1h; a two-stage vs joint A/B is a weekend including evals.
**Honest expected effect.** This attacks the greedy→envelope reliability gap (~18–21 pass@30 vs 3–7 greedy in our own measurements) — plausibly the cheapest few HumanEval points available to us, and it redirects effort away from GRPO cycles the field says are wasted at this scale.

### 4. FIM on the code streams (file-level, ~50% rate, Qwen special tokens)

**Mechanism.** Universal in code models (Qwen2.5-Coder staged file→repo FIM; LFM2 FIM on 50% of code; Phi-4-mini code-completion data "leading to significant performance improvements"). Trains suffix-conditioned infilling = the editing capability an agentic coder actually needs.
**Evidence strength: medium for L2R benchmarks** (it mostly adds a *capability*, roughly benchmark-neutral at fixed tokens), **high for the editing/agentic north star** we've committed to.
**Port.** Data-transform only; since we're adopting the Qwen tokenizer (#105), the FIM special tokens come for free. One DeltaNet-specific caveat: document permutation must respect our cu_seqlens/doc_ids isolation (machinery already exists); a linear-RNN reads the suffix into bounded state before generating the middle — this is exactly a bounded-state long-context recall exercise, i.e., it *exercises our WM/recurrence niche*.
**Cheapest validation.** Add FIM to the SFT/anneal streams first (not a fresh pretrain); eval on HumanEval-infilling / single-line SAFIM-style probes vs the no-FIM control. Rides existing runs, ~0 marginal GPU cost.
**Honest expected effect.** ~0 on HumanEval; real on infilling/editing evals (from ~0 to functional). Do it inside run #1's decay mix, not as its own campaign.

### 5. Deliberately *not* recommended despite prominence

- **Influence-based mixture optimization (MobileLLM-R1):** 6,800 GPU-hours of curation. Out of budget; our lean-mix + microannealing probes (#1) capture most of the value.
- **Trillion-token synthetic generation (Phi line):** generation cost alone exceeds our total budget. The affordable slice of the Phi insight is exactly OpenCoder's downloadable synthetic annealing corpus (covered in #1).
- **Multi-stage curricula for the stable phase:** SmolLM2's own sub-1B finding says at small token budgets you run single-stage quality-first — which we already do.

---

## Part 3 — What the reports say about our linearization/KD path (question 4)

**Strongly validating:**
- **Nobody in 2025-26 trains a competitive sub-1B model on anything like 5B tokens.** The floor is 2T curated (MobileLLM-R1, "fewer than 5T total"), typical is 4–11T, Gemma-3-270M used 6T, Qwen3 used 36T. From-scratch at 5B tokens is 400–7000× under the field. Inheritance is not just reasonable, it's the only coherent move.
- **RADLADS quantifies the conversion bargain:** 350–700M tokens retains ~90–100% of a softmax teacher in a linear-attention student, at $2k for 72B. Our MOHAWK-style 370M-token linearization sits exactly at the field-validated token scale. Steal their ablation lessons: hidden-state alignment step is load-bearing; never freeze MLPs; use DCLM (not FineWeb-Edu) as conversion data for Qwen teachers; optimizer steps > batch size; cosine-anneal to roughly the teacher's final LR.
- **KD-native small models are the norm** (Gemma3, LFM2's full-10T KD, Qwen3 strong-to-weak) — our KD bridge is the right chassis, worth the objective upgrade in #2.

**Warning flags:**
- **Pure-linear is the road not taken.** LFM2 ablated linear-attention/SSM operators *out* under an edge budget; Falcon-H1 found a small attention fraction sufficient and transformative; every efficient-architecture line that shipped competitive quality at sub-1B is hybrid. With sliding-window attention the bounded-state/O(1)-memory moat survives a hybrid. This independently confirms our hybrid-tax fork as *the* central question — the field's answer is already "hybrid, minimally."
- **RADLADS' reasoning-conversion caveat:** converted linear models gained less from longer outputs than their softmax teachers — directly relevant to our latent-thinking/long-CoT plans on a linearized base; measure inference-time-scaling retention explicitly after conversion.
- **Depth:** every successful sub-1B transformer is deep-thin (MobileLLM +2.7–4.3% from depth; Qwen2.5-Coder-0.5B = 24L×896; SmolLM2-135M = 30L). Our shallow-wide 10L is against the grain — with the notable exception of LFM2 (16L at 350M–1.2B), the one line latency-optimized like us. Conveniently, linearizing Qwen2.5-Coder-0.5B inherits 24 layers and resolves this in the field's direction without a decision on our part.

---

## Single best bet

**KD-through-anneal on the linearized base:** take the linearized+healed DeltaNet (Qwen2.5-Coder-0.5B donor), and run its WSD decay phase (~0.5–1B tokens, ~2–4 days on our rig) on a decay mixture built from the downloaded OpenCoder annealing corpus + Stack-Edu + FineMath4+ + instruction-formatted data + 50%-FIM'd code, with teacher logits active (decoupled top-K objective), run 2–3× on different data orders and soup'd. This stacks the four highest-evidence, fully-affordable elements (decay-phase quality upsample, KD-native training, instruct-in-decay, FIM) into one ~1-GPU-week campaign on the path we've already committed to — every component is independently ablation-validated by at least one production small-model report, and the data costs nothing but bandwidth.

Sources: [SmolLM2](https://arxiv.org/html/2502.02737v1) · [MobileLLM-R1](https://arxiv.org/html/2509.24945) · [OpenCoder](https://arxiv.org/html/2411.04905v1) · [Qwen3](https://arxiv.org/html/2505.09388v1) · [Qwen2.5-Coder](https://arxiv.org/pdf/2409.12186) · [Gemma 3](https://arxiv.org/html/2503.19786v1) · [LFM2](https://arxiv.org/html/2511.23404v1) · [OLMo 2](https://arxiv.org/pdf/2501.00656) · [Falcon-H1](https://arxiv.org/abs/2507.22448) · [RADLADS](https://arxiv.org/html/2505.03005v4) · [MobileLLM](https://arxiv.org/pdf/2402.14905) · [Strong Teacher Not Needed?](https://arxiv.org/abs/2605.23857) · [Phi-4-Mini](https://arxiv.org/pdf/2503.01743) · [SmolLM3 blog](https://huggingface.co/blog/smollm3)
