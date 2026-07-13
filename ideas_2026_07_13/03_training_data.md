# Lens: Training methods & data efficiency (verbatim agent report, 2026-07-13)

# Training Methods & Data Efficiency — 5 ideas for capability-per-token at 400M

Framing constraint accepted: you are token-poor, not capacity-poor. Every idea below is a *multiplier on effective tokens* or a *reuse of compute you already spend*. Throughput assumption for kill-tests: ~30–50k tok/s at 287–402M per 5090 → 500M tokens ≈ 3–5 h per arm, so all kill-tests fit in 1–2 days on 2 GPUs.

---

## Idea 1 — Teacher-scored token triage: Rho-1-style selective loss using the KD logits you already store (RANK 1)

**Mechanism.** [Rho-1 / Selective Language Modeling](https://arxiv.org/abs/2404.07965): score every pretrain token by *excess loss* (student CE − reference CE), train CE only on the top ~60% of tokens; noise/boilerplate tokens get zero loss. Result: Rho-1-1B matched DeepSeekMath with **3% of the tokens** on math continual-pretrain, +6.8% avg across 15 tasks on general tokens. Your twist: the reference model is **free** — your offline vLLM KD bridge already writes Qwen2.5-Coder-7B logits to disk. So run a three-way per-token routing in one pass: (a) teacher-confident & student-wrong → sparse top-k KD loss (the informative tokens); (b) both-wrong / high teacher entropy → drop or downweight (inherently unpredictable tokens — license headers, random hex, unique identifiers); (c) both-right → plain CE at low weight. This is "not all tokens are what you need" fused with "distill only where it pays" ([AdaKD](https://arxiv.org/html/2510.11615v1), [SelecTKD](https://arxiv.org/pdf/2510.24021), [Random Sampling KD](https://arxiv.org/pdf/2503.16870) show sparse/selective KD works with <10% overhead).

**Why it fits.** Directly attacks the binding constraint (under-trained, not under-capacity) as a per-token multiplier; zero new infra beyond a scoring pass over logits you already computed; your dep-distance-stratified CE gives you exactly the dev signal to see whether the selected tokens are the structure-bearing ones. Also consistent with your hygiene scars: the token-drop mask is auditable per-source before training (learn from min_content_len).

**Kill-test (~1 day, 2 arms × 1 GPU).** From the same base ckpt, continue-pretrain 400–500M code tokens: arm A full-CE baseline, arm B SLM top-60% (+ optional KD-on-disagreement). Gates: held-out HumanEval-solution CE, dep-distance-stratified CE, and a per-source audit of what got dropped (if it's dropping the long-range-reuse tokens, kill it).

**Expected effect (honest).** Rho-1's headline is math; code gains in the paper are smaller (~few points). Realistic: 1.5–2× effective-token multiplier on the code CE curve, i.e., 0.02–0.05 CE at iso-token. That's large for you — it compounds with everything else. Failure mode: at 400M with a 7B reference, "excess loss" may select tokens the student *can't* learn yet; the routing (c) mitigates.

**Novelty.** Low-moderate — Rho-1 is 2024, selective KD is active 2025–26. The specific fusion (reference scores as a free byproduct of an existing KD pipeline; three-way route CE/KD/drop) is an engineering-novel combination, not a paper claim.

---

## Idea 2 — Mutated multi-epoch: verified semantics-preserving transforms as "fresh" tokens (RANK 2)

**Mechanism.** [Muennighoff's data-constrained scaling laws](https://www.jmlr.org/papers/volume26/24-1000/24-1000.pdf): repeating data ≤4 epochs is nearly free, value decays to ~zero by ~16–40 epochs. [WRAP-style rephrasing](https://mbrenndoerfer.com/writing/data-constrained-scaling-llm-training-data-limits) is the known alternative to raw repetition. Code gives you something web text can't: **provably semantics-preserving mutations** — AST-level variable/function renaming, formatting/style permutation, statement reordering where dataflow allows, comment strip/paraphrase, dead-code insertion, decompose/inline refactors — each verifiable by re-running the tests you already have in the grader. Per epoch k, re-render the same corpus through a fresh mutation seed. The surface distribution changes (defeating memorization, which is what makes repeats decay) while the semantic distribution is fixed. Bonus alignment with your dep-distance finding: renaming forces the model to bind *this document's* identifier, not a memorized global one — it's targeted training for exactly the long-range-reuse capability your stratified probe rewards.

**Why it fits.** Token poverty says you *will* multi-epoch; the only question is whether epochs 4–12 are worth anything. You have the AST/execution infra to verify mutations. Hygiene lesson: mutation pipeline emits per-source token counts + a mutated/verbatim ratio log from day 1.

**Kill-test (~1 day, 3 arms).** Iso-compute at deliberately high repetition where the effect is visible: 8 epochs × 125M tokens — (A) verbatim × 8, (B) mutated × 8, (C) 1 × 1B fresh tokens (upper bound). Measure held-out CE + stratified CE. Decision rule: if B recovers >40% of the A→C gap, adopt for the production run.

**Expected effect.** At 4 epochs, small (verbatim is already cheap); the payoff is extending your usable-epoch horizon from ~4 to ~10–16, i.e., turning a 5B corpus into ~15–25B effective tokens of the *highest-quality* subset. Risk: mutations that break naturalness (weird names) could hurt; keep a verbatim fraction per epoch.

**Novelty.** Moderate. WRAP/rephrasing is established for web text; identifier-renaming augmentation exists in code-representation learning; systematic *mutation-as-epoch-schedule under data-constrained scaling laws* for generative code pretraining is thinly covered.

---

## Idea 3 — The trace-ladder: generalize your Stage A→B result into a verified curriculum of program families (RANK 3)

**Mechanism.** You just validated the recipe: teach a skill in TEXT scratchpad, then compress to latent; straight-line executor transferred +53% rel to real-code output prediction. Generalize into a **skill tower** of synthetic executable families with a generator+executor per rung: straight-line → branches → bounded loops (accumulators) → early-exit/search loops → recursion → dict/list state → multi-function call chains → *bug-localization* (mutate one line, trace diverges, predict where). Each rung: infinite free data, ground-truth traces from execution (process supervision at zero labeling cost), Stage A text-trace training → Stage B latent compression, with **spaced replay** of earlier rungs (you already learned consolidation is mandatory from latent-arith forgetting). This is your in-house version of what [CodeI/O](https://arxiv.org/abs/2502.07316) (ICML'25 oral) showed transfers broadly, and what [SemCoder](https://arxiv.org/pdf/2406.01006) showed lifts CRUXEval; [ExecVerify](https://arxiv.org/html/2603.11226) adds stepwise verifiable rewards in the same space.

**Why it fits.** It's the direct continuation of your single most recent validated result, and it's the one data engine where you own the whole verification loop. Crucial published caveat that matches your instincts: traced *human-written* functions can hurt via trace-format pattern-matching (["Code Execution as Grounded Supervision"](https://aclanthology.org/2025.emnlp-main.1260.pdf) line of work) — so the transfer gate must be held-out-*format* real code (CRUXEval-O style), never the synthetic format itself.

**Kill-test (1–2 days).** One new rung only: state-carrying loops. (i) Does Stage A→B hold (text-first works, latent-first fails — replicating your Stage-A finding on a harder family)? (ii) Does adding the loops rung on top of the straight-line executor *increase* real-code output-prediction transfer beyond the +53%, or is transfer saturated at rung 1? (ii) is the decisive question — it tells you whether the tower is worth building.

**Expected effect.** If transfer stacks: this is your CodeI/O — plausibly the biggest capability-per-token lever you own, since the data is free and process-supervised. If transfer saturates at rung 1, you've spent 2 days learning the ceiling early. Your depth-via-iteration finding (heterogeneous composition collapses) predicts trouble at the multi-function rung — that's informative either way.

**Novelty.** The trace-training space is crowded (CodeI/O, SemCoder, TracePile, NExT). Your text→latent compression stage + explicit family curriculum + spaced replay is a genuinely distinctive combination; nobody publishes latent-compressed trace curricula.

---

## Idea 4 — Branch-Train-Merge to dodge your DDP jail + soup the specialists (RANK 4)

**Mechanism.** Your latent co-train runs are forced single-GPU (DDP+latent doubly blocked). Branch-Train-Merge ([BTM](https://arxiv.org/pdf/2502.01804)-family; [Soup-of-Experts, ICML'25](https://arxiv.org/abs/2502.01804); merging-CPT case studies [2511.02451](https://arxiv.org/html/2511.02451v1)) converts the idle second GPU into a *parallel specialist branch*: from a shared WSD-plateau ckpt, GPU-0 continues on code-heavy mix, GPU-1 on trace/reasoning mix (or the Idea-3 ladder), then weight-average. You already validated the enabling fact: SWA over WSD-plateau ckpts gave a free ~3% CE and proved plateau ckpts live in one merge-compatible basin — short branches from a shared plateau are exactly the small-drift regime where averaging works (the literature's failure mode is *long divergent* mid-training).

**Why it fits.** Pure throughput arbitrage: 2× wall-clock on your most constrained resource, plus the soup variance-reduction you already measured. Zero interconnect needed (no NVLink is irrelevant).

**Kill-test (~1.5 days).** Iso-compute: (A) single-GPU 1B-token mixed run vs (B) 2 × 500M-token branches (code / trace) merged, optional light 50M-token anneal after merge. Compare held-out CE per source. Gate: B ≥ A − noise. Sweep merge weight {0.3, 0.5, 0.7} — cheap, no retraining.

**Expected effect.** Mostly a *2× wall-clock* win with parity quality; a small CE win if the specialist gradients interfere less than mixed-batch gradients. Failure: branches drift out of basin — keep branches ≤500M tokens and LR at plateau level.

**Novelty.** Low — this is applied BTM/soup. The value is fit-to-constraint, not research novelty.

---

## Idea 5 — Dense trace-matching RL rewards (RANK 5 — flagged low-EV for you specifically)

**Mechanism.** Replace pass/fail grader reward with per-checkpoint state matching: execute the rollout, compare intermediate variable states against reference execution, reward = fraction matched (+ pass bonus). [ExecVerify](https://arxiv.org/html/2603.11226) and [CodeRL+](https://arxiv.org/html/2510.18471v2) are the 2025–26 versions of this; it densifies GRPO advantage exactly where your dense-tier grader ladder left off.

**Why ranked last.** Your own pass@k autopsy is the counter-evidence: the envelope (~18–21) is SFT/base-set, RL only sharpens greedy into it, and round-2 RL saturated. Dense rewards make the sharpening cheaper, but can't add knowledge to the base set. Only worth running *after* Ideas 1–3 raise the envelope. Kill-test if ever: 200-step GRPO A/B, dense vs pass/fail reward, measure pass@30 envelope not greedy — half a day. Expected: faster convergence to the same envelope; envelope growth would be a surprise worth knowing.

---

## Ranking by EV per GPU-hour

| # | Idea | Cost of kill-test | P(win) × size | EV/GPU-h |
|---|---|---|---|---|
| 1 | Teacher-scored token triage (Rho-1 + sparse KD) | ~8 GPU-h | high × large | **highest** |
| 2 | Mutated multi-epoch | ~15 GPU-h | med-high × large (epoch-horizon) | high |
| 3 | Trace-ladder (Stage A→B tower) | ~20 GPU-h | med × potentially largest | high, higher variance |
| 4 | BTM soup (2-GPU parallelism) | ~35 GPU-h | high × small (mostly wall-clock) | medium |
| 5 | Dense trace RL | ~6 GPU-h | low × small (envelope-capped) | low |

## Single best bet

**Idea 1 — teacher-scored token triage.** It is the only idea that multiplies the value of *every* future token you train on, it reuses compute you have already spent (the stored Qwen2.5-Coder-7B logits make the Rho-1 reference model literally free), the kill-test is the cheapest and cleanest (two arms, one day, existing dev signals), and Rho-1's published magnitude — SOTA math with 3% of tokens, +6.8% avg on general continual pretrain — is exactly the shape of lever a 400M model at 5B tokens vs competitors' trillions needs. Ideas 2 and 3 then stack on top of it: triage decides *which* tokens carry loss, mutation decides *how often* you can re-see them, and the trace-ladder decides *what free tokens to add*.

Sources:
- [Rho-1: Not All Tokens Are What You Need (arXiv 2404.07965)](https://arxiv.org/abs/2404.07965)
- [Scaling Data-Constrained Language Models (Muennighoff et al., JMLR)](https://www.jmlr.org/papers/volume26/24-1000/24-1000.pdf)
- [Data-Constrained Scaling overview incl. rephrasing-as-alternative](https://mbrenndoerfer.com/writing/data-constrained-scaling-llm-training-data-limits)
- [CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction (ICML 2025)](https://arxiv.org/abs/2502.07316)
- [SemCoder: Training Code LMs with Comprehensive Semantics Reasoning](https://arxiv.org/pdf/2406.01006)
- [ExecVerify: White-Box RL with Verifiable Stepwise Rewards](https://arxiv.org/html/2603.11226)
- [Code Execution as Grounded Supervision (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1260.pdf)
- [SelfCodeAlign: Self-Alignment for Code Generation (NeurIPS 2024)](https://arxiv.org/abs/2410.24198)
- [SelecTKD: Selective Token-Weighted KD](https://arxiv.org/pdf/2510.24021)
- [AdaKD: LLM-Oriented Token-Adaptive KD](https://arxiv.org/html/2510.11615v1)
- [Sparse Logit Sampling: Accelerating KD in LLMs](https://arxiv.org/pdf/2503.16870)
- [Soup-of-Experts: Pretraining Specialist Models via Parameter Averaging (ICML 2025)](https://arxiv.org/abs/2502.01804)
- [Merging Continual Pretraining Models (finance case study)](https://arxiv.org/html/2511.02451v1)
- [CodeRL+: Execution Semantics Alignment RL](https://arxiv.org/html/2510.18471v2)
- [Curriculum Learning for LLM Pretraining: Learning Dynamics (2026)](https://arxiv.org/abs/2601.21698)
