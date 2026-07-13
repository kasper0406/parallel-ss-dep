# Lens: Test-time compute & adaptivity (verbatim agent report, 2026-07-13)

# Test-Time Compute & Adaptivity — 5 ideas for asymmetric exploitation

Grounding notes up front: literature searches confirm (a) no published work on constant-state branching as the enabler for program tree search — the transformer world is busy *mitigating* exactly this cost ([ArborKV, structure-aware KV management for tree reasoning](https://arxiv.org/pdf/2605.22106), [SGLang RadixAttention](https://arxiv.org/pdf/2312.07104)); (b) TTT-per-instance is validated on ARC ([Akyürek et al.](https://arxiv.org/html/2411.07279v1)) but not established for code with execution verifiers; (c) cascades/spec-decode are crowded ([Faster Cascades via Speculative Decoding](https://arxiv.org/html/2405.19261), [UCCI calibrated cascade routing](https://arxiv.org/html/2605.18796), [Mamba Drafters](https://arxiv.org/abs/2506.01206)) but use *confidence* deferral, which this project has already shown fails — execution-based deferral sidesteps that.

One honest frame constraining everything below (from the project's own pass@k finding): **pure selection cannot beat the base-set envelope (~18–21/164)**. Ideas are therefore split into envelope-*harvesters* (cheap, reliable) and envelope-*growers* (the real prize).

---

## Idea 1 — Repair-native decoding: train the error_text turn, then loop sample→grade→repair

**Mechanism.** The grader already returns dense `error_text` (traceback, failed-assert sources). Harvest (failed_code, error_text, passing_sibling) triples *for free* from existing BoN dumps (where one candidate in a group passed and others didn't — `gen_rejection_data.py` infra exists). SFT a repair turn: `problem + failed code + error_text → fixed code`. Deployment decode = BoN round 1 → grade → feed failures + diagnostics back → repair round 2/3. This is an envelope-**grower**: the repair distribution is conditioned on information (the actual failure) that no amount of round-1 sampling sees.

**Why asymmetric for us.** (i) Multi-round repair histories (problem + attempt + traceback + attempt2 + …) blow up transformer KV; ours is 8.2 MiB flat regardless of rounds. (ii) A 402M model can afford 10 repair rounds for less than one round of a 40B. (iii) Unique synergy: the exec-trace latent program is literally training the trunk to *simulate execution* — repair is the task where execution-comprehension cashes out.

**Kill-test (<1.5 days).** Harvest triples from prior MBPP/HumanEval BoN runs (~hours, CPU), SFT 1–2h on one 5090, then eval: repair@3-rounds vs iso-total-compute flat BoN (same total forward passes). Decisive metric: problems solved in round ≥2 that were 0/N in round 1 (true envelope growth, not reshuffled selection).

**Expected effect.** Self-Debug/Reflexion-class gains are +5–10 pp for large models; small models are worse at reading tracebacks, so honestly +2–5 problems on MBPP-scale, possibly less on HumanEval-hard where failure is knowledge-shaped. Even a null is informative (says the wall is knowledge, not feedback-use).

**Novelty.** Self-Debug/Reflexion (2023) established the loop for big models; training a *tiny* model's repair turn on self-harvested execution triples, co-trained with a neural-interpreter latent objective, is not in the literature I found. Modest novelty, high reliability.

---

## Idea 2 — Search-native decoding: state-checkpointed program tree search with partial-execution pruning + the latent exec-head as value function

**Mechanism.** At branch points (line boundaries or high-entropy/gate-flagged tokens), snapshot the recurrent state (8.2 MiB memcpy — `_clone_cache` already exists from soft-mixture decode), spawn k continuations batched through `forward_step`. Two value signals for pruning/backtracking: (a) **partial execution** — exec the program prefix up to the last complete statement (syntax-close it), grader tiers as reward; (b) where the prefix isn't executable, the **exec-trace latent head as a learned value function** — the neural interpreter predicts where execution is heading without running anything. Backtrack = restore an 8.2 MiB snapshot; hundreds of live branches fit in <2 GB.

**Why asymmetric for us.** This is the purest structural claim in the project. Transformer tree search is KV-memory-bound — the existence of [ArborKV](https://arxiv.org/pdf/2605.22106) and radix-tree prefix sharing as active 2025–26 research areas proves branching cost is *their* binding constraint. For us branching is O(1) in context length: a branch at token 8,000 costs the same 8.2 MiB as at token 100. A 40B transformer cannot hold 500 live search branches; we can on half a 5090.

**Kill-test (~2 days, the engineering is the cost).** MBPP + grader: line-level stochastic beam (width 8–16, branch every line, prune on partial-execution tier) vs **iso-forward-pass flat BoN with verifier**. That control is mandatory — the only honest question is whether *redirecting* compute via mid-generation feedback beats spending it on independent samples. Decisive metric: solved-problem set difference vs iso-compute BoN over ≥3 seeds. Phase 2 (only if phase 1 wins): swap partial-execution value for the latent exec-head value on non-executable prefixes.

**Expected effect.** Risky. MBPP programs are short (~10 lines), so mid-generation signal has few decision points; plausible outcomes range from null (knowledge wall dominates, search reshuffles the same base set) to +3–6 problems where round-1 sampling wastes mass on early wrong commitments (the documented early-commitment failure mode). The *systems* result (500-branch search in constant memory, benchmarked vs a transformer baseline OOMing) is publishable even if capability delta is small — and it composes with Ideas 1 and 3 as the decode substrate.

**Novelty.** MCTS-for-code with execution feedback exists ([AdverMCTS](https://arxiv.org/pdf/2604.10449), ToT, REx); [Mamba Drafters](https://arxiv.org/abs/2506.01206) use tree drafts but only for spec-decode alignment. **Constant-state branching as the enabler, plus a trained neural interpreter as the value function for non-executable prefixes, appears unpublished.** The value-function half is the genuinely novel science.

---

## Idea 3 — Execution-gated cascade: 402M solves what it can, escalates on grader failure (never on confidence)

**Mechanism.** Local-first agent: 402M runs BoN-with-verifier (N=64–128, nearly free). Deferral rule = **did anything pass the visible tests** — a deterministic, label-free oracle — not a confidence score. Only unsolved problems escalate to a big API model, optionally *with* the small model's best attempt + error_text as context (the big model does repair, not generation from scratch — cheaper tokens).

**Why asymmetric for us.** The whole cascade literature ([UCCI](https://arxiv.org/html/2605.18796), [speculative cascades](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)) struggles with calibrated deferral; this project *proved* confidence arbiters fail on its model. Execution-grounding deletes the calibration problem. And O(1) decode means the local tier's cost is essentially flat in context length — exactly the agent-economics moat in the north star.

**Kill-test (<1 day, ~zero GPU).** MBPP/HumanEval: measure (i) fraction solved locally, (ii) $/solved-problem vs big-model-only, (iii) whether small-model-attempt-as-context reduces big-model tokens. All infra exists.

**Expected effect.** No capability gain by construction — pure economics: expect 30–50 % of problems resolved locally at ~1–3 % of the cost, i.e., a 1.5–3× cost reduction for equal final accuracy. A guaranteed-positive result that quantifies the thesis ("what does the cost moat buy in a real pipeline").

**Novelty.** Low-moderate (EcoAssistant/FrugalGPT-adjacent), but as a *measurement* for the committed north star it's the cheapest credible number the project can produce.

---

## Idea 4 — TTT-on-the-problem: state-priming + per-problem self-training on verified candidates

**Mechanism.** Two tiers, pointing the meta-TTT machinery at the *problem* instead of the repo. **Tier A (zero-gradient, unique to us):** the delta-rule state is a Widrow-Hoff learner with a meta-learned per-token LR (β) — ingest augmented views of the problem (visible asserts first, docstring paraphrases, I/O examples reformatted) so the state *adapts* before decoding begins. **Tier B (gradient, ARC-style):** run BoN, take candidates passing visible tests, do k LoRA/SGD steps on them, resample — per-problem rejection-sampling finetuning, feasible only because the model is 402M (seconds per problem).

**Kill-test (~1–1.5 days).** Tier B first (no new training needed): MBPP, N=64 → filter on visible tests → 20 SGD steps → N=64 again; control = flat N=128. Decisive metric: envelope growth (new problems entering pass@k) vs control. Tier A rides on the meta-TTT run already planned: add a "problem-priming vs shuffled-priming" arm to the existing real-vs-shuffled kill-test.

**Expected effect.** [TTT gave up to 6× on ARC](https://arxiv.org/html/2411.07279v1), but ARC is adaptation-shaped and code is knowledge-shaped; honest expectation is +1–3 problems from Tier B (it's iterated rejection sampling, known-positive but small), Tier A unknown — it's the real bet and shares its fate with meta-TTT.

**Novelty.** TTT-for-ARC is established; per-problem TTT with an *execution verifier* selecting the training set, on a sub-1B model, is thinly covered. Tier A (meta-learned recurrent state as the TTT mechanism, no gradients) would be novel if it works.

---

## Idea 5 — Diversity engineering for BoN: prompt-permutation / state-init ensembling

**Mechanism.** Same problem, k prompt variants (assert order, docstring paraphrase, added scaffold comments) → k different recurrent states → decode from each; union candidates under the grader. Targets the finding that temperature-only diversity saturates the base set — orthogonal prompt-space diversity can enlarge it. The [Product-of-Experts ARC work](https://arxiv.org/html/2505.07859v2) validated perspective-permutation as a diversity source; [sampling-diversity scaling analyses](https://arxiv.org/pdf/2502.11027) show diversity, not temperature, is the binding variable at large N.

**Why asymmetric / kill-test / effect.** Weakest asymmetry (any model can permute prompts), but re-prefilling k variants is cheaper for us at long contexts. Kill-test is ~half a day: pass@64 via 8 prompts × 8 samples vs 1 prompt × 64 samples, same temperature. Expect +0–2 problems; worth running only as a rider on Idea 1/3's eval runs. Novelty: low — this is a recipe knob, not a paper.

---

## Ranking by expected-value-per-GPU-hour

| Rank | Idea | GPU cost | P(positive) | Payoff shape |
|---|---|---|---|---|
| 1 | **#3 execution-gated cascade** | ~0 | ~1.0 | Pure economics, quantifies the thesis |
| 2 | **#1 repair-native decoding** | ~0.5 day | ~0.6 | Most reliable envelope *growth* |
| 3 | **#2 search-native decoding** | ~2 days | ~0.4 | Structural differentiation + composes with everything |
| 4 | **#4 TTT-on-problem** | ~1 day | ~0.4 (B) / ? (A) | Small sure gain + one live research bet |
| 5 | **#5 diversity ensembling** | ~0.25 day | ~0.5 | Marginal, run as a rider |

## Single best bet

**Idea 2 — search-native decoding with state checkpointing, partial-execution pruning, and the exec-trace latent head as the value function on non-executable prefixes** — but sequenced *after* the half-day repair-triple harvest of Idea 1, because repair becomes the search's backtrack-and-edit operator. It is the only idea where all three project assets converge on something no one else can build: the O(1)-state branch cost (the transformer world's active pain point per ArborKV/RadixAttention), the execution grader (mid-generation ground truth), and the neural interpreter (a learned value function for prefixes that can't yet be run). Even under the pessimistic capability outcome, the artifact — hundreds-of-branches program search in constant memory on a consumer GPU — is the most concrete demonstration yet of what the committed cost moat is *for*.

Sources: [ArborKV](https://arxiv.org/pdf/2605.22106) · [SGLang](https://arxiv.org/pdf/2312.07104) · [Akyürek et al., TTT for Abstract Reasoning](https://arxiv.org/html/2411.07279v1) · [Product of Experts on ARC](https://arxiv.org/html/2505.07859v2) · [Faster Cascades via Speculative Decoding](https://arxiv.org/html/2405.19261) · [UCCI cascade routing](https://arxiv.org/html/2605.18796) · [Mamba Drafters for Speculative Decoding](https://arxiv.org/abs/2506.01206) · [AdverMCTS](https://arxiv.org/pdf/2604.10449) · [Sampling Diversity in Scaling LLM Inference](https://arxiv.org/pdf/2502.11027) · [Geiping et al., recurrent-depth test-time compute](https://huggingface.co/papers/2502.05171)
