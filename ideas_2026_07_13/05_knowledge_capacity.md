# Lens: Knowledge capacity (verbatim agent report, 2026-07-13)

# Knowledge Capacity for a 400M DeltaNet Coder — 5 Ideas

Framing constraint I held throughout: the wall is *knowledge*, our validated lever is PKM-when-engaged, and any mechanism must preserve the O(1)-decode / bounded-GPU-state moat. Host RAM (64-128GB) and NVMe are *not* part of the bounded-state budget — the moat is about GPU decode cost, and a read-only table is economically "weights", not "state". That asymmetry is the main unexploited resource.

---

## Idea 1 — Host-RAM Giant PKM: scale the value table 10-100x past VRAM ("UltraMem-at-home")

**Mechanism.** Our PKM is 4 heads x 256² = 262k slots (~470MB bf16 at d=896). Scale to 4-16M slots (7-28GB bf16) living in pinned host RAM, with the sub-key product lookup on GPU (keys are tiny — sqrt scaling is the whole point of product keys) and only the top-k selected value rows gathered over PCIe. Per token that is ~k×heads ≈ 32 rows × 896 × 2B ≈ 57KB — ~1-2µs of PCIe 5.0 bandwidth, hidden under the block's compute. Keep a GPU-resident LRU cache of hot slots (code-token slot distributions are Zipfian; our own diagnostics show `top` slot mass concentration). Training: our v7.1 bootstrap package (α-floor, ε-greedy, value-LR-mult 100x) already solved the "dead table" problem; the new part is a sharded CPU-side EmbeddingBag with async H2D overlap for the sparse update (rows touched per step are bounded by batch×T×k — fits in a transfer buffer).

**Why it fits.** (a) Meta's Memory Layers at Scale showed log-linear factual-knowledge gains up to 128B memory params — memory-augmented models beat dense models with >2x compute; UltraMem showed inference time stays ~constant as sparse params grow. We have never tested whether *our* +0.04-0.14 CE lift is the start of that log-linear curve or a plateau — this is exactly the "iso-param comparison untested" hole in our evidence. (b) The moat survives cleanly: GPU decode memory stays constant, per-token host traffic is O(k), and the table is read-only at deploy — it is "more weights stored where weights are cheap", not unbounded state. (c) Fact-heavy sources benefited most in our per-source probe — the signature of a knowledge store, and knowledge is the diagnosed wall.

**Kill-test (<2 days, 2×5090).** Continue-pretrain the wide-10L base (or a 150M trunk from scratch) for ~1B tokens at three table sizes {262k, 2M, 8M slots}, iso-token, iso-trunk, plus a no-table control. Metrics: per-source CE on fact-heavy streams, the dep-distance-stratified code CE probe (our first real feature win — use it), tokens/s (does the CPU gather cost <10% throughput?), and slot-utilization diagnostics (does ε-greedy still light up 8M slots, or does coverage collapse — the scaling failure mode). Decision rule: if 262k→8M gives ≥2-3x the 262k lift on fact-CE with <10% throughput tax, the curve is live; if flat, PKM is confirmed as sample-efficiency-only and we stop investing here permanently.

**Expected effect.** Stratified/fact CE: high confidence of a real, monotone win. HumanEval/MBPP greedy: honestly modest at 5B tokens (+0-3 problems) — knowledge tables need token exposure to fill; the *decisive* version is pairing this with a long code-token run (or the linearized-Qwen base). This is a capacity unlock whose realized value scales with training tokens.

**Novelty.** Architecture: Memory Layers at Scale (Meta, arXiv:2412.09764) and UltraMem/UltraMemV2 (ByteDance, ICLR 2025, arXiv:2411.12364 / 2508.18756) are the closest — both at datacenter scale with GPU-sharded tables. Host-RAM-offloaded memory tables for a sub-1B *edge* model, framed as a cost-moat (VRAM-free knowledge), is an unoccupied niche — engineering novelty plus a genuinely unpublished small-scale scaling question.

---

## Idea 2 — State Cartridges: precomputed DeltaNet states as loadable knowledge modules ("RAG into state")

**Mechanism.** Our recurrent state is 8.2MiB, flat. Precompute "cartridges" offline: ingest curated library docs (numpy, requests, re...) through the frozen model in one O(T) forward, snapshot the final DeltaNet state + WM buffer, store to disk. At task time, parse imports, load the matching cartridge(s) as the initial state (~ms from NVMe), generate as usual. Two tiers: (A) pure forward-ingest (zero training, deploy-free); (B) Cartridges-style *self-study* — generate synthetic QA about each doc corpus, then train the model (or just a state-space adapter/per-cartridge state via gradient) with a context-distillation objective so that answering *from state* actually works. Tier B is our parked meta-TTT machinery repurposed with a concrete, cheap target.

**Why it fits.** This is the only idea where the bounded state IS the mechanism rather than a constraint: a KV-cache cartridge for a transformer is 100s of MB and grows with doc length; ours is 8MiB *regardless of how many doc tokens were ingested*. It converts "we cannot stuff context" from a weakness into the product story: knowledge modules at fixed cost. It also directly tests the "delta rule = Widrow-Hoff fast-weight learner" thesis on a task that matters.

**Kill-test (<1-2 days).** Build 20 cartridges from real API docs. Probe: CE / exact-match on correct API-call tokens (function names, argument names, argument order) for held-out usage examples, three arms: matched cartridge vs *shuffled-docs cartridge* (the control from the meta-TTT note — kills "generic warm-state" explanations) vs cold state. Tier A is a half-day of pure inference. If Tier A is null (likely — our stateful-chunking probe showed generic state-carry gains are tiny), spend day 2 on Tier B self-study finetune on 5 libraries and re-run the same three arms. Decision rule: matched-vs-shuffled gap > 0 with p<0.01 on API-token accuracy, else park again.

**Expected effect.** Tier A: probably ~0 (state that was never trained to retain doc facts won't). Tier B: plausible real lift on the API probe; HumanEval unchanged (its APIs are builtins) — the right eval is an import-heavy MBPP slice or a small custom "unfamiliar library" suite, which is also the honest framing: this mechanism targets *long-tail/private* APIs, exactly what no 5B-token pretrain can cover.

**Novelty.** Cartridges (Hazy Research, arXiv:2506.06266) and Cartridges-at-Scale (arXiv:2606.04557) do this for transformer KV caches with per-corpus gradient training. Doing it in a bounded linear-RNN state — where ingest is a *forward pass* and the artifact is 8MiB constant — is genuinely novel, and the closest thing to a publishable differentiator on our stack.

---

## Idea 3 — Surgical knowledge transplant: failure-mined distillation into PKM slots only

**Mechanism.** Close the loop between our measured failures and the memory table. (1) Mine: run our model and Qwen2.5-Coder-7B (already deployed via our KD pipeline) over MBPP/HumanEval-adjacent code; collect positions where teacher is confident and we are wrong AND the token is knowledge-shaped (API identifier, argument keyword, constant, import path) — a cheap classifier over token type + teacher entropy. (2) Synthesize: for each mined knowledge item, generate 20-100 usage variants (teacher-generated). (3) Inject: finetune *only* the PKM value table (+ optionally query/keys) on this corpus, trunk frozen. Sparse slot updates touch a tiny fraction of parameters, so forgetting is structurally limited — this is exactly the finding of "Sparse Memory Finetuning" (arXiv:2605.03229): memory-layer finetuning forgets far less than LoRA or full FT.

**Why it fits.** It is the cheapest direct test of "the wall is knowledge, and PKM is where knowledge lives": if transplanting the *measured* missing facts into slots doesn't move the evals, then either the mining is wrong or the wall isn't PKM-addressable — both decision-relevant. Trunk frozen → zero risk to the base; bounded state untouched; the loop (deploy → observe failures → write slots overnight) is also the germ of the ingraining/sleep-consolidation tier of the north star.

**Kill-test (<1 day).** 50 mined knowledge items → ~20k synthetic examples → memory-only FT (single-GPU, ~2-4h) → three measurements: (a) targeted-item accuracy (should approach 100% — sanity), (b) held-out general per-source CE (forgetting guard, must be ~0 drift), (c) HumanEval/MBPP temp pass@k with bootstrap CI (never greedy-only, per our eval-noise lesson). Decision rule: (a) fixed AND (b) flat is already a positive infrastructure result; any (c) movement is upside.

**Expected effect.** (a) near-certain, (b) likely fine per the sparse-FT literature, (c) +0-3 problems — bounded by how many bench failures are truly discrete-fact-shaped rather than compositional. High information value either way.

**Novelty.** Composition is the novelty: model editing (ROME/MEMIT) meets memory-layer sparse finetuning (2605.03229) meets automated failure mining for code APIs. No paper I found closes this specific loop on a code model.

---

## Idea 4 — Bits-vs-slots: quantized value tables as a knowledge-Pareto question

**Mechanism.** PKM values are pure lookup — no matmul — so quantization costs one dequant on k rows/token, essentially free. At a *fixed byte budget*, compare fewer-fp16-slots vs more-int4-slots vs many-int2-slots (QAT: fp32 shadow weights, fake-quant forward — standard recipe from recsys embedding tables, which survive 4-bit well and 2-bit with QAT). The trunk stays bf16 — the asymmetry (2-bit knowledge, 16-bit compute) is principled: values encode *which* fact, the trunk computes *with* it; facts tolerate low precision.

**Why it fits.** Multiplier on Idea 1: if int4 is lossless, the same 28GB of host RAM holds 64M slots instead of 16M. Also GPU-only relevance: 4x slots inside the current VRAM budget without touching the trunk.

**Kill-test (~1 day).** Small trunk, iso-token, iso-*byte* table budget, three arms {262k fp16, 1M int4, 4M int2}, fact-source CE + slot-utilization. Clean, fully automated sweep.

**Expected effect.** Standalone HumanEval: negligible. As a scaling multiplier: material. Prior from recsys says int4-more-slots wins.

**Novelty.** Embedding-table quantization is old (arXiv:1911.02079, recsys mixed-precision); the bits-vs-slots *Pareto for LM memory layers trained with QAT* appears unpublished — a small, clean contribution.

---

## Idea 5 — API-surface pretraining stream: (signature, doc, example) tuples as a first-class format

**Mechanism.** Auto-generate a structured corpus from the top ~200 PyPI packages via `inspect` + doc scraping: canonical template of import path, signature, one-line semantics, 1-2 minimal runnable examples, common exception. Upsample it in the mix as its own source. Crucially, apply our const-recall lesson: supervise usage *at first occurrence* (where recurrence/memory must produce the API name), not at restatements — otherwise the stream trains nothing, like the recall-stream bug. Pair with PKM engagement kill-gates so we can see whether the new facts land in slots or trunk.

**Why it fits.** Our own synthesis says the headline moves via DATA, not mechanisms ("under-trained, not under-capacity"); this is the knowledge-wall attack that requires no new machinery and it manufactures exactly the token type the wall is made of, at ~1000x higher fact-density-per-token than scraped code.

**Kill-test (<1.5 days).** Generate ~200-500M tokens, continue-pretrain the wide-10L SWA base ~1 day, eval: held-out signature-completion probe (argument-name exact-match on packages *in* vs *out* of the corpus — the in/out split is the control), plus MBPP temp pass@k. Decision rule: in-corpus argument accuracy jump with out-of-corpus flat → format works, scale it into the next big run.

**Expected effect.** Probe win near-certain; MBPP +1-3; the *compound* value is as the default data format for every future token we spend.

**Novelty.** Low — DocPrompting, API-doc-conditioned generation, and "On Mitigating Code LLM Hallucinations with API Documentation" (arXiv:2407.09726) are adjacent; big labs do this implicitly. Value is practical, not publishable.

---

## Ranking by expected value per GPU-hour

1. **Idea 3 (surgical transplant)** — hours of GPU, frozen trunk, decision-relevant regardless of outcome, builds the ingraining loop.
2. **Idea 5 (API-surface stream)** — ~1 day, near-certain probe win, compounds into all future runs; ranked below 3 only because its novelty is nil and its full value needs the next big token budget.
3. **Idea 1 (host-RAM giant PKM)** — ~2 days, strong external prior (Meta/ByteDance scaling laws), answers our open iso-param/capacity question, preserves and *extends* the cost moat.
4. **Idea 2 (state cartridges)** — 1-2 days, highest novelty and best north-star fit, but highest null-risk (Tier A likely 0; Tier B is real work).
5. **Idea 4 (bits-vs-slots)** — cheap and clean but only matters conditional on Idea 1 succeeding.

## Single best bet

**Idea 1, with Idea 4 folded in as a fourth arm**: the host-RAM giant PKM scaling test. It is the only idea that can make a 400M model *structurally* know 10-100x more than its VRAM allows; the external scaling-law prior is the strongest of any idea here; it converts our one validated knowledge mechanism from "sample-efficiency lever" into a possible capacity unlock while leaving the GPU decode footprint constant — i.e., it strengthens rather than compromises the O(1)-decode moat, because "cheap bounded-state agent with a 30GB knowledge table in host RAM" is a *better* edge story than "small model that knows little". And its kill-test is genuinely decisive: a flat slot-scaling curve at iso-token would close this entire direction, which is worth two GPU-days on its own.

Sources: [Memory Layers at Scale (arXiv:2412.09764)](https://arxiv.org/abs/2412.09764), [UltraMem (arXiv:2411.12364)](https://arxiv.org/pdf/2411.12364), [UltraMemV2 (arXiv:2508.18756)](https://arxiv.org/html/2508.18756v1), [ByteDance UltraMem blog](https://seed.bytedance.com/en/blog/seed-research-new-ultra-sparse-architecture-reduces-inference-costs-by-up-to-83-compared-to-moe), [Cartridges: self-study KV caches (arXiv:2506.06266)](https://huggingface.co/papers/2506.06266), [Cartridges at Scale (arXiv:2606.04557)](https://arxiv.org/html/2606.04557v1), [Hazy Research Cartridges blog](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges), [Sparse Memory Finetuning (arXiv:2605.03229)](https://arxiv.org/pdf/2605.03229), [Memory Decoder (arXiv:2508.09874)](https://arxiv.org/html/2508.09874v1), [Great Memory, Shallow Reasoning: Limits of kNN-LMs (arXiv:2408.11815)](https://arxiv.org/pdf/2408.11815), [Post-Training 4-bit Quantization on Embedding Tables (arXiv:1911.02079)](https://arxiv.org/pdf/1911.02079), [Mixed-Precision Embeddings for Recommendation (arXiv:2409.20305)](https://arxiv.org/pdf/2409.20305), [Mitigating Code LLM Hallucinations with API Documentation (arXiv:2407.09726)](https://arxiv.org/pdf/2407.09726), [Meta memory layers — VentureBeat](https://venturebeat.com/ai/meta-proposes-new-scalable-memory-layers-that-improve-knowledge-reduce-hallucinations)
