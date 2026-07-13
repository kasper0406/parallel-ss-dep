# Lens: Contrarian wildcards (verbatim agent report, 2026-07-13)

Below is my deliverable as the CONTRARIAN WILDCARDS ideation agent. All ideas were checked against literature through mid-2026 (searches run; sources at the end).

---

# Contrarian wildcards for the 287–402M DeltaNet coder

Framing constraint I optimized against: token poverty binds, features bolted onto converged models stay inert, and the committed north star is a cost-moat agent (O(1) decode, adaptivity), not benchmark rank. So the wildcards below either (a) manufacture supervision without tokens, (b) exploit the delta-rule state as a mathematical object nobody else has, or (c) accept a measured ceiling instead of fighting it.

---

## 1. State Algebra — the 8.2 MiB state as a composable, searchable, mergeable artifact

**Idea.** Treat the DeltaNet recurrent state as a first-class object with an algebra: (a) **parallel ingestion** — split a repo into N shards, ingest each independently (parallel prefill across both GPUs), merge the states, and decode from the merged state; (b) **state libraries** — precompute per-library/per-file states offline, retrieve and mix at test time (a "cartridge" that costs one forward pass, no training); (c) **crossover as search** — for a hard problem, run N diverse partial rollouts, select states of the most promising ones, linearly mix them, resume generation (population-based search where the genome is the fast-weight matrix).

**Mechanism.** The delta rule is Widrow–Hoff: the state is (approximately) the running least-squares solution of an associative key→value regression. Sums of rank-1 updates from disjoint key subspaces are nearly additive, so merging states from disjoint content should approximately equal sequential ingestion — and *State Soup* (arXiv 2406.08423) already showed linear interpolation of Mamba-2.8b states improves perplexity and ICL, with query-based retrieval over a state library working zero-shot. Nobody has applied this to a delta-rule model, to repo ingestion, or as a search operator. If it works, it's a moat a transformer literally cannot copy (KV caches concatenate, they don't *mix*; Cartridges need training).

**Why it probably fails.** The delta rule is precisely NOT additive — it's interference-aware: each write first *subtracts* the state's current prediction for that key. Updates don't commute; merging two states that saw overlapping keys double-counts and produces off-manifold states the trunk never saw. The exact parallel combination is what LASP's inter-chunk correction term computes — naive averaging throws that term away, and the discarded term is largest exactly when content overlaps (real repos overlap heavily: imports, idioms). State Soup used a gated-linear model on soft tasks; precise recall (your regime) may shatter under mixing. Also β-gating means later shards' decay schedules differ from what sequential ingestion would apply.

**Cheapest experiment (<1 day, existing harness).** Take the saturating multibind probe (`wm_recall_cotrain` infrastructure): bindings 1–32 in context A, bindings 33–64 in context B, evaluate recall from (i) sequential A+B state, (ii) mean of states, (iii) sum, (iv) per-head norm-max select. Surprise threshold: merged recall within ~10 pp of sequential on *disjoint* keys, and graceful (not catastrophic) degradation with 25% key overlap. Then repeat with two halves of a real source file.

**Prior art.** [State Soup](https://arxiv.org/abs/2406.08423) (closest; validates the primitive, doesn't do delta rule / repos / search); [Cartridges](https://arxiv.org/abs/2506.06266) (composable, but gradient-trained); [LASP](https://arxiv.org/html/2404.02882v3) (gives you the exact math of what naive merging discards). The evolutionary-crossover use appears novel.

---

## 2. Amortized sleep — a meta-learned consolidation operator C(state) → LoRA delta

**Idea.** The radical version of LoRA-sleep: don't run gradient descent at consolidation time at all. Meta-train a hypernetwork that reads the per-layer DeltaNet states after context ingestion and *directly emits* low-rank weight deltas, with the meta-objective being context distillation: `model + C(state)` on an *empty* context must match `model` with the full context. Ingraining becomes a single forward pass — sleep in milliseconds.

**Mechanism.** The state is already the compressed sufficient statistic of the context (that's the whole Widrow-Hoff point), and — the underexploited symmetry — the state and the slow weights are objects of the *same type*: linear maps on the residual stream. A fast-weight matrix at layer ℓ and a LoRA on layer ℓ's k/v projections live in nearly the same space, so the consolidation map might be close to linear, i.e. learnable from few meta-tasks. This is the piece that would turn the meta-TTT bet from "session-scoped adaptation" into "permanent one-shot learning," the actual differentiator in NORTH_STAR.

**Why it probably fails.** Context distillation needs many gradient steps for a reason: the context→weights map is nonlinear and layer-entangled. Text-to-LoRA is lossy even when meta-trained on hundreds of adapters at 1–8B scale; you have neither the task diversity nor the token budget to meta-train a general C, so it will overfit to the meta-training task family (your recurring failure mode: mechanisms that ace their own probe). And "features bolted on stay inert" applies to the hypernetwork itself.

**Cheapest experiment (<2 days).** Single-layer, single-task probe: on multibind recall, freeze everything and train a *linear* map from layer-ℓ final state to a rank-4 LoRA on that layer, objective = context-free logp of the bound values. Surprise threshold: >50% context-free recall on *held-out* bindings (i.e., the map generalizes across content, not just tasks). If linear fails but a 2-layer MLP works, still interesting; if nothing works at one layer on one task, kill it.

**Prior art.** [Text-to-LoRA](https://sakana.ai/text-to-lora/) (text→LoRA, not state→LoRA); [Cartridges/self-study](https://arxiv.org/abs/2506.06266) and ["Do Language Models Need Sleep?"](https://arxiv.org/abs/2605.26099) (both gradient/recurrence-based, not amortized); ["Still: Amortized KV Cache Compaction in a Single Forward Pass"](https://arxiv.org/pdf/2606.07878) (closest in spirit, transformer KV). State→slow-weights amortization for a linear RNN appears open.

---

## 3. The interpreter as adversarial sparring partner — semantic-change self-play grounded by execution

**Idea.** Manufacture unbounded dense supervision from the corpus you already have: a mutator (can be a dumb operator library + the model at temperature) edits real functions; the *interpreter* labels each edit semantics-preserving vs semantics-changing by running both on fuzzed inputs; the model is trained ELECTRA-style to detect and *localize* semantic change without executing — i.e., by exercising its validated latent-execution skill on natural code. Escalate difficulty adversarially: keep the mutations the discriminator gets wrong.

**Mechanism.** This attacks the binding constraint directly. Token poverty means you can't buy more code; but you can buy *labels per token* with CPU cycles. Discriminative objectives are famously ~4× more sample-efficient than generative ones (ELECTRA), and semantic-equivalence discrimination is the one objective that *cannot* be solved by surface statistics — it forces the causal/semantic representation your 287M base demonstrably lacks. It's also the first task that puts the latent-interpreter finding on a path to natural code rather than synthetic traces.

**Why it probably fails.** The ELECTRA lesson cuts both ways: discriminative pretraining transfers poorly to *generation*. The mutation distribution is narrow, so the model may learn mutation-signatures, not semantics (your dep-distance forensics showed exactly this pattern of probe-local wins). Input fuzzing for arbitrary corpus functions is messy (crashes, non-determinism, coverage gaps), and "equivalent on 20 fuzzed inputs" ≠ equivalent. Realistic outcome: CRUXEval-O moves, HumanEval CE doesn't.

**Cheapest experiment (<2 days).** 50k mutation pairs from corpus functions (AST-level operators + trivial fuzzing, skip anything that doesn't run in 100ms), co-train a discriminator head + a small LM-loss weight on the wide-10L base for a day, then eval the things that would indicate *transfer*: CRUXEval-O, HumanEval-solution CE, and the dep-distance-stratified CE. Surprise = any movement on the stratified CE (a generation-side metric) from a discrimination-side objective.

**Prior art.** ["Semantic Equivalence Self-Play with Formal Verification"](https://arxiv.org/html/2604.17010) (LLM-scale, verifier-based — validates the loop, doesn't do it at pretrain scale on a tiny model with interpreter grounding); [semantic-preserving mutation operators](https://arxiv.org/abs/2503.23448); ELECTRA. The pretrain-scale, interpreter-labeled, latent-execution-tied version is open.

---

## 4. Latent microcode — accept the ~6-hop unit; define a calling convention and compose

**Idea.** Stop trying to extend latent depth (your own grid says heterogeneous composition collapses regardless of R). Instead: a latent "subroutine" runs ≤6 hops, then a learned **commit op** writes its result into the DeltaNet state/WM as a named binding, resets, and the next subroutine reads it back. Depth-12 reasoning = 3 chained 4-hop units linked through memory. The scaffold (or eventually the gate) sequences units — a tiny ISA where the ~6-hop latent horizon is the instruction, and memory is the register file.

**Mechanism.** Your evidence says: homogeneous latent iteration works; composition through *token space* works; direct deep latent chaining fails. The commit op is the minimal token-space-like bottleneck — a discrete *event* with full-bandwidth *content* — converting "reasoning = deep recurrence" (which one trunk can't do) into "reasoning = short program over shallow recurrences" (which is exactly what your exec-trace latent program needs to break the 6-hop wall). The per-hop interpreter supervision you already generate gives the commit targets for free.

**Why it probably fails.** Two of your own findings argue against it. (1) The heterogeneous-composition collapse was diagnosed as *op-selection* failure, not depth — checkpointing intermediate results doesn't fix picking the wrong op per step. (2) Commit = a state *write* at a think position, the exact corruption channel you closed with state-readonly β=0; reopening it risks the "thinking corrupts recall" regression, and errors compound across units with no verifier between them.

**Cheapest experiment (<2 days, existing harness).** Pointer-chase K=12 (a known latent failure): train with a forced commit every 4 hops, supervised by the per-hop interpreter truth; compare 3×(R=4)+commits vs single R=12 vs R=12 with per-hop supervision. Surprise = the chained version solves K=12 where monolithic fails — that would mean the wall is state-management, not op-selection, and the whole latent program scales by scaffolding.

**Prior art.** Coconut-plus-discrete-checkpoint hybrids are just emerging ([ICLR 2026 Latent & Implicit Thinking workshop](https://openreview.net/pdf/d14277b6518be41dfa71a1cd3224d4de191e2b8a.pdf); [looped transformers bridging latent/explicit](https://arxiv.org/pdf/2606.31779)); none define a memory-linked calling convention on a linear RNN. Mostly open.

---

## 5. Co-execution pretraining — the model's beliefs get written INTO the interpreter (DAgger on program state)

**Idea.** Bidirectional grounding: during exec-trace training, with probability p, decode the model's latent prediction of a variable's value and *substitute it into the live interpreter environment*; execution continues from the model's possibly-wrong belief. Supervision becomes the divergence trajectory — the model trains on the consequences of its own errors, on its own state distribution.

**Mechanism.** This is DAgger/scheduled sampling lifted from token space to *program state* space. Teacher-forced traces (your current N3 setup) have classic exposure bias: one wrong latent hop and the model is off-distribution for every subsequent hop — plausibly the real cause of the ~6-hop horizon. It also generates self-correction data for free (detect drift, recover), which is the skill an agentic coder needs most.

**Why it probably fails.** Scheduled sampling's biased-objective problems are well documented; injecting wrong values crashes or degenerates many programs, so much of the manufactured signal is junk; and the payoff lands on a mechanism (latent execution) that is currently a science probe, not on the inference path — high probability of technical success, low probability of headline movement.

**Cheapest experiment (<1 day).** Purely diagnostic first: on the trained N3 model, measure hop-horizon under *self-generated* intermediates vs teacher-forced (already computable). If the gap is big, fine-tune a few hours with p=0.3 substitution and re-measure. Surprise = horizon extends 6→10 from robustness alone, implying the horizon was exposure bias, not capacity.

**Prior art.** DAgger, scheduled sampling, SCoRe-style self-correction training, Dreamer-style imagination rollouts. Applying it to latent program-state in an LM appears novel but is a small delta.

---

## 6. Population-of-states — K meta-learned initial states + a router (specialists in state space, not weight space)

**Idea.** The "population of tiny specialists" seed, but contrarian about *where* the specialists live: one trunk, K trained initial DeltaNet states (~8 MiB each) specialized on data clusters (recursion-heavy code, string algorithms, pandas, SQL...), and a tiny router that picks/mixes the initial state per problem. Fixed FLOPs, no token-budget split (the fatal flaw of weight-space specialist populations under token poverty), and composable with Idea 1's algebra.

**Mechanism.** For a linear RNN the initial state is a fast-weight *prior* — a pre-loaded associative memory — which is strictly more expressive than a soft prompt (it conditions every layer's recurrence multiplicatively through the delta rule). PEFT-for-SSM work (state-offset tuning) shows state-side adaptation is competitive with weight-side at a fraction of the parameters.

**Why it probably fails.** Gating/decay erodes the initial state over ~2k tokens, so the prior washes out exactly when the problem gets long; prefix-tuning-class methods historically deliver small wins at small scale; and "bolted onto a converged trunk" is your documented inertness failure mode. Expected outcome: 1–2% clustered CE, no headline movement.

**Cheapest experiment (<2 days).** Train 4 initial states on 4 clusters (frozen trunk), eval held-out per-cluster CE with *oracle* routing vs one global tuned state. Only if oracle-routed wins by >2% is a router worth building.

**Prior art.** [State-offset tuning](https://arxiv.org/pdf/2503.03499), [PEFT of SSMs](https://arxiv.org/abs/2410.09016), prefix-tuning, Branch-Train-Merge (weight-space version), State Soup's retrieval mode. The per-domain state-library-with-router combo is lightly explored.

---

# Ranking (P(big win) × size of win)

| # | Idea | P(works) | Size if it works | Notes |
|---|---|---|---|---|
| 1 | **State Algebra** | ~30% | Huge | Parallel O(1) prefill + free cartridges + a search operator; feeds meta-TTT directly; transformers can't copy it |
| 2 | Interpreter adversarial self-play (#3) | ~25% | Large | Only idea that attacks token poverty itself; transfer-to-generation is the risk |
| 3 | Amortized sleep C(state)→LoRA (#2) | ~10% | Huge | The full differentiator if it works; probably needs meta-scale you don't have |
| 4 | Latent microcode (#4) | ~35% | Medium | Likely to pass its probe; latent is currently off the inference path, capping the win |
| 5 | Co-execution DAgger (#5) | ~60% | Small–medium | Near-certain to "work," uncertain to matter |
| 6 | Population-of-states (#6) | ~35% | Small | Cheap, composable, but prefix-tuning-class ceiling |

# The one I'd actually run first: **State Algebra (#1)**

Three reasons. It's the cheapest — under a day on probes that already exist (multibind recall + two ingestion contexts + three merge operators). It's the most asymmetric — if disjoint-content state merging works even approximately, you simultaneously unlock parallel repo prefill (a wall-clock moat on top of the memory moat), zero-training repo cartridges, and a population-search operator, all serving the committed north star; and Idea 6 and half of Idea 2 become downstream freebies. And **failure is equally load-bearing**: if delta-rule states turn out to be order-entangled and unmixable, that's a hard constraint the entire meta-TTT/repo-ingestion program needs to know about *now* — it says the state must always be built sequentially, which changes the design of everything from sleep-consolidation to the agentic scoreboard. No other idea on this list is informative in both directions at that price.

Sources: [State Soup (arXiv 2406.08423)](https://arxiv.org/abs/2406.08423), [Cartridges (arXiv 2506.06266)](https://arxiv.org/abs/2506.06266), [Hazy Research Cartridges blog](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges), [Text-to-LoRA (Sakana)](https://sakana.ai/text-to-lora/), [Semantic Equivalence Self-Play with Formal Verification (arXiv 2604.17010)](https://arxiv.org/html/2604.17010), [Semantic-Preserving Transformations as Mutation Operators (arXiv 2503.23448)](https://arxiv.org/abs/2503.23448), [Do Language Models Need Sleep? (arXiv 2605.26099)](https://arxiv.org/abs/2605.26099), [Still: Amortized KV Cache Compaction (arXiv 2606.07878)](https://arxiv.org/pdf/2606.07878), [LASP (arXiv 2404.02882)](https://arxiv.org/html/2404.02882v3), [State-offset Tuning (arXiv 2503.03499)](https://arxiv.org/pdf/2503.03499), [PEFT of State Space Models (arXiv 2410.09016)](https://arxiv.org/abs/2410.09016), [Bridging Latent and Explicit Reasoning with Looped Transformers (arXiv 2606.31779)](https://arxiv.org/pdf/2606.31779), [ICLR 2026 Latent & Implicit Thinking workshop paper](https://openreview.net/pdf/d14277b6518be41dfa71a1cd3224d4de191e2b8a.pdf).
