# 10 — Inference-Time Architectural Innovations

**Frame:** the (SO(n) + DeltaNet) hybrid lost on PPL despite winning the parity kill-gate (+32.6 pp end-tok at T=64). The premise here: maybe the cell is good enough. The lever we're under-using is the *inference policy* — turning train-time PPL into test-time correctness via dynamic depth, parallel state branches, and verifier-grounded search. For code in particular, pass@k scales hard with sample budget and execution feedback (S\* shows +8–18 pp Qwen2.5-Coder gains on top of the *same model*) — usually a bigger lever than the cell.

A second framing worth keeping in mind: linear-RNN states are *cheap to fork and merge* relative to KV caches (one matrix vs. T·d_k cache), so several inference-time tricks that are awkward for transformers are nearly free for us.

---

## Top 5 concrete ideas

### Idea 1: S\* / iterative test-time scaling for code, with state-forking BoN
- **What it is:** S\* (Li et al., 2025) is a 2-stage hybrid scaling recipe for code: (1) generate N parallel samples with iterative debugging — each sample is *executed against public tests*, error trace conditioning seeds the next round; (2) selection via paired adversarial input generation + execution. Since our cell carries a *single matrix state* per branch (not a full KV cache), forking N branches at any prefix point is one tensor copy, not O(T·d) memory — much cheaper than for transformers. Combine with "RadixAttention-style" prefix sharing on the SSM state (the SGLang MambaRadixCache work shows the API is real).
- **How it changes the curve:** S\* itself reports Qwen2.5-Coder gains of +8.4–18.2 pp pass@1, +9.9 pp over majority voting, +15.6 pp over self-debug. Compounded with a weaker-but-cheaper cell, you can spend the saved FLOPs on more N. With a 135 M / DeltaNet-class cell that's ~2× cheaper per token than a transformer of equal PPL, you afford 2× the N at the same wall-clock — which is the right side of the pass@k coverage curve.
- **Compatibility:** native — every linear-RNN branch is `(state_matrix, last_token)`, trivially forkable; merge is just discarding losers. Need test execution harness (sandbox subprocess + timeout) and a small differential-input generator (can prompt the model itself).
- **Key papers:** Li et al., *S\*: Test Time Scaling for Code Generation* (arXiv 2502.14382, EMNLP 2025); Snell et al., *Scaling LLM Test-Time Compute Optimally...* (arXiv 2408.03314, ICLR 2025 oral); SGLang/PyTorch, *Hybrid Models Meet SGLang* (2025).
- **Implementation cost:** **M** — execution sandbox + state-fork API is ~1k LOC; no model changes.
- **Code-LLM relevance:** **high** — direct on HumanEval/MBPP. This is the lever I'd bet on first if the question is "how do we beat a stronger cell on pass@1?"
- **Risk:** test-driven loops only help when public tests exist or differential generation is reliable; raw PPL on "the-stack-v2-python" doesn't move from this. Keep our existing cell-level kill-gates intact.

### Idea 2: Mixture-of-Recursions (MoR) over a single recurrent block — our cell as the recurred unit
- **What it is:** MoR (Bae et al., NeurIPS 2025; arXiv 2507.10524) replaces a deep stack with one *shared* block recursed K times, with a per-token router choosing depth K_t ∈ {1..K_max}. KV-share variant reuses keys from the first recursion. Reported new Pareto frontier 135 M – 1.7 B on PPL/few-shot vs vanilla and prior recursive baselines — *exactly our scale*. The natural marriage with linear-RNN: replace the recurred *block* with a Gated-DeltaNet/DeltaProduct cell, route per-token over depth. State-space recurrences carry small fixed-size state, so MoR's KV-cache memory advantage compounds.
- **How it changes the curve:** at fixed train FLOPs you get lower PPL and higher throughput — the rare both-axes win. Expressivity-wise, recursing the same cell K times with token-dependent K is equivalent to giving "hard" tokens (e.g., end-of-function, EOL after `def`) more state-update steps, which is the standard parity-tracking story without growing layer count.
- **Compatibility:** **high** — fla layers compose into a recurrence loop trivially; the router is one MLP. Caveat: their KV-cache-sharing trick is transformer-specific; for us, the analog is "share post-cell state between recursion steps" which is also natural.
- **Key papers:** Bae et al., *Mixture-of-Recursions* (arXiv 2507.10524, NeurIPS 2025); Raposo et al., *Mixture-of-Depths* (arXiv 2404.02258, 2024); Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (arXiv 2502.05171, 2025) — the 3.5 B Huginn model improves on math/coding with extra recurrence steps.
- **Implementation cost:** **M** — one-block model + router + recursion loop. Auxiliary load-balance loss like MoE.
- **Code-LLM relevance:** **medium-high** — code has a power-law distribution of "hard" positions (boundary tokens, identifier-bind sites, control-flow joins); routing them to deeper compute is the right inductive bias. Direct evidence missing for code specifically.
- **Risk:** training instability of the router at small scale (135 M); you lose some of the parity-kill-gate guarantee if the router learns to short-circuit hard positions early. Also — does MoR PPL reproduce when the block is a linear RNN vs. a transformer? Open.

### Idea 3: Self-speculative decoding with a Mamba-family drafter (target = our hybrid)
- **What it is:** Two recent results make this almost a free win: (a) *Mamba Drafters for Speculative Decoding* (EMNLP findings 2025) shows a small Mamba drafter calibrates *better* with a target than a transformer drafter of equal size (lower ECE, higher acceptance); (b) EAGLE-3 (Li et al., NeurIPS 2025) shows draft heads using *fused multi-level features* of the target give 2–6× speedup. Our hybrid is already a linear-RNN — its own hidden state is a fine drafter conditioning signal. So: a tiny ReDrafter-style RNN head (Apple, arXiv 2403.09919) consuming our DeltaNet state, trained by KD from the hybrid, with EAGLE-3-style training-time-test simulation.
- **How it changes the curve:** moves the cost/quality curve along the *latency* axis only — same PPL, ~2–4× faster decode. That FLOPs budget can then be spent on Idea 1 (more BoN samples) — the two stack multiplicatively.
- **Compatibility:** **high** — the drafter is a 1-layer linear RNN over our hidden states; verification is standard rejection sampling. The non-trivial bit is *state rollback* on rejected drafts: with a single matrix state per layer per sequence, rollback is just keeping a pre-draft snapshot (cheap), unlike transformer KV which requires position-level rollback.
- **Key papers:** Wang et al., *Mamba Drafters for Speculative Decoding* (EMNLP findings 2025); Li et al., *EAGLE-3: Scaling up Inference Acceleration via Training-Time Test* (arXiv 2503.01840, NeurIPS 2025); Cai et al., *Medusa* (arXiv 2401.10774, ICML 2024); Bhendawade et al., *Recurrent Drafter / ReDrafter* (Apple, 2024).
- **Implementation cost:** **M** — tiny drafter (1 fla layer + linear head) + KD training pass + verifier loop. EAGLE-3 codebase is open.
- **Code-LLM relevance:** **medium** — pure speed. But code generation is highly autoregressive and decode-bound at long contexts; for completion-style use (Copilot-style) latency *is* the product.
- **Risk:** tooling — vLLM's SSM support doesn't yet handle prefix caching cleanly (open RFC #17140); we may have to live in fla + a custom serving loop. Also: at 135 M scale a separate drafter is overkill; payoff is at 1 B+.

### Idea 4: Pause/thinking tokens at parse-boundaries, made cheap by linear-state forking
- **What it is:** Append k learnable `<pause>` tokens at training time; at inference, *insert* extra pause tokens at syntactically meaningful boundaries (after `def`, before `return`, at indent-changes). Theoretical result: pause tokens *strictly increase* expressivity of constant-depth bounded-precision transformers from a strict subset of AC⁰ to all of AC⁰ (Pfau et al., arXiv 2505.21024, ICLR 2025). For linear RNNs, the analog claim is unproved but the mechanics are the same — extra steps with no output token = more state mutations. A code-aware twist: trigger pause-token bursts at AST boundaries detected on the fly with a 1-token-ahead grammar (XGrammar-style, 100× faster than prior libs), so we *only* spend pause tokens where syntactic complexity is highest.
- **How it changes the curve:** flat-cost per generated token (decode latency unchanged for normal positions), localized compute spikes at boundaries. Combined with constrained decoding (Park et al. 2024 quotient-grammar FIM; XGrammar 2024), you get correctness-by-construction *and* extra reasoning compute exactly where ambiguity lives.
- **Compatibility:** **high** — a special-token vocab addition + boundary detection. Linear-RNN states absorb pause tokens as ordinary updates, no architectural change. Quiet-STaR's reinforcement of "rationale" tokens (arXiv 2403.09629) is a more invasive variant, but Fast Quiet-STaR (arXiv 2505.17746) showed +5.7–9 pp without inference latency cost via curriculum.
- **Key papers:** Pfau et al., *Pause Tokens Strictly Increase Expressivity...* (arXiv 2505.21024, ICLR 2025); Goyal et al., *Think Before You Speak: Pause Tokens* (ICLR 2024, arXiv 2310.02226); Zelikman et al., *Quiet-STaR* (arXiv 2403.09629, 2024); Hou et al., *Fast Quiet-STaR* (arXiv 2505.17746, 2025); XGrammar (2024).
- **Implementation cost:** **S-M** — special-token + curriculum + cheap parser; constrained decoding integration is M.
- **Code-LLM relevance:** **high** — code's syntactic structure gives free, well-defined boundaries; pause-at-`def` is conceptually principled. Constrained decoding alone reportedly removes most syntax errors (Park et al. 2024).
- **Risk:** training has to *teach* the pause tokens to be useful — naive insertion at inference without curriculum gives nothing. Also: filler-tokens-help results are sensitive to pretraining recipe; some replications (Wei et al. blog) only see effects in specific regimes.

### Idea 5: Tree search over linear-RNN states with execution-grounded pruning (RethinkMCTS-lite for SSMs)
- **What it is:** Beam/MCTS where each node holds a `(prefix_tokens, recurrent_state)` snapshot; child expansion forks the state, runs k tokens, and is *evaluated by partial program execution* (compile/parse check, optional runtime on stub inputs). RethinkMCTS (Li et al., 2024, arXiv 2409.09584) gets GPT-3.5-turbo from 70.1→89.0 pass@1 on HumanEval. With linear-RNN, every node clones in O(d²) memory not O(T·d), so we can run *much wider beams* than transformer baselines at the same memory budget. Combined with parser-level early rejection (the FIM grammar quotient work), you prune subtrees that can't possibly compile.
- **How it changes the curve:** big pass@k jump at high inference budget; the steep tail of pass@1 with N samples flattens into a near-pass@N because compile-check pruning ~3-4× the effective N for syntactic correctness alone, leaving the budget for semantic exploration.
- **Compatibility:** **high** — linear-RNN's "fork the state" primitive is what makes this affordable. The parser-quotient is language-agnostic for any CFG.
- **Key papers:** Li et al., *RethinkMCTS* (arXiv 2409.09584, EMNLP 2025); CodeTree (arXiv 2411.04329, 2024); Park et al., *Constrained Decoding for FIM via Grammar Quotients* (arXiv 2402.17988, 2024); SFS / Foresting (ICLR 2025).
- **Implementation cost:** **M-L** — state-fork API (S), parser + grammar quotient (M), MCTS loop with execution feedback (M). Reasonable in 1–2 weeks.
- **Code-LLM relevance:** **high** — designed for it.
- **Risk:** MCTS is gnarly to debug; execution feedback needs sandboxing; code with side effects (file I/O, network) won't run safely. Limit to pure-function HumanEval/MBPP-style first.

---

## Recommendation

**Bet: Idea 1 (S\* with state-forking BoN) first, Idea 2 (MoR-on-our-cell) second.**

Idea 1 is a 1-week ship, model-agnostic, hits HumanEval/MBPP directly, and exploits the fact that our linear-RNN state is *cheap to fork* — a real architectural advantage transformers don't have. It's also orthogonal to every cell-level experiment in `01_ttt_cells.md`: any cell improvement we land later just plugs in. Quickest path to a real, defensible coding-LLM number.

Idea 2 is the train-time, novelty-bearing bet. MoR's reported gains *at our exact scale* are the most directly relevant result in this whole sweep, and combining it with a Gated-DeltaProduct cell (per `01_ttt_cells.md`) gives a clean, mechanically-justified architecture: one shared linear-RNN block, recursed token-adaptively, with a router. If both work, they stack: MoR-cell trained once, then S\*-style inference.

Skip Idea 3 (drafter) until target is ≥1 B; pure speedup, low novelty, current vLLM/SSM tooling is rough. Idea 4 is cheap and theoretically clean — try it as a side-experiment with constrained decoding (Idea 4-lite: just XGrammar, no thinking tokens, gives us free syntax correctness today). Idea 5 is a longer-tail bet — only worth the engineering if we're chasing pass@k headlines on HumanEval.

---

## Open questions

1. Does our parity kill-gate result *transfer* through MoR's router? If the router shortcuts past hard-state-tracking positions, the +32.6 pp at T=64 may collapse. Need ablation: force K=K_max for the first M tokens and see if parity holds.
2. State-fork memory cost in fla: how big is the recurrent state matrix per layer at our 135 M config? If it's e.g. 64×64×N_layers, we can fork hundreds of branches in 1 GB; this is the key feasibility question for Idea 1 + 5.
3. For Idea 1: does iterative-debugging gain replicate when the underlying model is sub-1 B and never saw error traces in pretrain? S\* used Qwen2.5-Coder; at 135 M / TinyStories+Python the trace-conditioning may not work — need a small SFT pass on `(buggy_code, trace) → fix` pairs first.
4. Is there a clean pause-token training signal at 135 M? Quiet-STaR needed teacher forcing on rationales; Fast Quiet-STaR removes it — but small-model variants are unevaluated. Cheap experiment: 200-step continued pretrain w/ k=4 pause tokens and see if PPL on Python moves.
5. EAGLE-3 / drafter integration with fla: if we go that route, does fla expose hidden states cleanly enough to feed a drafter, or do we need to monkey-patch? Probably the latter.
6. Constrained decoding (XGrammar) — does Python's grammar admit fast quotienting for FIM? Unknown for our use case. Should be a 2-day spike.
