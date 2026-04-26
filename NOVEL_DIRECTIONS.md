# Genuinely Novel Directions (not frontier-tracking)

**Date:** 2026-04-26

The first brainstorm produced ten "swap in 2025 paper X" recommendations. The user's pushback: that's not novel, it's iteration.

This document is a first-principles rewrite. Question: **what direction would a small lab attempt that isn't already in the queue at Mistral / DeepMind / Together / fla-org?**

The frontier of 2025-2026 has densely tiled:
- Linear-RNN cells of every algebraic flavor (DeltaNet, GDN, DeltaProduct, PaTH, PD-SSM, Heisenberg)
- Linear-RNN + softmax hybrids (Samba, Jamba, Hymba, Granite-4)
- MoE / routing for cells (BlackMamba, Linear-MoE, MoR)
- Distillation (LoLCATs, MOHAWK, RADLADS, HALO)
- Long context tricks (Tiled FLA, Multi-Axis Checkpointing, Video-Ma2mba)

What they all share: **the state is a fixed-shape tensor, the recurrence is one update per token, and "code" is treated as just a different distribution.** Almost every contribution is "compose existing primitives differently."

Genuinely fresh directions break at least one of those assumptions.

---

## 5 directions that aren't a re-arrangement of existing primitives

### A. Symbol-grounded state — sequence-aware identifier tables, not vectors

**Frontier blind spot:** every published linear-RNN packs symbol-binding into a continuous tensor. Memorizing Transformer / kNN-cache externalize but treat memory as opaque text. PKM/Memory Layers index by content but don't *parse identity*. **No published autoregressive LM maintains an explicit, learned symbol table whose keys are tokens-as-identities and values are vectors-as-bindings, updated as the sequence streams.**

**Sketch:**
- State has two components: dense `h_t ∈ ℝ^d` (as today) plus sparse table `S_t : ID → ℝ^d` keyed by token-id
- On each token: cell decides via gate whether this token is a *binding event* (LHS of `=`, `def name`, `class name`, `import name`); if yes, write `S_t[id_t] ← f(h_t)`
- Reads: subsequent tokens with the same id retrieve `S_t[id_t]` and concat into the recurrence
- The table is sparse and symbolic — no softmax over all keys; equality test on token ids
- **Parallel-scan story:** the table update is a left-fold; equivalent to a sparse semilattice (last-write-wins per key) which IS associative ⇒ parallel-scan compatible
- **Why for code:** matches code's most distinctive property — variables and functions have *names*, not just embeddings. Pointer-chasing across scopes becomes O(1) lookup, not O(T) recall
- **Why nobody's done this:** the obvious version uses argmax-on-token-id, which is non-differentiable in the routing decision. The parallel-scan-friendly trick (last-write-wins with soft gating) sidesteps that

**Risks:** (1) tokenizer fragmentation breaks identity (BPE splits long names); needs a stable identifier abstraction. (2) graceful degradation on non-code is unclear.

**Novelty score:** 9/10. Closest published work is Memorizing Transformer + PKM, which differ in important ways.

---

### B. Multi-pass prefill — parallel scans with different reading modes, fused at output

**Frontier blind spot:** standard scan is single-pass. Multi-pass exists in encoder-only (BERT runs L layers), but **no autoregressive linear-RNN runs the same input through K parallel scans with K different parameter sets, then fuses.** Multi-head attention varies the projection but uses a single scan; this varies the *scan* itself.

**Sketch:**
- K parallel scans `S_1, ..., S_K`, each with its own cell parameters. K ∈ {2, 3, 4}
- Each scan can specialize: short-decay `λ_k = 0.7` for local n-grams, long-decay `λ_k = 0.999` for document state, mid-decay for paragraph
- Or: scan_1 is rotation, scan_2 is delta-rule, scan_3 is symbol-grounded (idea A) — *one layer*, K cells running in parallel, learned mixture at output
- **Difference from layer-level hybrid (our failed attempt):** information from all K reading modes is available *at every token*, not just at every other layer. The hybrid had to spend an ortho layer's compute to update the rotation state, lose it, recover it after a delta layer
- **Cost:** K× the compute of one scan, but trivially parallel on a single GPU (K independent scans)
- **Why for code:** code has at least 3 distinct timescales — bracket-local (1-30 toks), function-local (10-300 toks), file-global (300-10k toks). One cell with one decay can't do all three; K cells with K decays + learned mixing could

**Novelty score:** 8/10. ms-Mamba did multi-scale within a single state; this runs *separate cells* in parallel. ResNet-style multi-branch isn't applied to RNN cells in the literature.

---

### C. Event-driven irregular-time cells

**Frontier blind spot:** every published linear-RNN updates state on *every* token. Mixture-of-Depths skips compute via routing, but the cell still "runs" on every token. **No published cell makes the recurrence step explicitly conditional on syntactic event tokens, with cheap-update vs. expensive-update modes that differ by orders of magnitude.**

**Sketch:**
- Tag tokens by syntactic role (cheap, ~1% overhead): `WHITESPACE`, `IDENT`, `KEYWORD`, `BRACKET_OPEN`, `BRACKET_CLOSE`, `BINDING`
- Cell has two recurrences:
  - Cheap recurrence on every token: low-rank state update, near-free
  - Expensive recurrence on event tokens only: full delta-rule or rotation update
- The "event rate" in real Python is ~15-25%, so **expensive compute drops 4-7×** at minimal capacity loss
- **Different from MoD:** MoD routes *whole tokens* through deep stacks; this routes *cell update intensity* per token. Cheaper because the routing is rule-based not learned
- **Why for code:** Python's syntactic structure is real signal we throw away. A token like `:` after `def foo():` is genuinely more important to the model's state than the 8 spaces of indentation that follow

**Novelty score:** 7/10. ACT-style adaptive computation is the obvious cousin, but tied to *cell update granularity* and to *syntactic events* is unexplored.

---

### D. Tree-scan over the parse tree, not the token stream

**Frontier blind spot:** code has free hierarchical structure (the AST), and **every published autoregressive code LM throws it away at training time.** Tree-LSTM (2015) used trees for classification, not autoregressive generation. AST-T5 / AST-FIM use AST as a corruption mask for masked LM, not as the backbone of the recurrence.

**Sketch:**
- Streaming parser emits AST nodes as tokens arrive; left-children are sealed before parents
- Tree-scan: each AST node aggregates state from its children via an associative operator (any monoid we've already implemented works); state at the root contains the program's entire prefix
- Token-level prediction reads from current node + path-to-root via a fixed-depth chain
- **Parallel-scan story:** standard parallel tree reduction, well-understood from PRAM literature
- **Why for code:** matches code's actual structure. Function calls *literally are* tree compositions. `f(g(x))` should compose semantically, not be flattened into a token stream
- **Why nobody's done this for autoregressive:** the obvious blocker is "parsing is unstable on partial input." But Python's grammar is LL(1)-ish on a per-line basis; we can recover gracefully

**Risks:** (1) needs streaming parser at training time — annoying but cheap. (2) breaks for malformed code — but humans write malformed code constantly during edits, so the model needs a fallback. (3) tokenizer alignment with AST nodes is fiddly.

**Novelty score:** 8/10. Tree-LSTM cousin but autoregressive + parallel-scan kernel + Python-specific is fresh.

---

### E. Verifier-in-the-loop pretraining

**Frontier blind spot:** verifier feedback is **almost universally a post-training signal** (RLHF, on-policy distillation). For code, the verifier is *cheap and deterministic* — `python -m py_compile` is microseconds, AST parse is microseconds, even running short snippets is fast. **Nobody has used compile/parse signal as a *pretraining* aux objective at scale.** Reason: belief that PT-scale isn't compatible with such expensive supervision. But for code, parse/compile is cheap.

**Sketch:**
- During pretraining, after every K tokens, sample 2-5 candidate continuations from the model, compile them (fast), and use compile-success as a per-step aux signal
- Aux loss: KL between the policy's distribution and the "compile-success-weighted" distribution
- Or simpler: at each token, predict "will the next 32 tokens parse?" as a binary aux head
- The model gets gradient signal that surface-form fluent code that doesn't compile is *worse* than slightly-disfluent code that does
- **Why this matters:** PPL is a terrible proxy for code. Compile-pass is a closer proxy for "is this real code." Training on it directly closes the PPL-vs-HumanEval gap that Agent 07 flagged
- **Why nobody's done this:** PT pipelines are batch-jobs; injecting an external verifier looks engineering-heavy. But on 2× RTX 5090 with batch 8 it's trivial — fork a CPU thread per batch, use stale signal if it lags

**Novelty score:** 7/10. Closest cousin is "process reward models" in math RL; nobody has put this on a linear-RNN backbone or used it during pretraining.

---

## Honest comparative ranking

| # | Direction | Novelty | Eng cost | Code-LLM relevance | Risk-adjusted EV |
|---|-----------|---------|----------|---------------------|------------------|
| A | Symbol-grounded state | 9/10 | M (1-2 weeks; new sparse-state kernel) | very high | **highest** |
| B | Multi-pass prefill | 8/10 | S (3-5 days; reuses existing scans) | medium | **highest** |
| D | Tree-scan over AST | 8/10 | L (3-4 weeks; parser + tree kernel) | very high | medium-high |
| C | Event-driven cells | 7/10 | S (3-5 days; rule-based gate on existing fla cells) | medium-high | high |
| E | Verifier-in-loop PT | 7/10 | M (1-2 weeks; pipeline change) | high | medium-high |

A and B are both shippable in 1-2 weeks, both genuinely novel relative to the 2025 frontier, both code-aware in fundamentally different ways, both compose with whatever cell we settle on.

---

## Recommendation: ship A + B together

A symbol-grounded state inside a multi-pass scan framework is a *single coherent system*:
- Pass 1: standard delta-rule (handles general LM)
- Pass 2: rotation / heisenberg (handles state-tracking — keeps our existing kill-gate alive)
- Pass 3: symbol-grounded sparse table (handles code's pointer-chasing)
- Output: learned mixture per token

**Cell shape:** 3 scans in parallel, each ~half the head_dim of a standard cell, fused at output. Compute is comparable to a single scan at full head_dim. The novelty is the *combination*: nobody has run a symbol-table-as-state scan, and nobody has run multi-cell multi-pass linear-RNN as one layer.

**First experiment (1 week):**
- Implement A (symbol-grounded scan) standalone in a Triton kernel — last-write-wins update on a (key→value) sparse table; reads via id-equality lookup
- Eval on a code-pointer-chasing synthetic: "given a sequence of `x = N; ...; x = M; ...; print(x)`, predict the latest binding"
- Compare to DeltaNet, hybrid, Memorizing Transformer baseline
- **Decision gate:** if A wins on the synthetic, build the multi-pass framework. If A loses, drop and revisit B/C/D/E.

**Second experiment (week 2):**
- If A wins: stack A as one of K passes in B's multi-pass scan, train at 135M / 5K steps
- Eval: TinyStories, Python code, and the synthetic from week 1

**Why this isn't on the public frontier yet:** symbol-grounded scan needs both (1) a parallel-scan-friendly sparse update and (2) the realization that token-id is the right key for code. The first is a small kernel; the second is a viewpoint shift. Frontier labs are largely pursuing dense-tensor scaling laws, not domain-specific structure.
