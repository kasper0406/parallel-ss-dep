# 01 — Test-Time Training and Fast-Weight Cells

**Frame:** the hybrid (SO(n) + DeltaNet) lost on PPL despite winning on parity. The TTT family is the
natural next-rung *fast-weight* abstraction: instead of hand-designing a linear/orthogonal recurrence,
the cell is a small inner network whose weights are updated by an inner gradient step per token (or
per chunk). DeltaNet is the degenerate case (1-layer linear inner model + 1 SGD step on a squared
recall loss). Everything below is "DeltaNet on stronger inner geometry."

---

## Top 5 concrete ideas

### Idea 1: TTT-MLP layer (LaCT-flavoured) replacing both ortho-rotation and DeltaNet
- **What it is:** Replace each (rotation, DeltaNet) pair with a single TTT layer whose hidden state is
  a 2-layer MLP (Sun et al. 2024). At each chunk, run k inner SGD/Muon steps on a self-supervised
  reconstruction loss `||W_inner(k_t) − v_t||²`. Use LaCT's "large chunk" trick (chunk = 1k–8k tokens
  instead of 16) to push GPU utilization from ~5 % to 50–70 % without writing a custom kernel.
- **Mechanistic story:** an MLP inner state is a *nonlinear* fast weight, so it can store keys whose
  retrieval is non-linear-separable — exactly the regime where DeltaNet (linear inner) is tight under
  Zoology recall lower bounds. Online gradient descent across a sequence is itself non-TC⁰ (the loss
  surface depends on cumulative state), so the parity / state-tracking argument we built rotations
  for *plausibly transfers* to TTT-MLP without a separate ortho layer. Code-relevant: function-def-
  then-use is a 2-step in-context regression (define `f`, recall its semantics later) — a literal
  online-learning task.
- **Key papers:** Sun et al., *Learning to (Learn at Test Time)* (2024, arXiv 2407.04620);
  Zhang et al., *Test-Time Training Done Right / LaCT* (2025, arXiv 2505.23884);
  Li et al., *TNT: Chunkwise Training for Test-Time Memorization* (2025, arXiv 2511.07343).
- **Implementation cost:** **M** — fla has TTT-Linear; TTT-MLP exists in PyTorch ref impl
  (`test-time-training/ttt-lm-pytorch`); LaCT is "a few dozen lines of pure PyTorch." No custom
  Triton needed at first. XL only if we want decode-time fused kernels later.
- **Code-LLM relevance:** **high** — code is the canonical "learn from a definition, apply later"
  workload. TTT showed the strongest *long-context* PPL gap vs Mamba; coding files are exactly that
  regime (1k–32k tokens, with intra-file definitions).
- **Risk:** TTT-MLP famously has nasty memory I/O — chunk size 16 is unusable at 5090 throughput.
  Without LaCT-style chunking, training will be *slower* than our already-1.27× DeltaNet hybrid.
  Also, TTT-MLP's inner Muon optimizer is finicky to stabilize at small (135 M) scale — most
  reported wins are 350 M+.

### Idea 2: Gated DeltaProduct as the linear leg, replacing DeltaNet only
- **What it is:** Drop-in replacement for our DeltaNet layer with **Gated DeltaProduct** (Siems et al.
  Feb 2025): the recurrence is a product of `n_h` Householder reflections per token plus Mamba2-style
  scalar decay. With `n_h ≥ 2` and 3 layers, it provably solves any group-word problem (parity, mod-p,
  S₅) — i.e. it eats the parity benchmark *without* a separate rotation layer. Already in fla.
- **Mechanistic story:** Householder products = orthogonal updates with negative eigenvalues, which
  is exactly the structure our SO(n) layer provides — except now it's *fused* into the same recurrence
  as the rank-1 delta erase. The proven incompatibility we hit between SO(n) and DeltaNet was within
  a *single linear* state; DeltaProduct breaks the impasse by going past linear via a fixed `n_h`
  number of inner steps. Same TC⁰ escape, none of the layer-alternation cost.
- **Key papers:** Siems et al., *DeltaProduct* (2025, arXiv 2502.10297); Yang et al., *Gated Delta
  Networks* (ICLR 2025, arXiv 2412.06464); Peng et al., *RWKV-7 "Goose"* (Mar 2025, arXiv 2503.14456)
  — the same idea independently, with a relaxed value-replacement rule.
- **Implementation cost:** **S** — DeltaProduct kernels are in fla already; just swap the layer and
  pick `n_h ∈ {2, 3, 4}`. RWKV-7 is also a pure swap.
- **Code-LLM relevance:** **medium-high** — eats the synthetic state-tracking gap that motivated our
  hybrid, while keeping single-state simplicity. Whether it beats Gated DeltaNet (already in
  Qwen3-Next) on real code PPL is the open question; ablations exist on language modeling but not
  HumanEval/MBPP at 135 M scale.
- **Risk:** at 135 M / 5 k steps, the parity-style win may *still* not translate to PPL — same trap
  we hit. But at least we delete a full layer family (and the 1.27× wall-clock).

### Idea 3: Titans MAC (Memory-as-Context) hybrid with our existing DeltaNet
- **What it is:** Behrouz et al. Jan 2025 (Google). A 2-layer-MLP "neural memory" trained at test
  time with a *surprise* signal (`∇ℓ` magnitude as the write gate) and weight-decay forgetting,
  combined with sliding-window attention as short-term memory. MAC variant feeds memory output as
  extra context tokens; reportedly beats GPT-4 on BABILong at 760 M scale.
- **Mechanistic story:** the surprise gate is a learned, *data-dependent* per-token write rate — a
  generalization of Mamba2/RWKV-7 gating, but on a *deep* memory. For code, "surprise" = encountering
  a new identifier or import; storing those preferentially in long-term memory is the right inductive
  bias.
- **Key papers:** Behrouz et al., *Titans* (2024, arXiv 2501.00663); follow-up *MIRAS* framework
  (Google, Dec 2025).
- **Implementation cost:** **M-L** — no official open weights/code at our 135 M scale; "Titans
  Revisited" (arXiv 2510.09551) is a lightweight reimpl we can fork. MAC variant requires sliding-
  window attention alongside.
- **Code-LLM relevance:** **medium** — strongest reported gains are at long-context retrieval
  (BABILong, NIAH); code-LLM evals not reported. MAC's hybrid-with-attention story is exactly
  the architectural pattern the field is converging on.
- **Risk:** "Titans Revisited" already pushed back on reproducibility; the surprise metric is
  sensitive. Adding sliding-window attention re-introduces a quadratic (windowed) component —
  spiritually we're now hybrid + attention.

### Idea 4: RWKV-7 "Goose" as a same-family one-cell competitor
- **What it is:** Same delta-rule lineage as DeltaNet but with vector-valued gating, in-context
  learning rate, *and* a relaxed value-replacement rule that lets the recurrence do "copy"
  transitions. Formally proven to recognize all regular languages with constant layers (so >TC⁰)
  while staying parallelizable.
- **Mechanistic story:** identical TC⁰ escape as our hybrid, single-cell, with mature kernels and
  a public 2.9 B model (so we know it scales). Compares head-to-head with Gated DeltaNet in the
  paper — close, slightly different recall/state-tracking trade-offs.
- **Key papers:** Peng et al., *RWKV-7 "Goose"* (Mar 2025, arXiv 2503.14456).
- **Implementation cost:** **S** — RWKV-7 kernels and pretrained checkpoints public; fla mirrors
  the cell.
- **Code-LLM relevance:** **medium** — RWKV-7 has been pretrained on code; community ports exist
  on HumanEval/MBPP. Solid baseline more than novel direction.
- **Risk:** essentially "same horse, slightly different saddle" as Gated DeltaProduct. If we want
  novelty for a writeup, RWKV-7 is mostly a baseline.

### Idea 5: Modern-Hopfield / Energy-Transformer fast-weight head, alternated with DeltaNet
- **What it is:** Replace the SO(n) layer with a single-step modern-Hopfield update — a fixed-point
  energy minimization over a fast-weight associative store. Equivalent to softmax attention over a
  *learned* fast weight matrix that itself is updated online. Long-sequence variants (Hoover et al.
  2023; arXiv 2507.01052, 2025) propose temporal kernels.
- **Mechanistic story:** modern Hopfield is a *non-linear* attention with exponential capacity in
  the key dimension; complements DeltaNet's linear recall. Crucially, Hopfield retrieval is one
  fixed-point step (parallelizable), unlike the per-token unroll we needed for rotations.
- **Key papers:** Ramsauer et al., *Hopfield Networks Is All You Need* (NeurIPS 2020); Hoover et al.,
  *Energy Transformer* (NeurIPS 2023); recent *Long-Sequence Memory with Temporal Kernels* (2025,
  arXiv 2507.01052).
- **Implementation cost:** **L** — no off-the-shelf chunkwise-parallel Hopfield-as-RNN-cell exists;
  need to build it. Probably needs a Triton kernel for the energy step at chunk boundaries.
- **Code-LLM relevance:** **low-medium** — promising on synthetic associative recall; no published
  code-LLM results.
- **Risk:** highest research-risk option, lowest off-the-shelf payoff. Likely scope creep.

---

## Recommendation

**Prioritize Idea 2 (Gated DeltaProduct, n_h = 2 or 3) first, then Idea 1 (TTT-MLP + LaCT).**

Idea 2 is the *highest-EV near-term experiment*: it's an S-cost swap in fla, deletes the
ortho-rotation layer (and its 1.27× wall-clock), and inherits proven TC⁰ escape. We can have a
135 M / 5 k-step PPL number on TinyStories + Python within a day. If DeltaProduct ≥ pure DeltaNet
*and* still passes our parity kill-gate, the whole hybrid story collapses into a single cleaner cell
— which is the win we wanted.

Idea 1 (TTT-MLP) is the *higher-ceiling, longer-bet*. It's the only candidate that genuinely changes
the abstraction: the cell is a programmer, not just a fancier matrix. If we believe code is online
in-context regression, this is the cell that should win HumanEval/MBPP at long context. But TTT-MLP
at sub-350 M has a track record of underwhelming PPL until LaCT-style chunking is in place — so this
is a 1-week experiment, not a 1-day one. Run it second, with LaCT chunk size ≥ 2 k.

Skip Idea 3 (Titans) until someone with a 760 M budget reproduces the BABILong claims. Skip Idea 4
(RWKV-7) as a research direction (use as baseline). Skip Idea 5 unless DeltaProduct *and* TTT-MLP
both fail.

---

## Open questions

1. **Does our parity kill-gate survive the swap to Gated DeltaProduct?** It should by theorem
   (Householder products + 3 layers solve any group-word problem) — but with `n_h = 2` is the
   constant good enough at 135 M? Sanity-check before trusting.
2. **TTT-MLP under LaCT — has anyone done it at 135 M?** Most TTT papers report 350 M+; the
   "Test-Time Training Done Right" paper is at long-video / multimodal scales. We may be the
   first to try the small-scale LM regime on code.
3. **What's TTT-MLP's TC⁰ status, formally?** Online gradient descent on an MLP feels like it
   should escape TC⁰ — but no one has written the proof. If we do, it's a defensible
   theory contribution alongside the empirical run.
4. **Does Muon (the inner optimizer in TTT) play nice with bf16 + sm_120?** RTX 5090 has
   tensor-core quirks at low precision; TTT's inner-loop updates may underflow.
5. **Hybrid with TTT-MLP**: if TTT alone underperforms, does *one* TTT-MLP layer + *N−1* Gated
   DeltaProduct layers do better than uniform? (Mirrors our SO(n)+DeltaNet split, but with a
   stronger pair.)
6. **Code eval choice:** at 135 M nothing solves HumanEval. Better targets: pass@1 on tiny
   subsets of MBPP-S, or PPL on the-stack-v2-python — match our current TinyStories /
   Python-PPL setup so kill-gates compose.
