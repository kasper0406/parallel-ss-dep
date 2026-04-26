# 05 — MoE / Routing / Mixture-of-Cell-Types

**Frame:** the 25/75 ortho/Delta split lost on PPL because almost no tokens need parity-tracking.
A *fixed* layer ratio pays the SO(n) tax on every token. The MoE-style remedy is to keep the
specialized cell but route to it sparingly: most tokens see Delta; a small minority — bracket
opens, function definitions, indentation deltas — see ortho/state-tracking. Three structural
choices: **mix at layer level** (Jamba/Linear-MoE), **mix at head level** (Hymba), or
**mix at token level via a gate** (Switch / MoD / MoR). Code is exactly the regime where mixed
cell types should help: a Python file is mostly normal text plus rare structural moments.

---

## Top 5 concrete ideas

### Idea 1: Hymba-style parallel hybrid head — ortho head and Delta head fused per layer
- **What it is:** every layer is one block with `H` heads, *split* between ortho-rotation heads
  (e.g. 1 of 6) and Gated-DeltaNet heads (5 of 6), running **in parallel on the same input**, then
  re-projected via a shared output W_o. Same compute pattern as multi-head attention, but the
  heads have different recurrence kernels. Ratio is per-layer, *not* per-token.
- **Routing mechanism:** none — *static* head allocation. The model chooses how to use them via
  the projection mixing weights, which behave like a soft per-token selector.
- **Parallel-scan story:** trivial — each head runs its own scan, all in parallel along the batch
  dim. No coordination across heads. Output concat is a regular matmul.
- **Key papers:** Dong et al., *Hymba: A Hybrid-head Architecture for Small Language Models*
  (ICLR 2025, arXiv 2411.13676); Lieber et al., *Jamba* (2024, arXiv 2403.19887); Zamba2 (2024).
- **Implementation cost:** **S** — fla already exposes per-head DeltaNet; SO(n) head is a tiny
  rotation kernel we have. Just register two head types in one block.
- **Code-LLM relevance:** **high** — Hymba 1.5B beat Llama-3.2-1B with a 5:1 SSM:attn head ratio
  *at every layer*, where our 25/75 was a *layer-block* ratio. Per-layer per-head is finer-grained,
  and lets every layer have a tiny ortho contribution instead of clustering it in one quadrant.
- **Risk:** still a static ratio. If almost zero tokens need ortho, the head capacity is wasted —
  but the overhead is now `1/H` per layer (~17 % at H=6) instead of 25 % of layers. Weaker
  parity gains than a token-routed cell, possibly stronger PPL.

### Idea 2: Token-level switch over cell types (ortho / Delta / FFN-only) — Mixture-of-Cells
- **What it is:** literal Switch Transformer pattern, but the experts are **different recurrences**,
  not different MLPs. At each layer, a router computes `g_t = softmax(W_r · x_t)` over
  {ortho-cell, Delta-cell, identity}. Top-1 (Switch) or top-2 (Mixtral) routing. Each token's
  output is a (sparse) weighted sum of cells that fired.
- **Routing mechanism:** input-only `σ(W_r · x_t)` first; ablate input+state `σ(W_r · [x_t, h̄_{t-1}])`
  later. **Auxiliary-loss-free** balancing in the DeepSeek-V3 style (additive bias updated by EMA
  of expert load) — avoids the auxiliary-loss / specialization tradeoff. K-means / hash routing
  as a sanity baseline.
- **Parallel-scan story:** *hardest piece*. Trick: run **both scans densely** (each cell sees the
  whole sequence), but mask the input to each scan by its router gate, i.e.
  `k_t^ortho = g_t^ortho · k_t`, identical for value/state-update. The scan associativity is
  preserved per cell type because cells don't talk to each other within the layer; the router
  merely zeroes out non-routed tokens. Cost = sum of cell costs *with* full-length scans, no
  dynamic shapes. Megablocks-style block-sparse only helps if we accept variable-length per-cell
  scans, which breaks the chunked DeltaNet kernel.
- **Key papers:** Fedus et al., *Switch Transformer* (2021, arXiv 2101.03961); Wang et al.,
  *Auxiliary-Loss-Free Load Balancing* (DeepSeek, 2024, arXiv 2408.15664); Sun et al.,
  *Linear-MoE* (Mar 2025, arXiv 2503.05447) — first system that puts MoE on top of linear
  RNN/SSM/linear-attn cells (but their experts are still FFNs; ours would be cell types).
- **Implementation cost:** **L** — router + gating + dual scan + load-balance bookkeeping. Each
  cell already exists; the glue is the work. Triton fused gate-then-mask wrapper is needed for
  good throughput.
- **Code-LLM relevance:** **high** — directly tests the hypothesis "ortho only matters on the
  rare structural tokens." If true, the router will converge to firing ortho on bracket opens,
  function `def`/`class` lines, indentation steps. Inspectable post-hoc, which is itself valuable.
- **Risk:** at 135M / 5k steps, the router may not converge — Switch routers are notoriously
  data-hungry and need a warmup. Routing collapse to one cell is the failure mode; aux-loss-free
  bias should mitigate. Also: dense-scan-with-masking still pays full FLOPs, so the win is
  *quality at fixed compute*, not speedup.

### Idea 3: Soft-MoE over cell types — fully differentiable, no top-k
- **What it is:** instead of discrete routing, every cell processes a **weighted average** of
  tokens (or, dually, every token is a weighted sum of cell outputs):
  `y_t = sum_c α_{t,c} · Cell_c(x_t)`, with α from softmax over cells, **trained without aux loss**.
  This is Soft-MoE adapted from {experts} to {cell types}.
- **Routing mechanism:** continuous `α_{t,c} = softmax_c(W_r · x_t)` per token, fully
  differentiable. Effectively a *learned mixture* between ortho, Delta, and (optionally) attention,
  per-token.
- **Parallel-scan story:** every cell's scan runs over the full sequence — no masking needed,
  just unscaled inputs. Outputs are linearly combined per token. This is the cleanest scan story
  of all routing variants.
- **Key papers:** Puigcerver et al., *From Sparse to Soft Mixtures of Experts* (ICLR 2024,
  arXiv 2308.00951); Hymba uses an implicit version of this via parallel head fusion.
- **Implementation cost:** **M** — basically Hymba (Idea 1) with a learned per-token gate
  replacing the fixed head ratio. Still no token dropping, no aux loss.
- **Code-LLM relevance:** **high** — this is the *cleanest* answer to "why was 25/75 wrong?"
  Replace the static 25/75 with a per-token learned mix and let SGD discover the schedule.
  At the limit it can degenerate to "α_ortho ≈ 0 everywhere" (which would be a strong negative
  result) or "α_ortho spikes on `(`,`{`,`def`" (the desired behavior).
- **Risk:** runs all cells densely → no compute saving. Pure quality bet. Also: a soft mixture
  can be *worse* than a single best cell because it averages two functions whose composition was
  the original incompatibility we found between SO(n) and Delta.

### Idea 4: Mixture-of-Depths over cell-blocks — token-level skip of the ortho cell
- **What it is:** keep DeltaNet on every layer; insert a *skippable* ortho block whose router
  decides per-token whether to apply it (`y = x + g_t · OrthoCell(x)` with `g_t ∈ {0,1}`,
  expert-choice top-k routing). Every token always pays Delta cost; only `k%` of tokens pay ortho
  cost.
- **Routing mechanism:** **expert-choice routing** (Zhou et al. 2022) — the ortho block has a
  capacity `C = k · T`, and the router selects top-`C` tokens by score. Guarantees perfect load
  balance, no aux loss, no collapse. Gate input: `x_t` (post-Delta).
- **Parallel-scan story:** the ortho scan now operates on a **subsequence** of length `C` selected
  per layer per batch. Use the *gather-scan-scatter* pattern: gather the routed tokens, run a
  short ortho scan over the contiguous gathered buffer, scatter back. Caveat: gathered tokens lose
  positional adjacency, so the ortho recurrence between bracket-open at position 23 and
  bracket-close at position 41 is now between gathered indices 7 and 8 — *which is exactly what
  we want* if the cell is meant to track structural state.
- **Key papers:** Raposo et al., *Mixture-of-Depths* (2024, arXiv 2404.02258); Zhou et al.,
  *Expert Choice Routing* (NeurIPS 2022, arXiv 2202.09368); Bae et al., *Mixture-of-Recursions*
  (NeurIPS 2025, arXiv 2507.10524) — most recent token-routing-with-recurrence work.
- **Implementation cost:** **M-L** — gather/scatter is a single Triton kernel; the rest is plain
  PyTorch. Megablocks block-sparse not strictly needed.
- **Code-LLM relevance:** **medium-high** — and this is the most *interpretable* design: we can
  literally print which tokens got routed to ortho and check if they are bracket characters. If
  yes, we have a strong story for why the cell helps. If no, we cleanly disprove the hypothesis.
- **Risk:** subsequence scan breaks distance — a `(` at pos 23 paired with `)` at pos 41 now sees
  no recurrence-distance signal. May hurt parity-style tracking. Counter: structural state
  tracking is *exactly* about contraction across irrelevant tokens, so collapsing them is the
  point.

### Idea 5: Hash-routed Mixture-of-Cells — zero-parameter router as ablation
- **What it is:** route by token-id hash to one of {ortho, Delta} cells. No learned router. Used
  purely as a **sanity baseline** for Idea 2: if learned routing doesn't beat hash-routing
  significantly, the routing decision isn't doing useful work.
- **Routing mechanism:** `cell(t) = ortho if hash(token_id_t) % 100 < 25 else Delta`. Same 25/75
  budget as our failed hybrid, but distributed across tokens instead of layers.
- **Parallel-scan story:** identical to Idea 2 — masked dense scans.
- **Key papers:** Roller et al., *Hash Layers* (NeurIPS 2021, arXiv 2106.04426); Cerebras "Router
  Wars" comparison (2024).
- **Implementation cost:** **S** — strip the learned router off Idea 2's code path.
- **Code-LLM relevance:** **low as a primary**, **high as an ablation** — tells us whether the
  signal is in *which tokens* go where (learned beats hash) or merely in *having a mix of cells*
  (hash matches learned). Cerebras showed learned routing is 3× better than hash at 128 experts —
  but with only 2 cell types and a structurally meaningful split (e.g., hash on `(`, `[`, `{`,
  `:`, `\n` always to ortho), the gap may close.
- **Risk:** if hash-routing matches learned routing, all our routing engineering was overhead.
  Useful negative result.

---

## Recommendation

**Winner: Idea 3 (Soft-MoE over cell types)** as the cheapest first experiment, **Idea 4 (MoD
over the ortho cell)** as the principled scaling target.

Soft-MoE is essentially Hymba with a learned per-token gate replacing the fixed head split. It
is the minimal-risk way to test "is 25/75 wrong, or is mixing wrong?" — if even the soft mixture
loses to pure Delta, the cell type is the problem; if it wins, *some* mixing is right and we
graduate to Idea 4 for compute savings.

Idea 4 (MoD-over-ortho) has the killer interpretability story: we can literally inspect which
tokens get routed to the ortho cell. In a code corpus, if the routed tokens are `(`, `def`,
indentation, the design wins both PPL and a publishable mechanistic story; if they are random,
we learn the hybrid premise is wrong.

Skip Switch-style top-1 routing (Idea 2) for now — too much engineering for our 135M / 5k-step
budget, and DeepSeek-V3-style aux-loss-free balancing wants 1B+ scale to shine.

## Open questions

- **Gate input:** input-only or input+state? State adds expressivity (the gate can react to *what
  was just stored*), but breaks the parallel-scan property of the gate itself (gate now depends
  on prior states). Probably input-only is enough at our scale.
- **Granularity of routing:** per-token vs per-chunk-of-16. Chunk-routing is cheaper and matches
  DeltaNet's chunk size; per-token is finer. Hymba implicitly chunks at head-fusion; MoD is
  per-token.
- **Specialization vs collapse:** at 135M / 5k steps, will the router actually specialize? Or
  will it converge to "use Delta always" (= pure DeltaNet baseline, our best result)? An
  *aux-loss with very small weight* may be necessary to break ties early.
- **Composition with Idea 2 (Gated DeltaProduct from brief 01):** if DeltaProduct already
  absorbs the ortho cell's role into a single recurrence, we may not need routing at all. Order
  the experiments accordingly.
- **Inference-time:** routed cells need cell-specific KV/state caches. Fine for ortho (small
  state), but a routing decision that flips between ortho and Delta mid-sequence raises a stale-
  state question for the un-routed cell. Need to either keep both states warm (cheap) or accept
  warmup tokens after a flip.
