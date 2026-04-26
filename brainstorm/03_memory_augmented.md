# 03 — Memory-Augmented Recurrence

**Premise check.** Code is full of name-bound, long-range pointer-chasing: a function at line 200 must resolve a `def` at line 12, an import 8 files away, a type at the top of the module. A 256-dim DeltaNet hidden state is a poor associative store for this. The hybrid SO(n)+DeltaNet stack we tested escapes the parity wall, but it does **nothing** for the recall wall in code. External / sparse / content-addressable memory tackles that wall directly. Below: the few designs from 2024–2026 that are both *parallel-scan-friendly* and *plausibly worth a 5k-step bake-off*.

## Top 5 concrete ideas

### Idea 1: Trainable PKM Memory Layers (Lample/Meta-style, scaled 2024)
- **What it is:** Replace some FFN layers with a sparse trainable key–value lookup. Query produces a top-k mask over a huge (1M+) parameter table; a tiny softmax over k≈8 retrieved values returns the layer output. Meta showed up to 128B-param memory layers beating dense baselines with same FLOPs.
- **Memory mechanism:** Read O(√M) via product keys → top-k (k≈8). Write = backprop on the k touched rows; **fully parallel across tokens** since each token's lookup is independent. No cross-token dependency → drops cleanly into a parallel-scan RNN backbone.
- **Why for code:** Acts as a learned hash from "context summary" → "common identifier / API / type idiom". Massive parameter capacity at constant FLOPs is exactly what you want for memorizing the Python stdlib or a repo's symbol table without bloating the recurrent state.
- **Key papers:** Berges et al. *Memory Layers at Scale* (Meta, 2024); Lample et al. *Large Memory Layers with Product Keys* (2019); Zhao & Jones *Fast-weight PKM* (2026).
- **Implementation cost:** **M.** Meta released code (`facebookresearch/memory`); slot it into one mid-stack DeltaNet block.
- **Code-LLM relevance:** **High.** Direct match for "I need to know what `re.compile` returns" and similar parametric-knowledge lookups that crush small-model code PPL.
- **Risk:** Sparse-grad CUDA kernels are the bottleneck; on sm_120 with no NVLink the embedding-table sharding could be ugly. Also: gains may be redundant with attention layers in a hybrid model.

### Idea 2: Titans-style Test-Time Memory (2-layer MLP fast-weights, surprise-gated)
- **What it is:** A small MLP whose **weights** are the recurrent state. The "input" loss `‖MLP(k_t) − v_t‖²` produces a per-token gradient that updates the MLP — i.e. the recurrence rule **is** SGD on associative memory. Behrouz et al. (Titans, NeurIPS 2025) showed >2M-token tasks with O(1) state-size scaling.
- **Memory mechanism:** Read O(1) (one MLP fwd). Write O(1) (one SGD step with momentum + decay). Crucially, the chunkwise update is **mathematically equivalent to a generalized delta rule** → integrates into the same parallel-scan / chunkwise-matmul pipeline as DeltaNet/Gated DeltaNet (`fla` already supports this pattern).
- **Why for code:** Nonlinear key→value map handles polysemous identifiers far better than a linear outer-product memory. "Surprise gating" naturally writes when a new symbol is *defined* (high loss) and skips on routine repetition.
- **Key papers:** Behrouz et al. *Titans: Learning to Memorize at Test Time* (2025); Sun et al. *Test-Time Training Done Right / LaCT* (2025); Yang et al. *Gated DeltaNet* (ICLR 2025).
- **Implementation cost:** **M-L.** Triton kernel non-trivial (chunked Jacobian-vector products), but Sun's LaCT released chunked-TTT kernels that scale to 1M-token chunks.
- **Code-LLM relevance:** **High.** This is the *direct* generalization of DeltaNet that already beats DN on long-context recall in published numbers — and it's drop-in for our `fla` stack.
- **Risk:** Triton kernel correctness/perf on sm_120; "Titans Revisited" (2510.09551) reports the original numbers were partly artifacts of tokenization. Need to replicate carefully.

### Idea 3: Memorizing-Transformer-style kNN cache over a DeltaNet backbone
- **What it is:** Every K layers, store (k, v) into a non-differentiable FAISS-like index of recent activations. At read time, do approximate kNN top-32 lookup over the past N≈64k tokens and gate that into the current hidden state. Wu et al. (2022) + the 2024 MLP-Memory and MemLong revivals show steady PPL gains on **code (GitHub)** specifically up to 262k mem.
- **Memory mechanism:** Read O(log M) with HNSW / IVF indexes. Write O(1) (append + index insert). Reads are **per-token-independent** → parallel-scan compatible across the recurrent backbone; the index lookup itself is the only sequential dependency, and it can be pipelined behind the scan.
- **Why for code:** Genuinely the most natural fit. A function definition becomes an entry in the index; calling that name later does a content-based fetch of the original `def` line's KV. This is *literal* pointer-chasing.
- **Key papers:** Wu et al. *Memorizing Transformers* (ICLR 2022, code-section is the strongest result); Liu et al. *MemLong* (2024); *MLP Memory* (2025).
- **Implementation cost:** **M.** FAISS already integrates with PyTorch; the gating layer is trivial.
- **Code-LLM relevance:** **Very high.** Best-published evidence on the GitHub-PPL axis we actually care about.
- **Risk:** Index quality at small training scales is unproven; at 135M / 5k steps the model may not learn to *use* the kNN signal before the run ends. Mitigation: warm-start the gate to lean on kNN and let the model un-learn it.

### Idea 4: Modern Hopfield / Energy-Transformer slot memory
- **What it is:** Persistent set of M≈256–1024 learnable "slot" vectors. Read = single Hopfield update (`softmax(βQM^T)M`) which is *mathematically identical to softmax-attention over the slot set*. Slots are content-addressed, exponentially many fixed points → high storage capacity (Ramsauer et al., 2020; KHM 2024 for tight bounds).
- **Memory mechanism:** Read O(M) with M small and constant — basically a tiny attention layer per timestep. Write = either gradient on the slot params (slow, parametric) or a fast Hebbian update (1 step). Trivially parallel-scan-compatible.
- **Why for code:** Slots can specialize to recurring archetypes ("loop counter", "open file handle", "boolean flag"). Less directly suited for arbitrary identifier recall, more suited for *role* tracking.
- **Key papers:** Hu et al. *Provably Optimal Memory Capacity / KHM* (NeurIPS 2024); Sun et al. *Associative Transformer* (CVPR 2025); ICLR 2025 *Frontiers in Associative Memory* workshop.
- **Implementation cost:** **S.** A few lines on top of any block.
- **Code-LLM relevance:** **Medium.** Compelling, but slot count caps capacity; the kNN cache or PKM is a stronger bet for the symbol-table use-case.
- **Risk:** May just be reinventing soft-attention with extra steps; rarely beats vanilla attention on pure LM PPL.

### Idea 5: StreamingLLM-style persistent token sinks as a *learned* episodic buffer
- **What it is:** Reserve N≈64 "register tokens" that every layer cross-attends to (these *are* the memory). Optionally cycle them via a tiny RNN that compresses departing-window state into a sink slot. Combines cleanly with a DeltaNet backbone — sinks become globally visible *episodic* state, recurrence stays local.
- **Memory mechanism:** Read O(N) tiny attention. Write = either learned compressor (parallel) or queue-style FIFO (sequential but cheap). Fully parallel.
- **Why for code:** Sinks naturally become a per-file scope: imports, top-level type defs, the few currently-open scopes. Mirrors how humans read code.
- **Key papers:** Xiao et al. *StreamingLLM* (ICLR 2024); Munkhdalai et al. *Infini-Attention* (2024) — both ship this idea in different clothes; *Elastic Memory* (2026) and *Melodi* are the latest variants.
- **Implementation cost:** **S.**
- **Code-LLM relevance:** **Medium.** Cheap and likely a small win, but unlikely to be a *fundamental* upgrade over our current hybrid.
- **Risk:** Infini-Attention got a public failed-replication writeup (HF blog) — reproducibility is contested; the gains tend to be at very long context (>32k) which we're not even training on yet.

## Recommendation

**Bet on Idea 2 (Titans / TTT-MLP delta-rule generalization) and Idea 3 (kNN cache over DeltaNet) — in that order.**

- **Idea 2** is the lowest-friction next step from where we are today. The math is a clean superset of Gated DeltaNet (which `fla` already implements), the Triton effort reuses the chunkwise-update template we wrote for Heisenberg, and **the same kill-gate methodology** (parity test → end-tok PPL on TinyStories+code) maps over directly. If the Titans-style nonlinear fast-weights beats Gated DeltaNet on T=64 recall by even +5 pp, that's a stronger signal than the SO(n) parity result and points squarely at code.
- **Idea 3** is the ideological win: an external kNN over the GitHub-trained model is the only one of these designs that *can plausibly outperform attention on the symbol-resolution task at any scale*. Bigger build, bigger payoff. Stage it as a follow-up if Idea 2 lands.

Skip the SO(n) hybrid for memory work — it solves an orthogonal (parity / circuit-class) problem; layering external memory on top of plain DeltaNet will likely outrun the hybrid at 135M.

## Open questions

1. **Eval gap:** Can we synthesize a "pointer-chasing" microbenchmark in code (e.g. resolve-this-identifier-defined-N-tokens-back) where Idea 2 vs Idea 3 vs DeltaNet make distinguishable predictions at 135M scale? This is the missing companion to the T=64 parity gate.
2. **Parallel-scan compatibility of Titans variants (MAC vs MAG vs MAL):** the released paper conflates them; only MAL is provably equivalent to a delta-rule scan. Need to read 2510.09551 (Titans Revisited) before committing.
3. **PKM × DeltaNet interaction:** does a 1B-param PKM bolted onto a 135M DeltaNet behave like a 135M model + a knowledge cache, or does it actually reshape the model's recurrence dynamics? Worth a 2k-step ablation.
4. **kNN warm-up:** at 5k steps, will the gate even discover the cache signal? Need a curriculum (synthetic copy task → real code) to bootstrap.
5. **Sparse CUDA on sm_120:** memory layers bottleneck on irregular gather/scatter; how much does the Blackwell tensor-memory accelerator help vs hurt vs the released A100 kernels?

---

### Executive summary (~150 words)

External memory beats the recall wall that the SO(n) hybrid doesn't touch — especially for code. Of the 2024–26 work surveyed, two ideas matter: (1) **Titans-style test-time-trained MLP memory** (Behrouz, NeurIPS 2025) is a clean nonlinear generalization of Gated DeltaNet's delta rule, parallel-scan compatible, and the most plausible drop-in upgrade to our current `fla` stack; (2) **kNN cache à la Memorizing Transformer** (Wu 2022, MemLong 2024) is the *only* design here whose published gains are largest on **GitHub-PPL specifically**, i.e. our actual target. Memory layers (PKM, Meta 2024) are a strong third-place card mostly for parametric symbol-table capacity. Hopfield slots and StreamingLLM sinks are cheap but unlikely to be game-changing. Recommendation: implement Titans-MAL chunkwise Triton kernel as the immediate next step, with kNN cache as the high-payoff follow-on. Both compose with — and likely subsume — the existing DeltaNet hybrid.

---

**Sources:**
- [Titans: Learning to Memorize at Test Time (2501.00663)](https://arxiv.org/abs/2501.00663)
- [Titans Revisited (2510.09551)](https://arxiv.org/abs/2510.09551)
- [Gated Delta Networks (NVIDIA, 2412.06464)](https://arxiv.org/abs/2412.06464)
- [Memory Layers at Scale (Meta, 2412.09764)](https://arxiv.org/html/2412.09764v1)
- [facebookresearch/memory](https://github.com/facebookresearch/memory)
- [Memorizing Transformers (2203.08913)](https://arxiv.org/abs/2203.08913)
- [MemLong: Memory-Augmented Retrieval for Long Text (2408.16967)](https://arxiv.org/html/2408.16967v1)
- [MLP Memory (2508.01832)](https://arxiv.org/html/2508.01832v1)
- [Provably Optimal Memory Capacity for Modern Hopfield Models (2410.23126)](https://arxiv.org/abs/2410.23126)
- [StreamingLLM (2309.17453)](https://arxiv.org/abs/2309.17453)
- [Infini-Attention (2404.07143)](https://arxiv.org/abs/2404.07143) and the [HF failed-replication blog](https://huggingface.co/blog/infini-attention)
- [Test-Time Training Done Right / LaCT (2505.23884)](https://arxiv.org/html/2505.23884v1)
- [Megalodon (2404.08801)](https://arxiv.org/abs/2404.08801)
- [Fast-weight PKM (2601.00671)](https://arxiv.org/html/2601.00671)
- [YOCO (2405.05254)](https://arxiv.org/pdf/2405.05254)
- [Retrieval-Augmented Code Generation Survey (2510.04905)](https://arxiv.org/abs/2510.04905)
