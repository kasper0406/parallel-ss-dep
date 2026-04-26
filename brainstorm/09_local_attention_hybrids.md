# 09 — Local-Attention Layer-Level Hybrids

## Executive context

Our SO(n)-rotation + DeltaNet layer hybrid lost 1.07–1.22x PPL vs pure DeltaNet. The likely reason: *both* sublayers are linear-RNN, so neither offers content-addressable recall — and "do parity well + do erase well" is not what's missing at 135M / 5k steps. What every production hybrid in 2024–2026 (Samba, Jamba, Granite-4, Zamba2, Hymba, MiniMax-01, Qwen3.5) actually mixes is **softmax attention (usually windowed) + a linear backbone**. That's the bet to copy.

## Top 5 concrete ideas

### Idea 1: SambaNet-Delta — interleaved DeltaNet + SWA, 1:1, window=2048

- **What it is:** Replace half the DeltaNet layers with sliding-window softmax attention (FlashAttention SWA, window 2048, GQA with 4 KV heads). Pattern: `[DeltaNet, SWA-MLP] × N/2`. This is Samba's exact recipe but with DeltaNet swapped in for Mamba — Samba uses Mamba+SWA at 1:1 with window=2048 and beats Phi-3-mini on HumanEval, MMLU, GSM8K, with perfect NIAH up to 256K from 4K training. Drop the rotation entirely.
- **Why for code:** Variable refs, function defs, type signatures, import-then-use patterns are precisely "needle-in-haystack within a few KB" — a 2K window covers >95% of intra-function and intra-file recall edges. DeltaNet handles the long, low-resolution stuff (file-level themes, repeated patterns).
- **Key papers:** Samba (Ren et al., MSR, ICLR 2025), Gated DeltaNet (Yang/NVIDIA, ICLR 2025), Mistral-7B SWA (2023). Production: Samba-3.8B, Qwen3.5 backbone.
- **Implementation cost:** S. fla already has DeltaNet/Gated-DeltaNet kernels; PyTorch SDPA + FlashAttention-2 has SWA built-in (`is_causal=True, window_size=(2048,0)`).
- **Code-LLM relevance:** **High.** Samba's coding numbers are the proof point.
- **Risk:** At 135M / 5k steps and short context (probably 1–4K), a 2048 window is essentially full attention — we won't see the long-context win and the parameter overhead from softmax KV may slightly hurt PPL. Mitigation: also test window=512.

### Idea 2: Granite-Style 9:1 — DeltaNet-heavy backbone with rare full-attention checkpoints

- **What it is:** Pattern `[DeltaNet × 9, MHA-full] × M`. Granite-4.0-H (IBM, Oct 2025) ships a 9:1 Mamba-2:Transformer ratio at 3B/7B/32B with strong commercial performance and >70% RAM reduction vs pure transformers. The single full-attention layer per block acts as a "global integrator" that the Delta state can route through.
- **Why for code:** Code has a *small number* of strong long-range edges (the function-def → call site, the import → use). 1-in-10 full-attention layers is enough to capture them; 9-in-10 DeltaNet handles the rest cheaply.
- **Key papers:** Granite-4.0 technical report (IBM, 2025), Jamba (AI21, 2024) used 1:7 with similar conclusions.
- **Implementation cost:** S. Same components as Idea 1, just different pattern.
- **Code-LLM relevance:** **High** for production, **medium** for our 135M/5K research scale (you may not see the win until 1B+ or 16K+ context).
- **Risk:** At small scale and short context the recall benefit is tiny, but the attention-parameter penalty is fixed — PPL might wash. The Granite recipe is tuned for 512K-context regimes.

### Idea 3: SWAX-Delta — Short-window SWA (W=128–256) + DeltaNet, alternating

- **What it is:** Same as Idea 1 but **window=128 or 256**. The Feb 2025 "Sliding Window Attention Training for Efficient LLMs" paper and the Sept 2025 "Short window attention enables long-term memorization" finding both report that **W=128 SWA outperforms W=2048** on RULER NIAH up to 131K context — counterintuitive but reproduced. Gemma 3 reduced its window from 4096 → 1024 with no quality loss.
- **Why for code:** Most code recall is *very local* (tokens within a function, last few lines for FIM). The DeltaNet backbone already provides a "fading global memory"; SWA only needs to provide crisp short-range lookup. Also: a 256-window SWA is essentially free even at training-time at 4K context.
- **Key papers:** SWAT (Feb 2025), "Short window attention enables long-term memorization" (Sept 2025), Gemma 3 tech report.
- **Implementation cost:** S. Identical to Idea 1 modulo a number.
- **Code-LLM relevance:** **High.** Plausibly the best bang-per-flop at our scale. With short context budgets and small models, smaller windows may be strictly better.
- **Risk:** "Short window beats long window" is a recent and contested result; it might not transport to code.

### Idea 4: Hymba-Style Parallel Heads — DeltaNet head ∥ SWA head per layer

- **What it is:** *Within* a layer, run DeltaNet heads and SWA heads in parallel and concatenate, instead of alternating layers. NVIDIA's Hymba 1.5B does this for Mamba+attention and beats Llama-3.2-1B with 10x less KV cache. Practical recipe: split your existing head budget 50/50, route Q/K/V through both paths, fuse outputs.
- **Why for code:** Every layer gets *both* fading-state context and crisp recall — no need to wait for the next attention layer. Especially useful for FIM where the model needs both the prefix-state and the immediate-suffix tokens at every depth.
- **Key papers:** Hymba (NVIDIA, ICLR 2025).
- **Implementation cost:** **M.** Requires writing a fused parallel-head module; not just a layer-stacking change. fla supports both primitives but you assemble them yourself.
- **Code-LLM relevance:** **Medium-high.** More expressive per parameter, but more risk in the integration.
- **Risk:** More moving parts; harder to debug; less proven outside Hymba's specific recipe.

### Idea 5: YOCO-Style Decoder-Decoder — DeltaNet self-encoder + cross-attention reader

- **What it is:** Bottom half of the network is pure DeltaNet (cheap), top half is cross-attention layers reading a single shared global KV cache produced by the bottom half. From "You Only Cache Once" (Sun/Dong, MSR 2024). Effectively gives full-attention quality with O(N) memory, prefill drops 180s → 6s at 512K.
- **Why for code:** Code generation is heavily prefill-bound (huge prompts, short completions). YOCO's prefill speedup matters *commercially* even if PPL is just on par.
- **Key papers:** YOCO (Sun et al., MSR, May 2024).
- **Implementation cost:** **L.** Architecturally non-trivial; needs careful KV-sharing wiring.
- **Code-LLM relevance:** **Medium.** Strong inference story, weaker pretraining-PPL story; not ideal for our 5K-step research loop.
- **Risk:** Returns are mostly at long context; at 4K our infra won't see them. Implementation risk dominates.

## Recommendation

**Run Idea 3 (SWAX-Delta short-window) and Idea 1 (Samba-Delta long-window) head-to-head** as the next experiment. Concretely, at 135M / 5K steps / TinyStories+Python:

1. Pure DeltaNet baseline (you already have this).
2. **DeltaNet + SWA(W=256), 1:1 alternating** (Idea 3).
3. **DeltaNet + SWA(W=2048), 1:1 alternating** (Idea 1, Samba clone).
4. *Optional:* Granite 9:1 (Idea 2) if budget permits.

This gives you a clean read on (a) does softmax recall help DeltaNet at all at our scale, and (b) does window size matter when context is short. **Skip the SO(n) rotation entirely for now** — every successful production hybrid in 2024–2026 mixes softmax attention with a linear backbone, not two linear backbones. The parity story is real but secondary to recall, which is the actual bottleneck the literature consistently identifies.

## Production-frontier 2025 convergence

The field has converged on **~10–25% softmax attention + ~75–90% linear backbone, with sliding windows of 1024–2048**, and the variance across teams is mostly cosmetic:

| Model | Year | Linear | Attn ratio | Window |
|---|---|---|---|---|
| Jamba | 2024 | Mamba | 1:7 (12.5%) | full |
| Samba | 2024 | Mamba | 1:1 (50%) | 2048 |
| Hymba | 2024 | Mamba | parallel heads | 1024 SWA + 3 full |
| Zamba2 | 2024 | Mamba2 | 2 shared full | full |
| MiniMax-01 | 2025 | Lightning | 1:7 (12.5%) | full softmax |
| Granite-4 | 2025 | Mamba2 | 1:9 (10%) | full |
| Qwen3.5 | 2025 | Gated-DeltaNet | hybrid w/ SWA | SWA |

Three takeaways: (1) **everyone uses softmax attention somewhere**; pure SSM (Falcon Mamba) underperforms on retrieval; (2) the **1:7–1:9 ratio** is dominant for >1B with full attention, **1:1 with SWA** is dominant for <4B at long context (Samba); (3) **SWA window is shrinking over time** (Mistral 4096 → Gemma3 1024 → SWAT 128–256). For a coding LLM specifically, the InfiniteVL/Qwen3.5 recipe — Gated-DeltaNet + SWA — is exactly the production-frontier point.

## Open questions

- At 135M/5K does softmax attention parameter overhead net positive vs the pure DeltaNet baseline? (Hybrid wins are mostly reported at 1B+.)
- Does SWA window=128 actually help on *code* (vs natural language), or do code's longer-range deps (function refs) demand 2048?
- Do we need attention sinks (StreamingLLM, 4 dummy tokens) when the linear path *already* provides a stable compressed state? Mostly no, but a quick ablation is cheap.
- Is the better play swapping DeltaNet → Gated-DeltaNet (Qwen3.5 backbone) before adding any attention?
- For *code* specifically, does FIM (with explicit prefix/suffix tokens) benefit more from parallel-head Hymba-style mixing than from layer-alternating?

## Sources

- [Samba (Ren et al., MSR, ICLR 2025)](https://arxiv.org/abs/2406.07522)
- [Samba GitHub](https://github.com/microsoft/Samba)
- [Jamba (AI21, 2024)](https://arxiv.org/abs/2403.19887)
- [IBM Granite 4.0 announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- [Granite 4.0 hybrid Mamba2/Transformer (MarkTechPost)](https://www.marktechpost.com/2025/10/02/ibm-released-new-granite-4-0-models-with-a-novel-hybrid-mamba-2-transformer-architecture-drastically-reducing-memory-use-without-sacrificing-performance/)
- [StripedHyena-7B (Together AI)](https://www.together.ai/blog/stripedhyena-7b)
- [Hymba (NVIDIA, ICLR 2025)](https://arxiv.org/abs/2411.13676)
- [Zamba2 (Zyphra)](https://www.zyphra.com/post/zamba2-7b)
- [YOCO (Sun et al., MSR, 2024)](https://arxiv.org/abs/2405.05254)
- [MiniMax-01 (2025)](https://github.com/MiniMax-AI/MiniMax-01)
- [Falcon Mamba 7B](https://arxiv.org/abs/2410.05355)
- [SWAT — Sliding Window Attention Training (Feb 2025)](https://arxiv.org/abs/2502.18845)
- [Short window attention enables long-term memorization (Sept 2025)](https://arxiv.org/abs/2509.24552)
- [StreamingLLM attention sinks (Xiao et al., ICLR 2024)](https://arxiv.org/abs/2309.17453)
- [Gated DeltaNet (Yang/NVIDIA, ICLR 2025)](https://arxiv.org/abs/2412.06464)
- [Mistral-7B SWA](https://arxiv.org/abs/2310.06825)
- [InfiniteVL (DeltaNet + SWA)](https://github.com/hustvl/InfiniteVL)
