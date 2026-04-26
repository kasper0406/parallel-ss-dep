# 06 — State Checkpointing / Compression / Async Offload

**Premise check.** For our cells the per-token state is `B*H*d_head*d_state` floats, and the *naive* parallel-scan backward materialises one such state per timestep. On 32 GB at d_head=64, d_state=64, B=8, H=8, fp16, that's ~16 MB per timestep — T=8k already eats >120 GB of activation memory unless we recompute. Mamba/Mamba-2/DeltaNet/FLA *all* avoid this in their fused kernels by recomputing inside the chunk; the practical question is whether we can push **chunk size and segment count** much higher to reach T=32k–256k, without re-implementing the wheel for our SO(n)/Heisenberg/U_4 cells.

## Top 5 concrete ideas

### Idea 1: Tiled-Flash-Linear-Attention–style "two-level" chunking for our scan kernels
- **What it is:** TFLA (Beck et al., NeurIPS '25) splits the sequence into outer chunks (recurrent across, parallel within) and adds a *second* level of tiling inside each chunk. This lets you pick chunk sizes >> what fla currently allows (fla caps at 64–256), so far fewer per-chunk states get materialised in HBM.
- **Memory mechanism:** materialised inter-chunk states drop from O(T/c · d_state²) to O(T/c_big · d_state²) where c_big can be 2k+. Recomputation of intra-chunk states stays in SRAM.
- **Compatibility:** mid. Our `ortho_son` autograd.Function is *already* a chunkwise kernel — we'd need to re-derive the backward to support a tiled inner loop, but the math is the same `state_t = M_t · state_{t-1} + b_t` form. The Heisenberg multi-d kernel is the harder port.
- **Key papers:** Beck et al., *Tiled Flash Linear Attention* (arXiv 2503.14376, NeurIPS '25); Yang et al., *Gated Linear Attention* (2312.06635) for chunkwise backward derivation.
- **Implementation cost:** **L** (1–2 weeks of Triton work per cell type, but mlstm_kernels reference impl is open-source).
- **Long-context payoff:** unlocks T≈32k–64k on a single 5090 at our model size.
- **Risk:** numerical error from recomputed states with tight chunk size; sm_120 register pressure on inner tile.

### Idea 2: Multi-Axis Gradient Checkpointing (MA-GC), Mamba flavour
- **What it is:** Park et al. (Video-Ma2mba, arXiv 2411.19460) checkpoint along *both* layer and sequence axes. Sequence-axis checkpoints save the SSM hidden state at K equispaced time-points; the segment between two checkpoints is recomputed forward-then-backward at backward time. Mamba scan is associative ⇒ recomputation is exact.
- **Memory mechanism:** activation memory drops to O(sqrt(T)·d_state) per layer instead of O(T·d_state). They report ~hours of video on a single GPU; in token terms that's millions of tokens.
- **Compatibility:** **high**. Implements as a thin `torch.utils.checkpoint`-style wrapper *around* the existing fla kernels — no kernel rewrites. Plays nicely with our `autograd.Function` for ortho_son if we treat each segment as a black box that takes `(x_seg, state_in)` → `(y_seg, state_out)`.
- **Key papers:** Park et al. (2411.19460) — exact recipe; Chen et al. *Training Deep Nets with Sublinear Memory Cost* (1604.06174) — original sqrt(T) checkpoint analysis.
- **Implementation cost:** **S/M** (a few days; mostly correctness-testing).
- **Long-context payoff:** T=32k almost free; T=128k feasible at the cost of 2× backward FLOPs.
- **Risk:** lowest of the 5. Pure compute/memory tradeoff, no accuracy hit.

### Idea 3: ZeCO / LASP-2 sequence parallelism across our 2 GPUs (no NVLink path)
- **What it is:** Each GPU owns a sequence shard of length T/G. Pass *only* the chunk-final state across the PCIe link (it's d_state² bytes, not Td_state). ZeCO (NeurIPS '25, arXiv 2507.01004) introduces an "All-Scan" primitive that boils communication down to a single small message per layer; LASP-2 (ICLR '25) is a more conservative ring-style version that is already in the fla repo.
- **Memory mechanism:** activation memory per GPU scales 1/G; communication is O(layers · d_state²) per step, fits comfortably in PCIe Gen5 ×16 between two 5090s.
- **Compatibility:** mid–high for linear-attn; **medium** for Heisenberg/U_4 because the "carry" message is no longer a single matrix but a tuple — still sendable, just more bookkeeping.
- **Key papers:** Chou et al., *ZeCO* (2507.01004); Sun et al., *LASP-2* (ICLR '25, OpenReview c6TDOPEQ0e); Sun et al., *LASP* (2404.02882).
- **Implementation cost:** **M** (LASP-2 reference code exists; integrating with our autograd.Function needs care).
- **Long-context payoff:** doubles whatever single-GPU max T we have. Critical if we go 4× or 8× boxes later.
- **Risk:** PCIe bandwidth without NVLink may serialise — but message is tiny, expected to be hidden.

### Idea 4: O(T)-memory backward via per-chunk **state checkpointing** (the cheapest win)
- **What it is:** During forward, store *only* the chunk-final hidden states (one per chunk, ~T/c of them) to HBM. During backward, for each chunk, reload the boundary state, replay forward in SRAM, and run the backward kernel — never materialise per-timestep states to HBM. This is exactly what Mamba/fla do *inside* a chunk; we extend it across the whole sequence by treating boundaries as torch checkpoints.
- **Memory mechanism:** activation memory becomes O((T/c) · d_state²) on HBM (boundary states only) + O(c · d_head · d_state) per active backward chunk. With c=2k and d_state=64 this is ~MB-scale even at T=32k.
- **Compatibility:** **highest**. This is a thin wrapper on our existing kernels — no math change. Our `ortho_son` autograd.Function already exposes a `state_in` argument, so it slots in.
- **Key papers:** Gu & Dao, *Mamba* (2312.00752, §3.3 "Hardware-Aware Algorithm"); blog: justinchiu.netlify.app, *Differentiating through an associative parallel scan* (2024).
- **Implementation cost:** **S** (1–2 days; basically a Python wrapper using `torch.utils.checkpoint` with `use_reentrant=False`).
- **Long-context payoff:** T=16k–32k immediately on existing kernels; combined with Idea 2 → T=128k.
- **Risk:** essentially none beyond ~1.3× backward time. **This is the kill-gate prerequisite for any long-context experiment.**

### Idea 5: Reversible linear-RNN block (RevFFN-style coupling)
- **What it is:** Wrap our linear-RNN block in a reversible coupling à la RevNet/RevFFN (arXiv 2512.20920 — RevFFN reports 49% peak VRAM reduction vs. checkpointing on MoE). Two streams `(x1, x2)`; block updates `x1' = x1 + f(x2)`, `x2' = x2 + g(x1')`. Forward stores nothing; backward reconstructs.
- **Memory mechanism:** activation memory at the *block* level → ~constant (only inputs of first block). State recurrence inside the block is still our parallel-scan cell; it just becomes an "f" inside a reversible coupling.
- **Compatibility:** medium. Reformulating as coupled streams changes the macro-architecture (effectively a hybrid). MacKay et al. (2018, 1810.10999) showed *perfectly* reversible RNNs have a fundamental "no-forgetting" pitfall — but here reversibility is at the *block* level, not the token level, so that's avoided.
- **Key papers:** Mangalam et al. *Reversible ViT* (CVPR '22); MacKay et al. *Reversible RNNs* (NeurIPS '18); Zhu, *Making Reversible Transformers Accurate, Efficient, and Fast* (Berkeley 2023); RevFFN (2512.20920, 2025).
- **Implementation cost:** **L** (architecture change; needs ablation against non-reversible baseline at our 135M scale).
- **Long-context payoff:** orthogonal to Ideas 1–4 — multiplies layer-count savings on top of sequence savings. Could push T·L product 4–8×.
- **Risk:** non-trivial accuracy hit observed historically when adapting RevNet-style designs to language; a wrong tweak burns weeks.

## Recommendation

**Build Idea 4 first (this week, 1–2 days), then Idea 2 (next week, 3–5 days).** Together they should turn our T=64 parity-kill-gate into a clean T=8k–32k experiment on existing 5090 hardware *without touching the Triton kernels*. Idea 4 is essentially "do what Mamba already does, but at the autograd.Function boundary." Idea 2 stacks on top with the same code path (just N-level checkpointing).

**Idea 1 (TFLA-style) is the medium-term play** if Idea 2's recomputation overhead (~2× backward) is too painful. It's ~2 weeks of kernel work but actually *reduces* compute via larger arithmetic intensity — a net win, not a tradeoff. Worth doing once the long-context training run is locked in.

**Skip Idea 5 for now.** Architecture changes during a hyperparameter hunt is masochistic. Revisit if Ideas 1+2+4 saturate at T~64k and we need more.

**Ignore Idea 3 (sequence parallel) until single-GPU saturation.** Without NVLink we'd just be paying complexity tax for marginal headroom.

## Open questions

1. Does our `ortho_son` autograd.Function already support a `state_in`/`state_out` interface, or do we need to add it before Idea 4 is even possible? (Quick-check kernels/ortho_son/.)
2. How does TFLA's tiled inner loop interact with our **non-diagonal** state-transition matrices (SO(n), Heisenberg)? FLA assumes diagonal — we may need a custom WY-representation.
3. At T=32k, does the parity gap (Heisenberg vs linear, +32.6 pp end-tok at T=64) widen or shrink? This is the *real* question — we hypothesise widen, but it's untested.
4. Is the backward-pass numerical error from chunk-boundary recomputation tolerable for our SO(n) blocks? Heisenberg is non-unitary and could drift.

Sources:
- [Tiled Flash Linear Attention (Beck et al., 2025)](https://arxiv.org/abs/2503.14376)
- [Video-Ma2mba: Multi-Axis Gradient Checkpointing (Park et al., 2024)](https://arxiv.org/abs/2411.19460)
- [ZeCO: Zero Communication Overhead SP for Linear Attention (Chou et al., 2025)](https://arxiv.org/abs/2507.01004)
- [LASP-2 (Sun et al., ICLR 2025)](https://openreview.net/forum?id=c6TDOPEQ0e)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Reversible Recurrent Neural Networks (MacKay et al., 2018)](https://arxiv.org/abs/1810.10999)
- [RevFFN: Memory-Efficient Reversible MoE (2025)](https://arxiv.org/html/2512.20920v1)
- [Differentiating through an associative parallel scan (Justin Chiu, 2024)](https://justinchiu.netlify.app/blog/pscan_diff/)
- [LongMamba: Training-Free Receptive Field Enlargement (2025)](https://arxiv.org/abs/2504.16053)
- [Stuffed Mamba: State Collapse in RNN Long-Context (2024)](https://arxiv.org/html/2410.07145v1)
