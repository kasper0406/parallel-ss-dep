# Lens: Systems/compute co-design (verbatim agent report, 2026-07-13)

# Systems/Compute Co-Design — 5 Ideas, Ranked

Framing: single-GPU training at ~24-28k tok/s means every 10% throughput ≈ +0.5B tokens over a Chinchilla-scale run. The two structural inefficiencies are (a) one entire RTX 5090 sitting idle during big runs, and (b) a decode path paying 2.3× launch-overhead tax on the thing the north star says is the moat.

---

## Idea 1 — Two-GPU data parallel WITHOUT DDP: manual bucketed allreduce, DiLoCo fallback (rank #1)

**Mechanism.** The DDP-latent incompatibility is a property of `DistributedDataParallel`'s autograd-hook machinery (static_graph + no_sync bug, plus your changing latent graph). Sidestep the module entirely: two processes, each does `loss.backward()` normally (latent path, curricula, everything untouched), then a hand-rolled `dist.all_reduce` over flat bf16 gradient buckets before the optimizer step. No hooks, no graph assumptions — the "incompatibility" evaporates because you never register anything with autograd. If PCIe sync cost turns out worse than estimated (consumer Blackwell almost certainly has P2P disabled like the 4090, so traffic bounces through host memory), escalate to DiLoCo-style local SGD: H inner steps per worker, sync parameter deltas with an outer Nesterov step — communication drops 30-500×, making the interconnect irrelevant ([DiLoCo/OpenDiLoCo](https://arxiv.org/pdf/2407.07852), [Streaming DiLoCo lineage](https://arxiv.org/html/2506.21263v1)).

**Arithmetic.** 287-402M params → bf16 grads = 574-804 MB/step. Host-staged PCIe gen5 realistically ~12-25 GB/s → 25-70 ms of comm per step against a ~350-500 ms step. Overlapped bucketed reduction (launch allreduce on early-layer buckets while late-layer backward runs) hides most of it: **~1.85-1.95× effective throughput, i.e. +85-95% tokens**. DiLoCo with H=32: 1.15 GB fp32 delta every ~15 s → <1% overhead, ~1.95×+, at the cost of a small convergence penalty (literature shows near-parity at tiny worker counts, and 2 workers is the easiest possible case).

**Cheapest decisive validation (<1 day).** A ~150-line two-process script: measure (i) raw `all_reduce` bandwidth for a 600 MB bucket on your rig (this single number decides manual-allreduce vs DiLoCo), (ii) tok/s scaling on 500 real train steps with the latent aux ON, (iii) loss-curve overlay vs single-GPU at iso-tokens. Kill-gate: effective scaling <1.5× and DiLoCo loss gap >0.5% at iso-token.

**Risk/effort.** 2-4 days total. Risks: grad-accum interaction (trivial — reduce once per optimizer step, which also amortizes comm by `grad_accum`×, making overhead ~1-2% even unoverlapped); Muon must see the averaged grad before Newton-Schulz (it does — NS runs in the optimizer, after reduction); bf16 grad reduction noise (you already validated bf16 optimizer state; reduce in fp32 if paranoid, still cheap). Divergent RNG/data shards are standard.

**Prior art.** This is just pre-DDP-era data parallelism plus [local-SGD literature](https://nathan.rs/posts/research-log/); nothing novel, which is why it's low-risk.

**Tokens/engineering-day: ~+90% throughput / 3 days ≈ 30%/day. Nothing else comes close.**

---

## Idea 2 — FP8 linears via torchao float8 on sm_120 (rank #2)

**Mechanism.** Consumer Blackwell has FP8 tensor cores and torch nightly's `_scaled_mm` reportedly hits **~2.5× bf16 matmul throughput on sm_120** — with the documented footgun that the default kernel preference tries a CUTLASS kernel that doesn't load on sm_120 and *silently falls back to a slow dequant path*; you must force the TORCH kernel preference ([PyTorch Blackwell/torchao blog](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/), [torchao paper](https://arxiv.org/pdf/2507.16099)). Apply `convert_to_float8_training` to the MLP + q/k/v/β projections only; FLA's Triton chunk kernels stay bf16. Your shapes qualify: at b=14, T=2048, the MLP GEMM is (28k × 960 × 7680) — far above the size where dynamic-quant overhead dominates.

**Arithmetic.** Projections + MLP are ~60-70% of step FLOPs in a DeltaNet block (the chunk kernel is the rest). If FP8 GEMMs net 1.8× after quant overhead: step speedup ≈ 1/(0.35 + 0.65/1.8) ≈ **1.39×**. Conservative (fwd-only FP8, or per-tensor scaling overhead worse than expected): 1.15-1.25×. Call it **+20-35% tokens**.

**Cheapest decisive validation (<1 day).** Microbench `_scaled_mm` vs bf16 at your exact GEMM shapes (2 hours — this alone confirms or kills the 2.5× claim on your silicon/driver). Then 500-step train A/B: tok/s + loss-curve overlay + gnorm trace. Kill-gate: loss curve visibly above bf16 at iso-step, or net speedup <10%.

**Risk/effort.** 1-2 days. Numerics risk at 287M is real but bounded — master weights stay fp32, Muon/AdamW see full-precision grads; your `--z_loss` is already the right logit-stability insurance. torch.compile interplay is the main unknown (torchao float8 is designed for compile, but your strict-mode will surface issues immediately — which is what strict mode is for). Note DeepSeek-V3 trained 671B in FP8; 287M is not the hard case, small-batch quant granularity is.

**Tokens/day: ~+25% / 1.5 days ≈ 17%/day.**

---

## Idea 3 — CUDA-graph capture of `forward_step` decode (rank #3, strategic)

**Mechanism.** Your 2.3× decode-latency gap is kernel-launch-bound, and you have the *best possible* workload for CUDA graphs: a linear RNN's decode step is **perfectly static** — fixed-shape 8.2 MiB state, no growing KV cache, same kernel sequence every token. This is easier than what vLLM/SGLang graph-capture (they fight variable KV lengths; you have none) ([SGLang CUDA graph](https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph), [Fireworks writeup](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning)). What breaks with your state dict: (1) the Python dict cache must become pre-allocated static tensors mutated in place (functionalize `forward_step`: state in, state out, copy into static buffers — the standard recipe); (2) `think_run_len` per-row counters → keep as a static int tensor updated on-graph; (3) gate emit/think branching is data-dependent → either capture two graphs (emit-step, think-step) and select on host from the previous step's gate readback, or keep the gate compare on-device and capture the superset. Route A: `torch.compile(mode="reduce-overhead", fullgraph=True)` on the functionalized step. Route B (more robust given FLA graph breaks): manual `torch.cuda.CUDAGraph` capture — Triton kernels capture fine; only the Python glue dies, which is the point.

**Arithmetic.** Launch overhead is 20-40% of decode time even in optimized servers; your gap is 2.3×, i.e. ~57% of step time is overhead. Graphs eliminate nearly all of it → **~1.8-2.2× decode throughput**, closing the gap to near-parity. Payoff lands in three places: grader-RL rollouts (generation-dominated — this is *training* tokens too, since RL wall-clock is mostly rollout), HumanEval/eval turnaround, and the north-star cost-moat benchmark, where "O(1) memory but 2.3× slow" becomes "O(1) memory at parity latency" — that changes the headline claim of `DECODE_COST_BENCH.md`.

**Cheapest decisive validation (<1 day).** Capture ONE block's `forward_step` (lean config, no FiLM/WM — the path `test_incremental_decode.py` already validates) with manual graph capture; assert logit equivalence vs eager; measure single-stream tok/s. If one block graphs cleanly, the full stack will.

**Risk/effort.** 1-3 days for the lean config. Risks: hidden host-device syncs in the step (`.item()` calls on gate values — must be hunted); memory-pool pinning during capture; the FiLM-K3+WM production decode path is still unwired anyway, so scope to the lean config that the cost-moat benchmark uses.

**Tokens/day: modest for pretrain, ~1.5-2× for RL-phase wall-clock, plus direct north-star value. ≈ strategic tie-breaker rather than raw token yield.**

---

## Idea 4 — Incremental-state latent growing-thread: O(chunk+R) per aux position instead of O(T·R) (rank #4)

**Mechanism.** The latent co-train aux re-forwards the (padded, batched) prefix per latent slot. But DeltaNet is a linear RNN: the state at any position is recoverable from a per-chunk state checkpoint. Do one forward over the sequence storing per-chunk-boundary states (chunk=64 → 32 checkpoints × ~1.1 MiB whole-model state each ≈ 35 MiB — trivial), then for each sampled aux position: restore the nearest checkpoint, replay ≤64 tokens, run the R grad-enabled latent steps via the existing `forward_step` machinery. Gradient flows through the R latent steps and the ≤64-token replay window; the prefix state is treated as constant — a truncated-BPTT semantics change, but the aux loss's job is training the latent adapter/gate, not the trunk-through-prefix, so truncation is arguably the *right* semantics (and matches your adapter-only post-hoc findings).

**Arithmetic.** Per aux position, compute drops from O(T)=2048 token-positions to O(64+R)≈70 → ~29× on the aux-forward component. You already banked 2.6-3.5× from batching; if residual aux overhead is ~15-25% of step time on latent runs, this recovers most of it: **+15-30% tokens on every latent-co-train run**, and unlocks denser aux sampling (more positions per step at the same cost — a quality lever, not just speed).

**Cheapest decisive validation (<1 day).** Numerics first: assert restored-state + 64-token replay reproduces the full-forward hidden at the aux position to bf16 tolerance (FLA's chunk kernel already returns final states; extending to per-chunk output is a kwarg away in your fork). Then time one aux step both ways.

**Risk/effort.** 2-4 days. Risks: FLA fork surgery to emit intermediate chunk states; grad-through-`forward_step` is untested (it's inference-oriented — check for `no_grad`/detach in the cache path); truncated-gradient semantics needs one iso-step convergence check against the current path.

**Tokens/day: ~+20% on latent runs / 3 days ≈ 7%/day (higher if latent co-train is the main workload going forward).**

---

## Idea 5 — FLA-on-Blackwell throughput audit: chunk-size + autotune-config sweep (rank #5, cheap)

**Mechanism.** FLA's `chunk_delta_rule` Triton configs (chunk=64, num_warps/num_stages grids) were tuned on Hopper/Ada; sm_120 has different SMEM per SM, register file, and [Triton's Blackwell backend improvements land real gains](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/) (NVIDIA reports up to 1.5× attention-class kernels via Blackwell-aware codegen). You own the fork — sweep chunk ∈ {32, 64, 128}, widen the autotune grid, pin the autotune cache (avoid re-tuning per shape), and profile whether the `(B,T,d)→(1,B*T,d)` cu_seqlens flatten adds a hidden reshape/copy.

**Arithmetic.** The chunk kernel is ~30-40% of step time; a 20-30% kernel win → **+6-12% end-to-end**. Honest expected value is the low end.

**Validation (<1 day).** `profile_train.py` (already exists) + a chunk-size/warps sweep on the real config. One day, fully decisive.

**Risk/effort.** ~1 day, near-zero risk. **Tokens/day: ~8%/day — good ratio, small absolute cap.**

---

## Anti-idea, flagged for honesty: sequence-length warmup curriculum

For softmax transformers, [seq-length warmup buys ~1.5× wall-clock](https://arxiv.org/pdf/2108.06084) because attention is O(T²). **DeltaNet's per-token cost is ~flat in T** (chunkwise-linear), so the throughput mechanism mostly doesn't exist here — you'd gain only second-order effects (activation memory → slightly larger batch, marginal stability). Not worth a slot; mentioning it so nobody spends a day rediscovering this.

Also skipped: multi-tenancy (GPU1 already absorbs evals during single-GPU runs; Idea 1 consumes it better), PowerSGD (compression is unnecessary at PCIe bandwidth for a 287M model — DiLoCo dominates it if comm ever becomes the bottleneck), and paged/offloaded optimizer state (bf16 optim state already banked the win; you're activation-bound, not state-bound).

---

## Single best bet

**Idea 1: two-GPU gradient averaging without DDP.** It is the only idea that changes the binding constraint by ~2× rather than tens of percent, it's the least novel engineering on the list (flat-bucket allreduce is 2017-era tech; DiLoCo is the validated fallback if PCIe disappoints), and the decisive experiment — measure allreduce bandwidth for one 600 MB bucket, then 500 real latent-training steps across two processes — fits in a single day. At token-poverty as the stated constraint, "the second $2k GPU stops idling" beats every kernel-level optimization combined; FP8 (Idea 2) then stacks multiplicatively on top for a combined ~2.3-2.6× if both land.

Sources: [DiLoCo/OpenDiLoCo](https://arxiv.org/pdf/2407.07852), [DiLoCoX](https://arxiv.org/html/2506.21263v1), [local SGD notes](https://nathan.rs/posts/research-log/), [PyTorch Blackwell MXFP8/torchao blog](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/), [torchao paper](https://arxiv.org/pdf/2507.16099), [SGLang CUDA graph docs](https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph), [Fireworks CUDA graphs writeup](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning), [Triton on Blackwell (NVIDIA)](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/), [Sequence Length Warmup (Composer)](https://docs.mosaicml.com/projects/composer/en/stable/method_cards/seq_length_warmup.html), [stability-efficiency dilemma paper](https://arxiv.org/pdf/2108.06084), [FLA repo](https://github.com/fla-org/flash-linear-attention).
