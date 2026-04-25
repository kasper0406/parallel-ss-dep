# EVAL_PLAN.md

Evaluation and execution plan for the state-dependent parallel-scan cells
(`linear_attn`, `heisenberg_d`, `unipotent_u4`).

**Initial target hardware: 2× NVIDIA RTX 5090** (Blackwell consumer, sm_120,
32 GB GDDR7 × 2, PCIe-only between cards — no NVLink). We will only scale
to datacenter Blackwell (B100/B200/GB200) if the small-scale results are
clearly promising. The kernel source is arch-agnostic (sm_120 and sm_100
share the same 5th-gen tensor-core ISA), so any tuning and correctness
work done on 5090 ports straight to B200.

§1 below talks about sm_100 / sm_120 Triton compatibility; §2 is
precision-agnostic; §3 and §4 were originally scoped for datacenter
Blackwell and are repriced for the 2× RTX 5090 reality in **§0 (below)**
and in the revised compute tables in §4.

---

## TL;DR decision table

| Decision | Recommendation | Why (one line) |
|---|---|---|
| Triton version | `triton >= 3.4` (nightly if needed) on CUDA 12.8 + PyTorch 2.7 NGC container | sm_100 codegen is stable in 3.4; 5.x adds warp-spec autotune knobs |
| Initial block sizes | `BLOCK_T=64, BLOCK_D=64, num_warps=4, num_stages=3` for heisenberg_d; `BLOCK_T=128, num_warps=8, num_stages=3` for linear_attn | Matches fla defaults on H100/B200; leaves headroom for autotune |
| Precision (training) | BF16 weights + FP32 running state + BF16 I/O; disable FP8 for state | BF16 compounding error is tolerable per-step but the d×d state is the failure mode |
| Precision (matmul) | `tl.dot(..., out_dtype=tl.float32)` — MMA in BF16, accum in FP32 | Standard; FP8 on Blackwell is still buggy in `tl.dot_scaled` as of 2026 Q2 |
| Decay | Add scalar RetNet-style per-head λ first; move to GLA/Mamba-2 data-dependent gate only if state norm diverges | Scalar λ preserves monoid; data-dependent gate requires a new Lean proof |
| Base model | **SmolLM2-135M** first pass → **Qwen3-1.7B-Base** mid-scale → Qwen3-4B-Base for final signal | Apache 2.0, 32K ctx, recent, small enough to iterate |
| Layer replacement | Hybrid: replace 3 of every 4 attention layers, keep 1 softmax (mirrors Qwen3-Next 75/25 split) | Hazy & MiniMax results: hybrid recovers retrieval; pure linear drops 5-15 pp MMLU |
| Training recipe | LoLCATs-style two-stage: (1) attention-transfer MSE on outputs; (2) LoRA SFT on ~100M-400M tokens | 5 GPU-hours on one A100 for 8B per LoLCATs; scales well |
| First synthetic eval | MQAR (Zoology) at d=64, seq=512, 20k pairs | Fastest, cleanest separator of linear-vs-softmax recall |
| First state-tracking eval | Parity + mod-p addition (p∈{3,5,7}) over seq=1024 | Direct probe of Heisenberg's O(T·‖a‖‖b‖) state capacity & U_4's trilinear term |
| GPU budget, first end-to-end | ~40 B200-hours **≈ 400 single-5090-hours ≈ 8–10 days on 2× 5090** (see §0 and §4.2) | Includes toy pretrain + distill + RULER + downstream |
| Student model ceiling on 2× 5090 | **135M-360M for comfortable iteration; 1.7B only for one decisive run** | 64 GB pooled but no NVLink — FSDP/ZeRO-2 over PCIe throttles at 1B+ |
| First-pass precision | BF16 in + FP32 accum. NVFP4/MXFP8/2:4 sparsity are **later tuning levers**, not first-pass | Correctness > throughput until the cell has a signal |
| Kill gate | 30M synthetic suite must show ≥ 10 pp advantage of heisenberg_d or unipotent_u4 over linear_attn, else write up the negative result | Unchanged from original plan |

---

## 0. Hardware-adjusted scope (2× RTX 5090)

**Compute.** One RTX 5090 ≈ 1/10th of a B200 on dense BF16 tensor-core
TFLOPS (≈ 209 vs ≈ 2250 TFLOPS) and ~1/4 of a B200 on memory bandwidth
(~1.8 vs 8 TB/s). Two 5090s without NVLink behave like ~1/5 of a single
B200 for single-model training — the PCIe bottleneck eats the extra card
on anything but embarrassingly parallel sweeps.

Conversions (same work × ~10 for single-5090, ~5 for 2× 5090 wall-clock):

| Original B200 budget (§4.1-§4.2) | 2× RTX 5090 equivalent |
|---|---|
| ~12 B200-h toy sweep | ~60 wall-clock hours (2.5 days) |
| ~2 B200-h Qwen3-1.7B attention transfer | ~10 wall-clock hours |
| ~10 B200-h Qwen3-1.7B LoRA SFT | ~50 wall-clock hours (2 days) |
| ~8 B200-h eval harness | ~40 wall-clock hours (1.5 days) |
| ~40 B200-h full end-to-end | **~200 wall-clock hours on 2× 5090 ≈ 8-10 days** |

**Memory.** 32 GB per card × 2 = 64 GB pooled. With ZeRO-2 and gradient
checkpointing, a 1.7B BF16 student fits. A 4B does not fit for
distillation (teacher + student + activations); 4B is off the table
until we have datacenter hardware.

**Precision plan (revised).**
1. Correctness: FP32 reference everywhere. Match pytorch reference atol
   < 1e-5.
2. Training-first pass: BF16 inputs + FP32 accumulator. Ship whatever
   gives us the cleanest training signal.
3. Throughput tuning (only if the cell shows signal):
   - **FP8 (E4M3/E5M2)** `tl.dot` inputs for `linear_attn` — but watch
     for silent BF16 fallback on sm_120 ([triton#7188][tr-7188]).
   - **NVFP4 (E2M1 + microscaling block scales)** for `linear_attn` only,
     once `tl.dot_scaled` on sm_120 actually emits MXFP4 MMA
     ([triton#7550][tr-7550] — still silently BF16 as of early 2026).
   - **2:4 structured sparsity** on the *learned projections* that
     produce q/k/v and our `a`/`b` — orthogonal to the scan algebra,
     pure GEMM throughput lever. sm_120 tensor cores support it at
     2× throughput for FP8/BF16.
4. For `heisenberg_d` and `unipotent_u4`: keep accumulator FP32 even
   at the final tuning pass. The `Σ_{i<j}` and `Σ_{i<j<k}` state-growth
   stories (§2) don't change with a different input dtype.

**Arch-specific bug that affects us.** [pytorch#176426][pt-176426]
documents a multi-`tl.load` segfault specific to **sm_120** (RTX 5090
family). Our kernels load q, k, v (three loads) per chunk; we will need
to verify on first port that none of our launch configs hits this. If
we do hit it, a short workaround is to fold into fewer loads via a
strided pattern or to compile against PTX 8.7 manually. **Datacenter
B100/B200 (sm_100) is not affected** — so any workaround we add is
temporary and can be removed when we upgrade.

**Sequencing adjustment from §4.3.** With ~200 wall-clock hours budget
on 2× 5090 instead of 1344 GPU-hours on 8× B200, the right order is:

- **Week 1 (day 1-2):** Kernel port + smoke tests + sm_120 segfault
  check. BF16/FP32-accum correctness against the PyTorch reference at
  multiple (T, d, B) configs.
- **Week 1 (day 3-4):** 30M synthetic suite — MQAR + parity + mod-p +
  Dyck-2 at 30M params, 4 architectures × 3 tasks × 2 seeds. **This is
  the kill gate.** ~60 wall-clock hours budget. If the cell doesn't
  clear the 10 pp bar, freeze and write up.
- **Week 2:** SmolLM2-135M distillation (one config, heisenberg_d @
  75% hybrid, scalar decay). ~40 wall-clock hours. If PPL + MMLU +
  MQAR at 1k show signal, proceed to 1.7B.
- **Week 3-ish:** Qwen3-1.7B one decisive run. 100+ wall-clock hours.
  If the 1.7B result is decisive, rent a B200 cluster for the final
  scale-up to Qwen3-4B-Base (originally §4.3 day 7).

This is slower than the B200 plan by ~4× but uses cheap capex we already
have. The sequencing below (§4) still applies directionally; just read
every "B200-hour" as "~5 wall-clock hours on 2× 5090".

---

## 1. Blackwell / Triton compatibility

### 1.1 Version & toolchain

NVIDIA's Triton Inference Server 25.12 release notes confirm Blackwell (sm_100) is
officially in the supported-architecture list alongside Hopper, Ada and Ampere
([NVIDIA 25.12 release notes][triton-rel]). On the compiler side, the OpenAI Triton
project added Blackwell codegen in 3.2 and stabilized it through 3.3/3.4, with
direct lowering to the 5th-gen tensor core and TMEM for `tl.dot` in BF16/FP16
([NVIDIA dev blog on Triton/Blackwell][nv-triton-bw]). The minimum stack we should
target:

- CUDA 12.8 (PTX ISA 8.7, first release that exposes the Blackwell TMA
  descriptors and `tcgen05.mma` intrinsics — required for 5th-gen tensor cores).
- PyTorch 2.7 or the `nvcr.io/nvidia/pytorch:25.12-py3` NGC container (Triton
  3.4 + cuDNN 9.5 bundled; known-good on B200, per vLLM's B200 fix post
  ([vLLM B200 fix][vllm-b200])).
- `triton >= 3.4` stable. For warp-spec autotune (`num_consumer_groups`,
  `num_buffers_warp_spec`) you need 3.5 nightly
  ([PyTorch warp-spec blog][pt-warp-spec]).

### 1.2 Known issues vs Hopper

The non-trivial bugs we should expect:

- **FP8 `tl.dot` silently lowers to BF16/FP16 MMA** on consumer Blackwell
  (RTX 5090, sm_120) and *sometimes* on sm_100. Tracked in
  [triton#7188][tr-7188] and [triton#7550][tr-7550]. Performance implication
  only — numerically it's just BF16. **Action:** do not assume FP8 dot gives
  FP8 throughput without inspecting the emitted PTX.
- **Multi-`tl.load` kernels segfault on sm_120 (RTX PRO 6000)** per
  [pytorch#176426][pt-176426]. The report is sm_120-specific and datacenter
  sm_100 B200 is unaffected in our testing reports, but we should smoke-test
  every kernel with a plain compile-only invocation on the target arch before
  benchmarking.
- **Software pipelining regressions for large GEMMs + FP8** on early driver
  stacks; resolved in CUDA 12.8 + driver ≥575. Stay on the NGC container.

### 1.3 Our primitive coverage

We use `tl.load/store` with block pointers, `tl.dot`, `tl.trans`, `tl.where`,
`tl.zeros`, arithmetic, and a static-range inner loop. **None** of these have
Blackwell-specific regressions outside the FP8 `tl.dot` story. `tl.trans` + `tl.dot`
is the standard flash-attn transpose-then-MMA pattern and compiles to the 5th-gen
tensor core with BF16 inputs out of the box.

**One risk specific to heisenberg_d:** our combine does an outer-product update
`c_new += a_outer_b` inside the scan loop. This is a tiny d×d MMA (typically
d ∈ {32, 64, 128}) — well below the Blackwell MMA "sweet spot" of 128×128 per
instruction. For small d we may end up under-utilizing tensor cores and would be
better served by `tl.dot` with padded tiles or by unrolling to vector FMA. That's
a tuning question, not a correctness blocker.

### 1.4 Recommended starting configs

Based on flash-linear-attention's tuned configs (fla is tested on a single GB200
and H100, per its README [fla repo][fla-repo]) and the FA3/FA4 reverse-engineering
write-ups ([Modal FA4][modal-fa4], [FlashAttention-3 paper][fa3]):

| Kernel | BLOCK_T | BLOCK_D | num_warps | num_stages | Notes |
|---|---|---|---|---|---|
| `linear_attn` | 128 | 64 | 8 | 3 | FA2 defaults; stretch to 4 stages if SRAM allows |
| `heisenberg_d` | 64 | 64 | 4 | 3 | Smaller T because state is d×d — L1 cache pressure |
| `unipotent_u4` | 128 | — | 4 | 3 | 6-scalar state, compute-bound, widen T |

For a Blackwell-specific tuning pass, add `num_consumer_groups ∈ {0, 2}` and
`num_buffers_warp_spec ∈ {2, 3, 4}` to the autotune sweep once the correctness
pass is green. The PyTorch blog reports up to 100% speedups from warp-spec on
attention workloads on Hopper/Blackwell ([PyTorch warp-spec][pt-warp-spec]).

### 1.5 FP8 / FP4 recommendation per kernel

| Kernel | Inputs (a, b, q, k) | State accumulator | Output | Justification |
|---|---|---|---|---|
| `linear_attn` | BF16 | **FP32** | BF16 | state = Σ kvᵀ grows with T; FP32 accum is standard (GLA, DeltaNet) |
| `heisenberg_d` | BF16 | **FP32** | BF16 | state = Σ_{i<j} aᵢbⱼᵀ grows as T²/2 worst-case — FP32 mandatory |
| `unipotent_u4` | BF16 | **FP32** for cubic term | BF16 | trilinear Σ_{i<j<k} grows as T³/6 — even more aggressive |

**FP8 in the MMA inputs** is worth an ablation *only* for `linear_attn` after
numerics are pinned down; the numerical-stability literature around Blackwell FP8
is still unfriendly ([FP8-vs-BF16 stability][fp8-stability], [FP8 round-trip][fp8-roundtrip])
and the current `tl.dot` FP8 path is not reliably using FP8 cores anyway
([triton#7188][tr-7188]). **FP4/MXFP4: skip** until Triton's `tl.dot_scaled` actually
emits MXFP4 MMA on sm_100 — as of issue filings into 2026 Q1 it silently falls
back to BF16 ([triton#7550][tr-7550]).

---

## 2. Numerical precision and state stability

### 2.1 The state-growth problem

Our multi-d Heisenberg recurrence is literally `c_{t+1} = c_t + a_t · b_{t+1}ᵀ`
with state `c_t = Σ_{i<j≤t} aᵢ bⱼᵀ`. If `‖a‖, ‖b‖` are O(1) and independent
across steps, ‖c_T‖ grows like O(√T) in the random-walk case and O(T) in the
worst adversarial case — either way unbounded in T. At T=4096 this overflows BF16's
mantissa well before it overflows the exponent, and the matmul read-out `q·c·k`
accumulates the error. U_4's trilinear term is O(T²) worst case.

The existing linear-attention family solves this exactly as the brief hypothesizes:

- **RetNet** uses a fixed per-head scalar decay γ ([Sun 2023][retnet] via
  [Schoelkopf blog][schoelkopf]): `S_t = γ·S_{t-1} + kᵀv`. Geometric decay
  bounds ‖S‖ ≤ ‖kᵀv‖ / (1−γ).
- **GLA / Mamba-2** uses a data-dependent gate `G_t = f(x_t)`:
  `S_t = G_t ⊙ S_{t-1} + kᵀv` ([GLA paper][gla]).
- **Gated DeltaNet** combines the delta rule `(I - β kkᵀ)` with a scalar decay
  `α_t = exp(…)` and an update rate `β_t = σ(…)` ([DeltaNet blog][deltanet-blog]).

### 2.2 Does scalar decay preserve our monoid?

**Yes, scalar decay preserves the H_d monoid.** The current Heisenberg combine is
`(a₁, b₁, c₁) · (a₂, b₂, c₂) = (a₁+a₂, b₁+b₂, c₁+c₂+a₁b₂ᵀ)`. Introduce a
scalar λ ∈ (0,1) and lift to a "weighted" state where each step rescales the
accumulators:

```
(a, b, c, λ) · (a', b', c', λ') = (λ'·a + a',  λ'·b + b',  λ'·c + c' + a·b'ᵀ,  λ·λ')
```

This is still associative (it's the semidirect product of the scalar monoid
(ℝ, ·) with the Heisenberg algebra under the λ-action). A Lean proof of this
should be a ~30-line extension of `HeisenbergD.lean` — the bilinear cross term
now carries a λ factor but commutes through the ring-reshuffle. Same story for
U_4 (add λ and U_4 commute because U_4 acts linearly on itself after scaling).

**Data-dependent gates break associativity naively** — if λ is a function of
state or of position, the combine op no longer re-associates. The standard fix
(used in GLA and Mamba-2) is to make the gate depend *only* on the incoming
input `x_t`, which keeps λ a function on the input sequence and preserves the
"associate on left and right sub-products independently" property modulo a
prefix product of λs. **This is a distinct Lean proof** and should be formalized
before committing to the gate — we'd be replacing our current "pure monoid"
structure with a "monoid with a cocycle" structure. Doable but not free.

**Minimum intervention recommendation:** start with scalar-per-head λ (RetNet
style). It preserves the existing Lean proofs with a trivial extension, bounds
state norm, and matches what RetNet showed was sufficient for pretraining. Graduate
to Mamba-2/GLA style gating only if ablations at 135M-parameter scale show the
scalar decay is the bottleneck.

### 2.3 Numerical tolerance targets

Our fp64 reference has ~1e-12 noise. For BF16 vs reference we should expect:

- **Per-step error** ≈ 2⁻⁷ ≈ 8e-3 relative (BF16 has 7 mantissa bits).
- **Accumulated error after T steps** with FP32 accumulator: O(√T · ε_fp32 · ‖c‖)
  ≈ 1e-4 relative at T=4096.
- **Accumulated error with BF16 accumulator**: O(√T · ε_bf16 · ‖c‖) ≈ 0.5 to
  unbounded — **not acceptable**.

Concrete kernel-vs-reference test targets:

| Setting | atol | rtol | Reject if |
|---|---|---|---|
| FP64 reference vs naive loop | 1e-10 | 1e-12 | diff > 1e-8 |
| BF16 kernel vs FP32 reference, T≤512 | 5e-3 | 5e-3 | diff > 1e-2 |
| BF16 kernel vs FP32 reference, T=4096 | 5e-2 | 5e-2 | any NaN/Inf |

These match the tolerances used in fla's kernel tests. For training, the main
question isn't the absolute tolerance but whether gradients stay well-behaved —
the BF16 flash-attention literature ([Why Low-Precision Training Fails][bf16-fail])
shows that biased rounding in BF16 softmax can cause catastrophic loss explosion
after thousands of steps. We likely inherit some of this via the read-out `qᵀ·c·k`,
but without the softmax the primary failure mode is overflow not bias. **Mitigation:**
renormalize the state with an RMSNorm every layer (standard in GLA/DeltaNet) and
keep the accumulator in FP32.

For inference, BF16 end-to-end with occasional FP32 renormalization is the
industry standard (Gated DeltaNet, GLA all ship this). FP8 inference is
premature.

---

## 3. Evaluation methodology

### 3.1 Which base model

Decision matrix:

| Model | Params | License | Ctx | Linear-attn variants | Verdict |
|---|---|---|---|---|---|
| SmolLM2-135M | 135M | Apache 2.0 | 8K | — | **iteration-speed workhorse** ([SmolLM2 paper][smollm2], [HF card][smol135]) |
| SmolLM2-360M | 360M | Apache 2.0 | 8K | — | intermediate checkpoint |
| Qwen3-0.6B-Base | 0.6B | Apache 2.0 | 32K | — | small, recent, Apache |
| Qwen3-1.7B-Base | 1.7B | Apache 2.0 | 32K | — | **mid-scale target** ([Qwen3 blog][qwen3]) |
| Qwen3-4B-Base | 4B | Apache 2.0 | 32K | — | final signal |
| Qwen3-Next-80B-A3B | 80B MoE / 3B active | Apache 2.0 | 256K-1M | already hybrid Gated DeltaNet 75/25 | too big, but *most informative* teacher — skip unless we have 8×B200 budget ([vLLM Qwen3-Next][vllm-qwen3n]) |
| Qwen3.5 (0.8B, 2B, 4B, 9B) | — | Apache 2.0 | 262K | Gated DeltaNet hybrid inherited | newer (March 2026), also viable ([Qwen3.5 coverage][qwen35]) |

**Pick:** SmolLM2-135M for synthetic benchmarks and the first distillation dry
run. Qwen3-1.7B-Base for the main "does this actually work on real LM perplexity
and downstream evals" data point. Qwen3-4B-Base only after the 1.7B result is
decisive. Qwen3-Next is tempting because the teacher *already* uses Gated DeltaNet
in 75% of layers — we could swap only those layers for our heisenberg_d cell and
keep the 25% softmax — but 80B at any precision is out of scope for an initial
evaluation.

### 3.2 Layer replacement strategy

From the distillation literature:

- **MambaInLlama** (NeurIPS 2024) ran full-replace and 50%-hybrid; 50%-hybrid
  matched or beat full-attention baselines on chat benchmarks in 20B tokens,
  5 days on 8×A100 ([paper][mil-paper], [code][mil-code]).
- **LoLCATs** (Hazy, ICLR 2025) did full-replace on all attention layers with
  attention-transfer + LoRA; 0.2% trainable params, 40M tokens, ~5 hours on
  one A100 for an 8B model ([blog][lolcats-p1], [part 2][lolcats-p2]).
- **RADLADS** (2025) distills to RWKV-variant linear decoders for Qwen2.5
  7B/32B/72B; 350-700M tokens, <\$2000 to convert 72B, ~7 hours for 7B on
  8×MI300X ([paper][radlads]).
- **Qwen3-Next** ships 75% Gated DeltaNet + 25% softmax attention natively
  ([vLLM Qwen3-Next][vllm-qwen3n]).
- **AMD-HybridLM** and Samba-series hybrids converge on roughly 50-75% linear +
  the remainder softmax as the sweet spot.

**Pick:** hybrid 75% our cell + 25% softmax, matching Qwen3-Next's ratio. The
consensus in 2024-2026 is that pure linearization loses 3-10 perplexity points
on real LM and drops MMLU 5-15 points, and that the remaining softmax layers
carry most of the in-context recall capability. Full replace remains a useful
ablation but isn't the production recipe.

### 3.3 Distillation vs from-scratch

From-scratch at 30M is cheap and gives clean architecture-signal — do it for
synthetic tasks. Distillation is the right recipe at 135M+. Specifics:

**Two-stage LoLCATs-style recipe:**

1. **Stage 1 — attention transfer.** Freeze the teacher. Replace 75% of the
   attention layers with our cell (heisenberg_d + scalar decay). Loss: MSE
   between the teacher's layer output and the student's layer output, per
   replaced layer. Dataset: 40M-100M tokens of FineWeb-Edu or OpenWebMath
   (Hedgehog and LoLCATs both use mixtures of pre-training data). Budget:
   LoRA on the MLP side; ≤0.5% params trainable.

2. **Stage 2 — LoRA SFT.** On a downstream / instruction dataset (e.g.
   SlimOrca, OpenHermes) with KL-divergence to the teacher's logits. 200-400M
   tokens. 

**Loss details:**
- Stage 1: `‖h_student - h_teacher‖²` per replaced layer; optionally add a
  feature-map alignment loss as in Hedgehog ([Hedgehog paper][hedgehog]) if
  our cell has a learnable feature map (ours currently does not — the bilinear
  a/b are linear projections of the input).
- Stage 2: standard KD KL `KL(p_teacher ‖ p_student)` at temperature 2.0 plus a
  label-smoothed CE with the ground-truth token.

**Gotchas from the literature:**
- **Normalization mismatch.** Replacing attention often breaks the model's
  residual-stream scale; LoLCATs fixes this with an extra RMSNorm post-cell.
  Plan to include it.
- **Position encodings.** Teacher uses RoPE — we should either preserve RoPE on
  the softmax-retained layers and drop it on our cell (RetNet-style: decay
  replaces positional info) or feed rotated q/k into our cell. RetNet's
  per-head λ *is* a learned positional prior; when we add scalar decay this
  becomes natural.
- **Tokenizer.** Must match teacher exactly. Use the teacher's tokenizer file.
- **Layer index selection for hybrid.** MambaInLlama found that *preserving
  attention in the middle layers and replacing at the start/end* worked better
  than uniform stride. Plan: ablate stride-4, stride-8, and "keep middle N"
  configurations.

### 3.4 Concrete evals — real-world benchmarks

| Benchmark | Tests | Why it matters | Expected signal |
|---|---|---|---|
| **PPL on WikiText-103, PG19, The Pile (held-out slice)** | LM quality | Baseline sanity | Δ ≤ 0.5 vs teacher = good |
| **MMLU, HellaSwag, ARC-E/C, PIQA, WinoGrande** | zero-shot downstream | Standard LLM eval suite | ≤ 3 pp drop vs teacher = good |
| **RULER (1k→128k context)** | long-context retrieval | Where pure linear-attention architectures collapse ([RULER paper][ruler]) | Pure-linear drops >30 pp at 16k+; hybrid with our cell should be within 5-10 pp |
| **NIAH (Needle-in-a-Haystack, gkamradt)** | single-needle retrieval | Gated DeltaNet territory — our Heisenberg state *should* do well because bilinear cross-terms encode positional binding | Hybrid ≥ 80% pass at 16k |
| **MQAR (Zoology)** | multi-query associative recall | Key separator between architectures ([Zoology blog][zoology], [paper][zoology-paper]) | Our heisenberg_d expected to outperform pure linear attention at matched state size; this is the *core* experimental claim |
| **BABILong** | long-context QA | Orthogonal to NIAH — tests reasoning not just retrieval | |

### 3.5 State-tracking evals (the bilinear-RNN strength)

This is where the novelty story gets proven. Linear RNNs with positive-only
diagonal state transitions cannot learn parity; introducing negative eigenvalues
fixes it ([Unlocking State-Tracking][neg-eig]). Csordas et al.'s bilinear RNN
work (2505.21749) shows bilinear transitions extend the expressivity envelope
([Csordas 2505.21749][csordas]).

Our cells should be tested on:

- **Parity** over {0,1}^T: linear-attention baseline fails; our U_4 cell should
  pass because the trilinear ordered term encodes products of three odd bits.
- **Modular addition mod p** for p ∈ {3, 5, 7}: diagonal LRNN fails; block-
  diagonal 2D LRNN passes only for p=2. Our heisenberg_d with d=p should
  embed Z/p as a quotient of its bilinear accumulator.
- **Random state machine** (|S|=16, |Σ|=8): general state-tracking.
- **Dyck-n balance** (n=2, 3): classical formal-language test — pure softmax
  attention nails it, pure linear fails at deep nesting ([Dyck and attention][dyck-attn]).

These are **cheap to run** — 30M parameter models, a few GPU-hours each. They
should run on the first pass.

### 3.6 Cheaper first-pass evals (before touching 1B+)

Proposed 30M-parameter synthetic benchmark suite. Train from scratch, 1-2
GPU-hours each on a single B200:

1. **MQAR at small scale** (Zoology config): d_model=128, n_pairs=20k, seq=512,
   vocab=8192. Compare (a) pure softmax, (b) pure linear_attn, (c) pure
   heisenberg_d, (d) pure unipotent_u4 at matched state size. This is the
   clean "does the bilinear cross-term actually buy us recall?" experiment.
2. **Parity + mod-p addition** sequence labelling at length 1024. Same four
   architectures. Directly probes the Lean-proved trilinear structure in U_4.
3. **Multi-hop induction heads** (In-Context-Learning toy, Olsson et al.
   formulation via [induction heads][induction-heads]): 2-hop and 3-hop needle
   tasks. Tests whether our cross-term enables the "copy-then-match" circuit
   linear attention struggles with.

Success criterion for proceeding to distillation: **(2) or (3) shows a ≥ 10 pp
advantage for heisenberg_d or unipotent_u4 over linear_attn at matched parameter
count.** If not, the cell is probably a write-up but not a deploy.

---

## 4. Compute budget and sequencing

### 4.1 GPU-hour estimates

Grounded in the published numbers:

**(a) Reference training at 30M params on toy dataset.**
From scratch, ~500M tokens (Chinchilla-optimal would be 600M at 20 tok/param
[Hoffmann 2022][chinchilla]; we go a bit under since it's synthetic).
BF16, B200, batch 256×1024, ~50k steps. Estimated throughput ~150k tok/s on
one B200 for a 30M-param dense model. **~1 GPU-hour per toy run.** Full sweep
(4 architectures × 3 synthetic tasks) = **~12 B200-hours**.

**(b) Distillation into cell at 1-3B scale.**
LoLCATs: "~5 hours on a 40GB A100 for 8B attention transfer" — a B200 is
~3× faster and has 180GB, so attention transfer at 1.7B should be <2 B200-hours.
Add LoRA SFT on 200-400M tokens: ~6-10 B200-hours. Budget **~20 B200-hours for
Qwen3-1.7B distillation, end-to-end**. Qwen3-4B would be ~60 B200-hours (ratio
from RADLADS 7B→32B scaling).

MambaInLlama: "<5 days on 8×A100 for Llama3-8B hybrid" = ~960 A100-hours.
Scaling down to 1.7B linearly and up from A100→B200 (×3), this is another
independent estimate of ~60-80 B200-hours for a *heavy* distillation; LoLCATs'
~20 B200-hour estimate assumes much more aggressive LoRA freezing.

**(c) Real LM eval.**
MMLU + HellaSwag + ARC + PIQA + WinoGrande + RULER + NIAH + MQAR on Qwen3-1.7B-
scale student: ~4-6 B200-hours for the full harness (lm-eval-harness + the
long-context suites). Add another ~4 hours for BABILong.

### 4.2 First end-to-end eval budget

For Qwen3-1.7B student with heisenberg_d + scalar decay, 75% hybrid:
- Synthetic sanity (MQAR + parity + mod-p at 30M): ~12 B200-hours.
- Attention transfer stage 1: ~2 B200-hours.
- LoRA SFT stage 2: ~10 B200-hours.
- Eval harness: ~8 B200-hours.
- Buffer for failed runs & autotune: ~8 B200-hours.

**Total: ~40 B200-hours for first complete end-to-end signal.**
At ~$3.79/hr on-demand per Lambda Labs B200 pricing ([B200 pricing][b200-price])
or ~$2.25/hr reserved: **$90-150 per end-to-end eval**.

### 4.3 Sequencing on an 8×B200 week (≈1344 GPU-hours)

ROI-maximizing order (early kill for bad ideas, parallel where independent):

1. **Day 1 morning (8 GPU-hours):** Port kernels to Blackwell — compile-smoke,
   correctness against the pytorch reference at several (T, d, B) configs.
   No tuning yet. Verify BF16 accumulator-in-FP32 numerics against the FP64
   reference (§2.3 table).
2. **Day 1 afternoon (24 GPU-hours):** Autotune sweep over `BLOCK_T,
   BLOCK_D, num_warps, num_stages, num_consumer_groups` for all three kernels,
   one shape each. Collect peak TFLOPS / memory bandwidth. Separate B200 per
   kernel, in parallel.
3. **Day 2 (60 GPU-hours across 8 GPUs in parallel):** 30M synthetic suite.
   Four architectures × three tasks × two seeds = 24 runs, ~1.5 hours each
   on 1 B200 ≈ 36 core-hours sequential, fits in one day with 8-way parallel.
   **This is the gate.** If heisenberg_d or unipotent_u4 doesn't clearly beat
   linear_attn on parity/mod-p/MQAR, stop here and write up the kernels +
   Lean proofs as a "we identified a non-working cell" negative result.
4. **Days 3-4 (120 GPU-hours):** Distill heisenberg_d-hybrid into
   SmolLM2-135M. Use attention-transfer + LoRA. Evaluate PPL + MMLU. Fast
   iteration. Two ablations in parallel: (a) scalar decay λ, (b) no decay.
5. **Days 5-6 (300 GPU-hours):** Distill into Qwen3-1.7B-Base. Three
   configs in parallel: 50% hybrid, 75% hybrid, 100% replaced. LoRA only.
   Full eval harness at end.
6. **Day 7 (300 GPU-hours buffer):** Pick the winning config, scale to
   Qwen3-4B-Base if (and only if) 1.7B showed ≤5 pp MMLU drop. Otherwise use
   the day to tighten the numerical-stability story (FP32 accumulator ablations,
   state-norm tracking, longer-context extrapolation up to 32K).

If we only have a single 8×B200 node for a *single* week and the goal is
"publishable result," budget roughly:
- 15% compute → kernel tuning
- 30% compute → synthetic benchmarks (the novelty proof)
- 40% compute → distillation at 1.7B
- 15% compute → evals + writeup runs

---

## Sources

- [NVIDIA Triton Inference Server 25.12 release notes][triton-rel]
- [NVIDIA dev blog: OpenAI Triton on Blackwell][nv-triton-bw]
- [vLLM B200 compatibility fix][vllm-b200]
- [PyTorch warp-specialization blog][pt-warp-spec]
- [triton-lang/triton#7188 — FP8 tl.dot fallback][tr-7188]
- [triton-lang/triton#7550 — tl.dot_scaled BF16 fallback on 5090][tr-7550]
- [pytorch/pytorch#176426 — sm_120 multi-load segfault][pt-176426]
- [fla-org/flash-linear-attention][fla-repo]
- [Modal reverse-engineering Flash Attention 4][modal-fa4]
- [FlashAttention-3 paper][fa3]
- [Hedgehog & the Porcupine][hedgehog]
- [LoLCATs blog part 1][lolcats-p1], [part 2][lolcats-p2]
- [MambaInLlama paper][mil-paper], [code][mil-code]
- [RADLADS paper][radlads]
- [Qwen3 blog][qwen3]
- [Qwen3-Next / vLLM coverage][vllm-qwen3n]
- [Qwen3.5 coverage][qwen35]
- [SmolLM2 paper][smollm2], [135M card][smol135]
- [GLA paper][gla]
- [Gated DeltaNet blog][deltanet-blog]
- [RetNet intro via Schoelkopf][schoelkopf]
- [Hoffmann 2022 Chinchilla][chinchilla]
- [Why Low-Precision Transformer Training Fails][bf16-fail]
- [FP8 vs BF16 stability tradeoff][fp8-stability]
- [Zoology blog][zoology], [Zoology paper][zoology-paper]
- [RULER paper][ruler]
- [Unlocking State-Tracking via Negative Eigenvalues][neg-eig]
- [Csordas Bilinear RNN (2505.21749)][csordas]
- [Induction heads (Olsson et al.)][induction-heads]
- [Dyck languages and attention][dyck-attn]
- [B200 cloud pricing comparison][b200-price]

[triton-rel]: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-25-12.html
[nv-triton-bw]: https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/
[vllm-b200]: https://medium.com/@imen.selmi/fix-for-vllm-incompatibility-on-b200-keep-nvidia-pytorch-use-newer-vllm-8e0b3a0a4d16
[pt-warp-spec]: https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/
[tr-7188]: https://github.com/triton-lang/triton/issues/7188
[tr-7550]: https://github.com/triton-lang/triton/issues/7550
[pt-176426]: https://github.com/pytorch/pytorch/issues/176426
[fla-repo]: https://github.com/fla-org/flash-linear-attention
[modal-fa4]: https://modal.com/blog/reverse-engineer-flash-attention-4
[fa3]: https://www.together.ai/blog/flashattention-3
[hedgehog]: https://arxiv.org/abs/2402.04347
[lolcats-p1]: https://hazyresearch.stanford.edu/blog/2024-10-14-lolcats-p1
[lolcats-p2]: https://hazyresearch.stanford.edu/blog/2024-10-14-lolcats-p2
[mil-paper]: https://arxiv.org/abs/2408.15237
[mil-code]: https://github.com/jxiw/MambaInLlama
[radlads]: https://arxiv.org/abs/2505.03005
[qwen3]: https://qwenlm.github.io/blog/qwen3/
[vllm-qwen3n]: https://blog.vllm.ai/2025/09/11/qwen3-next.html
[qwen35]: https://stable-learn.com/en/qwen35-native-multimodal-agent-model/
[smollm2]: https://arxiv.org/html/2502.02737v1
[smol135]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
[gla]: https://arxiv.org/pdf/2312.06635
[deltanet-blog]: https://sustcsonglin.github.io/blog/2024/deltanet-2/
[schoelkopf]: https://haileyschoelkopf.github.io/blog/2024/linear-attn/
[retnet]: https://arxiv.org/abs/2307.08621
[chinchilla]: https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf
[bf16-fail]: https://arxiv.org/html/2510.04212v1
[fp8-stability]: https://arxiv.org/html/2411.08719v1
[fp8-roundtrip]: https://arxiv.org/html/2405.18710v2
[zoology]: https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis
[zoology-paper]: https://arxiv.org/abs/2312.04927
[ruler]: https://arxiv.org/abs/2404.06654
[neg-eig]: https://arxiv.org/abs/2411.12537
[csordas]: https://arxiv.org/html/2505.21749
[induction-heads]: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
[dyck-attn]: https://aclanthology.org/2020.findings-emnlp.384.pdf
[b200-price]: https://getdeploying.com/gpus/nvidia-b200

---

Written to `/Volumes/git/state-dep-parallel/EVAL_PLAN.md`. Recommended initial
student model size: **135M parameters** (SmolLM2-135M). First end-to-end eval
GPU budget: **~40 B200-hours**.

---

## 5. Refined plan after Phase 1-8 architectural exploration

After the 2× RTX 5090 dev session in April 2026 (full writeup in
[`RESULTS.md`](RESULTS.md)), the plan has sharpened considerably. The
findings:

- **Two architecturally distinct walls** identified mechanistically
  (Grazzi TC⁰ wall + Zoology recall wall).
- **Single-cell unification fails** (rotation conjugation breaks the
  fixed-frame structure delta-rule recall depends on; tested 4 cells).
- **Hybrid layer stack `[ortho, deltanet, ortho, deltanet]`** solves
  both (parity at T=512 = 100 %, induction = 100 %, 0.94 M params).

### Status of original §4.3 sequencing

| Original step | Status |
|---|---|
| Day 1 morning: kernel port + smoke tests on sm_120 | ✓ done |
| Day 1 afternoon: autotune sweep (skipped — used PyTorch refs to ship faster) | partial |
| Day 2: 30M synthetic suite (parity, MQAR, mod-p, Dyck) | ✓ parity done at 1M params; MQAR partially done; mod-p, Dyck not run |
| Day 3-4: SmolLM2-135M distillation | not started |
| Day 5-6: Qwen3-1.7B distillation | not started |
| Day 7: Qwen3-4B / negative-result writeup decision | superseded — the empirical findings reframe the project |

### Refined priority list (replaces §4.3 day 3-7)

The architectural exploration produced a sharper research story than
"distill SmolLM2 with `heisenberg_d` hybrid". The four work items below
cover both *empirical validation of the wall framing* (high priority,
new) and *the original distillation arc* (still relevant for scale).

#### 5.1 Modular counting mod p > 2 *(highest priority)*

The cleanest theoretical prediction of the wall framing: `Z_p ⊂ SO(2)`
for any p, so SO(n)-scan and our hybrid solve any modular counting,
while DeltaNet (even with `allow_neg_eigval=True`) only reaches `Z_2`
because Householder reflectors have eigenvalue ±1 along k.
**DeltaProduct's Householder-product eigenvalues are also ±1**, so it
should also fail mod-p for p > 2.

If this prediction holds at our 1M-param scale, it's the cleanest
single-task separator we have between hybrid and DeltaProduct.

- **Task**: predict `(x₁ + x₂ + … + x_t) mod p` at every position, for
  `p ∈ {3, 5, 7}`.
- **Architectures**: linear, heisenberg, ortho, rotconj, deltanet,
  deltaproduct (need to install / wrap), mamba2, hybrid.
- **Setup**: T=128, vocab=p, n_layers=4, d_head=32, 5000 steps.
- **Pass criterion**: hybrid solves all p ∈ {3, 5, 7}; DeltaNet /
  DeltaProduct fail for p > 2 (at chance = 1/p).
- **Compute**: ~2 GPU-hours per (arch × p) on a single 5090. Total
  ~50 GPU-hours.

#### 5.2 S₅ word problem *(high priority)*

The canonical NC¹-complete benchmark. SO(n) for n≥3 contains A₅, so
in-principle solvable by ortho / hybrid. The empirical question is
whether SGD finds non-abelian rotations or stays in the abelian
subgroup.

- **Task**: from [Grazzi et al. ICLR'25](https://arxiv.org/abs/2411.12537)
  + [DeltaProduct (Yang et al. NeurIPS'25)](https://arxiv.org/abs/2502.10297)
  benchmark suite — sequence of S₅ generators, predict whether
  composition is identity.
- **Architectures**: same set as 5.1.
- **Pass criterion**: any architecture solves > 95 % at T=512. Per
  Grazzi et al., DeltaNet+`allow_neg_eigval` partially solves; we
  expect hybrid to match or beat.
- **Compute**: ~30 GPU-hours.

#### 5.3 Lean: incompatibility theorem *(parallel work, no GPU)*

Formalise the empirical observation that single-cell rotation × delta
is impossible:

- **Theorem (sketch)**: let `M = (M_t)_{t=1}^T` be a sequence of
  transitions. If each `M_t = (I − β_t k_t k_tᵀ) O_t (I − β_t k_t k_tᵀ)ᵀ`
  with `O_t ∈ SO(n)` and `k_t` data-dependent, then there is no
  fixed-basis decoding `q · c_t` that recovers `(q · k_p)·v_p` for an
  arbitrary "stored" pair (k_p, v_p) at p < t.
- **Module**: new `StateDep/IncompatibilityTheorem.lean`.
- **Effort**: ~50-100 lines of Lean. Group-theoretic argument plus a
  standard inner-product calculation.

#### 5.4 SmolLM2-135M distillation with hybrid stack

The original §3.3 plan, but using the refined hybrid architecture
instead of pure heisenberg_d:

- **Hybrid spec**: replace 3 of every 4 attention layers in SmolLM2 with
  alternating ortho + DeltaNet, keeping every 4th as softmax. Mirrors
  Qwen3-Next's 75 % linear / 25 % softmax with finer-grained
  specialisation.
- **Recipe**: LoLCATs-style two-stage (attention transfer + LoRA SFT),
  per §3.3.
- **Compute**: per §4.2 estimate, ~12 B200-h ≈ 60 wall-clock-h on 2×
  5090. Doable but depends on whether 5.1 + 5.2 give signal.
- **Gating**: only proceed if (a) modular counting (5.1) shows clean
  separation hybrid ≫ DeltaProduct, *or* (b) S₅ (5.2) shows hybrid ≥
  DeltaProduct. If neither, the project is the formal+kernel+
  diagnostic contribution and distillation is overkill.

### Compute budget on 2× RTX 5090

Total budget for 5.1 + 5.2 + 5.3 ≈ ~80 GPU-hours of synthetic
experiments + Lean work. 5.4 distillation is another ~60 wall-clock
hours if pursued.

### Why this sequence

The wall-framing result is publishable on its own (formal Lean +
incompatibility theorem + minimal hybrid). 5.1 and 5.2 are the cleanest
empirical separators that distinguish *our* hybrid framing from the
existing fla / DeltaProduct line. 5.4 is the scale-up that proves the
finding generalises to LLMs; only worth pursuing once 5.1 / 5.2 give
signal.

---

## 6. Post-hybrid plan: hunting failure modes + selective rotation

After Phases 4-12 produced hybrid v2 — competitive at LM scale (1.19×
DeltaNet on TinyStories) and dominant on continuous-angle modular
arithmetic (mod-5 T=512: 100 % vs DeltaNet+negeig's 9 %) — the next
work is **systematically finding where hybrid fails and adding the
missing primitive.** Documented in `NEXT_DIRECTIONS.md` "Next iteration:
hunting hybrid's failure modes".

### Stress tests, in priority order

1. **Long-T S₅ word problem (T=512)**. Quickest test of whether
   SO(n)-rotation generalises to long-T non-abelian state-tracking.
   At T=128 deltanet_negeig wins; T=512 reveals which architecture
   actually scales.
2. **Selective copy** (Mamba paper). Sequence with mostly noise + a
   few signal tokens. Predict signal tokens in order. Predicts hybrid
   *fails* because ortho rotation is always-on with no skip mechanism.
3. **Multi-class mod-p with very large p (p ∈ {11, 13, 17})**. Tests
   the *optimisation regime* of SO(n) rather than expressivity — at
   small p the rotation angle 2π/p is easy; at p=17 (~21°) it's not.
4. **Stack-with-types** (deeper Dyck variants). Tests whether
   rotation can encode per-depth type memory or whether DeltaNet's
   recall is essential for matched-bracket type tracking.
5. **Long-context MQAR (T=4096+)**. Push recall ability past
   DeltaNet's known limit. If both fail, we need stronger memory.

### The likely missing primitive: selective rotation

Mamba2-style `Δ_t = f(x_t)` gating of the rotation magnitude:

```
λ_t = σ(W_λ · x_t) ∈ (0, 1)
R_t = exp(λ_t · skew(W_skew · x_t)) · R_{t-1}
```

`λ_t = 0` ⇒ identity rotation (skip token); preserves the rotation
primitive (still Grazzi-clean), parallel-scannability (cumulative
product unchanged), and the Lean associativity proof (semidirect
product structure). ~30-line extension to `OrthogonalScanAttention`;
no Triton kernel change.

Should fix selective-copy failure and improve LM PPL by allowing the
ortho layers to ignore irrelevant tokens.

### Recommended sequence (after current overnight runs land)

| Step | Task | Compute |
|---|---|---|
| 6.1 | S₅ T=512 sweep (deltanet vs hybrid v2) | ~30 min |
| 6.2 | Selective copy sweep | ~30 min |
| 6.3 | Implement `use_selective_lambda` in OrthogonalScanAttention | ~1 hour Python |
| 6.4 | Re-run mod-p / parity / induction with selective ortho (preserve wins) | ~1 hour |
| 6.5 | Re-run TinyStories + Python LM PPL with selective ortho | ~1 hour |
| 6.6 | Triton backward kernel for matmul-scan | ~2-3 hours |
| 6.7 | Distill from coding teacher (DeepSeek-Coder / StarCoder); HumanEval / MBPP | multi-day |

Total compute for 6.1-6.5 (the architectural iteration): ~4-5 hours
on 2× RTX 5090. 6.6 is engineering. 6.7 is the "demonstrate at LLM
scale on coding tasks" final push.

### Why this sequence

6.1-6.2 surface the next failure mode; 6.3 builds the fix; 6.4-6.5
prove the fix doesn't regress; 6.6 unlocks 6.7 by closing the
remaining wall-clock gap. The user's stated goal is "assess if this
works on LLM coding tasks", which 6.7 directly addresses.
