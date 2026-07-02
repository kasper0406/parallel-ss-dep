> ⚠️ **METROLOGY CORRECTIONS (audited 2026-07-01, `SESSION_FINDINGS.md`).** Read first.
> 1. **The constant-MEMORY result is SOLID and architecture-determined** (8.2 MiB flat state, 6.45×
>    peak advantage @131k, ≈624× on the state) — for the LEAN config only (the production WM stack has
>    an O(T) decode buffer). This is the robust part.
> 2. **The "1.7–1.8× batched throughput win" is EAGER-vs-EAGER** (both HF/naive, no
>    compile/paged-attention/continuous-batching). Throughput is exactly what vLLM optimizes for the
>    transformer → this number is unlikely to survive vs a production serving stack. Label it "eager
>    reference throughput"; the robust win is MEMORY / OOM-frontier (B vs B×L scaling), not tok/s.
> 3. **`forward_step` is NOT "logit-tested" for the benched config.** `test_incremental_decode.py` runs
>    on the PRODUCTION 10L×896d use_memory RL ckpt (not the lean linearized config), the threshold is
>    **argmax ≥14/16 under ~0.19 bf16 drift** (not exact logit equivalence, not "≥15/16"), and
>    **`prefill_state_only()`** (used for ALL batched numbers) is UNTESTED. Cost claims are unaffected
>    (cost is state-value-independent); the argmax-equivalence of the exact config is asserted. TODO:
>    add equivalence tests pinning `prefill`/`forward_step`/`prefill_state_only` on the benched ckpt.
> 4. Single-stream latency FAVORS the transformer (~2.3× faster/token until ~131k); "dramatic" applies
>    to the MEMORY/reach axis only. Bounded memory is also achievable by a windowed/streaming/paged
>    transformer without switching backbones.

> **2026-07-01 correction — bench autocast asymmetry removed.** OURS' timed decode
> loops (`bench_ours`, `bench_ours_batched` — prefill, warmup, and the timed `G`-step
> loop) were each wrapped in an outer `torch.autocast("cuda", bf16)`; the transformer
> arm's timed loop was not. Since `load_ours` already casts the model to bf16, this was
> suspected pure dispatch overhead charged only to ours (`SESSION_FINDINGS.md` audit).
> Verified numerically first: `RMSNorm`'s reduction (`x.pow(2).mean(-1).rsqrt()`) is one
> of the ops autocast promotes to fp32, so removing the wrapper does change raw logits
> (max|Δ| up to ~0.44 over a 16-step forward_step probe from a shared cloned prefill
> cache) — NOT a pure no-op. However, greedy argmax token sequences over that same
> 16-step probe were IDENTICAL with vs without the wrapper (both a self-fed greedy
> trajectory from a structured prompt and a fixed-random-continuation trajectory);
> mismatches only appeared under a maximally-degenerate fully-i.i.d.-random
> prefill+continuation probe with near-tied logits, a regime `decode_bench.py` itself
> doesn't exercise (it feeds a constant token every step; correctness was never its
> concern). Per the task's own equivalence criterion (argmax-sequence match) this
> counts as equivalent, so the wrapper was removed from all 6 call sites (prefill /
> warmup / timed loop × 2 functions); the inner narrow autocast inside
> `_FlaWrapper.forward_step` (`experiments/layers.py`, a safety net for non-bf16
> callers) is untouched. **Re-measured ours ms/tok (bf16, gen=64, warmup=8, median of 3
> reps each):** L=512: 14.53 → 13.33 ms/tok; L=8192: 14.50 → 13.51 ms/tok; L=32768:
> 14.48 → 13.40 ms/tok — a consistent **~7–8% reduction** (~1.0–1.2 ms/tok), constant
> across L as before (the O(1)-latency shape is unchanged, just less inflated). The
> single-stream "~2.3× slower than the transformer" figure in point 4 above should be
> read as ~2.1× post-fix (transformer ~6.3 ms/tok in the same probe, unaffected by
> this change); the qualitative verdict is unchanged.

# Decode-cost benchmark — testing the O(1)/constant-memory decode moat (2026-06-30)

Honest single-stream (batch=1, bf16) autoregressive-decode benchmark of the
project's core "bounded-state DeltaNet decodes at ~O(1)/token with constant
memory, vs a transformer's O(T) KV-cache growth" claim. Previously this was
only a code comment ("~6.5ms"); the flagship probe ran full-forward and
declared decode cost out of scope.

Harness: `experiments/decode_bench.py`. Raw: `checkpoints/decode_bench/results.json`.
Run: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python experiments/decode_bench.py`

## Step 0 — does a true O(1) incremental decode exist? YES (for the lean config)

`TinyLM.prefill()` + `TinyLM.forward_step()` (`experiments/model.py`) implement
real state-passing incremental decode:
- `prefill(prompt)` runs ONE full forward, building an FLA `Cache` of the
  per-layer **bounded recurrent state** (DeltaNet `recurrent_state (1,15,64,64)`
  + 3 short-conv states per layer).
- `forward_step(tok, cache)` processes **one** new token through FLA's
  `fused_recurrent` kernel reading that state — it does **NOT** re-process the
  prefix.

Verified directly: the recurrent-state tensors are **byte-identical in size at
L=512 and L=32768** (8.20 MiB total, 32 layers). The repo's own
`experiments/test_incremental_decode.py` pins logit-equivalence to the full
forward (argmax match >=15/16). This is the lean linearized config built by
`linearize_smollm2.build_bare_deltanet` (arch=deltanet, feedback=none, no
WM/PKM/gate/FiLM) — exactly the clean case.

Note: CLAUDE.md/AGENTS.md still say `forward_step` is "shipped but not yet wired
into `TinyLM.forward_step`" — that is **stale**; it is wired and tested. What
remains unwired is only the FiLM-K3 + WorkingMemory **production thinking stack**
at decode (xattn raises `NotImplementedError`; WM uses a growing buffer). For
the linearized deployment target the O(1) path is fully realized.

## The benchmark

OURS = `checkpoints/linearize/linearized_stage3.pt` (32L x 960d DeltaNet,
449.2M params, lean). BASELINE = `HuggingFaceTB/SmolLM2-360M` (HF transformer,
361.8M params, `use_cache=True` growing KV cache). bf16, eager (no compile) for
both, batch=1, prefill L then generate G=64; steady-state = median ms over the
64 decode steps after 8 warmup steps; peak = decode-phase peak
(`max_memory_allocated`, reset after prefill so the O(L) prefill transient does
not mask the decode footprint). "cache" = recurrent state (ours, measured
directly) / KV cache (transformer).

| L | ours ms/tok | ours peak MiB | ours state MiB | xf ms/tok | xf peak MiB | xf KV MiB | speedup (xf/ours) | mem ratio (xf/ours) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256    | 14.40 | 919.3 | 8.20 | 6.26 |  747.2 |   10.1 | 0.43x | 0.81x |
| 512    | 14.42 | 919.3 | 8.20 | 6.30 |  757.4 |   20.1 | 0.44x | 0.82x |
| 1024   | 14.36 | 919.3 | 8.20 | 6.32 |  777.7 |   40.1 | 0.44x | 0.85x |
| 2048   | 14.40 | 919.3 | 8.20 | 6.26 |  825.3 |   81.2 | 0.44x | 0.90x |
| 4096   | 14.43 | 919.3 | 8.20 | 6.34 |  901.5 |  162.8 | 0.44x | 0.98x |
| 8192   | 14.34 | 919.4 | 8.20 | 6.35 | 1064.4 |  321.8 | 0.44x | 1.16x |
| 16384  | 14.40 | 919.4 | 8.20 | 6.39 | 1387.7 |  640.2 | 0.44x | 1.51x |
| 32768  | 14.42 | 919.6 | 8.20 | 6.41 | 2037.3 | 1280.3 | 0.44x | 2.22x |
| 65536  | 14.45 | 919.8 | 8.20 | 8.69 | 3337.5 | 2560.6 | 0.60x | 3.63x |
| 131072 | 14.44 | 920.3 | 8.20 | 13.92 | 5938.0 | 5121.1 | 0.96x | 6.45x |

## Verdict

**Constant-memory moat: REAL and unambiguous.** Ours' decode footprint is flat —
peak ~919 MiB and recurrent state **8.20 MiB at every context length**, including
131k tokens. The transformer's KV cache grows strictly linearly (10 -> 5121 MiB)
and is unbounded. The memory advantage crosses 1x at ~4k tokens, reaches **2.2x
at 32k and 6.45x at 131k**, and keeps widening; the transformer would eventually
OOM where ours never does.

**O(1) latency moat: REAL in shape, but the constant is large/un-optimized.**
Ours' ms/token is flat (~14.4 ms, <1% variance over a 512x context range) =
genuine O(1) per-token compute. The transformer's ms/token is flat-then-growing
(6.3 ms to 32k, then 8.7 ms at 65k, 13.9 ms at 131k) = O(T) attention. BUT in
absolute terms ours is **~2.3x SLOWER per token** until the crossover at ~131k
tokens, because: (a) ours has more params (449M vs 362M — DeltaNet attention is
heavier than RoPE-GQA); (b) the FLA `fused_recurrent` T=1/batch=1 kernel is
launch-overhead-bound and un-optimized (no CUDA graphs / no `torch.compile` on
`forward_step`), while HF's softmax-attention decode at batch=1 short-context is
highly optimized and memory-bandwidth-cheap. So the **latency** win only converts
to a wall-clock win at very long single-stream context (>~131k); the **memory**
win arrives far earlier (>4k) and is unbounded.

## Honest caveats

- **Path measured = lean linearized config** (feedback=none, no WM/PKM/gate). This
  is the deployment target of the linearization direction and the clean O(1) case.
  It is NOT the production thinking stack (FiLM-K3 + WorkingMemory): at decode the
  generators set `_film_bypass=True` so FiLM adds no cost, but `WorkingMemory`'s
  incremental buffer (`_wm_append_one`) GROWS O(T) — so the WM-enabled config is
  **not** constant-memory at decode. The moat as measured holds for the lean model.
- The flat ~14.4 ms is an un-optimized eager constant; the *flatness* is the
  architectural property, the absolute value is improvable (CUDA graphs / compiled
  step) and would move the latency crossover to much shorter context.
- bf16 weights for both (deployment dtype). Random decode tokens — cost is
  data-independent for both. Eager for both (fair; deployed forward_step is eager).

## What could not be measured here

- **Transformer OOM point**: not reached at batch=1 within 131k (only 5.9 GB).
  The memory/throughput moat is far larger under **batched serving** (KV cache x
  batch), which this batch=1 single-stream spec does not exercise.
- **Beyond 131k for ours**: `prefill` still materializes a transient (1, L, vocab)
  logits tensor (~24 GB at 262k) that would OOM the *prefill* — a prefill
  implementation artifact, NOT a decode-moat failure (the decode state stays 8.2
  MiB). Decoding itself would remain flat.
- **Production FiLM-K3 + WM stack decode cost**: out of scope here (its true
  incremental path is only partly wired; WM buffer is O(T)).

---

# Batched serving sweep (2026-06-30) — where the moat actually lives

batch=1 is the WORST case for a bounded-state model (no batch to amortize the
recurrent kernel's launch overhead, and the transformer's KV cache is trivially
small). Realistic agentic serving is batched: the transformer's KV grows as
**B x L** while DeltaNet's recurrent state grows only with **B** (L-independent)
and its FLA kernels are matmul-bound (parallelize over batch). Sweep: B in
{1,8,32,64,128,256} x L in {2048,8192,32768}, prefill then G=64.

Fairness note: OURS's recurrent state is built via `prefill_state_only` (the
SAME state as `TinyLM.prefill`, validated decode-argmax-identical at B>1,
single-shot and chunked) which skips the all-position `lm_head` transient — the
exact analogue of giving the transformer `logits_to_keep=1` at prefill. Both
peaks are measured during the decode phase (reset after prefill). Both run eager
(un-optimized reference paths).

`tput x` = ours_throughput / xf_throughput (>1 ⇒ ours wins). `mem x` =
xf_peak / ours_peak (>1 ⇒ ours uses less). `tok/s` = total across the B parallel
sequences; `ms/tok` = per-sequence latency.

| B | L | ours tok/s | ours ms/tok | ours MiB | xf tok/s | xf ms/tok | xf MiB | tput x | mem x |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 2048  |    69.2 | 14.45 |  919 |  157.9 |  6.33 |   825 | 0.44x | 0.90x |
| 8   | 2048  |   540.3 | 14.79 |  979 | 1228.5 |  6.51 |  1408 | 0.44x | 1.44x |
| 32  | 2048  |  2157.5 | 14.83 | 1183 | 3444.7 |  9.29 |  3429 | 0.63x | 2.90x |
| 64  | 2048  |  4284.9 | 14.93 | 1454 | 4218.5 | 15.17 |  6124 | **1.02x** | 4.21x |
| 128 | 2048  |  8550.5 | 14.96 | 1996 | 4704.9 | 27.17 | 11515 | **1.82x** | 5.77x |
| 256 | 2048  | 17028.3 | 14.96 | 3081 |  OOM   |  OOM  |  OOM  | xf-OOM | xf-OOM |
| 1   | 8192  |    68.8 | 14.53 |  919 |  158.1 |  6.32 |  1064 | 0.44x | 1.16x |
| 8   | 8192  |   538.5 | 14.85 |  980 |  851.8 |  9.39 |  3358 | 0.63x | 3.43x |
| 32  | 8192  |  2135.6 | 14.99 | 1184 | 1266.9 | 25.24 | 11231 | **1.69x** | 9.48x |
| 64  | 8192  |  4244.3 | 15.07 | 1457 |  OOM   |  OOM  |  OOM  | xf-OOM | xf-OOM |
| 128 | 8192  |  8512.8 | 14.99 | 2002 |  OOM   |  OOM  |  OOM  | xf-OOM | xf-OOM |
| 256 | 8192  |  OOM*   |  OOM* | OOM* |  OOM   |  OOM  |  OOM  |   -    |   -    |
| 1   | 32768 |    68.3 | 14.57 |  920 |  155.2 |  6.44 |  2037 | 0.44x | 2.22x |
| 8   | 32768 |   538.4 | 14.81 |  981 |  316.8 | 25.25 | 11160 | **1.70x** | 11.37x |
| 32  | 32768 |  2154.6 | 14.84 | 1190 |  OOM   |  OOM  |  OOM  | xf-OOM | xf-OOM |
| 64  | 32768 |  4252.4 | 15.05 | 1469 |  OOM   |  OOM  |  OOM  | xf-OOM | xf-OOM |
| 128 | 32768 |  8501.1 | 15.05 | 2026 |  OOM   |  OOM  |  OOM  | xf-OOM | xf-OOM |
| 256 | 32768 |  OOM*   |  OOM* | OOM* |  OOM   |  OOM  |  OOM  |   -    |   -    |

`OOM*` = ours OOMs only at B=256, L>=8192 — and that is a **prefill-activation**
OOM (the chunked prefill still peaks at B x chunk x d_ff), **not** a decode-state
limit: ours' decode state at B=128/L=32768 is just 1050 MiB and would be ~2.1 GiB
at B=256, well within budget. Finer prefill chunking removes it.

> ⚠️ **CORRECTED 2026-07-01 (code audit + re-measurement).** The original claim
> here — "The transformer OOMs are fundamental (the KV cache it OOMs on IS the
> decode state)" — was **partly a harness artifact**: the transformer prefilled
> the full (B, L) prompt in ONE forward while ours got a chunked prefill, and
> the single try/except could not attribute the OOM phase. Re-run with the same
> chunked-prefill accommodation (chunk=512, phase-attributed;
> `bench_transformer_batched(chunk=...)` now supports this):
> - **(B=64, L=8192): NOT fundamental.** The transformer serves it —
>   1,313 tok/s at 21,730 MiB decode peak. Corrected cell: ours 4,244 tok/s @
>   1,457 MiB vs xf 1,313 tok/s @ 21,730 MiB = **3.2× throughput at ~15× less
>   memory** (a stronger honest claim than "xf-OOM").
> - **(B=256, L=2048): genuine decode-phase OOM** (prefill completes at
>   24,853 MiB peak with 20,516 MiB KV; decode then OOMs under the eager HF
>   DynamicCache). Note a paged-KV serving stack (vLLM) would likely serve it.
> - The remaining OOM cells (KV ≥ 40 GiB: B=128/L=8192, B≥32/L=32768) are
>   arithmetically fundamental on a 32 GiB card.
> The B-vs-B×L scaling conclusion is unchanged; the specific frontier
> attribution above is superseded by this note.

## Updated verdict — the batch=1 latency loss FLIPS to a throughput + memory WIN

1. **OURS's per-sequence latency is ~FLAT in batch** — 14.45 ms at B=1 to 14.96
   ms at B=256 (+3.5% over a 256x batch increase). The recurrent kernel is
   matmul-bound: batch amortizes the per-step launch overhead, so **throughput
   scales near-linearly** (69 -> 17,028 tok/s, ~246x at B=256). Peak memory grows
   only with B (state = B x 8.2 MiB) and is **independent of L** (919 MiB at
   B=1/L=2048 == 920 MiB at B=1/L=32768).

2. **The transformer degrades on BOTH axes with batch**: per-seq latency grows
   (6.3 -> 27.2 ms at L=2048 as B 1->128) because KV-attention is bandwidth/
   compute-bound on a B x L cache; throughput saturates (~4,700 tok/s at L=2048)
   and its peak memory explodes (11.5 GiB at B=128/L=2048), hitting OOM across a
   wide swath of the frontier.

3. **The latency verdict FLIPS.** The batch=1 "ours 2.3x slower" reverses to a
   throughput WIN at **B>=64 (short ctx)** and **earlier at longer context**
   (B=32 at L=8192 = 1.69x; B=8 at L=32768 = 1.70x) — because the transformer's
   per-token cost rises with context while ours stays constant. Peak: 1.82x
   throughput at 5.8x less memory (B=128/L=2048).

4. **Transformer-OOM frontier (32 GiB), where ours sails through:** the
   transformer OOMs at (B>=256, L=2048), (B>=64, L=8192), (B>=32, L=32768) — i.e.
   roughly when **B x L of KV exceeds ~30 GiB (~750k cached tokens)**. At every
   one of those points OURS serves at 2,100-8,500 tok/s using 1.2-2.0 GiB. The
   memory ratio reaches **11.4x at (B=8, L=32768)** and keeps widening.

**Honest read: yes, batched serving at non-trivial context is where the real
moat lives.** At batch=1 short-context the transformer is genuinely ~2.3x faster
per token (fewer params, optimized softmax-decode kernel) and the moat is just
"constant memory". But for the realistic agentic-serving regime — many
concurrent streams and/or long context — the bounded-state model wins decisively
on throughput-per-GPU and memory, and serves whole (batch, context) regions that
OOM the transformer outright. Caveat: both are eager reference paths; a paged-KV
serving stack (vLLM) would raise the transformer's OOM ceiling somewhat (~2x via
reduced fragmentation/waste) but cannot change its fundamental B x L scaling,
whereas ours' B-only / L-independent scaling is the architectural property.
Caveat 2: this is the LEAN linearized model; the production WM stack would
re-introduce an O(T) decode buffer.
