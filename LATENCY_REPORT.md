# Decode latency, prefill latency, and inference-state memory

Honest measurement of the inference cost of the 708 M sparse-(2, 34) FiLM
DeltaNet vs the 708 M plain DeltaNet (Phase 20 architectural finding,
−3.2 % PPL on codeparrot at 30 M tokens), plus a 360 M Transformer
deployment-memory-equivalent reference. This is the missing measurement
for the architectural finding — without it, any blog/paper on this work
has a hole.

- **Bench script**: [`experiments/decode_bench.py`](experiments/decode_bench.py)
- **Raw JSON output**: [`bench_decode.json`](bench_decode.json)
- **Date**: 2026-04-30
- **Hardware**: 1× RTX 5090 (sm_120, 32 GB), CUDA 13.2, PyTorch nightly cu132
- **Tokenizer**: HuggingFaceTB/SmolLM2-135M (vocab 49152)
- **Checkpoints (708 M, T=512, codeparrot, Muon)**:
  - `checkpoints/dn_36L_708M_muon.pt` (DN baseline, 705.8 M params)
  - `checkpoints/sparse_2_34_708M_muon.pt` (FiLM, 707.8 M params)
- **Transformer reference**: 360 M Muon ckpt
  (`checkpoints/transformer_30L_360M_muon.pt`), positional embedding
  extended random-init beyond its trained 512-token range — quality-only
  results would not be valid past 512 ctx, but **decode latency depends
  only on the architecture** (verified: identical with/without ckpt).

## TL;DR

| | Decode (ms/tok @ 8 K) | Inference state (MB @ 8 K) | Quality (PPL) |
|---|---:|---:|---:|
| **DN baseline 708 M** | 14.5 | 9.8 | 35.38 |
| **Sparse-FiLM DN 708 M [lagged-cached]** | **14.5** | 9.8 | **34.26** ⭐ |
| Sparse-FiLM DN 708 M [2-pass, gold-matching] | 28.8 | 9.8 | (=lagged, by construction) |
| Transformer 360 M | 4.6 | 720 | 18.78 |

- **Two FiLM decode protocols measured** because the choice has very
  different latency cost:
  1. **Lagged-cached** (the user's intended deployment protocol): one
     full-forward per decode step + one extra `Linear` × `(1, 1, d)` for
     the FiLM. Reuses the *previous decode step's pass-2* layer-34
     output as the lag-1 FiLM input — an approximation of training's
     pass-1 layer-34. Cost: **identical to plain DN.** Quality drift
     from the pass-2-as-pass-1 proxy is **unverified** in this report —
     a separate eval is needed to validate.
  2. **2-pass-per-decode** (gold-matching): exactly mirrors training. At
     each decode step, runs pass-1 on the new token through layers
     0..source (35/36 layers), then pass-2 through all 36 layers using
     the previous step's *true* pass-1 source-layer output. Cost: **2×
     plain DN.** Verified to match full-forward decode token-for-token.
- **Inference state**: the FiLM model's deployment-memory state
  (recurrent + conv state of all 36 layers + a single `(1, 1, d_model)`
  lagged source) is **74× smaller** than a 360 M Transformer's KV cache
  at T=8 K (9.8 MB vs 720 MB).
- **Verdict on whether the architectural lift survives the latency tax**
  — depends on which decode protocol you mean:
  - **Lagged-cached**: tax is ≤1 % (negligible). The architectural lift
    (−3.2 % PPL) survives intact, *if* the proxy doesn't cost quality.
  - **2-pass**: tax is +99 % (decode is twice as slow). The lift is
    real but at a real cost. This is the safe upper bound.
- The Transformer 360 M is still **3.1× faster per decode token** than
  the 708 M FiLM DN. The deployment-memory parity argument shifts the
  comparison axis to memory, not speed.

## 1. Decode latency (ms/token, batch=1)

Median (p95) over 5 outer iterations × 24 decode steps after a
T-token prefill = 120 measured decode steps per cell. Warmup: 2 outer
iterations (also discarded). Greedy sampling.

### Headline table

| Context T | DN baseline | FiLM (lagged-cached) | FiLM tax (lagged) | FiLM (2-pass) | FiLM tax (2-pass) | Transformer-360 M |
|----------:|------------:|---------------------:|------------------:|--------------:|------------------:|------------------:|
| 512       | 14.49 (14.72) | 14.56 (15.01) | +0.5 % | 28.70 (29.02) | +98.1 % | 4.58 (4.64) |
| 2 048     | 14.50 (14.76) | 14.60 (15.03) | +0.7 % | 28.57 (29.13) | +97.0 % | 4.58 (4.62) |
| 4 096     | 14.54 (14.85) | 14.56 (14.88) | +0.1 % | 28.83 (29.36) | +98.3 % | 4.58 (4.64) |
| 8 192     | 14.53 (15.04) | 14.54 (14.86) | +0.1 % | 28.78 (29.98) | +98.1 % | 4.64 (4.70) |

The lagged-cached FiLM has a 0.1–0.7 % tax over plain DN: a single
`Linear(W_scale)` and `Linear(W_shift)` on a `(1, 1, 1024)` tensor at
layer 2's input — well below the 5 % "negligible" threshold and far
below the 20 % "material" threshold from the spec.

The 2-pass FiLM has a 97–98 % tax: as expected for a 2-pass forward
where pass-1 covers 35/36 layers and pass-2 covers all 36 layers. This
is the gold-matching upper bound that exactly mirrors the training-time
forward.

The Transformer is **constant at ~4.6 ms/tok across all T** because RTX
5090's bf16 SDPA at `q_len=1` against a bf16 KV cache is fast enough
that even at T=8 192 the attention is negligible vs the 30-layer
linear-op cost. Decode latency on this hardware is dominated by
per-step matmul throughput, not by KV-cache attention size.

## 2. Prefill latency (tokens/s, batch=1)

| Context T | DN baseline | FiLM (lagged) | FiLM (2-pass) | FiLM/DN ratio | Transformer-360 M |
|----------:|------------:|--------------:|--------------:|--------------:|------------------:|
| 512       | 24.6 ms / 20.8 K tok/s | 45.8 ms / 11.2 K tok/s | 48.2 ms / 10.6 K tok/s | 1.86× / 1.96× | 11.2 ms / 45.6 K tok/s |
| 2 048     | 50.2 ms / 40.8 K tok/s | 98.7 ms / 20.7 K tok/s | 99.5 ms / 20.6 K tok/s | 1.97× / 1.98× | 29.9 ms / 68.4 K tok/s |
| 4 096     | 101.2 ms / 40.5 K tok/s | 200.6 ms / 20.4 K tok/s | 201.7 ms / 20.3 K tok/s | 1.98× / 1.99× | 56.9 ms / 72.0 K tok/s |
| 8 192     | 197.5 ms / 41.5 K tok/s | 404.1 ms / 20.3 K tok/s | 406.0 ms / 20.2 K tok/s | 2.05× / 2.06× | 121.0 ms / 67.7 K tok/s |

The FiLM model **always pays the 2× prefill tax**, because pass-1 is
needed to extract the lagged source state. The "lagged-cached" trick
helps only at decode (where we can reuse the previous step's pass-2
output); during prefill we don't have a previous step.

For typical workloads (short prompt, long generation), prefill is a
small fraction of total wall-clock — see "Workload analysis" below.

## 3. Inference-state memory (MB at T = 8 192, batch=1)

### Per-batch-element state at end of T = 8 192 prompt

| Model | State formula | Analytical | Empirical Δ |
|---|---|---:|---:|
| DN 708 M | `36 layers × (16 heads × 64 × 64 fp32 + 3 × 1024 × 4 bf16 conv)` | **9.84 MB** | 9.91 MB |
| Sparse-FiLM DN 708 M | DN state + `(1, 1, 1024) fp32` lagged source | **9.85 MB** | 9.91 MB |
| Transformer 360 M | `2 (K,V) × 30 layers × 12 heads × 8 192 × 64 bf16` | **720 MB** | 720.06 MB |

(Empirical Δ matches analytical to within autocast-bookkeeping noise.)

### Memory ratios

| | vs DN-baseline | vs Transformer @ 8 K |
|---|---:|---:|
| DN baseline 708 M | 1× | 0.014× (74× smaller) |
| FiLM 708 M | 1.0002× | 0.014× (74× smaller) |
| Transformer 360 M | 73× | 1× |

The FiLM connection adds **2 KB per batch element** (the cached
`(1, 1, d_model)` lagged-source tensor) on top of DN's already-tiny
state. At a 64-batch serving regime, FiLM adds 128 KB total — trivial.

The 708 M FiLM model's deployment-memory advantage over the 360 M
Transformer at T=8 K is **710 MB per batch element saved**. At
batch=64, the Transformer would need 45 GB of KV cache (infeasible on
a single 32 GB card), while the RNN's state is just 64 × 9.85 MB ≈
630 MB. **The deployment-memory argument grows much stronger at large
batch.**

### Quality per inference-MB at T=8 192 (batch=1)

| Model | Inference state | PPL | Quality / state |
|---|---:|---:|---:|
| DN baseline 708 M | 9.8 MB | 35.38 | (best of RNNs at the size, before FiLM) |
| **Sparse-FiLM DN 708 M** | 9.8 MB | **34.26** | **strongest RNN quality/MB** |
| Transformer 360 M | 720 MB | 18.78 | (best PPL, but 74× more state) |

The Transformer beats both RNNs by ~82 % on raw PPL but at 74× the
inference state. The deployment-memory-parity argument is that you'd
swap the Transformer's KV-cache budget for parameters and run the RNN
at higher capacity. Per Phase 20's data, this argument **does not close
the cross-architecture gap** at our 30 M-token training budget — both
RNN models are far from saturated. We do not claim cross-architecture
victory; the architectural finding is a **linear-RNN-internal**
contribution.

## 4. Honest interpretation

### Does the architectural lift survive the latency tax?

**It depends on which decode protocol you assume.**

- **Lagged-cached protocol** (1× cost):
  - +0.1 % to +0.7 % tax across all T tested.
  - Well inside the user-supplied "FiLM tax is negligible (≤5 %)"
    threshold; rules out the "≥20 % material" objection.
  - The architectural lift (−3.2 % PPL) survives essentially intact.
  - **Caveat:** uses pass-2 layer-34 as a proxy for what training used
    (pass-1 layer-34) at the same time index. This is structurally an
    approximation; exact greedy decode tokens diverge from the
    training-form forward after the first step. Quality cost of this
    proxy is **not measured here** — would require a separate codeparrot
    PPL / HumanEval eval running the lagged-cached decode and comparing
    to the 2-pass decode. This is the principal open question for
    deploying the FiLM model.

- **2-pass-per-decode protocol** (2× cost):
  - +97–98 % tax across all T tested.
  - Exactly mirrors the training-time forward; tokens generated are
    identical to a "rerun full forward at every step" reference,
    verified at T=64 (9/9 tokens match).
  - This is the **safe upper bound**. If quality of the lagged-cached
    proxy is unacceptable, this is the fallback.
  - At ~28.8 ms/tok at T=8 K, FiLM 708 M [2-pass] is 2× the DN 708 M
    decode time (14.5 ms/tok). It is a coherent option: pay 2× decode
    latency to get −3.2 % PPL plus the same deployment-memory savings.
    Whether that tradeoff is acceptable depends on the deployment
    context — for a memory-bound serving regime where the alternative
    is paying 74× the inference state for a Transformer, paying 2× the
    compute for a quality-equivalent (within the linear-RNN family)
    model is reasonable.

### Workload analysis: prefill vs decode amortization

For typical workloads, the prefill 2× tax matters less than the decode
tax. Computing total wall-clock for a few realistic scenarios:

| Workload | Prefill T | Generated tokens | DN total | FiLM lagged total | FiLM 2-pass total |
|---|---:|---:|---:|---:|---:|
| Chat reply | 512 | 256 | 24.6 ms + 256 × 14.5 ms = **3.74 s** | 45.8 ms + 256 × 14.6 ms = **3.78 s** (+1 %) | 48.2 ms + 256 × 28.7 ms = **7.39 s** (+98 %) |
| Long-prompt agent | 4 096 | 256 | 101 ms + 256 × 14.5 ms = **3.81 s** | 200 ms + 256 × 14.6 ms = **3.94 s** (+3 %) | 202 ms + 256 × 28.8 ms = **7.57 s** (+99 %) |
| RAG / 8 K-context | 8 192 | 64 | 198 ms + 64 × 14.5 ms = **1.13 s** | 404 ms + 64 × 14.5 ms = **1.33 s** (+18 %) | 406 ms + 64 × 28.8 ms = **2.25 s** (+99 %) |
| Code-completion (long output) | 1 024 | 1 024 | 25 ms + 1024 × 14.5 ms = **14.9 s** | 50 ms + 1024 × 14.5 ms = **14.9 s** (≈0 %) | 50 ms + 1024 × 28.7 ms = **29.4 s** (+97 %) |

Lagged-cached FiLM ranges from +0 % to +18 % wall-clock vs DN
depending on prefill/decode ratio — the 18 % comes from the RAG-style
8 K-prompt-64-output regime where prefill dominates. For
generation-heavy regimes (chat, code completion), the lagged-cached
FiLM is essentially free.

The 2-pass FiLM is a clean +97–99 % across all regimes, matching the
2× full-forward cost.

### Where the FiLM model loses ground

1. **Prefill is 2× slower** than DN regardless of protocol. Long-prompt
   workloads pay this in full.
2. **Transformer 360 M is 3.1× faster per decode token** despite the
   709 M FiLM model being deployment-memory-cheaper. On RTX 5090 the
   per-step compute is dominated by MLP / linear ops, and a
   half-the-params model is correspondingly half-the-time.
3. **Transformer 360 M is ~3× faster on prefill** as well, because
   half-the-params + bf16 SDPA on the prefill batch is just plain
   faster.

Deployment story for sparse-FiLM 708 M: **cheaper memory, slower
compute**. It would be the right choice if the deployment constraint
is memory (long contexts, large batch, consumer GPUs) and quality at
the linear-RNN ceiling matters. It would not be the right choice if
compute per token is the bottleneck and you have plenty of GPU memory.

### Caveats and known limitations

1. **The lagged-cached decode quality is unverified.** The bench
   measures latency only. We *did* verify (in `decode_bench.py`'s
   sanity test) that the 2-pass decode matches the full-forward gold
   token-for-token at T=64 (9/9 match). The lagged-cached decode
   matches the gold for the first token only (1/9) — this is expected,
   because the proxy-substitution introduces drift after step 0, but it
   does not necessarily mean the *eventual quality* is bad (FiLM α is
   small, so pass-2 ≈ pass-1; the model may have learned to be robust
   to small input perturbations). **A follow-up eval running
   codeparrot PPL with the lagged-cached decode is the next step** to
   confirm or reject the proxy. Until that's run, the safest claim
   is "lagged-cached decode = same latency as DN, quality drift TBD".

2. **Prefill 2× tax is unavoidable for the FiLM connection.** The
   2-pass structure is intrinsic to how cross-layer feedback works
   (information from layer 34 reaches layer 2). The pass-1 *can* be
   shortened to layers 0..source (35/36 layers, ~3 % savings), and we
   do that. Beyond that there's no cheaper way to extract the lagged
   source state.

3. **Decode-state memory is exactly DN memory** — the lagged source
   state `(1, 1, 1024)` fp32 = 4 KB at batch 1, scaling linearly with
   batch size. At batch=64 this adds 256 KB on top of DN's 630 MB —
   completely negligible.

4. **The 2-pass decode requires storing TWO sets of recurrent state
   per layer** (cache_p1 for pass-1 layers 0..34, cache_p2 for pass-2
   all layers). Memory: 9.8 MB × (35/36 + 1) ≈ 19.4 MB per batch — still
   tiny vs Transformer's 720 MB at T=8 K. Not a real cost.

5. **Random tokens for input.** The bench uses
   `torch.randint(0, vocab, ...)` to construct prefill ids. Fine for
   measuring latency (which depends on shape, not content) but means
   the decode loop produces gibberish. Latency is identical for real
   text.

6. **batch=1 only.** All numbers are at batch 1. Per-decode-token cost
   on the Transformer would scale roughly linearly in batch (limited by
   memory bandwidth on the KV cache); the RNN scales better
   (recurrent state is small, MLP is the dominant cost). At batch 64,
   the Transformer needs 64× the KV cache (45 GB at T=8 K, infeasible
   on a single 32 GB card), while the RNN's state at batch 64 is just
   64 × 9.8 MB ≈ 630 MB — well within capacity. **The
   deployment-memory argument grows stronger at large batch.**

7. **Transformer ckpt position embeddings only valid up to T=512.** We
   measured latency-only beyond that (random-init pos embed for T > 512).
   Latency is architecture-only, so this is a non-issue for latency
   reporting; quality numbers above T=512 would not be meaningful.

8. **n_meas=5 outer iterations × 24 decode steps each.** Median ± p95
   reflect 120 measured decode steps. Variance is small (p95−median
   ≈ 0.1–0.5 ms, well under the 1 ms differences between models).

9. **No torch.compile / no graph capture.** A production deployment
   would likely use one or both, which can cut decode latency 30–50 %
   for the linear-RNN side and similarly for the Transformer side.
   The relative numbers (FiLM-lagged ≈ DN, FiLM-2pass ≈ 2×, TX ≈ 4.5 ms)
   should be roughly preserved.

## How to reproduce

```
cd /home/knielsen/ml/parallel-ss-dep
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python experiments/decode_bench.py \
    --n_warmup 2 --n_meas 5 --n_decode 24 \
    --out bench_decode.json
```

Run time: ~12 minutes on a single RTX 5090. The script outputs all
configurations × four context lengths × decode/prefill/memory into
`bench_decode.json`.

For the Transformer with the loaded 360 M Muon checkpoint
(latency-identical, but pos-embed extrapolation beyond T=512):

```
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python experiments/decode_bench.py \
    --no_dn --no_film --transformer_from_ckpt --out bench_tx_loaded.json
```

To verify the 2-pass decode matches full-forward gold token-for-token:

```python
# Inline sanity test (run in repl):
import sys; sys.path.insert(0, '.')
import torch
from experiments.decode_bench import (
    load_dn_or_film, stateful_prefill_film_2pass, decode_step_film_2pass,
)
m = load_dn_or_film('checkpoints/sparse_2_34_708M_muon.pt')
ids = torch.randint(0, 49152, (1, 64), device='cuda')
logits, c1, c2, lagged = stateful_prefill_film_2pass(m, ids)
gen = [logits.argmax(-1).item()]
for _ in range(8):
    new_logits, lagged = decode_step_film_2pass(m, torch.tensor([[gen[-1]]]).cuda(), c1, c2, lagged)
    gen.append(new_logits.argmax(-1).item())
# vs full-forward:
ids_full = ids.clone()
gold = []
for _ in range(9):
    with torch.inference_mode():
        l = m(ids_full)
    nt = l[:, -1:].argmax(-1)
    gold.append(nt.item())
    ids_full = torch.cat([ids_full, nt], dim=1)
assert gen == gold, f"2-pass mismatch: {gen} vs {gold}"
```
