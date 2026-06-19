# Fused DeltaNet per-head NS preconditioner — wall-clock + memory vs Muon

**Date:** 2026-06-19 · **GPU:** RTX 5090 (sm_120), GPU 1 only · env `.venv/bin/python`,
`PYTHONPATH=.` · **Shapes:** production trunk 10 L × d_model 896 × 14 heads × d_head 64
(the 80 Muon-eligible 2-D matrices = q/k/v/o/b + MLP, 128.6 M params).

> Companion to `DELTANET_PRECONDITIONER.md` (the iso-step convergence study).
> This file is ONLY the fused-optimizer build + perf proof. Scripts:
> `experiments/exp_deltanet_precond_fused.py` (optimizer + `--mode
> verify|profile|memory|fraction`), tests `experiments/test_deltanet_precond_fused.py`.

## VERDICT

**Yes — the fused per-head NS optimizer is ≥ Muon on BOTH wall-clock AND memory at
production scale, and it is byte-identical (Δ = 0) to the validated 2-object
prototype.** Per-head NS does ~14× fewer NS FLOPs on q/k/v (small 64×64 grams
instead of one 896×896 gram), which buys both axes at once:

- **Wall-clock:** ~**1.15–1.26× faster** than `torch.optim.Muon` (fp32 state),
  ~**1.12× faster** (bf16 state) — faster on min AND median, across two
  independent contended round-robin runs.
- **Memory:** optimizer **state is identical** (one momentum buffer/param: 514 MB
  fp32 / 257 MB bf16), and the **peak transient NS workspace is *lower*** than
  Muon — **22.7 MB vs 29.1 MB (fp32)** and **22.7 MB vs 42.2 MB (bf16)** —
  confirming the hypothesis that per-head NS on small matrices ≤ Muon's
  whole-matrix NS workspace.

The one gap is the optional `batch_across_layers` (cross-layer) mode: marginally
faster wall-clock but **10× heavier** transient (296 MB, from the `cat` that
stacks all layers into one bmm). It is **not** the default — per-unit streaming
(batch over the 14 heads, one projection at a time) wins on the joint objective.

---

## 1. Fused design (`FusedDeltaNetMuon`)

A single `torch.optim.Optimizer` with one `step()` that replaces the validated
2-object pair (`torch.optim.Muon(o_proj+MLP)` + `DeltaNetProjMuon(q/k/v/b)`):

- **q/k/v projections** → per-head Newton–Schulz, **batched over the 14 heads**
  as one bmm on the `(n_heads, d_head, d_in)` stack (reuses the prototype's exact
  `_ns_batched`). Each head slice is Frobenius-normalized + NS'd independently;
  bmm never mixes batch elements.
- **b_proj** (β logits, `n_heads × d_in`) → per-row NS (the `(n_heads,1,d_in)`
  stack).
- **o_proj + MLP** (head-MIXING / generic matrices) → whole-matrix Muon, exactly
  `torch.optim.Muon` (reuses `_zeropower_via_newtonschulz`, same `(A,B,C)` coeffs,
  `eps=1e-7`, `ns_steps=5`, `_adjust_lr`).
- **one momentum buffer per param** (fp32, or bf16 via `bf16_state=True`),
  momentum/nesterov/decoupled-WD identical to torch Muon.

**Memory-efficient by construction** (this is what makes it ≤ Muon on memory):

- *Momentum is in-place.* fp32 uses a single foreach pass
  (`_foreach_lerp_(buf,g,1-m)` then `_foreach_lerp_(g,buf,m)` → `g` *is* the
  nesterov update) — no materialized update-list. bf16 lifts ONE param's
  buffer+grad to fp32 at a time (`_bf16_momentum_one`), matching `BF16StateMuon`'s
  streaming footprint.
- *NS + apply are streamed per projection* — only one projection's NS workspace
  is live at a time, so peak transient tracks the *largest single* NS, not the
  sum. (The earlier non-streaming foreach draft held all 30 updates + 30 outputs
  at once → 440 MB; streaming brought it to 22.7 MB.)

`batch_across_layers=True` is an opt-in variant that `cat`s all same-shape
projections from every layer into one big bmm (fewer kernel launches) — measured
below; heavier and not faster enough to justify, so default is `False`.

---

## 2. Byte-identical verification (mandatory pre-condition)

`--mode verify` builds two identically-seeded production models, feeds **identical
grads** to the 2-object prototype and to the fused optimizer for 10 steps, and
reports `max|Δ|` over all 80 matrix params. Both batching modes:

| comparison | device | max&#124;Δ&#124; |
|---|---|---|
| fused **per-unit** vs 2-object (`DeltaNetProjMuon`+`Muon`) | CUDA (prod shapes) | **0.000e+00** |
| fused **cross-layer** vs 2-object | CUDA (prod shapes) | **0.000e+00** |
| both | CPU (prod shapes & small) | **0.000e+00** |

Exactly bit-for-bit (tolerance was 1e-5; achieved 0). The cross-layer bmm is
bit-identical to per-layer batching here — cuBLAS picks the same per-matrix kernel
regardless of batch count for these shapes. Regression-pinned in
`experiments/test_deltanet_precond_fused.py` (5 tests, CPU-only, all pass).

```
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_precond_fused.py \
  --mode verify --device cuda --d_model 896 --n_layers 10 --n_heads 14 --d_head 64
```

---

## 3. Wall-clock — ms / optimizer-step at production shapes

**Measurement protocol.** Real production matrices for all 10 layers + fake grads;
warm up ≥40 steps; `torch.cuda.synchronize()` around every timed burst. GPU 1 was
**intermittently shared** by another experiment (it cycled 0→18 GB repeatedly), so
timing is done **round-robin**: warm up all configs, then cycle through them timing
short bursts for many reps, and report **MIN** (the least-contended sample =
closest to true compute) plus median. Round-robin makes the contention hit every
config equally, so the *relative* ordering is contention-invariant. Two independent
runs agree.

### Run A — round-robin, 30 reps × 40-step bursts (min ms / opt-step)

| optimizer | state | **min ms** | median ms | vs Muon (min) |
|---|---|---:|---:|---:|
| `torch.optim.Muon` | fp32 | 33.52 | 40.56 | — |
| 2-object prototype | fp32 | 25.67 | 33.26 | 1.31× |
| **FUSED per-unit** | fp32 | **26.69** | 35.33 | **1.26×** |
| FUSED cross-layer | fp32 | 24.11 | 34.01 | 1.39× |
| `BF16StateMuon` | bf16 | 29.62 | 38.80 | — |
| **FUSED per-unit** | bf16 | **26.54** | 34.68 | **1.12×** |

The fused per-unit optimizer is faster than Muon on min AND median, in both fp32
and bf16. (2-object and cross-layer are slightly faster still on raw wall-clock,
but lose on memory — see §4.)

### Run B — round-robin, 20 reps × 50-step bursts (independent window, confirms A)

| optimizer | state | min ms | median ms |
|---|---|---:|---:|
| `torch.optim.Muon` | fp32 | 34.56 | 37.87 |
| 2-object prototype | fp32 | 25.77 | 33.85 |
| **FUSED per-unit** | fp32 | **26.65** | 35.35 |
| FUSED cross-layer | fp32 | 24.14 | 31.91 |
| `BF16StateMuon` | bf16 | 29.62 | 38.67 |
| **FUSED per-unit** | bf16 | **27.97** | 34.73 |

Within contention noise of Run A (muon-bf16 min reproduced to 0.001 ms); same
ordering, same ~1.2–1.3× fp32 / ~1.06–1.12× bf16 fused-over-Muon advantage.

> Why "only" ~1.2× and not the ~14× FLOP ratio: per-head NS speeds up the **q/k/v**
> (30 of 80 matrices); **o_proj + MLP** are head-mixing and stay on whole-matrix
> Muon in both optimizers, so roughly half the matrix work is common. The win is
> the q/k/v half going ~14× cheaper. (The iso-step convergence win in
> `DELTANET_PRECONDITIONER.md` is the larger, separate lever; this file only shows
> the per-step cost is not a regression.)

```
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_precond_fused.py \
  --mode profile --d_model 896 --n_layers 10 --n_heads 14 --d_head 64 \
  --warmup 40 --iters 40 --reps 30
```

---

## 4. Memory — persistent state + peak transient workspace

`--mode memory`: per-process numbers (robust to the GPU co-tenant). State =
sum of momentum-buffer bytes. Peak transient = `max_memory_allocated()` across one
`step()` minus the baseline, after buffers already exist (so it isolates the NS
workspace, not the one-time buffer allocation).

| optimizer | state | **state MB** | **peak transient MB** |
|---|---|---:|---:|
| `torch.optim.Muon` | fp32 | 514.30 | 29.10 |
| 2-object prototype | fp32 | 514.30 | 29.10 |
| **FUSED per-unit** | fp32 | 514.30 | **22.68** |
| FUSED cross-layer | fp32 | 514.30 | 296.29 |
| `BF16StateMuon` | bf16 | 257.15 | 42.21 |
| **FUSED per-unit** | bf16 | 257.15 | **22.68** |

- **State is identical** to Muon for all variants — exactly one momentum buffer
  per param (514 MB fp32, halved to 257 MB with `bf16_state`). No extra
  preconditioner state (unlike SOAP/Shampoo).
- **Peak transient workspace is LOWER than Muon** for the per-unit fused: 22.7 vs
  29.1 MB (fp32), 22.7 vs 42.2 MB (bf16). Confirms the hypothesis — the per-head
  NS gram is `(14,64,64)` (tiny) vs Muon's `(896,896)` whole-matrix gram, so the
  largest single NS workspace is smaller even though the optimizer touches the
  same params. (The bf16 fused matches the fp32 fused at 22.7 MB because the NS
  math is fp32 in both; bf16 only shrinks the persistent buffers, and the fused
  streams the fp32 lift one param at a time.)
- **cross-layer = REFUTED on memory:** the `cat` into a single
  `(Σheads, 64, 896)` bmm workspace costs 296 MB transient (~10× Muon). This is
  the closeable-but-not-worth-it gap; per-unit streaming avoids it entirely.

```
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_precond_fused.py \
  --mode memory --d_model 896 --n_layers 10 --n_heads 14 --d_head 64
```

---

## 5. Opt-step as a fraction of a full training step

`--mode fraction`: real fwd+bwd (bf16 autocast) of the 10 L × d896 DeltaNet +
opt-step. The opt-step is **invariant** to batch/seqlen (it only touches params);
the fwd+bwd scales with tokens. Measured on GPU 1 (single microbatch, no FiLM
K=3 / WM / PKM → the *lightest possible* fwd+bwd, i.e. an **upper bound** on the
opt-step fraction):

| config | fwd+bwd | fused opt-step (invariant) | opt / (fwd+bwd + opt) |
|---|---:|---:|---:|
| B=4, T=1024 | 56.3–57.3 ms | 35.7–36.4 ms | ~39 % |
| B=4, T=2048 | 110.7 ms | 37.4 ms | ~25 % |

(B=4/T=1024 measured twice; the cleaner window reconciles end-to-end: full step
97.4 ms ≈ 57.3 fwd+bwd + 35.7 opt, and the fused opt-step 35.7 ms < Muon's 39.2
ms in the same window.) The opt-step holds at ~37 ms while fwd+bwd grows linearly
with tokens. Scaling to
the real production step shrinks the fraction to **well under 1 %**:

- production batch **14** (×3.5) × T **2048** × **FiLM K=3** self-feed (~2–3× the
  forward) → fwd+bwd per microbatch ≈ 0.7–1.0 s;
- production uses **grad_accum 8** → 8 fwd+bwd per single opt-step;
- ⟹ opt-step ≈ 37 ms / (8 × ~0.8 s) ≈ **0.5 %** of a real training step.

So the optimizer choice is negligible at production scale — and the fused–vs–Muon
delta (fused is ~1–2 ms *faster* per opt-step here) is a saving, not a cost. The
big-config "full step" timing is contention-noisy (separate timing windows didn't
add up), so the fraction is derived from the stable, separately-measured
fwd+bwd-vs-opt components rather than a single end-to-end number.

---

## 6. Caveats / honesty

- **Timing was under intermittent GPU contention** (another experiment cycled on
  GPU 1 throughout). Mitigated by round-robin min-of-many reps and reproduced
  across two runs; the *relative* fused-vs-Muon ordering is contention-invariant,
  but the absolute ms would be a few-ms lower on a fully exclusive GPU. (Memory is
  per-process and unaffected.)
- The micro-bench model is `feedback_mode="none"` (no FiLM/WM/PKM) so the matrix
  set is q/k/v/o/b + MLP at production width — the right set for an
  optimizer-only comparison; the FiLM/WM 2-D matrices would just add more
  whole-matrix-Muon params (common to both optimizers).
- The byte-identity proof covers the **fp32** path against the validated fp32
  prototype. The bf16-state fused is a correct bf16 Muon-family update (lift→step→
  store), validated to run finite and keep bf16 buffers (test), but there is no
  bf16 2-object prototype to diff against bit-for-bit.
- Headline metric for adopting this is the **iso-step convergence win** in
  `DELTANET_PRECONDITIONER.md`; this file only establishes that folding it into a
  single fused optimizer carries that win to wall-clock with **no** per-step time
  or memory regression vs Muon.

## 7. Independent review

The fused optimizer was independently code-reviewed against the prototype, torch
`Muon`, and `BF16StateMuon`. **No correctness bugs in the update math** — the
in-place foreach momentum, the non-nesterov "NS reads the buffer" path (NS copies
its fp32 input, never mutating the persisted buffer), disjoint param routing (each
param updated exactly once), and the cross-layer slice-back were all confirmed
correct, consistent with the Δ=0 byte-identity. Three minor items were fixed: an
empty-param-list guard in the fp32 momentum, an honest `batch_across_layers=False`
in `--mode fraction` (bf16 ignores cross-layer by design, so the fraction now
measures the per-unit streamed path it claims to), and a non-zero exit code from
`--mode verify` on failure.
