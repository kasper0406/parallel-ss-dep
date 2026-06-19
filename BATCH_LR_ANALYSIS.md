# Batch size & learning rate for the 287 M DeltaNet pretrain — gradient-noise-scale analysis

**Date:** 2026-06-19 · **Hardware:** 1× RTX 5090 (GPU 1), measurement-only · **Trunk:** 10L × d896 × 14h × d64 DeltaNet + FiLM(0,5..4,9) K=3, output gate, WM(ctx-namekey, read-α frozen 0), PKM, latent-feedback adapter (the v17/v18 production stack, 292 M params).

**TL;DR.** Measured the McCandlish *simple noise scale* `B_simple = tr(Σ)/|G|²` of the **plain LM cross-entropy** gradient on the real trunk + real data (`configs/pretrain_mix_v4.yaml`), at 4 checkpoints spanning 0.5→4.0 B tokens, globally and per optimizer group (Muon-matrix vs AdamW). **Result: `B_simple` ≈ 20–58 k tokens for every group at every stage — 4.5–12× *below* the current 262 k tokens/step.** We are firmly **above** the critical batch size (signal-dominated; noise is only 8–18 % of the production-step gradient), i.e. in the diminishing-returns / curvature-limited regime, **not** the noise-limited regime the per-head-NS hypothesis assumed. So: **do not increase the batch** (pure waste); 262 k is generous-but-defensible; the only efficiency lever is a *modest decrease* to ~131 k, gated by an aux-loss caveat. The current LRs (lr_muon 5e-3, lr_adamw 1.4e-3) sit at ~85–90 % of the large-batch LR ceiling → already correct, no change needed.

---

## Method (estimator + assumptions)

Two-batch unbiased estimator (McCandlish et al., *An Empirical Model of Large-Batch Training*, App. A). Production trains at **batch 4 × grad_accum 32 × T2048**, so the natural example unit is the **microbatch** (4 sequences); B_big = 32 microbatches = the production step (128 seqs = 262 144 tokens). With B_small = 1 microbatch, B_big = G = 32 microbatches, and the identity `E[|g_B|²] = |G|² + tr(Σ_mb)/B` (B in microbatches):

```
|G|²        = (G·sq_big − sq_small)/(G−1)
tr(Σ_mb)    = G·(sq_small − sq_big)/(G−1)
B_simple_mb = G·(sq_small − sq_big)/(G·sq_big − sq_small)
B_simple_tokens = B_simple_mb · (4 seqs · 2048 tok)      # = B_simple_seq · T
```
`sq_small` = mean over the 32 microbatches of `|one-microbatch grad|²`; `sq_big` = `|mean-over-32-microbatches grad|²`. Averaged over **25 independent draws** (3200 distinct cached sequences total, the *same* pool reused across all checkpoints so the across-checkpoint trend reflects the model, not the data draw). 68 % bands are bootstrap over draws.

Assumptions / fidelity choices, stated honestly:
- **Example unit = one sequence** (within-sequence token correlation is absorbed into the per-sequence gradient — the standard convention; OpenAI/GPT report B_crit in tokens = examples × seqlen). The tokens figure is invariant to the seq-vs-microbatch granularity.
- **Plain LM CE only** (think bursts off; gate-weighting, gist, gate-entropy, latent-reasoning, ctx-addr, z-loss all *excluded*). This is the clean, reproducible "pretraining batch size" object. See the Caveats section — the full multi-loss objective's B_crit is almost certainly *higher*.
- **Faithful forward:** eval-mode (no PKM-ε/dropout — both are 0 at these steps anyway), bf16 autocast (matches `--bf16`), FiLM K=3 self-feed active, doc-id cross-document isolation on, EOS targets masked (`--mask_eos_in_targets`).
- **Data:** `pretrain_mix_v4.yaml` per instruction. The checkpoints actually trained on the v17/v18-arxiv mixes (incremental descendants of v4). The noise-scale *ratio* is robust to mild distribution shift; absolute `|G|²` may differ a few %. Using one fixed v4 pool across checkpoints isolates the model effect.

Scripts (GPU 1 only, standalone): `experiments/noise_scale_data_cache.py` (caches the pool), `experiments/grad_noise_scale.py` (the estimator), `experiments/lr_sweep_fromscratch.py` (Part-2 validation). Raw output: `runs/noise_scale/results.json`, `runs/noise_scale/part1.log`.

---

## Part 1 — B_simple across training (the growth curve)

`B_simple` in **tokens** (68 % bootstrap band), and `noise_frac@262k` = fraction of the production-step gradient that is noise = `B_simple/(B_simple+262144)`:

| tokens | group  | B_simple (tok) | 68 % band            | seqs | `|G|²`  | noise_frac @262k |
|-------:|--------|---------------:|----------------------|-----:|--------:|-----------------:|
| 0.5 B  | global | **28,795**     | [25.7k, 32.9k]       | 14.1 | 0.758   | 0.099 |
| 0.5 B  | muon   | 26,885         | [24.4k, 30.0k]       | 13.1 | 0.608   | 0.093 |
| 0.5 B  | adamw  | 36,535         | [30.2k, 46.7k]       | 17.8 | 0.150   | 0.122 |
| 1.5 B  | global | **53,822**     | [45.5k, 67.2k]       | 26.3 | 0.968   | 0.170 |
| 1.5 B  | muon   | 36,925         | [33.0k, 42.3k]       | 18.0 | 0.186   | 0.123 |
| 1.5 B  | adamw  | 57,845         | [47.4k, 76.3k]       | 28.2 | 0.782   | 0.181 |
| 2.5 B  | global | **22,740**     | [20.2k, 26.1k]       | 11.1 | 1.774   | 0.080 |
| 2.5 B  | muon   | 55,837         | [50.5k, 62.5k]       | 27.3 | 0.067   | 0.176 |
| 2.5 B  | adamw  | 21,445         | [19.0k, 24.8k]       | 10.5 | 1.707   | 0.076 |
| 4.0 B  | global | **31,269**     | [27.9k, 34.8k]       | 15.3 | 1.236   | 0.107 |
| 4.0 B  | muon   | 49,962         | [47.4k, 52.6k]       | 24.4 | 0.064   | 0.160 |
| 4.0 B  | adamw  | 30,252         | [26.8k, 33.9k]       | 14.8 | 1.173   | 0.103 |

**Reading the table:**

1. **Every B_simple is 20–58 k tokens — 4.5–12× below the 262 k production step.** `noise_frac@262k` never exceeds 0.18: the production-step gradient is 82–92 % "true signal". This is the *opposite* of noise-limited.

2. **The Muon (trunk) noise scale genuinely grows ~2× over training: 27 k → 50 k tok** (the 0.5 B and 2.5/4.0 B bands don't overlap). This is the textbook "B_crit rises as loss drops", driven by the trunk gradient norm collapsing **~10×** (`|G|²_muon`: 0.61 → 0.064) while its variance falls more slowly. The AdamW (embed/lm_head) noise scale stays flat-and-noisy (~20–58 k).

3. **The *global* number is a non-monotonic mixture and should not be over-read.** Global `B_simple` mixes the two groups weighted by their `|G|²`. Early, the trunk dominates `|G|²` (0.61 vs 0.15) so global≈muon; late, the embed/lm_head dominates (1.7 vs 0.064) so global≈adamw. That re-weighting — not a real reversal — is why global wobbles 29k→54k→23k→31k. **The per-group rows are the reliable signal.**

**Verdict on the batch.** For the core LM objective the critical batch `B_crit ≈ B_simple` is ~30 k early rising to ~50 k (trunk) by 4 B tokens. The production **262 k tokens/step is ~5–9× above B_crit for the whole run.** Operating that far above B_crit means (McCandlish Pareto, `S/S_min = 1+B_crit/B`, `E/E_min = 1+B/B_crit`): we run at only **~1.1–1.2× the minimum *steps*** but **~6–9× the minimum *compute***. The repo's earlier 14 k→262 k bump (correctly) fixed a *below*-B_crit batch, but **overshot**: the compute-optimal point (`B≈2·B_crit`, the balanced 2×/2× knee) is around **80–130 k tokens**, i.e. grad_accum ≈ 12–16, not 32.

---

## Part 2 — LR for the chosen batch

### Theory (grounded in the measured noise scale)

McCandlish optimal-LR-vs-batch: `η_opt(B) = η_max / (1 + B_noise/B)`, with `B_noise ≈ B_simple`. Small batch → `η ∝ B` (linear); `B ≫ B_noise` → `η → η_max` (saturated). Our two optimizers differ in batch-robustness:
- **Muon** orthogonalises the update (per-head Newton–Schulz) → update *magnitude* is decoupled from gradient scale, so its `η_opt` is the *most* batch-robust and saturates earliest.
- **AdamW** is RMS-normalised → classic ≈`√B` scaling below B_crit, saturating at B_crit. (More robust than raw SGD's linear law, less than Muon.)

Both groups sit at **B = 262 k ≫ B_noise**, so both are in the saturated plateau. At the production batch:

| group | B_noise (range over run) | `η_opt(262k)/η_max = 262/(262+B_noise)` |
|-------|--------------------------|------------------------------------------|
| Muon  | 27 k (early) → 50 k (late) | 0.91 → 0.84 |
| AdamW | 21 k → 58 k               | 0.93 → 0.82 |

→ The current LRs are at **~84–93 % of their large-batch ceilings**. They are *already correct* for 262 k; pushing them higher buys almost nothing (you'd be chasing the last ~10–15 % toward `η_max ≈ lr/0.85`, i.e. lr_muon≈5.9e-3, lr_adamw≈1.65e-3 — within noise of current).

**Extrapolation to a different batch** `B'` (factor `η_opt(B')/η_opt(262k) = B'(262k+B_noise) / (262k(B'+B_noise))`, B_noise≈40–50k):

| target batch | grad_accum | LR factor | recommended lr_muon | recommended lr_adamw |
|--------------|-----------:|----------:|--------------------:|---------------------:|
| **262 k (current)** | 32 | 1.00 | **5.0e-3** | **1.4e-3** |
| 131 k | 16 | ~0.86 | ~4.3e-3 | ~1.2e-3 |
| 98 k  | 12 | ~0.82 | ~4.1e-3 | ~1.15e-3 |
| 524 k | 64 | ~1.09 | ~5.4e-3 | ~1.5e-3 |

**The headline LR conclusion: because we are above B_crit, the optimal LR is *batch-robust* — a ±2× change in batch moves the optimal LR by only ~±10–15 %.** Keep the current LRs for 262 k; if you drop to 131 k, nudge them down ~14 % (lr_muon 4.3e-3, lr_adamw 1.2e-3); do **not** raise them for a bigger batch.

### Empirical validation (short from-scratch sweep)

A non-invasive watcher (`runs/noise_scale/watch_and_sweep.sh`) runs a 250-step from-scratch sweep on the real trunk (stripped of memory/PKM/latent/gist to isolate the optimizer-LR effect), reporting plain-LM VAL CE on a *fixed* held-out pool, the moment GPU 1 frees from the user's concurrent `embed_ab` run. Arms: {0.5×, 1×, 2×} LR @ 262 k and {1×, theory} LR @ 131 k. Expected from theory: VAL CE roughly **flat** across the LR bracket at 262 k with the optimum at/above 1× (saturation), and 131 k @ ~0.86× LR ≈ 131 k @ 1× ≈ 262 k (batch-robust).

> **STATUS: PENDING.** At analysis time GPU 1 was reclaimed by the user's `embed_ab_lr2` run (batch 12, ~75 min/arm); per "GPU 1 only / don't disturb others" the sweep was **not** run co-resident. Results will be appended to `runs/noise_scale/sweep.log` when the watcher fires. If it has not run by the time you read this, launch manually:
> `bash runs/noise_scale/watch_and_sweep.sh` (waits for a free GPU 1, then runs ~35 min).

---

## Bottom line (recommended operating point for the next pretrain)

**Keep batch ≈ 262 k tokens (batch 4 × grad_accum 32 × T2048), lr_muon = 5e-3, lr_adamw = 1.4e-3.** The measured plain-LM critical batch is ~30–50 k tokens throughout training, so 262 k is comfortably *above* B_crit (the gradient is already 82–92 % signal) and the current LRs sit on their saturated plateau — there is **no case for increasing either the batch or the LR**, and **no need to ramp the batch up** (B_crit grows only to ~50 k, never near 262 k). The single defensible efficiency move is a **modest reduction to ~131 k (grad_accum 16) with LR nudged ~14 % lower (lr_muon ≈ 4.3e-3, lr_adamw ≈ 1.2e-3)**, which on the pure-LM analysis should reach the same loss in fewer total tokens (≈ wall-clock) — **but validate with a short A/B first** (watch gnorm + VAL stability), because of the caveat below.

**Honest uncertainty — the dominant caveat.** This is the **plain-LM** noise scale. The full v18 objective adds gist (0.1), gate-entropy (0.1), **latent-reasoning (0.05, computed on only n=4 synthetic examples → very high per-step variance)**, ctx-addr (0.2; ≈0 on v4), and z-loss. These raise the *effective* training B_crit above the plain-LM ~50 k — possibly a lot, for the tiny-batch latent term — which is exactly why 262 k is defensible and why cutting the batch is *risky*: a smaller LM batch makes the combined per-step gradient noisier, and this repo has repeatedly hit instabilities (gnorm spikes, the v12/v13 curricula that had to be *delayed*) that smaller batches would worsen. **To settle the batch question definitively, measure the noise scale of the *full* per-step training gradient** (all aux losses + the latent-reasoning stream); that is the one number this study deliberately did not compute.

**Implication for the per-head-NS preconditioner.** The hypothesis was "the per-head-NS edge compresses in the tail because we enter the noise-limited regime late." **The data does not support that** — `noise_frac@262k` stays ≤ 0.18 throughout and the trunk gradient (where per-head-NS acts) is 84–91 % signal even at 4 B tokens; we are *signal*-dominated, not noise-limited. The kernel of truth is that the **trunk (Muon-group) noise scale does roughly double (27 k→50 k)** late — but it stays 5× below 262 k, so a larger batch would only denoise the trunk gradient from ~16 % to ~8 % noise (marginal). The tail-compression of the per-head-NS edge is far better explained by the trunk gradient *magnitude* collapsing ~10× (`|G|²` 0.61→0.064 → less to precondition) than by gradient noise. **Net: per-head-NS is a conditioning lever, largely orthogonal to batch size; do not expect a bigger batch to grow its edge, and a smaller (noisier) batch would, if anything, shrink it.**
