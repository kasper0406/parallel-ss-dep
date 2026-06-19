# Better embedding optimizer vs shared-LR Adam — 287M DeltaNet pretrain A/B

**Config (every arm + both baselines identical):** real production trunk 10L × d896 × 14h, FiLM(0,5;1,6;2,7;3,8;4,9) K=3 (warmup 200), `--bf16 --tf32 --bf16_optim_state --activation_checkpointing --no-compile`, data `configs/pretrain_mix_v4.yaml`, T=2048, batch=12, grad_accum=6 (= 147,456 tok/step), WSD lr=1.4e-3 / lr_muon=5e-3, **Muon matrix optimizer in EVERY arm**, seed=0. Embedding arms run 1500 steps at constant-peak LR (decay_frac 0) = exactly the baseline's first 1500 steps (its decay starts at 2125), so matched-step comparison is apples-to-apples. The ONLY thing varied across arms is the embedding/lm_head treatment.

Baselines reused (NOT re-run): `runs/precond_ab/muon.log` (shared-LR AdamW embeddings — the thing we try to beat) and `runs/precond_ab/fused.log` (per-head Newton-Schulz matrix optimizer — the ~3% reference win), both seed-0, identical config.

## Throughput (median tok/s, steady-state)

| arm | median tok/s |
|---|---:|
| muon (baseline) | 50096 |
| fused (per-head NS) | 50133 |
| embed_lr 2x | 50084 |
| embed_lr 5x | n/a |
| embed_lr 10x | n/a |
| rownorm dualizer | 50008 |

## VAL ppl at matched steps

| step | muon | fused | embed_lr 2x | embed_lr 5x | embed_lr 10x | rownorm dualizer |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 267.35 | 232.27 | 185.67 | 126.29 | — | 153.12 |
| 200 | 56.33 | 53.61 | 50.89 | — | — | 64.67 |
| 300 | 31.35 | 30.39 | 31.27 | — | — | 44.56 |
| 400 | 27.24 | 26.40 | 27.81 | — | — | 41.75 |
| 500 | 21.72 | 20.72 | 22.47 | — | — | 32.26 |
| 600 | 15.93 | 14.97 | 16.48 | — | — | 22.47 |
| 700 | 14.35 | 13.52 | 14.93 | — | — | 20.51 |
| 800 | 12.87 | 12.34 | 13.31 | — | — | 17.31 |
| 900 | 11.93 | 11.56 | 12.25 | — | — | 15.20 |
| 1000 | 10.86 | 10.56 | 11.12 | — | — | 13.10 |
| 1100 | 10.45 | 10.07 | 10.58 | — | — | 12.18 |
| 1200 | 10.07 | 9.76 | 10.10 | — | — | 11.27 |
| 1300 | 9.24 | 8.98 | 9.34 | — | — | 10.23 |
| 1400 | 8.43 | 8.33 | 8.64 | — | — | 9.28 |
| 1500 | 7.86 | 7.76 | 8.10 | — | — | 8.42 |

## Each embedding arm vs the Muon baseline (positive ⇒ arm is BETTER)

`mean ppl gap` = mean over matched eval steps of (muon_ppl − arm_ppl); dominated by the steep early region. `mean VAL-CE gap` = mean(ln muon_ppl − ln arm_ppl) (scale-stable, the cleaner aggregate). `mid` = the stable region (steps ≥ 990): mean ppl gap and mean %.

| arm | mean ppl gap | mean VAL-CE gap | mid ppl gap | mid % | final-step ppl gap |
|---|---:|---:|---:|---:|---:|
| embed_lr 2x | +5.535 | +0.0117 | -0.162 | -1.76% | -0.240 (step 1500) |
| embed_lr 5x | +141.060 | +0.7500 | +nan | +nan% | +141.060 (step 100) |
| embed_lr 10x | n/a (no overlap / arm failed) | | | | |
| rownorm dualizer | +2.643 | -0.1810 | -1.262 | -12.84% | -0.560 (step 1500) |
| _ref: per-head NS (fused)_ | +2.983 | +0.0430 | +0.242 | +2.46% | +0.100 (step 1500) |

## Train-CE EMA (α=0.1) gap vs Muon baseline (positive ⇒ arm lower CE)

| arm | mean EMA gap | final-step EMA gap |
|---|---:|---:|
| embed_lr 2x | +0.0287 | -0.0090 (step 1500) |
| embed_lr 5x | +0.4330 | +0.5253 (step 100) |
| embed_lr 10x | n/a | n/a |
| rownorm dualizer | -0.1977 | -0.0909 (step 1500) |
| _ref: per-head NS (fused)_ | +0.0541 | +0.0152 (step 1500) |

## Verdict

**No better embedding optimizer was found. Shared-LR AdamW on the embedding/lm_head is already well-tuned; the one real optimizer lever is the MATRIX optimizer (per-head NS, +2.46% mid), not the embedding.**

Read the table carefully — the auto-picked "best arm = embed_lr 5x by mean VAL-CE gap" is an **artifact of incomplete data**: lr5 only reached step 100 (a single very-early eval where *every* higher-LR arm transiently leads because higher LR descends faster early). It is NOT a win. The honest per-arm reading:

- **embed_lr 2x (completed, 1500 steps): WORSE.** mid −1.76%, final ppl 8.10 vs muon 7.86 (−0.240). Classic LR-too-high crossover — it leads early (step 100: 185.7 vs 267.4) but is overtaken and ends behind. Raising the embedding LR μP-style does not help at our operating point.
- **embed_lr 5x (INCOMPLETE — killed at step 100), embed_lr 10x (NOT RUN).** Not full refutations on their own. But lr2 already establishes the early-lead→late-penalty crossover, and lr5 at step 100 is on exactly that faster-early trajectory (126.3 vs muon 267.4), so an even-higher LR reversing the crossover to a *late* win is very unlikely. The sweep is not exhaustive; the trend is one-directional.
- **rownorm (modular-norm) dualizer (completed): DECISIVELY WORSE.** mid −12.84%, every matched step behind, final ppl 8.42 vs 7.86. Spectral/row-normalizing the embedding gradient (treating it like a matrix-layer param) is the wrong geometry for a lookup table — the AdamW per-coordinate adaptive scaling is better for sparse-row embedding updates.
- **Reference — per-head NS matrix optimizer: +2.46% mid, +0.043 mean VAL-CE.** The only positive lever in this whole comparison, and it leaves the embedding alone.

**Conclusion:** the embedding/lm_head stays on shared-LR AdamW (no embed-LR multiplier, no rownorm dualizer). The wiring (`--embed_lr_mult`, `--embed_optimizer rownorm`) is committed default-OFF as a tested, documented **negative result** so we don't re-litigate it. The DeltaNet matrix optimizer (`--matrix_optimizer fused_deltanet_ns`) is the optimizer change worth carrying forward.

HONESTY: single seed; paired (shared init/data/schedule in-window) so the matched-step offset removes most run-to-run variance. The 2x result (−1.76%) and rownorm (−12.84%) are both larger than the step-to-step VAL wiggle, so they are real signals, not noise; lr5/lr10 are genuinely incomplete and labeled as such.
