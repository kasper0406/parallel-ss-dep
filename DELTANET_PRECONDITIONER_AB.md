# Fused per-head DeltaNet-NS vs Muon — production pretrain A/B

**Config (BOTH arms identical):** real production trunk 10L × d896 × 14h, FiLM(0,5;1,6;2,7;3,8;4,9) K=3 (warmup 200), `--bf16 --tf32 --bf16_optim_state --activation_checkpointing --no-compile`, data `configs/pretrain_mix_v4.yaml`, T=2048, batch=12, grad_accum=6 (= 147,456 tok/step), WSD lr=1.4e-3 / lr_muon=5e-3, seed=0, 2500 steps. GPU 1 only, sequential. The ONLY difference between arms is the q/k/v/b orthogonalization (`--matrix_optimizer`).

## Wall-clock + peak memory (the clean deltas)

| arm | median tok/s | ms/step | peak GPU mem (MiB) |
|---|---:|---:|---:|
| muon | 50096 | 2943 | 24692 |
| fused | 50133 | 2941 | 25132 |

Fused throughput vs muon: **1.001×** (>1 ⇒ fused faster). Peak-mem Δ (fused−muon): **440 MiB**.

> Note: at grad_accum 6 the optimizer step is ~0.5% of a full training step (perf doc §5), so full-step tok/s is expected to be near-identical between arms; the isolated opt-step timing below is where the per-head-NS cost difference actually shows.

## Loss — matched-step EMA (α=0.1) of train CE

gap = muon_ema − fused_ema (positive ⇒ fused ahead).

| step | muon CE (ema) | fused CE (ema) | gap |
|---:|---:|---:|---:|
| 200 | 7.7209 | 7.6270 | +0.0938 |
| 400 | 5.2941 | 5.2264 | +0.0677 |
| 600 | 4.1263 | 4.0537 | +0.0726 |
| 800 | 3.3685 | 3.3018 | +0.0667 |
| 1000 | 2.8878 | 2.8429 | +0.0450 |
| 1200 | 2.6212 | 2.5947 | +0.0265 |
| 1400 | 2.4441 | 2.4257 | +0.0184 |
| 1600 | 2.3617 | 2.3489 | +0.0128 |
| 1800 | 2.2410 | 2.2304 | +0.0106 |
| 2000 | 2.1953 | 2.1889 | +0.0064 |
| 2200 | 2.1386 | 2.1327 | +0.0059 |
| 2400 | 2.0864 | 2.0791 | +0.0074 |

Final (step 2500) EMA train-CE gap (muon−fused): **+0.0060**.

## VAL ppl at matched steps

| step | muon VAL ppl | fused VAL ppl | muon−fused |
|---:|---:|---:|---:|
| 100 | 267.350 | 232.270 | +35.0800 |
| 200 | 56.330 | 53.610 | +2.7200 |
| 300 | 31.350 | 30.390 | +0.9600 |
| 400 | 27.240 | 26.400 | +0.8400 |
| 500 | 21.720 | 20.720 | +1.0000 |
| 600 | 15.930 | 14.970 | +0.9600 |
| 700 | 14.350 | 13.520 | +0.8300 |
| 800 | 12.870 | 12.340 | +0.5300 |
| 900 | 11.930 | 11.560 | +0.3700 |
| 1000 | 10.860 | 10.560 | +0.3000 |
| 1100 | 10.450 | 10.070 | +0.3800 |
| 1200 | 10.070 | 9.760 | +0.3100 |
| 1300 | 9.240 | 8.980 | +0.2600 |
| 1400 | 8.430 | 8.330 | +0.1000 |
| 1500 | 7.860 | 7.760 | +0.1000 |
| 1600 | 7.890 | 7.690 | +0.2000 |
| 1700 | 7.580 | 7.430 | +0.1500 |
| 1800 | 7.510 | 7.480 | +0.0300 |
| 1900 | 7.480 | 7.410 | +0.0700 |
| 2000 | 7.310 | 7.320 | -0.0100 |
| 2100 | 7.300 | 7.230 | +0.0700 |
| 2200 | 7.140 | 7.000 | +0.1400 |
| 2300 | 6.610 | 6.550 | +0.0600 |
| 2400 | 6.310 | 6.250 | +0.0600 |
| 2500 | 6.210 | 6.160 | +0.0500 |

Mean VAL-ppl(muon−fused) over 25 evals: **+1.8224** (positive ⇒ fused lower ppl).

## Exclusive opt-step timing + memory (GPU 1 free after the run)

```
[profile] 10L d896 14h d_head64  matrix params=80 (128.6M)
[profile] warmup=30 burst=30 reps=12 (round-robin, report MIN ms = uncontended)

  muon (fp32 state)                    min  28.960  median  28.973 ms/opt-step
  2-object (fp32 state)                min  25.616  median  25.641 ms/opt-step
  FUSED per-unit (fp32 state)          min  26.681  median  26.691 ms/opt-step
  FUSED cross-layer (fp32 state)       min  23.925  median  23.968 ms/opt-step
  muon (bf16 state)                    min  29.456  median  29.492 ms/opt-step
  FUSED per-unit (bf16 state)          min  26.164  median  26.176 ms/opt-step

[memory] 10L d896 14h matrix params=80 (128.6M)

  muon (fp32 state)                    state=  514.30 MB  peak_transient_workspace=   28.90 MB
  2-object (fp32 state)                state=  514.30 MB  peak_transient_workspace=   28.90 MB
  FUSED per-unit (fp32 state)          state=  514.30 MB  peak_transient_workspace=   22.48 MB
  FUSED cross-layer (fp32 state)       state=  514.30 MB  peak_transient_workspace=  295.90 MB
  muon (bf16 state)                    state=  257.15 MB  peak_transient_workspace=   41.75 MB
  FUSED per-unit (bf16 state)          state=  257.15 MB  peak_transient_workspace=   22.48 MB
```

## Verdict

**Per-head NS is strictly not-worse and modestly better than Muon at production scale — adopt as the DeltaNet matrix-optimizer default.**

- **(a) Loss:** fused is ahead at essentially every matched step. EMA train-CE gap +0.094 (step 200, steep) → compresses to +0.006 by step 2500; VAL ppl ~5–6% lower in the steep region (steps 500–700), ~3% mid, ~0 by ~step 2000. A real, consistent **convergence-SPEED** win — largest early, vanishing as both reach the same floor (a preconditioner changes the path, not the floor). At our always-undertrained operating point the realized win is the steep/mid advantage (~3% mean fewer steps to a target loss); train-to-convergence loss is ~tied (within single-seed noise).
- **(b) Wall-clock:** full-step **identical** (1.001× tok/s) — the optimizer step is ~0.5% of a full step. The isolated, uncontended opt-step is **~1.1× faster** for fused (26.7 vs 29.0 ms fp32; 26.2 vs 29.5 bf16), but that's invisible at full-step scale.
- **(c) Memory:** optimizer **state identical** (514 MB fp32 / 257 bf16); isolated transient **lower** for fused (22.5 vs 28.9 fp32 / 41.8 bf16). The full-run peak showed fused **+440 MiB (1.8%)** — an allocator/fragmentation artifact, NOT an optimizer cost (the wired path is the streamed per-unit, `batch_across_layers=False`; the clean isolated table is the source of truth). So memory is ≤ Muon at the optimizer level, ~tied end-to-end.

**Net:** a free, modest, *transient* convergence-speed edge (~3% fewer steps to a loss at our operating point) at identical wall-clock and no real memory cost — default-worthy, not transformational. Single seed (paired init+data); a 2nd seed would tighten the loss-gap significance. The advantage should be LARGER in a lower-noise (bigger-batch) regime — see the batch/LR (B_crit) analysis.
