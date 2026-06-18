# v17 pretrain — loss-term weight balance report

**Date:** 2026-06-18 · **ckpt:** `checkpoints/pretrain_v17_step4769_tok1250164736.pt`
(step 4769, 1.25 B tokens) · **run:** still live on GPU 0 at ~step 5400.
**Measurement:** GPU 1 only, `runs/measure_loss_grads.py` (standalone; reuses the
trainer's own loss functions). No training/model/launcher files were modified.

## TL;DR verdict

The balance is **largely healthy and the run is empirically going well** (αL
committed +0.36, VAL 46.9 → 7.5 and still dropping, no destabilization, latent
gradient bounded). The current weights are mostly inherited+validated and the
gradient measurement supports keeping them. There are **two concrete findings**:

1. **`pkm_diversity` (weight 0.01) contributes EXACTLY ZERO gradient** — it is
   computed on the *detached* stashed slot indices **and** weights
   (`experiments/memory_layer.py:283-284`), so the loss has no grad path to any
   parameter. It is an inert no-op per step (the docstring's "affects the
   value-table grads" claim is not realized). Slot diversity in the run
   (`slots/H≈25k/65k`, `top≈0.01`) is held up by ε-greedy + LayerNorm score-norm
   + the value-LR, **not** this loss.
2. **`ctx_addr_aux` (weight 0.2) has converged to ~zero gradient** — WM
   addressing is *solved* on the training distribution (`addr_loss≈0`,
   `p_bind=1.000`). The 0.2 weight is now mostly irrelevant (0.2 × 0 ≈ 0), but it
   occasionally amplifies a spike when a hard recall position misses (`addr→5.8`
   seen at log step 5260 → loss contribution ~1.17).

Everything else (LM 1.0, gate_entropy_aux 0.1, gist 0.1, z_loss 1e-4, latent
0.05/0.05) is pulling a measurable, sensibly-sized gradient. Recommendation:
**keep all weights as-is**; optionally retire `pkm_diversity` (it does nothing)
and optionally halve `ctx_addr_aux` to 0.1 (spike insurance). The only
*speculative* re-weight worth a cheap A/B is `gate_entropy_aux` 0.1 → 0.05 (it is
the heaviest aux and competes for the same trunk/FiLM capacity as LM).

---

## 1. Per-term gradient distribution (the load-bearing measurement)

Method: build the v17 model faithfully via `eval_bracket_structure.build_model_from_ckpt`
(+ re-attach trained `gist_heads`), `model.train()`, bf16 autocast + TF32 +
activation-checkpointing exactly as the run. Build ONE real batch (B=4, T=2048)
from `configs/pretrain_mix_v14_wmrecall_maskfix.yaml` via `MixedSourceStream`
(`emit_read_mask=True`, think-bursts on; batch selected to contain recall
positions so `ctx_addr` fires — 4 recall positions, 17 think positions). Each
term reconstructed by IMPORTING the trainer's functions
(`_nonthink_forward_loss`, `_z_loss_term`, `_ctx_addr_aux_loss`,
`_pkm_diversity_loss`, `LatentReasoningCotrain` / `_answer_span_latent_loss`).
For each term: zero grads → `(weight·loss).backward()` → record `‖grad‖` globally
and grouped by module. **Weights are the EFFECTIVE per-step weights** (n_micro
scaling cancels at `(loss/n_micro).backward()`; ramps at this step are ≈1.0).

### Term × module  ‖weighted grad‖  (`.` = 0 to displayed precision)

| term (weight) | trunk | FiLM | lm_head | emb/norm | WM ctxkey | pkm_val | pkm_rtr | gist_h | lat_adpt | gate_head | **GLOBAL** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **LM CE** (1.0) | 4.9e-1 | **1.55e0** | 2.2e-1 | 3.8e-2 | 6e-5 | 1.9e-4 | 2.8e-2 | . | . | 5.3e-2 | **1.646** |
| gate_entropy_aux (0.1) | 1.0e-1 | 3.0e-1 | . | 1.4e-2 | 2e-7 | 1.4e-5 | 2.4e-3 | . | . | 6.0e-2 | **0.319** |
| gist (0.1) | 8.6e-3 | 6.9e-3 | . | 7.5e-4 | ~0 | 1.6e-6 | 7.5e-4 | 6.3e-3 | . | . | **0.0128** |
| z_loss (1e-4) | 1.2e-2 | 5.0e-2 | 1.2e-3 | 8.6e-4 | ~0 | 1.6e-6 | 5.7e-4 | . | . | . | **0.0514** |
| ctx_addr_aux (0.2) | 1e-6 | 1e-6 | . | 3e-8 | 5e-7 | 3e-10 | 3e-8 | . | . | . | **1.6e-6** |
| **pkm_diversity (0.01)** | . | . | . | . | . | . | . | . | . | . | **0.000 (DETACHED)** |
| latent ans-CE R=2 (0.05) | 1.4e-1 | . | 3.0e-1 | 5.0e-3 | . | 9.9e-5 | 6.9e-3 | . | 3.6e-3 | . | **0.335** |
| latent ans-CE R=4 (0.05) | 1.2e-1 | . | 2.2e-1 | 5.3e-3 | . | 9.9e-5 | 1.1e-2 | . | 2.1e-3 | . | **0.247** |
| latent ans-CE R=8 (0.05) | 1.4e-1 | . | 2.2e-1 | 1.2e-2 | . | 1.0e-4 | 7.0e-3 | . | 1.4e-3 | . | **0.259** |
| latent gate-BCE R=2 (eff 2.5e-3) | 9.4e-3 | . | . | 2.8e-3 | . | 7.6e-6 | 6.0e-4 | . | 3.5e-4 | 1.3e-2 | **0.0161** |
| latent gate-BCE R=4 (eff 2.5e-3) | 5.2e-3 | . | . | 1.5e-3 | . | 3.9e-6 | 2.9e-4 | . | 1.3e-4 | 7.0e-3 | **0.0089** |
| latent gate-BCE R=8 (eff 2.5e-3) | 1.3e-2 | . | . | 3.9e-3 | . | 1.1e-5 | 9.9e-4 | . | 4.2e-4 | 1.7e-2 | **0.0217** |

(latent gate-BCE effective weight = `latent_reasoning_weight × latent_reasoning_gate_weight = 0.05 × 0.05`,
folded inside `_answer_span_latent_loss`.)

### Per-step share (using ONE representative latent rung, R=4)

| term | ‖w·grad‖ | share of Σ‖·‖ |
|---|--:|--:|
| **LM CE** | 1.646 | **72.1 %** |
| gate_entropy_aux | 0.319 | 14.0 % |
| latent ans-CE (R=4) | 0.247 | 10.8 % |
| z_loss | 0.051 | 2.2 % |
| gist | 0.013 | 0.6 % |
| latent gate-BCE (R=4) | 0.009 | 0.4 % |
| ctx_addr_aux | ~0 | ~0 % |
| pkm_diversity | 0 | 0 % |

(Share-of-sum-of-norms is a heuristic — gradients do not add in norm; see the
combined measurement next.)

### Combined per-step gradient (what `--grad_clip 1.0` actually clips)

All terms summed + one latent rung (R=3), single backward:

```
global ‖grad‖ = 1.4291   (vs Σ term-norms 2.92 → ~50% alignment/cancellation)
  FiLM           1.30e0     ← dominates the actual optimizer step
  trunk          5.01e-1
  lm_head        3.07e-1
  pkm_router     2.88e-2
  embed/out_norm 2.30e-2
  gist_heads     6.35e-3
  latent_adapter 3.39e-3
  pkm_values     2.19e-4
  WM(ctxkey)     6.25e-5
```

Key reads:
- **LM dominates** (72 % of summed norm, and is what drives the FiLM/trunk
  gradient in the combined step) — correct for a weight-1.0 next-token loss.
- **The single biggest gradient sink is FiLM** (1.30 in the combined step, from
  only 8 M params). Both LM and gate_entropy push hard through FiLM — expected:
  FiLM K=3 self-feed runs each block 3× and modulates the residual at every
  layer. This is *intra-LM* concentration, not a loss-term imbalance.
- **`--grad_clip 1.0` is mildly active** (combined 1.43 > 1.0 → ~0.70× scaling);
  consistent with the stable `tloss`/VAL in the log. Occasional log gnorm spikes
  (L0 0.5–1.0) line up with hard latent rungs (R=4) and the rare `addr` spike.
- **WM(ctxkey) and copy_head are gradient-starved by design**: `read_alpha` is
  frozen at 0 (`--mem_freeze_read_alpha`, additive injection OFF), WM contributes
  only through the copy head at recall positions (4/8192 here), and ctx_addr is
  solved → the WM encoders see ~6e-5 gradient. This is fine *if* recall stays
  solved (it is: const +0.99), but it means the addresser is no longer learning.
- **`pkm_values` raw grad is tiny (1.9e-4)** but it carries `--pkm_value_lr_mult
  100.0`, so its *update* is ~100× the raw grad → comparable to a 0.02 effective.
  Grad-NORM ≠ update magnitude (see caveats). The committed `αL=+0.36` and
  `row=2.7` in the log confirm the value table is genuinely learning.

---

## 2. Trajectory read (from `runs/pretrain_v17.log`)

Bucketed averages (every 500 steps):

| step | gist | ce | emit_ce | addr | pkm αL | emit% | reason |
|---:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.557 | 4.66 | 2.02 | 1.428 | 0.021 | 30.8 | – |
| 500 | 0.203 | 2.34 | 0.46 | 0.005 | 0.001 | 44.2 | – |
| 1000 | 0.137 | 1.84 | 0.46 | 0.002 | 0.038 | 54.8 | – |
| 2000 | 0.105 | 1.68 | 0.42 | 0.134 | 0.188 | 59.4 | 1.762 |
| 3000 | 0.113 | 1.66 | 0.40 | 0.002 | 0.325 | 60.0 | 0.990 |
| 4000 | 0.113 | 1.56 | 0.37 | 0.000 | 0.360 | 62.2 | 0.921 |
| 5000 | 0.112 | 1.52 | 0.37 | 0.278* | 0.349 | 63.8 | 0.739 |

VAL ppl: 46.9 → 22.3 → 15.4 → … → 7.6 → **7.53** (last), monotone down.
(*step-5000 `addr` bucket inflated by a single hard-recall spike of 5.8.)

- **Still moving:** `ce` (1.84→1.52), `reason` (1.76→0.74, with R ramping 2→8 in
  parallel), gate `emit%` (31→64), VAL ppl (still dropping).
- **Converged / saturated:** `gist` (plateaued ~0.11 by step 1000 — cosine ≈0.89,
  not improving), `emit_ce` (flat ~0.37 since step 500), `pkm αL` (committed
  +0.36, stable — the v7.1 path, NOT the v7.0 α-decay failure).
- **≈0 (solved):** `addr` — ~0.000 with `p_bind=1.0` on all but ~2/80 logged
  steps; the WM addresser solved the recall distribution.

---

## 3. Per-term balance assessment

| term | weight | grad share | verdict |
|---|--:|--:|---|
| **LM CE** | 1.0 | 72 % | **Well-balanced.** Dominant, as it must be. Includes the WM copy loss at recall positions (sparse → sub-threshold grad here). |
| **gate_entropy_aux** | 0.1 | 14 % | **Well-balanced, but the heaviest aux.** Real gradient into trunk/FiLM/gate_head; gate emit% is climbing healthily (31→64 %). The *only* term I'd consider trimming for LM capacity (speculative). |
| **latent ans-CE** | 0.05 | ~11 % | **Well-balanced; NOT starved.** 3rd-largest gradient; a major co-trainer of lm_head (0.22–0.30, ≥ LM's own lm_head grad) + trunk. **R-independent**: norm flat at R=2/4/8 (0.33/0.25/0.26) → 0.05 stays effective as R→8. |
| **z_loss** | 1e-4 | 2 % | **Well-balanced; measurably active** (grad 0.051 > gist!). logsumexp²=0.0037 → logits controlled. Validated PaLM value; doing real work despite tiny weight. |
| **latent gate-BCE** | 0.05×0.05 | 0.4 % | **OK.** Trains gate invoke+halt on ptr10dict; ~5× smaller than gate_entropy at the gate_head, different data → no conflict. |
| **gist** | 0.1 | 0.6 % | **Saturated but harmless.** Plateaued (cos≈0.89); grad 0.013, mostly into gist_heads + a little trunk. Near-spent regularizer; raising it won't beat the plateau, lowering frees ~nothing. Keep. |
| **ctx_addr_aux** | 0.2 | ~0 % | **Over-weighted relative to need, but harmlessly so.** SOLVED → ~0 grad. The 0.2 only matters during rare hard-recall spikes (`addr→5.8`), where it amplifies a transient. Spike insurance, not capacity tax. |
| **pkm_diversity** | 0.01 | 0 % | **WASTED — inert no-op.** Detached inputs → zero gradient to all params. Not "starved" (any weight × 0 = 0); structurally does nothing. |

Capacity-tax cross-check (memory `project_why_mechanisms_synthesis`,
`project_cold_latent_cotrain_destabilizes_pretrain`): the documented 287 M
VAL-drift came from *general-text* `latent_cotrain` + `gate_calibration` cold
engagement — **both OFF in v17**. The depth-matched `latent_reasoning` used here
is the validated fix; its gradient is bounded (~0.25, no spikes), the combined
gnorm is a stable 1.43, and VAL is *dropping* — so no capacity-competition
pathology is visible in this run.

---

## 4. Recommendations

**Primary: keep the weights as-is.** The measured gradient distribution matches
intent — LM dominant, gate/latent meaningfully co-training, z_loss/gist as
controlled regularizers — and the run is empirically healthy.

Concrete, evidence-backed tweaks (all optional, none urgent):

1. **Retire `pkm_diversity` (weight → 0), or fix its detach.** It is a verified
   zero-gradient no-op (`memory_layer.py:283-284` detaches both slot idx AND
   weights). *Either*: set `--pkm_diversity_weight 0.0` (saves the per-step
   scatter-add over 4×65 536 slots; behavior byte-identical since grad=0) — but
   **cannot change this run** (no relaunch). *Or* (future): if diversity pressure
   is actually wanted, stop detaching `_last_weights` for this loss so it bites
   the value table. Validate: PKM `top`/`slots/H` health log should be unchanged
   when set to 0 (proving it was inert). **Cheap; for the next run only.**

2. **Optionally lower `ctx_addr_aux` 0.2 → 0.1** for the next run. It is spent on
   the average step; halving it halves the rare hard-recall spike contribution
   without affecting the solved steady state. Validate: `addr` trajectory should
   stay ~0 with `p_bind≈1.0`; transient gnorm spikes should shrink. Low priority
   (grad_clip already contains the spikes).

3. **(Speculative) A/B `gate_entropy_aux` 0.1 vs 0.05.** It is the heaviest aux
   (14 %, into the same FiLM/trunk LM uses). *If* a future run wants to free
   trunk capacity for LM/VAL, this is the lever — but the gate is fragile and
   currently calibrating well, so only A/B it with a clear VAL/HumanEval target.
   Cheaply A/B-able as two short continuation forks measuring VAL + gate emit%.

No fresh from-scratch run is justified by the loss balance alone.

---

## 5. Caveats (honest)

- **Single batch (B=4, T=2048), single ckpt step (4769).** Magnitudes carry
  batch noise, esp. the sparse recall/copy/WM terms (4 recall positions here).
  Treat ratios as order-of-magnitude, not 3-sig-fig.
- **Grad NORM ≠ parameter UPDATE.** The optimizer applies Muon (most matmul
  params) vs AdamW (embeddings/tables/α/gates), `--pkm_value_lr_mult 100`, and
  separate LR groups. A small-norm group (e.g. `pkm_values` 1.9e-4) can still
  update substantially. The log's `uratio` (~2.8e-3, uniform across layers) is
  the better "is it learning" signal; this report measures gradient *allocation*,
  which is what the task asked for.
- **Latent term is R-dependent and stochastic** (random rung + examples). I
  measured R∈{2,4,8} to bracket it; per step only ONE rung fires (curriculum
  frontier ≈ R 2–4 at this step).
- **Ramps ≈ full at this step** (latent warmup 2000→5000 → ramp≈0.92→1.0;
  ctx_addr warmup done at 3000 → 1.0); I used ramp=1.0 (steady state).
- The gist loss was reconstructed with fresh-then-restored trained `gist_heads`
  (the ckpt stores them; `build_model_from_ckpt` drops them as a trainer-side
  aux, so the script re-attaches and loads the saved weights).

Reproduce: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python runs/measure_loss_grads.py`
