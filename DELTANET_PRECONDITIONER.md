# DeltaNet-tailored preconditioner vs Muon — design + fair head-to-head

**Date:** 2026-06-19 · **Probe:** MQAR T=256 / K=32, small DeltaNet (4L, d_model 256), fp32, GPU 1.

## TL;DR / verdict

A **per-head Newton–Schulz dualizer** on DeltaNet's q/k/v projections is the
single most tractable + theory-justified idea, and it is a **real, consistent
iso-step win**: it reaches a recall threshold **~7–13 % fewer steps** than plain
Muon, holding across **every LR (3) and every seed (4)** and both thresholds
(0.90, 0.99). The win **scales with the number of heads** (nh=8 → +12.7 %, nh=2
→ +5.2 %), i.e. it tracks DeltaNet's *head-independence* structure exactly as the
modular-norm theory predicts — it is a genuine **direction** improvement, not a
step-size artifact.

At the **narrow probe** it does **not** beat Muon on wall-clock (the iso-step
gain is eaten by a small-matrix launch-overhead in the optimizer step), so on
this probe it is **iso-step-positive, wall-clock-neutral**. BUT — and this is the
key difference from SOAP/Shampoo (which lost because they are *fundamentally*
more expensive) — the per-head optimizer cost **reverses at production width**:
at d_model=896 / 14 heads the per-head step is measured **cheaper** than Muon
(10.87 vs 12.13 ms; many small 64×896 orthogonalizations beat one 896×896 gram).
So the wall-clock wash is an artifact of the tiny probe, not the method.

**Coupling q and k (the headline "QK-gauge" idea) is a clean negative**: the
`qk_coupled` arm is consistently *between* Muon and per-head (worse than plain
per-head). Per-head NS is already gauge-equivariant; forcing q⊥k over-constrains.
The win is **purely from per-head structure**, not from coupling.

**Recommendation: worth a larger test.** Re-implement per-head NS as a single
*fused* matrix optimizer (no second optimizer object, batched bmm — already
prototyped) and A/B it against Muon at the **287 M production scale on real
pretrain per-source CE** (the repo's discriminating metric). Unlike SOAP it has
no fundamental per-step cost penalty at scale, so the iso-step win should carry
to wall-clock.

---

## 1. Brainstorm — candidate DeltaNet-tailored updates

Muon = steepest descent under the spectral (RMS→RMS operator) norm via
Newton–Schulz orthogonalization — the modular-norm **dualizer** for a *generic*
linear layer. It is architecture-naive to DeltaNet:

> `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`,  `o_t = S_t q_t`

The relevant DeltaNet symmetries:

- **S1 — head independence.** q/k/v_proj produce `n_heads` *independent*
  associative memories; the recurrence never mixes heads. The modular-norm dual
  of a multi-head linear op is the **per-head** spectral norm.
- **S2 — QK key-space gauge.** `k` enters only via `k kᵀ` and pairs with `v` via
  `v kᵀ`; `q` reads via `S q`. The loss sees only relative q/k geometry → a
  per-head `O(d_head)` rotation `q→Uq, k→Uk` shared by q_proj/k_proj leaves the
  output invariant.
- **S3 — radial gauge.** `qk_norm='l2'` L2-normalizes each head's q,k → the
  per-head vector *magnitude* is gauge.
- **S4 — Stiefel drift.** gated / DeltaProduct variants push the state toward a
  near-orthogonal manifold.

Candidates:

| # | idea | symmetry | tractable? |
|---|------|----------|-----------|
| **(a) per-head NS** | NS each head's output slice of q/k/v independently instead of the whole d_model×d_model matrix | **S1** (+ is automatically equivariant under S2/S3, see below) | **yes — primary** |
| (b) QK-coupled NS | stack `[g_q[h]; g_k[h]]` per head, NS jointly → orthogonalize q against k | S2 | yes — secondary arm |
| (c) radial-quotient | project out the per-head radial gradient component before NS (Riemannian on the L2 sphere) | S3 | medium |
| (d) Stiefel retraction | constrain projections near-orthogonal via a retraction | S4 | low (gated only) |
| (e) learned Kron metric | PSGD-Kron / Lie-group whitening restricted to DeltaNet layers ("can preconditioners be learned") | S1–S3 in the Lie structure | high cost; deferred |

**Why per-head NS already respects S2 & S3 (a subtle, decisive point):**
Newton–Schulz is **equivariant under left-orthogonal multiplication**:
`NS(UG) = U·NS(G)` for orthogonal `U`. So applying per-head NS to q_proj and
k_proj *separately* already commutes with the S2 gauge `q→Uq, k→Uk`. That means
candidate (b)'s explicit coupling is **not needed for gauge-correctness** — it
adds an *extra* constraint (q⊥k bases). This prediction is exactly what the
experiment confirms: (b) helps *less* than (a).

**Chosen primary = (a) per-head NS.** Cleanest theory tie (S1 modular-norm
dual), cheapest, and at scale it is *fewer FLOPs* than whole-matrix NS. Kept (b)
as a second arm to test the QK-coupling hypothesis head-on. (c)/(d)/(e) deferred.

`o_proj` is **intentionally NOT tailored** — it is the one matrix that *mixes*
heads (the readout), so the whole-matrix Muon dual is correct there. `b_proj`
(β logits, n_heads×d_model) is treated per-head (per-row NS) per the task spec;
it is tiny and immaterial.

### RMS-consistency (why the same LR is fair across arms)

A `d_head×d_in` row-orthonormal slice has Frobenius² = d_head; summed over heads
= `n_heads·d_head = d_model` = the Frobenius² of the whole-matrix orthogonal
update. Measured: whole 209.6 vs per-head 187.9 (within ~10 %, NS5's
S'~U(0.5,1.5) + bf16). So the same `lr` produces ~the same effective step — and
the LR sweep absorbs the residual. (If anything per-head takes *slightly smaller*
steps yet still converges faster → the win is direction quality, not step size.)

---

## 2. Implementation

Standalone, does **not** touch the live v18 stack (`train_lm.py`, `optim_utils.py`,
`model.py`, …).

- **`experiments/exp_deltanet_precond_optim.py`** — `DeltaNetProjMuon`, a
  `torch.optim.Optimizer`. Reuses torch's **exact** Newton–Schulz
  (`_zeropower_via_newtonschulz`), momentum/nesterov, decoupled WD and `_adjust_lr`
  — the *only* thing that differs from `torch.optim.Muon` is where the
  orthogonalization boundary is drawn (per-head / qk-coupled vs whole-matrix).
  Per-head NS is vectorized with a batched bmm (`_ns_batched`), numerically
  identical to per-slice torch NS (max |Δ| 0.017, pure bf16 reduction-order
  noise). `build_units_from_model` auto-discovers q/k/v/b per block from the FLA
  DeltaNet wrapper and routes o_proj + MLP to plain Muon.
- **`experiments/exp_deltanet_precond.py`** — the fair MQAR harness.
- **`experiments/exp_deltanet_precond_analyze.py`** — single-seed speed table.
- **`experiments/exp_deltanet_precond_msagg.py`** — multi-seed mean±std aggregation.

---

## 3. Fair experiment setup

- **Arms** (matrix optimizer on q/k/v/b; *everything else identical*):
  **A `muon`** plain `torch.optim.Muon` on all 2D hidden matrices ·
  **B `perhead`** `DeltaNetProjMuon(per-head)` on q/k/v/b + Muon on o_proj+MLP ·
  **C `qk_coupled`** `DeltaNetProjMuon(qk-coupled)` on q/k/v/b + Muon on o_proj+MLP.
- **Identical init** (same `--seed` → `torch.manual_seed` before build) and
  **identical data + order** (per-step batch from a generator re-seeded per run;
  fixed val set). Verified: step-0 train loss is byte-identical across arms
  within a seed (4.30942 @ s0, 4.31912 @ s1), and differs across seeds.
- **Identical** step budget / batch (64) / T (256) / n_pairs (32) / cosine LR /
  AdamW group (embeddings/lm_head/pos/1D/conv at a held-constant `--lr 3e-3`).
- **Swept knob:** matrix LR `--lr_mat ∈ {5e-3, 1e-2, 2e-2}` (+ {2.5e-3, 4e-2} in
  the wide single-seed scan) — ≥3 LRs/arm, report each arm's best.
- **fp32** (MQAR's masked loss collapses in bf16 — documented; the FLA kernel
  still runs bf16 internally, as in `train_mqar.py`).
- **Discriminating metric = convergence SPEED.** The task saturates to
  recall=1.0 within the budget, so final recall does not discriminate; we report
  **(interpolated) steps-to-recall-threshold**, multi-seed mean±std, iso-step and
  iso-wallclock. 4 seeds × 3 LRs × 3 arms; eval every 5 steps.

---

## 4. Results

### Iso-step: steps-to-recall≥0.90 (mean ± std over 4 seeds)

| lr_mat | A muon | B perhead | C qk_coupled |
|--------|-------:|----------:|-------------:|
| 5e-3   | 244 ± 17 | **219 ± 12** | 227 ± 13 |
| 1e-2   | 172 ± 10 | **158 ± 6**  | 163 ± 7  |
| 2e-2   | 141 ± 7  | **130 ± 5**  | 136 ± 5  |

Per-head beats Muon at **every** LR and seed; qk_coupled is always *between*.

### Per-arm BEST (per-seed fastest LR; mean ± std over 4 seeds)

| arm | steps-to-0.90 (iso-step) | wall-to-0.90 (iso-wall) | opt ms/step | Δ vs Muon (iso-step) | Δ vs Muon (iso-wall) |
|-----|-------------------------:|------------------------:|------------:|---------------------:|---------------------:|
| A muon       | 141 ± 7 | **2.02 ± 0.13 s** | 4.20 | — | — |
| B **perhead**| **130 ± 5** | 2.05 ± 0.09 s | 6.59 | **−11 (−7 %)** | +0.02 s (+1 %) |
| C qk_coupled | 136 ± 5 | 2.09 ± 0.12 s | 5.85 | −5 (−4 %) | +0.07 s (+3 %) |

(steps-to-0.99 tells the same story: muon 152, **perhead 140 (−8 %)**, qk 146.)

**Iso-step:** per-head is a clean win. **Iso-wallclock (this probe):** neutral —
the per-head optimizer step costs 6.59 ms vs Muon's 4.20 (two matrix-optimizer
objects + small-matrix bmm launch overhead), which cancels the ~10-step gain.

### Mechanism — the win tracks head count (steps-to-0.90, lr 1e-2, n=2)

| config (d_model=256) | A muon | B perhead | per-head gain |
|----------------------|-------:|----------:|--------------:|
| n_heads=2, d_head=128 (big per-head gauge) | 133.4 | 126.5 | +5.2 % |
| n_heads=8, d_head=32 (many heads)          | 260.5 | 227.5 | **+12.7 %** |

The gain grows with the **number of heads**, *not* d_head → it is
**head-independence (S1)**, not the per-head gauge size, that plain Muon
mistreats. Production DeltaNet uses **14 heads** → the iso-step win should be
*larger* there than this 4-head probe's −7 %.

### Cost reverses at production width (opt ms/step, d_model=896 / 14 heads / d_head 64)

| arm | opt ms/step |
|-----|------------:|
| A muon       | 12.13 |
| B **perhead**| **10.87 (−10 %)** |

At the narrow probe per-head was 1.6× *slower* (launch overhead on 64×256
slices). At production width per-head is **cheaper** — 14 small (64×896) NS with
64×64 grams beat one 896×896-gram NS. So the iso-wallclock wash is a tiny-probe
artifact that **inverts** at scale. (Unlike SOAP, which is fundamentally
eigendecomposition-bound and lost on wall-clock — `project_soap_vs_muon_optimizer`.)

---

## 5. Verdict + next step

- **Does it beat Muon iso-step?** **Yes — modest but robust** (~7 % to 0.90,
  ~8 % to 0.99; up to +12.7 % at 8 heads). Consistent across all LRs/seeds; it is
  a direction improvement grounded in DeltaNet's head-independence (S1).
- **Does it beat Muon iso-wallclock on this probe?** **No — neutral** (+1 %),
  because the per-head optimizer step is more expensive *at tiny matrix sizes*.
- **Is that cost fundamental?** **No.** It is small-matrix launch overhead +
  running a second optimizer object. The per-head step is measured **cheaper than
  Muon at production width**, and per-head NS is asymptotically fewer FLOPs than
  whole-matrix NS. This is the decisive difference from SOAP.
- **QK-coupling (the headline gauge-coupling idea)?** **Negative.** Per-head NS is
  already gauge-equivariant; explicit coupling over-constrains and helps less.
  Don't pursue it.

**Recommended next step (worth it):**
1. Fold per-head NS into a **single fused** matrix optimizer (one foreach pass,
   batched bmm; route q/k/v per-head + o_proj/MLP whole-matrix in the same step)
   to remove the second-optimizer overhead — convert the iso-step win to
   wall-clock at all scales.
2. A/B **per-head vs plain Muon at the 287 M production scale on real pretrain
   per-source CE** (the repo's discriminating signal; MQAR saturates), with the
   14-head trunk where the win should be largest. Keep AdamW/embeddings identical;
   sweep matrix LR; report per-source CE iso-step *and* iso-wallclock.
3. If positive, the cheap follow-on is candidate (c) radial-quotient (S3) on top
   of per-head; skip (b) qk-coupling and (e) learned-Kron (cost not justified by
   this probe).

If the production A/B is flat, this is a clean negative at scale — but the probe
result (real iso-step win + cheaper-at-scale step) makes the test worth running.

---

## 6. Exact commands + script paths

Scripts (all standalone, GPU-1-only, do not touch the live stack):
- `experiments/exp_deltanet_precond_optim.py` — `DeltaNetProjMuon` + `build_units_from_model`
- `experiments/exp_deltanet_precond.py` — fair MQAR harness (one arm × one LR per run)
- `experiments/exp_deltanet_precond_analyze.py` — single-seed speed table
- `experiments/exp_deltanet_precond_msagg.py` — multi-seed mean±std aggregation

```bash
export PYTHONPATH="$PYTHONPATH:."

# Multi-seed fair sweep (4 seeds × 3 LRs × 3 arms; identical init+data per seed):
for seed in 0 1 2 3; do ds=$((1234+seed)); vs=$((999+seed));
 for arm in muon perhead qk_coupled; do for lr in 5e-3 1e-2 2e-2; do
   CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_precond.py \
     --arm $arm --lr_mat $lr --steps 500 --eval_every 5 --seed $seed \
     --data_seed $ds --val_seed $vs --tag ${arm}_lr${lr}_s${seed} \
     --out_dir runs/deltanet_precond_ms
 done; done; done

# Aggregate (mean±std steps-to-threshold, iso-step + iso-wallclock):
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_precond_msagg.py runs/deltanet_precond_ms 0.90
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_precond_msagg.py runs/deltanet_precond_ms 0.99

# Head-count mechanism ablation (nh2/dh128 vs nh8/dh32, d_model=256 fixed):
for cfg in "2 128" "8 32"; do set -- $cfg; for arm in muon perhead; do for seed in 0 1; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_precond.py \
    --arm $arm --lr_mat 1e-2 --n_heads $1 --d_head $2 --steps 500 --eval_every 5 \
    --seed $seed --data_seed $((1234+seed)) --val_seed $((999+seed)) \
    --tag ${arm}_nh$1dh$2_s${seed} --out_dir runs/deltanet_precond_head
done; done; done

# Production-width optimizer-cost check (timing only):
for arm in muon perhead; do CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  experiments/exp_deltanet_precond.py --arm $arm --lr_mat 1e-2 \
  --d_model 896 --n_heads 14 --d_head 64 --steps 80 --eval_every 80 \
  --tag wide_$arm --out_dir /tmp/dnp_widetime; done
```

Outputs: `runs/deltanet_precond_ms/*.json` (multi-seed), `runs/deltanet_precond_head/*.json`
(head ablation). Each JSON carries the full recall-vs-step-and-wall history plus
isolated cuda-synced `opt_ms_per_step`.
