# GPU pickup plan — PD-SSM family validation

This is a self-contained handoff doc. Anyone (including future-you on
the GPU box) should be able to pick this up cold and run the
experiments in order, knowing what numbers to expect.

## Branch & state

- Branch: `pd-ssm` (off `main` at commit 64a2956)
- Pre-flight: run `python experiments/test_pd_closure.py` — six fast
  numerical tests verifying PD closure, layer ↔ matmul-reference
  equivalence, complex-D unit-disk constraint, and Muon end-to-end. All
  pass on MPS in a few seconds.
- Modal app: `modal_app.py` at the repo root has six entrypoints —
  `smoke`, `priority1..5`, `all_priorities`, plus a `custom` escape hatch.
  See the docstring at the top of the file.
- Three new layers in `experiments/layers.py`:
  - `PDScanAttention` — vector-state PD-SSM (Terzić et al. 2025, arXiv:2509.22284)
  - `PDKVScanAttention` — matrix-state PD with rank-1 KV write (novel, the natural PD/DeltaNet hybrid)
  - `ComplexPDScanAttention` — PD with complex unit-disk D (novel, escapes Landau bound)
- New file `experiments/optim_muon.py` — Keller Jordan's Muon, standalone port
- Smoke scripts: `smoke_pd_ssm.py`, `smoke_pdkv_mkar.py`, `smoke_complex_pd_landau.py` (all flag `--optim adamw|muon`)
- ARCHES dicts in all `train_*.py` drivers updated with `pd_ssm`, `pd_kv`, `complex_pd`

## What's already validated on MPS

| Test | Config | Result | Verdict |
|---|---|---|---|
| Parity, T=64 | N=8, 500 steps | pd_ssm 100% / linear 49% | clean |
| Mod-2 (parity), T=32 | N=4, 800 steps | pd_ssm 100% / complex_pd 100% / linear 43% | both PD variants solve |
| Mod-3, T=32 | N=4, 800 steps | pd_ssm 31% / complex_pd 100% / linear 97%† | pd_ssm stalls on discrete σ optim |
| **Mod-5, T=32** | **N=4, 800 steps** | **pd_ssm 23% (chance) / complex_pd 82% / linear 21%** | **Landau bound g(4)=4 confirmed; complex_pd ✓** |

† linear at T=32 essentially memorizes the cumsum; should fail at T≥128.

**Two findings:**
1. **Expressivity (Landau bound):** pd_ssm-N4 cannot do mod-5 (Z₅ not embeddable in S₄). Complex-PD-N4 succeeds via e^{2πi/5}.
2. **Optimization (discrete σ):** pd_ssm-N4 fails *even mod-3*, which it should be able to do (Z₃ ≤ S₄). The straight-through estimator on σ_t is the bottleneck. Complex-PD avoids this — phase is continuous.

## The headline scientific claim to validate at scale

> **ComplexPDScanAttention Pareto-dominates PDScanAttention** on (a) the
> Krohn-Rhodes class of recognizable monoids and (b) the optimization
> landscape, at the same O(N²) per-token cost. The framework's eigenvalue
> axis (the Grazzi negative-eigenvalues theorem) reaches its full
> conclusion when D is allowed to range over the complex unit disk
> rather than just [-1, 1].

If the GPU experiments confirm this on (mod-p, T≥256, full-config), it's
publishable as an extension of PD-SSM (arXiv:2509.22284).

## Priority 1 — Mod-p capacity sweep (algebra-gap test)

**Goal:** confirm Landau-bound prediction at scale.

**Action:** Force state_dim=4 for pd_ssm/complex_pd. The `--state_dim`
arg is now wired through `train_modular.py` (and `train_mqar.py`).
Just run:

```bash
python experiments/train_modular.py \
    --arches deltanet,deltanet_negeig,pd_ssm,complex_pd \
    --p 2 3 5 7 11 \
    --T 128 \
    --state_dim 4 \
    --steps 5000 --batch 256 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
    --lr 3e-3
```

Or via Modal: `modal run modal_app.py::priority1`

**Predicted end_acc per (p, arch):**

| p  | deltanet | deltanet_negeig | pd_ssm-N4 | complex_pd-N4 |
|----|----------|-----------------|-----------|----------------|
| 2  | ✗ chance | ✓ 100%          | ✓ 100%    | ✓ 100%         |
| 3  | ✗        | ✓ ~80-100%      | ~30-50%‡  | ✓ ~95-100%     |
| 5  | ✗        | ✗ ~9% (collapse)| ✗ ~20% ✗  | ✓ ~95-100%     |
| 7  | ✗        | ✗               | ✗ ~14% ✗  | ✓ ~95-100%     |
| 11 | ✗        | ✗               | ✗ ~9% ✗   | ✓ ~95-100%     |

‡ pd_ssm-N4 *could* solve mod-3 (Z₃ ≤ S₄) but the discrete-σ optim
makes it unreliable. May solve with Muon (see Priority 4).

**Pass criterion for the framework:** complex_pd ≥ 95% on mod-5,7,11
where pd_ssm sits at chance. If complex_pd succeeds where pd_ssm fails
on mod-5 specifically, the Landau-bound claim is empirically validated.

**Time estimate:** ~5 min/run × 5 p-values × 4 arches = **~100 min total** on a single H100 / A100.

## Priority 2 — Mod-5 long-T (DeltaNet collapse reproduction)

**Goal:** reproduce the repo's existing finding that DeltaNetNegEig
catastrophically collapses on mod-5 at T=512, and demonstrate that
PD-family architectures don't.

```bash
python experiments/train_modular.py \
    --arches deltanet_negeig,pd_ssm,complex_pd \
    --p 5 \
    --T 512 \
    --state_dim 8 \
    --steps 5000 --batch 256 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32
```

**Predictions:**
- deltanet_negeig: ~9% (below 20% chance) — known result from RESULTS.md.
- pd_ssm-N8: 100% if discrete σ optim doesn't stall (g(8)=15 includes 5).
- complex_pd-N8: 100% confidently (no σ optim difficulty).

**Pass criterion:** Both PD variants > 95% where deltanet_negeig collapses to chance. This puts a clean number on the framework's claim that the Grazzi eigenvalue ceiling is broken by complex roots of unity.

**Time estimate:** ~30 min on H100.

## Priority 3 — MKAR head-to-head (capacity-gap test, novel arch)

**Goal:** validate that PD-KV (matrix state) closes the capacity gap that
vector-state PD-SSM cannot, at no expressivity cost.

```bash
# --n_pairs now accepts multiple values (one command sweeps all K).
python experiments/train_mqar.py \
    --arches linear,deltanet,pd_ssm,pd_kv \
    --T 256 \
    --n_pairs 4 8 16 32 \
    --vocab 64 \
    --steps 5000 --batch 256 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32
```

Or via Modal: `modal run modal_app.py::priority3`

**Predicted recall@K (MQAR query-positions accuracy):**

| K  | linear | deltanet | pd_ssm-N32† | pd_kv-N8×D32 |
|----|--------|----------|-------------|---------------|
| 4  | ≥0.95  | ≥0.95    | ≥0.95       | ≥0.95         |
| 8  | ≥0.95  | ≥0.95    | ~0.85-0.95  | ≥0.95         |
| 16 | ≥0.90  | ≥0.95    | ~0.40-0.70‡ | ≥0.90         |
| 32 | ~0.70  | ≥0.95    | ~0.05-0.15  | ~0.70-0.90    |

† Default `state_dim=d_head=32` for pd_ssm.

‡ pd_ssm vector state is N=32-dim, can hold ~32 real numbers; with K=16 KV bindings each takes ~2 dims of capacity, getting tight. With K=32, saturated.

**Pass criterion (the novel-architecture claim):** pd_kv ≥ 0.85 on K=32
where pd_ssm < 0.20. This is the cleanest demonstration that
matrix-state with PD transition combines both axes.

**Headline metric:** plot of recall vs K for each arch, looking for the
crossover point where pd_ssm falls off the cliff but pd_kv tracks
deltanet/linear.

**Time estimate:** ~10 min/run × 4 K-values × 4 arches = **~160 min**.

## Priority 4 — Muon vs AdamW on discrete σ

**Goal:** test whether Muon's orthogonalised momentum fixes the
discrete-σ optimization stall observed on mod-3 / mod-5 with pd_ssm.

```bash
# AdamW baseline (already known)
python experiments/smoke_complex_pd_landau.py \
    --p 3 5 \
    --T 128 --steps 3000 --batch 128 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
    --state_dim 8 \
    --optim adamw \
    --arches pd_ssm,complex_pd

# Muon variant
python experiments/smoke_complex_pd_landau.py \
    --p 3 5 \
    --T 128 --steps 3000 --batch 128 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
    --state_dim 8 \
    --optim muon \
    --arches pd_ssm,complex_pd
```

**Predictions:**
- complex_pd: AdamW and Muon both reach ≥95% on mod-3 and mod-5 (no
  discrete-search difficulty). Muon may converge faster (≥1500 steps).
- pd_ssm: AdamW stalls on mod-3 and mod-5 (chance ~20%). Muon may break
  through one or both — that's the open question.

**Pass criterion (the optimization claim):** Muon-pd_ssm reaches ≥80% on
mod-3 where AdamW-pd_ssm stalls at ~30%. If yes, the discrete-σ stall is
an optimizer issue (specifically: the AdamW per-parameter momentum is
not the right fit for one-hot-of-softmax gradients). If Muon also stalls,
the issue is the straight-through estimator itself, and Gumbel-softmax
+ temperature annealing (or Sinkhorn) would be the next thing to try.

**Time estimate:** ~10 min/run × 4 runs = **~40 min**.

## Priority 5 (optional but cheap) — S5 word-problem at T=128

**Goal:** see if PD/Complex-PD beats DeltaNetNegEig on the repo's existing
S₅ benchmark (currently DeltaNetNegEig 0.978 pos_recall vs Hybrid 0.71).
PD-SSM with N≥5 can encode S₅ permutations natively in σ.

```bash
python experiments/train_s5.py \
    --arches deltanet_negeig,pd_ssm,complex_pd \
    --T 128 \
    --steps 5000 --batch 256 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32
```

**Prediction:** pd_ssm-N32 and complex_pd-N32 should both ≥ 95%
pos_recall — beating DeltaNet's 0.978 because they realize S₅ as actual
permutations rather than via reflections-as-transpositions.

If this holds, it's a direct mechanism-level explanation for *why*
PD beats DeltaNet on this discrete-state task: PD natively realizes
the standard permutation representation, DeltaNet has to compose
Householder reflections.

**Time estimate:** ~30 min.

## Priority 6 (optional, longer) — TinyStories LM PPL

**Goal:** confirm none of the PD variants catastrophically fail on real
text. Hybrid v2 currently 1.19× DeltaNet PPL. PD variants should be
in the same range (slightly worse if anything because of the discrete
σ; complex_pd should be closer to deltanet baseline).

```bash
python experiments/train_lm.py \
    --arches deltanet,pd_ssm,pd_kv,complex_pd \
    --steps 5000
```

**Expected:** PPL within 1.0-1.5× DeltaNet baseline (5.65). pd_ssm may be
worse (1.5-2.0×) because of the discrete σ. complex_pd close to
DeltaNet. pd_kv may match or beat DeltaNet on real text because of the
extra structure.

**Time estimate:** ~3-4 hours on H100.

## Diagnostic: what to look at if predictions fail

- **Inspect σ histograms** for pd_ssm: per-head, plot the distribution
  of argmax-σ over a batch. If σ is highly diffuse (close to identity),
  the model hasn't found permutation structure. If σ is concentrated
  but not on a useful pattern, optimization is stuck in a local minimum.
- **Inspect D distributions** for pd_ssm and complex_pd: real D should
  have entries near ±1 for parity-style tasks. Complex D should have
  phases at multiples of 2π/p for mod-p tasks.
- **Loss curve shape:** if pd_ssm loss stays at log(p) with no decrease,
  the gradient signal isn't reaching σ at all — something's broken in
  the straight-through pathway.
- **Try smaller state_dim:** if mod-5 succeeds at N=4, the architecture
  is right; if it only succeeds at N=8 or 16, the discrete σ search
  has a bigger landscape than expected and may need Gumbel-softmax.

## Summary table — what counts as "shallow signal validates the framework"

| Priority | Pass criterion | Confidence delivered |
|----------|----------------|----------------------|
| 1 (Mod-p Landau) | complex_pd ≥ 95% on mod-5,7,11 where pd_ssm chance | Strongly confirms eigenvalue axis |
| 2 (Mod-5 long-T) | Both PD variants > 95% where deltanet_negeig 9% | Beats published collapse |
| 3 (MKAR cap-gap) | pd_kv ≥ 0.85 at K=32 where pd_ssm < 0.20 | Validates novel arch |
| 4 (Muon optim) | Muon-pd_ssm ≥ 80% on mod-3 where AdamW 30% | Confirms optim diagnosis |
| 5 (S₅) | pd_ssm/complex_pd ≥ 95% pos_recall | Mechanism explanation |
| 6 (LM) | PPL within 1.5× DeltaNet | No catastrophic regression |

If 1, 2, 3 all pass → the framework + novel architectures are
publishable. 4 nails the mod-3 mystery. 5 + 6 round out the story.
