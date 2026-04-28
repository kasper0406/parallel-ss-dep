# GPU run plan — PD-SSM family validation

Three new layer cells were added (`PDScanAttention`, `PDKVScanAttention`,
`ComplexPDScanAttention`) plus the Muon optimizer. MPS sequential scans
are too slow for full-scale validation; this doc lists the experiments
that should be run on GPU (CUDA) to validate the framework's predictions.

## Layer summary

| Cell                       | State            | Cost / token | Predicted wins                      |
|----------------------------|------------------|--------------|--------------------------------------|
| PDScanAttention            | vector R^N       | O(N²) gather | parity, mod-p (p ≤ Landau g(N)), S_n in 1 layer |
| PDKVScanAttention          | matrix R^{N×D}   | O(N·D)       | MKAR / associative recall + state-tracking |
| ComplexPDScanAttention     | complex R^{N×2}  | O(N²) gather | mod-p for any p (escapes Landau bound) |

## Already validated on MPS

- **Parity at T=64, N=8, 500 steps**: PDScanAttention 100% / LinearAttention 49% (chance). Margin +50.8 pp. (`smoke_pd_ssm.py`)
- **Mod-p Landau test, N=4, T=32, 800 steps** (`smoke_complex_pd_landau.py`):

  | p | linear | pd_ssm-N4 | complex_pd-N4 | takeaway |
  |---|--------|-----------|---------------|----------|
  | 2 | 0.43 ✗ | 1.00 ✓ | 1.00 ✓ | both solve parity |
  | 3 | 0.97 ✓† | **0.31 ✗** | 1.00 ✓ | pd_ssm stalls (optimization, not expressivity) |
  | 5 | 0.21 ✗ | **0.23 ✗** | **0.82 ✓** | pd_ssm fails (Landau g(4)=4), complex_pd ✓ |

  † linear at T=32 essentially memorizes; should fail at T≥128.

  **Two findings:** (1) pd_ssm fails mod-5 with N=4 *for fundamental Landau-bound reasons* — Z₅ not embeddable in S₄; complex_pd succeeds via e^{2πi/5}. (2) pd_ssm even fails mod-3 (which should fit in S₄ via a 3-cycle) — straight-through estimator on σ is the bottleneck. complex_pd bypasses this entirely because phase is continuous.

## To run on GPU — three priority experiments

### Priority 1 — Mod-p capacity sweep (algebra-gap test)

Validates the Landau-function prediction: `g(N)` bounds the maximum mod-p
solvable by PDScanAttention with state dim N.

```bash
# Reference DeltaNet baseline (the strongest comparable single-cell)
python experiments/train_modular.py \
    --arches deltanet,deltanet_negeig,pd_ssm,complex_pd \
    --p 2 3 5 7 11 \
    --T 128 \
    --steps 5000 \
    --batch 256 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
    --lr 3e-3
```

Sharp predictions:
- pd_ssm with default state_dim=d_head=32 → handles mod-p for p ≤ g(32). g(32) is large (32 has element of order 32 trivially via single 32-cycle). So all p ≤ 32 should be reachable. Actually g(32) ≥ 7 × 11 × 13 = 1001 if we count primes summing to ≤ 32. So mod-11 is well within reach.
- For the **sharp** Landau test, force state_dim=4: pd_ssm-N4 fails mod-5,7,11; complex_pd-N4 succeeds all.

To force state_dim, edit `train_modular.py`:`train_one()` to pass
`state_dim=4` when arch is `pd_ssm`/`complex_pd` (or expose it as a CLI arg).

### Priority 2 — Mod-5 at long-T (DeltaNet collapse reproduction)

The repo's existing finding: DeltaNetNegEig collapses to 9.1% on mod-5 at
T=512. Predicted: PDScanAttention solves it cleanly (assuming N ≥ 5 and
the discrete-σ optimization completes — see Priority 3).

```bash
python experiments/train_modular.py \
    --arches deltanet_negeig,pd_ssm,complex_pd \
    --p 5 \
    --T 512 \
    --steps 5000 \
    --batch 256 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32
```

If pd_ssm fails: run with `--steps 10000` and inspect σ histograms (whether
the model finds a 5-cycle structure). The MPS test at N=8, 1000 steps stuck
at chance ~20% — likely an optimization issue, not expressivity.

### Priority 3 — MKAR capacity gap (PD-SSM vs PD-KV)

The headline novel-architecture comparison: vector-state PD saturates
around K ≈ N; matrix-state PD-KV holds K ≈ N · D.

```bash
# K sweep with N=8 fixed for both PD variants
python experiments/train_mqar.py \
    --arches linear,deltanet,pd_ssm,pd_kv \
    --T 256 \
    --n_pairs 4 8 16 32 \
    --vocab 64 \
    --steps 5000 \
    --batch 256 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32
```

Note: train_mqar.py currently takes a single --n_pairs. To run the sweep,
either modify it to accept `nargs="+"` or run separately per K.

Predictions:
- K=4 (well below all capacities): all solve.
- K=8 (≈ N for pd_ssm): pd_ssm starts to saturate.
- K=16: pd_ssm clearly worse than pd_kv ≈ deltanet ≈ linear.
- K=32: pd_ssm random, others still strong.

### Priority 4 — Muon vs AdamW on the discrete σ problem

The mod-5 N=8 stall on MPS at 1000 steps is consistent with hard discrete
optimization. Muon's orthogonalized momentum may help the σ logits
converge faster.

```bash
# Compare on mod-5 specifically
python experiments/smoke_complex_pd_landau.py \
    --p 5 --T 128 --steps 3000 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
    --state_dim 8 \
    --optim adamw  # baseline

python experiments/smoke_complex_pd_landau.py \
    --p 5 --T 128 --steps 3000 \
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
    --state_dim 8 \
    --optim muon
```

If Muon converges 2× faster on mod-5, that's a clean signal that the
discrete-σ optimization landscape benefits from orthogonalized updates.

## What "shallow signal" looks like

- Priority 1 (Landau): pd_ssm-N4 mod-5 stays at ~20% chance, complex_pd-N4
  hits >95%. Clean Pareto.
- Priority 2 (mod-5 long-T): pd_ssm or complex_pd hits >95% where
  deltanet_negeig collapses — validates the framework against the repo's
  known result.
- Priority 3 (MKAR): pd_ssm starts dropping around K=8-12, pd_kv stays
  high through K=32. Confirms the capacity-vs-structure tradeoff.
- Priority 4 (Muon): Muon mod-5 converges in 1500-2000 steps, AdamW in
  3000-4000 (or never). Validates the optimization-not-expressivity
  diagnosis of the MPS mod-5 stall.

If all four show the predicted directions, that's a paper-grade
empirical validation of the framework.
