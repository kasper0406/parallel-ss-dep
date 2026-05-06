# Mechanism report — sparse-FiLM lift vs cell state capacity

**Date:** 2026-04-30
**Phase:** 21
**Question:** Does the diminishing sparse-FiLM lift across stronger linear-RNN
cells (Phase 17: DN −3.1 %, GDP −1.9 %, Mamba2 −0.27 %) reflect a
**state-capacity** mechanism — i.e. cells with larger per-token state
already carry the cross-layer context FiLM would otherwise inject?

## Setup

Vary plain DeltaNet's `d_head` (and matching `n_heads = d_model / d_head`)
to control per-layer recurrent state size, holding `d_model = 576` and
`n_layers = 30` fixed. Per-layer state = `n_heads × d_head²` (the
WY-repr hidden state in DeltaNet's chunked algorithm).

| Variant | `n_heads` | `d_head` | Per-layer state | State ratio vs Phase 17 |
|---------|-----------|----------|-----------------|--------------------------|
| Small state | 18 | 32 | 18 432 | 0.50× |
| Phase 17 baseline | 9 | 64 | 36 864 | 1.00× |
| Large state | 4 | 144 | 82 944 | 2.25× |

**Note on the d_head=144 deviation from the user's spec.** The original
brief specified `n_heads=4, d_head=128` for the "Large state" cell, but
that breaks `d_model = n_heads · d_head` (4 × 128 = 512 ≠ 576). FLA's
DeltaNet enforces this constraint. Closest valid option holding
`d_model = 576`: `n_heads = 4, d_head = 144`. This gives an even larger
state-size spread (2.25× vs the brief's intended 1.78×), which actually
*strengthens* the H1 test.

**Training config (matches Phase 17 exactly).** codeparrot/codeparrot-clean
Python, T=512, batch=8, 5 K AdamW steps, lr=3e-4 cosine, seed=0,
no warmup. Each `d_head` × `{baseline, +sparse-(2, 28)-FiLM}` pair = 4
new runs. Phase 17's `d_head=64` results (PPL 51.00 / 49.40, α=−0.054)
are reused from RESULTS.md.

## Results

PPL = final validation PPL at step 5 000.

| `d_head` | State | DN baseline PPL | + sparse (2, 28) FiLM | Δ vs baseline | Final α | Basin sign |
|---------:|------:|----------------:|----------------------:|--------------:|--------:|-----------:|
|  32 | 18 432 | 55.72 | 54.37 | **−2.4 %** | −0.056 | NEG (subtractive) |
|  64 | 36 864 | 51.00 | 49.40 | **−3.1 %** | −0.054 | NEG (subtractive) |
| 144 | 82 944 | 46.73 | 45.25 | **−3.2 %** | **+0.053** | **POS (additive)** |

(Phase 17 row is reproduced for reference.)

## Interpretation

**H1 (state-capacity) is rejected by this experiment.**

If the diminishing FiLM lift across stronger cells (Phase 17: DN −3.1 %,
GDP −1.9 %, Mamba2 −0.27 %) were due to per-token state size — "more
state means less to inject" — the lift should have *decreased*
monotonically with `d_head`. It did not. Instead:

- Going from `d_head=32` → `d_head=64` the lift *grew* (−2.4 % → −3.1 %).
- Going from `d_head=64` → `d_head=144` the lift was essentially
  unchanged (−3.1 % → −3.2 %).

So inside the *same* DeltaNet cell, varying state size by 4.5× changes
the FiLM lift by less than 1 PPL point absolute. State size alone is
not what gates whether sparse-FiLM helps — the cross-cell pattern
(DN > GDP > Mamba2) must be driven by some *other* property of the
cell (most likely: forget-gate redundancy, the SSM aggregation form,
or the basin's compatibility with the cell's gradient geometry).

**Surprising secondary finding: the basin sign flipped at the
larger-state cell.**

`d_head=32` and `d_head=64` both find the negative-α subtractive
("predictive coding") basin. `d_head=144` lands in the positive-α
additive basin instead, but with *the same |α| ≈ 0.054* and
*essentially the same architectural lift* (−3.2 % vs −3.1 %). This
reinforces the Phase 14b–g mechanism story: the architectural lift
exists in *both* polarities, and which polarity gets found depends on
the joint configuration of state size, optimizer, and init geometry.
The magnitude of |α| is the architecturally robust quantity, not
its sign — a recurring observation across this project (217 M /
AdamW = NEG, 360 M / Muon = POS, 708 M / Muon = NEG).

**What the data says about the cross-cell story:**

The Phase 17 finding that the lift diminishes from DN → GDP → Mamba2
is *not* explained by state size. The remaining hypotheses (in
descending plausibility based on this and prior data):

- **H2 (forget-gate redundancy).** GatedDeltaProduct and Mamba2 both
  have learnable forget gates that selectively retain or erase
  per-token state. FiLM's contribution overlaps that mechanism: it's
  also a learnable, content-dependent gate on per-token input.
  This is the natural next test — DN with an added forget gate, see
  if the FiLM lift drops.
- **H3 (SSM aggregation form).** Mamba2 aggregates with a structured
  SSM kernel; its aggregation has different gradient geometry than
  DN's outer-product update. The basin we identified (multiplicative
  + non-softmax) may simply not be reachable through Mamba2's update.
- **H4 (training noise dominating in Mamba2).** Mamba2 trained at
  ~4.5 K tok/s vs DN's ~16 K tok/s in Phase 17 — single-seed,
  potentially noisy. The −0.27 % could be within noise of zero.

## Suggested follow-up

1. **H2 test (cheapest, ~30 min):** add a scalar forget gate to plain
   DN and re-run the Phase 17 setup. If FiLM lift drops to near GDP's
   −1.9 % range, H2 is supported. If lift stays near DN's −3.1 %, H2
   is rejected and the residual cross-cell variance must be H3 or H4.
2. **H4 noise floor (~30 min):** repeat Mamba2 + sparse-FiLM at a
   second seed. If the second seed gives a similar near-zero lift,
   the Mamba2 result is real; otherwise it was noise.
3. **Re-frame the cross-cell narrative.** The clean architectural
   claim is "−3 % lift in plain DeltaNet, robust across 4.5× state-
   size variation, robust across both basin signs, robust to 3.3× scale-
   up." The diminishing lift in stronger cells is a *cell-level*
   compatibility effect, not a *state-capacity* effect. The blog/paper
   should be honest that this is an open question with H2-H3-H4 on the
   shortlist.

