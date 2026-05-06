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

1. ~~**H2 test:** add a scalar forget gate to plain DN and re-run the
   Phase 17 setup.~~ → **Done in Phase 21b below — H2 also rejected.**
2. **H4 noise floor (~30 min):** repeat Mamba2 + sparse-FiLM at a
   second seed. If the second seed gives a similar near-zero lift,
   the Mamba2 result is real; otherwise it was noise.
3. **Re-frame the cross-cell narrative.** The clean architectural
   claim is "−3 % lift in plain DeltaNet, robust across 4.5× state-
   size variation, robust across both basin signs, robust to 3.3× scale-
   up, robust to adding a forget gate." The diminishing lift in
   stronger cells is a *cell-level* compatibility effect with the
   specific structured-SSM aggregation form (most likely H3), not a
   state-capacity (H1) or forget-gate-redundancy (H2) effect.

# Phase 21b — H2 (forget-gate redundancy) test (2026-04-30)

**Question:** does adding a learnable forget gate to plain DeltaNet
reduce the sparse-FiLM lift toward GatedDeltaProduct's −1.9 % range
(vs plain DN's −3.1 %)? Phase 17 showed GDP benefits less from FiLM,
and a natural hypothesis is that GDP's forget gate already provides
a content-dependent gating mechanism that overlaps with what FiLM
contributes.

## Setup

Add a forget-gate-only variant of plain DN — instantiated as fla's
`GatedDeltaNet` with `use_gate=False` (output gate disabled,
forget-gate-only). New layer wrapper `DeltaNetForgetGateAttention`
in `experiments/layers.py`. Same training config as Phase 17:
codeparrot Python, T=512, batch=8, 5 K AdamW steps, lr=3e-4 cosine,
seed=0, no warmup. `d_model=576, n_heads=9, d_head=64, n_layers=30`.

Two new runs: forget-gated DN baseline + forget-gated DN + sparse-(2, 28)
FiLM. Plain DN reference is reused from Phase 17.

## Results

| Cell | Baseline | + sparse (2, 28) FiLM | Δ vs baseline | Final α | Basin |
|------|---------:|----------------------:|--------------:|--------:|-------|
| Plain DN (Phase 17) | 51.00 | 49.40 | −3.1 % | −0.054 | NEG |
| **DN + forget gate** | **46.36** | **44.86** | **−3.23 %** | **−0.044** | **NEG** |

(For reference: GatedDeltaProduct in Phase 17 had lift −1.9 %.)

## Interpretation

**H2 is rejected.**

Adding a forget gate to plain DN substantially improves the baseline
(51.00 → 46.36, **−9 % PPL**) — confirming the forget gate is a
genuinely useful mechanism in its own right. But the sparse-FiLM
lift on top of forget-gated DN is **−3.23 %**, virtually identical
to plain DN's −3.1 %. So the FiLM mechanism is *not* redundant with
the forget-gate mechanism — they're complementary contributions.

The basin sign is preserved (still NEG, |α| ≈ 0.044, consistent with
Phase 21's d_head=32/64 and Phase 17's plain DN at α=−0.054). The
mechanism story (multiplicative form + non-softmax aggregation finds
this basin) is unchanged by the forget gate.

**What this leaves as the explanation for Phase 17's cross-cell
diminishing lift.** With H1 (state capacity) and H2 (forget-gate
redundancy) both rejected, the most likely remaining hypothesis is
**H3 — SSM aggregation form**. Mamba2's structured SSM kernel
aggregates per-token information through a fundamentally different
update geometry (continuous-time discretization with channel-wise
diagonal SSM) than DeltaNet's outer-product Householder-style
update. The basin we identified — multiplicative form +
non-softmax aggregation, gradient direction `x · scale` — may
simply not be reachable through Mamba2's update geometry, regardless
of whether a forget gate is present.

GatedDeltaProduct sits in the middle (still uses outer-product
Householder updates like DN, but with multiple Householder products
per step + the forget gate). Its lift of −1.9 % is between DN's
−3.1 % and Mamba2's −0.27 %, which is consistent with a partial
geometric incompatibility rather than a clean redundancy effect.

## Updated cross-cell story

| Cell type | Update geometry | Forget gate | FiLM lift |
|-----------|-----------------|-------------|-----------|
| Plain DN  | Outer-product, rank-1 erase | None | **−3.1 %** |
| DN + forget gate | Outer-product, rank-1 erase | Scalar | **−3.2 %** |
| GatedDeltaProduct | Outer-product, K Householder products | Scalar | −1.9 % |
| Mamba2 | Structured SSM, diagonal | Per-channel | −0.27 % |

The FiLM lift correlates with **how close the cell's update is to
a plain rank-1 outer-product**, not with the presence of a forget
gate. Future blog/paper should frame this as: "sparse-FiLM is a
mechanism specifically for outer-product-style linear-RNN cells; it
does not transfer cleanly to structured-SSM aggregation."

## Suggested follow-up

1. **H4 noise floor (~30 min):** repeat Mamba2 + sparse-FiLM at a
   second seed to bound whether the −0.27 % is real.
2. **Test on GatedDeltaProduct *without* multiple Householders**
   (`num_householder=1`, equivalent to plain DN with forget gate +
   `allow_neg_eigval=True`). If the lift returns to ~−3 %, H3 is
   strongly supported: the multiple-Householder-product structure
   is what disrupts the basin's reachability, not the forget gate.

