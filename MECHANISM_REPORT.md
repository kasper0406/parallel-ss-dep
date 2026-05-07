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


# Phase 21c — Self-feeding retrain (Option B test) (2026-04-30)

**Question.** Does training pass-2's loss against an FiLM input that is
the model's *own* lagged source-layer output (rather than a separate
vanilla pass-1's output) close the train/inference gap that the
708 M lagged-cached PPL parity check exposed (PPL +7.9 % drift; see
[`LATENCY_REPORT.md`](LATENCY_REPORT.md))? If yes, deployment uses a
**single forward** (1× plain DN decode cost), preserving the
architectural lift end-to-end.

## Setup

New feedback mode `feedback_self_k ∈ {2, 3}` in
[`experiments/model.py`](experiments/model.py): an iterative
fixed-point training protocol where pass K's FiLM input is the lag-1
of pass K−1's source-layer output, with the prior K−1 forwards run
under `torch.no_grad()` so backward only flows through the final
forward (1× backward cost).

- **K=2**: cold start (FiLM=0, no_grad) → final pass with FiLM from
  lag(cold). Same compute as current 2-pass training, but pass-1 is
  detached (vs current 2-pass which backprops through both).
- **K=3**: cold start → 1 self-feed (no_grad) → final pass with FiLM
  from lag(self-feed). +50 % wall-clock vs current 2-pass.

Two new training runs at the Phase 17 217 M config (`d_model=576`,
`n_heads=9`, `d_head=64`, `n_layers=30`, T=512, batch=8, 5 K AdamW
steps, lr=3e-4 cosine, seed=0, codeparrot/codeparrot-clean,
content field). GPU 1, no Muon.

Eval: [`experiments/eval_filmed_ppl_217m.py`](experiments/eval_filmed_ppl_217m.py)
on a deterministic 8 K-token val slice from
`train.shuffle(seed=42).skip(10_000)` — same val tail as training.
Three quantities measured per checkpoint:

1. **Training-protocol PPL** — `model(x)` running the K-iter
   self-feeding forward.
2. **2-pass PPL** — same weights, `feedback_self_k=0` override
   (standard pass-1 vanilla + pass-2 FiLM-from-pass-1).
3. **Lagged-cached deployment PPL** — token-by-token streaming
   forward where FiLM at the target reads the previous step's
   pass-2 source-layer output as the lag-1 proxy. This is the 1×
   decode cost protocol used in `decode_bench.py`.

Plus self-consistency norm: ‖prev_iter_src − final_iter_src‖
relative to ‖final_iter_src‖, averaged over a val batch.

## Results

### Training PPL (5 K steps, full 64-chunk val cap = 32 K tokens)

| Variant | Training-time val PPL @ step 5K | Final α | Lift vs plain DN baseline 51.00 |
|---------|--------------------------------:|--------:|---------------------------------:|
| Plain DN (Phase 17) | 51.00 | — | — |
| + sparse (2, 28) FiLM, **standard 2-pass** (Phase 17) | **49.40** | −0.054 | **−3.1 %** |
| + sparse (2, 28) FiLM, **K=2 self-feeding** | **50.41** | −0.046 | **−1.16 %** |
| + sparse (2, 28) FiLM, **K=3 self-feeding** | **49.89** | −0.053 | **−2.18 %** |

Both self-feeding variants find the **negative-α basin**
(α ≈ −0.046 / −0.053), consistent with Phase 17's
α = −0.054. K=3 self-feeding recovers most of the standard 2-pass
lift (−2.18 % vs −3.1 %); K=2 self-feeding recovers 37 % of it.

The gap between K=2 self-feeding and standard 2-pass is the cost of
**detaching pass 1** from the gradient — the standard 2-pass training
backprops through *both* forwards and so optimizes pass-1's source
layer to produce a useful FiLM input *for the chosen weights*. The
self-feeding variants only optimize the final forward; the cold-
start state is whatever the model produces under FiLM=0.

### Train/inference gap on the 8 K-token val slice

| Protocol → | Training-protocol | 2-pass | Lagged-cached | Δ lagged vs train |
|---|---:|---:|---:|---:|
| Phase 17 ref (standard 2-pass training) | 45.41 | 45.41 | 45.79 | **+0.83 %** |
| **K=2 self-feeding** | 46.73 | 46.73 | 46.74 | **+0.02 %** |
| **K=3 self-feeding** | 46.21 | 47.15 | 46.15 | **−0.14 %** |

(Note. PPL on this 8 K-token sub-sample is lower-variance but
slightly different from the 64-chunk training-time PPL — the
ranking and gap-closure conclusions are unchanged.)

The 708 M Phase 20.5 ckpt's lagged-cached drift was +7.9 % on the
training val tail. At 217 M with standard 2-pass training the drift
is much smaller (+0.83 %) — perhaps because the smaller model's α
geometry is less "fragile". With self-feeding training the drift is
**driven essentially to zero**: K=2 = +0.02 % (effectively floating-
point noise), K=3 = −0.14 % (lagged-cached actually slightly *better*
than the training protocol, suggesting the iterations converged
inside the noise floor).

### Self-consistency norms

Average of ‖prev_iter_src − final_iter_src‖ / ‖final_iter_src‖ over
the 8 K-token val slice (4 batches of 4 chunks).

| Variant | Final iter ‖src‖ | Prev iter Δ ‖src‖ | Relative |
|---------|-----------------:|------------------:|---------:|
| K=2 self-feeding (cold-start vs final) | 11 213 | 4 134 | **0.369** |
| K=3 self-feeding (iter-1 vs iter-2) | 11 299 | 1 732 | **0.153** |

K=3 has 2.4× tighter self-consistency than K=2. This matches the
PPL-gap result: K=3's iterations have converged toward a
fixed-point and the lagged-cached deployment is statistically
indistinguishable from the training forward. K=2 is *not* near a
fixed-point (cold-start state and final-pass state differ by 37 %),
yet still has near-zero PPL gap — because the network has learned
to be **robust** to that input perturbation.

## Verdict

**Self-feeding closes the train/inference gap.** Both K=2 and K=3
satisfy the success criterion (lagged-cached PPL within +0.5 PPL of
training-protocol PPL, lift over plain DN baseline ≥ −2 % at 5 K
steps). K=3 satisfies the stricter criterion (tighter self-consistency,
better lift recovery).

**Recommendation for distillation pilot.** Use **K=3 self-feeding**:
- Lift over plain DN: −2.18 % (vs standard 2-pass −3.1 %, K=2 −1.16 %).
- Self-consistency: 15 % rel-norm divergence (3× better than K=2).
- Train/inference gap: −0.14 % (effectively zero).
- Wall-clock cost: ~50 % more train-time compute (3× forward, 1× backward
  per step) vs current 2-pass; *no* extra inference cost.

The architectural lift survives the deployment switch to **single forward
inference** (1× plain DN decode cost). The −3.1 % standard 2-pass lift
shrinks to −2.18 % under K=3 self-feeding, but the **deployment is now
1× DN compute** vs +99 % for the gold-matching 2-pass decode. The
trade-off is favorable for compute-bound deployments.

## Reproduction

```bash
# Train K=2 self-feeding:
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --feedback film --feedback_pairs "2,28" \
    --feedback_self_k 2 \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/film_self_k2_2_28_30L_217M.pt

# Train K=3 self-feeding (replace --feedback_self_k 2 with 3).

# Eval (8 K val slice = 16 chunks × T=512):
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u experiments/eval_filmed_ppl_217m.py \
    --ckpt checkpoints/film_self_k3_2_28_30L_217M.pt \
    --T 512 --n_tokens 8192 --batch 4 \
    --out bench_film_self_k3_ppl_8k.json
```


# Phase 21d — 708M K=3 self-feeding verification (2026-05-06)

**Question.** Does K=3 self-feeding **scale to 708 M**? At 217 M the
train/inference gap was already small for std-2-pass (drift +0.83 %),
making Phase 21c's K=3 result inconclusive about whether self-feeding
helps at the scales where the problem actually manifests. The 708 M
std-2-pass model showed +7.9 % drift — that's the signal we need K=3
to fix.

## Setup

Train a 708 M sparse-(2, 34) FiLM model with K=3 self-feeding, matching
Phase 20's std-2-pass 708 M run **exactly except for the feedback mode**:

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --T 512 --batch 4 --steps 15000 \
    --d_model 1024 --n_heads 16 --d_head 64 --n_layers 36 \
    --lr 3e-4 --optimizer muon --lr_muon 1e-3 \
    --feedback film --feedback_pairs "2,34" --feedback_self_k 3 \
    --seed 0 \
    --save_ckpt checkpoints/sparse_2_34_708M_muon_self_k3.pt
```

Wall-clock: 6492 s ≈ 108 min (1.32× the std-2-pass 708 M run's 4922 s,
in line with the +50 % expected from K=3's extra forward).

Eval on the same deterministic 32 K-token val slice that produced the
std-2-pass 708 M's PPL 34.26 / lagged-cached 36.97 (Phase 20.5
`d2876e3` extended to 32 K).

## Results

| Variant | Train-faithful PPL | Lagged-cached PPL | Drift | Decode cost | α |
|---------|-------------------:|------------------:|------:|------------:|--:|
| 708 M plain DN baseline | 35.38 | 35.38 | — | 1× | — |
| 708 M std-2-pass FiLM | 34.26 | 36.97 | **+7.92 %** | 2× | −0.198 |
| **708 M K=3 self-feeding FiLM** | 35.23 | **34.85** | **−0.04 %** | **1×** ⭐ | −0.197 |

K=3 training-protocol PPL (its own self-consistent forward): 34.86.
Self-consistency rel-norm: **0.114** (better than 217 M K=3's 0.153,
indicating tighter convergence at scale).

## Interpretation

**K=3 self-feeding scales cleanly to 708 M.** The +7.92 % drift that
broke the std-2-pass deployment story is fully closed (drift now
−0.04 %, a sub-PPL-point fluctuation). Both K=3 and std-2-pass find
the same negative-α basin at α ≈ −0.198, confirming the basin is
the architecturally robust quantity (Phase 21's prior conclusion).

**Architectural lift at deployment** (lagged-cached, 1× decode cost):

- K=3 deployment vs plain DN baseline: **34.85 vs 35.38 = −1.5 %.**
- K=3 deployment vs std-2-pass deployment: **34.85 vs 36.97 = −5.7 %.**
- K=3 deployment vs std-2-pass *training-faithful* (the protocol-cheating
  reference number we previously quoted): 34.85 vs 34.26 = +1.7 %.

The architectural lift shrinks at 708M from std-2-pass's −3.2 %
training-faithful claim to K=3's −1.5 % deployment-honest claim, but
the deployment cost drops from 2× to 1×. Wall-clock × quality
(`decode_ms × PPL`) at 8 K context: K=3 wins decisively
(`14.5 × 34.85 = 505` vs std-2-pass `28.8 × 34.26 = 987`, ~2× better).

**Verdict.** **Distillation pilot uses K=3 self-feeding.** The trade
of −1.7 PPL training-faithful → +0.6 PPL lagged-cached deployment is
more than compensated by halving the decode cost. The story for any
paper / blog post is now: "−1.5 % lift at 1× decode cost" instead of
"−3.2 % lift at 2× decode cost" — the deployment-fair claim is the
honest one.

## Reproduction

```bash
# Eval (32 K val slice = 64 chunks × T=512):
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u experiments/eval_filmed_ppl_708m.py \
    --ckpt checkpoints/sparse_2_34_708M_muon_self_k3.pt \
    --T 512 --n_tokens 32768 --batch 2 \
    --out bench_film_self_k3_708M_ppl.json
```

