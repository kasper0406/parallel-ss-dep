# Structural Surprise PoC — surprise-modulated α at K=3 self-feeding

**Branch:** `structural-surprise-loss`
**Status:** v0 PoC complete — **verdict: no signal on code**.

This is the v0 minimal PoC proposed in `STRUCTURAL_SURPRISE_DESIGN.md`:
replace the scalar α in the Phase 21c sparse-(2,28) FiLM K=3 self-feeding
217 M baseline with a per-token `α(t) = α₀ · σ(scale · s_z(t) + bias)`,
where `s_z(t)` is the per-batch z-scored inter-iter delta of the source
layer's output between the K=3 self-feeding model's two no-grad iterations
(iter 1 = cold, iter 2 = first self-feed). All other training settings
match Phase 21c exactly.

## TL;DR

| Variant | Training-protocol PPL | 2-pass PPL | Lagged-cached PPL |
|---|---:|---:|---:|
| Phase 21c K=3 baseline (scalar α) | 46.21 | 47.15 | 46.15 |
| **K=3 + surprise-modulated α (this PoC)** | **46.25** | **46.25** | **46.19** |
| Δ vs Phase 21c | +0.04 | n/a* | +0.04 |

*The 2-pass eval column for surprise_modulated runs the same K=3 forward
as training-protocol, because the surprise signal requires K=3's two
no-grad iterations. The Phase 21c K=3 2-pass column ran a flat-α 2-pass
forward (since flat-α has no per-iter structure to exploit). They are
not directly comparable in 2-pass form. The training-protocol and
lagged-cached numbers are the load-bearing comparisons.

The α(t) distribution is **non-degenerate** (clearly spread, see histograms
below), so the modulation IS doing something — it just doesn't translate
into a PPL improvement.

**Verdict: no signal.** Lagged-cached PPL = 46.19 ≥ Phase 21c's 46.15 by
+0.04 PPL absolute (–0.09 % relative). The success threshold (lagged-cached
≤ 45.5, i.e. ≥ 0.5 PPL absolute below baseline) is far from met. The
mild-signal threshold (lagged-cached ≤ 46.0) is also not met. This direction
**does not warrant the statement-boundary follow-up on code**.

## Implementation summary

- **Code touchpoints:**
  - `experiments/model.py` — `FeedbackProjection` gains `alpha_mode`
    parameter; `surprise_modulated` introduces three learnable scalars
    (`alpha_zero`, `surprise_scale`, `surprise_bias`) per FiLM target. The
    K=3 self-feeding loop in `TinyLM.forward()` tracks two no-grad
    iterations' source states and computes a per-token L2-norm delta as
    the surprise signal, fed to `FeedbackProjection.forward(...,
    surprise=)` for the loss-bearing pass. The `_last_surprise_per_target`
    and `_last_alpha_t_per_target` attributes are cached for diagnostics.
  - `experiments/train_lm.py` — adds `--feedback_alpha_mode {scalar,
    surprise_modulated}` CLI flag, persists it to the checkpoint config.
  - `experiments/eval_filmed_ppl_217m.py` — extends `load_film_217m`
    to read the alpha mode; in surprise_modulated mode the 2-pass eval
    keeps K=3 (because the surprise signal requires it). Adds
    `collect_surprise_alpha_distribution` which emits per-target
    surprise(t) / α(t) summary stats and ASCII histograms.

- **Surprise definition (this PoC):**
  - K=3 has three iterations: iter 1 (cold, no_grad), iter 2 (one
    self-feed, no_grad), iter 3 (loss-bearing, with-grad). At the time the
    FiLM at layer 2 needs to apply α(t), neither iter 3's source state nor
    a fresh "iter K vs iter K-1" delta is available without a fourth
    forward pass. We use the "free signal" interpretation from the design
    doc: `surprise(t) = ‖iter_2.src(t) − iter_1.src(t)‖₂`, both no-grad
    and detached. This is a measurement of how much the model's source
    representation shifted across one round of self-feeding, available
    before the loss-bearing pass starts. Per-batch z-score normalisation
    (over the whole (B, T) tensor, with `std.clamp(min=1e-6)`) before σ.

- **At lagged-cached deploy time** the surprise tensor is unavailable
  (single forward pass). The `FeedbackProjection.forward` falls back to
  `α₀ · σ(bias)`, which gives a uniform α across all tokens — equivalent
  to a flat-α form with magnitude `α₀ · σ(bias)`. After training:
  `α₀ ≈ 0.083`, `bias ≈ 0` (small), so the deploy-time effective α is
  `≈ 0.083 · 0.5 = 0.041`, smaller than the per-token 2-pass mean (0.046).
  This is the deployment-without-surprise mode and partially explains the
  ~0.07 PPL gap between training-protocol (46.25) and lagged-cached
  (46.19) — the lagged-cached numerically beats training-protocol because
  the smaller effective α at deploy reduces FiLM-driven drift. Not a true
  improvement, just a quirk of the σ-gate.

## Distributions over the 8 K val slice

### surprise(t) — heavy-tailed, right-skewed (good)

```
surprise(t) histogram (n=8192, min=0.00, max=317.12):
   0.0000 –  15.8562 |     17 
  15.8562 –  31.7124 |     19 
  31.7124 –  47.5687 |     84 █
  47.5687 –  63.4249 |    398 ████████
  63.4249 –  79.2811 |   1204 ████████████████████████
  79.2811 –  95.1373 |   1968 ████████████████████████████████████████
  95.1373 – 110.9936 |   1745 ███████████████████████████████████
 110.9936 – 126.8498 |   1178 ███████████████████████
 126.8498 – 142.7060 |    732 ██████████████
 142.7060 – 158.5622 |    384 ███████
 158.5622 – 174.4185 |    222 ████
 174.4185 – 190.2747 |    118 ██
 190.2747 – 206.1309 |     55 █
 206.1309 – 221.9871 |     33 
 221.9871 – 237.8434 |     13 
 237.8434 – 253.6996 |      7 
 253.6996 – 269.5558 |      9 
 269.5558 – 285.4120 |      0 
 285.4120 – 301.2683 |      2 
 301.2683 – 317.1245 |      4 
```
Stats: mean 103.13, std 31.94, p5/p50/p95 = 60.6 / 98.3 / 160.9. Heavy
right tail as expected — there are tokens with much-larger-than-typical
inter-iter source-state shifts, consistent with a meaningful "surprise"
signal in the data (not flat / not bimodal). This is the desired
sanity-check; the signal IS structural.

### α(t) — clearly spread, bell-shaped (non-degenerate)

```
α(t) histogram (n=8192, min=0.018, max=0.079):
   0.0180 –   0.0210 |      8 
   0.0210 –   0.0240 |      9 
   0.0240 –   0.0271 |      9 
   0.0271 –   0.0301 |     33 
   0.0301 –   0.0331 |    100 ██
   0.0331 –   0.0361 |    371 █████████
   0.0361 –   0.0392 |    927 ████████████████████████
   0.0392 –   0.0422 |   1353 ███████████████████████████████████
   0.0422 –   0.0452 |   1511 ████████████████████████████████████████
   0.0452 –   0.0483 |   1212 ████████████████████████████████
   0.0483 –   0.0513 |    917 ████████████████████████
   0.0513 –   0.0543 |    649 █████████████████
   0.0543 –   0.0574 |    418 ███████████
   0.0574 –   0.0604 |    306 ████████
   0.0604 –   0.0634 |    170 ████
   0.0634 –   0.0664 |    101 ██
   0.0664 –   0.0695 |     58 █
   0.0695 –   0.0725 |     22 
   0.0725 –   0.0755 |     13 
   0.0755 –   0.0786 |      5 
```
Stats: mean 0.046, std 0.0076, p5/p50/p95 = 0.035 / 0.045 / 0.060. Range
~4.4× from p5 to p95. The modulation produces a clear per-token gradient
in coupling strength — high-surprise tokens get α ~0.06, low-surprise
tokens get α ~0.035. **Not degenerate.** So the conclusion isn't "the
modulation didn't take effect"; it's "the modulation took effect and
didn't help".

### Learned modulator parameters

After 5 K steps:
- `α₀ ≈ +0.083` (positive-α basin; opposite sign from the Phase 21c
  scalar baseline's −0.054 — see "α basin observation" below).
- `surprise_scale ≈ 1.0` (essentially unchanged from init).
- `surprise_bias ≈ 0` (essentially unchanged from init).

The modulator parameters didn't move much from init, but α₀ did grow
substantially. Most of the per-token spread in α(t) comes from the
sigmoid acting on the natural variance of `surprise_z(t)`, not from
the modulator finding a non-trivial scale/bias.

## α basin observation

The Phase 21c scalar-α K=3 baseline converges to α ≈ **−0.054**
(predictive-coding negative-α basin). The surprise-modulated variant
converges to `α₀ ≈ +0.083` (positive-α basin). This is a real
divergence in the learned coupling direction.

Hypothesis: the σ-gate breaks the symmetry that makes negative-α
attractive in scalar mode. With α(t) = α₀·σ(...), the per-token coupling
is always same-signed as α₀ but its magnitude is data-dependent. The
positive-α basin found here corresponds to "amplify the signal where
surprise is high" rather than the predictive-coding "subtract the
prediction" behaviour of the scalar baseline. Both are local minima of
the loss; the K=3 surprise-modulated training found the positive one.

This is consistent with the no-signal verdict: the model found a
different basin, optimised it, and ended up matched-but-not-better
than the predictive-coding scalar baseline. The structural-surprise
signal didn't unlock a basin that beats both.

## Self-consistency

Final iter-3 source state vs prev iter-2 source state, relative norm:
**0.168** (mean across 4 batches). For comparison, Phase 21c K=3 had
0.153. Slightly worse self-consistency in surprise-modulated mode —
not a problem on its own, but consistent with the model finding a
less-converged fixed-point.

## Verdict

**No signal.** Lagged-cached PPL = 46.19 vs Phase 21c K=3's 46.15. We
trained for 5 K steps with the same compute as Phase 21c (~18 min on a
single 5090). The α(t) distribution is healthy (non-degenerate, spread
over a 4× range, sigmoid-saturated tails) and the surprise signal is
heavy-tailed (the desired structural-pivot indicator), but the modulation
does not translate into PPL gains on Python code.

**No follow-up recommended on code:**
- The statement-boundary variant (broadcast surprise across Python
  statements via `ast.parse`) was conditional on the per-token variant
  showing ≥ 0.5 PPL absolute lift. It didn't, so statement-boundary won't
  save it (would have to come back from −0.04 PPL to ≥ +0.5 PPL — that's
  ≥ 0.54 PPL of new lift the statement granularity has to provide).
- The Phase 18 prior was already unfavourable (Phase 18 surprise gating
  on code contributed ~0 % on PPL). This PoC reproduces that finding for
  a different surprise signal (inter-iter delta vs CE-vs-running-mean) at
  a different injection point (FiLM α modulation vs scratchpad attention
  bias). Two negative results on different surprise definitions on code
  — fairly strong evidence that "code doesn't have the right kind of
  per-token surprise to lift sparse-FiLM via scalar α modulation".

## What this rules out, what's still open

**Ruled out (or strongly disfavoured):**
- Per-token α modulation by inter-iter delta surprise on code → no PPL
  lift over Phase 21c K=3 baseline.
- Statement-boundary variant on code (because v0 must show signal first).

**Still open (would need a new branch, separate dispatch):**
- The proposal's *original* domain (dialogue / DailyDialog) — surprise on
  code has now twice failed to lift PPL, but dialogue's per-sentence
  intent shifts may be a fundamentally different signal density.
- The full Phase 1-3 pipeline (oracle + L_sem + meta-cognitive head),
  which adds external supervision (Sentence-BERT space) the inter-iter
  delta proxy doesn't have.
- A different *injection point* on code: e.g. surprise-weighted CE loss
  instead of α modulation. Loss reweighting was ruled in scope by the
  proposal's Phase 2; we did not test it here.

The minimum design doc recommendation — "front-load the cheap signal
checks, back-load the infrastructure spend" — has been served: this
1-day PoC says "don't invest in the multi-week infrastructure rebuild
for code". That's a clean go/no-go.

## Reproduction

```bash
# Train (~18 min on 5090, GPU 1)
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --feedback film --feedback_pairs "2,28" \
    --feedback_self_k 3 \
    --feedback_alpha_mode surprise_modulated \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/film_self_k3_surprise_2_28_30L_217M.pt

# Eval on 8 K val slice (~2 min)
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u experiments/eval_filmed_ppl_217m.py \
    --ckpt checkpoints/film_self_k3_surprise_2_28_30L_217M.pt \
    --T 512 --n_tokens 8192 --batch 4 \
    --out bench_film_self_k3_surprise_ppl_8k.json
```

## Artifacts

- Checkpoint: `checkpoints/film_self_k3_surprise_2_28_30L_217M.pt`
- Training log: `runs/film_self_k3_surprise.log`
- Eval log: `runs/eval_film_self_k3_surprise.log`
- Eval JSON (with full histograms): `bench_film_self_k3_surprise_ppl_8k.json`
- Reference (Phase 21c K=3 baseline): `bench_film_self_k3_ppl_8k.json`
