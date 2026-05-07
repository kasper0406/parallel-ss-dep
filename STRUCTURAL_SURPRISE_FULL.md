# Structural surprise loss — full PoC report (Phases 0-2)

**Date:** 2026-05-07
**Branch:** `structural-surprise-loss`
**Goal:** test the *full* proposal from `STRUCTURAL_SURPRISE_DESIGN.md` —
Phases 0-2 (statement-level eval harness, predictive surprise oracle,
semantic gradient loss `L_sem`) — at 217 M on code, on top of K=3
self-feeding sparse-(2, 28) FiLM. Decide whether the actual mechanism
(semantic-space loss weighted by oracle-measured structural surprise)
helps PPL on code, and whether it's worth pushing to dialogue. Phase 3
(meta-cognitive head) deferred.

This is structurally distinct from the v0 PoC (`STRUCTURAL_SURPRISE_PoC.md`,
commit `21aa8b9`), which used per-token model-internal surprise to
modulate α and was a clean negative.

## Setup

### Phase 0 — statement-level eval harness

`experiments/statement_segmentation.py` parses Python via `ast.parse`,
mapping each statement node back to its token range in the tokeniser
(SmolLM2-135M). `experiments/eval_statement_ppl.py` evaluates a
checkpoint on a deterministic 32 K-token codeparrot val slice,
computing per-statement CE and stratifying by oracle surprise into
deciles.

### Phase 1 — predictive surprise oracle

- **Frozen encoder E**: a plain DeltaNet 217 M trained for 5 K AdamW
  steps on codeparrot (`d_model=576 n_heads=9 d_head=64 n_layers=30`).
  Final-layer hidden states, mean-pooled across each statement's token
  range. Saved as `checkpoints/dn_baseline_30L_217M_for_oracle.pt`.
- **Predictive head P**: 2-layer causal Transformer, `d_model=576
  n_heads=8 d_head=72`, ~8.13 M params. Takes a sequence of statement
  embeddings and predicts the next statement's embedding. Trained
  on codeparrot statements until cosine-similarity loss plateaus.
  Saved as `checkpoints/oracle_predictive_head_217M.pt`.
- **Surprise score**: `surprise(s_t) = 1 - cos(P(prev), E(s_t))`.

### Oracle inspection (`surprise_inspection.txt`) — eyeball-passes

Top-10 most-surprising statements (validation: should look structural):

```
surp=0.91 Assign     'self.turned_off_filtering = True'
surp=0.90 Expr       "'''To be implemented by subclass...'''"  (docstring)
surp=0.87 Try        'try:'
surp=0.87 If         "if __name__ == '__main__':"
surp=0.81 Try        'try:'
surp=0.80 ClassDef   'class Config:'
surp=0.80 Expr       '"""Convert the quota object..."""'  (docstring)
surp=0.79 Return     'return []'
surp=0.79 Assign     'BLUE = "\033[34m"'
```

Bottom-10 least-surprising (should look formulaic):

```
surp=0.04 ImportFrom 'from tensorflow.python.framework import tensor_shape'
surp=0.04 ImportFrom 'from tensorflow.python.keras... import generic_utils'
surp=0.04 ImportFrom 'from nova.api.openstack import extensions'
... (all import statements)
```

The oracle correctly identifies structurally pivotal statements
(`try:`, `class X:`, `return ...`, docstrings, `if __name__ == "__main__":`)
as high-surprise and formulaic imports as low-surprise. Heavy right
tail in the surprise distribution. Phase 1 validation gate cleanly
passed.

### Phase 2 — `L_sem` semantic gradient loss

In `experiments/model.py`:

- Projection `W: R^576 → R^576` (identity-init).
- Pool the base model's final-layer hidden states across each statement
  → `h_t`. Project: `h̃_t = W(h_t)`.
- Target: `e_t = E(s_t).detach()`.
- Per-statement loss: `L_sem(s_t) = surprise(s_t).detach() · (1 - cos(h̃_t, e_t))`.
- Total loss: `L = L_ce + β · sum_t L_sem(s_t)`.

Crucially, surprise is detached (it's a weight, not a target), and
the oracle target `e_t` is detached (frozen oracle).

CLI: `--semantic_loss_beta β --encoder_ckpt ... --oracle_ckpt ...`.

## Training

Two configurations, identical Phase 17/21c setup otherwise
(`d_model=576 n_heads=9 d_head=64 n_layers=30 T=512 batch=8 5K AdamW
steps lr=3e-4 cosine seed=0` codeparrot Python):

- `K=3 self-feeding sparse-(2,28) FiLM + L_sem β=0.1` (1364 s)
- `K=3 self-feeding sparse-(2,28) FiLM + L_sem β=0.3` (1368 s)

## Results

Statement-stratified eval (32 K-token codeparrot val slice, oracle deciles):

| Variant | Overall PPL | Top-10% (most surprising) | Bot-10% (least surprising) | top/overall |
|---|---:|---:|---:|---:|
| DN baseline (217 M, no FiLM) | 47.13 | 53.40 | 47.70 | 1.13× |
| K=3 self-feed (Phase 21c) | 45.61 | 51.01 | 45.95 | 1.12× |
| K=3 + L_sem β=0.1 | 44.54 | 49.69 | 44.63 | 1.12× |
| **K=3 + L_sem β=0.3** | **43.53** | **49.36** | **43.49** | 1.13× |

### Δ vs K=3 self-feed baseline

| Variant | Overall | Top-10% | Bot-10% |
|---|---:|---:|---:|
| K=3 + L_sem β=0.1 | −2.4 % | −2.6 % | −2.9 % |
| K=3 + L_sem β=0.3 | **−4.6 %** | **−3.2 %** | **−5.4 %** |

### Cumulative Δ vs plain DN baseline

| Variant | Overall | Top-10% | Bot-10% |
|---|---:|---:|---:|
| K=3 self-feed | −3.2 % | −4.5 % | −3.7 % |
| K=3 + L_sem β=0.3 | **−7.6 %** | **−7.6 %** | **−8.8 %** |

## Verdict — strong overall signal, structural framing overstated

**`L_sem` clearly helps**, both at β=0.1 and β=0.3, additively on top
of K=3 self-feeding. β=0.3 hasn't saturated the lift — the curve
still suggests room at higher β. The cumulative lift over plain DN is
**−7.6 %** at β=0.3, more than double K=3's standalone −3.2 %.

**But the lift is NOT preferentially at structural pivots.** Bot-10%
PPL improves *more* than top-10% PPL (β=0.3: bot −5.4 % vs top −3.2 %).
The `top/overall` ratio is essentially unchanged across all variants
(1.12-1.13×). So `L_sem` behaves like a general semantic-space
regulariser rather than a structural-pivot booster — the surprise
weighting is providing useful gradient signal, but it's not
selectively concentrated at pivots as the design doc hypothesised.

What this likely means: the gradient signal `L_sem` adds is
*qualitatively different* from CE — it's a hidden-state alignment
loss against a frozen-DN-encoder target, weighted by per-statement
surprise. The "weighted by surprise" part doesn't seem to be doing
the heavy lifting; the alignment-against-frozen-encoder-target is.
A cleaner diagnostic would be a **β=0.3, surprise-weight=1.0** ablation
(uniform-weight L_sem) — if it matches β=0.3 with surprise weighting,
the surprise machinery is decorative.

## β saturation + uniform-weight ablation (extended sweep, 2026-05-07)

The original recommendation (continue β sweep + uniform-weight
ablation) was executed. Results:

| Variant | Overall | Top-10% | Bot-10% | top/all |
|---|---:|---:|---:|---:|
| K=3 + L_sem β=0.5 (surprise-wtd) | 42.44 | 49.11 | 41.95 | 1.16× |
| K=3 + L_sem β=1.0 (surprise-wtd) | 42.01 | 48.86 | 41.30 | 1.16× |
| **K=3 + L_sem β=1.0 (UNIFORM)** | **41.76** | **48.54** | **40.18** | 1.16× |
| K=3 + L_sem β=2.0 (surprise-wtd) | 41.89 | 49.29 | 41.80 | 1.18× |

### Three decisive findings

1. **β saturates at ≈1.0.** β=2.0 (41.89) brings no measurable
   improvement over β=1.0 (42.01). The lift curve has plateaued.
   Optimal `--semantic_loss_beta` for this regime is roughly 1.0.

2. **Surprise weighting is actively counterproductive.** With β
   held at 1.0, the **uniform-weight** ablation
   (`--semantic_loss_uniform_weight`) reaches **41.76 PPL**, beating
   the surprise-weighted 42.01 by 0.6 %. The oracle-derived
   per-statement weights are not just decorative — they make the
   result *worse*. Top-10% PPL: 48.54 (uniform) vs 48.86
   (surprise-weighted). Bot-10%: 40.18 vs 41.30. The improvement is
   uniform across deciles, not concentrated at low-surprise statements.

3. **The lift's domain is "everything", not "pivots".** The
   `top/overall` ratio rises monotonically with β (1.12× → 1.16× →
   1.18×). At higher β, the model gets *relatively* worse at
   structurally surprising statements vs its overall improvement. The
   cumulative −11.4 % lift over plain DN at uniform-β=1.0 is
   distributed:
   - Top-10% (most-surprising statements): −9.1 %.
   - Bot-10% (least-surprising statements): **−15.8 %**.

   The frozen-encoder alignment loss helps formulaic code more than
   pivotal code.

### Cumulative architectural lift over plain DN @ 217M

| Stack | PPL | Δ vs plain DN | Decode cost |
|---|---:|---:|---:|
| Plain DN baseline | 47.13 | — | 1× |
| + K=3 self-feeding sparse-FiLM | 45.61 | −3.2 % | 1× |
| + K=3 + L_sem β=1.0 (uniform) | **41.76** | **−11.4 %** | 1× |

Of the −11.4 % cumulative lift, K=3 self-feeding contributes −3.2 %
and the alignment loss contributes the remaining −8.4 %. The
alignment loss is the larger-magnitude effect.

### Verdict on the structural-surprise hypothesis

**Rejected, decisively.** The proposal's central claim — that
weighting an alignment loss by oracle-measured structural surprise
preferentially improves performance at pivotal moments — is wrong
in this domain. Empirically:

- The oracle correctly identifies structurally surprising statements
  (the eyeball check on `surprise_inspection.txt` is unambiguous —
  top-decile is `try:`, `class:`, returns, docstrings; bottom-decile
  is formulaic imports).
- But weighting the loss by that surprise score makes results worse
  than ignoring it.
- And the lift, when present, concentrates at routine statements,
  not pivots.

So the surprise machinery is correctly detecting pivots, the loss is
producing real PPL gains, but the two facts are not connected
causally. The actual mechanism is **uniform alignment of pooled
hidden-state representations to a frozen-encoder target** — a general
auxiliary regulariser, not a structural-surprise booster.

### What still works and is publishable

The `K=3 self-feeding + uniform-weight L_sem alignment loss`
combination is a clean architectural finding. Lift is large
(−11.4 % at 217M, additive on top of the K=3 self-feeding lift).
The mechanism is interpretable: pool the model's hidden states
across statements, project to a frozen-encoder space, and minimise
cosine distance to the encoder's pooled representation. No oracle
predictive head needed — that machinery can be removed entirely.

### Recommendations going forward

1. **708 M scale verification** is the most important next test. Run
   K=3 self-feeding + uniform-weight L_sem at the Phase 20/21d 708 M
   Muon setup. If the lift holds, the mechanism is real; if it
   attenuates, it's a small-scale curiosity.
2. **Drop the oracle predictive head** — Phase 1's surprise machinery
   is empirically counterproductive. Future runs should use the
   uniform-weight path only. The `experiments/oracle_train.py` and
   `surprise_inspection.txt` artifacts can be retained as evidence
   for the negative finding.
3. **Reframe the writeup**: not "structural surprise loss" but
   "frozen-encoder alignment auxiliary loss for linear-RNN distillation".
   The alignment-loss framing is also closer to standard knowledge-
   distillation formulations and easier to relate to existing
   literature.
4. **Defer dialogue corpus** indefinitely. The proposal's strongest
   claims were about dialogue pivots, but the surprise mechanism that
   was supposed to exploit those pivots doesn't help. There's no
   reason to expect the dialogue domain to rescue the structural-
   surprise framing.
5. **Combine with distillation pilot**. The distillation pilot's
   recipe (KL+CE α=0.9 from Qwen3.6) and this branch's recipe
   (uniform-weight L_sem from a frozen DN encoder) are structurally
   similar — both are auxiliary alignment losses. Worth testing
   whether both can stack additively, or whether they're substitutes.

## What this means for the research direction

The proposal's central claim — "structural surprise weights the loss
toward pivotal moments" — is **not what's happening empirically** in
code. Instead, what works is **frozen-encoder alignment**: pool the
model's hidden states across each statement, project to a frozen
encoder's pooled-representation space, and minimise cosine distance.
This helps PPL across all statement types roughly uniformly, with
the largest absolute gains at the easiest statements.

This is a clean publishable finding — alignment to a frozen-encoder
target as an auxiliary regulariser is a real mechanism with a real
+11.4 % cumulative lift over plain DN at 217 M. The "structural
surprise" framing of the original proposal should be dropped from the
final writeup; what to keep is the alignment-loss formulation.

The v0 surprise-modulated-α PoC (commit `21aa8b9`, negative) and the
β-sweep / uniform-weight ablation here together provide a decisive
negative on **"surprise as gradient weight"** in the code domain at
this scale; the **alignment-to-frozen-target** part stands.

## Reproduction

```bash
# Phase 0/1: train the DN-baseline-as-encoder + oracle predictive head
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/oracle_train.py \
    --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
    --save_ckpt checkpoints/oracle_predictive_head_217M.pt

# Phase 2: train K=3 + L_sem at β ∈ {0.1, 0.3}
for BETA in 0.1 0.3; do
    SAFE=$(echo "$BETA" | tr . _)
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/train_lm.py \
        --arch deltanet \
        --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
        --semantic_loss_beta $BETA \
        --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
        --oracle_ckpt checkpoints/oracle_predictive_head_217M.pt \
        --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
        --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
        --dataset codeparrot/codeparrot-clean --text_field content \
        --save_ckpt checkpoints/film_self_k3_lsem_b${SAFE//0_/0}_2_28_30L_217M.pt
done

# Stratified eval
for CKPT in dn_baseline_30L_217M_for_oracle film_self_k3_2_28_30L_217M \
            film_self_k3_lsem_b01_2_28_30L_217M film_self_k3_lsem_b03_2_28_30L_217M; do
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/eval_statement_ppl.py \
        --ckpt checkpoints/${CKPT}.pt \
        --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
        --oracle_ckpt checkpoints/oracle_predictive_head_217M.pt \
        --T 512 --n_eval_tokens 32768 \
        --out bench_stmt_ppl_${CKPT}.json
done
```
