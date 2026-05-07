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

## Recommendations

1. **Continue the β sweep**: run β ∈ {0.5, 1.0, 2.0} to find where the
   lift saturates. The trajectory β=0.0 → 0.1 → 0.3 → ? is monotone
   so far.
2. **Run a uniform-weight ablation** at the winning β: `L_sem` with
   `surprise(s_t) ≡ 1` (no surprise weighting). This isolates whether
   the surprise machinery contributes anything beyond the alignment
   loss itself. If uniform matches surprise-weighted, the surprise
   parts of the design are unnecessary in this domain.
3. **Verify at 708 M scale**: at 217 M the lift is large; at scale it
   may attenuate (similar to the smaller cross-cell pattern). Run
   K=3 + L_sem β=0.3 at 708 M with the same Phase 20/21d Muon setup,
   compare to K=3-only at 708 M (PPL 34.85).
4. **Defer Phase 3** (meta-cognitive head) until the surprise-weighting
   ablation answers whether structural surprise is doing meaningful
   work.
5. **Defer dialogue corpus** until the code result generalises at
   scale. The reframing ("alignment loss + general signal" rather
   than "structural-surprise loss") may not require dialogue-specific
   evaluation infrastructure.

## What this means for the research direction

The proposal's central claim — "structural surprise weights the loss
toward pivotal moments" — is **not what's happening empirically** in
code at 217 M. Instead, `L_sem` appears to be a general additive
auxiliary loss that pushes hidden-state representations toward a
frozen-encoder target, helping PPL across all statement types
roughly uniformly.

This is still a publishable finding — alignment to a frozen-encoder
target as a regulariser is a real mechanism — but the framing in the
final write-up should be honest about what's actually working. The
v0 surprise-modulated-α PoC (commit `21aa8b9`, also negative) and
this Phase 0-2 result together suggest the **"surprise as gradient
weight"** thesis has weak empirical support in the code domain at
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
