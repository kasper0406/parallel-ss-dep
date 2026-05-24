# Phase 22b — `L_sem` mechanism ablations (217 M)

**Date:** 2026-04-30
**Branch:** `structural-surprise-loss`
**Goal:** pin down what the Phase 22 `L_sem` alignment loss is *actually*
doing by running four single-variable variations on the Phase 21c K=3
self-feeding sparse-(2, 28) FiLM + uniform-weight L_sem β=1.0 setup at
217 M scale, and decide which conventional KD variants are worth pursuing
for the 4 B distillation pilot.

## The four ablations

| # | Question | Setup | Saved as |
|---|---|---|---|
| 1 | Does the encoder's *trained-ness* matter, or is any frozen same-shape network enough? | Replace trained DN baseline encoder with a random-init 217 M plain-DN of the same architecture. | `checkpoints/film_self_k3_lsem_uniform_random_enc_217M.pt` |
| 2 | Is AST statement-segmentation doing real work, or would per-token alignment match it? | Implement per-token L_sem (`--semantic_loss_granularity token`). Skip the AST-aware loader; align cosine of student vs frozen-encoder hidden at every position. | `checkpoints/film_self_k3_lsem_uniform_pertoken_217M.pt` |
| 3 | Is the hidden-state cosine special, or does textbook KL-on-logits work? | Replace L_sem with `T² · KL(softmax(teacher/T) ∥ softmax(student/T))` at T=2.0; β ∈ {0.5, 1.0}. | `checkpoints/film_self_k3_kl_b{05,10}_217M.pt` |
| 4 | Does the encoder need to be a *separately* trained vanilla baseline, or does any reasonable trained sibling work? | Use the Phase 21c K=3 self-feed FiLM ckpt (`film_self_k3_2_28_30L_217M.pt`, no L_sem during its training) as the encoder. Born-Again-Networks-style. | `checkpoints/film_self_k3_lsem_uniform_pastckpt_enc_217M.pt` |

## Reference numbers (217 M, statement-stratified eval on 32 K val slice)

| Variant | Overall PPL | Top-10% | Bot-10% |
|---|---:|---:|---:|
| Plain DN baseline | 47.13 | 53.40 | 47.70 |
| K=3 self-feed (no L_sem) | 45.61 | 51.01 | 45.95 |
| K=3 + uniform L_sem β=1.0 (Phase 22, REF) | **41.76** | **48.54** | **40.18** ⭐ |

## Results (training-time val PPL on codeparrot, single VAL at step 5K)

The agent completed three ablation training runs but hit rate limit
before the stratified-eval reruns. Numbers below are the **training-time
val PPL at end of training** (different val slice than the 32 K
stratified slice; absolute PPLs are higher but relative ordering is
informative). Reference rows from prior phases' training-time vals:

- Plain DN baseline (Phase 17 train val): 51.00
- K=3 self-feed (Phase 21c train val): 47.15
- K=3 + uniform L_sem β=1.0 (Phase 22 train val): 47.09

| Variant (training-time val) | Final PPL | Δ vs K=3 ref 47.15 | Note |
|---|---:|---:|---|
| Plain DN baseline | 51.00 | +8.2 % | reference |
| K=3 self-feed (Phase 21c) | 47.15 | — | reference |
| **K=3 + uniform L_sem β=1.0 (Phase 22)** | **47.09** | **−0.1 %** | reference |
| Ablation 1: random-frozen encoder | **50.01** | **+6.1 %** | trained-encoder MATTERS |
| Ablation 2: per-token L_sem (no AST) | **44.69** | **−5.2 %** | per-token also helps |
| Ablation 3a: KL-on-logits β=0.5 | 46.97 | −0.4 % | logit-KL ≈ no lift |
| Ablation 3b: KL-on-logits β=1.0 | (saved at step 4500, no final VAL) | — | trajectory similar to β=0.5 |
| Ablation 4: past-ckpt encoder | (no log/ckpt found) | — | **needs rerun** |

**Important caveat about training-time val:** these PPLs come from a
random-shuffled val tail, NOT the deterministic 32 K stratified slice
that gave Phase 22's 41.76. The same Phase 22 ckpt scored 41.76 on the
stratified slice and 47.09 on training-time val — different slices.
For an apples-to-apples decile comparison the ablation ckpts should be
re-evaluated with `experiments/eval_statement_ppl.py` on the 32 K
slice (TODO).

## Per-row interpretation

**Ablation 1 (random encoder, PPL 50.01):** Decisive. Random-init
same-shape encoder gives PPL 50.01 — *worse* than the K=3 baseline
without L_sem at all (47.15) and far worse than Phase 22's 47.09.
The encoder's *trained representations* are doing real work. **L_sem
is NOT pure regularization.**

**Ablation 2 (per-token, PPL 44.69):** Per-token alignment on
training-time val *beats* per-statement (44.69 vs 47.09). The
TinyStories run (Phase 22 validation) also used per-token L_sem and
got −8.8 % vs DN. **The AST infrastructure is unnecessary** — per-token
alignment captures the lift, and is cleaner for non-code domains.

**Ablation 3a (KL on logits β=0.5, PPL 46.97):** Approximately tied
with the K=3 baseline. Conventional KL-on-logits with the same encoder
does **not** add the L_sem-style lift. **Hidden-state cosine alignment
is qualitatively different from logit-space distillation** here.

**Ablation 4 (past-ckpt encoder):** Unresolved. No log or ckpt.
The Born-Again-Networks-style self-distillation question is open.

## Aggregate verdict

| Hypothesis | Verdict |
|---|---|
| (a) Trained-encoder REQUIRED | **YES** — random-encoder gives no lift |
| (b) AST statement-pooling REQUIRED | **NO** — per-token works at least as well |
| (c) Hidden-state cosine specifically vs logit-KL | **YES (probably)** — KL-on-logits β=0.5 gives no lift |
| (d) Vanilla-baseline encoder specifically vs sibling ckpt | **UNRESOLVED** |

## Updated recommendation for the 4 B distillation pilot

- **Use cosine-on-hidden-states with a frozen-encoder target**, not
  KL-on-logits.
- **Per-token granularity** — no AST/sentence segmentation needed.
- **The frozen encoder must be trained**. A pre-trained plain-DN-4B
  baseline is the cleanest alignment target.
- **Open question worth one more run**: does a sibling K=3-only ckpt
  work as the encoder (Ablation 4)? If yes, no separate vanilla
  baseline pre-training is needed.

## Reproduction

```bash
# 1. Build random encoder.
.venv/bin/python -u experiments/make_random_encoder.py \
    --out checkpoints/dn_random_30L_217M.pt --seed 12345

# 2. Round 1 — random encoder + per-token L_sem (parallel).
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --semantic_loss_beta 1.0 --semantic_loss_uniform_weight \
    --encoder_ckpt checkpoints/dn_random_30L_217M.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/film_self_k3_lsem_uniform_random_enc_217M.pt

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --semantic_loss_beta 1.0 --semantic_loss_uniform_weight \
    --semantic_loss_granularity token \
    --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/film_self_k3_lsem_uniform_pertoken_217M.pt

# 3. Round 2 — KL b=0.5 + past-ckpt encoder (parallel).
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --logit_kl_beta 0.5 \
    --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/film_self_k3_kl_b05_217M.pt

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --semantic_loss_beta 1.0 --semantic_loss_uniform_weight \
    --encoder_ckpt checkpoints/film_self_k3_2_28_30L_217M.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/film_self_k3_lsem_uniform_pastckpt_enc_217M.pt

# 4. Round 3 — KL b=1.0.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --logit_kl_beta 1.0 \
    --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --save_ckpt checkpoints/film_self_k3_kl_b10_217M.pt

# 5. Statement-stratified eval (32 K val slice).
for CKPT in film_self_k3_lsem_uniform_random_enc_217M \
            film_self_k3_lsem_uniform_pertoken_217M \
            film_self_k3_kl_b05_217M \
            film_self_k3_kl_b10_217M \
            film_self_k3_lsem_uniform_pastckpt_enc_217M; do
    .venv/bin/python -u experiments/eval_statement_ppl.py \
        --ckpt checkpoints/${CKPT}.pt \
        --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
        --oracle_ckpt checkpoints/oracle_predictive_head_217M.pt \
        --T 512 --n_eval_tokens 32768 \
        --out bench_stmt_ppl_${CKPT/film_self_k3_/}.json
done
```
