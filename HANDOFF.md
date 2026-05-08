# Handoff — state of the project at session reset (2026-05-08)

This doc is the single-page brief for the next agent picking up. It
covers (a) what's empirically settled, (b) what's open, (c) where to
find what, and (d) suggested next steps in priority order.

## TL;DR

We have a **deployment-honest −9.0 % ± 1.8 pp lift over plain DeltaNet
at 217 M** (multi-seed mean), and **−10.0 % at 708 M** (single seed),
on codeparrot Python, **at 1× decode cost**. The lift comes from the
combination of three components:

1. **Sparse cross-layer FiLM** — single FiLM modulation from layer
   N's lag-1 output to layer 2's input, one learnable scalar α.
2. **K=3 self-feeding** — iterative-fixed-point training that closes
   the train/inference gap so the cheap lagged-cached decode protocol
   (1× cost) matches the training-faithful 2-pass eval (would be 2×).
3. **Uniform-weight L_sem alignment loss** — auxiliary cosine-distance
   loss between the student's pooled hidden states and a frozen
   plain-DN baseline's pooled hidden states. **This is the single
   biggest contributor.**

The same recipe **also lifts TinyStories by −8.5 %** with per-token
L_sem (no AST), so it's not a code-specific artifact.

The original "structural surprise" framing of L_sem is **rejected**:
weighting the loss by the oracle's surprise score is empirically
counterproductive. The actual mechanism is **alignment to a trained
frozen-DN encoder's representations** (a sibling of the student).

## Current branches

| Branch | What's there | Last commit |
|---|---|---|
| `main` | Project root + the (paused) 4 B distillation pilot | `08882e4` (lit assessment) inherited |
| `structural-surprise-loss` | Phase 21c–22d structural-surprise work | `08882e4` (literature novelty assessment) — uncommitted ablation/validation results pending; this commit will land them. |
| Worktree `parallel-ss-dep-distill/` | DN-4B distillation pilot scripts (no full run yet) | `182796c` (BPB byte accounting) |

## What's empirically settled (the headline numbers)

### Architectural lift over plain DN at 708 M, deployment-honest (1× decode)

| Stack @ 708 M | PPL on 32 K val slice | Δ vs plain DN | Decode cost |
|---|---:|---:|---:|
| Plain DN baseline | 35.38 | — | 1× |
| + K=3 self-feeding sparse-FiLM (Phase 21d) | 34.85 | −1.5 % | 1× |
| + K=3 + uniform L_sem β=1.0 (Phase 22) | **31.83** | **−10.0 %** | **1×** |

Std-2-pass FiLM (Phase 20) had a "training-faithful" PPL of 34.26 but
its lagged-cached deployment quality was 36.97 — **broken** at 1× decode.
K=3 self-feeding fixed that train/inference gap.

### Multi-seed stability at 217 M (Phase 22 validation, statement-stratified eval)

| Seed | Overall | Top-10% | Bot-10% | Δ vs DN 47.13 |
|---|---:|---:|---:|---:|
| 0 (Phase 22) | 41.76 | 48.54 | 40.18 | −11.4 % |
| 1 | 43.69 | 49.66 | 43.60 | −7.3 % |
| 2 | 43.15 | 48.86 | 43.01 | −8.4 % |
| **mean ± σ** | **42.87 ± 0.85** | 49.02 ± 0.49 | 42.26 ± 1.50 | **−9.0 % ± 1.8 pp** |

σ ≈ 1.0 PPL ≈ 10 % of the lift component. **Reproduces.** Headline
should drop from "−11.4 %" (lucky seed=0) to "−9.0 % ± 1.8 pp" (mean).

### Natural-text validation (TinyStories, 217 M)

| Variant | Stratified PPL | Δ vs plain DN |
|---|---:|---:|
| Plain DN baseline | 6.71 | — |
| K=3 self-feed | 6.65 | −1.0 % |
| **K=3 + per-token L_sem β=1.0** | **6.14** | **−8.5 %** |

The L_sem lift survives the code → natural-text transition. **Per-token
alignment is sufficient** — no AST/sentence segmentation needed. This
**closes the largest open claim risk** from the literature search.

### Mechanism ablations at 217 M (Phase 22b — TRAIN-TIME val PPL, ≠ stratified)

| Ablation | Train-val PPL | Δ vs Phase 22 ref 47.09 | Verdict |
|---|---:|---:|---|
| Random-frozen encoder | 50.01 | +6.1 % | Trained encoder REQUIRED |
| Per-token L_sem (no AST) | 44.69 | −5.2 % | AST not required |
| KL-on-logits β=0.5 | 46.97 | −0.4 % | KL doesn't add lift |
| KL-on-logits β=1.0 | (no final VAL) | — | trajectory similar to β=0.5 |
| Past-ckpt encoder | (no log/ckpt) | — | **rerun needed** |

Caveat: ablation rows used training-time val PPL, not the 32 K
stratified slice. The Phase 22 ref's 47.09 is its training-time val,
NOT the 41.76 stratified figure. For an apples-to-apples decile
comparison, ablation ckpts need to be re-evaluated with
`experiments/eval_statement_ppl.py`.

### Inference economics (Phase 20.5 / 21d)

At 8 K context, batch=1, 5090:

| Variant | Decode ms/tok | State (MB) |
|---|---:|---:|
| Plain DN 708 M | 14.5 | 9.8 |
| K=3 + L_sem 708 M (lagged-cached) | 14.5 | 9.8 |
| Reference: Transformer 360 M | 4.6 | **720** |

State is **74× smaller** than the comparable Transformer's KV cache.
Decode latency for the K=3 + L_sem RNN is the same as plain DN.

## What's open / unresolved

1. **Phase 22b Ablation 4 (past-checkpoint encoder)** — never ran to
   completion. The Born-Again-Networks question (does a sibling
   K=3-only ckpt work as the encoder, vs requiring a pre-trained
   vanilla baseline?) is open. **Cheapest experiment to fill in next.**
2. **Stratified eval of all Phase 22b ablation ckpts** — to make the
   ablation table comparable to Phase 22's 41.76 reference. ~5 min
   each on idle GPU. The training is done; just need eval.
3. **708 M multi-seed** — Phase 22 at 708 M was seed=0 only. 217 M
   showed σ ≈ 0.85. ~3 h per additional seed at 708 M.
4. **DN-4B distillation pilot** — the headline "scales to 4 B" claim.
   Validated at 1 B / 1 K steps (DISTILL_PILOT_REPORT.md, α=0.9 winning
   recipe). The full pilot was scoped, scripts staged in the
   `parallel-ss-dep-distill` worktree, but the multi-day run was not
   started before the rate limit hit. **The pilot's recipe should now
   include K=3 + uniform L_sem, not just KL+CE** — an open design
   choice flagged in the L_SEM_MECHANISM_ABLATIONS.md recommendation.
5. **L_sem β > 1.0 is plateaued at 217 M** (β=0.5 → 0.9; β=1.0 → 0.93;
   β=2.0 → 0.94; uniform-weight at β=1.0 → 0.93). Don't push β higher
   without good reason.
6. **Literature gap that remains:** the "frozen-vanilla-baseline-as-
   alignment-target for a stronger architecture" recipe parallels Born-
   Again Networks (Furlanello 2018) closely. The novelty case rests on
   the *combination* with K=3 + sparse-FiLM + linear-RNN context, not
   on L_sem alone (per `LITERATURE_NOVELTY_ASSESSMENT.md`).

## Where things live (file map)

### Recipes / docs
- `STRUCTURAL_SURPRISE_FULL.md` — full Phase 21c–22 writeup. Largest
  reference doc. Has reproduction commands.
- `STRUCTURAL_SURPRISE_VALIDATION.md` — Phase 22 validation: multi-seed
  + natural-text (this commit).
- `L_SEM_MECHANISM_ABLATIONS.md` — Phase 22b: 4 ablations (this commit).
- `LITERATURE_NOVELTY_ASSESSMENT.md` — sober prior-art analysis (commit
  `08882e4`).
- `MECHANISM_REPORT.md` — Phase 21 (state-capacity rejected),
  Phase 21b (forget-gate rejected), Phase 21c (K=3 self-feeding works
  at 217 M), Phase 21d (K=3 verified at 708 M).
- `LATENCY_REPORT.md` — Phase 20.5 decode-latency benchmark + lagged-
  cached PPL parity (commit `d2876e3`); should be referenced when
  discussing inference economics.
- `RESULTS.md` — historical phases 1–21d. Older, doesn't include
  Phase 22 / Phase 22b yet — add a Phase 22 section if writing a paper.
- `README.md` — public-facing summary; honestly notes the L_sem result.
- `HISTORY.md` — pre-Phase-14 work.

### Code (`experiments/`)
- `model.py` — `TinyLM` with all knobs: `feedback_pairs`,
  `feedback_self_k`, `semantic_loss_dim`, `feedback_alpha_mode`. The
  K=3 self-feeding is `feedback_self_k=3`.
- `train_lm.py` — main trainer. New flags this branch added:
  `--feedback_self_k`, `--feedback_alpha_mode`, `--semantic_loss_beta`,
  `--semantic_loss_uniform_weight`, `--semantic_loss_granularity
  {statement, token}`, `--encoder_ckpt`, `--oracle_ckpt`. Encoder is
  required if β > 0; oracle is **optional** under uniform-weight.
- `eval_statement_ppl.py` — statement-stratified eval (decile
  breakdown by oracle surprise on the 32 K val slice).
- `eval_filmed_ppl_217m.py` / `eval_filmed_ppl_708m.py` — 2-pass /
  lagged-cached protocol comparison evals.
- `oracle_train.py` — predictive head training (Phase 21c, now mostly
  vestigial since uniform-weight L_sem doesn't use it).
- `make_random_encoder.py` — Phase 22b utility: builds a random-init
  same-shape DN ckpt for the random-encoder ablation.
- `statement_segmentation.py`, `statement_stream.py` — AST-based
  statement boundary infrastructure. Used by Phase 22 per-statement
  L_sem; per-token mode bypasses these (and gets the same lift).
- `decode_bench.py` — decode-latency benchmark with both 2-pass
  (sequential), 2-pass (overlap), lagged-cached protocols.

### Checkpoints (`checkpoints/`)
Important ones for the next agent:
- `dn_baseline_30L_217M_for_oracle.pt` — frozen 217 M DN encoder used
  by all Phase 22 / 22b / 22d L_sem runs. Same-architecture vanilla baseline.
- `dn_36L_708M_muon.pt` — frozen 708 M DN encoder used by Phase 22.
- `dn_random_30L_217M.pt` — random-init 217 M for Ablation 1.
- `film_self_k3_2_28_30L_217M.pt` — Phase 21c K=3-only at 217 M.
- `film_self_k3_lsem_uniform_b10_708M_muon.pt` — Phase 22 winner ckpt.
- `film_self_k3_lsem_uniform_*` family — Phase 22 / 22b / 22c variants.
- `film_self_k3_30L_217M_tinystories_seed0.pt` and the L_sem variant —
  TinyStories Phase 22d.

### Worktrees
- `/home/knielsen/ml/parallel-ss-dep` (this dir) — `structural-surprise-loss` branch.
- `/home/knielsen/ml/parallel-ss-dep-distill` — `main` branch with the
  paused 4 B distillation pilot (vLLM + KL+CE recipe scripts; no full
  run yet, only the 1 B / 1 K-step validation `f5b81e4`).

## Suggested next steps in priority order

1. **(20 min) Stratified eval of the Phase 22b ablation ckpts** so the
   ablation table is on the same scale as Phase 22's 41.76 ref. The
   ckpts already exist; just run `eval_statement_ppl.py` on each.
2. **(45 min) Rerun Phase 22b Ablation 4 (past-checkpoint encoder)**.
   This is the only ablation that didn't complete and it answers an
   important "is the vanilla-baseline-as-encoder special?" question.
   Use `checkpoints/film_self_k3_2_28_30L_217M.pt` as the
   `--encoder_ckpt`.
3. **(2 days, GPU 0) Full DN-4B distillation pilot** with the updated
   recipe (KL+CE α=0.9 from Qwen3.6 + per-token L_sem with frozen DN-4B
   baseline). The frozen DN-4B baseline doesn't exist yet; pre-training
   it adds another ~1 day on GPU 0. Net: ~3 days. Without L_sem (just
   KL+CE) is the original task #137 scope and is faster; the new
   recommendation is to fold L_sem in.
4. **(3 h, GPU 1) 708 M Phase 22 multi-seed (seed=1, seed=2)** to get
   σ at scale. The 217 M σ ≈ 0.85 might shrink at 708 M (more capacity,
   less seed variance) but should be measured.
5. **(in parallel with above) Writeup**. The data is publishable. The
   strongest framing per `LITERATURE_NOVELTY_ASSESSMENT.md` is:
   *"A Born-Again recipe for sparse-feedback DeltaNet — integrating
   cross-layer FiLM, K-pass self-feeding, and frozen-baseline alignment
   for a 9–10 % deployment-honest improvement at 1× decode cost."*
   Workshop track at NeurIPS / ICLR / ICML is realistic with current
   data. Full paper needs the 4 B distillation pilot result.

## Pitfalls and gotchas

- **Two val slices.** Training-time val PPL (random shuffle) ≠ the 32 K
  stratified slice (`shuffle(seed=42).skip(10000)`). They give different
  absolute PPLs (e.g., 47.09 vs 41.76 for the Phase 22 ckpt). Always
  compare apples-to-apples on the *same* val slice. The stratified one
  is the primary comparison for the writeup.
- **The `--oracle_ckpt` is optional under `--semantic_loss_uniform_weight`.** This
  was relaxed in the train_lm.py changes for Phase 22's 708 M run.
  Earlier Phase 22b ablations may have still required it.
- **The W_semantic projection** (added to `TinyLM` when
  `semantic_loss_dim > 0`) is part of the saved state_dict but unused
  at inference. The `load_film_ckpt` helper in `eval_filmed_ppl_708m.py`
  was patched to instantiate this layer when reading a Phase 22 ckpt.
- **fla TileLang sm_120 forget-gate bug.** GatedDeltaProduct with
  `use_forget_gate=True` and GatedDeltaNet (which has it implicitly)
  hit `CUDA_ERROR_MISALIGNED_ADDRESS` on RTX 5090 + cu132 nightly.
  Workaround: `import scripts.sm120_tilelang_workaround` and clear the
  TileLang cache. See `BUG_sm120_forget_gate.md` and
  `scripts/tilelang_sm120_fix.patch` (a one-line upstream fix that's
  ready to submit but hasn't been pushed).
- **Surprise-modulated α (commit `21aa8b9`)** was a clean negative
  result and the surprise machinery is **gone** from the recommended
  recipe. Keep it as evidence in the writeup; don't reuse it.

## Status of the agent task list as of session reset

Tasks #149–#152 are marked `in_progress` but their underlying work
landed (or partially landed) — see git log + this commit. The tasks
should be marked completed by the next agent after reviewing this doc.
