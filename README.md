# state-dep-parallel

Mapping the design space of **state-dependent, parallelizable RNN cells**
— with a specific architectural finding: a single sparse cross-layer
feedback connection in a 30-layer DeltaNet stack beats vanilla
Transformer (−23 %), Mamba2 (−12.5 %), and pure DeltaNet (−3.1 %) at
matched params on Python code, mechanistically explained, and
extrapolating cleanly to 16× training context.

- [`RESULTS.md`](RESULTS.md) — full empirical writeup, Phases 1-16
- [`NEXT_DIRECTIONS.md`](NEXT_DIRECTIONS.md) — current research plan
- [`HISTORY.md`](HISTORY.md) — earlier hybrid + Lean + Triton-kernel
  work that preceded the architectural finding (Phases 1-13)
- [`SESSION_FINDINGS.md`](SESSION_FINDINGS.md) — chronological session log
- [`StateDep/`](StateDep/) — Lean project (13 monoids proved associative)
- [`kernels/`](kernels/) — Triton kernels for sm_120
- [`experiments/`](experiments/) — training drivers and architecture modules

## Headline finding

**Sparse far-distance top-down feedback** beats the obvious SOTA
baselines at matched params on Python code:

@30L, 217 M params, T=512, codeparrot Python, 5 K AdamW steps,
identical block structure (RMSNorm + attention + RMSNorm + GLU FFN
with d_ff=4·d_model), lr=3e-4 cosine (no warmup), seed=0:

```
Vanilla Transformer (softmax + abs pos emb): 60.75 PPL
Mamba2 (SSM):                                55.60 PPL
DeltaNet (linear-RNN):                       51.00 PPL
Sparse-(2, 28)-FiLM DeltaNet (ours):         49.40 PPL ⭐
```

The architecture: a **single FiLM-style cross-layer connection** from
layer 28's output (lagged by 1 token) to layer 2's input, with one
learnable scalar α. +0.3 % extra parameters. The network discovers
α ≈ −0.054 from data — a *subtractive* predictive-coding filter that
forms a previously-unreported optimization basin in this class of
architectures.

3-seed reproducibility on the (2, 28) variant: **49.40 ± 0.31** (σ < 1 %).

```python
                     pass-1 (vanilla forward)
   x → L0 → L1 → L2 → ... → L28 → ... → L29 → out
                      ↑                ↓
                      └── lag(t-1) ────┤
                          FiLM α≈−0.054 │
                                         pass-2 input
```

## Validation matrix

| Metric | Result | Reference |
|---|---|---|
| Reproducibility (3 seeds) | 49.40 ± 0.31 (σ < 1 %) | RESULTS Phase 14 pre-flight |
| Depth ablation | beats DN at 8L / 15L / 30L (−2.6 % / −0.8 % / −3.1 %) | Phase 14 |
| TinyStories | −1.6 % vs DN (not code-specific) | Phase 14 |
| 16× T extrapolation | stable −4–5 % at T=512 → 8192 | Phase 14 |
| 8× T extrapolation, all SOTA baselines | wins at every T tested | Phase 16 |
| Mechanism: 7+ controlled ablations | direction/target/source/aggregation/form all matter | Phase 14b–g |
| Mechanism: cross-layer attention forms tested | sparse single-pair = FiLM-sum = sigmoid-attn (basin tied) | Phase 14d–g |

## Mechanism

The negative-α basin is reached **iff** two conditions hold:

1. **Multiplicative output form** (FiLM-shaped `x · (1 + α·s) + α·t`,
   not additive Q-K-V residual). The gradient on α flows through an
   `x · scale` term that gives a strong, x-correlated direction. Pure
   additive residuals lack this.
2. **Non-softmax aggregation** across sources. Sum (`film_sum`) and
   independent sigmoid gates (`film_sigmoid`) both work; softmax
   routing causes 1/K-dilution at init that forces α toward 0.

Either failure mode alone is sufficient to break the basin. With
both conditions met, multiple architectures converge to PPL ≈ 49.4 at
@30L:
- Sparse single-pair (2, 28) FiLM
- Multi-source FiLM-sum 3-source
- Sigmoid-gated cross-layer attention
- Distributed multi-pair (2, 28)+(4, 24)+(8, 20)

## What's next

Confirmation step before any paper writeup:

- **Literature search** — thorough check for prior work on backward /
  cross-layer / top-down state propagation in RNN architectures. The
  finding looks novel relative to what we know, but we need to verify
  before claiming so.

Then:

- **Generalize the cell** — swap DeltaNet for **DeltaProduct**
  ([Yang et al. NeurIPS 2025][deltaproduct]) or **PD-SSM** as the base
  RNN cell, test if sparse cross-layer feedback gives the same
  architectural lift on top of stronger linear-RNN cells.
- **Scale-up validation** — train at 350-500 M params on 5-10 B tokens
  to check the architectural ordering doesn't flip with more data.
- **Distillation revisit** — initial Qwen3.6 distillation (Phase 15)
  was a negative result due to teacher–data misalignment; a coder-
  aligned teacher (DeepSeek-Coder, Qwen3-Coder-Next) might change
  this.

## Build

Lean library — requires `elan`, `lake`, network access:
```bash
cd StateDep
source $HOME/.elan/env
lake exe cache get
lake build
```

Python environment (uv + cu132 nightly torch + triton + flash-linear-attention):
```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install numpy flash-linear-attention
```

For the distillation pipeline (vLLM teacher in a separate venv):
```bash
uv venv .venv-vllm
VIRTUAL_ENV=$(pwd)/.venv-vllm uv pip install vllm transformers datasets
```

## Reproduce the headline finding

```bash
# DeltaNet baseline
python experiments/train_lm.py --arch deltanet --feedback none \
  --steps 5000 --d_model 576 --n_heads 9 --d_head 64 --n_layers 30

# Sparse (2, 28) FiLM
python experiments/train_lm.py --arch deltanet --feedback film \
  --feedback_pairs "2,28" --steps 5000 --d_model 576 --n_heads 9 \
  --d_head 64 --n_layers 30

# Vanilla Transformer baseline (needs --max_T because softmax attention
# is permutation-invariant without positional info)
python experiments/train_lm.py --arch transformer --max_T 512 \
  --feedback none --steps 5000 --d_model 576 --n_heads 9 --d_head 64 \
  --n_layers 30

# Mamba2 baseline
python experiments/train_lm.py --arch mamba2 --feedback none \
  --steps 5000 --d_model 576 --n_heads 9 --d_head 64 --n_layers 30
```

All runs use codeparrot/codeparrot-clean, T=512, batch=8, lr=3e-4
cosine.

## Earlier work

The cell-level architectural exploration that preceded the sparse-
feedback finding (hybrid `[ortho, deltanet]` stack, AUSSM concurrent
prior art, Lean library, Triton kernels) is documented in
[`HISTORY.md`](HISTORY.md). That work led to the empirical observation
that motivated this finding: cell-level architecture saturates before
the inter-cell information flow. Sparse cross-layer feedback is the
inter-cell flow that turned out to matter.

[grazzi]: https://arxiv.org/abs/2411.12537
[zoology]: https://arxiv.org/abs/2312.04927
[aussm]: https://arxiv.org/abs/2507.05238
[deltaproduct]: https://arxiv.org/abs/2502.10297
