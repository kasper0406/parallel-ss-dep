# state-dep-parallel

Mapping the design space of **state-dependent, parallelizable RNN cells**
— with a specific finding: in a DeltaNet stack, a *single*
sparse late-to-early FiLM connection (a minimal-form descendant of
GF-RNN-style top-down feedback) gives a robust ~3-5 % PPL lift over
the underlying linear-RNN cell, **growing with scale**: from −3.1 %
at 217 M / AdamW to −5.4 % at 360 M / Muon. Mechanistically
characterized (Phase 14b–g) — multiplicative form + non-softmax
aggregation finds a previously-unreported optimization basin —
and extrapolates cleanly to 16× training context.

**Honest scale-up scoreboard** (360 M, Muon, codeparrot, 15 K steps):

```
Vanilla Transformer (Muon-tuned):   18.78 PPL  ← strongest at this scale
Sparse-(2, 28)-FiLM DeltaNet:        21.57 PPL  ← strongest in linear-RNN family
DeltaNet baseline:                   22.79 PPL
```

At smaller-scale + AdamW (217 M / 5 K) the Transformer was the worst
(60.75); at 360 M / Muon (the optimizer it was designed for) it
catches up and surpasses. The **architectural lift of sparse-FiLM
over DeltaNet holds and grows at scale**, but the comparative claim
against Transformer is scale- and optimizer-dependent.

The high-level idea of upper-to-lower-layer feedback in stacked RNNs
is not new — see *Related work* below — but the **specific minimal
form, the modern linear-RNN context, and the mechanism analysis are.**

- [`RESULTS.md`](RESULTS.md) — full empirical writeup, Phases 1-16
- [`NEXT_DIRECTIONS.md`](NEXT_DIRECTIONS.md) — current research plan
- [`HISTORY.md`](HISTORY.md) — earlier hybrid + Lean + Triton-kernel
  work that preceded the architectural finding (Phases 1-13)
- [`SESSION_FINDINGS.md`](SESSION_FINDINGS.md) — chronological session log
- [`StateDep/`](StateDep/) — Lean project (13 monoids proved associative)
- [`kernels/`](kernels/) — Triton kernels for sm_120
- [`experiments/`](experiments/) — training drivers and architecture modules

## Headline finding

**A single sparse late-to-early FiLM connection in a DeltaNet stack
gives a robust ~3-5 % PPL lift over the underlying linear-RNN cell,
with the lift growing at scale.**

| Setup | DN baseline | + Sparse (2, 28) FiLM | Δ | α |
|---|---|---|---|---|
| 217 M / AdamW / 5 K steps | 51.00 | 49.40 | **−3.1 %** | −0.054 |
| 360 M / Muon  / 15 K steps | 22.79 | 21.57 | **−5.4 %** | +0.158 |

The architecture: a **single FiLM-style cross-layer connection** from
layer 28's output (lagged by 1 token) to layer 2's input, with one
learnable scalar α. +0.3 % extra parameters. The network discovers
α from data; sign and magnitude depend on optimizer + scale, but
both reach the same architectural lift over the underlying cell.

3-seed reproducibility on the 217M variant: **49.40 ± 0.31** (σ < 1 %).

**Honest scale-up note** — the comparative ranking against attention
flips at scale. At 217 M / AdamW (no warmup) sparse-FiLM beat vanilla
Transformer by 23 %; at 360 M with Muon (which was *designed* for
Transformer attention matrices), Transformer wins:

```
Vanilla Transformer (Muon-tuned):     18.78 PPL  ← strongest at this scale
Sparse-(2, 28)-FiLM DeltaNet:          21.57 PPL  ← strongest in linear-RNN family
DeltaNet baseline:                     22.79 PPL
```

The architectural lift over DeltaNet holds across scales; the cross-
attention-class ranking depends on optimizer choice. Honest framing:
*sparse-FiLM is the strongest known modification within the linear-RNN
family at modern scale*.

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

## Related work — honest novelty accounting

The high-level construct (top-down feedback in stacked RNNs) is not
new. Three prior works are direct antecedents and must be cited:

| Paper | What's theirs | What's still ours |
|---|---|---|
| **[GF-RNN][gfrnn]** (Chung, Gulcehre, Cho, Bengio, ICML 2015) | The *idea* of top-down feedback from upper RNN layers to lower layers, with learnable gates per layer pair. Tested with tanh / LSTM / GRU on character LM and Python program eval. | They use *gated additive* updates (not multiplicative FiLM); feedback at *all* layer pairs (not single sparse pair); sequential RNNs (not parallel-scan-friendly); no mechanism characterization (no negative-α basin observation); small scale. |
| **[BRIMs][brims]** (Mittal, Lamb, Goyal et al., ICML 2020) | The *t−1 lag trick* for parallel-scan friendliness with this kind of feedback — the higher-layer-to-lower-layer signal looks at the previous timestep precisely to preserve causality. | Attention-based (not FiLM); modular RIM cells; every adjacent layer pair (not sparse single-pair); small models. |
| **[SparX][sparx]** (Lou, Cao et al., AAAI 2025) | The *"sparse cross-layer connection"* terminology and a sparse adjacency pattern in the Mamba/Transformer family. | They do **forward** DenseNet-style feature aggregation in a vision backbone (not late→early feedback in language); cross-attention with channel-wise routing (not FiLM); vision domain only. |

Adjacent but mechanically distinct:
**Feedback Transformer** ([Fan et al. 2020][feedbackt]),
**Staircase Attention** ([Ju et al. 2021][staircase]),
**TransformerFAM** ([Hwang et al. 2024][fam]) — feedback/memory
mechanisms in Transformer blocks, mostly within-block or temporal
rather than cross-layer in stacked RNN.
**Universal Transformer** ([Dehghani et al. 2019][ut]),
**Loop-Residual** ([Loop-Residual NN][loopres]),
**Depth-Recurrent Transformer** — depth-recurrence with parameter
sharing rather than a fixed sparse inter-layer connection.
**PredNet** ([Lotter, Kreiman, Cox 2017][prednet]) — predictive
coding for video with conv layers; theoretical inspiration only.
The predictive coding lineage from Rao & Ballard (1999) onward is the
theoretical frame the negative-α basin matches.

### What's actually new here

What the prior art does not have, that this project contributes:

1. **The "minimal form" empirical finding.** Among the family of
   GF-RNN-style top-down feedbacks, the *minimum* useful structure
   for a linear-RNN coding LM is a single sparse pair `(2, 28)` with
   FiLM modulation. Multi-pair, all-pair, attention-routed, dense-
   feedback, and cross-attention variants all *fail* to beat this —
   in many cases they fail to find the basin at all.
2. **The mechanism characterization.** 20+ controlled ablations
   identify that a previously-unreported *negative-α subtractive
   basin* exists, and is reachable iff (i) the modulation is
   *multiplicative* (FiLM-form, not Q-K-V additive) and (ii)
   aggregation across sources is *non-softmax* (sum or independent
   sigmoid; softmax dilutes by 1/K and forces α toward 0). Either
   failure alone breaks the basin.
3. **Modern linear-RNN, matched-everything-except-attention vs SOTA.**
   Identical block structure across Sparse-FiLM / DeltaNet / Mamba2
   / vanilla Transformer, sole variable is the attention class, at
   217 M params on Python code; sparse-FiLM beats by 23 / 12.5 / 3.1 %.
4. **Long-T extrapolation comparison** across all four architectures.
   Sparse-FiLM wins at every T from 512 to 4096 (8×); Transformer
   with absolute-position embeddings can't extrapolate at all.

So the claim shifts from "novel architecture" to **"minimal-form
empirical demonstration plus mechanism, of an idea that has been
around since GF-RNN 2015, in a modern linear-RNN coding LM where it
hasn't been tested."** That's still publishable, just honest.

## What's next

- **Generalize the cell** — swap DeltaNet for **DeltaProduct**
  ([Yang et al. NeurIPS 2025][deltaproduct]) or **PD-SSM** as the
  base RNN cell, test if the sparse cross-layer feedback gives the
  same architectural lift on top of stronger linear-RNN cells.
- **Scale-up validation** — train at 350-500 M params on 5-10 B
  tokens to check the architectural ordering doesn't flip with more
  data.
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
[gfrnn]: https://arxiv.org/abs/1502.02367
[brims]: https://arxiv.org/abs/2006.16981
[sparx]: https://arxiv.org/abs/2409.09649
[feedbackt]: https://arxiv.org/abs/2002.09402
[staircase]: https://arxiv.org/abs/2106.04279
[fam]: https://arxiv.org/abs/2404.09173
[ut]: https://arxiv.org/abs/1807.03819
[loopres]: https://arxiv.org/abs/2409.14199
[prednet]: https://arxiv.org/abs/1605.08104
