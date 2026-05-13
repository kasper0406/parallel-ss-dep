# state-dep-parallel

## Current direction — "Small super-coder"

The active goal of this repo is to build a **small, efficient code model that competes with much larger models on coding benchmarks** under tight compute (2× RTX 5090). The architectural research below feeds into that target.

Stack as of 2026-05-13:
- **DeltaNet** backbone (`--arch deltanet`) — bounded-state linear RNN, no KV-cache cost. *Note: `gated_deltanet` is broken on sm_120 RTX 5090 (FLA Triton kernel bug); plain `deltanet` works.*
- **Sparse FiLM feedback (2, 28)** + K=3 self-feeding — −3 % to −5 % PPL at 217 M / 360 M / 708 M, single forward at deploy.
- **Bounded working memory** ([`experiments/model.py::WorkingMemory`](experiments/model.py)) — write-gated buffer of past hidden states, read via soft attention at "think" / query positions. **+11.1 pp recall** on saturated MQAR (T=512, K=128) vs DeltaNet alone, no O(T²) attention cost. Validated 2026-05-12 — see [SESSION_FINDINGS](SESSION_FINDINGS.md).
- **Thinking gate** — per-position σ head choosing emit vs think. Allows extra recurrent passes at hard positions and triggers memory reads.
- **Mixed-corpus pretrain** (`experiments/data_mix.py`, `configs/pretrain_mix_v*.yaml`) — 9–11 weighted HuggingFace streams (code, instruct, CS textbooks, Wikipedia, optionally BigVul + CyberNative CVE data) with chunk-boundary think-burst injection so memory + gate train from step 0. Auto-stop on flat HumanEval over two 500 M-token intervals.
- **Speed**: `--bf16 --tf32` measured 2.28× over fp32 on 5090 (~18 k → ~42 k tok/s).

Active run (2026-05-13): [`launch_pretrain_mix_v2.sh`](launch_pretrain_mix_v2.sh) — 217 M, 2.13 B tokens, ~14 hr ETA.

Project framing: [`THESIS.md`](THESIS.md).
Post-pretrain RL plan: [`PHASE_C_RL.md`](PHASE_C_RL.md).
Architectural write-up: [`WORKING_MEMORY_FINDINGS.md`](WORKING_MEMORY_FINDINGS.md).

---

## Architecture-research background

Mapping the design space of **state-dependent, parallelizable RNN cells**
— with a specific finding: in a DeltaNet stack, a *single*
sparse late-to-early FiLM connection (a minimal-form descendant of
GF-RNN-style top-down feedback) gives a robust **~3-5 % PPL lift**
over the underlying linear-RNN cell that survives a **3.3× parameter
scale-up** (217 M → 708 M) and an optimizer change (AdamW → Muon).
Mechanistically characterized (Phase 14b–g) — multiplicative form +
non-softmax aggregation finds a previously-unreported optimization
basin — and extrapolates cleanly to 16× training context.

**Architectural lift over DeltaNet, all three scales:**

| Scale / optimizer | DN baseline | + Sparse FiLM | Δ | α |
|---|---|---|---|---|
| 217 M / AdamW / 5 K | 51.00 | 49.40 | **−3.1 %** | −0.054 |
| 360 M / Muon  / 15 K | 22.79 | 21.57 | **−5.4 %** | +0.158 |
| 708 M / Muon  / 15 K | 35.38 | 34.26 | **−3.2 %** | −0.198 |

**Cross-architecture scoreboard** (codeparrot, Muon-tuned, T=512):

```
Vanilla Transformer @ 360M:                   18.78 PPL  ← strongest at this scale
Sparse-(2, 34)-FiLM DeltaNet @ 708M:           34.26 PPL  ← deployment-memory-fair
DeltaNet @ 708M:                               35.38 PPL
```

The 708 M RNN is sized so its inference-time state is comparable to
a 360 M Transformer's KV cache (deployment-memory parity). At this
token budget (~30 M, far below Chinchilla-optimal for 708 M) the
Transformer still wins on raw quality — we do **not** close the
cross-class gap. What we do show is that the **architectural lift
of sparse-FiLM over DeltaNet is robust** across 217 M → 360 M →
708 M; the comparative claim against Transformer is scale- and
optimizer-dependent.

**Inference cost.** Originally the architecture required a 2-pass
forward at both training and decode time (pass 1 produces the FiLM
input, pass 2 is the actual model output) — and a naïve "lagged-
cached" inference shortcut breaks quality (PPL 36.97, worse than
plain DN at 35.38; see [`LATENCY_REPORT.md`](LATENCY_REPORT.md)).
**Phase 21c–d fixes this**: training with K=3 self-feeding (the
model's own lagged source-layer output as FiLM input) closes the
train/inference gap entirely. At 708M, K=3 self-feeding gives PPL
**34.85 at 1× decode cost** (vs std-2-pass 34.26 at 2× decode cost,
or std-2-pass lagged-cached 36.97 — broken). The deployment story:

| Variant @ 708M | PPL | Decode cost | vs DN baseline 35.38 |
|---|---:|---:|---:|
| Std-2-pass FiLM (2-pass eval) | 34.26 | 2× | −3.2 % |
| **K=3 self-feeding FiLM (lagged-cached)** | **34.85** | **1×** | **−1.5 %** ⭐ |
| Std-2-pass FiLM (lagged-cached, broken) | 36.97 | 1× | +4.5 % |

The deployment-memory advantage is intact: the RNN inference state
is **74× smaller** than a 360M Transformer's KV cache at 8K context
(9.8 MB vs 720 MB). The honest deployment-fair claim is **−1.5 %
lift at 1× decode cost** with K=3 self-feeding training.

The high-level idea of upper-to-lower-layer feedback in stacked RNNs
is not new — see *Related work* below — but the **specific minimal
form, the modern linear-RNN context, and the mechanism analysis are.**

- [`THESIS.md`](THESIS.md) — project framing: why methodology-stacking
  at small scale is the bet, what the claim is and isn't
- [`PHASE_C_RL.md`](PHASE_C_RL.md) — prediction-as-RL-signal proposal:
  reward code that's both correct *and* whose behaviour the model
  predicted; the RL plan for after Phase A pretrain + Phase B SFT
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
gives a robust ~3-5 % PPL lift over the underlying linear-RNN cell.
The lift survives 3.3× parameter scale-up and an optimizer change.**

| Setup | DN baseline | + Sparse FiLM | Δ | α |
|---|---|---|---|---|
| 217 M / AdamW / 5 K steps  | 51.00 | 49.40 (2, 28) | **−3.1 %** | −0.054 |
| 360 M / Muon  / 15 K steps | 22.79 | 21.57 (2, 28) | **−5.4 %** | +0.158 |
| 708 M / Muon  / 15 K steps | 35.38 | 34.26 (2, 34) | **−3.2 %** | −0.198 |

The architecture: a **single FiLM-style cross-layer connection** from
a late layer's output (lagged by 1 token) to layer 2's input, with one
learnable scalar α. +0.3 % extra parameters. The network discovers α
from data; sign and magnitude depend on optimizer + scale (sign is
*not* monotone in scale — see Phase 20), but the architectural lift
holds across all three configurations we tested. Pair index scales
with depth (`(2, 28)/30 ≈ (2, 34)/36`).

3-seed reproducibility on the 217 M variant: **49.40 ± 0.31** (σ < 1 %).

**Honest cross-architecture framing.** At 217 M / AdamW sparse-FiLM
beat vanilla Transformer by 23 %; at 360 M with Muon (which was
*designed* for Transformer attention matrices), Transformer wins.
The 708 M run was sized for **deployment-memory parity** with a
360 M Transformer (RNN state ≈ Transformer KV cache budget):

```
Vanilla Transformer @ 360M Muon:                    18.78 PPL
Sparse-(2, 34)-FiLM DeltaNet @ 708M Muon:            34.26 PPL  ← deployment-memory-fair
DeltaNet @ 708M Muon:                                35.38 PPL
```

At ~30 M training tokens (well below Chinchilla-optimal for 708 M),
deployment-memory parity does **not** close the cross-class gap.
What survives at scale is the architectural lift *within* the
linear-RNN family. Honest framing: *sparse-FiLM is the strongest
known modification within the linear-RNN family at the scales we
can afford to train; the cross-architecture comparison would need a
Chinchilla-scale or frontier-finetune follow-up*.

```python
                     pass-1 (vanilla forward)
   x → L0 → L1 → L2 → ... → L28 → ... → L29 → out
                      ↑                ↓
                      └── lag(t-1) ────┤
                          FiLM α≈−0.054 │
                                         pass-2 input
```

## Current Frontier: Reinforcement Learning (GRPO) & Deep Thinking

We are currently transitioning the architecture from supervised BPTT to **Reinforcement Learning (GRPO)**.

### Phase 24 Discovery: Autonomous Thinking "From Scratch"
Recent experiments (May 2026) have established that:
- **RL > Supervised:** Transitioning to GRPO has solved the "Maladaptive Thinking" trap. Models are now rewarded only when thinking directly improves next-token prediction.
- **Autonomous Discovery:** We successfully trained a thinking model **from scratch** (starting from a non-thinking DN baseline). The model autonomously discovered the utility of the [THINKING] token, reaching a ~12% thought rate through pure RL exploration.
- **Scaling Depth:** The "From SFT" model has already scaled to an average depth of 1.25 (max 10) while maintaining control over perplexity.

### Next Steps: Benchmark & RAG
1.  **Evaluation:** We are now running the RL-trained agents on existing benchmarks (**HumanEval**, **MBPP**, and mechanistic tasks) to quantify the reasoning lift from extra "thought" passes.
2.  **Continuous RAG:** Integrating the external vector database to give the model a concrete source of information to "think about."

See [`RL_RAG_ROADMAP.md`](RL_RAG_ROADMAP.md) for the technical integration plan.

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
