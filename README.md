# state-dep-parallel

A **small, efficient code model** built to punch above its weight on coding
benchmarks under tight compute (2× RTX 5090, 32 GB each, no NVLink).
Architectural research feeds that target. Engineering details and the running
work log live in [`GEMINI.md`](GEMINI.md); project framing in
[`THESIS.md`](THESIS.md).

## Architecture

![Model architecture](docs/architecture.svg)

## Stack

- **DeltaNet** backbone — bounded-state linear RNN, no KV-cache cost.
  (`gated_deltanet` is broken on sm_120; plain `deltanet` works.)
- **Shallow-wide trunk** — 10L × 896d + 5 dense reverse-FiLM pairs + K=3
  self-feed. Iso-param swap of the 30L × 576d trunk; ~18 % faster wall-clock,
  matches/beats it on VAL ppl.
- **Sparse FiLM feedback** — the headline architectural finding (below).
- **Working Memory** — write-gated bounded buffer, read at think positions.
  +11 pp on saturated MQAR recall; 98 % long-context recall.
- **Product-Key Memory** — 262 k learned KV slots after one block. Load-bearing
  on HumanEval (−5 in ablation) once the v7.1 bootstrap-fix package is used.
- **Thinking gate** — per-position emit/think head. *Status below.*
- **Mixed-corpus pretrain** with cross-document state isolation (`cu_seqlens`
  from per-position `doc_ids`).
- **Execution-grounded RL** ([`train_rl_grader.py`](experiments/train_rl_grader.py))
  — GRPO with a dense `code_grader` reward. The post-pretrain capability lever.

## Current status (honest)

- **Best HumanEval pass@1: 16/164** (`rl_grader_phase_c_v2_step300`, KL-stable
  GRPO on the Chinchilla-completed 287 M base).
- **Validated primitives**: FiLM (−3–5 % PPL), PKM (−5 HumanEval ablation),
  WM (98 % recall). These earn their place on their own metrics.
- **The open question — does *thinking* improve task correctness?** Currently
  **no** at this scale, and on a weak base it actively *hurts*: a held-out
  probe shows discrete-token thinking gives negative next-token Δlogp, and a
  thinking-on vs thinking-off HumanEval split is 3 vs 8 — the model thinks
  where it's *uncertain*, but uncertainty ≠ "thinking helps". Latent
  (high-bandwidth, Coconut-style) thinking works on synthetic reasoning tasks
  but, bolted onto a converged trunk, costs base quality without a measured
  win. The standing conclusion: a thinking mechanism has to be **co-trained
  into a strong base**, not added post-hoc — and we **require a measured
  thinking contribution before scaling on it**. Full trail in `GEMINI.md`.
- Dead ends documented so they aren't repeated: discrete-token thinking
  (never amplifies on this trunk), gate aux loss targeting the wrong
  mechanism, latent thinking as a post-hoc bolt-on (regresses VAL),
  continuation SFT/DPO on off-distribution data (regresses).

## Headline architecture finding

**A single sparse late-to-early FiLM connection in a DeltaNet stack gives a
robust ~3–5 % PPL lift that survives a 3.3× param scale-up and an optimizer
change.** One late-layer output (lagged 1 token) modulates an early layer via
FiLM with one learnable scalar α (+0.3 % params).

| Setup | DN baseline | + Sparse FiLM | Δ |
|---|---|---|---|
| 217 M / AdamW / 5 K  | 51.00 | 49.40 (2,28) | **−3.1 %** |
| 360 M / Muon / 15 K  | 22.79 | 21.57 (2,28) | **−5.4 %** |
| 708 M / Muon / 15 K  | 35.38 | 34.26 (2,34) | **−3.2 %** |

3-seed reproducibility at 217 M: 49.40 ± 0.31 (σ < 1 %). K=3 self-feeding
closes the train/inference gap → −1.5 % lift at **1× decode cost**, with RNN
inference state 74× smaller than a matched Transformer's KV cache.

**Mechanism**: the lift comes from a negative-α subtractive basin reachable
*iff* (i) modulation is multiplicative (FiLM-form, not Q-K-V additive) and
(ii) cross-source aggregation is non-softmax (sum or sigmoid; softmax dilutes
by 1/K). 20+ controlled ablations; either condition failing breaks the basin.

**Honest framing**: cross-architecture (vs Transformer) the comparison is
scale/optimizer-dependent — Transformer wins at 360 M/Muon. What's robust is
the lift *within* the linear-RNN family. The top-down-feedback idea isn't new
([GF-RNN][gfrnn] 2015, [BRIMs][brims] 2020); the contribution is the
*minimal-form demonstration + mechanism* in a modern linear-RNN coding LM.

## Build

```bash
# Python (uv + cu132 nightly torch + local flash-linear-attention fork)
uv venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install numpy
uv pip install -e /home/knielsen/ml/flash-linear-attention   # Blackwell fixes
export PYTHONPATH=$PYTHONPATH:.

# Lean library (StateDep/) — requires elan + lake
cd StateDep && source $HOME/.elan/env && lake exe cache get && lake build
```

## Reproduce the headline finding

```bash
# DeltaNet baseline vs Sparse-(2,28) FiLM (codeparrot, T=512, batch=8, lr=3e-4)
python experiments/train_lm.py --arch deltanet --feedback none \
  --steps 5000 --d_model 576 --n_heads 9 --d_head 64 --n_layers 30
python experiments/train_lm.py --arch deltanet --feedback film \
  --feedback_pairs "2,28" --steps 5000 --d_model 576 --n_heads 9 \
  --d_head 64 --n_layers 30
```

[gfrnn]: https://arxiv.org/abs/1502.02367
[brims]: https://arxiv.org/abs/2006.16981
