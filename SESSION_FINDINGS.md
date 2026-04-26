# Session findings — Symbol-grounded direction (Direction A from `NOVEL_DIRECTIONS.md`)

**Date:** 2026-04-26
**Question:** Does sequence-aware sparse identifier table (last-write-wins per token-id) help over DeltaNet?

## What was built
- `experiments/tasks/var_binding.py` — synthetic pointer-chasing task: `v = N ; ... ?v` predict latest binding.
- `experiments/layers.py::SymbolGroundedAttention` — per-head sparse table `S ∈ ℝ^{V×D}`, gathers/scatters by token-id, gated last-write-wins update. Pure-PyTorch O(T) loop ref impl. `n_symbols` hash-bucket arg for real-LM use.
- `experiments/train_var_binding.py` — driver mirroring `train_mqar.py`.
- Wired SymbolGrounded + `hybrid_sg`, `hybrid_sg_25_75` arches into `experiments/train_lm.py`.

## What was tested

### 1. Synthetic var_binding task (negative-utility test)
DeltaNet hits **acc=1.000 by step 200** at every config tested:
- T ∈ {64, 128, 256, 512}
- n_vars ∈ {4, 8, 32, 64, 128}
- d_head ∈ {8, 16, 32}
- Even at the tightest config (`d_head=8, n_vars=128`).

Verdict: **var_binding with explicit `=` delimiters is just MQAR**. DeltaNet's rank-1 erase rule was *literally designed* for this kind of update; it's saturated at every scale we can train. SymbolGrounded *also* solves it (acc 1.000 by step 100), so layer is verified working — but the synthetic cannot separate the two architectures.

### 2. TinyStories LM at 13.6M params, T=128, 1500 steps

| Arch | Val PPL @ 1500 | Δ vs DeltaNet |
|------|----------------|---------------|
| DeltaNet | **13.36** | baseline |
| hybrid_sg_25_75 (1 SG + 3 DN) | **13.25** | **−0.8%** |
| hybrid_sg (50/50) | 13.62 | +1.9% |
| symgrounded standalone | ~30+ (plateaued ~step 500, killed) | ~+125% |

### 3. Python code (codeparrot-clean) at 13.6M params, T=128, 1500 steps

| Arch | Val PPL @ 500 | Val PPL @ 1000 | Val PPL @ 1500 | Δ vs DeltaNet |
|------|----------------|------------------|------------------|---------------|
| DeltaNet | 198.66 | 124.17 | **113.68** | baseline |
| hybrid_sg_25_75 | 208.11 | 126.13 | **116.08** | **+2.1%** |

## Honest interpretation

- **Sparse SG hybrid is tied with DeltaNet** — within 2-3% in either direction on both datasets.
- **Standalone SG is decisively worse** (~2× PPL, plateaus early).
- **More SG (50/50) hurts** — worse than 25/75 sparse.
- **The "code-relevant story" did not show up** — code PPL gap (+2.1%) is not better than text PPL gap (−0.8%).
- **Wallclock penalty:** SG layer has Python-loop O(T); ~3-4× slower at 25% mixing, 7× slower standalone.
- **Hash-bucket lossy:** n_symbols=512 vs vocab=49152 means 96-way collisions, cap on what symbolic state can encode.

The hypothesis in `NOVEL_DIRECTIONS.md` was "code's most distinctive property is named identifiers; pointer-chasing across scopes becomes O(1) lookup." The data does not support this at small scale: DeltaNet's *learned soft KV-store* equally handles binding-style retrieval, and SG's hash-bucket limitation removes the one structural advantage (exact identity match) the design promised.

**Verdict on Direction A:** "no harm, no help, slower wallclock." Direction A is not the win.

## What stays valid

- The brainstorm / synthesis from earlier in this session still has 4 unexplored directions:
  - **B. Multi-pass parallel scans** (K cells in parallel within one layer, fused at output) — different from layer-level hybrid; nobody runs *separate scans* in parallel.
  - **C. Event-driven irregular-time cells** — cheap update on whitespace, expensive on syntactic events.
  - **D. Tree-scan over the parse tree** — autoregressive AST-aware sequence model with parallel tree reduction.
  - **E. Verifier-coupled pretraining** — compile/parse signal as a PT aux objective.

Direction E is the biggest "different from frontier" bet because no published autoregressive code LM uses verifier signal during pretraining (only post-training RLHF). It also addresses the *real* bottleneck Agent 07 flagged: "When Perplexity Lies" — a 7B distilled model matching teacher within 0.2pp PPL but losing by 20.8pp on HumanEval.

## What to do next

Two options, in order of expected info-per-effort:

1. **Direction B — multi-pass scan.** Drop-in replacement for one layer: 2-3 cells (delta + heisenberg + maybe sg) running in parallel, learned mixer at output. Tests "is parallel composition better than serial." ~2 days eng. The mechanistic question is: does information from multiple reading modes available at every token outperform alternating layer types? The answer is unknown.

2. **Direction E — verifier-coupled PT.** Aux loss: predict "will the next K tokens parse?" as binary head, applied during code-LM training. Cheap signal (microseconds per check). Code-specific. Tests whether the gap between PPL and HumanEval is closeable from the training side. ~1 week eng (parser pipeline + aux head + ablation).

Both are genuinely novel relative to the 2025 frontier. Both could yield real wins if they work. They're orthogonal — could ship both.
