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

---

## Direction B (multi-pass) — tested 2026-04-26 evening

**Setup:** `MultiPassAttention` runs K cells in parallel on the same residual, fuses outputs via softmax mixture (alpha learnable, init=zeros = uniform 1/K).

### Three multi-pass variants

1. **multipass_dh** (DeltaNet + Heisenberg): catastrophic failure.
   - TinyStories PPL 30.21 vs DeltaNet 13.36 (+126% worse)
   - Code PPL 237.69 vs DeltaNet 113.68 (+109% worse)
   - Heisenberg is poor at LM; convex combination drags everything down.

2. **multipass_dd** (DeltaNet + DeltaNet_negeig): marginal win vs same-depth baseline.
   - TinyStories PPL 12.96 vs DeltaNet@4L 13.36 (-3.0%)
   - Code PPL 108.50 vs DeltaNet@4L 113.68 (-4.6%)
   - But uses 2× compute per layer.

3. **DeltaNet@8L** (compute-parity baseline, 14.7M params):
   - TinyStories PPL 12.25 → **beats multipass_dd by 5.5%**
   - Code PPL 108.36 → **ties multipass_dd within 0.1%**

### Honest verdict on Direction B
- The "multi-pass win" is purely a compute artifact. At compute parity, deeper DeltaNet wins on text and ties on code.
- The framework only gains marginal value when both cells are strong LMs *and* slightly different (here: delta vs delta_negeig is basically two flavors of the same cell).
- Pairing a strong cell with a weak cell (multipass_dh) catastrophically destroys the strong cell.
- The mechanistic intent ("every reading mode at every token") doesn't yield a real win at this scale.

**Direction B is NOT the win.** Closest mechanism in published lit is Hymba's head-level mix of attention + Mamba, but that pairs cells of complementary type (recall + state-tracking). Our linear+linear pairing has no such complementarity to exploit.

---

## Updated state of the novel-directions queue

- ~~Direction A (symbol-grounded scan)~~ — tested, tied with DeltaNet, not the win.
- ~~Direction B (multi-pass parallel scans)~~ — tested, loses to deeper DeltaNet at compute parity.
- **Direction C (event-driven irregular-time cells)** — untested. Cheap to test (~1 day).
- **Direction D (tree-scan over the parse tree)** — untested. Expensive (~3-4 weeks).
- **Direction E (verifier-coupled pretraining)** — untested. Most novel, ~1 week eng.

**Next bet:** Direction E. Reasoning:
1. A and B both failed at the cell-design level. The remaining novel cell-level designs (C, D) likely fail similarly — the small-scale-LM landscape is dominated by depth-scaling and well-tuned cells.
2. Direction E is fundamentally different: it changes the *training objective*, not the cell. Less likely to fall into the "two cells averaged is worse than one cell deep" trap.
3. Direction E directly addresses the PPL-vs-HumanEval gap that Agent 07 flagged. Even if it doesn't improve PPL, it might improve actual code generation quality.
4. Distillation (literature) is the highest-confidence improvement, but the user wants novel.

---

## Direction E (verifier-coupled aux) — first variant tested

**Setup:** Bracket-depth auxiliary loss. Per-token aux head predicts current bracket depth (0..24 classification). Loss: cross-entropy with weight 0.1, added to LM loss.

Implementation: `experiments/aux_brackets.py` precomputes a (vocab_size,)
table of `#open_brackets - #close_brackets` per token text; per-token depth = cumsum along the sequence. Aux head is `nn.Linear(d_model -> 25)` on the final layer's hidden state.

### Results (codeparrot-clean, 1500 steps, T=128, ~14M params, seed 0)

| Arch | Val PPL @ 1500 | Δ vs baseline |
|------|----------------|---------------|
| DeltaNet baseline | 113.68 | — |
| DeltaNet + bracket aux (w=0.1) | **113.48** | **−0.18%** (noise) |

Train losses tracked nearly identically step-for-step. The aux head consumed gradient bandwidth without measurably reshaping the residual stream.

**Why it didn't bite:** the aux head at the final layer can learn to read off bracket depth without forcing earlier layers to encode it. With weight=0.1 the gradient pressure is small. The bracket signal at T=128 affects relatively few token transitions — most of the LM loss comes from per-token vocabulary distributions where bracket depth is a weak predictor.

Plans for follow-up (running now): aux_weight=0.5 to see if stronger pressure produces a visible LM effect. If still null, definitive negative on this aux signal at this scale.

**aux_weight=0.5 result:** PPL **116.14** = +2.2% worse than baseline. The stronger aux signal *actively hurts* LM performance. Combined with the weight=0.1 null result, this is a definitive negative: bracket-depth aux either does nothing (low weight) or steals capacity from the LM (high weight). The signal is wrong-shaped.

---

## Pattern across all four novel-direction experiments

| Direction | Variant | TinyStories | Code | Verdict |
|-----------|---------|-------------|------|---------|
| A | symgrounded standalone | 30+ (plateau) | — | catastrophic loss |
| A | hybrid_sg_25_75 | 13.25 (-0.8%) | 116.08 (+2.1%) | tied |
| A | hybrid_sg (50/50) | 13.62 (+1.9%) | — | small loss |
| B | multipass_dh | 30.21 (+126%) | 237.69 (+109%) | catastrophic |
| B | multipass_dd | 12.96 (-3.0%) | 108.50 (-4.6%) | wins vs same-depth, loses to deeper |
| B | DeltaNet @8L (compute-parity) | **12.25** | **108.36** | **best on text, ties on code** |
| E | DeltaNet + bracket aux | 113.48 (-0.18%) | — | tied (running w=0.5) |

**The pattern:** at our scale (~14M params, 1500 steps), the LM optimization landscape is well-explored by depth-scaled DeltaNet. Novel cells either:
- Catastrophically fail when paired with weak baselines (multipass_dh, symgrounded)
- Match depth-scaled DeltaNet within noise (hybrid_sg_25_75, multipass_dd, aux_brackets)
- Lose to depth-scaled DeltaNet on text while tying on code (multipass_dd)

No clear architectural wins from cell engineering, parallel composition, or aux losses *at this scale*.

## Strategic options going forward

Given 4 novel-direction experiments yielded zero clean wins:

**Option 1: Scale up.** Run experiments at 135M params / 5K-10K steps where
architectural differences may show. Cost: 4-8 hours per run.

**Option 2: Generation-side eval.** Maybe novel architectures help on
HumanEval-pass@1 even when PPL is tied. Need a generation eval pipeline.
Most novel directions (E especially) are about quality not PPL.

**Option 3: Frontier-with-a-twist.** Combine literature recipes (Gated
DeltaProduct, identity-init, distillation) in a way nobody else has.
Less novel per component, more novel as a system.

**Option 4: Accept depth-scaled DeltaNet as the local optimum.** Pivot
away from architecture research toward eval/inference innovations
(direction 10 from the original brainstorm: S* state-forking, MoR).

**Option 5: Test directions C (event-driven) or D (tree-scan over AST).**
But the pattern strongly suggests they'd be more null results.

## What to do next

Two options, in order of expected info-per-effort:

1. **Direction B — multi-pass scan.** Drop-in replacement for one layer: 2-3 cells (delta + heisenberg + maybe sg) running in parallel, learned mixer at output. Tests "is parallel composition better than serial." ~2 days eng. The mechanistic question is: does information from multiple reading modes available at every token outperform alternating layer types? The answer is unknown.

2. **Direction E — verifier-coupled PT.** Aux loss: predict "will the next K tokens parse?" as binary head, applied during code-LM training. Cheap signal (microseconds per check). Code-specific. Tests whether the gap between PPL and HumanEval is closeable from the training side. ~1 week eng (parser pipeline + aux head + ablation).

Both are genuinely novel relative to the 2025 frontier. Both could yield real wins if they work. They're orthogonal — could ship both.
