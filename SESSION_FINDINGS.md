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

---

## Update 2026-04-26 evening: Scale-up of multipass_dd flips the verdict

Followed Option 1 (scale up multipass_dd). Ran on Python code at intermediate
scale: T=256, 3000 steps, ~30M params.

### Compute-fair comparison (both ~16 attention-block-equivalents)

| Arch | n_layers | Params | Val PPL @ 3000 | Δ vs DN |
|------|----------|--------|----------------|---------|
| DeltaNet @16L | 16 | 42.0M | 69.57 | baseline |
| **multipass_dd @8L** | 8 (×2 cells/layer) | **35.7M** | **67.79** | **−2.6%** |

multipass_dd wins with **15% fewer parameters** AND ran 10-15% faster
wall-clock (fewer FFN blocks).

### Convergence trajectory tells the real story

| Step | DN @16L val PPL | multipass_dd @8L val PPL | Δ |
|------|-----------------|---------------------------|---|
| 1000 | 125.05 | 128.16 | DN ahead +2.5% |
| 2000 | 86.41 | 82.29 | multipass ahead **−4.8%** |
| 3000 | 69.57 | 67.79 | multipass ahead **−2.6%** |

multipass starts behind, catches up, then pulls ahead. Best gap mid-training, narrows slightly by end.

### What changed vs the 14M comparison?

At 14M (1500 steps, T=128):
- DN @8L code: 108.36
- multipass_dd @4L code: 108.50 → tied

At 30M (3000 steps, T=256):
- DN @16L code: 69.57
- multipass_dd @8L code: 67.79 → multipass wins

**The architectural advantage emerges with scale.** Possibilities:
1. The two-cell ensemble effect provides regularization that helps with longer training.
2. Each cell has more capacity (d_model=256 vs 128), so individual specialists can learn more diverse features.
3. T=256 gives more long-range dependencies where the ensemble of state-tracking cells helps.

### Promotion: Direction B is no longer "dead"

The earlier 14M-scale verdict ("loses to deeper DeltaNet at compute parity") only held at small scale. At 30M with longer training, multipass_dd wins clean. This is the first novel-direction positive result this session.

**Next test:** scale further. multipass_dd @ 100-135M params, 5000+ steps, T=512 on code. Cost: 4-8 hours per run. If the gap holds or widens, this is a real architectural finding.

### Caveats
- 30M is still small; need to confirm at 135M
- Tested only on Python code; need TinyStories at 30M to confirm gap is consistent
- Single seed; need 2-3 seeds for confidence
- The multipass_dd architecture uses `delta + delta_negeig` — two flavors of the same cell. Whether the win generalizes to other cell pairings (delta + ortho, delta + heisenberg with proper init) is unknown.

---

## Update 2026-04-26 late evening: 135M scale-up does NOT replicate

Ran the queued 135M / 5K-step experiment.

| Arch | Params | FFN blocks | Val PPL @ 5K | Wallclock |
|------|--------|------------|--------------|-----------|
| **DeltaNet @30L** | 216.3M | 30 | **51.00** | baseline |
| multipass_dd @15L | 156.5M | 15 | 52.97 (+3.9%) | 1.5× faster |

### Convergence trajectory

| Step | multipass | DeltaNet | Δ |
|------|-----------|----------|---|
| 1000 | 115.29 | 119.88 | mp −3.8% |
| 2000 | 86.93 | 86.69 | DN +0.3% |
| 3000 | 66.45 | 65.21 | DN +1.9% |
| 4000 | 57.70 | 55.94 | DN +3.1% |
| 5000 | 52.97 | 51.00 | **DN +3.9%** |

multipass starts ahead but DN's depth advantage compounds steadily. By
step 5000, DN wins by 3.9%.

### Honest re-reading of direction B

The 30M-scale "win" was an artifact of the specific compute regime:
short training (3000 steps), shorter T (256), and the "warmup phase"
dominating the comparison. At frontier scale (5000 steps, T=512), DN's
depth wins.

multipass_dd is genuinely more **parameter-efficient** (28% fewer params
for 3.9% PPL loss) and **wallclock-efficient** (1.5× faster). For
deployment scenarios where PARAM count matters more than PPL, this is a
valid trade-off. But it's not a "novel architecture beats DeltaNet on
LM" win — DN @30L wins on PPL at compute parity.

### The full session in one table

| Direction | Best variant | Best result | Verdict |
|-----------|--------------|-------------|---------|
| A | hybrid_sg_25_75 | tied within ±2% | not the win |
| B | multipass_dd @30M | -2.6% PPL vs DN | seemed like win... |
| B | multipass_dd @135M | +3.9% PPL vs DN | ...didn't replicate at scale |
| E | bracket aux | tied at w=0.1, hurts at w=0.5 | not the win |
| (compute baseline) | deeper DeltaNet | always wins on PPL | local optimum |

### Final session conclusion

After 7+ hours of compute across 2× RTX 5090 testing 4 novel directions
at multiple scales (14M, 30M, 135M):

**Zero clean architectural wins for "novel" cells over depth-scaled
DeltaNet at our compute budget.**

Pattern is robust:
- At small scale, novel cells tie within noise
- At medium scale, some novel cells appear to win (regression-to-mean
  effect of "warmup phase")
- At frontier scale, depth-scaled DeltaNet pulls ahead

This is consistent with the 10-agent brainstorm's frontier-convergence
finding: industry settled on softmax+linear hybrids and depth scaling
for good reasons. Novel cell engineering at small lab scale doesn't
easily produce wins.

**Recommended next steps (if user wants to continue):**

1. **Frontier-with-twist** (option C from earlier strategic options).
   Take Gated DeltaProduct (Siems Feb '25, 1-2 PPL gain documented at
   805M scale) + identity-init + structural aux. Novel as combination,
   not as components. Highest expected PPL win.

2. **Generation eval pivot**. Set up HumanEval pass@k evaluation. Tests
   whether architectural differences hide behind PPL parity. Direction
   E (verifier-coupled training) and Direction B (multipass) might both
   show generation-side wins even when PPL ties.

3. **Distillation from frontier coder** (literature recipe). LoLCATs from
   Qwen2.5-Coder-1.5B → ~3 GPU-weeks budget on 2× 5090. Highest-confidence
   path to a useful coder model. Not novel but functional.

4. **Accept depth-scaled DeltaNet as the local optimum**. Write up the
   negative results as a paper ("when novel cells don't beat well-tuned
   baselines: a small-lab perspective"). Pivot research focus to evals,
   inference-side innovations, or different problem entirely.

## What to do next

Two options, in order of expected info-per-effort:

1. **Direction B — multi-pass scan.** Drop-in replacement for one layer: 2-3 cells (delta + heisenberg + maybe sg) running in parallel, learned mixer at output. Tests "is parallel composition better than serial." ~2 days eng. The mechanistic question is: does information from multiple reading modes available at every token outperform alternating layer types? The answer is unknown.

2. **Direction E — verifier-coupled PT.** Aux loss: predict "will the next K tokens parse?" as binary head, applied during code-LM training. Cheap signal (microseconds per check). Code-specific. Tests whether the gap between PPL and HumanEval is closeable from the training side. ~1 week eng (parser pipeline + aux head + ablation).

Both are genuinely novel relative to the 2025 frontier. Both could yield real wins if they work. They're orthogonal — could ship both.

---

## Update 2026-04-26 night: Cross-layer top-down feedback (Day 1+2)

User proposed a fundamentally different direction: cross-layer top-down
feedback with t-1 lag (parallel-scan-friendly). 4 modes implemented:
none / additive / film / predictive. 2-pass forward — pass 1 collects
each layer's outputs, pass 2 modulates each layer's input by the
*next-deeper layer*'s pass-1 output, shifted right by 1.

α_L per-layer learnable strength, init=0 so model starts as feedforward.
Critical bug fix: also init W_fb (and W_scale/W_shift) non-zero —
otherwise gradient on α is zero forever.

### 14M / 1500 steps / T=128 (TinyStories + code)

| Mode | TinyStories PPL | Code PPL |
|------|-----------------|----------|
| none (baseline) | 13.36 | 113.68 |
| additive | 13.00 (−2.7%) | 110.01 (−3.2%) |
| predictive | 13.00 (−2.7%) | (not run) |
| **film** | **12.93 (−3.2%)** | **107.28 (−5.6%)** |

film wins on both, bigger on code than text. **First clean
positive-direction result of this session.**

### 30M / 3000 steps / T=256

| Eval | DN @16L | film @16L | Δ |
|------|---------|-----------|---|
| TinyStories | 6.96 | 7.03 | film **+1.0% worse** |
| Code | 69.57 | 68.06 | film **−2.2% better** |

The TinyStories win disappeared at 30M; the code win shrank from
−5.6% to −2.2%. Pattern: the win is **code-specific** and **shrinks
with scale**.

### 135M / 5000 steps / T=512 (running) — the critical test

multipass_dd's 30M code win disappeared completely at 135M (DN was
+3.9% better). Will film's 30M code win survive? Currently at step
400 of 5000 (~5 hours). film 0.02 nats ahead in train; need val
trajectory to know.

### Honest read

Cross-layer feedback **is** the first novel direction this session
that gives real LM PPL improvement at small scale. The fact that the
win is bigger on code than text matches the mechanistic story — code
has explicit hierarchy (function/class/scope) that benefits from
top-down modulation more than natural text does.

But: the win shrinks with scale. By 30M:
- TinyStories: gone
- Code: half its 14M magnitude

If the 135M run shows further shrinkage to 0% or negative, this is
the same "small-scale artifact" pattern as multipass_dd. If it holds
at ~2% positive on code, it's a genuine code-specific architectural
finding worth deeper investigation.
