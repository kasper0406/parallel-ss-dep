# Structural Surprise Loss × sparse-FiLM — design analysis

**Branch:** `structural-surprise-loss`  
**Parent commit:** `2900c51` (Phase 21d — 708M K=3 self-feeding works)  
**Status:** design only, no code or training in this branch yet.

This document captures (a) an honest assessment of integrating the
proposed Structural Surprise Loss with our sparse-(2, 34) FiLM
architecture, and (b) a minimal proof-of-concept that tests the *idea*
without committing to the full multi-phase infrastructure rebuild.

## TL;DR

The proposal is conceptually clean and well-motivated, but it is a
substantial *separate* research project (1-2 months for full v1) rather
than a 1-day extension of our sparse-FiLM work. Three structural
concerns to confront before committing:

1. **Domain mismatch.** Our entire training/eval stack runs on
   codeparrot (Python source). The proposal assumes dialogue
   (DailyDialog / OpenAssistant / PersonaChat) and "sentence
   segmentation" doesn't apply cleanly to source code. To do the full
   proposal we need to switch corpora, re-baseline the 708M sparse-FiLM
   on dialogue, and build a dialogue-side topic-pivot eval set
   (which the plan itself flags as "the single most important piece
   of the whole experiment"). That's ~1-2 weeks of pre-work.

2. **Phase 18 priors are unfavorable.** The
   surprise-gated cross-layer scratchpad (Phase 18) saw surprise
   gating contribute ~0 % on PPL — content-only routing was the
   marginal +0.5 % win, and the simple sparse FiLM beat the most
   expressive scratchpad by ~2.5 %. That was *per-token* surprise on
   *code*; the proposal is *per-sentence* surprise on *dialogue*. The
   prior is suggestive, not damning, but it does mean we have evidence
   that adding a surprise signal to our setup didn't help last time.

3. **K=3 self-feeding already exposes a self-surprise signal for free.**
   The proposal's Phase 3 trains a separate head to predict the
   model's own surprise. We already train a model where iter-`K-1`
   produces a "vanilla" estimate and iter-`K` produces the
   "self-fed" estimate — the per-token L2 norm of their difference
   is structurally a measure of "how much did the model's own
   prediction shift after one round of self-feeding." That's ~⅔ of
   the meta-cognitive signal Phase 3 wants, in a representation we
   already train. This is the cheapest version of Phase 3 and is
   worth validating before training a separate head.

## What composes well, what doesn't

| Proposal component | Composition with sparse-FiLM | Notes |
|---|---|---|
| Phase 1 oracle (frozen `E` + predictive `P`) | Independent of FiLM | Adds an external dependency (Sentence-BERT) and a separate predictive head. Useful artifact regardless of FiLM. Worth building once if pursuing the proposal. |
| Phase 2 `L_sem` (semantic gradient loss in oracle space) | Composes additively as auxiliary loss | The cleanest composition. `L = α·CE + β·L_sem` runs on top of our existing K=3 self-feeding training. The projection `W` and the FiLM module train independently. |
| Phase 2 surprise reweighting of CE | Composes additively | Same as above — orthogonal to FiLM. |
| Phase 3 meta-cognitive head | Composes; or use K=3 inter-iter delta as cheaper proxy | Phase 3 trains a *separate* MLP to predict per-sentence surprise. Our K=3 model already produces a self-surprise-ish signal at iter-K-1 vs iter-K. Try the free signal first. |
| Surprise-modulated FiLM α | Direct integration with FiLM | Replace the single scalar α in the FiLM module with a function `α(s_t) = α_0 · σ(predicted_surprise)`. High-surprise sentences/statements get amplified feedback. Tests whether the architectural lift concentrates at structural pivots. |
| Sentence-level FiLM source | Direct integration with FiLM | Instead of `lag-1` at token granularity, use `lag-1` at sentence/statement granularity for the FiLM input. Doesn't need a surprise oracle at all. |

## Minimal proof-of-concept (proposed; not yet run)

Goal: stay on codeparrot (no infrastructure rebuild), test the *idea*
that structural-surprise-weighted training adds value to our existing
708M sparse-FiLM. ~1 day end-to-end.

**Components needed:**

1. **Statement-level segmentation for Python** in lieu of NLP sentences.
   Either AST-based (`ast.parse` per top-level statement) or a cheap
   heuristic (split on `\n`, merge continuation lines). Track
   `(start_token_idx, end_token_idx)` per statement, the same way
   the proposal asks for sentences.

2. **Free surprise signal**: per-token CE divergence between iter-2
   and iter-3 of the K=3 self-feeding forward, pooled per statement.
   No oracle, no separate predictive head. The model already produces
   both passes during training.

3. **Loss reweighting only** (no `L_sem`, no oracle, no semantic loss
   in v0). At each statement, weight the CE loss by
   `1 + λ · normalized_surprise` for `λ ∈ {0.0, 0.1, 0.3, 1.0}`.

4. **Eval**: held-out PPL on codeparrot val, plus a code-pivot eval
   set (split on `def `/`class `/`import ` boundaries — the natural
   topic shifts for Python). Compare to Phase 21d's K=3 baseline.

If this v0 shows signal, the full proposal (oracle + `L_sem` + Phase 3)
is worth the infrastructure investment. If it doesn't, the 1-2 month
multi-phase plan likely won't either, and we save the time.

## Recommended sequencing

1. **v0 minimal PoC** on codeparrot (~1 day): K=3 inter-iter delta as
   surprise weight, statement-level granularity. ~30 min training at
   217M to compare against Phase 21c K=3 baseline.

2. **If v0 shows signal:** add `L_sem` (Phase 2 of the proposal) on
   top of v0, still on codeparrot. ~1 day. Tests whether the
   semantic-loss component compounds the lift.

3. **If v1 shows signal:** switch to dialogue. Build Phase 0
   infrastructure (DailyDialog pipeline, sentence segmentation,
   pivot eval set). ~1 week.

4. **If v2 dialogue baseline confirms the architectural lift survives
   the domain switch:** run the full 3-phase plan on dialogue with
   the proper oracle, `L_sem`, and meta-cognitive head. ~2-4 weeks.

This sequencing front-loads the cheap signal-checks and back-loads
the infrastructure spend. Each gate is a clean go/no-go for the next
commitment.

## Open questions (not blockers for v0)

- Does Python's natural unit (statement, function, class) carry the
  same kind of "structural surprise" that sentences carry in dialogue?
  The proposal's intuition is that pivots in *intent* and *framing*
  drive the lift; code has analogues (a new function definition is a
  framing shift; an exception handler is an intent shift), but the
  signal density is different.
- Does K=3's inter-iter delta actually correlate with structural
  pivots, or just with token rarity? The proposal's Phase 1
  validation gate ("inspect the top-10% most-surprising sentences")
  applies here too — the free signal is only useful if it eyeballs
  as genuine structural surprise.
- Does sparse-FiLM's existing lift already concentrate at pivots? If
  so, surprise-weighting may be redundant with what the FiLM already
  does. The Phase 14 ablations (depth, source layer) didn't break
  this down by token type — would be a useful retrospective analysis
  before investing in the v0 PoC.

## Out of scope for this branch (per fork directive — one shot)

- Implementing the v0 PoC (statement segmentation, training run).
- Building any of Phase 1, 2, or 3 of the proposal.
- Touching code, model, or eval infrastructure.

This document is design-only. Next step is for the parent agent to
decide whether to commit to the v0 PoC and dispatch a follow-up.
