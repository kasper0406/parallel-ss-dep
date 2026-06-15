# Verdict: PKM works (for facts)

## Summary
**PKM is the one side mechanism that is load-bearing out-of-the-box.** It contributes positively on every data source (largest on fact-heavy ones), is never negative, and ablating it costs −5/−50 % on HumanEval once the [[product-key-memory]] v7.1 bootstrap is used. The reason it works where [[working-memory-recall-saga]] doesn't: facts reduce next-token CE *everywhere* in text, so the ordinary LM loss automatically creates PKM's bottleneck — no special data needed ([[objective-function-alignment]]). But its benefit is **bounded** by under-exposure and catastrophic forgetting. Source: `CLAUDE.md`, `WHY_THINKING_MARGINAL_ON_CODE.md`, `project_code_thinking_ceiling.md`.

## The positive numbers
- **Per-source CE toggle (v7.1-pkm-film final ckpt)**: wikipedia **+0.099**, cybernative +0.063, code_exercises +0.046, bigvul +0.044, others +0.01–0.03 — **always positive, largest on fact-heavy sources** (exactly as the PKM literature predicts).
- **HumanEval ablation**: `pkm_off −5/−50 %` on the Phase C SFT base — genuinely contributing once given enough pretrain (v12 inherited +0.090, even stronger).
- Healthy bootstrap end-state: αL +0.346, value-row norm 2.54 (155 % drift from init), ~40 k active slots.

## What bounds it (the addendum findings)
- **Not capacity.** A 76 k-param model memorizes ~10 k facts; 287 M ≈ tens of millions; MBPP needs thousands. At matched params PKM ≈ dense for storage (no magic capacity edge).
- **Under-exposure is the real wall.** A precise fact needs **~100+ gradient exposures** to lock in (150 exp → 0.93 retention). `degrees_to_radians` got 22 exposures → ~0.07 dense / 0.12 PKM retention — and the model still emits the inverted formula. Rare coding algorithms are structurally under-exposed.
- **PKM's actual edge: 1.7–2.2× more sample-efficient than dense at low exposure** (acquires rare facts faster). Real but bounded — everything still needs many exposures.
- **Catastrophic forgetting / stability-plasticity.** PKM is read at every token via *shared learned addressing*, so storing a new fact re-routes all others → it forgets as badly as dense (continue-train on 40 facts: solved-control 30→6). Freeze addressing → no forgetting AND no learning. **PKM does NOT escape the dilemma** because it isn't an interference-free growable store. See [[thinking-on-code-verdict]].

## Implications
- PKM earns its place in the stack. Keep the v7.1 bootstrap settings (see [[product-key-memory]]).
- The "huge benefit from memory" intuition is partly right (faster tail acquisition) but bounded. The architecture direction for a real leap is a **growable sparse / non-parametric (kNN) store** with per-fact dedicated slots — append, no gradient, no interference. See [[open-questions-and-roadmap]].
- Cheap real levers that exploit the insight: tail up-sampling past ~100 exposures, SFT data cleaning (~6 % broken / ~70 % unverified targets), route tail facts through PKM.

## Related
[[product-key-memory]] · [[objective-function-alignment]] · [[thinking-on-code-verdict]] · [[mechanism-verdicts-overview]] · #verdict
