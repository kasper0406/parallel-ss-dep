# Thesis — methodology is under-claimed at small scale

## Summary
The bet: most training-methodology innovations of the last few years (curated synthetic data, execution/RL rewards, distillation, architectural lifts, working memory + gated computation) were invented and benchmarked at frontier scale, where a single technique's effect is squeezed between massive base capacity and noisy evals. **At ≤1 B params the same techniques compound visibly** because the base is weak enough that genuine signal moves the benchmark — so there is unclaimed per-parameter performance there. Source: `THESIS.md`. See [[small-super-coder-goal]].

## The claim, stated carefully
> Holding compute roughly fixed, **methodology** (data quality + curation, training stages, RL signal design, architecture refinements) is a much larger lever at small scale than the field's current research portfolio reflects.

This is NOT "scaling laws are wrong" and NOT "small dense nets match large ones." It is narrower: combining small-scale-validated methods in the right order can land **above the published per-parameter frontier** — not because the field doesn't know how, but because it hasn't combined the methods at this scale.

## Supporting evidence cited
- **Phi-1** (1.3 B, ~50 % HumanEval, ~7 B synthetic tokens) — data quality, not architecture.
- **Qwen2.5-Coder-0.5 B** (~31 % HumanEval) — methodology (RL + data) over scale.
- **DeepSeek-R1-Distill-Qwen-1.5 B** — distillation lifts a small base into competitive territory.
- **Internal**: [[working-memory]] +11 pp recall on saturated MQAR — an architectural lift that only *shows up* at small/saturated scale.

## What it explicitly does NOT support (honesty clauses)
- Frontier-class models are not 4 B-active MoEs; the very top still tracks scale. The bet narrows the per-parameter gap, doesn't eliminate it.
- "Lottery ticket" results don't imply small dense nets match large ones on every task — citing them as general justification is overreach.
- On hard multi-step reasoning / long-context multi-file code, the per-parameter gap *grows* with difficulty.

## Commitments
- **Eval at small scale or don't claim the result.** A 70 B effect is not evidence at 217 M.
- **Compound methods, don't single-shot them.**
- **Be honest about the gap to frontier.** "Per-parameter SOTA on benchmark X" is defensible; "small = large" is not.

## Related
[[small-super-coder-goal]] · [[fair-baselines]] · [[mechanism-verdicts-overview]] · #framing
