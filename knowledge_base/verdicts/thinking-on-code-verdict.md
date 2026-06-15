# Verdict: why thinking/memory is marginal on code

## Summary
The intuition is correct that a small model *should* gain hugely from parametric memory, context recall, and extra compute. The reality on MBPP/HumanEval at 287 M is a marginal +1/+3. The investigation found **why, and it is not that the mechanisms are broken**: short coding benchmarks are bottlenecked by **base knowledge / composition** — something none of the inference mechanisms address. The mechanisms ARE load-bearing where the bottleneck matches their function; MBPP is just the wrong probe. Source: `WHY_THINKING_MARGINAL_ON_CODE.md`, `project_code_thinking_ceiling.md`.

## The chain of evidence
1. **Per-token thinking ceiling is low and depth-independent**: a latent burst helps only ~10–13 % of code positions, by ~+0.3 logp, **FLAT across R=1/2/4/8** (arithmetic chains, by contrast, scale strongly with R). → code-token prediction is mostly recall/pattern. See [[code-is-recall-not-iteration]].
2. **No addressable retrieval content for code (WM ruled out cheaply)**: oracle best-of-16 buffer retrieval scored 0.249 vs a matched-norm **random** control 0.298 (random HIGHER) → oracle−random ≈ 0, the "oracle lift" was pure max-of-K selection noise. Always run the random max-of-K control. PKM is the only positive channel (+0.052, modest, already exploited).
3. **Failures are runnable but fundamentally wrong**: 87 % of failures RUN but pass ~6 % of tests — plausible-but-wrong algorithms (`degrees_to_radians` uses 180/π; `longest_chain` infinite-recurses), not off-by-one. Plus a non-trivial rate of **bad tests** (benchmark undercounts).
4. **Failures are KNOWLEDGE-bound, predominantly not search-bound**: the model rates its own wrong answer more likely than the gold. (The strong form — "0/93 search-bound, 4.6×" — was later **corrected** as a greedy-argmax artifact: gold is 62 % top-1 / 82 % top-5 reachable, pass@8 recovers ~9 %. So there is a ~10–15 % search-recoverable tail; failures are *predominantly* not *exclusively* knowledge-bound. See [[fair-baselines]].)

## Why it didn't learn it in training (the deeper layers)
- **Under-exposure, not capacity**: a fact needs ~100+ exposures; rare algorithms get ~22. PKM is 1.7–2.2× more sample-efficient but everything still needs many exposures. See [[pkm-verdict]].
- **Catastrophic forgetting / stability-plasticity**: cramming 40 clean facts fixes those problems (0→18) but destroys ~22/30 previously-solved (full FT AND PKM). Learning needs the same addressing plasticity that causes forgetting. (Recipe-bound, not fully fundamental: replay+KL halves forgetting; the clean fix is a **full retrain on combined corpus**, not continue-training.)
- **Non-parametric kNN-LM** clears forgetting + consumption (oracle 0→23/40) **but only for near-exact matches** — de-leaked realistic store collapses to 1/40. Retrieval-as-memorization-at-scale, not compose-from-similar. (The apparent realistic win was **leakage** — magicoder/codefeedback contain the exact MBPP problems; always de-leak datastores — see [[broken-probe-lessons]].)

## The defensible conclusion
On short, knowledge-bound MBPP code-gen at this base, **inference-time depth/recall/retrieval have little addressable surface**. Each mechanism is load-bearing on its matched bottleneck (WM +11 pp MQAR, latent +0.65–0.80 arithmetic, PKM −5 HumanEval). The honest levers for real coding gains:
1. **Base capability** (more params / more+cleaner data) — the only thing that buys composition.
2. **Datastore covering the task distribution** — works for problems resembling seen ones, not novel ones.
3. **Reserve latent thinking / WM for their matched bottlenecks** (multi-step computation, long-context/agentic recall).

The one real HumanEval thinking lift (7→11) came from **selective routing**, not depth — see [[humaneval-trajectory]].

## Related
[[code-is-recall-not-iteration]] · [[pkm-verdict]] · [[latent-thinking-verdict]] · [[fair-baselines]] · [[mechanism-verdicts-overview]] · #verdict
