# Principle: code is recall, not iteration

## Summary
**Per-token code prediction is mostly recall/pattern, not iterated computation.** This is why [[latent-thinking]] — a depth-extender for iterated computation — is marginal on code: only ~10 % of code positions benefit from a latent burst, by ~+0.3 logp, and crucially **FLAT across R=1/2/4/8** (no depth payoff). Contrast iterated-computation tasks (arithmetic/pointer-chase) where exact-R unlocks +0.65–0.80. A Coconut-style latent step only re-iterates the trunk's own computation; it cannot FETCH a fact the trunk doesn't already have in-state. Source: `project_code_thinking_ceiling.md`, `WHY_THINKING_MARGINAL_ON_CODE.md`.

## The numbers
- Fraction of code positions a latent burst helps / mean Δlogp where it helps: R=1 → 5.7 %/+0.38, R=2 → 11.8 %/+0.25, R=4 → 8.7 %/+0.32, R=8 → 8.0 %/+0.34. **~10 %, ~+0.3, flat in R.**
- Even an oracle gate at a 25 %-think budget captures only ~+0.11 mean Δlogp.
- Arithmetic chains, by contrast, scale strongly with R (the whole point of latent thinking).

## Why the gate can't fix it
- `corr(P_think, Δlogp) ≈ −0.10` at every R — the gate is **anti-aimed** (fires think where thinking hurts).
- A gate-head-only BCE calibration did NOT flip it (−0.10 → −0.17): "will thinking help here" is **not linearly decodable from `out_norm(h_t)`**. The label is mostly coin-flips at the margin, AND the value is *recall* (a property of what's retrievable), not a property of the current hidden. See [[thinking-gate]].

## Implications
- **Don't expect a general MBPP/HumanEval lift from depth-thinking** — the addressable surface is ~0 there. Stop spending inference-mechanism effort on it.
- The fix direction for code is **retrieval (PKM/WM)**, not more iteration depth — but even that is bounded by knowledge/under-exposure ([[thinking-on-code-verdict]]).
- Reserve latent thinking for **matched bottlenecks**: multi-step computation, long dependency chains, execution tracing — where it IS load-bearing ([[latent-thinking-verdict]]).
- Where code DOES decompose into a chain of cheap *uniform* steps, latent thinking helps (supplies depth). It breaks on per-step op *variety* (heterogeneity) — see [[latent-thinking]].

## Related
[[latent-thinking-verdict]] · [[thinking-gate]] · [[thinking-on-code-verdict]] · [[objective-function-alignment]] · #principle #thinking
