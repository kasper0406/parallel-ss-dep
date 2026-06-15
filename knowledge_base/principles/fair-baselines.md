# Principle: fair baselines / run the control

## Summary
**Before quoting "X helps by +Y", run the control that gives the baseline equal opportunity — and lead with that fair comparison.** This is a standing user requirement; multiple over-claims have been caught and the honest fair-baseline result was usually *stronger* for surviving the challenge. Source: `feedback_fair_baselines.md`, plus repeated catches across the latent-thinking and WM arcs.

## Over-claims that were caught (learn from these)
1. **Within-model ablation undertrains the off-path.** The "latent thinking lifts n=2 from 0.2 → 1.0" was partly an undertraining artifact: a dedicated **100 %-no-think control** showed a single forward CAN learn 2-hop chase (0.94) when fully trained. The real defensible lift was at n≥3. → **Always run the dedicated 100 %-no-think control before quoting a latent lift.**
2. **Max-of-K selection noise masquerades as an oracle lift.** "Oracle retrieval helps +0.6" collapsed when compared to a **matched-norm RANDOM-injection** control (oracle 0.249 vs random 0.298, random HIGHER). oracle−random ≈ 0. → **Always run the random max-of-K control on any oracle/best-of-K probe.**
3. **Datastore leakage.** kNN-LM "realistic 11/40" was pure leakage — magicoder/codefeedback contain the exact MBPP problems. De-leaked → 1/40. → **Always de-leak retrieval datastores.**
4. **Greedy-argmax artifact.** "0/93 search-bound, model rates wrong answer 4.6× over gold" measured `logp_own` on the model's own argmax decode (near per-token max by construction). Proper rank test: gold 62 % top-1 / 82 % top-5, pass@8 recovers ~9 %. The binary claim outran the data. → **Don't compare logp of an arbitrary alternative against the model's own greedy output.**
5. **Broken eval format inflated/deflated a result** — see [[broken-probe-lessons]].

## How to apply
- Ask: "is the baseline trained as hard as the treatment? Same data, same steps, same opportunity?" If the off-path got fewer gradient steps, run a dedicated control that gives it maximum opportunity, and report the lift vs THAT.
- State caveats proactively (what's matched, what isn't). Inference compute can't be matched for thinking — that's fine, it's the mechanism; say so.
- A modest greedy lift (+3–4 on 164 problems) is within the ±2–3 sample-noise band — confirm with temperature-sampled pass@1 before quoting it as a headline.
- Don't dress up an inflated or net-negative result as a win. The rigorous version is usually the more convincing story anyway.

## Related
[[broken-probe-lessons]] · [[latent-thinking-verdict]] · [[thinking-on-code-verdict]] · #principle #methodology
