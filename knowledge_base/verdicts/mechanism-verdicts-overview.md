# Mechanism verdicts — overview

## Summary
The payoff of the whole investigation: a clear, numbers-backed verdict on which side mechanisms are load-bearing and why. The unifying principle: **a mechanism helps exactly to the degree its training objective demanded its function AND the deployment workload actually has that bottleneck** (see [[objective-function-alignment]]). PKM wins because facts are everywhere in the LM loss; WM and latent thinking are load-bearing only on their matched bottlenecks (recall saturation, iterated depth), which short knowledge-bound coding benchmarks rarely stress. Source: `WHY_THINKING_MARGINAL_ON_CODE.md`, the `project_*` memory files.

## The one table
| mechanism | its function | training demanded it? | deployment bottleneck? | verdict |
|---|---|---|---|---|
| **[[pkm-verdict]]** (parametric store) | store/recall facts | **YES** — facts reduce next-token CE everywhere | **YES** — 287 M is knowledge-bound | **load-bearing** (pkm_off −5/−50 % on HumanEval; +0.06–0.09 per-source CE) ✓ |
| **[[working-memory-recall-saga]]** (content recall) | retrieve a specific bound value | NO — general text needs only recency (recurrence gives it) | rarely (real code seldom holds many live competing bindings) | **inert on code recall** — read collapses to recency; **niche = saturating multi-key / agentic** ✗/niche |
| **[[latent-thinking-verdict]]** (sequential compute) | multi-hop iterated reasoning | PARTIALLY — only on the reasoning co-train | rarely (per-token code is recall, not iteration) | **works on depth (+0.45–0.80), marginal on code** ~ |

## The two mechanisms behind the verdicts
1. **[[route-around-principle]]** — gradient descent takes the primary path (trunk/recurrence) whenever it can solve the task, leaving the side mechanism idle. A mechanism only becomes load-bearing when the primary path provably CANNOT do the task (capacity-exceeding AND non-memorizable).
2. **[[objective-function-alignment]]** — co-training is necessary but not sufficient: the co-training *loss* must contain the bottleneck. v10 co-trained WM for 7 B tokens and it still went recency, because the pretrain loss never required content recall.

## The blunt bottom line for the coding headline
At 287 M, MBPP/HumanEval are **knowledge/composition-bound** ([[thinking-on-code-verdict]], [[code-is-recall-not-iteration]]). None of these inference mechanisms supplies missing knowledge. They are each real on their matched probe; the coding headline needs base capability (params + clean data) + execution-grounded RL.

## Related
[[pkm-verdict]] · [[working-memory-recall-saga]] · [[latent-thinking-verdict]] · [[thinking-on-code-verdict]] · [[route-around-principle]] · [[objective-function-alignment]] · #verdict
