# Principle: objective-function alignment

## Summary
**A mechanism helps exactly to the degree its TRAINING OBJECTIVE demanded its FUNCTION — and the deployment workload actually has that bottleneck.** This is the unifying principle behind every mechanism verdict. It is a *stronger* claim than "co-train the modules": co-training is necessary but not sufficient, because the co-training **loss** must itself contain the bottleneck the mechanism relieves. Source: `WHY_THINKING_MARGINAL_ON_CODE.md` (unifying synthesis), `project_wm_addressing_root_cause.md`.

## The decision table
| mechanism | function | training demanded it? | deployment bottleneck? | result |
|---|---|---|---|---|
| **PKM** | store/recall facts | YES — facts cut next-token CE everywhere | YES — model is knowledge-bound | **load-bearing** ✓ |
| **WM** | content recall | NO — general text needs only recency | rarely | **inert** (read → recency) ✗ |
| **latent thinking** | iterated compute | PARTIALLY — only on the reasoning co-train | rarely (code is recall) | **marginal** ~ |

## Why PKM is the exception
PKM works out-of-the-box because the ordinary LM loss **automatically creates its bottleneck** — facts are everywhere in text, so reducing CE *requires* storing/recalling them. WM's bottleneck (many-simultaneous-binding recall) and latent thinking's (multi-hop iteration) are rare in general text, so the loss never pressured the trunk to build the representations those mechanisms need. Measured: WM's read is recency/positional, never content-addressable.

## The actionable corollary
To make WM or latent thinking load-bearing, **the bottleneck must exist in TRAINING** — bake tasks whose function IS the mechanism into the pretrain loss:
- WM → capacity-exceeding, non-memorizable multi-key recall (multibind/MQAR) with the read **driven at the query position via `mem_read_mask`** and the loss **at that position**.
- latent thinking → depth-requiring chains (pointer-chase) with a depth curriculum, co-trained from day 1.

And it must satisfy [[route-around-principle]]: even with the right loss, if the trunk can fit the data another way, gradient routes around the mechanism. The loss must make the mechanism the *only* low-loss path.

## Why the user's intuition AND the result both hold
"Memory should be a HUGE benefit" is true — on a workload bottlenecked on memory/iteration. The actual workload (general code at 287 M) is bottlenecked on KNOWLEDGE, which PKM already covers. So WM/latent's marginal benefit there is *correct given this workload*, not a failure of the idea.

## Related
[[route-around-principle]] · [[mechanism-verdicts-overview]] · [[key-separability]] · [[code-is-recall-not-iteration]] · #principle
