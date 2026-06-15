# Knowledge Base — Small Super-Coder

## Summary
This is an Obsidian-style vault distilling everything the project has learned building a **small (~287 M) code model that punches above its weight** on coding tasks under a 2× RTX 5090 budget. The highest-value notes are the **Mechanism verdicts** (which architectural pieces actually earn their place, and why) and the **Cross-cutting principles** (the reusable methodology lessons) — both sections below. Every note starts with a `## Summary`; links are Obsidian wiki-style `[[note-name]]` references (they resolve by filename, so folders don't matter).

> Read order for a newcomer: [[small-super-coder-goal]] → [[mechanism-verdicts-overview]] → [[route-around-principle]] → [[objective-function-alignment]] → then dive into whichever mechanism you're touching.

---

## Goal & framing
- [[small-super-coder-goal]] — the north star: a deployable small code model competitive per-parameter on HumanEval/MBPP.
- [[thesis]] — the bet: methodology is a bigger lever at small scale than the field's research portfolio reflects.
- [[hardware-and-compute]] — 2× RTX 5090 (sm_120, 32 GB, no NVLink); the Blackwell/FLA gotchas and token budgets.

## Architecture stack
One note per mechanism. What it is, how it's wired, the validated settings.
- [[deltanet-backbone]] — bounded-state linear-RNN backbone (plain `deltanet`; gated form broken on sm_120).
- [[film-feedback]] — sparse late→early FiLM modulation; the headline −3–5 % PPL lift + K=3 self-feed.
- [[shallow-wide-trunk]] — 10L×896d + dense FiLM; iso-param, ~18 % faster than the 30L trunk.
- [[product-key-memory]] — 262 k-slot parametric KV store + the v7.1 bootstrap-fix package.
- [[working-memory]] — bounded write-gated buffer with decoupled-KV addressing; the read-α gate.
- [[latent-thinking]] — Coconut-style state-readonly hidden-feedback "thinking"; the depth-matched co-train recipe.
- [[thinking-gate]] — the per-position emit/think gate; calibration, temperature-fragility.
- [[cross-document-isolation]] — `cu_seqlens` from `doc_ids` so packed docs don't leak state.
- [[training-and-optim-knobs]] — bf16/tf32, WSD schedule, WD=0.01, activation checkpointing, grad-accum, Muon+AdamW, the gate-floor trap.

## Mechanism verdicts
The heart of the vault — the empirical verdict on whether each mechanism is load-bearing, with numbers and the "why".
- [[mechanism-verdicts-overview]] — the one-table summary + the unifying principle.
- [[pkm-verdict]] — **works** for facts (+0.06–0.09 per-source CE; −5 HumanEval ablation); bounded by under-exposure & forgetting.
- [[latent-thinking-verdict]] — **works on DEPTH** (homogeneous iterated computation, +0.45–0.80); ~nothing on code generation.
- [[working-memory-recall-saga]] — the full arc: gradient-disconnect → key-separability root cause → name-key+copy-readout fix (100 % synthetic) → but real code recall is solved by the recurrence, so WM's only real headroom is agentic/long-context.
- [[thinking-on-code-verdict]] — why thinking is marginal on MBPP/HumanEval; the knowledge/under-exposure bottleneck; the selective-thinking lift (7→11).

## Cross-cutting principles
The reusable wisdom — read these before designing any new mechanism or experiment.
- [[route-around-principle]] — a mechanism stays idle unless the task forces the primary path to FAIL **and** the gradient reaches it.
- [[objective-function-alignment]] — a mechanism helps iff training demanded its function AND deployment has that bottleneck.
- [[key-separability]] — cosine addressing needs separable keys; near-identical keys collapse to recency.
- [[code-is-recall-not-iteration]] — per-token code is recall, not iterated computation; why depth-thinking is marginal on code.
- [[fair-baselines]] — always run the control that gives the baseline equal opportunity; multiple over-claims were caught this way.
- [[pareto-safe-thinking]] — get thinking's upside without downside: structural floor + verifier best-of.
- [[broken-probe-lessons]] — training-matched formats, max-of-K controls, de-leaking datastores; probes that inflate/deflate.

## Experimental history
- [[pretrain-run-history]] — v4 → v14: what each run was and what it produced.
- [[recall-investigation-arc]] — the WM-recall saga as a chronological investigation.
- [[latent-thinking-arc]] — discrete-token (failed) → latent (works on depth) → real-model port → code.
- [[humaneval-trajectory]] — 8 → 10 → 14 → 16/164, and why 16 is the current best.

## Tooling & how-to
- [[evals-and-probes]] — the key evals & diagnostic probes, what each measures.
- [[data-generators]] — the synthetic task/recall generators.
- [[launchers-and-metrics-to-watch]] — the autoresume launchers + the live metrics that tell you a run is healthy.

## Open questions & roadmap
- [[open-questions-and-roadmap]] — does WM help AGENTIC (v14 pending)? the strategic fork (code-headline vs agentic vs unified base); deferred items.
