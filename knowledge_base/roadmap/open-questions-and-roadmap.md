# Open questions & roadmap

## Summary
The live questions and the strategic fork. The single biggest open empirical question: **does [[working-memory]] become load-bearing on AGENTIC recall** (v14, running on GPU0) — the one regime with real headroom that the recurrence can't cover. Beyond that, the strategic decision is where to point effort: the **code headline** (base scale + RL), the **agentic/long-context** capability (where WM/latent are matched), or a **unified base** carrying all mechanisms. Sources: `run_v14_autoresume.sh`, the `project_*` memory files, [[mechanism-verdicts-overview]].

## The live question: does WM help agentic?
- **v14** = WM-recall continuation from v12 with the validated **embedding-key addressing + copy head + answer-span `mem_read_mask`**. The code-recall probe has **no headroom** (recurrence at 100 %), so usefulness must be measured on **`data/agentic_recall_heldout.jsonl`** (baseline ~0.42, a real capacity wall) once the WM features train (~0.5–1 B tokens), via `eval_code_recall.py` arms {on, full_off, no_think}. WM is load-bearing iff `on ≫ full_off` and `≫ no_think`.
- **Risk**: even validated, WM's payoff regime (capacity-exceeding, non-memorizable, multi-key recall) is rare in code; PKM + recurrence + latent may cover the realistic code workload, making WM droppable inert weight for the *code* model. Its niche is agentic/long-context.
- If v14 is negative on agentic too, the honest call is to **drop WM from the code stack** (keep it only if the agentic direction is pursued). A fundamentally different addressing scheme (discrete/VQ keys over the identifier span) would be a separate research bet, not a v14 bake.

## The strategic fork
1. **Code headline** — the only levers that buy *composition* are base capability (more params / more+cleaner data) and execution-grounded RL. Concrete: pure-code SFT → rejection-sampling self-distillation → DPO → grader-RL (v2 reached 16/164). Plus the cheap knowledge levers: tail up-sampling past ~100 exposures, SFT data cleaning (~6 % broken / ~70 % unverified), route tail facts through PKM. See [[thinking-on-code-verdict]], [[humaneval-trajectory]].
2. **Agentic / long-context** — the matched bottleneck for WM (recall) and latent thinking (depth). Build/borrow a long-context-code or agentic eval and *show* the mechanisms load-bearing there (v14 is the first step). This is where the "small super-coder with memory" thesis can actually win.
3. **Unified base** — carry PKM + WM + latent all live. Blocker: day-1 latent **competes with the PKM α-bootstrap** in the 0–3000-step window. Fix = stagger latent past the PKM commit window and/or strengthen the PKM floor; v12 owns PKM, v13 owns latent, neither owns both. See [[pretrain-run-history]].

## Architectural bets worth considering
- **Growable/non-parametric memory** — the current PKM/WM forget like dense weights (shared addressing). An interference-free per-fact-slot or kNN store (append, no gradient) is the concrete direction for a real memory leap — but kNN only helps near-exact matches, not novel composition. See [[pkm-verdict]], [[thinking-on-code-verdict]].
- **Discrete/VQ-key associative head** — a delta-rule layer fed one-hot/VQ keys = content-addressable recall with zero cross-talk, reusing the FLA kernel. The principled fix for [[key-separability]] if WM recall is revived.
- **More state / hybrid attention** — the residual reasoning↔code trade-off and the recall capacity wall are ultimately capacity limits; bigger state / a small think-time attention would escape them (architecture, not training).

## Deferred / known-incomplete
- Per-token incremental decode path for `state_readonly_at_think` (β unmasked in decode).
- RL replay-packing passes `doc_ids=None` ([[cross-document-isolation]]).
- Latent-thinking HumanEval magnitude needs temperature-sampled confirmation through `eval_humaneval.evaluate()`.
- `torch.compile` + aux extra-forwards (Inductor crash) — currently `--no-compile` on the aux-loss runs.

## Related
[[mechanism-verdicts-overview]] · [[working-memory-recall-saga]] · [[thinking-on-code-verdict]] · [[pretrain-run-history]] · [[small-super-coder-goal]] · #roadmap
