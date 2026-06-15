# Principle: broken-probe lessons

## Summary
Probes lie in both directions — they can manufacture a "recall=0", inflate a think_rate, or hide a working mechanism behind a parsing bug. Several major conclusions in this project were **eval/probe artifacts** that flipped once the probe was fixed. The standing rules: **measure in the training-matched format, run the matched control, de-leak datastores, and parse the answer the way training optimized it.** Sources: `project_wm_recall_probe_broken_and_routed_around.md`, `project_latent_thinking_real_model.md` (eval-bug correction), `feedback_fair_baselines.md`.

## The artifacts that were caught
1. **Format-OOD probe manufactured "recall = 0".** The in-run recall probe used the SFT `# {flattened one-line program}` format, which is OOD for a PRETRAIN ckpt → the same v12 1 B ckpt read **0.000** (broken) vs **43.8 %** (training-matched). Worse, the broken format also **inflated think_frac and manufactured a spurious positive WM delta** in the near-zero regime. Fix: `--prompt_format training_matched`. **Lesson: always re-measure on the training-matched format before believing a probe.**
2. **Digit-run parsing turned correct deep traces into "collapse".** `latent_arith_real` scored the answer with "first contiguous digit run" of the full decode; the model emits the correct answer token then keeps emitting digits → "5" + "19999…" parsed as 519999 → wrong. This produced the entirely-wrong "latent thinking can't do heterogeneous reasoning" conclusion. Fix: read the answer from the FIRST digit-bearing emitted token only (= the last think-slot argmax = what per-hop training optimizes). **Lesson: parse the answer the way training optimized it.**
3. **`force_prefix_think=0` means "GATE DECIDES", not "no think".** With a trained gate the "none" column silently becomes the autonomous column. For a TRUE no-think baseline pass `emit_threshold=0.0`. **Tell: if `none == R=n` exactly, your no-think arm is contaminated.**
4. **Max-of-K / oracle inflation** and **datastore leakage** — see [[fair-baselines]].
5. **Broken downstream harness** — MBPP `eval_broader` scored 0/200 (baseline) due to a format bug; an ad-hoc flip script reproduced 0/80 vs the canonical 8/164. **Lesson: trust the canonical `eval_humaneval.evaluate()`; a confirmation eval must reuse it verbatim.**

## How to apply
- Probe in the **same distribution/format the ckpt was trained on**; an OOD probe is uninformative at best and actively misleading at worst.
- Always include a **matched control arm** (true no-think; random max-of-K; de-leaked datastore).
- A delta on a **0 %-scored task is trivially 0** — fix the task to have headroom before reading the delta ([[working-memory-recall-saga]], single-binding recall has no headroom).
- When a result seems too catastrophic (0/164) or too clean (100 %), suspect the probe first.

## Related
[[fair-baselines]] · [[working-memory-recall-saga]] · [[latent-thinking-verdict]] · [[evals-and-probes]] · #principle #methodology
