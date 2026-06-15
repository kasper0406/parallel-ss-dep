# Evals & diagnostic probes

## Summary
The reusable evals and probes, grouped by what they measure. The recurring discipline: **use the matched format, run the matched control, and pick the probe whose bottleneck matches the mechanism** ([[broken-probe-lessons]], [[fair-baselines]]). All paths are `experiments/*.py`. The CLAUDE.md per-source PKM probes (`probe_pkm_per_source`, `probe_pkm_contribution`) are referenced historically but the present-on-disk PKM probes are `probe_pkm_capacity.py` and `probe_v5_pkm_utilization.py`.

## Capability evals
- `eval_humaneval.py` — the **canonical** HumanEval harness. For distilled ckpts use `--prompt_style sft_comment --extract_code_block --min_emit_before_eos 30`. Latent thinking via `--generator latent_think --emit_threshold <t>`. `--diagnose_json` dumps per-problem tier/think-count. **Trust this harness; confirmation evals must reuse `evaluate()` verbatim.**
- `eval_code_recall.py` — realistic code/agentic recall (arms `on`/`full_off`/`no_think`, modes `teacher_forced`/`generate`, dist×N cross-tab). The probe for [[working-memory]] usefulness on real structure.
- `eval_longctx_recall.py` — long-context single/multi-binding recall by distance bucket. `--prompt_format {comment_oneline, training_matched}` (use training_matched for pretrain ckpts!). `--wm_ablate {mean, full_off, coop_off}`, `force_prefix_think`.
- `latent_arith_real.py --eval_only` — the depth probe: `none` (true no-think, `emit_threshold=0`) vs `R=n` (depth-matched latent) vs `--autonomous_halt`. **The canonical "does latent thinking help" eval** (success = net-positive vs the fair no-think control, lift grows with depth). See [[latent-thinking-verdict]].
- `train_mqar.py` / `tasks/mqar.py` — the canonical "does this help recall" synthetic. Use T=256/K=32 for architectural ablations; T=512/K=128 for the memory-vs-no-memory saturation regime. **Run MQAR in fp32** (bf16 collapses sparse loss).

## Mechanism / addressing probes
- `wm_namekey_probe.py` — addressing: name (input-embedding) key vs cosine-on-hidden (top1=1.00 vs chance). The probe that solved [[key-separability]].
- `wm_multitok_readout.py` — copy/pointer multi-token readout (EXACT 1.00 at N=48/64/96).
- `probe_wm_recall_addressing.py` — write/read/readout localization at a forced think (mass-on-binding).
- `probe_wm_utilization.py` — WM read-hit-rate / attention concentration / coverage (the MQAR variance diagnosis).
- `probe_v5_pkm_utilization.py` — PKM row-norm drift, slot concentration, residual contribution (found v5-pkm 97 % dead).
- `probe_pkm_capacity.py` — param-matched dense-vs-PKM fact memorization (capacity vs exposure).

## Thinking / ceiling probes
- `probe_gate_placement.py` — per-position think ceiling: frac-helpful, mean Δlogp, corr(P_think, Δlogp), oracle headroom. Found the ~10%/flat-in-R code ceiling.
- `probe_retrieval_channels.py` — PKM vs WM channel decomposition (+ no-train cosine).
- `probe_oracle_retrieval.py` — oracle-vs-**random** retrieval upper bound (the max-of-K control). Always run the random arm.
- `probe_thinking_passk.py` — pass@K solvable-set diff (thinking-unique solves).
- `probe_overstep.py` / `probe_stability_fix.py` — the over-step cliff + fixed-point fix.

## Code-bottleneck probes
- `probe_knowledge_vs_search.py` — knowledge- vs search-bound failures (mind the greedy-argmax artifact, [[fair-baselines]]).
- `probe_failure_bottleneck.py` — failure tiers (runnable-but-wrong).
- `probe_exposure_lever.py` — exposure / forgetting (modes `full`/`pkm`/`pkmval`, + replay/KL).
- `probe_knn_oracle.py` — non-parametric kNN-LM (modes `oracle`/`realistic`/`corpus_clean`; **de-leak the datastore**).

## Architecture diagnostics
- `diag_ckpt.py` — per-layer logit-lens CE, hidden rank, ||h||, per-source CE. Found the WD=0.1 residual-stream collapse. `diag_reference_lm.py` = the SmolLM2-135M matched reference.
- `best_of_think.py` — verifier best-of arbiter (strictly ≥ no-think). See [[pareto-safe-thinking]].

## Related
[[broken-probe-lessons]] · [[fair-baselines]] · [[data-generators]] · [[launchers-and-metrics-to-watch]] · #tooling
