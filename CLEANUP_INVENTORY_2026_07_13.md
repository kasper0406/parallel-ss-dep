# Repo Cleanup Inventory — 2026-07-13 (read-only classification pass)

**Scope**: `experiments/*.py` (245 files incl. `__init__.py`, + `experiments/tasks/`), repo-root `*.sh` (70) +
`experiments/*.sh` (7), repo-root `*.md` (38), `configs/*.yaml` (13). `checkpoints/`, `data/`, `runs/` excluded
per instructions. **No file was modified; this report is the only write.**

**Method**: (a) full cross-reference map — every module grepped for importers/launcher/doc references;
(b) static AST scan of every `experiments.*` import (result: **zero unresolved imports repo-wide**);
(c) git last-touch dates (note: commit `691dd7d` "backup sweep" 2026-07-02 stamps ~51 files that are really
June-20→July-2 work); (d) pytest baseline: **1067 tests collected, 0 collection errors**
(`PYTHONPATH=. .venv/bin/python -m pytest experiments/ --collect-only -q`); (e) full argparse-consumer audit of
`train_lm_args.py` (218 dests), `sft_code.py` (41), `train_rl_grader.py` (69); (f) every launcher read and its
scripts/flags/configs verified to exist.

**Headline**: the repo is in much better shape than the "2 months of fast churn" prior suggests. All previously
deprecated mechanisms (`process_reward`, FIX A `mem_write_only_at_think`, corpus-RAG, the no-op SFT flags,
`launch_pretrain_v2_thinking.sh`) are **confirmed fully removed** — nothing in the tree references them (the only
hits are two stale comments in `speed_knobs.py`/`test_speed_knobs.py`). **Zero dead CLI flags. Zero broken
imports. 13 DEAD files total**, all launchers/wrappers except one superseded generator.

---

## Summary counts

| category | experiments/*.py | tests (test_*.py) | *.sh | *.md | configs/ | total |
|---|---|---|---|---|---|---|
| ACTIVE | 58 (19 core + `__init__` + 38 one-off) + 8 `tasks/` | 69 | 14 | 14 | 4 | 167 |
| KEEP-HISTORICAL | 97 | 20 | 51 | 24 | 9 | 201 |
| DEAD | 1 | 0 | 12 | 0 | 0 | 13 |
| DEAD-FLAGS | — | — | — | — | — | **0** |

---

## ACTIVE

### Core modules (the train_lm / model / eval stack — all heavily imported)

| file | reason |
|---|---|
| experiments/model.py, layers.py | TinyLM + attention/DeltaNet blocks; imported by ~55 files |
| experiments/model_builder.py, build_arch.py | model construction from args; train_lm + evals |
| experiments/train_lm.py, train_lm_args.py | the pretrain trainer + CLI (touched 2026-07-13) |
| experiments/optim_utils.py, bf16_optim.py, speed_knobs.py | optimizer/scheduler/bf16/compile stack |
| experiments/soap.py, embed_optim.py, exp_deltanet_precond_optim.py, exp_deltanet_precond_fused.py | live `optim_utils.build_optimizer` branches (`matrix_optimizer`/`embed_optimizer`); the fused-NS branch is the production `--matrix_optimizer fused_deltanet_ns` used by v19 |
| experiments/data_mix.py | mixed-corpus streaming; every pretrain launcher |
| experiments/memory_layer.py | PKM; imported by model.py |
| experiments/thinking.py, gate_calibration.py, gist_loss.py, aux_brackets.py, curriculum.py | aux losses / gate / latent primitive, all imported by train_lm or train_rl_grader |
| experiments/code_grader.py | execution-grounded grader; ~24 importers |
| experiments/sft_code.py, train_rl_grader.py | SFT + grader-RL trainers |
| experiments/iterative_repair.py, rl_multiturn.py | top-level imports of train_rl_grader |
| experiments/teacher_logits_io.py | KD logit store I/O; imported by train_lm |
| experiments/train_mqar.py + experiments/tasks/ (8 files) | canonical synthetic-recall validation mandate (AGENTS.md) |
| experiments/__init__.py | package marker |

### Active one-off / program scripts (38, per-file evidence in the sweep)

| file | program |
|---|---|
| gen_exec_traces.py, eval_exec_trace_latent.py, eval_exec_trace_latent_trace.py, eval_exec_trace_text.py, eval_stage_a_killgate.py, latent_reasoning_cotrain.py, bench_latent_reasoning_batched.py | exec-trace latent (Stage A/B), EXEC_TRACE_LATENT_PLAN.md, commits 2026-07-04..13 |
| eval_cruxeval_transfer.py | CRUXEval-O transfer probe (2026-07-13) |
| gen_repo_episodes.py, eval_repo_adaptive.py | meta-TTT P0 (2026-07-13) |
| gen_teacher_logits.py, gen_teacher_logits_vllm.py, distill_solutions.py | KD/distill pipeline (v19 program) |
| linearize_qwen.py, linearize_smollm2.py, linearize_hybrid_ablation.py, widen_mlp_ckpt.py | linearization pivot |
| eval_humaneval.py, eval_callback.py, eval_bracket_structure.py, eval_hf_humaneval.py, humaneval_solution_ce.py | eval harnesses (`build_model_from_ckpt` has ~30 importers) |
| eval_longctx_recall.py, ablate_memory_mechanisms.py, eval_code_recall.py, feature_probe.py, probe_humaneval.py | wired into train_lm mid-eval callbacks / each other |
| scoreboard_longctx_cost.py, decode_bench.py, flagship_recall_probe.py, flagship_recall_probe_gen.py | north-star cost/recall scoreboard |

### Active launchers (14)

| file | evidence |
|---|---|
| launch_pretrain_v19_codeup.sh | current mainline pretrain (HANDOFF; config exists; flags valid) |
| launch_gen_teacher_logits_v19.sh, launch_distill_v19_1b.sh | v19 KD steps 1+2 |
| launch_stageB_latent_trace.sh, launch_stageB_depthfix.sh | exec-trace Stage B (commits 2026-07-13; input ckpts exist) |
| launch_linearize_qwen.sh, launch_sft_qwen.sh, launch_rl_grader_qwen.sh, launch_gate_1b_qwen.sh | Qwen-donor pivot (STRATEGY_2026_06_28) |
| launch_hybrid_ablation.sh | linear-attn-tax fork |
| launch_feature_pilot_A.sh / _B.sh / _Bprime.sh, launch_feature_prewarm_phase0.sh | feature-pilot arms (SESSION_FINDINGS 2026-07-03; bases exist) |

### Active configs (4)

pretrain_mix_v19_codeup.yaml, pretrain_mix_stageA_executor.yaml, pretrain_mix_v18_arxiv.yaml (used by v19
gate/distill + phase1 launchers), pretrain_mix_feature_pilot.yaml.

### Active docs (14)

AGENTS.md (+ CLAUDE.md / GEMINI.md symlinks), README.md, HANDOFF.md, SESSION_FINDINGS.md, THESIS.md,
EXEC_TRACE_LATENT_PLAN.md, META_TTT_PLAN_2026_07_13.md, LITERATURE_LATENT_EXEC_2026_07_13.md,
IDEAS_2026_07_13.md, NORTH_STAR_2026_06_30.md, STRATEGY_2026_06_28.md, SCOREBOARD.md.

### Active tests (69)

All test files import cleanly (0 collection errors) and none tests removed code. The 69 not listed under
KEEP-HISTORICAL below pin core-module behavior (model/layers mechanics incl. think-adapter/soft-mixture/WM/state-readonly
regression guards, data_mix, cu_seqlens, graders, RL trainer, KD I/O, exec-trace program, repo-episodes,
CRUXEval, feature probes, engagement checks, optimizer wiring). Notable: `test_deltanet_precond_fused.py` and
`test_matrix_optimizer_wiring.py` guard the **production** fused-NS optimizer; `test_engagement_checks.py`
(53 tests) guards the mechanism-engagement kill-gates.

---

## KEEP-HISTORICAL

Concluded work whose findings are cited in docs/memory; cheap to keep, retained for reproducibility. Per-file
evidence was verified in the sweep; compact reasons here.

### One-off scripts (97)

| group | files | documented in |
|---|---|---|
| DeltaNet preconditioner study (drivers; winning classes live in optim_utils) | exp_deltanet_precond.py, _analyze.py, _bf16.py, _msagg.py; exp_deltanet_bf16_agg.py, _data.py, _gapsteps.py, _lm.py; precond_ab_analyze.py | DELTANET_PRECONDITIONER*.md |
| Optimizer benches | bench_optimizer.py, bench_analyze.py, bench_data_cache.py; embed_ab_analyze.py | AGENTS.md (keep Muon), EMBED_OPTIMIZER_AB.md |
| Batch/LR noise-scale study | grad_noise_scale.py, lr_sweep_fromscratch.py, noise_scale_data_cache.py | BATCH_LR_ANALYSIS.md |
| Depth-via-iteration / op-selector series | depth_via_iteration.py, depth_via_iteration_run.py, op_selector_depth.py, rope_selector_depth.py, sel_counter_depth.py, teachwean_selector_depth.py | memory `project_depth_via_iteration`, STRATEGY_2026_06_28 |
| Latent-thinking arc (concluded probes/trainers; live path is latent_reasoning_cotrain) | latent_think.py, latent_arith.py, latent_arith_real.py, latent_sft.py, latent_transfer.py, latent_mem.py, latent_code_cotrain.py, latent_cot_distill.py, latent_rl.py, probe_latent_thread.py, probe_overstep.py, probe_stability_fix.py, probe_think_depth.py, eval_arith_ladder_thinking.py | THINKING_LATENT_2026_05_28.md, THINKING_MEMORY_PLAN.md, LATENT_COT_DISTILL_PROPOSAL.md |
| "Why thinking is marginal on code" probes | probe_gate_placement.py, probe_gate_calibration.py, probe_gate_routing.py, probe_knn_oracle.py, probe_knowledge_vs_search.py, probe_oracle_retrieval.py, probe_retrieval_channels.py, probe_exposure_lever.py, probe_failure_bottleneck.py, probe_pkm_capacity.py, probe_thinking_passk.py, per_problem_flip.py, gate_calibrate_code.py, route_emit_finetune.py, best_of_think.py | WHY_THINKING_MARGINAL_ON_CODE.md, RETRIEVAL_AUGMENTED_LATENT_THINKING.md, THINKING_GATE_SELECTIVITY / THINKING_HUMANEVAL docs |
| WM addressing/recall saga | wm_namekey_probe.py, wm_vqkey_probe.py, wm_recall_cotrain.py, wm_output_sft.py, wm_multitok_readout.py, wm_largerstate_control.py, probe_wm_recall_addressing.py, probe_wm_utilization.py, alias_addressing_probe.py, probe_v5_pkm_utilization.py | AGENTS.md WM sections, WM_ADDRESSING_PROPOSAL, knowledge_base verdicts |
| Recall-cotrain / scoreboard probes | eval_recall_ours.py, eval_recall_tf.py, eval_recall_variants.py, probe_depdist_stratified_ce.py | RECALL_COTRAIN_PROBE.md, SESSION_FINDINGS (metrology corrections) |
| Data generators (some feed still-runnable evals — re-confirm before any deletion) | gen_agentic_recall_tasks.py, gen_alias_recall.py, gen_arith_ladder_pm.py, gen_code_recall_tasks.py, gen_compose.py, gen_cot_distill_data.py, gen_framings.py, gen_incontext_ops.py, gen_incontext_tables.py, gen_longctx_recall_tasks.py, gen_multibind_recall.py, gen_natural_reuse_recall.py, gen_pointer_chase.py, gen_recall_diverse_train.py, gen_recall_variants.py, gen_rejection_data.py, gen_scoreboard_recall_train.py, gen_state_track.py, gen_surface_chase.py, gen_synthetic_memory_tasks.py, gen_synthetic_pyfunc_data.py, gen_synthetic_reasoning_tasks.py, gen_variant_recall.py, build_cot_sft_data.py, build_pretrain_repair_corpus.py, build_probe_dataset.py | respective probe docs; `gen_code_recall_tasks`/`gen_longctx_recall_tasks`/`build_probe_dataset`/`gen_state_track`/`gen_scoreboard_recall_train`/`gen_rejection_data` feed ACTIVE evals |
| Diagnostics / misc | diag_ckpt.py, analyze_diag.py, humaneval_diagnose.py, eval_broader.py, eval_self_repair.py | AGENTS.md diagnostic-tools section, RL_NEXT_DESIGN.md |

(Only file groups shown; the sweep verified each file individually — every one has resolving imports and a doc citation or a
"<90% sure → keep" note.)

### Tests pinned to concluded modules (20)

test_best_of_think, test_cot_distill, test_depth_via_iteration, test_gen_longctx_recall_tasks,
test_gen_synthetic_memory_tasks, test_latent_cot_distill, test_latent_think, test_op_selector_depth,
test_rope_selector_depth, test_sel_counter_depth, test_teachwean_selector_depth, test_pretrain_repair_corpus,
test_probe_gate_calibration, test_synthetic_pyfunc, test_synthetic_reasoning, and the 5 GPU teacher smokes
test_qwen_{gen,logprobs,teacher,throughput,vllm} (KD pipeline validation, memory `project_kd_distillation_pipeline`).
They follow their tested modules: keep while the module is kept.

### Launchers (51 = 46 root + 5 experiments/)

| group | files | note |
|---|---|---|
| Pretrain run history | launch_pretrain_mix_v4.sh, launch_pretrain_mix_v7_pkm_film.sh, launch_pretrain_phase_c.sh, launch_pretrain_v8_wide.sh, launch_pretrain_v8_wide_ddp.sh, launch_pretrain_v9.sh, launch_pretrain_v10_all_features.sh, launch_pretrain_v10_FRESH.sh, launch_pretrain_v11.sh, launch_pretrain_v16_code_fresh.sh, launch_pretrain_v17_singlegpu.sh, launch_pretrain_v18_arxiv_resume.sh, run_v12_autoresume.sh, run_v13_autoresume.sh, run_v14_autoresume.sh | configs-of-record for documented runs (v12/v13/v14 autoresumes are the ONLY launchers of record for those runs, unlike run_v11_autoresume) |
| Latent bake/adapter/RL | launch_latent_bake_probe.sh, launch_latent_bake_code.sh, launch_latent_adapter_cotrain.sh, launch_latent_rl.sh, launch_depthiter.sh | documented latent arc |
| Grader-RL history | launch_rl_grader_phase_c_v2.sh (the "16/164" run of record), launch_rl_grader_pure.sh, launch_rl_grader_wide10L_v2.sh, launch_rl_arith_stateread.sh, launch_rl_grader_v8{,_selective,_judge,_signal,_strong,_strong_judge}.sh, launch_judge_server.sh | v8 RL saga + judge infra |
| SFT history | launch_sft_phase_c.sh (robust-best 14/164), launch_sft_v12.sh, launch_sft_v18.sh, launch_sft_phasec_clean.sh, launch_sft_phasec_astclean.sh, launch_sft_synth_pyfunc.sh, launch_sft_baked.sh, launch_sft_gatecal2.sh, launch_sft_v8_combined.sh | several were HANDOFF "ready-to-run"; some have gitignored-missing base ckpts (weak evidence only) |
| Probes / A-Bs | launch_pkm_prewarm_probe.sh, launch_pkm_unfreeze_probe.sh, launch_phase1_ab_A.sh, launch_phase1_ab_B.sh, run_phase1_10L.sh, run_phase1_evals.sh | PHASE1_AB_RESULTS.md, SESSION_FINDINGS 2026-07-02 |
| experiments/*.sh | ablation_mechanisms_seq.sh (run of record; supersedes _run), distill_kd_ab.sh, embed_ab_run.sh, precond_ab_run.sh, run_deltanet_bf16_grid.sh | each backs a documented A/B |

### Configs (9)

pretrain_mix_v4.yaml (v4/v7/phase-C mix of record, still referenced by ~24 scripts incl. active harnesses),
pretrain_mix_v10.yaml, pretrain_mix_v11.yaml (also v12/v13), pretrain_mix_v14_wmrecall.yaml,
pretrain_mix_v14_wmrecall_maskfix.yaml (v17), pretrain_mix_v15.yaml, pretrain_mix_recall_cotrain.yaml,
pretrain_mix_recall_diverse.yaml (RECALL_COTRAIN_PROBE runs), natural_reuse_recall_mini.yaml
(referenced by gen_natural_reuse_recall.py).

### Docs (24)

BATCH_LR_ANALYSIS, BEST_OPTIONS_BRAINSTORM, DECODE_COST_BENCH, DELTANET_PRECONDITIONER{,_AB,_PERF},
EMBED_OPTIMIZER_AB, EXEC_PRETRAIN_PROPOSAL, FLAGSHIP_PROBE_RESULTS, INTELLIGENT_AGENT_BRAINSTORM,
LATENT_COT_DISTILL_PROPOSAL, LINEARIZATION_PROOF_RESULTS, LOSS_BALANCE_REPORT, PHASE1_AB_RESULTS,
RECALL_COTRAIN_PROBE, RETRIEVAL_AUGMENTED_LATENT_THINKING, RL_NEXT_DESIGN, STATEFUL_CHUNKING_PROPOSAL,
THINKING_GATE_SELECTIVITY_2026_05_30, THINKING_HUMANEVAL_2026_06_06, THINKING_LATENT_2026_05_28,
THINKING_MEMORY_PLAN, WHY_THINKING_MARGINAL_ON_CODE, WM_ADDRESSING_PROPOSAL_agentic_semantics — all are
result/decision records cited from AGENTS.md or memory notes. No dead docs.

---

## DEAD (13) — with evidence

| file | evidence |
|---|---|
| experiments/gen_exec_trace.py | Superseded by `gen_exec_traces.py` (2026-07-04), whose docstring explicitly labels it "the OLD, pre-fair-control generator … templated, not executed". Zero references from any .md/.sh/.py (only mention is gen_exec_traces' description of it); no test file. |
| launch_pretrain_v10_SMOKE.sh | 30-step smoke saving `SMOKE_v10_DELETEME.pt`; run concluded; superseded by launch_pretrain_v10_FRESH.sh. |
| launch_pretrain_v11_SMOKE.sh | Smoke saving `SMOKE_v11_DELETEME.pt`; superseded by launch_pretrain_v11.sh (kept). |
| launch_pretrain_v15_SMOKE.sh | 150-step smoke (`_smoke_v15.pt`); superseded by launch_pretrain_v15.sh. |
| launch_pretrain_v15.sh | Discrete-key-WM continuation superseded along the v15→v17/v18/v19 chain (ctx_namekey absorbed the discrete-key WM); no unique documented result beyond the WM saga (which cites the probes, not this launcher). |
| launch_pretrain_v17_allfeatures.sh | Abandoned DDP path: its own successor `launch_pretrain_v17_singlegpu.sh` header says the DDP path was abandoned (static_graph + no_sync PyTorch bug, memory `project_ddp_latent_incompatibility`); never produced a validated run. |
| launch_sft_gatecal.sh | Strictly superseded by `launch_sft_gatecal2.sh`: identical script incl. save path `checkpoints/sft_gatecal.pt`, gatecal2 only adds `--gate_calibration_max_positions 16 --gate_calibration_sample_frac 0.05` (verified by diff). |
| resume_v9.sh | Autoresume wrapper of the finished v9 run; its seed glob `checkpoints/pretrain_v9_step*.pt` matches nothing → self-aborts (verified); config preserved in launch_pretrain_v9.sh (kept). |
| run_v11_autoresume.sh | Redundant autoresume duplicate of launch_pretrain_v11.sh (the config of record, kept); v11 concluded. |
| trigger_v9_sft_3b.sh | One-shot trigger that blocks forever waiting for `pretrain_v9_step*_tok*.pt ≥ 3B` (none exist, v9 done); its gating question was answered (latent-cotrain Δlogp negative, v9 arm closed). |
| run_phase1_evals_C.sh | Evaluates `checkpoints/phase1_ab_C.pt`, which does not exist (verified); also depends on transient `/tmp/probe_he_ce.py`. Arm-C never materialized; run_phase1_evals.sh (kept) covers A/B. |
| experiments/ablation_mechanisms_run.sh | Superseded by `ablation_mechanisms_seq.sh`, whose header records that this parallel/GPU0 variant failed twice (GPU0 crash + OOM); HANDOFF cites the _seq run as the run of record. |
| experiments/precond_ab_smoke.sh | 3-step wiring smoke for the concluded preconditioner A/B; superseded by precond_ab_run.sh; result documented in DELTANET_PRECONDITIONER_AB.md. |

No test file belongs to any DEAD item (pytest baseline unaffected by pruning: **1067 collected before = after**).

---

## DEAD-FLAGS: none found

Full consumer audit of all three argparse surfaces (218 + 41 + 69 dests): **every flag is consumed by a live
code path**. Verified specifically:

- `--process_reward_*`, `--mem_write_only_at_think`, `--wm_future_pred_weight`, `--future_emb_T_max`,
  `--future_emb_T_ramp_frac`: **absent** from all parsers and all launchers (removal was complete).
- `--future_emb_loss_weight` (sft_code) is ALIVE — despite the legacy name it feeds the live trunk-gist heads
  (`sft_code.py:1098,1267,1354`), not the removed lexical future-emb mechanism.
- `--layer_drop_max` is not a CLI flag at all (model_builder `getattr` only) — the deliberately-kept rejected option.
- Single-read flags spot-checked as genuine consumers, e.g. `--latent_reasoning_depth_weighted`
  (train_lm.py:1537 → LatentReasoningCotrain), `--think_checkpointing` (train_lm.py:1725), engagement-check
  flags (train_lm.py:2900-2929), `--batch_turn0` (train_rl_grader.py:1714).

Only residue: two stale **comments** naming `compute_process_reward_loss` in `experiments/speed_knobs.py:135`
and `experiments/test_speed_knobs.py:101` — harmless; reword whenever those files are next touched.

---

## Proposed prune order (safest first)

1. **Orphaned superseded one-off** — `experiments/gen_exec_trace.py` (zero inbound references, explicit
   supersession note in its replacement's docstring, no test).
2. **Smoke scripts of finished runs** — `launch_pretrain_v10_SMOKE.sh`, `launch_pretrain_v11_SMOKE.sh`,
   `launch_pretrain_v15_SMOKE.sh`, `experiments/precond_ab_smoke.sh`.
3. **Self-aborting / dangling wrappers** — `resume_v9.sh`, `trigger_v9_sft_3b.sh`, `run_v11_autoresume.sh`,
   `run_phase1_evals_C.sh`, `experiments/ablation_mechanisms_run.sh`.
4. **Superseded launchers with a kept successor** — `launch_sft_gatecal.sh` (→ gatecal2),
   `launch_pretrain_v17_allfeatures.sh` (→ v17_singlegpu), `launch_pretrain_v15.sh` (→ v17/v18/v19 chain).
5. **Dead flags + their parser hunks** — nothing to do (0 found).
6. **Test files of deleted code** — nothing to do (0 exist).

Total prunable: 13 files, all category DEAD. Everything else is ACTIVE or cheap-to-keep history.

## Housekeeping findings (not prunes)

- **AGENTS.md/CLAUDE.md doc index is stale**: it cites 8 docs that no longer exist anywhere in the repo —
  `MILESTONE_ARCH.md`, `PHASE_C_RL.md`, `PLAN.md`, `NEXT_DIRECTIONS.md`, `THINKING_RAG_DIRECTION.md`,
  `RL_RAG_ROADMAP.md`, `THINKING_PLAN.md`, `CROSS_DOC_ISOLATION_PLAN.md` (not moved to `knowledge_base/` or
  `docs/` either). The index should be refreshed in a future docs pass.
- Several KEEP-HISTORICAL launchers reference gitignored checkpoints that are no longer on disk
  (`launch_pretrain_v9.sh`, `launch_sft_baked.sh`, `launch_sft_gatecal2.sh`, `launch_sft_v8_combined.sh`,
  `run_v14_autoresume.sh`) — kept because missing ckpts are weak evidence and each run is documented, but they
  are the next-weakest tier if a deeper prune is ever wanted.
- Borderline calls resolved to KEEP-HISTORICAL per the <90% rule: `eval_recall_ours.py` (partially superseded by
  `eval_recall_tf.py` but still cited in RECALL_COTRAIN_PROBE.md), `launch_pretrain_v16_code_fresh.sh`
  (transitional but runnable, no missing inputs), and the six data generators feeding ACTIVE evals (listed above).
