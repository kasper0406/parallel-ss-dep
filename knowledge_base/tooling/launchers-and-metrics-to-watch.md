# Launchers & metrics to watch

## Summary
The autoresume launchers and the live training-log metrics that tell you a run is healthy (and the specific failure signatures to catch early). The autoresume pattern (`run_v*_autoresume.sh`) loops the trainer, resuming from the latest mid-eval ckpt, so a crash/OOM/GPU-fall-off doesn't lose progress. **Live runs: v12 (GPU1), v14 (GPU0) — do not co-locate or touch.** Sources: `run_v14_autoresume.sh`, `CLAUDE.md`, the `project_*` memory files.

## Launchers
- **`run_v{11,12,13,14}_autoresume.sh`** — the production pretrain loop. Each `common_args()` block is the full config; the loop seeds from the latest `_step*.pt` (resume) or a prior-version ckpt (continuation). v14 is a **v12 continuation** (inherits code competence + trained DKV WM; embedding-key addressing adds no params so v12 loads byte-identical).
- **`launch_pretrain_*`** — the historical run configs (v4, v7-pkm-film, phase_c, v8-wide, v9, v10). See [[pretrain-run-history]].
- **Post-training**: `launch_sft_*` (SFT), `launch_rl_grader_*` (execution-grounded GRPO; v2 is the KL-stable 16/164 recipe), `launch_latent_*` (latent bake / adapter cotrain / RL), `route_emit_finetune.py` (gate→emit on code).
- **Standalone fix/validation scripts** (not in a launcher): `latent_reasoning_cotrain.py`, `wm_recall_cotrain.py` (`--unfreeze_trunk`), `latent_code_cotrain.py --freeze_trunk`, `best_of_think.py`.

## Live metrics to watch (and their failure signatures)
- **VAL ppl** — a sudden **+10 %** jump = a new aux loss destabilized the run (the v12 cold-latent failure). Use real PPL (exp(−CE)), don't include ponder cost.
- **Per-block gnorm** (`gnorm(L0,Lmid,Llast,last/first)`) — uniform jump to ~20× = full-trunk aux-dominated gradient (cold-latent). `last_over_first ≫ 1` = early layers gradient-starved.
- **PKM health** (`pkm(αL, αeff, row, slots/H, top, ε, φ)`):
  - **`αL` must commit** to +0.27..+0.35 by the α-floor window end. αL stuck ~0 = the v7.0/v11/v13 **decay-failure path**.
  - **`row` > 1 and growing** = values learning; frozen at ~1 = dead. (`v_std` is useless.)
  - `top` (hottest-slot mass) spiking to ~0.17 (from ~0.008) + `slots/H` collapsing = routing damaged.
- **`latent(Δlogp)`** — should climb toward / above 0 as the op trains; stuck net-negative = cold/OOD latent path.
- **`gc(tgt1, σ, Δlogp)`** (gate-calibration) — `tgt1` high + `σ` low = gate miscalibrated (thinking helps but the gate wants to emit).
- **WM**: `read_alpha` (decaying 1.0→~0.08 = optimizer dialing WM out, no recall gradient), and on the recall probe **think_frac** (0 = recurrence solves it, WM never exercised — route-around). Measure WM deltas only in **training-matched format** on a **headroom-bearing** probe.

## Operational rules
- **`--mid_eval_save_only`** on production runs (HumanEval-during-pretrain is noisy + OOMs; the ckpt is the load-bearing artifact).
- Kill wedged CUDA processes by **process group**, not PID.
- Re-verify `fla.utils.IS_NVIDIA_BLACKWELL is True` after any fork pull. See [[hardware-and-compute]].

## Related
[[pretrain-run-history]] · [[product-key-memory]] · [[working-memory-recall-saga]] · [[training-and-optim-knobs]] · #tooling #infra
