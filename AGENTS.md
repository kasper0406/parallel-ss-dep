# AGENTS.md — Project Instructions

> Canonical, tool-agnostic agent instructions for this repo (the
> [AGENTS.md](https://agents.md/) standard). `CLAUDE.md` and `GEMINI.md` are
> symlinks to this file, so Claude Code, Gemini CLI, Codex, Cursor, etc. all
> read the same source of truth.
>
> This file is the *current* state, resolved. Chronological history, superseded
> claims, and the full experimental arc live in **`AGENTS_HISTORY.md`** — read
> that only when you need the "why" behind a resolved decision.

## Top-level goal — cost-moat coding agent (north star committed 2026-06-30)

**Build a cheap, bounded-state, long-context / high-concurrency coding agent
whose moat is COST and ADAPTIVITY — not benchmark rank.**

- **Cost moat (proven):** O(1) / constant-memory decode. The lean DeltaNet
  config holds a flat 8.2 MiB recurrent state and ~14.4 ms/tok, ~919 MiB decode
  peak from L=512 to L=131072, while a transformer's KV cache grows 20→5121 MiB
  (6.45× memory gap at 131k, still widening). See `DECODE_COST_BENCH.md`,
  `SCOREBOARD.md`.
- **Adaptivity (the bet, unmeasured):** learn from deployment — meta-train
  DeltaNet's recurrent state (delta rule = Widrow-Hoff GD step, β = learned
  per-token LR) into a test-time learner over full-repo ingestion, plus a
  LoRA-sleep consolidation tier. This is the main differentiation program now
  (see `META_TTT_PLAN_2026_07_13.md`).
- **The benchmark-rank goal was CONCEDED.** A pure-DeltaNet 287M "beats big
  models on HumanEval" is structurally unreachable at this budget: token-poverty
  (half-size SmolLM2-135M models code CE better — 0.798 vs our 0.969 — from a
  ~400–800× token deficit) + a linear-attention tax that *grows* with donor
  strength. Do not chase a HumanEval headline; the tokens can't buy it.

**Hard compute constraint:** 2× RTX 5090 (32 GB each, no NVLink), sm_120
(Blackwell). One GPU fell off the bus once under sustained load (2026-07-11; a
reboot recovered it) — periodic checkpoints are what saved that run; keep them
mandatory. Latent-thinking runs are single-GPU (DDP+latent is doubly
incompatible; see conventions). We are token-limited: our bases see ~5B tokens; nobody trains a
competitive sub-1B model on <2T, so **inheritance (linearization / KD) is the
only coherent path to a competent base**.

Reference framing: `NORTH_STAR_2026_06_30.md`, `STRATEGY_2026_06_28.md`,
`THESIS.md`.

## Current architectural stack

Backbone is **DeltaNet** (`--arch deltanet`, plain `chunk_delta_rule` from FLA —
efficient linear-RNN, bounded state). `--arch gated_deltanet` is broken on sm_120
(FLA `prepare_wy_repr_bwd` CUDA misaligned-address in the Triton autotuner) — do
not use until upstream FLA fixes it.

Two live trunk shapes:
- **Shallow-wide from-scratch trunk** — `--n_layers 10 --d_model 896 --n_heads 14
  --d_head 64 --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9"
  --feedback_self_k 3`. Iso-param swap of the old 30L×576d trunk; matches/beats
  it on VAL CE at ~18% less wall-clock.
- **Linearized inherited base (the competent-base path)** — SmolLM2-360M weights
  copied into a 32L×960d DeltaNet (bit-exact embed/MLP/norms + MOHAWK attn
  transfer + e2e KD): HE-solution CE 0.759 lean (`checkpoints/linearize/
  linearized_stage3.pt`), preserves O(1) decode. **The production base is
  `checkpoints/production_lean_soup3.pt` (HE-CE 0.6614, 2026-07-18)**: an
  equal-weight soup of three 0.3x-anneal decay seeds — beats every single
  seed AND the control on BOTH axes (depdist 0.8126 vs control 0.8254, every
  stratum improved); N-seed decay + soup is now the standard anneal recipe: the
  3B-token composition run — 2.56B-token constant-LR plateau on the pilot mix
  from `stageA_executor.pt` (0.7343) + 459M-token WSD decay on a 0.3x-strength
  curated anneal mix (`anneal_mix_v4.yaml`), guard-clean on depdist (+0.0079 ≤
  +0.01). Anneal strength does NOT transfer across scales (0.4x passed the
  micro-run projection but tripped the production guard at +0.0146) —
  re-calibrate per run; full trade curve in IDEAS_2026_07_13.md Tier-1 log.
  Earlier trunks `feature_pilot_A.pt` (0.7429) / `stageA_executor.pt` (0.7343)
  remain for lineage A/Bs. The stronger Qwen2.5-Coder-0.5B donor was tried and
  **refuted** (+0.276 tax, only 15/164) — inheritance buys a cost substrate,
  not a free quality jump.

Validated mechanisms (each earns its place on its OWN matched probe, NOT on the
noisy HumanEval headline):

- **Sparse FiLM feedback**: −3.1% to −5.4% PPL at 217M/360M/708M (README table).
  K=3 self-feeding closes the train/inference gap → single-forward 1× decode.
  `--alpha_wd 0.0` (no weight decay on FiLM α). NOTE: `prefill`/`forward_step`
  currently bypass FiLM at decode — a FiLM-trained ckpt generates as a different
  function than it trained as; the K=1-lag decode wiring is the fix (in progress,
  commit 58c7e70).
- **Product-Key Memory** (`experiments/memory_layer.py::PKMLayer`), 262k learned
  KV slots after one block. Load-bearing **when engaged** (+0.04..0.14 CE on
  recall-heavy tokens; −5 HumanEval in the Phase-C ablation), but honestly a
  **bounded sample-efficiency lever, not a capacity win** (≈ dense at matched
  params). Requires the **v7.1 bootstrap-fix package** to train at all
  (v5-pkm was 97% dead value rows): `--pkm_use_output_gate` (α init 0),
  `--pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 2000` (sign-
  preserving α floor), `--pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 2000`
  (ε-greedy slot replacement), `--pkm_value_init_std 1.0`, `--pkm_score_norm
  layer`, `--pkm_diversity_weight 0.01`, `--pkm_value_lr_mult 100.0` (separate
  AdamW group). **No weight decay on value tables** (commit b12bbd3 — WD erodes
  cold rows). Live diag: `pkm(αL,αeff,row,slots/H,top,ε,φ)` — `row` (mean row-
  norm vs init) is the health metric, `v_std` is misleading. Probes:
  `probe_v5_pkm_utilization.py`, `probe_pkm_per_source.py` (canonical toggle).
- **Bounded Working Memory** (`experiments/model.py::WorkingMemory`) — write-
  gated buffer of past hidden states. Final addresser is **`mem_ctx_namekey`**: a
  learned, no-hash **contextual name-span key** (dot-product attention over the
  trunk's contextual hidden pooled on the identifier name-span) + copy/pointer
  readout, `--ctx_addr_aux_weight` supervision. Co-trained into pretrain it is
  **load-bearing leak-free**: first-occurrence const recall 0.03→0.98, syn64
  +0.29→+0.76 and syn128 +0.07→+0.53 (500M→1B, strengthening). +11.1pp on
  saturated MQAR (T=512,K=128). **BUT for the code model WM is inert/drop**:
  recurrence absorbs the recall the code path needs (realistic N≤24, below WM's
  validated N≥48 niche); frozen it's dead weight, forcing it on crashes recall
  99%→35%. Keep WM only for the synthetic-recall research probe and the meta-TTT
  substrate; do NOT reflexively bolt it onto code. Discrete lexical-hash and VQ
  keys are dead (spelling-locked) — the right key is the continuous name-span
  key. Open gap: the copy head only fires on data-provided masks, not at
  inference (self-calibrating trigger being wired, 58c7e70).
- **Latent-space thinking** (the ONLY working thinking primitive; discrete-token
  thinking is dead) — `experiments/latent_think.py`, Coconut-style: at an
  appended think slot feed the trunk's own continuous hidden back as the next
  input for R iterations, **state-readonly** (`--state_readonly_at_think`,
  DeltaNet β=0 at think positions) so the recurrent state is never corrupted.
  Full `d_model` bandwidth each step (token feedback FAILS: 0.09 vs 1.00).
  **Scope, resolved:** it substitutes trunk *depth* for *homogeneous iteration*
  (shallow L2+R=K hits 1.00 through K=8; off-diagonal R≠K collapses → genuine
  per-step compute) but hits a **heterogeneous-composition wall** — random-access
  op-selection collapses to chance for any answer-only-trained learned selector.
  On general per-token code it is **net-negative / salvage-narrow** (code is
  recall, not iteration). Structural Pareto floor is solid: adapter-at-think-slot-
  only ⇒ no-think byte-identical to base (max|Δlogits|=0). See the live
  exec-trace program below for where latent DOES work.
- **Emit/think gate** — per-position sigmoid head, `σ=sigmoid(gate_logit)=P(EMIT)`
  (LM loss, eval `emit iff g≥threshold`, entropy-aux all agree). Optional
  supervision:
  - **Gate-calibration aux** (`--gate_calibration_weight`, wired in both
    `sft_code.py` and `train_lm.py`): sampled positions with σ∈[low,high] get an
    extra latent-R forward; `target=1{Δlogp>0}`; trains `P(think)=σ(−gate_logit)
    →target` (polarity fixed 2026-06-13 — the gate THINKS where thinking helps;
    baseline lp0 on the SAME left-padded prefix). Uses the latent primitive
    (`latent_R`), not discrete tokens. Guard: `test_gate_calibration.py::
    test_gate_calibration_polarity_thinks_where_helpful`.
  - **Entropy-grounded target** (`--gate_entropy_aux_weight 0.1
    --gate_entropy_aux_temperature 2.0`): BCE to `exp(-H_t/T)` from the same
    forward's logits. No extra forward.
  - Gate is **temperature-fragile**: healthy think_rate≈0.30 at greedy collapses
    bimodal under τ=0.9 RL sampling. Don't expect train-time selectivity to
    survive rollouts without robustification.
- **Cross-document state isolation** (`cu_seqlens` from per-position `doc_ids`) —
  `data_mix.py::MixedSourceStream(emit_doc_ids=True)` → `model._build_cu_seqlens`
  → FLA `chunk_delta_rule`; `WorkingMemory` gets a same-document read mask.
  `doc_ids=None` is byte-identical to the old path. Tests `test_cu_seqlens.py`.
- **Execution-grounded RL** (`experiments/train_rl_grader.py`) — GRPO with
  `code_grader.grade()` dense reward (tier ladder syntax 0.0 < exec 0.05 <
  runtime 0.2 < partial < pass 1.0; `error_text` for repair). KL-to-reference
  stable (`--kl_coef 0.05`, frozen ref). RL's real effect is greedy sharpening,
  not new capability (pass@k envelope is SFT/base-set, not RL-created).
- **Mixed-corpus pretrain + day-1 co-trained stack** (`data_mix.py`,
  `--data_mix configs/pretrain_mix_v*.yaml`) — features must be co-trained from
  day 1 (bolt-on-to-converged-trunk = inert), attached via frozen-trunk pre-warm,
  NOT forced contribution floors (floors on a converged base are sabotage — see
  recipe rules).

Dead directions (do **not** revive without explicit justification): Corpus-RAG
(`rag_projection`, gradients structurally zero); `--arch gated_deltanet` on
sm_120; process-reward aux loss (`process_reward.py` deleted); `--mem_write_only_
at_think` (FIX A, empirically rejected); discrete-token thinking measurement;
LayerDrop/`--layer_drop_max`; embedding-specific optimizers (tuned embed LR −1.76%,
rownorm dualizer −12.84%); SOAP / Sketchy-Shampoo (Pareto-dominated by Muon at
287M); scaling to a stronger same-vocab donor for a quality jump (Qwen refuted).

## Key mandates — evaluation

- **Measure recall LEAK-FREE.** `eval_code_recall.py --mode teacher_forced` on
  `*_recall_heldout` scores a RESTATED answer → recurrence recency-copies →
  WM-ON==WM-OFF==~100%, uninformative. Use the **first-occurrence** protocol:
  `wm_recall_cotrain.py --mode validate`, the saturating multibind probe, or
  `scoreboard_longctx_cost.py` (bind once near the start, never restate before
  the query). There WM-OFF const is ~0.03, not 1.0.
- **One committed HumanEval harness; treat greedy 164 as noisy.** Canonical
  config: `--prompt_style sft_comment --extract_code_block --min_emit_before_eos
  30 --max_gen 512`. HumanEval-164 greedy is ±2–3 noise at this scale — the
  "0→10→14→16/164 arc" was config-drift + noise (real cluster ~13–15, SFT≈RL,
  "16" not reproducible). **Do not trust ≥1-point deltas**; use temp pass@k /
  bootstrap CI / a bigger bench (HumanEval+ / MBPP+ / CRUXEval) for real A/Bs.
- **Fair baselines — run the control that gives the baseline equal opportunity.**
  For a thinking claim, compare against a **separately-trained** no-think control
  (same data/steps), NOT the within-model `emit_threshold=0` path. State caveats
  proactively; the user catches inflated lifts. Report best-of-N+grader (~21/164
  pass@30) only beside the SAME arbiter on a matched competitor.
- **Dependency-distance-stratified CE is the standing feature dev signal**
  (in-repo probe). Stop deciding feature questions on HumanEval greedy or
  ≤512-token CE (all code evals were short-context → WM's niche was invisible).
- **Use real PPL** (exp(−CE)); don't fold ponder cost into displayed ppl.

## Key mandates — training config (validated defaults)

- **Always `--bf16 --tf32`** (fp32 path leaves ~2.3× speed on the table).
- **`--wd 0.01`** (now the default; `0.1` is a Moonlight 5.7T-token setting that
  collapses the residual stream at our budget). `--alpha_wd 0.0` on FiLM α.
- **`--lr_schedule wsd`** (default; warmup → constant peak → short cosine decay
  over `--lr_decay_frac 0.15`). We keep extending runs, so cosine's upfront
  `T_max` is a poor fit. Never let WSD hit LR=0 at a curriculum's end.
- **`--batch` × K-self-feed OOM rule:** K=3 self-feed spikes activations +3.76 GiB
  when K engages past the warmup. Use `--batch 14` whenever K=3 will engage;
  `--batch 20` only for K=1-the-whole-way runs. Always
  `--activation_checkpointing` for production (bit-identical, ~Nlayers× less
  stored activation, ~30% more compute).
- **`--grad_accum N`** (pretrain path only; errors with `--enable_thinking_token`)
  — target ~229k tok/step (`--batch 14 --grad_accum 8`). `--grad_clip 1.0`,
  `--z_loss 1e-4` (logit-drift regulariser).
- **Gate-floor traps:** In BPTT pretrain never let `--gate_floor_min` reach 0.0
  (maladaptive-thinking trap: VAL ppl 49→940). Use `--gate_floor_min 0.5
  --gate_warmup_steps 20000`. In RL rollouts that must preserve thinking, keep
  **`gate_floor < emit_threshold`** (e.g. 0.3 vs 0.5) — `clamp_min(gate_floor) ≥
  emit_threshold` silently makes the model never-think (`test_rl_grader_gate_
  floor.py`).
- **Halt-after-docstring fix** (pretrain teaches `"""\n → EOS`): eval with
  `--min_emit_before_eos 30` + `--gate_floor` (mirrors train clamp); train with
  `--mask_eos_in_targets`.
- **Optimizers:** keep **Muon** (Shampoo-adjacent, Pareto-dominant). The one free
  training-speed win is the per-head Newton-Schulz DeltaNet matrix optimizer
  (`--matrix_optimizer fused_deltanet_ns`, ~3–13% iso-step, byte-identical
  step-0) — adopt for the next from-scratch pretrain; it is speed, not capability.
- **GRPO ponder shaping** (`compute_grpo_advantages`): `--grpo_ponder_shape
  quadratic --grpo_ponder_counterfactual --grpo_ponder_warmup_steps 300`.
- **Sparse-loss validation tasks (MQAR, recall) run in fp32** — bf16 collapses
  logits to uniform when only ~25% of positions carry gradient. Production
  pretrain (every token contributes) is unaffected. Use MQAR **T=256/K=32** for
  architectural ablations (the saturation T=512/K=128 regime doesn't discriminate
  feedback choices at our scale).

## Key mandates — launch / ops hygiene

- **Periodic checkpoints are mandatory.** Stage-A survived a GPU-off-the-bus
  crash by resuming from its step-382 periodic ckpt. On resume, re-pass
  `--mid_eval_every_tokens` and all curriculum flags (a v4 run lost 4000 steps of
  resume points by dropping it). Production pretrain: `--mid_eval_save_only`
  (skip the noisy HumanEval subprocess, save the ckpt), `--mid_eval_min_free_gib
  2.0` (auto-skip subprocess under GPU pressure).
- **Converged-base attach needs a frozen-trunk PRE-WARM, not forced floors.**
  Forced-contribution floors on a converged base are sabotage (feature-pilot B:
  WM read-α floor injected 15%-RMS noise every position for 800 steps → worse
  everywhere). Pre-warm each feature frozen-trunk with NO floors until genuinely
  useful, THEN unfreeze (`launch_feature_prewarm_phase0.sh`). Attach latent
  adapter-only (never full-trunk latent aux on a converged trunk).
- **Curricula ≤ ~40% of run length**; leave a constant-LR plateau after
  bootstrap. Add an engagement kill-gate per feature. ≥2 seeds (or a variance
  estimate) for any ≤0.02-nat claim.
- **Fixes are defaults, not flags.** Ship a validated fix ON by default; use
  ckpt-cfg legacy-on-missing-key for backward compat; default-off flags only for
  genuine experiments. Repo history is littered with runs lost to misconfigured
  flags.
- **Review implemented code.** After any implementation (yours or a delegated
  agent's), launch a background review agent on the diff before relying on it or
  scaling. Tests are a mandatory deliverable; personally skim correctness-critical
  hunks, re-run key tests, spot-verify one number.

## Conventions

- Python env via `uv`, torch nightly cu132 (`...whl/nightly/cu132`).
- **FLA must be the local fork** at `~/ml/flash-linear-attention`
  editable-installed (`uv pip install -e ...`) — the pip build lacks the
  Blackwell `global_scratch` allocator fix + our `cache.py` restore_value patch,
  and mis-detects sm_120 as non-Blackwell. Verify `import fla` resolves to the
  fork.
- `export PYTHONPATH=$PYTHONPATH:.` before `experiments/*.py`.
- Checkpoints: save `{state_dict, step, config}`, load `weights_only=False`.
  `eval_bracket_structure.build_model_from_ckpt` auto-detects gate/memory/adapter
  from the state-dict — reuse it, don't hand-roll model construction.
- **DDP + latent is doubly incompatible → latent runs are single-GPU.** (a)
  static_graph + grad-accum no_sync is a genuine unfixed PyTorch regression
  (upstream fix staged at `~/ml/pytorch`, not pushed); (b) the latent reentrant
  adapter needs static_graph, which our R=2..8 + step-2000-engage curriculum
  breaks. The second GPU could be recovered via manual bucketed allreduce /
  DiLoCo (no DDP hooks) — a queued ~1.9× lever (IDEAS Tier 0).
- **`data_mix` min_content_len fix** (a662e84): `_filter_min_content_len` only
  checked content/text fields → magicoder+textbooks were 0 tokens across v10–v17;
  fixed with a longest-string fallback. Watch per-source token counts.
- **Trainer module layout:** `train_lm_args.py` (`build_parser()`), `build_arch.py`
  (`build_arch`/`parse_layers_arg` + attention registry), `model_builder.py`
  (`build_model_from_args`), `optim_utils.py` (`build_optimizer`, Muon/AdamW
  split, WSD/cosine), `speed_knobs.py` (`apply_speed_knobs`: bf16 autocast + TF32
  + optional compile). `model.py::TinyLM._finalize` is the shared forward exit-
  tail — call it in new branches, don't duplicate `out_norm→memory→lm_head`.
- **`--compile` default-on, strict** (`torch._dynamo.config.suppress_errors=False`
  — silent eager fallback was a repeated footgun). `--no-compile` is the escape
  hatch (needed with aux-extra-forwards: Inductor symbolic-shape crash). FiLM
  K-warmup bypass (`--feedback_self_k_warmup_steps N`, `TinyLM._film_bypass`) is
  the bigger speed lever (+40%). Also: `--bf16_optim_state` (~550 MB saved,
  lossless), compiled AdamW step (default on, Muon left eager).
- **Speed knob for generation:** `model._film_bypass=True` around all generate
  loops. True O(1) incremental decode (`TinyLM.prefill`+`forward_step`) is wired
  and logit-tested for the lean/plain-DeltaNet config (constant 8.2 MiB state);
  FiLM-K3 + WorkingMemory production stack remains unwired at decode (WM buffer is
  O(T)) — the deferred part.
- **Tests** live under `experiments/test_*.py`, pytest-discoverable. Run:
  `PYTHONPATH=. .venv/bin/python -m pytest experiments/test_*.py -v`. Pin CUDA
  tests to a free GPU (`CUDA_VISIBLE_DEVICES`). Add a test with every new builder,
  filter, or shaping mode.
- **Diagnostics** (CPU-or-single-GPU, safe on any ckpt): `diag_ckpt.py` (logit-
  lens CE, effective rank, ‖h‖, per-source CE — found the residual-stream
  collapse), `diag_reference_lm.py` (matched HF reference, SmolLM2-135M),
  `profile_train.py`, PKM probes above. Per-layer grad/update diagnostics log
  every `--log_every` step (read `last_over_first`: ≫1 → early layers gradient-
  starved; ≈1 → healthy, bottleneck is tokens/data).

## Current programs in flight

- **Exec-trace latent thinking — "the neural interpreter"** (`EXEC_TRACE_LATENT_
  PLAN.md`). Teach the model to SIMULATE program execution — per-step ground
  truth is free and infinite (`sys.settrace`; `gen_exec_traces.py`, 30,100
  examples K=2–8). **Resolved arc:** latent-first FAILS (N1′: latents learn the
  value prior, hop CE plateaus at ~ln 10; the staging step was skipped) →
  **Stage A** text-scratchpad executor **DECISIVE PASS** (heldout answer-with-
  trace 0.94–0.97 vs direct 0.12–0.16 = +80–84pp; HE-CE 0.7343, best code ckpt;
  `stageA_executor.pt`) → **Stage B** Coconut text→latent compression **NOT
  KILLED**: latent(R=K) answer 0.63/0.63/0.59 at K=4/5/6, per-hop decode 0.844
  @K4 (N1′ was 0.11), depth-true signature (R=1 AND R=K+4 both collapse), HE-CE
  0.7491. Limit = a **~6-hop latent horizon** (curriculum under-exposure of deep
  slots; depth-weighted sampling arm `--latent_reasoning_depth_weighted` built).
  **Durable methodology finding: externalize before you internalize** — latent
  compression is learnable only once the function pre-exists in token space; dense
  per-step supervision alone is NOT sufficient. CRUXEval-O transfer probe:
  mechanism does NOT transfer, but direct-answer internalization +53% rel (z=2.58,
  n=800, token-count confound flagged). Winding down to a **writeup** ("Latent
  Execution: internalizing interpreter traces into continuous thoughts"); the
  trained executor stays as a *component* of the eventual agent, not the headline.
  Novelty map: `LITERATURE_LATENT_EXEC_2026_07_13.md`.
- **Meta-TTT repo-adaptive coder** (`META_TTT_PLAN_2026_07_13.md`) — the main
  differentiation bet, unparked 2026-07-13. Meta-train the recurrent state as a
  deliberate test-time learner over full-repo ingestion (O(1)/8.2 MiB at any
  length). Pre-registered kill-test: episodes `[repo 8–32k → task]`, arms
  real-repo / shuffled-repo / no-ingestion; kill if lift(real−shuffled) after
  meta-training doesn't exceed the non-meta-trained base's incidental lift.
  P0 (episode builder + 3-arm harness) built (commit 7223772); P1 meta-train
  pilot next. Run the kill-test on the CURRENT base first (cheap, decisive), port
  to the linearized base in P3 if it passes.
- **Linearize-a-donor competent base** (`LINEARIZATION_PROOF_RESULTS.md`,
  STRATEGY fork) — composes with meta-TTT (needs a base worth adapting), doesn't
  compete. SmolLM2-360M path validated (CE 0.759); Qwen-Coder-0.5B refuted for
  quality. Open fork: is the linear-attn tax reducible via a **moat-preserving
  sliding-window hybrid** (`linearize_hybrid_ablation.py`)? Full attention is off
  the table (breaks O(1)).
- **Decode-kernel optimization** — compile / CUDA-graph `forward_step` to fix the
  un-optimized-eager latency constant (the O(1) flatness is architectural; the
  bad constant is fixable). Turns single-stream latency from a blemish into a win
  earlier.
- **Ideas backlog** (`IDEAS_2026_07_13.md`, 10-lens synthesis). Tier-0 near-free
  probes: State Algebra (merge delta-rule states), exposure-bias diagnostic on the
  6-hop horizon, two-GPU manual-allreduce validation, execution-gated cascade
  cost number. Tier-1 next production run: **KD-through-anneal on the linearized
  base** (decay-phase data upgrade + Rho-1 token triage + per-head-NS + FIM +
  SAM-on-decay + SWA soup). Tier-2 research: search-native decoding, repair-trace
  midtraining + EdgeBench-mini, host-RAM PKM scaling.

## Documentation index

- `NORTH_STAR_2026_06_30.md` — the committed cost-moat-agent thesis + build order.
- `STRATEGY_2026_06_28.md` — ranked directions, the two real forks, kill-signals,
  "kill now" list (benchmark goal unwinnable here).
- `META_TTT_PLAN_2026_07_13.md` — the repo-adaptive-coder differentiation bet.
- `EXEC_TRACE_LATENT_PLAN.md` — neural-interpreter program (Stage A/B results).
- `LITERATURE_LATENT_EXEC_2026_07_13.md` — novelty map for the latent-exec writeup.
- `IDEAS_2026_07_13.md` — 10-lens idea synthesis + recommended sequencing.
- `SCOREBOARD.md` / `DECODE_COST_BENCH.md` — the cost-moat measurements (the
  board where the moat is the metric).
- `THESIS.md` / `README.md` — project framing + headline results.
- `SESSION_FINDINGS.md` — chronological empirical log (tail = freshest).
- `HANDOFF.md` — session-to-session handoff notes; append at end of each session.
- `THINKING_LATENT_2026_05_28.md`,
  `THINKING_HUMANEVAL_2026_06_06.md`, `WHY_THINKING_MARGINAL_ON_CODE.md`,
  `DELTANET_PRECONDITIONER*.md`, `STATEFUL_CHUNKING_PROPOSAL.md`,
  `EXEC_PRETRAIN_PROPOSAL.md` — mechanism/probe deep-dives.
- **`AGENTS_HISTORY.md`** — the full chronological arc, superseded claims with
  their corrections, and detailed run history. Read it for the "why" behind any
  resolved decision above; do NOT treat its dated claims as current.

## Historical appendix

All chronological findings, superseded headlines (with their correction notes),
pretrain run history (v4 / v5-pkm / v6 / v7.1 / Phase C), the SFT/RL/DPO trajectory,
the WM/thinking arc, and per-date engineering-bug logs have been moved to
**`AGENTS_HISTORY.md`**. This file states only the resolved current state; where an
older doc conflicts with a bullet here, this file wins.
