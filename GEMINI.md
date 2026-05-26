# GEMINI.md / CLAUDE.md — Project Instructions

## Top-level goal — "Small Super-coder"

**Build a small code model that punches above its weight on coding benchmarks (HumanEval / MBPP / long-context recall) under tight compute and memory.** Architectural research in this repo is in service of that target. Every experiment should ladder up to it.

Hard compute constraint: 2× RTX 5090 (32 GB each, no NVLink). Training-token budgets should be planned accordingly — Chinchilla-optimal for a 217 M model is ~4 B tokens; budget accordingly.

## Current architectural stack

Backbone is **DeltaNet** (`--arch deltanet`, plain `chunk_delta_rule` from FLA — efficient linear-RNN, bounded state). Validated lifts to keep:

- **Sparse FiLM feedback (2, 28)**: −3.1 % to −5.4 % PPL at 217 M / 360 M / 708 M (see README headline table). Use `--feedback film --feedback_pairs "2,28" --feedback_self_k 3`.
- **K=3 self-feeding** for FiLM: closes train/inference gap so deployment uses a single forward at 1× decode cost.
- **Bounded working memory** (`experiments/model.py::WorkingMemory`, validated 2026-05-12): write-gated buffer of past hidden states, read at "think" / query positions via soft attention. **+11.1 pp recall** on saturated MQAR (T=512, K=128) vs DeltaNet alone. Read cost O(T·K·d), no O(T²) attention. Enable with `--use_memory --mem_size 1024`.
- **State-readonly thinking** (`--state_readonly_at_think`, plain-DeltaNet only, validated 2026-05-26). Forces the DeltaNet per-token write-gate β to 0 at think positions via a forward-hook on the inner FLA layer's `b_proj` (pre-sigmoid logits clamped to -1e4). Think tokens still READ from the recurrent state (so the local h_t carries useful info for the gate / WM / lm_head) but never WRITE to it — preserves long-range bindings across multi-think bursts, the documented "thinking corrupts recall" failure mode (100% → 20% at distance 512). Default OFF (backwards-compat). Threaded as `think_mask=(input_ids==thinking_token_id)` through `Block.forward → _FlaWrapper.forward`; `b_proj` parameter names are unchanged so existing ckpts load byte-identically. Synthetic 1-layer DeltaNet probe: train with a 4-think burst, eval with 32 thinks → ON 0.88 recall, OFF 0.41 (`experiments/test_state_readonly_thinking.py`). Phase 2 of `THINKING_PLAN.md`. Only the full-sequence forward path is wired; the per-token `forward_step` decode path leaves β unmasked (open follow-up — non-trivial because the cache state would also need a sentinel for "this step is a think").
- **Process-reward auxiliary loss** (Phase A of THINKING_PLAN v5, `--process_reward_weight`, default 0.0 = off, added 2026-05-26). SFT-only aux loss that finally gives the trunk + WM gradient through the *consequence* of thinking. On a sampled fraction of positions where σ(gate) > `--process_reward_apply_min_sigma` (default 0.3), an extra forward over `[prefix, K * THINK_ID]` (K from `--process_reward_K`, default 4) computes log p_after(true_next_token) and the loss is `mean(log p_before - log p_after)` — minimising this pushes thinks to put more probability mass on the truth. Bounded by `--process_reward_sample_frac` × `--process_reward_max_positions`. Helper at `experiments/process_reward.py::compute_process_reward_loss`; wired in `sft_code.py` legacy forward path. **Probe** (`experiments/probe_process_reward.py`) on the SFT base `sft_phase_c_combined.pt` confirms the canonical "thinks aren't productive" baseline: mean Δlogp = -0.165, only 30.5 % of high-gate positions benefit from K=4 thinks. This is exactly what Phase A is trained to flip. **Pad-id MUST differ from `thinking_token_id`** (the helper asserts this) — pad-as-think silently triggers `state_readonly_at_think` / `mem_write_only_at_think` on padding positions, corrupting the after-forward's recurrent state. Caller uses `pad_id=0`. Tests in `experiments/test_process_reward.py` (13).
- **ThinkAdapter** (Phase B of THINKING_PLAN v5, `--use_think_adapter`, default off, added 2026-05-26). Small 2-layer MLP per Block (`d_model → hidden_mult·d_model → d_model`, GELU) + learnable scalar α (init 0). Applied as `h_out = h_in + α · think_mask · ThinkAdapter(h_in)` at think positions only. The trunk now has parameters DEDICATED to thinking-time computation — previously every block ran identical computation at think vs emit positions, only differing by the input embedding. Wired in `Block.forward` AND `TinyLM.prefill` AND `TinyLM._step_block` (the v5 code review caught that the incremental-decode path was bypassing `Block.forward` entirely — fixed by threading `think_mask` through `_step_block`, regression test in `test_think_adapter.py::test_adapter_fires_during_incremental_decode`). Adapter weights + α route to AdamW via `optim_utils._is_think_adapter` predicate (NOT Muon). α init 0 → ckpts trained without it load byte-identical; α init 0 + non-zero MLP weights → α moves first under loss gradient, MLP weights follow once α drifts off zero (FiLM-α pattern). `eval_bracket_structure.build_model_from_ckpt` auto-detects adapter from state_dict; pass `force_use_think_adapter=True` to attach a fresh adapter to a pretrain ckpt during SFT (`sft_code.py --use_think_adapter --think_adapter_hidden_mult 2`). Tests in `experiments/test_think_adapter.py` (11).
- **Soft-mixture decode** (Phase C of THINKING_PLAN v5, `--gate_mode soft`, default `hard`, added 2026-05-26). Eval-only mechanism in `experiments/eval_humaneval.py::generate_soft_mixture`: at each emit step run BOTH branches (emit and think) and mix the resulting probabilities by σ(gate): `p_mix = σ · p_emit + (1-σ) · p_think`, sample from p_mix. The emit-branch FLA cache / WM buffer / FiLM lagged sources / per-row think-run counter are the canonical state; the think-branch runs on a deep-cloned cache that is discarded. Cache-clone helpers `_clone_cache` / `_clone_fla_cache` isolate all mutable state per call. Cost: ~2× per step. **The reason this is novel, not a "closing of train/eval gap"**: at training time the gate is a soft loss-weight (`g·CE_real + (1-g)·λ_ponder`), NOT a soft mixture of emit/think outputs — the think-path computation never enters training-time loss anywhere. Soft-mixture decode is the first time the gate's continuous σ has actually mixed real outputs at any point in the pipeline. Tests in `experiments/test_soft_mixture_decode.py` (8).
- **Per-position think-index embedding** (`--think_index_emb_size N`, default 0 = off, added 2026-05-26). Small zero-init `nn.Embedding(N, d_model)` added on top of the [THINKING] input embedding, indexed by the token's position within its consecutive-think burst (clamped to N-1 for bursts longer than N). Phase 3 of `THINKING_PLAN.md`: directly attacks the multi-think homogenization failure (8 consecutive thinks share one input embedding → median pairwise cos +0.146 vs +0.060 at emit, effective rank ~210 vs ~560; `diag_think_position_diversity.py`). Zero-init means a cold-start / freshly-loaded ckpt is byte-identical to the pre-Phase-3 path; the model opts in via gradient only if the diversity is actually useful. The per-position index is computed via a vectorized cumsum-reset (`c = cumsum(think_mask)`; `reset = cummax(c * (1-think_mask))`; `burst_idx = (c - reset - 1).clamp(0)`) — no Python loop, GPU-friendly. Wired in `TinyLM.forward` AND `prefill` AND `forward_step` (the latter maintains a per-row running-counter in `cache["think_run_len"]` so incremental decoding gets the same index a one-shot forward would). Applied AFTER any caller-supplied `inputs_embeds` (retrieval-as-input mode also benefits — the additive index breaks homogeneity regardless of how the base embedding was built). Tests in `experiments/test_think_index_emb.py` (8: default-off no-param, default-off byte-identical, index assignment, burst clamp, isolated-think → index 0, distinct vectors across burst, state-dict round-trip, non-zero table changes output).
- **Shallow-wide trunk (v6 / v7 / v7.1, validated 2026-05-17–18).** Iso-param swap of the 30L × 576d production trunk for **10L × 896d + 5 dense reverse FiLM pairs (0,5)(1,6)(2,7)(3,8)(4,9) + K=3 self-feed**. Hypothesis: "the brain is shallow with heavy feedback loops" — at 10 layers, gradient paths are shorter, every early layer reads from a late layer (dense fan-in across depth), wider hidden absorbs the freed parameter budget. **Result**: at iso-token, the 10L × 896d trunk matches the 30L baseline on overall VAL CE and trains ~18 % faster in wall-clock (v7.1-film 13.3 h vs v4 16.3 h at the same step budget). Enable with `--n_layers 10 --d_model 896 --n_heads 14 --d_head 64 --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" --feedback_self_k 3`. Sister run available with **cross-layer attention** instead of FiLM (`--feedback_xattn "0:5,6,7,8,9;1:5,6,7,8,9;2:5,6,7,8,9" --feedback_xattn_form film_sigmoid --feedback_xattn_heads 8`); both converge to similar VAL ppl by step 9300 (film 5.83, xattn 5.94, v4 5.89), so the choice is FiLM-cheaper-per-step vs xattn-more-expressive at the same budget.
- **xattn no_grad pass-1 (validated 2026-05-18).** The cross-layer attention forward needs pass-1 source-layer hidden states for pass-2 to attend over. Originally **both passes were grad-enabled** → 2 fwd + 2 bwd = 4 work-units per step (vs FiLM K=3's 3 fwd + 1 bwd = 3 work-units). Profile showed the cost was entirely the 2nd forward, not the xattn modules themselves (film_sigmoid vs scalar-α: −0.8 ms negligible). Fix: wrap pass-1 in `with torch.no_grad()` — mirrors the FiLM K=3 protocol where K-1 warmup passes are no_grad. Source-layer parameters still receive gradient via pass-2 (which also runs the full block stack); pass-1's only function is producing the lagged source states. **Equalises xattn cost with FiLM K=3** (both ~3 units/step). Located at `experiments/model.py` xattn forward branch.
- **Product-Key Memory side-table** (`experiments/memory_layer.py::PKMLayer`). Multi-head learned KV side-table dropped in after a single block: per query, top-k retrieve from `n_heads × n_keys²` slots via factorised sub-key product (default 4 heads × 256² = 262 k effective slots). Value tables route to AdamW (not Muon — table-shaped); `optim_utils._is_embedding_like` handles this. **v5-pkm probe (2026-05-17) showed 97 % of value rows still at random init** after 2.13 B tokens — the original PKM training was broken at our scale; the v5-pkm "wins" were trunk lifts, not PKM lifts. The v7.1 5-fix bootstrap package below makes it actually learn.
- **PKM-v7.1 bootstrap-fix package (`experiments/memory_layer.py`, validated 2026-05-18).** Five interlocking fixes that turn PKM from "97 % dead at 2.13 B tokens" (v5-pkm) into "always-positive per-source contribution" (v7.1-pkm-film, final ppl 5.83 vs v4 final 5.89 = 1 % aggregate lift). Per-source PKM toggle Δ on v7.1-film final: **wikipedia +0.099 CE**, cybernative +0.063, code_exercises +0.046, bigvul +0.044, others +0.01–0.03 — fact-heavy wins largest, exactly as Lample predicted; **never negative on any source**. The fixes:
  - FIX 1: `--pkm_use_output_gate` (default on) — scalar α (init 0) wrapping PKM output. Mirrors the FiLM α curriculum: gradient grows α only as PKM proves useful, instead of forcing a random-init contribution that v5-pkm's training had to fight.
  - FIX 1B: `--pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 2000` — sign-preserving additive floor on α (`α_eff = α + sign(α)·floor`). Forces minimum PKM contribution magnitude during the value-table-bootstrap window. Without this, α grew 0 → +0.085 by step 280 and **collapsed back to +0.04 by step 400** (v7.0 trace, before fix), because random-init values produced structured noise and the optimizer correctly said "use less". With the floor, value rows get meaningful gradient long enough that they learn useful patterns; α then settles at +0.30–0.35 on its own.
  - FIX 2: `--pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 2000` — ε-greedy random slot replacement at training time. With probability ε, each top-k retrieval is replaced by a uniform random slot index — forces *every* slot to receive gradient even before the learned router would have picked it (v5-pkm probe: only ~4 % of slots ever fired; v7.1 hits 65 k / 65 k unique slots per microbatch during warmup).
  - FIX 3: `--pkm_value_init_std 1.0` — init value rows at residual-stream magnitude instead of `1/sqrt(d) ≈ 0.04`. PKM output magnitude scales linearly with init std; the small init in v5-pkm gave residual contribution ≈ 1 %.
  - FIX 4: `--pkm_score_norm layer` (default) — LayerNorm on slot-selection scores, avoids the noisy running-stats of Lample's BatchNorm at our token scale. `--pkm_score_norm batch` reproduces v5-pkm behaviour.
  - FIX 5: `--pkm_diversity_weight 0.01` — auxiliary `-H(slot distribution)` loss penalising peaky retrieval (the "one slot eats 40 % of a head's mass" pattern observed in v5-pkm).
  - **Value-LR multiplier** `--pkm_value_lr_mult 100.0` — separate AdamW group for `pkm_layer.values.*` at 100× the base LR. Compensates for the multiplicative dampening of per-row gradient (`α · w_k · ∂loss` ≈ 10⁴× smaller than other params). Without this, value rows stayed at init even with α-floor active. Implementation in `optim_utils._is_pkm_value` + `build_optimizer(pkm_value_lr_mult=...)`.
  - Live diagnostics logged per `--log_every` step: `pkm(αL=+0.32, αeff=+0.34, row=1.79, slots/H=33k/65k, top=0.003, ε=0.00, φ=0.00)` — `αL` is the learned scalar, `αeff` is the effective (αL + sign·floor), `row` is mean value-row norm / expected-init norm (>1 = rows learning, ≈ 1 = frozen), `slots/H` is unique slots hit per microbatch, `top` is the hottest slot's mass share, `ε` and `φ` are the ε-greedy and floor curricula. **`v_std` is NOT a useful diagnostic** — it is invariant under updates that preserve approximate Gaussian distribution; use `row` instead (it's what the v5-pkm post-hoc probe measured).
  - **PKM-utilization probe**: `experiments/probe_v5_pkm_utilization.py` (post-hoc on any PKM ckpt: row-norm drift from init, per-head slot-concentration, residual contribution magnitude). **Per-source contribution probe**: `experiments/probe_pkm_per_source.py` (toggles αL on/off and reports per-source CE delta on each data-mix source). The contribution-probe is the canonical "is PKM actually pulling its weight?" test — run it on any new PKM ckpt before scoping a follow-up.
- **Output / thinking gate**: per-position sigmoid head deciding emit vs think. Wired in `train_lm.py` (BPTT) and `train_rl.py` (GRPO).
- **Mixed-corpus pretrain + chunk-boundary think-burst injection** (`experiments/data_mix.py`, 2026-05-12/13): 9–11 weighted HuggingFace streams + random think-burst insertion at chunk boundaries so the memory + gate-head get dense supervised gradient from step 0 of pretrain. Targets at think positions are masked to −100 in the loss. Wired via `train_lm.py --data_mix configs/pretrain_mix_v*.yaml`.

Dead direction (do **not** revive without explicit justification):
- **Corpus-RAG** (`rag_projection`, the old in-memory codeparrot DB). Its loss-bearing forward didn't pass `rag_hidden`, so gradients were structurally zero and the projection never trained. Replaced by `WorkingMemory`.
- **`--arch gated_deltanet`** on sm_120 (RTX 5090): FLA's `gated_delta_rule.prepare_wy_repr_bwd` hits CUDA "misaligned address" inside the Triton autotuner during the backward pass. Plain `--arch deltanet` works and is what the validated 217 M ckpts use. Don't try the gated form again on Blackwell consumer cards until upstream FLA has a fix.

## Key mandates

- **Validate architecture on synthetic recall tasks before scaling.** MQAR (`experiments/tasks/mqar.py`, `train_mqar.py`) is the canonical test for "does this help recall". Choose configs *on or above* DeltaNet's saturation threshold (the T=512/K=128 regime), otherwise the mechanism is invisible.
- **Don't run natural-text RL on the existing base.** `checkpoints/think_sweep_fresh_tokens_final.pt` scored **0 / 20 on HumanEval** (8.2 M training tokens, 0.04 tokens/param — way below Chinchilla). Architectural wins drown in the noise of an undertrained base.
- **Memory only belongs in RL or in tasks that explicitly pass `mem_read_mask`.** `WorkingMemory` injects only at positions matching `thinking_token_id`. Pretraining/SFT data does not contain thinking tokens, so the default read mask is empty and the memory module's gradient is **structurally zero** — exactly the failure mode that killed the old corpus-RAG. `train_distill.py` no longer accepts `--use_memory`; `sft_code.py` warns when loading a ckpt that has memory weights. Memory should be added during the RL stage, where rollouts naturally insert think tokens.
- **Thinking-token embedding** must be initialised to the embedding-mean when added — PyTorch's default Linear init puts random noise into the recurrence at every think step. `TinyLM.__init__` does this automatically when `use_memory=True`.
- **`mem_read_mask`** is the right way to drive memory reads at non-think tokens (e.g. MQAR query positions). Pass it as a kwarg to `TinyLM.forward`.
- **Activation checkpointing** (`--think_checkpointing`) is mandatory for `safety_max_depth > 2` in RL / queue-based training.
- **Don't let `--gate_floor_min` reach 0.0 during BPTT pretrain with the thinking gate.** v2 attempt 2 (2026-05-13) showed the maladaptive-thinking trap: with the default curriculum `1.0 → 0.0` over 2 k steps, VAL ppl went 49 → 940 within 2 k steps of floor hitting 0. Symptoms in the log: `emit_ce` stays low (~0.85), overall `ce` and VAL ppl explode together, gate emit-rate stays around 0.27 — the model routes huge probability mass into the think token at every position because `(1-g)·λ < g·CE_real` for hard tokens. Fix: `--gate_floor_min 0.5 --gate_warmup_steps 20000` so the gate is forced to weight real-token loss ≥ 0.5 everywhere. Loss-bearing callsite at `train_lm.py:1969-1986`. For natural RL where the model should learn full thinking autonomy, this trap doesn't apply (no `gate_terms` masking).
- **Block-level activation checkpointing** (`--activation_checkpointing`) wraps each transformer Block's loss-bearing forward in `torch.utils.checkpoint(use_reentrant=False)`. Trades ~30 % extra compute for a ~Nlayers× reduction in stored activations, which lets us push batch significantly higher (**validated target: 14 at T=2048 bf16** vs the current 7; **do NOT exceed 14** with `--feedback_self_k 3` self-feed enabled — see next bullet). Output is bit-identical (no dropout; preserve_rng_state default). Wire-up: `TinyLM(activation_checkpointing=True)`, helpers in `experiments/model.py::_run_block / _ckpt_run_block / TinyLM._block_fwd`, test in `experiments/test_activation_checkpointing.py`. Use it for the *next* pretrain / SFT / RL run.
- **`--batch` × K-self-feed × activation memory** (2026-05-16, learned the hard way twice). K-self-feed runs every block K times in the forward, so peak activation memory scales roughly with K. **At `--batch 20 --activation_checkpointing --feedback_self_k 3` on the 32 GB 5090**: steady-state at K=1 (during `--feedback_self_k_warmup_steps`) fits in ~27 GiB, but the moment K=3 kicks in at step (warmup+1), activations spike +3.76 GiB and the run OOMs. **The b=20 smoke test missed this because it never crossed the K-warmup boundary.** Yesterday's "mystery" 22:11 v4 crash AND today's relaunch attempt were both this. Use `--batch 14` whenever K=3 self-feed will engage (validated steady-state ~25 GiB at K=3); reserve `--batch 20` only for runs that stay K=1 the whole way (e.g. `--feedback_self_k 1` or `--feedback none`).
- **Use real PPL** (i.e. exp(−CE)) for reported numbers; don't include ponder cost in displayed "ppl=" values.
- **Always train with `--bf16 --tf32`** on this hardware. The 5090 fp32 path leaves ~2.3× speed on the table (measured: 18 k → 42 k tok/s at the same batch). bf16 wraps `model.forward` in `torch.autocast`; master weights stay fp32 so Muon/AdamW are unaffected. TF32 covers the residual fp32 matmul. No GradScaler needed (bf16 has fp32 exponent range).
- **Don't apply weight decay to FiLM α** (`--alpha_wd 0.0`). The WD-equilibrium probe (2026-05-13, `experiments/probe_alpha_wd.py` on the v1 500 M-token ckpt) showed |grad_α| ≈ 1.83 × WD·|α| — the gradient consistently wants α higher and the Muon WD=0.1 was the brake, not a true loss-flat ceiling. Run with `--alpha_wd 0.0` and let α find its real equilibrium.
- **Train with `--wd 0.01`, not the legacy 0.1** (2026-05-14). `diag_ckpt` on the v2 mid-eval ckpts found **residual-stream collapse**: ||h||@L0 *shrank* 8.1 → 3.5 between the 500 M and 1 B ckpts, vs SmolLM2-135M (same 30L×576d shape) which has ||h||@L0 ≈ 44 growing 20× over depth. WD=0.1 was holding weights too small → block contributions vanish → the residual stream goes thin-and-diffuse. v3a (`--wd 0.01`) un-collapsed it (||h||@L0 23.5, growing 7×) and beat v2 on **every** per-source CE — including the hard CVE streams v2 was *regressing* on. WD=0.1 is a Moonlight-scale (5.7 T-token) setting; at our ~5–10 tok/param it's pure brake. CLI: `--wd` now **defaults to 0.01** (the validated value); pass `--wd 0.1` only to deliberately reproduce the old Moonlight-scale setting.
- **Use `--lr_schedule wsd` (the default for new runs).** We heavily undertrain and keep extending runs, so cosine's commit-`T_max`-upfront is a poor fit — v3a's 70 k cosine had the LR floored exactly when token counts got interesting, making the 500 M→1 B gain look artificially flat. WSD (warmup → constant peak LR for the bulk → short cosine decay over the last `--lr_decay_frac`, default 0.15) has no wasted low-LR tail and can be stopped anywhere to cash out a checkpoint. `--lr_schedule cosine` kept as the escape hatch (byte-identical to old behaviour). Implemented in `optim_utils.py::_wsd_lambda`.
- **Do NOT use LayerDrop / Stochastic Depth** (`--layer_drop_max`, default 0.0 = off). v3b (`--layer_drop_max 0.2`, linear 0→0.2 across depth) was a clean negative result: it *did* what Stochastic Depth promises — logit-lens saturates ~5 layers earlier, depth utilisation redistributed — but it made L25–L29 vestigial and cost ~0.02–0.10 CE on **every** source vs the WD-only v3a. It "fixes" depth-concentration, but depth-concentration was never the bottleneck (SmolLM2 has the identical lens shape and trains fine). The flag stays in the code as a tested, working option; just don't enable it.
- **GRPO ponder cost shaping** (`experiments/thinking.py::compute_grpo_advantages`) supports four orthogonal knobs, all backwards-compatible defaults: `--grpo_ponder_shape {linear,quadratic}`, `--grpo_ponder_counterfactual` (clamps task component at depth-0 baseline so thinking can never *worsen* the task reward but the depth cost still always applies), `--grpo_separate_ponder_norm` (z-scores task only and subtracts absolute ponder after, fixing the wash-out where GRPO group-norm squashes the small ponder magnitude into CE noise), `--grpo_ponder_warmup_steps N` (curriculum ramping ponder cost 0→full over N steps to avoid cold-start collapse). Recommended Phase-C config: `--grpo_ponder_shape quadratic --grpo_ponder_counterfactual --grpo_ponder_warmup_steps 300`.
- **Grader-RL (execution-grounded GRPO)** (`experiments/train_rl_grader.py`, added 2026-05-16). GRPO with `code_grader.grade()` as the reward — runs N rollouts per MBPP problem at temperature τ, grades each subprocess, computes advantages from the (B, N) dense reward tensor, PPO clipped policy update. Distinct from `train_rl.py` (prediction-as-reward on codeparrot): this trains the model on whether the emitted code actually *works*, the only signal that can teach productive thinking. **First validated lift**: 1 pass / 16 rollouts at step 14 → 14 cumulative passes in 82 steps; `depth_mean` dropped from ~25 to <20 once `--ponder_warmup_steps` completed (gate became selective rather than always-on). Defaults: `--batch 4 --grpo_n_group 4 --max_gen 96 --temperature 0.9 --gate_floor 0.0 --emit_threshold 0.5 --ponder_cost 0.005 --ponder_shape quadratic --counterfactual --ponder_warmup_steps 50 --lr 5e-6 --clip_eps 0.2`. Run with `--smoke` for a 2-step end-to-end smoke. **Perf**: rollouts and policy-update are batched (Phase D + E shipped). State-passing incremental decoding (Phase A foundation in `_FlaWrapper.forward_step`) is shipped but not yet wired into `TinyLM.forward_step` — FiLM K=3 self-feed + `WorkingMemory` buffer make true per-token incremental decoding non-trivial; deferred.
- **Stochastic gate as a policy variable** (`--stochastic_gate`, default off, added 2026-05-26 to `train_rl_grader.py`). Replaces the deterministic `gate ≥ emit_threshold` rollout decision with a Bernoulli draw from `clamp(sigmoid(gate), gate_floor, 1-eps)` so the gate becomes an exploratory RL action — same group-relative advantage that rewards the emit-token PPO also rewards the gate's choices (separate per-rollout-mean gate-PPO surrogate added to the loss). `force_emit` positions (think-budget exhausted / finished rows) are EXCLUDED from the policy gradient since the model had no choice. Per-rollout `Rollout.gate_decisions/gate_log_probs/gate_positions` are recorded for the offline PPO ratio; the new-policy gate log-prob is extracted from `model._last_gate` (DDP-safe via `.module` lookup) at `gate_positions − 1`. `--gate_entropy_bonus` (default 0.01) adds a Bernoulli-entropy regularizer that prevents collapse to never-think / always-think; subtracted from the loss so higher entropy → lower loss. Logs `gate(fire=..., H=..., ratio=...)` when on. The architectural pivot: the optimal thinking pattern is unique to FiLM K=3 + WM + retrieval-as-input and not whatever Qwen's CoT looks like, so let RL DISCOVER it instead of imitating. Tests in `experiments/test_stochastic_gate.py` (10).
- **Mid-eval ckpt-save robustness** (added 2026-05-16). When the trainer is GPU-pressured (typical mid-training: 25-30 GiB / 32 GiB on the 5090), the HumanEval *subprocess* spawned by mid-eval would OOM trying to load its own copy of the model on the same GPU. Worse, in v4's first run the user restarted with `--load_ckpt` and forgot to re-pass `--mid_eval_every_tokens`, losing 4000 steps of resume points. Two fixes: (a) `--mid_eval_save_only` flag explicitly skips the HumanEval subprocess and just saves the ckpt + advances the counter; (b) `--mid_eval_min_free_gib` (default 2.0) auto-skips if `torch.cuda.mem_get_info()` shows less than that much free GiB. The ckpt save happens BEFORE the subprocess regardless, so resume artifacts always land. Production runs should pass `--mid_eval_save_only` — HumanEval during pretrain is noisy and the ckpts are the load-bearing thing. Tests in `experiments/test_eval_callback.py`.
- **Pretrain knobs (`--grad_accum`, `--grad_clip`, `--z_loss`), added 2026-05-14** from the "techniques we should be using" review. (1) `--grad_accum N` accumulates N microbatches per optimizer step on the **non-thinking (pretrain) path only** — the thinking-token queue path has its own `--think_queue_accum_steps`, and `--grad_accum > 1` errors if combined with `--enable_thinking_token`. The v3-and-earlier effective batch was 7×2048 = 14 k tok/step, far below the 0.5–4 M typical for pretraining; the tiny-batch gradient noise was a real convergence-efficiency drag. Use `--activation_checkpointing --batch 14 --grad_accum 8` for ~229 k tok/step. The non-thinking forward+loss is factored into `train_lm.py::_nonthink_forward_loss`; `tokens_seen` and `tok/s` both multiply by `grad_accum`. (2) `--grad_clip` (default 1.0) exposes the previously-hardcoded global grad-norm clip; 0 disables. (3) `--z_loss` (**defaults to 1e-4**; 0 disables) adds `weight·mean(logsumexp(logits)²)` — the PaLM/Chinchilla logit-drift regulariser, cheap bf16-stability insurance. Tests in `experiments/test_pretrain_knobs.py`.
- **Cross-document state isolation — implemented 2026-05-14.** `data_mix.py` packs multiple documents into each T=2048 sequence separated only by an EOS *token*; DeltaNet is a linear RNN so its recurrent state (and `WorkingMemory` reads) would otherwise flow straight across those boundaries. Fix: `MixedSourceStream(emit_doc_ids=True)` emits a per-position `doc_id` array (kept aligned through `insert_think_bursts` via its new optional `aligned=` arg); `train_lm.py` threads `doc_ids` into `TinyLM.forward`; `model._build_cu_seqlens` turns it into a ragged `cu_seqlens` (int32) that `_FlaWrapper` feeds to FLA's `chunk_delta_rule` after flattening `(B,T,d)→(1,B*T,d)` — verified in the local fork, fwd+bwd. `WorkingMemory.forward` gained a same-document read mask. `doc_ids=None` (any non-`data_mix` stream, eval, MQAR) is byte-identical to the old path — only `DeltaNetAttention` opts in (`accepts_cu_seqlens=True`). Tests: `experiments/test_cu_seqlens.py` (8, incl. the packed==unpacked CUDA equality test). The `doc_id` representation is also what FIM (#65) reuses. Full design: `CROSS_DOC_ISOLATION_PLAN.md`. **Open follow-up**: the RL replay-packing path still passes `doc_ids=None`; the FiLM `_shift_right` lag is not doc-aware (second-order, documented non-fix).

## Documentation index

- `MILESTONE_ARCH.md` — north-star milestone: PKM, WM, thinking gate (+ long chains), and selective memory writes must each be measurably load-bearing. Current status, ablation table, and workstreams (A–D) to get there.
- `README.md` — headline results (FiLM lift, MQAR recall, deployment-fair scoring) and the small-super-coder framing.
- `THESIS.md` — project framing: small-model performance is under-served by training-methodology research; what we claim and what we explicitly don't.
- `PHASE_C_RL.md` — prediction-as-RL-signal proposal for post-SFT. Multiplicative reward of correctness × prediction-match; includes the ponder-cost shaping infrastructure ready to use.
- `PLAN.md` (or `/home/knielsen/.claude/plans/noble-beaming-beacon.md`) — active and near-term experiments.
- `NEXT_DIRECTIONS.md` — strategic roadmap + queued ablations (cross-attention vs scalar-α, no-WD-on-α follow-up).
- `THINKING_RAG_DIRECTION.md` — mechanistic rationale for the thinking head (still valid for the gate; the corpus-RAG section is historical).
- `RL_RAG_ROADMAP.md` — superseded by the working-memory direction; keep for history.
- `SESSION_FINDINGS.md` — chronological log of empirical results.
- `HANDOFF.md` — session-to-session handoff notes; append the current state at end of each session.

## Conventions

- Python env via `uv`, torch nightly cu132 (`...whl/nightly/cu132`).
- **FLA must be the local fork** at `~/ml/flash-linear-attention` editable-installed into the `.venv`. The pip-published `flash-linear-attention 0.5.1` is missing the Blackwell `global_scratch` allocator fix (commit `8b05e2f1` upstream) and our in-progress `cache.py` restore_value patch. To set up a fresh venv: `rm -rf .venv/lib/python3.12/site-packages/fla` then `uv pip install -e /home/knielsen/ml/flash-linear-attention`. Verify with `python -c "import fla; print(fla.__file__)"` — should resolve to `~/ml/flash-linear-attention/fla/__init__.py`.
- Run scripts in this repo expect `export PYTHONPATH=$PYTHONPATH:.` before invoking `experiments/*.py`.
- Save checkpoints with `{state_dict, step, config}` dict; load with `weights_only=False` (the `config` contains class references).
- `experiments/eval_bracket_structure.py::build_model_from_ckpt` auto-detects `gate_head` and `memory.*` in the state-dict and constructs the model with matching kwargs; reuse it for new evals rather than hand-rolling.
- **Tests live under `experiments/test_*.py`** and are pytest-discoverable. **122 tests** covering `data_mix`, `thinking` (GRPO advantages), `eval_callback`, `sft_code` (think-burst insertion), `activation_checkpointing` (bit-identical fwd/bwd), `eos_mask`, `code_grader` (dense tier ladder + `error_text`), `pretrain_knobs` (grad-accum gradient-equivalence + z-loss + gate-entropy-aux BCE direction + flags), `cu_seqlens` (cross-document isolation: doc_ids emission, `_build_cu_seqlens`, packed==unpacked DeltaNet equality, WorkingMemory doc masking), and the **PKM-v7 + v7.1 mechanics** (`test_memory_layer.py`: output-gate-α init / learns, ε-greedy slot replacement + eval-mode-inactive, value-init std configurable, LayerNorm score-norm, sign-preserving α-floor, value-LR-mult creates separate optim group). Run with `PYTHONPATH=. .venv/bin/python -m pytest experiments/test_*.py -v`. Note: the CUDA tests OOM if co-resident with a training run on the same GPU — pin to a free GPU with `CUDA_VISIBLE_DEVICES`. Add a test when you add a new builder, filter, or shaping mode.
- **Trainer module layout** (`experiments/train_lm.py` was split for readability, 2026-05-13):
  - `experiments/train_lm_args.py` — `build_parser()` returns the full argparse. CLI surface lives here.
  - `experiments/build_arch.py` — `build_arch()` / `parse_layers_arg()` + the `_NAME_TO_CLS` attention registry. Extracted from `train_lm.py` so `model_builder.py` can import without depending on the trainer (2026-05-14; the import was a latent bug — `model_builder` imported a module that didn't exist yet, only masked because the running v2 process had cached the old in-process definition).
  - `experiments/model_builder.py` — `build_model_from_args(args, vocab_size, thinking_token_id)` constructs the `TinyLM` and returns `(model, ModelBuildInfo)`. Reusable from eval/SFT scripts.
  - `experiments/optim_utils.py` — `build_optimizer(model, optimizer=..., lr=..., lr_muon=..., alpha_wd=..., steps=..., wd=..., lr_schedule=..., warmup_steps=..., decay_frac=...)` returns `(opts, scheds)`. Handles Muon + AdamW split, FiLM-α param-group split, and the cosine-vs-WSD scheduler choice.
  - `experiments/speed_knobs.py` — `apply_speed_knobs(model, bf16=True, tf32=True, compile_model=False)` does the bf16-autocast wrap + TF32 + optional `torch.compile`. Call AFTER model construction.
  - `experiments/model.py::TinyLM._finalize(h, input_ids, mem_read_mask, return_aux, return_hidden, return_gate, maybe_gate, extra=())` — shared exit-tail for every forward branch. Don't duplicate `out_norm → memory → lm_head → packing` in a new branch; call `_finalize` instead.
- **Diagnostic tools** (built 2026-05-14, all CPU-or-single-GPU, safe to run on any ckpt):
  - `experiments/diag_ckpt.py` — per-layer logit-lens CE, hidden effective rank, ||h|| + Δh/||h_prev||, and per-source held-out CE. The tool that found the residual-stream collapse. `run_diag(...)` is importable.
  - `experiments/diag_reference_lm.py` — same metrics for a HuggingFace causal LM (SmolLM2-135M is the matched-shape reference). Use it to tell "our run is broken" from "this is normal for the architecture".
  - `experiments/profile_train.py` — `torch.profiler` harness on the real train config; A/B flags `--compile` / `--film_bypass`. Needs a dedicated GPU (don't run co-resident).
  - `experiments/profile_v6_xattn.py` — narrow profile for the cross-layer-attention 2-pass vs 1-pass overhead. Decomposes the cost into "2nd-pass forward" vs "xattn module compute" — the 2026-05-17 run on this confirmed the second forward is ~98 % of the overhead and the xattn modules themselves are nearly free, motivating the no_grad-pass-1 fix.
  - `experiments/probe_v5_pkm_utilization.py` — post-hoc on any PKM ckpt: per-head value-row magnitude vs init, slot-hit concentration (top-1 / top-10 / top-100 / top-1000 shares per head), residual contribution `‖pkm(h)‖ / ‖h‖`. The tool that discovered v5-pkm was 97 % dead.
  - `experiments/probe_pkm_per_source.py` — runs every (model, α-on/off) pair on a held-out batch from each data-mix source; reports per-source CE matrix + Δ per source. Canonical "is PKM contributing usefully?" test.
  - `experiments/probe_pkm_contribution.py` — same idea but stratified by per-token CE bucket (does PKM help at uncertain positions or uniformly?).
  - `experiments/inspect_v5_pkm.py` — greedy decode with per-token gate / WM-read / PKM-slot capture, dumps a JSON for plotting. Useful for behaviour debugging.
- **Per-layer learning diagnostics are logged every `--log_every` steps** (default-on, no flag). Console: `gnorm(L0,Lmid,Llast,last/first)` + `uratio(...)`; TB: `layer_grad_norm/L**`, `layer_update_ratio/L**`, `layer_grad_norm/last_over_first`. Helpers `_block_grad_norms` / `_block_weight_snapshot` / `_block_update_ratios` in `train_lm.py`. Read `last_over_first`: ≫1 → early layers gradient-starved (vanishing-gradient signature → deep supervision worth trying); ≈1 → gradient healthy, "slow early layers" was a logit-lens misread, bottleneck is tokens/data.
- **`--compile` is default-on** (`apply_speed_knobs`, `torch.compile(model.forward, fullgraph=False)`). FLA Triton kernels are graph breaks; compile fuses the PyTorch glue between them. **Validated 2026-05-14** via `profile_train.py`: +10 % (343.7 → 312.3 ms/step), no errors on the nightly-torch/FLA/Blackwell stack. `--no-compile` is the escape hatch.
- **FiLM K-self-feed warmup curriculum** (`--feedback_self_k_warmup_steps N`, default 0 = off): before step N the forward takes the plain 1-pass block loop (`TinyLM._film_bypass`), skipping the multi-pass K=3 overhead while early gradient is just noise; after N it runs the configured `--feedback_self_k`. The bypass path *is* the well-tested `feedback_mode=="none"` path. **Measured 2026-05-14**: the bypass is **+40 %** (343.7 → 245.9 ms/step); stacks with `--compile` to **+57 %** (219.0 ms/step, 65 k tok/s). Use a generous warmup — the bypass is the bigger speed lever of the two.
- **bf16 optimizer state** (`--bf16_optim_state`, default off, added 2026-05-15): stores AdamW `exp_avg`/`exp_avg_sq` and Muon `momentum_buffer` as bf16 instead of fp32. Math runs in fp32 (lift state → step → cast back), so it's lossless in the precision sense — validated against stock AdamW/Muon on a small DeltaNet (max |Δ_loss| < 0.005 over 200 steps with grad_accum=8 + bf16 autocast). Saves **~550 MB persistent** on the 218 M v4 model (227 MB AdamW + 322 MB Muon). Implementation: `experiments/bf16_optim.py::BF16StateAdamW / BF16StateMuon`; tests in `experiments/test_bf16_optim_state.py`. **Does NOT save peak GPU memory from gradients** — `bench_bf16_grad_memory.py` showed `grad_dtype=bf16` only halves persistent grad storage; backward still allocates fp32 intermediates regardless. For a peak-mem win on grads you'd need bf16 master weights with stochastic rounding (open follow-up).
- **Strict compile (no silent fallback)** — `apply_speed_knobs(..., compile_model=True)` sets `torch._dynamo.config.suppress_errors = False` (added 2026-05-18). Without this, `torch.compile` install succeeds, then on the first compile failure dynamo silently reverts that frame to eager and prints a warning that's easy to miss — multiple runs in this repo have shipped at production speed because of that footgun. With the strict flag, compile errors throw at install/first-call so we notice immediately. `--compile` is on by default; the strict-mode is unconditional whenever compile is requested. Validated by the v6-xattn run (2026-05-17), which DOES compile cleanly under strict mode despite the two-pass + dynamic ModuleDict dispatch.
- **Entropy-grounded gate target (`--gate_entropy_aux_weight`, validated 2026-05-17).** Auxiliary BCE loss supervising the output-gate logit with a predictive-uncertainty target derived from the same forward's logits (detached): `target_t = exp(-H_t / T)` where `H_t` is per-position next-token entropy. Confident position → target ≈ 1 → gate trained to emit; uncertain position → target ≈ 0 → gate trained to think. **No extra forward, no extra compute** — just turns the existing gate logit into a position-grounded uncertainty head. Use `--gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0`. Implementation in `train_lm.py::_nonthink_forward_loss` (the BCE term is added to total loss). Tests in `experiments/test_pretrain_knobs.py`. Used in v6 + v7 launchers as a cheap fix for the "SFT collapses gate to always-emit" failure mode observed on v5-pkm.
- **`torch.compile` on the optimizer step** (`BF16StateAdamW(compile_step=True)`, default on, added 2026-05-16). The per-param AdamW update body (bf16→fp32 cast → lerp/addcmul/sqrt → cast back) is factored into a module-level `_adamw_param_step(...)` and compiled. Per-step scalars (`wd_scale`, `neg_step_size`, `bias_c2_sqrt_inv`) are passed as 0-dim tensors so compile doesn't specialize on their values and re-trace every step; the per-group constants (`beta1`, `beta2`, `eps`) stay as Python floats. First-step compile cost is ~1-2 s per distinct param shape (~20 shapes for the 218 M v4 model → ~20-30 s one-time). Cuts the optim-step kernel-launch count from ~4500/step (446 params × ~10 eager ops each) down to roughly one fused kernel per shape class. Validated bit-identical to the eager path (max |Δ_param| < 1e-5 over 20 steps; see `test_compiled_adamw_matches_eager_adamw`). **Muon is left eager** — Newton-Schulz is matmul-bound, the per-param Python overhead is small relative to NS compute, and compiling around the upstream `_zeropower_via_newtonschulz` (in-place ops + fixed iteration count) is fragile. Kill switch: `BF16StateAdamW(..., compile_step=False)`.

## Halt-after-docstring trap (pretrain artifact, 2026-05-13)

Mixed-corpus pretrain with EOS at small-document boundaries teaches the model `"""\n → EOS` (most short docs end with a docstring + body + EOS). HumanEval prompts end with `"""\n    ` → model halts on the first emitted token. Three layers of fix, in increasing cost:

- **Inference, already shipped**: `--min_emit_before_eos N` in `eval_humaneval.py` (and `eval_callback.run_humaneval`) suppresses EOS for the first N emitted tokens. `train_lm.py` defaults `--mid_eval_min_emit_before_eos 30`.
- **Inference, also shipped**: `--gate_floor F` in `eval_humaneval.py` mirrors the train-time `g.clamp(min=gate_floor_min)`. Without it, σ(gate) at deploy can land arbitrarily in [0, floor], systematically below the inference `emit_threshold` because training made the model indifferent to those gate values. `train_lm.py` auto-passes `--gate_floor=args.gate_floor_min` at mid-eval.
- **Data, shipped not yet on by default**: `--mask_eos_in_targets` in `train_lm.py` (consumed by `data_mix.py`) sets targets equal to eos_token_id to -100 so the model never gets a gradient on predicting EOS. Enable for v3+.
- **Data, queued (Task #65)**: FIM augmentation in `data_mix.py`. Highest-leverage real fix per StarCoder/CodeLlama recipe; do after EOS-mask data run validates.

## Execution-grounded RL reward (Phase-C prep, 2026-05-14)

`experiments/code_grader.py` returns a **dense** `GradingResult`, not binary pass/fail — so GRPO groups have advantage variance before the model can fully solve a task. Tier ladder + `score ∈ [0,1]`: `syntax_error` 0.0 < `exec_error` 0.05 < `runtime_error` 0.2 < `partial` 0.2 + 0.7·(n_passed/n_tests) < `pass` 1.0. `_exec_target` compiles, exec's the solution, then AST-splits `check()` and runs it statement-by-statement so one failing assert doesn't mask the rest. `GradingResult.error_text` carries the formatted diagnosis (SyntaxError + line, exec traceback, failed-assert sources) — the feedback the **iterative self-repair loop** will put back into a re-added prompt so the model learns to diagnose. Do **not** use embedding-similarity-to-reference as a reward: hackable, no single target, code embeddings barely track correctness — lean into the verifier. Tests: `experiments/test_code_grader.py` (15).

## Current state (2026-05-23)

Headline (full arc): SFT v7 8/164 → Phase C SFT 10/164 → grader-RL v2
step-300 **16/164 = 9.8 % HumanEval pass@1**, project best. Phase C
pretrain is the new architectural baseline (5.28 B tokens,
Chinchilla-complete for 287 M, strict per-source CE win over v7.1
pretrain). RL v2 added the long-overdue KL-to-reference penalty to
`train_rl_grader.py`. PyTorch upstream fix for the AOTAutograd
ViewMeta-replay segfault is on a separate branch at `~/ml/pytorch`,
PR pending. See the "Phase C pretrain + grader-RL trajectory
(2026-05-21..23)" section below for the full story.

Pretrain history (kept for context):

## Pretrain run history (2026-05-17..18)

- **v4** (30L × 576d + FiLM(2,28) K=3 + WM + thinking-gate, `launch_pretrain_mix_v4.sh`, 9300 steps / 2.13 B tokens) — completed 2026-05-17. **Final VAL ppl 5.89**. This is the deep-trunk baseline.
- **v5-pkm** (30L × 576d + PKM-v5 + FiLM(2,28) K=3, `launch_pretrain_mix_v5_pkm.sh`) — completed ~step 4500 (ppl 7.02). Looked like a token-efficiency win in early reading, but the 2026-05-17 PKM-utilization probe revealed **97 % of the value table was still at random init** with 4 % slot coverage and 1 % residual contribution magnitude. The "win" was the trunk; PKM was structurally inert. Triggered the v7.1 PKM-bootstrap-fix package above. **Don't reuse this PKM training recipe** — use the v7.1 settings.
- **v6-shallow** (10L × 896d + 5 dense FiLM, `launch_pretrain_mix_v6_shallow.sh`) — killed at step ~2800 to make GPU room for v7. At iso-step it was tracking 5–7 % ahead of v4 on VAL ppl. The shallow-wide hypothesis was validated.
- **v7.1-pkm-film** (10L × 896d + 5 dense FiLM K=3 + PKM-v7.1 + WM + entropy-aux gate, `launch_pretrain_mix_v7_pkm_film.sh`, 9300 steps) — **completed 2026-05-18, final VAL ppl 5.83**. Strongest pretrain we've produced. **Beats v4 final (5.89) at the same step budget in 18 % less wall time (13.3 h vs 16.3 h).** PKM live diagnostics at end of run: αL=+0.346, row=2.542 (155 % drift from init), 40 k active slots, αeff=+0.346. Per-source PKM-toggle Δ on this ckpt: always positive, largest on wikipedia (+0.099), cybernative (+0.063), code_exercises (+0.046), bigvul (+0.044). Ckpts: `checkpoints/pretrain_mix_v7_pkm_film.pt` + 4 mid-eval ckpts at 500 M / 1 B / 1.5 B / 2 B tokens.
- **v7.1-pkm-xattn** (sister run, 10L × 896d + cross-layer attention + PKM-v7.1, `launch_pretrain_mix_v7_pkm_xattn.sh`, 9300 steps) — **completed 2026-05-18, final VAL ppl 5.94**. Slightly worse than film for ~2.4× the per-step compute pre-fix; the no_grad-pass-1 fix landed mid-flight and equalised step cost. The two runs are within VAL-ppl noise; **film is the cheaper-per-step default**.
- **Persistent unsolved problem**: pretrain-only HumanEval is 0/50 on every ckpt we've ever scored (v5-pkm, v6, v7.1, distilled). VAL ppl 5.83 vs 5.89 is a real but small win; **the actual capability bottleneck is post-pretrain (SFT + execution-grounded RL)**. The bigvul/cybernative *upward drift* across 500 M → 1 B was the v4-mix lever; v6 / v7.1 still see this trend but the shallow-wide trunk's overall trajectory absorbs it.
- **Next planned**: SFT v7.1-pkm-film on a larger curated (problem, solution) corpus than the 10 M-token MBPP+CodeAlpaca we tried before → execution-grounded RL via `train_rl_grader.py` (validated lift on v5-pkm-SFT base, ready to use). The thinking-gate failure mode observed on v5-pkm-SFT (`mean gate at emit = 0.978, think_rate = 0.000` after SFT) is the open problem alongside data scale.

## Distillation + thinking-token findings (2026-05-19)

After v7.1 pretrain hit the 0/164 HumanEval ceiling, the post-pretrain
pipeline was rebuilt. Key results below.

### What worked: distillation + future-emb prediction + synthetic memory
**Combined SFT v1** (`checkpoints/sft_v7_pkm_film_combined.pt`,
`launch_sft_v7_combined.sh`) → **HumanEval pass@1 = 11/164 (6.7 %)** — first
non-zero result on the codebase. The lift came from three stacked changes
vs. the prior `sft_v7_pkm_film_thinking.pt` (0/164):

  1. **Qwen 3.6 AWQ distillation** (`experiments/distill_solutions.py`,
     vLLM-driven, ~38 k (problem, CoT, code) pairs from MBPP/LeetCode/
     Magicoder/CodeFeedback). Replaces the prior 10 M-token codeparrot
     distill. The student learns Qwen's reasoning prose around the code.
     New code-grader loaders: `mbpp_all`, `mbpp_plus`, `mbpp_combined`,
     `leetcode`, `super_combined`, `magicoder_oss`, `codefeedback`,
     `distill_corpus`. **Critical eval flags for distilled ckpts:**
     `eval_humaneval.py --prompt_style sft_comment --extract_code_block`
     (prepends `# Complete the following Python function.\n` to match the
     SFT format and pulls the ```python``` fence out of the model output
     before grading; without these, scoring is structurally 0).
  2. **Future-embedding prediction auxiliary loss** (added to
     `sft_code.py` as `--future_emb_loss_weight` / `--future_emb_T_max`
     / `--future_emb_T_ramp_frac`). At position t, a small `Linear`
     head predicts the input embedding at t + T_eff via `1 - cosine`,
     with T_eff ramping 1 → 8 over the first 30 % of training. Forces
     position-t representations to encode high-level structure of what
     comes next (algorithm choice, library imports) — directly attacks
     the early-commitment failure mode (model emits `return 0.5` because
     the docstring example showed 0.5).
  3. **Synthetic memory-required tasks**
     (`experiments/gen_synthetic_memory_tasks.py`, 12.5 k examples
     across 5 families: var_binding, chain_arithmetic, list_index_recall,
     dict_lookup, multi_step_arithmetic). Each problem sets a value
     early, has distractor lines, then asks for it — forces the model to
     either use WM or attend long-range.

  All three landed in `launch_sft_v7_combined.sh`. Combined SFT v2 (same
  recipe + FIX A active during training) scored **9/164 — strictly
  worse** than v1.

### What didn't work: FIX A (write-only-at-think)
`WorkingMemory(..., write_only_at_think=True)` masks the per-position
write-gate to a large-negative value at non-think positions before the
top-K buffer-selection topk, forcing the WM buffer to contain only
think-position content. Motivated by `diag_thinking_machinery.py`'s
finding that the write-gate was uniform across think/emit despite
WM-reads being sharper at think.

  - Inference-only A/B on the existing distilled ckpt (small sample):
    **1 vs 3 passes** in favor of FIX A.
  - SFT'd from pretrain WITH FIX A (combined v2): **11 → 9 pass@1**,
    LOSS of 2.
  - Root cause: the `[THINKING]` token has ONE learned input embedding.
    Burst of 8 consecutive thinks → 8 hidden states driven by the same
    input → high pairwise correlation (median cos +0.146 vs +0.060 at
    emit) → think-position effective rank ~210 vs ~560 at emit (see
    `experiments/diag_think_position_diversity.py`, 2026-05-19). FIX A
    fills the WM buffer with this low-rank pool; sharp read queries can
    only retrieve from a homogeneous information manifold.
  - **Don't enable `--mem_write_only_at_think` for further work** —
    the flag survives as code (with its own tests) but the empirical
    answer is no. Solve the homogeneity problem at the source instead
    (next section).

### What we're building next: retrieval-as-input thinking tokens
Instead of appending a discrete `[THINKING]` token (homogeneous input
embedding), the gate's "think" decision triggers a WM lookup whose
**retrieval result IS the next position's input embedding**, bypassing
the embedding table. Each think step now gets a unique input signal —
different queries → different retrievals → diverse hidden states →
diverse buffer over multiple thinks. This is the architectural fix
suggested by the user (2026-05-19) and the diagnostic.

  Plumbing already shipped:
  - `TinyLM.forward(... inputs_embeds: Tensor | None = None)`. When
    set, the embedding-table lookup is bypassed and the provided
    embeddings drive the trunk.
  - `WorkingMemory.forward` stashes `self._last_injection`
    (`(B, T, d_model)`, detached) — the per-position retrieval result
    PRE-masking. The generate loop reads this at think positions and
    uses it as `inputs_embeds[t + 1]`.
  - `eval_humaneval.generate_with_retrieval_as_input(...)` is the
    drop-in replacement for `generate(use_thinking=True, ...)` that
    implements the new mechanism.

  Training the new mechanism end-to-end is the next step. For now the
  generator works at inference time only (the SFT ckpt didn't see
  retrieval-as-input during training).

### WM gist supervision — retrieval-as-input is the lever (2026-05-20)
Two SFT runs trained the WM mechanism end-to-end:
- **v5** (`launch_sft_v7_combined_v5_wm_load_bearing.sh`):
  retrieval-as-input + "Option A v5" direct WM supervision targeting
  `embed(input_ids[t+4])` — the input embedding of one token 4
  positions ahead — + long-context recall data.
- **v6** (`launch_sft_v7_combined_v6_gist.sh`): identical recipe, but
  the Option-A target swapped to the **multi-horizon hidden-state
  gist** (see below). Everything else held fixed to isolate the
  target change.

**HumanEval re-ablation results** (full 164, retrieval-as-input
generator, `ablate_memory_mechanisms.py`):

| ckpt | baseline | wm_off | pkm_off | both_off |
|---|---|---|---|---|
| v1 (no retrieval-as-input) | 11 | **+1** (decorative) | −6 | — |
| v5 (lexical target)        | 10 | **5 (−5)** | 3 (−7) | 2 (−8) |
| v6 (gist target)           |  9 | **4 (−5)** | 1 (−8) | 3 (−6) |

**The finding: retrieval-as-input — not the supervision target — is
what made WM load-bearing.** v1's `wm_off` was +1 (WM decorative);
v5 *and* v6 both drop −5 (−50 %). But v6's gist target produced **no
measurable change over v5's lexical target** — headline flat (9 vs
10/164, v6 marginally worse, within noise), `wm_off` delta identical
(−5). So the gist-vs-lexical hypothesis is **not supported by the
HumanEval ablation**.

**Two reasons the HumanEval ablation cannot see a gist effect, both
real:**
1. **Confound.** In retrieval-as-input mode the WM injection *is* the
   think-token input embedding. `ablate_memory_mechanisms.py` ablates
   by zeroing `memory.W_proj.weight` → think tokens get a *zero* input.
   So `wm_off −5` partly measures "think mechanism broken", not
   "retrieved content useless". **Fix: ablate by replacing the WM read
   with a mean vector, not zero** (so think tokens still get *a*
   signal) — `eval_longctx_recall.py --wm_ablate mean`.
2. **Wrong probe.** HumanEval problems are short — they fit inside
   DeltaNet's recurrent state, so long-range memory is never needed.
   WM's actual value (long-range recall) is structurally invisible
   here; all headline numbers sit at 9–11/164 regardless. The right
   probe is **held-out long-context recall** (`eval_longctx_recall.py`,
   `data/longctx_recall_heldout.jsonl`) — that's where gist-vs-lexical
   can actually differ.

### Long-context recall eval — thinking corrupts recall (2026-05-20)
`eval_longctx_recall.py` on a 600-task held-out set (6 distance
buckets, 64–768 tokens; the 768 bucket exceeds the 1024-trained
context and is skipped). var_binding tasks: bind `x = N` at the top,
distractor, `print(x)`. Per-distance recall accuracy:

| distance | v1 (plain think) | v5 (lexical) | v6 (gist) |
|---|---|---|---|
| 64  | 100% | 99%  | 99% |
| 128 |  95% | 100% | 99% |
| 256 |  87% | 100% | 90% |
| 384 |  68% | 100% | 75% |
| 512 |  20% | 100% | 61% |
| **overall** | **74.0%** | **99.8%** | **84.8%** |
| **think_rate** | **0.36** | **0.012** | **0.23** |

**The finding: thinking corrupts long-range recall, and the damage
scales with think volume — not with the injection mechanism.** v1
(plain `[THINKING]` token, no retrieval-as-input) is the WORST (20% at
distance 512) and thinks the most (0.36). v5 thinks almost never
(0.012) and is perfect. v6 is in between on both. A `wm_off` ablation
of v6 (zero `W_proj` → think tokens get a *zero* input) was *worse*
than v6 baseline — so it is not the retrieval *content* that corrupts,
it is that every think token steps the DeltaNet recurrence and
perturbs the precise binding the linear-RNN state was carrying.
**Corollary: a gist and a precise pointer are in tension** — and more
deeply, *any* think token in a linear-RNN recurrence is a recall risk.

### v7 — additive α-gated injection + trunk gist (2026-05-20)
`launch_sft_v7_combined_v7_additive.sh`. Two fixes:

- **Fix B — additive α-gated retrieval injection.** retrieval-as-input
  (v5/v6) *replaced* the think-token embedding with the WM retrieval.
  v7 *adds* it: `input[think] = think_embed + α·retrieval`, α a learned
  scalar `TinyLM.retrieval_input_alpha` (init 0.1, no weight decay —
  the FiLM-α lesson). A useless retrieval contributes ≈0; the
  think_embed baseline always survives, so a bad retrieval can never
  overwrite trunk state. Mode is recorded as
  `cfg["retrieval_input_additive"]`; `generate_with_retrieval_as_input`
  takes `additive=` (True for v7, False replays v5/v6 replacement).
- **Fix C — gist moved to the trunk, WM precision-only.** The v6 WM
  gist supervision is removed. The multi-horizon windowed-gist target
  (`windowed_future_gist`, `K ∈ {16,64,256}`) now supervises the
  TRUNK's heads (predict mean-pooled `h[t+1:t+1+K]` from `h[t]`), so
  "direction" lives in the trunk and WM is free to learn precise
  retrieval from the LM loss alone. CLI: `--future_emb_loss_weight 0.1
  --wm_gist_horizons "16,64,256"`; heads saved as
  `ckpt["future_gist_heads_state_dict"]`. The v5/v6 flags
  `--wm_future_pred_weight`, `--wm_future_pred_T`, `--future_emb_T_max`,
  `--future_emb_T_ramp_frac` are deprecated no-ops.

Tests: `test_trunk_gist_loss.py` (gist math + multi-horizon loss),
`test_retrieval_as_input.py` (additive formula, α param, additive vs
replace). v7 also tests a hypothesis: removing the WM-gist loss should
revert think_rate toward v5's 0.012 (v6's 0.23 was plausibly *caused*
by that loss perturbing the gate). **If v7 still over-thinks and recall
degrades, the next fix is architectural — make think tokens
state-read-only in the DeltaNet recurrence (β=0 on the delta-rule
write gate) so they cannot corrupt long-range state, with thinking
influencing the emit via the WM channel instead.**

### Phase C pretrain + grader-RL trajectory (2026-05-21..23)

**Phase C — Chinchilla-completion pretrain**
(`launch_pretrain_phase_c.sh`, 20.6 h on 1× RTX 5090). Continuation of
the v7.1-pkm-film ckpt from 2.13 B → 5.28 B tokens
(Chinchilla-optimal for 287 M), with the v7 trunk multi-horizon gist
loss added. Result: **strict win over the v7.1 base on all 8
per-source held-out CEs** (~0.17 CE mean; biggest wins on
cybernative −0.29, python_codes −0.24, bigvul −0.21; wikipedia which
was a tie at the 3.0 B midpoint also flipped positive by the end).
In-run VAL ppl 5.83 → 4.90. PKM value-row norm grew 2.54 → 3.58
during the continuation — the value table genuinely kept learning.

**torch.compile + AOTAutograd ViewMeta bug, workaround + upstream
fix.** The gist loss's backward gradient flow added a new graph
output and tripped a SymInt-corruption bug in AOTAutograd's
ViewMeta-replay (pytorch issue #124382 territory). Three principled
fix attempts in `model.py` all crashed (return-tuple → segfault;
attribute stash → garbage-shape RuntimeError; compute-inside-forward
→ segfault). **Workaround**:
`torch._functorch.config.view_replay_for_aliased_outputs = False`
toggle in `speed_knobs.py` before `torch.compile` — verified 3×
against the real trainer; held for the full 20.6 h Phase C run.
**Upstream fix**: a proper PyTorch source patch was root-caused,
implemented, and tested on a checkout at `~/ml/pytorch` on branch
`fix-viewmeta-replay-garbage-shape` (commit 78e6245); two layered
bugs — a C++ use-after-free in `_unsafe_view_ViewMeta`'s
`SerializableTuple` (reference-typed element bound to a temporary
from `std::make_tuple`), and an overlapping-stride tangent layout
for `expand`'d outputs in `MemoryFormatMeta.from_tensor`.
Equivalence-tested on CPU; PR pending.

**Combined SFT on the Phase C base** (`launch_sft_phase_c.sh`, 43 min)
→ HumanEval **10/164** (+2 over SFT v7 on the v7.1 base). Long-context
recall **98.2 %**. **PKM is now load-bearing on HumanEval**:
`pkm_off −5/−50 %` on the Phase C SFT ablation — the static-memory
side-table genuinely contributes once given enough pretrain training
(consistent with the row-norm growth above).

**Grader-RL v1** (`launch_rl_grader_phase_c.sh`, 800-step GRPO):
**14/164 at step-100** (+4 over Phase C SFT — the project's first
execution-grounded RL lift). Plateaued at step-200 (13), still 14 at
step-300, then **catastrophically collapsed around step-350** when
the ponder cost finally bit and depth dropped from 120 → 30, taking
the SFT-distilled CoT-+-code output format down with it (reward → 0,
all rollouts syntax_error). Diagnosis: no KL-to-reference penalty.

**Grader-RL v2** (`launch_rl_grader_phase_c_v2.sh`, KL-stable GRPO).
Implemented the missing KL term (`--kl_coef 0.05`, frozen reference =
the starting SFT ckpt) in `train_rl_grader.py`. Recipe also dropped
ponder cost (depth was the v1 trigger), halved LR (5e-6 → 2e-6),
tightened PPO clip (0.2 → 0.1), lowered temperature (0.9 → 0.7),
capped at 400 steps. **Monotonically climbed**: step-100 14/164 →
step-200 15/164 → step-300 **16/164 (9.8 %, new project best, +100 %
relative over SFT v7's 8/164)**. KL bounded 0.05–0.10 throughout —
the principled stability mechanism preventing v1-style drift.
`train_rl_grader.py` now exposes `--kl_coef` and loads a frozen ref
model; the docstring's long-promised KL term is finally implemented.

**Decode-speedup pass (2026-05-23)**. Profile showed every generated
token re-processes the entire prefix (~30 ms/tok at T=512). Two-tier
fix: (1) **shipped**: `model._film_bypass = True` toggled around all
generation loops (`eval_humaneval.generate`,
`generate_with_retrieval_as_input`, `train_rl_grader.rollout_group_
batched`) — K=3 FiLM self-feed at T=1 is pure waste. ~2× measured.
(2) **in progress**: wire true state-passing incremental decoding
through `TinyLM.forward_step` (foundation in
`_FlaWrapper.forward_step` already exists from earlier work);
estimated additional ~3.5–6×.

**Self-distillation infra ready** (`gen_rejection_data.py` with
`--keep_all`, `train_dpo.py`) for the next push past v2's 16/164 —
the user-approved order is rejection-sampling SFT → DPO → iterative
repair.

### Headline HumanEval trajectory across this arc

| stage | HumanEval | Δ |
|---|---|---|
| SFT v7 (v7.1 base, replacement retrieval, v6 WM-gist) | 8/164 | — |
| Phase C SFT (Chinchilla base + additive + trunk gist) | 10/164 | +2 |
| RL v1 step-100 (peak, then collapse) | 14/164 | +4 |
| **RL v2 step-300 (KL-stable, current best)** | **16/164 (9.8 %)** | **+2** |

Each architectural piece is now measurably load-bearing on its
PROPER probe: PKM on HumanEval (−5 in ablation), WM on long-context
recall (98.2 %), thinking gate is task-adaptive (0 % think on
recall, ~33 % on code), FiLM on PPL. The remaining lever for the
coding headline at 287 M is post-training scale + model size (the
~12–15 % HumanEval band typical for this size class).

### Other engineering bugs caught & fixed (2026-05-19 audit)
  - **sft_code control-flow bug**: modern-load path (ckpt has memory +
    gate) was sandwiched between three branches; the final `else: model,
    cfg = build_model_from_ckpt(...)` re-ran for every modern-path
    invocation, silently overwriting any flags the modern path had set
    (FIX A, sft_with_thinking, etc.). Combined-SFT-v1 trained for 30
    minutes WITHOUT FIX A despite the launcher passing
    `--mem_write_only_at_think`. Fixed by guarding the legacy/original
    paths behind `if not args_with_thinking_done`. Regression test:
    `experiments/test_sft_code_loading.py::test_modern_path_honors_mem_write_only_at_think_flag`.
  - **eval_bracket_structure reload bug**: `build_model_from_ckpt`
    didn't pass `mem_write_only_at_think` from cfg to the
    `WorkingMemory` constructor, so a ckpt trained with FIX A would be
    re-evaluated with FIX A off (the flag is a plain bool attribute,
    not in state_dict). Fixed by reading from cfg with default False.
    Regression tests: `test_eval_reload_preserves_woat_flag_*`.
  - **gate_floor/emit_threshold saturation in train_rl_grader rollout**:
    `gate.clamp_min(gate_floor) >= emit_threshold` is trivially True
    when `gate_floor >= emit_threshold`, so setting
    `--gate_floor 0.5 --emit_threshold 0.5` (the v2-RL launcher)
    silently made the model never-think regardless of the actual gate
    output. Pinned by `experiments/test_rl_grader_gate_floor.py`. The
    operational rule: **for RL rollouts that should preserve thinking,
    use `gate_floor < emit_threshold`** (e.g. 0.3 vs 0.5).
  - **future_head_state_dict was saved but never loaded** — broke
    continuation training (eval is unaffected). Fixed by re-loading in
    `sft_code.py` when `--future_emb_loss_weight > 0` and the source
    ckpt has the state-dict key.

### Open architectural finding: thinking gate is temperature-fragile
At τ=0 (greedy eval) the gate produces a healthy `think_rate ≈ 0.30`
on HumanEval. At τ=0.9 (RL rollout sampling) it collapses to bimodal:
either ~0 (never think) or ~1 (always think) depending on whether
`gate_floor` is below or at/above `emit_threshold`. Hypothesised cause:
sampling pushes the hidden-state distribution into regions the gate
wasn't trained on, so its output is essentially noise around 0.5; tiny
config changes flip it to a stuck regime. Diagnostic
`gate_collapse_diag` is planned. Implication: **don't expect train-time
gate selectivity to survive RL rollouts** without explicit
robustification (train under sampling, or replace the discrete gate
with a continuous "soft think" attention-style head).

## Lessons learnt (don't relive these)

- **PKM α-decay is real and means the bootstrap is incomplete.** v7.0 (before the α-floor + value-LR-mult fixes) showed αL grow 0 → +0.085 by step 280, then steadily decay back to +0.04 by step 400 — the optimizer correctly sees random-init values as noise. The fix is to keep αL high enough long enough that value rows actually learn (α-floor curriculum), AND give values a separate higher LR (`--pkm_value_lr_mult 100`). Without both, PKM stays inert.
- **`v_std` is a misleading PKM diagnostic.** It's invariant under gradient updates that preserve the overall Gaussian distribution of values, which is exactly what early-training noise looks like. Use `row` (mean row-norm vs expected init) — that's what the v5-pkm probe used to discover the table was dead.
- **MQAR at the saturation regime (T=512/K=128) doesn't discriminate architectural feedback choices at our model scale.** At those settings the DeltaNet baseline saturates at chance recall AND we never converge in 3 k steps anyway. Tried as an early-validation for v6 and was inconclusive; we got better signal from per-source pretrain CE. Use MQAR T=256/K=32 (the original 4L sweet-spot) for architectural ablations, not the memory-vs-no-memory saturation regime.
- **`compile_model` silently failing is a footgun** (fixed by strict mode, 2026-05-18). Multiple runs in this repo trained at eager speed because the silent dynamo fallback wasn't noticed. Always-on strict mode prevents this.
- **bf16-on-sparse-loss collapses logits to uniform.** Discovered while debugging MQAR (sparse mask, only ~25 % of positions carry gradient). The per-token gradient ends up below bf16's mantissa precision when scaled by activation_checkpointing + Adam's normalization. **Fix for sparse-loss validation tasks (MQAR, recall benchmarks): run in fp32**. Production pretrain is unaffected (every token contributes loss, ~4× more gradient signal per step).
