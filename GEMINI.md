# GEMINI.md / CLAUDE.md — Project Instructions

## Top-level goal — "Small Super-coder"

**Build a small code model that punches above its weight on coding benchmarks (HumanEval / MBPP / long-context recall) under tight compute and memory.** Architectural research in this repo is in service of that target. Every experiment should ladder up to it.

Hard compute constraint: 2× RTX 5090 (32 GB each, no NVLink). Training-token budgets should be planned accordingly — Chinchilla-optimal for a 217 M model is ~4 B tokens; budget accordingly.

## Current architectural stack

Backbone is **DeltaNet** (`--arch deltanet`, plain `chunk_delta_rule` from FLA — efficient linear-RNN, bounded state). Validated lifts to keep:

- **Sparse FiLM feedback (2, 28)**: −3.1 % to −5.4 % PPL at 217 M / 360 M / 708 M (see README headline table). Use `--feedback film --feedback_pairs "2,28" --feedback_self_k 3`.
- **K=3 self-feeding** for FiLM: closes train/inference gap so deployment uses a single forward at 1× decode cost.
- **Bounded working memory** (`experiments/model.py::WorkingMemory`, validated 2026-05-12): write-gated buffer of past hidden states, read at "think" / query positions via soft attention. **+11.1 pp recall** on saturated MQAR (T=512, K=128) vs DeltaNet alone. Read cost O(T·K·d), no O(T²) attention. Enable with `--use_memory --mem_size 1024`.
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
- **Use real PPL** (i.e. exp(−CE)) for reported numbers; don't include ponder cost in displayed "ppl=" values.
- **Always train with `--bf16 --tf32`** on this hardware. The 5090 fp32 path leaves ~2.3× speed on the table (measured: 18 k → 42 k tok/s at the same batch). bf16 wraps `model.forward` in `torch.autocast`; master weights stay fp32 so Muon/AdamW are unaffected. TF32 covers the residual fp32 matmul. No GradScaler needed (bf16 has fp32 exponent range).
- **Don't apply weight decay to FiLM α** (`--alpha_wd 0.0`). The WD-equilibrium probe (2026-05-13, `experiments/probe_alpha_wd.py` on the v1 500 M-token ckpt) showed |grad_α| ≈ 1.83 × WD·|α| — the gradient consistently wants α higher and the Muon WD=0.1 was the brake, not a true loss-flat ceiling. Run with `--alpha_wd 0.0` and let α find its real equilibrium.
- **GRPO ponder cost shaping** (`experiments/thinking.py::compute_grpo_advantages`) supports four orthogonal knobs, all backwards-compatible defaults: `--grpo_ponder_shape {linear,quadratic}`, `--grpo_ponder_counterfactual` (clamps task component at depth-0 baseline so thinking can never *worsen* the task reward but the depth cost still always applies), `--grpo_separate_ponder_norm` (z-scores task only and subtracts absolute ponder after, fixing the wash-out where GRPO group-norm squashes the small ponder magnitude into CE noise), `--grpo_ponder_warmup_steps N` (curriculum ramping ponder cost 0→full over N steps to avoid cold-start collapse). Recommended Phase-C config: `--grpo_ponder_shape quadratic --grpo_ponder_counterfactual --grpo_ponder_warmup_steps 300`.

## Documentation index

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
- **Tests live under `experiments/test_*.py`** and are pytest-discoverable. 52 tests covering `data_mix`, `thinking` (GRPO advantages), `eval_callback`, and `sft_code` (think-burst insertion). Run with `PYTHONPATH=. .venv/bin/python -m pytest experiments/test_*.py -v`. Add a test when you add a new builder, filter, or shaping mode — the test suite has already caught two real bugs this session (BigVul vul=0 filter, GRPO NaN on size-1 groups).

## Current state (2026-05-13)

- `pretrain_mix_v2.pt` is the active run target: 217 M DeltaNet + FiLM(2,28) + memory(1024) + thinking-gate on the v2 mixed corpus (9 v1 sources + BigVul + CyberNative DPO at ~7 % CVE/security weight). bf16 + TF32, batch 8 × T 2048 × 130 k steps ≈ 2.13 B tokens. `--alpha_wd 0.0` per the WD-equilibrium probe. ETA ~14 hr at ~42 k tok/s. Launcher: `launch_pretrain_mix_v2.sh`; logs in `runs/pretrain_mix_v2.log`; TB in `runs/tb/pretrain_mix_v2`.
- v1 (fp32 / batch 4) was killed at step ~80 k (~30 % through) after the bf16+TF32 fast-mode confirmed a 2.28× speedup. The α-WD probe result was the main usable data from v1.
- Next planned: post-v2, run the cross-attention vs scalar-α ablation if no-WD-on-α doesn't push α much higher; then SFT on instruction-code pairs; then GRPO + memory + thinking with the new ponder shaping; then PHASE_C_RL (prediction-as-signal).
