# GEMINI.md / CLAUDE.md — Project Instructions

## Top-level goal — "Small Super-coder"

**Build a small code model that punches above its weight on coding benchmarks (HumanEval / MBPP / long-context recall) under tight compute and memory.** Architectural research in this repo is in service of that target. Every experiment should ladder up to it.

Hard compute constraint: 2× RTX 5090 (32 GB each, no NVLink). Training-token budgets should be planned accordingly — Chinchilla-optimal for a 217 M model is ~4 B tokens; budget accordingly.

## Current architectural stack

Backbone is **gated DeltaNet** (efficient linear-RNN, bounded state). Validated lifts to keep:

- **Sparse FiLM feedback (2, 28)**: −3.1 % to −5.4 % PPL at 217 M / 360 M / 708 M (see README headline table). Use `--feedback film --feedback_pairs "2,28" --feedback_self_k 3`.
- **K=3 self-feeding** for FiLM: closes train/inference gap so deployment uses a single forward at 1× decode cost.
- **Bounded working memory** (`experiments/model.py::WorkingMemory`, validated 2026-05-12): write-gated buffer of past hidden states, read at "think" / query positions via soft attention. **+11.1 pp recall** on saturated MQAR (T=512, K=128) vs DeltaNet alone. Read cost O(T·K·d), no O(T²) attention. Enable with `--use_memory --mem_size 1024`.
- **Output / thinking gate**: per-position sigmoid head deciding emit vs think. Wired in `train_lm.py` (BPTT) and `train_rl.py` (GRPO).

Dead direction (do **not** revive without explicit justification):
- **Corpus-RAG** (`rag_projection`, the old in-memory codeparrot DB). Its loss-bearing forward didn't pass `rag_hidden`, so gradients were structurally zero and the projection never trained. Replaced by `WorkingMemory`.

## Key mandates

- **Validate architecture on synthetic recall tasks before scaling.** MQAR (`experiments/tasks/mqar.py`, `train_mqar.py`) is the canonical test for "does this help recall". Choose configs *on or above* DeltaNet's saturation threshold (the T=512/K=128 regime), otherwise the mechanism is invisible.
- **Don't run natural-text RL on the existing base.** `checkpoints/think_sweep_fresh_tokens_final.pt` scored **0 / 20 on HumanEval** (8.2 M training tokens, 0.04 tokens/param — way below Chinchilla). Architectural wins drown in the noise of an undertrained base.
- **Thinking-token embedding** must be initialised to the embedding-mean when added — PyTorch's default Linear init puts random noise into the recurrence at every think step. `TinyLM.__init__` does this automatically when `use_memory=True`.
- **`mem_read_mask`** is the right way to drive memory reads at non-think tokens (e.g. MQAR query positions). Pass it as a kwarg to `TinyLM.forward`.
- **Activation checkpointing** (`--think_checkpointing`) is mandatory for `safety_max_depth > 2` in RL / queue-based training.
- **Use real PPL** (i.e. exp(−CE)) for reported numbers; don't include ponder cost in displayed "ppl=" values.

## Documentation index

- `README.md` — headline results (FiLM lift, MQAR recall, deployment-fair scoring) and the small-super-coder framing.
- `PLAN.md` (or `/home/knielsen/.claude/plans/noble-beaming-beacon.md`) — active and near-term experiments.
- `NEXT_DIRECTIONS.md` — strategic roadmap.
- `THINKING_RAG_DIRECTION.md` — mechanistic rationale for the thinking head (still valid for the gate; the corpus-RAG section is historical).
- `RL_RAG_ROADMAP.md` — superseded by the working-memory direction; keep for history.
- `SESSION_FINDINGS.md` — chronological log of empirical results.

## Conventions

- Python env via `uv`, torch nightly cu132 (`...whl/nightly/cu132`).
- Run scripts in this repo expect `export PYTHONPATH=$PYTHONPATH:.` before invoking `experiments/*.py`.
- Save checkpoints with `{state_dict, step, config}` dict; load with `weights_only=False` (the `config` contains class references).
- `experiments/eval_bracket_structure.py::build_model_from_ckpt` auto-detects `gate_head` and `memory.*` in the state-dict and constructs the model with matching kwargs; reuse it for new evals rather than hand-rolling.
