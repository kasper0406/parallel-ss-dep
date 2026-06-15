# Training & optimizer knobs

## Summary
The validated training recipe for this stack on the 5090s: **bf16 + tf32 always**, Muon (matmul params) + AdamW (embeddings/tables/α) split, **WSD schedule**, **WD=0.01** (not the Moonlight-scale 0.1), activation checkpointing + grad-accum for real batch sizes, and a handful of stability regularizers. These were each learned the hard way; the non-obvious ones are flagged. Source: `CLAUDE.md`, `experiments/optim_utils.py`, `speed_knobs.py`, `bf16_optim.py`.

## Always-on
- **`--bf16 --tf32`** — ≈2.3× over the fp32 path (18 k→42 k tok/s). Master weights stay fp32, no GradScaler. **Exception**: sparse-loss validation tasks (MQAR, recall) collapse to uniform under bf16 (per-token gradient below mantissa precision) → run those in **fp32**.
- **`--compile`** (default-on, strict mode `suppress_errors=False`) — +10 %; strict mode prevents the silent dynamo-fallback footgun that shipped multiple runs at eager speed. `torch.compile` on the AdamW step is also on (bit-identical to eager).
- **`--activation_checkpointing`** — ~N_layers× activation reduction for ~30 % compute; bit-identical. Lets batch go higher (target 14 at T=2048 with K=3 self-feed).
- **`--lr_schedule wsd`** (default) — warmup → constant peak → short cosine decay (`--lr_decay_frac 0.15`). We undertrain and keep extending runs, so cosine's commit-T_max-upfront is a poor fit; WSD can stop anywhere to cash a ckpt.

## The hard-won settings
- **`--wd 0.01`** (default now), NOT 0.1. WD=0.1 caused **residual-stream collapse** (||h||@L0 shrank across ckpts; blocks vanished). WD=0.1 is a 5.7 T-token Moonlight setting; at our ~5–10 tok/param it's pure brake. `diag_ckpt.py` found this.
- **`--alpha_wd 0.0`** — no weight decay on FiLM α (the gradient wants α higher; WD was the brake). See [[film-feedback]].
- **`--gate_floor_min 0.5 --gate_warmup_steps 20000`** — avoids the maladaptive-thinking trap (VAL ppl 49→940 when the floor reaches 0). See [[thinking-gate]].
- **`--grad_accum N`** (pretrain path only) — effective batch was ~14 k tok/step (far below the 0.5–4 M typical); `--batch 14 --grad_accum 8` ≈ 229 k tok/step.
- **`--grad_clip 1.0`**, **`--z_loss 1e-4`** (PaLM logit-drift regularizer), **`--gate_entropy_aux_weight 0.1`** (entropy-grounded gate target, no extra forward).

## Optimizer split (`optim_utils.build_optimizer`)
- **Muon** for matmul/recurrent params; **AdamW** for embeddings, FiLM-α, PKM value tables (`_is_pkm_value` → 100× LR group), think-adapter, gate.
- bf16 optimizer state optional (`--bf16_optim_state`) — math in fp32, ~550 MB saved; lossless.

## Mid-eval robustness
- `--mid_eval_save_only` (skip the GPU-pressured HumanEval subprocess, just save the ckpt + advance the counter) and `--mid_eval_min_free_gib` auto-skip. Production runs use save-only — the ckpts are the load-bearing artifact.

## Related
[[film-feedback]] · [[thinking-gate]] · [[hardware-and-compute]] · #architecture #infra
