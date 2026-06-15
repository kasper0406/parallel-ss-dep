# Hardware & compute constraints

## Summary
Everything runs on a **2× RTX 5090 dev rig** (Blackwell sm_120, 32 GB each, **no NVLink**). This drives the whole design: bounded-state linear RNN (no growing KV cache), bf16 everywhere, activation checkpointing, and a Chinchilla token budget of ~5.3 B for the 287 M model. The Blackwell GPUs also create a recurring class of bugs in the FLA (flash-linear-attention) kernels that must be watched. Sources: `hardware.md`, `feedback_python_env.md`, `project_fla_blackwell_detection_bug.md`, `CLAUDE.md`.

## The rig
- 2× RTX 5090, **sm_120** (capability `(12, 0)`), 32 GB each, no NVLink → DDP only, no model-parallel sharding across the link.
- Hard compute constraint. Chinchilla-optimal for a 217 M model ≈ 4 B tokens; for 287 M ≈ 5.3 B. Budget runs accordingly.
- GPUs can fall off the PCIe bus under sustained load ("Unable to determine the device handle … Unknown Error") — corrupts CUDA node-wide; needs nvidia module reload / reboot.

## Python env
- `uv` for env management; **torch nightly cu132** (`...whl/nightly/cu132`).
- **FLA must be the local fork** at `~/ml/flash-linear-attention`, editable-installed. The pip `flash-linear-attention 0.5.1` is missing the Blackwell `global_scratch` allocator fix. Verify `import fla; fla.__file__` resolves to the fork.
- `export PYTHONPATH=$PYTHONPATH:.` before running `experiments/*.py`.

## Blackwell / FLA gotchas (recurring)
- **`gated_deltanet` is broken on sm_120** — FLA's `gated_delta_rule` hits a CUDA "misaligned address" in the Triton autotuner backward. Use plain `--arch deltanet`. See [[deltanet-backbone]].
- **FLA Blackwell-detection bug**: `IS_NVIDIA_BLACKWELL` only matched capability `== 10`; sm_120 is `(12,0)` → every Blackwell workaround was silently OFF → intermittent `cudaErrorLaunchFailure` in latent forwards. Fix in the fork (`[0] in (10, 12)`), but it **reverts on any fork pull** — re-apply and verify `fla.utils.IS_NVIDIA_BLACKWELL is True`.
- Wedged/runaway CUDA processes after a launch failure must be killed by **process group**, not PID.

## Standing performance knobs (always on)
- `--bf16 --tf32` (≈2.3× over fp32 path). See [[training-and-optim-knobs]].
- `--activation_checkpointing`, `--compile` (default-on, strict mode), FiLM K-self-feed warmup bypass.
- bf16 optimizer state optional (`--bf16_optim_state`, ~550 MB saved on the 218 M model).

## Related
[[deltanet-backbone]] · [[training-and-optim-knobs]] · #infra
