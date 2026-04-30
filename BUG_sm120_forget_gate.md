# `RuntimeError: Triton Error [CUDA]: misaligned address` — fla forget-gate backward on RTX 5090 (sm_120)

> **Root cause (2026-04-30):** TileLang `SharedMemoryAlignmentPlanner` only
> enforces 1024-byte alignment for TMA-touched smem buffers when
> `TargetIsHopper(target)` is true, so sm_120 falls back to 16-byte alignment
> while `cp.async.bulk.tensor.*.shared::cta.global.*` requires 128-byte. The
> originally-reported `prepare_wy_repr_bwd_kernel` crash is a deferred-error
> red herring — the actual misalignment is in the TileLang-compiled
> `chunk_bwd_dqkwg_tilelang` kernel that runs immediately before. See
> `scripts/tilelang_sm120_issue_draft.md` for the upstream report and
> `scripts/sm120_tilelang_workaround.py` for the in-process Python
> workaround.

A `prepare_wy_repr_bwd_kernel` launch in **`fla` 0.5.0** appears to crash on
the first backward pass when the **forget gate is enabled**, on **RTX 5090
(sm_120)** running **PyTorch cu132 nightly** with **Triton 3.7**. Forward
succeeds; the crash happens at backward kernel-handle initialisation, before
any user code runs.

With `CUDA_LAUNCH_BLOCKING=1` the crash actually surfaces in the
TileLang-compiled `chunk_bwd_dqkwg_tilelang` kernel that runs *before*
`prepare_wy_repr_bwd_kernel`. Without blocking, CUDA's deferred error
machinery raises the misalignment when the next launch (`prepare_wy_repr_bwd_kernel`)
tries to initialise its handles.

## Affected paths

The crash is in `fla.ops.gated_delta_rule.wy_fast.prepare_wy_repr_bwd`. Any
fla cell whose backward routes through `chunk_gated_delta_rule_bwd` with
`g is not None` (forget gate active) hits it:

| fla layer                | Default forget gate | Crashes? | Workaround in this repo                              |
|--------------------------|---------------------|----------|------------------------------------------------------|
| `DeltaNet`               | n/a (no forget gate) | No      | n/a — used as-is                                      |
| `GatedDeltaNet`          | always on           | Yes     | Avoided in long-T runs                                |
| `GatedDeltaProduct`      | `use_forget_gate=True` | Yes  | `use_forget_gate=False` in `experiments/layers.py:271` |

## Crash signature

Captured from a real run:

```
File "fla/ops/gated_delta_rule/chunk.py", line 322, in backward
    dq, dk, dv, db, dg, dh0, dA_log, ddt_bias = chunk_gated_delta_rule_bwd(
File "fla/ops/gated_delta_rule/chunk.py", line 230, in chunk_gated_delta_rule_bwd
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
File "fla/ops/gated_delta_rule/wy_fast.py", line 329, in prepare_wy_repr_bwd
    prepare_wy_repr_bwd_kernel[(NT, B * HV)](
...
File "triton/compiler/compiler.py", line 468, in _init_handles
    self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
RuntimeError: Triton Error [CUDA]: misaligned address
```

The error is raised during `_init_handles` (kernel-binary load), not at the
launch — so it's the **compiled kernel object itself** Triton rejects on
sm_120, not the input pointer alignment per call. Forward kernels in the
same module load and run fine; only the backward `prepare_wy_repr_bwd_kernel`
trips it.

Full log: `/tmp/lm_fb/sparse_gated.log`.

## Where the code lives

All paths relative to the fla 0.5.0 site-packages tree
(`.venv/lib/python3.12/site-packages/fla/`):

- **Backward dispatch** — `ops/gated_delta_rule/chunk.py:322`
  (calls `chunk_gated_delta_rule_bwd` from the autograd `backward`)
- **Backward driver** — `ops/gated_delta_rule/chunk.py:230`
  (calls `prepare_wy_repr_bwd`)
- **Failing kernel launch** — `ops/gated_delta_rule/wy_fast.py:329`
  (`prepare_wy_repr_bwd_kernel[(NT, B * HV)](...)` with grid `(NT, B*HV)`,
  `BT=64`, `BK,BV ∈ {16,32,64}` chosen via `check_shared_mem()`)
- **Kernel definition (decorator + body)** — `ops/gated_delta_rule/wy_fast.py`
  (search for `prepare_wy_repr_bwd_kernel`; the kernel is `@triton.jit`)
- **GatedDeltaProduct routing into this path** — when
  `use_forget_gate=True`, `chunk_gated_delta_product_fwd` in
  `ops/gated_delta_product/chunk.py:67` calls
  `gdn_recompute_w_u_fwd` (i.e. `gated_delta_rule.wy_fast.recompute_w_u_fwd`),
  which establishes the wy-repr that `prepare_wy_repr_bwd` later inverts in
  the backward pass — that's how the forget-gate flip lights up this kernel
  for GDP too.
- **Layer-level switch** — `fla/layers/gated_deltaproduct.py:49,61,102,232,253`
  (every `if self.use_forget_gate:` site)

Our wrapper that pins the workaround:
`experiments/layers.py:271`

```python
super().__init__(GatedDeltaProduct(
    ...
    use_forget_gate=False,  # forget-gate kernel hits sm_120 misalign bug
    ...
))
```

## Environment

| Component | Version                                         |
|-----------|-------------------------------------------------|
| GPU       | NVIDIA GeForce RTX 5090 (sm_120, 32 GB)        |
| Driver    | 580.126.20                                      |
| PyTorch   | `2.13.0.dev20260427+cu132` (cu132 nightly wheel) |
| Triton    | `3.7.0` (bundled with the nightly torch)        |
| fla       | `0.5.0`                                         |
| OS        | Linux 6.8.0-110-generic                         |
| Python    | 3.12                                            |

Install reproducer for the env:

```bash
uv venv .venv-repro --python 3.12
source .venv-repro/bin/activate
uv pip install -U --pre --index-url https://download.pytorch.org/whl/nightly/cu132 'torch'
uv pip install fla-core==0.5.0 einops
```

## Minimal repro

The smallest end-to-end repro is one `GatedDeltaProduct` layer with
`use_forget_gate=True`, a single forward and `loss.backward()`. Forward
returns; backward raises.

`scripts/repro_sm120_forget_gate.py`:

```python
import torch
from fla.layers import GatedDeltaProduct

torch.manual_seed(0)
device = "cuda"
dtype  = torch.bfloat16

B, T, D, H = 1, 64, 256, 4
layer = GatedDeltaProduct(
    hidden_size=D,
    head_dim=D // H,
    num_heads=H,
    num_v_heads=H,
    mode="chunk",
    use_output_gate=True,
    use_short_conv=True,
    use_forget_gate=True,    # <-- flip to False to bypass the crash
    allow_neg_eigval=True,
    num_householder=2,
).to(device=device, dtype=dtype)

x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
y, *_ = layer(x)              # forward succeeds
y.sum().backward()            # ← Triton Error [CUDA]: misaligned address
print("ok")
```

Run:

```bash
python scripts/repro_sm120_forget_gate.py
```

Expected on sm_120 + cu132 nightly: forward prints nothing, backward raises
`RuntimeError: Triton Error [CUDA]: misaligned address` from
`triton/compiler/compiler.py: _init_handles`.

Flipping `use_forget_gate=False` makes the same script return `ok`.

## What we know / don't know

- **Forward is fine.** Only the backward `prepare_wy_repr_bwd_kernel` hits it
  — both DN-style WY recompute (forward) and GDP forward kernels load and
  run on the same device.
- **Failure is at handle load, not launch.** The `_init_handles` site means
  the compiled cubin / module fails to load on sm_120 — not a runtime
  pointer-alignment issue per call. Strides, dtypes and tensor shapes look
  16-byte aligned at the Python level (B=1, T=64, H=4, K=V=64, BT=64,
  BK=BV ∈ {16,32,64}).
- **Likely a Triton ↔ sm_120 codegen issue**, not an fla data-layout bug:
  the same kernel works on Ampere/Hopper per the upstream CI. A clean
  upstream report should land against Triton 3.7 / sm_120 with this fla
  kernel as the reproducer.
- **Cost of the workaround.** Disabling the forget gate handicaps the
  GatedDeltaProduct baseline by roughly +2.5 PPL (52.80 vs an expected ~50
  with the gate on, on our 217M codeparrot setup). The cross-layer FiLM
  delta we report on top of GDP is measured on this handicapped baseline —
  a clean GDP+FiLM number requires either fixing this kernel or running on
  a non-sm_120 GPU.

## Pointers for the upstream report

The bug belongs upstream at **`tile-ai/tilelang`** — see the prepared
draft at `scripts/tilelang_sm120_issue_draft.md` and the unified diff
at `scripts/tilelang_sm120_fix.patch`.

- TileLang repo: <https://github.com/tile-ai/tilelang> — file the issue,
  attach `scripts/repro_sm120_pure_tilelang.py` as the minimal repro.
- One-line fix in `src/transform/merge_shared_memory_allocations.cc:393`:
  swap `TargetIsHopper(target)` → `TargetHasBulkCopy(target)` so any
  TMA-capable target (sm_90 / sm_100 / sm_120 / future) gets the strict
  alignment automatically.

End-to-end verification on this hardware (RTX 5090, cu132 nightly) of
the workaround at `scripts/sm120_tilelang_workaround.py`:

1. `rm -rf ~/.tilelang/cache/*/kernels/*` (clear stale cached kernels).
2. `import scripts.sm120_tilelang_workaround` before any fla import.
3. Run `scripts/repro_sm120_forget_gate.py` with `USE_FORGET_GATE=True`.

→ Forward and backward both succeed; `chunk_bwd_dqkwg_tilelang`
recompiles under the 1024-byte aligner and the deferred Triton
`misaligned address` error disappears.
