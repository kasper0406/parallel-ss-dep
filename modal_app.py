"""Modal app for the PD-SSM family GPU experiments.

Usage:

  pip install modal && modal token new          # one-time auth setup

  # Quick smoke (~1 min, validates the image works at all)
  modal run modal_app.py::smoke

  # Priority 1 — mod-p Landau sweep at state_dim=4 (the algebra-gap test)
  modal run modal_app.py::priority1

  # Priority 2 — mod-5 long-T DeltaNet collapse reproduction
  modal run modal_app.py::priority2

  # Priority 3 — MKAR head-to-head (capacity-gap test)
  modal run modal_app.py::priority3

  # Priority 4 — Muon vs AdamW on discrete sigma
  modal run modal_app.py::priority4

  # Priority 5 — S5 word problem
  modal run modal_app.py::priority5

  # Run everything (~6-8 hours)
  modal run modal_app.py::all_priorities

  # Custom command — passes through to a single python invocation
  modal run modal_app.py::custom --command "python experiments/train_modular.py --arches pd_ssm --p 5 --T 128 --steps 1000"

GPU choice: defaults to H100. Override with --gpu flag (A100, H100, B200, etc.):
  modal run modal_app.py::priority1 --gpu A100

Cost notes (rough, Modal pricing):
  H100 80GB : ~$3.95/hr → priority 1 ~$7, all six ~$30-50
  A100 80GB : ~$2.10/hr → priority 1 ~$3.50, all six ~$20-30
  B200      : ~$5/hr (when available) → ~2x faster than H100, similar total cost.

Notes:
  - The image pre-installs torch, fla-flash-linear-attention, numpy, datasets,
    transformers, and clones the repo. First-build is ~3-5 min; subsequent
    runs reuse the cached image (~10-30s warmup).
  - The repo is baked into the image at the BRANCH commit, so every run is
    on the exact code state at modal-image-build time. Re-build the image
    by editing BRANCH/COMMIT below or with `modal run --force-build`.
  - All output is streamed back to your terminal; results files live on the
    container's ephemeral disk and are dumped to stdout at the end.
"""

import modal

# ---------------------------------------------------------------------------
# Image: PyTorch + fla + the repo at a specific branch.
# ---------------------------------------------------------------------------

REPO_URL = "https://github.com/kasper0406/parallel-ss-dep"
BRANCH = "pd-ssm"  # change to "main" once merged

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # Match the CUDA build to what Modal's GPU runtime expects (CUDA 12.x).
    .pip_install(
        "torch",  # pulls the latest stable; CUDA-aware on Modal GPUs
        "numpy",
        "transformers",
        "datasets",
        "einops",
        "ninja",  # fla's chunked kernels need this
    )
    # fla provides DeltaNet / GatedDeltaNet / Mamba2 wrappers we use as baselines.
    # If `flash-linear-attention` is unavailable, the smoke tests still run
    # because the PD-family layers don't need fla.
    .pip_install("flash-linear-attention")
    # Clone the repo at the specific branch (snapshot at image-build time).
    .run_commands(
        f"git clone --branch {BRANCH} --depth 1 {REPO_URL} /workspace"
    )
    .workdir("/workspace")
)

app = modal.App("pd-ssm-experiments", image=image)


# ---------------------------------------------------------------------------
# The single GPU function — runs an arbitrary shell command.
# All experiments dispatch through this so the image / GPU spec is shared.
# ---------------------------------------------------------------------------

@app.function(gpu="H100", timeout=8 * 3600)
def _run_command(command: str) -> str:
    """Run a shell command in /workspace and return combined stdout+stderr."""
    import subprocess
    print(f"\n>>> {command}\n", flush=True)
    proc = subprocess.Popen(
        command, shell=True, cwd="/workspace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    chunks = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    rc = proc.wait()
    out = "".join(chunks)
    if rc != 0:
        raise RuntimeError(f"command exited with code {rc}\n{out[-2000:]}")
    return out


# Helper to launch _run_command with a chosen GPU at call-time.
def _launch(command: str, gpu: str = "H100") -> str:
    if gpu == "H100":
        return _run_command.remote(command)
    # Reconfigure GPU per-call — Modal supports per-invocation overrides via
    # `with_options` (reads from the function instance).
    fn = _run_command.with_options(gpu=gpu)
    return fn.remote(command)


# ---------------------------------------------------------------------------
# Smoke test — fast sanity that the image and GPU work end-to-end.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke(gpu: str = "H100"):
    """Fastest end-to-end check: parity at T=64, ~1 min."""
    out = _launch("python experiments/smoke_pd_ssm.py", gpu=gpu)
    print("\n=== SMOKE COMPLETE ===")


# ---------------------------------------------------------------------------
# Priority 1 — Mod-p Landau sweep at state_dim=4.
# Predicted: complex_pd >= 95% on mod-5,7,11 where pd_ssm at chance.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def priority1(gpu: str = "H100"):
    """Landau-bound sweep — pd_ssm vs complex_pd at N=4, mod-{2,3,5,7,11}."""
    cmd = (
        "python experiments/train_modular.py "
        "--arches deltanet,deltanet_negeig,pd_ssm,complex_pd "
        "--p 2 3 5 7 11 "
        "--T 128 "
        "--state_dim 4 "
        "--steps 5000 --batch 256 "
        "--d_model 128 --n_layers 4 --n_heads 4 --d_head 32 "
        "--lr 3e-3 "
        "--log_every 500"
    )
    _launch(cmd, gpu=gpu)
    print("\n=== PRIORITY 1 COMPLETE ===")
    print("Pass criterion: complex_pd >= 0.95 end_acc on mod-5/7/11 where pd_ssm at chance.")


# ---------------------------------------------------------------------------
# Priority 2 — Mod-5 long-T DeltaNet collapse reproduction.
# Predicted: PD variants > 95% where deltanet_negeig at 9% (collapse).
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def priority2(gpu: str = "H100"):
    """Long-T mod-5: reproduce DeltaNetNegEig collapse, show PD doesn't."""
    cmd = (
        "python experiments/train_modular.py "
        "--arches deltanet_negeig,pd_ssm,complex_pd "
        "--p 5 "
        "--T 512 "
        "--state_dim 8 "
        "--steps 5000 --batch 128 "
        "--d_model 128 --n_layers 4 --n_heads 4 --d_head 32 "
        "--lr 3e-3 "
        "--log_every 500"
    )
    _launch(cmd, gpu=gpu)
    print("\n=== PRIORITY 2 COMPLETE ===")
    print("Pass criterion: pd_ssm AND complex_pd > 0.95 where deltanet_negeig collapses to ~0.09.")


# ---------------------------------------------------------------------------
# Priority 3 — MKAR head-to-head (capacity-gap test).
# Predicted: pd_kv >= 0.85 at K=32 where pd_ssm < 0.20.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def priority3(gpu: str = "H100"):
    """MKAR sweep — pd_ssm (vector state) vs pd_kv (matrix state) vs baselines."""
    cmd = (
        "python experiments/train_mqar.py "
        "--arches linear,deltanet,pd_ssm,pd_kv "
        "--T 256 "
        "--n_pairs 4 8 16 32 "
        "--vocab 64 "
        "--steps 5000 --batch 256 "
        "--d_model 128 --n_layers 4 --n_heads 4 --d_head 32 "
        "--lr 3e-3 "
        "--log_every 500"
    )
    _launch(cmd, gpu=gpu)
    print("\n=== PRIORITY 3 COMPLETE ===")
    print("Pass criterion: pd_kv recall >= 0.85 at K=32 where pd_ssm < 0.20 (capacity gap closed).")


# ---------------------------------------------------------------------------
# Priority 4 — Muon vs AdamW on discrete sigma.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def priority4(gpu: str = "H100"):
    """Muon vs AdamW: does Muon fix the discrete-σ stall on mod-3 / mod-5?"""
    base = (
        "python experiments/smoke_complex_pd_landau.py "
        "--p 3 5 "
        "--T 128 --steps 3000 --batch 128 "
        "--d_model 128 --n_layers 4 --n_heads 4 --d_head 32 "
        "--state_dim 8 "
        "--arches pd_ssm,complex_pd "
        "--log_every 500"
    )
    print(">>> AdamW baseline")
    _launch(base + " --optim adamw", gpu=gpu)
    print("\n>>> Muon variant")
    _launch(base + " --optim muon", gpu=gpu)
    print("\n=== PRIORITY 4 COMPLETE ===")
    print("Pass criterion: Muon-pd_ssm >= 0.80 on mod-3 where AdamW-pd_ssm stalls at ~0.30.")


# ---------------------------------------------------------------------------
# Priority 5 — S5 word problem.
# Predicted: pd_ssm and complex_pd >= 0.95 pos_recall — beats DeltaNet's 0.978.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def priority5(gpu: str = "H100"):
    """S5 word problem — PD natively realises the standard permutation rep."""
    cmd = (
        "python experiments/train_s5.py "
        "--arches deltanet_negeig,pd_ssm,complex_pd "
        "--T 128 "
        "--steps 5000 --batch 256 "
        "--d_model 128 --n_layers 4 --n_heads 4 --d_head 32 "
        "--lr 3e-3 "
        "--log_every 500"
    )
    _launch(cmd, gpu=gpu)
    print("\n=== PRIORITY 5 COMPLETE ===")


# ---------------------------------------------------------------------------
# Run everything in order.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def all_priorities(gpu: str = "H100"):
    """Run priority 1-5 in sequence (~6-8 hours, ~$30-50 on H100)."""
    smoke(gpu=gpu)
    priority1(gpu=gpu)
    priority2(gpu=gpu)
    priority3(gpu=gpu)
    priority4(gpu=gpu)
    priority5(gpu=gpu)


# ---------------------------------------------------------------------------
# Escape hatch — run an arbitrary command on a GPU.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def custom(command: str, gpu: str = "H100"):
    """Run an arbitrary shell command on the GPU container.

    Example:
      modal run modal_app.py::custom --command "python experiments/train_modular.py --arches pd_ssm --p 5 --T 128 --steps 2000 --state_dim 4"
    """
    _launch(command, gpu=gpu)
