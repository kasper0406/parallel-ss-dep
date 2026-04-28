#!/usr/bin/env bash
# run_all_priorities.sh — Hyperstack/SSH-friendly driver for the GPU sweep.
#
# Usage on the Hyperstack instance (after SSH'ing in):
#
#   curl -O https://raw.githubusercontent.com/kasper0406/parallel-ss-dep/pd-ssm/run_all_priorities.sh
#   chmod +x run_all_priorities.sh
#   ./run_all_priorities.sh                   # all priorities, auto-shutdown
#   ./run_all_priorities.sh priority1         # just priority 1
#   ./run_all_priorities.sh priority1 --no-shutdown
#
# Logs land in ./logs/<priority>_<timestamp>.log.
# By default: tear down the box on completion (so you don't pay for idle).
# Pass --no-shutdown to keep it running for inspection.

set -euo pipefail

# ----- config -----
BRANCH="pd-ssm"
REPO_URL="https://github.com/kasper0406/parallel-ss-dep"
WORK_DIR="${HOME}/parallel-ss-dep"
LOG_DIR="${WORK_DIR}/logs"

# ----- arg parse -----
PRIORITIES=()
SHUTDOWN=1
for arg in "$@"; do
    case "$arg" in
        priority1|priority2|priority3|priority4|priority5|smoke)
            PRIORITIES+=("$arg")
            ;;
        --no-shutdown)
            SHUTDOWN=0
            ;;
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $arg"; exit 1
            ;;
    esac
done
if [[ ${#PRIORITIES[@]} -eq 0 ]]; then
    PRIORITIES=(smoke priority1 priority2 priority3 priority4 priority5)
fi

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

# ----- env setup (idempotent) -----
log "Setting up environment..."
if ! command -v python3.11 &>/dev/null && ! command -v python3 &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip git
fi
PY=$(command -v python3.11 || command -v python3)

if [[ ! -d "$WORK_DIR" ]]; then
    log "Cloning $REPO_URL @ $BRANCH..."
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORK_DIR"
fi
cd "$WORK_DIR"

if [[ ! -d ".venv" ]]; then
    log "Creating venv..."
    $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing deps (skipping if already installed)..."
pip install --quiet --upgrade pip
pip install --quiet torch numpy einops ninja
# fla provides DeltaNet/GatedDeltaNet/Mamba2 baselines.
pip install --quiet flash-linear-attention || \
    log "WARN: flash-linear-attention install failed — fla baselines will skip."

# ----- pre-flight -----
mkdir -p "$LOG_DIR"
PRE_LOG="$LOG_DIR/preflight_$(ts | tr ':' '-').log"
log "Pre-flight numerical sanity checks..."
python experiments/test_pd_closure.py 2>&1 | tee "$PRE_LOG"

# ----- experiments -----
run_smoke() {
    python experiments/smoke_pd_ssm.py
}

run_priority1() {
    python experiments/train_modular.py \
        --arches deltanet,deltanet_negeig,pd_ssm,complex_pd \
        --p 2 3 5 7 11 \
        --T 128 \
        --state_dim 4 \
        --steps 5000 --batch 256 \
        --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
        --lr 3e-3 --log_every 500
}

run_priority2() {
    python experiments/train_modular.py \
        --arches deltanet_negeig,pd_ssm,complex_pd \
        --p 5 \
        --T 512 \
        --state_dim 8 \
        --steps 5000 --batch 128 \
        --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
        --lr 3e-3 --log_every 500
}

run_priority3() {
    python experiments/train_mqar.py \
        --arches linear,deltanet,pd_ssm,pd_kv \
        --T 256 \
        --n_pairs 4 8 16 32 \
        --vocab 64 \
        --steps 5000 --batch 256 \
        --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
        --lr 3e-3 --log_every 500
}

run_priority4() {
    BASE_ARGS=(
        --p 3 5
        --T 128 --steps 3000 --batch 128
        --d_model 128 --n_layers 4 --n_heads 4 --d_head 32
        --state_dim 8
        --arches pd_ssm,complex_pd
        --log_every 500
    )
    log "Priority 4 — AdamW baseline"
    python experiments/smoke_complex_pd_landau.py "${BASE_ARGS[@]}" --optim adamw
    log "Priority 4 — Muon variant"
    python experiments/smoke_complex_pd_landau.py "${BASE_ARGS[@]}" --optim muon
}

run_priority5() {
    python experiments/train_s5.py \
        --arches deltanet_negeig,pd_ssm,complex_pd \
        --T 128 \
        --steps 5000 --batch 256 \
        --d_model 128 --n_layers 4 --n_heads 4 --d_head 32 \
        --lr 3e-3 --log_every 500
}

# ----- run loop -----
for p in "${PRIORITIES[@]}"; do
    LOG_FILE="$LOG_DIR/${p}_$(ts | tr ':' '-').log"
    log "=========== Running $p — log: $LOG_FILE ==========="
    t0=$SECONDS
    if "run_$p" 2>&1 | tee "$LOG_FILE"; then
        log "$p OK in $((SECONDS - t0))s"
    else
        log "$p FAILED — see $LOG_FILE — continuing"
    fi
done

log "All priorities completed. Logs:"
ls -la "$LOG_DIR"

if [[ $SHUTDOWN -eq 1 ]]; then
    log "Shutting down in 60 seconds (run with --no-shutdown to keep alive)..."
    sleep 60
    sudo shutdown -h now
fi
