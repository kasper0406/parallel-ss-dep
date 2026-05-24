#!/bin/bash
# Phase 22 validation: multi-seed (codeparrot) + natural-text (TinyStories).
# Runs sequentially on GPU 1.
#
# Multi-seed: K=3 + uniform L_sem β=1.0 at seed=1, seed=2 on codeparrot.
# Natural-text: plain DN, K=3 only, K=3 + per-token L_sem β=1.0 on TinyStories.
#
# Each codeparrot run = ~22 min train + ~5 min eval.
# Each TinyStories run = ~22 min train + ~30 s eval (no AST harness).
# Lagged-cached eval is skipped for natural-text runs (irrelevant for the
# cross-corpus check; only meaningful for FiLM-on-code deployment).
#
# Usage:
#   bash launch_validation.sh
set -e
cd /home/knielsen/ml/parallel-ss-dep

PY=.venv/bin/python
GPU=1
LOG=logs/validation
mkdir -p "$LOG"

# Wait for GPU 1 to be free (no python process running on it).
wait_for_gpu() {
    local target="$1"
    local count=0
    while true; do
        # Count python processes on the target GPU.
        local n=$(nvidia-smi --query-compute-apps=gpu_uuid,process_name \
                  --format=csv,noheader 2>/dev/null \
                  | awk -F, -v g="$target" 'NR>0 {print}' | wc -l)
        # Use index-based query.
        local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$target")
        if [[ "$mem" -lt 2000 ]]; then
            echo "[wait_for_gpu] GPU $target free (mem $mem MiB)"
            return 0
        fi
        count=$((count + 1))
        if (( count % 12 == 0 )); then
            echo "[wait_for_gpu] GPU $target busy: mem=$mem MiB (waited $((count*10))s)"
        fi
        sleep 10
    done
}

run_codeparrot_lsem_seed() {
    local seed="$1"
    local ckpt="checkpoints/film_self_k3_lsem_uniform_b10_30L_217M_seed${seed}.pt"
    local trainlog="$LOG/film_self_k3_lsem_uniform_seed${seed}_train.log"
    local evallog="$LOG/film_self_k3_lsem_uniform_seed${seed}_eval.log"
    local outjson="bench_stmt_ppl_lsem_uniform_seed${seed}.json"
    if [[ -f "$ckpt" ]]; then
        echo "[seed$seed] skipping training (ckpt $ckpt exists)"
    else
        echo "[seed$seed] training -> $ckpt"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/train_lm.py \
            --arch deltanet \
            --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
            --semantic_loss_beta 1.0 --semantic_loss_uniform_weight \
            --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
            --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
            --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed "$seed" \
            --dataset codeparrot/codeparrot-clean --text_field content \
            --log_every 500 --val_every 2500 \
            --save_ckpt "$ckpt" \
            > "$trainlog" 2>&1
        echo "[seed$seed] training done. Tail:"
        tail -5 "$trainlog"
    fi
    if [[ -f "$outjson" ]]; then
        echo "[seed$seed] skipping eval ($outjson exists)"
    else
        echo "[seed$seed] eval -> $outjson"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/eval_statement_ppl.py \
            --ckpt "$ckpt" \
            --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
            --oracle_ckpt checkpoints/oracle_predictive_head_217M.pt \
            --T 512 --n_eval_tokens 32768 \
            --out "$outjson" \
            > "$evallog" 2>&1
        echo "[seed$seed] eval done. Tail:"
        grep -E "Overall:|Top decile:|Bottom decile:|overall PPL" "$evallog" | head -8
    fi
}

run_tinystories_dn_baseline() {
    local ckpt="checkpoints/dn_baseline_30L_217M_tinystories_seed0.pt"
    local trainlog="$LOG/dn_baseline_tinystories_train.log"
    local evallog="$LOG/dn_baseline_tinystories_eval.log"
    local outjson="bench_filmed_ppl_217M_tinystories_dn.json"
    if [[ -f "$ckpt" ]]; then
        echo "[ts-dn] skipping training (ckpt exists)"
    else
        echo "[ts-dn] training plain DN on TinyStories -> $ckpt"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/train_lm.py \
            --arch deltanet \
            --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
            --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
            --dataset roneneldan/TinyStories --text_field text \
            --log_every 500 --val_every 2500 \
            --save_ckpt "$ckpt" \
            > "$trainlog" 2>&1
        echo "[ts-dn] training done. Tail:"
        tail -5 "$trainlog"
    fi
    if [[ -f "$outjson" ]]; then
        echo "[ts-dn] skipping eval"
    else
        echo "[ts-dn] eval"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/eval_filmed_ppl_217m.py \
            --ckpt "$ckpt" \
            --T 512 --n_tokens 32768 --batch 4 \
            --dataset roneneldan/TinyStories --text_field text \
            --skip_lagged_cached \
            --out "$outjson" \
            > "$evallog" 2>&1
        grep -E "PPL =|training-protocol" "$evallog" | head -5
    fi
}

run_tinystories_k3() {
    local ckpt="checkpoints/film_self_k3_30L_217M_tinystories_seed0.pt"
    local trainlog="$LOG/film_self_k3_tinystories_train.log"
    local evallog="$LOG/film_self_k3_tinystories_eval.log"
    local outjson="bench_filmed_ppl_217M_tinystories_k3.json"
    if [[ -f "$ckpt" ]]; then
        echo "[ts-k3] skipping training (ckpt exists)"
    else
        echo "[ts-k3] training K=3 self-feeding on TinyStories -> $ckpt"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/train_lm.py \
            --arch deltanet \
            --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
            --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
            --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
            --dataset roneneldan/TinyStories --text_field text \
            --log_every 500 --val_every 2500 \
            --save_ckpt "$ckpt" \
            > "$trainlog" 2>&1
        echo "[ts-k3] training done. Tail:"
        tail -5 "$trainlog"
    fi
    if [[ -f "$outjson" ]]; then
        echo "[ts-k3] skipping eval"
    else
        echo "[ts-k3] eval"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/eval_filmed_ppl_217m.py \
            --ckpt "$ckpt" \
            --T 512 --n_tokens 32768 --batch 4 \
            --dataset roneneldan/TinyStories --text_field text \
            --skip_lagged_cached \
            --out "$outjson" \
            > "$evallog" 2>&1
        grep -E "PPL =|training-protocol" "$evallog" | head -5
    fi
}

run_tinystories_k3_lsem() {
    # Per-token L_sem: AST segmentation doesn't apply to TinyStories prose.
    # Use --semantic_loss_granularity token (already implemented in train_lm.py).
    # The encoder is the TinyStories DN baseline (must be trained first).
    local enc="checkpoints/dn_baseline_30L_217M_tinystories_seed0.pt"
    local ckpt="checkpoints/film_self_k3_lsem_pertoken_b10_30L_217M_tinystories_seed0.pt"
    local trainlog="$LOG/film_self_k3_lsem_pertoken_tinystories_train.log"
    local evallog="$LOG/film_self_k3_lsem_pertoken_tinystories_eval.log"
    local outjson="bench_filmed_ppl_217M_tinystories_k3_lsem.json"
    if [[ ! -f "$enc" ]]; then
        echo "[ts-k3-lsem] encoder $enc not found — TinyStories DN must be trained first"
        return 1
    fi
    if [[ -f "$ckpt" ]]; then
        echo "[ts-k3-lsem] skipping training (ckpt exists)"
    else
        echo "[ts-k3-lsem] training K=3 + per-token L_sem β=1 on TinyStories -> $ckpt"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/train_lm.py \
            --arch deltanet \
            --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
            --semantic_loss_beta 1.0 --semantic_loss_uniform_weight \
            --semantic_loss_granularity token \
            --encoder_ckpt "$enc" \
            --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
            --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
            --dataset roneneldan/TinyStories --text_field text \
            --log_every 500 --val_every 2500 \
            --save_ckpt "$ckpt" \
            > "$trainlog" 2>&1
        echo "[ts-k3-lsem] training done. Tail:"
        tail -5 "$trainlog"
    fi
    if [[ -f "$outjson" ]]; then
        echo "[ts-k3-lsem] skipping eval"
    else
        echo "[ts-k3-lsem] eval"
        wait_for_gpu "$GPU"
        CUDA_VISIBLE_DEVICES=$GPU $PY -u experiments/eval_filmed_ppl_217m.py \
            --ckpt "$ckpt" \
            --T 512 --n_tokens 32768 --batch 4 \
            --dataset roneneldan/TinyStories --text_field text \
            --skip_lagged_cached \
            --out "$outjson" \
            > "$evallog" 2>&1
        grep -E "PPL =|training-protocol" "$evallog" | head -5
    fi
}

# Sequence:
# 1. Multi-seed: codeparrot K=3+L_sem at seeds 1, 2.
# 2. TinyStories: Plain DN (encoder for the L_sem run).
# 3. TinyStories: K=3 only.
# 4. TinyStories: K=3 + per-token L_sem.
echo "=== Phase 22 validation launcher ==="
echo "Date: $(date)"
echo
echo "--- 1. Multi-seed: K=3 + L_sem on codeparrot, seed=1 ---"
run_codeparrot_lsem_seed 1
echo
echo "--- 2. Multi-seed: K=3 + L_sem on codeparrot, seed=2 ---"
run_codeparrot_lsem_seed 2
echo
echo "--- 3. TinyStories: plain DN baseline ---"
run_tinystories_dn_baseline
echo
echo "--- 4. TinyStories: K=3 self-feeding only ---"
run_tinystories_k3
echo
echo "--- 5. TinyStories: K=3 + per-token L_sem ---"
run_tinystories_k3_lsem
echo
echo "=== All done $(date) ==="
