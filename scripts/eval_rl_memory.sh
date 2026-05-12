#!/bin/bash
# Post-training eval for the RL+memory experiment.
#
# For each of {base SFT, RL mem-on, RL mem-off}:
#   - probe via experiments/probe_thinking.py (gate-vs-CE Spearman,
#     depth histogram, memory diagnostics)
#   - HumanEval pass@1 on 50 problems
#
# Runs the two RL ckpts in parallel on GPUs 0 and 1; base SFT goes after.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p runs/rl_eval

# 1. HumanEval (50 problems, greedy) in parallel on the two RL ckpts.
echo "[$(date +%H:%M:%S)] HumanEval on RL mem-on (GPU 0)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -u experiments/eval_humaneval.py \
    --ckpt checkpoints/rl_memon_d16.pt \
    --max_problems 50 --max_gen 256 --temperature 0.0 \
    > runs/rl_eval/heval_rl_memon.log 2>&1 &
PID_ON=$!

echo "[$(date +%H:%M:%S)] HumanEval on RL mem-off (GPU 1)"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -u experiments/eval_humaneval.py \
    --ckpt checkpoints/rl_memoff_d16.pt \
    --max_problems 50 --max_gen 256 --temperature 0.0 \
    > runs/rl_eval/heval_rl_memoff.log 2>&1 &
PID_OFF=$!

wait "$PID_ON" "$PID_OFF"

# 2. Probes — sequential, short.
echo "[$(date +%H:%M:%S)] Probe on RL mem-on"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -u experiments/probe_thinking.py \
    --ckpt checkpoints/rl_memon_d16.pt --use_memory --mem_size 512 \
    --n_samples 32 --max_depth 5 --grpo_n_group 4 --T 512 \
    --min_decision_pos 16 --seed 42 \
    > runs/rl_eval/probe_rl_memon.log 2>&1

echo "[$(date +%H:%M:%S)] Probe on RL mem-off"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -u experiments/probe_thinking.py \
    --ckpt checkpoints/rl_memoff_d16.pt \
    --n_samples 32 --max_depth 5 --grpo_n_group 4 --T 512 \
    --min_decision_pos 16 --seed 42 \
    > runs/rl_eval/probe_rl_memoff.log 2>&1

# 3. Summary
echo
echo '====================== SUMMARY ======================'
echo
echo '=== HumanEval ==='
for tag in memon memoff; do
    grep -E '^pass@' runs/rl_eval/heval_rl_${tag}.log | head -1 | xargs -I{} echo "  rl_${tag}: {}"
done
echo
echo '=== Probe — mem-on ==='
sed -n '/PROBE REPORT/,/===========/p' runs/rl_eval/probe_rl_memon.log
echo '=== Probe — mem-off ==='
sed -n '/PROBE REPORT/,/===========/p' runs/rl_eval/probe_rl_memoff.log
