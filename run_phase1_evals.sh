#!/bin/bash
# Clean Phase-1 eval runner. SINGLE process, runs ONLY after Arm B exits, on the
# then-free GPU (no concurrency — the previous orchestrator ran evals while the
# dying Arm B still held the GPU → garbage). Waits for Arm B PID, verifies the
# ckpt, runs CE (base/A/B) + pass@1 (A/B), writes results, touches DONE2.
exec >/tmp/phase1_evals.log 2>&1
set -x
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
GPU=1
BPID=${1:?need Arm B pid}

echo "[eval] waiting for Arm B pid $BPID at $(date)"
while kill -0 "$BPID" 2>/dev/null; do sleep 60; done
echo "[eval] Arm B exited at $(date)"
sleep 10

if [ ! -f checkpoints/phase1_ab_B.pt ]; then
  echo "[eval] FATAL: checkpoints/phase1_ab_B.pt missing — Arm B did not save (crash?)."
  tail -15 runs/phase1_ab_B.log
  touch /tmp/phase1_DONE2; exit 1
fi
ls -la checkpoints/phase1_ab_B.pt

echo "[eval] CE probes (base / A / B) at $(date)"
: > /tmp/phase1_ce2.txt
for CK in checkpoints/linearize/linearized_stage3.pt checkpoints/phase1_ab_A.pt checkpoints/phase1_ab_B.pt; do
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python /tmp/probe_he_ce.py "$CK" 2>>/tmp/phase1_ce2.err | tee -a /tmp/phase1_ce2.txt
done

echo "[eval] pass@1 (greedy, A / B) at $(date)"
: > /tmp/phase1_pass2.txt
for CK in checkpoints/phase1_ab_A.pt checkpoints/phase1_ab_B.pt; do
  echo "=== $CK ===" >> /tmp/phase1_pass2.txt
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python experiments/eval_humaneval.py \
    --ckpt "$CK" --n_samples 1 --temperature 0.0 --min_emit_before_eos 30 \
    >> /tmp/phase1_pass2.txt 2>>/tmp/phase1_pass2.err || echo "pass@1 FAILED for $CK" >> /tmp/phase1_pass2.txt
done

echo "[eval] ALL DONE at $(date)"
touch /tmp/phase1_DONE2
