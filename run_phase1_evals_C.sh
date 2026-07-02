#!/bin/bash
# Arm C (freeze-trunk pre-warm) eval runner. Waits for Arm C to exit, then evals
# the FINAL ckpt + the post-freeze mid-eval ckpt CE on the free GPU, single-process.
exec >/tmp/phase1_evals_C.log 2>&1
set -x
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
GPU=1
CPID=${1:?need Arm C pid}
echo "[evalC] waiting for Arm C pid $CPID at $(date)"
while kill -0 "$CPID" 2>/dev/null; do sleep 60; done
echo "[evalC] Arm C exited at $(date)"; sleep 10

: > /tmp/phase1_ce3.txt
echo "ANCHORS: base=0.7585  ArmA(feat OFF)=0.7523  ArmB(feat ON)=0.7573" | tee -a /tmp/phase1_ce3.txt
# Eval the final + every mid-eval ckpt (trajectory: post-freeze -> end)
for CK in checkpoints/phase1_ab_C.pt checkpoints/phase1_ab_C_step*.pt; do
  [ -f "$CK" ] || continue
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python /tmp/probe_he_ce.py "$CK" 2>>/tmp/phase1_ce3.err | tee -a /tmp/phase1_ce3.txt
done

echo "[evalC] pass@1 (C final) at $(date)"
: > /tmp/phase1_pass3.txt
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python experiments/eval_humaneval.py \
  --ckpt checkpoints/phase1_ab_C.pt --n_samples 1 --temperature 0.0 --min_emit_before_eos 30 \
  >> /tmp/phase1_pass3.txt 2>>/tmp/phase1_pass3.err || echo "pass@1 FAILED" >> /tmp/phase1_pass3.txt

echo "[evalC] DONE at $(date)"; touch /tmp/phase1_DONE3
