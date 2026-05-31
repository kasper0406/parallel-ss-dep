#!/bin/bash
# LLM-ranker judge server: Qwen2.5-Coder-3B-AWQ via vLLM, OpenAI API on :8000.
# Co-resides with the RL trainer on one GPU, capped at ~6.5 GB (util 0.20).
# Started in the .venv-vllm (vLLM 0.20.0; ABI-incompatible with the training
# venv). The trainer reaches it via --judge_url http://localhost:8000.
cd /home/knielsen/ml/parallel-ss-dep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs
CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv-vllm/bin/vllm serve \
    Qwen/Qwen2.5-Coder-3B-Instruct-AWQ \
    --port ${JUDGE_PORT:-8000} \
    --gpu-memory-utilization ${JUDGE_UTIL:-0.20} \
    --max-model-len 4096 \
    > runs/judge_server.log 2>&1 &
echo "judge server PID $!"
