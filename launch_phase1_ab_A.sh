#!/bin/bash
# PHASE-1 A/B  —  ARM A (CONTROL): continue-pretrain the linearized 402M base
# with FEATURES OFF. Identical data / tokens / schedule / optimizer to ARM B;
# the ONLY difference is the absence of FiLM / WM / PKM / latent / gate.
#
#   base   : checkpoints/linearize/linearized_stage3.pt
#            (DeltaNet d960 x 32L x 15h x d_head64, d_ff2560, TIED 49152, feedback=none)
#   data   : configs/pretrain_mix_v18_arxiv.yaml  (tok HuggingFaceTB/SmolLM2-135M == 49152)
#   budget : 1900 steps x (batch2 x grad_accum32 x T2048 = 131072 tok) ~= 249.0M tokens
#            (batch2 chosen so ARM B's K=3 self-feed + latent stack fits 32GB;
#             total compute for a fixed token budget is invariant to batch/ga split.
#             IDENTICAL batch/grad_accum/steps in both arms -> identical token stream.)
#   vocab  : --keep_base_vocab 49152  -> NO embedding resize (load is 0-mismatch)
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-1}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --d_head 64 --n_heads 15 --tie_embeddings --d_ff 2560 \
  --feedback none \
  --load_ckpt checkpoints/linearize/linearized_stage3.pt --keep_base_vocab 49152 \
  --data_mix configs/pretrain_mix_v18_arxiv.yaml --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --num_workers 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --wd 0.01 --alpha_wd 0.0 --grad_clip 1.0 --z_loss 1e-4 \
  --optimizer muon --lr 3e-4 --lr_muon 1e-3 --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.15 \
  --steps 1900 --val_every 200 --log_every 20 --seed 1234 \
  --mid_eval_every_tokens 100000000 --mid_eval_save_only \
  --save_ckpt checkpoints/phase1_ab_A.pt --tb_dir runs/tb/phase1_ab_A \
  > runs/phase1_ab_A.log 2>&1 &
echo "Launched PHASE1 ARM A (control), GPU $GPU (PID $!).  Watch: tail -f runs/phase1_ab_A.log"
