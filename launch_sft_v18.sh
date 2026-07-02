#!/bin/bash
# SFT on the v18 base (data-hygiene fix: revived magicoder+textbooks dead
# sources + arXiv; day-1 co-trained PKM/WM(mem_ctx_namekey)/latent). Same
# ast-clean data-hygiene corpus + recipe as launch_sft_phasec_astclean.sh
# (which gave Phase-C-SFT = 14/164 same-config) so the comparison isolates the
# BASE-pretrain difference (v18 data fix + integrated mechanisms) vs Phase C.
#
# DIFFERENCES vs the phasec launcher, all deliberate:
#   - base = checkpoints/pretrain_v18.pt (final).
#   - DROP --retrieval_as_input_thinking: DEPRECATED. v18's WM is the no-hash
#     mem_ctx_namekey addresser + copy head (--mem_always_read --emit_read_mask),
#     auto-detected from the ckpt cfg by build_model_from_ckpt. Do NOT force the
#     legacy retrieval-as-input path on it.
#   - ADD --state_readonly_at_think: v18 trained with it (DeltaNet beta=0 at think
#     positions so thinking can't corrupt recall). The modern load path flips it
#     on + installs the Block hook.
#   - DROP --mem_size override: let build_model_from_ckpt read v18's mem_size
#     (2048) from cfg instead of forcing 1024.
#
# VALIDATE AT LAUNCH (cheap build_model_from_ckpt sanity check, see below) that
# the final ckpt round-trips: memory.* present, gate_head present, pkm_layer
# present, mem_ctx_namekey + copy head in cfg, state_readonly flips on.
#
# Eval thinking-OFF (the headline number; the apparatus is for downstream RL +
# recall probes): --prompt_style sft_comment --extract_code_block --max_gen 512.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

BASE=${BASE:-checkpoints/pretrain_v18.pt}
if [ ! -f "$BASE" ]; then echo "ERROR: base ckpt $BASE missing"; exit 1; fi

# Rebuild the ast-clean corpus (clean-distill + synthmem + longctx), same as phasec.
if [ ! -f data/distill_v7_phase1_astclean.jsonl ]; then
  echo "ERROR: data/distill_v7_phase1_astclean.jsonl missing"; exit 1
fi
cat data/distill_v7_phase1_astclean.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_v18_astclean.jsonl
echo "[sft-v18] corpus rows: $(wc -l < data/sft_v18_astclean.jsonl)"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt "$BASE" \
    --save_ckpt checkpoints/sft_v18.pt \
    --distilled_jsonl data/sft_v18_astclean.jsonl \
    --with_thinking \
    --state_readonly_at_think \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 100 \
    > runs/sft_v18.log 2>&1 &
echo "Launched v18 SFT, GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v18.log"
