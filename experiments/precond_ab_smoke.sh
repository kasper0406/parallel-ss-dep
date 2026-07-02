#!/bin/bash
# SMOKE for the matrix-optimizer A/B wiring (GPU 1 only).
# Runs 3 short jobs on the real harness + real trunk:
#   1. muon, NO --matrix_optimizer flag   (legacy / backwards-compat baseline)
#   2. --matrix_optimizer muon            (must be byte-identical to #1)
#   3. --matrix_optimizer fused_deltanet_ns
# Then verifies: all train w/o error; step-1 tloss byte-identical across all 3
# (same init+data, optimizer only acts from step 1); #1 == #2 at EVERY step
# (backwards-compat); #3 diverges from #1 after step 1 (optimizer differs).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/precond_ab_smoke
mkdir -p $OUT

# Real production trunk (10L x 896d x 14h), clean optimizer probe: no
# thinking/WM/PKM/gate, think-burst injection OFF. --no-compile for fast,
# deterministic smoke. K=3 FiLM engaged from step 0 (warmup 0) to exercise the
# real path + peak memory. Small batch/grad_accum/steps purely for smoke speed.
common() {
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 --feedback_self_k_warmup_steps 0 \
    --data_mix configs/pretrain_mix_v4.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0 \
    --T 2048 --batch 8 --grad_accum 2 \
    --activation_checkpointing \
    --bf16 --tf32 --no-compile --bf16_optim_state \
    --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.15 \
    --steps 3 --val_every 1000 --log_every 1 --seed 0 "$@"
}

echo "=== [1/3] muon (NO flag) ==="; common                                  > $OUT/noflag.log 2>&1
echo "=== [2/3] --matrix_optimizer muon ==="; common --matrix_optimizer muon > $OUT/muon.log 2>&1
echo "=== [3/3] --matrix_optimizer fused_deltanet_ns ==="; common --matrix_optimizer fused_deltanet_ns > $OUT/fused.log 2>&1

echo; echo "=== SMOKE VERDICT ==="
.venv/bin/python - "$OUT" <<'PY'
import sys, re, pathlib
out = pathlib.Path(sys.argv[1])
def tloss(log):
    # data lines: "  step    tok/s    tloss    lr"; tloss is 3rd float col.
    rows = {}
    for ln in log.read_text().splitlines():
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+-]+)", ln)
        if m:
            rows[int(m.group(1))] = float(m.group(3))
    return rows
def trained_ok(log):
    t = log.read_text()
    return ("Traceback" not in t) and ("Error" not in t.split("WARNING")[0] if False else "Traceback" not in t)
nf, mu, fu = tloss(out/"noflag.log"), tloss(out/"muon.log"), tloss(out/"fused.log")
print("noflag:", nf); print("muon  :", mu); print("fused :", fu)
ok = True
for name, log in [("noflag","noflag.log"),("muon","muon.log"),("fused","fused.log")]:
    txt = (out/log).read_text()
    if "Traceback" in txt:
        print(f"FAIL: {name} crashed (Traceback in log)"); ok = False
    if "matrix_optimizer=fused_deltanet_ns" in txt:
        for l in txt.splitlines():
            if "matrix_optimizer=fused_deltanet_ns" in l: print("  [fused build]", l.strip())
# (a) step-1 identical across all arms (same init+data)
s1 = [r.get(1) for r in (nf,mu,fu)]
if len(set(s1)) == 1 and s1[0] is not None:
    print(f"PASS (b): step-1 tloss byte-identical across arms = {s1[0]}")
else:
    print(f"FAIL (b): step-1 tloss differs across arms: {s1}"); ok = False
# (c) noflag == muon at EVERY logged step (backwards-compat)
if nf == mu and nf:
    print(f"PASS (c): --matrix_optimizer muon byte-identical to no-flag at all steps")
else:
    print(f"FAIL (c): noflag vs muon differ: {nf} vs {mu}"); ok = False
# fused should diverge after step 1 (sanity the optimizer actually differs)
later = [s for s in fu if s > 1]
if later and any(fu[s] != mu.get(s) for s in later):
    print(f"OK: fused diverges from muon after step 1 (optimizer differs as expected)")
else:
    print(f"WARN: fused did not diverge from muon at steps>1 (check it ran enough steps): fused={fu} muon={mu}")
print("\nSMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
