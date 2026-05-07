#!/usr/bin/env bash
# Run the DN-4B distillation pilot validation: 1B plain-DN student
# trained for 1K steps in two modes (KL+CE vs CE-only). Same data,
# same schedule, same seed. The only variable is the loss.
set -euo pipefail

WORKTREE=/home/knielsen/ml/parallel-ss-dep-distill
PYTHON=/home/knielsen/ml/parallel-ss-dep/.venv/bin/python
SHARDS="${WORKTREE}/data/distill_pilot_1M"
LOGS="${WORKTREE}/logs/distill_pilot"
METRICS="${WORKTREE}/logs/distill_pilot/metrics"

mkdir -p "${LOGS}" "${METRICS}"

# Common flags. Batch 4 is the validation baseline (matches the spec's
# "small batch" guidance and ~2 M training tokens at 1 K steps). Bump
# to 8 if you want faster wall-clock and your card has the headroom
# (947 M plain-DN bf16 at T=512 fits comfortably in 30+ GB free
# after the teacher exits).
COMMON=(--shards "${SHARDS}" --steps 1000 --batch 4
        --d_model 1280 --n_heads 20 --d_head 64 --n_layers 24
        --top_k 32 --val_chunks 64 --val_every 250 --log_every 50
        --seed 0 --lr 3e-4)

cd "${WORKTREE}"

echo "== KL+CE distillation run (alpha=0.5) =="
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u experiments/distill_pilot.py \
  "${COMMON[@]}" --mode kl_ce --alpha 0.5 \
  --save_metrics "${METRICS}/kl_ce.json" \
  2>&1 | tee "${LOGS}/kl_ce.log"

echo "== KL+CE distillation run (alpha=0.8 — light regularizer) =="
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u experiments/distill_pilot.py \
  "${COMMON[@]}" --mode kl_ce --alpha 0.8 \
  --save_metrics "${METRICS}/kl_ce_a08.json" \
  2>&1 | tee "${LOGS}/kl_ce_a08.log"

echo "== KL+CE distillation run (alpha=0.9 — recipe to scale) =="
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u experiments/distill_pilot.py \
  "${COMMON[@]}" --mode kl_ce --alpha 0.9 \
  --save_metrics "${METRICS}/kl_ce_a09.json" \
  2>&1 | tee "${LOGS}/kl_ce_a09.log"

echo "== CE-only baseline run =="
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u experiments/distill_pilot.py \
  "${COMMON[@]}" --mode ce \
  --save_metrics "${METRICS}/ce.json" \
  2>&1 | tee "${LOGS}/ce.log"

echo
echo "== summary =="
"${PYTHON}" -c "
import json, pathlib
m = pathlib.Path('${METRICS}')
results = {}
for name in ['kl_ce', 'kl_ce_a08', 'kl_ce_a09', 'ce']:
    fp = m / f'{name}.json'
    if not fp.exists():
        print(f'  {name}: missing'); continue
    d = json.loads(fp.read_text())
    fv = d['final_val']
    results[name] = (fv['val_ppl'], d['alpha'])
    print(f'  {name:12s}  alpha={d[\"alpha\"]:.1f}  val_ppl={fv[\"val_ppl\"]:.2f}  val_ce={fv[\"val_ce\"]:.4f}  val_kl={fv[\"val_kl\"]:.4f}  ({d[\"wallclock_s\"]:.0f}s)')
ce_ppl = results.get('ce', (None, 0))[0]
if ce_ppl:
    print()
    for k, (ppl, a) in results.items():
        if k == 'ce': continue
        rel = (ppl - ce_ppl) / ce_ppl * 100
        sign = '+' if rel >= 0 else ''
        verdict = 'WORSE' if rel > 0 else 'BETTER'
        print(f'  {k:12s}  vs CE-only: {sign}{rel:.1f} % PPL  ({verdict})')
"
