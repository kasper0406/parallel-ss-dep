#!/usr/bin/env bash
# Extract final PPL and final α from Phase 21 logs.
LOG_DIR=/home/knielsen/ml/parallel-ss-dep/logs/phase21
for f in "$LOG_DIR"/phase21_*.log; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .log)
    final_ppl=$(grep "VAL  loss" "$f" | tail -1 | awk -F'ppl=' '{print $2}')
    last_alpha_line=$(grep -E "α=\[" "$f" | tail -1)
    alpha=$(echo "$last_alpha_line" | awk -F'α=\\[' '{print $2}' | awk -F'\\]' '{print $1}')
    done_line=$(grep "Done in" "$f")
    echo "[$label]"
    echo "  final_ppl: ${final_ppl:-<not done>}"
    echo "  alpha:     ${alpha:-<no FiLM>}"
    echo "  $done_line"
done
