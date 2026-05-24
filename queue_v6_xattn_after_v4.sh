#!/bin/bash
# Wait for v4 pretrain (PID 4130767) to exit, then launch v6-xattn on GPU 0.
V4_PID=4130767
echo "[queue] watching v4 PID $V4_PID; will fire v6-xattn on exit"
while kill -0 $V4_PID 2>/dev/null; do
  sleep 60
done
echo "[queue] v4 exited at $(date -Iseconds); launching v6-xattn"
GPU=0 /home/knielsen/ml/parallel-ss-dep/launch_pretrain_mix_v6_xattn.sh
