"""Tiny GPU smoke test: build the 1B plain-DN student + 1 forward+backward.

Purpose: verify the student infra is sane at 1B scale before kicking off
the full 1K-step validation runs. Uses just 1 batch of synthetic data;
no shards needed.

Run:
  CUDA_VISIBLE_DEVICES=0 /home/knielsen/ml/parallel-ss-dep/.venv/bin/python \
    experiments/distill_pilot_gpu_smoke.py
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"  free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB / "
          f"{torch.cuda.mem_get_info()[1]/1e9:.1f} GB total")

    V = 248064
    T = 512
    B = 4
    print(f"Building TinyLM (vocab={V}, d=1280, h=20, dh=64, L=24, T={T}, B={B}) ...")
    t0 = time.time()
    model = TinyLM(
        vocab_size=V, d_model=1280, n_heads=20, d_head=64, n_layers=24,
        attention_cls=DeltaNetAttention, feedback_mode="none",
        tie_embeddings=True,
    ).cuda()
    print(f"  build+to-cuda in {time.time()-t0:.1f}s; "
          f"params={model.num_params()/1e6:.1f} M")
    print(f"  free after model: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # Synthetic batch.
    torch.manual_seed(0)
    x = torch.randint(0, V, (B, T), device="cuda")
    tk_ids = torch.randint(0, V, (B, T, 20), device="cuda")
    tk_lps = torch.randn(B, T, 20, device="cuda")  # all finite

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Forward + backward with bf16 autocast.
    print("Forward+backward (bf16 autocast) ...")
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        # Toy KL+CE loss matching distill_pilot.compute_loss shape.
        sl = logits[:, :-1, :]
        tgt = x[:, 1:]
        ce = F.cross_entropy(sl.reshape(-1, V), tgt.reshape(-1))
        gathered = sl.gather(2, tk_ids[:, 1:, :].long())
        teacher_log_p = F.log_softmax(tk_lps[:, 1:, :].float(), dim=-1)
        teacher_p = teacher_log_p.exp()
        student_log_p = F.log_softmax(gathered.float(), dim=-1)
        kl = (teacher_p * (teacher_log_p - student_log_p)).sum(-1).mean()
        loss = 0.5 * kl + 0.5 * ce
    loss.backward()
    fwd_bwd = time.time() - t0
    optim.step()
    print(f"  loss={loss.item():.4f}  ce={ce.item():.4f}  kl={kl.item():.4f}")
    print(f"  fwd+bwd+step in {fwd_bwd:.2f}s -> {B*T/fwd_bwd:.0f} tok/s")
    print(f"  free after step: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB  "
          f"max_mem={torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    print("OK")


if __name__ == "__main__":
    main()
