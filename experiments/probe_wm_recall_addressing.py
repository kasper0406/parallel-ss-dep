"""Localize WHY trained WM barely helps multibind recall: write vs read vs readout.

On the trained Stage A ckpt, for each multibind example we run ONE forward over
`[flattened_program, THINK]` and capture the WM read at the think slot
(`memory._capture_read` → `_last_read_attn` (B,T,Kbuf), `_last_top_idx` (B,Kbuf)
buffer source positions). The queried variable's binding (`vX = NNNN`) lives at a
known token span (the FIRST occurrence of the answer value in the prompt). We ask:

  - WRITE: is the binding's value token even IN the buffer (a top_idx slot whose
    source position lands in the binding span)? If not, the write-gate didn't
    select it → write-side failure.
  - READ: does the read-attention at the think slot put its mass on that slot?
    top1-on-binding and mass-on-binding vs a diffuse/wrong distribution → read/
    addressing failure.

If the binding IS in the buffer and the read attends to it but recall still
fails, the bottleneck is the READOUT (W_proj/emit), not addressing. This
disambiguates the +2.7pp ceiling.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/probe_wm_recall_addressing.py \
      --ckpt checkpoints/wm_cotrain_stage_a.pt --n 80
"""
from __future__ import annotations

import argparse
import json

import torch


def _find_sub(hay: list[int], needle: list[int]) -> int:
    """First index where `needle` occurs in `hay`, else -1."""
    if not needle:
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/wm_cotrain_stage_a.pt")
    p.add_argument("--tasks", default="data/multibind_recall_heldout.jsonl")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.sft_code import _flatten_to_oneline
    from transformers import AutoTokenizer

    model, cfg = build_model_from_ckpt(args.ckpt, force_state_readonly=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    tid = int(getattr(model, "thinking_token_id", cfg.get("thinking_token_id")))
    mem = model.memory
    mem._capture_read = True

    recs = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs = recs[:args.n]

    n = 0
    write_hit = 0          # binding value token is in the buffer
    read_top1_hit = 0      # read-attn top-1 slot is the binding
    read_mass = 0.0        # read-attn mass on binding slot(s)
    top1_mass_sum = 0.0    # how peaky is the read in general
    with torch.no_grad():
        for rec in recs:
            prompt = "# " + _flatten_to_oneline(rec["problem_prompt"]) + "\n"
            ids = tok.encode(prompt, add_special_tokens=False)
            ans_ids = tok.encode(str(rec["answer"]), add_special_tokens=False)
            # binding value span = FIRST occurrence of the answer value in the
            # prompt (it appears once, at `vX = NNNN`).
            bpos = _find_sub(ids, ans_ids)
            if bpos < 0:
                continue
            bind_span = set(range(bpos, bpos + len(ans_ids)))
            # forward [prompt, THINK] → read at the think slot (last position)
            seq = torch.tensor([ids + [tid]], dtype=torch.long, device=args.device)
            model(seq, return_hidden=False)
            attn = mem._last_read_attn          # (1, T, Kbuf)
            top_idx = mem._last_top_idx          # (1, Kbuf) source positions
            if attn is None or top_idx is None:
                continue
            a = attn[0, -1]                      # (Kbuf,) read at think slot
            src = top_idx[0]                     # (Kbuf,)
            # which buffer slots are the binding's value tokens?
            in_bind = torch.tensor(
                [int(s.item()) in bind_span for s in src],
                device=a.device, dtype=torch.bool)
            n += 1
            write_hit += int(in_bind.any().item())
            top1 = int(a.argmax().item())
            top1_mass_sum += float(a.max().item())
            if in_bind.any():
                read_mass += float(a[in_bind].sum().item())
                read_top1_hit += int(in_bind[top1].item())

    if n == 0:
        print("no usable examples")
        return 1
    print(f"[probe] {args.ckpt}  n={n}  Kbuf={int(top_idx.shape[-1])}")
    print(f"  WRITE: binding value in buffer        = {write_hit/n:.2%}")
    print(f"  READ : top-1 read slot IS the binding = {read_top1_hit/n:.2%}")
    print(f"  READ : mean read-mass on binding slot = {read_mass/n:.3f}")
    print(f"  (mean top-1 read mass overall, peakiness) = {top1_mass_sum/n:.3f}")
    print("\nINTERPRETATION:")
    print("  write low  → trunk/write-gate doesn't store the binding (write-side)")
    print("  write high + read low → addressing (W_q/W_k) can't find it (read-side)")
    print("  write+read high but recall low → readout (W_proj/emit) is the limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
