"""Mechanistic probe: trace the latent-thinking thread (2026-06-02).

Tests the theory that latent thinking is an ITERATED MAP  z_k = g(z_{k-1}; S)
(S = recurrent state frozen at the prefix) that helps only iterated-map-shaped
tasks, and otherwise DRIFTS OFF the input-embedding manifold (degrading/hurting
the emit). Runs R forced latent steps on three position classes and records, per
step k (k=0 = no-think):
  - dlogp(true_next): does thinking improve the true-next prediction? helps vs hurts
  - max cos(z_k, embedding table): does z stay ON the input-embedding manifold?
  - ||z_k - z_{k-1}||: does it iterate, or collapse to a fixed point (no-op)?
  - ||z_k||
  - decode = argmax lm_head at the think slot (qualitative: orbit vs garbage)

Classes: pointer-chase (depth-bound, iterated-map) | general text | code.
If the theory holds: pointer-chase dlogp climbs + z stays on-manifold + nonzero dz;
text/code dlogp flat-or-negative + z drifts off-manifold (cos drops) + dz collapses.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/probe_latent_thread.py \
      --ckpt checkpoints/latent_transfer_v6.pt --R 5 --n 24
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.sft_code import load_distilled_jsonl

GENERAL_TEXT = [
    "The capital of France is Paris, a city famous for its art and history.",
    "She opened the door slowly and saw that the room was completely empty.",
    "Climate scientists warned that global temperatures would continue to rise.",
    "He picked up the phone and dialed the number he had memorized years ago.",
    "The committee agreed that the proposal needed further review before approval.",
    "After the long winter, the first flowers finally began to bloom in spring.",
    "The teacher explained that the experiment required careful measurement and patience.",
    "They drove through the mountains for hours before reaching the small village.",
    "Economists disagree about whether the new policy will reduce unemployment.",
    "The old library held thousands of books that nobody had read in decades.",
    "Researchers discovered a new species of frog deep in the rainforest canopy.",
    "Every morning she ran along the river before the city woke up and traffic began.",
]


@torch.no_grad()
def trace(model, prefix_ids, R, thinking_id, true_next, device, emb_norm):
    base = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    cur_ids, cur_emb = base, model.embed(base)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long, device=device)
    rows, prev = [], None
    for k in range(R + 1):                                  # k=0 == no-think
        out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        logits = out[0] if isinstance(out, tuple) else out
        h = out[1]
        last = logits[0, -1].float()
        lp = float(torch.log_softmax(last, -1)[true_next])
        dec = int(last.argmax())
        z = model.apply_latent_feedback_adapter(h[:, -1:, :])[0, 0].float()
        zn = z / (z.norm() + 1e-6)
        max_cos = float((emb_norm @ zn).max())
        dz = float((z - prev).norm()) if prev is not None else 0.0
        prev = z
        rows.append(dict(k=k, lp=lp, max_cos=max_cos, znorm=float(z.norm()),
                         dz=dz, dec=dec))
        if k < R:
            cur_ids = torch.cat([cur_ids, think_tok], dim=1)
            cur_emb = torch.cat([cur_emb, z.view(1, 1, -1).to(cur_emb.dtype)], dim=1)
    return rows


def _agg(all_rows, R):
    """Mean over samples of each metric at each k; dlp[k] = lp[k]-lp[0]."""
    out = []
    for k in range(R + 1):
        lp = [r[k]["lp"] for r in all_rows]
        dlp = [r[k]["lp"] - r[0]["lp"] for r in all_rows]
        mc = [r[k]["max_cos"] for r in all_rows]
        dz = [r[k]["dz"] for r in all_rows]
        zn = [r[k]["znorm"] for r in all_rows]
        m = lambda v: sum(v) / len(v)
        out.append((k, m(dlp), m(mc), m(dz), m(zn)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/latent_transfer_v6.pt")
    ap.add_argument("--ptr_heldout", default="data/ptrdict_heldout_n5.jsonl")
    ap.add_argument("--code_jsonl", default="data/sft_phase_c_combined.jsonl")
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--code_prefix_len", type=int, default=48)
    args = ap.parse_args()
    device = "cuda"

    model, cfg = build_model_from_ckpt(args.ckpt, force_state_readonly=True,
                                       force_use_latent_feedback_adapter=True)
    model = model.to(device).eval()
    model._film_bypass = True
    if getattr(model, "use_memory", False):
        model.use_memory = False
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    emb = model.embed.weight.float()
    emb_norm = emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)
    print(f"# ckpt={args.ckpt} R={args.R} n={args.n}", flush=True)

    cases = {}

    # 1) POINTER-CHASE (depth-bound iterated map): true_next = the answer.
    ptr = [json.loads(l) for l in open(args.ptr_heldout) if l.strip()][:args.n]
    rows = []
    for r in ptr:
        pfx = r["prompt"] + "\ndef solve():\n    return "
        ids = tok.encode(pfx, add_special_tokens=False)
        tn = tok.encode(str(r["answer"]), add_special_tokens=False)[0]
        rows.append(trace(model, ids, args.R, thinking_id, tn, device, emb_norm))
    cases["pointer_chase"] = (rows, ptr)

    # 2) CODE: true_next = the next code token mid-solution.
    pairs = load_distilled_jsonl(args.code_jsonl, prefer_full_completion=False,
                                 require_extracted_code=True)[:400]
    rows = []
    for _prob, sol in pairs:
        ids = tok.encode(sol, add_special_tokens=False)
        if len(ids) < args.code_prefix_len + 2:
            continue
        pfx, tn = ids[:args.code_prefix_len], ids[args.code_prefix_len]
        rows.append(trace(model, pfx, args.R, thinking_id, tn, device, emb_norm))
        if len(rows) >= args.n:
            break
    cases["code"] = (rows, None)

    # 3) GENERAL TEXT: true_next = the next word token.
    rows = []
    for s in GENERAL_TEXT:
        ids = tok.encode(s, add_special_tokens=False)
        cut = max(4, len(ids) - 4)
        pfx, tn = ids[:cut], ids[cut]
        rows.append(trace(model, pfx, args.R, thinking_id, tn, device, emb_norm))
    cases["general_text"] = (rows, None)

    print("\n=== per-step means (dlp = logp(true_next) gain vs no-think) ===")
    for name, (rows, _) in cases.items():
        print(f"\n[{name}]  n={len(rows)}")
        print(f"  {'k':>2} {'dlp':>8} {'max_cos_emb':>12} {'||z_k-z_{k-1}||':>15} {'||z_k||':>9}")
        for (k, dlp, mc, dz, zn) in _agg(rows, args.R):
            print(f"  {k:>2} {dlp:>+8.3f} {mc:>12.3f} {dz:>15.3f} {zn:>9.2f}")

    # Qualitative: decode trajectory of the thread on 2 pointer-chase examples
    print("\n=== thread decode trajectory (pointer-chase: does it trace the orbit?) ===")
    for i, r in enumerate(ptr[:3]):
        toks = [tok.decode([row["dec"]]).strip() for row in cases["pointer_chase"][0][i]]
        inter = r.get("intermediates", [])
        print(f"  ex{i}: start->...->{r['answer']}  true orbit {inter}")
        print(f"        thread decodes per step: {toks}")
    print("\n=== thread decode trajectory (code) ===")
    for i in range(min(3, len(cases['code'][0]))):
        toks = [tok.decode([row["dec"]]).strip() for row in cases["code"][0][i]]
        print(f"  code ex{i}: thread decodes per step: {toks}")


if __name__ == "__main__":
    main()
