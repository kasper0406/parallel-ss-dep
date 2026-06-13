"""Placement diagnostic: does the gate fire WHERE latent thinking helps?

The marginal aggregate win on code is consistent with two very different
worlds, and the fix is different in each:

  (A) PLACEMENT-bound: the trunk thinks usefully (Δlogp>0 at some positions)
      but the gate's think-decision is UNCORRELATED with Δlogp — it fires in
      the wrong places. Fix = gate calibration (dense per-position Δlogp>0
      teacher). Cheap, high-leverage.

  (B) CONTENT-bound: Δlogp ≈ 0 essentially everywhere, so there is nothing to
      place toward. Fix = better thought content (richer adapter, deeper R,
      execution-grounded targets). Expensive.

This probe measures both, per real code position, on the co-trained base:
  - P(think) = 1 - σ(gate_logit)   (the gate's actual decision tendency)
  - Δlogp(R) = logp_think(true_next) - logp_nothink(true_next)

Then reports:
  - correlation(P_think, Δlogp)            — is the gate aimed at all?
  - mean Δlogp where the gate WANTS to think vs where it wants to emit
  - gate RECALL: of positions where Δlogp>0, what frac does the gate fire?
  - the headroom: mean Δlogp over the top-k positions an ORACLE gate would
    pick vs the mean over the positions THIS gate picks.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_gate_placement.py checkpoints/latent_code_adapteronly.pt
"""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from experiments.thinking import latent_think_logp, load_latent_model


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latent_code_adapteronly.pt"
    jsonl = sys.argv[2] if len(sys.argv) > 2 else "data/sft_phase_c_combined.jsonl"
    n_problems = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    pos_per_problem = 8
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg, tid, tok, eos = load_latent_model(ckpt, device, train=False)
    comment = "# Complete the following Python function.\n"

    # Collect (prefix_ids, true_next) sampled across the CODE span of passing
    # solutions, plus the gate σ and Δlogp at each.
    rows = []
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            if float(d.get("score", 0)) < 0.99 or not d.get("extracted_code"):
                continue
            pre = tok.encode(comment + d["problem_prompt"], add_special_tokens=False)
            code = tok.encode(d["extracted_code"], add_special_tokens=False)
            ids = pre + code
            if len(code) < 12 or len(ids) > 320:
                continue
            rows.append((ids, len(pre)))
            if len(rows) >= n_problems:
                break
    print(f"[placement] ckpt={ckpt} problems={len(rows)} R={R} "
          f"pos/problem={pos_per_problem}", flush=True)

    p_think_all, dlogp_all = [], []
    for ids, plen in rows:
        L = len(ids)
        # sample code positions (need a t+1 target): from plen..L-2
        lo, hi = plen, L - 2
        if hi <= lo:
            continue
        step = max(1, (hi - lo) // pos_per_problem)
        for t in range(lo, hi, step):
            prefix = torch.tensor([ids[:t + 1]], device=device)
            true_next = torch.tensor([ids[t + 1]], device=device)
            with torch.no_grad():
                out = model(prefix)
                logits = (out[0] if isinstance(out, tuple) else out)[0, -1].float()
                nothink_lp = F.log_softmax(logits, -1)[ids[t + 1]].item()
                gl = model._last_gate_logits[0, -1].item()
                p_think = 1.0 - torch.sigmoid(torch.tensor(gl)).item()
                think_lp = latent_think_logp(
                    model, prefix, true_next, R=R, thinking_token_id=tid,
                    pad_id=0, wm_off=True)[0].item()
            p_think_all.append(p_think)
            dlogp_all.append(think_lp - nothink_lp)

    pt = torch.tensor(p_think_all)
    dl = torch.tensor(dlogp_all)
    n = dl.numel()
    helps = (dl > 0)
    # correlation
    corr = torch.corrcoef(torch.stack([pt, dl]))[0, 1].item()
    # gate's working point: it "fires think" where p_think > 0.5 (decision rule),
    # but on code the gate runs negative, so also report the top-quartile-by-
    # p_think slice (where the gate is RELATIVELY most inclined to think).
    fire = pt > 0.5
    q = torch.quantile(pt, 0.75)
    top_pt = pt >= q
    def mean(mask):
        return dl[mask].mean().item() if mask.any() else float("nan")
    print(f"\n=== Δlogp landscape (n={n} positions) ===")
    print(f"  mean Δlogp (all)            = {dl.mean().item():+.3f}")
    print(f"  median Δlogp (all)          = {dl.median().item():+.3f}")
    print(f"  frac positions Δlogp>0      = {helps.float().mean().item():.3f}")
    print(f"  mean Δlogp | Δlogp>0        = {mean(helps):+.3f}  (the upside if placed right)")
    print(f"\n=== Is the gate AIMED at the upside? ===")
    print(f"  corr(P_think, Δlogp)        = {corr:+.3f}   (>0 = gate aimed correctly)")
    print(f"  mean Δlogp | gate fires(>.5)= {mean(fire):+.3f}  (n={int(fire.sum())})")
    print(f"  mean Δlogp | top-quartile pt= {mean(top_pt):+.3f}  (n={int(top_pt.sum())})")
    print(f"  mean Δlogp | bottom 75% pt  = {mean(~top_pt):+.3f}")
    # gate recall: of the helps, what frac is in the gate's top quartile?
    if helps.any():
        recall = (helps & top_pt).float().sum().item() / helps.float().sum().item()
        print(f"  gate recall of Δlogp>0 (topQ)= {recall:.3f}  "
              f"(random baseline 0.25)")
    # oracle headroom: mean Δlogp the SAME budget would capture if placed by an
    # oracle (top-k by true Δlogp) vs by this gate (top-k by p_think).
    k = max(1, int(0.25 * n))
    oracle = dl.topk(k).values.mean().item()
    gate_idx = pt.topk(k).indices
    gate_cap = dl[gate_idx].mean().item()
    print(f"\n=== Headroom at a 25%-think budget (k={k}) ===")
    print(f"  oracle placement mean Δlogp = {oracle:+.3f}")
    print(f"  this-gate placement mean    = {gate_cap:+.3f}")
    print(f"  CAPTURED FRACTION           = "
          f"{(gate_cap / oracle) if oracle > 0 else float('nan'):.2f}  "
          f"(1.0 = gate already optimal; ~0 = placement is the bottleneck)")


if __name__ == "__main__":
    main()
