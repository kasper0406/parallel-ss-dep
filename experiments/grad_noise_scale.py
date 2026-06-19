"""Gradient-noise-scale / critical-batch-size measurement for the 287M trunk.

McCandlish et al. "An Empirical Model of Large-Batch Training" — the SIMPLE
noise scale B_simple = tr(Σ)/|G|², estimated with the unbiased two-batch
estimator.  We measure on the REAL model + REAL data (the cached pretrain pool
from `noise_scale_data_cache.py`, drawn from configs/pretrain_mix_v4.yaml), for
the PLAIN LM cross-entropy gradient (think bursts off; aux losses excluded — see
the analysis doc for that caveat).

UNIT.  Production trains at batch 4 (micro_b) x grad_accum 32 (accum) x T 2048,
so the natural example unit is the MICROBATCH (micro_b sequences) and the
production step is B_big = accum microbatches.  With B_small=1 microbatch and
B_big=G=accum microbatches, and the per-batch identity
    E[|g_B|^2] = |G|^2 + tr(Σ_mb)/B          (B in microbatches),
the two-batch solve has the closed form
    |G|^2          = (G*sq_big - sq_small)/(G-1)
    tr(Σ_mb)       = G*(sq_small - sq_big)/(G-1)
    B_simple_mb    = G*(sq_small - sq_big)/(G*sq_big - sq_small)
where sq_small = E[|one-microbatch grad|^2] (mean over the G microbatches of a
draw) and sq_big = |mean-over-G-microbatches grad|^2.  Averaged over independent
draws.  Convert to tokens: B_simple_tokens = B_simple_mb * micro_b * T
(equivalently B_simple_seq * T).  This is invariant to seq-vs-microbatch choice.

Per-optimizer-group: the muon (2D hidden matrices) and adamw (embeds/lm_head/1D)
splits are measured separately (disjoint params → |g|^2 = sum of group |g|^2).

GPU 1 ONLY.  New standalone script; imports the existing builders.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.optim_utils import (
    _is_embedding_like, _is_think_adapter, _is_refinement_head,
    _is_latent_feedback_adapter,
)


def _param_group(name: str, p: torch.Tensor) -> str:
    """Replicate optim_utils.build_optimizer's muon/adamw split exactly."""
    if _is_think_adapter(name) or _is_refinement_head(name) \
            or _is_latent_feedback_adapter(name):
        return "adamw"
    if _is_embedding_like(name) or p.ndim != 2:
        return "adamw"
    return "muon"


def _solve(sq_small: float, sq_big: float, G: int):
    """Two-batch closed form. Returns (B_simple_mb, Gnorm2, trSigma_mb)."""
    Gnorm2 = (G * sq_big - sq_small) / (G - 1)
    trSig = G * (sq_small - sq_big) / (G - 1)
    denom = (G * sq_big - sq_small)
    Bmb = (G * (sq_small - sq_big) / denom) if denom != 0 else float("nan")
    return Bmb, Gnorm2, trSig


def measure_ckpt(ckpt_path, pool, *, n_draws_use, device="cuda",
                 autocast=True):
    inputs = pool["inputs"]
    targets = pool["targets"]
    doc_ids = pool["doc_ids"]
    T = pool["T"]
    accum = pool["accum"]        # = G microbatches (B_big)
    micro_b = pool["micro_b"]
    n_draws_total = pool["n_draws"]
    n_draws = min(n_draws_use, n_draws_total)
    G = accum

    model, cfg = build_model_from_ckpt(ckpt_path)
    model.eval()
    model._gist_loss_enabled = False

    # Fixed list of trainable params + their group label.
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [_param_group(n, p) for n, p in named]
    params = [p for _, p in named]
    n_muon = sum(g == "muon" for g in groups)
    n_adamw = sum(g == "adamw" for g in groups)
    nM_muon = sum(p.numel() for (g, (_, p)) in zip(groups, named) if g == "muon")
    nM_adamw = sum(p.numel() for (g, (_, p)) in zip(groups, named) if g == "adamw")
    print(f"  trainable params: muon {n_muon} tensors ({nM_muon/1e6:.1f}M), "
          f"adamw {n_adamw} tensors ({nM_adamw/1e6:.1f}M)")

    # Persistent accumulator buffers (reused per draw).
    g_sum = [torch.zeros_like(p) for p in params]

    per_draw = []  # list of dicts with sq_small/sq_big per group
    seq_idx = 0
    base = 0
    t0 = time.time()
    for d in range(n_draws):
        for buf in g_sum:
            buf.zero_()
        sq_small = {"global": 0.0, "muon": 0.0, "adamw": 0.0}
        for m in range(accum):
            sl = slice(base, base + micro_b)
            x = inputs[sl].to(device, non_blocking=True)
            y = targets[sl].to(device, non_blocking=True)
            dids = doc_ids[sl].to(device, non_blocking=True)
            base += micro_b
            model.zero_grad(set_to_none=True)
            if autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x, doc_ids=dids)
            else:
                out = model(x, doc_ids=dids)
            logits = out[0] if isinstance(out, tuple) else out
            V = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, V).float(),
                                   y.reshape(-1), ignore_index=-100)
            loss.backward()
            # per-group squared-norm of this single-microbatch gradient + accum.
            with torch.no_grad():
                for gi, (grp, p, acc) in enumerate(zip(groups, params, g_sum)):
                    if p.grad is None:
                        continue
                    s = float(p.grad.double().pow(2).sum())
                    sq_small[grp] += s
                    sq_small["global"] += s
                    acc.add_(p.grad)
        # average over the G microbatches -> E[|g at B_small=1 mb|^2]
        for k in sq_small:
            sq_small[k] /= accum
        # big-batch gradient = mean over G microbatches.
        sq_big = {"global": 0.0, "muon": 0.0, "adamw": 0.0}
        with torch.no_grad():
            for grp, acc in zip(groups, g_sum):
                s = float((acc / accum).double().pow(2).sum())
                sq_big[grp] += s
                sq_big["global"] += s
        per_draw.append({"sq_small": sq_small, "sq_big": sq_big})
        if (d + 1) % 5 == 0 or d == 0:
            dt = time.time() - t0
            print(f"    draw {d+1}/{n_draws}  ({dt:.0f}s)  "
                  f"sq_small/sq_big(global)={sq_small['global']:.4e}/"
                  f"{sq_big['global']:.4e}", flush=True)

    # Aggregate: mean sq_small / sq_big across draws, then solve.
    results = {}
    for grp in ("global", "muon", "adamw"):
        ss = sum(pd["sq_small"][grp] for pd in per_draw) / len(per_draw)
        sb = sum(pd["sq_big"][grp] for pd in per_draw) / len(per_draw)
        Bmb, Gn2, trS = _solve(ss, sb, G)
        # Bootstrap over draws for an uncertainty band on B_simple_tokens.
        import random
        rng = random.Random(0)
        boots = []
        nd = len(per_draw)
        for _ in range(2000):
            idx = [rng.randrange(nd) for _ in range(nd)]
            ssb = sum(per_draw[i]["sq_small"][grp] for i in idx) / nd
            sbb = sum(per_draw[i]["sq_big"][grp] for i in idx) / nd
            b, _, _ = _solve(ssb, sbb, G)
            boots.append(b * micro_b * T)
        boots.sort()
        lo = boots[int(0.16 * len(boots))]
        hi = boots[int(0.84 * len(boots))]
        results[grp] = {
            "sq_small": ss, "sq_big": sb,
            "Gnorm2": Gn2, "trSigma_mb": trS,
            "B_simple_mb": Bmb,
            "B_simple_seq": Bmb * micro_b,
            "B_simple_tokens": Bmb * micro_b * T,
            "B_simple_tokens_lo": lo, "B_simple_tokens_hi": hi,
            # noise fraction of the production-step gradient (B_big=128 seqs):
            # var-contribution / total = (trS/G)/sq_big.
            "noise_frac_at_prod": (trS / G) / sb if sb > 0 else float("nan"),
        }
    del model, g_sum
    torch.cuda.empty_cache()
    return results, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="runs/noise_scale/pool_v4.pt")
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="checkpoint paths (early..late)")
    p.add_argument("--tokens", nargs="+", type=float, default=None,
                   help="token count (B) per ckpt, for labeling")
    p.add_argument("--n_draws_use", type=int, default=25)
    p.add_argument("--out", default="runs/noise_scale/results.json")
    p.add_argument("--no_autocast", action="store_true")
    args = p.parse_args()

    pool = torch.load(args.pool, map_location="cpu", weights_only=False)
    print(f"Pool: {pool['inputs'].shape[0]} seqs, T={pool['T']}, "
          f"accum(G)={pool['accum']}, micro_b={pool['micro_b']}, "
          f"n_draws={pool['n_draws']}")
    B_big_tok = pool["accum"] * pool["micro_b"] * pool["T"]
    print(f"Production step B_big = {pool['accum']*pool['micro_b']} seqs "
          f"= {B_big_tok} tokens\n")

    all_res = []
    for i, ck in enumerate(args.ckpts):
        tok_b = args.tokens[i] if args.tokens else None
        print(f"=== ckpt {ck}"
              + (f"  (~{tok_b:.2f}B tokens)" if tok_b else "") + " ===")
        res, cfg = measure_ckpt(ck, pool, n_draws_use=args.n_draws_use)
        for grp in ("global", "muon", "adamw"):
            r = res[grp]
            print(f"  [{grp:6s}] B_simple = {r['B_simple_tokens']:,.0f} tok "
                  f"[{r['B_simple_tokens_lo']:,.0f}, {r['B_simple_tokens_hi']:,.0f}] "
                  f"(= {r['B_simple_seq']:.1f} seqs); "
                  f"noise_frac@prod(128seq)={r['noise_frac_at_prod']:.3f}")
        all_res.append({"ckpt": ck, "tokens_B": tok_b, "results": res})
        with open(args.out, "w") as f:
            json.dump({"pool_B_big_tokens": B_big_tok, "ckpts": all_res}, f,
                      indent=2)
        print(f"  (saved -> {args.out})\n", flush=True)

    print("DONE. Summary (global B_simple in tokens):")
    for r in all_res:
        g = r["results"]["global"]
        print(f"  {r['tokens_B']}B: {g['B_simple_tokens']:,.0f} tok "
              f"(noise_frac@prod={g['noise_frac_at_prod']:.2f})")


if __name__ == "__main__":
    main()
