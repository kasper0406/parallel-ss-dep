"""Channel-decomposition probe: WHERE does (if anywhere) a latent think step
have retrieval headroom on code — trunk-only, +PKM, or +WM?

Motivated by the adversarial review of probe_gate_placement.py: that probe ran
with wm_off=True, so its "10% helpful / +0.3 / flat-R" ceiling is the
trunk+PKM-only, WM-DISABLED ceiling. It cannot tell us whether retrieval has
headroom, because it disabled the retrieval channel. This probe holds the
no-think baseline FIXED (the deployed config) and measures the latent-think
Δlogp ceiling under each channel configuration, on the EXISTING base, NO
training:

  T      : trunk only           (use_pkm=False, WM off)    — pure iteration
  T+P    : trunk + PKM           (use_pkm=True,  WM off)    — what the old probe measured
  T+P+WM : trunk + PKM + native WM (use_pkm=True, use_memory=True, wm_off=False)
           — the read_alpha≈0.66 channel the old probe turned off

Decomposition: PKM's marginal value during thinking = ceiling(T+P) − ceiling(T);
native WM's marginal value = ceiling(T+P+WM) − ceiling(T+P).

Falsification rule (from the review): if NO config clears the T+P 10%/+0.3 wall,
there is no retrievable headroom and a WM/PKM-retrieval trainer cannot
manufacture it. If T+P+WM clears it, native WM has headroom (RALT-via-WM worth
training). If T+P >> T, PKM is the channel (lever = gate/re-query PKM, maybe no
training).

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_retrieval_channels.py checkpoints/latent_code_adapteronly.pt
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
    has_pkm = getattr(model, "use_pkm", False) and hasattr(model, "pkm_layer")
    has_wm = hasattr(model, "memory")
    print(f"[channels] ckpt={ckpt} R={R} has_pkm={has_pkm} has_wm={has_wm} "
          f"read_alpha={float(getattr(model.memory, 'read_alpha', float('nan'))) if has_wm else None}",
          flush=True)
    comment = "# Complete the following Python function.\n"

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
    print(f"[channels] problems={len(rows)} pos/problem={pos_per_problem}", flush=True)

    # Channel configs: (label, use_pkm, use_memory, wm_off)
    configs = [("T", False, False, True),
               ("T+P", True, False, True),
               ("T+P+WM", True, True, False)]
    dlogp = {c[0]: [] for c in configs}

    base_use_pkm = getattr(model, "use_pkm", False)
    base_use_mem = getattr(model, "use_memory", False)

    def latent_logp(prefix, true_next, use_pkm, use_mem, wm_off):
        model.use_pkm = use_pkm if has_pkm else False
        model.use_memory = use_mem if has_wm else False
        try:
            return latent_think_logp(model, prefix, true_next, R=R,
                                     thinking_token_id=tid, pad_id=0,
                                     wm_off=wm_off)[0].item()
        finally:
            model.use_pkm = base_use_pkm
            model.use_memory = base_use_mem

    import time as _time
    for pi, (ids, plen) in enumerate(rows):
        _t = _time.time()
        L = len(ids)
        lo, hi = plen, L - 2
        if hi <= lo:
            continue
        step = max(1, (hi - lo) // pos_per_problem)
        for t in range(lo, hi, step):
            prefix = torch.tensor([ids[:t + 1]], device=device)
            true_next = torch.tensor([ids[t + 1]], device=device)
            # FIXED no-think baseline = the deployed config (PKM on, no think
            # token so WM never fires anyway).
            with torch.no_grad():
                model.use_pkm = base_use_pkm
                model.use_memory = base_use_mem
                out = model(prefix)
                logits = (out[0] if isinstance(out, tuple) else out)[0, -1].float()
                nothink_lp = F.log_softmax(logits, -1)[ids[t + 1]].item()
                for label, up, um, wo in configs:
                    tl = latent_logp(prefix, true_next, up, um, wo)
                    dlogp[label].append(tl - nothink_lp)
        if pi % 5 == 0:
            print(f"  [{pi}/{len(rows)}] L={L} {_time.time()-_t:.2f}s", flush=True)

    print(f"\n=== Latent-think Δlogp ceiling by channel "
          f"(n={len(dlogp['T'])} positions, baseline=deployed no-think) ===")
    print(f"{'config':<10}{'frac>0':>9}{'mean|helps':>12}{'mean(all)':>11}"
          f"{'median':>9}")
    ceil = {}
    for label, *_ in configs:
        dl = torch.tensor(dlogp[label])
        helps = dl > 0
        mh = dl[helps].mean().item() if helps.any() else float("nan")
        ceil[label] = (helps.float().mean().item(), mh)
        print(f"{label:<10}{helps.float().mean().item():>9.3f}{mh:>12.3f}"
              f"{dl.mean().item():>11.3f}{dl.median().item():>9.3f}")
    print(f"\n=== Marginal channel value (Δ in frac-helpful, Δ in mean|helps) ===")
    print(f"  PKM during thinking  (T+P)-(T)     = "
          f"{ceil['T+P'][0]-ceil['T'][0]:+.3f} frac, "
          f"{ceil['T+P'][1]-ceil['T'][1]:+.3f} mag")
    print(f"  native WM (T+P+WM)-(T+P)            = "
          f"{ceil['T+P+WM'][0]-ceil['T+P'][0]:+.3f} frac, "
          f"{ceil['T+P+WM'][1]-ceil['T+P'][1]:+.3f} mag")
    print(f"\nVERDICT: if T+P+WM clears the T+P wall → WM has headroom (train "
          f"RALT-WM). If T+P>>T → PKM is the channel. If all flat → no "
          f"retrieval headroom, rethink.")


if __name__ == "__main__":
    main()
