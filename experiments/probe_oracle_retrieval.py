"""Oracle-retrieval upper bound: is WM addressing the bottleneck on code, or is
the buffer content simply useless for code-token prediction?

The cheap no-train tests (legacy + cosine WM) both fail to help code thinking.
But that could be "addressing untrained" rather than "nothing to retrieve". This
probe removes addressing from the equation: at a think position it tries
injecting EACH past position's WM value (one-hot oracle slot selection) and keeps
the BEST — the upper bound any addressing scheme (trained DKV included) could
reach from the existing buffer content.

Decision rule:
  - oracle clears the T+P wall (frac-helpful ↑↑, mean(all) → ≥0): addressing IS
    the bottleneck → a trained DKV continuation is worth the GPU-hours.
  - oracle ≈ no-retrieval baseline: the buffer content is useless for code
    thinking → NO addressing scheme will help → skip DKV-WM-for-code.

Mechanism mirrors the real latent step (state-readonly think slot, adapter),
adding alpha·W_proj(W_v(h[j])) as the oracle retrieval at the think slot.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_oracle_retrieval.py checkpoints/latent_code_adapteronly.pt
"""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from experiments.thinking import load_latent_model, _logits_hidden


@torch.no_grad()
def latent_logp_with_injection(model, prefix_ids, true_next, *, R, tid,
                               inject=None, alpha=1.0):
    """R-step state-readonly latent loop; optionally add alpha*inject (d_model)
    at the think slot input. Returns logp(true_next) at the final think slot."""
    base_emb = model.embed(prefix_ids)                      # (1, P, d)
    think_col = torch.full((1, 1), int(tid), dtype=prefix_ids.dtype,
                           device=prefix_ids.device)
    ids = torch.cat([prefix_ids, think_col], dim=1)
    _l0, h0 = _logits_hidden(model(prefix_ids, return_hidden=True))
    z = h0[:, -1:, :]
    last = None
    for _ in range(max(1, int(R))):
        zi = model.apply_latent_feedback_adapter(z)
        if inject is not None:
            zi = zi + alpha * inject.view(1, 1, -1).to(zi.dtype)
        ie = torch.cat([base_emb, zi.to(base_emb.dtype)], dim=1)
        logits, h = _logits_hidden(model(ids, inputs_embeds=ie, return_hidden=True))
        z = h[:, -1:, :]
        last = logits[:, -1, :]
    return F.log_softmax(last.float(), -1)[0, int(true_next)].item()


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latent_code_adapteronly.pt"
    jsonl = sys.argv[2] if len(sys.argv) > 2 else "data/sft_phase_c_combined.jsonl"
    n_problems = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    R = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n_cand = 16          # sampled past positions to try as oracle retrieval
    pos_per_problem = 5
    alpha = 1.0
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg, tid, tok, eos = load_latent_model(ckpt, device, train=False)
    model.use_memory = False           # we inject the oracle value manually
    mem = model.memory
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
            if len(code) < 12 or len(ids) > 256:
                continue
            rows.append((ids, len(pre)))
            if len(rows) >= n_problems:
                break
    print(f"[oracle] ckpt={ckpt} problems={len(rows)} R={R} n_cand={n_cand} "
          f"alpha={alpha}", flush=True)

    d_base, d_oracle, d_rand = [], [], []
    import time as _t
    for pi, (ids, plen) in enumerate(rows):
        t0 = _t.time()
        L = len(ids)
        lo, hi = plen, L - 2
        if hi <= lo:
            continue
        full = torch.tensor([ids], device=device)
        # out-normed hiddens at every position (the vectors WM would store).
        with torch.no_grad():
            _l, H = _logits_hidden(model(full, return_hidden=True))   # (1,L,d)
        step = max(1, (hi - lo) // pos_per_problem)
        for t in range(lo, hi, step):
            prefix = torch.tensor([ids[:t + 1]], device=device)
            tn = ids[t + 1]
            # no-retrieval baseline (== T+P, PKM is on inside the trunk)
            model.use_pkm = True
            base = latent_logp_with_injection(model, prefix, tn, R=R, tid=tid,
                                              inject=None)
            # no-think baseline for Δ
            with torch.no_grad():
                lo_g = (lambda o: o[0] if isinstance(o, tuple) else o)(model(prefix))
                nothink = F.log_softmax(lo_g[0, -1].float(), -1)[tn].item()
            d_base.append(base - nothink)
            # oracle: try injecting each of n_cand past positions' WM value.
            cand_pos = list(range(0, t + 1))
            if len(cand_pos) > n_cand:
                idx = torch.linspace(0, t, n_cand).long().tolist()
                cand_pos = sorted(set(idx))
            best = base
            inj_norms = []
            for j in cand_pos:
                inj = mem.W_proj(mem.W_v(H[:, j, :]))[0]   # (d_model,)
                inj_norms.append(inj.norm().item())
                lp = latent_logp_with_injection(model, prefix, tn, R=R, tid=tid,
                                                inject=inj, alpha=alpha)
                if lp - nothink > best:
                    best = lp - nothink
            d_oracle.append(best)
            # CONTROL: matched-norm RANDOM injections, same max-of-K. If the
            # oracle lift is just max-of-K-noise, this matches it; if oracle >>
            # random, the buffer content is genuinely addressable signal.
            mean_norm = sum(inj_norms) / max(1, len(inj_norms))
            best_rand = base
            g = torch.Generator(device=device).manual_seed(1234 + t)
            for _ in range(len(cand_pos)):
                rv = torch.randn(H.shape[-1], generator=g, device=device)
                rv = rv / rv.norm() * mean_norm
                lp = latent_logp_with_injection(model, prefix, tn, R=R, tid=tid,
                                                inject=rv, alpha=alpha)
                if lp - nothink > best_rand:
                    best_rand = lp - nothink
            d_rand.append(best_rand)
        if pi % 5 == 0:
            print(f"  [{pi}/{len(rows)}] L={L} {_t.time()-t0:.2f}s", flush=True)

    b = torch.tensor(d_base)
    o = torch.tensor(d_oracle)
    r = torch.tensor(d_rand)
    print(f"\n=== Oracle-retrieval upper bound (n={b.numel()} positions) ===")
    print(f"  baseline T+P     : frac>0={ (b>0).float().mean():.3f}  "
          f"mean(all)={b.mean():+.3f}  median={b.median():+.3f}")
    print(f"  RANDOM-inj max-of-K: frac>0={ (r>0).float().mean():.3f}  "
          f"mean(all)={r.mean():+.3f}  median={r.median():+.3f}")
    print(f"  ORACLE retrieve  : frac>0={ (o>0).float().mean():.3f}  "
          f"mean(all)={o.mean():+.3f}  median={o.median():+.3f}")
    print(f"  oracle vs RANDOM : +{ (o>0).float().mean()-(r>0).float().mean():.3f} frac, "
          f"{o.mean()-r.mean():+.3f} mean   "
          f"(this is the REAL addressable-content signal; oracle-vs-baseline is "
          f"inflated by max-of-K)")
    print(f"\nDECISION: if oracle >> RANDOM → buffer content is addressable signal, "
          f"train DKV. If oracle ≈ RANDOM → the 'lift' is max-of-K noise, SKIP "
          f"DKV-WM-for-code.")


if __name__ == "__main__":
    main()
