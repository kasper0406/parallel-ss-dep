"""Is PKM a more parameter-efficient FACT STORE than dense weights?

This is the project's core thesis, and the reason it matters: the 287M model has
a load-bearing PKM and STILL lost facts it saw 22x (degrees_to_radians). Either
PKM doesn't realize the memorization-efficiency Lample claims, or the facts
aren't routed to it. This isolates the first question with a controlled,
PARAM-MATCHED memorization-capacity curve.

Setup (pure fact storage — no in-context recall):
  - N facts: a FIXED random key vector r_i (frozen, carries no value info) ->
    a random class value_i in [0, V).
  - The model must memorize r_i -> value_i from training alone. Capacity lives
    ONLY in the mapping f (dense MLP vs PKM); the keys and the d->V head are
    shared and frozen/small.
  - Sweep N; report retention accuracy. The architecture that retains more facts
    at MATCHED params is the more efficient store.

DENSE:  logits = head( MLP(r) )          capacity in the MLP hidden
PKM:    logits = head( PKM(r) )          capacity in the PKM value tables

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/probe_pkm_capacity.py
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from experiments.memory_layer import PKMLayer

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True


class DenseStore(nn.Module):
    def __init__(self, d, h, V):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, r):
        return self.head(self.norm(self.net(r)))


class PKMStore(nn.Module):
    def __init__(self, d, V, n_heads, n_keys, k_dim, v_dim, top_k):
        super().__init__()
        self.pkm = PKMLayer(d_model=d, n_heads=n_heads, n_keys=n_keys,
                            k_dim=k_dim, top_k=top_k, v_dim_per_head=v_dim,
                            value_bf16=False, use_output_gate=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, r):
        out = self.pkm(r.unsqueeze(1)).squeeze(1)        # (B, d)
        return self.head(self.norm(out))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_and_eval(model, keys, vals, steps=4000, bs=1024, lr=2e-3):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    N = keys.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (min(bs, N),), device=DEVICE)
        logits = model(keys[idx])
        loss = F.cross_entropy(logits, vals[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = 0
        for s in range(0, N, 4096):
            e = min(s + 4096, N)
            pred = model(keys[s:e]).argmax(-1)
            acc += (pred == vals[s:e]).sum().item()
    return acc / N


def main():
    d, V = 48, 256
    # Small, PARAM-MATCHED models so the capacity cliff is at reachable N.
    pkm_cfg = dict(n_heads=2, n_keys=48, k_dim=16, v_dim=12, top_k=24)
    pkm0 = PKMStore(d, V, **pkm_cfg)
    pp = n_params(pkm0)
    # set dense_h to match pkm params: 2*d*h + (head d*V) + norms ≈ pp
    head = d * V
    dense_h = max(8, int((pp - head) / (2 * d)))
    pd = n_params(DenseStore(d, dense_h, V))
    print(f"[pkm-capacity] d={d} V={V}  PKM params={pp:,} "
          f"(slots/head={48*48}, v_dim=12)  DENSE params={pd:,} (h={dense_h})  "
          f"ratio={pp/pd:.2f}", flush=True)

    # Part 1: capacity cliff (ample-exposure: each fact seen ~40x).
    print(f"\n=== Part 1: capacity cliff (ample exposure) ===")
    print(f"{'N facts':>8}{'DENSE acc':>11}{'PKM acc':>10}")
    for N in [2000, 5000, 10000, 20000, 40000, 80000]:
        g = torch.Generator(device=DEVICE).manual_seed(N)
        keys = F.normalize(torch.randn(N, d, generator=g, device=DEVICE), dim=-1)
        vals = torch.randint(0, V, (N,), generator=g, device=DEVICE)
        steps = max(4000, 40 * N // 1024)
        da = train_and_eval(DenseStore(d, dense_h, V), keys, vals, steps=steps)
        pa = train_and_eval(PKMStore(d, V, **pkm_cfg), keys, vals, steps=steps)
        print(f"{N:>8}{da:>11.3f}{pa:>10.3f}  (steps={steps})", flush=True)

    # Part 2: FEW-SHOT acquisition — fix N well under capacity, vary exposures.
    # This is the property that matters for the real model: degrees_to_radians
    # was seen ~22x. Does PKM lock a fact in with FEWER exposures than dense?
    print(f"\n=== Part 2: few-shot acquisition (N=4000, vary exposures/fact) ===")
    print(f"{'exp/fact':>9}{'DENSE acc':>11}{'PKM acc':>10}")
    N = 4000
    g = torch.Generator(device=DEVICE).manual_seed(7)
    keys = F.normalize(torch.randn(N, d, generator=g, device=DEVICE), dim=-1)
    vals = torch.randint(0, V, (N,), generator=g, device=DEVICE)
    for exp in [2, 5, 10, 22, 50, 150]:
        steps = max(50, exp * N // 1024)
        da = train_and_eval(DenseStore(d, dense_h, V), keys, vals, steps=steps)
        pa = train_and_eval(PKMStore(d, V, **pkm_cfg), keys, vals, steps=steps)
        print(f"{exp:>9}{da:>11.3f}{pa:>10.3f}  (steps={steps})", flush=True)

    print(f"\nINTERPRETATION: Part 1 cliff far above MBPP's ~few-thousand facts "
          f"=> raw capacity is NOT the bottleneck. Part 2 is the real question: "
          f"if PKM acquires facts in FEWER exposures than dense, then targeted "
          f"REPEATED exposure (or PKM-routed facts) is the lever for rare-fact "
          f"retention — much cheaper than scaling the model.")


if __name__ == "__main__":
    main()
