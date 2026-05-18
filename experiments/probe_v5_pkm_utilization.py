"""Did v5-pkm actually USE its 38M-param product-key table?

Diagnostics on the trained ckpt + a forward pass on held-out text:
  - Value-table magnitude: how far did rows move from init? Per-head
    histogram of ||v_row|| — if most rows are still at init std, the
    table is largely dead weight.
  - Sub-key utilisation: per-head sub-key magnitude / spread.
  - Forward hit-rate: per-head, distribution of top-1 slot indices over
    a real codeparrot batch. Concentrated hits = a few hot slots
    dominate (= small effective capacity). Uniform = full table used.
  - Per-head residual-stream contribution: ||pkm(h)|| / ||h|| averaged
    across positions. If this is tiny everywhere, the PKM output is
    being squashed regardless of how rich the lookups are.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/probe_v5_pkm_utilization.py \\
        --ckpt checkpoints/pretrain_mix_v5_pkm.pt
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.memory_layer import PKMLayer


def init_row_std(d_model: int) -> float:
    """The std the value table was initialised at in PKMLayer.__init__."""
    return 1.0 / math.sqrt(d_model)


def value_table_magnitude_stats(pkm: PKMLayer, init_std: float) -> dict:
    """Per-head distribution of row magnitudes vs init."""
    out = {}
    for h, emb in enumerate(pkm.values):
        # rows: (n_keys^2, v_dim_per_head)
        rows = emb.weight.float()
        # expected ||v_row|| at init = sqrt(v_dim_per_head) * init_std
        expected_init_norm = math.sqrt(rows.shape[1]) * init_std
        row_norms = rows.norm(dim=1)
        out[f"head_{h}"] = {
            "n_rows": rows.shape[0],
            "v_dim": rows.shape[1],
            "expected_init_norm": expected_init_norm,
            "mean_row_norm": row_norms.mean().item(),
            "median_row_norm": row_norms.median().item(),
            "p95_row_norm": row_norms.quantile(0.95).item(),
            "frac_above_2x_init": (
                (row_norms > 2 * expected_init_norm).float().mean().item()
            ),
        }
    return out


def forward_slot_hits(model, pkm: PKMLayer, x: torch.Tensor) -> dict:
    """Run forward, capture slot indices the PKM actually picked."""
    captured = {}
    orig_forward = pkm.forward

    def patched(h):
        B, T, _ = h.shape
        H, K, kd, tk = pkm.n_heads, pkm.n_keys, pkm.k_dim, pkm.top_k
        h_n = pkm.norm(h)
        q = pkm.query_proj(h_n).float().view(B, T, H, 2, kd)
        sk1 = pkm.subkeys[:, 0].float()
        sk2 = pkm.subkeys[:, 1].float()
        s1 = torch.einsum("bthd,hkd->bthk", q[:, :, :, 0], sk1)
        s2 = torch.einsum("bthd,hkd->bthk", q[:, :, :, 1], sk2)
        s1 = pkm.bn_s1(s1.reshape(B * T * H, K)).reshape(B, T, H, K)
        s2 = pkm.bn_s2(s2.reshape(B * T * H, K)).reshape(B, T, H, K)
        s1k, i1 = s1.topk(tk, dim=-1)
        s2k, i2 = s2.topk(tk, dim=-1)
        full = s1k.unsqueeze(-1) + s2k.unsqueeze(-2)
        full = full.reshape(B, T, H, tk * tk)
        idx_flat = (i1.unsqueeze(-1) * K + i2.unsqueeze(-2)).reshape(B, T, H, tk * tk)
        best_scores, best_pos = full.topk(tk, dim=-1)
        slot_idx = idx_flat.gather(-1, best_pos)            # (B,T,H,tk)
        weights = F.softmax(best_scores, dim=-1)
        captured["slot_idx"] = slot_idx.detach()
        captured["weights"] = weights.detach()
        # Compute output the standard way so model fwd proceeds
        return orig_forward(h)

    pkm.forward = patched
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        pkm.forward = orig_forward
    return captured


def slot_hit_summary(captured: dict, n_heads: int, n_slots_per_head: int) -> dict:
    """Per-head: how concentrated were the picks?"""
    slot_idx = captured["slot_idx"]        # (B, T, H, top_k)
    weights = captured["weights"]          # (B, T, H, top_k)
    B, T, H, tk = slot_idx.shape
    out = {}
    for h in range(H):
        # Flatten over (B, T, top_k) — every retrieval event is a (slot, weight) pair.
        idx = slot_idx[:, :, h, :].reshape(-1)
        w = weights[:, :, h, :].reshape(-1)
        # Aggregate weight per slot.
        slot_mass = torch.zeros(n_slots_per_head, device=idx.device,
                                dtype=torch.float32)
        slot_mass.scatter_add_(0, idx, w.float())
        slot_mass = slot_mass / slot_mass.sum().clamp_min(1e-9)
        # Sort and report concentration.
        sorted_mass = slot_mass.sort(descending=True).values
        out[f"head_{h}"] = {
            "n_unique_hit": int((slot_mass > 0).sum().item()),
            "top1_share": float(sorted_mass[0].item()),
            "top10_share": float(sorted_mass[:10].sum().item()),
            "top100_share": float(sorted_mass[:100].sum().item()),
            "top1000_share": float(sorted_mass[:1000].sum().item()),
        }
    return out


def pkm_residual_contribution(model, pkm: PKMLayer, x: torch.Tensor) -> float:
    """||PKM output|| / ||residual stream|| at the PKM injection layer."""
    captured = {}
    orig_forward = pkm.forward

    def patched(h):
        out = orig_forward(h)
        captured["norm_in"] = h.detach().float().norm(dim=-1)
        captured["norm_out"] = out.detach().float().norm(dim=-1)
        return out

    pkm.forward = patched
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        pkm.forward = orig_forward
    ratio = (captured["norm_out"] / captured["norm_in"].clamp_min(1e-6))
    return {
        "mean_ratio": float(ratio.mean().item()),
        "median_ratio": float(ratio.median().item()),
        "p95_ratio": float(ratio.quantile(0.95).item()),
        "p5_ratio": float(ratio.quantile(0.05).item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/pretrain_mix_v5_pkm.pt")
    p.add_argument("--n_tokens", type=int, default=2048,
                   help="Total tokens fed through the probe (batch*T).")
    args = p.parse_args()

    print(f"Loading {args.ckpt} ...")
    model, _cfg = build_model_from_ckpt(args.ckpt)
    model = model.cuda()
    model.eval()
    if not hasattr(model, "pkm_layer"):
        raise SystemExit("ckpt has no pkm_layer — not a PKM model")

    pkm: PKMLayer = model.pkm_layer
    d = model.embed.embedding_dim
    n_slots = pkm.n_keys * pkm.n_keys
    print(f"\nPKM config: n_heads={pkm.n_heads}, n_keys={pkm.n_keys}, "
          f"slots/head={n_slots}, top_k={pkm.top_k}, v_dim/head={pkm.v_dim_per_head}")
    print(f"Total PKM value params: ~{pkm.n_heads * n_slots * pkm.v_dim_per_head / 1e6:.1f} M")

    # 1) Value-table magnitude stats — did rows move from init?
    print("\n=== Value-table magnitude vs init ===")
    init_std = init_row_std(d)
    vstats = value_table_magnitude_stats(pkm, init_std)
    for h, s in vstats.items():
        print(f"  {h:>8}: expected_init={s['expected_init_norm']:.4f}  "
              f"mean={s['mean_row_norm']:.4f}  "
              f"median={s['median_row_norm']:.4f}  "
              f"p95={s['p95_row_norm']:.4f}  "
              f"frac>2x_init={s['frac_above_2x_init']:.3f}")

    # 2) Slot-hit distribution on real text.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    sample_text = (
        "def fibonacci(n):\n    if n < 2: return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n\n"
        "class Stack:\n    def __init__(self):\n        self.items = []\n"
        "    def push(self, x): self.items.append(x)\n"
        "    def pop(self): return self.items.pop()\n\n"
        "# Python is a high-level programming language with dynamic typing.\n"
        "# Common data structures include lists, tuples, dicts, and sets.\n"
    ) * 12
    ids = tok.encode(sample_text)[: args.n_tokens]
    x = torch.tensor(ids, device="cuda").unsqueeze(0)
    print(f"\nForward on {x.shape[1]} held-out tokens ...")
    captured = forward_slot_hits(model, pkm, x)
    slot_stats = slot_hit_summary(captured, pkm.n_heads, n_slots)

    print(f"\n=== Slot-hit concentration per head (n_slots/head = {n_slots}) ===")
    for h, s in slot_stats.items():
        print(f"  {h:>8}: hit={s['n_unique_hit']:>6}/{n_slots} "
              f"(top1={s['top1_share']:.3f}, top10={s['top10_share']:.3f}, "
              f"top100={s['top100_share']:.3f}, top1000={s['top1000_share']:.3f})")

    # 3) PKM residual contribution magnitude.
    print("\n=== PKM residual contribution: ||pkm(h)|| / ||h|| ===")
    r = pkm_residual_contribution(model, pkm, x)
    print(f"  mean={r['mean_ratio']:.4f}  median={r['median_ratio']:.4f}  "
          f"p5={r['p5_ratio']:.4f}  p95={r['p95_ratio']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
