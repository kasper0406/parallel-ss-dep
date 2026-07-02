"""Net2Net-style MLP widening for the shallow-wide 10L DeltaNet.

Widens the GLU MLP of an existing (donor-seeded + healed) DeltaNet checkpoint
from d_ff=SRC -> d_ff=DST while PRESERVING the healed weights exactly.

GLU layout per layer L (experiments/model.py::GLU):
  - W_g.weight, W_u.weight : Linear(d_model, d_ff) -> shape [d_ff, d_model]
  - W_d.weight             : Linear(d_ff, d_model) -> shape [d_model, d_ff]
  forward: W_d( silu(W_g x) * W_u x )

Widen rule (the only loss-bearing decisions):
  - W_g, W_u rows [0:SRC]   = copied EXACTLY (preserve donor-seed + heal)
                rows [SRC:] = fresh init, std = 1/sqrt(d_model)  (real features)
  - W_d cols    [0:SRC]     = copied EXACTLY
                cols [SRC:]  = fresh init, std = 0.1 * source-W_d std (per layer).
                ZERO here would stall the bootstrap (new units contribute 0 AND
                get 0 grad on their input weights -> width never fills); LARGE
                here spikes init CE. Small-nonzero gives a trainable bump.
  - Everything else (embed, tied lm_head, attention, norms) copied UNCHANGED.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/widen_mlp_ckpt.py \
      --src checkpoints/linearize/linearized_10L_stage3.pt \
      --dst checkpoints/linearize/linearized_10L_w7680_init.pt \
      --new_d_ff 7680 --wd_newcol_std_frac 0.1 --seed 1234
"""
import argparse
import math
import torch


def widen_ckpt(src_path: str, dst_path: str, new_d_ff: int,
               wd_newcol_std_frac: float = 0.1, seed: int = 1234) -> dict:
    g = torch.Generator().manual_seed(seed)
    ckpt = torch.load(src_path, weights_only=False, map_location="cpu")
    sd = ckpt["state_dict"]
    cfg = dict(ckpt["config"])
    d_model = int(cfg["d_model"])
    n_layers = int(cfg["n_layers"])
    src_d_ff = int(cfg["d_ff"])
    assert new_d_ff > src_d_ff, f"new_d_ff {new_d_ff} must exceed src {src_d_ff}"

    wg_new_std = 1.0 / math.sqrt(d_model)  # normal fresh-init for real features
    new_sd = {}
    report = []
    for k, v in sd.items():
        new_sd[k] = v.clone()  # default: copy unchanged

    for L in range(n_layers):
        wg = sd[f"blocks.{L}.mlp.W_g.weight"]   # [src_d_ff, d_model]
        wu = sd[f"blocks.{L}.mlp.W_u.weight"]   # [src_d_ff, d_model]
        wd = sd[f"blocks.{L}.mlp.W_d.weight"]   # [d_model, src_d_ff]
        assert wg.shape == (src_d_ff, d_model)
        assert wu.shape == (src_d_ff, d_model)
        assert wd.shape == (d_model, src_d_ff)

        n_new = new_d_ff - src_d_ff
        wd_src_std = float(wd.std().item())
        wd_new_std = wd_newcol_std_frac * wd_src_std

        new_wg = torch.empty(new_d_ff, d_model, dtype=wg.dtype)
        new_wu = torch.empty(new_d_ff, d_model, dtype=wu.dtype)
        new_wd = torch.empty(d_model, new_d_ff, dtype=wd.dtype)

        # preserve the healed weights EXACTLY
        new_wg[:src_d_ff].copy_(wg)
        new_wu[:src_d_ff].copy_(wu)
        new_wd[:, :src_d_ff].copy_(wd)

        # fresh init for the new width
        new_wg[src_d_ff:].normal_(0.0, wg_new_std, generator=g)
        new_wu[src_d_ff:].normal_(0.0, wg_new_std, generator=g)
        new_wd[:, src_d_ff:].normal_(0.0, wd_new_std, generator=g)

        new_sd[f"blocks.{L}.mlp.W_g.weight"] = new_wg
        new_sd[f"blocks.{L}.mlp.W_u.weight"] = new_wu
        new_sd[f"blocks.{L}.mlp.W_d.weight"] = new_wd
        report.append((L, wd_src_std, wd_new_std, n_new))

    cfg["d_ff"] = new_d_ff
    cfg["widened_from_d_ff"] = src_d_ff
    cfg["widen_wd_newcol_std_frac"] = wd_newcol_std_frac
    out = {"state_dict": new_sd, "step": ckpt.get("step", 0), "config": cfg}
    torch.save(out, dst_path)

    print(f"[widen] {src_path} -> {dst_path}")
    print(f"[widen] d_ff {src_d_ff} -> {new_d_ff}  (wg/wu new-std={wg_new_std:.5f})")
    for L, s, n, nn in report:
        print(f"[widen]   L{L}: W_d src-std={s:.5f}  new-col-std={n:.5f}  (+{nn} cols)")
    n_params = sum(t.numel() for t in new_sd.values())
    # tied embeddings: lm_head.weight duplicates embed.weight in the state dict
    tied = bool(cfg.get("tie_embeddings", False))
    dedup = n_params - (sd["embed.weight"].numel() if tied else 0)
    print(f"[widen] state_dict params={n_params:,}  tied-dedup={dedup:,}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="checkpoints/linearize/linearized_10L_stage3.pt")
    ap.add_argument("--dst", default="checkpoints/linearize/linearized_10L_w7680_init.pt")
    ap.add_argument("--new_d_ff", type=int, default=7680)
    ap.add_argument("--wd_newcol_std_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    widen_ckpt(a.src, a.dst, a.new_d_ff, a.wd_newcol_std_frac, a.seed)
