"""Equal-weight checkpoint soup (OLMo-2-style decay soup / model soups).

Averages the state_dicts of N same-architecture checkpoints (fp32
accumulation, cast back to each tensor's original dtype). Non-floating
entries (step counters, int buffers) are taken from the FIRST checkpoint,
as are `step` and `config`. Refuses to soup checkpoints whose key sets or
tensor shapes differ.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/soup_ckpts.py \
    --ckpts a.pt b.pt c.pt --out soup.pt
"""

import argparse

import torch


def soup_state_dicts(sds: list[dict]) -> dict:
    ref = sds[0]
    keys = set(ref.keys())
    for i, sd in enumerate(sds[1:], 1):
        if set(sd.keys()) != keys:
            missing = keys ^ set(sd.keys())
            raise ValueError(f"ckpt {i} key mismatch vs ckpt 0: {sorted(missing)[:5]}")
    out = {}
    for k, v in ref.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            out[k] = v
            continue
        acc = v.detach().to(torch.float32).clone()
        for sd in sds[1:]:
            w = sd[k]
            if w.shape != v.shape:
                raise ValueError(f"shape mismatch at {k}: {v.shape} vs {w.shape}")
            acc += w.detach().to(torch.float32)
        out[k] = (acc / len(sds)).to(v.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.ckpts) < 2:
        raise SystemExit("need >= 2 checkpoints to soup")

    loaded = [torch.load(p, map_location="cpu", weights_only=False)
              for p in args.ckpts]
    souped = soup_state_dicts([c["state_dict"] for c in loaded])
    torch.save({"state_dict": souped,
                "step": loaded[0].get("step"),
                "config": loaded[0].get("config"),
                "soup_of": list(args.ckpts)}, args.out)
    n_params = sum(v.numel() for v in souped.values() if torch.is_tensor(v))
    print(f"[soup] {len(args.ckpts)} ckpts -> {args.out}  ({n_params/1e6:.1f}M elements)")


if __name__ == "__main__":
    main()
