"""Time-breakdown probe for v6-xattn: where's the per-step cost going?

We compare these configurations on the same data + shape:

  baseline (no xattn)         single-pass forward (= what v6-shallow K=1 does)
  v6-xattn (film_sigmoid)     full 2-pass + film_sigmoid xattn (= production v6-xattn)
  v6-xattn (attn form)        full 2-pass + scalar-α attn xattn (cheaper module)

Difference (baseline → v6-xattn) tells us the *combined* cost of "2nd pass +
xattn module compute". Difference (attn-form → film_sigmoid) tells us how much
of that is the film_sigmoid module specifically (vs. cheap scalar α).

Relative costs scale linearly with B*T, so a B=2 T=512 probe (~3 GB) is
representative of the 14 × 2048 production-shape per-step breakdown.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.model import TinyLM
from experiments.layers import DeltaNetAttention
from experiments.speed_knobs import apply_speed_knobs


def make_model(use_xattn: bool, xattn_form: str = "film_sigmoid",
                d_model: int = 896, n_layers: int = 10,
                n_heads: int = 14, d_head: int = 64,
                xattn_heads: int = 8,
                xattn_pairs_spec: tuple = (
                    (0, (5, 6, 7, 8, 9)),
                    (1, (5, 6, 7, 8, 9)),
                    (2, (5, 6, 7, 8, 9)),
                )) -> TinyLM:
    """Configurable so the same script can run at production shape OR at a
    fits-in-4GB tiny shape (for co-residency profiling). The cost RATIOS
    are what we care about and they're shape-invariant in the limit."""
    xattn_pairs = xattn_pairs_spec if use_xattn else ()
    m = TinyLM(
        vocab_size=49216, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head,
        attention_cls=DeltaNetAttention,
        feedback_mode="none",
        feedback_xattn_pairs=xattn_pairs,
        feedback_xattn_form=xattn_form,
        feedback_xattn_heads=xattn_heads,
        activation_checkpointing=True,
        output_gate=True,
        use_memory=True, mem_size=1024,
        thinking_token_id=49152,
    ).cuda()
    apply_speed_knobs(m, bf16=True, tf32=True, compile_model=False)
    return m


def bench(model, x, y, n_iter=10, label="", do_backward=True):
    """Forward + (optional) backward; return ms/step averaged over n_iter."""
    V = model.embed.num_embeddings
    # warmup
    for _ in range(3):
        out = model(x)
        if do_backward:
            loss = F.cross_entropy(out.reshape(-1, V), y.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = model(x)
        if do_backward:
            loss = F.cross_entropy(out.reshape(-1, V), y.reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / n_iter
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"  {label:<30} {ms:>8.1f} ms/step   peak={peak_gb:.1f} GB")
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--tiny", action="store_true",
                   help="4L × 256d, 3 targets × 3 sources — fits in ~2 GB "
                        "for co-residency profiling. Production-shape "
                        "cost ratios still hold (linear in d×L).")
    args = p.parse_args()
    if args.tiny:
        cfg = dict(d_model=256, n_layers=4, n_heads=4, d_head=64,
                   xattn_heads=4,
                   xattn_pairs_spec=((0, (2, 3)), (1, (2, 3))))
    else:
        cfg = {}

    B, T = args.batch, args.T
    torch.manual_seed(0)
    x = torch.randint(0, 49000, (B, T), device="cuda")
    y = torch.randint(0, 49000, (B, T), device="cuda")

    print(f"\nProfile: B={B} T={T}, n_iter={args.n_iter}, bf16+ckpt, no compile\n")

    print("=== fwd+bwd ms/step ===")
    m_base = make_model(use_xattn=False, **cfg)
    t_base = bench(m_base, x, y, args.n_iter, "baseline (1-pass, no xattn)")
    del m_base
    torch.cuda.empty_cache()

    m_attn = make_model(use_xattn=True, xattn_form="attn", **cfg)
    t_attn = bench(m_attn, x, y, args.n_iter, "2-pass + xattn[attn] (scalar α)")
    del m_attn
    torch.cuda.empty_cache()

    m_fsig = make_model(use_xattn=True, xattn_form="film_sigmoid", **cfg)
    t_fsig = bench(m_fsig, x, y, args.n_iter, "2-pass + xattn[film_sigmoid] (production)")

    # Attribution: subtract isolates the cost of each phase.
    cost_2pass_only = t_attn - t_base    # 2nd pass + tiny scalar-α gate cost
    cost_filmsig_extra = t_fsig - t_attn # extra cost of film_sigmoid vs attn
    print(f"\n=== Attribution ===")
    print(f"  baseline 1-pass cost:                  {t_base:>8.1f} ms")
    print(f"  cost of 2nd pass (vs 1-pass baseline): {cost_2pass_only:>8.1f} ms  "
          f"(+{cost_2pass_only/t_base*100:.0f}%)")
    print(f"  cost of film_sigmoid vs scalar-α:      {cost_filmsig_extra:>8.1f} ms  "
          f"(+{cost_filmsig_extra/t_base*100:.0f}%)")
    print(f"  total xattn overhead (vs baseline):    {t_fsig - t_base:>8.1f} ms  "
          f"(+{(t_fsig - t_base)/t_base*100:.0f}%)")

    # Forward-only comparison (no backward / no opt). Tells us whether the
    # cost is in the forward kernels themselves vs the backward.
    print("\n=== fwd-only ms/step (no backward) ===")
    t_base_f = bench(make_model(use_xattn=False), x, y, args.n_iter,
                     "baseline fwd only", do_backward=False)
    t_fsig_f = bench(m_fsig, x, y, args.n_iter,
                     "film_sigmoid fwd only", do_backward=False)
    print(f"  fwd overhead ratio fsig/base:   {t_fsig_f/t_base_f:.2f}×")
    print(f"  fwd+bwd overhead ratio fsig/base: {t_fsig/t_base:.2f}×")


if __name__ == "__main__":
    sys.exit(main())
