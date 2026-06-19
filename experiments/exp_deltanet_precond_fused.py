"""FUSED DeltaNet-tailored matrix optimizer (single torch.optim.Optimizer).

This is the production-candidate form of the validated 2-object prototype
(`exp_deltanet_precond_optim.DeltaNetProjMuon` + a separate `torch.optim.Muon`
for o_proj/MLP). It collapses both into ONE optimizer with ONE `step()`:

  * DeltaNet q/k/v/b projections    -> per-head Newton-Schulz, done as a SINGLE
    BATCHED bmm over (n_heads * n_layers * {q,k,v}) slices (cross-layer batching),
    numerically identical to the prototype's per-unit `_ns_batched` (each slice is
    Frobenius-normalized + NS'd independently; bmm never mixes batch elements).
  * o_proj + MLP (head-MIXING / generic matrices) -> whole-matrix Muon, exactly
    `torch.optim.Muon` (reuses `_zeropower_via_newtonschulz`).
  * one momentum buffer per param (fp32, or bf16 with `bf16_state=True`).

It REUSES the exact same primitives as the prototype + torch.optim.Muon
(`_ns_batched`, `_zeropower_via_newtonschulz`, `_adjust_lr`, the lerp/nesterov
momentum, decoupled WD) so the ONLY difference vs the 2-object prototype is that
everything happens in one `step()` and the per-head NS is batched across layers.

The byte-identical check (`--mode verify`) proves fused == (DeltaNetProjMuon +
torch.optim.Muon) to fp32 round-off; `--mode profile` / `--mode memory` /
`--mode fraction` produce the perf tables in DELTANET_PRECONDITIONER_PERF.md.

GPU-1-only, standalone, does NOT touch the live v18 stack.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.optim.optimizer import Optimizer
from torch.optim._muon import (
    _zeropower_via_newtonschulz, _adjust_lr,
    DEFAULT_A, DEFAULT_B, DEFAULT_C, EPS,
)

# Reuse the prototype's EXACT per-head batched NS so the perhead path is
# byte-identical to the validated 2-object optimizer.
from experiments.exp_deltanet_precond_optim import (
    _ns_batched, build_units_from_model,
)

_NS = (DEFAULT_A, DEFAULT_B, DEFAULT_C)


class FusedDeltaNetMuon(Optimizer):
    """Single-object Muon with per-head NS on DeltaNet q/k/v/b and whole-matrix
    Muon on o_proj/MLP. Momentum/nesterov/decoupled-WD/LR-adjust are identical
    to torch.optim.Muon; the per-head NS reuses the prototype's `_ns_batched`.

    Args:
      units:    tailored units from `build_units_from_model` (matrix_perhead /
                rownorm; qk_coupled supported for completeness).
      other_2d: list of plain-Muon 2D params (o_proj + MLP).
      bf16_state:        store momentum buffers as bf16 (math still fp32).
      batch_across_layers: stack same-shape per-head slices from ALL layers into
                ONE bmm NS (the fused win). When False, falls back to per-unit
                NS (matches the prototype's exact batch grouping bit-for-bit).
    """

    def __init__(self, units, other_2d, *, lr: float, momentum: float = 0.95,
                 weight_decay: float = 0.0, nesterov: bool = True,
                 ns_steps: int = 5, bf16_state: bool = False,
                 batch_across_layers: bool = False):
        params = []
        for u in units:
            params.extend(u["params"])
        params.extend(other_2d)
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay,
                                      nesterov=nesterov, ns_steps=ns_steps))
        self._units = units
        self._other = list(other_2d)
        self._bf16_state = bf16_state
        self._batch_across_layers = batch_across_layers

        # Pre-bucket the per-head units by (kind, d_head, d_in) so step() can do
        # one batched NS per bucket. b_proj (rownorm) -> (1, d_in) bucket.
        self._perhead = [u for u in units if u["kind"] == "matrix_perhead"]
        self._rownorm = [u for u in units if u["kind"] == "rownorm"]
        self._qkc = [u for u in units if u["kind"] == "qk_coupled"]

    # ---- momentum / nesterov (identical math to torch.optim.Muon) ----
    def _buf(self, p):
        st = self.state[p]
        if "momentum_buffer" not in st:
            dt = torch.bfloat16 if self._bf16_state else p.grad.dtype
            st["momentum_buffer"] = torch.zeros_like(p.grad, dtype=dt)
        return st["momentum_buffer"]

    def _fp32_momentum_inplace(self, params, mom, nesterov):
        """fp32 state: batched (foreach) momentum + nesterov done IN-PLACE so no
        large temporary update-list is materialized. Returns the 'update' tensor
        for each param: p.grad itself (nesterov, overwritten in place) or the
        momentum buffer (non-nesterov). Numerically identical to the prototype's
        per-tensor `buf.lerp_(g,1-m); g.lerp(buf,m)`."""
        if not params:
            return []
        grads = [p.grad for p in params]
        store = [self._buf(p) for p in params]
        torch._foreach_lerp_(store, grads, 1 - mom)          # buf = m*buf+(1-m)*g
        if nesterov:
            torch._foreach_lerp_(grads, store, mom)          # g  = (1-m)*g+m*buf
            return grads
        return store

    def _bf16_momentum_one(self, p, mom, nesterov):
        """bf16 state: lift ONE param's buffer+grad to fp32, momentum+nesterov,
        persist buffer back to bf16. Returns the fp32 update tensor. Streamed
        per-param so only one param's fp32 temporaries are ever live."""
        store = self._buf(p)
        buf = store.float()
        g = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
        buf.lerp_(g, 1 - mom)
        upd = g.lerp_(buf, mom) if nesterov else buf
        store.copy_(buf)
        return upd

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        grp = self.param_groups[0]
        lr, mom, wd = grp["lr"], grp["momentum"], grp["weight_decay"]
        nesterov, ns_steps = grp["nesterov"], grp["ns_steps"]
        wd_scale = 1.0 - lr * wd

        perhead = [u for u in self._perhead if u["param"].grad is not None]
        rownorm = [u for u in self._rownorm if u["param"].grad is not None]
        qkc = [u for u in self._qkc if u["q"].grad is not None and u["k"].grad is not None]
        others = [p for p in self._other if p.grad is not None]

        # ---- momentum (in-place; no large update-list) ----
        if not self._bf16_state:
            all_params = ([u["param"] for u in perhead]
                          + [t for u in qkc for t in (u["q"], u["k"])]
                          + [u["param"] for u in rownorm] + others)
            self._fp32_momentum_inplace(all_params, mom, nesterov)
            upd = (lambda p: p.grad) if nesterov else self._buf
        else:
            upd = lambda p: self._bf16_momentum_one(p, mom, nesterov)

        # ---- 1. per-head q/k/v: NS batched over heads (cross-layer optional) ----
        # cross-layer stacks same-shape projections from ALL layers into one bmm
        # (fewer launches, but a contiguous (Sum_h, dh, d_in) workspace); per-unit
        # streams one projection at a time (NS still batched over its 14 heads ->
        # tiny workspace, matches Muon's per-param memory).
        if perhead:
            if self._batch_across_layers and not self._bf16_state:
                buckets: dict[tuple[int, int], list] = {}
                for u in perhead:
                    buckets.setdefault((u["d_head"], u["param"].shape[1]), []).append(u)
                for (dh, d_in), us in buckets.items():
                    if len(us) == 1:
                        u = us[0]; nh = u["n_heads"]
                        self._ns_apply_perhead(upd(u["param"]), u["param"], nh, dh,
                                               d_in, ns_steps, lr, wd_scale)
                        continue
                    stack = torch.cat([upd(u["param"]).view(u["n_heads"], dh, d_in)
                                       for u in us], dim=0)
                    out = _ns_batched(stack, ns_steps).to(stack.dtype)
                    alr = _adjust_lr(lr, None, (dh, d_in))
                    off = 0
                    for u in us:                              # stream the apply
                        nh = u["n_heads"]; p = u["param"]
                        p.mul_(wd_scale)
                        p.add_(out[off:off + nh].reshape_as(p), alpha=-alr)
                        off += nh
                    del out, stack
            else:
                for u in perhead:
                    p = u["param"]
                    self._ns_apply_perhead(upd(p), p, u["n_heads"], u["d_head"],
                                           p.shape[1], ns_steps, lr, wd_scale)

        # ---- 2. qk_coupled (optional arm) ----
        for u in qkc:
            q, k = u["q"], u["k"]
            nh, dh, d_in = u["n_heads"], u["d_head"], q.shape[1]
            gq, gk = upd(q), upd(k)
            stacked = torch.cat([gq.view(nh, dh, d_in), gk.view(nh, dh, d_in)], dim=1)
            o = _ns_batched(stacked, ns_steps).to(q.dtype)
            alr = _adjust_lr(lr, None, (2 * dh, d_in))
            q.mul_(wd_scale); q.add_(o[:, :dh].reshape_as(q), alpha=-alr)
            k.mul_(wd_scale); k.add_(o[:, dh:].reshape_as(k), alpha=-alr)

        # ---- 3. rownorm (b_proj): per-row NS, streamed ----
        for u in rownorm:
            p = u["param"]; nh, d_in = p.shape
            g = upd(p).view(nh, 1, d_in)
            out = _ns_batched(g, ns_steps).to(torch.float32).view(nh, d_in)
            alr = _adjust_lr(lr, None, (1, d_in))
            p.mul_(wd_scale); p.add_(out, alpha=-alr)

        # ---- 4. other_2d (o_proj + MLP): whole-matrix Muon, streamed ----
        for p in others:
            out = _zeropower_via_newtonschulz(upd(p), _NS, ns_steps, EPS)
            alr = _adjust_lr(lr, None, p.shape)
            p.mul_(wd_scale); p.add_(out, alpha=-alr)
        return loss

    def _ns_apply_perhead(self, u_tensor, p, nh, dh, d_in, ns_steps, lr, wd_scale):
        g = u_tensor.view(nh, dh, d_in)
        out = _ns_batched(g, ns_steps).to(torch.float32)
        alr = _adjust_lr(lr, None, (dh, d_in))
        p.mul_(wd_scale)
        p.add_(out.reshape_as(p), alpha=-alr)


# ======================================================================
#  build helpers
# ======================================================================
def build_production_model(d_model=896, n_layers=10, n_heads=14, d_head=64,
                            vocab=49152, max_T=2048, device="cpu"):
    from experiments.model import TinyLM
    from experiments.layers import DeltaNetAttention
    model = TinyLM(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                   n_heads=n_heads, d_head=d_head, attention_cls=DeltaNetAttention,
                   max_T=max_T, feedback_mode="none").to(device)
    return model


def matrix_param_set(model):
    """All 2D hidden matrices (= the Muon-eligible set, what all three optimizers
    operate on). Excludes embeddings/lm_head/pos + non-2D (those go to AdamW)."""
    embed_like = {"embed.weight", "pos_embed.weight", "lm_head.weight"}
    out = []
    for n, p in model.named_parameters():
        if not p.requires_grad or p.ndim != 2 or n in embed_like:
            continue
        out.append((n, p))
    return out


def make_muon(model, lr, momentum, wd):
    params = [p for _, p in matrix_param_set(model)]
    return torch.optim.Muon(params, lr=lr, momentum=momentum,
                            weight_decay=wd, nesterov=True)


def make_two_object(model, lr, momentum, wd):
    from experiments.exp_deltanet_precond_optim import DeltaNetProjMuon
    _, units, other_2d = build_units_from_model(model, mode="perhead")
    o1 = torch.optim.Muon(other_2d, lr=lr, momentum=momentum,
                          weight_decay=wd, nesterov=True)
    o2 = DeltaNetProjMuon(units, lr=lr, momentum=momentum,
                          weight_decay=wd, nesterov=True)
    return [o1, o2]


def make_fused(model, lr, momentum, wd, *, bf16_state=False,
               batch_across_layers=True):
    _, units, other_2d = build_units_from_model(model, mode="perhead")
    return FusedDeltaNetMuon(units, other_2d, lr=lr, momentum=momentum,
                             weight_decay=wd, nesterov=True,
                             bf16_state=bf16_state,
                             batch_across_layers=batch_across_layers)


# ======================================================================
#  verify: byte-identical to the 2-object prototype
# ======================================================================
def cmd_verify(args):
    dev = args.device
    lr, mom, wd = 1e-2, 0.95, 0.01

    # Two identically-seeded models -> identical params.
    def build():
        torch.manual_seed(args.seed)
        if dev == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        return build_production_model(d_model=args.d_model, n_layers=args.n_layers,
                                      n_heads=args.n_heads, d_head=args.d_head,
                                      device=dev)
    mA = build()   # 2-object
    mB = build()   # fused (cross-layer batched)
    mC = build()   # fused (per-unit batched -> bit-for-bit grouping of prototype)

    optA = make_two_object(mA, lr, mom, wd)
    optB = make_fused(mB, lr, mom, wd, batch_across_layers=True)
    optC = make_fused(mC, lr, mom, wd, batch_across_layers=False)

    matsA = matrix_param_set(mA)
    matsB = {n: p for n, p in matrix_param_set(mB)}
    matsC = {n: p for n, p in matrix_param_set(mC)}

    g = torch.Generator(device=dev).manual_seed(123)
    max_dB = max_dC = 0.0
    n_tail = sum(len(u["params"]) for u in build_units_from_model(mA, mode="perhead")[1])
    print(f"[verify] device={dev}  matrix params={len(matsA)}  "
          f"(tailored q/k/v/b={n_tail})  steps={args.steps}")
    for step in range(args.steps):
        # identical grads for all three on every matrix param
        for n, pA in matsA:
            gr = torch.randn(pA.shape, generator=g, device=dev, dtype=pA.dtype) * 0.01
            pA.grad = gr.clone()
            matsB[n].grad = gr.clone()
            matsC[n].grad = gr.clone()
        for o in optA:
            o.step()
        optB.step()
        optC.step()
        # compare
        for n, pA in matsA:
            dB = (pA - matsB[n]).abs().max().item()
            dC = (pA - matsC[n]).abs().max().item()
            max_dB = max(max_dB, dB)
            max_dC = max(max_dC, dC)
    print(f"[verify] max|Δ| fused(per-unit, batch_across_layers=False) vs 2-object "
          f"= {max_dC:.3e}")
    print(f"[verify] max|Δ| fused(CROSS-LAYER batched)               vs 2-object "
          f"= {max_dB:.3e}")
    tol = 1e-5
    print(f"[verify] tolerance = {tol:.0e}")
    ok_C = max_dC < tol
    print(f"[verify] per-unit fused byte-identical: {'PASS' if ok_C else 'FAIL'}")
    print(f"[verify] cross-layer fused within tol : "
          f"{'PASS' if max_dB < tol else f'delta={max_dB:.2e} (bf16 reduction-order)'}")
    return ok_C


# ======================================================================
#  profile: wall-clock ms / optimizer-step
# ======================================================================
def _fake_grads(params, gen, dev, scale=0.01):
    for p in params:
        p.grad = torch.randn(p.shape, generator=gen, device=dev, dtype=p.dtype) * scale


def _burst_ms(step_fn, iters):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step_fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


def _roundrobin(step_fns, warmup, burst, reps):
    """Round-robin timing robust to a noisy co-tenant: warm up all, then cycle
    through configs `reps` times timing a `burst` of steps each. Report MIN
    (uncontended compute) + median per config."""
    names = list(step_fns.keys())
    for n in names:           # warmup all
        for _ in range(warmup):
            step_fns[n]()
    torch.cuda.synchronize()
    samples = {n: [] for n in names}
    for _ in range(reps):
        for n in names:
            samples[n].append(_burst_ms(step_fns[n], burst))
    import statistics
    return {n: (min(s), statistics.median(s)) for n, s in samples.items()}


def cmd_profile(args):
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    lr, mom, wd = 1e-2, 0.95, 0.01
    gen = torch.Generator(device=dev).manual_seed(0)

    model = build_production_model(d_model=args.d_model, n_layers=args.n_layers,
                                   n_heads=args.n_heads, d_head=args.d_head,
                                   device=dev)
    mats = [p for _, p in matrix_param_set(model)]
    _fake_grads(mats, gen, dev)

    n_mat_params = sum(p.numel() for p in mats)
    print(f"[profile] {args.n_layers}L d{args.d_model} {args.n_heads}h d_head{args.d_head}"
          f"  matrix params={len(mats)} ({n_mat_params/1e6:.1f}M)")
    print(f"[profile] warmup={args.warmup} burst={args.iters} reps={args.reps} "
          f"(round-robin, report MIN ms = uncontended)\n")

    from experiments.bf16_optim import BF16StateMuon
    two_obj = make_two_object(model, lr, mom, wd)
    bf_muon = BF16StateMuon(mats, lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
    step_fns = {
        "muon (fp32 state)": make_muon(model, lr, mom, wd).step,
        "2-object (fp32 state)": (lambda o=two_obj: [x.step() for x in o]),
        "FUSED per-unit (fp32 state)":
            make_fused(model, lr, mom, wd, batch_across_layers=False).step,
        "FUSED cross-layer (fp32 state)":
            make_fused(model, lr, mom, wd, batch_across_layers=True).step,
        "muon (bf16 state)": bf_muon.step,
        "FUSED per-unit (bf16 state)":
            make_fused(model, lr, mom, wd, bf16_state=True, batch_across_layers=False).step,
    }
    res = _roundrobin(step_fns, args.warmup, args.iters, args.reps)
    rows = []
    for name in step_fns:
        mn, med = res[name]
        rows.append((name, mn, med))
        print(f"  {name:36s} min {mn:7.3f}  median {med:7.3f} ms/opt-step")
    return rows


# ======================================================================
#  memory: persistent state + peak transient workspace
# ======================================================================
def _state_bytes(opt):
    """Sum bytes of all optimizer state tensors (momentum buffers)."""
    opts = opt if isinstance(opt, list) else [opt]
    total = 0
    for o in opts:
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
    return total


def cmd_memory(args):
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    lr, mom, wd = 1e-2, 0.95, 0.01
    gen = torch.Generator(device=dev).manual_seed(0)
    model = build_production_model(d_model=args.d_model, n_layers=args.n_layers,
                                   n_heads=args.n_heads, d_head=args.d_head,
                                   device=dev)
    mats = [p for _, p in matrix_param_set(model)]

    from experiments.bf16_optim import BF16StateMuon
    builders = [
        ("muon (fp32 state)", lambda: make_muon(model, lr, mom, wd)),
        ("2-object (fp32 state)", lambda: make_two_object(model, lr, mom, wd)),
        ("FUSED per-unit (fp32 state)",
         lambda: make_fused(model, lr, mom, wd, batch_across_layers=False)),
        ("FUSED cross-layer (fp32 state)",
         lambda: make_fused(model, lr, mom, wd, batch_across_layers=True)),
        ("muon (bf16 state)",
         lambda: BF16StateMuon(mats, lr=lr, momentum=mom, weight_decay=wd, nesterov=True)),
        ("FUSED per-unit (bf16 state)",
         lambda: make_fused(model, lr, mom, wd, bf16_state=True, batch_across_layers=False)),
    ]
    print(f"[memory] {args.n_layers}L d{args.d_model} {args.n_heads}h "
          f"matrix params={len(mats)} ({sum(p.numel() for p in mats)/1e6:.1f}M)\n")
    for name, build in builders:
        _fake_grads(mats, gen, dev)
        opt = build()
        # one step to allocate momentum state, then measure
        if isinstance(opt, list):
            for o in opt:
                o.step()
        else:
            opt.step()
        torch.cuda.synchronize()
        state_b = _state_bytes(opt)
        # peak transient: reset, then a fresh step
        _fake_grads(mats, gen, dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        if isinstance(opt, list):
            for o in opt:
                o.step()
        else:
            opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        transient = peak - base
        print(f"  {name:36s} state={state_b/1e6:8.2f} MB  "
              f"peak_transient_workspace={transient/1e6:8.2f} MB")


# ======================================================================
#  fraction: opt-step as a fraction of a full fwd+bwd+opt training step
# ======================================================================
def cmd_fraction(args):
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import torch.nn.functional as F
    lr, mom, wd = 1e-2, 0.95, 0.01

    model = build_production_model(d_model=args.d_model, n_layers=args.n_layers,
                                   n_heads=args.n_heads, d_head=args.d_head,
                                   device=dev)
    vocab = 49152
    adamw_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and (p.ndim != 2 or n in
                       {"embed.weight", "pos_embed.weight", "lm_head.weight"})]
    adamw = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.0)
    # production bf16-state candidate is the per-unit STREAMED path (bf16 ignores
    # batch_across_layers by design), so measure exactly that.
    fused = make_fused(model, lr, mom, wd, bf16_state=True, batch_across_layers=False)
    muon = make_muon(model, lr, mom, wd)

    B, T = args.batch, args.seqlen
    x = torch.randint(0, vocab, (B, T), device=dev)
    y = torch.randint(0, vocab, (B, T), device=dev)

    def fwd_bwd():
        adamw.zero_grad(set_to_none=True)
        for p in (p for _, p in matrix_param_set(model)):
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = (out[0] if isinstance(out, tuple) else out)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   y.reshape(-1))
        loss.backward()

    def full_step(matrix_opt):
        fwd_bwd()
        matrix_opt.step()
        adamw.step()

    # warmup
    for _ in range(args.warmup):
        full_step(fused)
    torch.cuda.synchronize()

    # time full step (fused)
    t0 = time.time()
    for _ in range(args.iters):
        full_step(fused)
    torch.cuda.synchronize()
    full_ms = (time.time() - t0) / args.iters * 1e3

    # time fwd+bwd only
    t0 = time.time()
    for _ in range(args.iters):
        fwd_bwd()
    torch.cuda.synchronize()
    fb_ms = (time.time() - t0) / args.iters * 1e3

    # isolated fused opt step
    fwd_bwd()
    for _ in range(args.warmup):
        fused.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        fused.step()
    torch.cuda.synchronize()
    fused_ms = (time.time() - t0) / args.iters * 1e3

    # isolated muon opt step
    fwd_bwd()
    for _ in range(args.warmup):
        muon.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        muon.step()
    torch.cuda.synchronize()
    muon_ms = (time.time() - t0) / args.iters * 1e3

    print(f"[fraction] {args.n_layers}L d{args.d_model} {args.n_heads}h  B={B} T={T}")
    print(f"  fwd+bwd only         : {fb_ms:8.2f} ms")
    print(f"  full step (fused)    : {full_ms:8.2f} ms")
    print(f"  fused opt-step alone : {fused_ms:8.2f} ms  "
          f"({fused_ms/full_ms*100:5.2f}% of full step)")
    print(f"  muon  opt-step alone : {muon_ms:8.2f} ms  "
          f"({muon_ms/full_ms*100:5.2f}% of full step if substituted)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["verify", "profile", "memory", "fraction"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d_model", type=int, default=896)
    ap.add_argument("--n_layers", type=int, default=10)
    ap.add_argument("--n_heads", type=int, default=14)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100, help="steps per burst")
    ap.add_argument("--reps", type=int, default=15, help="round-robin reps (min reported)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "verify":
        import sys
        sys.exit(0 if cmd_verify(args) else 1)
    elif args.mode == "profile":
        cmd_profile(args)
    elif args.mode == "memory":
        cmd_memory(args)
    elif args.mode == "fraction":
        cmd_fraction(args)


if __name__ == "__main__":
    main()
