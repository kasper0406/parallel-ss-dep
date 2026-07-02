"""Is the DeltaNet linear-attention tax REDUCIBLE? — stage-2 ablation.

`experiments/linearize_smollm2.py` linearizes SmolLM2-360M into a 32L DeltaNet
by inheriting all non-attention weights and training the per-layer DeltaNet
sublayers (MOHAWK-style "stage 2" attention transfer) to match the donor's
attention outputs. The per-layer relative-MSE floored at ~0.20 and the
stage-2 full-forward HumanEval-solution CE floored at ~1.005 — DeltaNet
cannot fully reproduce softmax attention's token mixing. That residual is the
"linear-attention tax".

THIS script asks: is that tax reducible by a MORE EXPRESSIVE attention
replacement, measured cheaply at stage-2 only (~45M tokens/arm)?

Arms (all 32L full-depth, all inherit donor non-attention weights):
  - baseline     : plain DeltaNet (the existing path). Reference: rel-MSE
                   ~0.20, CE ~1.005.
  - deltaproduct : GatedDeltaProduct (num_householder K Householder products
                   per token). GATED FLA kernel — historically hit a CUDA
                   "misaligned address" on sm_120; wrapped in try/except and
                   reported cleanly if it crashes. (use_forget_gate=False to
                   dodge the known-buggy forget-gate kernel.)
  - wide_dhead   : plain DeltaNet with head_dim=128 (vs 64). Expressed via
                   FLA DeltaNet expand_k=expand_v so num_heads stays 15 and
                   input/output stay d_model=960 (the o_proj already maps
                   value_dim=1920 -> 960). Cleanly expressible — verified.
  - hybrid       : a SET of layers kept as REAL softmax attention (RoPE +
                   GQA matching the donor: 15 q heads / 5 KV heads / head_dim
                   64 / donor rope_theta=100000), DeltaNet everywhere else.
                   The Jamba/Zamba escape from the linear-attention tax. The
                   kept-attention sublayers INHERIT the donor q/k/v/o weights
                   byte-exact (our RoPE is Llama-faithful, verified rel-MSE
                   ~1e-13 vs the donor attention) and are ALSO trained in
                   stage 2. Expectation: their per-layer rel-MSE << 0.20 and
                   the full-forward CE drops meaningfully vs baseline.

NOTE on softmax impl: the task originally specified `from fla.layers import
Attention`. That module hard-requires `flash_attn`, which is NOT installed on
this rig (it raises ImportError at construction). We therefore implement a
Llama-faithful SDPA softmax attention (`HybridSoftmaxAttention`). This is
strictly better for the experiment: matching Llama's rotate_half RoPE
convention lets the inherited donor weights be byte-exact (FLA's RotaryEmbedding
uses a different convention, so FLA Attention would NOT inherit byte-exact).

Eval: teacher-forced HumanEval-solution CE — byte-identical protocol to
linearize_smollm2 (reuses humaneval_solution_ce + make_forward_logits, fp32).

Run (single GPU, GPU1):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      experiments/linearize_hybrid_ablation.py --arms baseline,deltaproduct,wide_dhead,hybrid
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/knielsen/ml/parallel-ss-dep")

from experiments.model import TinyLM, RMSNorm
from experiments.layers import (
    DeltaNetAttention,
    GatedDeltaProductAttention,
    _FlaWrapper,
)
# Reuse the linearization primitives verbatim so the eval / data / weight-copy
# protocols are byte-identical to the existing proof.
from experiments.linearize_smollm2 import (
    ARCH,
    DONOR,
    TOKID,
    DEV,
    copy_donor_weights,
    humaneval_solution_ce,
    make_forward_logits,
    code_token_stream,
    save_ckpt,
    verify_copy_zero_attn,
    stage3_e2e,
)

# Donor GQA / RoPE constants (verified from AutoConfig + inv_freq inspection:
# rope_theta lives in rope_parameters/rope_scaling, NOT the top-level field).
DONOR_NUM_KV_HEADS = 5
DONOR_ROPE_THETA = 100000.0


# --------------------------------------------------------------------------- #
# wide_dhead — plain DeltaNet with head_dim=128 instead of 64.
#
# FLA DeltaNet computes head_k_dim = int(hidden_size*expand_k)//num_heads. With
# hidden_size=960, num_heads=15 and expand_k=expand_v=2.0 -> key_dim=value_dim=
# 1920, head_dim=128, and o_proj already maps value_dim(1920) -> hidden_size(960).
# So "wider heads" is cleanly expressible WITHOUT changing the residual width.
# DeltaNetAttention hardcodes expand=1.0 + asserts d_model==n_heads*d_head, so
# we need this dedicated wrapper.
# --------------------------------------------------------------------------- #
class WideDeltaNetAttention(_FlaWrapper):
    """Plain DeltaNet with a configurable per-head dim via expand_k/expand_v.

    Same FLA DeltaNet kernel as `DeltaNetAttention` (plain, ungated, no
    misalign risk), but with `head_dim = target_head_dim` realized through the
    key/value expansion ratio. `n_heads` is preserved; the input and output
    stay at `d_model` (o_proj maps value_dim -> d_model)."""

    accepts_cu_seqlens = True

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 target_head_dim: int = 128):
        from fla.layers import DeltaNet
        key_dim = n_heads * target_head_dim
        # FLA only requires key_dim/value_dim divisible by num_heads (here by
        # construction: key_dim = n_heads*target_head_dim). expand may be
        # fractional; int(hidden_size*expand) must round back to key_dim.
        expand = key_dim / d_model
        assert int(d_model * expand) == key_dim, \
            f"wide_dhead: key_dim {key_dim} not recoverable from expand {expand}"
        super().__init__(DeltaNet(
            mode="chunk",
            d_model=d_model,
            hidden_size=d_model,
            num_heads=n_heads,
            expand_k=expand,
            expand_v=expand,
            use_beta=True,
            use_gate=False,
            use_short_conv=True,
            qk_activation="silu",
            qk_norm="l2",
            allow_neg_eigval=False,
        ))


# --------------------------------------------------------------------------- #
# hybrid — Llama-faithful softmax attention (RoPE + GQA), donor-inheritable.
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Llama (non-interleaved) RoPE: split last dim in two halves."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class HybridSoftmaxAttention(nn.Module):
    """Real softmax attention with Llama-faithful rotate_half RoPE + GQA.

    Exposes the repo `Block` contract: `forward(x, **kw) -> (B, T, d_model)`.
    Tolerates and ignores the extra `cu_seqlens` / `think_mask` kwargs that
    `Block.forward` forwards to FLA wrappers (mirrors `_FlaWrapper`'s tolerance).

    Projections match the donor Llama attention shapes exactly:
        q_proj [d_model, d_model], k_proj/v_proj [kv_dim, d_model],
        o_proj [d_model, d_model]  (no bias).
    so donor q/k/v/o weights inherit byte-exact (verified rel-MSE ~1e-13 vs
    the donor self_attn output). RoPE convention is HF-Llama (rotate_half,
    non-interleaved); the inv_freq base is the donor's rope_theta.
    """

    # We DON'T advertise accepts_cu_seqlens — packed cross-doc isolation is a
    # DeltaNet-recurrence concern; the linearize stream uses doc_ids=None so
    # cu_seqlens is always None here anyway. forward() still tolerates the kwarg.
    accepts_cu_seqlens = False
    accepts_think_mask = False

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 num_kv_heads: int = DONOR_NUM_KV_HEADS,
                 rope_theta: float = DONOR_ROPE_THETA):
        super().__init__()
        assert n_heads % num_kv_heads == 0, \
            f"n_heads({n_heads}) must be divisible by num_kv_heads({num_kv_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.num_kv_heads = num_kv_heads
        self.kv_dim = num_kv_heads * d_head
        self.rope_theta = float(rope_theta)
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor,
                cu_seqlens: torch.Tensor | None = None,
                think_mask: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        B, T, _ = x.shape
        H, KV, D = self.n_heads, self.num_kv_heads, self.d_head
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)      # (B, H, T, D)
        k = self.k_proj(x).view(B, T, KV, D).transpose(1, 2)     # (B, KV, T, D)
        v = self.v_proj(x).view(B, T, KV, D).transpose(1, 2)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(x.device))     # (T, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)                  # (T, D)
        cos = emb.cos()[None, None, :, :].to(x.dtype)
        sin = emb.sin()[None, None, :, :].to(x.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        # Expand KV heads to query-head count for GQA (exact repeat, like HF).
        rep = H // KV
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, H * D)
        return self.o_proj(o)


# --------------------------------------------------------------------------- #
# Arm -> per-layer attention class list.
# --------------------------------------------------------------------------- #
def _evenly_spaced_hybrid_layers(n_hybrid: int, n_layers: int) -> list[int]:
    """Evenly-spaced, biased to the back so the last layer is always kept
    (default for n_hybrid=4 over 32L -> [7, 15, 23, 31])."""
    n = max(1, n_hybrid)
    step = n_layers / n
    return sorted({int(round((i + 1) * step)) - 1 for i in range(n)})


def resolve_hybrid_layers(args, n_layers: int) -> list[int]:
    """Resolve the layer indices kept as softmax attention for the `hybrid` arm.

    `--hybrid_layers` (explicit list) wins when passed; otherwise the
    selection is evenly spaced across `n_layers`, derived from `--n_hybrid`.
    If BOTH are passed explicitly, it's only valid when they agree on the
    *count* — this guards the historical bug here: `--hybrid_layers` used to
    default to a non-empty string that always won, so a run only varying
    `--n_hybrid` silently trained identical hybrids.
    """
    hybrid_layers_explicit = bool(args.hybrid_layers)
    n_hybrid_explicit = bool(getattr(args, "n_hybrid_explicit", False))

    even_sel = None
    if not hybrid_layers_explicit or n_hybrid_explicit:
        even_sel = _evenly_spaced_hybrid_layers(args.n_hybrid, n_layers)

    if hybrid_layers_explicit:
        sel = sorted({int(x) for x in args.hybrid_layers.replace(";", ",").split(",")
                      if x.strip() != ""})
        if n_hybrid_explicit and len(sel) != len(even_sel):
            raise ValueError(
                f"--hybrid_layers ({sel}, count={len(sel)}) and --n_hybrid "
                f"({args.n_hybrid} -> evenly-spaced count={len(even_sel)}) were "
                f"BOTH passed explicitly and disagree in count. Pass only one "
                f"of the two, or make the counts agree.")
    else:
        sel = even_sel
    assert all(0 <= s < n_layers for s in sel), f"hybrid_layers out of range: {sel}"
    return sel


def build_arm_model(arm: str, args, n_layers: int):
    """Return (model, hybrid_set_or_None). Inherited non-attention weights are
    copied by the caller; the per-layer attention class is chosen here."""
    if arm == "baseline":
        cls_list = [DeltaNetAttention] * n_layers
        hybrid_set = None
    elif arm == "deltaproduct":
        cls = functools.partial(GatedDeltaProductAttention,
                                num_householder=args.num_householder)
        cls_list = [cls] * n_layers
        hybrid_set = None
    elif arm == "wide_dhead":
        cls = functools.partial(WideDeltaNetAttention,
                                target_head_dim=args.wide_head_dim)
        cls_list = [cls] * n_layers
        hybrid_set = None
    elif arm == "hybrid":
        hybrid_set = resolve_hybrid_layers(args, n_layers)
        hyb = functools.partial(HybridSoftmaxAttention,
                                num_kv_heads=DONOR_NUM_KV_HEADS,
                                rope_theta=DONOR_ROPE_THETA)
        cls_list = [hyb if i in hybrid_set else DeltaNetAttention
                    for i in range(n_layers)]
    else:
        raise ValueError(f"unknown arm: {arm}")

    model = TinyLM(
        vocab_size=ARCH["vocab_size"], d_model=ARCH["d_model"],
        n_layers=n_layers, n_heads=ARCH["n_heads"], d_head=ARCH["d_head"],
        d_ff=ARCH["d_ff"], max_T=0,
        attention_cls_per_layer=cls_list,
        feedback_mode="none",
        tie_embeddings=ARCH["tie_embeddings"],
    ).to(DEV)
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.eps = ARCH["rms_eps"]
    return model, hybrid_set


@torch.no_grad()
def init_hybrid_attn_from_donor(model, donor, hybrid_set, layer_select):
    """Copy donor self_attn q/k/v/o into the hybrid student layers when shapes
    match. Returns {inherited:[...], random:[...]}."""
    out = {"inherited": [], "random": []}
    d = donor.model
    for i in hybrid_set:
        donor_idx = layer_select[i]
        sa = d.layers[donor_idx].self_attn
        attn = model.blocks[i].attn
        try:
            shapes_ok = (
                attn.q_proj.weight.shape == sa.q_proj.weight.shape and
                attn.k_proj.weight.shape == sa.k_proj.weight.shape and
                attn.v_proj.weight.shape == sa.v_proj.weight.shape and
                attn.o_proj.weight.shape == sa.o_proj.weight.shape and
                sa.q_proj.bias is None and sa.o_proj.bias is None
            )
        except AttributeError:
            shapes_ok = False
        if shapes_ok:
            attn.q_proj.weight.data.copy_(sa.q_proj.weight.data)
            attn.k_proj.weight.data.copy_(sa.k_proj.weight.data)
            attn.v_proj.weight.data.copy_(sa.v_proj.weight.data)
            attn.o_proj.weight.data.copy_(sa.o_proj.weight.data)
            out["inherited"].append(i)
        else:
            out["random"].append(i)
    return out


# --------------------------------------------------------------------------- #
# Per-layer relative-MSE measurement (held-out, reusable at init + final).
# Same protocol as the stage-2 loss: donor attn input (post input_layernorm) ->
# student attn sublayer, compared to donor attn output, normalized by target
# power. Captures donor under bf16 autocast exactly like stage 2.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_per_layer_relmse(model, donor, tok, args, layer_select,
                             n_layers: int, n_batches: int = 4,
                             seed_offset: int = 7777) -> list[float]:
    donor_n_layers = ARCH["n_layers"]
    cap_in = [None] * donor_n_layers
    cap_out = [None] * donor_n_layers
    handles = []
    for i, L in enumerate(donor.model.layers):
        def mk_in(idx):
            def h(_m, _i, o):
                cap_in[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return h
        def mk_out(idx):
            def h(_m, _i, o):
                cap_out[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return h
        handles.append(L.input_layernorm.register_forward_hook(mk_in(i)))
        handles.append(L.self_attn.register_forward_hook(mk_out(i)))

    # try/finally so a fault mid-measurement can't leak donor forward-hooks
    # into the next arm (stale hooks pin activations + waste compute).
    try:
        model.eval()
        sums = [0.0] * n_layers
        cnt = 0
        stream = code_token_stream(tok, args.T, args.batch, seed=args.seed + seed_offset)
        for _ in range(n_batches):
            try:
                x = next(stream)
            except StopIteration:
                break
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                donor(x)
            for i in range(n_layers):
                donor_idx = layer_select[i]
                inp = cap_in[donor_idx].float()
                tgt = cap_out[donor_idx].float()
                our = model.blocks[i].attn(inp).float()
                denom = tgt.pow(2).mean().clamp_min(1e-6)
                sums[i] += (F.mse_loss(our, tgt) / denom).item()
            cnt += 1
    finally:
        for h in handles:
            h.remove()
    if cnt == 0:
        return [float("nan")] * n_layers
    return [s / cnt for s in sums]


# --------------------------------------------------------------------------- #
# Stage 2 — layerwise attention transfer (modified copy: per-layer rel-MSE +
# trainable-param manifest + arm-agnostic). Loss / data identical to
# linearize_smollm2.stage2_layerwise.
# --------------------------------------------------------------------------- #
def stage2_layerwise_arm(model, donor, tok, args, log, layer_select):
    n_layers = len(layer_select)
    donor_n_layers = ARCH["n_layers"]

    # Freeze inherited weights; train ONLY attention sublayers. The selector
    # captures BOTH DeltaNet (blocks.i.attn.layer.*) AND hybrid-softmax
    # (blocks.i.attn.{q,k,v,o}_proj.*) params — both live under blocks.i.attn.
    attn_params = []
    train_names = []
    for n, p in model.named_parameters():
        train = n.startswith("blocks.") and ".attn." in n and ".attn_norm." not in n
        p.requires_grad_(train)
        if train:
            attn_params.append(p)
            train_names.append(n)
    log(f"[stage2] trainable attention params: "
        f"{sum(p.numel() for p in attn_params)/1e6:.1f}M across {len(attn_params)} tensors")
    # Manifest: per-layer count of trainable attn tensors (sanity that hybrid
    # softmax-attn params are actually in the trainable set).
    per_layer_trainable = {}
    for n in train_names:
        try:
            li = int(n.split(".")[1])
        except (IndexError, ValueError):
            continue
        per_layer_trainable[li] = per_layer_trainable.get(li, 0) + 1
    sample_layers = sorted(per_layer_trainable)[:4] + sorted(per_layer_trainable)[-2:]
    log("[stage2] trainable-attn tensors per layer (sample): " +
        ", ".join(f"L{li}={per_layer_trainable[li]}" for li in sorted(set(sample_layers))))

    opt = torch.optim.AdamW(attn_params, lr=args.lr_layerwise, betas=(0.9, 0.95),
                            weight_decay=0.0)
    total_steps = max(1, args.layerwise_tokens // (args.batch * args.T))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr_layerwise, total_steps=total_steps,
        pct_start=0.03, anneal_strategy="cos", div_factor=10, final_div_factor=20)

    donor.eval()
    cap_in = [None] * donor_n_layers
    cap_out = [None] * donor_n_layers
    handles = []
    for i, L in enumerate(donor.model.layers):
        def mk_in(idx):
            def h(_m, _i, o):
                cap_in[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return h
        def mk_out(idx):
            def h(_m, _i, o):
                cap_out[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return h
        handles.append(L.input_layernorm.register_forward_hook(mk_in(i)))
        handles.append(L.self_attn.register_forward_hook(mk_out(i)))

    stream = code_token_stream(tok, args.T, args.batch, seed=args.seed)
    model.train()
    t0 = time.time(); toks_seen = 0; next_eval = args.eval_every_tokens
    running = 0.0; rcount = 0
    # try/finally so an OOM / CUDA fault mid-stage2 can't leak donor
    # forward-hooks into the next arm (stale hooks pin activations across arms).
    try:
        for step in range(total_steps):
            try:
                x = next(stream)
            except StopIteration:
                break
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                donor(x)
            opt.zero_grad(set_to_none=True)
            layer_loss_sum = 0.0
            for i in range(n_layers):
                donor_idx = layer_select[i]
                inp = cap_in[donor_idx].float()
                tgt = cap_out[donor_idx].float()
                our = model.blocks[i].attn(inp).float()
                denom = tgt.pow(2).mean().clamp_min(1e-6)
                loss_i = F.mse_loss(our, tgt) / denom
                (loss_i / n_layers).backward()
                layer_loss_sum += loss_i.item()
            torch.nn.utils.clip_grad_norm_(attn_params, 1.0)
            opt.step(); sched.step()
            running += layer_loss_sum / n_layers; rcount += 1
            toks_seen += args.batch * args.T

            if step % args.log_every == 0:
                tps = toks_seen / max(1e-9, time.time() - t0)
                log(f"[stage2] step {step}/{total_steps} tok {toks_seen/1e6:.1f}M "
                    f"rel_nmse {running/max(1,rcount):.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{tps/1e3:.1f}k tok/s")
                running = 0.0; rcount = 0
            if toks_seen >= next_eval or step == total_steps - 1:
                next_eval += args.eval_every_tokens
                ce, nt = humaneval_solution_ce(make_forward_logits(model), tok)
                log(f"[stage2] *** HumanEval-solution CE @ {toks_seen/1e6:.0f}M tok = {ce:.4f} ***")
                save_ckpt(model, os.path.join(args.out_dir, f"{args.ckpt_prefix}_stage2.pt"),
                          step=step, stage="stage2", n_layers=n_layers, layer_select=layer_select)
                model.train()
            if args.smoke and step >= 3:
                break
    finally:
        for h in handles:
            h.remove()
    ce, _ = humaneval_solution_ce(make_forward_logits(model), tok)
    return ce


# --------------------------------------------------------------------------- #
def run_arm(arm: str, donor, tok, args, log) -> dict:
    n_layers = ARCH["n_layers"]                  # full-depth 32L proof config
    layer_select = list(range(n_layers))         # identity (no depth reduction)
    args.ckpt_prefix = f"hybrid_ablation_{arm}"
    log("=" * 70)
    log(f"==== ARM: {arm} ====")
    rec: dict = {"arm": arm, "n_layers": n_layers, "status": "ok"}

    try:
        model, hybrid_set = build_arm_model(arm, args, n_layers)
        manifest = copy_donor_weights(model, donor, layer_select)
        log(f"  copied {len(manifest['copied'])} non-attn tensors; "
            f"{len(manifest['left_random'])} left-random (attention sublayers)")
        rec["n_params"] = int(model.num_params())
        log(f"  total model params: {model.num_params()/1e6:.1f}M "
            f"(baseline DeltaNet ~402M; deltaproduct/wide_dhead are LARGER — NOT "
            f"iso-param; hybrid ~iso-param ~397M → its kill-signal is unconfounded)")

        hybrid_init = None
        if arm == "hybrid":
            rec["hybrid_layers"] = hybrid_set
            hybrid_init = init_hybrid_attn_from_donor(model, donor, hybrid_set, layer_select)
            log(f"  hybrid layers: {hybrid_set}")
            log(f"  donor-attn inherited byte-exact into layers {hybrid_init['inherited']}; "
                f"random-init layers {hybrid_init['random']}")
            rec["hybrid_attn_init"] = hybrid_init

        # Copy-correctness of the inherited NON-attention weights (zero both
        # models' attention, compare logits). Type-agnostic — works for any arm.
        vr = verify_copy_zero_attn(model, donor, tok)
        log(f"  [verify] zero-attn logit match vs donor: max|Δ|={vr['max_abs_diff']:.3e} "
            f"argmax-agree={vr['argmax_agree']*100:.1f}%")
        rec["verify_zero_attn_maxdiff"] = vr["max_abs_diff"]

        # Per-layer rel-MSE at INIT (before any training). For hybrid this shows
        # the inherited-attn layers already near 0; for the rest, random ~O(1).
        relmse_init = measure_per_layer_relmse(model, donor, tok, args, layer_select,
                                               n_layers, n_batches=2 if args.smoke else 4)
        rec["per_layer_relmse_init"] = relmse_init
        log(f"  [init] per-layer rel-MSE: mean={_mean(relmse_init):.4f} "
            f"min={min(relmse_init):.4f} max={max(relmse_init):.4f}")

        ce_init, nt = humaneval_solution_ce(make_forward_logits(model), tok)
        rec["ce_init"] = ce_init
        log(f"  [init] HumanEval-solution CE = {ce_init:.4f} "
            f"(uniform={math.log(ARCH['vocab_size']):.3f}, {nt} sol tokens)")

        # Stage 2 — the actual ablation.
        ce_s2 = stage2_layerwise_arm(model, donor, tok, args, log, layer_select)
        rec["ce_stage2"] = ce_s2
        log(f"  [stage2] FINAL HumanEval-solution CE = {ce_s2:.4f}")

        relmse_final = measure_per_layer_relmse(model, donor, tok, args, layer_select,
                                                n_layers, n_batches=2 if args.smoke else 4)
        rec["per_layer_relmse"] = relmse_final
        rec["mean_relmse"] = _mean(relmse_final)
        log(f"  [stage2] per-layer rel-MSE FINAL: mean={rec['mean_relmse']:.4f} "
            f"min={min(relmse_final):.4f} max={max(relmse_final):.4f}")
        if arm == "hybrid" and hybrid_set:
            hyb_vals = [relmse_final[i] for i in hybrid_set]
            dn_vals = [relmse_final[i] for i in range(n_layers) if i not in hybrid_set]
            rec["hybrid_layer_relmse_mean"] = _mean(hyb_vals)
            rec["deltanet_layer_relmse_mean"] = _mean(dn_vals)
            log(f"  [stage2] hybrid-layer rel-MSE mean={_mean(hyb_vals):.4f}  "
                f"deltanet-layer rel-MSE mean={_mean(dn_vals):.4f}")

        # Optional stage 3 (default off — strategic decision is at stage-2 level).
        if args.with_stage3:
            ce_s3 = stage3_e2e(model, donor, tok, args, log,
                               n_layers=n_layers, layer_select=layer_select)
            rec["ce_stage3"] = ce_s3
            log(f"  [stage3] FINAL HumanEval-solution CE = {ce_s3:.4f}")

        del model
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 — deltaproduct may CUDA-crash on sm_120
        import traceback
        tb = traceback.format_exc()
        log(f"  !!! ARM {arm} CRASHED: {type(e).__name__}: {str(e)[:300]}")
        log(tb)
        rec["status"] = "crashed"
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return rec


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="",
                    help="single arm to run (baseline/deltaproduct/wide_dhead/hybrid)")
    ap.add_argument("--arms", type=str, default="baseline,wide_dhead,hybrid,deltaproduct",
                    help="comma-separated arms to run sequentially. deltaproduct "
                         "runs LAST: a CUDA 'misaligned address' on sm_120 is "
                         "sticky (poisons the process), so an isolated crash there "
                         "can't cost the strategic hybrid kill-signal answer.")
    ap.add_argument("--out_dir", type=str, default="checkpoints/hybrid_ablation")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--layerwise_tokens", type=int, default=45_000_000)
    ap.add_argument("--lr_layerwise", type=float, default=1.0e-3)
    ap.add_argument("--eval_every_tokens", type=int, default=25_000_000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    # deltaproduct
    ap.add_argument("--num_householder", type=int, default=2)
    # wide_dhead
    ap.add_argument("--wide_head_dim", type=int, default=128)
    # hybrid
    ap.add_argument("--n_hybrid", type=int, default=None,
                    help="number of layers kept as softmax attention, evenly "
                         "spaced (biased to the back) across n_layers; used "
                         "whenever --hybrid_layers is empty. Default 4, which "
                         "on the 32L proof config resolves to the same "
                         "[7,15,23,31] as the old hardcoded --hybrid_layers "
                         "default.")
    ap.add_argument("--hybrid_layers", type=str, default="",
                    help="comma-separated layer indices kept as softmax attention; "
                         "empty (default) = evenly-spaced from --n_hybrid. "
                         "Passing both explicitly is a hard error unless their "
                         "counts agree (see resolve_hybrid_layers).")
    # optional stage 3 (off by default)
    ap.add_argument("--with_stage3", action="store_true")
    ap.add_argument("--e2e_batch", type=int, default=4)
    ap.add_argument("--e2e_tokens", type=int, default=40_000_000)
    ap.add_argument("--lr_e2e", type=float, default=1.0e-4)
    ap.add_argument("--kd_temp", type=float, default=2.0)
    ap.add_argument("--ce_anchor", type=float, default=0.1)
    args = ap.parse_args()
    # Sentinel-default so resolve_hybrid_layers can tell "--n_hybrid was passed
    # explicitly" apart from "using the default" (needed for the both-explicit
    # count-mismatch hard-error, and to preserve the old [7,15,23,31] default).
    args.n_hybrid_explicit = args.n_hybrid is not None
    if args.n_hybrid is None:
        args.n_hybrid = 4
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    def log(msg):
        print(msg, flush=True)

    arms = [args.arm] if args.arm else [a.strip() for a in args.arms.split(",") if a.strip()]
    log(f"ARMS: {arms}  (smoke={args.smoke}, layerwise_tokens={args.layerwise_tokens:,}, "
        f"with_stage3={args.with_stage3})")
    if "hybrid" in arms:
        resolved = resolve_hybrid_layers(args, ARCH["n_layers"])
        source = ("--hybrid_layers (explicit)" if args.hybrid_layers
                   else f"--n_hybrid={args.n_hybrid}"
                        f"{' (explicit)' if args.n_hybrid_explicit else ' (default)'}")
        log(f"[startup] hybrid arm resolved layers: {resolved}  (source: {source})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading tokenizer {TOKID} + donor {DONOR} (once, reused across arms) ...")
    tok = AutoTokenizer.from_pretrained(TOKID)
    donor = AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float32).to(DEV).eval()
    for p in donor.parameters():
        p.requires_grad_(False)

    results = {}
    for arm in arms:
        rec = run_arm(arm, donor, tok, args, log)
        results[arm] = rec
        with open(os.path.join(args.out_dir, f"{arm}.json"), "w") as f:
            json.dump(rec, f, indent=2)
        log(f"  wrote {os.path.join(args.out_dir, arm + '.json')}")

    # Combined summary.
    summary = {
        "reference": {"deltanet_baseline_relmse": 0.20,
                      "deltanet_baseline_ce": 1.005,
                      "donor_smollm2_360M_ce": 0.6142,
                      "our_from_scratch_ce": 0.9716},
        "fair_compute_note": (
            "deltaproduct (~491M) and wide_dhead (~520M) are NOT iso-param with "
            "baseline (~402M) — a CE drop there is partly more params, not a more "
            "expressive cell. hybrid (~397M) IS ~iso-param with baseline, so the "
            "hybrid kill-signal is NOT confounded by param count. See per-arm "
            "n_params."),
        "config": {"layerwise_tokens": args.layerwise_tokens, "T": args.T,
                   "batch": args.batch, "smoke": args.smoke,
                   "with_stage3": args.with_stage3,
                   "num_householder": args.num_householder,
                   "wide_head_dim": args.wide_head_dim,
                   "hybrid_layers": args.hybrid_layers or f"even-{args.n_hybrid}"},
        "arms": {a: {k: v for k, v in r.items()
                     if k not in ("per_layer_relmse", "per_layer_relmse_init")}
                 for a, r in results.items()},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    log("==== SUMMARY (stage-2 CE floor + mean rel-MSE) ====")
    log(f"  reference: DeltaNet baseline rel-MSE ~0.20 / CE ~1.005 ; "
        f"donor 0.614 ; from-scratch 0.972")
    base_ce = results.get("baseline", {}).get("ce_stage2")
    for a, r in results.items():
        if r.get("status") != "ok":
            log(f"  {a:13s}: CRASHED ({r.get('error','?')[:80]})")
            continue
        ce = r.get("ce_stage2", float('nan'))
        mr = r.get("mean_relmse", float('nan'))
        drop = ""
        if base_ce and a != "baseline" and ce == ce:
            rel = (base_ce - ce) / base_ce * 100
            drop = f"  (CE drop vs baseline: {rel:+.1f}%)"
        log(f"  {a:13s}: stage2 CE={ce:.4f}  mean rel-MSE={mr:.4f}{drop}")
    log("  KILL-SIGNAL: hybrid must drop stage-2 CE >15-20% vs baseline, "
        "else the linear-attention tax is structural.")
    log(f"wrote {os.path.join(args.out_dir, 'summary.json')}")
    os._exit(0)


if __name__ == "__main__":
    main()
