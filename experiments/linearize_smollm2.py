"""SmolLM2-360M -> DeltaNet LINEARIZATION proof.

Hypothesis: a DeltaNet whose NON-attention weights are inherited from
SmolLM2-360M, and whose attention sublayers are DISTILLED from the donor's
attention (MOHAWK / Mamba-in-Llama style), retains the donor's knowledge —
landing near SmolLM2-360M's teacher-forced HumanEval-solution CE (~0.614)
rather than our-from-scratch (~0.969).

The donor (SmolLM2-360M, a Llama block) is STRUCTURALLY identical to our
`TinyLM` `Block` except the attention sublayer (RoPE-GQA vs DeltaNet). So
these copy directly:
    embed_tokens            -> embed
    input_layernorm[i]      -> blocks[i].attn_norm
    post_attention_ln[i]    -> blocks[i].mlp_norm
    mlp.gate/up/down_proj   -> blocks[i].mlp.W_g/W_u/W_d
    model.norm              -> out_norm
    (tied) lm_head          -> embed (tied)
Only the per-layer DeltaNet attention sublayer is LEARNED.

Stages:
  0. INIT  : build bare DeltaNet TinyLM, copy weights, verify the copy is
             exact (zero-attention logit match vs donor), save init ckpt.
  2. LAYERWISE attention transfer: freeze inherited weights; train each
             DeltaNet sublayer to match (MSE) the donor attention sublayer's
             output given the donor's own attention-sublayer input.
  3. END-TO-END distill: unfreeze all; KL to donor logits (+ small CE anchor).

Eval (identical protocol to /tmp/probe_capacity_ref.py & /tmp/probe_he_ce.py):
teacher-forced HumanEval-solution CE.

Run (single GPU, GPU1):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python experiments/linearize_smollm2.py \
      --layerwise_tokens 200_000_000 --e2e_tokens 80_000_000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/knielsen/ml/parallel-ss-dep")

from experiments.model import TinyLM, RMSNorm
from experiments.layers import DeltaNetAttention

DONOR = "HuggingFaceTB/SmolLM2-360M"
TOKID = "HuggingFaceTB/SmolLM2-135M"   # identical 49152 vocab; the probe's tokenizer
DEV = "cuda"

# SmolLM2-360M architecture (verified from AutoConfig). This is the DONOR arch;
# the STUDENT may keep fewer layers (depth-reduction, see resolve_layer_select).
ARCH = dict(
    vocab_size=49152, d_model=960, n_layers=32, n_heads=15, d_head=64,
    d_ff=2560, rms_eps=1e-5, tie_embeddings=True,
)

# --------------------------------------------------------------------------- #
# Depth-reduction: a STUDENT with n_layers_out < donor depth keeps a SUBSET of
# donor layers, mapping student layer i -> donor layer LAYER_SELECT[i]. Identity
# (n_layers_out == 32, no explicit list) preserves the original 32L path
# byte-for-byte. The default 10L selection is first + last + 8 evenly spaced.
# NOTE: strided layer-drop BREAKS the residual composition (donor layer SEL[i]
# was trained to read SEL[i]-1's output, not the previous KEPT layer's), so the
# stage-3 end-to-end KD is the workhorse that heals the composition break.
# --------------------------------------------------------------------------- #
DEFAULT_LAYER_SELECT_10L = [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]


def resolve_layer_select(n_layers_out: int, explicit: str | None) -> list[int]:
    """Return the donor-layer indices kept by the student, length n_layers_out.

    `explicit` (comma/semicolon-separated) overrides everything. Otherwise:
    full-depth -> identity; n_layers_out==10 -> DEFAULT_LAYER_SELECT_10L;
    else evenly-spaced incl. first+last.
    """
    donor_L = ARCH["n_layers"]
    if explicit:
        sel = [int(x) for x in explicit.replace(";", ",").split(",") if x.strip()]
    elif n_layers_out == donor_L:
        sel = list(range(donor_L))
    elif n_layers_out == 10:
        sel = list(DEFAULT_LAYER_SELECT_10L)
    else:
        sel = sorted({round(i * (donor_L - 1) / (n_layers_out - 1))
                      for i in range(n_layers_out)})
    assert len(sel) == n_layers_out, \
        f"layer_select has {len(sel)} entries but n_layers_out={n_layers_out}: {sel}"
    assert all(0 <= s < donor_L for s in sel), f"layer_select out of range: {sel}"
    return sel


# --------------------------------------------------------------------------- #
# Model construction + weight copy
# --------------------------------------------------------------------------- #
def build_bare_deltanet(n_layers: int | None = None) -> TinyLM:
    """A clean Llama-with-DeltaNet-attention: no memory / PKM / gate / FiLM /
    latent. RMSNorm eps set to the donor's 1e-5 so the inherited norms behave
    byte-identically to SmolLM2. `n_layers` defaults to the donor depth (32);
    pass a smaller value for a depth-reduced student."""
    if n_layers is None:
        n_layers = ARCH["n_layers"]
    model = TinyLM(
        vocab_size=ARCH["vocab_size"], d_model=ARCH["d_model"],
        n_layers=n_layers, n_heads=ARCH["n_heads"],
        d_head=ARCH["d_head"], d_ff=ARCH["d_ff"], max_T=0,
        attention_cls=DeltaNetAttention,
        feedback_mode="none",
        tie_embeddings=ARCH["tie_embeddings"],
    ).to(DEV)
    # Match the donor's RMSNorm epsilon (TinyLM defaults to 1e-6).
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.eps = ARCH["rms_eps"]
    return model


@torch.no_grad()
def copy_donor_weights(model: TinyLM, donor, layer_select: list[int] | None = None) -> dict:
    """Copy donor non-attention weights into `model` in-place. Student layer i
    inherits donor layer `layer_select[i]`'s MLP + the two norms. Returns a
    manifest dict {copied:[...], left_random:[...]} with shape assertions."""
    d = donor.model  # LlamaModel
    if layer_select is None:
        layer_select = list(range(ARCH["n_layers"]))
    copied, left_random = [], []

    def cp(dst: torch.Tensor, src: torch.Tensor, name: str):
        assert dst.shape == src.shape, \
            f"SHAPE MISMATCH {name}: dst {tuple(dst.shape)} vs src {tuple(src.shape)}"
        dst.data.copy_(src.data.to(dst.dtype))
        copied.append(name)

    # Token embedding (lm_head is tied to embed -> updated automatically).
    cp(model.embed.weight, d.embed_tokens.weight, "embed.weight (<-embed_tokens, tied lm_head)")
    # Per-layer norms + MLP: student block i <- donor layer layer_select[i].
    for i, donor_idx in enumerate(layer_select):
        L = d.layers[donor_idx]
        B = model.blocks[i]
        tag = f" (<-donor L{donor_idx})"
        cp(B.attn_norm.weight, L.input_layernorm.weight, f"blocks.{i}.attn_norm.weight{tag}")
        cp(B.mlp_norm.weight, L.post_attention_layernorm.weight, f"blocks.{i}.mlp_norm.weight{tag}")
        cp(B.mlp.W_g.weight, L.mlp.gate_proj.weight, f"blocks.{i}.mlp.W_g.weight{tag}")
        cp(B.mlp.W_u.weight, L.mlp.up_proj.weight, f"blocks.{i}.mlp.W_u.weight{tag}")
        cp(B.mlp.W_d.weight, L.mlp.down_proj.weight, f"blocks.{i}.mlp.W_d.weight{tag}")
    # Final norm.
    cp(model.out_norm.weight, d.norm.weight, "out_norm.weight (<-model.norm)")

    # Everything under blocks.*.attn.* is the DeltaNet sublayer -> left random.
    copied_set = set()
    # rebuild the exact param-name set we touched for a clean diff
    for n, _ in model.named_parameters():
        is_copied = (
            n == "embed.weight" or n == "lm_head.weight" or n == "out_norm.weight"
            or (n.startswith("blocks.") and (
                ".attn_norm." in n or ".mlp_norm." in n or ".mlp." in n))
        )
        if is_copied:
            copied_set.add(n)
        else:
            left_random.append(n)
    return {"copied": copied, "left_random": left_random,
            "n_copied_params": len(copied_set)}


# --------------------------------------------------------------------------- #
# Copy-correctness verification: zero attention in BOTH models, compare logits.
# If the copy is exact, the two MLP-only forwards must match to ~fp32 noise.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_copy_zero_attn(model: TinyLM, donor, tok, n_tokens: int = 256) -> dict:
    """Zero every attention sublayer in both models and compare the resulting
    logits on the same input. A near-zero max-abs-diff proves the inherited
    embed/norm/MLP weights are wired correctly."""
    model.eval(); donor.eval()
    # Build a short input from HumanEval text (deterministic).
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    ids = tok.encode(ds[0]["prompt"] + ds[0]["canonical_solution"],
                     add_special_tokens=False)[:n_tokens]
    x = torch.tensor([ids], device=DEV)

    # --- donor with attention zeroed ---
    d_handles = []
    for L in donor.model.layers:
        def z(_m, _i, o):
            if isinstance(o, tuple):
                return (torch.zeros_like(o[0]),) + tuple(o[1:])
            return torch.zeros_like(o)
        d_handles.append(L.self_attn.register_forward_hook(z))
    donor_logits = donor(x).logits.float()
    for h in d_handles:
        h.remove()

    # --- our model with attention zeroed ---
    m_handles = []
    for B in model.blocks:
        def z2(_m, _i, o):
            return torch.zeros_like(o[0] if isinstance(o, tuple) else o)
        m_handles.append(B.attn.register_forward_hook(z2))
    our = model(x)
    our_logits = (our[0] if isinstance(our, (tuple, list)) else our).float()
    for h in m_handles:
        h.remove()

    diff = (our_logits - donor_logits).abs()
    # Agreement of next-token argmax (sanity on the whole distribution).
    agree = (our_logits.argmax(-1) == donor_logits.argmax(-1)).float().mean().item()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "donor_logit_scale": donor_logits.abs().mean().item(),
        "argmax_agree": agree,
        "n_tokens": x.shape[1],
    }


# --------------------------------------------------------------------------- #
# Depth-reduced copy verification: the zero-attn full forward can NOT match the
# donor (different layer count), so instead check that every COPIED tensor is
# bit-identical to its donor source. (cp() copies bit-exact, so this is a guard
# against a wiring/shape bug, not a numerical test.)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_copy_params(model: TinyLM, donor, layer_select: list[int]) -> dict:
    d = donor.model
    max_diff = 0.0
    n_checked = 0

    def chk(dst, src):
        nonlocal max_diff, n_checked
        diff = (dst.float() - src.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        n_checked += 1

    chk(model.embed.weight, d.embed_tokens.weight)
    chk(model.out_norm.weight, d.norm.weight)
    for i, donor_idx in enumerate(layer_select):
        L = d.layers[donor_idx]
        B = model.blocks[i]
        chk(B.attn_norm.weight, L.input_layernorm.weight)
        chk(B.mlp_norm.weight, L.post_attention_layernorm.weight)
        chk(B.mlp.W_g.weight, L.mlp.gate_proj.weight)
        chk(B.mlp.W_u.weight, L.mlp.up_proj.weight)
        chk(B.mlp.W_d.weight, L.mlp.down_proj.weight)
    return {"max_abs_diff": max_diff, "mean_abs_diff": max_diff,
            "donor_logit_scale": None, "argmax_agree": None,
            "n_tokens": n_checked, "param_level": True}


# --------------------------------------------------------------------------- #
# Data: stream codeparrot, tokenize, pack into (B, T) blocks.
# --------------------------------------------------------------------------- #
def code_token_stream(tok, block_size: int, batch: int, seed: int = 0):
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=2000, seed=seed)
    eos = tok.eos_token_id or 0
    buf: list[int] = []
    rows: list[list[int]] = []
    for ex in ds:
        toks = tok.encode(ex["content"], add_special_tokens=False)
        buf.extend(toks)
        buf.append(eos)
        while len(buf) >= block_size:
            rows.append(buf[:block_size])
            buf = buf[block_size:]
            if len(rows) == batch:
                yield torch.tensor(rows, device=DEV)
                rows = []


# --------------------------------------------------------------------------- #
# HumanEval-solution CE — verbatim protocol from /tmp/probe_capacity_ref.py.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def humaneval_solution_ce(forward_logits, tok) -> tuple[float, int]:
    """`forward_logits(x)` -> logits (B=1, T, V). Returns (CE, n_sol_tokens)."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    tot, ntok = 0.0, 0
    for ex in ds:
        prompt, sol = ex["prompt"], ex["canonical_solution"]
        pids = tok.encode(prompt, add_special_tokens=False)
        fids = tok.encode(prompt + sol, add_special_tokens=False)
        if len(fids) <= len(pids) + 1:
            continue
        x = torch.tensor([fids], device=DEV)
        lg = forward_logits(x)[0, :-1, :]
        tgt = x[0, 1:]
        s = max(len(pids) - 1, 0)
        tot += F.cross_entropy(lg[s:].float(), tgt[s:], reduction="sum").item()
        ntok += tgt[s:].numel()
    return tot / ntok, ntok


def make_forward_logits(model: TinyLM, bf16: bool = False):
    """fp32 forward by default — matches the reference `probe_he_ce.py` protocol
    that produced the our-from-scratch floor (0.9716), so the linearized number
    is directly comparable. The DeltaNet FLA kernel still runs in bf16 internally
    (handled inside `_FlaWrapper`); only the MLP/embed/lm_head matmuls differ
    between fp32 and the bf16-autocast path (≈<0.02 CE)."""
    @torch.no_grad()
    def f(x):
        model.eval()
        if bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
        else:
            out = model(x)
        return (out[0] if isinstance(out, (tuple, list)) else out)
    return f


# --------------------------------------------------------------------------- #
# Stage 2 — layerwise attention transfer (MOHAWK-style block matching).
# --------------------------------------------------------------------------- #
def stage2_layerwise(model, donor, tok, args, log, layer_select):
    n_layers = len(layer_select)          # student layers
    donor_n_layers = ARCH["n_layers"]     # donor layers (hooks span all of these)
    # Freeze inherited weights; train ONLY DeltaNet sublayers.
    attn_params = []
    for n, p in model.named_parameters():
        train = n.startswith("blocks.") and ".attn." in n and ".attn_norm." not in n
        p.requires_grad_(train)
        if train:
            attn_params.append(p)
    log(f"[stage2] trainable DeltaNet params: "
        f"{sum(p.numel() for p in attn_params)/1e6:.1f}M across {len(attn_params)} tensors")

    opt = torch.optim.AdamW(attn_params, lr=args.lr_layerwise, betas=(0.9, 0.95),
                            weight_decay=0.0)
    total_steps = max(1, args.layerwise_tokens // (args.batch * args.T))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr_layerwise, total_steps=total_steps,
        pct_start=0.03, anneal_strategy="cos", div_factor=10, final_div_factor=20)

    donor.eval()
    # Capture donor per-layer attn input (post input_layernorm) + output, for
    # ALL donor layers; student block i trains against donor layer layer_select[i].
    cap_in: list[torch.Tensor] = [None] * donor_n_layers
    cap_out: list[torch.Tensor] = [None] * donor_n_layers
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
    for step in range(total_steps):
        try:
            x = next(stream)
        except StopIteration:
            break
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            donor(x)  # populates cap_in / cap_out via hooks
        opt.zero_grad(set_to_none=True)
        layer_loss_sum = 0.0
        for i in range(n_layers):
            donor_idx = layer_select[i]
            inp = cap_in[donor_idx].float()
            tgt = cap_out[donor_idx].float()
            our = model.blocks[i].attn(inp).float()
            # Relative MSE so all layers are weighted equally regardless of
            # their activation magnitude.
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
    for h in handles:
        h.remove()
    ce, _ = humaneval_solution_ce(make_forward_logits(model), tok)
    return ce


# --------------------------------------------------------------------------- #
# Stage 3 — end-to-end logit distillation (KL + small CE anchor).
# --------------------------------------------------------------------------- #
def stage3_e2e(model, donor, tok, args, log, n_layers=None, layer_select=None):
    if n_layers is None:
        n_layers = ARCH["n_layers"]
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_e2e,
                            betas=(0.9, 0.95), weight_decay=0.0)
    bs = args.e2e_batch
    total_steps = max(1, args.e2e_tokens // (bs * args.T))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr_e2e, total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos", div_factor=10, final_div_factor=20)
    T = args.kd_temp
    donor.eval()
    stream = code_token_stream(tok, args.T, bs, seed=args.seed + 101)
    model.train()
    t0 = time.time(); toks_seen = 0; next_eval = args.eval_every_tokens
    run_kl = 0.0; run_ce = 0.0; rc = 0
    for step in range(total_steps):
        try:
            x = next(stream)
        except StopIteration:
            break
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t_logits = donor(x).logits.float()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            s_logits = (out[0] if isinstance(out, (tuple, list)) else out).float()
        # KL(teacher || student) on next-token positions. Reshape to
        # (N_tok, V) so reduction='batchmean' divides by N_tok = batch*time
        # (per-token KL), not just the batch dim.
        V = s_logits.shape[-1]
        sl = s_logits[:, :-1, :].reshape(-1, V)
        tl = t_logits[:, :-1, :].reshape(-1, V)
        kl = F.kl_div(F.log_softmax(sl / T, -1),
                      F.softmax(tl / T, -1),
                      reduction="batchmean") * (T * T)
        # small hard-target CE anchor (Hinton soft+hard).
        ce = F.cross_entropy(sl, x[:, 1:].reshape(-1))
        loss = kl + args.ce_anchor * ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_kl += kl.item(); run_ce += ce.item(); rc += 1
        toks_seen += bs * args.T
        if step % args.log_every == 0:
            tps = toks_seen / max(1e-9, time.time() - t0)
            log(f"[stage3] step {step}/{total_steps} tok {toks_seen/1e6:.1f}M "
                f"kl {run_kl/max(1,rc):.4f} ce {run_ce/max(1,rc):.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} {tps/1e3:.1f}k tok/s")
            run_kl = run_ce = 0.0; rc = 0
        if toks_seen >= next_eval or step == total_steps - 1:
            next_eval += args.eval_every_tokens
            ce_he, _ = humaneval_solution_ce(make_forward_logits(model), tok)
            log(f"[stage3] *** HumanEval-solution CE @ {toks_seen/1e6:.0f}M tok = {ce_he:.4f} ***")
            # Rolling main ckpt (overwritten) + a token-tagged snapshot so the
            # heal trajectory is recoverable.
            save_ckpt(model, os.path.join(args.out_dir, f"{args.ckpt_prefix}_stage3.pt"),
                      step=step, stage="stage3", n_layers=n_layers, layer_select=layer_select)
            snap = os.path.join(args.out_dir,
                                f"{args.ckpt_prefix}_stage3_{toks_seen//1_000_000}M.pt")
            save_ckpt(model, snap, step=step, stage="stage3",
                      n_layers=n_layers, layer_select=layer_select)
            log(f"[stage3]     saved snapshot {os.path.basename(snap)}")
            model.train()
        if args.smoke and step >= 3:
            break
    ce_he, _ = humaneval_solution_ce(make_forward_logits(model), tok)
    return ce_he


# --------------------------------------------------------------------------- #
def save_ckpt(model, path, step, stage, n_layers=None, layer_select=None):
    if n_layers is None:
        n_layers = ARCH["n_layers"]
    cfg = dict(arch="deltanet", n_layers=n_layers, d_model=ARCH["d_model"],
               n_heads=ARCH["n_heads"], d_head=ARCH["d_head"], d_ff=ARCH["d_ff"],
               vocab_size=ARCH["vocab_size"], tie_embeddings=ARCH["tie_embeddings"],
               feedback="none", rms_eps=ARCH["rms_eps"], stage=stage,
               donor=DONOR, layer_select=layer_select)
    torch.save({"state_dict": model.state_dict(), "step": step, "config": cfg}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--e2e_batch", type=int, default=4)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--layerwise_tokens", type=int, default=200_000_000)
    ap.add_argument("--e2e_tokens", type=int, default=80_000_000)
    ap.add_argument("--lr_layerwise", type=float, default=1.0e-3)
    ap.add_argument("--lr_e2e", type=float, default=1.0e-4)
    ap.add_argument("--kd_temp", type=float, default=2.0)
    ap.add_argument("--ce_anchor", type=float, default=0.1)
    ap.add_argument("--eval_every_tokens", type=int, default=50_000_000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_stage2", action="store_true")
    ap.add_argument("--skip_stage3", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # --- depth-reduction (default: full-depth 32L identity = original behaviour) ---
    ap.add_argument("--n_layers_out", type=int, default=ARCH["n_layers"],
                    help="student layer count; <32 keeps a subset of donor layers")
    ap.add_argument("--layer_select", type=str, default="",
                    help="explicit comma-separated donor-layer indices to keep "
                         "(len must == n_layers_out); empty = auto")
    ap.add_argument("--ckpt_prefix", type=str, default="linearized",
                    help="prefix for saved ckpts ({prefix}_init/stage2/stage3.pt)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    def log(msg):
        print(msg, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading tokenizer {TOKID} + donor {DONOR} ...")
    tok = AutoTokenizer.from_pretrained(TOKID)
    donor = AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float32).to(DEV).eval()
    for p in donor.parameters():
        p.requires_grad_(False)

    # --- resolve depth-reduction layer selection ---
    n_layers_out = args.n_layers_out
    layer_select = resolve_layer_select(n_layers_out, args.layer_select)
    is_identity = (layer_select == list(range(ARCH["n_layers"])))
    log(f"DEPTH: donor {ARCH['n_layers']}L -> student {n_layers_out}L  "
        f"(identity={is_identity})")
    log(f"  layer_select (student i <- donor): {layer_select}")

    log("building bare DeltaNet TinyLM + copying donor weights ...")
    model = build_bare_deltanet(n_layers_out)
    manifest = copy_donor_weights(model, donor, layer_select)
    log(f"  COPIED {len(manifest['copied'])} tensors "
        f"({manifest['n_copied_params']} param-tensors in name-set)")
    log(f"  LEFT RANDOM {len(manifest['left_random'])} param-tensors "
        f"(DeltaNet attention sublayers)")
    log("  copied (sample): " + ", ".join(manifest["copied"][:4]) + ", ...")
    log("  left-random (sample): " + ", ".join(manifest["left_random"][:4]) + ", ...")
    log(f"  total model params: {model.num_params()/1e6:.1f}M")

    # --- copy-correctness ---
    if is_identity:
        # full-depth: zero-attention logit match vs donor (strong numeric test).
        vr = verify_copy_zero_attn(model, donor, tok)
        log(f"[verify] zero-attn logit match vs donor: max|Δ|={vr['max_abs_diff']:.4e} "
            f"mean|Δ|={vr['mean_abs_diff']:.4e} (donor logit scale {vr['donor_logit_scale']:.3f}) "
            f"argmax-agree={vr['argmax_agree']*100:.1f}%")
        copy_ok = vr["max_abs_diff"] < 0.05
        log(f"[verify] COPY {'EXACT (PASS)' if copy_ok else 'MISMATCH (FAIL)'}")
    else:
        # depth-reduced: full forward can't match donor; verify copied tensors
        # are bit-identical to their selected donor sources instead.
        vr = verify_copy_params(model, donor, layer_select)
        copy_ok = vr["max_abs_diff"] < 1e-6
        log(f"[verify] depth-reduced param copy: max|Δ|={vr['max_abs_diff']:.2e} "
            f"over {vr['n_tokens']} tensors -> "
            f"{'EXACT (PASS)' if copy_ok else 'MISMATCH (FAIL)'}")

    save_ckpt(model, os.path.join(args.out_dir, f"{args.ckpt_prefix}_init.pt"),
              step=0, stage="init", n_layers=n_layers_out, layer_select=layer_select)

    # --- CE at init: random DeltaNet attn (sanity: << uniform = ln(49152)) ---
    uniform_ce = math.log(ARCH["vocab_size"])
    ce_init, nt = humaneval_solution_ce(make_forward_logits(model), tok)
    log(f"[init] HumanEval-solution CE (random DeltaNet attn) = {ce_init:.4f} "
        f"(uniform={uniform_ce:.3f}, {nt} sol tokens)")

    results = {"verify": vr, "uniform_ce": uniform_ce, "ce_init": ce_init,
               "manifest": {k: (v if not isinstance(v, list) else len(v))
                            for k, v in manifest.items()}}

    results["layer_select"] = layer_select
    results["n_layers_out"] = n_layers_out

    ce_s2 = None
    if not args.skip_stage2:
        ce_s2 = stage2_layerwise(model, donor, tok, args, log, layer_select)
        log(f"[stage2] FINAL HumanEval-solution CE = {ce_s2:.4f}")
        results["ce_stage2"] = ce_s2

    ce_s3 = None
    if not args.skip_stage3:
        ce_s3 = stage3_e2e(model, donor, tok, args, log,
                           n_layers=n_layers_out, layer_select=layer_select)
        log(f"[stage3] FINAL HumanEval-solution CE = {ce_s3:.4f}")
        results["ce_stage3"] = ce_s3

    log("==== SUMMARY ====")
    log(f"  SmolLM2-360M teacher (target) : 0.6142")
    log(f"  our-from-scratch SFT (floor)  : 0.9716")
    log(f"  (a) init random-attn          : {ce_init:.4f}")
    if ce_s2 is not None:
        log(f"  (b) after layerwise transfer  : {ce_s2:.4f}")
    if ce_s3 is not None:
        log(f"  (c) after end-to-end distill  : {ce_s3:.4f}")
    import json
    rj = "linearize_results.json" if args.ckpt_prefix == "linearized" \
        else f"linearize_results_{args.ckpt_prefix}.json"
    with open(os.path.join(args.out_dir, rj), "w") as f:
        json.dump(results, f, indent=2)
    log("results json written")
    os._exit(0)


if __name__ == "__main__":
    main()
