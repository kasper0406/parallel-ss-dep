"""Short fresh-from-scratch LR sweep on the REAL trunk, for the batch/LR study.

Validates the noise-scale LR recommendation empirically: trains the production
trunk (10L x d896 x 14h x d64, FiLM(0,5..4,9) K=3) from scratch on
configs/pretrain_mix_v4.yaml for a few hundred steps under Muon(matrix)+AdamW
(rest), at a chosen (lr_muon, lr_adamw, batch), and reports plain-LM VAL CE on a
FIXED held-out pool (identical across runs so only the LR/batch differs).

Deliberately STRIPPED to isolate the optimizer LR-vs-batch effect: no memory /
PKM / latent-reasoning / gist / gate aux (those add their own gradient noise and
would confound the trunk-LR comparison). Matrix params still go to Muon and
embed/lm_head/1D to AdamW exactly as production. No checkpoints saved.

GPU 1 ONLY. Reuses build_parser/build_model_from_args/build_optimizer.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.train_lm_args import build_parser
from experiments.model_builder import build_model_from_args
from experiments.optim_utils import build_optimizer
from experiments.speed_knobs import apply_speed_knobs
from experiments.data_mix import MixedSourceStream, load_sources_from_yaml


def build_trunk_args(steps, warmup):
    a = build_parser().parse_args([])
    a.arch = "deltanet"
    a.d_model = 896
    a.n_layers = 10
    a.n_heads = 14
    a.d_head = 64
    a.feedback = "film"
    a.feedback_pairs = "0,5;1,6;2,7;3,8;4,9"
    a.feedback_self_k = 3
    a.feedback_self_k_warmup_steps = 0        # K=3 active (the bulk-training regime)
    a.output_gate = False                      # stripped: isolate trunk LR
    a.enable_thinking_token = False
    a.use_memory = False
    a.use_pkm = False
    a.aux_brackets = False
    a.activation_checkpointing = True
    a.max_T = 2048
    a.load_ckpt = None
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr_muon", type=float, required=True)
    p.add_argument("--lr_adamw", type=float, required=True)
    p.add_argument("--batch", type=int, default=4)         # microbatch seqs
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--config", default="configs/pretrain_mix_v4.yaml")
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--val_pool", default="runs/noise_scale/pool_v4.pt")
    p.add_argument("--val_n", type=int, default=64)
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--base_seed", type=int, default=777)
    p.add_argument("--z_loss", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tag", default="sweep")
    args = p.parse_args()

    eff_tok = args.batch * args.grad_accum * args.T
    print(f"[{args.tag}] lr_muon={args.lr_muon:.2e} lr_adamw={args.lr_adamw:.2e} "
          f"batch={args.batch}x{args.grad_accum} ({eff_tok} tok/step) "
          f"steps={args.steps}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    base_vocab = tok.vocab_size
    vocab_with_think = ((base_vocab + 1 + 63) // 64) * 64
    thinking_token_id = base_vocab

    targs = build_trunk_args(args.steps, args.warmup)
    targs.T = args.T
    model, _info = build_model_from_args(
        targs, vocab_size=vocab_with_think, thinking_token_id=thinking_token_id)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_par/1e6:.1f}M", flush=True)

    opts, scheds = build_optimizer(
        model, optimizer="muon", lr=args.lr_adamw, lr_muon=args.lr_muon,
        alpha_wd=0.0, steps=args.steps, wd=0.01, lr_schedule="wsd",
        warmup_steps=args.warmup, decay_frac=0.15, bf16_optim_state=True,
        verbose=True)
    apply_speed_knobs(model, bf16=True, tf32=True, compile_model=False)

    # Fixed val pool (identical across every LR run).
    vp = torch.load(args.val_pool, map_location="cpu", weights_only=False)
    vN = min(args.val_n, vp["inputs"].shape[0])
    v_in = vp["inputs"][-vN:]
    v_y = vp["targets"][-vN:]
    v_doc = vp["doc_ids"][-vN:]
    print(f"  val pool: last {vN} seqs of {args.val_pool}", flush=True)

    # Train stream (plain LM, no think bursts).
    sources = load_sources_from_yaml(args.config)
    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.T,
        thinking_token_id=thinking_token_id, think_burst_prob=0.0,
        base_seed=args.base_seed, mask_eos_in_targets=True, emit_doc_ids=True)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=1)
    it = iter(loader)

    def run_val():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, vN, args.batch):
                x = v_in[i:i+args.batch].to("cuda")
                y = v_y[i:i+args.batch].to("cuda")
                d = v_doc[i:i+args.batch].to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, doc_ids=d)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                       y.reshape(-1), ignore_index=-100)
                tot += loss.item() * x.numel()
                n += x.numel()
        model.train()
        return tot / n

    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        for o in opts:
            o.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _m in range(args.grad_accum):
            xb, yb, db = next(it)
            xb, yb, db = xb.to("cuda"), yb.to("cuda"), db.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb, doc_ids=db)
            logits = out[0] if isinstance(out, tuple) else out
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                 yb.reshape(-1), ignore_index=-100)
            zl = args.z_loss * (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
            loss = (ce + zl) / args.grad_accum
            loss.backward()
            acc_loss += float(ce.detach()) / args.grad_accum
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            print(f"  step {step}/{args.steps}  train_ce={acc_loss:.4f}  "
                  f"({dt/step:.2f}s/step)", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            vce = run_val()
            history.append((step, vce))
            print(f"    >>> VAL ce={vce:.4f} ppl={torch.tensor(vce).exp():.2f} "
                  f"@ step {step}", flush=True)

    print(f"[{args.tag}] FINAL VAL ce={history[-1][1]:.4f} "
          f"(lr_muon={args.lr_muon:.2e} lr_adamw={args.lr_adamw:.2e} "
          f"batch={eff_tok}tok)", flush=True)
    print(f"[{args.tag}] history: " +
          " ".join(f"{s}:{v:.4f}" for s, v in history), flush=True)


if __name__ == "__main__":
    main()
