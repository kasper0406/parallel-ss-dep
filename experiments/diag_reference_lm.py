"""Logit-lens / effective-rank diagnostics for a HuggingFace causal LM
(e.g. SmolLM2-135M). Same metrics as `diag_ckpt.py` but adapted to the HF
LlamaDecoderLayer structure so we can compare a well-trained 135 M
reference against our 217 M v2 ckpt at matched architecture
(SmolLM2-135M is 30L × 576d × 9h — same shape as our backbone).

This decides whether the dead-depth pattern we see in our v2 ckpt is
(a) "normal for tied-embedding 30-layer decoders at this scale" or
(b) "specific to this run — depth genuinely being wasted".

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
    .venv/bin/python experiments/diag_reference_lm.py \\
        --model HuggingFaceTB/SmolLM2-135M \\
        --data_mix configs/pretrain_mix_v2_with_cve.yaml \\
        --batch 4 --T 1024
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from experiments.diag_ckpt import _effective_rank


@torch.no_grad()
def _lens_ce_hf(final_norm, lm_head, h, targets):
    h = final_norm(h)
    logits = lm_head(h)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1), ignore_index=-100).item()


class _Hooks:
    def __init__(self, layers):
        self.captured = [None] * len(layers)
        self._h = []
        for i, l in enumerate(layers):
            self._h.append(l.register_forward_hook(self._mk(i)))

    def _mk(self, i):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            self.captured[i] = t.detach()
        return hook

    def close(self):
        for h in self._h:
            h.remove()


@torch.no_grad()
def _per_source_ce(model, tokenizer, yaml_path, T, batch, n_chunks_per_source):
    from experiments.data_mix import load_sources_from_yaml, MixedSourceStream
    sources = load_sources_from_yaml(yaml_path)
    out: dict[str, float] = {}
    device = next(model.parameters()).device
    for src in sources:
        one = type(src)(**{**src.__dict__, "weight": 1.0})
        stream = MixedSourceStream(
            sources=[one], tokenizer=tokenizer, block_size=T,
            thinking_token_id=None, think_burst_prob=0.0, base_seed=12345,
        )
        it = iter(stream)
        ce_sum, n_valid = 0.0, 0
        for _ in range(max(1, n_chunks_per_source // batch + 1)):
            xs, ys = [], []
            try:
                for _ in range(batch):
                    x, y = next(it)
                    xs.append(x); ys.append(y)
            except StopIteration:
                break
            if not xs:
                break
            inp = torch.stack(xs).to(device)
            tgt = torch.stack(ys).to(device)
            logits = model(inp).logits
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  tgt.reshape(-1), ignore_index=-100).item()
            valid = (tgt != -100).sum().item()
            ce_sum += ce * valid
            n_valid += valid
            if n_valid >= n_chunks_per_source * T:
                break
        if n_valid > 0:
            out[src.name] = ce_sum / n_valid
    return out


def _batch_from_yaml(yaml_path, tokenizer, T, batch):
    from experiments.data_mix import load_sources_from_yaml, MixedSourceStream
    sources = load_sources_from_yaml(yaml_path)
    # Use just the first source for the lens pass — matches the trainer.
    first = type(sources[0])(**{**sources[0].__dict__, "weight": 1.0})
    stream = MixedSourceStream(
        sources=[first], tokenizer=tokenizer, block_size=T,
        thinking_token_id=None,  # reference model has no think token
        think_burst_prob=0.0, base_seed=12345,
    )
    it = iter(stream)
    xs, ys = [], []
    for _ in range(batch):
        x, y = next(it)
        xs.append(x); ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--data_mix", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per_source", action="store_true")
    ap.add_argument("--n_chunks_per_source", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    inputs, targets = _batch_from_yaml(args.data_mix, tok, args.T, args.batch)
    inputs = inputs.to(args.device); targets = targets.to(args.device)

    layers = model.model.layers
    final_norm = model.model.norm
    lm_head = model.lm_head
    n_layers = len(layers)
    d_model = model.config.hidden_size

    hooks = _Hooks(layers)
    try:
        logits = model(inputs).logits
        overall_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      targets.reshape(-1),
                                      ignore_index=-100).item()
    finally:
        pass

    print(f"\n=== {args.model}  ({n_layers}L × {d_model}d)  ===")
    print(f"overall CE = {overall_ce:.4f}  ppl = {math.exp(overall_ce):.2f}")
    print(f"per-layer  logit-lens CE | effrank | ||h|| | Δh/||h_prev||")
    prev = None
    for L, h in enumerate(hooks.captured):
        ce = _lens_ce_hf(final_norm, lm_head, h, targets)
        er = _effective_rank(h)
        hn = h.float().norm(dim=-1).mean().item()
        if prev is None:
            add = float("nan")
        else:
            d = (h - prev).float().norm(dim=-1).mean().item()
            add = d / (prev.float().norm(dim=-1).mean().item() + 1e-9)
        prev = h
        print(f"  L{L:02d}  lensCE={ce:6.3f}  effrank={er:5.1f}/{d_model}"
              f"  ||h||={hn:7.2f}  Δh/||h_prev||={add:6.3f}")
    hooks.close()

    if args.per_source:
        print("per-source CE (same eval as diag_ckpt — comparable to our ckpts):")
        psc = _per_source_ce(model, tok, args.data_mix, args.T, args.batch,
                              args.n_chunks_per_source)
        for name, ce in sorted(psc.items(), key=lambda kv: kv[1]):
            print(f"  {name:24s}  CE={ce:.3f}  ppl={math.exp(ce):.2f}")


if __name__ == "__main__":
    main()
