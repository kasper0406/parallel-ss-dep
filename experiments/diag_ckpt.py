"""Checkpoint diagnostics: per-layer logit-lens, hidden effective rank,
per-source held-out CE.

Why this exists: when overall val PPL stops moving, we want to distinguish
"model is too small" from "model is not being asked to learn this" from
"depth/width is being wasted". The three diagnostics here cover the three
hypotheses:

- **Per-layer logit-lens CE**: project every block's residual stream through
  `out_norm + lm_head` and compute CE. A "dead zone" — consecutive layers
  where CE doesn't improve — means those layers aren't contributing.
- **Hidden effective rank**: SVD of per-layer hidden activations across
  positions. Eff-rank << d_model = residual stream is bandwidth-starved,
  some directions are wasted.
- **Per-source held-out CE**: bucket loss by mix source. Flat overall PPL
  can hide one source improving and another regressing.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/diag_ckpt.py \
        --ckpt checkpoints/pretrain_mix_v2_step69755_tok1000007680.pt \
        --data_mix configs/pretrain_mix_v2_with_cve.yaml \
        --device cuda:0 --batch 4 --T 1024 --n_chunks_per_source 12

Importable: `run_diag(ckpt_path, ...) -> DiagReport` for use from
`eval_callback.py`.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight ckpt loader (mirrors eval_bracket_structure but device-flex).

def _load_model(ckpt_path: str, device: str):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    # build_model_from_ckpt hardcodes cuda → we hot-swap device after load.
    model, cfg = build_model_from_ckpt(ckpt_path)
    model = model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Diag report container.

@dataclass
class DiagReport:
    overall_ce: float
    per_layer_lens_ce: list[float]
    per_layer_eff_rank: list[float]
    per_layer_resid_add: list[float] = field(default_factory=list)  # ||h[L]-h[L-1]||/||h[L-1]||
    per_layer_h_norm: list[float] = field(default_factory=list)     # mean ||h[L]||
    per_source_ce: dict[str, float] = field(default_factory=dict)
    n_tokens_scored: int = 0

    def to_lines(self, d_model: int | None = None) -> list[str]:
        out = [f"overall CE = {self.overall_ce:.4f} "
               f"(ppl {math.exp(self.overall_ce):.2f}, "
               f"n_tokens={self.n_tokens_scored})"]
        n = len(self.per_layer_lens_ce)
        if n:
            out.append("per-layer  logit-lens CE | effrank | ||h|| | Δh/||h_prev||")
            for L in range(n):
                er = self.per_layer_eff_rank[L]
                er_str = (f"{er:5.1f}/{d_model}" if d_model else f"{er:5.1f}")
                hn = self.per_layer_h_norm[L] if L < len(self.per_layer_h_norm) else float("nan")
                add = self.per_layer_resid_add[L] if L < len(self.per_layer_resid_add) else float("nan")
                out.append(f"  L{L:02d}  lensCE={self.per_layer_lens_ce[L]:6.3f}"
                           f"  effrank={er_str}"
                           f"  ||h||={hn:7.2f}"
                           f"  Δh/||h_prev||={add:6.3f}")
        if self.per_source_ce:
            out.append("per-source CE (lower = model fits that stream better):")
            for name, ce in sorted(self.per_source_ce.items(),
                                   key=lambda kv: kv[1]):
                out.append(f"  {name:24s}  CE={ce:.3f}  ppl={math.exp(ce):.2f}")
        return out


# ---------------------------------------------------------------------------
# Logit-lens + effective rank hooks.

class _LayerHooks:
    """Stash each block's output for later analysis. Hooks fire during a
    plain `model.forward()`; we then run `out_norm + lm_head` on each
    captured h to compute the per-layer logit-lens CE."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.captured: list[torch.Tensor | None] = [None] * len(model.blocks)
        self._handles = []
        for i, blk in enumerate(model.blocks):
            self._handles.append(blk.register_forward_hook(self._mk(i)))

    def _mk(self, i: int):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.captured[i] = t.detach()
        return hook

    def close(self):
        for h in self._handles:
            h.remove()


def _effective_rank(h: torch.Tensor, eps: float = 1e-9) -> float:
    """Participation ratio = (Σσ)² / Σσ². Cheaper and more interpretable
    than entropy of the singular-value distribution; equals d_model for
    isotropic activations, equals 1 for rank-1 collapse."""
    flat = h.reshape(-1, h.shape[-1])
    # Cap rows for tractable SVD on huge batches.
    if flat.shape[0] > 4096:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:4096]
        flat = flat[idx]
    flat = flat - flat.mean(0, keepdim=True)
    # Centre then SVD on small side.
    s = torch.linalg.svdvals(flat.float())
    s = s.clamp_min(0)
    num = s.sum().item()
    den = (s ** 2).sum().item() + eps
    return float(num * num / den)


@torch.no_grad()
def _logit_lens_ce(model, captured_h: torch.Tensor,
                    targets: torch.Tensor) -> float:
    """Project a layer's residual stream through the SAME exit-tail the
    final layer uses (`out_norm + lm_head`) and compute CE. Memory is
    skipped here — the read mask is empty for pretrain-style batches and
    its presence would just add noise.

    Lower than the final-layer CE = that layer already had enough signal
    to predict; higher = it didn't."""
    h = model.out_norm(captured_h)
    logits = model.lm_head(h)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            ignore_index=-100).item()


# ---------------------------------------------------------------------------
# Per-source CE via the existing MixedSourceStream (one source at a time).

def _per_source_iters(yaml_path: str, tokenizer, block_size: int,
                       thinking_token_id: int):
    """Yield (source_name, iterator) for each enabled source in the YAML.
    Each iterator produces `(inputs, targets)` chunks with the SAME
    think-burst + EOS-mask semantics the trainer uses, so the CE numbers
    are comparable to train/val loss."""
    from experiments.data_mix import load_sources_from_yaml, MixedSourceStream
    sources = load_sources_from_yaml(yaml_path)
    for src in sources:
        one = type(src)(**{**src.__dict__, "weight": 1.0})
        stream = MixedSourceStream(
            sources=[one], tokenizer=tokenizer, block_size=block_size,
            thinking_token_id=thinking_token_id,
            think_burst_prob=0.0,  # cleaner per-source signal
            base_seed=12345,
        )
        yield src.name, iter(stream)


@torch.no_grad()
def _ce_on_batch(model, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            ignore_index=-100).item()


# ---------------------------------------------------------------------------
# Top-level entry point.

def run_diag(ckpt_path: str, *,
              data_mix_yaml: str | None = None,
              tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M",
              device: str = "cuda:0",
              batch: int = 4, T: int = 1024,
              n_chunks_per_source: int = 12,
              ) -> DiagReport:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    model, cfg = _load_model(ckpt_path, device=device)
    d_model = int(cfg["d_model"])

    # 1) Per-layer logit-lens + effective rank on a single batch from the
    # FIRST source in the YAML (proxy for "natural" data; we just need
    # hidden states, the source isn't load-bearing here).
    per_layer_lens_ce: list[float] = []
    per_layer_eff_rank: list[float] = []
    overall_ce = float("nan")

    if data_mix_yaml is not None:
        thinking_token_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
        # Take one batch worth from the FIRST source for the lens pass.
        first_name, first_iter = next(_per_source_iters(
            data_mix_yaml, tok, T, thinking_token_id))
        xs, ys = [], []
        for _ in range(batch):
            x, y = next(first_iter)
            xs.append(x); ys.append(y)
        inputs = torch.stack(xs).to(device)
        targets = torch.stack(ys).to(device)
        hooks = _LayerHooks(model)
        per_layer_resid_add: list[float] = []
        per_layer_h_norm: list[float] = []
        try:
            logits = model(inputs)
            overall_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                          targets.reshape(-1),
                                          ignore_index=-100).item()
            prev = None
            for h in hooks.captured:
                per_layer_lens_ce.append(_logit_lens_ce(model, h, targets))
                per_layer_eff_rank.append(_effective_rank(h))
                hn = h.float().norm(dim=-1).mean().item()
                per_layer_h_norm.append(hn)
                if prev is None:
                    per_layer_resid_add.append(float("nan"))
                else:
                    delta = (h - prev).float().norm(dim=-1).mean().item()
                    prev_n = prev.float().norm(dim=-1).mean().item() + 1e-9
                    per_layer_resid_add.append(delta / prev_n)
                prev = h
        finally:
            hooks.close()

    # 2) Per-source CE — one fresh iterator per source for honest sampling.
    per_source_ce: dict[str, float] = {}
    n_tokens_scored = 0
    if data_mix_yaml is not None:
        for name, src_iter in _per_source_iters(
                data_mix_yaml, tok, T, thinking_token_id):
            ce_sum = 0.0
            ce_count = 0
            for _ in range(n_chunks_per_source // batch + 1):
                xs, ys = [], []
                try:
                    for _ in range(batch):
                        x, y = next(src_iter)
                        xs.append(x); ys.append(y)
                except StopIteration:
                    break
                if not xs:
                    break
                inputs = torch.stack(xs).to(device)
                targets = torch.stack(ys).to(device)
                ce = _ce_on_batch(model, inputs, targets)
                # Weight by # scored tokens to be exact.
                valid = (targets != -100).sum().item()
                ce_sum += ce * valid
                ce_count += valid
                n_tokens_scored += valid
                if ce_count >= n_chunks_per_source * T:
                    break
            if ce_count > 0:
                per_source_ce[name] = ce_sum / ce_count

    return DiagReport(
        overall_ce=overall_ce,
        per_layer_lens_ce=per_layer_lens_ce,
        per_layer_eff_rank=per_layer_eff_rank,
        per_layer_resid_add=per_layer_resid_add if data_mix_yaml else [],
        per_layer_h_norm=per_layer_h_norm if data_mix_yaml else [],
        per_source_ce=per_source_ce,
        n_tokens_scored=n_tokens_scored,
    )


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_mix", default=None,
                    help="Optional: YAML for per-source CE + logit-lens.")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--n_chunks_per_source", type=int, default=12)
    return ap.parse_args()


def main():
    args = _parse_args()
    print(f"[diag] ckpt={args.ckpt}")
    print(f"[diag] device={args.device}  batch={args.batch}  T={args.T}")
    report = run_diag(
        args.ckpt, data_mix_yaml=args.data_mix,
        tokenizer_name=args.tokenizer, device=args.device,
        batch=args.batch, T=args.T,
        n_chunks_per_source=args.n_chunks_per_source,
    )
    import torch as _t
    ckpt = _t.load(args.ckpt, map_location="cpu", weights_only=False)
    d_model = ckpt["config"].get("d_model")
    print()
    print("=" * 70)
    print("CHECKPOINT DIAGNOSTICS")
    print("=" * 70)
    for line in report.to_lines(d_model=d_model):
        print(line)


if __name__ == "__main__":
    main()
