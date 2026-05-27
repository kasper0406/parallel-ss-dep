"""Per-source CE comparison probe for v7.1 PKM analysis.

Two questions answered:

 1. Where does PKM HELP vs HURT?
    For each data-mix source, forward a held-out batch through the
    v7.1-pkm-film FINAL ckpt with α at its trained value, then again
    with α forced to 0. Per-source CE delta reveals the niches.

 2. Apples-to-apples vs v6-shallow (no PKM)?
    Forward the same batches through the v6-shallow step-2180 ckpt vs
    v7.1-pkm-film step-2180 ckpt (same token budget, 500M).

Sources (from configs/pretrain_mix_v4.yaml): codeparrot, python_codes_25k,
magicoder_oss, code_exercises_jinaai, python_instr_18k_alpaca,
codealpaca_20k, bigvul, cybernative_vuln_dpo, textbooks_lite_sciphi,
textbook_quality_programming, wikipedia_en.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/probe_pkm_per_source.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import yaml

from experiments.eval_bracket_structure import build_model_from_ckpt


SOURCES = [
    "codeparrot",
    "python_codes_25k",
    "magicoder_oss",
    "code_exercises_jinaai",
    "python_instr_18k_alpaca",
    "codealpaca_20k",
    "bigvul",
    "cybernative_vuln_dpo",
    "textbooks_lite_sciphi",
    "textbook_quality_programming",
    "wikipedia_en",
]


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_single_source_loader(yaml_path: str, source_name: str,
                                 tokenizer_name: str, block_size: int,
                                 batch_size: int, seed: int,
                                 thinking_token_id: int):
    """Return a one-shot iterable that yields a single (x, y, doc_ids)
    batch from one specific source. Constructs a 1-element
    MixedSourceStream with weight=1 to reuse the existing pipeline."""
    from experiments.data_mix import (
        MixedSourceStream, load_sources_from_yaml,
    )
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    cfg = _load_yaml(yaml_path)
    sources_cfg = cfg["sources"]
    src_dict = next((s for s in sources_cfg if s["name"] == source_name), None)
    if src_dict is None:
        raise ValueError(f"source '{source_name}' not in {yaml_path}")
    tmp_yaml = pathlib.Path(f"/tmp/probe_single_{source_name}.yaml")
    tmp_yaml.write_text(yaml.safe_dump(
        {"sources": [{**src_dict, "weight": 1.0}]}
    ))
    sources = load_sources_from_yaml(str(tmp_yaml))
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=block_size,
        thinking_token_id=thinking_token_id,
        think_burst_prob=0.0,  # clean held-out distribution, no think bursts
        base_seed=seed,
        mask_eos_in_targets=True,
        emit_doc_ids=True,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


def _ce_on_batch(model, x, y, doc_ids, use_pkm: bool):
    """Forward (under bf16 autocast) and return per-token CE."""
    pkm = getattr(model, "pkm_layer", None)
    α_save = None
    if pkm is not None and pkm.use_output_gate:
        α_save = pkm.out_alpha.detach().clone()
        if not use_pkm:
            with torch.no_grad():
                pkm.out_alpha.zero_()
            # also kill any leftover floor curriculum from training
            pkm.alpha_floor = 0.0
    try:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, doc_ids=doc_ids).float()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1).clamp_min(0),
            reduction="none",
        ).reshape(y.shape)
        valid = (y != -100).float()
        return float((ce * valid).sum() / valid.sum().clamp_min(1.0))
    finally:
        if α_save is not None and not use_pkm:
            with torch.no_grad():
                pkm.out_alpha.copy_(α_save)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", type=str,
                   default="configs/pretrain_mix_v4.yaml")
    p.add_argument("--tokenizer", type=str,
                   default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=123,
                   help="Fixed across all forwards so the same tokens are "
                        "scored by every (model, α) combination.")
    p.add_argument("--ckpts", type=str, nargs="+", default=[
        "checkpoints/pretrain_mix_v7_pkm_film.pt",        # final, PKM-on/off
        "checkpoints/pretrain_mix_v7_pkm_film_step2180_tok500039680.pt",
        "checkpoints/pretrain_mix_v6_shallow_step2180_tok500039680.pt",
        "checkpoints/pretrain_mix_v4.pt",                  # final, no PKM
    ])
    args = p.parse_args()

    print(f"Probe config: block={args.block_size} batch={args.batch_size} "
          f"sources={len(SOURCES)} (one held-out batch each)\n")

    # ------------------------------------------------------------------
    # Step 1. Build a fixed (x, y, doc_ids) batch PER SOURCE so every model
    # is scored on the exact same tokens. Done before loading any model.
    # ------------------------------------------------------------------
    # Need a thinking_token_id placeholder for MixedSourceStream — use the
    # tokenizer vocab size + 0 (any unused id is fine, no bursts injected).
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.tokenizer)
    thinking_token_id = len(_tok)
    print("Building held-out batches (1 per source) ...")
    src_batches = {}
    for src in SOURCES:
        try:
            loader = _build_single_source_loader(
                args.yaml, src, args.tokenizer, args.block_size,
                args.batch_size, args.seed,
                thinking_token_id=thinking_token_id,
            )
            x, y, doc_ids = next(iter(loader))
            src_batches[src] = (x.cuda(), y.cuda(), doc_ids.cuda())
            n_valid = int((y != -100).sum())
            print(f"  {src:<32} valid tokens = {n_valid}")
        except Exception as e:
            print(f"  {src:<32} ERROR: {e}")

    # ------------------------------------------------------------------
    # Step 2. For each ckpt, compute per-source CE. For v7.1-pkm ckpts
    # also compute CE with α=0 (PKM disabled).
    # ------------------------------------------------------------------
    results: dict[str, dict[str, float]] = {}
    for ckpt_path in args.ckpts:
        if not pathlib.Path(ckpt_path).exists():
            print(f"\nSKIP {ckpt_path} (missing)")
            continue
        print(f"\nLoading {ckpt_path} ...")
        model, cfg = build_model_from_ckpt(ckpt_path)
        model = model.cuda().eval()
        has_pkm = hasattr(model, "pkm_layer")
        label = pathlib.Path(ckpt_path).stem
        if has_pkm and model.pkm_layer.use_output_gate:
            print(f"  PKM on. α_trained = {float(model.pkm_layer.out_alpha.detach()):+.4f}")
            for tag in ["pkm_on", "pkm_off"]:
                use_pkm = (tag == "pkm_on")
                key = f"{label}::{tag}"
                results[key] = {}
                for src, (x, y, doc_ids) in src_batches.items():
                    ce = _ce_on_batch(model, x, y, doc_ids, use_pkm=use_pkm)
                    results[key][src] = ce
        else:
            key = label
            results[key] = {}
            for src, (x, y, doc_ids) in src_batches.items():
                ce = _ce_on_batch(model, x, y, doc_ids, use_pkm=False)
                results[key][src] = ce
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3. Pretty-print.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'source':<32} " + " ".join(f"{k[-25:]:>14}" for k in results.keys()))
    print("-" * 100)
    for src in SOURCES:
        if src not in src_batches:
            continue
        row = f"{src:<32} "
        for k in results.keys():
            ce = results[k].get(src, float("nan"))
            row += f"{ce:>14.4f}"
        print(row)
    print("=" * 100)

    # PKM contribution Δ on the v7.1-film final ckpt (if both tags present).
    final_on = next((k for k in results if "pkm_film.pt::pkm_on" in k or
                     (k.endswith("::pkm_on") and "step2180" not in k and "step" not in k)), None)
    final_off = next((k for k in results if "pkm_film.pt::pkm_off" in k or
                      (k.endswith("::pkm_off") and "step2180" not in k and "step" not in k)), None)
    if final_on is None:
        final_on = next((k for k in results if k.endswith("::pkm_on")), None)
    if final_off is None:
        final_off = next((k for k in results if k.endswith("::pkm_off")), None)
    if final_on and final_off:
        print(f"\n=== PKM contribution per source (Δ = CE_off - CE_on; +ve means PKM HELPS) ===")
        deltas = []
        for src in SOURCES:
            if src in results[final_on]:
                d = results[final_off][src] - results[final_on][src]
                deltas.append((src, d))
        for src, d in sorted(deltas, key=lambda t: -t[1]):
            tag = "PKM HELPS" if d > 0.01 else ("PKM HURTS" if d < -0.01 else "neutral  ")
            print(f"  {src:<32} Δ = {d:+.4f}  {tag}")


if __name__ == "__main__":
    sys.exit(main())
