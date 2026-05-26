"""
Bracket / structure quality eval on generated code.

Phase 1 of the post-PPL eval pipeline: tests whether a model produces
*structurally valid* code completions, independent of next-token PPL.

Metrics (all per generation):
  - parse_success: prefix + generation parses as Python
  - bracket_imbalance: |#open - #close| over the generation only
  - indent_consistency: generated tokens preserve indentation patterns
  - n_unique_idents: heuristic for "did the model produce a single
    repeating token vs varied identifiers" (sanity check)

Usage:
  python experiments/eval_bracket_structure.py \\
      --ckpt /path/to/model.pt \\
      --n_samples 200 --gen_len 64
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import (
    DeltaNetAttention, DeltaNetNegEigAttention,
    OrthogonalScanAttention, SymbolGroundedAttention,
    HeisenbergAttention, MultiPassAttention,
)
from experiments.model import TinyLM
from experiments.train_lm import build_arch, parse_layers_arg


def build_model_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("layers_spec"):
        cls_list = parse_layers_arg(cfg["layers_spec"])
        attn_kw = dict(attention_cls_per_layer=cls_list)
    elif cfg.get("arch"):
        attn_kw = build_arch(cfg["arch"], cfg["n_layers"])
    else:
        raise ValueError("checkpoint has neither arch nor layers_spec")
    # Linear-RNN architectures train with max_T=0 (no pos embed); the
    # Transformer baseline trains with max_T=args.T (learnable abs pos
    # embed). Detect from the saved state_dict whether the ckpt has a
    # pos_embed and infer the correct max_T.
    sd_keys = ckpt["state_dict"].keys() if "state_dict" in ckpt else {}
    if "pos_embed.weight" in sd_keys:
        max_T_inferred = ckpt["state_dict"]["pos_embed.weight"].shape[0]
    else:
        max_T_inferred = 0
    # Auto-detect optional heads from the state_dict so we build the model
    # with the right kwargs and avoid strict-load mismatches.
    has_gate = any(k.startswith("gate_head.") for k in sd_keys)
    has_memory = any(k.startswith("memory.") for k in sd_keys)
    has_pkm = any(k.startswith("pkm_layer.") for k in sd_keys)
    mem_kwargs = {}
    if has_memory:
        # Infer mem_dim from W_proj's input dimension and mem_size has no
        # state-dict footprint (it's compile-time). Default to 1024.
        mem_dim_inferred = ckpt["state_dict"]["memory.W_proj.weight"].shape[1]
        mem_kwargs = dict(
            use_memory=True,
            mem_dim=int(mem_dim_inferred),
            mem_size=int(cfg.get("mem_size", 1024)),
            thinking_token_id=int(cfg.get("thinking_token_id",
                                          cfg["vocab_size"] - 1)),
            # FIX A (added 2026-05-18): the write-only-at-think flag isn't
            # in state_dict (it's a plain bool attribute), so when a
            # WorkingMemory is reconstructed for eval/inference we must
            # read the flag from cfg and pass it to __init__. Without
            # this, a model trained with mem_write_only_at_think=True
            # gets evaluated with write_only_at_think=False, silently
            # losing the architectural mask. Default False = backward
            # compat for older ckpts.
            mem_write_only_at_think=bool(cfg.get("mem_write_only_at_think",
                                                  False)),
        )
    pkm_kwargs = {}
    if has_pkm:
        # Infer all shape-derived PKM hyperparams from state dict.
        # subkeys: (n_heads, 2, n_keys, k_dim)
        sk = ckpt["state_dict"]["pkm_layer.subkeys"]
        n_heads_pkm, _, n_keys, k_dim = sk.shape
        # Count value-table heads (one nn.Embedding per head).
        n_head_tables = sum(
            1 for k in sd_keys
            if k.startswith("pkm_layer.values.") and k.endswith(".weight")
        )
        assert n_head_tables == n_heads_pkm, (
            f"PKM subkeys say n_heads={n_heads_pkm} but found "
            f"{n_head_tables} value tables")
        # value_bf16 from the dtype of values.0.weight.
        v0_dtype = ckpt["state_dict"]["pkm_layer.values.0.weight"].dtype
        # top_k and pkm_after_layer have no state-dict footprint; use cfg or
        # the canonical defaults (32, 14) the v5-pkm run used.
        # v7 PKM-bootstrap-fix package: score_norm + value_init_std +
        # use_output_gate. Auto-detect from state dict for back-compat
        # with v5/v6 ckpts; cfg overrides the auto-detect when present.
        if "pkm_layer.out_alpha" in sd_keys:
            pkm_use_output_gate = True
        else:
            pkm_use_output_gate = bool(cfg.get("pkm_use_output_gate", False))
        if "pkm_layer.bn_s1.weight" in sd_keys:
            pkm_score_norm = "batch"
        elif "pkm_layer.ln_s1.weight" in sd_keys:
            pkm_score_norm = "layer"
        else:
            pkm_score_norm = str(cfg.get("pkm_score_norm", "batch"))
        pkm_kwargs = dict(
            use_pkm=True,
            pkm_after_layer=int(cfg.get("pkm_after_layer", 14)),
            pkm_n_keys=int(n_keys),
            pkm_n_heads=int(n_heads_pkm),
            pkm_k_dim=int(k_dim),
            pkm_top_k=int(cfg.get("pkm_top_k", 32)),
            pkm_value_bf16=(v0_dtype == torch.bfloat16),
            pkm_score_norm=pkm_score_norm,
            pkm_value_init_std=float(cfg.get("pkm_value_init_std", 1.0)),
            pkm_use_output_gate=pkm_use_output_gate,
        )
    # Phase-2 (state_readonly_at_think) and Phase-3 (think_index_emb_size)
    # are plain init kwargs on TinyLM with no state-dict footprint when
    # zero. Read from cfg so a ckpt trained with them stays consistent
    # when reloaded for eval/SFT/RL; default off for back-compat.
    sr_at_think = bool(cfg.get("state_readonly_at_think", False))
    think_idx_emb = int(cfg.get("think_index_emb_size", 0))
    # Auto-detect think_index_emb from state dict when not in cfg (older
    # ckpts that predate the cfg key but were trained with it on).
    if think_idx_emb == 0 and "think_index_emb.weight" in sd_keys:
        think_idx_emb = int(ckpt["state_dict"]["think_index_emb.weight"].shape[0])
    model = TinyLM(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        d_head=cfg["d_head"], max_T=max_T_inferred,
        feedback_mode=cfg.get("feedback_mode", "none"),
        feedback_distances=tuple(cfg.get("feedback_distances", (1,))),
        feedback_pairs=tuple(cfg.get("feedback_pairs", ()) or ()),
        feedback_self_k=int(cfg.get("feedback_self_k", 0)),
        output_gate=bool(has_gate),
        state_readonly_at_think=sr_at_think,
        think_index_emb_size=think_idx_emb,
        **mem_kwargs,
        **pkm_kwargs,
        **attn_kw,
    )
    # Backward-compat: old single-scale film checkpoints saved keys as
    # `feedback.L.alpha` etc. (no MultiScale wrapper). Remap to the new
    # nested form `feedback.L.projs.0.alpha`.
    sd = ckpt["state_dict"]
    if any(k.startswith("feedback.") and ".projs." not in k for k in sd):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("feedback.") and ".projs." not in k:
                parts = k.split(".", 2)
                new_k = f"{parts[0]}.{parts[1]}.projs.0.{parts[2]}"
                new_sd[new_k] = v
            else:
                new_sd[k] = v
        sd = new_sd
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda").eval()
    return model, cfg


def _make_val_iter(tokenizer_name: str, dataset: str, text_field: str,
                   T_prefix: int, batch: int = 8):
    """Build a streaming iterator yielding (B, T_prefix) prefix batches."""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    val_stream = load_dataset(dataset, split="train", streaming=True
                              ).shuffle(seed=42).skip(20_000)
    buf: list[int] = []
    eos = tok.eos_token_id or tok.bos_token_id or 0
    for example in val_stream:
        text = example[text_field]
        ids = tok.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= batch * T_prefix:
            chunk = buf[: batch * T_prefix]
            buf = buf[batch * T_prefix:]
            yield torch.tensor(chunk, dtype=torch.long).view(batch, T_prefix), tok


@torch.no_grad()
def greedy_generate(model, prefix_ids: torch.Tensor, gen_len: int) -> torch.Tensor:
    """Generate gen_len tokens greedily by re-running full forward each step.

    Slow but simple. For our small models (~100-250M, T <= 512) this is OK.
    """
    out = prefix_ids.clone()
    for _ in range(gen_len):
        logits = model(out)                      # (B, T_so_far, V)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
    return out


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _indent_consistency(text: str) -> float:
    """Fraction of indents that are multiples of 4 (or 0)."""
    lines = text.split("\n")
    indents = [_line_indent(L) for L in lines if L.strip()]
    if not indents:
        return 1.0
    return sum(1 for i in indents if i % 4 == 0) / len(indents)


def evaluate(ckpt_path: str, n_samples: int = 200, gen_len: int = 64,
             T_prefix: int = 256, batch: int = 8,
             dataset: str = "codeparrot/codeparrot-clean",
             text_field: str = "content"):
    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    print(f"  config: feedback={cfg.get('feedback_mode')}, "
          f"n_layers={cfg['n_layers']}, d_model={cfg['d_model']}")

    tok_name = cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M")
    val_iter = _make_val_iter(tok_name, dataset, text_field, T_prefix, batch)

    parse_success = 0
    bracket_imbalance = []
    indent_consistencies = []
    n_unique_ratio = []
    total = 0

    for prefix_ids, tokenizer in val_iter:
        prefix_ids = prefix_ids.to("cuda")
        # Generate
        gen_full = greedy_generate(model, prefix_ids, gen_len)
        gen_only_ids = gen_full[:, T_prefix:]
        prefix_only_ids = gen_full[:, :T_prefix]

        for b in range(prefix_ids.shape[0]):
            prefix_text = tokenizer.decode(prefix_only_ids[b].cpu().tolist(),
                                           skip_special_tokens=True)
            gen_text = tokenizer.decode(gen_only_ids[b].cpu().tolist(),
                                        skip_special_tokens=True)
            full_text = prefix_text + gen_text

            # parse_success: full text parses as Python
            try:
                ast.parse(full_text)
                parse_success += 1
            except (SyntaxError, ValueError):
                pass

            # bracket imbalance over generation
            opens = sum(gen_text.count(c) for c in "({[")
            closes = sum(gen_text.count(c) for c in ")}]")
            bracket_imbalance.append(abs(opens - closes))

            # indent consistency over generation
            indent_consistencies.append(_indent_consistency(gen_text))

            # token diversity
            gen_tokens = gen_only_ids[b].cpu().tolist()
            n_unique_ratio.append(len(set(gen_tokens)) / max(1, len(gen_tokens)))

            total += 1

        if total >= n_samples:
            break

    results = {
        "n_samples": total,
        "parse_success_rate": parse_success / total,
        "mean_bracket_imbalance": sum(bracket_imbalance) / total,
        "indent_consistency_rate": sum(indent_consistencies) / total,
        "mean_token_diversity": sum(n_unique_ratio) / total,
    }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, action="append",
                   help="path to checkpoint (can be repeated for comparison)")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--gen_len", type=int, default=64)
    p.add_argument("--T_prefix", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean")
    p.add_argument("--text_field", type=str, default="content")
    args = p.parse_args()

    all_results = {}
    for ckpt in args.ckpt:
        print(f"\n{'=' * 70}\nEvaluating: {ckpt}\n{'=' * 70}")
        results = evaluate(ckpt, n_samples=args.n_samples, gen_len=args.gen_len,
                           T_prefix=args.T_prefix, batch=args.batch,
                           dataset=args.dataset, text_field=args.text_field)
        all_results[ckpt] = results
        print(f"\nResults for {pathlib.Path(ckpt).name}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ckpt':<60} {'parse%':>8} {'brack_imb':>10} {'indent%':>9} {'tok_div':>8}")
    for ckpt, r in all_results.items():
        name = pathlib.Path(ckpt).stem
        print(f"{name:<60} {r['parse_success_rate']*100:>7.1f}% "
              f"{r['mean_bracket_imbalance']:>10.2f} "
              f"{r['indent_consistency_rate']*100:>8.1f}% "
              f"{r['mean_token_diversity']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
