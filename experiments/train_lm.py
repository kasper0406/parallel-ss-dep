"""
Minimal language modelling training driver — practical demonstration.

The point: show that the hybrid scaffold trains stably on real text at
~135M-param scale and reaches competitive perplexity vs pure DeltaNet.

Uses TinyStories (HuggingFaceH4/TinyStories, ~500MB of children's
stories) tokenised with SmolLM2's tokeniser (49152 vocab). This is the
smallest realistic LM benchmark; if hybrid can't match DeltaNet here,
distillation is futile.

Usage:
    python experiments/train_lm.py --arch deltanet --steps 5000
    python experiments/train_lm.py --arch hybrid   --steps 5000
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from experiments.layers import (
    DeltaNetAttention, DeltaNetNegEigAttention, GatedDeltaNetAttention,
    DeltaNetForgetGateAttention,
    GatedDeltaProductAttention,
    OrthogonalScanAttention, SymbolGroundedAttention,
    HeisenbergAttention, MultiPassAttention,
    SoftmaxAttention, Mamba2Attention,
)
from experiments.model import TinyLM
from experiments.aux_brackets import compute_bracket_deltas, bracket_depth


_NAME_TO_CLS = {
    "deltanet":   DeltaNetAttention,
    "deltanet_negeig": DeltaNetNegEigAttention,
    "deltanet_forgetgate": DeltaNetForgetGateAttention,
    "gated_deltanet": GatedDeltaNetAttention,
    "gated_deltaproduct": GatedDeltaProductAttention,
    "ortho":      OrthogonalScanAttention,
    "transformer": SoftmaxAttention,
    "mamba2":     Mamba2Attention,
}


def _make_sg_factory(vocab_size: int, n_symbols: int):
    """Factory closure for SymbolGroundedAttention with vocab params bound."""
    def _f(**kw):
        return SymbolGroundedAttention(vocab_size=vocab_size, n_symbols=n_symbols, **kw)
    return _f


def build_arch(name: str, n_layers: int, vocab_size: int = 0, n_symbols: int = 512):
    """Map an arch name to either a single attention class or a per-layer list.

    For shorthand patterns:
      hybrid        — alternating ortho/deltanet (50/50, ortho first).
      hybrid_25_75  — 1 ortho + 3 deltanet, repeating.
      hybrid_75_25  — 3 ortho + 1 deltanet, repeating.
    Or use --layers for an explicit comma-separated list.
    """
    if name in _NAME_TO_CLS:
        return dict(attention_cls=_NAME_TO_CLS[name])
    if name == "hybrid":
        cls = [OrthogonalScanAttention if i % 2 == 0 else DeltaNetAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_25_75":
        # 1 ortho + 3 deltanet pattern, ortho at every 4th position.
        cls = [OrthogonalScanAttention if i % 4 == 0 else DeltaNetAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_75_25":
        # 3 ortho + 1 deltanet pattern.
        cls = [DeltaNetAttention if i % 4 == 0 else OrthogonalScanAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_negeig":
        cls = [OrthogonalScanAttention if i % 2 == 0 else DeltaNetNegEigAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    # SymbolGrounded variants: need vocab_size + n_symbols bound at build time.
    if name == "symgrounded":
        sg = _make_sg_factory(vocab_size, n_symbols)
        return dict(attention_cls=sg)
    if name == "hybrid_sg":
        # 50/50 alternating SymGrounded + DeltaNet (sg first).
        sg = _make_sg_factory(vocab_size, n_symbols)
        cls = [sg if i % 2 == 0 else DeltaNetAttention for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_sg_25_75":
        # 1 sg + 3 delta pattern (sparse symbolic).
        sg = _make_sg_factory(vocab_size, n_symbols)
        cls = [sg if i % 4 == 0 else DeltaNetAttention for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    # Multi-pass — direction B. K cells in parallel within each layer.
    if name == "multipass_dh":
        # DeltaNet + Heisenberg in parallel at every layer.
        def _mp(**kw):
            return MultiPassAttention(
                cells=[DeltaNetAttention, HeisenbergAttention], **kw
            )
        return dict(attention_cls=_mp)
    if name == "multipass_dho":
        # DeltaNet + Heisenberg + Ortho in parallel.
        def _mp(**kw):
            return MultiPassAttention(
                cells=[DeltaNetAttention, HeisenbergAttention,
                       OrthogonalScanAttention], **kw
            )
        return dict(attention_cls=_mp)
    if name == "multipass_dd":
        # Two strong LM cells: DeltaNet + DeltaNet_negeig (allow_neg_eigval).
        # Tests whether multi-pass helps when both cells are competent LMs.
        def _mp(**kw):
            return MultiPassAttention(
                cells=[DeltaNetAttention, DeltaNetNegEigAttention], **kw
            )
        return dict(attention_cls=_mp)
    raise ValueError(f"unknown arch: {name}")


def parse_layers_arg(spec: str) -> list:
    """Parse comma-separated --layers spec into a class list."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [_NAME_TO_CLS[p] for p in parts]


class TokenisedStream(IterableDataset):
    """Streaming IterableDataset of fixed-length tokenised chunks."""

    def __init__(self, dataset, tokenizer, block_size, text_field="text",
                 shuffle_buffer=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        buf: list[int] = []
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.bos_token_id
        if eos is None:
            eos = 0
        for example in self.dataset:
            text = example[self.text_field]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= self.block_size + 1:
                chunk = buf[: self.block_size + 1]
                buf = buf[self.block_size :]
                inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield inputs, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default=None,
                   choices=["deltanet", "deltanet_negeig", "deltanet_forgetgate",
                            "gated_deltanet",
                            "gated_deltaproduct",
                            "ortho", "transformer", "mamba2",
                            "hybrid", "hybrid_25_75", "hybrid_75_25",
                            "hybrid_negeig",
                            "symgrounded", "hybrid_sg", "hybrid_sg_25_75",
                            "multipass_dh", "multipass_dho", "multipass_dd"])
    p.add_argument("--n_symbols", type=int, default=512,
                   help="Hash bucket size for SymbolGrounded layer.")
    p.add_argument("--aux_brackets", action="store_true",
                   help="Add bracket-depth auxiliary loss (direction E).")
    p.add_argument("--aux_weight", type=float, default=0.1,
                   help="Weight on the aux loss term.")
    p.add_argument("--aux_max_depth", type=int, default=24,
                   help="Cap bracket depth at this value for the aux head.")
    p.add_argument("--feedback", type=str, default="none",
                   choices=["none", "additive", "film", "predictive"],
                   help="Cross-layer top-down feedback mode (Day 1).")
    p.add_argument("--surprise_weight", type=float, default=0.0,
                   help="Weight on the surprise (prediction-error) aux "
                        "loss for predictive feedback mode.")
    p.add_argument("--freeze_alpha", action="store_true",
                   help="Diagnostic: build film/additive arch with all α=0 "
                        "frozen. Tests dead-weight effect of feedback "
                        "machinery without active feedback.")
    p.add_argument("--save_ckpt", type=str, default=None,
                   help="Path to save final model checkpoint (for downstream "
                        "evals like HumanEval, bracket-structure, long-T PPL).")
    p.add_argument("--feedback_distances", type=str, default="1",
                   help="Comma-separated distances for multi-scale top-down "
                        "feedback. '1' = single-step (default), "
                        "'1,2,4,8,16' = exponential reach.")
    p.add_argument("--feedback_pairs", type=str, default="",
                   help="Sparse (target,source) feedback connections; "
                        "semicolon-separated. E.g. '2,28;3,27' means "
                        "layer 2's input gets feedback from layer 28, "
                        "and layer 3's input gets feedback from layer 27. "
                        "If non-empty, overrides --feedback_distances.")
    p.add_argument("--feedback_xattn", type=str, default="",
                   help="Cross-layer attention feedback. Each target attends "
                        "over multiple source layers' lagged hidden states. "
                        "Syntax 'target:src1,src2,...; target2:src3,src4,...'. "
                        "Example '2:14,21,28' = layer 2's input attends over "
                        "layers 14, 21, 28. 'all' = every layer attends over "
                        "every later layer (Idea 2). If non-empty, overrides "
                        "BOTH --feedback_distances and --feedback_pairs and "
                        "ignores --feedback (attention is its own form).")
    p.add_argument("--feedback_xattn_heads", type=int, default=4,
                   help="Number of heads inside the cross-layer attention.")
    p.add_argument("--feedback_lag", type=int, default=1,
                   help="Lag in tokens for the source state before feeding "
                        "to target (default 1 = t-1, parallel-scan friendly).")
    p.add_argument("--feedback_position", type=str, default="pre",
                   choices=["pre", "post"],
                   help="'pre' (default) modulates target's input; "
                        "'post' modulates target's output.")
    p.add_argument("--feedback_per_channel_alpha", action="store_true",
                   help="Use per-channel α (d_model floats) instead of scalar α "
                        "for sparse FiLM. Tests whether channel-dependent gating "
                        "can mix the negative- and positive-α basins per channel.")
    p.add_argument("--feedback_self_k", type=int, default=0,
                   choices=[0, 2, 3],
                   help="K-iteration self-feeding training for sparse FiLM. "
                        "0 (default) = standard 2-pass training. "
                        "2 = cold start + one self-feed; loss on iter 2 "
                        "(same compute as default 2-pass). "
                        "3 = cold start + two self-feeds; loss on iter 3 "
                        "(~50%% more compute). Trains the model to be "
                        "self-consistent under lagged-cached deployment, "
                        "closing the train/inference gap. Requires "
                        "--feedback_pairs to be set.")
    p.add_argument("--feedback_scratchpad", type=str, default="",
                   help="Surprise-gated scratchpad pairs, semicolon-separated. "
                        "Each is 'target,source' where target attends causally "
                        "over source-layer pass-1 outputs with attention scores "
                        "biased by per-position surprise. E.g. '2,28' applies "
                        "the scratchpad at layer 2 reading from layer 28.")
    p.add_argument("--feedback_scratchpad_heads", type=int, default=9,
                   help="Number of heads inside the scratchpad attention.")
    p.add_argument("--feedback_scratchpad_routing", type=str, default="softmax",
                   choices=["softmax", "sigmoid", "sigmoid_uniform",
                            "uniform_surprise"],
                   help="Routing for the scratchpad attention. Disentangling: "
                        "'softmax' = content (Q·K) softmax + surprise log-bias; "
                        "'sigmoid' = content sigmoid × surprise (multiplicative); "
                        "'sigmoid_uniform' = content only, surprise IGNORED "
                        "(tests pure content retrieval); "
                        "'uniform_surprise' = surprise only, content IGNORED "
                        "(tests pure saliency-based retrieval).")
    p.add_argument("--feedback_xattn_form", type=str, default="attn",
                   choices=["attn", "film_sum", "film_sum_mlp", "film_sum_glu",
                            "film_target_gated",
                            "film_attn", "film_sigmoid", "all_sigmoid"],
                   help="Form of the cross-layer feedback module: "
                        "'attn' = additive Q-K-V softmax residual (default); "
                        "'film_sum' = multi-source FiLM, sum (no softmax); "
                        "'film_sum_mlp' = film_sum but each W is an MLP "
                        "(Linear-GELU-Linear) — tests nonlinear expressivity; "
                        "'film_attn' = softmax routing + multiplicative FiLM "
                        "output (lets attention learn negative-α basin); "
                        "'film_sigmoid' = sigmoid (not softmax) per-source "
                        "gates + FiLM output (no 1/K dilution, per-token "
                        "routing preserved); "
                        "'all_sigmoid' = all-to-all sigmoid-gated FiLM with "
                        "shared K/W projections per source (Idea 2); pair "
                        "with --feedback_xattn 'all_above' or 'all'.")
    p.add_argument("--layers", type=str, default=None,
                   help="explicit comma-separated layer arch list, "
                        "e.g. 'ortho,deltanet,deltanet,deltanet,ortho,...'. "
                        "Overrides --arch.")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--max_T", type=int, default=0,
                   help="Max sequence length for the (optional) absolute "
                        "positional embedding. Set to >= --T (e.g. equal) "
                        "for the Transformer baseline (softmax attention "
                        "is permutation-invariant without position info). "
                        "Linear-RNN architectures (DeltaNet, Mamba2, etc.) "
                        "have implicit position via state and do not need "
                        "this. Default 0 = no positional embedding.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "muon"],
                   help="'adamw' (default) — single AdamW for all params. "
                        "'muon' — Muon for ≥2D hidden-layer matrices, AdamW "
                        "for embeddings, lm_head, and 1D params. Typically "
                        "30-50% faster convergence per Keller Jordan / NanoGPT "
                        "speedrunning. Pair with --lr_muon ~1e-3.")
    p.add_argument("--lr_muon", type=float, default=1e-3,
                   help="Muon learning rate (used only when --optimizer muon). "
                        "AdamW LR for the remaining params is taken from --lr.")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--tokenizer", type=str,
                   default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                   help="HF dataset id; supports also 'codeparrot/codeparrot-clean' "
                        "and bigcode/the-stack-smol/data/python")
    p.add_argument("--dataset_config", type=str, default=None,
                   help="Optional config name (e.g. 'python' for the-stack-smol)")
    p.add_argument("--text_field", type=str, default="text",
                   help="Field in the dataset that contains the text "
                        "('text' for TinyStories, 'content' for codeparrot/the-stack)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    arch_label = args.arch if args.arch else f"layers={args.layers}"
    print(f"GPU: {torch.cuda.get_device_name(0)}  arch={arch_label}")

    # 1. Tokeniser + dataset.
    from transformers import AutoTokenizer
    from datasets import load_dataset
    print(f"Loading tokeniser {args.tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  vocab size: {tok.vocab_size}")

    print(f"Loading dataset {args.dataset} (streaming) ...")
    ds_kwargs = dict(streaming=True)
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    try:
        train_stream = load_dataset(args.dataset, split="train", **ds_kwargs)
    except ValueError:
        # Some datasets only have "train".
        train_stream = load_dataset(args.dataset, **ds_kwargs)["train"]
    try:
        val_stream = load_dataset(args.dataset, split="validation", **ds_kwargs)
    except (ValueError, KeyError):
        # No validation split — split off a slice of train as held-out.
        # We just take a separate streaming pass with a different seed-shuffle.
        try:
            val_stream = load_dataset(args.dataset, split="test", **ds_kwargs)
        except (ValueError, KeyError):
            print("  no val/test split — using shuffled train tail as validation")
            val_stream = load_dataset(args.dataset, split="train",
                                      **ds_kwargs).shuffle(seed=42).skip(10_000)

    train_ds = TokenisedStream(train_stream, tok, args.T,
                               text_field=args.text_field)
    val_ds = TokenisedStream(val_stream, tok, args.T,
                             text_field=args.text_field)
    train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=1)

    # 2. Model.
    if args.layers:
        cls_list = parse_layers_arg(args.layers)
        n_layers_actual = len(cls_list)
        attn_kw = dict(attention_cls_per_layer=cls_list)
    else:
        if args.arch is None:
            raise SystemExit("specify --arch or --layers")
        attn_kw = build_arch(args.arch, args.n_layers,
                             vocab_size=tok.vocab_size, n_symbols=args.n_symbols)
        n_layers_actual = args.n_layers
    aux_dim = (args.aux_max_depth + 1) if args.aux_brackets else 0
    fb_distances = tuple(int(d) for d in args.feedback_distances.split(",") if d)
    fb_pairs = ()
    if args.feedback_pairs:
        fb_pairs = tuple(
            tuple(int(x) for x in pair.split(","))
            for pair in args.feedback_pairs.split(";") if pair
        )
    # Cross-layer attention pairs.
    # 'all'  -> every layer attends over every layer above it.
    # else   -> 'tgt:src1,src2; tgt2:src3,...' explicit form.
    fb_scratchpad_pairs = ()
    if args.feedback_scratchpad:
        fb_scratchpad_pairs = tuple(
            tuple(int(x) for x in pair.split(","))
            for pair in args.feedback_scratchpad.split(";") if pair
        )
    fb_xattn_pairs = ()
    if args.feedback_xattn:
        if args.feedback_xattn.strip() == "all":
            fb_xattn_pairs = tuple(
                (tgt, tuple(s for s in range(n_layers_actual) if s != tgt))
                for tgt in range(n_layers_actual)
            )
        elif args.feedback_xattn.strip() == "all_above":
            # Every layer attends only over LATER layers (higher indices).
            # Last layer has no sources and is skipped.
            fb_xattn_pairs = tuple(
                (tgt, tuple(range(tgt + 1, n_layers_actual)))
                for tgt in range(n_layers_actual - 1)
            )
        else:
            tmp = []
            for group in args.feedback_xattn.split(";"):
                group = group.strip()
                if not group:
                    continue
                target_str, src_str = group.split(":")
                tgt = int(target_str.strip())
                srcs = tuple(int(s) for s in src_str.split(",") if s.strip())
                tmp.append((tgt, srcs))
            fb_xattn_pairs = tuple(tmp)
    model = TinyLM(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_layers=n_layers_actual,
        n_heads=args.n_heads, d_head=args.d_head, aux_dim=aux_dim,
        max_T=args.max_T,
        feedback_mode=args.feedback, feedback_distances=fb_distances,
        feedback_pairs=fb_pairs,
        feedback_xattn_pairs=fb_xattn_pairs,
        feedback_xattn_heads=args.feedback_xattn_heads,
        feedback_xattn_form=args.feedback_xattn_form,
        feedback_lag=args.feedback_lag,
        feedback_position=args.feedback_position,
        feedback_per_channel_alpha=args.feedback_per_channel_alpha,
        feedback_scratchpad_pairs=fb_scratchpad_pairs,
        feedback_scratchpad_heads=args.feedback_scratchpad_heads,
        feedback_scratchpad_routing=args.feedback_scratchpad_routing,
        feedback_self_k=args.feedback_self_k,
        **attn_kw,
    ).to("cuda")
    if args.freeze_alpha and (args.feedback != "none" or fb_xattn_pairs):
        model.freeze_alpha()
    if fb_xattn_pairs:
        n_total_pairs = sum(len(srcs) for _, srcs in fb_xattn_pairs)
        feedback_desc = (f"xattn[{args.feedback_xattn_form}]"
                         f"(targets={len(fb_xattn_pairs)},"
                         f"src-edges={n_total_pairs},"
                         f"heads={args.feedback_xattn_heads})")
    else:
        feedback_desc = f"{args.feedback}@d={args.feedback_distances}"
    print(f"  params: {model.num_params() / 1e6:.1f}M  aux_dim={aux_dim}  "
          f"feedback={feedback_desc}"
          f"{' (α frozen=0)' if args.freeze_alpha else ''}")

    if args.aux_brackets:
        print("Computing bracket-deltas table for tokenizer ...")
        bracket_deltas = compute_bracket_deltas(tok)
        print(f"  table shape: {bracket_deltas.shape}, "
              f"non-zero count: {(bracket_deltas != 0).sum().item()}")
    else:
        bracket_deltas = None

    # Optimizer construction — single AdamW (default) or Muon + AdamW split.
    if args.optimizer == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)]
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(
            opts[0], T_max=args.steps, eta_min=args.lr * 0.1)]
    else:  # muon
        # Split params: ≥2D hidden-layer matrices → Muon; embeddings, lm_head,
        # and 1D params (alphas, RMSNorm scales) → AdamW.
        embed_or_head_names = {"embed.weight", "pos_embed.weight", "lm_head.weight"}
        muon_params, adamw_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            # Muon supports strictly 2D matrices. Embeddings/lm_head + 1D
            # params + 3D+ tensors (short_conv kernels, etc.) go to AdamW.
            if name in embed_or_head_names or p.ndim != 2:
                adamw_params.append(p)
            else:
                muon_params.append(p)
        print(f"  optimizer split: {len(muon_params)} Muon params "
              f"({sum(p.numel() for p in muon_params)/1e6:.1f}M), "
              f"{len(adamw_params)} AdamW params "
              f"({sum(p.numel() for p in adamw_params)/1e6:.1f}M)")
        opts = [
            torch.optim.Muon(muon_params, lr=args.lr_muon,
                             momentum=0.95, weight_decay=0.1),
            torch.optim.AdamW(adamw_params, lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=0.1),
        ]
        scheds = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opts[0], T_max=args.steps, eta_min=args.lr_muon * 0.1),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opts[1], T_max=args.steps, eta_min=args.lr * 0.1),
        ]
    # Backwards-compat aliases used elsewhere in the loop.
    opt = opts[0]
    scheduler = scheds[0]

    # 3. Train loop.
    print(f"\n{'step':>6}  {'tok/s':>8}  {'tloss':>8}  {'lr':>9}")
    t0 = time.perf_counter()
    train_iter = iter(train_loader)
    last_log = t0
    last_log_step = 0
    losses = []
    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to("cuda"), y.to("cuda")
        want_surprise = (args.feedback == "predictive" and args.surprise_weight > 0)
        if args.aux_brackets and want_surprise:
            logits, aux_logits, surprise = model(x, return_aux=True, return_surprise=True)
        elif args.aux_brackets:
            logits, aux_logits = model(x, return_aux=True)
            surprise = torch.zeros((), device="cuda")
        elif want_surprise:
            logits, surprise = model(x, return_surprise=True)
            aux_logits = None
        else:
            logits = model(x)
            aux_logits = None
            surprise = torch.zeros((), device="cuda")
        if args.aux_brackets:
            depth = bracket_depth(x, bracket_deltas).clamp(0, args.aux_max_depth)
            aux_loss = F.cross_entropy(
                aux_logits.reshape(-1, args.aux_max_depth + 1),
                depth.reshape(-1),
            )
        else:
            aux_loss = torch.zeros((), device="cuda")
        lm_loss = F.cross_entropy(
            logits.reshape(-1, tok.vocab_size), y.reshape(-1),
        )
        loss = lm_loss + args.aux_weight * aux_loss + args.surprise_weight * surprise
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        losses.append(lm_loss.item())  # track LM loss alone for comparison

        if step % args.log_every == 0 or step == args.steps:
            now = time.perf_counter()
            tok_per_sec = (step - last_log_step) * args.batch * args.T / (now - last_log)
            line = (f"{step:>6d}  {tok_per_sec:>8.0f}  "
                    f"{sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:])):>8.4f}  "
                    f"{scheduler.get_last_lr()[0]:>9.2e}")
            if args.feedback != "none" or fb_xattn_pairs:
                alphas = model.feedback_alphas()
                if not alphas:
                    pass
                elif isinstance(alphas[0], tuple):
                    # Sparse-FiLM:    (target, source_int, alpha)
                    # Cross-attn:     (target, sources_tuple, alpha)
                    if fb_xattn_pairs:
                        # Show first 4 (target<-K:α) entries to keep line short.
                        head_n = 4
                        items = [f"{t}<-{len(srcs)}:{a:+.3f}"
                                 for t, srcs, a in alphas[:head_n]]
                        suffix = f",… ({len(alphas) - head_n} more)" if len(alphas) > head_n else ""
                        line += "  α=[" + ",".join(items) + suffix + "]"
                    else:
                        line += "  α=[" + ",".join(
                            f"{t}<-{s}:{a:+.3f}" for t, s, a in alphas
                        ) + "]"
                elif isinstance(alphas[0], list):
                    # Dense multi-scale: max |α| per layer.
                    summary = [max(abs(a) for a in row) for row in alphas]
                    line += f"  max|α|=[{','.join(f'{a:.3f}' for a in summary)}]"
                else:
                    line += f"  α=[{','.join(f'{a:+.3f}' for a in alphas)}]"
            print(line)
            last_log = now
            last_log_step = step

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to("cuda"), vy.to("cuda")
                    vlogits = model(vx)
                    vloss = F.cross_entropy(
                        vlogits.reshape(-1, tok.vocab_size), vy.reshape(-1),
                    )
                    val_loss += vloss.item() * vx.numel()
                    n_val += vx.numel()
                    if n_val >= 64 * args.T:                # cap val
                        break
            val_loss /= n_val
            ppl = float(torch.tensor(val_loss).exp())
            print(f"        VAL  loss={val_loss:.4f}  ppl={ppl:.2f}")
            model.train()

    secs = time.perf_counter() - t0
    print(f"\nDone in {secs:.0f}s ({secs/args.steps*1000:.0f} ms/step avg).")

    if args.save_ckpt:
        ckpt = {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": tok.vocab_size,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "d_head": args.d_head, "n_layers": n_layers_actual,
                "max_T": args.T, "feedback_mode": args.feedback,
                "feedback_distances": fb_distances,
                "feedback_pairs": fb_pairs,
                "feedback_xattn_pairs": fb_xattn_pairs,
                "feedback_xattn_heads": args.feedback_xattn_heads,
                "feedback_xattn_form": args.feedback_xattn_form,
                "feedback_self_k": args.feedback_self_k,
                "arch": args.arch, "layers_spec": args.layers,
                "n_symbols": args.n_symbols,
                "tokenizer": args.tokenizer,
            },
        }
        pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save_ckpt)
        print(f"Checkpoint saved to {args.save_ckpt}")


if __name__ == "__main__":
    sys.exit(main())
