"""Model-construction glue extracted from train_lm.main().

`build_model_from_args(args, vocab_size, thinking_token_id)` parses the
feedback-related CLI strings, resolves attention class(es), and constructs
the TinyLM with the right kwargs. Returns the model plus a small
`ModelBuildInfo` containing derived data the train loop needs later
(`fb_pairs`, `fb_xattn_pairs`, `n_layers`, `aux_dim`).

Kept separate so unit tests can construct equivalent models without
spinning up the full train loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.build_arch import build_arch, parse_layers_arg
from experiments.model import TinyLM


@dataclass
class ModelBuildInfo:
    fb_pairs: tuple
    fb_xattn_pairs: tuple
    n_layers: int
    aux_dim: int


def _parse_feedback_pairs(spec: str) -> tuple:
    if not spec:
        return ()
    return tuple(
        tuple(int(x) for x in pair.split(","))
        for pair in spec.split(";") if pair
    )


def _parse_xattn_pairs(spec: str, n_layers: int) -> tuple:
    if not spec:
        return ()
    s = spec.strip()
    if s == "all":
        return tuple(
            (tgt, tuple(src for src in range(n_layers) if src != tgt))
            for tgt in range(n_layers)
        )
    if s == "all_above":
        return tuple(
            (tgt, tuple(range(tgt + 1, n_layers)))
            for tgt in range(n_layers - 1)
        )
    tmp = []
    for group in s.split(";"):
        group = group.strip()
        if not group:
            continue
        target_str, src_str = group.split(":")
        tgt = int(target_str.strip())
        srcs = tuple(int(x) for x in src_str.split(",") if x.strip())
        tmp.append((tgt, srcs))
    return tuple(tmp)


def build_model_from_args(args, *, vocab_size: int,
                           thinking_token_id: int | None
                           ) -> tuple[TinyLM, ModelBuildInfo]:
    """Build a TinyLM from argparse `args`. Returns (model, info)."""
    # Attention layers — explicit list overrides --arch.
    if args.layers:
        cls_list = parse_layers_arg(args.layers)
        n_layers = len(cls_list)
        attn_kw = dict(attention_cls_per_layer=cls_list)
    else:
        if args.arch is None:
            raise SystemExit("specify --arch or --layers")
        attn_kw = build_arch(args.arch, args.n_layers)
        n_layers = args.n_layers

    aux_dim = (args.aux_max_depth + 1) if args.aux_brackets else 0
    fb_pairs = _parse_feedback_pairs(args.feedback_pairs)
    fb_xattn_pairs = _parse_xattn_pairs(args.feedback_xattn, n_layers)

    mem_kwargs: dict = {}
    if args.use_memory:
        if thinking_token_id is None:
            raise SystemExit(
                "--use_memory requires a thinking token. Set --data_mix "
                "(auto-assigns one) or --enable_thinking_token.")
        mem_kwargs = dict(
            use_memory=True,
            mem_size=int(args.mem_size),
            mem_dim=int(args.mem_dim) if args.mem_dim > 0 else int(args.d_model),
            thinking_token_id=int(thinking_token_id),
            mem_write_only_at_think=bool(
                getattr(args, "mem_write_only_at_think", False)),
        )

    pkm_kwargs: dict = {}
    if getattr(args, "use_pkm", False):
        pkm_kwargs = dict(
            use_pkm=True,
            pkm_after_layer=int(args.pkm_after_layer),
            pkm_n_keys=int(args.pkm_n_keys),
            pkm_n_heads=int(args.pkm_n_heads),
            pkm_k_dim=int(args.pkm_k_dim),
            pkm_top_k=int(args.pkm_top_k),
            pkm_value_bf16=bool(args.pkm_value_bf16),
            # v7 PKM-bootstrap-fix package (forwarded to PKMLayer).
            pkm_score_norm=str(getattr(args, "pkm_score_norm", "layer")),
            pkm_value_init_std=float(getattr(args, "pkm_value_init_std", 1.0)),
            pkm_use_output_gate=bool(getattr(args, "pkm_use_output_gate", True)),
        )

    model = TinyLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=n_layers,
        n_heads=args.n_heads, d_head=args.d_head, aux_dim=aux_dim,
        max_T=args.max_T,
        feedback_mode=args.feedback,
        feedback_pairs=fb_pairs,
        feedback_xattn_pairs=fb_xattn_pairs,
        feedback_xattn_heads=args.feedback_xattn_heads,
        feedback_xattn_form=args.feedback_xattn_form,
        feedback_lag=args.feedback_lag,
        feedback_position=args.feedback_position,
        feedback_per_channel_alpha=args.feedback_per_channel_alpha,
        feedback_self_k=args.feedback_self_k,
        feedback_alpha_mode=args.feedback_alpha_mode,
        output_gate=(args.output_gate
                     or (args.enable_thinking_token
                         and args.think_decision == "gate")),
        activation_checkpointing=args.activation_checkpointing,
        layer_drop_max=float(getattr(args, "layer_drop_max", 0.0)),
        state_readonly_at_think=bool(
            getattr(args, "state_readonly_at_think", False)),
        **mem_kwargs,
        **pkm_kwargs,
        **attn_kw,
    ).to("cuda")

    if args.activation_checkpointing:
        print("Activation checkpointing ON for transformer blocks "
              "(~30% extra compute, large activation-memory savings)")

    # Optional resume.
    if args.load_ckpt is not None:
        ck = torch.load(args.load_ckpt, weights_only=False, map_location="cuda")
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded ckpt {args.load_ckpt!r} "
              f"(missing={len(missing)} unexpected={len(unexpected)} keys)")
        if missing:
            print(f"  first missing: {missing[:5]}")
        if unexpected:
            print(f"  first unexpected: {unexpected[:5]}")

    return model, ModelBuildInfo(
        fb_pairs=fb_pairs, fb_xattn_pairs=fb_xattn_pairs,
        n_layers=n_layers, aux_dim=aux_dim,
    )
