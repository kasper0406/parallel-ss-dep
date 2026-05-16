"""Supervised fine-tune of a distilled TinyLM on (problem, solution) pairs.

Goal: bridge the gap from "produces code-shaped text" (the distilled base)
to "produces code that passes unit tests" by training on instruction-shaped
data: MBPP train split + CodeAlpaca.

Format: we present each (problem, solution) as a Python comment followed by
the solution code, so the SFT distribution matches plain-code distribution
at HumanEval inference time:

    # <problem text, one-line>
    <solution code>
    <eos>

The loss is computed only on the solution tokens (everything from the
newline after the comment to <eos>). Problem-comment tokens are masked.

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/sft_code.py \\
        --load_ckpt checkpoints/distill_qwen36_dn217_mem.pt \\
        --save_ckpt checkpoints/sft_dn217_mem.pt \\
        --epochs 2 --batch 4 --lr 3e-5
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


def _flatten_to_oneline(s: str) -> str:
    """Squash multi-line problem text into a one-line `# ...` comment-safe string."""
    return " ".join(s.split())


def load_pairs(max_codealpaca: int | None = None) -> list[tuple[str, str]]:
    """Return a list of (problem, solution) string pairs.

    Sources combined:
      - MBPP train split (~370 problems with `text` and `code` fields).
      - CodeAlpaca-20k (filtered to Python-looking outputs).
    """
    from datasets import load_dataset

    pairs: list[tuple[str, str]] = []

    print("loading MBPP train split...")
    mbpp = load_dataset("mbpp", split="train")
    for x in mbpp:
        pairs.append((x["text"], x["code"]))

    print("loading CodeAlpaca-20k...")
    try:
        ca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception as e:
        print(f"  CodeAlpaca load failed ({e}); proceeding with MBPP only.")
        return pairs

    n_added = 0
    for x in ca:
        if max_codealpaca is not None and n_added >= max_codealpaca:
            break
        instr = x["instruction"]
        inp = x.get("input") or ""
        out = x["output"]
        # Filter to Python-looking outputs (heuristic): must contain `def `
        # OR start with python keywords / imports.
        if not any(t in out for t in ("def ", "import ", "for ", "while ",
                                       "if ", "return ", "print(")):
            continue
        prompt = instr if not inp else f"{instr}\n{inp}"
        pairs.append((prompt, out))
        n_added += 1
    print(f"  added {n_added} CodeAlpaca examples")
    return pairs


def build_example(prompt: str, solution: str, tokenizer,
                  max_len: int) -> tuple[list[int], list[int]]:
    """Tokenize one example to (input_ids, labels) where labels are -100 on
    the prompt-comment tokens and the actual token ids on the solution
    tokens (causal LM convention: predict each token given prefix).
    """
    comment_line = f"# {_flatten_to_oneline(prompt)}\n"
    comment_ids = tokenizer.encode(comment_line, add_special_tokens=False)
    solution_text = solution + ("\n" if not solution.endswith("\n") else "")
    sol_ids = tokenizer.encode(solution_text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        sol_ids = sol_ids + [int(eos)]
    full = comment_ids + sol_ids
    # Truncate from the right if needed (keep the comment intact).
    if len(full) > max_len:
        full = full[:max_len]
        sol_len = max(0, len(full) - len(comment_ids))
    else:
        sol_len = len(sol_ids)
    # Labels: -100 on the comment portion, real ids on the solution.
    labels = [-100] * (len(full) - sol_len) + full[len(full) - sol_len:]
    return full, labels


def insert_think_bursts(
    input_ids: list[int], labels: list[int],
    thinking_token_id: int, max_len: int,
    max_bursts: int = 3, max_burst_depth: int = 8,
    rng: torch.Generator | None = None,
    aligned: list[int] | None = None,
) -> tuple[list[int], list[int]] | tuple[list[int], list[int], list[int]]:
    """Insert random-depth thinking-token bursts into a (input_ids, labels) pair.

    Each burst is a run of `depth` think tokens; depth ~ U[1, max_burst_depth].
    Labels at inserted think positions are set to -100 (no loss). The result
    is truncated to `max_len` if too long.

    Purpose: give the think-token embedding, the gate head, and the working-
    memory module dense supervised gradient *before* RL ever sees them. The
    SFT loss is still next-token CE on real tokens (think positions don't
    contribute), but every real-token prediction that follows a think burst
    requires the model to have processed those think tokens gracefully, so
    the think-embedding and memory weights have to learn to be useful.

    `aligned`: an optional per-token array (same length as `input_ids`, e.g.
    document ids) that must stay aligned through the same insertions. Each
    inserted think token copies the doc id of the preceding real token, so a
    think burst belongs to the document it sits inside. When given, the
    return is a 3-tuple `(new_ids, new_labels, new_aligned)`; otherwise the
    2-tuple `(new_ids, new_labels)` — existing callers are unaffected.
    """
    def _ret(ids, labs, al):
        return (ids, labs, al) if aligned is not None else (ids, labs)

    if rng is None:
        rng = torch.Generator().manual_seed(0)
    if max_bursts <= 0:
        return _ret(input_ids, labels, aligned)
    n = len(input_ids)
    if n < 4:
        return _ret(input_ids, labels, aligned)
    n_bursts = int(torch.randint(0, max_bursts + 1, (1,), generator=rng).item())
    if n_bursts == 0:
        return _ret(input_ids, labels, aligned)
    burst_positions = sorted(
        torch.randperm(n, generator=rng)[:n_bursts].tolist()
    )
    new_ids: list[int] = []
    new_labels: list[int] = []
    new_aligned: list[int] = []
    last = 0
    for p in burst_positions:
        new_ids.extend(input_ids[last:p])
        new_labels.extend(labels[last:p])
        depth = int(
            torch.randint(1, max_burst_depth + 1, (1,), generator=rng).item()
        )
        new_ids.extend([int(thinking_token_id)] * depth)
        new_labels.extend([-100] * depth)
        if aligned is not None:
            new_aligned.extend(aligned[last:p])
            # Inserted think tokens belong to the document of the preceding
            # real token (or the first document, at position 0).
            fill = aligned[p - 1] if p > 0 else aligned[0]
            new_aligned.extend([fill] * depth)
        last = p
    new_ids.extend(input_ids[last:])
    new_labels.extend(labels[last:])
    if aligned is not None:
        new_aligned.extend(aligned[last:])
    if len(new_ids) > max_len:
        new_ids = new_ids[:max_len]
        new_labels = new_labels[:max_len]
        new_aligned = new_aligned[:max_len]
    return _ret(new_ids, new_labels, new_aligned)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--load_ckpt", type=str, required=True)
    p.add_argument("--save_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_codealpaca", type=int, default=10000,
                   help="Cap on CodeAlpaca samples (full set is 20k).")
    p.add_argument("--seed", type=int, default=0)
    # --- Thinking-during-SFT: forces the model to handle think tokens ---
    p.add_argument("--with_thinking", action="store_true",
                   help="Enable working memory + output gate + random think "
                        "burst insertion during SFT. This trains the "
                        "think-embedding, memory weights, and trunk to handle "
                        "think tokens gracefully BEFORE RL evaluates the "
                        "gate's emit/think decision. Without this, RL has to "
                        "bootstrap thinking from random init — which it "
                        "rationally refuses (see WORKING_MEMORY_FINDINGS.md).")
    p.add_argument("--think_max_bursts", type=int, default=3,
                   help="Max think-token bursts inserted per example.")
    p.add_argument("--think_max_depth", type=int, default=8,
                   help="Max depth per inserted burst.")
    p.add_argument("--mem_size", type=int, default=1024)
    p.add_argument("--mem_dim", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    # --- 1. Build model from ckpt ----------------------------------------
    print(f"loading checkpoint: {args.load_ckpt}")
    if args.with_thinking:
        # Detect whether the loaded ckpt ALREADY has thinking infrastructure
        # (memory + output gate + thinking-token vocab slot). v5-pkm and later
        # ckpts have all of this baked in from pretrain; older distilled
        # ckpts (the original SFT use case) do not, and we have to add it.
        import torch as _t
        raw_ckpt = _t.load(args.load_ckpt, map_location="cpu", weights_only=False)
        sd_keys = set(raw_ckpt["state_dict"].keys())
        ckpt_has_memory = any(k.startswith("memory.") for k in sd_keys)
        ckpt_has_gate = any(k.startswith("gate_head.") for k in sd_keys)
        if ckpt_has_memory and ckpt_has_gate:
            # Modern path: ckpt already has memory + gate (and possibly PKM).
            # Use build_model_from_ckpt — it autodetects all three and gets
            # the architecture exactly right. Skip the "expand vocab" code
            # because the thinking-token slot is already in the saved vocab.
            from experiments.eval_bracket_structure import build_model_from_ckpt
            model, cfg = build_model_from_ckpt(args.load_ckpt)
            thinking_token_id = cfg.get("thinking_token_id")
            if thinking_token_id is None:
                # Fall back to "last vocab slot" if cfg didn't store it.
                thinking_token_id = int(cfg["vocab_size"]) - 1
            print(f"  with-thinking + ckpt-already-has-thinking: loaded as-is "
                  f"(memory + gate {'+ pkm ' if any(k.startswith('pkm_layer.') for k in sd_keys) else ''}"
                  f"think_id={thinking_token_id})")
            cfg["sft_with_thinking"] = True
            base_vocab_for_loss = int(cfg["vocab_size"]) - 1
            model.train()
            # Skip the rest of the with-thinking branch below.
            args_with_thinking_done = True
        else:
            args_with_thinking_done = False
    else:
        args_with_thinking_done = False

    if args.with_thinking and not args_with_thinking_done:
        # Legacy path: ckpt has no memory + gate. Build the model directly
        # with thinking + memory ON, then load the ckpt state with
        # strict=False so memory + gate heads stay freshly-initialised. The
        # think-token embedding gets fresh init at the new last vocab slot
        # (later over-written to embed-mean inside TinyLM.__init__ when
        # use_memory=True).
        from experiments.model import TinyLM
        from experiments.layers import DeltaNetAttention
        cfg = dict(raw_ckpt["config"])  # copy
        sd = raw_ckpt["state_dict"]
        base_vocab = int(cfg["vocab_size"])
        new_vocab = base_vocab + 1
        thinking_token_id = base_vocab
        # Expand embed + lm_head rows if needed (ckpt had output_gate=False,
        # so no extra slot was reserved).
        for key in ("embed.weight", "lm_head.weight"):
            if key in sd and sd[key].shape[0] < new_vocab:
                old = sd[key]
                pad = _t.zeros(new_vocab - old.shape[0], old.shape[1], dtype=old.dtype)
                sd[key] = _t.cat([old, pad], dim=0)
        fb_pairs = tuple(tuple(p) for p in cfg.get("feedback_pairs", ()) or ())
        model = TinyLM(
            vocab_size=new_vocab,
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]),
            d_head=int(cfg["d_head"]),
            max_T=int(cfg.get("max_T", 0)),
            feedback_mode=str(cfg.get("feedback_mode", "none")),
            feedback_pairs=fb_pairs,
            feedback_self_k=int(cfg.get("feedback_self_k", 0)),
            tie_embeddings=bool(cfg.get("tie_embeddings", True)),
            output_gate=True,
            use_memory=True,
            mem_size=int(args.mem_size),
            mem_dim=int(args.mem_dim) if args.mem_dim > 0 else int(cfg["d_model"]),
            thinking_token_id=thinking_token_id,
            attention_cls=DeltaNetAttention,
        ).cuda()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  with-thinking build: +1 vocab slot for think token "
              f"({thinking_token_id=}), missing={len(missing)} unexpected={len(unexpected)}")
        cfg["use_memory"] = True
        cfg["output_gate"] = True
        cfg["thinking_token_id"] = thinking_token_id
        cfg["vocab_size"] = new_vocab
        cfg["mem_size"] = int(args.mem_size)
        cfg["mem_dim"] = int(args.mem_dim) if args.mem_dim > 0 else int(cfg["d_model"])
        cfg["sft_with_thinking"] = True
    else:
        # Original path: load whatever was saved, leave memory inert if present.
        from experiments.eval_bracket_structure import build_model_from_ckpt
        model, cfg = build_model_from_ckpt(args.load_ckpt)
        thinking_token_id = cfg.get("thinking_token_id")
    model.train()
    # base_vocab_for_loss = index BELOW which targets are valid emit tokens.
    # Slicing logits[..., :base_vocab_for_loss] removes the thinking-token
    # slot (and any kernel-alignment padding above it) from the CE softmax.
    if args.with_thinking and thinking_token_id is not None:
        base_vocab_for_loss = int(thinking_token_id)
    else:
        base_vocab_for_loss = int(cfg["vocab_size"])
    print(f"  model: {cfg['n_layers']}L  d_model={cfg['d_model']}  "
          f"params={model.num_params() / 1e6:.1f}M  "
          f"with_thinking={args.with_thinking}  "
          f"vocab={cfg['vocab_size']} loss_slice=:{base_vocab_for_loss}")

    # --- 2. Tokenizer ------------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer",
                                                "HuggingFaceTB/SmolLM2-135M"))
    print(f"  tokenizer: {cfg.get('tokenizer')}  vocab={tok.vocab_size}")

    # --- 3. Data ----------------------------------------------------------
    pairs = load_pairs(max_codealpaca=args.max_codealpaca)
    print(f"  total pairs: {len(pairs)}")
    print(f"  tokenizing...")
    encoded: list[tuple[list[int], list[int]]] = []
    skipped = 0
    for prompt, sol in pairs:
        full, labels = build_example(prompt, sol, tok, args.max_len)
        if len(full) < 8 or all(l == -100 for l in labels):
            skipped += 1
            continue
        encoded.append((full, labels))
    print(f"  encoded: {len(encoded)} (skipped {skipped})")
    pad_id = int(tok.eos_token_id) if tok.eos_token_id is not None else 0

    def make_batch(rows):
        max_t = max(len(ids) for ids, _ in rows)
        max_t = min(max_t, args.max_len)
        bsz = len(rows)
        x = torch.full((bsz, max_t), pad_id, dtype=torch.long, device=device)
        y = torch.full((bsz, max_t), -100, dtype=torch.long, device=device)
        for i, (ids, labels) in enumerate(rows):
            ids = ids[:max_t]; labels = labels[:max_t]
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
        return x, y

    # --- 4. Optim + train loop -------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    n_steps = (len(encoded) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, n_steps), eta_min=args.lr * 0.1,
    )
    print(f"  total train steps: {n_steps}")
    print()

    rng = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    losses: list[float] = []
    step = 0
    for epoch in range(args.epochs):
        # Shuffle each epoch.
        idx = torch.randperm(len(encoded), generator=rng).tolist()
        for i in range(0, len(encoded) - args.batch + 1, args.batch):
            rows = [encoded[idx[j]] for j in range(i, i + args.batch)]
            if args.with_thinking and thinking_token_id is not None:
                # Inject random think bursts per row. Each row independently
                # samples 0..K bursts; depth ∈ [1, max_burst_depth] per burst.
                rows = [
                    insert_think_bursts(ids, lbls, int(thinking_token_id),
                                          args.max_len,
                                          args.think_max_bursts,
                                          args.think_max_depth, rng)
                    for ids, lbls in rows
                ]
            x, y = make_batch(rows)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                # Slice off the +1 thinking-token slot before CE so the
                # think token is never a valid target (label space ==
                # base_vocab_for_loss). The slice is a no-op when not
                # in with_thinking mode (sizes match).
                if args.with_thinking:
                    logits = logits[..., :base_vocab_for_loss]
                # Standard causal-LM shift: logits[:, :-1] predict y[:, 1:].
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = y[:, 1:].contiguous()
                # ignore_index=-100 masks the comment tokens AND the
                # inserted think tokens.
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                ppl = math.exp(min(loss.item(), 20))
                tok_s = (step * args.batch * args.max_len) / max(1, time.time() - t0)
                print(f"  step {step:>5}/{n_steps}  loss={loss.item():.4f}  "
                      f"ppl={ppl:.2f}  lr={lr:.2e}  tok/s={tok_s:.0f}")

    print(f"\nDone in {time.time() - t0:.0f}s.  Final loss: {losses[-1]:.4f}")

    # --- 5. Save ---------------------------------------------------------
    pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    new_cfg = dict(cfg)
    new_cfg["sft_source"] = "MBPP+CodeAlpaca"
    new_cfg["sft_epochs"] = args.epochs
    torch.save({"state_dict": model.state_dict(), "config": new_cfg},
               args.save_ckpt)
    print(f"saved: {args.save_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
