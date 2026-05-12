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
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    # --- 1. Build model from ckpt ----------------------------------------
    # NOTE: Loading via build_model_from_ckpt auto-detects memory.* keys in
    # the state dict and re-enables WorkingMemory. That's *fine* for eval,
    # but SFT here does not pass `mem_read_mask`, so the memory module
    # would inject only at thinking_token_id positions — which never appear
    # in MBPP/CodeAlpaca text. The module's gradient would therefore be
    # zero (same failure mode as the old corpus-RAG). Make this explicit:
    # if the loaded ckpt has memory weights, we still build the model with
    # memory ON (so state_dict matches), but the caller should be aware
    # that those weights are not being trained here.
    print(f"loading checkpoint: {args.load_ckpt}")
    from experiments.eval_bracket_structure import build_model_from_ckpt
    model, cfg = build_model_from_ckpt(args.load_ckpt)
    model.train()
    mem_in_ckpt = bool(cfg.get("use_memory", False)) or hasattr(model, "memory")
    print(f"  model: {cfg['n_layers']}L  d_model={cfg['d_model']}  "
          f"params={model.num_params() / 1e6:.1f}M")
    if mem_in_ckpt:
        print("  ⚠️  ckpt has WorkingMemory weights. SFT does NOT pass "
              "mem_read_mask, so memory.* weights will receive zero gradient "
              "during this run (inert path). Use RL for memory training.")

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
            x, y = make_batch(rows)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                # Standard causal-LM shift: logits[:, :-1] predict y[:, 1:].
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = y[:, 1:].contiguous()
                # ignore_index=-100 masks the comment tokens.
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
