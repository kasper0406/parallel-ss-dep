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

from experiments.gist_loss import (
    build_gist_heads, trunk_gist_loss, parse_horizons,
)


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


def load_distilled_jsonl(path: str, *,
                          prefer_full_completion: bool = True,
                          require_extracted_code: bool = True,
                          keep_only_passing: bool = False,
                          ) -> list[tuple[str, str]]:
    """Load (problem, solution) pairs from a distill_solutions.py JSONL.

    Each JSONL row has {task_id, problem_prompt, qwen_completion,
    extracted_code, has_tests, tier, score, sample_idx}.

      prefer_full_completion:
        True  (default) → solution = qwen_completion (CoT + code block).
                          Student learns to reason before emitting code,
                          which exercises the thinking gate during SFT.
        False → solution = extracted_code only (cleaner but no reasoning
                signal).

      require_extracted_code:
        Drop rows where Qwen ran out of tokens before producing a
        ```python``` block. Default True — those rows are noisy.

      keep_only_passing:
        If True, drop rows where has_tests and tier != "pass" (rejection
        sampling: train only on solutions known to work). Distillation-
        only sources (no tests) are kept regardless.
    """
    import json
    pairs: list[tuple[str, str]] = []
    n_total = n_dropped_no_code = n_dropped_failed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if require_extracted_code and not r.get("extracted_code"):
                n_dropped_no_code += 1
                continue
            if keep_only_passing and r.get("has_tests") and r.get("tier") != "pass":
                n_dropped_failed += 1
                continue
            problem = r["problem_prompt"]
            solution = (r["qwen_completion"] if prefer_full_completion
                        else r["extracted_code"])
            if not solution:
                continue
            pairs.append((problem, solution))
    print(f"  loaded {len(pairs)} pairs from {path} "
          f"(total rows={n_total}, dropped no-code={n_dropped_no_code}, "
          f"dropped failed={n_dropped_failed})")
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
    p.add_argument("--distilled_jsonl", type=str, default=None,
                   help="If set, load (problem, solution) pairs from a "
                        "distill_solutions.py JSONL output INSTEAD OF "
                        "MBPP+CodeAlpaca. Each row contributes one pair where "
                        "the solution is Qwen's full completion (CoT + code).")
    p.add_argument("--distilled_keep_only_passing", action="store_true",
                   help="If set with --distilled_jsonl, drop rows where the "
                        "problem had tests and Qwen's sample didn't pass "
                        "(rejection sampling). Distillation-only rows "
                        "(magicoder/codefeedback, no tests) are always kept.")
    p.add_argument("--distilled_code_only", action="store_true",
                   help="If set with --distilled_jsonl, use ONLY the extracted "
                        "code block as the solution target (drop Qwen's "
                        "reasoning prose). Default keeps the full completion.")
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
    p.add_argument("--mem_write_only_at_think", action="store_true",
                   help="FIX A: force WorkingMemory writes to come only "
                        "from think positions. See train_lm_args.py for "
                        "full rationale. When the loaded ckpt was trained "
                        "without this flag, retraining a few epochs with "
                        "it on lets the model adapt its write-gate to "
                        "actually be selective at think positions.")
    p.add_argument("--retrieval_as_input_thinking", action="store_true",
                   help="Replace the discrete [THINKING] token's input "
                        "embedding with the WorkingMemory retrieval at the "
                        "previous position. Solves the think-position "
                        "homogeneity that caused FIX A to fail (the "
                        "[THINKING] token has one embedding so successive "
                        "thinks are highly correlated). With this flag, "
                        "each think step's input is the model's own "
                        "retrieval → diverse per-step → diverse buffer → "
                        "useful sharp reads. See GEMINI.md "
                        "'retrieval-as-input' for the architectural "
                        "rationale.")
    p.add_argument("--disable_wm_during_sft", action="store_true",
                   help="Zero WorkingMemory.W_proj weight and freeze it, "
                        "so WM injections are always zero during this SFT "
                        "run. Diagnostic flag — not currently used in "
                        "production; the milestone calls for fixing WM, "
                        "not removing it.")
    # --- v7 trunk multi-horizon GIST loss (Fix C, 2026-05-20) ----------
    # The trunk's job is "high-level direction": at position t each head
    # predicts the GIST of the upcoming window — the mean-pooled hidden
    # state over h[t+1 : t+1+K], stop-grad'd. The trunk is causal so
    # each h[t] is a running contextualised summary; the windowed mean
    # is a genuine "where this is going" vector. Multi-horizon (K in
    # {16,64,256}) gives local tactic + mid plan + global direction.
    #
    # History: v5 supervised the WM read to predict embed(input_ids[t+4])
    # (context-free lexical). v6 supervised the WM read to predict this
    # gist — but routing a blurry gist through WM broke precise recall
    # (longctx eval 2026-05-20: 99%→61%). v7 supervises the TRUNK with
    # the gist and leaves WM free to learn precise retrieval.
    p.add_argument("--future_emb_loss_weight", type=float, default=0.0,
                   help="Weight for the v7 trunk multi-horizon gist "
                        "loss. 0 = disabled. Recommended 0.1.")
    p.add_argument("--wm_gist_horizons", type=str, default="16,64,256",
                   help="Comma-separated future-window sizes K for the "
                        "trunk gist loss (one head per horizon).")
    # Deprecated flags — kept so older launchers still parse.
    p.add_argument("--wm_future_pred_weight", type=float, default=0.0,
                   help="DEPRECATED since v7 (WM gist supervision "
                        "removed). Ignored.")
    p.add_argument("--wm_future_pred_T", type=int, default=4,
                   help="DEPRECATED (v5 single-offset embed target). "
                        "Ignored.")
    p.add_argument("--future_emb_T_max", type=int, default=8,
                   help="DEPRECATED (v5 lexical-target ramp). Ignored.")
    p.add_argument("--future_emb_T_ramp_frac", type=float, default=0.3,
                   help="DEPRECATED (v5 lexical-target ramp). Ignored.")
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
            # FIX A: honour the new flag even on a pre-built model — flip
            # the bit on the already-constructed WorkingMemory module.
            if bool(args.mem_write_only_at_think):
                model.memory.write_only_at_think = True
                cfg["mem_write_only_at_think"] = True
                print(f"  FIX A enabled: WorkingMemory.write_only_at_think = True "
                      "(non-think positions masked to -1.0 before topk)")
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

    # Three branches: modern (already done above), legacy (build fresh +
    # expand vocab), or original (no thinking).
    if args_with_thinking_done:
        # Modern path already loaded `model` and `cfg` above and applied
        # FIX A if requested. Don't re-load here — that would silently
        # overwrite the FIX A flag and any other modern-path setup.
        pass
    elif args.with_thinking and not args_with_thinking_done:
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
            mem_write_only_at_think=bool(args.mem_write_only_at_think),
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
        cfg["mem_write_only_at_think"] = bool(args.mem_write_only_at_think)
        cfg["sft_with_thinking"] = True
    else:
        # Original path: load whatever was saved, leave memory inert if present.
        from experiments.eval_bracket_structure import build_model_from_ckpt
        model, cfg = build_model_from_ckpt(args.load_ckpt)
        thinking_token_id = cfg.get("thinking_token_id")
    # --- Optional: disable WorkingMemory injection for this run ----------
    # Motivated by the 2026-05-19 ablation on v1: wm_off scored 12/164 vs
    # baseline 11/164, suggesting WM is at best neutral, slightly hurting.
    # Zero W_proj.weight (the output projection) and freeze it — every
    # injection becomes 0, so WM is structurally inert. The gate still
    # fires and think tokens are still inserted (so the rest of the
    # think-burst infrastructure works), but the WM-read path contributes
    # nothing to h.
    if (getattr(args, "disable_wm_during_sft", False)
            and hasattr(model, "memory")):
        with torch.no_grad():
            model.memory.W_proj.weight.zero_()
        model.memory.W_proj.weight.requires_grad_(False)
        cfg["disable_wm_during_sft"] = True
        print("  disable_wm_during_sft: WM.W_proj zeroed + frozen "
              "(injection = 0 for all positions)")
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
    if args.distilled_jsonl:
        pairs = load_distilled_jsonl(
            args.distilled_jsonl,
            prefer_full_completion=not args.distilled_code_only,
            require_extracted_code=True,
            keep_only_passing=args.distilled_keep_only_passing,
        )
    else:
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
    # v7 trunk multi-horizon gist heads (Fix C, 2026-05-20). The trunk's
    # job is "high-level direction": at position t each head predicts the
    # mean-pooled future hidden state over h[t+1:t+1+K] (the windowed
    # gist), one head per horizon K. This REPLACES both the old lexical
    # future-emb target (embed(input_ids[t+T]) — context-free) and the
    # v6 WM-injection gist supervision (which forced WM to emit blurry
    # gists and broke precise recall). WM is now left to learn precise
    # retrieval via the LM loss alone; "direction" lives in the trunk.
    d_model = int(cfg["d_model"])
    gist_horizons = parse_horizons(args.wm_gist_horizons)
    future_gist_heads = None
    if args.future_emb_loss_weight > 0:
        future_gist_heads = build_gist_heads(d_model, gist_horizons).cuda()
        _ck = locals().get("raw_ckpt")
        if _ck is None:
            _ck = torch.load(args.load_ckpt, map_location="cpu",
                              weights_only=False)
        if "future_gist_heads_state_dict" in _ck:
            try:
                future_gist_heads.load_state_dict(
                    _ck["future_gist_heads_state_dict"])
                print("  trunk-gist heads: restored from ckpt")
            except (RuntimeError, KeyError):
                print("  trunk-gist heads: ckpt horizons differ — fresh")
        print(f"  trunk-gist heads (v7 Fix C): d_model={d_model}, "
              f"horizons={gist_horizons}, "
              f"weight={args.future_emb_loss_weight}")
    if args.wm_future_pred_weight > 0:
        print("  NOTE: --wm_future_pred_weight is deprecated since v7 "
              "(WM gist supervision removed) — ignored.")

    # Optimizer: retrieval_input_alpha gets NO weight decay (the FiLM-α
    # lesson — WD on a gate scalar manufactures a false low equilibrium).
    alpha_params, decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (alpha_params if n.endswith("retrieval_input_alpha")
         else decay_params).append(p)
    head_params = (list(future_gist_heads.parameters())
                   if future_gist_heads is not None else [])
    opt = torch.optim.AdamW(
        # WD=0.01 is the project default since 2026-05-14 (the v3a
        # residual-stream-collapse finding — see GEMINI.md). WD=0.1
        # was a Moonlight-scale (5.7 T-token) setting; at our
        # ~10 tok/param it acts as pure brake on the residual stream.
        [{"params": decay_params + head_params, "weight_decay": 0.01},
         {"params": alpha_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
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
                # Need hidden states for the trunk gist aux loss.
                want_hidden = future_gist_heads is not None
                # --- Retrieval-as-input thinking (v7 additive, Fix B) ---
                # 2-pass forward:
                #   pass 1 (no_grad): capture the WM injection at every
                #           position (model.memory._last_injection).
                #   build inputs_embeds: at think positions ADD the
                #           previous position's WM retrieval, scaled by
                #           the learned scalar α (model.retrieval_input_
                #           alpha), to the think-token embedding.
                #   pass 2: forward with inputs_embeds.
                # v5/v6 REPLACED the think embedding with the retrieval
                # — destructive: a blurry retrieval overwrote the precise
                # binding the trunk carried (v6 longctx: 99%→61% recall).
                # v7 ADDS instead: input[think] = think_embed + α·retr.
                # A useless retrieval contributes ≈0 (gradient shrinks
                # α); the think_embed baseline always survives. WM still
                # gets gradient via the residual _inject_memory path
                # inside pass 2, so pass 1 can stay no_grad.
                if (args.retrieval_as_input_thinking
                        and args.with_thinking
                        and thinking_token_id is not None):
                    with torch.no_grad():
                        _ = model(x)
                    inj = model.memory._last_injection  # (B, T, d) detached
                    base_emb = model.embed(x)               # (B, T, d)
                    is_think = (x == int(thinking_token_id)).unsqueeze(-1)
                    # Think at t consumes the retrieval from t-1.
                    shifted_inj = torch.cat(
                        [torch.zeros_like(inj[:, :1]), inj[:, :-1]],
                        dim=1,
                    )
                    alpha = model.retrieval_input_alpha
                    inputs_embeds = (
                        base_emb
                        + is_think.to(base_emb.dtype)
                        * alpha
                        * shifted_inj.to(base_emb.dtype)
                    )
                    if want_hidden:
                        logits, h = model(x, inputs_embeds=inputs_embeds,
                                          return_hidden=True)
                    else:
                        logits = model(x, inputs_embeds=inputs_embeds)
                elif want_hidden:
                    logits, h = model(x, return_hidden=True)
                else:
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
                lm_loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = lm_loss
                # ----- v7 trunk multi-horizon gist loss (Fix C) -----
                # Each head predicts, from the trunk hidden state h[t],
                # the GIST of the upcoming window — the mean-pooled
                # hidden state over h[t+1 : t+1+K], stop-grad'd. The
                # trunk is causal so each h[t] is a running
                # contextualised summary; the windowed mean is a
                # genuine "where this is going" vector. Multi-horizon K
                # gives local tactic + mid plan + global direction.
                # This is the v6 gist target, but supervising the TRUNK
                # rather than the WM read — so "direction" lives in the
                # trunk and WM is left free to learn precise retrieval
                # (v6 put the blurry gist into WM and broke recall).
                if future_gist_heads is not None:
                    loss = loss + args.future_emb_loss_weight * (
                        trunk_gist_loss(h, future_gist_heads, gist_horizons))
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
    # v7: record that retrieval-as-input used the additive α-gated form
    # so the eval generator picks the matching injection mode.
    new_cfg["retrieval_input_additive"] = bool(
        args.retrieval_as_input_thinking)
    if future_gist_heads is not None:
        new_cfg["future_emb_loss_weight"] = float(args.future_emb_loss_weight)
        new_cfg["wm_gist_horizons"] = list(gist_horizons)
    ckpt_dict = {"state_dict": model.state_dict(), "config": new_cfg}
    if future_gist_heads is not None:
        ckpt_dict["future_gist_heads_state_dict"] = \
            future_gist_heads.state_dict()
    torch.save(ckpt_dict, args.save_ckpt)
    print(f"saved: {args.save_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
