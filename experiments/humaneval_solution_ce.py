"""Teacher-forced CE on HumanEval canonical solutions — the artifact-free
"how good a code model is this" metric (no decoding / halt pathology).

For each HumanEval problem: tokenize prompt+canonical_solution, compute mean
cross-entropy ONLY on the solution-span tokens (prompt tokens masked), then
token-weighted-average across all 164 problems. Handles BOTH HuggingFace causal
LMs (--hf NAME) and our TinyLM checkpoints (--ckpt PATH).

COMPARABILITY CAVEAT: the reported per-TOKEN CE is only directly comparable
across two runs that use the SAME tokenizer. The --ckpt path defaults to the
cosmo2/SmolLM2 vocab (49152 tokens) unless the ckpt's own cfg says otherwise
(e.g. a linearized-Qwen ckpt uses the Qwen tokenizer, ~151643 tokens); the
--hf path always uses whatever tokenizer the named HF model ships with, which
is frequently NOT cosmo2/SmolLM2 (e.g. Qwen models). This script prints a
prominent WARNING whenever the loaded tokenizer isn't the cosmo2/SmolLM2
vocab. Per-token CE across DIFFERENT tokenizers is not comparable (fewer,
bigger tokens => lower nats/token for the same information content) — use
the reported `ce_per_byte` for cross-tokenizer comparisons instead (see
SESSION_FINDINGS.md #6: the "+0.145 vs +0.276" tax comparison was nats/token
across two different tokenizers; per-byte the ratio is 1.78, not 1.90).

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/humaneval_solution_ce.py --hf HuggingFaceTB/SmolLM2-360M
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/humaneval_solution_ce.py --ckpt checkpoints/sft_wide10L.pt
"""
import argparse
import contextlib
import json

import torch
import torch.nn.functional as F
from datasets import load_dataset

# cosmo2 / SmolLM2 vocab size (shared across the 135M/360M/1.7B SmolLM2
# tokenizers). Used only as a comparability heuristic for the WARNING below.
COSMO2_VOCAB_SIZE = 49152


def _autocast_ctx(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def hf_logits(model, ids, use_bf16: bool):
    with torch.no_grad(), _autocast_ctx(use_bf16):
        return model(ids).logits


def ckpt_logits(model, ids, use_bf16: bool):
    # Our ckpts keep fp32 master weights (repo convention, see
    # speed_knobs.py::apply_speed_knobs) — bf16 is applied via autocast, not
    # by casting the weights, so the FLA kernels see the dtype they expect.
    with torch.no_grad(), _autocast_ctx(use_bf16):
        out = model(ids)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _tokenizer_is_cosmo2(tok) -> bool:
    return getattr(tok, "vocab_size", None) == COSMO2_VOCAB_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"],
                    help="dtype applied to BOTH arms (default bf16, for parity "
                         "between --hf and --ckpt). Historical `ours` numbers "
                         "in this repo were measured at fp32 (the --hf arm was "
                         "always bf16) — pass --dtype fp32 to reproduce those.")
    ap.add_argument("--json", type=str, default=None,
                    help="optional path to also dump the results as JSON "
                         "(adds fields; never renames the printed keys)")
    args = ap.parse_args()
    use_bf16 = (args.dtype == "bf16")

    if args.hf:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf)
        model = AutoModelForCausalLM.from_pretrained(
            args.hf, dtype=(torch.bfloat16 if use_bf16 else torch.float32)).cuda().eval()
        fwd = hf_logits
        name = args.hf
    else:
        from experiments.eval_bracket_structure import build_model_from_ckpt
        from transformers import AutoTokenizer
        model, cfg = build_model_from_ckpt(args.ckpt)
        model = model.cuda().eval()
        # use the ckpt's own tokenizer (Qwen for linearized-Qwen, else SmolLM2)
        tok = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-360M"))
        fwd = ckpt_logits
        name = args.ckpt

    is_cosmo2 = _tokenizer_is_cosmo2(tok)
    if not is_cosmo2:
        print(f"*** WARNING: tokenizer for {name!r} is NOT the cosmo2/SmolLM2 "
              f"vocab (got vocab_size={getattr(tok, 'vocab_size', '?')}, expected "
              f"{COSMO2_VOCAB_SIZE}). The per-token `ce` printed below is NOT "
              f"comparable to a run using a different tokenizer — compare "
              f"`ce_per_byte` instead. ***", flush=True)

    ds = load_dataset("openai_humaneval", split="test")
    total_nll = 0.0
    total_tok = 0
    total_bytes = 0
    for problem in ds:
        prompt = problem["prompt"]
        sol = problem["canonical_solution"]
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(prompt + sol, add_special_tokens=False)["input_ids"]
        n_sol = len(full_ids) - len(p_ids)
        if n_sol <= 0:
            continue
        ids = torch.tensor([full_ids], device="cuda")
        logits = fwd(model, ids, use_bf16).float()
        # predict token t from position t-1; solution targets are full_ids[len(p):]
        # shift: logits[:, :-1] predict ids[:, 1:]
        logp = F.log_softmax(logits[0, :-1], dim=-1)
        tgt = ids[0, 1:]
        # solution target positions in the shifted frame: indices >= len(p_ids)-1
        start = len(p_ids) - 1
        sel = logp[start:start + n_sol]
        seltgt = tgt[start:start + n_sol]
        nll = -sel.gather(1, seltgt.unsqueeze(1)).squeeze(1)
        total_nll += nll.sum().item()
        total_tok += n_sol
        # Byte length (UTF-8) of the exact scored solution span. Using the
        # ground-truth `canonical_solution` text (rather than re-decoding the
        # scored token ids) sidesteps tokenizer merge effects right at the
        # prompt/solution boundary, and is tokenizer-independent — this is
        # what makes ce_per_byte comparable ACROSS tokenizers.
        total_bytes += len(sol.encode("utf-8"))

    ce = total_nll / max(1, total_tok)
    ce_per_byte = total_nll / max(1, total_bytes)
    print(f"{name}: HumanEval-solution CE = {ce:.4f}  "
          f"(over {total_tok} solution tokens, {len(ds)} problems)", flush=True)
    print(f"{name}: HumanEval-solution CE per byte = {ce_per_byte:.4f} nats/byte "
          f"(over {total_bytes} UTF-8 solution bytes) "
          f"[{'cosmo2/SmolLM2 vocab' if is_cosmo2 else 'NON-cosmo2 tokenizer -- use this for cross-tokenizer comparisons'}]",
          flush=True)

    if args.json:
        result = {
            "name": name,
            "ce": ce,
            "total_tok": total_tok,
            "n_problems": len(ds),
            "ce_per_byte": ce_per_byte,
            "total_bytes": total_bytes,
            "total_nll_nats": total_nll,
            "dtype": args.dtype,
            "tokenizer_vocab_size": getattr(tok, "vocab_size", None),
            "tokenizer_is_cosmo2": is_cosmo2,
        }
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
