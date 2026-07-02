"""FLAGSHIP capability probe — long-context multi-key recall, 3 fair arms.

THE CLAIM (capability only; cost/O(1)-decode deferred): a bounded-state
DeltaNet + WorkingMemory can RECALL facts bound EARLY in a long sequence and
queried LATE — at sequence lengths BEYOND a fixed transformer's training
window — where a fixed-window transformer goes blind.

ARMS (same task content / values / queries; same answer extraction):
  - ours            : checkpoints/pretrain_v12.pt (DeltaNet + WorkingMemory +
                      thinking gate). NO positional limit (max_T=0) so it
                      processes arbitrary L; trained at T=2048 so L>2048 is
                      length-extrapolation. Natural deployment (gate decides
                      whether to think). Full-forward-equivalent decode.
  - ours_forcethink : same model, but FORCE `force_prefix_think` latent/WM
                      think steps before the answer (diagnostic: does forcing
                      the WM channel rescue long recall the gate won't use?).
  - control_full    : HuggingFaceTB/SmolLM2-360M, FULL context (perfect-but-
                      expensive attention; O(L) KV). max_position_embeddings
                      8192 → degrades/￼OOMs past that.
  - control_window2048 : SmolLM2-360M but the context is TRUNCATED to the LAST
                      2048 tokens before the query — a fixed-cost agent. Past
                      L=2048 the early bindings leave the window → recall must
                      collapse to ~chance. The key foil OURS must beat.

FAIRNESS NOTE (equal-opportunity, per repo mandate): OURS and the base
transformer have DISJOINT best-shot elicitations — OURS was pretrained only on
the "Run the following Python program ... Answer: N" PROSE format (6/6 at
L=1024 there, 0/6 on bare completion), while the base SmolLM2-360M is a
*base* LM that does not follow that instruction (0/6) but completes a
"The value of vX is " cue at 6/6. Forcing ONE shared format would cripple one
arm and inflate the other. So each arm uses ITS OWN best-shot wrapper around
the IDENTICAL task body / bindings / queried key, and the SAME first-4-digit
answer extractor. The task CONTENT (which value must be recalled at what
distance) is identical across arms. A single-shared-format sensitivity run is
available with --shared_format to show the wrapper effect explicitly.

Leak-free by construction: the bound value is NEVER restated before the query,
so a recency-copy cannot answer; the model must recall across the full
binding→query distance ≈ L.

CAVEAT: CAPABILITY ONLY. This does NOT measure or claim O(1)/constant-cost
decode — true incremental forward_step decode is an unwired perf detail; the
capability numbers here are decode-path-independent (incremental == full
forward, verified). Cost/throughput is deliberately out of scope.

DTYPE PARITY (fixed 2026-07-01): OURS defaults to bf16 (`--ours_dtype bf16`)
to match the SmolLM2 controls (always bf16, see `run_control`) and
`decode_bench.load_ours`'s deployment dtype — an fp32 OURS vs bf16 controls
was an unaudited precision asymmetry favoring OURS. Historical runs of this
script (pre-2026-07-01) used fp32; pass `--ours_dtype fp32` to reproduce them.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \\
    experiments/flagship_recall_probe.py \\
      --tasks data/flagship_recall.jsonl \\
      --ckpt checkpoints/pretrain_v12.pt \\
      --hf_model HuggingFaceTB/SmolLM2-360M \\
      --keys_per_task 6 --out_json results/flagship_recall.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

NL = "\n"
_FENCE = "```"

# ---- per-arm best-shot prompt wrappers (identical body/key across arms) ----

def prompt_ours(body: str, key: str) -> str:
    """OURS best-shot: the multibind_recall PROSE format v12 was pretrained on
    (problem_prompt + the '\\n\\n' the data_mix join inserts)."""
    return ("Run the following Python program and report what it prints." + NL + NL
            + _FENCE + "python" + NL + body + NL + f"print({key})" + NL + _FENCE
            + NL + NL)


def prompt_completion(body: str, key: str) -> str:
    """Base-LM best-shot: a pure value-completion cue (no instruction
    following needed). Leak-free — the value is not restated before the cue."""
    return (_FENCE + "python" + NL + body + NL + _FENCE + NL + NL
            + f"The value of {key} is ")


_INT_RE = re.compile(r"-?\d+")
_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)", re.IGNORECASE)
_FOUR_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")


def extract_answer(text: str) -> str | None:
    """Identical extractor for ALL arms. Values are 4-digit and distractors
    contain NO 4-digit literals, so the first isolated 4-digit number is the
    model's recalled value. Fallbacks: explicit `Answer: N`, then first int."""
    m = _FOUR_RE.search(text)
    if m:
        return m.group(0)
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1)
    m = _INT_RE.search(text)
    return m.group(0) if m else None


def load_tasks(path: str):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# ---------------------------- OURS arm ------------------------------------

def run_ours(model, tok, tasks, *, keys_per_task, max_gen, force_think,
             shared_format, max_problems_per_bucket):
    from experiments.eval_humaneval import (
        generate, generate_with_retrieval_as_input)
    tid = int(model.thinking_token_id)
    wrap = prompt_completion if shared_format else prompt_ours
    per_bucket: dict[int, list[int]] = {}
    tok_lens: dict[int, list[int]] = {}
    agg_think = agg_emit = 0
    seen_bucket: dict[int, int] = {}
    model.eval()
    with torch.no_grad():
        for rec in tasks:
            L = rec["bucket"]
            if (max_problems_per_bucket and
                    seen_bucket.get(L, 0) >= max_problems_per_bucket):
                continue
            seen_bucket[L] = seen_bucket.get(L, 0) + 1
            keys = rec["keys"][:keys_per_task]
            for key in keys:
                gold = str(rec["answers"][key])
                prompt = wrap(rec["body"], key)
                ids = tok.encode(prompt, add_special_tokens=False)
                t = torch.tensor(ids, device="cuda").unsqueeze(0)
                if force_think > 0:
                    out, diag = generate_with_retrieval_as_input(
                        model, t, max_gen=max_gen, temperature=0.0,
                        eos_token_id=tok.eos_token_id, thinking_token_id=tid,
                        total_think_budget=force_think + 8,
                        emit_threshold=0.5, gate_floor=0.0,
                        additive=bool(getattr(model, "retrieval_input_additive",
                                              False)),
                        force_prefix_think=force_think, use_incremental=True)
                else:
                    out, diag = generate(
                        model, t, max_gen=max_gen, temperature=0.0,
                        eos_token_id=tok.eos_token_id, use_thinking=True,
                        thinking_token_id=tid, total_think_budget=24,
                        emit_threshold=0.5, gate_floor=0.0,
                        use_incremental=True)
                gen_only = [x for x in out[0, t.shape[1]:].tolist()
                            if x != tid]
                text = tok.decode(gen_only, skip_special_tokens=True)
                pred = extract_answer(text)
                per_bucket.setdefault(L, [0, 0])
                per_bucket[L][1] += 1
                per_bucket[L][0] += int(pred == gold)
                tok_lens.setdefault(L, []).append(len(ids))
                agg_think += diag.get("think_total", 0)
                agg_emit += diag.get("emit_count", 0)
    think_rate = agg_think / max(1, agg_think + agg_emit)
    return per_bucket, tok_lens, think_rate


# --------------------------- CONTROL arms ---------------------------------

def run_control(hf, htok, tasks, *, keys_per_task, max_gen, window,
                shared_format, max_problems_per_bucket):
    # Controls always use the base-LM completion cue UNLESS shared_format
    # forces OURS's prose wrapper on everyone.
    wrap = prompt_ours if shared_format else prompt_completion
    per_bucket: dict[int, list[int]] = {}
    tok_lens: dict[int, list[int]] = {}
    n_overflow = 0
    seen_bucket: dict[int, int] = {}
    eos = htok.eos_token_id
    with torch.no_grad():
        for rec in tasks:
            L = rec["bucket"]
            if (max_problems_per_bucket and
                    seen_bucket.get(L, 0) >= max_problems_per_bucket):
                continue
            seen_bucket[L] = seen_bucket.get(L, 0) + 1
            keys = rec["keys"][:keys_per_task]
            for key in keys:
                gold = str(rec["answers"][key])
                prompt = wrap(rec["body"], key)
                ids = htok.encode(prompt, add_special_tokens=False)
                full_len = len(ids)
                if window is not None:
                    ids = ids[-window:]            # keep the LAST `window` toks
                tok_lens.setdefault(L, []).append(len(ids))
                if len(ids) > 8192:
                    n_overflow += 1               # beyond SmolLM2's window
                t = torch.tensor(ids, device="cuda").unsqueeze(0)
                out = hf.generate(t, max_new_tokens=max_gen, do_sample=False,
                                  pad_token_id=eos)
                text = htok.decode(out[0, len(ids):], skip_special_tokens=True)
                pred = extract_answer(text)
                per_bucket.setdefault(L, [0, 0])
                per_bucket[L][1] += 1
                per_bucket[L][0] += int(pred == gold)
    return per_bucket, tok_lens, n_overflow


def _fmt_table(results: dict, buckets: list[int]) -> str:
    arms = list(results.keys())
    w = max(len(a) for a in arms) + 2
    head = f"{'arm':<{w}}" + "".join(f"{('L=' + str(b)):>14}" for b in buckets)
    lines = [head, "-" * len(head)]
    for a in arms:
        pb = results[a]["per_bucket"]
        row = f"{a:<{w}}"
        for b in buckets:
            if str(b) in pb:
                c, t = pb[str(b)]
                row += f"{f'{100*c/max(1,t):.1f}% ({c}/{t})':>14}"
            else:
                row += f"{'-':>14}"
        lines.append(row)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="data/flagship_recall.jsonl")
    p.add_argument("--ckpt", default="checkpoints/pretrain_v12.pt")
    p.add_argument("--hf_model", default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--keys_per_task", type=int, default=6)
    p.add_argument("--max_gen", type=int, default=16)
    p.add_argument("--max_problems_per_bucket", type=int, default=0,
                   help="0 = all tasks in the file per bucket")
    p.add_argument("--force_think", type=int, default=8,
                   help="ours_forcethink: # forced WM/think steps before answer")
    p.add_argument("--arms", type=str,
                   default="ours,ours_forcethink,control_full,control_window2048")
    p.add_argument("--shared_format", action="store_true",
                   help="sensitivity: force OURS's prose wrapper on the controls "
                        "and the completion wrapper on OURS (shows wrapper effect)")
    p.add_argument("--ours_dtype", choices=["bf16", "fp32"], default="bf16",
                   help="dtype OURS runs at. Defaults to bf16 for deployment "
                        "parity with decode_bench.load_ours (the SmolLM2 "
                        "controls below always load bf16, so an fp32 OURS "
                        "was an unfair precision advantage). HISTORICAL "
                        "runs of this script (pre-2026-07-01) used fp32 — "
                        "pass --ours_dtype fp32 to reproduce them.")
    p.add_argument("--out_json", default="results/flagship_recall.json")
    args = p.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    tasks = load_tasks(args.tasks)
    buckets = sorted({r["bucket"] for r in tasks})
    mppb = args.max_problems_per_bucket or None
    print(f"{len(tasks)} tasks, buckets={buckets}, keys_per_task="
          f"{args.keys_per_task}, arms={arms}")

    from transformers import AutoTokenizer
    results: dict[str, dict] = {}

    # ---- OURS arms ----
    if any(a.startswith("ours") for a in arms):
        from experiments.eval_bracket_structure import build_model_from_ckpt
        model, cfg = build_model_from_ckpt(args.ckpt)
        if args.ours_dtype == "bf16":
            model.to(torch.bfloat16)   # deployment dtype; matches decode_bench.load_ours
        # else fp32 (build_model_from_ckpt's default, unmodified) — historical/opt-in.
        otok = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
        if "ours" in arms:
            t0 = time.time()
            pb, tl, tr = run_ours(
                model, otok, tasks, keys_per_task=args.keys_per_task,
                max_gen=args.max_gen, force_think=0,
                shared_format=args.shared_format,
                max_problems_per_bucket=mppb)
            results["ours"] = {
                "per_bucket": {str(k): v for k, v in pb.items()},
                "tok_len_mean": {str(k): sum(v) / len(v) for k, v in tl.items()},
                "think_rate": tr, "fmt": "completion" if args.shared_format
                else "prose", "secs": time.time() - t0}
            print(f"[ours] done in {results['ours']['secs']:.0f}s "
                  f"think_rate={tr:.3f}")
        if "ours_forcethink" in arms:
            t0 = time.time()
            pb, tl, tr = run_ours(
                model, otok, tasks, keys_per_task=args.keys_per_task,
                max_gen=args.max_gen, force_think=args.force_think,
                shared_format=args.shared_format,
                max_problems_per_bucket=mppb)
            results["ours_forcethink"] = {
                "per_bucket": {str(k): v for k, v in pb.items()},
                "tok_len_mean": {str(k): sum(v) / len(v) for k, v in tl.items()},
                "think_rate": tr, "force_think": args.force_think,
                "secs": time.time() - t0}
            print(f"[ours_forcethink] done in "
                  f"{results['ours_forcethink']['secs']:.0f}s "
                  f"think_rate={tr:.3f}")
        del model
        torch.cuda.empty_cache()

    # ---- CONTROL arms ----
    if any(a.startswith("control") for a in arms):
        from transformers import AutoModelForCausalLM
        htok = AutoTokenizer.from_pretrained(args.hf_model)
        hf = AutoModelForCausalLM.from_pretrained(
            args.hf_model, dtype=torch.bfloat16).to("cuda").eval()
        for arm, window in (("control_full", None),
                            ("control_window2048", 2048)):
            if arm not in arms:
                continue
            t0 = time.time()
            pb, tl, nov = run_control(
                hf, htok, tasks, keys_per_task=args.keys_per_task,
                max_gen=args.max_gen, window=window,
                shared_format=args.shared_format,
                max_problems_per_bucket=mppb)
            results[arm] = {
                "per_bucket": {str(k): v for k, v in pb.items()},
                "tok_len_mean": {str(k): sum(v) / len(v) for k, v in tl.items()},
                "n_overflow_8192": nov, "secs": time.time() - t0}
            print(f"[{arm}] done in {results[arm]['secs']:.0f}s "
                  f"overflow>8192={nov}")
        del hf
        torch.cuda.empty_cache()

    # ---- report ----
    print("\n" + "=" * 78)
    print("FLAGSHIP RECALL — accuracy vs sequence length L "
          f"(keys_per_task={args.keys_per_task})")
    print("=" * 78)
    print(_fmt_table(results, buckets))
    print("\ntokens per bucket (SmolLM2 tokenizer, shared across arms):")
    ref = results.get("ours") or next(iter(results.values()))
    for b in buckets:
        if str(b) in ref["tok_len_mean"]:
            print(f"  L={b}: ~{ref['tok_len_mean'][str(b)]:.0f} tokens"
                  + (" (control_window2048 caps at 2048)" if b > 2048 else ""))
    for a in results:
        if a.startswith("ours"):
            print(f"  {a}: think_rate={results[a].get('think_rate'):.3f}")

    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "buckets": buckets,
                   "results": results}, f, indent=2)
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
