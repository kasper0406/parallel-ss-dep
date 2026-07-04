"""Stage-A executor eval — TEXT scratchpad (EXEC_TRACE_LATENT_PLAN.md phase N3,
2026-07-04). The "does the model learn to execute" harness for the Stage-A
run (`configs/pretrain_mix_stageA_executor.yaml`, `checkpoints/stageA_executor.pt`):
teach the executor function in TEXT with native token machinery ("# trace:"
scratchpad) before any latent compression. Distinct from
`eval_exec_trace_latent.py` (that harness tests the LATENT-Coconut mechanism
on the same held-out data; this one tests the plain discrete-token
scratchpad the Stage-A pretrain mix is actually co-training right now).

Data: the ORIGINAL exec-trace schema (`gen_exec_traces.py`'s output) —
per-rung files `<heldout_prefix>_n{K}.jsonl`, each record a dict with
`prompt` / `intermediates` (list of K per-step tracked-variable values,
real-execution ground truth) / `answer` (== intermediates[-1]) / `rung`
(== K) / `tracked_var`. NOT `data/exec_trace_text_train.jsonl` (that file
has already been flattened to a single `text` field for the data-mix
loader) — inspecting it is what fixes the exact training cue reproduced
here: `<prompt>\n# trace:\n# step 1: <var> = <v1>\n...\n# step K: <var> =
<vK>\n# final: <answer>`.

Two generation prompts per held-out record, SAME ckpt:
  with-trace : prompt.rstrip() + "\n# trace:\n"          (the trained cue)
  direct     : prompt.rstrip() + "\n# final: "            (skip the scratchpad)
Both are greedy-decoded via the TRUE incremental prefill/forward_step path
(`experiments/decode_bench.py::load_ours` for bf16 weight-cast + film_bypass,
the exact loading convention `eval_recall_ours.py` / `scoreboard_longctx_
cost.py::run_ours_quality` use; autocast is applied explicitly around the
prefill/forward_step calls here because `apply_speed_knobs`'s bf16 wrap
only patches `model.forward`, which prefill/forward_step never call through).

Metrics per rung K (see `aggregate_rung`):
  trace_state_acc     micro-avg over all (example, step) pairs: does the
                       generated "# step j: var = VALUE" value match
                       intermediates[j-1]? Missing/malformed step lines
                       count as wrong (parse_step_lines never invents a
                       value it didn't find in the text).
  trace_exact         fraction of examples where ALL K steps are correct.
  answer_exact        fraction where the "# final: N" value matches
                       `answer` (with-trace prompt).
  answer_direct       same accuracy metric on the direct (no-cue) prompt.
  lift_pp             100*(answer_exact - answer_direct) -- does forcing
                       the scratchpad actually help the final answer, or
                       is it decorative?
  full_format_rate / answer_parse_rate / direct_parse_rate
                       FORMAT-COMPLIANCE, separated from accuracy: did the
                       model even emit a parseable structure? A model that
                       never uses the "# step"/"# final:" cue scores 0 on
                       everything above for a structural reason, not a
                       reasoning one -- this is what makes the BASE-ckpt
                       reference run (`--base_ckpt`) an interpretable null.

Pre-registered Stage-A verdict (EXEC_TRACE_LATENT_PLAN.md N3 kill-line, this
harness's PASS/FAIL): trace_state_acc >= 0.5 at K=4  AND  lift_pp >= 5 for
EVERY evaluated rung with K >= 4.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/eval_exec_trace_text.py \\
      --ckpt checkpoints/stageA_executor.pt \\
      --heldout_prefix data/exec_trace_heldout --lengen \\
      --rungs 2,3,4,5,6,7,8 --n_per_rung 300 \\
      --out results/exec_trace_text_eval.json

  # fast harness self-check (2 rungs x 5 examples):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/eval_exec_trace_text.py --ckpt checkpoints/feature_pilot_A.pt --smoke
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

# --------------------------------------------------------------------------- #
# Parsing -- pure functions, no torch/model dependency (unit tested standalone
# in test_eval_exec_trace_text.py without any GPU / FLA / HF requirement).
# --------------------------------------------------------------------------- #

# "# step 1: x = 8" -- lenient: tolerant of extra/missing spaces around ':'
# and '=', case-insensitive, doesn't care what the variable name literally
# is (only the step index and the value matter for scoring).
STEP_RE = re.compile(r"#\s*step\s*(\d+)\s*:\s*\S+\s*=\s*(-?\d+)", re.IGNORECASE)
# "# final: 4" -- the labelled answer line in the with-trace continuation.
FINAL_RE = re.compile(r"#\s*final\s*:\s*(-?\d+)", re.IGNORECASE)
# Direct-baseline continuation has no label (the "# final: " cue is already
# part of the PROMPT, not generated) -- just pull the first integer literal
# out of whatever the model emits next.
FIRST_INT_RE = re.compile(r"-?\d+")


def parse_step_lines(text: str) -> dict[int, int]:
    """Return {step_number: value} for every '# step J: var = VALUE' match
    in `text`. First occurrence of a given J wins (mirrors the trained cue's
    strictly-increasing step order); a step whose line is missing or
    malformed (no match at all) simply never gets an entry -- callers must
    treat an absent key as wrong, never crash trying to look it up."""
    out: dict[int, int] = {}
    for m in STEP_RE.finditer(text):
        j = int(m.group(1))
        if j not in out:
            out[j] = int(m.group(2))
    return out


def parse_final(text: str) -> int | None:
    """First '# final: VALUE' match, or None if the label/value never
    appears (malformed continuation -- counted as wrong by the caller, not
    a crash)."""
    m = FINAL_RE.search(text)
    return int(m.group(1)) if m else None


def parse_direct(text: str) -> int | None:
    """First integer literal anywhere in the direct-baseline continuation,
    or None if the model never emits one."""
    m = FIRST_INT_RE.search(text)
    return int(m.group(0)) if m else None


def _has_complete_final_line(text: str) -> bool:
    """True once a '# final: VALUE' match is followed by at least one more
    character (whitespace/newline/EOF-adjacent) -- i.e. the value token
    itself looks finished, not still being emitted. Used only to trigger
    early-stop generation; a false negative just means we keep generating
    up to the max-token budget, which is always still parseable."""
    m = FINAL_RE.search(text)
    return bool(m) and m.end() < len(text) and text[m.end()] in "\n\r \t."


def _has_complete_direct_value(text: str) -> bool:
    m = FIRST_INT_RE.search(text)
    return bool(m) and m.end() < len(text) and text[m.end()] in "\n\r \t."


def build_prompts(rec: dict) -> tuple[str, str]:
    """(with_trace_prompt, direct_prompt) generation prefixes for one
    held-out record -- exactly the two cue suffixes
    `data/exec_trace_text_train.jsonl`'s flattened `text` field trains on
    (inspected directly: "...of x.\\n# trace:\\n# step 1: x = 8\\n...\\n#
    final: 4"), plus the no-cue control."""
    base = rec["prompt"].rstrip()
    return base + "\n# trace:\n", base + "\n# final: "


# --------------------------------------------------------------------------- #
# Data loading.
# --------------------------------------------------------------------------- #

def load_rung(prefix: str, K: int) -> list[dict]:
    path = f"{prefix}_n{K}.jsonl"
    p = pathlib.Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Generation -- TRUE incremental decode (prefill + forward_step), mirroring
# decode_bench._ours_greedy / scoreboard_longctx_cost._ours_greedy exactly:
# bf16 weights (from decode_bench.load_ours) PLUS an explicit bf16 autocast
# around the calls (prefill/forward_step bypass model.forward, so
# apply_speed_knobs's forward-only autocast wrap never reaches them).
# --------------------------------------------------------------------------- #

@torch.no_grad()
def greedy_generate(model, tok, prompt_ids: list[int], max_gen: int, eos_id: int,
                    stop_fn=None, grace: int = 4) -> str:
    """Greedy-decode up to `max_gen` new tokens. Stops early `grace` tokens
    after `stop_fn(text_so_far)` first turns true (lets a value that
    happens to span >1 BPE token finish before we cut generation), on EOS,
    or at the `max_gen` budget -- whichever comes first."""
    device = next(model.parameters()).device
    t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen_ids: list[int] = []
    stop_at = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cache, last_logits = model.prefill(t)
        nxt = last_logits[:, -1, :].float().argmax(-1, keepdim=True)
        for step in range(max_gen):
            tid = int(nxt.item())
            if tid == eos_id:
                break
            gen_ids.append(tid)
            if stop_fn is not None:
                text = tok.decode(gen_ids, skip_special_tokens=False)
                if stop_fn(text):
                    if stop_at is None:
                        stop_at = step
                    if step - stop_at >= grace:
                        break
            logits, cache = model.forward_step(nxt, cache)
            nxt = logits[:, -1, :].float().argmax(-1, keepdim=True)
    return tok.decode(gen_ids, skip_special_tokens=False)


@torch.no_grad()
def eval_record(model, tok, eos_id: int, rec: dict, K: int, max_gen_trace: int,
                max_gen_direct: int, grace: int = 4) -> dict:
    inter = list(rec.get("intermediates", []))
    answer = rec.get("answer")
    with_trace_prompt, direct_prompt = build_prompts(rec)

    trace_ids = tok.encode(with_trace_prompt, add_special_tokens=False)
    direct_ids = tok.encode(direct_prompt, add_special_tokens=False)

    trace_text = greedy_generate(model, tok, trace_ids, max_gen_trace, eos_id,
                                 stop_fn=_has_complete_final_line, grace=grace)
    direct_text = greedy_generate(model, tok, direct_ids, max_gen_direct, eos_id,
                                  stop_fn=_has_complete_direct_value,
                                  grace=min(2, grace))

    step_vals = parse_step_lines(trace_text)
    step_found = [int(j in step_vals) for j in range(1, K + 1)]
    step_correct = [int(j in step_vals and (j - 1) < len(inter)
                        and step_vals[j] == inter[j - 1])
                    for j in range(1, K + 1)]
    ans_pred = parse_final(trace_text)
    ans_correct = int(ans_pred is not None and ans_pred == answer)
    direct_pred = parse_direct(direct_text)
    direct_correct = int(direct_pred is not None and direct_pred == answer)

    return {
        "task_id": rec.get("task_id"), "K": K,
        "step_found": step_found, "step_correct": step_correct,
        "trace_exact": int(K > 0 and sum(step_correct) == K),
        "ans_pred": ans_pred, "ans_correct": ans_correct,
        "direct_pred": direct_pred, "direct_correct": direct_correct,
        "trace_text": trace_text, "direct_text": direct_text,
    }


# --------------------------------------------------------------------------- #
# Aggregation + reporting.
# --------------------------------------------------------------------------- #

def aggregate_rung(K: int, per_record: list[dict]) -> dict:
    n = len(per_record)
    total_steps = sum(len(r["step_correct"]) for r in per_record)
    correct_steps = sum(sum(r["step_correct"]) for r in per_record)
    full_format = sum(1 for r in per_record
                      if r["step_found"] and sum(r["step_found"]) == len(r["step_found"]))
    trace_exact_n = sum(r["trace_exact"] for r in per_record)
    ans_correct_n = sum(r["ans_correct"] for r in per_record)
    ans_parsed_n = sum(1 for r in per_record if r["ans_pred"] is not None)
    direct_correct_n = sum(r["direct_correct"] for r in per_record)
    direct_parsed_n = sum(1 for r in per_record if r["direct_pred"] is not None)

    trace_state_acc = (correct_steps / total_steps) if total_steps else None
    answer_exact = (ans_correct_n / n) if n else None
    answer_direct = (direct_correct_n / n) if n else None
    lift_pp = (100.0 * (answer_exact - answer_direct)
              if (answer_exact is not None and answer_direct is not None) else None)
    return {
        "K": K, "n": n,
        "trace_state_acc": trace_state_acc,
        "full_format_rate": (full_format / n) if n else None,
        "trace_exact": (trace_exact_n / n) if n else None,
        "answer_exact": answer_exact,
        "answer_parse_rate": (ans_parsed_n / n) if n else None,
        "answer_direct": answer_direct,
        "direct_parse_rate": (direct_parsed_n / n) if n else None,
        "lift_pp": lift_pp,
    }


def compute_verdict(rung_results: list[dict]) -> dict:
    """Stage-A pre-registered kill-line: trace_state_acc >= 0.5 at K=4 AND
    lift_pp >= 5 for EVERY evaluated rung with K >= 4."""
    by_k = {r["K"]: r for r in rung_results}
    k4 = by_k.get(4)
    state_acc_k4 = k4["trace_state_acc"] if k4 else None
    state_pass = bool(state_acc_k4 is not None and state_acc_k4 >= 0.5)
    lift_by_k = {r["K"]: r["lift_pp"] for r in rung_results
                if r["K"] >= 4 and r["lift_pp"] is not None}
    lift_pass = bool(lift_by_k) and all(v >= 5.0 for v in lift_by_k.values())
    return {
        "trace_state_acc_at_K4": state_acc_k4,
        "trace_state_acc_at_K4_pass (>=0.5)": state_pass,
        "lift_pp_by_K_for_K_ge_4": lift_by_k,
        "lift_pass_all_K_ge_4 (>=5pp each)": lift_pass,
        "overall_pass": bool(state_pass and lift_pass),
    }


def print_table(title: str, rung_results: list[dict]):
    print(f"\n=== {title} ===")
    hdr = (f"{'K':>3} {'n':>4} | {'state_acc':>9} {'fmt%':>5} {'trace_ex':>8} | "
          f"{'ans_ex':>7} {'ans_p%':>7} | {'direct':>7} {'dir_p%':>7} | "
          f"{'lift_pp':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rung_results:
        sa = f"{r['trace_state_acc']:.3f}" if r["trace_state_acc"] is not None else "  n/a"
        fmt = f"{100*r['full_format_rate']:.0f}" if r["full_format_rate"] is not None else " n/a"
        te = f"{r['trace_exact']:.3f}" if r["trace_exact"] is not None else "  n/a"
        ae = f"{r['answer_exact']:.3f}" if r["answer_exact"] is not None else "  n/a"
        ap = f"{100*r['answer_parse_rate']:.0f}" if r["answer_parse_rate"] is not None else " n/a"
        de = f"{r['answer_direct']:.3f}" if r["answer_direct"] is not None else "  n/a"
        dp = f"{100*r['direct_parse_rate']:.0f}" if r["direct_parse_rate"] is not None else " n/a"
        lift = f"{r['lift_pp']:+.1f}" if r["lift_pp"] is not None else "   n/a"
        print(f"{r['K']:>3} {r['n']:>4} | {sa:>9} {fmt:>4}% {te:>8} | "
             f"{ae:>7} {ap:>6}% | {de:>7} {dp:>6}% | {lift:>8}")
    print("  (fmt% = full_format_rate: ALL K '# step' lines found, regardless "
         "of value correctness; ans_p%/dir_p% = parse rate, regardless of "
         "correctness -- separates format compliance from capability)")


# --------------------------------------------------------------------------- #
# Model loading + per-ckpt driver.
# --------------------------------------------------------------------------- #

def load_eval_model(ckpt_path: str):
    """bf16 weights + film_bypass via decode_bench.load_ours (the
    incremental prefill/forward_step path's blessed loading convention --
    see eval_recall_ours.py / scoreboard_longctx_cost.run_ours_quality)."""
    from experiments.decode_bench import load_ours
    from transformers import AutoTokenizer
    model, cfg = load_ours(ckpt_path)
    if getattr(model, "use_memory", False):
        model.use_memory = False
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos_id = tok.eos_token_id
    return model, cfg, tok, eos_id


def eval_ckpt_on_rungs(ckpt_path: str, rungs: list[int], prefix: str,
                       n_per_rung: int, max_gen_direct: int, grace: int,
                       tag: str = "") -> list[dict]:
    model, cfg, tok, eos_id = load_eval_model(ckpt_path)
    print(f"  [{tag}] loaded {ckpt_path} ({cfg.get('n_layers')}L x "
         f"{cfg.get('d_model')}d, arch={cfg.get('arch')}, "
         f"thinking_token_id={cfg.get('thinking_token_id')} eos_id={eos_id})",
         flush=True)
    rung_results = []
    t0 = time.time()
    for K in rungs:
        recs = load_rung(prefix, K)[:n_per_rung]
        if not recs:
            print(f"  [{tag}] rung {K}: no data at {prefix}_n{K}.jsonl -- skipped")
            continue
        max_gen_trace = K * 12 + 24
        per_record = [eval_record(model, tok, eos_id, rec, K, max_gen_trace,
                                  max_gen_direct, grace)
                     for rec in recs]
        rung_results.append(aggregate_rung(K, per_record))
        print(f"  [{tag}] rung {K} done ({len(recs)} recs, "
             f"{time.time()-t0:.0f}s elapsed)", flush=True)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return rung_results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout_prefix", default="data/exec_trace_heldout")
    ap.add_argument("--lengen", action="store_true",
                    help="also evaluate the length-generalization rungs")
    ap.add_argument("--lengen_prefix", default="data/exec_trace_lengen_heldout")
    ap.add_argument("--rungs", default="2,3,4,5,6,7,8")
    ap.add_argument("--lengen_rungs", default="9,10,12")
    ap.add_argument("--n_per_rung", type=int, default=300)
    ap.add_argument("--max_gen_direct", type=int, default=8,
                    help="max new tokens for the no-cue direct-answer prompt")
    ap.add_argument("--grace_tokens", type=int, default=4,
                    help="extra tokens generated past the first early-stop "
                         "trigger, in case the value spans >1 BPE token")
    ap.add_argument("--base_ckpt", default="checkpoints/feature_pilot_A.pt",
                    help="reference ckpt for the format-vs-capability null "
                         "check (item 5): same eval, expect near-0 trace "
                         "compliance since it was never trained on this cue")
    ap.add_argument("--skip_base_control", action="store_true")
    ap.add_argument("--base_n_per_rung", type=int, default=30,
                    help="smaller sample for the reference run -- it's a "
                         "compliance smell test, not a precision measurement")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run for a fast harness self-check")
    args = ap.parse_args()

    if args.smoke:
        args.rungs = "2,4"
        args.n_per_rung = 5
        args.base_n_per_rung = 3
        args.lengen = False

    assert torch.cuda.is_available(), "needs CUDA"

    all_rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    t0 = time.time()

    print(f"[ckpt] evaluating {args.ckpt}")
    main_heldout = eval_ckpt_on_rungs(
        args.ckpt, all_rungs, args.heldout_prefix, args.n_per_rung,
        args.max_gen_direct, args.grace_tokens, tag="ckpt/heldout")
    print_table(f"HELDOUT ({args.heldout_prefix})", main_heldout)

    main_lengen: list[dict] = []
    if args.lengen:
        lengen_rungs = [int(x) for x in args.lengen_rungs.split(",") if x.strip()]
        main_lengen = eval_ckpt_on_rungs(
            args.ckpt, lengen_rungs, args.lengen_prefix, args.n_per_rung,
            args.max_gen_direct, args.grace_tokens, tag="ckpt/lengen")
        print_table(f"LENGTH-GENERALIZATION ({args.lengen_prefix})", main_lengen)

    base_heldout: list[dict] = []
    base_path = pathlib.Path(args.base_ckpt)
    ckpt_path = pathlib.Path(args.ckpt)
    if args.skip_base_control:
        print("\n[base control] skipped (--skip_base_control)")
    elif not base_path.exists():
        print(f"\n[base control] skipped -- {args.base_ckpt} not found")
    elif base_path.resolve() == ckpt_path.resolve():
        print(f"\n[base control] skipped -- base_ckpt == ckpt ({args.ckpt})")
    else:
        print(f"\n[base control] {args.base_ckpt} "
             f"(n_per_rung={args.base_n_per_rung}, format-vs-capability null check)")
        base_heldout = eval_ckpt_on_rungs(
            args.base_ckpt, all_rungs, args.heldout_prefix, args.base_n_per_rung,
            args.max_gen_direct, args.grace_tokens, tag="base/heldout")
        print_table(f"BASE CONTROL ({args.base_ckpt})", base_heldout)

    verdict = compute_verdict(main_heldout)
    print("\n=== STAGE-A PRE-REGISTERED VERDICT (heldout) ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nPASS/FAIL: {'PASS' if verdict['overall_pass'] else 'FAIL'}")

    elapsed = time.time() - t0
    print(f"\n[done] total runtime {elapsed:.1f}s")

    results = {
        "ckpt": args.ckpt, "heldout": main_heldout, "lengen": main_lengen,
        "base_control": {"ckpt": args.base_ckpt, "heldout": base_heldout},
        "verdict": verdict, "runtime_s": elapsed,
    }
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
