"""Stage-B latent-trace eval — the LATENT twin of `eval_exec_trace_text.py`
(EXEC_TRACE_LATENT_PLAN.md "Staged addendum", 2026-07-13).

For a `--latent_reasoning_trace_mode`-trained ckpt (Coconut text->latent
replacement from `stageA_executor.pt`): does the model absorb into R latent
(hidden-feedback) slots what the Stage-A TEXT scratchpad demonstrably carried?

Arms per rung K on the held-out exec-trace set (ORIGINAL schema
`<heldout_prefix>_n{K}.jsonl`; NOT the flattened text file):

  latent_RK    prompt.rstrip()+"\\n# trace:\\n" -> inject R=K latent slots
               (growing thread, bit-matching the TRAINING mechanics via
               `eval_exec_trace_latent.grow_latent_thread`) -> greedy-decode the
               text continuation -> parse "# final: N" (answer_exact). Also
               argmax-decode each latent slot through out_norm->lm_head for the
               per-hop state accuracy vs intermediates.
  direct       prompt.rstrip()+"\\n# final: " greedy -> the same-ckpt no-trace
               baseline (identical to eval_exec_trace_text.py's direct arm).
  text_trace   the Stage-A TEXT arm (generate the "# step j:" trace as text) on
               THIS ckpt -> how much text-executor skill survived Stage B.
  latent_R1    R=1  (under-thinking) — depth-dependence signature, cheap.
  latent_RKp4  R=K+4 (over-thinking) — depth-dependence signature, cheap.

Pre-registered verdict (EXEC_TRACE_LATENT_PLAN.md Stage-B lines):
  KILL          latent_RK answer_exact - direct answer_exact < 5pp at K>=4
                -> latent cannot absorb what text carries; close the arc.
  Mechanism     per-hop decode acc at latent slots >= 0.50 at K=4
                (vs N1'-latent-first's 11% = value-prior matching).
  Success bar   retain >=50% of the Stage-A lift at K=4-8, i.e. latent_RK
                answer_exact >= ~0.55.
  Code guard    (checked separately via HE-CE, not here.)

Loading mirrors `eval_exec_trace_latent.py` (build_model_from_ckpt + film_bypass
+ apply_speed_knobs bf16, fp32 master weights + autocast) so the latent forwards
match the bf16-autocast training thread exactly; the direct/text_trace arms
reuse `eval_exec_trace_text.py`'s incremental prefill/forward_step generator
(its own explicit autocast) unchanged.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/eval_exec_trace_latent_trace.py \\
      --ckpt checkpoints/stageB_latent_trace.pt \\
      --heldout_prefix data/exec_trace_heldout \\
      --rungs 2,3,4,5,6,7,8 --n_per_rung 300 \\
      --out results/exec_trace_latent_trace_eval.json

  # fast harness self-check (2 rungs x 4 examples):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/eval_exec_trace_latent_trace.py --ckpt <ckpt> --smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

# Reuse (import, don't copy) the Stage-A parsers/loaders/generator ...
from experiments.eval_exec_trace_text import (
    build_prompts,
    eval_record as text_eval_record,
    load_rung,
    parse_final,
    _has_complete_final_line,
)
# ... and the validated latent growing-thread injection mechanics.
from experiments.eval_exec_trace_latent import grow_latent_thread
from experiments.thinking import clean_latent_thread


# --------------------------------------------------------------------------- #
# Model loading — mirrors eval_exec_trace_latent.load_eval_model (fp32 master
# weights + bf16 autocast, so the latent forwards match the training thread).
# --------------------------------------------------------------------------- #

def load_eval_model(ckpt_path: str, device: str = "cuda", bf16: bool = True,
                    tf32: bool = True):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.speed_knobs import apply_speed_knobs
    from transformers import AutoTokenizer
    model, cfg = build_model_from_ckpt(ckpt_path)
    model = model.to(device).eval()
    if getattr(model, "use_memory", False):
        model.use_memory = False
    model._film_bypass = True
    apply_speed_knobs(model, bf16=bf16, tf32=tf32, compile_model=False,
                      verbose=False)
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos_id = tok.eos_token_id
    return model, cfg, thinking_id, tok, eos_id


# --------------------------------------------------------------------------- #
# Latent decode — reuses grow_latent_thread (the training-thread injection) then
# greedy-continues the text via the SAME full-forward path with an early stop.
#
# Bit-equivalent to eval_humaneval.generate_latent_think(force_prefix_think=R)
# for the emitted tokens (grow_latent_thread + the length-(P+R) forward produce
# the identical first-emit logits; each subsequent emit re-forwards the same
# way; the same think-token id is masked out of the emission), but with an
# early-stop on the "# final:" line so a long text continuation (the low-R arms)
# does not run the full max_gen every time. thinking_token_id is aliased to
# eos_token_id on these --keep_base_vocab ckpts, so masking it also (correctly)
# forbids emitting eos — matching generate_latent_think's own emit masking.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def latent_greedy_answer(model, prompt_ids: list[int], R: int, thinking_id: int,
                         tok, max_gen: int, device, grace: int = 4) -> str:
    cur_ids, cur_emb = grow_latent_thread(model, prompt_ids, R, thinking_id,
                                          device)
    gen_ids: list[int] = []
    stop_at = None
    for step in range(max_gen):
        with clean_latent_thread(model, film_bypass=True,
                                 no_activation_ckpt=True):
            out = model(cur_ids, inputs_embeds=cur_emb, skip_lm_head=False)
        logits = out[0] if isinstance(out, tuple) else out
        nl = logits[:, -1, :].float().clone()
        nl[..., int(thinking_id)] = -float("inf")   # never emit think/eos-alias
        nxt = int(nl.argmax(dim=-1).item())
        gen_ids.append(nxt)
        text = tok.decode(gen_ids, skip_special_tokens=False)
        if _has_complete_final_line(text):
            if stop_at is None:
                stop_at = step
            if step - stop_at >= grace:
                break
        tok_t = torch.full((1, 1), nxt, dtype=cur_ids.dtype, device=device)
        cur_ids = torch.cat([cur_ids, tok_t], dim=1)
        cur_emb = torch.cat(
            [cur_emb, model.embed(tok_t).to(cur_emb.dtype)], dim=1)
    return tok.decode(gen_ids, skip_special_tokens=False)


def encode_inter_token_ids(tok, inter_vals: list) -> list:
    """Intermediate VALUES -> single TOKEN IDS (the training loader's encoding;
    `None` for multi-token values, which are then skipped in the read). The
    original harness compared the argmax token id against the RAW int value —
    units that can never match, reading structurally 0.000 (bug caught
    2026-07-13 on the first Stage-B eval; answer_exact arms were unaffected)."""
    out = []
    for v in inter_vals:
        enc = tok.encode(str(v), add_special_tokens=False)
        out.append(int(enc[0]) if len(enc) == 1 else None)
    return out


@torch.no_grad()
def latent_perhop_reads(model, prompt_ids: list[int], R: int, thinking_id: int,
                        inter: list[int], device) -> list[int]:
    """Argmax-decode each latent slot j (1..R) at absolute position P+j-1 through
    out_norm->lm_head and compare to intermediates[j-1] AS A TOKEN ID (encode
    via `encode_inter_token_ids` first). Returns the per-hop correctness list
    over j=1..min(R, len(inter)) (the unshifted-logit convention, exactly
    `_answer_span_latent_loss`'s per-hop read)."""
    P = len(prompt_ids)
    cur_ids, cur_emb = grow_latent_thread(model, prompt_ids, R, thinking_id,
                                          device)
    with clean_latent_thread(model, film_bypass=True, no_activation_ckpt=True):
        out = model(cur_ids, inputs_embeds=cur_emb, skip_lm_head=False)
    logits = out[0] if isinstance(out, tuple) else out
    correct = []
    for j in range(1, R + 1):
        if (j - 1) < len(inter) and inter[j - 1] is not None:
            pred = int(logits[0, P + j - 1, :].argmax().item())
            correct.append(int(pred == inter[j - 1]))
    return correct


# --------------------------------------------------------------------------- #
# Per-record evaluation.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_record(model, tok, eos_id: int, thinking_id: int, rec: dict, K: int,
                max_gen_direct: int, grace: int, device) -> dict:
    inter = list(rec.get("intermediates", []))
    answer = rec.get("answer")
    with_trace_prompt, _direct_prompt = build_prompts(rec)
    prompt_ids = tok.encode(with_trace_prompt, add_special_tokens=False)

    # Text arms (direct + text_trace) via the Stage-A incremental generator.
    max_gen_trace = K * 12 + 24
    td = text_eval_record(model, tok, eos_id, rec, K, max_gen_trace,
                          max_gen_direct, grace)

    # Latent arms. latent_RK (fully latent) emits only "# final:" (short); the
    # low-R arm emits residual text steps, so it gets the full trace budget.
    def _latent_answer(R, budget):
        txt = latent_greedy_answer(model, prompt_ids, R, thinking_id, tok,
                                   budget, device, grace)
        pred = parse_final(txt)
        return int(pred is not None and pred == answer), pred, txt

    rk_correct, rk_pred, rk_txt = _latent_answer(K, 24)
    perhop = latent_perhop_reads(model, prompt_ids, K, thinking_id,
                                 encode_inter_token_ids(tok, inter), device)
    r1_correct, _r1_pred, _r1_txt = _latent_answer(1, max_gen_trace)
    rkp4_correct, _rp_pred, _rp_txt = _latent_answer(K + 4, 24)

    return {
        "task_id": rec.get("task_id"), "K": K,
        # text arms
        "text_step_correct": td["step_correct"],
        "text_trace_answer": td["ans_correct"],
        "direct_answer": td["direct_correct"],
        # latent arms
        "latent_RK_answer": rk_correct, "latent_RK_pred": rk_pred,
        "latent_R1_answer": r1_correct,
        "latent_RKp4_answer": rkp4_correct,
        "perhop": perhop,
        "latent_RK_text": rk_txt,
    }


# --------------------------------------------------------------------------- #
# Aggregation + verdict.
# --------------------------------------------------------------------------- #

def aggregate_rung(K: int, per_record: list[dict]) -> dict:
    n = len(per_record)
    total_steps = sum(len(r["text_step_correct"]) for r in per_record)
    correct_steps = sum(sum(r["text_step_correct"]) for r in per_record)
    perhop_total = sum(len(r["perhop"]) for r in per_record)
    perhop_correct = sum(sum(r["perhop"]) for r in per_record)
    # per-step latent accuracy across the K slots (micro over examples)
    perhop_by_step = [0] * K
    perhop_by_step_n = [0] * K
    for r in per_record:
        for j, c in enumerate(r["perhop"]):
            perhop_by_step_n[j] += 1
            perhop_by_step[j] += c

    def _rate(key):
        return (sum(r[key] for r in per_record) / n) if n else None

    return {
        "K": K, "n": n,
        "text_state_acc": (correct_steps / total_steps) if total_steps else None,
        "text_trace_answer": _rate("text_trace_answer"),
        "direct_answer": _rate("direct_answer"),
        "latent_RK_answer": _rate("latent_RK_answer"),
        "latent_R1_answer": _rate("latent_R1_answer"),
        "latent_RKp4_answer": _rate("latent_RKp4_answer"),
        "latent_RK_perhop": (perhop_correct / perhop_total) if perhop_total
        else None,
        "perhop_by_step": [
            (perhop_by_step[j] / perhop_by_step_n[j])
            if perhop_by_step_n[j] else None for j in range(K)],
    }


def compute_verdict(rung_results: list[dict]) -> dict:
    """Pre-registered Stage-B lines (see module docstring)."""
    by_k = {r["K"]: r for r in rung_results}
    # KILL: latent_RK - direct >= 5pp at every K>=4.
    lift_by_k = {}
    for K, r in by_k.items():
        if (K >= 4 and r["latent_RK_answer"] is not None
                and r["direct_answer"] is not None):
            lift_by_k[K] = 100.0 * (r["latent_RK_answer"] - r["direct_answer"])
    kill_pass = bool(lift_by_k) and all(v >= 5.0 for v in lift_by_k.values())
    # Mechanism gate: per-hop decode acc >= 0.50 at K=4.
    k4 = by_k.get(4)
    perhop_k4 = k4["latent_RK_perhop"] if k4 else None
    mech_pass = bool(perhop_k4 is not None and perhop_k4 >= 0.50)
    # Success bar: latent_RK answer_exact >= 0.55 at every K in 4..8.
    succ_by_k = {K: by_k[K]["latent_RK_answer"] for K in range(4, 9)
                 if K in by_k and by_k[K]["latent_RK_answer"] is not None}
    succ_pass = bool(succ_by_k) and all(v >= 0.55 for v in succ_by_k.values())
    return {
        "kill_lift_pp_by_K_for_K_ge_4": lift_by_k,
        "beats_direct_by_5pp_all_K_ge_4 (NOT killed)": kill_pass,
        "perhop_acc_at_K4": perhop_k4,
        "mechanism_gate_pass (perhop>=0.50 @K4)": mech_pass,
        "latent_RK_answer_by_K_4to8": succ_by_k,
        "success_bar_pass (>=0.55 @K4-8)": succ_pass,
        # "overall" = the kill-line NOT tripped (the go/no-go on the arc);
        # mechanism + success are the interpretation, reported alongside.
        "not_killed": kill_pass,
    }


def print_table(title: str, rung_results: list[dict]):
    print(f"\n=== {title} ===")
    hdr = (f"{'K':>3} {'n':>4} | {'txt_st':>6} {'txt_ans':>7} {'direct':>7} | "
           f"{'latRK':>6} {'latR1':>6} {'latRK+4':>7} | {'perhop':>7} | "
           f"{'lift_pp':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rung_results:
        def _f(x, w=6, p=3):
            return (f"{x:>{w}.{p}f}" if x is not None else f"{'n/a':>{w}}")
        lift = (r["latent_RK_answer"] - r["direct_answer"]
                if (r["latent_RK_answer"] is not None
                    and r["direct_answer"] is not None) else None)
        lift_s = f"{100 * lift:>+8.1f}" if lift is not None else f"{'n/a':>8}"
        print(f"{r['K']:>3} {r['n']:>4} | {_f(r['text_state_acc'])} "
              f"{_f(r['text_trace_answer'], 7)} {_f(r['direct_answer'], 7)} | "
              f"{_f(r['latent_RK_answer'])} {_f(r['latent_R1_answer'])} "
              f"{_f(r['latent_RKp4_answer'], 7)} | "
              f"{_f(r['latent_RK_perhop'], 7)} | {lift_s}")
    print("  (lift_pp = latent_RK - direct; kill-line = lift < 5pp at K>=4)")


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def eval_ckpt_on_rungs(model, tok, eos_id, thinking_id, rungs, prefix,
                       n_per_rung, max_gen_direct, grace, device, tag=""):
    rung_results = []
    t0 = time.time()
    for K in rungs:
        recs = load_rung(prefix, K)[:n_per_rung]
        if not recs:
            print(f"  [{tag}] rung {K}: no data at {prefix}_n{K}.jsonl -- skipped")
            continue
        per_record = [eval_record(model, tok, eos_id, thinking_id, rec, K,
                                  max_gen_direct, grace, device)
                      for rec in recs]
        rung_results.append(aggregate_rung(K, per_record))
        print(f"  [{tag}] rung {K} done ({len(recs)} recs, "
              f"{time.time() - t0:.0f}s elapsed)", flush=True)
    return rung_results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout_prefix", default="data/exec_trace_heldout")
    ap.add_argument("--lengen", action="store_true")
    ap.add_argument("--lengen_prefix", default="data/exec_trace_lengen_heldout")
    ap.add_argument("--rungs", default="2,3,4,5,6,7,8")
    ap.add_argument("--lengen_rungs", default="9,10,12")
    ap.add_argument("--n_per_rung", type=int, default=300)
    ap.add_argument("--max_gen_direct", type=int, default=8)
    ap.add_argument("--grace_tokens", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.rungs = "2,4"
        args.n_per_rung = 4
        args.lengen = False

    assert torch.cuda.is_available(), "needs CUDA"
    device = args.device
    t0 = time.time()
    model, cfg, thinking_id, tok, eos_id = load_eval_model(
        args.ckpt, device, bf16=not args.no_bf16, tf32=not args.no_tf32)
    print(f"[loaded] {args.ckpt} thinking_id={thinking_id} eos_id={eos_id} "
          f"use_latent_feedback_adapter="
          f"{getattr(model, 'use_latent_feedback_adapter', False)} "
          f"state_readonly_at_think="
          f"{getattr(model, 'state_readonly_at_think', None)} "
          f"use_memory={getattr(model, 'use_memory', False)}", flush=True)

    all_rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    heldout = eval_ckpt_on_rungs(
        model, tok, eos_id, thinking_id, all_rungs, args.heldout_prefix,
        args.n_per_rung, args.max_gen_direct, args.grace_tokens, device,
        tag="heldout")
    print_table(f"HELDOUT ({args.heldout_prefix})", heldout)

    lengen = []
    if args.lengen:
        lengen_rungs = [int(x) for x in args.lengen_rungs.split(",")
                        if x.strip()]
        lengen = eval_ckpt_on_rungs(
            model, tok, eos_id, thinking_id, lengen_rungs, args.lengen_prefix,
            args.n_per_rung, args.max_gen_direct, args.grace_tokens, device,
            tag="lengen")
        print_table(f"LENGTH-GENERALIZATION ({args.lengen_prefix})", lengen)

    verdict = compute_verdict(heldout)
    print("\n=== STAGE-B PRE-REGISTERED VERDICT (heldout) ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nNOT KILLED (latent beats direct >=5pp @K>=4): "
          f"{verdict['not_killed']}")

    elapsed = time.time() - t0
    print(f"\n[done] total runtime {elapsed:.1f}s")

    results = {"ckpt": args.ckpt, "heldout": heldout, "lengen": lengen,
               "verdict": verdict, "runtime_s": elapsed}
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
