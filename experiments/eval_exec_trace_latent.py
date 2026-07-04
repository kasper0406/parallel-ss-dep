"""Kill-test harness for the exec-trace latent program (EXEC_TRACE_LATENT_PLAN.md
phase N3 — the pre-registered evals, 2026-07-04).

Mirrors the TRAINING-side latent mechanics exactly (any train/eval mismatch here
invalidates the kill-test — this repo's most documented failure class):

  - Growing-thread Coconut feedback, identical to
    `latent_reasoning_cotrain._answer_span_latent_loss` / `latent_sft.latent_sft_loss`:
    at each of R latent steps, append ONE new `[thinking_token_id]` position whose
    INPUT EMBEDDING is `model.apply_latent_feedback_adapter(h[:, -1:, :])` from the
    previous forward over the (growing) sequence so far. This is NOT the same
    mechanism as `thinking._latent_think_logits` (which iterates a SINGLE fixed
    think slot R times) — that primitive is for the general-text latent_cotrain
    probe, a different training path. `train_lm.py`'s `LatentReasoningCotrain`
    (what actually produced the N0/N1 ckpts) uses the growing-thread form, so
    that is what this harness reproduces.
  - `state_readonly_at_think` / `use_memory` / `feedback_mode` are read from the
    ckpt's own config (never forced) — these ckpts were trained with
    state_readonly_at_think=False (state-WRITABLE thinking; see ckpt cfg), so
    forcing it True here would itself be a mechanics mismatch.
  - `thinking_token_id` is read from cfg, not assumed — the N0/N1 base
    (`feature_pilot_A.pt`) was pretrained with `--keep_base_vocab`, so
    thinking_token_id is ALIASED to eos_token_id (=0 for the SmolLM2 tokenizer).
    No gate_head exists on these ckpts (pure `--latent_reasoning_weight`
    co-train, no `--latent_reasoning_gate_weight`), so R is always externally
    forced via `force_prefix_think` — there is no autonomous think/emit
    decision to evaluate here.
  - Greedy decode reuses `eval_humaneval.generate_latent_think` (the exact
    inference codepath `latent_arith_real.py._eval_rung` uses), so the
    "greedy-decode" arm is a genuine integration check of the real deploy path,
    while the "teacher-forced" arm is the pure diagnostic (argmax at each
    latent slot / answer position from ONE combined forward — causally
    identical to the greedy decode's logits at those same positions, since
    appending the gold answer span cannot change earlier positions' hidden
    states). For this corpus every answer is a single BPE token (mod-10 by
    construction), so teacher-forced-exact and greedy-exact are mathematically
    equivalent for the "none" and "R=K" arms; computing BOTH via independent
    code paths is a live self-consistency check baked into the harness (see
    `--assert_tf_greedy_agree`).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
      experiments/eval_exec_trace_latent.py \
      --ckpt checkpoints/latent_exectrace_N1.pt \
      --heldout_prefix data/exec_trace_heldout \
      --lengen_prefix data/exec_trace_lengen_heldout \
      --rungs 2,3,4,5,6,7,8 --lengen_rungs 9,10,12 \
      --n_per_rung 300 --out results/exec_trace_latent_eval.json
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate_latent_think
from experiments.speed_knobs import apply_speed_knobs
from experiments.thinking import clean_latent_thread


@contextlib.contextmanager
def _dedup_repeated_notice(substr: str):
    """Suppress a repeated informational `print(...)` after its first
    occurrence, WITHOUT editing the module that emits it.

    `eval_humaneval.generate_latent_think` prints a one-line note every single
    call when `state_readonly_at_think=False` (true for these ckpts, an
    expected condition, not a bug — see module docstring). At thousands of
    greedy-decode calls per run that would flood the log; this is read-only
    wrt existing files, so we filter at the `builtins.print` level for the
    scope of this context manager only (restored immediately after)."""
    orig_print = builtins.print
    seen = False

    def _filtered(*args, **kwargs):
        nonlocal seen
        if args and isinstance(args[0], str) and substr in args[0]:
            if seen:
                return
            seen = True
        orig_print(*args, **kwargs)

    builtins.print = _filtered
    try:
        yield
    finally:
        builtins.print = orig_print


# ---------------------------------------------------------------------------
# Data loading — mirrors latent_reasoning_cotrain._load_rung /
# latent_arith_real._load_rung exactly (same prompt->completion framing), plus
# the `intermediates` field for the per-hop metric.
# ---------------------------------------------------------------------------

def load_rung(prefix: str, n: int, tok, max_len: int) -> list[dict]:
    path = f"{prefix}_n{n}.jsonl"
    p = pathlib.Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.open():
        if not line.strip():
            continue
        r = json.loads(line)
        pfx = r["prompt"] + "\ndef solve():\n    return "
        c = tok.encode(pfx, add_special_tokens=False)
        s = tok.encode(str(r["answer"]), add_special_tokens=False)
        inter_ids = [tok.encode(str(v), add_special_tokens=False)[0]
                     for v in r.get("intermediates", [])]
        # +8 margin over the rung's own R covers the K+4 wrong-R arm and the
        # +eos token; real prompts are well under this in practice (see
        # module docstring measurements: max ~665 tok at K=12).
        if len(c) + len(s) + n + 8 <= max_len:
            out.append({"task_id": r.get("task_id"), "comment_ids": c,
                       "sol_ids": s, "answer": r["answer"],
                       "inter_ids": inter_ids})
    return out


# ---------------------------------------------------------------------------
# Model bootstrap — mirrors latent_arith_real.py's train() setup.
# ---------------------------------------------------------------------------

def load_eval_model(ckpt_path: str, device: str, bf16: bool = True,
                    tf32: bool = True):
    model, cfg = build_model_from_ckpt(ckpt_path)
    model = model.to(device).eval()
    # Clean latent thread, AMBIENT for the whole eval process (mirrors
    # latent_arith_real.py's train() setup, not the scoped clean_latent_thread
    # context — there is no combined-loss/backward hazard here to guard
    # against, this is pure no-grad eval).
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


# ---------------------------------------------------------------------------
# Core latent mechanics — bit-for-bit mirror of
# latent_reasoning_cotrain._answer_span_latent_loss's growing-thread loop
# (no_grad twin; skip_lm_head=True on the intermediate steps is a pure
# computation-skip — it does not touch the hidden states, only avoids the
# unused (B,T,V) vocab projection on the R-1 positions we never read logits
# from at that point, so it changes nothing about the mechanics being tested).
# ---------------------------------------------------------------------------

@torch.no_grad()
def grow_latent_thread(model, comment_ids: list[int], R: int, thinking_id: int,
                       device):
    """Return (cur_ids, cur_emb) of length P+R: prompt + R latent think slots,
    each fed back through `model.apply_latent_feedback_adapter`."""
    base_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long,
                           device=device)
    for _ in range(int(R)):
        with clean_latent_thread(model, film_bypass=True,
                                 no_activation_ckpt=True):
            h = model(cur_ids, inputs_embeds=cur_emb, skip_lm_head=True)
        if isinstance(h, tuple):
            h = h[0]
        z = h[:, -1:, :].to(cur_emb.dtype)
        z = model.apply_latent_feedback_adapter(z).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    return cur_ids, cur_emb


@torch.no_grad()
def final_forward_logits(model, cur_ids, cur_emb, sol_ids: list[int],
                         eos_id: int, device):
    """ONE forward over [prompt, R think slots, sol, eos]. Logits at position
    P+j-1 (j=1..R) are the per-hop read (causally identical whether or not the
    answer span is appended afterward — appending it cannot change any
    earlier position's hidden state); logits at P+R-1 .. P+R-1+len(sol)-1
    teacher-force the answer span (`_answer_span_latent_loss`'s exact
    `start = P + R - 1` convention)."""
    sol_t = torch.tensor([list(sol_ids) + [int(eos_id)]], dtype=torch.long,
                         device=device)
    full_ids = torch.cat([cur_ids, sol_t], dim=1)
    full_emb = torch.cat([cur_emb, model.embed(sol_t)], dim=1)
    with clean_latent_thread(model, film_bypass=True, no_activation_ckpt=True):
        out = model(full_ids, inputs_embeds=full_emb, skip_lm_head=False)
    logits = out[0] if isinstance(out, tuple) else out
    return logits


@torch.no_grad()
def eval_record_teacher_forced(model, ex: dict, R: int, thinking_id: int,
                               eos_id: int, device) -> dict:
    """Teacher-forced arm for one example at depth R: per-hop argmax vs
    intermediates[j-1] for j=1..R, and answer-span exact match."""
    P = len(ex["comment_ids"])
    cur_ids, cur_emb = grow_latent_thread(model, ex["comment_ids"], R,
                                         thinking_id, device)
    logits = final_forward_logits(model, cur_ids, cur_emb, ex["sol_ids"],
                                  eos_id, device)
    inter = ex["inter_ids"]
    perhop = []
    for j in range(1, R + 1):
        pos = P + j - 1
        pred = int(logits[0, pos, :].argmax().item())
        tgt = inter[j - 1] if (j - 1) < len(inter) else None
        perhop.append(None if tgt is None else int(pred == tgt))
    ans_positions = list(range(P + R - 1, P + R - 1 + len(ex["sol_ids"])))
    preds = [int(logits[0, p, :].argmax().item()) for p in ans_positions]
    tf_exact = int(preds == list(ex["sol_ids"]))
    return {"tf_exact": tf_exact, "perhop": perhop, "tf_pred": preds}


@torch.no_grad()
def eval_record_greedy(model, ex: dict, R: int, thinking_id: int, eos_id: int,
                       device) -> dict:
    """Greedy-decode arm via the real inference codepath
    (`eval_humaneval.generate_latent_think`, `force_prefix_think=R`) — exactly
    `latent_arith_real._eval_rung`'s convention (emit_threshold=0.0 for the
    R=0 no-think control so it is a TRUE no-think baseline regardless of any
    gate; irrelevant for R>0 since force_prefix_think ignores the gate then)."""
    prompt_ids = torch.tensor([ex["comment_ids"]], dtype=torch.long,
                              device=device)
    plen = prompt_ids.shape[1]
    out, diag = generate_latent_think(
        model, prompt_ids, max_gen=len(ex["sol_ids"]), temperature=0.0,
        eos_token_id=eos_id, thinking_token_id=thinking_id,
        force_prefix_think=R, emit_threshold=(0.0 if R == 0 else 0.5))
    new = [t for t in out[0, plen:].tolist() if t != thinking_id]
    pred = new[:len(ex["sol_ids"])]
    exact = int(pred == list(ex["sol_ids"]))
    return {"greedy_exact": exact, "greedy_pred": pred,
           "think_total": diag.get("think_total", 0)}


# ---------------------------------------------------------------------------
# Structural-floor check: for an adapter-only / frozen-trunk ckpt, the R=0
# (no latent steps ever inserted) forward must be BYTE-IDENTICAL to the
# unmodified base ckpt on the same prompts, since the adapter module is never
# invoked when no think positions exist and eval-mode never touches it either
# (the DDP dummy-zero-touch in model.py's _finalize is training-mode only).
# ---------------------------------------------------------------------------

@torch.no_grad()
def structural_floor_check(ckpt_path: str, base_ckpt_path: str,
                           token_id_lists: list[list[int]],
                           device: str, bf16: bool = True, tf32: bool = True) -> dict:
    """R=0 (no latent steps) forward from `ckpt_path` vs the unmodified
    `base_ckpt_path`, on the SAME token-id sequences (passed as raw ids, not
    decoded text, so there is no lossy decode/re-encode round-trip)."""
    if not pathlib.Path(base_ckpt_path).exists():
        return {"skipped": True, "reason": f"{base_ckpt_path} not found"}
    m_ckpt, _cfg_ckpt, _tid_ckpt, _tok_ckpt, _eos_ckpt = load_eval_model(
        ckpt_path, device, bf16=bf16, tf32=tf32)
    m_base, _cfg_base, _tid_base, _tok_base, _eos_base = load_eval_model(
        base_ckpt_path, device, bf16=bf16, tf32=tf32)
    max_delta = 0.0
    per_prompt = []
    for ids_list in token_id_lists:
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        l1 = m_ckpt(ids)
        l1 = l1[0] if isinstance(l1, tuple) else l1
        l2 = m_base(ids)
        l2 = l2[0] if isinstance(l2, tuple) else l2
        d = float((l1.float() - l2.float()).abs().max().item())
        per_prompt.append(d)
        max_delta = max(max_delta, d)
    del m_ckpt, m_base
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"skipped": False, "base_ckpt": base_ckpt_path,
           "n_prompts": len(token_id_lists), "per_prompt_max_abs_delta": per_prompt,
           "max_abs_delta": max_delta, "pass": max_delta == 0.0}


# ---------------------------------------------------------------------------
# Per-rung driver.
# ---------------------------------------------------------------------------

ARM_NAMES = ("no_think", "wrong_low", "latent_R=K", "wrong_high")


def _arms_for_rung(K: int) -> dict:
    return {"no_think": 0, "wrong_low": max(0, K // 2), "latent_R=K": K,
           "wrong_high": K + 4}


def eval_rung(model, recs: list[dict], K: int, thinking_id: int, eos_id: int,
             device, assert_tf_greedy_agree: bool = False) -> dict:
    arms = _arms_for_rung(K)
    res = {name: {"R": R, "n": 0, "tf_correct": 0, "greedy_correct": 0}
           for name, R in arms.items()}
    perhop_correct = 0
    perhop_total = 0
    perhop_by_step = [0] * K
    perhop_by_step_n = [0] * K
    disagreements = 0
    with _dedup_repeated_notice("[generate_latent_think] note:"):
        for ex in recs:
            for name, R in arms.items():
                tf = eval_record_teacher_forced(model, ex, R, thinking_id,
                                                eos_id, device)
                gr = eval_record_greedy(model, ex, R, thinking_id, eos_id,
                                        device)
                res[name]["n"] += 1
                res[name]["tf_correct"] += tf["tf_exact"]
                res[name]["greedy_correct"] += gr["greedy_exact"]
                if tf["tf_exact"] != gr["greedy_exact"]:
                    disagreements += 1
                    if assert_tf_greedy_agree:
                        raise AssertionError(
                            f"tf/greedy mismatch @ rung {K} arm {name} "
                            f"(R={R}) task {ex.get('task_id')}: "
                            f"tf_pred={tf['tf_pred']} vs greedy_pred="
                            f"{gr['greedy_pred']} gold={ex['sol_ids']}")
                if name == "latent_R=K":
                    for j, c in enumerate(tf["perhop"]):
                        if c is not None:
                            perhop_total += 1
                            perhop_correct += c
                            perhop_by_step_n[j] += 1
                            perhop_by_step[j] += c
    out = {"K": K, "n": len(recs), "arms": {}, "disagreements": disagreements}
    for name in ARM_NAMES:
        r = res[name]
        n = max(1, r["n"])
        out["arms"][name] = {
            "R": r["R"], "n": r["n"],
            "tf_exact": r["tf_correct"] / n,
            "greedy_exact": r["greedy_correct"] / n,
        }
    out["perhop_acc"] = (perhop_correct / perhop_total) if perhop_total else None
    out["perhop_by_step"] = [
        (perhop_by_step[j] / perhop_by_step_n[j]) if perhop_by_step_n[j] else None
        for j in range(K)]
    # Pre-registered kill-line: latent(R=K) - no-think >= 5pp on final-answer
    # exact (greedy decode = the deployment-realistic metric; teacher-forced
    # shown alongside for diagnostic parity — see module docstring, the two
    # are mathematically equivalent here since every answer is single-token).
    lift_greedy = out["arms"]["latent_R=K"]["greedy_exact"] - out["arms"]["no_think"]["greedy_exact"]
    lift_tf = out["arms"]["latent_R=K"]["tf_exact"] - out["arms"]["no_think"]["tf_exact"]
    out["kill_line"] = {
        "lift_pp_greedy": 100.0 * lift_greedy,
        "lift_pp_tf": 100.0 * lift_tf,
        "applies": K >= 4,
        "pass": (lift_greedy >= 0.05) if K >= 4 else None,
    }
    return out


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------

def print_table(title: str, rung_results: list[dict]):
    print(f"\n=== {title} ===")
    hdr = (f"{'K':>3} {'n':>4} | {'none_tf':>7} {'none_gr':>7} | "
          f"{'lowR_tf':>7} {'lowR_gr':>7} | {'R=K_tf':>7} {'R=K_gr':>7} | "
          f"{'hiR_tf':>7} {'hiR_gr':>7} | {'perhop':>7} | "
          f"{'lift_pp':>8} {'kill':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rung_results:
        a = r["arms"]
        kl = r["kill_line"]
        verdict = ("N/A" if kl["pass"] is None
                  else ("PASS" if kl["pass"] else "FAIL"))
        ph = f"{r['perhop_acc']:.3f}" if r["perhop_acc"] is not None else "  n/a"
        print(f"{r['K']:>3} {r['n']:>4} | "
              f"{a['no_think']['tf_exact']:>7.3f} {a['no_think']['greedy_exact']:>7.3f} | "
              f"{a['wrong_low']['tf_exact']:>7.3f} {a['wrong_low']['greedy_exact']:>7.3f} | "
              f"{a['latent_R=K']['tf_exact']:>7.3f} {a['latent_R=K']['greedy_exact']:>7.3f} | "
              f"{a['wrong_high']['tf_exact']:>7.3f} {a['wrong_high']['greedy_exact']:>7.3f} | "
              f"{ph:>7} | {kl['lift_pp_greedy']:>+8.2f} {verdict:>6}")
    print("  (lowR = R=K//2, hiR = R=K+4 -- depth-matching diagnostic; "
         "kill = 'latent(R=K) - no_think >= 5pp on greedy-exact, K>=4')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout_prefix", default="data/exec_trace_heldout")
    ap.add_argument("--lengen_prefix", default="data/exec_trace_lengen_heldout")
    ap.add_argument("--rungs", default="2,3,4,5,6,7,8")
    ap.add_argument("--lengen_rungs", default="9,10,12")
    ap.add_argument("--n_per_rung", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=900)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--floor_check_ckpt", default="checkpoints/feature_pilot_A.pt")
    ap.add_argument("--skip_floor_check", action="store_true")
    ap.add_argument("--floor_check_n", type=int, default=3)
    ap.add_argument("--assert_tf_greedy_agree", action="store_true",
                    help="hard-fail on any teacher-forced/greedy-decode "
                         "disagreement (both should be mathematically "
                         "equivalent for single-token answers -- a mismatch "
                         "signals a mechanics bug in the harness itself)")
    args = ap.parse_args()

    device = args.device
    t0 = time.time()
    model, cfg, thinking_id, tok, eos_id = load_eval_model(
        args.ckpt, device, bf16=not args.no_bf16, tf32=not args.no_tf32)
    n_params = model.num_params() if hasattr(model, "num_params") else -1
    print(f"[loaded] {args.ckpt} thinking_id={thinking_id} eos_id={eos_id} "
         f"state_readonly_at_think={getattr(model, 'state_readonly_at_think', None)} "
         f"use_latent_feedback_adapter={getattr(model, 'use_latent_feedback_adapter', False)} "
         f"use_memory={getattr(model, 'use_memory', False)} params={n_params:,}",
         flush=True)

    all_rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    lengen_rungs = [int(x) for x in args.lengen_rungs.split(",") if x.strip()]

    heldout_recs = {K: load_rung(args.heldout_prefix, K, tok, args.max_len)
                    for K in all_rungs}
    lengen_recs = {K: load_rung(args.lengen_prefix, K, tok, args.max_len)
                   for K in lengen_rungs}

    floor = {"skipped": True, "reason": "--skip_floor_check"}
    if not args.skip_floor_check:
        sample_recs = []
        for K in all_rungs:
            sample_recs.extend(heldout_recs.get(K, []))
            if len(sample_recs) >= args.floor_check_n:
                break
        id_lists = [ex["comment_ids"] for ex in sample_recs[:args.floor_check_n]]
        if id_lists:
            floor = structural_floor_check(
                args.ckpt, args.floor_check_ckpt, id_lists, device,
                bf16=not args.no_bf16, tf32=not args.no_tf32)
        else:
            floor = {"skipped": True, "reason": "no heldout prompts available"}
    print(f"\n=== STRUCTURAL FLOOR CHECK (R=0 vs {args.floor_check_ckpt}) ===")
    print(json.dumps(floor, indent=2))

    results = {"ckpt": args.ckpt, "config_subset": {
                  k: cfg.get(k) for k in (
                      "thinking_token_id", "state_readonly_at_think",
                      "use_memory", "feedback_mode", "arch", "n_layers",
                      "d_model", "vocab_size")},
              "structural_floor_check": floor, "heldout": [], "lengen": []}

    for K in all_rungs:
        recs = heldout_recs.get(K, [])[:args.n_per_rung]
        if not recs:
            print(f"  rung {K}: no heldout data at {args.heldout_prefix}_n{K}.jsonl -- skipped")
            continue
        r = eval_rung(model, recs, K, thinking_id, eos_id, device,
                     assert_tf_greedy_agree=args.assert_tf_greedy_agree)
        results["heldout"].append(r)
        print(f"  rung {K} done ({r['n']} recs, {time.time()-t0:.0f}s elapsed)",
             flush=True)
    print_table(f"HELDOUT ({args.heldout_prefix})", results["heldout"])

    for K in lengen_rungs:
        recs = lengen_recs.get(K, [])[:args.n_per_rung]
        if not recs:
            print(f"  lengen rung {K}: no data at {args.lengen_prefix}_n{K}.jsonl -- skipped")
            continue
        r = eval_rung(model, recs, K, thinking_id, eos_id, device,
                     assert_tf_greedy_agree=args.assert_tf_greedy_agree)
        results["lengen"].append(r)
        print(f"  lengen rung {K} done ({r['n']} recs, {time.time()-t0:.0f}s elapsed)",
             flush=True)
    print_table(f"LENGTH-GENERALIZATION ({args.lengen_prefix})", results["lengen"])

    results["runtime_s"] = time.time() - t0
    print(f"\n[done] total runtime {results['runtime_s']:.1f}s")

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
