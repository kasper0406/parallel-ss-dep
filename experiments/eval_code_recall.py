"""Scorable recall eval for the REALISTIC code / agentic recall sets.

The measuring stick for "is the validated WM recall mechanism USEFUL on the
project's real workload (coding + agentic)". Drop-in sibling of
eval_longctx_recall.py, but:
  - answers are not just integers (identifiers, filenames, aliases, hashes), so a
    GENERAL `Answer: X` extractor is used + a lenient gold-substring fallback;
  - it reports recall as a CURVE over distance bucket AND stratified by family /
    n_bindings (the interference / capacity axis — the real WM headroom driver);
  - it offers the SAME fair arms used elsewhere: WM-ON vs WM-OFF (use_memory off,
    the honest full_off ablation) vs a NO-THINK control;
  - two modes:
      * generate      : greedy decode, training-matched prompt format (the
                        realistic deployment measure; format-dependent).
      * teacher_forced: per-answer-token argmax exact-match using the annotated
                        answer span (format-INDEPENDENT, the clean headroom
                        probe — this is what establishes "where does the
                        recurrence fail on real code", i.e. where WM must help).
        teacher_forced is what wm_namekey_probe/wm_multitok_readout use; it does
        not depend on the model knowing the `Answer:` output format, so it is the
        right tool for a dry-run on a PRETRAIN base.

Usage:
  # clean WM-OFF headroom curve on a pretrain base (recommended dry-run):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/eval_code_recall.py \
      --ckpt checkpoints/pretrain_v12_step13352_tok3500146688.pt \
      --tasks data/code_recall_heldout.jsonl --mode teacher_forced \
      --wm_arm full_off --max_problems 400

  # realistic generation arm:
  ... --mode generate --generator retrieval_as_input --wm_arm on
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

_ANS_RE = re.compile(r"Answer\s*:\s*(.+)", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """General answer extractor: the token-run after the first `Answer:`.
    Falls back to the last non-empty stripped line."""
    m = _ANS_RE.search(text)
    if m:
        cand = m.group(1).splitlines()[0]
        cand = cand.strip().strip("`").strip().rstrip(".,;:")
        # answers are single tokens/identifiers/filenames — take first ws run
        cand = cand.split()[0] if cand.split() else cand
        return cand.strip("`.,;:") or None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _norm(s: str) -> str:
    return s.strip().strip("`").strip().rstrip(".,;:")


def bucket_of(record: dict) -> int:
    m = re.search(r"/d(\d+)/", record.get("task_id", ""))
    if m:
        return int(m.group(1))
    return int(record.get("approx_distance_tokens", 0))


# ---- WM-OFF / no-think ablation (mirrors eval_longctx_recall._full_off) -------
@contextlib.contextmanager
def _full_off(model):
    orig = getattr(model, "use_memory", False)
    try:
        model.use_memory = False
        yield
    finally:
        model.use_memory = orig


def tokenize_with_span(tok, full_text: str, c0: int, c1: int):
    """Tokenize `full_text` and return (ids, t0, t1) where ids[t0:t1] are exactly
    the tokens overlapping the char span [c0, c1) — the answer tokens.

    Uses the fast tokenizer's offset mapping (exact, handles BPE boundaries such
    as a leading-space token absorbing the start of an identifier). Falls back to
    prefix-length counting if offsets are unavailable. A token whose char range
    overlaps [c0, c1) is included; this is the right unit for both teacher-forced
    exact-match (compare predicted id to the actual answer token id) and for the
    v14 mem_read_mask (mark the answer-token positions)."""
    try:
        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        idxs = [i for i, (a, b) in enumerate(offs) if a < c1 and b > c0]
        if idxs:
            return ids, idxs[0], idxs[-1] + 1
    except Exception:
        pass
    ids = tok.encode(full_text, add_special_tokens=False)
    n0 = len(tok.encode(full_text[:c0], add_special_tokens=False))
    n1 = len(tok.encode(full_text[:c1], add_special_tokens=False))
    return ids, n0, max(n1, n0 + 1)


def _char_to_tok_span(tok, full_text: str, c0: int, c1: int):
    """Back-compat shim returning just (t0, t1)."""
    _, t0, t1 = tokenize_with_span(tok, full_text, c0, c1)
    return t0, t1


def eval_teacher_forced(model, tok, records, *, wm_arm="on", device="cuda",
                        max_T=2048):
    """Per-answer-token argmax exact-match using the annotated answer span.
    Format-independent. Returns per-bucket and per-(family,N) accuracy."""
    abl = _full_off(model) if wm_arm == "full_off" else contextlib.nullcontext()
    per_bucket, per_fam, per_n, per_dn = {}, {}, {}, {}
    n_correct = n_total = n_skip = n_first = 0
    was_train = model.training
    model.eval()
    with torch.no_grad(), abl:
        for rec in records:
            prompt = rec["problem_prompt"] + "\n\n"
            comp = rec["qwen_completion"]
            full_text = prompt + comp
            ans_cs = rec.get("answer_char_span")
            if not ans_cs:
                n_skip += 1
                continue
            ans_c0 = len(prompt) + ans_cs[0]
            ans_c1 = len(prompt) + ans_cs[1]
            ids, t0, t1 = tokenize_with_span(tok, full_text, ans_c0, ans_c1)
            if len(ids) > max_T:
                n_skip += 1
                continue
            if t1 - t0 < 1 or t0 < 1:
                n_skip += 1
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            # v14: drive the WM read / copy at the PREDICTING positions of the
            # answer span ([t0-1, t1-1) — logits[p] predicts token p+1). This is
            # what makes the trained recall mechanism (embedding-key + copy head)
            # actually fire here; without a mem_read_mask the WM only reads at
            # think positions, which the eval prompt has none of (the documented
            # "recall probe structurally 0"). For wm_arm=full_off the WM/copy are
            # disabled regardless, so the mask is a harmless no-op there.
            rmask = torch.zeros((1, len(ids)), dtype=torch.long, device=device)
            rmask[0, max(0, t0 - 1):max(1, t1 - 1)] = 1
            out = model(x, mem_read_mask=rmask)
            logits = out[0] if isinstance(out, tuple) else out
            # predicting position p emits token p+1 → for answer token at index t,
            # the predicting logits are at t-1.
            span_ids = ids[t0:t1]
            ok = True
            first_ok = False
            for j, tgt in enumerate(span_ids):
                pred = int(logits[0, t0 - 1 + j].argmax().item())
                if j == 0:
                    first_ok = (pred == tgt)
                if pred != tgt:
                    ok = False
                    break
            b = bucket_of(rec)
            fam = rec.get("family", "?")
            nb = rec.get("n_bindings", 0)
            per_bucket.setdefault(b, [0, 0]); per_bucket[b][1] += 1; per_bucket[b][0] += int(ok)
            per_fam.setdefault(fam, [0, 0]); per_fam[fam][1] += 1; per_fam[fam][0] += int(ok)
            per_n.setdefault(nb, [0, 0]); per_n[nb][1] += 1; per_n[nb][0] += int(ok)
            per_dn.setdefault((b, nb), [0, 0]); per_dn[(b, nb)][1] += 1; per_dn[(b, nb)][0] += int(ok)
            n_total += 1; n_correct += int(ok); n_first += int(first_ok)
    if was_train:
        model.train()
    return dict(recall=n_correct / max(1, n_total), n_correct=n_correct,
                first_recall=n_first / max(1, n_total), n_first=n_first,
                n_total=n_total, n_skip=n_skip, per_bucket=per_bucket,
                per_family=per_fam, per_n=per_n,
                per_dn={f"d{b}_N{nb}": v for (b, nb), v in sorted(per_dn.items())})


def eval_generate(model, tok, records, *, wm_arm="on", generator="retrieval_as_input",
                  max_gen=40, total_think_budget=64, emit_threshold=0.5,
                  gate_floor=0.0, device="cuda", max_T=2048):
    """Greedy decode in training-matched format; strict (Answer:) + lenient
    (gold substring) recall vs distance. wm_arm in {on, full_off, no_think}."""
    from experiments.eval_humaneval import generate, generate_with_retrieval_as_input
    thinking_token_id = getattr(model, "thinking_token_id", None)
    additive = bool(getattr(model, "retrieval_input_additive", False))
    eff_budget = 0 if wm_arm == "no_think" else total_think_budget
    abl = _full_off(model) if wm_arm == "full_off" else contextlib.nullcontext()
    per_bucket, per_fam = {}, {}
    n_strict = n_lenient = n_total = n_skip = 0
    agg_think = agg_emit = 0
    was_train = model.training
    model.eval()
    with torch.no_grad(), abl:
        for rec in records:
            prompt = rec["problem_prompt"] + "\n\n"
            pids = tok.encode(prompt, add_special_tokens=False)
            if len(pids) + max_gen + eff_budget > max_T:
                n_skip += 1
                continue
            pt = torch.tensor([pids], dtype=torch.long, device=device)
            if generator == "retrieval_as_input" and hasattr(model, "memory") \
                    and thinking_token_id is not None and wm_arm != "no_think":
                gen, diag = generate_with_retrieval_as_input(
                    model, pt, max_gen=max_gen, temperature=0.0,
                    eos_token_id=tok.eos_token_id,
                    thinking_token_id=thinking_token_id,
                    total_think_budget=eff_budget, emit_threshold=emit_threshold,
                    gate_floor=gate_floor, additive=additive)
            else:
                gen, diag = generate(
                    model, pt, max_gen=max_gen, temperature=0.0,
                    eos_token_id=tok.eos_token_id,
                    use_thinking=(thinking_token_id is not None and wm_arm != "no_think"),
                    thinking_token_id=thinking_token_id,
                    total_think_budget=eff_budget, emit_threshold=emit_threshold,
                    gate_floor=gate_floor)
            tid = int(thinking_token_id) if thinking_token_id is not None else -1
            gen_only = [t for t in gen[0, len(pids):].tolist() if t != tid]
            text = tok.decode(gen_only, skip_special_tokens=True)
            gold = _norm(rec["answer"])
            pred = extract_answer(text)
            strict = (pred is not None and _norm(pred) == gold)
            lenient = gold in text
            b = bucket_of(rec); fam = rec.get("family", "?")
            per_bucket.setdefault(b, [0, 0, 0]); per_bucket[b][2] += 1
            per_bucket[b][0] += int(strict); per_bucket[b][1] += int(lenient)
            per_fam.setdefault(fam, [0, 0, 0]); per_fam[fam][2] += 1
            per_fam[fam][0] += int(strict); per_fam[fam][1] += int(lenient)
            n_total += 1; n_strict += int(strict); n_lenient += int(lenient)
            agg_think += diag.get("think_total", 0); agg_emit += diag.get("emit_count", 0)
    if was_train:
        model.train()
    think_rate = agg_think / max(1, agg_think + agg_emit)
    return dict(strict=n_strict / max(1, n_total), lenient=n_lenient / max(1, n_total),
                n_strict=n_strict, n_lenient=n_lenient, n_total=n_total,
                n_skip=n_skip, think_rate=think_rate,
                per_bucket=per_bucket, per_family=per_fam)


def _load(path, n=None):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs[:n] if n else recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", default="data/code_recall_heldout.jsonl")
    p.add_argument("--mode", choices=["generate", "teacher_forced"],
                   default="teacher_forced")
    p.add_argument("--wm_arm", choices=["on", "full_off", "no_think"], default="on")
    p.add_argument("--generator", default="retrieval_as_input",
                   choices=["standard", "retrieval_as_input"])
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--family", type=str, default=None,
                   help="restrict to one family (const/signature/import/fname/"
                        "toolout/userinstr/setvar) — clean per-family headroom")
    p.add_argument("--max_gen", type=int, default=40)
    p.add_argument("--total_think_budget", type=int, default=64)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_log", default=None)
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    print(f"[load] {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model.to(args.device).eval()
    if hasattr(model, "_latent_feedback_premem"):
        model._latent_feedback_premem = True
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    max_T = cfg.get("max_T", 2048) or 2048
    records = _load(args.tasks)
    if args.family:
        records = [r for r in records if r.get("family") == args.family]
    if args.max_problems:
        records = records[:args.max_problems]
    print(f"[tasks] {len(records)} from {args.tasks}  mode={args.mode} "
          f"wm_arm={args.wm_arm}" + (f" family={args.family}" if args.family else ""))

    if args.mode == "teacher_forced":
        r = eval_teacher_forced(model, tok, records, wm_arm=args.wm_arm,
                                device=args.device, max_T=max_T)
        print(f"\n{'='*64}\nTEACHER-FORCED exact recall — {pathlib.Path(args.ckpt).name}"
              f"  [wm_arm={args.wm_arm}]\n{'='*64}")
        print(f"{'distance':>10} {'recall':>8} {'(corr/tot)':>14}")
        for b in sorted(r["per_bucket"]):
            c, t = r["per_bucket"][b]
            print(f"{b:>10} {c/max(1,t)*100:>7.1f}% {f'{c}/{t}':>14}")
        cc, nt = r["n_correct"], r["n_total"]
        print(f"{'-'*36}\n{'OVERALL':>10} {r['recall']*100:>7.1f}% {f'{cc}/{nt}':>14}"
              f"   first-token={r['first_recall']*100:.1f}%")
        print("  by family: " + "  ".join(
            f"{k}={v[0]/max(1,v[1])*100:.0f}%({v[0]}/{v[1]})" for k, v in sorted(r["per_family"].items())))
        print("  by n_bindings(exact): " + "  ".join(
            f"N{k}={v[0]/max(1,v[1])*100:.0f}%" for k, v in sorted(r["per_n"].items())))
        print("  distance x N (exact recall):")
        for k, v in r["per_dn"].items():
            print(f"      {k:>10}: {v[0]/max(1,v[1])*100:5.0f}% ({v[0]}/{v[1]})")
        print(f"  skipped(too long/no-span)={r['n_skip']}")
        summary = r
    else:
        r = eval_generate(model, tok, records, wm_arm=args.wm_arm,
                          generator=args.generator, max_gen=args.max_gen,
                          total_think_budget=args.total_think_budget,
                          emit_threshold=args.emit_threshold,
                          gate_floor=args.gate_floor, device=args.device, max_T=max_T)
        print(f"\n{'='*64}\nGENERATE recall — {pathlib.Path(args.ckpt).name}"
              f"  [wm_arm={args.wm_arm} gen={args.generator}]\n{'='*64}")
        print(f"{'distance':>10} {'strict':>8} {'lenient':>9} {'(tot)':>7}")
        for b in sorted(r["per_bucket"]):
            s, l, t = r["per_bucket"][b]
            print(f"{b:>10} {s/max(1,t)*100:>7.1f}% {l/max(1,t)*100:>8.1f}% {t:>7}")
        print(f"{'-'*40}\n{'OVERALL':>10} {r['strict']*100:>7.1f}% {r['lenient']*100:>8.1f}%")
        print("  by family: " + "  ".join(
            f"{k}=s{v[0]/max(1,v[2])*100:.0f}%/l{v[1]/max(1,v[2])*100:.0f}%" for k, v in sorted(r["per_family"].items())))
        print(f"  think_rate={r['think_rate']:.3f}  skipped={r['n_skip']}")
        summary = r

    if args.out_log:
        with open(args.out_log, "w") as f:
            json.dump({"ckpt": args.ckpt, "tasks": args.tasks, "mode": args.mode,
                       "wm_arm": args.wm_arm,
                       **{k: v for k, v in summary.items()
                          if k not in ("per_bucket", "per_family", "per_n")},
                       "per_bucket": {str(k): v for k, v in summary["per_bucket"].items()},
                       "per_family": summary.get("per_family", {})}, f, indent=2)
        print(f"  summary -> {args.out_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
