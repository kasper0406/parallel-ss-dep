"""Probe the K=2 latent-answer anomaly (paper §5 open anomaly).

Stage-B eval: latent R=K answer at K=2 reads 0.140 — BELOW the direct
baseline (0.2567) and far below its own per-hop decode (0.925) — while
K=3-6 sit at 0.59-0.63. This probe re-runs the K=2 rung capturing full
transcripts and classifies every failure:

  - format:   parse_final returned None (emission/format failure)
  - hop1:     emitted number == intermediates[0] (answered the FIRST hop —
              an off-by-one/came-up-short signature)
  - hop2ok:   emitted number != answer but slot-2 perhop read WAS correct
              (state recovered, emission diverges from own state)
  - other:    wrong number, not explained above

Also cross-tabs failure vs per-hop correctness and prints sample transcripts.
Reuses the committed eval harness functions verbatim (no re-implementation).
"""

import argparse
import collections
import json

import torch

from experiments.eval_exec_trace_latent_trace import (
    load_eval_model,
    latent_greedy_answer,
    latent_perhop_reads,
    encode_inter_token_ids,
)
from experiments.eval_exec_trace_text import build_prompts, parse_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/stageB_latent_trace.pt")
    ap.add_argument("--heldout_prefix", default="data/exec_trace_heldout")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--grace", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_samples", type=int, default=12,
                    help="sample transcripts to print per failure class")
    ap.add_argument("--out", default="runs/probe_k2_anomaly.json")
    args = ap.parse_args()

    model, cfg, thinking_id, tok, eos_id = load_eval_model(
        args.ckpt, device=args.device)

    path = f"{args.heldout_prefix}_n{args.K}.jsonl"
    recs = [json.loads(l) for l in open(path)][: args.n]
    print(f"[probe] {len(recs)} records from {path}; ckpt={args.ckpt}")

    per, classes = [], collections.Counter()
    for i, rec in enumerate(recs):
        answer = rec.get("answer")
        inter = list(rec.get("intermediates", []))
        prompt_ids = tok.encode(build_prompts(rec)[0], add_special_tokens=False)

        txt = latent_greedy_answer(model, prompt_ids, args.K, thinking_id, tok,
                                   args.budget, args.device, args.grace)
        pred = parse_final(txt)
        perhop = latent_perhop_reads(model, prompt_ids, args.K, thinking_id,
                                     encode_inter_token_ids(tok, inter),
                                     args.device)

        if pred is not None and pred == answer:
            cls = "correct"
        elif pred is None:
            cls = "format"
        elif len(inter) >= 1 and pred == inter[0] and inter[0] != answer:
            cls = "hop1"
        elif len(perhop) >= args.K and perhop[-1] == 1:
            cls = "hop2ok_emit_diverges"
        else:
            cls = "other"
        classes[cls] += 1
        # The last latent slot's logits are ALSO the first emission step
        # (unshifted per-hop convention) — check whether the first emitted
        # token is the answer itself (objective-collision signature).
        first_tok = tok.decode(
            tok.encode(txt, add_special_tokens=False)[:1]) if txt else ""
        first_is_ans = int(first_tok.strip() == str(answer))
        per.append({"task_id": rec.get("task_id"), "answer": answer,
                    "intermediates": inter, "pred": pred, "class": cls,
                    "perhop": perhop, "first_tok_is_answer": first_is_ans,
                    "text": txt})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(recs)}  {dict(classes)}")

    n = len(per)
    print("\n=== K=%d failure taxonomy (n=%d) ===" % (args.K, n))
    for cls, c in classes.most_common():
        print(f"  {cls:24s} {c:4d}  ({100*c/n:.1f}%)")

    # cross-tab: emitted-correct vs slot-K perhop-correct
    both = sum(1 for r in per if r["class"] == "correct"
               and len(r["perhop"]) >= args.K and r["perhop"][-1] == 1)
    hop_ok = sum(1 for r in per
                 if len(r["perhop"]) >= args.K and r["perhop"][-1] == 1)
    print(f"\nslot-{args.K} perhop correct: {hop_ok}/{n} "
          f"({100*hop_ok/n:.1f}%); of those, answer also correct: "
          f"{both}/{hop_ok}" if hop_ok else "no perhop reads")
    fta = sum(r["first_tok_is_answer"] for r in per)
    print(f"first emitted token == answer: {fta}/{n} ({100*fta/n:.1f}%) "
          f"[objective-collision signature: last-slot logits are both the "
          f"per-hop answer decode AND the first emission step]")

    for cls in [c for c in classes if c != "correct"]:
        print(f"\n--- sample transcripts: {cls} ---")
        for r in [r for r in per if r["class"] == cls][: args.n_samples]:
            print(f"  ans={r['answer']} inter={r['intermediates']} "
                  f"pred={r['pred']} perhop={r['perhop']} "
                  f"text={r['text']!r}")

    json.dump({"ckpt": args.ckpt, "K": args.K, "n": n,
               "classes": dict(classes), "records": per},
              open(args.out, "w"), indent=1)
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
