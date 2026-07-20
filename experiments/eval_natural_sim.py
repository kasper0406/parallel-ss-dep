"""Internal gate for SEARCH_NATIVE revival attempt A: natural-trace heldout
final-answer exact-match (bar: >= 0.30 to proceed to the 3a re-run).

Each heldout record's `text` already contains the full rendered prompt
(program + x = f(args) + question + `# trace:` + gold trace + `# final:`).
We cut the prompt at the END of the `# trace:` line, generate with the same
executor decode used by the 3a probe, parse `# final:`, and compare to the
record's `final_repr` (value-equality with type guard, via the probe's
`answer_matches`).
"""

import argparse
import ast
import json

from experiments.repair_value_probe import (
    answer_matches,
    executor_generate_natural,
    parse_final_answer,
)


def split_prompt(text: str) -> tuple[str, str] | None:
    """Prompt = everything through the `# trace:` line (inclusive, with its
    newline); gold = the rest. None if the marker is absent."""
    marker = "# trace:\n"
    idx = text.find(marker)
    if idx < 0:
        return None
    cut = idx + len(marker)
    return text[:cut], text[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/executor_natural.pt")
    ap.add_argument("--heldout", default="data/natural_traces_heldout.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dedup", action="store_true", default=True,
                    help="score each unique program text once (default; the "
                         "corpus has ~18x multiplicity)")
    ap.add_argument("--max_gen", type=int, default=512)
    ap.add_argument("--out", default="runs/eval_natural_sim.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from experiments.eval_exec_trace_latent_trace import load_eval_model
    model, cfg, thinking_id, tok, eos_id = load_eval_model(
        args.ckpt, device=args.device)

    recs = [json.loads(l) for l in open(args.heldout)]
    if args.dedup:
        seen, uniq = set(), []
        for r in recs:
            key = r["problem_key"]
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        recs = uniq
    if args.limit:
        recs = recs[: args.limit]
    print(f"[gate] {len(recs)} unique heldout programs from {args.heldout}")

    n_match, per = 0, []
    for i, r in enumerate(recs):
        sp = split_prompt(r["text"])
        if sp is None:
            continue
        prompt, _gold = sp
        ids = tok.encode(prompt, add_special_tokens=False)
        gen, _alp = executor_generate_natural(model, tok, ids, args.max_gen,
                                              eos_id)
        parsed = parse_final_answer(gen)
        try:
            expected = ast.literal_eval(r["final_repr"])
        except (ValueError, SyntaxError):
            expected = r["final_repr"]
        match = answer_matches(parsed, expected)
        n_match += int(match)
        per.append({"task_id": r["task_id"], "expected": r["final_repr"],
                    "got": parsed["raw"], "match": match})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(recs)}  acc so far {n_match/(i+1):.3f}",
                  flush=True)

    acc = n_match / max(1, len(per))
    verdict = "PROCEED (>=0.30)" if acc >= 0.30 else "KILL (<0.30)"
    print(f"\n[gate] natural-heldout sim exact-match: {n_match}/{len(per)} "
          f"= {acc:.3f}  ->  {verdict}")
    json.dump({"ckpt": args.ckpt, "n": len(per), "acc": acc,
               "verdict": verdict, "per": per}, open(args.out, "w"), indent=1)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
