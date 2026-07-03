"""Teacher-forced, FULL-FORWARD leak-free recall eval (2026-07-03).

Why this exists: `eval_recall_ours` generates via the incremental
`prefill`/`forward_step` path, but `TinyLM.prefill` deliberately runs the
FiLM-BYPASS forward — so a checkpoint trained WITH FiLM engaged (e.g. the
feature-pilot arm B) is evaluated as a DIFFERENT function than it trained
as, and its generated-recall numbers are an eval-path artifact (flat ~20%
at every distance, including in-window). This variant scores the SAME
leak-free first-occurrence tasks teacher-forced under the full training
forward instead: for each task, forward(prompt + gold) once and check the
argmax at every gold position (tf_exact) plus the gold total logprob.
Teacher-forced-but-leak-free: the gold value never appears in the PROMPT,
so predicting its tokens still requires recall across the full distance —
only the metric (argmax under forcing vs sampled generation) changes, and
it is applied identically to every arm.
"""
import argparse
import json

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.scoreboard_longctx_cost import prompt_pyout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--keys_per_task", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.to("cuda").eval()

    tasks = [json.loads(l) for l in open(args.tasks)]
    per_bucket = {}
    with torch.no_grad():
        for rec in tasks:
            L = rec["bucket"]
            pb = per_bucket.setdefault(L, [0, 0, 0])   # tf_exact, top1, total
            for key in rec["keys"][:args.keys_per_task]:
                # Identical elicitation to the generated-recall eval
                # (scoreboard prompt_pyout + answers field).
                prompt = prompt_pyout(rec["body"], key)
                gold = str(rec["answers"][key])
                p_ids = torch.tensor(
                    [tok.encode(prompt, add_special_tokens=False)],
                    device="cuda")
                g_ids = torch.tensor(
                    [tok.encode(gold, add_special_tokens=False)],
                    device="cuda")
                full = torch.cat([p_ids, g_ids], dim=1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(full)
                logits = out[0] if isinstance(out, tuple) else out
                # logits[t] predicts full[t+1]; gold spans positions
                # [P, P+G) so their predictors are logits[P-1 .. P+G-2].
                P, G = p_ids.shape[1], g_ids.shape[1]
                pred = logits[0, P - 1:P + G - 1].argmax(-1)
                tgt = full[0, P:P + G]
                pb[0] += int(bool((pred == tgt).all().item()))
                pb[1] += int(pred[0].item() == tgt[0].item())
                pb[2] += 1
    res = {str(L): {"tf_exact": v[0] / v[2], "top1": v[1] / v[2], "n": v[2]}
           for L, v in sorted(per_bucket.items())}
    json.dump({"ckpt": args.ckpt, "recall_tf": res}, open(args.out, "w"),
              indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
