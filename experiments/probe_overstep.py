"""R-sweep diagnostic: at a FIXED task-depth n, vary the latent depth R.

The deep-dep probe showed R=n -> ~1.00 but the autonomous gate (which chose
~22 latent steps for n=3..6 tasks) collapsed to ~0.10. Two mechanisms could
explain that, and they need different fixes:

  (a) over-stepping DRIFTS the answer: extra latent steps past the fixed point
      corrupt the carried pointer  -> accuracy should DROP as R grows past n.
      Fix = teach the gate to HALT earlier.
  (b) the gate halts at the WRONG position (sane step count, bad placement)
      -> a clean fixed point would hold: accuracy ~constant as R grows past n.
      Fix = different (positional) supervision, not just step-count.

This sweeps R in {n, n+2, n+5, n+9, n+13, ~22} on the trace heldout for a few
fixed n and prints accuracy at each R. Reuses latent_arith_real's loaders/eval.
"""
import argparse
import torch

from experiments.eval_bracket_structure import build_model_from_ckpt
from transformers import AutoTokenizer
import experiments.latent_arith_real as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/latent_trace_lineselect.pt")
    ap.add_argument("--heldout_prefix", default="data/trace_heldout")
    ap.add_argument("--task_ns", default="3,5")
    ap.add_argument("--r_offsets", default="0,2,5,9,13,19")
    ap.add_argument("--max_problems", type=int, default=150)
    ap.add_argument("--max_gen", type=int, default=12)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # mirror latent_arith_real eval_only build (no line_selector, no_memory)
    model, cfg = build_model_from_ckpt(
        args.base, force_state_readonly=True,
        force_use_latent_feedback_adapter=True)
    model = model.to(device).train()
    model._film_bypass = True
    if getattr(model, "use_memory", False):
        model.use_memory = False
        print("  [no_memory] WorkingMemory DISABLED — clean latent feedback", flush=True)
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    L.model_tok = tok
    eos_id = tok.eos_token_id
    print(f"[overstep] base={args.base} thinking_id={thinking_id} "
          f"adapter={bool(getattr(model,'use_latent_feedback_adapter',False))} "
          f"film={cfg.get('feedback_mode')} params={model.num_params():,}", flush=True)

    task_ns = [int(x) for x in args.task_ns.split(",") if x.strip()]
    offsets = [int(x) for x in args.r_offsets.split(",") if x.strip()]

    for n in task_ns:
        recs = L._load_rung(args.heldout_prefix, n, tok, args.max_len)
        if not recs:
            print(f"n={n}: (no heldout)", flush=True)
            continue
        print(f"\n=== task-depth n={n}  ({min(len(recs),args.max_problems)} problems) ===", flush=True)
        print(f"{'R':>4} {'acc':>7}   (none=no-think baseline below)", flush=True)
        none_acc = None
        for off in offsets:
            R = n + off
            res = L._eval_rung(model, recs, R, thinking_id, eos_id, device,
                               max_gen=args.max_gen, max_problems=args.max_problems)
            none_acc = res["none"]
            tag = "  <- R=n" if off == 0 else ""
            print(f"{R:>4} {res['r']:>7.3f}{tag}", flush=True)
        print(f"none {none_acc:>7.3f}  (R=0, gate forced to emit)", flush=True)


if __name__ == "__main__":
    main()
