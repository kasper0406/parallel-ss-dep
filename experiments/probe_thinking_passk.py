"""Execution-grounded headroom: does latent thinking EXPAND the solvable set
under sampling — the headroom gate-driven RL would exploit — or is it capped?

Per-token Δlogp probes can't see pass@1 pivotality. This generates K temperature
samples per problem with thinking-ON (gate-driven latent) and thinking-OFF
(no-think), grades all, and reports the set difference that matters for RL:

  THINK-ONLY solves   = problems some think-sample passes but NO no-think sample does
  NOTHINK-ONLY solves = the reverse (thinking broke it)

If THINK-ONLY >> NOTHINK-ONLY, gate-driven RL has real exploitable headroom on
code (run latent_rl.py on the co-trained base). If ≈ or worse, latent thinking's
code ceiling is genuinely ~neutral and the honest move is to stop polishing it.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_thinking_passk.py checkpoints/latent_code_adapteronly.pt 80 6
"""
import sys

import torch

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate, generate_latent_think
from experiments.latent_rl import grade_clean
from experiments.thinking import clean_latent_thread
import experiments.code_grader as CG


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latent_code_adapteronly.pt"
    n_prob = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    temp = 0.8
    think_thr = 0.5
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    m, cfg = build_model_from_ckpt(ckpt, force_state_readonly=True,
                                   force_use_latent_feedback_adapter=True)
    m = m.to(device).eval()
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id
    probs = CG.LOADERS["mbpp_combined"]()[off:off + n_prob]
    comment = "# Complete the following Python function.\n"
    print(f"[passk] ckpt={ckpt} n={len(probs)} K={K} temp={temp} thr={think_thr}",
          flush=True)

    def sample(prob, thinking):
        cids = tok.encode(comment + prob.prompt, add_special_tokens=False)
        pt = torch.tensor([cids], device=device)
        with torch.no_grad():
            if thinking:
                with clean_latent_thread(m):
                    gen, _ = generate_latent_think(
                        m, pt, max_gen=200, temperature=temp, eos_token_id=eos,
                        thinking_token_id=tid, max_think_per_step=8,
                        total_think_budget=400, emit_threshold=think_thr,
                        min_emit_before_eos=10, gate_floor=0.0)
            else:
                gen, _ = generate(m, pt, max_gen=200, temperature=temp,
                                  eos_token_id=eos, min_emit_before_eos=10)
        out = gen[0, len(cids):].tolist()
        code = tok.decode([t for t in out if t != tid], skip_special_tokens=True)
        return grade_clean(prob, code).passed

    nt_only, th_only, both, neither = [], [], 0, 0
    nt_solved_n = th_solved_n = 0
    for i, prob in enumerate(probs):
        nt = any(sample(prob, False) for _ in range(K))
        th = any(sample(prob, True) for _ in range(K))
        nt_solved_n += int(nt)
        th_solved_n += int(th)
        if nt and th:
            both += 1
        elif nt and not th:
            nt_only.append(prob.task_id)
        elif th and not nt:
            th_only.append(prob.task_id)
        else:
            neither += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(probs)}] nt_solved={nt_solved_n} "
                  f"th_solved={th_solved_n} th_only={len(th_only)} "
                  f"nt_only={len(nt_only)}", flush=True)

    print(f"\n=== pass@{K} solvable-set diff (n={len(probs)}) ===")
    print(f"  no-think pass@{K} = {nt_solved_n}")
    print(f"  thinking pass@{K} = {th_solved_n}")
    print(f"  BOTH solve        = {both}")
    print(f"  THINK-ONLY solves = {len(th_only)}  (RL headroom)  ids={th_only}")
    print(f"  NOTHINK-ONLY      = {len(nt_only)}  (thinking broke)  ids={nt_only}")
    print(f"  neither           = {neither}")
    print(f"\nVERDICT: THINK-ONLY >> NOTHINK-ONLY → gate-driven RL has headroom "
          f"(run latent_rl.py on this base). Else code-thinking ceiling ~neutral.")


if __name__ == "__main__":
    main()
