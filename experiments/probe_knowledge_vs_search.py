"""The crux of WHY thinking/memory don't help MBPP: is failure a KNOWLEDGE gap
or a SEARCH/decoding gap?

For each problem, compare under the model (teacher-forced, no generation needed
for the gold side):
  logp_gold = mean per-token log-prob of the GOLD reference solution
  logp_own  = mean per-token log-prob of the model's OWN greedy completion

On FAILED problems:
  - logp_gold > logp_own  → the model RANKS the correct solution above its own
    wrong one: the knowledge is LATENT, the failure is search/decoding —
    thinking / sampling / RL HAS real headroom (it just isn't reaching the
    solution it already prefers).
  - logp_gold < logp_own  → the model genuinely BELIEVES its wrong answer is
    more likely than the correct one: a true KNOWLEDGE gap — no inference-time
    mechanism (depth, memory, recall) can supply an algorithm the weights don't
    contain. Only scale / more training data fixes it.

This is the experiment that says whether the marginal benefit is "we built the
wrong mechanism" (search gap, fixable) or "wrong bottleneck" (knowledge gap,
needs scale).

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_knowledge_vs_search.py checkpoints/latent_code_adapteronly.pt 120
"""
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate
import experiments.code_grader as CG


def mean_logp(model, prompt_ids, comp_ids, device):
    """Mean per-token logp of comp_ids given prompt_ids (teacher-forced)."""
    ids = torch.tensor([prompt_ids + comp_ids], device=device)
    with torch.no_grad():
        out = model(ids)
        logits = (out[0] if isinstance(out, tuple) else out)[0].float()
    lp = F.log_softmax(logits, -1)
    P = len(prompt_ids)
    tot = 0.0
    for k in range(len(comp_ids)):
        tot += lp[P + k - 1, comp_ids[k]].item()
    return tot / max(1, len(comp_ids))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latent_code_adapteronly.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    m, cfg = build_model_from_ckpt(ckpt, force_state_readonly=True)
    m = m.to(device).eval()
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id
    probs = [p for p in CG.LOADERS["mbpp_combined"]()[:n] if p.gold_solution]
    comment = "# Complete the following Python function.\n"
    print(f"[know-vs-search] ckpt={ckpt} n={len(probs)} (with gold)", flush=True)

    know_latent = search_gap = 0   # failed & gold>own  (search-bound)
    know_gap = 0                   # failed & gold<own  (knowledge-bound)
    n_pass = 0
    dgold_fail, down_fail = [], []
    for i, prob in enumerate(probs):
        pids = tok.encode(comment + prob.prompt, add_special_tokens=False)
        # model's own greedy completion + grade
        with torch.no_grad():
            gen, _ = generate(m, torch.tensor([pids], device=device), max_gen=200,
                              temperature=0.0, eos_token_id=eos, min_emit_before_eos=10)
        own_ids = [t for t in gen[0, len(pids):].tolist() if t != tid]
        own_code = tok.decode(own_ids, skip_special_tokens=True)
        lines = own_code.split("\n"); st = 0
        for k, l in enumerate(lines):
            if re.match(r"^(def |import |from |class |@)", l):
                st = k; break
        body = "\n".join(lines[st:])
        names = re.findall(r"^def (\w+)", body, re.M)
        if names and prob.entry_point not in names:
            body = body + f"\n{prob.entry_point} = {names[0]}\n"
        passed = CG.grade(prob, body).passed
        if passed:
            n_pass += 1
            continue
        # logp of gold vs own (re-encode own body for a fair comparison)
        gold_ids = tok.encode(prob.gold_solution, add_special_tokens=False)[:200]
        own_enc = tok.encode(body, add_special_tokens=False)[:200]
        if not gold_ids or not own_enc:
            continue
        lg = mean_logp(m, pids, gold_ids, device)
        lo = mean_logp(m, pids, own_enc, device)
        dgold_fail.append(lg); down_fail.append(lo)
        if lg > lo:
            search_gap += 1
        else:
            know_gap += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(probs)}] pass={n_pass} search_gap={search_gap} "
                  f"know_gap={know_gap}", flush=True)

    nf = search_gap + know_gap
    import statistics as S
    print(f"\n=== Knowledge vs Search on FAILED problems (failures={nf}) ===")
    print(f"  pass                          = {n_pass}")
    print(f"  SEARCH-bound (logp_gold>own)  = {search_gap}  ({search_gap/max(1,nf):.1%})"
          f"  <- knowledge latent; thinking/sampling/RL HAS headroom")
    print(f"  KNOWLEDGE-bound (logp_gold<own)= {know_gap}  ({know_gap/max(1,nf):.1%})"
          f"  <- model prefers its wrong answer; needs SCALE, not thinking")
    if dgold_fail:
        print(f"  mean logp_gold={S.fmean(dgold_fail):+.3f}  "
              f"mean logp_own={S.fmean(down_fail):+.3f}  "
              f"(gold-own={S.fmean(dgold_fail)-S.fmean(down_fail):+.3f})")
    print(f"\nVERDICT: SEARCH-bound dominant → build better search/thinking/RL. "
          f"KNOWLEDGE-bound dominant → the mechanisms can't help MBPP; the "
          f"bottleneck is base knowledge at 287M (scale/data).")


if __name__ == "__main__":
    main()
