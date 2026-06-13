"""KILL-TEST: can NON-PARAMETRIC retrieval (kNN-LM) override the model's
CONFIDENTLY-WRONG generation on real MBPP — or does the consumption barrier kill
it the way it killed parametric memory?

Non-parametric retrieval structurally escapes the catastrophic-forgetting wall
(append to a datastore, no gradient). But the validation agent flagged the real
risk: the model rates its OWN WRONG answer ~4.6x over gold on 100% of failures,
its hidden encodes the wrong algorithm (-> confirmatory retrieval), and failures
are GLOBAL (-> per-token interpolation may not flip them). So before building any
real datastore / FAISS / latent-query coupling, test the UPPER BOUND:

  Datastore = the GOLD solutions of the very failing problems (answer literally
  retrievable). Generate with kNN-LM interpolation. If even this ORACLE datastore
  can't flip materially more than baseline, the consumption barrier is fatal and
  no realistic datastore will rescue it -> stop the kNN direction. If it flips a
  meaningful fraction, consumption works -> green-light real datastore + latent
  query formation.

Design notes from the agent: gate lambda on RETRIEVAL confidence (distance to
nearest neighbor), NOT fixed-lambda or LM-uncertainty (both weakest vs
confident-wrong). Key = last pre-lm_head hidden (return_hidden); value = next
token.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_knn_oracle.py checkpoints/sft_baked_pure.pt 40
"""
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.thinking import _logits_hidden
import experiments.code_grader as CG

DEVICE = "cuda"


def clean_grade(prob, code):
    lines = code.split("\n"); st = 0
    for k, l in enumerate(lines):
        if re.match(r"^(def |import |from |class |@)", l):
            st = k; break
    body = "\n".join(lines[st:])
    names = re.findall(r"^def (\w+)", body, re.M)
    if names and prob.entry_point not in names:
        body = body + f"\n{prob.entry_point} = {names[0]}\n"
    return CG.grade(prob, body).passed


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_baked_pure.pt"
    n_target = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    lam_max = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    # datastore: "oracle" = gold of the eval problems (upper bound, leaky);
    # "realistic" = gold of a DISJOINT set of problems (near-neighbor retrieval).
    ds_mode = sys.argv[4] if len(sys.argv) > 4 else "oracle"
    n_ds = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    conf_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 0.15
    topk = 8
    temp_knn = 10.0          # softmax temp on -distance (cosine in [0,2])
    # conf_scale: retrieval-confidence gate lam_eff = lam_max*exp(-d_min/scale).
    # Small (0.15) = only near-exact matches contribute (right for oracle);
    # large (>=1.0) = let distant near-neighbors contribute too (tests whether
    # cross-problem similar solutions are usable at all).
    torch.backends.cuda.matmul.allow_tf32 = True
    m, cfg = build_model_from_ckpt(ckpt, force_state_readonly=True)
    m = m.to(DEVICE).eval()
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id
    comment = "# Complete the following Python function.\n"
    probs_all = CG.LOADERS["mbpp_combined"]()

    def hidden_logits(ids):
        with torch.no_grad():
            out = m(ids, return_hidden=True)
        lg, h = _logits_hidden(out)
        return lg, h          # (1,L,V), (1,L,d)

    def greedy(prob, datastore=None, dmin_out=None):
        cids = tok.encode(comment + prob.prompt, add_special_tokens=False)
        ids = torch.tensor([cids], device=DEVICE)
        for _ in range(200):
            lg, h = hidden_logits(ids)
            p_lm = F.softmax(lg[0, -1].float(), -1)
            if datastore is not None:
                q = F.normalize(h[0, -1:].float(), dim=-1)      # (1,d)
                K, vals = datastore
                sim = (q @ K.T.float()).squeeze(0)              # (M,) cosine
                tv, ti = torch.topk(sim, min(topk, sim.numel()))
                w = F.softmax(tv * temp_knn, dim=-1)            # sharper on closer neighbors
                p_knn = torch.zeros_like(p_lm)
                p_knn.scatter_add_(0, vals[ti], w)
                d_min = (1.0 - tv[0]).item()
                if dmin_out is not None:
                    dmin_out.append(d_min)
                lam = lam_max * pow(2.718281828, -d_min / conf_scale)
                p = lam * p_knn + (1 - lam) * p_lm
            else:
                p = p_lm
            nxt = int(p.argmax())
            if nxt == tid:
                lg2 = lg[0, -1].float().clone(); lg2[tid] = -1e9
                nxt = int(lg2.argmax())
            ids = torch.cat([ids, torch.tensor([[nxt]], device=DEVICE)], 1)
            if nxt == eos:
                break
        out = [t for t in ids[0, len(cids):].tolist() if t != tid]
        return tok.decode(out, skip_special_tokens=True)

    # 1) find failing problems in an eval window (first 200); reserve the rest
    #    as a disjoint datastore pool for the realistic test.
    eval_window = probs_all[:200]
    pool = probs_all[200:]
    fails = []
    for prob in eval_window:
        if not prob.gold_solution:
            continue
        if not clean_grade(prob, greedy(prob)):
            fails.append(prob)
        if len(fails) >= n_target:
            break
    fail_ids = {p.task_id for p in fails}
    print(f"[knn-oracle] ckpt={ckpt} mode={ds_mode} failing={len(fails)} "
          f"lam_max={lam_max} topk={topk} conf_scale={conf_scale}", flush=True)

    # 2) build datastore.
    #   oracle    : the failing problems' own gold (leaky upper bound)
    #   realistic : gold of n_ds DISJOINT MBPP problems
    #   corpus    : gold of n_ds problems from a LARGE EXTERNAL corpus
    #               (distill_corpus 147k / codefeedback / magicoder) — the
    #               coverage test: does a big diverse store give near-exact
    #               matches for held-out MBPP? (the agent-recommended pivot)
    if ds_mode in ("corpus", "corpus_clean"):
        src = CG.LOADERS["distill_corpus"]()
        cand = [p for p in src if p.gold_solution]
        if ds_mode == "corpus_clean":
            # DE-LEAK: distill_corpus (magicoder/codefeedback) CONTAINS the exact
            # MBPP problems — exclude any entry whose normalized prompt matches an
            # eval problem OR whose gold contains the eval gold's prefix. Only then
            # is "coverage" honest (similar, not identical, solutions).
            import re as _re
            def _norm(s):
                return _re.sub(r"\s+", " ", (s or "")).strip().lower()
            ev_prompts = {_norm(p.prompt) for p in fails}
            ev_golds = [_norm(p.gold_solution)[:80] for p in fails if p.gold_solution]
            kept = []
            for p in cand:
                if _norm(p.prompt) in ev_prompts:
                    continue
                g = _norm(p.gold_solution)
                if any(eg and eg in g for eg in ev_golds):
                    continue
                kept.append(p)
                if len(kept) >= n_ds:
                    break
            print(f"[knn-oracle] de-leak: {len(cand)} cand -> kept {len(kept)} "
                  f"(removed exact-prompt/gold matches to eval)", flush=True)
            ds_probs = kept
        else:
            ds_probs = cand[:n_ds]
    elif ds_mode == "realistic":
        ds_probs = [p for p in pool if p.gold_solution
                    and p.task_id not in fail_ids][:n_ds]
    else:
        ds_probs = fails
    keys, values = [], []
    for prob in ds_probs:
        pre = tok.encode(comment + prob.prompt, add_special_tokens=False)
        gold = tok.encode("\n" + prob.gold_solution.replace("\r\n", "\n"),
                          add_special_tokens=False)
        ids = torch.tensor([(pre + gold)[:320]], device=DEVICE)
        _lg, h = hidden_logits(ids)
        P = len(pre)
        for t in range(P - 1, ids.shape[1] - 1):
            keys.append(F.normalize(h[0, t].float(), dim=-1).to(torch.bfloat16))
            values.append(ids[0, t + 1].item())
    K = torch.stack(keys)                               # (M,d) bf16, pre-normed
    vals = torch.tensor(values, device=DEVICE)
    print(f"[knn-oracle] datastore keys={K.shape[0]} from {len(ds_probs)} solutions",
          flush=True)

    # 3) eval baseline vs kNN; log d_min coverage (fraction of steps with a
    #    near-exact neighbor — the oracle band is d_min < 0.1).
    base_pass = knn_pass = 0
    flipped = []
    dmins = []
    for prob in fails:
        b = clean_grade(prob, greedy(prob, datastore=None))
        k = clean_grade(prob, greedy(prob, datastore=(K, vals), dmin_out=dmins))
        base_pass += int(b); knn_pass += int(k)
        if k and not b:
            flipped.append(prob.task_id)
    import statistics as S
    dm = torch.tensor(dmins) if dmins else torch.zeros(1)
    sharp = (dm < 0.1).float().mean().item()
    print(f"\n=== kNN-LM coverage test (mode={ds_mode}, n={len(fails)} failing) ===")
    print(f"  baseline pass            = {base_pass}/{len(fails)}")
    print(f"  kNN pass                 = {knn_pass}/{len(fails)}  (+{knn_pass-base_pass})")
    print(f"  flipped fail->pass: {flipped}")
    print(f"  d_min: mean={dm.mean():.3f} median={dm.median():.3f}  "
          f"frac steps with near-exact (d_min<0.1) = {sharp:.3f}")
    print(f"\nVERDICT: pass climbing toward oracle (23/40) with rising near-exact "
          f"coverage => small-model + BIG datastore is a real strategy (scale the "
          f"store). Pass stuck ~0-2 with d_min never reaching the oracle band => "
          f"near-exact coverage unattainable at this scale; composition wall holds "
          f"=> spend compute on base capability, reserve thinking/WM for matched "
          f"bottlenecks (arith depth, long-context recall).")


if __name__ == "__main__":
    main()
