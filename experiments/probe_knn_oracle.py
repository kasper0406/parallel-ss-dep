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
    topk = 8
    temp_knn = 10.0          # softmax temp on -distance (cosine in [0,2])
    conf_scale = 0.15        # retrieval-confidence gate: lam_eff = lam_max*exp(-d_min/scale)
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

    def greedy(prob, datastore=None):
        cids = tok.encode(comment + prob.prompt, add_special_tokens=False)
        ids = torch.tensor([cids], device=DEVICE)
        for _ in range(200):
            lg, h = hidden_logits(ids)
            p_lm = F.softmax(lg[0, -1].float(), -1)
            if datastore is not None:
                q = F.normalize(h[0, -1:].float(), dim=-1)      # (1,d)
                K, vals = datastore
                sim = (q @ K.T).squeeze(0)                       # (M,) cosine
                d = 1.0 - sim                                    # distance in [0,2]
                tv, ti = torch.topk(sim, min(topk, sim.numel()))
                w = F.softmax(tv * temp_knn, dim=-1)            # sharper on closer neighbors
                p_knn = torch.zeros_like(p_lm)
                p_knn.scatter_add_(0, vals[ti], w)
                d_min = (1.0 - tv[0]).item()
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

    # 1) find failing problems
    fails = []
    for prob in probs_all:
        if not prob.gold_solution:
            continue
        if not clean_grade(prob, greedy(prob)):
            fails.append(prob)
        if len(fails) >= n_target:
            break
    print(f"[knn-oracle] ckpt={ckpt} failing={len(fails)} lam_max={lam_max} "
          f"topk={topk} conf_scale={conf_scale}", flush=True)

    # 2) build ORACLE datastore: gold solutions of the failing problems
    keys, values = [], []
    for prob in fails:
        pre = tok.encode(comment + prob.prompt, add_special_tokens=False)
        gold = tok.encode("\n" + prob.gold_solution.replace("\r\n", "\n"),
                          add_special_tokens=False)
        ids = torch.tensor([(pre + gold)[:320]], device=DEVICE)
        _lg, h = hidden_logits(ids)
        P = len(pre)
        for t in range(P - 1, ids.shape[1] - 1):
            keys.append(h[0, t].float())
            values.append(ids[0, t + 1].item())
    K = F.normalize(torch.stack(keys), dim=-1)         # (M,d)
    vals = torch.tensor(values, device=DEVICE)
    print(f"[knn-oracle] datastore keys={K.shape[0]}", flush=True)

    # 3) eval baseline vs oracle-kNN on the failing problems
    base_pass = knn_pass = 0
    flipped = []
    for prob in fails:
        b = clean_grade(prob, greedy(prob, datastore=None))
        k = clean_grade(prob, greedy(prob, datastore=(K, vals)))
        base_pass += int(b); knn_pass += int(k)
        if k and not b:
            flipped.append(prob.task_id)
    print(f"\n=== ORACLE kNN-LM kill-test (n={len(fails)} failing problems) ===")
    print(f"  baseline pass            = {base_pass}/{len(fails)}")
    print(f"  ORACLE-kNN pass          = {knn_pass}/{len(fails)}  (+{knn_pass-base_pass})")
    print(f"  flipped fail->pass: {flipped}")
    print(f"\nVERDICT: even with the gold answer IN the datastore, if kNN can't "
          f"flip materially more than baseline => CONSUMPTION barrier is fatal "
          f"(confident-wrong + off-path query), kNN direction is dead too. If it "
          f"flips a meaningful fraction => consumption works, green-light real "
          f"datastore + latent-thinking query formation.")


if __name__ == "__main__":
    main()
