"""Re-keyed WorkingMemory prototype — attack KEY SEPARABILITY for recall.

ESTABLISHED DIAGNOSIS (MEMORY.md, don't redo):
  WM is inert for multibind recall because the read keys on the trunk hidden at
  the binding line, and near-identical lines `v17 = 4530` / `v31 = 9912` are NOT
  separable by the 287M trunk -> diffuse, query-independent (recency) read,
  mass_on_binding ~ chance, top1_bind = 0. MQAR worked because its keys were
  DISTINCT RANDOM TOKENS. Freeze-trunk / joint-trunk / explicit attn-placement
  supervision all failed to lift mass-on-binding above chance.

HYPOTHESIS (this probe):
  The discriminable signal IS present: the variable-NAME (v17 vs v31 are distinct
  digit-token sequences). If the WM key is derived from the NAME representation
  (write key = f(name at binding); query key = f(name at the recall query)) the
  buffer becomes content-addressable WITHOUT O(T^2) attention.

WHAT THIS DOES (pure geometry, NO training first):
  One frozen trunk forward per example -> premem hidden h (=out_norm(h_raw), the
  exact tensor WM reads) + input embeddings. Using ORACLE parse of the binding
  lines (every `vK = NNNN`) and the recall query (`vK` mention in the completion),
  build a buffer with ONE slot per binding and compare KEY SOURCES:
    - cur_q_vs_v : query=W_q(h_answer), key=W_v(h_value)  [v12's own trained WM]
    - h_value    : cosine(h_answerpos, h_valuepos)        [entangled baseline]
    - h_nameend  : cosine(h at last name-digit, ...)       [trunk name rep, causal-clean]
    - emb_name   : cosine over INPUT EMBEDDINGS of the name tokens (token identity)
  For each: top1 addressing accuracy (argmax buffer == queried binding) and mean
  attention mass on the correct binding, vs chance=1/N. This isolates ADDRESSING
  (can we find the right slot) from DECODING (can we read the value out).

  Then a RECALL-DECODE test for the winning key: read value = h(value pos),
  add (raw, or via a quick-trained linear W_proj) to the answer-position hidden,
  measure P(answer digits) / exact-match WM-ON vs WM-OFF (fair kill-gate).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/wm_namekey_probe.py \
      --ckpt checkpoints/pretrain_v12_step12398_tok3250061312.pt \
      --eval_sets mb_N48:/tmp/wm_ladder/mb_N48.jsonl,mb_N64:/tmp/wm_ladder/mb_N64.jsonl,mb_N96:/tmp/wm_ladder/mb_N96.jsonl \
      --n 96
"""
from __future__ import annotations

import argparse
import json
import re

import torch
import torch.nn.functional as F

DIGITS = set(range(32, 42))   # token ids for '0'..'9'
V_TOK = 102                   # 'v'
EQ_TOK = 446                  # ' ='


def parse_example(full, plen, answer):
    """Return (bindings, query) with ORACLE positions, all in token space.

    bindings: list of dict(name_start, name_end, val_start, val_end, value:int,
                           name_toks:list[int])  — one per `vK = NNNN` line.
    query: dict(name_start, name_end, name_toks, ans_pos, ans_toks) — the recall
           query (the `vK` mention in the COMPLETION) + the first completion
           occurrence of the value (the recall target span).
    """
    bindings = []
    name_to_b = {}
    for eq in range(len(full)):
        if full[eq] != EQ_TOK:
            continue
        # name = v + digits, ending at eq-1
        ne = eq - 1
        ns = ne
        while ns >= 0 and full[ns] in DIGITS:
            ns -= 1
        if ns < 0 or full[ns] != V_TOK:
            continue
        name_start = ns                       # the 'v'
        name_end = eq - 1                      # last name digit
        # value = digits after ' ='(eq) ' '(eq+1)
        vs = eq + 2
        ve = vs
        while ve < len(full) and full[ve] in DIGITS:
            ve += 1
        if ve == vs:
            continue
        digs = "".join("0123456789"[full[t] - 32] for t in range(vs, ve))
        try:
            value = int(digs)
        except ValueError:
            continue
        name_toks = full[name_start:name_end + 1]
        b = dict(name_start=name_start, name_end=name_end, val_start=vs,
                 val_end=ve, value=value, name_toks=name_toks)
        bindings.append(b)
        name_to_b[tuple(name_toks)] = len(bindings) - 1

    ans = int(answer)
    # queried binding = the (unique) binding whose value == answer
    cands = [i for i, b in enumerate(bindings) if b["value"] == ans]
    if len(cands) != 1:
        return None
    qb = cands[0]
    qname = tuple(bindings[qb]["name_toks"])

    # query name occurrence in COMPLETION: first occurrence of [v, *digits] in
    # full[plen:] matching qname (the `vK` in "`vK` is set to ...").
    L = len(qname)
    q_name_end = None
    for i in range(plen, len(full) - L + 1):
        if tuple(full[i:i + L]) == qname:
            q_name_end = i + L - 1
            break
    if q_name_end is None:
        return None

    # recall target = first completion occurrence of the value digit span
    ans_digs = [32 + int(c) for c in str(ans)]
    al = len(ans_digs)
    ans_pos = None
    for i in range(plen, len(full) - al + 1):
        if full[i:i + al] == ans_digs:
            ans_pos = i
            break
    if ans_pos is None:
        return None
    query = dict(name_start=q_name_end - L + 1, name_end=q_name_end,
                 name_toks=list(qname), ans_pos=ans_pos, ans_toks=ans_digs,
                 qb=qb)
    return bindings, query


def load_examples(path, tok, n, max_len):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompt_ids = tok.encode(r["problem_prompt"] + "\n\n",
                                    add_special_tokens=False)
            comp_ids = tok.encode(r["qwen_completion"], add_special_tokens=False)
            full = prompt_ids + comp_ids
            if len(full) > max_len:
                continue
            parsed = parse_example(full, len(prompt_ids), r["answer"])
            if parsed is None:
                continue
            bindings, query = parsed
            out.append(dict(full=full, plen=len(prompt_ids),
                            bindings=bindings, query=query,
                            n_vars=r.get("n_vars")))
            if len(out) >= n:
                break
    return out


@torch.no_grad()
def trunk_feats(model, full, device):
    """One frozen forward -> (h_premem[T,d], emb[T,d])."""
    ids = torch.tensor([full], dtype=torch.long, device=device)
    _, h = model(ids, return_hidden=True)         # premem (out_norm(h_raw))
    emb = model.embed(ids)
    return h[0], emb[0]


def name_emb_key(emb, name_start, name_end):
    """Order-sensitive pooled INPUT-EMBEDDING name key (token identity only).

    Position-weighted sum so `v13`!=`v31` (mean-pool would collide). Includes the
    shared 'v' (harmless constant) + the digit tokens in order."""
    idx = torch.arange(name_start, name_end + 1, device=emb.device)
    w = (torch.arange(1, len(idx) + 1, device=emb.device).float())
    return (emb[idx] * w.unsqueeze(-1)).sum(0)


def addressing_eval(model, examples, device, tau):
    """For each key source, top1 addressing accuracy + mass on correct slot."""
    mem = model.memory
    srcs = ["cur_q_vs_v", "h_value", "h_nameend", "emb_name"]
    agg = {s: dict(top1=0, mass=0.0) for s in srcs}
    chance = 0.0
    used = 0
    for ex in examples:
        h, emb = trunk_feats(model, ex["full"], device)
        B = ex["bindings"]
        qb = ex["query"]["qb"]
        N = len(B)
        chance += 1.0 / N
        used += 1
        # buffer key/val source positions
        val_pos = [b["val_start"] for b in B]
        nameend_pos = [b["name_end"] for b in B]
        ans_pos = ex["query"]["ans_pos"]
        q_nameend = ex["query"]["name_end"]

        def score(qvec, kmat):
            qn = F.normalize(qvec, dim=-1)
            kn = F.normalize(kmat, dim=-1)
            s = (kn @ qn) * tau
            return torch.softmax(s, dim=-1)

        # 1) v12's own trained WM projections: query=W_q(h at answer-1 pos),
        #    key=W_v(h at value pos). (the read predicting the answer queries at
        #    ans_pos-1; use the trunk hidden there.)
        qh = h[ans_pos - 1]
        kv = h[torch.tensor(val_pos, device=device)]
        a = score(mem.W_q(qh), mem.W_v(kv))
        agg["cur_q_vs_v"]["top1"] += int(a.argmax().item() == qb)
        agg["cur_q_vs_v"]["mass"] += float(a[qb])
        # 2) raw h at value pos vs h at answer-1 (entangled baseline)
        a = score(h[ans_pos - 1], kv)
        agg["h_value"]["top1"] += int(a.argmax().item() == qb)
        agg["h_value"]["mass"] += float(a[qb])
        # 3) trunk hidden at last name digit (causal-clean name rep); query =
        #    trunk hidden at the query name-end in the completion.
        kn = h[torch.tensor(nameend_pos, device=device)]
        a = score(h[q_nameend], kn)
        agg["h_nameend"]["top1"] += int(a.argmax().item() == qb)
        agg["h_nameend"]["mass"] += float(a[qb])
        # 4) pure input-embedding name key (token identity)
        kemb = torch.stack([name_emb_key(emb, b["name_start"], b["name_end"])
                            for b in B])
        qemb = name_emb_key(emb, ex["query"]["name_start"], q_nameend)
        a = score(qemb, kemb)
        agg["emb_name"]["top1"] += int(a.argmax().item() == qb)
        agg["emb_name"]["mass"] += float(a[qb])
    return srcs, agg, chance / max(1, used), used


@torch.no_grad()
def recall_decode(model, examples, device, tau, key_src, alpha):
    """Kill-gate recall using a chosen KEY SOURCE for addressing and value =
    h(value pos), injected RAW (no W_proj) at the answer position, scaled by
    alpha. WM-ON = lm_head(h_ans + alpha*read); WM-OFF = lm_head(h_ans).
    Reports exact-match (all digits argmax) + mean per-digit logp."""
    on_exact = off_exact = 0
    on_lp = off_lp = 0.0
    used = 0
    for ex in examples:
        h, emb = trunk_feats(model, ex["full"], device)
        B = ex["bindings"]
        q = ex["query"]
        qb = q["qb"]
        val_pos = [b["val_start"] for b in B]
        nameend_pos = [b["name_end"] for b in B]
        kv = h[torch.tensor(val_pos, device=device)]      # value hiddens

        def addr(qvec, kmat):
            qn = F.normalize(qvec, dim=-1)
            kn = F.normalize(kmat, dim=-1)
            return torch.softmax((kn @ qn) * tau, dim=-1)

        ans_toks = q["ans_toks"]
        vlen = len(ans_toks)
        ap = q["ans_pos"]
        on_ok = off_ok = True
        on_l = off_l = 0.0
        for j in range(vlen):
            ppos = ap - 1 + j                  # predicts ans token j
            # build query key at this predicting position
            if key_src == "h_nameend":
                # query addresses with the trunk hidden at the COMPLETION name
                # mention (fixed within the answer span — the name is just before)
                a = addr(h[q["name_end"]], h[torch.tensor(nameend_pos, device=device)])
            elif key_src == "emb_name":
                kemb = torch.stack([name_emb_key(emb, b["name_start"], b["name_end"]) for b in B])
                qemb = name_emb_key(emb, q["name_start"], q["name_end"])
                a = addr(qemb, kemb)
            else:  # entangled baseline: query = current predicting hidden
                a = addr(h[ppos], kv)
            read = (a.unsqueeze(0) @ kv).squeeze(0)        # (d,)
            lo = model.lm_head(h[ppos] + alpha * read)
            lf = model.lm_head(h[ppos])
            on_ok &= (int(lo.argmax().item()) == ans_toks[j])
            off_ok &= (int(lf.argmax().item()) == ans_toks[j])
            on_l += float(F.log_softmax(lo, -1)[ans_toks[j]])
            off_l += float(F.log_softmax(lf, -1)[ans_toks[j]])
        on_exact += int(on_ok)
        off_exact += int(off_ok)
        on_lp += on_l / vlen
        off_lp += off_l / vlen
        used += 1
    nz = max(1, used)
    return dict(n=used, on_acc=on_exact / nz, off_acc=off_exact / nz,
                on_lp=on_lp / nz, off_lp=off_lp / nz,
                d_acc=(on_exact - off_exact) / nz,
                d_lp=(on_lp - off_lp) / nz)


class NameKeyedRead(torch.nn.Module):
    """Content-addressable read keyed on the variable-NAME embedding.

    Addressing (FROZEN, no params): cosine(name_emb_query, name_emb_buffer)*tau
    -> softmax. Pure token-identity -> perfectly separable (probe: top1=1.00).
    Value path (TRAINED): read = Σ attn · W_v(h_value); inject α·W_proj(read).
    This is the v14 candidate: keep O(T·K), drop the entangled hidden key."""

    def __init__(self, d_model, d_mem=128, tau=20.0, alpha_init=1.0):
        super().__init__()
        self.W_v = torch.nn.Linear(d_model, d_mem, bias=False)
        self.W_proj = torch.nn.Linear(d_mem, d_model, bias=False)
        torch.nn.init.normal_(self.W_v.weight, std=0.02)
        torch.nn.init.normal_(self.W_proj.weight, std=0.02)
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha_init)))
        self.tau = float(tau)


def _ex_tensors(ex, h, emb, device):
    """Precompute per-example addressing + value sources."""
    B = ex["bindings"]
    qb = ex["query"]["qb"]
    val_end_pos = torch.tensor([b["val_end"] - 1 for b in B], device=device)
    kemb = torch.stack([name_emb_key(emb, b["name_start"], b["name_end"])
                        for b in B])                              # (N, d)
    qemb = name_emb_key(emb, ex["query"]["name_start"], ex["query"]["name_end"])
    return qb, val_end_pos, F.normalize(kemb, dim=-1), F.normalize(qemb, dim=-1)


def train_namekeyed(model, train_ex, eval_ex_sets, device, *, steps, lr,
                    d_mem, tau, unfreeze_trunk):
    """Train the name-keyed read (frozen trunk by default; --unfreeze_trunk for
    joint). WM-OFF in frozen mode == the v12 base exactly -> honest kill-gate."""
    read = NameKeyedRead(model.embed.weight.shape[1], d_mem=d_mem, tau=tau).to(device)

    def feats(ex, grad):
        ids = torch.tensor([ex["full"]], dtype=torch.long, device=device)
        if grad:
            _, h = model(ids, return_hidden=True)
            return h[0], model.embed(ids)[0]
        with torch.no_grad():
            _, h = model(ids, return_hidden=True)
            return h[0], model.embed(ids)[0]

    read_params = list(read.parameters())
    groups = [dict(params=read_params, lr=lr)]
    if unfreeze_trunk:
        for p in model.parameters():
            p.requires_grad_(True)
        trunk_params = [p for p in model.parameters() if p.requires_grad]
        groups.append(dict(params=trunk_params, lr=lr * 0.2))   # slower trunk
        params = read_params + trunk_params
    else:
        params = read_params
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

    import random as _r
    rng = _r.Random(0)
    model.eval()
    for step in range(1, steps + 1):
        ex = train_ex[rng.randrange(len(train_ex))]
        h, emb = feats(ex, unfreeze_trunk)
        qb, vend, kn, qn = _ex_tensors(ex, h, emb, device)
        attn = torch.softmax((kn @ qn) * read.tau, dim=-1)         # (N,)
        valbuf = read.W_v(h[vend])                                  # (N, d_mem)
        rd = attn @ valbuf                                          # (d_mem,)
        inj = read.alpha * read.W_proj(rd)                         # (d,)
        q = ex["query"]
        ce = 0.0
        for j, dtok in enumerate(q["ans_toks"]):
            ppos = q["ans_pos"] - 1 + j
            logits = model.lm_head(h[ppos] + inj)
            ce = ce + F.cross_entropy(logits.unsqueeze(0),
                                      torch.tensor([dtok], device=device))
        ce = ce / len(q["ans_toks"])
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d} ce={float(ce):.4f} alpha={float(read.alpha):.3f}")
    return read


@torch.no_grad()
def eval_namekeyed(model, read, exs, device):
    """Kill-gate: WM-ON (h+inj) vs WM-OFF (h). Exact-match + first-digit."""
    on_e = off_e = on_f = off_f = 0
    on_lp = off_lp = 0.0
    used = 0
    for ex in exs:
        ids = torch.tensor([ex["full"]], dtype=torch.long, device=device)
        _, h = model(ids, return_hidden=True)
        h = h[0]
        emb = model.embed(ids)[0]
        qb, vend, kn, qn = _ex_tensors(ex, h, emb, device)
        attn = torch.softmax((kn @ qn) * read.tau, dim=-1)
        rd = attn @ read.W_v(h[vend])
        inj = read.alpha * read.W_proj(rd)
        q = ex["query"]
        on_ok = off_ok = True
        on_l = 0.0
        for j, dtok in enumerate(q["ans_toks"]):
            ppos = q["ans_pos"] - 1 + j
            lo = model.lm_head(h[ppos] + inj)
            lf = model.lm_head(h[ppos])
            on_hit = int(lo.argmax().item()) == dtok
            off_hit = int(lf.argmax().item()) == dtok
            on_ok &= on_hit
            off_ok &= off_hit
            if j == 0:
                on_f += int(on_hit); off_f += int(off_hit)
            on_l += float(F.log_softmax(lo, -1)[dtok])
        on_e += int(on_ok); off_e += int(off_ok)
        on_lp += on_l / len(q["ans_toks"])
        used += 1
    nz = max(1, used)
    return dict(n=used, on_exact=on_e / nz, off_exact=off_e / nz,
                on_first=on_f / nz, off_first=off_f / nz,
                on_lp=on_lp / nz)


def train_rai(model, train_ex, eval_sets, device, *, steps, lr, d_mem, tau,
              unfreeze_trunk):
    """RETRIEVAL-AS-INPUT decode: name-keyed addressing (frozen, perfect), value
    = W_v(h at value-end) compressed to ONE vector, injected as the INPUT
    embedding across the answer span -> the trunk RECURRENCE unrolls the digits
    (the additive-output read structurally cannot). Kill-gate: ON=injected
    forward, OFF=clean forward."""
    d = model.embed.weight.shape[1]
    W_v = torch.nn.Linear(d, d_mem, bias=False).to(device)
    W_in = torch.nn.Linear(d_mem, d, bias=False).to(device)
    torch.nn.init.normal_(W_v.weight, std=0.02)
    torch.nn.init.normal_(W_in.weight, std=0.02)
    alpha = torch.nn.Parameter(torch.tensor(1.0, device=device))
    read_params = list(W_v.parameters()) + list(W_in.parameters()) + [alpha]
    groups = [dict(params=read_params, lr=lr)]
    if unfreeze_trunk:
        for p in model.parameters():
            p.requires_grad_(True)
        tp = [p for p in model.parameters() if p.requires_grad]
        groups.append(dict(params=tp, lr=lr * 0.2))
        params = read_params + tp
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        params = read_params
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    import random as _r
    rng = _r.Random(0)
    model.eval()

    def build_inj(ex):
        ids = torch.tensor([ex["full"]], dtype=torch.long, device=device)
        with torch.no_grad():
            _, h = model(ids, return_hidden=True)
        h = h[0]
        emb = model.embed(ids)[0]
        qb, vend, kn, qn = _ex_tensors(ex, h, emb, device)
        attn = torch.softmax((kn @ qn) * tau, dim=-1)
        read = attn @ W_v(h[vend].detach())
        return ids, alpha * W_in(read)

    def fwd_logits(ids, inj, ex):
        ie = model.embed(ids).clone()
        q = ex["query"]
        a, b = q["ans_pos"] - 1, q["ans_pos"] - 1 + len(q["ans_toks"])
        ie[0, a:b] = ie[0, a:b] + inj.unsqueeze(0)
        return model(ids, inputs_embeds=ie)

    def report(tag):
        print(f"\n===== RAI KILL-GATE [{tag}] =====")
        for nm, exs in eval_sets:
            if not exs:
                continue
            on_e = off_e = on_f = off_f = 0; on_lp = 0.0; used = 0
            with torch.no_grad():
                for ex in exs:
                    ids, inj = build_inj(ex)
                    lo = fwd_logits(ids, inj, ex)[0]
                    lf = model(ids)[0]
                    q = ex["query"]; ok_on = ok_off = True; lp = 0.0
                    for j, dt in enumerate(q["ans_toks"]):
                        pp = q["ans_pos"] - 1 + j
                        h_on = int(lo[pp].argmax()) == dt
                        h_off = int(lf[pp].argmax()) == dt
                        ok_on &= h_on; ok_off &= h_off
                        if j == 0:
                            on_f += int(h_on); off_f += int(h_off)
                        lp += float(F.log_softmax(lo[pp], -1)[dt])
                    on_e += int(ok_on); off_e += int(ok_off); on_lp += lp / len(q["ans_toks"]); used += 1
            nz = max(1, used)
            print(f"  [{nm}] n={used}  EXACT ON={on_e/nz:.3f} OFF={off_e/nz:.3f} (Δ{(on_e-off_e)/nz:+.3f})"
                  f"  |  FIRST ON={on_f/nz:.3f} OFF={off_f/nz:.3f} (Δ{(on_f-off_f)/nz:+.3f})  |  ON_lp={on_lp/nz:.2f}")

    report("BEFORE (untrained)")
    for step in range(1, steps + 1):
        ex = train_ex[rng.randrange(len(train_ex))]
        ids, inj = build_inj(ex)
        logits = fwd_logits(ids, inj, ex)[0]
        q = ex["query"]
        ce = 0.0
        for j, dt in enumerate(q["ans_toks"]):
            pp = q["ans_pos"] - 1 + j
            ce = ce + F.cross_entropy(logits[pp].unsqueeze(0),
                                      torch.tensor([dt], device=device))
        ce = ce / len(q["ans_toks"])
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d} ce={float(ce):.4f} alpha={float(alpha):.3f}")
    report("AFTER")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrain_v12_step12398_tok3250061312.pt")
    ap.add_argument("--eval_sets", default=(
        "mb_N48:/tmp/wm_ladder/mb_N48.jsonl,"
        "mb_N64:/tmp/wm_ladder/mb_N64.jsonl,"
        "mb_N96:/tmp/wm_ladder/mb_N96.jsonl"))
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--max_len", type=int, default=4000)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train", action="store_true",
                    help="train the name-keyed read head + kill-gate eval")
    ap.add_argument("--train_data", default="/tmp/wm_ladder/mb_train_mix.jsonl")
    ap.add_argument("--train_n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_mem", type=int, default=128)
    ap.add_argument("--unfreeze_trunk", action="store_true")
    ap.add_argument("--rai", action="store_true",
                    help="retrieval-as-input decode (recurrence unrolls value)")
    args = ap.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    print(f"[load] {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model.to(args.device).eval()
    model._latent_feedback_premem = True
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    # --------- TRAIN mode: learn the name-keyed read, then kill-gate eval -----
    if args.train:
        eval_sets = []
        for spec in args.eval_sets.split(","):
            spec = spec.strip()
            if not spec:
                continue
            name, path = spec.split(":", 1)
            exs = load_examples(path, tok, args.n, args.max_len)
            eval_sets.append((name, exs))
            print(f"[eval] {name} usable={len(exs)}")
        train_ex = load_examples(args.train_data, tok, args.train_n, args.max_len)
        print(f"[train] {len(train_ex)} examples  steps={args.steps} lr={args.lr}"
              f" d_mem={args.d_mem} unfreeze_trunk={args.unfreeze_trunk} rai={args.rai}")

        if args.rai:
            train_rai(model, train_ex, eval_sets, args.device, steps=args.steps,
                      lr=args.lr, d_mem=args.d_mem, tau=args.tau,
                      unfreeze_trunk=args.unfreeze_trunk)
            return

        def report(tag):
            print(f"\n===== KILL-GATE [{tag}] =====")
            for nm, exs in eval_sets:
                if not exs:
                    continue
                r = eval_namekeyed(model, read, exs, args.device)
                print(f"  [{nm}] n={r['n']}  EXACT ON={r['on_exact']:.3f} OFF={r['off_exact']:.3f}"
                      f" (Δ{r['on_exact']-r['off_exact']:+.3f})  |  FIRST-DIGIT ON={r['on_first']:.3f}"
                      f" OFF={r['off_first']:.3f} (Δ{r['on_first']-r['off_first']:+.3f})  |  ON_lp={r['on_lp']:.2f}")

        read = NameKeyedRead(model.embed.weight.shape[1], d_mem=args.d_mem, tau=args.tau).to(args.device)
        report("BEFORE (untrained read; ON≈OFF expected)")
        read = train_namekeyed(model, train_ex, eval_sets, args.device,
                               steps=args.steps, lr=args.lr, d_mem=args.d_mem,
                               tau=args.tau, unfreeze_trunk=args.unfreeze_trunk)
        report("AFTER (name-keyed read trained)")
        return

    for spec in args.eval_sets.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, path = spec.split(":", 1)
        exs = load_examples(path, tok, args.n, args.max_len)
        if not exs:
            print(f"[{name}] no usable examples"); continue
        srcs, agg, chance, used = addressing_eval(model, exs, args.device, args.tau)
        print(f"\n===== [{name}] addressing  (n={used}, chance={chance:.3f}, tau={args.tau}) =====")
        for s in srcs:
            t = agg[s]["top1"] / max(1, used)
            m = agg[s]["mass"] / max(1, used)
            print(f"  {s:12s}: top1_addr_acc={t:.3f}   mass_on_correct={m:.3f}   (x chance: {m/max(1e-9,chance):.1f})")
        # recall decode for the two name-keyed sources + entangled baseline
        for ks in ["h_value", "h_nameend", "emb_name"]:
            for alpha in (1.0,):
                r = recall_decode(model, exs, args.device, args.tau, ks, alpha)
                print(f"  recall[{ks:9s} a={alpha}]: WM-ON acc={r['on_acc']:.3f} lp={r['on_lp']:.2f}"
                      f" | OFF acc={r['off_acc']:.3f} lp={r['off_lp']:.2f}"
                      f" | dacc={r['d_acc']:+.3f} dlp={r['d_lp']:+.2f}")


if __name__ == "__main__":
    main()
