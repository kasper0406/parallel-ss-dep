"""Multi-token VALUE READOUT for the name-keyed WM recall mechanism.

CONTEXT (established, do not redo — see project_recall_discrete_key_direction):
  ADDRESSING is SOLVED. Keying the WM read on the variable-NAME input-embedding
  window (position-weighted pool of the name tokens' INPUT embeddings) makes
  retrieval perfectly separable on the saturating multibind regime: top1=1.00,
  mass 39-65x chance, to N=96, while v12's own cosine-on-hidden keys sit at
  chance. It is LOAD-BEARING: a single additive read lifts FIRST-DIGIT recall
  by +0.16..+0.21 (kill-gate ON-OFF) with OFF pinned at chance.

THE ONLY RESIDUAL BLOCKER (this script): MULTI-TOKEN VALUE READOUT.
  A single additive read vector is a constant shift -> it can only bias ONE
  output position, so 4-digit values fail EXACT match. We keep the proven
  embedding-key ADDRESSING fixed and change ONLY the readout, building two arms:

  ARM A  perdigit  (sequential per-position re-read):
    The value is buffered PER DIGIT. To emit answer digit j (j=0..3) we re-query
    with (name-key, output-position-offset j): addressing picks the binding slot
    (frozen, name-key), then we index digit j inside that slot's value span ->
    read_j = sum_i attn_i * W_v(h[val_start_i + j]). Because each digit position
    gives a DIFFERENT hidden, the additive injection carries a different shift
    per output position -> all 4 digits decodable. O(T*K), no attention. This is
    the additive-WM channel generalised to multi-token (the v14 candidate).

  ARM B  copy  (pointer/copy):
    Once addressed, COPY the verbatim source digit tokens. For output j the copy
    distribution over the vocab is p_copy[j] = sum_i attn_i * onehot(src_tok[i,j])
    (src_tok[i,j] = the literal token id at val_start_i + j). A learned copy gate
    g (from the answer-position hidden) mixes: p = (1-g) p_lm + g p_copy. With
    addressing one-hot this copies the exact 4 digits. Robust upper bound.

ADDRESSING is identical & FROZEN in both arms (name_emb_key from wm_namekey_probe).
TRUNK is FROZEN by default => WM-OFF == the v12 base EXACTLY (honest kill-gate);
--joint_trunk co-trains the trunk (slower LR) as an additional arm.

Oracle parse (wm_namekey_probe.parse_example) only LOCATES the name spans, the
value spans, and the answer span. The read NEVER peeks at which binding is
correct: it addresses purely by name-key cosine. This mirrors the v14 plan where
data_mix supplies a mem_read_mask over the answer span and the key comes from a
causal input-embedding window over the queried name.

Usage (GPU courtesy: pick the free GPU, expandable segments, one job):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/wm_multitok_readout.py \
      --ckpt checkpoints/pretrain_v12_step12398_tok3250061312.pt \
      --train /tmp/wm_mt/train.jsonl --n_train 2000 --steps 800 \
      --eval mb_N48:/tmp/wm_mt/held_N48.jsonl,mb_N64:/tmp/wm_mt/held_N64.jsonl,mb_N96:/tmp/wm_mt/held_N96.jsonl \
      --n_eval 200 --arm both
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.wm_namekey_probe import load_examples, name_emb_key


# --------------------------------------------------------------------------- caches
@torch.no_grad()
def extract(model, ex, device):
    """One frozen trunk forward -> only the slices the readout needs (CPU).

    Returns None if the example is not clean (value spans shorter than the answer
    length, etc.). Addressing keys come from INPUT EMBEDDINGS (name spans); value
    sources come from the premem hidden (the exact tensor WM reads)."""
    full = ex["full"]
    q = ex["query"]
    B = ex["bindings"]
    vlen = len(q["ans_toks"])                      # 4 for these tasks
    # every binding must have >= vlen value digits (true for randint(1000,9999))
    for b in B:
        if b["val_end"] - b["val_start"] < vlen:
            return None
    ids = torch.tensor([full], dtype=torch.long, device=device)
    _, h = model(ids, return_hidden=True)          # premem = out_norm(h_raw)
    h = h[0]
    emb = model.embed(ids)[0]

    kemb = torch.stack([name_emb_key(emb, b["name_start"], b["name_end"])
                        for b in B])               # (N, d)
    qemb = name_emb_key(emb, q["name_start"], q["name_end"])   # (d,)
    kn = F.normalize(kemb, dim=-1)
    qn = F.normalize(qemb, dim=-1)

    val_h = torch.stack([h[b["val_start"]:b["val_start"] + vlen] for b in B])  # (N,vlen,d)
    src_tok = torch.tensor([full[b["val_start"]:b["val_start"] + vlen] for b in B],
                           dtype=torch.long)        # (N, vlen)
    ap = q["ans_pos"]
    ans_h = h[ap - 1:ap - 1 + vlen]                 # (vlen, d) hidden predicting each digit
    ans_toks = torch.tensor(q["ans_toks"], dtype=torch.long)  # (vlen,)
    return dict(kn=kn.cpu(), qn=qn.cpu(), val_h=val_h.cpu(), src_tok=src_tok,
                ans_h=ans_h.cpu(), ans_toks=ans_toks, qb=q["qb"], N=len(B),
                vlen=vlen)


def build_cache(model, examples, device, tag):
    cache = []
    for ex in examples:
        c = extract(model, ex, device)
        if c is not None:
            cache.append(c)
    print(f"[cache:{tag}] {len(cache)}/{len(examples)} usable")
    return cache


# --------------------------------------------------------------------------- arms
class PerDigitRead(nn.Module):
    """ARM A: per-output-position additive read. read_j = attn @ W_v(val_h[:,j]).
    inject = alpha * W_proj(read_j) added to the answer-position hidden."""

    def __init__(self, d_model, d_mem=128, tau=20.0, alpha_init=1.0):
        super().__init__()
        self.W_v = nn.Linear(d_model, d_mem, bias=False)
        self.W_proj = nn.Linear(d_mem, d_model, bias=False)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.normal_(self.W_proj.weight, std=0.02)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.tau = float(tau)

    def inject(self, c, kn, qn, val_h):
        attn = torch.softmax((kn @ qn) * self.tau, dim=-1)        # (N,)
        valbuf = self.W_v(val_h)                                   # (N, vlen, d_mem)
        read = torch.einsum("n,njm->jm", attn, valbuf)            # (vlen, d_mem)
        return self.alpha * self.W_proj(read)                     # (vlen, d)


class CopyHead(nn.Module):
    """ARM B: pointer/copy. copy distribution = attn-weighted onehot(src_tok);
    learned gate g (from answer hidden) mixes with the frozen lm distribution."""

    def __init__(self, d_model, vocab, tau=20.0):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.0)
        self.tau = float(tau)
        self.vocab = vocab

    def final_logp(self, lm_logits, ans_h, kn, qn, src_tok):
        """Return per-digit log p_final (vlen, vocab)."""
        attn = torch.softmax((kn @ qn) * self.tau, dim=-1)        # (N,)
        vlen = lm_logits.shape[0]
        p_lm = torch.softmax(lm_logits, dim=-1)                   # (vlen, vocab)
        p_copy = lm_logits.new_zeros(vlen, self.vocab)
        # p_copy[j, src_tok[i,j]] += attn[i]
        for j in range(vlen):
            p_copy[j].scatter_add_(0, src_tok[:, j], attn)
        g = torch.sigmoid(self.gate(ans_h)).squeeze(-1)          # (vlen,)
        p = (1 - g).unsqueeze(-1) * p_lm + g.unsqueeze(-1) * p_copy
        return torch.log(p + 1e-9), g


# --------------------------------------------------------------------------- train/eval
def _move(c, device):
    return (c["kn"].to(device), c["qn"].to(device), c["val_h"].to(device),
            c["ans_h"].to(device), c["src_tok"].to(device),
            c["ans_toks"].to(device))


def train_arm(arm, model, cache, device, *, steps, lr, d_mem, tau):
    """Train the readout head on the FROZEN trunk.

    Frozen trunk => WM-OFF (no injection) reproduces the v12 base teacher-forced
    prediction EXACTLY, so the ON-vs-OFF kill-gate is an honest measure of what
    the readout adds. (Joint-trunk co-training is the v14 integration step, but it
    would shift the OFF baseline and so muddies the control; the frozen result
    already establishes whether the readout recovers the full value.)"""
    d = model.embed.weight.shape[1]
    vocab = model.lm_head.weight.shape[0]
    if arm == "perdigit":
        head = PerDigitRead(d, d_mem=d_mem, tau=tau).to(device)
    else:
        head = CopyHead(d, vocab, tau=tau).to(device)
    head_params = list(head.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(head_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(0)
    model.eval()

    for step in range(1, steps + 1):
        c = cache[rng.randrange(len(cache))]
        kn, qn, val_h, ans_h, src_tok, ans_toks = _move(c, device)
        if arm == "perdigit":
            inj = head.inject(c, kn, qn, val_h)                  # (vlen, d)
            logits = model.lm_head(ans_h + inj)                  # (vlen, vocab)
            ce = F.cross_entropy(logits, ans_toks)
        else:
            lm_logits = model.lm_head(ans_h)
            logp, g = head.final_logp(lm_logits, ans_h, kn, qn, src_tok)
            ce = F.nll_loss(logp, ans_toks)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            extra = (f" alpha={head.alpha.item():.3f}" if arm == "perdigit" else "")
            print(f"  [{arm}] step {step:4d} ce={float(ce):.4f}{extra}")
    return head


@torch.no_grad()
def eval_arm(arm, model, head, cache, device):
    """Kill-gate: WM-ON vs WM-OFF. Reports exact (all digits) + first-digit +
    addressing top1 + ON logp."""
    on_e = off_e = on_f = off_f = top1 = 0
    on_lp = off_lp = 0.0
    used = 0
    for c in cache:
        kn, qn, val_h, ans_h, src_tok, ans_toks = _move(c, device)
        attn = torch.softmax((kn @ qn) * head.tau, dim=-1)
        top1 += int(attn.argmax().item() == c["qb"])
        lm_logits = model.lm_head(ans_h)                         # WM-OFF logits
        if arm == "perdigit":
            inj = head.inject(c, kn, qn, val_h)
            on_logits = model.lm_head(ans_h + inj)
            on_lp_v = F.log_softmax(on_logits, -1)
            off_lp_v = F.log_softmax(lm_logits, -1)
            on_pred = on_logits.argmax(-1)
            off_pred = lm_logits.argmax(-1)
        else:
            logp, g = head.final_logp(lm_logits, ans_h, kn, qn, src_tok)
            off_lp_v = F.log_softmax(lm_logits, -1)
            on_lp_v = logp
            on_pred = logp.argmax(-1)
            off_pred = lm_logits.argmax(-1)
        on_ok = bool((on_pred == ans_toks).all())
        off_ok = bool((off_pred == ans_toks).all())
        on_e += int(on_ok); off_e += int(off_ok)
        on_f += int(on_pred[0].item() == ans_toks[0].item())
        off_f += int(off_pred[0].item() == ans_toks[0].item())
        idx = torch.arange(len(ans_toks), device=device)
        on_lp += float(on_lp_v[idx, ans_toks].mean())
        off_lp += float(off_lp_v[idx, ans_toks].mean())
        used += 1
    nz = max(1, used)
    return dict(n=used, on_exact=on_e / nz, off_exact=off_e / nz,
                on_first=on_f / nz, off_first=off_f / nz,
                top1=top1 / nz, on_lp=on_lp / nz, off_lp=off_lp / nz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrain_v12_step12398_tok3250061312.pt")
    ap.add_argument("--train", default="/tmp/wm_mt/train.jsonl")
    ap.add_argument("--eval", default=("mb_N48:/tmp/wm_mt/held_N48.jsonl,"
                                       "mb_N64:/tmp/wm_mt/held_N64.jsonl,"
                                       "mb_N96:/tmp/wm_mt/held_N96.jsonl"))
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_mem", type=int, default=128)
    ap.add_argument("--tau", type=float, default=20.0)
    ap.add_argument("--arm", choices=["perdigit", "copy", "both"], default="both")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    print(f"[load] {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model.to(args.device).eval()
    model._latent_feedback_premem = True
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    train_ex = load_examples(args.train, tok, args.n_train, args.max_len)
    eval_sets = []
    for spec in args.eval.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, path = spec.split(":", 1)
        eval_sets.append((name, load_examples(path, tok, args.n_eval, args.max_len)))

    train_cache = build_cache(model, train_ex, args.device, "train")
    eval_caches = [(nm, build_cache(model, exs, args.device, nm)) for nm, exs in eval_sets]

    arms = ["perdigit", "copy"] if args.arm == "both" else [args.arm]
    for arm in arms:
        print(f"\n########## ARM = {arm}  (frozen trunk; OFF == v12 base) ##########")
        head = train_arm(arm, model, train_cache, args.device, steps=args.steps,
                         lr=args.lr, d_mem=args.d_mem, tau=args.tau)
        print(f"\n===== [{arm}] KILL-GATE (WM-ON vs WM-OFF) =====")
        print(f"{'set':8s} {'n':>4s} | {'EXACT_ON':>9s} {'EXACT_OFF':>9s} {'dEXACT':>7s} "
              f"| {'FIRST_ON':>9s} {'FIRST_OFF':>9s} | {'top1':>5s} {'ON_lp':>7s} {'OFF_lp':>7s}")
        for nm, cache in eval_caches:
            r = eval_arm(arm, model, head, cache, args.device)
            print(f"{nm:8s} {r['n']:>4d} | {r['on_exact']:>9.3f} {r['off_exact']:>9.3f} "
                  f"{r['on_exact']-r['off_exact']:>+7.3f} | {r['on_first']:>9.3f} {r['off_first']:>9.3f} "
                  f"| {r['top1']:>5.2f} {r['on_lp']:>7.2f} {r['off_lp']:>7.2f}")


if __name__ == "__main__":
    main()
