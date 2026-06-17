"""CONTEXTUAL (semantic) vs SURFACE addressing on ALIAS recall — REAL frozen v12.

THE DECISIVE EXPERIMENT (research, fair-baseline discipline — a clean negative
is a fully valid result). Every WM addressing key we have validated keys on
SURFACE FORM (exact hash / normalized hash / learned name-span embedding key).
The user's verdict: surface addressing is "not general" — it breaks on aliases /
paraphrase / renames where the recall query does not textually match the binding
that holds the value. The general primitive is addressing by MEANING/REFERENCE:
a learned attention whose Q/K come from the trunk's CONTEXTUAL HIDDEN states
(what the context says a token refers to), like real transformer attention.

Data: experiments/gen_alias_recall.py — `A = 1234` ... `B = A` ... query B → 1234
(1-hop) and `C = B` (2-hop). The value digits appear ONCE (A's line), far back;
we score the model predicting them at the recall position (leak-free); A is never
the most-recent binding (no recency shortcut); held-out names disjoint from train.

----------------------------------------------------------------------------------
FAIR-BASELINE REWRITE (2026-06-17) — the first run's contextual arm was
UNDER-POWERED and never converged (train CE 1.8→2.3, failed even EXACT recall at
0.03 vs surface/hash 0.68/0.74), so its alias failure was uninterpretable. This
version applies the full fairness package so a negative (or positive) is
trustworthy:

  1 CAPACITY MATCH — contextual q/k are GELU MLPs (≥ surface's encoder), d_key
    raised to 128-256 (default 192).
  2 GATE-BOOTSTRAP TRAP fix — (a) copy gate warmed OPEN (bias +2, was -2 closed),
    AND (b) a DIRECT attention-supervision aux loss (CE that the read attention
    puts its mass on the CORRECT binding slot) so the addresser gets gradient
    INDEPENDENT of the gate. Applied uniformly to every LEARNED arm (surface +
    contextual); hash has no params.
  3 EXACT-FIRST CURRICULUM + SANITY GATE — train+report EXACT recall FIRST (phase
    1, exact-only) with TRAIN-set recall (does it even fit?). THE GATE: if the
    fair, capacity-matched, attention-supervised contextual arm cannot solve
    EXACT recall, the contextual hidden genuinely lacks usable separable
    reference info on the frozen trunk → fair negative. If it can, proceed to
    alias (phase 2).
  4 KEY ANCHOR — the prior arm keyed on h_vstart (the digit position, the wrong
    anchor for NAME identity). We now report THREE contextual anchors, all
    DECOUPLED from the copy source (copy always reads the value span):
      ctx_value    : query=recall-pos hidden, key=value-start hidden (old anchor)
      ctx_name     : query=query-name-END hidden, key=binding-name-END hidden
      ctx_namepool : query=pooled query-name-span hidden, key=pooled binding
                     name-span hidden  (the NAME-carrying contextual register)
  5 WHITENING DIAGNOSTIC — raw cosine bunches all binding hiddens at +0.82 (the
    textbook "dominant shared component" regime). We add (a) PCA top-k removal +
    raw cosine, and (b) a LEARNED LINEAR projection trained to identify the
    holder, on BOTH exact and alias. Separates "raw cosine can't" from "no
    learned key can".
  6 DE-NOISE — contextual exact-only first (phase 1), so unlearnable-alias
    gradient does not swamp the exact signal.

ARMS (frozen trunk, identical copy readout / data / budget; only the KEY differs):
  surface      : learned soft key over pooled name-span INPUT EMBEDDINGS.
  ctx_value / ctx_name / ctx_namepool : the CONTEXTUAL (strengthened) arms above.
  hash         : surface lexical FNV hash → one-hot.
  WM-OFF       : recurrence only (frozen trunk lm_head, no addressing) == the
                 honest kill-gate (the trunk is frozen so this is exact).

Usage (pin a FREE gpu — GPU 0 runs the v14 trainer; use GPU 1):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/alias_addressing_probe.py
  # quick wiring check:
  ... experiments/alias_addressing_probe.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- tokenization
class _Builder:
    """Build a text string while recording the CHAR span of each emitted piece.
    Spans are mapped to TOKEN spans after a single full tokenization (via the
    fast tokenizer's offset_mapping) — robust to BPE boundary merges, unlike
    incremental re-encoding."""

    def __init__(self):
        self.parts: list[str] = []
        self.pos = 0

    def emit(self, s: str) -> tuple[int, int]:
        start = self.pos
        self.parts.append(s)
        self.pos += len(s)
        return (start, self.pos)

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _char_to_tok_span(offs, c0, c1):
    """Tokens whose char-interval MIDPOINT lands in [c0, c1) → robust to
    leading-space token merges (a ` 1234` token is assigned to the digit span).
    Returns (t0, t1) exclusive, or None."""
    sel = [i for i, (o0, o1) in enumerate(offs)
           if o1 > o0 and c0 <= (o0 + o1) / 2.0 < c1]
    return (sel[0], sel[-1] + 1) if sel else None


def tokenize_record(rec, tok):
    """Build the token sequence + oracle spans for one alias-recall record.

    Returns None if a span fails to map cleanly or the value-holder's value
    tokens differ from the answer tokens (so exact-match scoring stays
    well-defined).
    """
    b = _Builder()
    b.emit("# config\n")
    slot_c = []                                       # per binding: char spans
    for (name, value) in rec["lines"]:
        nc = b.emit(name)
        b.emit(" = ")
        vc = b.emit(value)
        b.emit("\n")
        slot_c.append((nc, vc))
    b.emit("\n?")                                     # recall query on a new line
    qn_c = b.emit(rec["query_name"])
    b.emit(" = ")
    ans_c = b.emit(rec["answer"])

    enc = tok(b.text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    slots = []
    for (nc, vc) in slot_c:
        nspan = _char_to_tok_span(offs, *nc)
        vspan = _char_to_tok_span(offs, *vc)
        if nspan is None or vspan is None:
            return None
        slots.append(dict(nspan=nspan, vspan=vspan))
    qn_span = _char_to_tok_span(offs, *qn_c)
    ans_span = _char_to_tok_span(offs, *ans_c)
    if qn_span is None or ans_span is None:
        return None

    ap, ae = ans_span                                 # answer span [ap, ae)
    vlen = ae - ap
    if vlen < 1 or ap < 2:
        return None
    qb = rec["qb_line"]
    holder_v0, holder_v1 = slots[qb]["vspan"]
    # exact-match scoring requires the copy SOURCE (holder value tokens) to equal
    # the teacher-forced answer tokens.
    if (holder_v1 - holder_v0) < vlen:
        return None
    src_ans = ids[holder_v0:holder_v0 + vlen]
    if src_ans != ids[ap:ae]:
        return None
    return dict(
        ids=ids, slots=slots, ap=ap, vlen=vlen,
        qn_span=qn_span, qb=qb,
        alias_match=rec["alias_match_line"],
        n_vars=rec["n_vars"], hops=rec["hops"], kind=rec["kind"],
    )


# --------------------------------------------------------------- frozen trunk
def load_trunk(ckpt, device):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    model, cfg = build_model_from_ckpt(ckpt)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # return_hidden hands back the PRE-memory out_norm(h_raw) — the clean
    # contextual hidden (memory is inert here: no think tokens, read_mask=None).
    model._latent_feedback_premem = True
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    d_model = int(cfg["d_model"])
    return model, tok, d_model


@torch.no_grad()
def build_cache(model, tok, records, device, d_model, tag, max_len=1600,
                limit=None):
    """One frozen fp32 forward per record → the small slices the heads need
    (kept on CPU; lm_logits in fp16 to bound RAM). Skips records that exceed
    max_len or fail span consistency."""
    cache = []
    n_skip = 0
    for rec in records:
        ex = tokenize_record(rec, tok)
        if ex is None:
            n_skip += 1
            continue
        ids = ex["ids"]
        if len(ids) > max_len:
            n_skip += 1
            continue
        t = torch.tensor([ids], dtype=torch.long, device=device)
        logits, h = model(t, return_hidden=True)          # h = pre-mem out_norm
        h = h[0].float()                                   # (T, d)
        emb = model.embed(t)[0].float()                    # (T, d) input embeds
        ap, vlen = ex["ap"], ex["vlen"]
        qpos = ap - 1
        Nslots = len(ex["slots"])

        # per-slot: contextual key anchors + pooled name (input emb AND
        # contextual hidden) + value src
        h_vstart = torch.empty(Nslots, d_model)
        h_nameend = torch.empty(Nslots, d_model)
        name_pool = torch.empty(Nslots, d_model)          # INPUT-EMBEDDING pool
        name_pool_ctx = torch.empty(Nslots, d_model)      # CONTEXTUAL-HIDDEN pool
        src = torch.zeros(Nslots, vlen, dtype=torch.long)
        name_ids = []
        for j, sl in enumerate(ex["slots"]):
            ns, ne = sl["nspan"]
            v0, v1 = sl["vspan"]
            h_vstart[j] = h[v0]
            h_nameend[j] = h[ne - 1]
            name_pool[j] = emb[ns:ne].mean(0)
            name_pool_ctx[j] = h[ns:ne].mean(0)
            take = min(vlen, v1 - v0)
            src[j, :take] = torch.tensor(ids[v0:v0 + take])
            name_ids.append(ids[ns:ne])

        qns, qne = ex["qn_span"]
        cache.append(dict(
            h_query=h[qpos].cpu(),                          # (d) recall-pos hidden
            h_query_nameend=h[qne - 1].cpu(),               # (d) query name-end
            h_vstart=h_vstart.cpu(),                        # (N, d)
            h_nameend=h_nameend.cpu(),                      # (N, d)
            name_pool=name_pool.cpu(),                      # (N, d) input-emb pool
            name_pool_ctx=name_pool_ctx.cpu(),              # (N, d) ctx-hidden pool
            query_pool=emb[qns:qne].mean(0).cpu(),          # (d) input-emb pool
            query_pool_ctx=h[qns:qne].mean(0).cpu(),        # (d) ctx-hidden pool
            h_ans=h[qpos:qpos + vlen].cpu(),                # (vlen, d) copy gate
            lm_logits=logits[0, qpos:qpos + vlen].half().cpu(),  # (vlen, V)
            src=src,                                        # (N, vlen)
            ans_toks=torch.tensor(ids[ap:ap + vlen]),       # (vlen)
            name_ids=name_ids, query_name_ids=ids[qns:qne],
            qb=ex["qb"], alias_match=ex["alias_match"],
            N=Nslots, vlen=vlen, kind=ex["kind"], n_vars=ex["n_vars"],
            hops=ex["hops"],
        ))
        if limit is not None and len(cache) >= limit:
            break
    print(f"  [cache:{tag}] {len(cache)} examples (+{n_skip} skipped)")
    return cache


# ------------------------------------------------------------------- addressing
def _fnv1a(ids) -> int:
    h = 0x811C9DC5
    for x in ids:
        h ^= (int(x) & 0xFFFFFFFF)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class NameEncoder(nn.Module):
    """d_model pooled vector → d_key query/key (Linear → GELU → Linear)."""

    def __init__(self, d_model, d_key, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, d_key))

    def forward(self, x):
        return self.mlp(x)


class AddressingHead(nn.Module):
    """mode in {surface, contextual, hash}. `forward_attn(cache)` → (weights,
    logits) over the N binding slots. Shared copy readout + copy gate downstream
    (identical across arms — only the KEY differs)."""

    # contextual anchor → (query-hidden cache key, binding-hidden cache key)
    _CTX_ANCHOR = {"value": ("h_query", "h_vstart"),
                   "name": ("h_query_nameend", "h_nameend"),
                   "namepool": ("query_pool_ctx", "name_pool_ctx")}

    def __init__(self, mode, d_model, d_key=192, vocab=49216, hash_K=1 << 20,
                 ctx_anchor="value", gate_bias=2.0):
        super().__init__()
        self.mode = mode
        self.ctx_anchor = ctx_anchor
        self.vocab = vocab
        self.hash_K = hash_K
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        # FIX 2a: warm the gate OPEN (was -2.0 → sigmoid 0.12, starving the
        # addresser of gradient). +2.0 → sigmoid 0.88 copy mass from step 0.
        nn.init.constant_(self.gate.bias, gate_bias)
        hidden = max(d_key, d_model // 2)
        if mode == "surface":
            # Faithful repro of model.py `mem_soft_namekey`: ONE shared 2-layer
            # encoder over the pooled name-span input embeddings.
            self.enc = NameEncoder(d_model, d_key, hidden)
            self.log_tau = nn.Parameter(torch.tensor(math.log(20.0)))
        elif mode == "contextual":
            # FIX 1: equal-OR-GREATER capacity than surface — separate 2-layer
            # query/key MLPs so a contextual NEGATIVE means "even a capable
            # learned head can't extract it", not "a single linear was too weak".
            if ctx_anchor not in self._CTX_ANCHOR:
                raise ValueError(ctx_anchor)
            self.q_enc = NameEncoder(d_model, d_key, hidden)
            self.k_enc = NameEncoder(d_model, d_key, hidden)
            self.log_tau = nn.Parameter(torch.tensor(math.log(20.0)))
        elif mode != "hash":
            raise ValueError(mode)
        self.d_key = d_key

    def forward_attn(self, c, device):
        """Returns (weights, logits). weights = softmax read distribution over
        the N slots; logits = pre-softmax scores (for the attention-supervision
        aux loss). For hash, logits is None (no learnable params)."""
        if self.mode == "hash":
            bc = [_fnv1a(x) % self.hash_K for x in c["name_ids"]]
            qc = _fnv1a(c["query_name_ids"]) % self.hash_K
            m = torch.tensor([float(x == qc) for x in bc], device=device)
            w = m / m.sum().clamp_min(1e-9) if float(m.sum()) > 0 else m
            return w, None
        if self.mode == "surface":
            q = self.enc(c["query_pool"].to(device))             # (d_key,)
            k = self.enc(c["name_pool"].to(device))              # (N, d_key)
        else:  # contextual
            qk, kk = self._CTX_ANCHOR[self.ctx_anchor]
            q = self.q_enc(c[qk].to(device))                     # (d_key,)
            k = self.k_enc(c[kk].to(device))                     # (N, d_key)
        tau = self.log_tau.exp().clamp(2.0, 100.0)
        logits = F.cosine_similarity(q.unsqueeze(0), k, dim=-1) * tau  # (N,)
        return torch.softmax(logits, dim=-1), logits

    def recall_logp(self, c, attn, device):
        """Copy-mixed log p over the answer span: (vlen, vocab)."""
        lm_logits = c["lm_logits"].to(device).float()            # (vlen, V)
        h_ans = c["h_ans"].to(device)                            # (vlen, d)
        src = c["src"].to(device)                                # (N, vlen)
        vlen = c["vlen"]
        p_lm = torch.softmax(lm_logits, dim=-1)
        p_copy = lm_logits.new_zeros(vlen, self.vocab)
        for j in range(vlen):
            p_copy[j].scatter_add_(0, src[:, j], attn)
        copy_mass = attn.sum()
        g = torch.sigmoid(self.gate(h_ans)).squeeze(-1)          # (vlen,)
        g = g * (copy_mass > 0).float()
        p = (1 - g).unsqueeze(-1) * p_lm + g.unsqueeze(-1) * p_copy
        return torch.log(p + 1e-9)


def train_arm(head, cache, order, device, lr, attn_sup_weight, tag):
    """One curriculum phase. `order` is a list of indices into `cache`. Loss =
    recall CE + attn_sup_weight · attention-CE(target = correct binding qb).
    The attention term (FIX 2b) gives the addresser gradient INDEPENDENT of the
    copy gate."""
    opt = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    head.train()
    for step, idx in enumerate(order, 1):
        c = cache[idx]
        attn, logits = head.forward_attn(c, device)
        logp = head.recall_logp(c, attn, device)
        ce = F.nll_loss(logp, c["ans_toks"].to(device))
        loss = ce
        attn_ce = torch.tensor(0.0)
        if logits is not None and attn_sup_weight > 0:
            tgt = torch.tensor([c["qb"]], device=device)
            attn_ce = F.cross_entropy(logits.unsqueeze(0), tgt)
            loss = loss + attn_sup_weight * attn_ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"    [{head.mode:11s} {tag:5s}] step {step:4d} "
                  f"ce={float(ce):.4f} attn_ce={float(attn_ce):.4f}")
    head.eval()


@torch.no_grad()
def eval_arm(head, cache, device):
    top1 = mass = 0.0
    alias_top1 = 0.0
    on_e = on_f = 0
    for c in cache:
        attn, _ = head.forward_attn(c, device)
        qb, am = c["qb"], c["alias_match"]
        top1 += int(attn.argmax().item() == qb)
        alias_top1 += int(attn.argmax().item() == am)
        mass += float(attn[qb])
        logp = head.recall_logp(c, attn, device)
        ans = c["ans_toks"].to(device)
        pred = logp.argmax(-1)
        on_e += int(bool((pred == ans).all()))
        on_f += int(pred[0].item() == ans[0].item())
    n = max(1, len(cache))
    return dict(n=len(cache), top1=top1 / n, mass=mass / n,
                alias_top1=alias_top1 / n, on_exact=on_e / n, on_first=on_f / n)


@torch.no_grad()
def wmoff(cache):
    e = f = 0
    for c in cache:
        off = c["lm_logits"].float().argmax(-1)
        ans = c["ans_toks"]
        e += int(bool((off == ans).all()))
        f += int(off[0].item() == ans[0].item())
    n = max(1, len(cache))
    return e / n, f / n


# --------------------------------------------------------------- whitening diag
@torch.no_grad()
def _pca_components(cache, k_anchor, device, max_vecs=20000):
    """Mean + top-component basis of the stacked binding hiddens for one anchor.
    Returns (mean (d,), V (d, kmax)) where V columns are the top principal
    directions (the 'dominant shared component' the raw-cosine diagnostic showed
    bunching all hiddens at +0.82)."""
    vecs = []
    for c in cache:
        vecs.append(c[k_anchor])              # (N, d)
        if sum(v.shape[0] for v in vecs) >= max_vecs:
            break
    X = torch.cat(vecs, 0).to(device).float()  # (M, d)
    mean = X.mean(0)
    Xc = X - mean
    # economy SVD → right-singular vectors are principal directions
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return mean, Vh                            # Vh: (min(M,d), d)


@torch.no_grad()
def pca_whiten_top1(cache, q_anchor, k_anchor, mean, Vh, kremove, device):
    """Project OUT the top-`kremove` principal directions from both query and
    binding hiddens, then raw cosine → holder-identification top1 + beats_alias.
    kremove=0 reproduces plain raw cosine."""
    V = Vh[:kremove].to(device) if kremove > 0 else None
    top1 = beats = 0
    n = 0
    for c in cache:
        q = c[q_anchor].to(device).float() - mean
        k = c[k_anchor].to(device).float() - mean
        if V is not None:
            q = q - (q @ V.t()) @ V
            k = k - (k @ V.t()) @ V
        sim = F.cosine_similarity(q.unsqueeze(0), k, dim=-1)
        qb, am = c["qb"], c["alias_match"]
        top1 += int(sim.argmax().item() == qb)
        beats += int(float(sim[qb]) > float(sim[am]))
        n += 1
    nz = max(1, n)
    return top1 / nz, beats / nz


class _LinearAddr(nn.Module):
    """Learned LINEAR projection (no GELU) trained ONLY to identify the holder
    via attention-CE. Isolates 'can a learned LINEAR key separate the holder?'
    from 'can raw cosine?' (PCA) and 'can a nonlinear MLP?' (the main ctx arm)."""

    def __init__(self, d_model, d_proj=128):
        super().__init__()
        self.q = nn.Linear(d_model, d_proj, bias=False)
        self.k = nn.Linear(d_model, d_proj, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(20.0)))

    def logits(self, c, q_anchor, k_anchor, device):
        q = self.q(c[q_anchor].to(device).float())
        k = self.k(c[k_anchor].to(device).float())
        tau = self.log_tau.exp().clamp(2.0, 100.0)
        return F.cosine_similarity(q.unsqueeze(0), k, dim=-1) * tau


def fit_linear_addr(train_cache, q_anchor, k_anchor, device, steps, lr,
                    d_proj=128):
    head = _LinearAddr(train_cache[0][q_anchor].shape[-1], d_proj).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    rng = random.Random(1234)
    head.train()
    for _ in range(steps):
        c = train_cache[rng.randrange(len(train_cache))]
        logits = head.logits(c, q_anchor, k_anchor, device)
        tgt = torch.tensor([c["qb"]], device=device)
        loss = F.cross_entropy(logits.unsqueeze(0), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def eval_linear_addr(head, cache, q_anchor, k_anchor, device):
    top1 = beats = 0
    n = 0
    for c in cache:
        logits = head.logits(c, q_anchor, k_anchor, device)
        qb, am = c["qb"], c["alias_match"]
        top1 += int(logits.argmax().item() == qb)
        beats += int(float(logits[qb]) > float(logits[am]))
        n += 1
    nz = max(1, n)
    return top1 / nz, beats / nz


# ===================================================================== LoRA arm
# THE DECISIVE CONTROL (2026-06-17). The frozen-trunk result is: EXACT name->
# binding identity is linearly recoverable from the name-span contextual hidden
# (~0.9), but ALIAS/reference resolution (C->B->A->value) is at CHANCE for every
# key. Open question: is that because this FROZEN task-naive trunk never LEARNED
# reference-following, or because the linear-RNN architecture CAN'T do it?
#
# Control: attach a small LoRA adapter to the trunk's DeltaNet q/k/v/o (and
# optionally MLP) linear layers, UNFREEZE only {LoRA + the contextual name-span
# addresser + copy head}, co-train on the ALIAS recall TRAIN split, then evaluate
# on HELD-OUT alias names (disjoint from train -> generalization, not memory).
#   * LoRA recovers heldout alias holder-id >> chance, exact still works, general
#     CE no-harm, train ~= heldout  -> the linear-RNN trunk CAN learn reference-
#     following with light training; the fix for indirect recall is trunk
#     training, not a smarter key.
#   * LoRA stays ~chance on heldout, OR only memorizes (train >> heldout) -> a
#     deeper architectural limit of the attention-free linear RNN; general recall
#     needs an architectural change (attention/depth), not just training.


# Trunk-forward precision for extract_live. The frozen-trunk established result
# was measured in fp32 (build_cache, no autocast); on this sparse precision-
# sensitive recall task fp32 is the safe choice (CLAUDE.md: "bf16-on-sparse-loss
# collapses logits"). The fp32-vs-bf16 diagnostic showed only a ~0.05 holder-id
# difference, but we run the LoRA control in fp32 to match the established
# apparatus exactly. Set False -> fp32, True -> bf16 autocast.
EXTRACT_AUTOCAST = True


class LoRALinear(nn.Module):
    """Wraps a FROZEN nn.Linear with a low-rank update  base(x) + s*B(A(x)).

    Standard LoRA init: A ~ kaiming, B = 0  -> at init the adapter contributes
    EXACTLY zero (byte-identical to the base trunk; the no-harm delta starts at
    0). Only A, B carry gradient; the wrapped base weight stays requires_grad
    False. `enabled=False` disables the low-rank branch (used to measure the base
    LM / the frozen-trunk baseline through the IDENTICAL extraction path)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # A random, B zero
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            lo = F.linear(F.linear(x, self.A), self.B) * self.scaling
            out = out + lo.to(out.dtype)
        return out


def attach_lora(model, rank, alpha, target_mlp=False):
    """Replace the DeltaNet q/k/v/o projections (and optionally the GLU MLP) of
    every block with LoRALinear wrappers. Returns (lora_modules, n_params)."""
    from experiments.layers import _FlaWrapper
    mods = []
    for blk in model.blocks:
        attn = getattr(blk, "attn", None)
        if isinstance(attn, _FlaWrapper):
            layer = attn.layer
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                base = getattr(layer, nm, None)
                if isinstance(base, nn.Linear):
                    w = LoRALinear(base, rank, alpha).to(base.weight.device)
                    setattr(layer, nm, w)
                    mods.append(w)
        if target_mlp and hasattr(blk, "mlp"):
            for nm in ("W_g", "W_u", "W_d"):
                base = getattr(blk.mlp, nm, None)
                if isinstance(base, nn.Linear):
                    w = LoRALinear(base, rank, alpha).to(base.weight.device)
                    setattr(blk.mlp, nm, w)
                    mods.append(w)
    n_params = sum(m.A.numel() + m.B.numel() for m in mods)
    return mods, n_params


def set_lora_enabled(mods, flag):
    for m in mods:
        m.enabled = bool(flag)


def lora_params(mods):
    ps = []
    for m in mods:
        ps += [m.A, m.B]
    return ps


@torch.no_grad()
def assert_base_frozen(model, lora_mods):
    """Sanity: EXACTLY the LoRA A/B params are trainable; every base trunk
    weight (including each wrapped Linear's `.base`) is frozen."""
    lora_ids = {id(p) for m in lora_mods for p in (m.A, m.B)}
    n_train = n_train_lora = n_base = 0
    for p in model.parameters():
        if p.requires_grad:
            n_train += 1
            if id(p) in lora_ids:
                n_train_lora += 1
        else:
            n_base += 1
    leaked = n_train - n_train_lora
    return dict(n_trainable=n_train, n_trainable_is_lora=n_train_lora,
                n_frozen_base=n_base, leaked=leaked)


def extract_live(model, ex, dev, *, detach):
    """One trunk forward (the SAME normal forward build_cache uses, incl. FiLM
    K=3 self-feed) -> the small slices the namepool addresser + copy head need.
    detach=True  -> run under no_grad, return CPU tensors (lm_logits fp16) for a
                    static eval/baseline cache.
    detach=False -> run WITH grad, return live GPU tensors so the LoRA + heads
                    receive gradient through the trunk."""
    ids = ex["ids"]
    t = torch.tensor([ids], dtype=torch.long, device=dev)
    grad_ctx = torch.no_grad() if detach else nullcontext()
    ac = (torch.autocast("cuda", dtype=torch.bfloat16) if EXTRACT_AUTOCAST
          else nullcontext())
    with grad_ctx:
        with ac:
            logits, h = model(t, return_hidden=True)
    h = h[0].float()                                    # (T, d)  pre-mem out_norm
    logits = logits[0].float()                          # (T, V)
    ap, vlen = ex["ap"], ex["vlen"]
    qpos = ap - 1
    slots = ex["slots"]
    N = len(slots)
    name_pool_ctx = torch.stack(
        [h[s["nspan"][0]:s["nspan"][1]].mean(0) for s in slots])   # (N, d)
    h_nameend = torch.stack([h[s["nspan"][1] - 1] for s in slots])  # (N, d)
    qns, qne = ex["qn_span"]
    query_pool_ctx = h[qns:qne].mean(0)                            # (d,)
    h_query_nameend = h[qne - 1]                                   # (d,)
    h_ans = h[qpos:qpos + vlen]                                    # (vlen, d)
    lm_logits = logits[qpos:qpos + vlen]                          # (vlen, V)
    src = torch.zeros(N, vlen, dtype=torch.long, device=dev)
    for j, s in enumerate(slots):
        v0, v1 = s["vspan"]
        take = min(vlen, v1 - v0)
        src[j, :take] = torch.tensor(ids[v0:v0 + take], device=dev)
    ans_toks = torch.tensor(ids[ap:ap + vlen], device=dev)
    c = dict(query_pool_ctx=query_pool_ctx, name_pool_ctx=name_pool_ctx,
             h_nameend=h_nameend, h_query_nameend=h_query_nameend,
             h_ans=h_ans, lm_logits=lm_logits, src=src, ans_toks=ans_toks,
             vlen=vlen, qb=ex["qb"], alias_match=ex["alias_match"], N=N,
             kind=ex["kind"], n_vars=ex["n_vars"], hops=ex["hops"])
    if detach:
        out = {}
        for k, v in c.items():
            if torch.is_tensor(v):
                out[k] = (v.half() if k == "lm_logits" else v).detach().cpu()
            else:
                out[k] = v
        return out
    return c


def tokenize_records(records, tok, max_len, limit):
    """records -> list of `ex` metadata dicts (ids+spans), filtered for clean
    span mapping + length. Same filter build_cache applies, decoupled from the
    forward so the SAME record set drives the frozen-cache and live LoRA paths."""
    out, n_skip = [], 0
    for rec in records:
        ex = tokenize_record(rec, tok)
        if ex is None or len(ex["ids"]) > max_len:
            n_skip += 1
            continue
        out.append(ex)
        if limit is not None and len(out) >= limit:
            break
    return out, n_skip


def live_cache(model, ex_list, dev, lora_mods, lora_enabled):
    set_lora_enabled(lora_mods, lora_enabled)
    return [extract_live(model, ex, dev, detach=True) for ex in ex_list]


@torch.no_grad()
def general_ce_noharm(model, tok, dev, lora_mods, n_windows=32, T=512):
    """No-harm probe: mean next-token CE on general OUT-OF-alias-domain Python
    (stdlib source), with LoRA OFF (base LM) vs ON. If the LoRA wrecks the base
    LM the alias 'win' is hollow. Same model, same windows; only the adapter
    toggles."""
    import argparse as _ap
    import difflib as _df
    import json as _js
    import pprint as _pp
    texts = []
    for m in (_ap, _js, _df, _pp):
        f = getattr(m, "__file__", None)
        if f and f.endswith(".py") and os.path.exists(f):
            with open(f) as fh:
                texts.append(fh.read())
    big = "\n\n".join(texts)
    ids = tok(big, add_special_tokens=False)["input_ids"]
    windows = [ids[i:i + T] for i in range(0, max(0, len(ids) - T), T)][:n_windows]
    if not windows:
        return None, None, 0

    def _pass():
        tot = 0.0
        ntok = 0
        for w in windows:
            t = torch.tensor([w], dtype=torch.long, device=dev)
            ac = (torch.autocast("cuda", dtype=torch.bfloat16)
                  if EXTRACT_AUTOCAST else nullcontext())
            with ac:
                logits, _ = model(t, return_hidden=True)
            logits = logits[0].float()
            loss = F.cross_entropy(logits[:-1], t[0, 1:])
            tot += float(loss) * (len(w) - 1)
            ntok += (len(w) - 1)
        return tot / max(1, ntok)

    set_lora_enabled(lora_mods, False)
    base = _pass()
    set_lora_enabled(lora_mods, True)
    lora = _pass()
    return base, lora, len(windows)


def eval_linaddr_groups(addr, lora_mods, enabled, model, dev, q_anchor,
                        k_anchor, eval_groups, probe_groups):
    """Holder-id top1 (addr argmax == qb) per (kind,N) on heldout + per-kind on
    the train-name probe. Caches built live (no_grad) for THIS arm."""
    held = {}
    for (kind, N), ex in eval_groups.items():
        cache = live_cache(model, ex, dev, lora_mods, enabled)
        t1, beats = eval_linear_addr(addr, cache, q_anchor, k_anchor, dev)
        am = 0
        for c in cache:
            lg = addr.logits(c, q_anchor, k_anchor, dev)
            am += int(lg.argmax().item() == c["alias_match"])
        held.setdefault(kind, {})[N] = dict(top1=t1, beats=beats,
                                            alias_t1=am / max(1, len(cache)),
                                            chance=1.0 / max(1, N))
    tr = {}
    for (kind, _), ex in probe_groups.items():
        cache = live_cache(model, ex, dev, lora_mods, enabled)
        t1, _ = eval_linear_addr(addr, cache, q_anchor, k_anchor, dev)
        tr[kind] = t1
    return held, tr


def train_linaddr(addr, lora_mods, ex_list, model, dev, q_anchor, k_anchor, *,
                  steps, accum, addr_lr, lora_lr, lora_enabled, seed, tag):
    """Co-train a learned-LINEAR addresser (+ LoRA, if enabled) on the PURE
    holder-id objective (attention-CE: the read must land on the true value-
    holder slot qb). No copy/recall head -- this IS the established 'learned-
    linear holder-id' metric, so any held-out alias lift is attributable to the
    TRUNK (LoRA) making the resolved reference linearly recoverable."""
    groups = [{"params": list(addr.parameters()), "lr": addr_lr}]
    if lora_enabled and lora_mods:
        groups.append({"params": lora_params(lora_mods), "lr": lora_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    set_lora_enabled(lora_mods, lora_enabled)
    detach = not (lora_enabled and lora_mods)
    addr.train()
    rng = random.Random(seed)
    opt.zero_grad(set_to_none=True)
    run = 0.0
    micro = 0
    for step in range(1, steps + 1):
        ex = ex_list[rng.randrange(len(ex_list))]
        c = extract_live(model, ex, dev, detach=detach)
        logits = addr.logits(c, q_anchor, k_anchor, dev)
        tgt = torch.tensor([c["qb"]], device=dev)
        loss = F.cross_entropy(logits.unsqueeze(0), tgt)
        (loss / accum).backward()
        run += float(loss.detach())
        micro += 1
        if step % accum == 0:
            allp = [p for g in groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % (accum * 400) == 0 or step == 1:
            print(f"    [{tag:16s}] step {step:5d} attn_ce={run / micro:.4f}")
            run = 0.0
            micro = 0
    addr.eval()
    opt.zero_grad(set_to_none=True)
    return addr


def reset_lora(lora_mods):
    """Re-init every LoRA adapter to the at-init state (A kaiming, B zero) so a
    fresh lr point in the sweep starts from the byte-identical base trunk."""
    with torch.no_grad():
        for m in lora_mods:
            nn.init.kaiming_uniform_(m.A, a=math.sqrt(5))
            nn.init.zeros_(m.B)
            m.enabled = True


def run_lora_control(args, model, tok, d_model, vocab, dev):
    global EXTRACT_AUTOCAST
    EXTRACT_AUTOCAST = bool(args.lora_autocast)     # default False -> fp32
    print("\n" + "#" * 100)
    print("# LoRA CONTROL -- can a LIGHTLY-TRAINED linear-RNN trunk learn "
          "reference-following?")
    print(f"#   metric = learned-LINEAR holder-id on the '{args.lora_anchor}' "
          f"name-span anchor (the established exact metric).  "
          f"trunk fwd = {'bf16' if EXTRACT_AUTOCAST else 'fp32'}")
    print("#" * 100)
    kinds = ["exact", "alias1", "alias2"]
    nvs = [int(x) for x in args.n_vars.split(",")]
    ANCHORS = {"name": ("h_query_nameend", "h_nameend"),
               "namepool": ("query_pool_ctx", "name_pool_ctx")}
    qA, kA = ANCHORS[args.lora_anchor]

    print("[tokenize] train + heldout")
    train_recs = _load_jsonl(args.train_data)
    random.Random(args.seed + 1).shuffle(train_recs)
    train_ex, sk = tokenize_records(train_recs, tok, args.max_len,
                                    args.n_lora_train)
    print(f"  train: {len(train_ex)} ex (+{sk} skipped)")
    train_by_kind = {k: [e for e in train_ex if e["kind"] == k] for k in kinds}
    probe_groups = {(k, -1): train_by_kind[k][:args.n_eval] for k in kinds
                    if train_by_kind[k]}
    eval_groups = {}
    for kind in kinds:
        for N in nvs:
            p = os.path.join(args.data_dir,
                             f"alias_recall_{kind}_N{N}_heldout.jsonl")
            if not os.path.exists(p):
                continue
            ex, _ = tokenize_records(_load_jsonl(p), tok, args.max_len,
                                     args.n_eval)
            eval_groups[(kind, N)] = ex

    d_proj = args.whiten_dproj

    # ======================= FROZEN baseline =================================
    print("\n[FROZEN baseline] co-train a learned-linear addresser on the FROZEN "
          "trunk (LoRA off)")
    torch.manual_seed(args.seed + 100)
    fz_addr = _LinearAddr(d_model, d_proj).to(dev)
    train_linaddr(fz_addr, [], train_ex, model, dev, qA, kA,
                  steps=args.lora_steps, accum=args.lora_accum, addr_lr=args.lr,
                  lora_lr=0.0, lora_enabled=False, seed=args.seed + 3,
                  tag="frozen")
    fz_held, fz_tr = eval_linaddr_groups(fz_addr, [], False, model, dev, qA, kA,
                                         eval_groups, probe_groups)

    # ======================= LoRA arm (lr sweep) =============================
    print(f"\n[LoRA arm] attach LoRA (rank={args.lora_rank} "
          f"alpha={args.lora_alpha} mlp={args.lora_mlp}); "
          f"sweep lr={args.lora_lr_sweep}")
    lora_mods, n_lora = attach_lora(model, args.lora_rank, args.lora_alpha,
                                    target_mlp=args.lora_mlp)
    frz = assert_base_frozen(model, lora_mods)
    print(f"  LoRA params: {n_lora:,} over {len(lora_mods)} linear layers")
    print(f"  freeze check: trainable={frz['n_trainable']} "
          f"(all LoRA={frz['n_trainable_is_lora']}, leaked={frz['leaked']}) "
          f"frozen_base={frz['n_frozen_base']}")
    if frz["leaked"] != 0:
        raise RuntimeError("base-weight leak: non-LoRA trunk params trainable")
    b0, l0, _ = general_ce_noharm(model, tok, dev, lora_mods)
    print(f"  no-harm @init (must be ~0): base={b0:.4f} lora={l0:.4f} "
          f"Δ={l0 - b0:+.4f}")

    lrs = [float(x) for x in str(args.lora_lr_sweep).split(",")]
    lora_runs = []
    for lr in lrs:
        print(f"\n  --- LoRA lr={lr:g} ---")
        reset_lora(lora_mods)
        torch.manual_seed(args.seed + 100)
        addr = _LinearAddr(d_model, d_proj).to(dev)
        train_linaddr(addr, lora_mods, train_ex, model, dev, qA, kA,
                      steps=args.lora_steps, accum=args.lora_accum,
                      addr_lr=args.lr, lora_lr=lr, lora_enabled=True,
                      seed=args.seed + 3, tag=f"lora lr={lr:g}")
        held, tr = eval_linaddr_groups(addr, lora_mods, True, model, dev, qA, kA,
                                       eval_groups, probe_groups)
        base_ce, lora_ce, nw = general_ce_noharm(model, tok, dev, lora_mods)
        lora_runs.append(dict(lr=lr, held=held, tr=tr, base_ce=base_ce,
                              lora_ce=lora_ce, dce=lora_ce - base_ce, nw=nw))
        print(f"    no-harm: base={base_ce:.4f} lora={lora_ce:.4f} "
              f"Δ={lora_ce - base_ce:+.4f}")

    # =============================== REPORT ==================================
    def _agg(held, kind, key):
        d = held.get(kind, {})
        return (sum(v[key] for v in d.values()) / len(d)) if d else float("nan")

    print("\n" + "=" * 110)
    print(f"LoRA CONTROL RESULTS -- held-out alias names (disjoint from train); "
          f"holder-id via learned-linear on '{args.lora_anchor}' anchor.")
    print("=" * 110)

    print("\n(1) HOLDER-ID top1 per kind/N  (frozen vs LoRA@each lr)")
    cols = "".join(f"{('lora' + format(r['lr'], 'g')):>12s}" for r in lora_runs)
    print(f"  {'kind':7s} {'N':>3s} {'chance':>7s} {'frozen':>8s}{cols}")
    for kind in kinds:
        for N in nvs:
            fh = fz_held.get(kind, {}).get(N)
            if fh is None:
                continue
            row = f"  {kind:7s} {N:>3d} {fh['chance']:>7.3f} {fh['top1']:>8.3f}"
            for r in lora_runs:
                v = r["held"].get(kind, {}).get(N, {})
                row += f"{v.get('top1', float('nan')):>12.3f}"
            print(row)

    print("\n(2) PER-KIND aggregate (mean over N): frozen_held frozen_train | "
          "per-lr  held train gap")
    verdict = {}
    for kind in kinds:
        ch = _agg(fz_held, kind, "chance")
        row = (f"  {kind:7s} chance={ch:>6.3f} | frozen {_agg(fz_held, kind, 'top1'):>6.3f}"
               f" {fz_tr.get(kind, float('nan')):>6.3f} |")
        kv = {}
        for r in lora_runs:
            h = _agg(r["held"], kind, "top1")
            t = r["tr"].get(kind, float("nan"))
            row += f"  lr{r['lr']:g}: {h:>6.3f} {t:>6.3f} gap{t - h:>+6.3f}"
            kv[r["lr"]] = dict(held=h, train=t, gap=t - h)
        verdict[kind] = dict(chance=ch, frozen=_agg(fz_held, kind, "top1"),
                             lrs=kv)
        print(row)

    print("\n(3) NO-HARM -- general stdlib-Python next-token CE (LoRA off vs on)")
    for r in lora_runs:
        print(f"  lr={r['lr']:g}: base={r['base_ce']:.4f} lora={r['lora_ce']:.4f} "
              f"Δ={r['dce']:+.4f} ({'OK' if r['dce'] < 0.10 else 'HARM'})")

    # ------------------------------- verdict ---------------------------------
    print("\n" + "=" * 110)
    print("VERDICT")
    print("=" * 110)
    ok = [r for r in lora_runs if r['dce'] < 0.10]
    pick = (max(ok, key=lambda r: r['lr']) if ok
            else min(lora_runs, key=lambda r: r['lr']))['lr']
    print(f"  exact (name-identity) — frozen {verdict['exact']['frozen']:.3f} vs "
          f"LoRA@{pick:g} {verdict['exact']['lrs'][pick]['held']:.3f} "
          f"(chance {verdict['exact']['chance']:.3f})  "
          f"[exact must STAY high = no regression]")
    learned = False
    for kind in ("alias1", "alias2"):
        v = verdict[kind]
        hop = "1-hop" if kind == "alias1" else "2-hop"
        print(f"  {hop} ALIAS  (frozen held={v['frozen']:.3f}, "
              f"chance {v['chance']:.3f}):")
        for lr, d in v['lrs'].items():
            r = next(rr for rr in lora_runs if rr['lr'] == lr)
            above = (d['held'] > max(v['chance'] * 3, v['chance'] + 0.10)
                     and d['held'] - v['frozen'] > 0.08)
            memo = (d['train'] > max(v['chance'] * 3, 0.20)
                    and d['held'] <= v['chance'] + 0.08)
            tag = ("ABOVE-CHANCE+generalizes" if above and d['gap'] < 0.20
                   else "MEMORIZED(train>>held)" if memo else "at~chance")
            print(f"    lr={lr:g}: held={d['held']:.3f} train={d['train']:.3f} "
                  f"gap={d['gap']:+.3f} noharm_Δ={r['dce']:+.4f} -> {tag}")
            if above and d['gap'] < 0.20 and r['dce'] < 0.10:
                learned = True
    print()
    if learned:
        print("  >>> TRAINABLE-TRUNK LEARNS REFERENCE-FOLLOWING: a light, no-harm "
              "LoRA recovers HELD-OUT alias holder-id above chance and "
              "generalizes (train~=held). The fix for indirect/general recall is "
              "TRUNK TRAINING, not a smarter key.")
    else:
        any_memorize = any(
            d['train'] > max(verdict[k]['chance'] * 3, 0.20)
            for k in ('alias1', 'alias2') for d in verdict[k]['lrs'].values())
        print("  >>> DEEPER ARCHITECTURAL LIMIT: no no-harm LoRA recovered "
              "HELD-OUT alias above chance.")
        if any_memorize:
            print("      The trunk COULD fit alias on TRAIN names (memorization) "
                  "but did NOT generalize to held-out names -> light LoRA forms a "
                  "name-specific lookup, not a general reference-following "
                  "circuit.")
        else:
            print("      The trunk could not even FIT alias on train names with a "
                  "linear addresser -> reference-following is not representable in "
                  "the contextual hidden under light adaptation.")
        print("      Either way: general indirect recall needs an architectural "
              "change (attention/depth), not just light trunk training.")



# ------------------------------------------------------------------------- main
def _load_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# label -> (mode, ctx_anchor)
ARMS = [("surface", "surface", None),
        ("ctx_value", "contextual", "value"),
        ("ctx_name", "contextual", "name"),
        ("ctx_namepool", "contextual", "namepool"),
        ("hash", "hash", None)]
ARM_LABELS = [a[0] for a in ARMS]
# the three contextual anchors for the whitening diagnostic
WHITEN_ANCHORS = [("ctx_value", "h_query", "h_vstart"),
                  ("ctx_name", "h_query_nameend", "h_nameend"),
                  ("ctx_namepool", "query_pool_ctx", "name_pool_ctx")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="checkpoints/pretrain_v12_step14306_tok3750232064.pt")
    ap.add_argument("--data_dir", default="/tmp/alias_data")
    ap.add_argument("--train_data",
                    default="/tmp/alias_data/alias_recall_train.jsonl")
    ap.add_argument("--n_vars", default="16,32,64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_key", type=int, default=192)
    ap.add_argument("--gate_bias", type=float, default=2.0)
    ap.add_argument("--attn_sup_weight", type=float, default=1.0)
    ap.add_argument("--exact_steps", type=int, default=2000)   # phase 1
    ap.add_argument("--alias_steps", type=int, default=2000)   # phase 2
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_exact_train", type=int, default=1400)
    ap.add_argument("--n_alias_train", type=int, default=1800)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_train_eval", type=int, default=200)  # train-recall probe
    ap.add_argument("--max_len", type=int, default=1600)
    ap.add_argument("--whiten_steps", type=int, default=2000)
    ap.add_argument("--whiten_dproj", type=int, default=128)
    # ---- LoRA control (the decisive trainable-trunk arm) -------------------
    ap.add_argument("--lora_control", action="store_true",
                    help="run ONLY the trunk-LoRA reference-following control")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_lr", type=float, default=2e-4)
    ap.add_argument("--lora_lr_sweep", default="1e-4,3e-4",
                    help="comma list of LoRA LRs to sweep (each resets LoRA)")
    ap.add_argument("--lora_steps", type=int, default=3000)
    ap.add_argument("--lora_accum", type=int, default=1)
    ap.add_argument("--lora_mlp", action="store_true",
                    help="also adapt the GLU MLP (default: q/k/v/o only)")
    ap.add_argument("--lora_anchor", default="name",
                    choices=["name", "namepool"],
                    help="name-span contextual anchor for holder-id")
    ap.add_argument("--lora_autocast", action="store_true",
                    help="bf16-autocast the trunk fwd (default fp32, matches "
                         "the established frozen-trunk apparatus)")
    ap.add_argument("--n_lora_train", type=int, default=2000)
    ap.add_argument("--n_fit", type=int, default=600,
                    help="records for the learned-linear confirmation fit")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.exact_steps = 300
        args.alias_steps = 300
        args.n_exact_train = 120
        args.n_alias_train = 120
        args.n_eval = 30
        args.n_train_eval = 30
        args.whiten_steps = 300
        args.n_vars = "8,16"
        args.lora_steps = 120
        args.n_lora_train = 120
        args.n_eval = 20
        args.lora_lr_sweep = "3e-4"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = args.device
    nvs = [int(x) for x in args.n_vars.split(",")]
    kinds = ["exact", "alias1", "alias2"]

    print(f"[cfg] ckpt={args.ckpt} device={dev} d_key={args.d_key} "
          f"gate_bias={args.gate_bias} attn_sup={args.attn_sup_weight} "
          f"exact_steps={args.exact_steps} alias_steps={args.alias_steps} "
          f"n_vars={nvs}")
    print("[load trunk]")
    model, tok, d_model = load_trunk(args.ckpt, dev)
    vocab = int(model.lm_head.weight.shape[0])
    print(f"  d_model={d_model} vocab={vocab}")

    if args.lora_control:
        run_lora_control(args, model, tok, d_model, vocab, dev)
        return

    # ---- training pools (split by kind for the exact-first curriculum) -------
    print("[cache train]")
    train_recs = _load_jsonl(args.train_data)
    random.Random(args.seed + 1).shuffle(train_recs)
    exact_recs = [r for r in train_recs if r["kind"] == "exact"]
    alias_recs = [r for r in train_recs if r["kind"] in ("alias1", "alias2")]
    exact_train = build_cache(model, tok, exact_recs, dev, d_model, "exact_tr",
                              max_len=args.max_len, limit=args.n_exact_train)
    alias_train = build_cache(model, tok, alias_recs, dev, d_model, "alias_tr",
                              max_len=args.max_len, limit=args.n_alias_train)
    # phase-2 corpus = exact + alias mix (so the head keeps EXACT while learning
    # alias — a general addresser must do both).
    mix_train = exact_train + alias_train
    # held-in train-recall probes (does the arm even FIT what it trained on?)
    exact_train_probe = exact_train[:args.n_train_eval]
    alias_train_probe = alias_train[:args.n_train_eval]

    # ---- per (kind, N) held-out eval sets -----------------------------------
    print("[cache eval]")
    eval_sets = []
    for kind in kinds:
        for N in nvs:
            p = os.path.join(args.data_dir,
                             f"alias_recall_{kind}_N{N}_heldout.jsonl")
            if not os.path.exists(p):
                continue
            recs = _load_jsonl(p)
            cache = build_cache(model, tok, recs, dev, d_model,
                                f"{kind}_N{N}", max_len=args.max_len,
                                limit=args.n_eval)
            eval_sets.append((f"{kind}_N{N}", kind, cache))

    # ---- train each arm: PHASE 1 exact-only, snapshot exact, PHASE 2 mix -----
    ex_rng = random.Random(args.seed + 7)
    al_rng = random.Random(args.seed + 9)
    order_exact = [ex_rng.randrange(len(exact_train))
                   for _ in range(args.exact_steps)]
    order_mix = [al_rng.randrange(len(mix_train))
                 for _ in range(args.alias_steps)]

    heads = {}
    # exact[label] = eval after phase 1; the SANITY-GATE result for the table.
    exact_eval = {}            # label -> {set_name: eval_arm dict}
    exact_train_rec = {}       # label -> train-recall (exact) after phase 1
    for label, mode, anchor in ARMS:
        print(f"\n[arm {label}] PHASE 1 (exact-only)")
        head = AddressingHead(mode, d_model, d_key=args.d_key, vocab=vocab,
                              ctx_anchor=(anchor or "value"),
                              gate_bias=args.gate_bias).to(dev)
        train_arm(head, exact_train, order_exact, dev, args.lr,
                  args.attn_sup_weight, "exact")
        # snapshot EXACT (the sanity gate) BEFORE any alias training
        exact_eval[label] = {nm: eval_arm(head, cache, dev)
                             for nm, kind, cache in eval_sets if kind == "exact"}
        exact_train_rec[label] = eval_arm(head, exact_train_probe, dev)
        print(f"\n[arm {label}] PHASE 2 (exact+alias mix)")
        train_arm(head, mix_train, order_mix, dev, args.lr,
                  args.attn_sup_weight, "alias")
        heads[label] = head

    # ================= SANITY GATE ======================================
    print("\n" + "=" * 100)
    print("SANITY GATE — EXACT recall after PHASE 1 (exact-only training). "
          "TRAIN-recall confirms the arm FITS.")
    print("=" * 100)
    print(f"  {'arm':12s} | {'train_addr':>10s} {'train_rec':>9s} | "
          f"{'held_addr':>9s} {'held_rec':>8s}  (held = mean over exact N)")
    for label in ARM_LABELS:
        tr = exact_train_rec[label]
        hv = list(exact_eval[label].values())
        h_addr = sum(d["top1"] for d in hv) / max(1, len(hv))
        h_rec = sum(d["on_exact"] for d in hv) / max(1, len(hv))
        print(f"  {label:12s} | {tr['top1']:>10.3f} {tr['on_exact']:>9.3f} | "
              f"{h_addr:>9.3f} {h_rec:>8.3f}")
    ctx_best = max(
        sum(exact_eval[lb][nm]["on_exact"] for nm in exact_eval[lb])
        / max(1, len(exact_eval[lb]))
        for lb in ("ctx_value", "ctx_name", "ctx_namepool"))
    print(f"\n  >>> best CONTEXTUAL exact heldout recall = {ctx_best:.3f}  "
          f"({'PASS' if ctx_best >= 0.5 else 'FAIL'} the >=0.5 sanity gate)")

    # ================= MAIN TABLE =======================================
    print("\n" + "=" * 116)
    print("ALIAS RECALL — CONTEXTUAL (semantic) vs SURFACE vs HASH vs WM-OFF "
          "(frozen v12; held-out names)")
    print("  exact rows = phase-1 (exact-only) model; alias rows = phase-2 "
          "(exact+alias) model. dRecall = recall_ON - WM-OFF.")
    print("=" * 116)
    hdr = (f"{'set':12s} | {'arm':12s} | {'addr_top1':>9s} {'addr_mass':>9s} "
           f"{'alias_t1':>8s} | {'recall_ON':>9s} {'recall_OFF':>10s} "
           f"{'dRecall':>8s} | {'first_ON':>8s} {'first_OFF':>9s}")
    print(hdr)
    print("-" * len(hdr))
    agg = {}   # (arm, kind) -> [on_sum, off_sum, n]
    for nm, kind, cache in eval_sets:
        off_e, off_f = wmoff(cache)
        for label in ARM_LABELS:
            if kind == "exact":
                r = exact_eval[label][nm]            # phase-1 snapshot
            else:
                r = eval_arm(heads[label], cache, dev)
            print(f"{nm:12s} | {label:12s} | {r['top1']:>9.3f} {r['mass']:>9.3f} "
                  f"{r['alias_top1']:>8.3f} | {r['on_exact']:>9.3f} "
                  f"{off_e:>10.3f} {r['on_exact'] - off_e:>+8.3f} | "
                  f"{r['on_first']:>8.3f} {off_f:>9.3f}")
            a = agg.setdefault((label, kind), [0.0, 0.0, 0])
            a[0] += r["on_exact"] * r["n"]
            a[1] += off_e * r["n"]
            a[2] += r["n"]
        print(f"{'  (WM-OFF)':12s} | {'--':12s} | {'--':>9s} {'--':>9s} "
              f"{'--':>8s} | {'--':>9s} {off_e:>10.3f} {'--':>8s} | "
              f"{'--':>8s} {off_f:>9.3f}")
        print("-" * len(hdr))

    print("\n  PER-KIND aggregate recall (over all N) — ctx_* are the SEMANTIC "
          "arms:")
    hh = "  " + f"{'kind':8s} |" + "".join(f" {l:>12s}" for l in ARM_LABELS) \
        + f" {'WM-OFF':>9s}"
    print(hh)
    for kind in kinds:
        row = f"  {kind:8s} |"
        off = None
        for label in ARM_LABELS:
            a = agg.get((label, kind))
            if a and a[2]:
                row += f" {a[0] / a[2]:>12.3f}"
                off = a[1] / a[2]
            else:
                row += f" {'--':>12s}"
        row += f" {'--':>9s}" if off is None else f" {off:>9.3f}"
        print(row)

    # ================= TRAIN-RECALL on ALIAS (does the head FIT alias?) ===
    print("\n  ALIAS TRAIN-recall (held-in) after phase 2 — does the head even "
          "FIT alias on what it trained on?")
    print(f"  {'arm':12s} | {'addr_top1':>9s} {'recall':>8s}")
    for label in ARM_LABELS:
        r = eval_arm(heads[label], alias_train_probe, dev)
        print(f"  {label:12s} | {r['top1']:>9.3f} {r['on_exact']:>8.3f}")

    # ================= WHITENING DIAGNOSTIC =============================
    print("\n" + "=" * 116)
    print("WHITENING DIAGNOSTIC — can ANY learned/whitened key on the FROZEN "
          "contextual hidden identify the holder?")
    print("  raw=cosine top1 (PCA kremove=0); pca-k = top1 after projecting out "
          "the top-k shared PCs; lin = LEARNED LINEAR proj top1.")
    print("  (top1 = holder-id accuracy; beats = cos(holder) > cos(surface "
          "alias-match). Compare to 1/N chance.)")
    print("=" * 116)
    # exact-train + alias-train caches to FIT the linear/PCA on the matching kind
    for label, q_anchor, k_anchor in WHITEN_ANCHORS:
        print(f"\n  anchor [{label}]  q={q_anchor}  k={k_anchor}")
        # fit PCA basis + linear addr ONCE per kind-family (exact / alias).
        for fit_name, fit_cache, eval_filter in (
                ("exact", exact_train, "exact"),
                ("alias", alias_train, "alias")):
            mean, Vh = _pca_components(fit_cache, k_anchor, dev)
            lin = fit_linear_addr(fit_cache, q_anchor, k_anchor, dev,
                                  args.whiten_steps, args.lr, args.whiten_dproj)
            for nm, kind, cache in eval_sets:
                if not kind.startswith(eval_filter):
                    continue
                raw_t1, raw_b = pca_whiten_top1(cache, q_anchor, k_anchor,
                                                mean, Vh, 0, dev)
                p4_t1, _ = pca_whiten_top1(cache, q_anchor, k_anchor,
                                           mean, Vh, 4, dev)
                p16_t1, _ = pca_whiten_top1(cache, q_anchor, k_anchor,
                                            mean, Vh, 16, dev)
                p32_t1, _ = pca_whiten_top1(cache, q_anchor, k_anchor,
                                            mean, Vh, 32, dev)
                lin_t1, lin_b = eval_linear_addr(lin, cache, q_anchor,
                                                 k_anchor, dev)
                N = cache[0]["N"] if cache else 0
                print(f"    {nm:12s} 1/N={1.0 / max(1, N):.3f} | raw={raw_t1:.3f} "
                      f"pca4={p4_t1:.3f} pca16={p16_t1:.3f} pca32={p32_t1:.3f} | "
                      f"lin={lin_t1:.3f} (beats_alias raw={raw_b:.2f} "
                      f"lin={lin_b:.2f})")

    print("\nREAD:\n"
          "  * ctx_* exact recall PASS (>=0.5) AND ctx_* >> surface/hash on "
          "alias  ⇒ semantic addressing WORKS on the frozen trunk (general "
          "recall feasible).\n"
          "  * ctx_* exact recall PASS but alias ~ chance (and alias TRAIN-recall "
          "also low, lin/pca top1 ~ 1/N on alias) ⇒ the trunk encodes NAME "
          "identity but NOT the resolved reference; bottleneck = reference "
          "propagation in the trunk, not the addresser.\n"
          "  * ctx_* exact recall FAIL even with full capacity + attention "
          "supervision (train addr_top1 also low, lin/pca exact ~ 1/N) ⇒ the "
          "frozen contextual registers are not separable by ANY learned key; "
          "strong NEGATIVE → the fix is in the TRUNK (co-train), not the key.")


if __name__ == "__main__":
    main()
