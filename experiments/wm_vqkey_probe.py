"""VQ / learned addressing key vs the deployed lexical HASH — head-to-head probe.

QUESTION (research, fair-baseline discipline — a clean negative is a valid result):
  The deployed WorkingMemory recall key is a LEXICAL HASH of the identifier text
  (model.py::_identifier_code_lexical / _fnv1a). It is perfectly separable (one-hot,
  zero cross-talk) but EXACT-SPELLING: `cache_size` and `CACHE_SIZE` hash to
  different codes, so a recall query under a surface VARIANT of the bound name
  misses entirely. Can a LEARNED / VQ addressing key (hard attention over a learned
  codebook, argmax read) MATCH the hash on separability under saturation AND
  recover the surface-variant robustness the hash structurally cannot?

THREE ADDRESSING ARMS — identical copy readout + data + budget downstream, only the
key differs (so we isolate ADDRESSING):
  A HASH : deployed-style poly/FNV hash of the exact identifier char-spelling. No
           key training. code(name)=hash(chars) -> one-hot match. Exact-spelling.
  B VQ   : a small learned char-level encoder over the NAME-SPAN token embeddings
           -> query vector q; a learned codebook C (Kc slots); code = argmax(cos(q,C)).
           The store is keyed by the discrete code (one-hot -> zero cross-talk, like
           the hash, but LEARNED). Trained with a soft (temperature-annealed) code-
           agreement surrogate so gradient flows (Gumbel/straight-through spirit),
           plus the standard VQ commitment loss; HARD argmax code-equality at eval.
  C SOFT : same learned encoder -> q; read = softmax(cos(q, q_buffer)) over the
           buffered binding keys, NO quantization (continuous). The cross-talk
           baseline that should diffuse under saturation.

DOWNSTREAM (shared, identical for all arms): the validated COPY/pointer readout
  (wm_multitok_readout.py ARM B): p_copy[j] = sum_k attn[k]*onehot(src_tok[k,j]);
  p_final = (1-g)*p_lm + g*p_copy with a learned per-position copy gate g. With one-
  hot addressing this copies the exact value digits; with diffuse addressing it
  blurs across bindings. The TRUNK (tiny DeltaNet) is FROZEN -> WM-OFF (no copy)
  reproduces the recurrence-only baseline EXACTLY (honest kill-gate control).

TWO INDEPENDENT AXES — reported SEPARATELY, never averaged:
  (1) SATURATION : multibind N in {32,64,128}, query reuses the EXACT binding
      spelling. Hash ~100%; soft should diffuse as N grows; does VQ keep ~100%
      (no separability loss)?
  (2) ROBUSTNESS : query surface form DIFFERS from the binding surface but refers
      to the same entity. Variant pairs learnable FROM FORM: case (cache_size /
      CACHE_SIZE) and snake<->camel (queue_capacity / queueCapacity). Bind under
      one form, query under the related form. Hash -> ~0; does VQ learn to map
      variants to the SAME code and recover recall? HELD-OUT words (encoder must
      learn the char-level transformation, not memorize). Also a clearly-labeled
      IN-CONTEXT-ALIAS set (`X=..; Y=X; query Y`) which is NOT learnable from form
      (needs context) -> documented honest limit; all form-based arms expected ~0.

Self-contained: tiny char-level DeltaNet trained from scratch on synthetic data,
fp32 (sparse-loss bf16 collapse, per CLAUDE.md), runs in minutes. Does NOT load any
production checkpoint. Leaves no artifacts; prints two tables + a VQ codebook
equivalence diagnostic.

Usage (pin a FREE gpu):
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/wm_vqkey_probe.py
  # quick wiring check:
  ... experiments/wm_vqkey_probe.py --smoke
"""
from __future__ import annotations

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention

# --------------------------------------------------------------------------- vocab
_SPECIAL = ["<pad>", "<bos>", "<eos>"]
_CHARS = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_ =?\n")
_VOCAB = _SPECIAL + _CHARS
_CH2ID = {c: i for i, c in enumerate(_VOCAB)}
PAD, BOS, EOS = 0, 1, 2
VOCAB = len(_VOCAB)
_DIGIT_IDS = [_CH2ID[c] for c in "0123456789"]


def _enc_chars(s: str) -> list[int]:
    return [_CH2ID[c] for c in s]


# --------------------------------------------------------------------------- data
class WordBank:
    """Pronounceable random word generator with a disjoint train/eval split so no
    ENTITY ever leaks across the split (held-out robustness must be learned from the
    char-level FORM transformation, not memorized word->code)."""

    _CONS = "bcdfghjklmnpqrstvwxz"
    _VOW = "aeiou"

    def __init__(self, seed: int, forbidden: set[str] | None = None):
        self.rng = random.Random(seed)
        self._seen: set[str] = set()
        # words in `forbidden` (pass the train bank's LIVE _seen set) are ALSO
        # rejected → guarantees cross-disjoint train/eval banks: no word ever
        # leaks across the split, so held-out robustness must be learned from the
        # char-level FORM transform, not memorized word→code (review fix).
        self._forbidden = forbidden if forbidden is not None else set()

    def _word(self) -> str:
        n = self.rng.randint(3, 4)  # syllables — large space (~10^6) so train and
        w = "".join(self.rng.choice(self._CONS) + self.rng.choice(self._VOW)
                    for _ in range(n))          # eval can be fully disjoint
        if self.rng.random() < 0.3:  # occasional trailing consonant
            w += self.rng.choice(self._CONS)
        return w

    def fresh_word(self) -> str:
        for _ in range(200):
            w = self._word()
            if w not in self._seen and w not in self._forbidden:
                self._seen.add(w)
                return w
        # fallback: append letters until unique across BOTH banks
        while True:
            w = self._word() + self.rng.choice(self._CONS)
            if w not in self._seen and w not in self._forbidden:
                self._seen.add(w)
                return w


def _entity_parts(bank: WordBank, rng: random.Random, min_parts: int = 1
                  ) -> list[str]:
    """An entity is 1-2 word-parts -> supports snake/camel/case forms.

    `min_parts=2` forces a genuinely multi-token name so snake/camel/upper forms
    actually DIFFER in spelling (a 1-word entity has snake==camel, which would
    trivially pass the hash and corrupt the robustness measurement)."""
    k = 2 if (min_parts >= 2 or rng.random() < 0.5) else 1
    return [bank.fresh_word() for _ in range(k)]


def _form(parts: list[str], style: str) -> str:
    if style == "snake":
        return "_".join(parts)
    if style == "upper":
        return "_".join(parts).upper()
    if style == "camel":
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    if style == "pascal":
        return "".join(p.capitalize() for p in parts)
    raise ValueError(style)


# variant axes: bind-form -> query-form (both refer to the SAME entity)
_VARIANT_PAIRS = {
    "case":  ("snake", "upper"),   # cache_size  -> CACHE_SIZE
    "camel": ("snake", "camel"),   # queue_capacity -> queueCapacity
}


def _val_str(rng: random.Random) -> str:
    return str(rng.randint(1000, 9999))  # always 4 digits -> vlen=4


def make_example(bank: WordBank, rng: random.Random, n_vars: int, kind: str):
    """Build ONE example. kind in {exact, case, camel, alias}.

    Returns dict with token ids `full` and oracle metadata: per-binding name char
    span + value start, the query name span, the answer position + answer digit
    ids, and qb (the correct binding index). NO peeking at qb downstream — it is
    only used to score / supervise; the arms address purely by the name key."""
    # distinct entities (each its own parts); pick the queried one
    ents = [_entity_parts(bank, rng) for _ in range(n_vars)]
    vals = [_val_str(rng) for _ in range(n_vars)]
    qb = rng.randrange(n_vars)

    if kind == "exact":
        bind_styles = [rng.choice(["snake", "camel"]) for _ in range(n_vars)]
        bind_names = [_form(ents[i], bind_styles[i]) for i in range(n_vars)]
        query_name = bind_names[qb]                 # EXACT spelling reuse
        answer = vals[qb]
        alias_lines = []
    elif kind in ("case", "camel"):
        bstyle, qstyle = _VARIANT_PAIRS[kind]
        ents[qb] = _entity_parts(bank, rng, min_parts=2)  # ensure forms DIFFER
        bind_names = [_form(ents[i], rng.choice(["snake", "camel"]))
                      for i in range(n_vars)]
        bind_names[qb] = _form(ents[qb], bstyle)    # the queried binding's form
        query_name = _form(ents[qb], qstyle)        # related VARIANT form
        answer = vals[qb]
        alias_lines = []
    elif kind == "alias":
        # X = NNNN ; Y = X ; query Y -> answer is X's value (needs CONTEXT, not form)
        bind_names = [_form(ents[i], rng.choice(["snake", "camel"]))
                      for i in range(n_vars)]
        xname = bind_names[qb]
        yname = _form(_entity_parts(bank, rng), "snake")
        query_name = yname
        answer = vals[qb]
        alias_lines = [(yname, xname)]              # appended after bindings
    else:
        raise ValueError(kind)

    # ---- assemble tokens + spans -----------------------------------------
    toks: list[int] = [BOS]
    bindings = []
    order = list(range(n_vars))
    rng.shuffle(order)
    pos_of_qb = None
    for slot, i in enumerate(order):
        name = bind_names[i]
        ns = len(toks)
        toks += _enc_chars(name)
        ne = len(toks)                              # exclusive name end
        toks += _enc_chars(" = ")
        vs = len(toks)
        toks += _enc_chars(vals[i])
        toks += _enc_chars("\n")
        bindings.append(dict(name_ids=_enc_chars(name), val_start=vs))
        if i == qb:
            pos_of_qb = slot
    # alias lines: `Y = X\n` (value span is the alias-target NAME, not digits)
    for (yn, xn) in alias_lines:
        toks += _enc_chars(yn + " = " + xn + "\n")
    # query line: `?<qname> = <answer><eos>`
    toks += _enc_chars("?")
    qns = len(toks)
    toks += _enc_chars(query_name)
    qne = len(toks)                                 # exclusive query name end
    toks += _enc_chars(" = ")
    ans_pos = len(toks)
    toks += _enc_chars(answer)
    ans_end = len(toks)
    toks += [EOS]

    vlen = ans_end - ans_pos
    val_src = [b["val_start"] for b in bindings]
    return dict(
        full=toks,
        bind_name_ids=[b["name_ids"] for b in bindings],
        val_start=val_src,                          # source positions (len N)
        qname_ids=_enc_chars(query_name),
        ans_pos=ans_pos, vlen=vlen,
        ans_toks=toks[ans_pos:ans_end],
        qb=pos_of_qb,                               # index into the SHUFFLED bindings
        n_vars=n_vars, kind=kind,
    )


# --------------------------------------------------------------------------- trunk
class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_ff):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = DeltaNetAttention(d_model, n_heads, d_head)
        self.n2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class TinyTrunk(nn.Module):
    """Char-level DeltaNet LM. `forward(ids, return_hidden=True)` returns
    (logits, h_post) where h_post = out_norm(h) is exactly the tensor the copy
    gate / lm_head consume (mirrors the production WM pre-mem hidden)."""

    def __init__(self, d_model=256, n_heads=4, d_head=64, n_layers=3, d_ff=512):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_head, d_ff) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, VOCAB, bias=False)
        self.d_model = d_model

    def forward(self, ids, return_hidden=False):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        h = self.out_norm(x)
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        return logits


def pretrain_trunk(trunk, bank, rng, device, *, steps, batch, n_choices, kinds,
                   lr):
    """Next-token LM over the synthetic mix so p_lm / h_post are calibrated and
    the recurrence-only baseline is honest. Frozen afterwards."""
    opt = torch.optim.AdamW(trunk.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    trunk.train()
    for step in range(1, steps + 1):
        exs = [make_example(bank, rng, rng.choice(n_choices), rng.choice(kinds))
               for _ in range(batch)]
        T = max(len(e["full"]) for e in exs)
        ids = torch.full((batch, T), PAD, dtype=torch.long, device=device)
        for b, e in enumerate(exs):
            ids[b, :len(e["full"])] = torch.tensor(e["full"], device=device)
        logits = trunk(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                               ids[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  [trunk] step {step:4d} ce={float(loss):.4f}")
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)


# --------------------------------------------------------------------------- cache
@torch.no_grad()
def build_cache(trunk, examples, device, tag):
    """One frozen forward per example -> the slices the heads need (kept on CPU).
    h_ans = h_post predicting each answer digit; lm_logits there for WM-OFF/p_lm."""
    cache = []
    for e in examples:
        ids = torch.tensor([e["full"]], dtype=torch.long, device=device)
        logits, h = trunk(ids, return_hidden=True)
        ap, vlen = e["ans_pos"], e["vlen"]
        h_ans = h[0, ap - 1:ap - 1 + vlen].cpu()                 # (vlen, d)
        lm_logits = logits[0, ap - 1:ap - 1 + vlen].cpu()        # (vlen, vocab)
        # value source token ids per binding (first vlen digits of its value span)
        full = e["full"]
        src = torch.tensor([full[vs:vs + vlen] for vs in e["val_start"]],
                           dtype=torch.long)                      # (N, vlen)
        cache.append(dict(
            bind_name_ids=e["bind_name_ids"], qname_ids=e["qname_ids"],
            src=src, h_ans=h_ans, lm_logits=lm_logits,
            ans_toks=torch.tensor(e["ans_toks"], dtype=torch.long),
            qb=e["qb"], N=e["n_vars"], vlen=vlen, kind=e["kind"]))
    print(f"  [cache:{tag}] {len(cache)} examples")
    return cache


# --------------------------------------------------------------------------- keys
def _poly_hash(ids: list[int], K: int) -> int:
    """Deterministic salt-free polynomial hash of the EXACT char-id spelling.
    Mirrors model.py::_poly_str_hash (affine fold); case/separator sensitive."""
    h = 0
    for x in ids:
        h = (h * 257 + x + 1) % 2147483647
    return h % K


def _pad_names(name_id_lists, device):
    """list of variable-len id-lists -> (M, Lmax) long + (M, Lmax) bool mask."""
    M = len(name_id_lists)
    L = max(len(x) for x in name_id_lists)
    ids = torch.full((M, L), PAD, dtype=torch.long, device=device)
    mask = torch.zeros((M, L), dtype=torch.bool, device=device)
    for i, x in enumerate(name_id_lists):
        ids[i, :len(x)] = torch.tensor(x, device=device)
        mask[i, :len(x)] = True
    return ids, mask


class NameEncoder(nn.Module):
    """Char-level name encoder: own embedding -> masked mean-pool -> MLP -> q.
    Order-free pooling; case/camel equivalence is learnable because it can map
    'C'~'c' and de-weight '_' at the CHAR level (generalizes to held-out words)."""

    def __init__(self, d_enc=128, d_key=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_enc, padding_idx=PAD)
        self.mlp = nn.Sequential(nn.Linear(d_enc, 256), nn.GELU(),
                                 nn.Linear(256, d_key))

    def forward(self, ids, mask):
        e = self.emb(ids)                                        # (M, L, d_enc)
        m = mask.unsqueeze(-1).float()
        pooled = (e * m).sum(1) / m.sum(1).clamp_min(1.0)        # (M, d_enc)
        return self.mlp(pooled)                                  # (M, d_key)


class AddressingHead(nn.Module):
    """Unified head: addressing arm in {hash, vq, soft} + shared copy gate.

    `attn(cache, hard)` -> (attn over the N bindings [N], aux_vq_loss). hard=False
    uses the differentiable training surrogate (soft code-agreement for vq, softmax
    for soft); hard=True is the deployed read (argmax code-equality for hash/vq,
    sharp softmax for soft).
    """

    def __init__(self, mode, d_model, d_key=64, codebook=1024):
        super().__init__()
        self.mode = mode
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        self.codebook_K = codebook
        if mode in ("vq", "soft"):
            self.enc = NameEncoder(d_key=d_key)
        if mode == "vq":
            C = torch.randn(codebook, d_key) / math.sqrt(d_key)
            self.C = nn.Parameter(C)
        if mode == "soft":
            self.log_tau = nn.Parameter(torch.tensor(math.log(20.0)))
        self.d_key = d_key

    # -- per-arm code / query vectors -------------------------------------
    def _codes_hard(self, cache, device):
        """HARD discrete codes for hash/vq: (bind_codes [N], query_code int)."""
        if self.mode == "hash":
            bc = [_poly_hash(x, self.codebook_K) for x in cache["bind_name_ids"]]
            qc = _poly_hash(cache["qname_ids"], self.codebook_K)
            return torch.tensor(bc, device=device), qc
        # vq
        bids, bmask = _pad_names(cache["bind_name_ids"], device)
        qids, qmask = _pad_names([cache["qname_ids"]], device)
        qb_vec = self.enc(bids, bmask)                           # (N, d_key)
        qq_vec = self.enc(qids, qmask)                           # (1, d_key)
        Cn = F.normalize(self.C, dim=-1)
        bc = (F.normalize(qb_vec, dim=-1) @ Cn.t()).argmax(-1)   # (N,)
        qc = int((F.normalize(qq_vec, dim=-1) @ Cn.t()).argmax(-1)[0])
        return bc, qc

    def attn(self, cache, device, hard, vq_temp=20.0):
        aux = torch.zeros((), device=device)
        N = cache["N"]
        if self.mode == "hash":
            bc, qc = self._codes_hard(cache, device)
            match = (bc == qc).float()
            attn = match / match.sum().clamp_min(1e-9) if match.sum() > 0 \
                else match
            return attn, aux

        bids, bmask = _pad_names(cache["bind_name_ids"], device)
        qids, qmask = _pad_names([cache["qname_ids"]], device)
        qb_vec = self.enc(bids, bmask)                           # (N, d_key)
        qq_vec = self.enc(qids, qmask)[0]                        # (d_key,)

        if self.mode == "soft":
            tau = self.log_tau.exp().clamp(2.0, 100.0)
            sim = F.cosine_similarity(qq_vec.unsqueeze(0), qb_vec, dim=-1)  # (N,)
            attn = torch.softmax(sim * tau, dim=-1)
            return attn, aux

        # ---- vq ----------------------------------------------------------
        Cn = F.normalize(self.C, dim=-1)
        qn_b = F.normalize(qb_vec, dim=-1)
        qn_q = F.normalize(qq_vec, dim=-1)
        # standard VQ commitment loss (raw vectors, nearest code)
        with torch.no_grad():
            bc = (qn_b @ Cn.t()).argmax(-1)
            qc = (qn_q @ Cn.t()).argmax(-1)
        zb = self.C[bc]
        zq = self.C[qc]
        aux = (F.mse_loss(qb_vec, zb.detach()) + F.mse_loss(qq_vec, zq.detach())
               + 0.25 * (F.mse_loss(qb_vec.detach(), zb)
                         + F.mse_loss(qq_vec.detach(), zq)))
        if hard:
            match = (bc == qc).float()
            attn = match / match.sum().clamp_min(1e-9) if match.sum() > 0 \
                else match
            return attn, aux
        # soft, temperature-annealed code-agreement (differentiable surrogate)
        a_b = torch.softmax((qn_b @ Cn.t()) * vq_temp, dim=-1)   # (N, K)
        a_q = torch.softmax((qn_q @ Cn.t()) * vq_temp, dim=-1)   # (K,)
        psame = a_b @ a_q                                        # (N,) P[same code]
        attn = psame / psame.sum().clamp_min(1e-9)
        return attn, aux

    def recall_logp(self, cache, attn, device):
        """Copy-mixed log p over the answer span: (vlen, vocab)."""
        h_ans = cache["h_ans"].to(device)                        # (vlen, d)
        lm_logits = cache["lm_logits"].to(device)               # (vlen, vocab)
        src = cache["src"].to(device)                            # (N, vlen)
        vlen = cache["vlen"]
        p_lm = torch.softmax(lm_logits, dim=-1)                  # (vlen, vocab)
        p_copy = lm_logits.new_zeros(vlen, VOCAB)
        for j in range(vlen):
            p_copy[j].scatter_add_(0, src[:, j], attn)          # (vocab,)
        copy_mass = attn.sum()
        g = torch.sigmoid(self.gate(h_ans)).squeeze(-1)         # (vlen,)
        g = g * (copy_mass > 0).float()                         # no copy -> p_lm
        p = (1 - g).unsqueeze(-1) * p_lm + g.unsqueeze(-1) * p_copy
        return torch.log(p + 1e-9)


def train_arm(head, train_cache, order, device, *, vq_weight, vq_temp_sched):
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    head.train()
    nsteps = len(order)
    for step, idx in enumerate(order, 1):
        c = train_cache[idx]
        temp = vq_temp_sched(step / nsteps)
        attn, aux = head.attn(c, device, hard=False, vq_temp=temp)
        logp = head.recall_logp(c, attn, device)
        ans = c["ans_toks"].to(device)
        ce = F.nll_loss(logp, ans)
        loss = ce + vq_weight * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 300 == 0 or step == 1:
            print(f"    [{head.mode}] step {step:4d} ce={float(ce):.4f} "
                  f"vq={float(aux):.4f}")
    head.eval()


@torch.no_grad()
def eval_arm(head, cache, device):
    """Kill-gate: addressing (top1 / mass) + WM-ON vs WM-OFF EXACT recall."""
    top1 = 0
    mass = 0.0
    on_e = off_e = on_f = off_f = 0
    used = len(cache)
    for c in cache:
        attn, _ = head.attn(c, device, hard=True)
        qb = c["qb"]
        if attn.sum() > 0:
            top1 += int(attn.argmax().item() == qb)
        mass += float(attn[qb])
        logp = head.recall_logp(c, attn, device)
        ans = c["ans_toks"].to(device)
        off_logits = c["lm_logits"].to(device)
        on_pred = logp.argmax(-1)
        off_pred = off_logits.argmax(-1)
        on_e += int(bool((on_pred == ans).all()))
        off_e += int(bool((off_pred == ans).all()))
        on_f += int(on_pred[0].item() == ans[0].item())
        off_f += int(off_pred[0].item() == ans[0].item())
    nz = max(1, used)
    return dict(n=used, top1=top1 / nz, mass=mass / nz,
                on_exact=on_e / nz, off_exact=off_e / nz,
                on_first=on_f / nz, off_first=off_f / nz)


@torch.no_grad()
def vq_codebook_diag(head, robustness_caches, saturation_caches, device):
    """Did the VQ codebook LEARN the equivalences (vs memorize)?
      - variant same-code rate: code(bind form) == code(query form) of same entity
      - distinct-entity collision rate (separability) on a saturation set
      - codebook utilization (unique codes used / Kc)"""
    out = {}
    # variant pair equivalence (per robustness set)
    for nm, cache in robustness_caches:
        same = 0
        for c in cache:
            bc, qc = head._codes_hard(c, device)
            same += int(bc[c["qb"]].item() == qc)
        out[f"same_code[{nm}]"] = same / max(1, len(cache))
    # separability + utilization on a saturation set (distinct entities)
    all_codes = []
    coll_num = coll_den = 0
    for c in saturation_caches:
        bids, bmask = _pad_names(c["bind_name_ids"], device)
        Cn = F.normalize(head.C, dim=-1)
        codes = (F.normalize(head.enc(bids, bmask), dim=-1) @ Cn.t()).argmax(-1)
        all_codes.append(codes)
        u = len(set(codes.tolist()))
        coll_num += (len(codes) - u)
        coll_den += len(codes)
    out["distinct_collision_rate"] = coll_num / max(1, coll_den)
    flat = torch.cat(all_codes) if all_codes else torch.zeros(0)
    out["codebook_util"] = len(set(flat.tolist())) / head.codebook_K
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--codebook", type=int, default=1024)
    ap.add_argument("--trunk_steps", type=int, default=1500)
    ap.add_argument("--trunk_batch", type=int, default=4)
    ap.add_argument("--head_steps", type=int, default=1500)
    ap.add_argument("--n_train", type=int, default=1400)
    ap.add_argument("--n_eval", type=int, default=150)
    ap.add_argument("--vq_weight", type=float, default=0.25)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.trunk_steps = 60
        args.head_steps = 120
        args.n_train = 80
        args.n_eval = 30
        sat_N = [16, 32]
        rob_N = [16, 32]
    else:
        sat_N = [32, 64, 128]
        rob_N = [32, 128]

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"[cfg] device={dev} d_model={args.d_model} n_layers={args.n_layers} "
          f"codebook={args.codebook} trunk_steps={args.trunk_steps} "
          f"head_steps={args.head_steps} n_train={args.n_train} n_eval={args.n_eval}")

    # disjoint word banks: train vs eval (no entity leakage)
    train_bank = WordBank(seed=args.seed + 1)
    # eval bank rejects any word the train bank ever generated (live reference):
    # by the time eval words are drawn, train_bank._seen is fully populated, so
    # the split is genuinely entity-disjoint (no robustness leakage).
    eval_bank = WordBank(seed=args.seed + 9999, forbidden=train_bank._seen)
    train_rng = random.Random(args.seed + 2)
    eval_rng = random.Random(args.seed + 3)

    # ---- trunk ----------------------------------------------------------
    trunk = TinyTrunk(d_model=args.d_model, n_layers=args.n_layers).to(dev)
    pretrain_kinds = ["exact", "exact", "case", "camel", "alias"]
    pretrain_N = sat_N + [8, 16, 24, 48, 96] if not args.smoke else [8, 16, 32]
    print("[pretrain trunk]")
    pretrain_trunk(trunk, train_bank, train_rng, dev, steps=args.trunk_steps,
                   batch=args.trunk_batch, n_choices=pretrain_N,
                   kinds=pretrain_kinds, lr=3e-4)

    # ---- training pool (mixed) -----------------------------------------
    print("[gen+cache train]")
    train_pool_N = pretrain_N
    train_ex = [make_example(train_bank, train_rng,
                             train_rng.choice(train_pool_N),
                             train_rng.choice(pretrain_kinds))
                for _ in range(args.n_train)]
    train_cache = build_cache(trunk, train_ex, dev, "train")

    # ---- held-out eval sets --------------------------------------------
    sat_caches = []
    for N in sat_N:
        exs = [make_example(eval_bank, eval_rng, N, "exact")
               for _ in range(args.n_eval)]
        sat_caches.append((f"N={N}", build_cache(trunk, exs, dev, f"sat_N{N}")))

    rob_caches = []           # (label, cache) for case/camel across rob_N
    for kind in ("case", "camel"):
        for N in rob_N:
            exs = [make_example(eval_bank, eval_rng, N, kind)
                   for _ in range(args.n_eval)]
            rob_caches.append((f"{kind}_N={N}",
                               build_cache(trunk, exs, dev, f"{kind}_N{N}")))
    alias_caches = []
    for N in rob_N:
        exs = [make_example(eval_bank, eval_rng, N, "alias")
               for _ in range(args.n_eval)]
        alias_caches.append((f"alias_N={N}",
                             build_cache(trunk, exs, dev, f"alias_N{N}")))

    # shared training order (identical data + budget for every arm)
    ord_rng = random.Random(args.seed + 7)
    order = [ord_rng.randrange(len(train_cache)) for _ in range(args.head_steps)]

    def vq_temp_sched(frac):    # anneal sharpness 5 -> 40 (toward hard argmax)
        return 5.0 + 35.0 * frac

    # ---- train the three arms ------------------------------------------
    heads = {}
    for mode in ("hash", "vq", "soft"):
        print(f"\n[train arm: {mode}]")
        head = AddressingHead(mode, args.d_model, codebook=args.codebook).to(dev)
        train_arm(head, train_cache, order, dev,
                  vq_weight=args.vq_weight, vq_temp_sched=vq_temp_sched)
        heads[mode] = head

    # ---- WM-OFF control (arm-independent) ------------------------------
    def wmoff(cache):
        e = f = 0
        for c in cache:
            off = c["lm_logits"].argmax(-1)
            ans = c["ans_toks"]
            e += int(bool((off == ans).all()))
            f += int(off[0].item() == ans[0].item())
        n = max(1, len(cache))
        return e / n, f / n

    # ===================================================================
    def print_table(title, sets):
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)
        hdr = (f"{'set':12s} | {'arm':5s} | {'addr_top1':>9s} {'addr_mass':>9s} | "
               f"{'recall_ON':>9s} {'recall_OFF':>10s} {'dRecall':>8s} | {'first_ON':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for nm, cache in sets:
            off_e, off_f = wmoff(cache)
            for mode in ("hash", "vq", "soft"):
                r = eval_arm(heads[mode], cache, dev)
                print(f"{nm:12s} | {mode:5s} | {r['top1']:>9.3f} {r['mass']:>9.3f} | "
                      f"{r['on_exact']:>9.3f} {off_e:>10.3f} "
                      f"{r['on_exact']-off_e:>+8.3f} | {r['on_first']:>8.3f}")
            print(f"{'  (WM-OFF)':12s} | {'--':5s} | {'--':>9s} {'--':>9s} | "
                  f"{'--':>9s} {off_e:>10.3f} {'--':>8s} | {off_f:>8.3f}")
            print("-" * len(hdr))

    print_table("AXIS 1 — SATURATION (exact-spelling query; query reuses the "
                "bound name). Hash should hold ~1.0; does VQ match it; soft "
                "diffuse?", sat_caches)
    print_table("AXIS 2 — ROBUSTNESS (query is a surface VARIANT of the bound "
                "name; HELD-OUT words). Hash should ~0; does VQ recover recall?",
                rob_caches)
    print_table("AXIS 2b — IN-CONTEXT ALIAS (`Y = X`; query Y). NOT learnable "
                "from form (needs context) -> documented limit; ALL form-based "
                "arms expected ~0.", alias_caches)

    # ---- VQ codebook-equivalence diagnostic ----------------------------
    print("\n" + "=" * 92)
    print("VQ CODEBOOK DIAGNOSTIC — did the learned key map variants to the SAME "
          "code (learn) vs memorize?")
    print("=" * 92)
    diag = vq_codebook_diag(heads["vq"], rob_caches + alias_caches,
                            sat_caches[-1][1], dev)
    for k, v in diag.items():
        print(f"  {k:28s} = {v:.3f}")
    print("\n  Reading: same_code[case/camel] HIGH = VQ learned form-invariance "
          "(variants -> same code).\n  same_code[alias] expected LOW (form gives "
          "no signal). distinct_collision_rate LOW = separability kept.\n  "
          "codebook_util = fraction of the codebook actually used.")


if __name__ == "__main__":
    main()
