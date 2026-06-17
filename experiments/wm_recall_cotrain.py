"""Freeze-trunk WM-only co-train: make the WorkingMemory read CONTENT-ADDRESSABLE
and LOAD-BEARING for recall on the v12 base.

THE DIAGNOSIS (established, MEMORY.md):
  On the v12 base, WM is inert for recall because the WM read gets NO gradient
  from the recall objective. WM injects only at think-token positions (targets
  -100, state-readonly, injection feeds only lm_head), while the recall target
  ("...Answer: N") is an EMIT position whose gradient flows through the
  recurrence — never through W_proj(read). Result: read_alpha decayed 1.0 -> 0.07;
  the read is RECENCY (addressing probe: WRITE 100% binding-in-buffer, but READ
  ~0% mass on the queried binding). The DeltaNet recurrence saturates at N>=32,
  so there IS a regime WM could fill.

THE FIX (this script):
  Drive WM reads at the ANSWER-SPAN EMIT positions (via the existing
  `mem_read_mask` kwarg) so the answer-token CE backprops into W_proj(read) /
  W_q / W_k / W_v. Freeze the trunk; train ONLY the WM params. Re-bootstrap
  read_alpha (now ~0.07) to 1.0 so the read has gradient while addressing locks
  in. Optional content-recall aux pushes W_proj(read_q) toward the queried
  value's unembedding row (lm_head is UNTIED on v12, so the unembedding row — not
  embed — is the direction that raises the value logit).

EFFICIENCY:
  The frozen trunk is run ONCE per batch under torch.no_grad() to obtain the
  pre-memory hidden h_pre = out_norm(h_raw) (via return_hidden +
  `_latent_feedback_premem`). The WM module + lm_head are then run WITH grad on
  h_pre.detach(). This is bit-exact to TinyLM._finalize's post-memory logits
  (memory is called on exactly that h) but backprops through the tiny WM module
  only — no activation checkpointing, no wasted backward through 10 frozen
  layers.

VALIDATION (fair BEFORE vs AFTER, same probes the diagnosis used):
  * addressing: read mass-on-binding + attn-weighted-mean-pos/T (recency=1.0),
    at the think slot (diagnosis-matched) AND at the recall-query position
    (training-matched).
  * teacher-forced recall kill-gate at N=32/48: P(value)+argmax-accuracy of the
    value token, WM-ON (read injected at the query pos) vs WM-OFF (no injection).
  * read_alpha after training.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/wm_recall_cotrain.py \
      --steps 400 --batch 4 --lr 1e-3

  # validate-only on any ckpt (BEFORE numbers):
  ... --mode validate --ckpt checkpoints/pretrain_v12_step9537_tok2500067328.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers
def find_sub(hay: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return -1


def find_all(hay: list[int], needle: list[int]) -> list[int]:
    out = []
    if not needle:
        return out
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            out.append(i)
    return out


def build_examples(path, tok, *, n_vars=None, max_len=2048, limit=None,
                   skip_first=0, family=None):
    """Tokenize records into the training_matched format.

    Returns list of dicts: prompt_ids, comp_ids, full_ids, plen, bpos (binding,
    first occ in full), vpos (first completion occ of value), value_tok, n_vars.
    `family` (str|set): keep only records whose `family` field matches (used to
    select e.g. the `const` rows out of the mixed code_recall corpus).
    """
    fam_set = ({family} if isinstance(family, str) else
               set(family) if family is not None else None)
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs = recs[skip_first:]
    out = []
    for r in recs:
        if n_vars is not None and r.get("n_vars") not in n_vars:
            continue
        if fam_set is not None and r.get("family") not in fam_set:
            continue
        prompt = r["problem_prompt"] + "\n\n"
        comp = r["qwen_completion"]
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        comp_ids = tok.encode(comp, add_special_tokens=False)
        full = prompt_ids + comp_ids
        if len(full) > max_len:
            continue
        ans_ids = tok.encode(str(r["answer"]), add_special_tokens=False)
        plen = len(prompt_ids)
        bpos = find_sub(full, ans_ids)            # first occ overall (binding)
        comp_occ = [o for o in find_all(full, ans_ids) if o >= plen]
        if bpos < 0 or not comp_occ:
            continue
        vpos = comp_occ[0]                          # first completion occ
        if vpos - 1 < plen - 1:
            continue
        out.append(dict(
            full=full, plen=plen, bpos=bpos, blen=len(ans_ids),
            vpos=vpos, value_tok=ans_ids[0], ans_ids=ans_ids,
            n_vars=r.get("n_vars"),
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def build_general_examples(path, tok, *, limit=2000, max_len=512, min_len=48,
                           skip_first=0):
    """GENERAL-code NEGATIVES (no recall query): tokenize the `extracted_code`
    solution of distill records. The gate must learn NOT to copy here — these
    test selectivity (copy_g≈0) and no-harm (general CE WM-ON==WM-OFF)."""
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip_first:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = r.get("extracted_code") or ""
            if not code.strip():
                continue
            ids = tok.encode(code, add_special_tokens=False)
            if len(ids) < min_len:
                continue
            out.append(dict(full=ids[:max_len]))
            if len(out) >= limit:
                break
    return out


# ----------------------------------------------------------------------- probes
@torch.no_grad()
def selectivity_probe(model, recall_examples, general_examples, *, n, device,
                      gen_stride=8):
    """copy_g (the learned copy gate σ) at RECALL-ANSWER positions vs GENERAL-CODE
    positions. Selective ⇔ high on answers, ≈0 on general code (so it does not
    corrupt ordinary generation)."""
    mem = model.memory
    if not bool(getattr(model, "use_copy_head", False)):
        return None
    g_ans, g_gen = [], []
    if True:
        for ex in recall_examples[:n]:
            full = ex["full"]
            v0, vlen = ex["vpos"], len(ex["ans_ids"])
            seq = torch.tensor([full[:v0 + vlen]], dtype=torch.long, device=device)
            h = trunk_hpre(model, seq)
            mask = torch.zeros_like(seq, dtype=torch.float)
            mask[0, max(0, v0 - 1):v0 + vlen - 1] = 1.0
            inj = _mem_forward(model, h, seq, read_mask=mask)
            model._apply_copy_head(model.lm_head(inj), inj, seq, mask)
            if getattr(model, "_last_copy_gate", None) is not None:
                g_ans.append(float(model._last_copy_gate.mean()))
        for ex in general_examples[:n]:
            full = ex["full"]
            L = len(full)
            if L < 4:
                continue
            seq = torch.tensor([full], dtype=torch.long, device=device)
            h = trunk_hpre(model, seq)
            mask = torch.zeros_like(seq, dtype=torch.float)
            mask[0, 2:L - 1:gen_stride] = 1.0          # isolated → copy offset 0
            if float(mask.sum()) == 0:
                continue
            inj = _mem_forward(model, h, seq, read_mask=mask)
            model._apply_copy_head(model.lm_head(inj), inj, seq, mask)
            if getattr(model, "_last_copy_gate", None) is not None:
                g_gen.append(float(model._last_copy_gate.mean()))
    return dict(
        g_answer=(sum(g_ans) / max(1, len(g_ans))),
        g_general=(sum(g_gen) / max(1, len(g_gen))),
        n_ans=len(g_ans), n_gen=len(g_gen),
    )


@torch.no_grad()
def general_ce_killgate(model, general_examples, *, n, device, gen_stride=8):
    """CE on GENERAL code (decomposed no-harm), at isolated sampled positions:
      OFF       = plain trunk lm_head (the v12 base) — the reference.
      COPY-ONLY = lm_head(h) + the GATED copy mix (clean h, no W_proj inject).
                  Isolates the LEARNED COPY GATE's harm — the headline no-harm
                  number; ≈OFF iff the gate stays closed on general code.
      FULL      = lm_head(h + read_alpha·W_proj(read)) + gated copy. Includes the
                  always-on read INJECTION (read_alpha) on top of the gate.
    Target: COPY-ONLY Δ ≈ 0."""
    use_copy = bool(getattr(model, "use_copy_head", False))
    ce_full = ce_copy = ce_off = 0.0
    ntok = 0
    for ex in general_examples[:n]:
        full = ex["full"]
        L = len(full)
        if L < 4:
            continue
        seq = torch.tensor([full], dtype=torch.long, device=device)
        h = trunk_hpre(model, seq)
        pos = list(range(2, L - 1, gen_stride))            # isolated → offset 0
        if not pos:
            continue
        mask = torch.zeros_like(seq, dtype=torch.float)
        mask[0, pos] = 1.0
        inj = _mem_forward(model, h, seq, read_mask=mask)  # stashes read attn
        off_logits = model.lm_head(h)
        full_logits = model.lm_head(inj)
        copy_logits = off_logits
        if use_copy:
            full_logits = model._apply_copy_head(full_logits, inj, seq, mask)
            copy_logits = model._apply_copy_head(model.lm_head(h), h, seq, mask)
        tgt = seq[0, [p + 1 for p in pos]]
        ce_full += float(F.cross_entropy(full_logits[0, pos], tgt, reduction="sum"))
        ce_copy += float(F.cross_entropy(copy_logits[0, pos], tgt, reduction="sum"))
        ce_off += float(F.cross_entropy(off_logits[0, pos], tgt, reduction="sum"))
        ntok += len(pos)
    ntok = max(1, ntok)
    return dict(ce_full=ce_full / ntok, ce_copyonly=ce_copy / ntok,
                ce_off=ce_off / ntok,
                delta_copyonly=(ce_copy - ce_off) / ntok,
                delta_full=(ce_full - ce_off) / ntok, n_tok=ntok)


def load_model(ckpt, device, *, mem_always_read=False, force_wm=False,
               mem_key_from_embedding=False, mem_key_window=4,
               use_copy_head=False, mem_discrete_key=False, force_mem_size=None,
               mem_discrete_key_lexical=True, copy_require_match=True,
               match_window=32, mem_soft_namekey=False, soft_namekey_dim=64,
               soft_namekey_match_threshold=0.5, mem_ctx_namekey=False,
               ctx_namekey_dim=192, ctx_namekey_match_threshold=0.5):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    # FORCE-ATTACH the validated v14 WM-recall mechanism onto the base. None ->
    # leave whatever the cfg/state-dict say (back-compat); explicit values force
    # them on (embedding-key + discrete-key have no new params; copy head +
    # always-read attach fresh/zero-init so the base still loads via strict=False).
    # discrete-key needs the buffer to span the whole sequence → bump mem_size.
    model, cfg = build_model_from_ckpt(
        ckpt,
        force_mem_key_from_embedding=(True if mem_key_from_embedding else None),
        force_mem_key_window=(int(mem_key_window) if mem_key_from_embedding else None),
        force_use_copy_head=(True if use_copy_head else None),
        force_mem_always_read=(True if mem_always_read else None),
        # soft-namekey / ctx-namekey / discrete-key are mutually exclusive (the
        # WorkingMemory ctor asserts it) — force discrete OFF when soft/ctx is
        # requested so a base whose cfg carries discrete_key=True doesn't trip it.
        force_mem_discrete_key=(False if (mem_soft_namekey or mem_ctx_namekey)
                                else (True if mem_discrete_key else None)),
        force_mem_discrete_key_lexical=(bool(mem_discrete_key_lexical)
                                        if mem_discrete_key else None),
        # match-existence gate + locality window are SHARED by the discrete,
        # soft-namekey, and ctx-namekey paths (copy head reads `_last_match_exists`).
        force_mem_copy_require_match=(bool(copy_require_match)
                                      if (mem_discrete_key or mem_soft_namekey
                                          or mem_ctx_namekey) else None),
        force_mem_discrete_key_match_window=(int(match_window)
                                             if (mem_discrete_key
                                                 or mem_soft_namekey
                                                 or mem_ctx_namekey) else None),
        force_mem_soft_namekey=(True if mem_soft_namekey else
                                (False if mem_ctx_namekey else None)),
        force_mem_soft_namekey_dim=(int(soft_namekey_dim)
                                    if mem_soft_namekey else None),
        force_mem_soft_namekey_match_threshold=(
            float(soft_namekey_match_threshold) if mem_soft_namekey else None),
        force_mem_ctx_namekey=(True if mem_ctx_namekey else None),
        force_mem_ctx_namekey_dim=(int(ctx_namekey_dim)
                                   if mem_ctx_namekey else None),
        force_mem_ctx_namekey_match_threshold=(
            float(ctx_namekey_match_threshold) if mem_ctx_namekey else None),
        force_mem_size=(int(force_mem_size) if force_mem_size else None),
    )
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    # Pre-memory hidden as the clean WM input (matches _finalize: memory is
    # applied to out_norm(h_raw); return_hidden + this flag hands that back).
    model._latent_feedback_premem = True
    if force_wm and not model.state_readonly_at_think:
        raise RuntimeError(
            "--force_wm requires the base built with state_readonly_at_think=True "
            "(the β=0 hook). This ckpt's cfg has it OFF.")
    return model, cfg, tok


def _mem_forward(model, h, ids, read_mask=None):
    """Call WorkingMemory, passing input_emb when embedding-key addressing is on.

    The cotrain calls `mem()` DIRECTLY, bypassing `TinyLM._apply_memory` which is
    where `input_emb` is normally computed — without this the embedding-key path
    (`key_from_embedding=True`) silently falls back to cosine-on-h."""
    input_emb = None
    if (getattr(model.memory, "key_from_embedding", False)
            or getattr(model.memory, "soft_namekey", False)):
        input_emb = model.embed(ids)
    return model.memory(h, ids, read_mask=read_mask, input_emb=input_emb)


def trunk_hpre(model, ids, force_wm_mask=None):
    """Frozen trunk forward (no grad) -> pre-memory hidden out_norm(h_raw).
    `force_wm_mask` (B,T) bool: state-readonly (β=0) those positions so the
    recurrence can't carry the binding across the recall span."""
    with torch.no_grad():
        _, h_pre = model(ids, return_hidden=True, force_wm_mask=force_wm_mask)
    return h_pre


# ----------------------------------------------------------------------------- probes
@torch.no_grad()
def addressing_probe(model, tok, examples, *, mode, n, device, line_win=6):
    """mode='think_slot': prompt+'\n\n'+[THINK], read at the THINK slot.
       mode='recall_query': prompt+completion[:vpos], read at the query pos.
    Reports mass on the binding's VALUE digits (mass_on_binding), mass on the
    whole binding LINE (`vX = NNNN`, ±line_win — the read may address the
    variable-name/`=` token rather than the value digits), attn-weighted-mean
    source position / T (1.0=recency), and write-hit.
    """
    tid = int(getattr(model, "thinking_token_id"))
    mem = model.memory
    mem._capture_read = True
    n_used = 0
    write_hit = read_top1 = 0
    read_mass = line_mass = recency = wmean = bind_frac = 0.0
    for ex in examples[:n]:
        full = ex["full"]
        if mode == "think_slot":
            seq_ids = full[:ex["plen"]] + [tid]
            qpos = len(seq_ids) - 1
        else:  # recall_query: up to (excluding) the first completion value
            seq_ids = full[:ex["vpos"]]
            qpos = len(seq_ids) - 1
        if qpos < 1:
            continue
        bind_span = set(range(ex["bpos"], ex["bpos"] + ex["blen"]))
        line_span = set(range(ex["bpos"] - line_win, ex["bpos"] + ex["blen"]))
        seq = torch.tensor([seq_ids], dtype=torch.long, device=device)
        h_pre = trunk_hpre(model, seq)
        _mem_forward(model, h_pre, seq)             # populate capture
        attn = mem._last_read_attn[0]               # (T, K)
        top_idx = mem._last_top_idx[0]              # (K,)
        a = attn[qpos]                               # (K,)
        T = seq.shape[1]
        src = [int(s.item()) for s in top_idx]
        in_bind = torch.tensor([s in bind_span for s in src],
                               device=a.device, dtype=torch.bool)
        in_line = torch.tensor([s in line_span for s in src],
                               device=a.device, dtype=torch.bool)
        n_used += 1
        write_hit += int(in_bind.any().item())
        top1 = int(a.argmax().item())
        if in_bind.any():
            read_mass += float(a[in_bind].sum().item())
            read_top1 += int(in_bind[top1].item())
        if in_line.any():
            line_mass += float(a[in_line].sum().item())
        recent = (top_idx.float() >= T * 0.95)
        recency += float(a[recent].sum().item())
        wmean += float((a * top_idx.float()).sum().item()) / max(1, T)
        bind_frac += ex["bpos"] / max(1, T)
    mem._capture_read = False
    nz = max(1, n_used)
    return dict(
        mode=mode, n=n_used,
        write_in_buffer=write_hit / nz,
        read_top1_is_binding=read_top1 / nz,
        mass_on_binding=read_mass / nz,
        mass_on_binding_line=line_mass / nz,
        recency_mass=recency / nz,
        wmean_pos_frac=wmean / nz,
        binding_pos_frac=bind_frac / nz,
    )


@torch.no_grad()
def teacher_forced_recall(model, tok, examples, *, n, device,
                          force_read_alpha=None):
    """Kill-gate at the recall position. Teacher-forces the FULL value (all
    digits) and reports EXACT-MATCH accuracy (all digits argmax-correct) plus
    mean per-digit logp. WM-ON = lm_head(memory(h_pre)), WM-OFF = lm_head(h_pre);
    both share the SAME frozen trunk forward, so the only difference is the WM
    read injection -> an honest 'is the read load-bearing for recall' measure.
    Exact-match (not just the first digit) is the load-bearing metric: a value is
    1000-9999, so first-digit-only is ~10-way and too noisy to read a lift from.
    """
    mem = model.memory
    saved_alpha = mem.read_alpha.data.clone()
    if force_read_alpha is not None:
        mem.read_alpha.data.fill_(float(force_read_alpha))
    on_exact = off_exact = 0
    on_lp = off_lp = 0.0
    n_used = 0
    try:
        use_copy = bool(getattr(model, "use_copy_head", False))
        for ex in examples[:n]:
            full = ex["full"]
            v0 = ex["vpos"]
            vlen = len(ex["ans_ids"])
            seq_ids = full[:v0 + vlen]               # incl. all value digits
            seq = torch.tensor([seq_ids], dtype=torch.long, device=device)
            h_pre = trunk_hpre(model, seq)
            # Read mask = the ANSWER span [v0-1, v0+vlen). Starting at v0-1 (the
            # query position) aligns the copy-head per-answer-token OFFSET (run
            # index) with the addressed source span, matching how training drove
            # it. The leak-free recall position is v0-1 (predicts the FIRST,
            # uncopied value occurrence) — the loop never reads the restated
            # "Answer:" copy.
            mask = torch.zeros_like(seq, dtype=torch.float)
            mask[0, max(0, v0 - 1):v0 + vlen - 1] = 1.0
            inj_h = _mem_forward(model, h_pre, seq, read_mask=mask)
            # WM-ON logits include the v14 copy/pointer mix (the validated
            # multi-token recall readout). WM-OFF = plain base recurrence.
            if use_copy:
                on_logits = model.lm_head(inj_h)             # (1, T, V)
                on_logits = model._apply_copy_head(on_logits, inj_h, seq, mask)
            n_used += 1
            on_ok = off_ok = True
            on_lps = off_lps = 0.0
            for j in range(vlen):
                ppos = v0 - 1 + j                     # predicts digit full[v0+j]
                dtok = full[v0 + j]
                lo = on_logits[0, ppos] if use_copy else model.lm_head(inj_h[0, ppos])
                lf = model.lm_head(h_pre[0, ppos])
                on_ok &= (int(lo.argmax().item()) == dtok)
                off_ok &= (int(lf.argmax().item()) == dtok)
                on_lps += float(F.log_softmax(lo, -1)[dtok].item())
                off_lps += float(F.log_softmax(lf, -1)[dtok].item())
            on_exact += int(on_ok)
            off_exact += int(off_ok)
            on_lp += on_lps / vlen
            off_lp += off_lps / vlen
    finally:
        mem.read_alpha.data.copy_(saved_alpha)
    nz = max(1, n_used)
    return dict(
        n=n_used,
        wm_on_acc=on_exact / nz, wm_off_acc=off_exact / nz,
        wm_on_logp=on_lp / nz, wm_off_logp=off_lp / nz,
        delta_acc=(on_exact - off_exact) / nz,
        delta_logp=(on_lp - off_lp) / nz,
    )


def run_validation(model, tok, *, device, n_addr, n_recall, tag, eval_sets):
    print(f"\n===== VALIDATION [{tag}] read_alpha={float(model.memory.read_alpha.detach()):.4f} =====")
    for ev_name, exs in eval_sets:
        if not exs:
            print(f"  [{ev_name}] no usable examples")
            continue
        addr_t = addressing_probe(model, tok, exs, mode="think_slot",
                                  n=n_addr, device=device)
        addr_q = addressing_probe(model, tok, exs, mode="recall_query",
                                  n=n_addr, device=device)
        rec = teacher_forced_recall(model, tok, exs, n=n_recall, device=device)
        print(f"  [{ev_name}] addressing (think_slot ):"
              f" mass_bind={addr_t['mass_on_binding']:.3f}"
              f" mass_line={addr_t['mass_on_binding_line']:.3f}"
              f" wmean_pos/T={addr_t['wmean_pos_frac']:.3f}"
              f" write_in_buf={addr_t['write_in_buffer']:.2f}")
        print(f"  [{ev_name}] addressing (recall_query):"
              f" mass_bind={addr_q['mass_on_binding']:.3f}"
              f" mass_line={addr_q['mass_on_binding_line']:.3f}"
              f" wmean_pos/T={addr_q['wmean_pos_frac']:.3f}"
              f" top1_bind={addr_q['read_top1_is_binding']:.2f}")
        print(f"  [{ev_name}] recall kill-gate (n={rec['n']}):"
              f" WM-ON acc={rec['wm_on_acc']:.3f} logp={rec['wm_on_logp']:.2f}"
              f" | WM-OFF acc={rec['wm_off_acc']:.3f} logp={rec['wm_off_logp']:.2f}"
              f" | Δacc={rec['delta_acc']:+.3f} Δlogp={rec['delta_logp']:+.2f}")


# ----------------------------------------------------------------------------- train
def train(model, tok, train_examples, *, device, steps, batch, lr,
          content_aux_weight, log_every, val_cb, freeze_read_alpha=False,
          unfreeze_trunk=False, addr_aux_weight=0.0, force_wm=False,
          general_examples=None, general_frac=0.0, gen_stride=8,
          copy_gate_lr=None, read_alpha_init=1.0):
    mem = model.memory
    use_copy = bool(getattr(model, "use_copy_head", False))
    general_examples = general_examples or []
    # # general (negative) rows per batch — the SELECTIVITY signal: the gate must
    # learn to STAY CLOSED where copying the addressed binding's value is wrong.
    n_gen = int(round(batch * float(general_frac))) if general_examples else 0
    n_gen = min(n_gen, max(0, batch - 1))               # keep >=1 recall row
    n_rec = batch - n_gen
    # Re-bootstrap read_alpha (decayed to ~0.07 on v12) so the read has gradient
    # while content-addressing locks in. The v12 base shows the documented
    # "starvation loop": a weak read looks useless -> alpha shrinks -> the read
    # gets less gradient -> addressing never locks in. freeze_read_alpha pins it
    # at 1.0 so the read ALWAYS contributes (max gradient on W_q/W_k/W_proj) —
    # giving content-addressing its best shot to emerge before we judge the fix.
    # With DISCRETE-key + copy head, the recall flows through the (parameter-free
    # discrete address +) gated COPY head, NOT the W_proj injection — so the
    # always-on injection (read_alpha) is set SMALL to avoid corrupting general
    # code, and the copy gate carries selective recall.
    mem.read_alpha.data.fill_(float(read_alpha_init))
    if freeze_read_alpha:
        mem.read_alpha.requires_grad = False
    # enable the grad-keeping attn stash when the direct addressing aux is on OR
    # the copy head is on (the copy/pointer mix needs the read attention graph to
    # train the cosine addressing).
    mem._stash_read_attn_grad = bool(addr_aux_weight > 0.0) or use_copy
    # Optimizer. The copy gate (CopyReadout, bias init -6 → σ≈0.002, zero-init
    # weight) has a near-flat gradient at cold start; a dedicated higher LR lets
    # it actually CLIMB to open on recall answers (and the weight to learn the
    # answer-vs-general direction) within a short cotrain. Default = base lr.
    copy_gate_lr = float(copy_gate_lr) if copy_gate_lr else lr
    copy_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (copy_params if n.startswith("copy_head.") else base_params).append(p)
    groups = [dict(params=base_params, lr=lr)]
    if copy_params:
        groups.append(dict(params=copy_params, lr=copy_gate_lr))
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    params = base_params + copy_params

    # length-bucketed batches to minimize padding
    train_examples = sorted(train_examples, key=lambda e: len(e["full"]))
    rng = random.Random(0)

    def make_batch():
        i = rng.randrange(0, max(1, len(train_examples) - max(1, n_rec)))
        rec = train_examples[i:i + n_rec]
        gen = (random.sample(general_examples, n_gen)
               if n_gen and len(general_examples) >= n_gen else [])
        return rec, gen

    model.eval()  # deterministic trunk (no PKM eps-greedy / gist / floor); grad still flows
    step = 0
    while step < steps:
        opt.zero_grad(set_to_none=True)
        rec, gen = make_batch()
        bex = rec + gen                                  # recall rows then general
        is_gen = [False] * len(rec) + [True] * len(gen)
        Tmax = max(len(e["full"]) for e in bex)
        B = len(bex)
        ids = torch.zeros((B, Tmax), dtype=torch.long, device=device)
        mask = torch.zeros((B, Tmax), dtype=torch.float, device=device)
        tgt = torch.full((B, Tmax), -100, dtype=torch.long, device=device)
        gen_pos_mask = torch.zeros((B, Tmax), dtype=torch.bool, device=device)
        # FORCE-WM curriculum: state-readonly (β=0) the recall span — the queried
        # binding's value digits AND the query+answer span — so the recurrence
        # can no longer transport the binding to the answer; recall must flow
        # through the WM read. None when --force_wm off.
        fwm = (torch.zeros((B, Tmax), dtype=torch.bool, device=device)
               if force_wm else None)
        for b, e in enumerate(bex):
            full = e["full"]
            L = len(full)
            ids[b, :L] = torch.tensor(full, device=device)
            if is_gen[b]:
                # GENERAL-code NEGATIVE: copy head applied at ISOLATED positions
                # (offset 0 each); target = the true next token. Copying the
                # addressed binding's value is WRONG here → ∂CE/∂g closes the gate.
                gpos = list(range(2, L - 1, gen_stride))
                for t in gpos:
                    tgt[b, t] = full[t + 1]
                    mask[b, t] = 1.0
                    gen_pos_mask[b, t] = True
                continue
            # FOCUS the read + CE on the FIRST value occurrence — the ONLY true
            # long-range recall. (Later "outputs V"/"Answer: V" are copyable from
            # local context; prose is trunk-easy. Including them dilutes the WM
            # gradient ~7x, which is why the broad-mask smoke barely moved recall.)
            v0 = e["vpos"]
            vlen = len(e["ans_ids"])
            for t in range(max(0, v0 - 1), min(L - 1, v0 + vlen - 1)):
                tgt[b, t] = full[t + 1]
                mask[b, t] = 1.0
            if fwm is not None:
                bpos, blen = e["bpos"], e["blen"]
                fwm[b, bpos:min(L, bpos + blen)] = True          # binding digits
                fwm[b, max(0, v0 - 1):min(L, v0 + vlen)] = True  # query+answer
        # trunk forward.
        # JOINT (unfreeze_trunk): run the trunk WITH grad so the recall CE
        # backprops into the trunk's recurrence/embeddings — making the trunk
        # hiddens content-addressable (the MQAR ingredient freeze-trunk lacked).
        # The frozen trunk_hpre() path wraps the forward in no_grad AND detaches,
        # so it would starve the trunk of ALL gradient even with requires_grad=True
        # on the trunk params (the live --unfreeze_trunk no-op bug, fixed here).
        # _latent_feedback_premem=True -> h_pre is the pre-memory out_norm(h_raw);
        # memory is re-applied manually below so the masked answer-span read
        # carries the WM gradient.
        if unfreeze_trunk:
            _, h_pre = model(ids, return_hidden=True,     # grad flows into trunk
                             force_wm_mask=fwm)
        else:
            h_pre = trunk_hpre(model, ids, force_wm_mask=fwm).detach()
        inj_h = _mem_forward(model, h_pre, ids, read_mask=mask)  # (B,T,d) grad on WM
        pos = tgt != -100
        if use_copy:
            # v14 copy/pointer mix at the answer span (trains the cosine
            # addressing through the pointer distribution).
            lm_logits = model.lm_head(inj_h)            # (B, T, V)
            lm_logits = model._apply_copy_head(lm_logits, inj_h, ids, mask)
            sel_logits = lm_logits[pos]
        else:
            sel_logits = model.lm_head(inj_h[pos])      # (n_sel, V)
        sel_tgt = tgt[pos]
        ce = F.cross_entropy(sel_logits, sel_tgt)
        loss = ce
        aux_val = 0.0
        if content_aux_weight > 0.0:
            inj_grad = mem._last_injection_grad         # (B, T, d), grad
            aux_terms = []
            for b, e in enumerate(bex):
                if is_gen[b]:
                    continue
                qpos = e["vpos"] - 1
                if qpos < e["plen"] - 1 or qpos >= h_pre.shape[1]:
                    continue
                vec = inj_grad[b, qpos]
                tgt_row = model.lm_head.weight[e["value_tok"]].detach()
                aux_terms.append(1.0 - F.cosine_similarity(
                    vec.unsqueeze(0), tgt_row.unsqueeze(0)).squeeze(0))
            if aux_terms:
                aux = torch.stack(aux_terms).mean()
                aux_val = float(aux.item())
                loss = ce + content_aux_weight * aux
        # Direct attention-PLACEMENT aux (diagnostic best-shot): push the read
        # attention at the recall-query position ONTO the buffer slot(s) whose
        # source token lies in the queried binding's value-digit span. This is a
        # cross-entropy on the read attention (−log Σ_{slot∈binding} attn). If the
        # WM read CANNOT be driven to address the binding even with this explicit
        # supervision, content-addressing is architecturally out of reach here.
        addr_val = 0.0
        if addr_aux_weight > 0.0:
            attn_g = mem._last_read_attn_grad           # (B, T, K) w/ grad
            top_buf = mem._last_top_idx_buf             # (B, K) src positions
            addr_terms = []
            for b, e in enumerate(bex):
                if is_gen[b]:
                    continue
                qpos = e["vpos"] - 1
                if qpos < e["plen"] - 1 or qpos >= attn_g.shape[1]:
                    continue
                bind_span = set(range(e["bpos"], e["bpos"] + e["blen"]))
                srcs = top_buf[b].tolist()
                in_bind = torch.tensor(
                    [int(s) in bind_span for s in srcs],
                    device=attn_g.device, dtype=torch.bool)
                if not bool(in_bind.any()):
                    continue
                p_bind = attn_g[b, qpos][in_bind].sum().clamp_min(1e-9)
                addr_terms.append(-torch.log(p_bind))
            if addr_terms:
                addr = torch.stack(addr_terms).mean()
                addr_val = float(addr.item())
                loss = loss + addr_aux_weight * addr
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % log_every == 0 or step == 1:
            copy_g = ""
            if use_copy and getattr(model, "_last_copy_gate", None) is not None:
                g = model._last_copy_gate.squeeze(-1)        # (N,) in mask order
                gen_flat = gen_pos_mask[mask.bool()]         # (N,) aligned
                g_ans = g[~gen_flat]
                g_gen = g[gen_flat]
                copy_g = (f"  copy_bias={float(model.copy_head.gate.bias):.3f}"
                          f"  g_ans={float(g_ans.mean()) if g_ans.numel() else float('nan'):.4f}"
                          f"  g_gen={float(g_gen.mean()) if g_gen.numel() else float('nan'):.4f}")
            print(f"  step {step:4d}  ce={float(ce):.4f}  aux={aux_val:.4f}"
                  f"  addr={addr_val:.4f}"
                  f"  read_alpha={float(mem.read_alpha):.4f}"
                  f"  logit_scale={float(mem.logit_scale):.3f}"
                  f"  gate_bias_beta={float(mem.gate_bias_beta):.4f}{copy_g}")
        if val_cb is not None and step % (log_every * 5) == 0:
            val_cb(step)
    return model


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrain_v12_step9537_tok2500067328.pt")
    ap.add_argument("--train_data", default="/tmp/wm_train/compact_highN.jsonl")
    # eval sets: "name:path" — chosen in the HEADROOM regime (base WM-OFF recall
    # well below 1.0 so WM has room to be load-bearing). mb_* compact = in-dist
    # to compact training; d3_* verbose = transfer test.
    ap.add_argument("--eval_sets", default=(
        "mb_N64:/tmp/wm_ladder/mb_N64.jsonl,"
        "mb_N96:/tmp/wm_ladder/mb_N96.jsonl,"
        "d3_N48:/tmp/wm_ladder/d3_N48.jsonl,"
        "d3_N64:/tmp/wm_ladder/d3_N64.jsonl"))
    ap.add_argument("--out", default="checkpoints/wm_recall_cotrain_v12.pt")
    ap.add_argument("--mode", default="train_and_validate",
                    choices=["train_and_validate", "validate"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--content_aux_weight", type=float, default=0.5)
    ap.add_argument("--addr_aux_weight", type=float, default=0.0,
                    help="direct attention-placement aux: push read attention "
                         "onto the binding slot(s). Diagnostic best-shot for "
                         "whether the WM read CAN be made content-addressable.")
    ap.add_argument("--unfreeze_trunk", action="store_true",
                    help="JOINT trunk+WM training — the missing MQAR ingredient. "
                         "Also trains the trunk so its hiddens become "
                         "content-addressable (freeze-trunk proved insufficient). "
                         "Throwaway proof ckpt (forgets general; that's fine).")
    ap.add_argument("--freeze_read_alpha", action="store_true",
                    help="pin read_alpha=1.0 (anti-starvation) so addressing "
                         "gets max gradient before judging the fix.")
    ap.add_argument("--train_const_data", default="",
                    help="extra recall-POSITIVE rows: the `const` family of the "
                         "code_recall corpus (real NAME = value identifiers).")
    ap.add_argument("--train_const_limit", type=int, default=2000)
    ap.add_argument("--train_setvar_data", default="",
                    help="extra recall-POSITIVE rows: the `setvar` family of the "
                         "agentic_recall corpus (NAME = \"value\" name-reuse).")
    ap.add_argument("--train_setvar_limit", type=int, default=2000)
    ap.add_argument("--train_n_vars", default="48,64,96,128")
    ap.add_argument("--train_limit", type=int, default=8000)
    ap.add_argument("--train_max_len", type=int, default=1800)
    ap.add_argument("--eval_max_len", type=int, default=4000)
    ap.add_argument("--n_addr", type=int, default=64)
    ap.add_argument("--n_recall", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    # --- revive-WM: the validated v14 recall mechanism, force-attached ----------
    ap.add_argument("--mem_always_read", action="store_true",
                    help="WM reads at EVERY position (not just think tokens) so "
                         "it's always on the gradient path (PKM-style).")
    ap.add_argument("--force_wm", action="store_true",
                    help="FORCE-WM curriculum: state-readonly (β=0) the recall "
                         "span (binding+query+answer) so the recurrence can't "
                         "carry the binding — recall MUST flow through WM.")
    ap.add_argument("--mem_key_from_embedding", action="store_true",
                    help="v14 EMBEDDING-KEY addressing (causal input-embedding "
                         "window over the identifier; top1=1.00 in isolation).")
    ap.add_argument("--mem_key_window", type=int, default=4,
                    help="# recent tokens pooled for the embedding-key.")
    ap.add_argument("--use_copy_head", action="store_true",
                    help="v14 COPY/POINTER readout: mix LM dist with a copy dist "
                         "over the WM-addressed source span (exact recall).")
    ap.add_argument("--mem_discrete_key", action="store_true",
                    help="DISCRETE-HASH addressing: key the WM read on a "
                         "deterministic per-position integer code → onehot match "
                         "→ zero cross-talk. Default extractor = GENERAL lexical "
                         "identifier-span hash.")
    ap.add_argument("--mem_discrete_key_vstart", action="store_true",
                    help="Use the task-specific `vN` parser for the discrete-key "
                         "code instead of the general lexical identifier hash.")
    ap.add_argument("--mem_soft_namekey", action="store_true",
                    help="SOFT NAME-SPAN addressing: learned continuous key = "
                         "enc(pooled name-span input emb); cosine soft read over "
                         "binding slots. Adds surface-variant robustness the hash "
                         "cannot. Mutually exclusive with --mem_discrete_key.")
    ap.add_argument("--soft_namekey_dim", type=int, default=64,
                    help="soft name-key vector dim.")
    ap.add_argument("--soft_namekey_match_threshold", type=float, default=0.5,
                    help="min top-attn over binding slots to count a soft match "
                         "(gates the copy head).")
    ap.add_argument("--mem_ctx_namekey", action="store_true",
                    help="CONTEXTUAL NAME-SPAN addressing (2026-06-17): a "
                         "FULLY-LEARNED, NO-static-hash addresser. Key/query = "
                         "the trunk's CONTEXTUAL HIDDEN pooled over the "
                         "identifier name-span; DOT-PRODUCT read (learned scale). "
                         "Train with --addr_aux_weight (attention supervision). "
                         "Mutually exclusive with discrete/soft.")
    ap.add_argument("--ctx_namekey_dim", type=int, default=192,
                    help="ctx name-key vector dim (matches the probe's d_key).")
    ap.add_argument("--ctx_namekey_match_threshold", type=float, default=0.5,
                    help="min top-attn over binding slots to count a ctx match "
                         "(gates the copy head).")
    ap.add_argument("--copy_require_match", dest="copy_require_match",
                    action="store_true", default=True,
                    help="MATCH-EXISTENCE copy gating (default ON for discrete): "
                         "the copy head only fires where the discrete address "
                         "matched a real buffered binding → no garbage copy on "
                         "non-name-reuse families (userinstr/toolout). No-harm.")
    ap.add_argument("--no_copy_require_match", dest="copy_require_match",
                    action="store_false",
                    help="Disable match-existence gating (reproduce the pre-fix "
                         "cross-family over-firing behaviour).")
    ap.add_argument("--match_window", type=int, default=32,
                    help="LOCALITY window for the match-existence gate: a code "
                         "match only counts if the addressing identifier was re-"
                         "mentioned within this many tokens (rejects stale cross-"
                         "family false matches; preserves long-range recall). "
                         "0 disables locality (pure code-equality).")
    ap.add_argument("--mem_size_override", type=int, default=2048,
                    help="WM buffer size when discrete-key is on (must span the "
                         "whole sequence so every value-start is buffered).")
    ap.add_argument("--copy_gate_bias_init", type=float, default=0.0,
                    help="re-init the (force-attached) copy-gate bias to this "
                         "value so the gate has gradient to OPEN during the "
                         "short cotrain (default 0.0 → g=0.5; base built at -6).")
    ap.add_argument("--zero_shot_copy", action="store_true",
                    help="VALIDATE-ONLY: force copy gate g=1 (no training) to "
                         "measure the near-zero-shot discrete-addressed lookup.")
    # --- end-to-end: LEARNED gate + selectivity (mix with general-code negatives)
    ap.add_argument("--train_lm_head", action="store_true",
                    help="also train lm_head (co-train WM head + copy gate + "
                         "lm_head; trunk stays frozen so WM-OFF stays the base).")
    ap.add_argument("--copy_gate_lr", type=float, default=0.0,
                    help="dedicated LR for copy_head.* (the cold -6 gate has a "
                         "near-flat gradient; a higher LR lets it climb). 0=base lr.")
    ap.add_argument("--general_data", default="",
                    help="GENERAL-code NEGATIVES (distill extracted_code) for "
                         "SELECTIVITY: the gate must learn NOT to copy here.")
    ap.add_argument("--general_limit", type=int, default=3000)
    ap.add_argument("--general_max_len", type=int, default=512)
    ap.add_argument("--general_frac", type=float, default=0.5,
                    help="fraction of each batch that is general-code negatives.")
    ap.add_argument("--gen_stride", type=int, default=8,
                    help="stride for the isolated general-code copy positions.")
    ap.add_argument("--read_alpha_init", type=float, default=1.0,
                    help="initial W_proj read-injection strength. With discrete-"
                         "key+copy the copy head carries recall, so keep this "
                         "SMALL (e.g. 0.1) to keep the always-on read no-harm.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    print(f"[load] {args.ckpt}")
    model, cfg, tok = load_model(
        args.ckpt, device,
        mem_always_read=args.mem_always_read, force_wm=args.force_wm,
        mem_key_from_embedding=args.mem_key_from_embedding,
        mem_key_window=args.mem_key_window, use_copy_head=args.use_copy_head,
        mem_discrete_key=args.mem_discrete_key,
        mem_discrete_key_lexical=(not args.mem_discrete_key_vstart),
        copy_require_match=args.copy_require_match,
        match_window=args.match_window,
        mem_soft_namekey=args.mem_soft_namekey,
        soft_namekey_dim=args.soft_namekey_dim,
        soft_namekey_match_threshold=args.soft_namekey_match_threshold,
        mem_ctx_namekey=args.mem_ctx_namekey,
        ctx_namekey_dim=args.ctx_namekey_dim,
        ctx_namekey_match_threshold=args.ctx_namekey_match_threshold,
        force_mem_size=(args.mem_size_override
                        if (args.mem_discrete_key or args.mem_soft_namekey
                            or args.mem_ctx_namekey)
                        else None))
    # Open the copy gate so the short cotrain has gradient to climb it (base
    # copy head is built at bias -6 → g≈0.0025, near-flat gradient).
    if args.use_copy_head and getattr(model, "use_copy_head", False):
        with torch.no_grad():
            model.copy_head.gate.bias.fill_(float(args.copy_gate_bias_init))
    if args.zero_shot_copy and getattr(model, "use_copy_head", False):
        model._force_copy_gate = 1.0
    print(f"[revive-WM] always_read={model.memory.always_read} "
          f"discrete_key={getattr(model.memory, 'discrete_key', False)} "
          f"soft_namekey={getattr(model.memory, 'soft_namekey', False)} "
          f"ctx_namekey={getattr(model.memory, 'ctx_namekey', False)} "
          f"mem_size={model.memory.mem_size} "
          f"key_from_emb={model.memory.key_from_embedding} "
          f"key_window={model.memory.key_window} "
          f"use_copy_head={getattr(model, 'use_copy_head', False)} "
          f"copy_require_match={getattr(model.memory, 'copy_require_match', False)} "
          f"match_window={getattr(model.memory, 'discrete_key_match_window', 0)} "
          f"zero_shot_copy={args.zero_shot_copy} "
          f"force_wm={args.force_wm} state_readonly={model.state_readonly_at_think}")

    # ---- freeze trunk, leave only WM (+ copy head [+ lm_head]) trainable ----
    def wm_pred(n):
        if (n.startswith("memory.") or n.startswith("copy_head.")
                or n == "mem_alpha" or "retrieval_input_alpha" in n):
            return True
        if args.train_lm_head and n.startswith("lm_head."):
            return True
        return False
    for n, p in model.named_parameters():
        p.requires_grad = bool(wm_pred(n)) or bool(getattr(args, "unfreeze_trunk", False))
    if getattr(args, "unfreeze_trunk", False):
        print("[JOINT] trunk UNFROZEN — co-training trunk+WM (MQAR ingredient #2)")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_ct = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train_ct = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] trainable params ({train_ct:,}):")
    for n in trainable:
        print(f"    + {n}")
    print(f"[freeze] frozen params: {frozen_ct:,}  ({frozen_ct/(frozen_ct+train_ct):.4%} of total)")

    # ---- eval sets ---------------------------------------------------------
    eval_sets = []
    for spec in args.eval_sets.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, path = spec.split(":", 1)
        # Family filter inferred from the eval-set NAME prefix so a mixed-family
        # file (code_recall_heldout has const/import/signature/fname; agentic has
        # toolout/userinstr/setvar) is sliced to one family. Synthetic multibind
        # files have NO `family` field → None keeps all rows.
        low = name.lower()
        if low.startswith(("const", "realconst")):
            fam = "const"
        elif low.startswith("setvar"):
            fam = "setvar"
        elif low.startswith("import"):
            fam = "import"
        elif low.startswith(("sig", "signature")):
            fam = "signature"
        elif low.startswith(("fname", "func")):
            fam = "fname"
        elif low.startswith(("toolout", "tool")):
            fam = "toolout"
        elif low.startswith(("userinstr", "instr")):
            fam = "userinstr"
        else:
            fam = None
        exs = build_examples(path, tok, max_len=args.eval_max_len,
                             limit=args.n_recall * 2, family=fam)
        eval_sets.append((name, exs))
        print(f"[eval] {name} usable={len(exs)}"
              + (f" (family={fam})" if fam else ""))

    # ---- general-code negatives (selectivity + no-harm probes & training) ----
    general_ex = []
    if args.general_data:
        general_ex = build_general_examples(
            args.general_data, tok, limit=args.general_limit,
            max_len=args.general_max_len)
        print(f"[general] {len(general_ex)} general-code negatives from "
              f"{args.general_data}")
    gen_eval = general_ex[-128:] if len(general_ex) > 256 else general_ex
    gen_train = general_ex[:-128] if len(general_ex) > 256 else general_ex

    def run_selectivity(tag):
        if not (general_ex and getattr(model, "use_copy_head", False)):
            return
        sel = selectivity_probe(model, eval_sets[0][1], gen_eval,
                                n=args.n_recall, device=device,
                                gen_stride=args.gen_stride)
        gce = general_ce_killgate(model, gen_eval, n=64, device=device,
                                  gen_stride=args.gen_stride)
        if sel is not None:
            print(f"  [{tag}] SELECTIVITY copy_g: answer={sel['g_answer']:.4f}"
                  f" (n={sel['n_ans']}) | general={sel['g_general']:.4f}"
                  f" (n={sel['n_gen']})")
        print(f"  [{tag}] GENERAL-CODE CE: OFF(base)={gce['ce_off']:.4f}"
              f" | COPY-ONLY={gce['ce_copyonly']:.4f} (Δ={gce['delta_copyonly']:+.4f})"
              f" | FULL+inject={gce['ce_full']:.4f} (Δ={gce['delta_full']:+.4f})"
              f" (n_tok={gce['n_tok']})")

    def val_cb(step):
        run_validation(model, tok, device=device, n_addr=args.n_addr,
                       n_recall=args.n_recall, tag=f"step {step}",
                       eval_sets=eval_sets)
        run_selectivity(f"step {step}")

    # ---- BEFORE ------------------------------------------------------------
    run_validation(model, tok, device=device, n_addr=args.n_addr,
                   n_recall=args.n_recall, tag="BEFORE (v12 base, native alpha)",
                   eval_sets=eval_sets)
    run_selectivity("BEFORE")
    # extra BEFORE arm: force read_alpha=1.0 to prove it's ADDRESSING, not just
    # the small alpha, that makes the base WM non-load-bearing.
    print("\n----- BEFORE control: read_alpha forced to 1.0 (isolates addressing) -----")
    for ev_name, exs in eval_sets:
        if not exs:
            continue
        rec = teacher_forced_recall(model, tok, exs, n=args.n_recall,
                                    device=device, force_read_alpha=1.0)
        print(f"  [{ev_name}] recall (alpha=1.0): WM-ON acc={rec['wm_on_acc']:.3f}"
              f" | WM-OFF acc={rec['wm_off_acc']:.3f} | Δacc={rec['delta_acc']:+.3f}"
              f" Δlogp={rec['delta_logp']:+.2f}")

    if args.mode == "validate":
        return

    # ---- TRAIN -------------------------------------------------------------
    nv = set(int(x) for x in args.train_n_vars.split(","))
    train_ex = build_examples(args.train_data, tok, n_vars=nv,
                              max_len=args.train_max_len, limit=args.train_limit,
                              skip_first=0)
    n_mb = len(train_ex)
    if args.train_const_data:
        const_ex = build_examples(args.train_const_data, tok, family="const",
                                  max_len=args.train_max_len,
                                  limit=args.train_const_limit, skip_first=0)
        train_ex = train_ex + const_ex
        print(f"[train] +{len(const_ex)} real const recall rows")
    if args.train_setvar_data:
        setvar_ex = build_examples(args.train_setvar_data, tok, family="setvar",
                                   max_len=args.train_max_len,
                                   limit=args.train_setvar_limit, skip_first=0)
        train_ex = train_ex + setvar_ex
        print(f"[train] +{len(setvar_ex)} real setvar recall rows")
    print(f"\n[train] {len(train_ex)} recall-positive examples "
          f"(multibind={n_mb} n_vars={sorted(nv)} + const={len(train_ex)-n_mb})  "
          f"steps={args.steps} batch={args.batch} lr={args.lr} "
          f"aux_w={args.content_aux_weight}")
    train(model, tok, train_ex, device=device, steps=args.steps,
          batch=args.batch, lr=args.lr,
          content_aux_weight=args.content_aux_weight,
          log_every=args.log_every, val_cb=val_cb,
          freeze_read_alpha=args.freeze_read_alpha,
          unfreeze_trunk=bool(getattr(args, "unfreeze_trunk", False)),
          addr_aux_weight=float(getattr(args, "addr_aux_weight", 0.0)),
          force_wm=bool(args.force_wm),
          general_examples=gen_train, general_frac=args.general_frac,
          gen_stride=args.gen_stride, copy_gate_lr=args.copy_gate_lr,
          read_alpha_init=args.read_alpha_init)

    # ---- AFTER -------------------------------------------------------------
    run_validation(model, tok, device=device, n_addr=args.n_addr,
                   n_recall=args.n_recall, tag="AFTER (co-trained)",
                   eval_sets=eval_sets)
    run_selectivity("AFTER")

    # ---- base-preserved check ---------------------------------------------
    if getattr(args, "unfreeze_trunk", False):
        print("\n[base-check] JOINT mode: trunk was co-trained, so WM-OFF is NO"
              " LONGER the v12 base — the kill-gate (WM-ON vs WM-OFF on the SAME"
              " co-trained model) is the load-bearing test; compare AFTER vs"
              " BEFORE for the addressing/recency shift.")
    else:
        print("\n[base-check] trunk frozen -> WM-OFF path is the v12 base by"
              " construction (only memory.* updated).")

    # ---- save --------------------------------------------------------------
    # Record the force-attached mechanism in cfg so the ckpt reloads with it
    # (embedding-key + always-read have no state-dict footprint; the copy head's
    # params auto-detect, but the cfg flags keep the reload unambiguous).
    save_cfg = dict(cfg)
    save_cfg["mem_key_from_embedding"] = bool(model.memory.key_from_embedding)
    save_cfg["mem_key_window"] = int(model.memory.key_window)
    save_cfg["mem_always_read"] = bool(model.memory.always_read)
    save_cfg["mem_discrete_key"] = bool(getattr(model.memory, "discrete_key", False))
    save_cfg["mem_discrete_key_lexical"] = bool(
        getattr(model.memory, "discrete_key_lexical", True))
    save_cfg["mem_copy_require_match"] = bool(
        getattr(model.memory, "copy_require_match", True))
    save_cfg["mem_discrete_key_match_window"] = int(
        getattr(model.memory, "discrete_key_match_window", 32))
    save_cfg["mem_soft_namekey"] = bool(
        getattr(model.memory, "soft_namekey", False))
    if getattr(model.memory, "soft_namekey", False):
        save_cfg["mem_soft_namekey_dim"] = int(model.memory.soft_namekey_dim)
        save_cfg["mem_soft_namekey_match_threshold"] = float(
            model.memory.soft_namekey_match_threshold)
    save_cfg["mem_ctx_namekey"] = bool(
        getattr(model.memory, "ctx_namekey", False))
    if getattr(model.memory, "ctx_namekey", False):
        save_cfg["mem_ctx_namekey_dim"] = int(model.memory.ctx_namekey_dim)
        save_cfg["mem_ctx_namekey_match_threshold"] = float(
            model.memory.ctx_namekey_match_threshold)
    save_cfg["mem_size"] = int(model.memory.mem_size)
    save_cfg["use_copy_head"] = bool(getattr(model, "use_copy_head", False))
    torch.save({"state_dict": model.state_dict(),
                "step": int(args.steps), "config": save_cfg},
               args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
