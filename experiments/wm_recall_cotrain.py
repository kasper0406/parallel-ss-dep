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
                   skip_first=0):
    """Tokenize records into the training_matched format.

    Returns list of dicts: prompt_ids, comp_ids, full_ids, plen, bpos (binding,
    first occ in full), vpos (first completion occ of value), value_tok, n_vars.
    """
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


def load_model(ckpt, device):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    model, cfg = build_model_from_ckpt(ckpt)
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    # Pre-memory hidden as the clean WM input (matches _finalize: memory is
    # applied to out_norm(h_raw); return_hidden + this flag hands that back).
    model._latent_feedback_premem = True
    return model, cfg, tok


def trunk_hpre(model, ids):
    """Frozen trunk forward (no grad) -> pre-memory hidden out_norm(h_raw)."""
    with torch.no_grad():
        _, h_pre = model(ids, return_hidden=True)
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
        mem(h_pre, seq)                              # populate capture
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
        for ex in examples[:n]:
            full = ex["full"]
            v0 = ex["vpos"]
            vlen = len(ex["ans_ids"])
            seq_ids = full[:v0 + vlen]               # incl. all value digits
            seq = torch.tensor([seq_ids], dtype=torch.long, device=device)
            h_pre = trunk_hpre(model, seq)
            mask = torch.zeros_like(seq, dtype=torch.float)
            mask[0, max(0, ex["plen"] - 1):v0 + vlen - 1] = 1.0
            inj_h = mem(h_pre, seq, read_mask=mask)
            n_used += 1
            on_ok = off_ok = True
            on_lps = off_lps = 0.0
            for j in range(vlen):
                ppos = v0 - 1 + j                     # predicts digit full[v0+j]
                dtok = full[v0 + j]
                lo = model.lm_head(inj_h[0, ppos])
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
    print(f"\n===== VALIDATION [{tag}] read_alpha={float(model.memory.read_alpha):.4f} =====")
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
          unfreeze_trunk=False, addr_aux_weight=0.0):
    mem = model.memory
    # Re-bootstrap read_alpha (decayed to ~0.07 on v12) so the read has gradient
    # while content-addressing locks in. The v12 base shows the documented
    # "starvation loop": a weak read looks useless -> alpha shrinks -> the read
    # gets less gradient -> addressing never locks in. freeze_read_alpha pins it
    # at 1.0 so the read ALWAYS contributes (max gradient on W_q/W_k/W_proj) —
    # giving content-addressing its best shot to emerge before we judge the fix.
    mem.read_alpha.data.fill_(1.0)
    if freeze_read_alpha:
        mem.read_alpha.requires_grad = False
    # enable the grad-keeping attn stash only when the direct addressing aux is on
    mem._stash_read_attn_grad = bool(addr_aux_weight > 0.0)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    # length-bucketed batches to minimize padding
    train_examples = sorted(train_examples, key=lambda e: len(e["full"]))
    rng = random.Random(0)

    def make_batch():
        i = rng.randrange(0, max(1, len(train_examples) - batch))
        return train_examples[i:i + batch]

    model.eval()  # deterministic trunk (no PKM eps-greedy / gist / floor); grad still flows
    step = 0
    while step < steps:
        opt.zero_grad(set_to_none=True)
        bex = make_batch()
        Tmax = max(len(e["full"]) for e in bex)
        B = len(bex)
        ids = torch.zeros((B, Tmax), dtype=torch.long, device=device)
        mask = torch.zeros((B, Tmax), dtype=torch.float, device=device)
        tgt = torch.full((B, Tmax), -100, dtype=torch.long, device=device)
        for b, e in enumerate(bex):
            full = e["full"]
            L = len(full)
            ids[b, :L] = torch.tensor(full, device=device)
            # FOCUS the read + CE on the FIRST value occurrence — the ONLY true
            # long-range recall. (Later "outputs V"/"Answer: V" are copyable from
            # local context; prose is trunk-easy. Including them dilutes the WM
            # gradient ~7x, which is why the broad-mask smoke barely moved recall.)
            v0 = e["vpos"]
            vlen = len(e["ans_ids"])
            for t in range(max(0, v0 - 1), min(L - 1, v0 + vlen - 1)):
                tgt[b, t] = full[t + 1]
                mask[b, t] = 1.0
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
            _, h_pre = model(ids, return_hidden=True)    # grad flows into trunk
        else:
            h_pre = trunk_hpre(model, ids).detach()
        inj_h = mem(h_pre, ids, read_mask=mask)         # (B, T, d), grad on WM
        pos = tgt != -100
        sel_logits = model.lm_head(inj_h[pos])          # (n_sel, V)
        sel_tgt = tgt[pos]
        ce = F.cross_entropy(sel_logits, sel_tgt)
        loss = ce
        aux_val = 0.0
        if content_aux_weight > 0.0:
            inj_grad = mem._last_injection_grad         # (B, T, d), grad
            aux_terms = []
            for b, e in enumerate(bex):
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
            print(f"  step {step:4d}  ce={float(ce):.4f}  aux={aux_val:.4f}"
                  f"  addr={addr_val:.4f}"
                  f"  read_alpha={float(mem.read_alpha):.4f}"
                  f"  logit_scale={float(mem.logit_scale):.3f}"
                  f"  gate_bias_beta={float(mem.gate_bias_beta):.4f}")
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
    ap.add_argument("--train_n_vars", default="48,64,96,128")
    ap.add_argument("--train_limit", type=int, default=8000)
    ap.add_argument("--train_max_len", type=int, default=1800)
    ap.add_argument("--eval_max_len", type=int, default=4000)
    ap.add_argument("--n_addr", type=int, default=64)
    ap.add_argument("--n_recall", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    print(f"[load] {args.ckpt}")
    model, cfg, tok = load_model(args.ckpt, device)

    # ---- freeze trunk, leave only WM trainable -----------------------------
    wm_pred = lambda n: (n.startswith("memory.") or n == "mem_alpha"
                         or "retrieval_input_alpha" in n)
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
        exs = build_examples(path, tok, max_len=args.eval_max_len,
                             limit=args.n_recall * 2)
        eval_sets.append((name, exs))
        print(f"[eval] {name} usable={len(exs)}")

    def val_cb(step):
        run_validation(model, tok, device=device, n_addr=args.n_addr,
                       n_recall=args.n_recall, tag=f"step {step}",
                       eval_sets=eval_sets)

    # ---- BEFORE ------------------------------------------------------------
    run_validation(model, tok, device=device, n_addr=args.n_addr,
                   n_recall=args.n_recall, tag="BEFORE (v12 base, native alpha)",
                   eval_sets=eval_sets)
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
    print(f"\n[train] {len(train_ex)} examples (n_vars={sorted(nv)})  "
          f"steps={args.steps} batch={args.batch} lr={args.lr} "
          f"aux_w={args.content_aux_weight}")
    train(model, tok, train_ex, device=device, steps=args.steps,
          batch=args.batch, lr=args.lr,
          content_aux_weight=args.content_aux_weight,
          log_every=args.log_every, val_cb=val_cb,
          freeze_read_alpha=args.freeze_read_alpha,
          unfreeze_trunk=bool(getattr(args, "unfreeze_trunk", False)),
          addr_aux_weight=float(getattr(args, "addr_aux_weight", 0.0)))

    # ---- AFTER -------------------------------------------------------------
    run_validation(model, tok, device=device, n_addr=args.n_addr,
                   n_recall=args.n_recall, tag="AFTER (co-trained)",
                   eval_sets=eval_sets)

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
    save_cfg = dict(cfg)
    torch.save({"state_dict": model.state_dict(),
                "step": int(args.steps), "config": save_cfg},
               args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
