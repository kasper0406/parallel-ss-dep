"""Frozen-trunk OUTPUT-POLICY SFT: teach the discrete-WM model to EMIT the
recalled value in FREE generation (close the "output-policy gap").

THE DIAGNOSIS (validated, 2026-06-16; see report):
  The discrete-WM ckpt `wm_discrete_v12.pt` recalls perfectly when the answer
  format is given. With the `"Answer: "` prefix FORCED, free greedy generation +
  the copy head scores 96.7% (const) WM-ON vs 0% WM-OFF — i.e. the validated
  copy/pointer recall mechanism (read_alpha=0; recall is 100% from the
  parameter-free discrete copy head) ALREADY works in the short answer format.
  The ONLY failure in unconstrained generation is the OUTPUT POLICY: the model
  emits generic code and never attempts `"Answer: <value>"`, because it was
  co-trained on the recall MECHANISM but never SFT'd to ANSWER recall queries.

THE FIX (this script):
  A short SFT on the recall-instruction format. Each example is
  `(recall_prompt, "Answer: <value>")`. The trunk is FROZEN (so the base
  general-LM is preserved EXACTLY on the WM-OFF path) and ONLY the output / WM
  path trains: `lm_head` (learn to EMIT "Answer:" at recall-prompt ends) plus the
  copy head (`copy_head.gate`, so the copy fires at the short-format value span).
  A GENERAL-code anti-forget slice (distill `extracted_code`) keeps `lm_head`
  from over-fitting the answer format / forgetting code.

  Mechanically this is the autoregressive twin of the validated teacher-forced
  answer-span mask: CE on the FULL completion ("Answer:" prefix + value), with
  the v14 copy head fired (via `mem_read_mask`) on the contiguous VALUE span so
  its per-token offset copies the buffered binding token-by-token.

WHY frozen trunk: a v15 co-adapt run degraded the general LM. With the trunk
  frozen, `use_memory=False` is BYTE-IDENTICAL to the base trunk, so WM-OFF stays
  the honest base — the WM-ON >> WM-OFF generation gap proves the WM (not lm_head
  memorization) supplies the value (lm_head is shared across all problems and
  cannot store hundreds of distinct values off a frozen end-of-prompt hidden).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/wm_output_sft.py \
      --steps 400 --accum 8 --lr 2e-4 --out checkpoints/_smoke_wm_sft.pt
"""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F

from experiments.wm_recall_cotrain import (
    build_general_examples, find_sub, trunk_hpre, _mem_forward,
)


# --------------------------------------------------------------------------- data
def build_answer_sft_examples(path, tok, *, family=None, n_vars=None, limit=None,
                              max_len=1800, skip_first=0):
    """`(prompt, "Answer: <value>")` recall-instruction pairs.

    Returns dicts: full ids, plen (prompt length), vpos (first VALUE-token pos in
    full), vlen (value token count). The completion is the SHORT canonical
    `"Answer: <value>"`; the value is the leak-free annotated `answer`. We keep
    only records whose completion tokenizes as `[Answer][:][ ] + value` (a stable
    3-token prefix `[21350,42,216]`) so the generation run-start trigger is exact;
    and whose value digits actually occur in the PROMPT (a copy SOURCE exists)."""
    PREFIX = tok.encode("Answer: ", add_special_tokens=False)   # [21350, 42, 216]
    fam_set = ({family} if isinstance(family, str) else
               set(family) if family is not None else None)
    out = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines[skip_first:]:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if fam_set is not None and r.get("family") not in fam_set:
            continue
        if n_vars is not None and r.get("n_vars") not in n_vars:
            continue
        ans = str(r["answer"])
        prompt = r["problem_prompt"] + "\n\n"
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        comp_ids = tok.encode("Answer: " + ans, add_special_tokens=False)
        if comp_ids[:len(PREFIX)] != PREFIX:
            continue                                            # space merged → skip
        value_ids = comp_ids[len(PREFIX):]
        if not value_ids:
            continue
        full = prompt_ids + comp_ids
        if len(full) > max_len:
            continue
        # require a copy SOURCE in the prompt (the binding); the discrete address
        # resolves it, but this filters degenerate records up front.
        if find_sub(prompt_ids, tok.encode(ans, add_special_tokens=False)) < 0:
            continue
        plen = len(prompt_ids)
        out.append(dict(full=full, plen=plen,
                        vpos=plen + len(PREFIX), vlen=len(value_ids), answer=ans))
        if limit is not None and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------------- forward
def _example_loss(model, ex, *, device, is_general):
    """Per-example CE (micro-batch=1, no padding).

    Recall: CE on the FULL completion (the "Answer:" prefix teaches the output
    policy via lm_head; the value span fires the copy head via the read mask).
    General: CE on every code position, copy mask off (no-harm anti-forget)."""
    full = ex["full"]
    ids = torch.tensor([full], dtype=torch.long, device=device)
    L = len(full)
    # frozen trunk -> pre-memory out_norm(h_raw); detach so only WM/lm_head train.
    h_pre = trunk_hpre(model, ids).detach()
    mask = torch.zeros((1, L), dtype=torch.float, device=device)
    tgt = torch.full((1, L), -100, dtype=torch.long, device=device)
    if is_general:
        # predict every next token (positions 1..L-1 from 0..L-2).
        for t in range(0, L - 1):
            tgt[0, t] = full[t + 1]
    else:
        plen, vpos, vlen = ex["plen"], ex["vpos"], ex["vlen"]
        # CE on the whole completion: predicting positions [plen-1, L-2].
        for t in range(max(0, plen - 1), L - 1):
            tgt[0, t] = full[t + 1]
        # copy mask = contiguous VALUE-span query positions [vpos-1, vpos+vlen-2]
        # → run-index 0..vlen-1 → copy src+0..src+(vlen-1) (matches generation).
        for t in range(max(0, vpos - 1), min(L - 1, vpos + vlen - 1)):
            mask[0, t] = 1.0
    inj_h = _mem_forward(model, h_pre, ids, read_mask=mask)   # read_alpha=0 → ==h_pre
    lm_logits = model.lm_head(inj_h)
    lm_logits = model._apply_copy_head(lm_logits, inj_h, ids, mask)
    pos = tgt != -100
    return F.cross_entropy(lm_logits[pos], tgt[pos])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/wm_discrete_v12.pt")
    ap.add_argument("--out", default="checkpoints/_smoke_wm_sft.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=8,
                    help="micro-batch-1 examples accumulated per optimizer step.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--copy_gate_lr", type=float, default=1e-3,
                    help="dedicated LR for copy_head.* (cold -6 gate, near-flat).")
    ap.add_argument("--general_frac", type=float, default=0.5,
                    help="fraction of each accum that is general-code anti-forget.")
    ap.add_argument("--train_lm_head", dest="train_lm_head", action="store_true",
                    default=True)
    ap.add_argument("--train_memory", action="store_true",
                    help="also unfreeze memory.* (inert for recall here since "
                         "read_alpha=0 + discrete address is param-free).")
    # data
    ap.add_argument("--multibind_data", default="data/multibind_compact_train.jsonl")
    ap.add_argument("--const_data", default="data/code_recall_train.jsonl")
    ap.add_argument("--setvar_data", default="data/agentic_recall_train.jsonl")
    ap.add_argument("--general_data", default="data/distill_v7_phase1_astclean.jsonl")
    ap.add_argument("--multibind_n_vars", default="32,48,64")
    ap.add_argument("--per_source_limit", type=int, default=1500)
    ap.add_argument("--general_limit", type=int, default=3000)
    ap.add_argument("--max_len", type=int, default=1400)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer
    print(f"[load] {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model.to(device).eval()
    model._latent_feedback_premem = True
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    print(f"[wm] use_copy_head={getattr(model,'use_copy_head',False)} "
          f"discrete_key={getattr(model.memory,'discrete_key',False)} "
          f"always_read={model.memory.always_read} "
          f"read_alpha={float(model.memory.read_alpha.detach()):.3f} "
          f"copy_bias={float(model.copy_head.gate.bias):.2f}")

    # ---- freeze trunk; train only lm_head + copy_head (+ optional memory) ----
    def trainable(n):
        if n.startswith("copy_head."):
            return True
        if args.train_lm_head and n.startswith("lm_head."):
            return True
        if args.train_memory and n.startswith("memory."):
            return True
        return False
    # Tied-embeddings footgun: when embed.weight IS lm_head.weight, the shared
    # tensor surfaces in named_parameters() under "embed.weight" only, so the
    # "lm_head."-prefix predicate would silently leave the output projection
    # FROZEN → recall collapses without --force_answer_prefix. Detect the tie and
    # route the shared tensor through trainable().
    _tied = (getattr(model, "lm_head", None) is not None
             and getattr(model, "embed", None) is not None
             and model.lm_head.weight is model.embed.weight)
    if _tied and args.train_lm_head:
        _orig_trainable = trainable
        def trainable(n):  # noqa: F811 — extend to the tied output weight
            return _orig_trainable(n) or n == "embed.weight"
    for n, p in model.named_parameters():
        p.requires_grad = trainable(n)
    tr = [n for n, p in model.named_parameters() if p.requires_grad]
    if args.train_lm_head and not any(
            t.startswith("lm_head.") or t == "embed.weight" for t in tr):
        raise RuntimeError("train_lm_head requested but no output-projection "
                           "param is trainable — the SFT would be a no-op "
                           f"(tied={_tied}, trainable={tr})")
    tr_ct = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fz_ct = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[freeze] trainable ({tr_ct:,}): {tr}")
    print(f"[freeze] frozen {fz_ct:,} ({fz_ct/(fz_ct+tr_ct):.3%})")

    # ---- data --------------------------------------------------------------
    nv = set(int(x) for x in args.multibind_n_vars.split(","))
    recall = []
    mb = build_answer_sft_examples(args.multibind_data, tok, n_vars=nv,
                                   limit=args.per_source_limit, max_len=args.max_len)
    cn = build_answer_sft_examples(args.const_data, tok, family="const",
                                   limit=args.per_source_limit, max_len=args.max_len)
    sv = build_answer_sft_examples(args.setvar_data, tok, family="setvar",
                                   limit=args.per_source_limit, max_len=args.max_len)
    recall = mb + cn + sv
    print(f"[data] recall: multibind={len(mb)} const={len(cn)} setvar={len(sv)} "
          f"(total {len(recall)})")
    general = build_general_examples(args.general_data, tok, limit=args.general_limit,
                                     max_len=512)
    print(f"[data] general anti-forget: {len(general)}")
    random.shuffle(recall)

    # ---- optimizer (split LR for the cold copy gate) -----------------------
    copy_p = [p for n, p in model.named_parameters()
              if p.requires_grad and n.startswith("copy_head.")]
    base_p = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith("copy_head.")]
    groups = [dict(params=base_p, lr=args.lr)]
    if copy_p:
        groups.append(dict(params=copy_p, lr=args.copy_gate_lr))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    params = base_p + copy_p

    n_gen = int(round(args.accum * args.general_frac))
    n_rec = max(1, args.accum - n_gen)
    rng = random.Random(args.seed)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        rec_b = [recall[rng.randrange(len(recall))] for _ in range(n_rec)]
        gen_b = ([general[rng.randrange(len(general))] for _ in range(n_gen)]
                 if general and n_gen else [])
        ce_rec = ce_gen = 0.0
        nb = len(rec_b) + len(gen_b)
        for ex in rec_b:
            l = _example_loss(model, ex, device=device, is_general=False) / nb
            l.backward()
            ce_rec += float(l) * nb
        for ex in gen_b:
            l = _example_loss(model, ex, device=device, is_general=True) / nb
            l.backward()
            ce_gen += float(l) * nb
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:4d}  ce_recall={ce_rec/max(1,len(rec_b)):.4f}"
                  f"  ce_general={ce_gen/max(1,len(gen_b)):.4f}"
                  f"  copy_bias={float(model.copy_head.gate.bias):.3f}")

    # ---- save --------------------------------------------------------------
    save_cfg = dict(cfg)
    save_cfg["use_copy_head"] = bool(getattr(model, "use_copy_head", False))
    save_cfg["mem_discrete_key"] = bool(getattr(model.memory, "discrete_key", False))
    save_cfg["mem_always_read"] = bool(model.memory.always_read)
    torch.save({"state_dict": model.state_dict(), "step": int(args.steps),
                "config": save_cfg}, args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
