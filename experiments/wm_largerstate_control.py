"""FAIR-BASELINE CONTROL: does bigger DeltaNet recurrent state reach the
saturating-regime multibind recall that the WM copy-readout hits (~100%)?

The validated WM mechanism (embedding-key addressing + copy readout,
`wm_multitok_readout.py`) gets ~100% EXACT recall on the saturating multibind
regime (N>=48) where the production d_head=64 DeltaNet recurrence scores ~0.
BUT the delta-rule recurrence IS itself a content-addressable associative
memory (read = S.q); it saturates at N>=48 because of finite state RANK, not
because addressing is impossible. So before committing v14 to the WM side-table,
this control asks: does simply enlarging the recurrent state (bigger d_head ->
higher state rank) close the gap, cheaper, with NO side memory?

ARMS (fair: identical data / steps / batch / lr / seed; ONLY d_head changes).
With d_model held fixed the q/k/v/o projections are d_model x d_model regardless
of the head split, so PARAMETERS are essentially identical across arms (b_proj
differs by a few scalars). The ONLY thing that changes is the per-head state
matrix d_head x d_head and hence the total recurrent state n_heads*d_head^2 =
d_model*d_head (linear in d_head):
  - d_head=64  (n_heads = d_model/64)  : 1x state  (production setting)
  - d_head=128 (n_heads = d_model/128) : 2x state
  - d_head=256 (n_heads = d_model/256) : 4x state
  - deltaproduct: d_head=64 + K Householder products/token (higher EFFECTIVE
    rank per token) -- gated kernel, verify it runs on sm_120 first.

TASK: gen_multibind_recall saturating regime. Assign N 4-digit vars, then
`print(vX)`; completion = "...`vX` is set to {ans}, ...". Train with LM loss on
the COMPLETION tokens only (the program/prompt is masked -> the ONLY gradient
to encode the bindings is the recall at "is set to ___"). Eval = teacher-forced
EXACT-match of the answer tokens at the FIRST (long-range-recall) occurrence,
located via the fast-tokenizer offset map (NOT the trivial later "Answer:" copy).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. .venv/bin/python experiments/wm_largerstate_control.py \
      --arch deltanet --d_head 64 --steps 1500
"""
from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn.functional as F

from experiments.model import TinyLM
from experiments.layers import DeltaNetAttention, GatedDeltaProductAttention


def load_examples(path, tok, n, prompt_len_field="problem_prompt"):
    import json
    out = []
    with open(path) as f:
        for line in f:
            if len(out) >= n:
                break
            d = json.loads(line)
            prompt = d["problem_prompt"]
            comp = d["qwen_completion"]
            ans = str(d["answer"])
            full = prompt + comp
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offs = enc["offset_mapping"]
            pchars = len(prompt)
            # completion-token mask (target gets gradient iff it starts in completion)
            comp_mask = [int(o[0] >= pchars) for o in offs]
            # answer span: FIRST occurrence of `ans` inside the completion text
            try:
                a_rel = comp.index(ans)
            except ValueError:
                continue
            a_cs = pchars + a_rel
            a_ce = a_cs + len(ans)
            ans_tok_idx = [i for i, o in enumerate(offs)
                           if o[1] > o[0] and o[0] >= a_cs and o[1] <= a_ce]
            if not ans_tok_idx:
                continue
            out.append(dict(ids=ids, comp_mask=comp_mask,
                            ans_idx=ans_tok_idx, n_vars=d.get("n_vars")))
    return out


def build_model(arch, d_model, n_layers, d_head, num_householder, vocab, device,
                seed):
    torch.manual_seed(seed)
    assert d_model % d_head == 0, f"d_model {d_model} % d_head {d_head} != 0"
    n_heads = d_model // d_head
    if arch == "deltanet":
        cls = DeltaNetAttention
    elif arch == "deltaproduct":
        import functools
        cls = functools.partial(GatedDeltaProductAttention,
                                num_householder=num_householder)
    else:
        raise ValueError(arch)
    model = TinyLM(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        d_ff=4 * d_model,          # identical across arms (fairness)
        attention_cls=cls,
        max_T=0,
        feedback_mode="none",
        use_memory=False,
        output_gate=False,
        tie_embeddings=True,
    ).to(device)
    return model, n_heads


def collate(batch, pad_id, device, loss_mode="completion"):
    L = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    tmask = torch.zeros((len(batch), L), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"])
        if loss_mode == "answer":
            # supervise ONLY the first-occurrence answer tokens (pure recall
            # objective; matches the WM-readout's clean answer-position loss)
            for t in b["ans_idx"]:
                tmask[i, t] = True
        else:
            tmask[i, :n] = torch.tensor([bool(m) for m in b["comp_mask"]])
    return ids.to(device), tmask.to(device)


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def train(model, data, *, steps, batch, lr, pad_id, device, seed,
          loss_mode="completion", log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    rng = random.Random(seed)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        bs = [data[rng.randrange(len(data))] for _ in range(batch)]
        ids, tmask = collate(bs, pad_id, device, loss_mode)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = _logits(model(ids))
        logits = logits[:, :-1].float()
        tgt = ids[:, 1:].clone()
        # target t is supervised iff token t is a completion token
        sup = tmask[:, 1:]
        tgt[~sup] = -100
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"  step {step:4d}/{steps}  ce={float(loss):.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)


@torch.no_grad()
def evaluate(model, data, *, batch, pad_id, device):
    """Teacher-forced EXACT-match recall of the answer tokens (first/long-range
    occurrence). Returns (exact_rate, first_tok_rate, n)."""
    model.eval()
    exact = first = n = 0
    for s in range(0, len(data), batch):
        bs = data[s:s + batch]
        ids, _ = collate(bs, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = _logits(model(ids)).float()
        pred = logits.argmax(-1)             # pred[t] predicts token t+1
        for i, b in enumerate(bs):
            ai = b["ans_idx"]
            ok_all = True
            ok_first = None
            for t in ai:
                p = int(pred[i, t - 1].item())
                tru = int(b["ids"][t])
                if ok_first is None:
                    ok_first = (p == tru)
                if p != tru:
                    ok_all = False
            exact += int(ok_all)
            first += int(bool(ok_first))
            n += 1
    return exact / max(1, n), first / max(1, n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["deltanet", "deltaproduct"],
                    default="deltanet")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--num_householder", type=int, default=2)
    ap.add_argument("--train", default="/tmp/wm_mt/train.jsonl")
    ap.add_argument("--eval", default=("N48:/tmp/wm_mt/held_N48.jsonl,"
                                       "N64:/tmp/wm_mt/held_N64.jsonl,"
                                       "N96:/tmp/wm_mt/held_N96.jsonl"))
    ap.add_argument("--n_train", type=int, default=2500)
    ap.add_argument("--n_eval", type=int, default=250)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss_mode", choices=["completion", "answer"],
                    default="completion",
                    help="completion=LM loss on all completion tokens; "
                         "answer=loss ONLY on first-occurrence answer tokens "
                         "(pure recall objective)")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    vocab = len(tok)

    print(f"[load] train={args.train}", flush=True)
    train_data = load_examples(args.train, tok, args.n_train)
    eval_sets = []
    for spec in args.eval.split(","):
        nm, path = spec.split(":", 1)
        eval_sets.append((nm, load_examples(path, tok, args.n_eval)))
    print(f"[data] train={len(train_data)}  "
          + "  ".join(f"{nm}={len(d)}" for nm, d in eval_sets), flush=True)

    model, n_heads = build_model(args.arch, args.d_model, args.n_layers,
                                 args.d_head, args.num_householder, vocab,
                                 device, args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    state_per_layer = n_heads * args.d_head * args.d_head
    print(f"\n########## ARM arch={args.arch} d_model={args.d_model} "
          f"d_head={args.d_head} n_heads={n_heads} "
          f"householder={args.num_householder if args.arch=='deltaproduct' else '-'} "
          f"##########", flush=True)
    print(f"  params={n_params/1e6:.2f}M  recurrent_state/layer="
          f"{state_per_layer} floats (= d_model*d_head"
          f"{'*householder' if args.arch=='deltaproduct' else ''})", flush=True)

    print(f"  loss_mode={args.loss_mode}", flush=True)
    train(model, train_data, steps=args.steps, batch=args.batch, lr=args.lr,
          pad_id=pad_id, device=device, seed=args.seed,
          loss_mode=args.loss_mode)

    print(f"\n===== RECALL (teacher-forced exact-match) "
          f"arch={args.arch} d_head={args.d_head} =====", flush=True)
    print(f"{'set':6s} {'n':>4s} | {'EXACT':>7s} {'FIRST_TOK':>9s}", flush=True)
    for nm, d in eval_sets:
        ex, fr, n = evaluate(model, d, batch=8, pad_id=pad_id, device=device)
        print(f"{nm:6s} {n:>4d} | {ex:>7.3f} {fr:>9.3f}", flush=True)
    print("ARM DONE", flush=True)


if __name__ == "__main__":
    main()
