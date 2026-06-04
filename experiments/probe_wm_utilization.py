"""Working-memory utilization probe (2026-06-04).

Answers: "how well does the model actually USE the WM buffer, and is buffer
SIZE or ADDRESSING the bottleneck?" — the WM analog of
probe_v5_pkm_utilization.py.

Trains a small WM model on saturated MQAR (d=64,L=2,T=512,K=128 by default —
the regime where DeltaNet's state saturates at ~0.5 recall and WM rescues it),
then on a held-out batch measures:

  - recall (sanity)
  - read-addressing HIT RATE: for each query (key k_i), does the read
    attention's top slot point at that pair's source position {2i (key),
    2i+1 (value)}? This is the core "is the lookup learned" metric.
  - read concentration: mean top-attention mass + entropy at query positions
  - write concentration: fraction of buffer slots drawn from the lookup phase
    (pos < 2K) and how many of the K value-positions are actually in the buffer
  - buffer READ coverage: # distinct slots the queries retrieve from / K

Run two seeds (a good one and a bad one) and compare to explain the variance:
    PYTHONPATH=. .venv/bin/python experiments/probe_wm_utilization.py --seed 1
    PYTHONPATH=. .venv/bin/python experiments/probe_wm_utilization.py --seed 4
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM
from experiments.tasks.mqar import make_batch as mqar_batch


def build_and_train(seed, T, K, vocab, d_model, n_layers, n_heads, d_head,
                    steps, lr, mem_size, device="cuda",
                    floor_start=0.0, floor_warmup=0):
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    thinking_id = vocab  # extra slot; never appears in MQAR inputs
    model = TinyLM(
        vocab_size=vocab + 1, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=DeltaNetAttention,
        max_T=T, feedback_mode="none",
        use_memory=True, mem_size=mem_size, mem_dim=d_model,
        mem_read_alpha_init=1.0, thinking_token_id=thinking_id,
        mem_read_alpha_floor_start=floor_start,
        mem_read_alpha_floor_warmup_steps=floor_warmup,
        activation_checkpointing=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 0.1)
    model.train()
    for step in range(1, steps + 1):
        x, y, mask = mqar_batch(256, T, vocab_size=vocab, n_pairs=K,
                                device=device)
        rmask = mask.bool()
        logits = model(x, mem_read_mask=rmask)
        logits = logits[..., :vocab]
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                               reduction="none").reshape(x.shape)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    return model


@torch.no_grad()
def probe(model, T, K, vocab, device="cuda"):
    model.eval()
    model.memory._capture_read = True
    x, y, mask = mqar_batch(256, T, vocab_size=vocab, n_pairs=K, device=device)
    rmask = mask.bool()
    logits = model(x, mem_read_mask=rmask)[..., :vocab]
    preds = logits.argmax(-1)
    recall = ((preds == y) & mask).float().sum() / mask.sum().clamp_min(1)

    attn = model.memory._last_read_attn         # (B, T, Kbuf)
    top_idx = model.memory._last_top_idx        # (B, Kbuf) source positions
    B = x.shape[0]
    twoK = 2 * K

    # ---- write concentration -------------------------------------------
    # buffer slots whose source position is in the lookup phase (< 2K)
    frac_lookup = (top_idx < twoK).float().mean().item()
    # of the K value positions {1,3,...,2K-1}, how many are in the buffer?
    val_pos = torch.arange(1, twoK, 2, device=device)          # (K,)
    in_buf = (top_idx.unsqueeze(-1) == val_pos.view(1, 1, -1)).any(1)  # (B,K)
    val_coverage = in_buf.float().mean().item()               # frac of vals stored
    key_pos = torch.arange(0, twoK, 2, device=device)
    in_buf_k = (top_idx.unsqueeze(-1) == key_pos.view(1, 1, -1)).any(1)
    key_coverage = in_buf_k.float().mean().item()

    # ---- read addressing -----------------------------------------------
    # For each query position p, which pair i does its key belong to?
    keys_seq = x[:, 0:twoK:2]                                  # (B, K)
    q_tok = x[:, twoK:]                                        # (B, Q)
    # pair index i for each query: match q_tok against keys_seq
    match = (q_tok.unsqueeze(-1) == keys_seq.unsqueeze(1))     # (B, Q, K)
    pair_i = match.float().argmax(-1)                          # (B, Q)
    # top attention slot for each query position
    qa = attn[:, twoK:, :]                                     # (B, Q, Kbuf)
    top_slot = qa.argmax(-1)                                   # (B, Q)
    src_of_top = torch.gather(top_idx, 1, top_slot)            # (B, Q) source pos
    want_val = 2 * pair_i + 1
    want_key = 2 * pair_i
    val_hit = (src_of_top == want_val).float().mean().item()
    key_hit = (src_of_top == want_key).float().mean().item()
    either_hit = ((src_of_top == want_val) | (src_of_top == want_key)).float().mean().item()
    # concentration at query positions
    top_mass = qa.max(-1).values.mean().item()
    ent = -(qa.clamp_min(1e-9) * qa.clamp_min(1e-9).log()).sum(-1).mean().item()
    # buffer read coverage: mean fraction of unique retrieved source
    # positions per row / K (low => queries collapse onto a few slots).
    uniq = []
    for b in range(B):
        uniq.append(src_of_top[b].unique().numel())
    read_coverage = (sum(uniq) / B) / K

    alpha = model.memory.read_alpha.item()
    return {
        "recall": recall.item(),
        "read_alpha": alpha,
        "val_hit": val_hit, "key_hit": key_hit, "either_hit": either_hit,
        "top_mass": top_mass, "attn_entropy": ent,
        "write_frac_lookup": frac_lookup,
        "val_coverage": val_coverage, "key_coverage": key_coverage,
        "read_coverage": read_coverage,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=16)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--mem_size", type=int, default=256)
    p.add_argument("--floor_start", type=float, default=0.0)
    p.add_argument("--floor_warmup", type=int, default=0)
    args = p.parse_args()

    model = build_and_train(args.seed, args.T, args.K, args.vocab,
                            args.d_model, args.n_layers, args.n_heads,
                            args.d_head, args.steps, args.lr, args.mem_size,
                            floor_start=args.floor_start,
                            floor_warmup=args.floor_warmup)
    m = probe(model, args.T, args.K, args.vocab)
    print(f"\n=== WM utilization (seed={args.seed}, K={args.K}, "
          f"mem_size={args.mem_size}) ===")
    print(f"  recall              {m['recall']:.3f}")
    print(f"  read_alpha          {m['read_alpha']:+.3f}")
    print(f"  -- read addressing (did the lookup learn?) --")
    print(f"  value-hit rate      {m['val_hit']:.3f}   (top slot == correct value pos)")
    print(f"  key-hit rate        {m['key_hit']:.3f}")
    print(f"  either-hit rate     {m['either_hit']:.3f}")
    print(f"  top-attn mass       {m['top_mass']:.3f}   (1.0 = peaked, ~0 = diffuse)")
    print(f"  attn entropy        {m['attn_entropy']:.3f}")
    print(f"  -- write side / buffer --")
    print(f"  buf frac from lookup {m['write_frac_lookup']:.3f}  (1.0 = stores dictionary, not queries)")
    print(f"  value coverage       {m['val_coverage']:.3f}  (frac of K values present in buffer)")
    print(f"  key coverage         {m['key_coverage']:.3f}")
    print(f"  read coverage        {m['read_coverage']:.3f}  (distinct retrieved slots / K)")


if __name__ == "__main__":
    main()
