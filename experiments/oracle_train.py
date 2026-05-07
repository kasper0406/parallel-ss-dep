"""
Phase 1 — train the predictive surprise oracle on top of the frozen DN
baseline encoder.

Pipeline:
  1. Encoder E = frozen plain-DN 217M baseline (Phase 17 reference).
     Output of E for a statement = mean-pooled post-out_norm hidden state
     over the statement's token range. d_oracle = d_model = 576.
  2. Predictive head P = small 2-layer causal Transformer
     (d_model=576, n_heads=8, d_head=72, layers=2, ~2 M params).
     Input  : sequence of statement embeddings  (1, S, 576)
     Output : prediction of the next-statement embedding  (1, S, 576)
              where preds[t] = predicted embedding of statement t+1, given
              [emb_0, .., emb_t].
  3. Loss = 1 - cosine_similarity(preds[:-1], targets[1:]) averaged.

Data: we tokenize codeparrot Python files (same train tail used by the
base model), parse statements with `experiments.statement_segmentation`,
and emit one (file_idx, stmt_emb_seq) example per file.  Each forward
pass through the encoder produces the embeddings for one file's
statements; we feed those into the predictive head as a single sequence.

Training: ~30 min on GPU 1. Held-out cosine distance plateau check.

Validation gate: at the end, on a held-out batch:
  - Plot surprise distribution → heavy right tail expected.
  - Inspect top-10 / bottom-10 most/least-surprising statements manually.
  - Save to `surprise_inspection.txt`.

Usage:
  CUDA_VISIBLE_DEVICES=1 python experiments/oracle_train.py \
      --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
      --steps 20000 \
      --save_ckpt checkpoints/oracle_predictive_head_217M.pt \
      --inspection_out surprise_inspection.txt
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM
from experiments.statement_segmentation import parse_file_statements


# ---------------------------------------------------------------------------
# Predictive head — 2-layer causal Transformer with absolute pos embed
# (statement-position, not token-position; tiny vocab of S<256 typical).
# ---------------------------------------------------------------------------


class OraclePredictor(nn.Module):
    """A small 2-layer causal Transformer predicting next-statement embedding.

    Uses standard sinusoidal-ish learned positional embeddings (max_len=256)
    over statement positions, RMSNorm + causal multi-head self-attention +
    SwiGLU MLP. Output is the prediction in the same d-dim space.
    """

    def __init__(self, d_model: int = 576, n_heads: int = 8,
                 d_head: int = 72, layers: int = 2, max_len: int = 256,
                 ff_mult: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.layers = layers
        self.max_len = max_len
        assert n_heads * d_head == d_model, \
            f"n_heads*d_head ({n_heads}*{d_head}={n_heads*d_head}) != d_model ({d_model})"
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            _OracleBlock(d_model, n_heads, d_head, ff_mult) for _ in range(layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (B, S, d) — sequence of statement embeddings (E(s_0..s_{S-1})).
        Returns preds: (B, S, d) where preds[t] is the prediction of E(s_{t+1})
        given the prefix [E(s_0)..E(s_t)]. Causal-shifted at position t.
        """
        B, S, D = emb.shape
        pos = torch.arange(S, device=emb.device).clamp(max=self.max_len - 1)
        h = self.in_norm(emb) + self.pos_embed(pos).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)


class _OracleBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, ff_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        # Causal mask.
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool),
                          diagonal=1)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder loader — same layout as eval_filmed_ppl_217m.load_film_217m, but
# tolerant of plain-DN checkpoints (no feedback, no self-feeding).
# ---------------------------------------------------------------------------


def load_encoder(path: str, device: str = "cuda") -> TinyLM:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TinyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_head=cfg["d_head"],
        attention_cls=DeltaNetAttention,
        max_T=0,
        feedback_mode=cfg.get("feedback_mode", "none"),
        feedback_pairs=cfg.get("feedback_pairs", ()),
        feedback_self_k=cfg.get("feedback_self_k", 0),
        feedback_alpha_mode=cfg.get("feedback_alpha_mode", "scalar"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for L, blk in enumerate(model.blocks):
        blk.attn.layer.layer_idx = L
    return model


@torch.no_grad()
def encode_file_statements(encoder: TinyLM, token_ids: list[int],
                             statements: list, T: int = 512,
                             device: str = "cuda") -> torch.Tensor:
    """Forward `token_ids` through the encoder (truncated to T) and return
    mean-pooled hidden states per statement.

    Returns (S, d_model) tensor where S = number of statements that fully
    fit within [0, T). Statements that fall partly outside are dropped.
    """
    if not token_ids:
        return torch.empty(0, encoder.embed.embedding_dim, device=device)
    n_actual = min(len(token_ids), T)
    if n_actual < T:
        ids = token_ids + [0] * (T - n_actual)
    else:
        ids = token_ids[:T]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    # Encoder is plain DN: simple forward, capture out_norm hidden.
    h = encoder.embed(x)
    if encoder.max_T > 0:
        pos = torch.arange(T, device=device)
        h = h + encoder.pos_embed(pos)
    for blk in encoder.blocks:
        h = blk(h, input_ids=x)
    h_norm = encoder.out_norm(h)              # (1, T, d)
    # Pool per statement.
    valid_stmts = [s for s in statements if s.end_tok_idx <= n_actual]
    if not valid_stmts:
        return torch.empty(0, encoder.embed.embedding_dim, device=device)
    embs = []
    for s in valid_stmts:
        embs.append(h_norm[0, s.start_tok_idx:s.end_tok_idx].mean(dim=0))
    return torch.stack(embs, dim=0)            # (S, d)


# ---------------------------------------------------------------------------
# Streaming dataset of (file embedding sequences) for the oracle.
# ---------------------------------------------------------------------------


class StatementEmbeddingStream:
    """Stream files from a HF dataset, parse statements, encode via the
    frozen encoder, and yield (S, d) embedding sequences.

    We yield ONE example per file. The oracle head batches by stacking
    sequences of compatible length (or running each file as B=1 for
    simplicity at training time).
    """

    def __init__(self, encoder, tokenizer, dataset: str, text_field: str,
                 split: str, T: int, device: str, skip: int = 0,
                 seed: int = 0, min_stmts: int = 5,
                 max_stmts: int | None = 200):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.text_field = text_field
        self.split = split
        self.T = T
        self.device = device
        self.skip = skip
        self.seed = seed
        self.min_stmts = min_stmts
        self.max_stmts = max_stmts

    def stream_files(self):
        from datasets import load_dataset
        ds = (
            load_dataset(self.dataset, split=self.split, streaming=True)
            .shuffle(seed=self.seed)
            .skip(self.skip)
        )
        for ex in ds:
            text = ex.get(self.text_field, "") or ""
            if not text.strip():
                continue
            token_ids, stmts = parse_file_statements(text, self.tokenizer)
            if not stmts or len(stmts) < self.min_stmts:
                continue
            # Truncate to T and drop overflow statements.
            n_actual = min(len(token_ids), self.T)
            valid_stmts = [s for s in stmts if s.end_tok_idx <= n_actual]
            if len(valid_stmts) < self.min_stmts:
                continue
            if self.max_stmts is not None and len(valid_stmts) > self.max_stmts:
                valid_stmts = valid_stmts[:self.max_stmts]
            yield text, token_ids, valid_stmts


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", type=str, required=True)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=4,
                    help="Number of files per micro-batch (variable S, "
                         "stacked with padding).")
    p.add_argument("--d_oracle", type=int, default=576)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_head", type=int, default=72)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_ckpt", type=str, default=None)
    p.add_argument("--inspection_out", type=str, default="surprise_inspection.txt")
    p.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean")
    p.add_argument("--text_field", type=str, default="content")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading encoder: {args.encoder_ckpt}")
    encoder = load_encoder(args.encoder_ckpt, device=args.device)
    for q in encoder.parameters():
        q.requires_grad_(False)
    encoder.eval()
    print(f"  encoder params: {encoder.num_params() / 1e6:.1f}M, d_model="
          f"{encoder.embed.embedding_dim}")

    # Tokenizer.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # Predictive head.
    oracle = OraclePredictor(
        d_model=args.d_oracle, n_heads=args.n_heads, d_head=args.d_head,
        layers=args.layers, max_len=args.max_len,
    ).to(args.device)
    oracle_params = sum(p.numel() for p in oracle.parameters())
    print(f"Oracle predictor: {oracle_params/1e6:.2f}M params "
          f"(d={args.d_oracle}, h={args.n_heads}, layers={args.layers})")

    # Optimizer.
    opt = torch.optim.AdamW(oracle.parameters(), lr=args.lr,
                             betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)

    # Train data: same val-tail-skipped train stream as the base model.
    train_stream = StatementEmbeddingStream(
        encoder=encoder, tokenizer=tok,
        dataset=args.dataset, text_field=args.text_field, split="train",
        T=args.T, device=args.device, skip=0, seed=args.seed,
    )
    # Held-out val: skip past train, use the same skip=10_000 boundary as
    # the base model val tail.
    val_stream = StatementEmbeddingStream(
        encoder=encoder, tokenizer=tok,
        dataset=args.dataset, text_field=args.text_field, split="train",
        T=args.T, device=args.device, skip=10_000, seed=42,
    )

    # Pre-cache a small held-out batch of (token_ids, stmts) pairs so val
    # cost is consistent across runs.
    print("\nCaching held-out val files (50 files) ...")
    val_iter = val_stream.stream_files()
    val_cache = []
    for i, (text, ids, stmts) in enumerate(val_iter):
        if i >= 50:
            break
        val_cache.append((text, ids, stmts))
    print(f"  cached {len(val_cache)} files")

    # Train loop. We stream files; for each file, encode statements with the
    # frozen encoder and run the oracle. Loss = 1 - mean_cosine_sim of preds
    # to actual next-statement embeddings.
    print(f"\n{'step':>6}  {'tloss':>8}  {'cos':>8}  {'lr':>9}")
    losses = []
    cos_window = []
    t0 = time.perf_counter()
    train_files = train_stream.stream_files()
    step = 0
    while step < args.steps:
        try:
            text, token_ids, stmts = next(train_files)
        except StopIteration:
            print("  [info] train stream exhausted; restarting")
            train_files = train_stream.stream_files()
            continue
        # Encode statements with frozen encoder (B=1).
        with torch.no_grad():
            embs = encode_file_statements(
                encoder, token_ids, stmts, T=args.T, device=args.device,
            )                                                 # (S, d)
        if embs.shape[0] < 3:
            continue
        embs = embs.unsqueeze(0)                               # (1, S, d)
        # Forward oracle.
        preds = oracle(embs)                                   # (1, S, d)
        # Loss: cosine distance between preds[:, :-1] and embs[:, 1:].
        # preds[t] = prediction of emb[t+1].
        if embs.shape[1] < 2:
            continue
        cos = F.cosine_similarity(
            preds[:, :-1], embs[:, 1:], dim=-1,
        )                                                       # (1, S-1)
        loss = (1.0 - cos).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        losses.append(float(loss.item()))
        cos_window.append(float(cos.mean().item()))
        if step % args.log_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - t0
            print(f"{step:>6d}  "
                  f"{sum(losses[-args.log_every:])/min(args.log_every, len(losses)):>8.4f}  "
                  f"{sum(cos_window[-args.log_every:])/min(args.log_every, len(cos_window)):>8.4f}  "
                  f"{sched.get_last_lr()[0]:>9.2e}  "
                  f"({step / elapsed:.1f} step/s)")
        if step % args.val_every == 0 or step == args.steps:
            # Held-out cosine distance.
            oracle.eval()
            with torch.no_grad():
                vlosses = []
                vcos = []
                for text, token_ids, stmts in val_cache:
                    embs = encode_file_statements(
                        encoder, token_ids, stmts, T=args.T,
                        device=args.device,
                    )
                    if embs.shape[0] < 3:
                        continue
                    embs = embs.unsqueeze(0)
                    preds = oracle(embs)
                    cos = F.cosine_similarity(
                        preds[:, :-1], embs[:, 1:], dim=-1,
                    )
                    vlosses.append(float((1.0 - cos).mean().item()))
                    vcos.append(float(cos.mean().item()))
            print(f"        VAL  cos_dist={sum(vlosses)/max(1,len(vlosses)):.4f}  "
                  f"cos={sum(vcos)/max(1,len(vcos)):.4f}  "
                  f"({len(vlosses)} files)")
            oracle.train()

    # Save.
    if args.save_ckpt:
        ckpt = {
            "state_dict": oracle.state_dict(),
            "config": {
                "d_model": args.d_oracle, "n_heads": args.n_heads,
                "d_head": args.d_head, "layers": args.layers,
                "max_len": args.max_len,
            },
        }
        pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save_ckpt)
        print(f"\nOracle checkpoint saved to {args.save_ckpt}")

    # Inspection — dump top/bottom-10 statements by oracle surprise on val.
    print("\n--- Validation gate: surprise distribution + top/bottom-10 ---")
    oracle.eval()
    all_surprises: list[tuple[float, str, str, int]] = []
    # (surprise, kind, text, file_idx)
    with torch.no_grad():
        for fi, (text, token_ids, stmts) in enumerate(val_cache):
            embs = encode_file_statements(
                encoder, token_ids, stmts, T=args.T, device=args.device,
            )
            if embs.shape[0] < 3:
                continue
            embs1 = embs.unsqueeze(0)
            preds = oracle(embs1)
            for k in range(1, embs.shape[0]):
                # Surprise of stmt[k] given prefix [emb_0..emb_{k-1}].
                pred = preds[0, k - 1]
                tgt = embs[k]
                sim = F.cosine_similarity(pred.unsqueeze(0),
                                           tgt.unsqueeze(0)).item()
                surp = 1.0 - sim
                all_surprises.append((surp, stmts[k].kind, stmts[k].text, fi))

    if not all_surprises:
        print("  no surprise data collected — skipping inspection")
    else:
        n = len(all_surprises)
        all_surprises.sort(key=lambda t: t[0])
        # Histogram + summary stats.
        import numpy as np
        sup_arr = np.array([s[0] for s in all_surprises])
        bins = np.linspace(sup_arr.min(), sup_arr.max(), 21)
        counts, edges = np.histogram(sup_arr, bins=bins)
        max_c = max(int(counts.max()), 1)
        hist_lines = [f"surprise distribution (n={n}, "
                       f"min={sup_arr.min():.4f}, max={sup_arr.max():.4f}, "
                       f"mean={sup_arr.mean():.4f}, std={sup_arr.std():.4f}):"]
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            bar = "█" * int(40 * c / max_c)
            hist_lines.append(f"  {lo:8.4f} – {hi:8.4f} | {c:6d} {bar}")
        hist_text = "\n".join(hist_lines)
        print(hist_text)

        # Top-10 most surprising.
        print("\nTop-10 most-surprising statements:")
        top_lines = []
        for surp, kind, text, fi in reversed(all_surprises[-10:]):
            preview = text.replace("\n", "\\n")[:140]
            top_lines.append(f"  surp={surp:.4f}  kind={kind:14s}  file={fi:3d}  {preview!r}")
            print(top_lines[-1])

        # Bottom-10 least surprising.
        print("\nBottom-10 least-surprising statements:")
        bot_lines = []
        for surp, kind, text, fi in all_surprises[:10]:
            preview = text.replace("\n", "\\n")[:140]
            bot_lines.append(f"  surp={surp:.4f}  kind={kind:14s}  file={fi:3d}  {preview!r}")
            print(bot_lines[-1])

        if args.inspection_out:
            with open(args.inspection_out, "w") as f:
                f.write(f"Oracle inspection — encoder={args.encoder_ckpt}\n")
                f.write(f"oracle config: d={args.d_oracle}, h={args.n_heads}, "
                        f"layers={args.layers}\n")
                f.write(f"validation file count: {len(val_cache)}\n")
                f.write(f"total statements scored: {n}\n\n")
                f.write(hist_text + "\n\n")
                f.write("Top-10 most-surprising statements:\n")
                f.write("\n".join(top_lines) + "\n\n")
                f.write("Bottom-10 least-surprising statements:\n")
                f.write("\n".join(bot_lines) + "\n")
            print(f"\nInspection written to {args.inspection_out}")

    print(f"\nTotal time: {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
