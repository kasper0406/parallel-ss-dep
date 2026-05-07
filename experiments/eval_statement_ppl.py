"""
Phase 0 — statement-level perplexity eval harness.

Given a checkpoint, compute:
  1. Overall held-out PPL (all valid tokens within statements).
  2. Per-statement CE = mean(log-prob) over each statement's token range.
  3. If an oracle surprise score is provided per-statement, bin statements
     by oracle-surprise (z-scored over the eval set):
        - top decile surprise   → "structural-pivot" PPL
        - bottom decile surprise → "routine-token" PPL

The eval set is built from whole codeparrot Python files, parsed with
`ast` and segmented into statements via `experiments.statement_segmentation`.
We collect files until `n_eval_tokens` is met, truncating each file at
`T` tokens (matching training context).

Statement embeddings (mean-pooled hidden states over the token range) are
also returned, so the same eval can be reused as the input to the oracle
predictive head.

Usage:
  CUDA_VISIBLE_DEVICES=1 python experiments/eval_statement_ppl.py \
      --ckpt checkpoints/film_self_k3_2_28_30L_217M.pt \
      --T 512 --n_eval_tokens 32768 \
      --oracle_ckpt checkpoints/oracle_predictive_head_217M.pt \
      --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
      --out bench_stmt_ppl.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM
from experiments.statement_segmentation import parse_file_statements, Statement


# ---------------------------------------------------------------------------
# Loading checkpoints — both base/encoder ("plain DN") and FiLM-flavoured.
# ---------------------------------------------------------------------------


def load_model(path: str, device: str = "cuda",
                override_self_k: int | None = None) -> TinyLM:
    """Load any 217M checkpoint into a TinyLM. Generic loader that handles:
      - plain DN (no feedback) — for the encoder.
      - K=3 self-feeding sparse FiLM (Phase 21c) — for the base model.
      - K=3 + L_sem (Phase 22) — for the experiment.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    alpha_mode = cfg.get("feedback_alpha_mode", "scalar")
    self_k_ckpt = cfg.get("feedback_self_k", 0)
    if override_self_k is not None and alpha_mode == "surprise_modulated":
        self_k = self_k_ckpt
    else:
        self_k = (override_self_k if override_self_k is not None
                  else self_k_ckpt)
    model = TinyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_head=cfg["d_head"],
        attention_cls=DeltaNetAttention,
        max_T=0,
        feedback_mode=cfg.get("feedback_mode", "none"),
        feedback_pairs=cfg.get("feedback_pairs", ()),
        feedback_self_k=self_k,
        feedback_alpha_mode=alpha_mode,
        semantic_loss_dim=cfg.get("semantic_loss_dim", 0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for L, blk in enumerate(model.blocks):
        blk.attn.layer.layer_idx = L
    return model


# ---------------------------------------------------------------------------
# Eval data construction.
# ---------------------------------------------------------------------------


@dataclass
class FileChunk:
    """A single file's tokens (truncated to T) plus its parsed statements."""
    token_ids: torch.Tensor          # (T,) — left-padded with EOS if shorter? No: padded with 0 + mask via stmt ranges.
    statements: list                 # list of Statement (token ranges within token_ids)
    n_actual_tokens: int             # how many of the T positions are real
    file_idx: int


def build_file_chunks(tokenizer, T: int, n_eval_tokens: int,
                       dataset: str = "codeparrot/codeparrot-clean",
                       text_field: str = "content",
                       skip: int = 10_000,
                       min_stmts: int = 3,
                       seed: int = 42) -> list[FileChunk]:
    """Stream files from the val tail, parse each into statements, truncate
    to T tokens, and return a list of FileChunk.

    n_eval_tokens caps the total *actual* tokens collected (sum of
    n_actual_tokens). We discard files with fewer than `min_stmts` parseable
    statements (leaves nothing useful to score per-statement on).
    """
    from datasets import load_dataset
    val_stream = (
        load_dataset(dataset, split="train", streaming=True)
        .shuffle(seed=seed)
        .skip(skip)
    )
    chunks: list[FileChunk] = []
    total = 0
    file_idx = 0
    for ex in val_stream:
        text = ex.get(text_field, "")
        if not text or not text.strip():
            continue
        token_ids, statements = parse_file_statements(text, tokenizer)
        if len(statements) < min_stmts:
            continue
        # Truncate to T and drop statements that fall partly outside.
        if len(token_ids) >= T:
            token_ids_trunc = token_ids[:T]
            n_actual = T
        else:
            # Pad with 0s; n_actual records the real length.
            n_actual = len(token_ids)
            token_ids_trunc = token_ids + [0] * (T - n_actual)
        valid_stmts = [s for s in statements if s.end_tok_idx <= n_actual]
        if len(valid_stmts) < min_stmts:
            continue
        chunks.append(FileChunk(
            token_ids=torch.tensor(token_ids_trunc, dtype=torch.long),
            statements=valid_stmts,
            n_actual_tokens=n_actual,
            file_idx=file_idx,
        ))
        total += n_actual
        file_idx += 1
        if total >= n_eval_tokens:
            break
    return chunks


# ---------------------------------------------------------------------------
# Forward + per-statement CE / embedding computation.
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_with_hidden(model: TinyLM, x: torch.Tensor,
                          input_ids_for_aux: torch.Tensor | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a model forward and return (logits, last_hidden_states) where
    last_hidden_states is the post-out_norm hidden state at every position.

    We monkey-patch by re-running the same forward path inline. For our
    purposes (read out hidden states for pooling + read out logits for CE)
    we replicate the K-iter self-feeding loop here so we can intercept
    the final-layer hidden.
    """
    # Fast path: bypass model.forward and run the iterations explicitly,
    # capturing the post-out_norm hidden state ourselves.
    from experiments.model import _shift_right_by_k
    device = x.device
    B, T = x.shape

    # Plain forward (no feedback at all).
    if (model.feedback_mode == "none"
            and not model.feedback_xattn_pairs
            and not model.feedback_scratchpad_pairs):
        h = model.embed(x)
        if model.max_T > 0:
            pos = torch.arange(T, device=device)
            h = h + model.pos_embed(pos)
        for blk in model.blocks:
            h = blk(h, input_ids=x)
        h_norm = model.out_norm(h)
        logits = model.lm_head(h_norm)
        return logits, h_norm

    # Sparse-pair self-feeding path (K-iter).
    if model.feedback_pairs and model.feedback_self_k > 0:
        K = model.feedback_self_k
        source_layers = set(s for _, s in model.feedback_pairs)
        h_x = model.embed(x)
        if model.max_T > 0:
            pos = torch.arange(T, device=device)
            h_x = h_x + model.pos_embed(pos)
        # Iter 0 (cold).
        with torch.no_grad():
            src_states = model._sparse_pass_collect_sources(
                h_x, source_layers, film_sources_lagged=None, input_ids=x,
            )
        # Iters 1..K-2 (self-feed, no_grad).
        prev_src_states = None
        for _ in range(K - 2):
            with torch.no_grad():
                film_in = {s: _shift_right_by_k(v, model.feedback_lag)
                            for s, v in src_states.items()}
                prev_src_states = src_states
                src_states = model._sparse_pass_collect_sources(
                    h_x, source_layers, film_sources_lagged=film_in,
                    input_ids=x,
                )
        # Surprise (for surprise_modulated mode).
        film_in = {s: _shift_right_by_k(v, model.feedback_lag)
                    for s, v in src_states.items()}
        surprise_per_target: dict = {}
        if (getattr(model, "feedback_alpha_mode", "scalar")
                == "surprise_modulated" and prev_src_states is not None):
            for t, s in model.feedback_pairs:
                delta = src_states[s] - prev_src_states[s]
                surprise_per_target[t] = (
                    delta.float().norm(dim=-1).to(h_x.dtype).detach()
                )
        # Final iteration: collect post-norm.
        h = h_x
        for L, blk in enumerate(model.blocks):
            if (L in model.sparse_target_to_source
                    and model.feedback_position == "pre"):
                src = model.sparse_target_to_source[L]
                sup = surprise_per_target.get(L) if surprise_per_target else None
                h = model.sparse_feedback[str(L)](h, film_in[src], surprise=sup)
            h = blk(h, input_ids=x)
            if (L in model.sparse_target_to_source
                    and model.feedback_position == "post"):
                src = model.sparse_target_to_source[L]
                sup = surprise_per_target.get(L) if surprise_per_target else None
                h = model.sparse_feedback[str(L)](h, film_in[src], surprise=sup)
        h_norm = model.out_norm(h)
        logits = model.lm_head(h_norm)
        return logits, h_norm

    # Standard 2-pass sparse FiLM (for scalar K=0 eval of FiLM models).
    if model.feedback_pairs:
        from experiments.model import _shift_right_by_k as _sh
        source_layers = set(s for _, s in model.feedback_pairs)
        h_x = model.embed(x)
        if model.max_T > 0:
            pos = torch.arange(T, device=device)
            h_x = h_x + model.pos_embed(pos)
        # Pass 1.
        h1 = h_x
        pass1_at_sources: dict = {}
        for L, blk in enumerate(model.blocks):
            h1 = blk(h1, input_ids=x)
            if L in source_layers:
                pass1_at_sources[L] = h1
        # Pass 2 with feedback.
        h = h_x
        for L, blk in enumerate(model.blocks):
            if (L in model.sparse_target_to_source
                    and model.feedback_position == "pre"):
                src = model.sparse_target_to_source[L]
                state_above_lagged = _sh(pass1_at_sources[src],
                                           model.feedback_lag)
                h = model.sparse_feedback[str(L)](h, state_above_lagged)
            h = blk(h, input_ids=x)
            if (L in model.sparse_target_to_source
                    and model.feedback_position == "post"):
                src = model.sparse_target_to_source[L]
                state_above_lagged = _sh(pass1_at_sources[src],
                                           model.feedback_lag)
                h = model.sparse_feedback[str(L)](h, state_above_lagged)
        h_norm = model.out_norm(h)
        logits = model.lm_head(h_norm)
        return logits, h_norm

    # Fallback: just call the model.
    logits = model(x)
    return logits, None


@torch.no_grad()
def compute_per_statement_metrics(
    model: TinyLM,
    chunks: list[FileChunk],
    batch: int = 4,
    return_embeddings: bool = False,
) -> dict:
    """Forward each chunk, accumulate per-statement (sum_neg_logprob, n_tokens)
    and (if requested) per-statement mean-pooled hidden state.

    Returns dict with:
      'per_stmt_ce': list[float] — one per statement, in order of (file_idx, stmt_idx).
      'per_stmt_n_tokens': list[int]
      'per_stmt_kind': list[str]
      'per_stmt_text_preview': list[str]
      'per_stmt_file': list[int]
      'per_stmt_embedding': torch.Tensor or None — (N_stmts, d_model)
      'overall_ce': float
      'overall_n_tokens': int
    """
    device = next(model.parameters()).device
    d_model = model.embed.embedding_dim
    per_stmt_ce: list[float] = []
    per_stmt_n_tokens: list[int] = []
    per_stmt_kind: list[str] = []
    per_stmt_text: list[str] = []
    per_stmt_file: list[int] = []
    per_stmt_emb: list[torch.Tensor] = []
    sum_logprob_total = 0.0
    n_tokens_total = 0

    # Gather all chunk tensors and pad-stack into batches.
    n = len(chunks)
    for i in range(0, n, batch):
        batch_chunks = chunks[i:i + batch]
        x = torch.stack([c.token_ids for c in batch_chunks], dim=0).to(device)
        # Forward with hidden capture.
        logits, h_norm = forward_with_hidden(model, x)
        # Per-token CE on shifted labels (predict t+1 from h_t).
        # logits: (B, T, V); targets shift left by 1, so loss at pos t is
        # CE(logits[:, t], x[:, t+1]).
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for b, fc in enumerate(batch_chunks):
            for s in fc.statements:
                # Per-statement CE = mean of -log p(token_{t+1} | h_t)
                # for t in [s.start_tok_idx, s.end_tok_idx).
                # We need predictions for the *next* token: model predicts
                # x[:, t+1] from h[:, t]. So the loss-bearing positions
                # are [s.start_tok_idx, s.end_tok_idx) ∩ [0, T-1).
                start = s.start_tok_idx
                end = min(s.end_tok_idx, fc.n_actual_tokens, x.shape[1] - 1)
                if end <= start:
                    continue
                # Sum over positions; targets are x[b, t+1].
                pos_idx = torch.arange(start, end, device=device)
                gathered = log_probs[b, pos_idx]                        # (L, V)
                target_ids = x[b, pos_idx + 1]                          # (L,)
                lp = gathered.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (L,)
                neg_lp_sum = -lp.sum().item()
                n_t = pos_idx.numel()
                per_stmt_ce.append(neg_lp_sum / n_t)
                per_stmt_n_tokens.append(int(n_t))
                per_stmt_kind.append(s.kind)
                per_stmt_text.append(s.text[:200])
                per_stmt_file.append(fc.file_idx)
                sum_logprob_total += neg_lp_sum
                n_tokens_total += n_t
                if return_embeddings and h_norm is not None:
                    # Mean-pool hidden states over the statement's token range.
                    # We pool the hidden states of positions [start, end+1)
                    # — these are the inputs to the prediction head, i.e.
                    # representations of the statement's content as encoded
                    # by the model. This is *the* representation the design
                    # doc's L_sem aligns to E(s_t).
                    end_emb = min(s.end_tok_idx, fc.n_actual_tokens)
                    if end_emb > s.start_tok_idx:
                        emb = h_norm[b, s.start_tok_idx:end_emb].mean(dim=0)
                    else:
                        emb = h_norm.new_zeros(d_model)
                    per_stmt_emb.append(emb.cpu())

    overall_ce = sum_logprob_total / max(1, n_tokens_total)
    out = {
        "per_stmt_ce": per_stmt_ce,
        "per_stmt_n_tokens": per_stmt_n_tokens,
        "per_stmt_kind": per_stmt_kind,
        "per_stmt_text_preview": per_stmt_text,
        "per_stmt_file": per_stmt_file,
        "overall_ce": overall_ce,
        "overall_n_tokens": n_tokens_total,
    }
    if return_embeddings:
        out["per_stmt_embedding"] = (torch.stack(per_stmt_emb)
                                      if per_stmt_emb else None)
    return out


# ---------------------------------------------------------------------------
# Oracle / surprise loading.
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_oracle_surprise(encoder: TinyLM, oracle_head, chunks: list[FileChunk],
                              batch: int = 4) -> tuple[list[float], torch.Tensor]:
    """For each statement, compute oracle surprise = 1 - cos(pred, E(s_t)).

    `encoder` is the frozen DN baseline. We pool its post-out_norm hidden
    over each statement's token range to get E(s_t). `oracle_head` is the
    autoregressive predictive head — given the sequence of past statement
    embeddings, it predicts E(s_t).

    Returns (surprise_list, embedding_tensor) — `surprise_list` is a list
    aligned with the per-statement indexing of `compute_per_statement_metrics`.
    """
    # 1. Mean-pool encoder hidden states per statement.
    enc_metrics = compute_per_statement_metrics(
        encoder, chunks, batch=batch, return_embeddings=True
    )
    embeddings = enc_metrics["per_stmt_embedding"]   # (N, d)
    files = enc_metrics["per_stmt_file"]

    # 2. Group statements by file, compute surprise per file (autoregressive
    # over the file's statement sequence).
    surprise_list: list[float] = [float("nan")] * embeddings.shape[0]
    device = next(oracle_head.parameters()).device
    # Group consecutive same-file indices.
    n = embeddings.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n and files[j] == files[i]:
            j += 1
        # Sequence of statement embeddings for this file: [i, j).
        emb_seq = embeddings[i:j].to(device).unsqueeze(0)  # (1, S, d)
        S = emb_seq.shape[1]
        # Oracle head: given prefix [emb_0..emb_{t-1}], predict emb_t.
        # We support batch eval by passing the whole sequence with causal mask.
        preds = oracle_head(emb_seq)  # (1, S, d) — preds[t] = predict emb_t
        # cosine sim per statement.
        for k in range(S):
            if k == 0:
                # No prior context — define surprise as max (1.0) so it doesn't
                # bias the binning. (Or skip: use NaN and the binner drops it.)
                surprise_list[i + k] = float("nan")
                continue
            pred = preds[0, k - 1]                 # prediction OF emb_k
            tgt = emb_seq[0, k]
            sim = F.cosine_similarity(pred.unsqueeze(0),
                                       tgt.unsqueeze(0)).item()
            surprise_list[i + k] = 1.0 - sim
        i = j
    return surprise_list, embeddings


# ---------------------------------------------------------------------------
# Decile binning.
# ---------------------------------------------------------------------------


def bin_and_summarize(per_stmt_ce: list[float],
                       per_stmt_n_tokens: list[int],
                       per_stmt_surprise: list[float] | None,
                       ) -> dict:
    """Bin statements by surprise (drop NaNs), compute decile-wise PPL.

    PPL is computed as exp(token-weighted mean CE), so each bin's PPL
    correctly weighs longer statements more.
    """
    n = len(per_stmt_ce)
    out = {
        "n_stmts": n,
        "overall": _bin_summary(per_stmt_ce, per_stmt_n_tokens, list(range(n))),
    }
    if per_stmt_surprise is None:
        return out
    valid = [(i, s) for i, s in enumerate(per_stmt_surprise)
             if s == s and not math.isnan(s)]   # filter NaN
    if not valid:
        out["note"] = "all surprise NaN; no decile breakdown"
        return out
    # Sort by surprise ascending.
    valid.sort(key=lambda iv: iv[1])
    n_v = len(valid)
    # Top decile = highest surprise.
    top_n = max(1, n_v // 10)
    bot_n = max(1, n_v // 10)
    top_indices = [iv[0] for iv in valid[-top_n:]]
    bot_indices = [iv[0] for iv in valid[:bot_n]]

    top_summary = _bin_summary(per_stmt_ce, per_stmt_n_tokens, top_indices)
    bot_summary = _bin_summary(per_stmt_ce, per_stmt_n_tokens, bot_indices)
    out["valid_n_stmts"] = n_v
    out["top_decile"] = top_summary
    out["top_decile"]["min_surprise"] = float(valid[-top_n][1])
    out["top_decile"]["max_surprise"] = float(valid[-1][1])
    out["bottom_decile"] = bot_summary
    out["bottom_decile"]["min_surprise"] = float(valid[0][1])
    out["bottom_decile"]["max_surprise"] = float(valid[bot_n - 1][1])
    # Per-decile breakdown for context.
    deciles_n = [n_v * d // 10 for d in range(11)]
    out["all_deciles"] = []
    for d in range(10):
        s, e = deciles_n[d], deciles_n[d + 1]
        if e <= s:
            continue
        idxs = [iv[0] for iv in valid[s:e]]
        summ = _bin_summary(per_stmt_ce, per_stmt_n_tokens, idxs)
        summ["surprise_min"] = float(valid[s][1])
        summ["surprise_max"] = float(valid[e - 1][1])
        out["all_deciles"].append(summ)
    return out


def _bin_summary(per_stmt_ce, per_stmt_n_tokens, indices) -> dict:
    """Token-weighted mean CE over a subset of statement indices."""
    s = 0.0
    n = 0
    for i in indices:
        nt = per_stmt_n_tokens[i]
        s += per_stmt_ce[i] * nt
        n += nt
    if n == 0:
        return {"n_stmts": 0, "n_tokens": 0, "ce": float("nan"), "ppl": float("nan")}
    ce = s / n
    return {
        "n_stmts": len(indices),
        "n_tokens": int(n),
        "ce": float(ce),
        "ppl": float(math.exp(ce)),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                    help="Checkpoint to evaluate (plain DN, K=3 self-feeding, "
                         "or K=3 + L_sem variants).")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n_eval_tokens", type=int, default=32 * 1024,
                    help="Total real tokens to evaluate on (default 32K).")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--encoder_ckpt", type=str, default=None,
                    help="Path to the frozen encoder DN baseline checkpoint. "
                         "If provided together with --oracle_ckpt, statements "
                         "are binned by oracle surprise and top/bottom decile "
                         "PPL are reported.")
    p.add_argument("--oracle_ckpt", type=str, default=None,
                    help="Path to the trained oracle predictive head checkpoint.")
    p.add_argument("--out", type=str, default="bench_stmt_ppl.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--surprise_npy", type=str, default=None,
                    help="If provided, load pre-computed per-statement "
                         "surprise from this .npy/.pt file (skip oracle "
                         "computation; must match statement order).")
    p.add_argument("--save_surprise_npy", type=str, default=None,
                    help="If provided, save the computed per-statement "
                         "surprise array to this .pt file (for reuse by "
                         "subsequent evals).")
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, device=args.device)
    cfg_d = {
        "d_model": model.embed.embedding_dim,
        "n_layers": len(model.blocks),
        "feedback_pairs": list(model.feedback_pairs),
        "feedback_self_k": model.feedback_self_k,
        "feedback_alpha_mode": getattr(model, "feedback_alpha_mode", "scalar"),
    }
    print(f"  config: {cfg_d}")

    print(f"\nBuilding val-tail file chunks at T={args.T}, "
          f"n_eval_tokens={args.n_eval_tokens} ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    chunks = build_file_chunks(tok, T=args.T, n_eval_tokens=args.n_eval_tokens)
    n_files = len(chunks)
    n_real_toks = sum(c.n_actual_tokens for c in chunks)
    n_total_stmts = sum(len(c.statements) for c in chunks)
    print(f"  {n_files} files, {n_real_toks} real tokens, "
          f"{n_total_stmts} statements")

    print("\n--- Per-statement metrics (overall PPL) ---")
    t0 = time.perf_counter()
    metrics = compute_per_statement_metrics(model, chunks, batch=args.batch)
    elapsed = time.perf_counter() - t0
    overall_ce = metrics["overall_ce"]
    overall_ppl = math.exp(overall_ce)
    print(f"  overall CE  = {overall_ce:.4f}")
    print(f"  overall PPL = {overall_ppl:.4f}")
    print(f"  ({metrics['overall_n_tokens']} stmt-internal tokens, "
          f"{elapsed:.1f}s)")

    # Free the model now if we need to load the encoder.
    del model
    torch.cuda.empty_cache()

    # Compute or load oracle surprise.
    surprise_list = None
    if args.surprise_npy is not None:
        print(f"\nLoading pre-computed surprise from {args.surprise_npy} ...")
        surprise_loaded = torch.load(args.surprise_npy, weights_only=False)
        if isinstance(surprise_loaded, dict):
            surprise_list = surprise_loaded["per_stmt_surprise"]
        else:
            surprise_list = list(surprise_loaded)
        if len(surprise_list) != len(metrics["per_stmt_ce"]):
            raise SystemExit(
                f"surprise length {len(surprise_list)} != per-stmt count "
                f"{len(metrics['per_stmt_ce'])}"
            )
    elif args.encoder_ckpt and args.oracle_ckpt:
        print(f"\nLoading frozen encoder from {args.encoder_ckpt} ...")
        encoder = load_model(args.encoder_ckpt, device=args.device)
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
        print(f"Loading oracle head from {args.oracle_ckpt} ...")
        from experiments.oracle_train import OraclePredictor
        oracle_ckpt = torch.load(args.oracle_ckpt, map_location=args.device,
                                  weights_only=False)
        oracle_cfg = oracle_ckpt["config"]
        oracle_head = OraclePredictor(**oracle_cfg).to(args.device)
        oracle_head.load_state_dict(oracle_ckpt["state_dict"])
        oracle_head.eval()
        for p in oracle_head.parameters():
            p.requires_grad_(False)
        print("\n--- Computing oracle surprise per statement ---")
        t0 = time.perf_counter()
        surprise_list, _ = compute_oracle_surprise(
            encoder, oracle_head, chunks, batch=args.batch,
        )
        print(f"  ({time.perf_counter() - t0:.1f}s)")
        del encoder
        torch.cuda.empty_cache()
        if args.save_surprise_npy:
            torch.save({"per_stmt_surprise": surprise_list,
                         "per_stmt_kind": metrics["per_stmt_kind"],
                         "per_stmt_text_preview": metrics["per_stmt_text_preview"],
                         "per_stmt_file": metrics["per_stmt_file"]},
                        args.save_surprise_npy)
            print(f"  surprise saved to {args.save_surprise_npy}")

    print("\n--- Decile-binned metrics ---")
    summary = bin_and_summarize(
        metrics["per_stmt_ce"], metrics["per_stmt_n_tokens"], surprise_list,
    )
    print(f"  Overall:        n_stmts={summary['overall']['n_stmts']}, "
          f"n_tokens={summary['overall']['n_tokens']}, "
          f"PPL={summary['overall']['ppl']:.4f}")
    if "top_decile" in summary:
        print(f"  Top decile:     n_stmts={summary['top_decile']['n_stmts']}, "
              f"n_tokens={summary['top_decile']['n_tokens']}, "
              f"PPL={summary['top_decile']['ppl']:.4f}, "
              f"surprise_range=[{summary['top_decile']['min_surprise']:.4f},"
              f"{summary['top_decile']['max_surprise']:.4f}]")
        print(f"  Bottom decile:  n_stmts={summary['bottom_decile']['n_stmts']}, "
              f"n_tokens={summary['bottom_decile']['n_tokens']}, "
              f"PPL={summary['bottom_decile']['ppl']:.4f}, "
              f"surprise_range=[{summary['bottom_decile']['min_surprise']:.4f},"
              f"{summary['bottom_decile']['max_surprise']:.4f}]")
        if "all_deciles" in summary:
            print("  Per-decile PPL (low to high surprise):")
            for d, dec in enumerate(summary["all_deciles"]):
                print(f"    D{d}: n_stmts={dec['n_stmts']}, "
                      f"n_toks={dec['n_tokens']}, "
                      f"PPL={dec['ppl']:.4f}, "
                      f"surp=[{dec['surprise_min']:.4f},{dec['surprise_max']:.4f}]")

    results = {
        "ckpt": args.ckpt,
        "T": args.T,
        "n_eval_tokens": args.n_eval_tokens,
        "n_files": n_files,
        "n_real_tokens": int(n_real_toks),
        "n_total_stmts": int(n_total_stmts),
        "config": cfg_d,
        "summary": summary,
        # Per-statement raw arrays (for downstream analysis / inspection).
        "per_stmt_ce": metrics["per_stmt_ce"],
        "per_stmt_n_tokens": metrics["per_stmt_n_tokens"],
        "per_stmt_kind": metrics["per_stmt_kind"],
        "per_stmt_text_preview": metrics["per_stmt_text_preview"],
        "per_stmt_file": metrics["per_stmt_file"],
        "per_stmt_surprise": surprise_list,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
