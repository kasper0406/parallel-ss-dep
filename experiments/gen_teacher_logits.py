"""Offline generation of a teacher TOP-K logit store for distillation (task #104).

Streams the SAME packed token stream the student trainer would consume
(`MixedSourceStream` with the same mix / seed / T / batch / think-burst
settings), runs a frozen teacher forward at each position, keeps the top-k
next-token logits, and writes them to a sharded store (`teacher_logits_io.py`)
in STREAM ORDER. The student then trains by reading this store with
`train_lm.py --distill_logits_dir <out_dir>` — no teacher in the training loop.

  ┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────┐
  │ gen_teacher_logits  │ ───► │ sharded top-k store   │ ───► │ train_lm.py  │
  │ (this file, ONCE)   │      │ (manifest + shards)   │      │ --distill_   │
  └─────────────────────┘      └──────────────────────┘      │  logits_dir  │
                                                              └──────────────┘

HARD REQUIREMENT — row order == training token order
----------------------------------------------------
The store's row j MUST be the j-th token the trainer sees. The trainer asserts
the stored `input_ids` equal its live batch tokens per block and ABORTS on any
mismatch, so a desync is caught loudly — but to avoid that abort you MUST run BOTH
this generator AND the offline-KD training run with the SAME:

    --data_mix / --tokenizer / --seed / --T / --batch / --num_workers
    and the SAME think-burst settings (default: think-burst OFF on both).

The DataLoader interleaving across `num_workers` is deterministic for a fixed
worker count + seed, but it is the most fragile knob: --num_workers MUST be 0
on both this generator and the trainer (MixedSourceStream seeds each worker as
base_seed+17*worker_id; worker_id==0 for both num_workers=0 [main-process
iteration] and a single num_workers=1 worker, so 0/1 happen to coincide, but
this generator only accepts 0 — see the --num_workers flag below). Any
non-zero worker count here that the trainer doesn't mirror exactly produces a
DIFFERENT global token stream and the trainer's alignment assertion will fire.
Think-burst insertion is OFF by default (--think_burst_prob 0.0) because the
teacher has no think token and inserted think tokens shift alignment; the
trainer FORCES --think_burst_prob 0 under --distill_logits_dir, so leave it off
here too.

Teacher forward: this reference implementation uses HF `AutoModelForCausalLM`
forward + `torch.topk` on the next-token logits — the CORRECTNESS reference.
For production speed, vLLM `prompt_logprobs` (or an offline vLLM logits dump) is
the throughput path; it returns the same top-k log-probs far faster. This file
keeps the simple HF path so the store format and alignment are easy to verify.

CROSS-DOCUMENT STATE ISOLATION (default ON) — the teacher context MUST match
what the student can see
--------------------------------------------------------------------------
`MixedSourceStream` packs multiple documents into each T-length block,
separated by EOS; the student trains with `emit_doc_ids=True` and its DeltaNet
recurrent state is RESET at every document boundary (`doc_ids` -> cu_seqlens,
see AGENTS.md "Cross-document state isolation"). If this generator conditioned
the teacher on the full, un-isolated packed block (the old behaviour), every
stored teacher row after the FIRST document in a block would be a KD target
conditioned on context the student structurally cannot see — a silent
train/target mismatch that gets worse the further a position sits from the
start of its document.

Fix (default ON, `--no_doc_isolation` reproduces the old behaviour): stream
with `emit_doc_ids=True` (does NOT change the token stream — it only attaches
an aligned per-position document-id channel, see the comment at the
`MixedSourceStream(...)` call site) and split each packed row at its document
boundaries via `_doc_segments`; the teacher runs on each per-document segment
SEPARATELY (positions restart at 0 per document, matching the student, and
matching what the teacher itself saw during its own — presumably
per-document — training; RoPE is relative so a position restart is exactly
what a fresh document looks like to the teacher too). Results are stitched
back into the ORIGINAL packed order via `teacher_topk_doc_isolated` so the
store's row-order/format is completely unchanged — store[t] is still the
teacher's prediction "at packed position t", just conditioned on the correct
(same-document) prefix.

Boundary convention (do not change the store format for this): at the LAST
position of each document, the stored row is whatever the teacher predicts
after that document's last token — for a multi-doc packed block this is NOT
the packed stream's actual next token (which belongs to an unrelated
following document), and is not meant to be. This is intentional: the
CONSUMER (`train_lm.py`) masks the KD loss at any position whose *target*
crosses a document boundary, using its own `doc_ids` — so this "boundary" row
is simply a row the trainer never uses as a KD target, and needs no special
handling on the write side.

~1B DeltaNet student note: the store is independent of student size — the SAME
store trains a 287M or a ~1B student. A ~1B DeltaNet config is roughly
`--d_model 1536 --n_layers 24 --n_heads 24 --d_head 64` (≈1.0B params; scale
d_model/n_layers from the 287M `--d_model 896 --n_layers 10` shape). Do not add
launchers here.

Offline KD requires a teacher that SHARES the student tokenizer (no
cross-tokenizer alignment is implemented). Use a SmolLM2 teacher
(e.g. HuggingFaceTB/SmolLM2-1.7B) — same tokenizer as the SmolLM2-135M
student vocab. A Qwen teacher would need the (unimplemented) tokenizer-switch
path and is rejected at startup.

Example (production; doc isolation is default ON, no flag needed):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \\
      experiments/gen_teacher_logits.py \\
      --teacher_model HuggingFaceTB/SmolLM2-1.7B \\
      --data_mix configs/pretrain_mix_v19_codeup.yaml \\
      --tokenizer HuggingFaceTB/SmolLM2-135M \\
      --out_dir data/teacher_logits_smollm2_1p7b --top_k 24 \\
      --max_tokens 4_000_000_000 --T 2048 --batch 8 --seed 0 --num_workers 0
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from experiments.teacher_logits_io import LogitStoreWriter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher_model", type=str, required=True,
                   help="HF model id of the teacher. MUST share the student "
                        "tokenizer (e.g. HuggingFaceTB/SmolLM2-1.7B); a "
                        "cross-tokenizer teacher is rejected at startup.")
    p.add_argument("--data_mix", type=str, required=True,
                   help="YAML data-mix config; MUST match the trainer's "
                        "--data_mix for the offline-KD run.")
    p.add_argument("--tokenizer", type=str,
                   default="HuggingFaceTB/SmolLM2-135M",
                   help="Tokenizer shared by teacher + student. MUST match the "
                        "trainer's --tokenizer.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for the sharded logit store.")
    p.add_argument("--top_k", type=int, default=24,
                   help="Number of top teacher logits stored per position.")
    p.add_argument("--max_tokens", type=int, default=1_000_000,
                   help="Stop after writing at least this many tokens (rounded "
                        "up to a whole batch).")
    p.add_argument("--T", type=int, default=2048,
                   help="Sequence length (block_size). MUST match the trainer.")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size for the teacher forward AND the stream "
                        "packing. MUST match the trainer for lockstep alignment.")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. MUST be 0 — the trainer hard-"
                        "rejects offline-KD stores generated at num_workers>=2 "
                        "(worker round-robin interleaving changes the global "
                        "token stream vs. the trainer's own num_workers=0/1 "
                        "flat stream; MixedSourceStream seeds each worker "
                        "base_seed+17*worker_id). Any non-zero value is a "
                        "startup error, not just a mismatch warning.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for MixedSourceStream. MUST match the "
                        "trainer's --seed.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for the teacher forward (cuda / cpu).")
    p.add_argument("--think_burst_prob", type=float, default=0.0,
                   help="Think-burst insertion probability. Default 0 (off) and "
                        "STRONGLY recommended off — the teacher has no think "
                        "token. If >0, the trainer MUST use the same value.")
    p.add_argument("--no_doc_isolation", action="store_true",
                   help="Disable cross-document teacher-context isolation "
                        "(default: ON). With isolation ON, the teacher is run "
                        "SEPARATELY on each document within a packed block "
                        "(matching the student's DeltaNet doc_ids state reset); "
                        "with this flag, the teacher instead sees the full "
                        "packed block as one context (the OLD, defective "
                        "behaviour) — only for reproducing old stores.")
    p.add_argument("--mask_eos_in_targets", action="store_true",
                   help="Match the trainer's --mask_eos_in_targets (does not "
                        "change the token stream; kept for parity).")
    p.add_argument("--shard_max_bytes", type=float, default=1.5 * 1024 ** 3,
                   help="Approx bytes per shard (~1-2 GB). Default 1.5 GiB.")
    p.add_argument("--teacher_dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="dtype for the teacher forward.")
    p.add_argument("--log_every", type=int, default=20,
                   help="Print progress every N batches.")
    return p


def _doc_segments(doc_ids_row) -> list[tuple[int, int]]:
    """Contiguous [start, end) index spans over a 1-D per-position doc_ids
    sequence — one span per document, in packed order. `doc_ids_row` may be a
    list or a 1-D tensor. A row with a single document (all-equal doc_ids,
    e.g. isolation disabled / a single-doc block) returns one full-length
    span, so callers can use this unconditionally."""
    ids = doc_ids_row.tolist() if torch.is_tensor(doc_ids_row) else list(doc_ids_row)
    if not ids:
        return []
    spans = []
    start = 0
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            spans.append((start, i))
            start = i
    spans.append((start, len(ids)))
    return spans


@torch.no_grad()
def teacher_topk_doc_isolated(teacher, x: torch.Tensor, doc_ids: torch.Tensor,
                              top_k: int, teacher_vocab: int):
    """Run `teacher` on each row of `x` (B, T) SEPARATELY per document (per
    `doc_ids`, a same-shape tensor of per-position document ids) instead of
    over the full packed block — so no teacher target is conditioned on
    context that crosses a document boundary the student's DeltaNet state
    resets at. Positions restart at 0 within each document segment. See the
    module docstring's "CROSS-DOCUMENT STATE ISOLATION" section for the full
    rationale and the boundary convention at each document's last position.

    Returns `(topv, topi)`, each shaped (B, T, top_k) in PACKED order — i.e.
    identical shape/order to a plain `torch.topk(teacher(x).logits, top_k)`,
    just computed per-document instead of over the whole row.
    """
    B, T = x.shape
    x_teacher = x.clamp(max=teacher_vocab - 1)
    topv = torch.zeros(B, T, top_k, dtype=torch.float32, device=x.device)
    topi = torch.zeros(B, T, top_k, dtype=torch.long, device=x.device)
    for b in range(B):
        for s0, s1 in _doc_segments(doc_ids[b]):
            seg = x_teacher[b, s0:s1].unsqueeze(0)          # (1, seg_len)
            seg_out = teacher(input_ids=seg)
            seg_logits = getattr(seg_out, "logits", seg_out)  # (1, seg_len, Vt)
            v, i = torch.topk(seg_logits.float(), top_k, dim=-1)
            topv[b, s0:s1] = v[0]
            topi[b, s0:s1] = i[0]
    return topv, topi


def main():
    args = build_parser().parse_args()
    if int(args.num_workers) != 0:
        raise SystemExit(
            f"--num_workers must be 0 (got {args.num_workers}): the offline-KD "
            "store must be a batch-independent, worker-independent flat token "
            "stream, and train_lm.py only accepts a store generated at "
            "num_workers 0 (or 1, which coincides with 0 — see the module "
            "docstring) under --distill_logits_dir.")
    doc_isolation = not args.no_doc_isolation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.data_mix import MixedSourceStream, load_sources_from_yaml

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    base_vocab = int(tok.vocab_size)
    # Reserve one slot above the base vocab for the think token — IDENTICAL to
    # train_lm.py so the packed stream (and any inserted think ids) match exactly.
    thinking_token_id = base_vocab

    # FAIL FAST on a cross-tokenizer teacher. We feed the STUDENT tokenizer's ids
    # straight into the teacher forward with NO cross-tokenizer alignment, so a
    # teacher whose tokenizer differs from --tokenizer would produce meaningless
    # logits over the wrong vocab AND a store the trainer later rejects — and the
    # mistake is only discovered after the WHOLE (multi-day) generation. Check the
    # teacher's own tokenizer up front, before loading the (large) teacher model.
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model)
    if int(teacher_tok.vocab_size) != base_vocab:
        raise SystemExit(
            f"teacher tokenizer (vocab {int(teacher_tok.vocab_size)}) != student "
            f"tokenizer (vocab {base_vocab}); offline KD requires a "
            "SHARED-tokenizer teacher (e.g. HuggingFaceTB/SmolLM2-1.7B). A "
            "cross-tokenizer (e.g. Qwen) teacher needs the tokenizer-switch "
            "path, not yet implemented."
        )
    t_name = getattr(teacher_tok, "name_or_path", "")
    s_name = getattr(tok, "name_or_path", "")
    if t_name and s_name and t_name != s_name:
        print(f"  NOTE: teacher tokenizer '{t_name}' and student tokenizer "
              f"'{s_name}' have matching vocab size ({base_vocab}) but different "
              "names — assuming they are the same tokenizer family (e.g. SmolLM2 "
              "checkpoints). Verify if unsure.")

    sources = load_sources_from_yaml(args.data_mix)
    print(f"Teacher-logit gen: teacher={args.teacher_model}  "
          f"tokenizer={args.tokenizer}  vocab={base_vocab}  top_k={args.top_k}")
    print(f"  data_mix={args.data_mix}  T={args.T}  batch={args.batch}  "
          f"num_workers={args.num_workers}  seed={args.seed}  "
          f"think_burst_prob={args.think_burst_prob}")
    if doc_isolation:
        print("  doc isolation: ON — teacher conditioned on same-document "
              "prefix only (matches the student's DeltaNet cross-doc state "
              "reset); see module docstring.")
    else:
        print("  doc isolation: OFF (--no_doc_isolation) — teacher conditioned "
              "on the FULL packed block, including OTHER documents' content. "
              "This reproduces the pre-fix behaviour; the resulting store's KD "
              "targets past the first document in a block are conditioned on "
              "context the student cannot see. Only use this to reproduce old "
              "stores.")
    for s in sources:
        print(f"    - {s.name:24s} weight={s.weight:.3f}")

    # MUST mirror train_lm.py's MixedSourceStream construction (same mix / seed
    # / T / think-burst). `emit_doc_ids` does NOT change the token stream (it
    # only attaches an aligned id channel that reuses the same RNG draws), so
    # the trainer's input_ids alignment assertion is unaffected either way —
    # but the doc_ids channel IS what drives per-document teacher isolation
    # below, so we request it whenever doc_isolation is on (the default).
    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.T,
        thinking_token_id=thinking_token_id,
        think_burst_prob=args.think_burst_prob,
        base_seed=args.seed,
        mask_eos_in_targets=bool(args.mask_eos_in_targets),
        emit_doc_ids=doc_isolation,
    )
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.teacher_dtype]
    print(f"Loading teacher {args.teacher_model} ({args.teacher_dtype}) ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=dtype)
    teacher.to(args.device).eval()
    teacher_vocab = int(teacher.get_output_embeddings().weight.shape[0])
    tp = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"  {tp:.0f}M params, teacher vocab={teacher_vocab}")
    if args.top_k > teacher_vocab:
        raise SystemExit(f"--top_k {args.top_k} > teacher vocab {teacher_vocab}")

    writer = LogitStoreWriter(
        args.out_dir, k=args.top_k, vocab_size=teacher_vocab,
        teacher_model=args.teacher_model, tokenizer_name=args.tokenizer,
        shard_max_bytes=args.shard_max_bytes)

    written = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch[0]                                   # (B, T) — the inputs
            x = x.to(args.device)
            B, T = x.shape
            # The teacher has no think token; clamp any inserted think id (==
            # base_vocab, OOB for the teacher) to a valid id for the forward.
            # The REAL ids (with think token) are still stored, so the trainer's
            # alignment check sees the exact stream; think positions are excluded
            # from KD by the trainer's valid mask. (teacher_topk_doc_isolated
            # does the equivalent clamp internally for the doc_isolation path.)
            if doc_isolation:
                doc_ids = batch[2].to(args.device)          # (B, T)
                topv, topi = teacher_topk_doc_isolated(
                    teacher, x, doc_ids, args.top_k, teacher_vocab)
            else:
                x_teacher = x.clamp(max=teacher_vocab - 1)
                out = teacher(input_ids=x_teacher)
                logits = getattr(out, "logits", out)           # (B, T, Vt)
                topv, topi = torch.topk(logits.float(), args.top_k, dim=-1)
            writer.append(
                ids=topi.reshape(-1, args.top_k),
                logits=topv.reshape(-1, args.top_k),
                input_ids=x.reshape(-1))
            written += B * T
            if bi % args.log_every == 0:
                rate = written / max(1e-6, time.time() - t0)
                print(f"  batch {bi:>6d}  tokens={written:,}  "
                      f"{rate:,.0f} tok/s")
            if written >= args.max_tokens:
                break

    manifest = writer.close()
    dt = time.time() - t0
    print(f"Done: wrote {manifest['total_tokens']:,} tokens across "
          f"{len(manifest['shards'])} shard(s) to {args.out_dir} in {dt:.1f}s")
    print(f"  manifest: k={manifest['k']} vocab={manifest['vocab_size']} "
          f"dtype={manifest['dtype']}")


if __name__ == "__main__":
    main()
