"""Offline teacher top-k logit store via **vLLM** — the production gen path
(#104/#106), the fast/AWQ counterpart to the HF reference `gen_teacher_logits.py`.

Same store format (`teacher_logits_io.LogitStoreWriter`) and the SAME
stream-alignment contract the trainer enforces, but the teacher runs under vLLM
`prompt_logprobs` (AWQ-quantized, ~10k tok/s on a 5090 for Qwen-Coder-7B) instead
of an HF eager forward. Run under the vLLM venv:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    .venv-vllm/bin/python experiments/gen_teacher_logits_vllm.py \\
      --teacher_model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \\
      --data_mix configs/pretrain_mix_v18_arxiv.yaml \\
      --tokenizer Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \\
      --out_dir data/teacher_logits_qwen_coder7b --top_k 16 --max_tokens 1_000_000_000

ALIGNMENT (load-bearing) — the trainer reads this store in LOCKSTEP with its data
iterator and asserts stored input_ids == live tokens. We GUARANTEE a match by
reusing the IDENTICAL `MixedSourceStream` (same mix/tokenizer/seed/T, think-burst
OFF) at **num_workers=0**, which makes the flat token order batch-INDEPENDENT, and
by feeding vLLM the packed token-ids straight (TokensPrompt) — NO re-tokenization.
So STEP-2 training MUST use the same --data_mix/--tokenizer/--seed/--T and
--think_burst_prob 0 --num_workers 0. (batch/grad_accum may differ.)

OFF-BY-ONE (verified against the vLLM API): vLLM `prompt_logprobs[t]` is
P(x[t] | x[0..t-1]) with [0]=None; the trainer's store convention (matching the HF
forward `logits[:, t, :]`) is store[t] = P(x[t+1] | x[0..t]). Therefore
  store[t]   = prompt_logprobs[t+1]      for t in [0, T-2]
  store[T-1] = outputs[0].logprobs[0]    (P of the token AFTER the block)
We request `--vllm_top_logprobs` (>= top_k) and use max_tokens=1 so the generated
slot supplies that last distribution. Under doc isolation (below) this convention
applies PER DOCUMENT SEGMENT, not per packed block — see that section.

VOCAB FILTER (fixed 2026-07-01 — see "vocab-filter boundary" below): the filter
keeps teacher token ids `tid < real_vocab` where `real_vocab = len(tok)` is the
tokenizer's REAL vocabulary. Teacher top-k ids >= real_vocab are padding /
reserved-special-token rows (near-zero prob, rarely in top-k) OR the student's
[THINKING] slot itself (id == thinking_token_id, which sits at exactly
`len(tok)` — one past the last real id, IDENTICAL construction to train_lm.py);
none of those are a meaningful "teacher prediction" and must never be stored as
one. We DROP them so the trainer's gather(student_logits, teacher_ids) is always
in range AND never distils toward a padding/think slot. The store's declared
`vocab_size` is still the student's round64-PADDED model vocab (`student_vocab`)
— that's the SLOT-LAYOUT bound the trainer's `vocab_size <= model_vocab_size`
sanity check compares against, a looser (and always-satisfied) superset of the
`real_vocab` filter bound actually used above.

  Prior bug: the filter used `student_vocab` (151680, the round64-PADDED size)
  as the boundary instead of `len(tok)` (151665 for the Qwen tokenizer in use),
  so teacher lm_head PADDING rows in [151665, 151680) — including id 151665,
  which IS the student's [THINKING] slot — could be kept as a stored "top-k
  prediction". Any position where the teacher's padding-row logit happened to
  be large would silently train the student to predict a padding id, or worse,
  to predict [THINKING] the way a real token is predicted (thinking-token
  targets are separately masked to -100 in `y`, but not in the offline-KD
  target — the KD loss doesn't consult `y` for its own valid-id set beyond the
  `y != thinking_token_id` check in `train_lm.py`, only the `tid < vocab_size`
  gather-range bound, which the old `student_vocab` boundary satisfied for id
  151665 without excluding it).

logprob-as-logit: vLLM returns log-softmax logprobs. Top-k, temperature-scaled KD
is shift-invariant — softmax(logprob_topk / T) == softmax(logit_topk / T) (the
per-position logsumexp constant cancels) — so storing logprobs in the 'logits'
field is EXACT for the trainer's `_kd_loss_term_topk` math.

CROSS-DOCUMENT STATE ISOLATION (default ON) — same fix + rationale as
`gen_teacher_logits.py`'s "CROSS-DOCUMENT STATE ISOLATION" docstring section;
read that first. Here the mechanism is: stream with `emit_doc_ids=True` (an
aligned per-position document-id channel; does NOT change the token stream),
split each packed row into per-document `[s0, s1)` spans via `_doc_segments`,
and submit EACH SEGMENT as its OWN `TokensPrompt` (instead of one TokensPrompt
per full T-length row) — so vLLM's `prompt_logprobs` restart at position 0 for
every document, exactly like the student's DeltaNet state reset. Results are
stitched back into the packed `(B, T, top_k)` layout in the original order, so
the store format/alignment is unchanged. The OFF-BY-ONE convention above still
applies, just computed per document segment: `store[global t] =
prompt_logprobs[local_j + 1]` for `local_j < seg_len - 1`, and `= outputs[0].
logprobs[0]` at `local_j == seg_len - 1` (the document's last position) — i.e.
"the token after this segment", which for a non-final document is NOT the
packed stream's actual next token (that belongs to a different, unrelated
document). This is the same intentional boundary convention documented in
`gen_teacher_logits.py`: `train_lm.py` masks KD loss at any position whose
target crosses a document boundary, using its own `doc_ids`, so this row is
simply never used as a KD target on the consumer side. `--no_doc_isolation`
reproduces the old (single TokensPrompt per full row) behaviour.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.teacher_logits_io import LogitStoreWriter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher_model", type=str, required=True,
                   help="vLLM model id of the teacher (e.g. "
                        "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ).")
    p.add_argument("--data_mix", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True,
                   help="Student tokenizer == teacher tokenizer (shared-vocab "
                        "KD). MUST match the trainer's --tokenizer.")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--top_k", type=int, default=16,
                   help="Top-k teacher entries stored per position.")
    p.add_argument("--vllm_top_logprobs", type=int, default=20,
                   help="How many logprobs to request from vLLM per position "
                        "(>= top_k; the surplus covers vocab-filtered ids). "
                        "vLLM caps prompt_logprobs at 20.")
    p.add_argument("--max_tokens", type=int, default=1_000_000,
                   help="Stop after writing >= this many tokens (whole batches).")
    p.add_argument("--T", type=int, default=2048,
                   help="Block size. MUST match the trainer.")
    p.add_argument("--batch", type=int, default=48,
                   help="Sequences fed to vLLM per generate() call. Throughput "
                        "knob only; the flat token order is batch-independent at "
                        "num_workers=0, so this need NOT match the trainer.")
    p.add_argument("--num_workers", type=int, default=0,
                   help="MUST be 0 (batch-independent deterministic stream). A "
                        "non-zero value is rejected.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for MixedSourceStream. MUST match the "
                        "trainer's --seed (single-GPU offline KD => ddp_rank 0).")
    p.add_argument("--think_burst_prob", type=float, default=0.0,
                   help="MUST be 0 for offline KD (think bursts shift the "
                        "stream). Rejected if non-zero.")
    p.add_argument("--no_doc_isolation", action="store_true",
                   help="Disable cross-document teacher-context isolation "
                        "(default: ON). With isolation ON, each document "
                        "within a packed block is submitted to vLLM as its "
                        "OWN TokensPrompt (positions restart at 0 per doc), "
                        "matching the student's DeltaNet doc_ids state reset; "
                        "with this flag, one TokensPrompt per full packed row "
                        "is used instead (the OLD, defective behaviour) — only "
                        "for reproducing old stores.")
    p.add_argument("--mask_eos_in_targets", action="store_true",
                   help="Match the trainer (does not change the token stream).")
    p.add_argument("--shard_max_bytes", type=float, default=1.5 * 1024 ** 3)
    p.add_argument("--max_model_len", type=int, default=4096,
                   help=">= T + 1 (prompt + the 1 generated slot).")
    p.add_argument("--gpu_mem_fraction", type=float, default=0.9)
    p.add_argument("--max_num_seqs", type=int, default=48)
    p.add_argument("--log_every", type=int, default=20)
    return p


def _topk_row(src: dict, top_k: int, vocab_bound: int):
    """src: {token_id: Logprob(logprob, rank, ...)} from vLLM. Return
    (ids[top_k] int64, logps[top_k] float32) sorted by rank, restricted to
    ids < vocab_bound, padded (id repeat + very-low logprob) if fewer than
    top_k survive the filter.

    `vocab_bound` MUST be the teacher's REAL vocabulary (`len(tok)`), NOT the
    student's round64-padded model vocab — see the module docstring's "VOCAB
    FILTER" section. A too-large bound lets teacher lm_head padding rows (and
    the student's own [THINKING] slot, which sits at id == len(tok)) through
    as if they were real top-k predictions."""
    # vLLM orders by rank; sort defensively.
    items = sorted(src.items(), key=lambda kv: kv[1].rank)
    ids = []
    lps = []
    for tid, lp in items:
        if tid < vocab_bound:
            ids.append(tid)
            lps.append(lp.logprob)
            if len(ids) == top_k:
                break
    if not ids:                       # impossible in practice (>=1 valid id)
        ids = [0]
        lps = [-30.0]
    while len(ids) < top_k:           # pad: duplicate last id, ~0 softmax weight
        ids.append(ids[-1])
        lps.append(-30.0)
    return ids, lps


def _doc_segments(doc_ids_row) -> list[tuple[int, int]]:
    """Contiguous [start, end) index spans over a 1-D per-position doc_ids
    sequence — one span per document, in packed order. Mirrors
    `gen_teacher_logits.py::_doc_segments` (kept duplicated rather than
    imported so this file's only dependency at module-import time stays
    numpy — no torch/vllm needed to load `_topk_row`/`_doc_segments` for
    testing). `doc_ids_row` may be a list or a 1-D numpy array. A row with a
    single document (all-equal doc_ids) returns one full-length span."""
    ids = doc_ids_row.tolist() if hasattr(doc_ids_row, "tolist") else list(doc_ids_row)
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


def main():
    args = build_parser().parse_args()
    if int(args.num_workers) != 0:
        raise SystemExit("--num_workers must be 0 (batch-independent stream).")
    if float(args.think_burst_prob) != 0.0:
        raise SystemExit("--think_burst_prob must be 0 for offline KD.")
    if args.vllm_top_logprobs < args.top_k:
        raise SystemExit("--vllm_top_logprobs must be >= --top_k")
    doc_isolation = not args.no_doc_isolation

    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from experiments.data_mix import MixedSourceStream, load_sources_from_yaml

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Mirror train_lm.py EXACTLY: think id is the first slot above every real
    # token; student model vocab rounds that up to a multiple of 64.
    thinking_token_id = int(max(tok.vocab_size, len(tok)))
    student_vocab = ((thinking_token_id + 1 + 63) // 64) * 64
    # real_vocab: the FILTER boundary for teacher top-k ids (see the module
    # docstring's "VOCAB FILTER" section) — NOT student_vocab, which is the
    # round64-padded model-vocab used only for the manifest's declared
    # vocab_size / the trainer's <= model_vocab_size sanity check. For every
    # tokenizer we use, thinking_token_id is placed at exactly len(tok) (one
    # past the last real id — the "max(vocab_size, len(tok))" above is >=
    # len(tok) by construction, and equals it whenever, as for Qwen,
    # tok.vocab_size names a real/special token below len(tok)), so filtering
    # at len(tok) is precisely "exclude the [THINKING] slot and everything
    # beyond it" — kept as its own variable for readability at each call site.
    real_vocab = int(len(tok))

    sources = load_sources_from_yaml(args.data_mix)
    print(f"vLLM teacher-logit gen: teacher={args.teacher_model}")
    print(f"  tokenizer={args.tokenizer} base_vocab={tok.vocab_size} "
          f"len={len(tok)} think_id={thinking_token_id} student_vocab={student_vocab} "
          f"real_vocab(filter)={real_vocab}")
    print(f"  data_mix={args.data_mix} T={args.T} seed={args.seed} "
          f"top_k={args.top_k} vllm_top_logprobs={args.vllm_top_logprobs} "
          f"think_burst={args.think_burst_prob} num_workers=0")
    if doc_isolation:
        print("  doc isolation: ON — each document within a packed block is "
              "its own TokensPrompt (positions restart at 0 per doc), matching "
              "the student's DeltaNet cross-doc state reset.")
    else:
        print("  doc isolation: OFF (--no_doc_isolation) — one TokensPrompt "
              "per full packed row; teacher conditioned on OTHER documents' "
              "content past the first doc in a block. Pre-fix behaviour; only "
              "use to reproduce old stores.")
    for s in sources:
        print(f"    - {s.name:28s} weight={s.weight:.3f}")

    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.T,
        thinking_token_id=thinking_token_id,
        think_burst_prob=args.think_burst_prob,
        base_seed=args.seed,
        mask_eos_in_targets=bool(args.mask_eos_in_targets),
        emit_doc_ids=doc_isolation,
    )
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    print(f"Loading teacher via vLLM ({args.teacher_model}) ...")
    llm = LLM(model=args.teacher_model, quantization="awq",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_fraction,
              trust_remote_code=True, dtype="float16",
              enforce_eager=True, max_num_seqs=args.max_num_seqs)
    sampling = SamplingParams(max_tokens=1, temperature=0.0,
                              prompt_logprobs=args.vllm_top_logprobs,
                              logprobs=args.vllm_top_logprobs)

    writer = LogitStoreWriter(
        args.out_dir, k=args.top_k, vocab_size=student_vocab,
        teacher_model=args.teacher_model, tokenizer_name=args.tokenizer,
        shard_max_bytes=args.shard_max_bytes)

    written = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch  # (B,T)
        x_np = x.cpu().numpy().astype(np.int64)
        B, T = x_np.shape

        # seg_specs[i] = (row b, [s0, s1) span) for prompts[i]. Doc isolation
        # ON: one TokensPrompt PER DOCUMENT (positions restart at 0 per doc —
        # see the module docstring's "CROSS-DOCUMENT STATE ISOLATION"
        # section). Doc isolation OFF: one span per row == the old behaviour.
        if doc_isolation:
            doc_ids_np = (batch[2] if isinstance(batch, (list, tuple))
                          else None)
            if doc_ids_np is None:
                raise RuntimeError(
                    "doc_isolation is on but the stream did not yield "
                    "doc_ids — MixedSourceStream(emit_doc_ids=True) wiring "
                    "bug.")
            doc_ids_np = doc_ids_np.cpu().numpy()
            seg_specs = [(b, s0, s1) for b in range(B)
                        for s0, s1 in _doc_segments(doc_ids_np[b])]
        else:
            seg_specs = [(b, 0, T) for b in range(B)]
        prompts = [TokensPrompt(prompt_token_ids=x_np[b, s0:s1].tolist())
                  for b, s0, s1 in seg_specs]
        outs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)

        batch_ids = np.zeros((B, T, args.top_k), dtype=np.int64)
        batch_lps = np.zeros((B, T, args.top_k), dtype=np.float32)
        for (b, s0, s1), o in zip(seg_specs, outs):
            seg_len = s1 - s0
            pl = o.prompt_logprobs          # len seg_len, [0]=None
            og = o.outputs[0].logprobs      # len 1
            for j in range(seg_len):
                src = pl[j + 1] if j < seg_len - 1 else og[0]
                ids, lps = _topk_row(src, args.top_k, real_vocab)
                batch_ids[b, s0 + j] = ids
                batch_lps[b, s0 + j] = lps

        writer.append(
            ids=batch_ids.reshape(-1, args.top_k),
            logits=batch_lps.reshape(-1, args.top_k),
            input_ids=x_np.reshape(-1))
        written += B * T
        if bi % args.log_every == 0:
            rate = written / max(1e-6, time.time() - t0)
            print(f"  batch {bi:>6d}  tokens={written:,}  {rate:,.0f} tok/s",
                  flush=True)
        if written >= args.max_tokens:
            break

    manifest = writer.close()
    dt = time.time() - t0
    print(f"Done: wrote {manifest['total_tokens']:,} tokens across "
          f"{len(manifest['shards'])} shard(s) to {args.out_dir} in {dt:.1f}s "
          f"({written/max(1e-6,dt):,.0f} tok/s)")
    print(f"  manifest: k={manifest['k']} vocab={manifest['vocab_size']} "
          f"dtype={manifest['dtype']}")


if __name__ == "__main__":
    main()
