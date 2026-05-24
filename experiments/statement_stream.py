"""
Statement-aware streaming dataset for codeparrot.

Yields (input_ids, target_ids, stmt_starts, stmt_ends) tuples where the
last two are 1D LongTensors describing the token ranges (within the
input_ids slice) of the parseable statements that fully fit inside this
chunk.

Rationale: in the structural-surprise loss training (Phase 2), we need
per-batch knowledge of which token positions belong to which statement,
so we can pool the model's hidden state over each statement and compute
L_sem(s_t). Doing this with on-the-fly AST parsing of decoded chunks is
brittle because the chunk often straddles file boundaries (mixed-syntax
text, AST fails). Doing it at the dataloader level — where we still have
each file's raw text — is cleaner.

Implementation:
- For each streamed file, parse statements with `experiments.statement_segmentation`.
- Walk the statements with a running "current chunk start" cursor in the
  global token buffer. When a chunk fills, slice out
  (input_ids, target_ids), and project each statement's [start, end)
  range from "global" to "chunk-local" coordinates, dropping statements
  that don't fully fall inside the chunk.
- We deliberately track statements only on the *next-token-prediction*
  positions (i.e. positions [start, end) where the model receives the
  inputs and should emit targets). A statement that crosses an EOS
  boundary is dropped because its semantics break across files.

Padding: stmt_starts and stmt_ends are emitted as variable-length 1D
tensors. The trainer pads them to a fixed `max_stmts_per_chunk` (with
sentinel -1) before batching.
"""
from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from experiments.statement_segmentation import parse_file_statements


class StatementAwareTokenisedStream(IterableDataset):
    """Streaming IterableDataset that yields (inputs, targets, stmt_starts,
    stmt_ends) tuples per chunk. stmt_starts/ends are 1D LongTensors of
    length n_stmts_in_chunk (variable per chunk).

    `max_stmts_per_chunk` is used by the collator (custom collate_fn in
    the trainer) — at the dataset level we just emit the raw lists.
    """

    def __init__(self, dataset, tokenizer, block_size,
                 text_field: str = "content",
                 max_stmts_per_chunk: int = 64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.max_stmts_per_chunk = max_stmts_per_chunk

    def __iter__(self) -> Iterator[tuple]:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.bos_token_id
        if eos is None:
            eos = 0
        # Global token buffer. Each entry is an int.
        buf: list[int] = []
        # Statement boundaries in *buffer-global* coords. Each entry is
        # (start_in_buf, end_in_buf, file_idx) — the file_idx is for
        # debugging / tracing (not consumed downstream).
        buf_stmts: list[tuple[int, int, int]] = []
        # Where the current file started in the buffer — needed because
        # each file's local token offsets become global by adding this.
        cur_file_start = 0
        file_idx = 0

        for example in self.dataset:
            text = example.get(self.text_field, "")
            if not text:
                continue
            # Parse the file's statements. parse_file_statements returns
            # token_ids + statements with token offsets. We use those
            # offsets directly (matches the encode that follows since we
            # tokenize the same text without special tokens).
            tokens, stmts = parse_file_statements(text, self.tokenizer)
            # Append to buffer with file-global offsets translated.
            cur_file_start = len(buf)
            buf.extend(tokens)
            buf.append(eos)
            for s in stmts:
                # The end_tok_idx must not exceed len(tokens).
                if s.end_tok_idx > len(tokens):
                    continue
                buf_stmts.append(
                    (cur_file_start + s.start_tok_idx,
                     cur_file_start + s.end_tok_idx,
                     file_idx)
                )
            file_idx += 1
            # Emit chunks while buffer is large enough.
            while len(buf) >= self.block_size + 1:
                chunk = buf[: self.block_size + 1]
                # The chunk spans buf indices [0, block_size+1).
                # Inputs are positions 0..T-1, targets are 1..T.
                inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                # Find statements that fully fit within [0, T-1) — we use
                # T-1 because the per-statement loss uses positions [start,
                # end) where the next-token target is at [start+1, end+1)
                # so end must be <= T-1 for the target to exist.
                stmt_starts = []
                stmt_ends = []
                for st, en, _ in buf_stmts:
                    if st < 0 or en <= st:
                        continue
                    # Within this chunk's input range [0, T-1) (we want
                    # full statements with valid targets, so en must be
                    # <= block_size which equals T).
                    if st >= 0 and en <= self.block_size:
                        stmt_starts.append(st)
                        stmt_ends.append(en)
                # Cap to max_stmts_per_chunk.
                if len(stmt_starts) > self.max_stmts_per_chunk:
                    # Keep first N (chronological) — could also random-sample.
                    stmt_starts = stmt_starts[:self.max_stmts_per_chunk]
                    stmt_ends = stmt_ends[:self.max_stmts_per_chunk]
                ss = torch.tensor(stmt_starts, dtype=torch.long)
                se = torch.tensor(stmt_ends, dtype=torch.long)
                yield inputs, targets, ss, se
                # Slide buffer.
                buf = buf[self.block_size :]
                # Shift statement coordinates back by block_size (those
                # that fall negative are dropped); also drop statements
                # whose start is now past the kept buffer (they were fully
                # consumed in the prior chunk).
                new_stmts: list[tuple[int, int, int]] = []
                for st, en, fi in buf_stmts:
                    new_st = st - self.block_size
                    new_en = en - self.block_size
                    # Keep statements that fully or partially overlap the
                    # remaining buffer (st >= 0 in buffer-local coords
                    # equivalent to original st >= block_size). End
                    # remains an exclusive bound; if new_en <= 0 then
                    # the statement is wholly in the previous chunk.
                    if new_en <= 0:
                        continue
                    # Also drop statements that start before the new
                    # start (i.e. originally crossed the chunk boundary):
                    # we don't want partial statements.
                    if new_st < 0:
                        continue
                    new_stmts.append((new_st, new_en, fi))
                buf_stmts = new_stmts


def collate_with_statements(batch: list, pad_to: int = 64,
                              pad_value: int = -1) -> tuple:
    """Custom collator that pads variable-length statement-range arrays.

    Inputs:
        batch: list of (inputs, targets, stmt_starts, stmt_ends)
    Returns:
        (inputs_b, targets_b, stmt_starts_b, stmt_ends_b)
        stmt_starts_b, stmt_ends_b are (B, pad_to) with `pad_value` (-1)
        marking unused slots. Real entries are positive integers.
    """
    inputs_list = []
    targets_list = []
    starts_list = []
    ends_list = []
    for inp, tgt, ss, se in batch:
        inputs_list.append(inp)
        targets_list.append(tgt)
        starts_list.append(ss)
        ends_list.append(se)
    inputs_b = torch.stack(inputs_list, dim=0)
    targets_b = torch.stack(targets_list, dim=0)
    B = inputs_b.shape[0]
    starts_b = torch.full((B, pad_to), pad_value, dtype=torch.long)
    ends_b = torch.full((B, pad_to), pad_value, dtype=torch.long)
    for b in range(B):
        n = min(starts_list[b].numel(), pad_to)
        if n > 0:
            starts_b[b, :n] = starts_list[b][:n]
            ends_b[b, :n] = ends_list[b][:n]
    return inputs_b, targets_b, starts_b, ends_b
