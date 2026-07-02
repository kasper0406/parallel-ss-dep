"""Sharded top-k teacher-logit store for OFFLINE knowledge distillation.

Task #104. This is the storage format that lets a big teacher's top-k logits be
precomputed ONCE to disk (`gen_teacher_logits.py`) and then read by the student
trainer (`train_lm.py --distill_logits_dir`) with NO teacher in the training
loop.

Storage schema
--------------
A "store" is a directory containing:

  * one or more shard files ``shard_00000.safetensors`` (~1-2 GB each), each a
    safetensors file with three tensors covering a contiguous range of the
    global token stream:

        ids        uint32  [n, k]   teacher's top-k token ids at each position
        logits     fp16    [n, k]   RAW top-k logits (NOT softmaxed) — keeping
                                     them raw preserves temperature flexibility
                                     at training time.
        input_ids  uint32  [n]      the ACTUAL token at each position (i.e. the
                                     trainer's flattened input id). Stored for
                                     (a) alignment verification — the load-bearing
                                     safety check in the offline-KD path — and
                                     (b) it doubles as the CE target reference.

  * ``manifest.json`` at the store root::

        {
          "k": 24,
          "vocab_size": 49152,
          "teacher_model": "Qwen/Qwen3-Coder-30B",
          "tokenizer_name": "HuggingFaceTB/SmolLM2-135M",
          "total_tokens": 10485760,
          "dtype": "float16",
          "shards": [
            {"name": "shard_00000.safetensors", "start": 0, "end": 5242880,
             "n": 5242880},
            ...
          ]
        }

    ``start``/``end`` are GLOBAL token indices, half-open ``[start, end)``;
    consecutive shards tile ``[0, total_tokens)`` with no gaps/overlap.

Row order == training token order
---------------------------------
Row ``j`` of the store corresponds to the ``j``-th token the trainer sees when
its data iterator is flattened in stream order (each batch ``x`` of shape
``(B, T)`` is flattened row-major to ``B*T`` tokens). The generator
(`gen_teacher_logits.py`) MUST stream the identical packed token stream the
trainer uses (same data_mix / seed / T / batch / think-burst settings). The
trainer asserts ``input_ids`` match per block, so any desync is caught loudly
rather than silently corrupting KD.

dtypes
------
* ``logits`` are fp16 (half the bytes of fp32, plenty for a KL target).
* ``ids`` / ``input_ids`` are uint32 (vocab < 2^32). torch lacks many uint32
  ops, so the reader casts them to int64 on the way out for indexing/equality;
  storage stays uint32.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import torch
from safetensors.torch import safe_open, save_file


# Bytes per stored token = k * (2 bytes fp16 logit + 4 bytes uint32 id)
#                          + 4 bytes uint32 input_id.
def _bytes_per_token(k: int) -> int:
    return k * (2 + 4) + 4


def _default_shard_max_tokens(k: int, shard_max_bytes: float) -> int:
    """How many tokens fit in a ~`shard_max_bytes` shard at top-k = k."""
    return max(1, int(shard_max_bytes // _bytes_per_token(k)))


def _shard_name(idx: int) -> str:
    return f"shard_{idx:05d}.safetensors"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class LogitStoreWriter:
    """Append-and-shard writer for a teacher top-k logit store.

    Usage::

        w = LogitStoreWriter(out_dir, k=24, vocab_size=49152,
                             teacher_model="...", tokenizer_name="...")
        for batch in stream:
            w.append(ids, logits, input_ids)   # each [n, k] / [n, k] / [n]
        w.close()                              # flushes tail shard + manifest

    Shards are flushed automatically once the buffer reaches
    ``shard_max_tokens`` (derived from ``shard_max_bytes``, default ~1.5 GB).
    """

    def __init__(self,
                 out_dir: str,
                 k: int,
                 vocab_size: int,
                 teacher_model: str = "",
                 tokenizer_name: str = "",
                 dtype: str = "float16",
                 shard_max_bytes: float = 1.5 * 1024 ** 3,
                 shard_max_tokens: Optional[int] = None):
        if dtype != "float16":
            raise ValueError("only dtype='float16' is supported for logits")
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.k = int(k)
        self.vocab_size = int(vocab_size)
        self.teacher_model = str(teacher_model)
        self.tokenizer_name = str(tokenizer_name)
        self.dtype = dtype
        self.shard_max_tokens = int(
            shard_max_tokens if shard_max_tokens is not None
            else _default_shard_max_tokens(self.k, shard_max_bytes))
        if self.shard_max_tokens < 1:
            raise ValueError("shard_max_tokens must be >= 1")

        self._buf_ids: list[torch.Tensor] = []
        self._buf_logits: list[torch.Tensor] = []
        self._buf_input_ids: list[torch.Tensor] = []
        self._buf_tokens = 0          # tokens currently buffered (un-flushed)
        self._total_tokens = 0        # global tokens written + buffered
        self._shards: list[dict] = []  # manifest shard records (flushed only)
        self._shard_idx = 0
        self._closed = False

    # -- public -----------------------------------------------------------
    def append(self, ids, logits, input_ids) -> None:
        """Buffer one block of `n` token rows.

        ids:        [n, k] integer token ids
        logits:     [n, k] float top-k logits (cast to fp16 on store)
        input_ids:  [n]    integer actual-token ids
        """
        if self._closed:
            raise RuntimeError("append() after close()")
        ids = torch.as_tensor(ids)
        logits = torch.as_tensor(logits)
        input_ids = torch.as_tensor(input_ids)
        if ids.ndim != 2 or ids.shape[1] != self.k:
            raise ValueError(f"ids must be [n, {self.k}], got {tuple(ids.shape)}")
        if logits.shape != ids.shape:
            raise ValueError(
                f"logits {tuple(logits.shape)} must match ids {tuple(ids.shape)}")
        n = ids.shape[0]
        if input_ids.ndim != 1 or input_ids.shape[0] != n:
            raise ValueError(
                f"input_ids must be [{n}], got {tuple(input_ids.shape)}")
        # Coerce to storage dtypes (detach + cpu; uint32 / fp16).
        self._buf_ids.append(ids.detach().cpu().to(torch.int64).to(torch.uint32))
        # Clamp to the fp16 finite range BEFORE the cast — a pathologically large
        # teacher logit (|x| > 65504) would otherwise become +/-inf in fp16 and
        # corrupt the stored top-k distribution. fp32 values within range are
        # unchanged.
        self._buf_logits.append(
            logits.detach().cpu().float().clamp_(-65504.0, 65504.0)
            .to(torch.float16))
        self._buf_input_ids.append(
            input_ids.detach().cpu().to(torch.int64).to(torch.uint32))
        self._buf_tokens += n
        self._total_tokens += n
        while self._buf_tokens >= self.shard_max_tokens:
            self._flush_shard(self.shard_max_tokens)

    def close(self) -> dict:
        """Flush any buffered tail into a final shard and write the manifest.

        Returns the manifest dict (also written to ``manifest.json``)."""
        if self._closed:
            raise RuntimeError("close() called twice")
        if self._buf_tokens > 0:
            self._flush_shard(self._buf_tokens)
        manifest = {
            "k": self.k,
            "vocab_size": self.vocab_size,
            "teacher_model": self.teacher_model,
            "tokenizer_name": self.tokenizer_name,
            "total_tokens": self._total_tokens,
            "dtype": self.dtype,
            "shards": self._shards,
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        self._closed = True
        return manifest

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._closed:
            self.close()

    # -- internal ---------------------------------------------------------
    def _flush_shard(self, n_take: int) -> None:
        """Write the first `n_take` buffered tokens to a new shard file."""
        ids = torch.cat(self._buf_ids, dim=0)
        logits = torch.cat(self._buf_logits, dim=0)
        input_ids = torch.cat(self._buf_input_ids, dim=0)
        take_ids = ids[:n_take].contiguous()
        take_logits = logits[:n_take].contiguous()
        take_input_ids = input_ids[:n_take].contiguous()
        # Keep the remainder buffered as single tensors.
        rem_ids = ids[n_take:].contiguous()
        rem_logits = logits[n_take:].contiguous()
        rem_input_ids = input_ids[n_take:].contiguous()
        self._buf_ids = [rem_ids] if rem_ids.shape[0] else []
        self._buf_logits = [rem_logits] if rem_logits.shape[0] else []
        self._buf_input_ids = [rem_input_ids] if rem_input_ids.shape[0] else []
        self._buf_tokens = rem_ids.shape[0]

        name = _shard_name(self._shard_idx)
        start = self._shards[-1]["end"] if self._shards else 0
        save_file({
            "ids": take_ids,
            "logits": take_logits,
            "input_ids": take_input_ids,
        }, str(self.out_dir / name))
        self._shards.append(
            {"name": name, "start": start, "end": start + n_take, "n": n_take})
        self._shard_idx += 1


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class LogitStoreReader:
    """Memory-mapped reader over a sharded teacher-logit store.

    * ``len(reader)`` == total tokens.
    * ``reader.next_block(n)`` — sequential cursor: returns the next `n` tokens
      as ``(ids[n,k] int64, logits[n,k] fp16, input_ids[n] int64)`` and advances
      the cursor. Transparently spans shard boundaries.
    * ``reader.get_range(start, end)`` — random access by global token index.
    * ``reader.seek(pos)`` / ``reader.tell()`` / ``reader.remaining()``.

    Shards are mmap'd lazily via safetensors and the handles are cached, so
    random/sequential access never loads a whole shard into RAM.
    """

    def __init__(self, directory: str):
        self.dir = pathlib.Path(directory)
        with open(self.dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.k = int(self.manifest["k"])
        self.vocab_size = int(self.manifest["vocab_size"])
        self.total_tokens = int(self.manifest["total_tokens"])
        self.teacher_model = self.manifest.get("teacher_model", "")
        self.tokenizer_name = self.manifest.get("tokenizer_name", "")
        self.shards = list(self.manifest["shards"])
        # Integrity: shards must tile [0, total_tokens) with no gaps/overlap.
        expect = 0
        for sh in self.shards:
            if int(sh["start"]) != expect:
                raise ValueError(
                    f"shard {sh['name']} start {sh['start']} != expected "
                    f"{expect} (gap/overlap in manifest)")
            if int(sh["end"]) - int(sh["start"]) != int(sh["n"]):
                raise ValueError(f"shard {sh['name']} n != end-start")
            expect = int(sh["end"])
        if expect != self.total_tokens:
            raise ValueError(
                f"shards cover {expect} tokens but total_tokens="
                f"{self.total_tokens}")
        self._cursor = 0
        self._handles: dict[str, object] = {}

    def __len__(self) -> int:
        return self.total_tokens

    def tell(self) -> int:
        return self._cursor

    def remaining(self) -> int:
        return self.total_tokens - self._cursor

    def seek(self, pos: int) -> None:
        if not (0 <= pos <= self.total_tokens):
            raise ValueError(f"seek {pos} out of range [0, {self.total_tokens}]")
        self._cursor = int(pos)

    def next_block(self, n: int):
        """Return the next `n` tokens and advance the cursor."""
        if self._cursor + n > self.total_tokens:
            raise IndexError(
                f"next_block({n}) would read past end: cursor={self._cursor}, "
                f"total={self.total_tokens}. The store is too small for this "
                f"training run — regenerate with a larger --max_tokens.")
        out = self.get_range(self._cursor, self._cursor + n)
        self._cursor += n
        return out

    def get_range(self, start: int, end: int):
        """Random access by global token index, half-open ``[start, end)``.

        Returns ``(ids[m,k] int64, logits[m,k] fp16, input_ids[m] int64)`` where
        ``m = end - start``, concatenated across whichever shards overlap.
        """
        if not (0 <= start <= end <= self.total_tokens):
            raise ValueError(
                f"get_range({start}, {end}) out of [0, {self.total_tokens}]")
        ids_parts, logit_parts, inid_parts = [], [], []
        for sh in self.shards:
            s0, s1 = int(sh["start"]), int(sh["end"])
            lo, hi = max(start, s0), min(end, s1)
            if lo >= hi:
                continue
            a, b = lo - s0, hi - s0   # local offsets within this shard
            h = self._handle(sh["name"])
            ids_parts.append(h.get_slice("ids")[a:b].to(torch.int64))
            logit_parts.append(h.get_slice("logits")[a:b])
            inid_parts.append(h.get_slice("input_ids")[a:b].to(torch.int64))
        if not ids_parts:        # zero-length range
            return (torch.empty(0, self.k, dtype=torch.int64),
                    torch.empty(0, self.k, dtype=torch.float16),
                    torch.empty(0, dtype=torch.int64))
        ids = torch.cat(ids_parts, dim=0) if len(ids_parts) > 1 else ids_parts[0]
        logits = (torch.cat(logit_parts, dim=0) if len(logit_parts) > 1
                  else logit_parts[0])
        input_ids = (torch.cat(inid_parts, dim=0) if len(inid_parts) > 1
                     else inid_parts[0])
        return ids, logits, input_ids

    # -- internal ---------------------------------------------------------
    def _handle(self, name: str):
        h = self._handles.get(name)
        if h is None:
            # safe_open mmaps the file; holding the handle keeps the mmap alive.
            h = safe_open(str(self.dir / name), framework="pt")
            self._handles[name] = h
        return h
