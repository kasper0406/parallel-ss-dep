"""
Mixed-corpus streaming loader for the 217 M super-coder pretrain.

Reads a YAML config that lists weighted HuggingFace streaming sources
(the-stack code, GitHub issues, Stack Exchange, arXiv, Wikipedia, open
textbooks, commitpackft) and yields fixed-length tokenised chunks with
periodic random think-burst insertion (so the working-memory + gate-head
get dense supervised gradient from step 0 of pretrain).

Output protocol matches `train_lm.py::TokenisedStream`: each iteration
yields `(inputs, targets)` of shape `(block_size,)`. At positions where
the next token is a think token, `targets` is set to -100 so cross-entropy
ignores it.

CLI smoke (read 100 chunks, print per-source counts + decoded samples):

    PYTHONPATH=. python -m experiments.data_mix \\
        --config configs/pretrain_mix_v1.yaml --n_chunks 100 --block_size 2048
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Callable, Iterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import IterableDataset

from experiments.sft_code import insert_think_bursts


# ---------------------------------------------------------------------------
# Named filters: referenced by string from the YAML config so we don't need
# to eval arbitrary Python from data files. Each takes a HuggingFace example
# (dict) and returns bool. Register new filters here.

def _filter_min_content_len(min_chars: int) -> Callable[[dict], bool]:
    def f(ex):
        c = ex.get("content") or ex.get("text") or ""
        return isinstance(c, str) and len(c) >= min_chars
    return f


def _filter_se_score(min_score: int, programming_only: bool
                     ) -> Callable[[dict], bool]:
    def f(ex):
        score = ex.get("score") or ex.get("question_score") or 0
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0
        if score < min_score:
            return False
        if not programming_only:
            return True
        tags = ex.get("tags") or ex.get("question_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        prog_tags = {"python", "javascript", "java", "c++", "c", "rust", "go",
                      "typescript", "algorithm", "data-structures",
                      "performance", "debugging", "refactoring", "regex",
                      "sql", "linux", "git", "react", "django", "numpy",
                      "pandas", "machine-learning"}
        return any(t in prog_tags for t in tags) if tags else True
    return f


def _filter_gh_issue_resolved() -> Callable[[dict], bool]:
    def f(ex):
        # Accept if has a code block OR explicitly closed/resolved.
        text = ex.get("content") or ex.get("text") or ""
        if "```" in text:
            return True
        state = (ex.get("state") or ex.get("status") or "").lower()
        return state in ("closed", "resolved", "merged")
    return f


def _filter_always() -> Callable[[dict], bool]:
    return lambda ex: True


def _filter_bigvul_vulnerable() -> Callable[[dict], bool]:
    """BigVul has two records per CVE: vul=0 (safe state, used for classifier
    training, before==after) and vul=1 (the actually-vulnerable code paired
    with its patch, before≠after). For generative training we want vul=1
    only — that's where the bug→fix lesson lives. ~7% of records by stream
    order, so the source weight should be bumped to compensate."""
    def f(ex):
        return int(ex.get("vul", 0)) == 1
    return f


FILTER_REGISTRY: dict[str, Callable[..., Callable[[dict], bool]]] = {
    "min_content_len": lambda min_chars=100: _filter_min_content_len(min_chars),
    "se_score": lambda min_score=3, programming_only=True: _filter_se_score(
        min_score, programming_only),
    "gh_issue_resolved": lambda: _filter_gh_issue_resolved(),
    "bigvul_vulnerable": lambda: _filter_bigvul_vulnerable(),
    "always": lambda: _filter_always(),
}


def _build_filter(spec: str | dict | None) -> Callable[[dict], bool]:
    """spec is either a string name with default args, or a dict
    `{name: 'se_score', args: {min_score: 5}}`."""
    if spec is None:
        return _filter_always()
    if isinstance(spec, str):
        if spec not in FILTER_REGISTRY:
            raise ValueError(f"unknown filter: {spec!r}")
        return FILTER_REGISTRY[spec]()
    name = spec.get("name")
    args = spec.get("args", {}) or {}
    if name not in FILTER_REGISTRY:
        raise ValueError(f"unknown filter: {name!r}")
    return FILTER_REGISTRY[name](**args)


# ---------------------------------------------------------------------------
# Source-level streaming + tokenisation.

@dataclass
class SourceConfig:
    name: str
    dataset_id: str = ""             # HF dataset id; "" iff jsonl_path is set
    text_field: str | list = "text"  # str: single field; list[str]: join with \n\n
    text_builder: str | None = None  # named builder in TEXT_BUILDER_REGISTRY;
                                      # if set, overrides text_field.
    weight: float = 1.0
    split: str = "train"
    hf_extra: dict | None = None     # passed to load_dataset (e.g. {'name': 'python'})
    filter_spec: str | dict | None = None
    skip_first: int = 0              # for shuffled streaming, optional
    fim_rate: float = 0.0            # PSM Fill-in-the-Middle augmentation rate
                                      # (per-document); 0.0 disables.
    jsonl_path: str | None = None    # if set, stream rows from a local JSONL
                                      # file instead of HF. `text_field`
                                      # selects which field to extract.


# ---------------------------------------------------------------------------
# Fill-in-the-Middle (PSM variant, per DeepSeek-Coder). Sentinels are
# plain text strings — the tokenizer will encode them as multi-token
# sequences and the model learns them as fixed patterns. No vocab change.

_FIM_PREFIX = "<|fim_prefix|>"
_FIM_SUFFIX = "<|fim_suffix|>"
_FIM_MIDDLE = "<|fim_middle|>"


def maybe_apply_fim(text: str, *, rng: random.Random, fim_rate: float) -> str:
    """With probability `fim_rate`, reformat `text` as PSM Fill-in-the-Middle:

        <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}

    The split picks two distinct char boundaries (0 < a < b < len(text)).
    Otherwise the text is returned unchanged. Short texts (< 4 chars) are
    always returned unchanged since there is no room for prefix/middle/suffix.
    """
    if fim_rate <= 0.0:
        return text
    if rng.random() >= fim_rate:
        return text
    n = len(text)
    if n < 4:
        return text
    # Two distinct boundaries in (0, n); sort to get (a, b) with a < b.
    a, b = sorted(rng.sample(range(1, n), 2))
    prefix = text[:a]
    middle = text[a:b]
    suffix = text[b:]
    return f"{_FIM_PREFIX}{prefix}{_FIM_SUFFIX}{suffix}{_FIM_MIDDLE}{middle}"


# ---------------------------------------------------------------------------
# Text builders: for sources where the training text is a *composition* of
# multiple fields (e.g. BigVul = CVE-id + commit-msg + before/after code).
# Each builder takes an HF example dict and returns either the assembled
# string, or None to skip the example (missing required fields, etc.).

def _builder_bigvul(ex: dict) -> str | None:
    """BigVul (vulnerable→fixed C/C++ function pairs from CVE patches).
    Schema: 'CVE ID', 'CWE ID', 'commit_message', 'func_before', 'func_after',
    'lang', 'project'. Format as a CVE-prefixed before/after block so the
    model sees vulnerability description + bug pattern + fix together."""
    before = (ex.get("func_before") or "").strip()
    after = (ex.get("func_after") or "").strip()
    # Skip examples missing either side — both are required for the lesson
    # to make sense.
    if not before or not after or before == after:
        return None
    cve = ex.get("CVE ID") or "CVE-?"
    cwe = ex.get("CWE ID") or "CWE-?"
    lang = ex.get("lang") or "code"
    project = ex.get("project") or ""
    msg = (ex.get("commit_message") or "").strip()
    if len(msg) > 1200:
        msg = msg[:1200] + " ..."
    proj_str = f"in {project}" if project else ""
    return (
        f"Security fix {proj_str}\n"
        f"{cve}  ({cwe})\n"
        f"Fix description:\n{msg}\n\n"
        f"Vulnerable {lang} code:\n```{lang}\n{before}\n```\n\n"
        f"Patched {lang} code:\n```{lang}\n{after}\n```"
    )


def _builder_cybernative_dpo(ex: dict) -> str | None:
    """CyberNative/Code_Vulnerability_Security_DPO. Preference-pair dataset
    with `vulnerability`, `question`, `chosen`, `rejected`. For pretraining
    we use only the *chosen* (safe) side — the rejected side teaches bad
    patterns we don't want to reinforce."""
    vuln = (ex.get("vulnerability") or "").strip()
    question = (ex.get("question") or "").strip()
    chosen = (ex.get("chosen") or "").strip()
    if not question or not chosen:
        return None
    lang = ex.get("lang") or "code"
    return (
        f"Vulnerability awareness ({lang}): {vuln}\n\n"
        f"Task: {question}\n\n"
        f"Safe implementation:\n{chosen}"
    )


TEXT_BUILDER_REGISTRY: dict[str, callable] = {
    "bigvul": _builder_bigvul,
    "cybernative_dpo": _builder_cybernative_dpo,
}


def _extract_text(example: dict, text_field, text_builder: str | None = None
                  ) -> str | None:
    """Pull text from an example.

    - If `text_builder` is set, dispatch to that builder (see
      TEXT_BUILDER_REGISTRY) which takes the full example dict.
    - Else `text_field` is a string (single key) or a list of keys whose
      values are joined with '\\n\\n'.
    Returns None when nothing usable is present.
    """
    if text_builder:
        if text_builder not in TEXT_BUILDER_REGISTRY:
            raise ValueError(f"unknown text_builder: {text_builder!r}")
        return TEXT_BUILDER_REGISTRY[text_builder](example)
    if isinstance(text_field, str):
        val = example.get(text_field)
        if val is None:
            # Backwards-compat fallback to common alternative names.
            val = example.get("content") or example.get("text")
        return val if isinstance(val, str) and val else None
    # List of fields — join with double newline. Skip missing/empty.
    parts = []
    for k in text_field:
        v = example.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    if not parts:
        return None
    return "\n\n".join(parts)


def _jsonl_stream(path: str, seed: int = 0, skip_first: int = 0):
    """Stream a local JSONL file as an HF-like iterable. Yields dict
    rows. Cycles forever so the source doesn't exhaust mid-training;
    each cycle is shuffled with a different seed (seed + cycle_idx)."""
    p = pathlib.Path(path)
    if not p.exists():
        raise RuntimeError(f"jsonl_path does not exist: {p}")
    # Load all rows once (synthetic JSONLs are small enough — 50–100k
    # rows of ~500 chars = ~50 MB). For larger files, stream + reservoir
    # shuffle would be needed; we don't go there yet.
    rows: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise RuntimeError(f"jsonl_path is empty: {p}")
    rng = random.Random(seed)
    cycle = 0
    while True:
        order = list(range(len(rows)))
        rng_cycle = random.Random(seed + cycle)
        rng_cycle.shuffle(order)
        start = skip_first if cycle == 0 else 0
        for i in order[start:]:
            yield rows[i]
        cycle += 1


def _open_stream(src: SourceConfig, seed: int = 0):
    # Local JSONL branch: skip HF entirely.
    if src.jsonl_path:
        return _jsonl_stream(src.jsonl_path, seed=seed,
                             skip_first=src.skip_first)
    from datasets import load_dataset
    kw = dict(streaming=True)
    if src.hf_extra:
        kw.update(src.hf_extra)
    try:
        ds = load_dataset(src.dataset_id, split=src.split, **kw)
    except Exception as e:
        raise RuntimeError(
            f"failed to open dataset {src.dataset_id} (split={src.split}, "
            f"hf_extra={src.hf_extra}): {e}"
        )
    ds = ds.shuffle(seed=seed, buffer_size=1024)
    if src.skip_first > 0:
        ds = ds.skip(src.skip_first)
    return ds


# ---------------------------------------------------------------------------
# MixedSourceStream — the actual IterableDataset used by train_lm.

class MixedSourceStream(IterableDataset):
    """Weighted-sample multi-source HF streaming → fixed-size token chunks
    with periodic think-burst injection.

    Yields `(inputs, targets)` where:
      - `inputs` is a LongTensor of shape (block_size,).
      - `targets` is the next-token shift; positions where the *next* token
        is a think token are set to -100 so F.cross_entropy ignores them.

    Multi-worker safety: each worker gets a different `worker_seed` so the
    HF shuffle is disjoint per worker. There is still potential overlap with
    streaming + shuffle_buffer=1024 — acceptable for pretraining (data is
    huge relative to a single worker's window).
    """

    def __init__(self,
                 sources: list[SourceConfig],
                 tokenizer,
                 block_size: int,
                 thinking_token_id: int | None,
                 think_burst_prob: float = 0.5,
                 think_max_bursts: int = 2,
                 think_max_burst_depth: int = 6,
                 base_seed: int = 0,
                 mask_eos_in_targets: bool = False,
                 emit_doc_ids: bool = False,
                 ):
        if not sources:
            raise ValueError("sources must be non-empty")
        if abs(sum(s.weight for s in sources) - 1.0) > 1e-4:
            # Re-normalise.
            total = sum(s.weight for s in sources)
            sources = [SourceConfig(**{**s.__dict__, "weight": s.weight / total})
                       for s in sources]
        self.sources = sources
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.thinking_token_id = thinking_token_id
        self.think_burst_prob = float(think_burst_prob)
        self.think_max_bursts = int(think_max_bursts)
        self.think_max_burst_depth = int(think_max_burst_depth)
        self.base_seed = int(base_seed)
        self.mask_eos_in_targets = bool(mask_eos_in_targets)
        self.emit_doc_ids = bool(emit_doc_ids)
        self._filters = [_build_filter(s.filter_spec) for s in sources]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info else 0
        n_workers = info.num_workers if info else 1
        seed = self.base_seed + 17 * worker_id

        iters = [iter(_open_stream(src, seed=seed + i))
                 for i, src in enumerate(self.sources)]
        # Per-source token buffers + per-source running counts (for sampling
        # diagnostics). Sampling is at the *chunk* level — when we need a
        # new chunk we pick a source by weight, draw examples from it until
        # we have block_size+1 tokens, then yield.
        weights = [s.weight for s in self.sources]
        rng = random.Random(seed)
        torch_rng = torch.Generator().manual_seed(seed)
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.bos_token_id
        if eos is None:
            eos = 0

        # One persistent buffer per source so leftover tokens are reused.
        buffers: list[list[int]] = [[] for _ in self.sources]
        # Parallel per-position document ids (only filled when emit_doc_ids):
        # every token of a document shares an id; the trailing EOS belongs to
        # the *closing* document; the next document's first token increments.
        buffer_docids: list[list[int]] = [[] for _ in self.sources]
        doc_counters = [0] * len(self.sources)
        source_counts = [0] * len(self.sources)  # for smoke diagnostics

        def fill_buffer(idx: int) -> bool:
            """Pull examples from source idx until its buffer has >= block_size+1
            tokens. Returns False if the stream is exhausted."""
            target = self.block_size + 1
            tok = self.tokenizer
            f = self._filters[idx]
            src = self.sources[idx]
            text_field = src.text_field
            text_builder = src.text_builder
            fim_rate = float(src.fim_rate)
            while len(buffers[idx]) < target:
                try:
                    ex = next(iters[idx])
                except StopIteration:
                    return False
                if not f(ex):
                    continue
                text = _extract_text(ex, text_field, text_builder=text_builder)
                if text is None:
                    continue
                if fim_rate > 0.0:
                    text = maybe_apply_fim(text, rng=rng, fim_rate=fim_rate)
                ids = tok.encode(text, add_special_tokens=False)
                buffers[idx].extend(ids)
                buffers[idx].append(eos)
                if self.emit_doc_ids:
                    c = doc_counters[idx]
                    buffer_docids[idx].extend([c] * (len(ids) + 1))  # +1 = EOS
                    doc_counters[idx] = c + 1
            return True

        while True:
            # Sample a source by weight.
            idx = rng.choices(range(len(self.sources)), weights=weights, k=1)[0]
            if not fill_buffer(idx):
                # That source is exhausted — drop its weight and try again.
                weights[idx] = 0.0
                if sum(weights) <= 0.0:
                    return
                continue
            source_counts[idx] += 1
            chunk = buffers[idx][: self.block_size + 1]
            buffers[idx] = buffers[idx][self.block_size:]
            if self.emit_doc_ids:
                chunk_docids = buffer_docids[idx][: self.block_size + 1]
                buffer_docids[idx] = buffer_docids[idx][self.block_size:]
            else:
                chunk_docids = None
            # Random think-burst insertion at chunk boundary.
            if (self.thinking_token_id is not None
                    and self.think_burst_prob > 0.0
                    and rng.random() < self.think_burst_prob):
                # Reuse `insert_think_bursts` with labels = ids; we'll re-derive
                # the loss mask after shifting. Labels at think positions are
                # set to -100 by `insert_think_bursts` which we don't use here;
                # what matters is the inserted think tokens at random positions.
                ids_only = chunk[:]
                fake_labels = chunk[:]  # placeholder; rebuilt below
                if chunk_docids is not None:
                    ids_with_thinks, _, docids_with_thinks = insert_think_bursts(
                        ids_only, fake_labels,
                        thinking_token_id=int(self.thinking_token_id),
                        max_len=self.block_size + 1,
                        max_bursts=self.think_max_bursts,
                        max_burst_depth=self.think_max_burst_depth,
                        rng=torch_rng,
                        aligned=chunk_docids[:],
                    )
                else:
                    ids_with_thinks, _ = insert_think_bursts(
                        ids_only, fake_labels,
                        thinking_token_id=int(self.thinking_token_id),
                        max_len=self.block_size + 1,
                        max_bursts=self.think_max_bursts,
                        max_burst_depth=self.think_max_burst_depth,
                        rng=torch_rng,
                    )
                    docids_with_thinks = None
                # Re-pad if shortened (insert_think_bursts caps at max_len; can
                # be shorter if the bursts pushed the tail off and we already
                # had < max_len real tokens). Pad with eos.
                while len(ids_with_thinks) < self.block_size + 1:
                    ids_with_thinks.append(eos)
                    if docids_with_thinks is not None:
                        # Padding belongs to the last document seen.
                        docids_with_thinks.append(
                            docids_with_thinks[-1] if docids_with_thinks else 0)
                chunk = ids_with_thinks[: self.block_size + 1]
                if docids_with_thinks is not None:
                    chunk_docids = docids_with_thinks[: self.block_size + 1]
            # Shift to (inputs, targets); mask targets at think positions.
            inputs = torch.tensor(chunk[:-1], dtype=torch.long)
            targets = torch.tensor(chunk[1:], dtype=torch.long)
            if self.thinking_token_id is not None:
                think_mask = (targets == int(self.thinking_token_id))
                if think_mask.any():
                    targets = targets.masked_fill(think_mask, -100)
            # Optionally mask EOS in targets. Stops the model from learning
            # "predict EOS at sample boundary" (the halt-after-docstring
            # artifact observed at v2 attempt 3, 500M-token ckpt: HumanEval
            # generations terminated on the first emit because the model
            # learned EOS-after-`"""` from short documents). Off by default
            # for backwards compat with already-checked-in v2 launchers.
            if self.mask_eos_in_targets:
                eos_mask = (targets == int(eos))
                if eos_mask.any():
                    targets = targets.masked_fill(eos_mask, -100)
            if self.emit_doc_ids:
                # Align with `inputs` (drop the last position) and normalise
                # to 0-based per chunk so doc_ids start at 0 in every row.
                row = chunk_docids[:-1]
                mn = min(row) if row else 0
                doc_ids = torch.tensor([d - mn for d in row], dtype=torch.long)
                yield inputs, targets, doc_ids
            else:
                yield inputs, targets


# ---------------------------------------------------------------------------
# YAML loader + smoke entry-point.

def load_sources_from_yaml(path: str) -> list[SourceConfig]:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "sources" not in cfg:
        raise ValueError(f"YAML at {path} missing 'sources' key")
    out = []
    for src in cfg["sources"]:
        if not src.get("enabled", True):
            continue
        jsonl_path = src.get("jsonl_path")
        dataset_id = src.get("dataset_id", "")
        if not jsonl_path and not dataset_id:
            raise ValueError(
                f"source {src.get('name')!r}: must set either dataset_id "
                f"(HF) or jsonl_path (local JSONL)")
        out.append(SourceConfig(
            name=src["name"],
            dataset_id=dataset_id,
            text_field=src.get("text_field", "text"),
            text_builder=src.get("text_builder"),
            weight=float(src.get("weight", 1.0)),
            split=src.get("split", "train"),
            hf_extra=src.get("hf_extra"),
            filter_spec=src.get("filter"),
            skip_first=int(src.get("skip_first", 0)),
            fim_rate=float(src.get("fim_rate", 0.0)),
            jsonl_path=jsonl_path,
        ))
    if not out:
        raise ValueError(f"YAML at {path} has no enabled sources")
    return out


def _smoke(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Make space for the think token: snap (vocab+1) up to multiple of 64.
    base_vocab = tok.vocab_size
    vocab_with_think = ((base_vocab + 1 + 63) // 64) * 64
    thinking_token_id = base_vocab  # first slot after the real vocab
    sources = load_sources_from_yaml(args.config)
    print(f"Loaded {len(sources)} sources from {args.config}:")
    for s in sources:
        print(f"  - {s.name:18s} weight={s.weight:.3f}  id={s.dataset_id}"
              f"  split={s.split}  filter={s.filter_spec}")
    print(f"Tokeniser: {args.tokenizer}  vocab={base_vocab}  "
          f"vocab_with_think={vocab_with_think}  thinking_id={thinking_token_id}")
    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.block_size,
        thinking_token_id=thinking_token_id,
        think_burst_prob=args.think_burst_prob,
        think_max_bursts=args.think_max_bursts,
        think_max_burst_depth=args.think_max_burst_depth,
    )
    it = iter(ds)
    src_count = [0] * len(sources)
    burst_hits = 0
    chunk_lens = []
    samples: list[tuple[str, str]] = []   # (source_name_guess, decoded_preview)
    for i in range(args.n_chunks):
        try:
            x, y = next(it)
        except StopIteration:
            print(f"Stream exhausted at chunk {i}")
            break
        chunk_lens.append(int(x.numel()))
        # Detect think bursts: targets containing -100.
        if (y == -100).any():
            burst_hits += 1
        if i < 3:
            ids = x.tolist()
            txt = tok.decode([t for t in ids if t != thinking_token_id],
                              skip_special_tokens=True)
            n_think = sum(1 for t in ids if t == thinking_token_id)
            samples.append((f"chunk_{i}", f"think_tokens={n_think}  "
                            f"preview={txt[:160]!r}"))
    print(f"\nProcessed {len(chunk_lens)} chunks (block_size={args.block_size})")
    print(f"  burst injection rate: {burst_hits}/{len(chunk_lens)} = "
          f"{burst_hits/max(1,len(chunk_lens)):.3f} "
          f"(target ~{args.think_burst_prob})")
    print("\nSample chunks:")
    for name, prev in samples:
        print(f"  [{name}] {prev}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to pretrain mix YAML.")
    p.add_argument("--n_chunks", type=int, default=100)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--think_burst_prob", type=float, default=0.5)
    p.add_argument("--think_max_bursts", type=int, default=2)
    p.add_argument("--think_max_burst_depth", type=int, default=6)
    args = p.parse_args()
    _smoke(args)


if __name__ == "__main__":
    main()
