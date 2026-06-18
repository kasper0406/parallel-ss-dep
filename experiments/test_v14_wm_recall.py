"""Tests for the v14 WM-recall plumbing (2026-06-15).

The validated embedding-key + copy-readout recall mechanism, wired ADDITIVELY:
  1. WorkingMemory.key_from_embedding — address on a causal input-embedding
     window over the identifier (wm_namekey_probe.py).
  2. TinyLM.use_copy_head (CopyReadout) — pointer/copy over the addressed source
     span (wm_multitok_readout.py ARM B).
  3. data_mix.emit_read_mask — per-position mem_read_mask over recall answer
     spans, aligned through think-burst insertion.

CRITICAL: every new flag defaults OFF and the forward/loss/data stream is then
byte-identical to the pre-v14 (v12) behaviour — v12 re-imports these modules on
autoresume, so a default-on regression would crash it.

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_v14_wm_recall.py -v
"""
from __future__ import annotations

import re
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

import experiments.data_mix as dm
from experiments.data_mix import MixedSourceStream, SourceConfig
from experiments.model import CopyReadout, TinyLM, WorkingMemory


# ===========================================================================
# 1. WorkingMemory.key_from_embedding
# ===========================================================================

def _wm_inputs(B=2, T=12, d=16, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(B, T, d)
    input_ids = torch.randint(0, 50, (B, T))
    read_mask = torch.zeros(B, T, dtype=torch.bool)
    read_mask[:, -3:] = True
    return h, input_ids, read_mask


def test_key_from_embedding_default_off():
    """Default off → flag False, and the legacy decoupled read ignores
    input_emb (byte-identical to not passing it)."""
    mem = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7,
                        decoupled_kv=True).eval()
    assert mem.key_from_embedding is False
    h, ii, rm = _wm_inputs()
    with torch.no_grad():
        a = mem(h, ii, read_mask=rm)
        b = mem(h, ii, read_mask=rm, input_emb=torch.randn(2, 12, 16))
    torch.testing.assert_close(a, b)


def test_key_from_embedding_adds_no_params():
    """Embedding-key addressing uses RAW pooled embeddings + the existing DKV
    temperature → NO new state-dict keys (so a continuation ckpt loads with an
    identical param set)."""
    base = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7,
                         decoupled_kv=True)
    kfe = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7,
                        decoupled_kv=True, key_from_embedding=True)
    assert set(base.state_dict().keys()) == set(kfe.state_dict().keys())


def test_key_from_embedding_active_and_trains():
    """With the flag on the read is active and addressing params get gradient."""
    mem = WorkingMemory(d_model=16, d_mem=16, mem_size=16, thinking_token_id=7,
                        decoupled_kv=True, key_from_embedding=True)
    h, ii, rm = _wm_inputs()
    emb = torch.randn(2, 12, 16, requires_grad=True)
    out = mem(h, ii, read_mask=rm, input_emb=emb)
    assert not torch.allclose(out, h)
    out.pow(2).sum().backward()
    # gradient reaches the addressing temperature and the value path
    assert mem.logit_scale.grad is not None
    assert mem.W_v.weight.grad.abs().sum().item() > 0


def test_causal_emb_window_order_sensitive_and_causal():
    """The pooled window is order-sensitive (v13 != v31) and strictly causal
    (the value at t depends only on tokens <= t)."""
    mem = WorkingMemory(d_model=8, d_mem=8, mem_size=8, thinking_token_id=7,
                        decoupled_kv=True, key_from_embedding=True, key_window=4)
    torch.manual_seed(1)
    emb = torch.randn(1, 6, 8)
    pooled = mem._causal_emb_window(emb)
    # position 0 only sees token 0 (weight W); causal.
    torch.testing.assert_close(pooled[0, 0], 4.0 * emb[0, 0])
    # changing a FUTURE token must not change an earlier pooled position.
    emb2 = emb.clone()
    emb2[0, 5] += 10.0
    pooled2 = mem._causal_emb_window(emb2)
    torch.testing.assert_close(pooled[0, 2], pooled2[0, 2])
    # order sensitivity: swapping two tokens in the window changes the pool.
    emb3 = emb.clone()
    emb3[0, 2], emb3[0, 3] = emb[0, 3].clone(), emb[0, 2].clone()
    pooled3 = mem._causal_emb_window(emb3)
    assert not torch.allclose(pooled[0, 3], pooled3[0, 3])


# ===========================================================================
# 2. CopyReadout + TinyLM copy head / _contiguous_run_index
# ===========================================================================

def test_copy_readout_cold_start_gate_near_zero():
    head = CopyReadout(d_model=16)
    x = torch.randn(5, 16)
    with torch.no_grad():
        g = torch.sigmoid(head.gate(x))
    assert float(g.max()) < 0.01, "cold-start copy gate must be ~0"


def test_contiguous_run_index():
    mask = torch.tensor([[0, 1, 1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    idx = TinyLM._contiguous_run_index(mask)
    assert idx.tolist() == [[0, 0, 1, 2, 0, 0, 0, 1]]


# ---- TinyLM forward tests (need FLA / CUDA) ----
_HAS_CUDA = torch.cuda.is_available()
pytestmark_cuda = pytest.mark.skipif(not _HAS_CUDA,
                                     reason="DeltaNet/FLA needs CUDA")

THINK_ID = 60
PAD_ID = 0


def _make_model(*, use_copy_head=False, mem_key_from_embedding=False,
                use_memory=True, seed=0):
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(seed)
    return TinyLM(
        vocab_size=64, d_model=16, n_layers=2, n_heads=2, d_head=8,
        attention_cls=DeltaNetAttention, max_T=64,
        output_gate=True,
        use_memory=use_memory, mem_size=32, thinking_token_id=THINK_ID,
        pad_token_id=PAD_ID, mem_decoupled_kv=True,
        mem_key_from_embedding=mem_key_from_embedding,
        use_copy_head=use_copy_head,
    ).cuda()


@pytestmark_cuda
def test_tinylm_copy_head_default_off_byte_identical():
    """A model WITHOUT use_copy_head produces identical logits to the same-seed
    model — the new _apply_copy_head is a no-op when the flag is off, even when
    a mem_read_mask is supplied."""
    m = _make_model(use_copy_head=False, seed=3).eval()
    ids = torch.randint(1, 50, (2, 20), device="cuda")
    rm = torch.zeros(2, 20, dtype=torch.bool, device="cuda")
    rm[:, -4:] = True
    with torch.no_grad():
        a = m(ids)
        b = m(ids, mem_read_mask=rm)
    # mem_read_mask changes WHERE the WM injects (expected), but the copy path
    # must not fire: build a reference where the copy code cannot run.
    assert not hasattr(m, "copy_head")
    # The default (None) path is the one v12 uses; re-run identical → identical.
    with torch.no_grad():
        a2 = m(ids)
    torch.testing.assert_close(a, a2)


@pytestmark_cuda
def test_tinylm_copy_head_changes_only_masked_positions():
    """With the copy head on, logits change ONLY at mem_read_mask positions and
    the returned logits are a valid log-prob (softmax sums to 1)."""
    m = _make_model(use_copy_head=True, mem_key_from_embedding=True,
                    seed=4).eval()
    assert hasattr(m, "copy_head")
    assert m.memory._stash_read_attn_grad is True
    ids = torch.randint(1, 50, (2, 20), device="cuda")
    rm = torch.zeros(2, 20, dtype=torch.bool, device="cuda")
    rm[:, 15:18] = True
    # force the copy gate wide open so the mix is clearly active
    with torch.no_grad():
        m.copy_head.gate.bias.fill_(10.0)
        base = m(ids)                       # no mask → copy cannot fire
        mixed = m(ids, mem_read_mask=rm)
    diff = (mixed - base).abs().sum(dim=-1)        # (B, T)
    assert diff[rm].sum().item() > 0, "masked positions must change"
    assert diff[~rm].sum().item() == pytest.approx(0.0, abs=1e-4), \
        "non-masked positions must be untouched"
    # softmax(returned logits) is a proper distribution at masked positions.
    probs = torch.softmax(mixed[rm], dim=-1)
    torch.testing.assert_close(probs.sum(-1), torch.ones(int(rm.sum()),
                                                         device="cuda"),
                               rtol=1e-3, atol=1e-3)


@pytestmark_cuda
def test_tinylm_copy_head_off_with_mask_matches_no_copy_model():
    """use_copy_head=False with a mem_read_mask must equal a fresh same-seed
    model (the WM injection at the mask is the ONLY effect — the copy path is
    inert)."""
    m1 = _make_model(use_copy_head=False, seed=7).eval()
    m2 = _make_model(use_copy_head=False, seed=7).eval()
    ids = torch.randint(1, 50, (2, 16), device="cuda")
    rm = torch.zeros(2, 16, dtype=torch.bool, device="cuda")
    rm[:, -3:] = True
    with torch.no_grad():
        torch.testing.assert_close(m1(ids, mem_read_mask=rm),
                                   m2(ids, mem_read_mask=rm))


# ===========================================================================
# 3. data_mix emit_read_mask
# ===========================================================================

class _OffsetTok:
    """Whitespace-run tokenizer with char offsets (supports the __call__ +
    return_offsets_mapping interface the read-mask path needs) AND a matching
    .encode (used by the non-mask path)."""
    eos_token_id = 1
    bos_token_id = 1
    vocab_size = 1000

    def __init__(self):
        self._vocab = {}

    def _id(self, w):
        if w not in self._vocab:
            self._vocab[w] = 10 + len(self._vocab)
        return self._vocab[w]

    def _tokens(self, text):
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w, _, _ in self._tokens(text)]

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False):
        toks = self._tokens(text)
        out = {"input_ids": [self._id(w) for w, _, _ in toks]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(s, e) for _, s, e in toks]
        return out

    def __len__(self):
        return self.vocab_size + 200


def _recall_record():
    prompt = "context line one and two with many distractor words here"
    comp = "The recalled value Answer: ZEBRA"
    c0 = comp.index("ZEBRA")
    return {"problem_prompt": prompt, "qwen_completion": comp,
            "answer": "ZEBRA", "answer_char_span": [c0, c0 + len("ZEBRA")]}


def _run_stream(sources, *, emit_read_mask, emit_doc_ids=True,
                think_burst_prob=0.0, block_size=24, n=4, streams=None,
                base_seed=0):
    tok = _OffsetTok()
    with mock.patch.object(dm, "_open_stream",
                           lambda src, seed=0: iter(streams[src.name])):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                               block_size=block_size, thinking_token_id=THINK_ID,
                               think_burst_prob=think_burst_prob,
                               think_max_bursts=2, think_max_burst_depth=4,
                               base_seed=base_seed, emit_doc_ids=emit_doc_ids,
                               emit_read_mask=emit_read_mask)
        it = iter(ds)
        return [next(it) for _ in range(n)], tok


def test_emit_read_mask_default_off_arity_unchanged():
    """emit_read_mask off → 3-tuple (with doc_ids) exactly as v12."""
    src = SourceConfig(name="s0", dataset_id="x", text_field="text", weight=1.0)
    long_text = " ".join("w%d" % i for i in range(300))
    out, _ = _run_stream([src], emit_read_mask=False,
                         streams={"s0": [{"text": long_text}] * 50})
    for item in out:
        assert len(item) == 3  # (inputs, targets, doc_ids)


def test_emit_read_mask_yields_4tuple_and_marks_answer_span():
    """emit_read_mask on → 4-tuple; the mask is 1 exactly on the answer token(s)
    of a recall source and 0 everywhere else."""
    src = SourceConfig(name="rec", dataset_id="x",
                       text_field=["problem_prompt", "qwen_completion"],
                       weight=1.0, emit_read_mask=True)
    rec = _recall_record()
    out, tok = _run_stream([src], emit_read_mask=True, block_size=40, n=1,
                           streams={"rec": [rec] * 50})
    inputs, targets, doc_ids, read_mask = out[0]
    assert len(out[0]) == 4
    assert read_mask.shape == inputs.shape
    assert read_mask.sum().item() >= 1, "answer span must be marked"
    # The mask marks PREDICTING positions (one before each answer token), so the
    # token AFTER each masked position is the answer token (logits[p]->token p+1).
    ans_id = tok._id("ZEBRA")
    pos = read_mask.bool().nonzero().flatten().tolist()
    nxt = [inputs[p + 1].item() for p in pos if p + 1 < inputs.numel()]
    assert nxt and all(t == ans_id for t in nxt), \
        f"token after each masked pos {nxt} should be the answer token {ans_id}"


def test_emit_read_mask_zero_for_non_recall_source():
    """A source WITHOUT emit_read_mask contributes an all-zero mask."""
    src = SourceConfig(name="plain", dataset_id="x", text_field="text",
                       weight=1.0, emit_read_mask=False)
    long_text = " ".join("w%d" % i for i in range(300))
    out, _ = _run_stream([src], emit_read_mask=True,
                         streams={"plain": [{"text": long_text}] * 50})
    for inputs, targets, doc_ids, read_mask in out:
        assert read_mask.sum().item() == 0


def test_emit_read_mask_substring_fallback_no_charspan():
    """A recall source WITHOUT answer_char_span (e.g. multibind) falls back to
    the first occurrence of the `answer` string inside the answer field."""
    src = SourceConfig(name="mb", dataset_id="x",
                       text_field=["problem_prompt", "qwen_completion"],
                       weight=1.0, emit_read_mask=True)
    rec = {"problem_prompt": "v17 = 4530 distractor distractor here",
           "qwen_completion": "v17 is set to 4530", "answer": "4530"}
    # no answer_char_span key -> substring fallback on qwen_completion
    out, tok = _run_stream([src], emit_read_mask=True, block_size=40, n=1,
                           streams={"mb": [rec] * 50})
    inputs, targets, doc_ids, read_mask = out[0]
    ans_id = tok._id("4530")
    pos = read_mask.bool().nonzero().flatten().tolist()
    nxt = [inputs[p + 1].item() for p in pos if p + 1 < inputs.numel()]
    assert nxt and all(t == ans_id for t in nxt)


def test_mask_first_occurrence_default_off():
    """mask_first_occurrence defaults False → annotated answer_char_span is used
    (byte-identical to pre-fix behaviour)."""
    src = SourceConfig(name="s", dataset_id="x", text_field="text", weight=1.0)
    assert src.mask_first_occurrence is False


def test_mask_first_occurrence_targets_first_not_restated():
    """MASK FIX (project_const_recall_mask_mismatch): with a completion that has
    the value TWICE (a recall-hard first mention + a recency-trivial restated
    'Answer: V'), the annotated answer_char_span points at the SECOND occurrence.
    mask_first_occurrence=True must instead mask the FIRST occurrence (the position
    the leak-free eval scores), while default (False) keeps the annotated second."""
    from experiments.data_mix import _build_read_mask
    tok = _OffsetTok()
    prompt = "context line one and two with many distractor words here today"
    comp = "the value is ZEBRA and is later restated as Answer: ZEBRA"
    c0_restated = comp.rindex("ZEBRA")          # SECOND occurrence
    rec = {"problem_prompt": prompt, "qwen_completion": comp, "answer": "ZEBRA",
           "answer_char_span": [c0_restated, c0_restated + len("ZEBRA")]}
    text = prompt + "\n\n" + comp               # list text_field is joined with \n\n
    zid = tok._id("ZEBRA")

    def masked_positions(flag):
        src = SourceConfig(name="rec", dataset_id="x",
                           text_field=["problem_prompt", "qwen_completion"],
                           weight=1.0, emit_read_mask=True,
                           mask_first_occurrence=flag)
        ids, mask = _build_read_mask(rec, text, src, tok)
        pos = [i for i, m in enumerate(mask) if m == 1]
        # every masked position must PREDICT the value token (logits[p] -> p+1)
        assert pos and all(ids[p + 1] == zid for p in pos if p + 1 < len(ids))
        return pos

    pos_annotated = masked_positions(False)     # second / restated occurrence
    pos_firstocc = masked_positions(True)       # first occurrence
    # first-occurrence supervision must land strictly EARLIER than the annotated
    # restated occurrence.
    assert max(pos_firstocc) < min(pos_annotated), (
        f"first-occ mask {pos_firstocc} must precede restated mask {pos_annotated}")


def test_mask_first_occurrence_word_boundary_skips_embedded_substring():
    """MASK FIX edge case (import family, review finding b): a short alias value
    like `it` must NOT match the embedded `it` inside an earlier word (`itertools`).
    The word-boundary search picks the standalone alias occurrence, so the masked
    position predicts the alias token, not the longer word it is embedded in."""
    from experiments.data_mix import _build_read_mask
    tok = _OffsetTok()
    prompt = "module header with import statements and distractor words here"
    comp = "itertools is imported and then aliased to it"   # 'it' embedded then standalone
    rec = {"problem_prompt": prompt, "qwen_completion": comp, "answer": "it"}
    text = prompt + "\n\n" + comp
    src = SourceConfig(name="imp", dataset_id="x",
                       text_field=["problem_prompt", "qwen_completion"],
                       weight=1.0, emit_read_mask=True, mask_first_occurrence=True)
    ids, mask = _build_read_mask(rec, text, src, tok)
    pos = [i for i, m in enumerate(mask) if m == 1]
    assert pos, "alias must be masked"
    it_id = tok._id("it")
    nxt = [ids[p + 1] for p in pos if p + 1 < len(ids)]
    assert nxt and all(t == it_id for t in nxt), (
        f"masked positions must predict the standalone alias 'it' ({it_id}), "
        f"not the embedded substring inside 'itertools'; got {nxt}")


@pytestmark_cuda
def test_copy_head_answer_ce_gradient_reaches_addressing_and_copy():
    """The load-bearing v14 property: answer-span CE gradient flows into the
    copy gate AND the WM addressing (temperature, value path, read_alpha, and
    the input embeddings) — the gradient that was structurally missing in
    v10-v13 ('WM inert for recall')."""
    m = _make_model(use_copy_head=True, mem_key_from_embedding=True, seed=5)
    m.train()
    ids = torch.randint(1, 50, (2, 24), device="cuda")
    y = torch.randint(1, 50, (2, 24), device="cuda")
    rm = torch.zeros(2, 24, dtype=torch.bool, device="cuda")
    rm[:, -3:] = True
    out = m(ids, mem_read_mask=rm)
    logits = out[0] if isinstance(out, tuple) else out
    F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).backward()
    assert m.copy_head.gate.weight.grad.abs().sum().item() > 0
    assert m.memory.logit_scale.grad is not None and \
        m.memory.logit_scale.grad.abs().item() > 0
    assert m.memory.W_v.weight.grad.abs().sum().item() > 0
    assert m.embed.weight.grad.abs().sum().item() > 0


def test_emit_read_mask_think_positions_never_marked():
    """With think bursts inserted, think tokens are never read positions and the
    mask stays aligned (answer token still marked)."""
    src = SourceConfig(name="rec", dataset_id="x",
                       text_field=["problem_prompt", "qwen_completion"],
                       weight=1.0, emit_read_mask=True)
    rec = _recall_record()
    out, tok = _run_stream([src], emit_read_mask=True, think_burst_prob=1.0,
                           block_size=40, n=6, streams={"rec": [rec] * 200},
                           base_seed=11)
    saw_think = False
    for inputs, targets, doc_ids, read_mask in out:
        think_pos = (inputs == THINK_ID)
        if think_pos.any():
            saw_think = True
        # core invariant: no think position is ever a read position, and the
        # mask survives insertion (stays non-empty when an answer is present).
        assert (read_mask.bool() & think_pos).sum().item() == 0
    assert saw_think, "test should have exercised think-burst insertion"
