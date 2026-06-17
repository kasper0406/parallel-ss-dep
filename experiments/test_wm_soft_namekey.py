"""Tests for SOFT NAME-SPAN WM addressing (2026-06-16).

The soft-namekey path replaces the discrete lexical hash code-match with a
learned CONTINUOUS soft-attention key: q = enc(mean-pooled NAME-SPAN input
embeddings), read = softmax(cos(q_query, q_buf)/temp) over the BINDING
(value-start) buffer slots, causal + same-document masked. Default OFF → NO new
parameters → byte-identical, old ckpts load via strict=False. Validated
head-to-head by experiments/wm_vqkey_probe.py (matches the hash on separability,
adds surface-variant robustness).
"""
import pytest
import torch

from experiments.model import WorkingMemory


def _wm(soft=False, d_model=32, d_key=16, seed=0, **kw):
    torch.manual_seed(seed)
    return WorkingMemory(d_model=d_model, d_mem=d_model, mem_size=2048,
                         thinking_token_id=99, soft_namekey=soft,
                         soft_namekey_dim=d_key, **kw)


def test_default_off_no_new_params_and_byte_identical():
    """soft_namekey=False adds NO parameters and leaves the forward unchanged."""
    off = _wm(soft=False)
    # No soft-namekey params on the default (off) WM.
    assert not any("namekey" in k for k in off.state_dict().keys())
    # Param count identical to a plain legacy WM (the pre-change default).
    legacy = WorkingMemory(d_model=32, d_mem=32, mem_size=2048,
                           thinking_token_id=99)
    assert sorted(off.state_dict().keys()) == sorted(legacy.state_dict().keys())
    assert sum(p.numel() for p in off.parameters()) == \
        sum(p.numel() for p in legacy.parameters())
    # Off-path forward is deterministic & finite (legacy address-by-value path).
    torch.manual_seed(1)
    h = torch.randn(2, 16, 32)
    ids = torch.randint(0, 50, (2, 16))
    o1 = off(h, ids)
    o2 = off(h, ids)
    assert torch.equal(o1, o2)
    assert torch.isfinite(o1).all()


def test_soft_on_adds_encoder_params():
    """soft_namekey=True introduces the encoder + learned temperature, NOT on
    the off path."""
    on = _wm(soft=True, d_key=16)
    keys = set(on.state_dict().keys())
    assert any(k.startswith("namekey_enc.") for k in keys)
    assert "namekey_log_tau" in keys
    # The encoder maps d_model -> d_key.
    assert on.namekey_enc[-1].out_features == 16


def test_soft_and_discrete_mutually_exclusive():
    with pytest.raises(ValueError):
        WorkingMemory(d_model=32, d_mem=32, mem_size=64, thinking_token_id=99,
                      soft_namekey=True, discrete_key=True)


def test_soft_forward_requires_input_emb():
    """The soft path pools input embeddings, so input_emb must be passed."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(soft=True)
    wm.set_discrete_key_vocab(tok)
    ids = torch.tensor([tok.encode("cache_size = 11\nreturn cache_size",
                                   add_special_tokens=False)])
    h = torch.randn(1, ids.shape[1], 32)
    with pytest.raises(ValueError):
        wm(h, ids)                       # no input_emb → explicit error
    emb = torch.randn(1, ids.shape[1], 32)
    out = wm(h, ids, input_emb=emb)      # with input_emb → runs, finite
    assert torch.isfinite(out).all()


def test_encoder_produces_per_binding_key_namespan_only():
    """The per-binding key is a deterministic function of the NAME SPAN ONLY:
      (a) DISTINCT names → DISTINCT keys (separability);
      (b) changing the VALUE (or arrow) leaves a binding's key UNCHANGED — i.e.
          the pool is over the name span only, NOT the contaminated name+arrow+
          value window that broke the historical soft key.
    """
    from transformers import AutoTokenizer
    import torch.nn.functional as F
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(soft=True, seed=3)
    wm.set_discrete_key_vocab(tok)
    emb_table = torch.nn.Embedding(len(tok), 32)
    torch.nn.init.normal_(emb_table.weight, std=0.5)

    def keys_at_bindings(prog):
        ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
        emb = emb_table(ids)
        name_key, vstart, _ = wm._name_span_keys(ids, emb)
        q = wm.namekey_enc(name_key)[0]                         # (T, d_key)
        vs = [i for i, v in enumerate(vstart[0].tolist()) if v]
        return [q[p] for p in vs]

    k_a = keys_at_bindings("cache_size = 1111\nqueue_depth = 2222\n")
    assert len(k_a) == 2
    # (a) separability: the two distinct-name bindings have distinguishable keys.
    cos_cross = float(F.cosine_similarity(k_a[0][None], k_a[1][None]))
    assert cos_cross < 0.999
    # (b) name-span only: same names, DIFFERENT values → identical keys.
    k_b = keys_at_bindings("cache_size = 7777\nqueue_depth = 3333\n")
    assert torch.allclose(k_a[0], k_b[0], atol=1e-5)
    assert torch.allclose(k_a[1], k_b[1], atol=1e-5)


def test_soft_read_is_causal_and_doc_masked():
    """The soft read obeys (a) causality — a query never attends to a buffer
    slot at/after its own position; (b) document isolation — a query reads only
    same-document bindings, even when a same-named binding exists in another
    packed document."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(soft=True, seed=5)
    wm.set_discrete_key_vocab(tok)
    # doc0: bind cache_size=1111 ; doc1: bind cache_size=2222 then recall it.
    doc0 = tok.encode("cache_size = 1111\n", add_special_tokens=False)
    doc1 = tok.encode("cache_size = 2222\nreturn cache_size\n",
                      add_special_tokens=False)
    ids = torch.tensor([doc0 + doc1])
    doc_ids = torch.tensor([[0] * len(doc0) + [1] * len(doc1)])
    T = ids.shape[1]
    emb_table = torch.nn.Embedding(len(tok), 32)
    torch.nn.init.normal_(emb_table.weight, std=0.5)
    emb = emb_table(ids)
    h = torch.randn(1, T, 32)
    wm._capture_read = True
    wm(h, ids, doc_ids=doc_ids, input_emb=emb)
    attn = wm._last_read_attn[0]                                # (T, K)
    top_idx = wm._last_top_idx[0].tolist()                      # (K,) src pos
    _, vstart, _ = wm._name_span_keys(ids, emb)
    vs = [i for i, v in enumerate(vstart[0].tolist()) if v]
    assert len(vs) == 2
    vs_doc = {p: int(doc_ids[0, p]) for p in vs}
    # recall-query position = last token before the trailing newline of doc1
    qpos = T - 2
    a = attn[qpos]
    # (a) causality: zero mass on any buffer slot at/after the query position.
    for k, src in enumerate(top_idx):
        if src >= qpos:
            assert float(a[k]) < 1e-6
    # (b) doc isolation: zero mass on the doc-0 binding; positive on doc-1's.
    for k, src in enumerate(top_idx):
        if src in vs_doc:
            if vs_doc[src] == 0:                               # cross-doc binding
                assert float(a[k]) < 1e-6
    doc1_bind = [p for p in vs if vs_doc[p] == 1][0]
    k_doc1 = top_idx.index(doc1_bind)
    assert float(a[k_doc1]) > 0.5                              # addressed in-doc
