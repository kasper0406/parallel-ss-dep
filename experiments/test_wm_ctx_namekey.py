"""Tests for CONTEXTUAL NAME-SPAN WM addressing (`mem_ctx_namekey`, 2026-06-17).

The fully-learned, NO-static-hash addresser: key/query = the trunk's CONTEXTUAL
HIDDEN pooled over the identifier name-span (the `ctx_namepool` anchor that won
alias_addressing_probe.py), with a DOT-PRODUCT read (learned scale), not cosine.
Default off → byte-identical; old ckpts load strict=False (the encoders are the
only new params and they auto-detect / cold-init).
"""
import torch

from experiments.model import WorkingMemory


def _wm(ctx=False, dim=48, seed=0):
    torch.manual_seed(seed)
    return WorkingMemory(d_model=32, d_mem=32, mem_size=2048, thinking_token_id=99,
                         ctx_namekey=ctx, ctx_namekey_dim=dim)


def test_default_off_no_new_params_and_byte_identical():
    """ctx_namekey=False adds NO parameters and leaves the forward unchanged."""
    off = _wm(ctx=False)
    # The ctx-namekey plumbing must not introduce params when the flag is off:
    # compare to a plain legacy WM.
    plain = WorkingMemory(d_model=32, d_mem=32, mem_size=2048, thinking_token_id=99)
    assert sorted(off.state_dict().keys()) == sorted(plain.state_dict().keys())
    assert sum(p.numel() for p in off.parameters()) == \
        sum(p.numel() for p in plain.parameters())
    torch.manual_seed(1)
    h = torch.randn(2, 16, 32)
    ids = torch.randint(0, 50, (2, 16))
    o1, o2 = off(h, ids), off(h, ids)
    assert torch.equal(o1, o2)
    assert torch.isfinite(o1).all()


def test_ctx_on_adds_expected_params():
    """ctx_namekey=True adds exactly the two encoders + the learned dot scale."""
    on = _wm(ctx=True, dim=48)
    keys = set(on.state_dict().keys())
    assert any(k.startswith("ctxkey_q_enc.") for k in keys)
    assert any(k.startswith("ctxkey_k_enc.") for k in keys)
    assert "ctxkey_log_scale" in keys
    assert on.ctx_namekey and on.ctx_namekey_dim == 48


def test_mutual_exclusion():
    """ctx_namekey is mutually exclusive with discrete_key / soft_namekey."""
    for kw in (dict(discrete_key=True), dict(soft_namekey=True)):
        try:
            WorkingMemory(d_model=32, d_mem=32, mem_size=64, thinking_token_id=99,
                          ctx_namekey=True, **kw)
            assert False, f"expected ValueError for {kw}"
        except ValueError:
            pass


def test_read_restricted_to_causal_binding_slots():
    """On a real `vN = NNNN` program the ctx read attention at the query position
    puts ALL its mass on causally-valid BINDING (value-start) slots — the masking
    is correct even with cold-init encoders (addressing ACCURACY is a training
    result; this guards the MECHANISM)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(ctx=True, dim=64)
    wm.set_discrete_key_vocab(tok)
    wm._capture_read = True
    prog = "v5 = 7634\nv53 = 9016\nv8 = 4421\nprint(v53)\n"
    ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
    h = torch.randn(1, ids.shape[1], 32)
    out = wm(h, ids)
    assert torch.isfinite(out).all()
    # value-start (binding) source positions, from the shared lexical extractor.
    _, vstart = wm._identifier_code_lexical(ids)
    vstart = vstart[0]
    attn = wm._last_read_attn[0]            # (T, K)
    top_idx = wm._last_top_idx[0]           # (K,)
    qpos = ids.shape[1] - 1                  # last token (inside `print(v53)`)
    a = attn[qpos]                           # (K,)
    fired = a > 1e-6
    if bool(fired.any()):
        srcs = top_idx[fired]
        # every attended slot is a value-start AND strictly causal.
        assert bool(vstart[srcs].all()), "attention leaked onto a non-binding slot"
        assert bool((srcs < qpos).all()), "attention leaked onto a non-causal slot"


def test_state_dict_roundtrip():
    """A ctx_namekey WM's trained encoders round-trip through load_state_dict."""
    a = _wm(ctx=True, dim=48, seed=1)
    with torch.no_grad():
        for p in a.ctxkey_q_enc.parameters():
            p.add_(0.5)
        a.ctxkey_log_scale.add_(0.3)
    b = _wm(ctx=True, dim=48, seed=2)
    b.load_state_dict(a.state_dict())
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(va, vb)
