"""Tests for DISCRETE-HASH WM addressing (2026-06-16).

The discrete-key path keys the WM read on a deterministic per-position integer
CODE derived from input_ids (the most-recent `vN` binding number) → onehot
match → zero-cross-talk address on the slot whose binding shares the code. No
new parameters, so default-off is byte-identical and old ckpts load unchanged.
"""
import torch

from experiments.model import WorkingMemory


def _wm(discrete=False, decoupled=True, seed=0):
    torch.manual_seed(seed)
    return WorkingMemory(d_model=32, d_mem=32, mem_size=512, thinking_token_id=99,
                         decoupled_kv=decoupled, discrete_key=discrete)


def test_default_off_no_new_params_and_byte_identical():
    """discrete_key=False adds NO parameters and leaves the forward unchanged."""
    wm_off = _wm(discrete=False)
    wm_on = _wm(discrete=True)
    # No new nn.Parameters introduced by the discrete-key plumbing.
    assert sorted(wm_off.state_dict().keys()) == sorted(wm_on.state_dict().keys())
    assert sum(p.numel() for p in wm_off.parameters()) == \
        sum(p.numel() for p in wm_on.parameters())
    # Forward with discrete off is deterministic & finite (legacy decoupled path).
    torch.manual_seed(1)
    h = torch.randn(2, 16, 32)
    ids = torch.randint(0, 50, (2, 16))
    o1 = wm_off(h, ids)
    o2 = wm_off(h, ids)
    assert torch.equal(o1, o2)
    assert torch.isfinite(o1).all()


def test_identifier_code_and_vstart():
    """code carries the most-recent `vN` number; vstart marks the binding value
    start; references (`print(vN)`) update the code but mark no value start."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(discrete=True)
    wm.set_discrete_key_vocab(tok)
    prog = "v5 = 7634\nv53 = 9016\nprint(v53)\n"
    ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
    toks = [tok.decode([i]) for i in ids[0].tolist()]
    code, vstart = wm._identifier_code_vstart(ids)
    code, vstart = code[0].tolist(), vstart[0].tolist()
    # value-start fires exactly on the first VALUE digit of each binding.
    vs_pos = [i for i, v in enumerate(vstart) if v]
    assert toks[vs_pos[0]] == "7" and code[vs_pos[0]] == 5
    assert toks[vs_pos[1]] == "9" and code[vs_pos[1]] == 53
    # `print(v53)` is a reference → carries code 53, no extra value-start.
    assert len(vs_pos) == 2
    assert code[-2] == 53  # `)` carries the reference's code


def test_lexical_default_off_no_new_params():
    """The lexical-extractor plumbing adds NO parameters (default-off byte-id)."""
    off = WorkingMemory(d_model=32, d_mem=32, mem_size=64, thinking_token_id=99,
                        decoupled_kv=True, discrete_key=False)
    on = WorkingMemory(d_model=32, d_mem=32, mem_size=64, thinking_token_id=99,
                       decoupled_kv=True, discrete_key=True)  # lexical default
    assert sorted(off.state_dict().keys()) == sorted(on.state_dict().keys())
    assert sum(p.numel() for p in off.parameters()) == \
        sum(p.numel() for p in on.parameters())
    assert on.discrete_key_lexical is True


def test_lexical_vectorized_matches_python_loop():
    """The VECTORIZED `_identifier_code_lexical` (tensor ops, on-device) is
    byte-identical to the python-loop reference `_identifier_code_lexical_loop`
    on real recall sequences (synthetic vN multibind + real CONST code)."""
    import json
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = WorkingMemory(d_model=32, d_mem=32, mem_size=2048, thinking_token_id=99,
                       decoupled_kv=True, discrete_key=True)
    wm.set_discrete_key_vocab(tok)

    # a hand-written program exercising prose, references, multi-token names
    progs = [
        "v5 = 7634\nv53 = 9016\nv2 = 1234\nprint(v53)\n",
        ("import os\nCACHE_SIZE = 256\nQUEUE_CAP = 1994\n"
         "def f():\n    return CACHE_SIZE + QUEUE_CAP\n"
         "The constant CACHE_SIZE is assigned the value 256 here.\n"),
    ]
    # plus real records from disk if present
    for path, n in [("data/multibind_compact_heldout_N32.jsonl", 4),
                    ("data/code_recall_heldout.jsonl", 4)]:
        try:
            with open(path) as fh:
                for _, line in zip(range(n), fh):
                    r = json.loads(line)
                    progs.append(r["problem_prompt"] + "\n\n" + r["qwen_completion"])
        except FileNotFoundError:
            pass

    for prog in progs:
        ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
        cv, vv = wm._identifier_code_lexical(ids)
        cl, vl = wm._identifier_code_lexical_loop(ids)
        assert torch.equal(cv, cl), "vectorized code != loop code"
        assert torch.equal(vv, vl), "vectorized vstart != loop vstart"
    # batched (padding) path must also match the loop row-for-row.
    seqs = [tok.encode(p, add_special_tokens=False) for p in progs[:3]]
    Tm = max(len(s) for s in seqs)
    batch = torch.zeros((len(seqs), Tm), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = torch.tensor(s)
    cv, vv = wm._identifier_code_lexical(batch)
    cl, vl = wm._identifier_code_lexical_loop(batch)
    assert torch.equal(cv, cl) and torch.equal(vv, vl)


def test_lexical_code_matches_binding_to_reference_despite_prose():
    """GENERALITY: the SAME identifier text at its binding and at a later
    reference hashes to the SAME code, and intervening PROSE words (not bound
    names) do NOT clobber the carried code — the general version of the vN
    parser's 'only variables create codes' trick. Covers both a synthetic
    `vN = NUM` and a real `CONST = NUM` binding."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = WorkingMemory(d_model=32, d_mem=32, mem_size=512, thinking_token_id=99,
                       decoupled_kv=True, discrete_key=True)
    wm.set_discrete_key_vocab(tok)
    assert wm.discrete_key_lexical
    for binding, name, value in [("v53 = 9016", "v53", "9016"),
                                 ("QUEUE_CAPACITY = 1994", "QUEUE_CAPACITY", "1994")]:
        prog = (f"x1 = 11\n{binding}\nx2 = 22\n"
                f"The constant `{name}` is assigned the value {value} here.\n")
        ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
        toks = [tok.decode([i]) for i in ids[0].tolist()]
        code, vstart = wm._identifier_code_lexical(ids)
        code, vstart = code[0].tolist(), vstart[0].tolist()
        # value-start fires on a digit, with the BOUND NAME's code (not the
        # number's own hash).
        vs = [i for i, v in enumerate(vstart) if v]
        # the binding for `name` is the middle one; find the vstart whose digit
        # text is the first char of `value`.
        bind_vs = [i for i in vs if toks[i].strip() and value.startswith(toks[i].strip()[0])]
        assert bind_vs, f"no value-start for {name}"
        bind_code = code[bind_vs[-1]]
        # the recall position = the token right before the prose `value` digits.
        # find the LAST run of the value digits in the prose tail.
        # the token before it carries the queried name's code.
        # locate the prose digits: scan from the end for the first digit token.
        last_digit_pos = max(i for i, t in enumerate(toks) if t.strip()[:1].isdigit()) \
            if any(t.strip()[:1].isdigit() for t in toks) else -1
        recall_code = code[last_digit_pos - 1]
        assert bind_code == recall_code, \
            f"{name}: binding code {bind_code} != recall-carried {recall_code}"
        # the prose words `is assigned the value` must NOT have clobbered it.


def test_match_existence_true_only_where_a_binding_matches():
    """`_last_match_exists` (B,T) is True at a query position whose carried
    identifier code equals a CAUSALLY-VISIBLE buffered binding, and False before
    that binding's value-start exists / where the code is null. This is the
    free signal the match-existence copy gate keys on."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(discrete=True)              # mem_size=512 >> T → every pos buffered
    wm.set_discrete_key_vocab(tok)
    prog = "hello world\nv53 = 9016\nprint(v53)\n"
    ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
    toks = [tok.decode([i]) for i in ids[0].tolist()]
    _code, vstart = wm._identifier_code_lexical(ids)
    fvs = [i for i, v in enumerate(vstart[0].tolist()) if v][0]   # first val-start
    qpos = len(toks) - 2                 # `)` after print(v53), carries code 53
    h = torch.randn(1, ids.shape[1], 32)
    wm(h, ids)                            # populates _last_match_exists
    me = wm._last_match_exists[0]         # (T,) bool
    # nothing can match before the binding's value-start is causally visible.
    assert not bool(me[:fvs + 1].any()), "match before the binding existed"
    # the reference `print(v53)` query carries code 53 → a real prior match.
    assert bool(me[qpos]), "reference query failed to find its binding"


def test_match_locality_rejects_stale_carried_code():
    """LOCALITY: a code-equality match counts only if the addressing identifier
    was re-mentioned within `discrete_key_match_window` tokens. A binding whose
    name drifted far from the query (carried through prose, never re-mentioned)
    is rejected — this is the cross-family false-match fix — while a LOCAL
    re-mention is accepted. Long-range recall is preserved because the gate is on
    the NAME-mention distance, not the binding/value distance."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(discrete=True)
    wm.set_discrete_key_vocab(tok)
    # binding, then a long unrelated-prose gap with NO re-mention of v53.
    prog = "v53 = 9016\n" + "the cat sat on a mat and ran far away today okay\n" * 2
    ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
    h = torch.randn(1, ids.shape[1], 32)
    qpos = ids.shape[1] - 1               # deep in prose, carries stale code 53
    # window 0 (locality off) → pure code-equality match is True (stale matches).
    wm.discrete_key_match_window = 0
    wm(h, ids)
    assert bool(wm._last_match_exists[0, qpos]), "code-equality match expected"
    # window 8 → the bound name is >8 tokens back → match rejected (no harm).
    wm.discrete_key_match_window = 8
    wm(h, ids)
    assert not bool(wm._last_match_exists[0, qpos]), "stale match not rejected"
    # a LOCAL re-mention (print(v53) right after the binding) is still accepted.
    prog2 = "v53 = 9016\nprint(v53)\n"
    ids2 = torch.tensor([tok.encode(prog2, add_special_tokens=False)])
    h2 = torch.randn(1, ids2.shape[1], 32)
    wm(h2, ids2)
    toks2 = [tok.decode([i]) for i in ids2[0].tolist()]
    assert bool(wm._last_match_exists[0, len(toks2) - 2]), "local re-mention lost"


def test_match_existence_gating_suppresses_copy_on_no_match():
    """The match-existence copy gate: with copy_require_match ON, the copy head
    fires at a MATCH position but is SUPPRESSED at a NO-MATCH position (falls
    back to the plain LM); with it OFF, the copy fires at BOTH (the pre-fix
    cross-family over-firing). Tested via `_apply_copy_head` with hand-set WM
    stashes (no trunk forward / tokenizer needed)."""
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    from experiments.model import TinyLM
    model = TinyLM(
        vocab_size=10, d_model=8, n_layers=1, n_heads=2, d_head=4,
        attention_cls=DeltaNetAttention, max_T=32, output_gate=True,
        use_memory=True, mem_size=32, thinking_token_id=9, pad_token_id=0,
        mem_decoupled_kv=True, mem_discrete_key=True, use_copy_head=True,
    )
    mem = model.memory
    with torch.no_grad():
        model.copy_head.gate.bias.fill_(10.0)     # g ≈ 1 → copy clearly active
        model.copy_head.gate.weight.zero_()
    B, T, K, V, d = 1, 6, 2, 10, 8
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])      # source tokens to copy from
    h = torch.randn(B, T, d)
    lm_logits = torch.randn(B, T, V)
    rm = torch.zeros(B, T, dtype=torch.bool)
    rm[0, 3] = True                                # MATCH query
    rm[0, 4] = True                                # NO-MATCH query
    # buffer slots point at source positions 0 and 1; attention picks slot 0.
    mem._last_top_idx_buf = torch.tensor([[0, 1]])
    attn = torch.zeros(B, T, K)
    attn[0, 3, 0] = 1.0                            # pos 3 copies src @ pos 0
    attn[0, 4, 0] = 1.0                            # pos 4 would copy src @ pos 1
    mem._last_read_attn_grad = attn
    mem._last_match_exists = torch.tensor([[False, False, False, True, False, False]])

    def dist(out, t):
        return torch.softmax(out[0, t], dim=-1)
    plain = torch.softmax(lm_logits, dim=-1)

    mem.copy_require_match = True
    out_on = model._apply_copy_head(lm_logits.clone(), h, ids, rm)
    # MATCH (pos 3): copy fires → distribution shifts away from plain LM.
    assert (dist(out_on, 3) - plain[0, 3]).abs().sum() > 0.5
    # NO-MATCH (pos 4): copy suppressed → distribution == plain LM.
    assert torch.allclose(dist(out_on, 4), plain[0, 4], atol=1e-5)

    mem.copy_require_match = False
    out_off = model._apply_copy_head(lm_logits.clone(), h, ids, rm)
    # with gating OFF the copy fires at the NO-MATCH position too (the regression).
    assert (dist(out_off, 4) - plain[0, 4]).abs().sum() > 0.5


def test_discrete_addressing_is_sharp_and_zero_crosstalk():
    """A recall query keys onehot(code) → attention concentrates ~1.0 on the
    val-start of the binding that shares its variable number."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    wm = _wm(discrete=True)
    wm.set_discrete_key_vocab(tok)
    wm._capture_read = True
    prog = "v5 = 7634\nv53 = 9016\nv2 = 1234\nprint(v53)\n"
    ids = torch.tensor([tok.encode(prog, add_special_tokens=False)])
    toks = [tok.decode([i]) for i in ids[0].tolist()]
    vstart_v53 = toks.index("9")        # first value digit of v53's binding
    qpos = len(toks) - 2                 # `)` after print(v53), code=53
    h = torch.randn(1, ids.shape[1], 32)
    wm(h, ids)
    attn = wm._last_read_attn[0]         # (T, K)
    top_idx = wm._last_top_idx[0]        # (K,)
    a = attn[qpos]
    src = int(top_idx[int(a.argmax())])
    assert src == vstart_v53
    # mass on the correct slot is essentially 1.0 (zero cross-talk).
    assert float(a[top_idx == vstart_v53].sum()) > 0.99
