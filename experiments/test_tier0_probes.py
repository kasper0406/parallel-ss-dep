"""CPU tests for the Tier-0 GPU probes (2026-07-13):
  - experiments/probe_state_algebra.py       (DeltaNet state merging)
  - experiments/probe_latent_exposure_bias.py (latent-horizon stratification)

Covers: state-merge ops (mean/sum/normmax) shape+math, cache round-trip
(merged state -> decode proceeds, replicas mutation-isolated), binding-context
construction determinism + overlap, tokenizer prefix protocol, stratification
math on hand-computed per-slot patterns, Wilson CI sanity, and smoke-mode
plumbing for both scripts. All CPU, tiny models / mocks only.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_tier0_probes.py -v
"""
from __future__ import annotations

import argparse
import random

import pytest
import torch

import experiments.probe_state_algebra as psa
import experiments.probe_latent_exposure_bias as peb


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #

def _mk_states(n_layers=2, B=1, H=2, dk=4, dv=4, fill=0.0, conv=True):
    out = []
    for L in range(n_layers):
        rec = torch.full((B, H, dk, dv), fill) + L  # layer-distinct
        entry = {
            "recurrent_state": rec,
            "attn_state": None,
            "conv_state": tuple(torch.full((B, 3, 4), fill + 10 + L)
                                for _ in range(3)) if conv else None,
            "ffn_state": None,
        }
        out.append(entry)
    return out


class _CharTok:
    """Char-level tokenizer: every char -> one id. Prefix-consistent by
    construction (the property line_and_query_ids asserts)."""

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % 200) for c in text]


class _MockModel:
    """Minimal forward_step model: predicts (last input token + 1) % vocab,
    and MUTATES the layer-0 recurrent state in place (so the tests prove
    replicas are isolated from the pristine source states)."""

    def __init__(self, vocab=300):
        self.vocab = vocab

    def forward_step(self, input_id, cache):
        B = input_id.shape[0]
        cache["fla_cache"][0]["recurrent_state"].add_(1.0)  # in-place, on purpose
        logits = torch.zeros(B, 1, self.vocab)
        nxt = (input_id.squeeze(-1) + 1) % self.vocab
        logits[torch.arange(B), 0, nxt] = 1.0
        cache["seen"] = int(cache["seen"]) + 1
        return logits, cache


# =========================================================================== #
# 1. State-merge ops.
# =========================================================================== #

def test_merge_mean_and_sum_shapes_and_math():
    a = _mk_states(fill=1.0)
    b = _mk_states(fill=3.0)
    mean = psa.merge_state_dicts(a, b, "mean")
    s = psa.merge_state_dicts(a, b, "sum")
    for L in range(2):
        ra, rb = a[L]["recurrent_state"], b[L]["recurrent_state"]
        assert mean[L]["recurrent_state"].shape == ra.shape
        assert torch.allclose(mean[L]["recurrent_state"], (ra + rb) / 2)
        assert torch.allclose(s[L]["recurrent_state"], ra + rb)


def test_merge_normmax_per_head_select():
    a = _mk_states(n_layers=1, H=2, fill=0.0)
    b = _mk_states(n_layers=1, H=2, fill=0.0)
    # head 0 big in A, head 1 big in B
    a[0]["recurrent_state"][0, 0] = 5.0
    b[0]["recurrent_state"][0, 1] = 7.0
    m = psa.merge_state_dicts(a, b, "normmax")
    assert torch.allclose(m[0]["recurrent_state"][0, 0],
                          a[0]["recurrent_state"][0, 0])
    assert torch.allclose(m[0]["recurrent_state"][0, 1],
                          b[0]["recurrent_state"][0, 1])


def test_merge_b_only_and_conv_passthrough_and_isolation():
    a = _mk_states(fill=1.0)
    b = _mk_states(fill=3.0)
    m = psa.merge_state_dicts(a, b, "b_only")
    assert torch.allclose(m[0]["recurrent_state"], b[0]["recurrent_state"])
    # conv_state comes from B by default and is a fresh copy
    assert torch.allclose(m[0]["conv_state"][0], b[0]["conv_state"][0])
    m[0]["conv_state"][0].add_(99.0)
    m[0]["recurrent_state"].add_(99.0)
    assert float(b[0]["conv_state"][0].max()) < 50.0     # source untouched
    assert float(b[0]["recurrent_state"].max()) < 50.0
    with pytest.raises(ValueError):
        psa.merge_state_dicts(a, b, "definitely_not_an_op")


def test_merge_conv_from_mean():
    a = _mk_states(fill=1.0)
    b = _mk_states(fill=3.0)
    m = psa.merge_state_dicts(a, b, "mean", conv_from="mean")
    want = (a[0]["conv_state"][0] + b[0]["conv_state"][0]) / 2
    assert torch.allclose(m[0]["conv_state"][0], want)


# =========================================================================== #
# 2. Cache round-trip / replication.
# =========================================================================== #

def test_replicate_states_batch_and_isolation():
    st = _mk_states(fill=2.0)
    rep = psa.replicate_states(st, 3)
    assert rep[0]["recurrent_state"].shape[0] == 3
    assert rep[0]["conv_state"][1].shape[0] == 3
    rep[0]["recurrent_state"].add_(50.0)
    assert float(st[0]["recurrent_state"].max()) < 10.0  # source untouched


def test_states_to_cache_round_trip():
    st = _mk_states()
    cache = psa.states_to_cache(st, seen=17)
    assert cache["seen"] == 17
    assert cache["wm_buf"] is None and cache["lagged_sources"] is None
    got = cache["fla_cache"][1]["recurrent_state"]
    assert torch.allclose(got, st[1]["recurrent_state"])
    assert len(cache["fla_cache"]) == 2


def test_run_queries_decode_proceeds_teacher_forced():
    """Set a (merged) state, decode proceeds, and teacher-forced scoring is
    correct: the mock predicts last-token+1, so a value that continues the
    +1 chain is scored correct and any other value is not."""
    st = _mk_states(fill=0.0)
    queries = [
        # after feeding 7 the mock predicts 8; teacher-forcing 8 predicts 9.
        {"name": "good", "query_ids": [3, 7], "value_ids": [8, 9],
         "alt_value_ids": None, "group": "A"},
        {"name": "half", "query_ids": [3, 7], "value_ids": [8, 5],
         "alt_value_ids": None, "group": "A"},
        {"name": "bad", "query_ids": [3, 7], "value_ids": [4, 4],
         "alt_value_ids": None, "group": "B"},
    ]
    res = psa.run_queries(_MockModel(), st, seen=10, queries=queries,
                          device="cpu")
    by_name = {r["name"]: r for r in res}
    assert by_name["good"]["correct"] == 1
    assert by_name["good"]["per_tok_correct"] == [1, 1]
    assert by_name["half"]["correct"] == 0
    assert by_name["half"]["per_tok_correct"] == [1, 0]
    assert by_name["bad"]["correct"] == 0
    # pristine source states never touched by the (mutating) mock decode
    assert float(st[0]["recurrent_state"].abs().max()) == 0.0


def test_run_queries_groups_mixed_shapes():
    """Queries with different (query_len, value_len) shapes run in separate
    batches but all come back."""
    st = _mk_states()
    queries = [
        {"name": "q1", "query_ids": [3, 7], "value_ids": [8],
         "alt_value_ids": None, "group": "A"},
        {"name": "q2", "query_ids": [3, 7, 9], "value_ids": [10, 11],
         "alt_value_ids": None, "group": "A"},
    ]
    res = psa.run_queries(_MockModel(), st, seen=0, queries=queries,
                          device="cpu")
    assert {r["name"] for r in res} == {"q1", "q2"}
    by_name = {r["name"]: r for r in res}
    assert by_name["q1"]["correct"] == 1     # 7 -> 8
    assert by_name["q2"]["correct"] == 1     # 9 -> 10 -> 11


# =========================================================================== #
# 3. Binding-context construction.
# =========================================================================== #

def test_binding_context_determinism():
    c1 = psa.build_binding_contexts(16, random.Random(3), overlap_frac=0.25)
    c2 = psa.build_binding_contexts(16, random.Random(3), overlap_frac=0.25)
    c3 = psa.build_binding_contexts(16, random.Random(4), overlap_frac=0.25)
    assert c1 == c2
    assert c1 != c3


def test_binding_context_overlap_construction():
    c = psa.build_binding_contexts(16, random.Random(0), overlap_frac=0.25)
    assert len(c["a"]) == 16 and len(c["b"]) == 16
    assert len(c["shared"]) == 4                       # round(0.25 * 16)
    a_vals = dict(c["a"])
    b_vals = dict(c["b"])
    for nm, av, bv in c["shared"]:
        assert av != bv                                # later value must differ
        assert a_vals[nm] == av and b_vals[nm] == bv
    # disjoint variant: no shared names at all
    d = psa.build_binding_contexts(16, random.Random(0), overlap_frac=0.0)
    assert d["shared"] == []
    assert set(dict(d["a"])).isdisjoint(dict(d["b"]))


def test_line_and_query_ids_prefix_protocol():
    tok = _CharTok()
    l_ids, q_ids, v_ids = psa.line_and_query_ids(tok, "va007", "1234")
    assert l_ids[: len(q_ids)] == q_ids                # query = line prefix
    assert v_ids == tok.encode("1234")                 # newline stripped
    assert l_ids == q_ids + v_ids + tok.encode("\n")


def test_build_trial_groups_and_scoring_targets():
    tok = _CharTok()
    tr = psa.build_trial(tok, 8, random.Random(1), overlap_frac=0.25)
    groups = [q["group"] for q in tr["queries"]]
    assert groups.count("shared") == 2                 # round(0.25 * 8)
    assert groups.count("A") == 6 and groups.count("B") == 6
    for q in tr["queries"]:
        if q["group"] == "shared":
            # scored against the B/later value; alt is the A value
            assert q["alt_value_ids"] is not None
            assert q["value_ids"] != q["alt_value_ids"]
        else:
            assert q["alt_value_ids"] is None
    assert tr["ids_a"] and tr["ids_b"]


# =========================================================================== #
# 4. Stratification math (probe 2).
# =========================================================================== #

def test_stratify_hand_computed():
    # 4 records, K=3 (1 = slot decoded correctly)
    recs = [[1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]]
    s = peb.stratify(recs, 3)
    # slot 1: unconditional only
    assert s[1]["n_uncond"] == 4 and s[1]["k_uncond"] == 3
    assert s[1]["n_ok"] == 0 and s[1]["n_bad"] == 0
    # slot 2 | slot1: prefix-ok = recs 0,1,3 -> acc (1+0+1)/3; bad = rec 2 -> 1/1
    assert (s[2]["n_ok"], s[2]["k_ok"]) == (3, 2)
    assert (s[2]["n_bad"], s[2]["k_bad"]) == (1, 1)
    # slot 3 | slots1-2: ok = recs 0,3 -> (1+0)/2; bad = recs 1,2 -> (1+1)/2
    assert (s[3]["n_ok"], s[3]["k_ok"]) == (2, 1)
    assert (s[3]["n_bad"], s[3]["k_bad"]) == (2, 2)
    assert s[3]["n_excluded"] == 0


def test_stratify_none_exclusion():
    # None = unmeasurable slot: excluded from strata whose prefix (or self)
    # contains it, but still counted for other slots.
    recs = [[1, None, 1],
            [1, 1, None],
            [None, 1, 1]]
    s = peb.stratify(recs, 3)
    # slot 2: rec0 self-None -> excluded; rec2 prefix-None -> excluded
    assert (s[2]["n_ok"], s[2]["n_bad"], s[2]["n_excluded"]) == (1, 0, 2)
    # slot 3: rec0 prefix has None; rec1 self None; rec2 prefix None
    assert (s[3]["n_ok"], s[3]["n_bad"], s[3]["n_excluded"]) == (0, 0, 3)
    # unconditional counts skip only the None slot itself
    assert s[2]["n_uncond"] == 2
    assert s[1]["n_uncond"] == 2


def test_merge_strata_pools_by_slot():
    recs_a = [[1, 1], [1, 0]]
    recs_b = [[0, 1]]
    pooled = peb.merge_strata([peb.stratify(recs_a, 2),
                               peb.stratify(recs_b, 2)])
    assert pooled[1]["n_uncond"] == 3 and pooled[1]["k_uncond"] == 2
    assert pooled[2]["n_ok"] == 2 and pooled[2]["k_ok"] == 1
    assert pooled[2]["n_bad"] == 1 and pooled[2]["k_bad"] == 1


def test_reindex_perhop_reconstruction():
    inter_tok = [11, None, 13, 14, None]
    flat = [1, 0, 1]                       # reads for slots 1, 3, 4
    idx = peb.reindex_perhop(flat, inter_tok, R=5)
    assert idx == [1, None, 0, 1, None]
    # shorter R truncates cleanly
    assert peb.reindex_perhop([1], inter_tok, R=1) == [1]
    # leftover flat entries are a hard error (contract violation)
    with pytest.raises(AssertionError):
        peb.reindex_perhop([1, 0, 1, 0], inter_tok, R=3)


def test_indexed_perhop_reads_matches_flat_on_tiny_model():
    """Integration: indexed reads are the flat `latent_perhop_reads` output
    re-indexed onto slots (None exactly at multi-token intermediates)."""
    from experiments.layers import SoftmaxAttention
    from experiments.model import TinyLM
    from experiments.eval_exec_trace_latent_trace import latent_perhop_reads

    THINK_ID = 5
    torch.manual_seed(0)
    m = TinyLM(vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
               attention_cls=SoftmaxAttention, max_T=256,
               output_gate=True, state_readonly_at_think=True,
               use_latent_feedback_adapter=True)
    m.thinking_token_id = THINK_ID
    m.eval()

    prompt = [7, 8, 9, 10]
    inter_tok = [11, None, 13]
    idx = peb.indexed_perhop_reads(m, prompt, 3, THINK_ID, inter_tok, "cpu")
    flat = latent_perhop_reads(m, prompt, 3, THINK_ID, inter_tok, "cpu")
    assert len(idx) == 3 and idx[1] is None
    assert [x for x in idx if x is not None] == flat


# =========================================================================== #
# 5. Wilson CI sanity.
# =========================================================================== #

def test_wilson_ci_sanity():
    lo, hi = psa.wilson_ci(50, 100)
    assert abs(lo - 0.404) < 0.01 and abs(hi - 0.596) < 0.01
    lo0, hi0 = psa.wilson_ci(0, 50)
    assert lo0 == 0.0 and hi0 < 0.1
    lo1, hi1 = psa.wilson_ci(50, 50)
    assert hi1 == 1.0 and lo1 > 0.9
    # tighter with larger n at the same rate
    w_small = psa.wilson_ci(5, 10)
    w_big = psa.wilson_ci(500, 1000)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0])
    # degenerate n=0 -> maximally uninformative, no crash
    assert psa.wilson_ci(0, 0) == (0.0, 1.0)


# =========================================================================== #
# 6. Smoke-mode plumbing (both scripts).
# =========================================================================== #

def test_smoke_plumbing_state_algebra():
    args = argparse.Namespace(n_bindings=32, n_trials=5)
    psa.apply_smoke(args)
    assert args.n_bindings == 8 and args.n_trials == 1


def test_smoke_plumbing_exposure_bias():
    args = argparse.Namespace(rungs="6,7,8", n_per_rung=200, no_lengen=False)
    peb.apply_smoke(args)
    assert args.rungs == "6" and args.n_per_rung == 20
    assert args.no_lengen is True


def test_verdict_thresholds():
    """compute_verdict applies the pre-registered read mechanically."""
    def _summary(acc_ok, n_ok, acc_bad=0.1, n_bad=30):
        return {j: {"prefix_ok": {"n": n_ok, "k": int(acc_ok * n_ok),
                                  "acc": acc_ok, "wilson95": [0, 1]},
                    "prefix_bad": {"n": n_bad, "k": int(acc_bad * n_bad),
                                   "acc": acc_bad, "wilson95": [0, 1]},
                    "uncond": {"n": n_ok + n_bad, "k": 0, "acc": 0.0,
                               "wilson95": [0, 1]},
                    "n_excluded": 0}
                for j in peb.VERDICT_SLOTS}

    v = peb.compute_verdict(_summary(0.85, 40), min_stratum_n=20)
    assert all("ERROR_PROPAGATION" in v[j]["verdict"]
               for j in peb.VERDICT_SLOTS)
    v = peb.compute_verdict(_summary(0.2, 40), min_stratum_n=20)
    assert all("SLOT_DEPTH_COLLAPSE" in v[j]["verdict"]
               for j in peb.VERDICT_SLOTS)
    v = peb.compute_verdict(_summary(0.6, 40), min_stratum_n=20)
    assert all("MIXED" in v[j]["verdict"] for j in peb.VERDICT_SLOTS)
    v = peb.compute_verdict(_summary(0.85, 5), min_stratum_n=20)
    assert all(v[j]["verdict"] == "INSUFFICIENT_N" for j in peb.VERDICT_SLOTS)
