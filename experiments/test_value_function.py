"""Tests for the interpreter-as-value-function kill-test harness
(`experiments/value_function.py`, SEARCH_NATIVE_PLAN_2026_07_19.md Phase 2).

CPU-only and fast: the truncation/mutation/ranking machinery is pure Python +
sys.settrace (no CUDA / FLA / HF / model), the log-prob scorer is checked
against a hand-rolled cross-entropy on a tiny CPU nn.Module. The 3-item GPU
end-to-end smoke is opt-in (env VALUE_FN_GPU_SMOKE=1) so a plain pytest run
never touches the GPU.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_value_function.py -v
"""
from __future__ import annotations

import math
import os
import random

import pytest
import torch
import torch.nn.functional as F

from experiments.value_function import (
    execute_synthetic, split_top_level, choose_split, line_mutations,
    enumerate_distractors, build_item, _mean_cont_logprob,
    teacher_forced_mean_logprob, top_indices, success_strict, success_random,
    bootstrap_ci, _full_program_text, render_exec_prompt,
)


# --------------------------------------------------------------------------- #
# Fixtures: a hand-built synthetic record with a known trace.
# --------------------------------------------------------------------------- #

SETUP = "x = 2\na = 1\nb = 3\ntbl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]\n"
# x: 2 -> +3 -> 5 -> *2 -> 0 -> -4 -> 6 -> =7 -> 7   (a/b lines are distractors)
TRACED = ("x = (x + 3) % 10\n"
          "a = (a + 1) % 10\n"
          "x = (x * 2) % 10\n"
          "b = 5\n"
          "x = (x - 4) % 10\n"
          "x = 7\n")


def _make_rec(task_id="exec_synth/n4/test0"):
    events = execute_synthetic(SETUP, TRACED, "x")
    return {
        "task_id": task_id, "source": "synthetic", "rung": len(events),
        "answer": events[-1], "intermediates": list(events),
        "setup_src": SETUP, "traced_src": TRACED, "tracked_var": "x",
    }


def test_fixture_trace_is_what_we_think():
    events = execute_synthetic(SETUP, TRACED, "x")
    assert events == [5, 0, 6, 7]


# --------------------------------------------------------------------------- #
# Truncation: prefix + true-continuation reconstructs the program exactly.
# --------------------------------------------------------------------------- #

def test_split_top_level_reconstructs_exactly():
    chunks = split_top_level(TRACED)
    assert "".join(chunks) == TRACED
    # one chunk per top-level statement (all single-line here)
    assert len(chunks) == 6


def test_split_top_level_keeps_multiline_blocks_intact():
    src = ("if x >= 3:\n    x = (x - 1) % 10\nelse:\n    x = (x + 2) % 10\n"
           "for _i in range(2):\n    x = (x + 1) % 10\n")
    chunks = split_top_level(src)
    assert "".join(chunks) == src
    assert len(chunks) == 2  # the if/else block and the for-loop
    assert chunks[0].startswith("if x >= 3:")
    assert chunks[1].startswith("for _i in range(2):")


def test_choose_split_reconstructs_and_hits_event_counts():
    events = execute_synthetic(SETUP, TRACED, "x")
    K = len(events)
    for target_k in range(1, K):
        split = choose_split(SETUP, TRACED, "x", K, target_k)
        assert split is not None
        prefix, cont, actual_k = split
        # exact textual reconstruction of the traced body
        assert prefix + cont == TRACED
        # and of the whole program
        assert SETUP + prefix + cont == SETUP + TRACED
        # the prefix really produces `actual_k` tracked events, cont the rest
        pe = execute_synthetic(SETUP, prefix, "x")
        assert len(pe) == actual_k
        assert 1 <= actual_k <= K - 1
        # here every boundary is hittable, so actual_k == target_k
        assert actual_k == target_k


def test_choose_split_none_when_no_interior_boundary():
    # a single loop that emits all K events in ONE chunk -> no interior split
    setup = "x = 0\n"
    traced = "for _i in range(3):\n    x = (x + 1) % 10\n"
    events = execute_synthetic(setup, traced, "x")
    assert events == [1, 2, 3]
    assert choose_split(setup, traced, "x", len(events), 1) is None


# --------------------------------------------------------------------------- #
# Mutation surface + validity.
# --------------------------------------------------------------------------- #

def test_line_mutations_arith_const_and_opswap():
    muts = line_mutations("x = (x + 3) % 10")
    bodies = {b for b, _ in muts}
    classes = {c for _, c in muts}
    assert "x = (x + 5) % 10" in bodies       # a constant edit
    assert "x = (x - 3) % 10" in bodies       # the +/- op swap
    assert "x = (x + 3) % 10" not in bodies   # never the identity
    assert classes == {"const", "opswap"}


def test_line_mutations_assign_const():
    muts = line_mutations("x = 7")
    bodies = {b for b, _ in muts}
    assert "x = 3" in bodies and "x = 7" not in bodies
    assert all(c == "const" for _, c in muts)


def test_line_mutations_preserves_indent():
    muts = line_mutations("    x = (x - 2) % 10")
    assert all(b.startswith("    x = ") for b, _ in muts)


def test_line_mutations_none_for_lookup_or_header():
    assert line_mutations("x = tbl[x]") == []
    assert line_mutations("if x >= 3:") == []
    assert line_mutations("a = (a + 1) % 10") == []  # not the tracked var


def test_enumerate_distractors_valid_distinct_and_wrong():
    events = execute_synthetic(SETUP, TRACED, "x")
    true_answer = events[-1]
    split = choose_split(SETUP, TRACED, "x", len(events), 1)
    prefix, cont, _ = split
    rng = random.Random(123)
    ds = enumerate_distractors(SETUP, prefix, cont, "x", true_answer, rng, 3)
    assert len(ds) == 3
    texts = set()
    for d in ds:
        # (a) syntactically valid full program
        full = prefix + d["cont_text"]
        compile(SETUP + full, "<t>", "exec")
        # (b) executes, (c) answer differs from the true answer
        re_events = execute_synthetic(SETUP, full, "x")
        assert re_events[-1] == d["answer"]
        assert d["answer"] != true_answer
        # (d) distinct continuations, and different from the true continuation
        assert d["cont_text"] != cont
        texts.add(d["cont_text"])
    assert len(texts) == 3


def test_enumerate_distractors_deterministic_under_seed():
    events = execute_synthetic(SETUP, TRACED, "x")
    prefix, cont, _ = choose_split(SETUP, TRACED, "x", len(events), 2)
    a = enumerate_distractors(SETUP, prefix, cont, "x", events[-1],
                              random.Random(7), 3)
    b = enumerate_distractors(SETUP, prefix, cont, "x", events[-1],
                              random.Random(7), 3)
    assert [d["cont_text"] for d in a] == [d["cont_text"] for d in b]


# --------------------------------------------------------------------------- #
# build_item end-to-end (CPU, no model).
# --------------------------------------------------------------------------- #

def test_build_item_shape_and_semantics():
    rec = _make_rec()
    kind, item = build_item(rec, seed=0)
    assert kind == "item"
    assert len(item["candidates"]) == 4
    # exactly one true candidate, at true_index
    trues = [i for i, c in enumerate(item["candidates"]) if c["is_true"]]
    assert trues == [item["true_index"]]
    # the true continuation reconstructs the original program and hits answer
    true_c = item["candidates"][item["true_index"]]
    assert item["prefix_traced"] + true_c["cont_text"] == TRACED
    assert execute_synthetic(SETUP, TRACED, "x")[-1] == item["true_answer"]
    # every distractor really executes to its (wrong) recorded answer
    for c in item["candidates"]:
        full = item["prefix_traced"] + c["cont_text"]
        ev = execute_synthetic(SETUP, full, "x")
        assert ev[-1] == c["answer"]
        if not c["is_true"]:
            assert c["answer"] != item["true_answer"]


def test_build_item_deterministic():
    rec = _make_rec()
    k1, i1 = build_item(rec, seed=0)
    k2, i2 = build_item(rec, seed=0)
    assert [c["cont_text"] for c in i1["candidates"]] == \
           [c["cont_text"] for c in i2["candidates"]]
    assert i1["true_index"] == i2["true_index"]


def test_build_item_skips_non_synthetic():
    rec = _make_rec()
    rec["source"] = "mbpp"
    assert build_item(rec, seed=0) == ("skip", "non_synthetic")


def test_full_program_text_normalization_matches_template():
    rec = _make_rec()
    _, item = build_item(rec, seed=1)
    true_c = item["candidates"][item["true_index"]]
    prog = _full_program_text(item, true_c["cont_text"])
    # normalized program == setup(rstrip)\n + traced(rstrip)
    assert prog == SETUP.rstrip("\n") + "\n" + TRACED.rstrip("\n")
    prompt = render_exec_prompt(prog, "x")
    assert prompt.startswith("You are given the following Python program.\n")
    assert "final value of x" in prompt


# --------------------------------------------------------------------------- #
# Ranking math incl. tie handling.
# --------------------------------------------------------------------------- #

def test_top_indices_and_strict_unique_max():
    keys = [(1, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)]
    assert top_indices(keys) == [0]
    assert success_strict(keys, 0) == 1
    assert success_strict(keys, 1) == 0


def test_strict_counts_ties_as_failure():
    keys = [(1, float("-inf")), (1, float("-inf")), (0, 0.0), (0, 0.0)]
    assert top_indices(keys) == [0, 1]
    assert success_strict(keys, 0) == 0  # tie at the top -> failure


def test_random_tiebreak_true_not_in_top_is_zero():
    keys = [(0, 0.0), (1, 0.0), (1, 0.0), (1, 0.0)]
    for s in range(20):
        assert success_random(keys, 0, random.Random(s)) == 0


def test_random_tiebreak_is_uniform_over_tied_top():
    keys = [(1, float("-inf")), (1, float("-inf")), (0, 0.0), (0, 0.0)]
    wins = sum(success_random(keys, 0, random.Random(s)) for s in range(2000))
    assert 0.4 < wins / 2000 < 0.6  # ~1/2 for a 2-way tie


def test_ranking_scalar_logprob_keys_pick_max():
    keys = [(-1.0,), (-2.0,), (-0.5,), (-3.0,)]
    assert success_strict(keys, 2) == 1
    assert success_strict(keys, 0) == 0


def test_executor_secondary_tiebreak_resolves_on_logprob():
    # both match_bit==1 but the higher answer-logprob wins uniquely
    keys = [(1, -0.2), (1, -0.9), (0, 0.0), (0, 0.0)]
    assert top_indices(keys) == [0]
    assert success_strict(keys, 0) == 1
    assert success_strict(keys, 1) == 0


def test_bootstrap_ci_bounds():
    assert bootstrap_ci([1, 1, 1, 1], 200, seed=0) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([0, 0, 0, 0], 200, seed=0) == (0.0, 0.0, 0.0)
    m, lo, hi = bootstrap_ci([0, 1] * 50, 1000, seed=0)
    assert abs(m - 0.5) < 1e-9
    assert lo <= m <= hi
    assert bootstrap_ci([], 100, seed=0) == (None, None, None)


# --------------------------------------------------------------------------- #
# Log-prob scorer vs a hand-rolled cross-entropy (tiny CPU model).
# --------------------------------------------------------------------------- #

class _ByteTok:
    """Byte-level tokenizer: id == byte value (vocab 256). Concatenation-safe
    (no merges), so encode(prefix) is always a prefix of encode(prefix+cont)."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab=256, d=16):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def test_teacher_forced_mean_logprob_matches_cross_entropy():
    tok, model = _ByteTok(), _TinyModel()
    model.eval()
    prefix = "x = 2\nx = (x + 3) % 10\n"
    cont = "x = (x - 4) % 10\nx = 7\n"
    got = teacher_forced_mean_logprob(model, tok, prefix, prefix + cont)

    # independent hand-roll via F.cross_entropy
    fid = tok.encode(prefix + cont)
    start = len(tok.encode(prefix))
    with torch.no_grad():
        logits = model(torch.tensor([fid]))[0]
    inp = logits[start - 1:len(fid) - 1]          # predict positions [start, end)
    tgt = torch.tensor(fid[start:])
    ce = F.cross_entropy(inp, tgt, reduction="mean")
    assert math.isclose(got, float(-ce), rel_tol=1e-5, abs_tol=1e-5)


def test_mean_cont_logprob_matches_manual_gather():
    torch.manual_seed(1)
    ids = [5, 9, 2, 7, 3]
    logits = torch.randn(1, len(ids), 12)
    start = 2
    got = _mean_cont_logprob(logits, ids, start)
    logp = torch.log_softmax(logits[0], dim=-1)
    manual = sum(float(logp[j - 1, ids[j]]) for j in range(start, len(ids)))
    manual /= (len(ids) - start)
    assert math.isclose(got, manual, rel_tol=1e-6, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# GPU-optional end-to-end smoke (opt-in: VALUE_FN_GPU_SMOKE=1).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    os.environ.get("VALUE_FN_GPU_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="opt-in GPU smoke (set VALUE_FN_GPU_SMOKE=1 with a CUDA device)")
def test_gpu_smoke_three_items():
    import pathlib
    from experiments.value_function import (
        load_items, evaluate)
    from experiments.eval_exec_trace_text import load_eval_model
    ckpt = "checkpoints/stageA_executor.pt"
    if not pathlib.Path(ckpt).exists():
        pytest.skip(f"{ckpt} not present")
    items_by_k, _ = load_items([4], n_per_k=3, prefix="data/exec_trace_heldout",
                               seed=0)
    assert len(items_by_k[4]) == 3
    model, cfg, tok, eos_id = load_eval_model(ckpt)
    results, dbg = evaluate(model, tok, eos_id, items_by_k, [4], seed=0,
                            n_boot=100)
    ex = results["per_k"][4]["executor"]["ties_as_failures"]
    assert ex["n"] == 3 and ex["acc"] is not None
