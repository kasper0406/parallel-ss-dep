"""Tests for eval_longctx_recall.py — the held-out long-context recall
eval (2026-05-20). Covers the answer-extraction, distance-bucket, and
gold-answer helpers, plus the generator's exact-distance bucket mode.

These are the scoring primitives: if they are wrong, the whole recall
curve is wrong, so they are pinned tightly.
"""
import json
import random
import subprocess
import sys
import pathlib

from experiments.eval_longctx_recall import (
    extract_predicted_answer, bucket_of, gold_answer)
from experiments.gen_longctx_recall_tasks import (
    _gen_var_binding_long, _to_jsonl_record)


# --------------------------------------------------------------------
# extract_predicted_answer
# --------------------------------------------------------------------
def test_extract_answer_scaffold_wins():
    """The SFT target ends with `Answer: N` — that pattern takes
    priority over any other integer in the text."""
    text = ("The variable was set to 5 early, then 8 distractor lines "
            "ran. Answer: 1234\n")
    assert extract_predicted_answer(text) == "1234"


def test_extract_answer_case_insensitive():
    assert extract_predicted_answer("answer: 77") == "77"


def test_extract_answer_negative():
    assert extract_predicted_answer("Answer: -42") == "-42"


def test_extract_answer_fallback_last_integer():
    """No `Answer:` scaffold → fall back to the last bare integer."""
    assert extract_predicted_answer("it prints 3 then finally 99") == "99"


def test_extract_answer_none_when_no_integer():
    assert extract_predicted_answer("no number here at all") is None
    assert extract_predicted_answer("") is None


def test_extract_answer_first_scaffold_when_multiple():
    """If the model emits the scaffold more than once, take the first
    (the model's committed answer, before any rambling)."""
    assert extract_predicted_answer("Answer: 5 ... oops Answer: 8") == "5"


# --------------------------------------------------------------------
# bucket_of
# --------------------------------------------------------------------
def test_bucket_from_task_id():
    """Bucket-mode task_ids encode the exact distance: longctx/d768/12."""
    assert bucket_of({"task_id": "longctx/d768/12"}) == 768
    assert bucket_of({"task_id": "longctx/d64/0"}) == 64


def test_bucket_fallback_to_approx():
    """Random-mode records have no /dNNN/ — fall back to the rounded
    approx_distance_tokens."""
    rec = {"task_id": "longctx/431", "approx_distance_tokens": 612}
    assert bucket_of(rec) == 612


# --------------------------------------------------------------------
# gold_answer
# --------------------------------------------------------------------
def test_gold_answer_explicit_field():
    assert gold_answer({"answer": 1234}) == "1234"
    assert gold_answer({"answer": "55"}) == "55"


def test_gold_answer_parsed_from_extracted_code():
    """Pre-2026-05-20 records have no `answer` field — parse print(N)."""
    assert gold_answer({"extracted_code": "print(8080)"}) == "8080"


def test_gold_answer_matches_generator_output():
    """Round-trip: the record a generator emits must score against
    itself."""
    rng = random.Random(3)
    for _ in range(20):
        ex = _gen_var_binding_long(rng, target_distance_tokens=300)
        rec = _to_jsonl_record("longctx/d300/0", ex)
        assert gold_answer(rec) == ex["answer"]
        # The solution prose ends with `Answer: N` — extraction must
        # recover the same value.
        assert extract_predicted_answer(rec["qwen_completion"]) == ex["answer"]


# --------------------------------------------------------------------
# generator bucket mode
# --------------------------------------------------------------------
def test_record_has_answer_field():
    rng = random.Random(1)
    ex = _gen_var_binding_long(rng, 500)
    rec = _to_jsonl_record("longctx/d500/0", ex)
    assert rec["answer"] == ex["answer"]


def test_bucket_mode_generates_fixed_counts(tmp_path):
    """--distance_buckets + --per_bucket emits exactly per_bucket
    examples per distance, with the bucket encoded in the task_id."""
    out = tmp_path / "heldout.jsonl"
    repo = pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "experiments/gen_longctx_recall_tasks.py",
         "--out", str(out), "--distance_buckets", "64,256,512",
         "--per_bucket", "7", "--seed", "0"],
        cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    records = [json.loads(l) for l in out.read_text().splitlines() if l]
    assert len(records) == 21          # 3 buckets × 7
    by_bucket: dict[int, int] = {}
    for rec in records:
        by_bucket[bucket_of(rec)] = by_bucket.get(bucket_of(rec), 0) + 1
    assert by_bucket == {64: 7, 256: 7, 512: 7}
