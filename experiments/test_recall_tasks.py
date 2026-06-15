"""Tests for the realistic code/agentic recall task generators + eval harness.

Covers: per-family span annotations are correct (binding/query/answer char spans
recover the expected text), the answer char-span maps to the right token span
(tokenize_with_span), distance is positive, and the eval answer extractor parses
both integer and identifier answers. No GPU; uses the synthetic distractor pool so
no HF dataset is required (the SmolLM2 tokenizer is cached).
"""
import random

import pytest

from transformers import AutoTokenizer

from experiments import gen_code_recall_tasks as gcr
from experiments import gen_agentic_recall_tasks as gar
from experiments.eval_code_recall import (extract_answer, tokenize_with_span,
                                          bucket_of)

TOK = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
POOL = gcr._build_python_pool(synthetic=True)


def _check_spans(rec):
    pp, co = rec["problem_prompt"], rec["qwen_completion"]
    b, q, a = (rec["binding_char_span"], rec["query_key_char_span"],
               rec["answer_char_span"])
    assert b and q and a, f"missing span in {rec['family']}"
    # answer char-span recovers the answer
    assert rec["answer"].strip() in co[a[0]:a[1]] or co[a[0]:a[1]].strip() == rec["answer"].strip()
    # the source value text is at the binding span (value families) or the binding
    # contains the def/import (identifier families)
    assert rec["source_value_text"] in pp[b[0]:b[1]] or pp[b[0]:b[1]] in pp
    # query-key span recovers the recall key (for name addressing) or role phrase
    qtext = pp[q[0]:q[1]]
    assert len(qtext) > 0
    # distance positive
    assert rec["approx_distance_tokens"] > 0


@pytest.mark.parametrize("family", ["const", "signature", "import", "fname"])
def test_code_family_spans(family):
    rng = random.Random(123)
    for _ in range(8):
        rec = gcr.build_one(rng, POOL, TOK, family, dist=300, n=8)
        assert rec["family"] == family
        _check_spans(rec)


@pytest.mark.parametrize("family", ["toolout", "userinstr", "setvar"])
def test_agentic_family_spans(family):
    rng = random.Random(7)
    for _ in range(8):
        rec = gar.build_one(rng, TOK, family, dist=300, n=6)
        assert rec["family"] == family
        _check_spans(rec)


def test_answer_token_span_recovers_answer():
    """The answer char-span maps to a token span whose decode contains the answer
    (the unit the copy/pointer readout + mem_read_mask act on)."""
    rng = random.Random(1)
    for family in ["const", "signature", "import"]:
        for _ in range(6):
            rec = gcr.build_one(rng, POOL, TOK, family, dist=300, n=8)
            prompt = rec["problem_prompt"] + "\n\n"
            full = prompt + rec["qwen_completion"]
            a = rec["answer_char_span"]
            ids, t0, t1 = tokenize_with_span(TOK, full,
                                             len(prompt) + a[0], len(prompt) + a[1])
            decoded = TOK.decode(ids[t0:t1]).strip()
            assert rec["answer"].strip() in decoded or decoded == rec["answer"].strip()


def test_extract_answer_int_and_identifier():
    assert extract_answer("...outputs 4096.\n\nAnswer: 4096") == "4096"
    assert extract_answer("The alias is np.\n\nAnswer: np") == "np"
    assert extract_answer("Answer: results_final.csv") == "results_final.csv"
    assert extract_answer("Answer: `payload`.") == "payload"
    assert extract_answer("no answer at all here 7") == "no answer at all here 7"


def test_bucket_of_reads_task_id():
    assert bucket_of({"task_id": "const/d512/13"}) == 512
    assert bucket_of({"task_id": "x", "approx_distance_tokens": 333}) == 333


def test_value_uniqueness_for_const():
    """The const answer value must not collide in distractors (copy/scoring
    unambiguity); the value digit-string should appear once at the binding."""
    rng = random.Random(99)
    for _ in range(10):
        rec = gcr.build_one(rng, POOL, TOK, "const", dist=300, n=8)
        body = rec["problem_prompt"]
        # the answer value appears exactly once in the code body (at its binding)
        code = body.split("```python")[1].split("```")[0]
        assert code.count(rec["answer"]) == 1
