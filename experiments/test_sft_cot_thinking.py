"""Tests for the Phase-4 CoT-thinking SFT example builder
(`experiments.sft_code.build_example_with_cot_thinking`) and the
loader/dispatch path (`load_distilled_jsonl_with_cot`).

The contract these tests pin (Option A, THINKING_DECISIONS.md
2026-05-26):
  * CoT prose is REPLACED by N consecutive [THINKING] tokens, not
    interleaved or learned-by-content.
  * Prompt + think positions get -100 labels; solution + eos get real
    ids — so the model is trained ONLY to emit the solution, after
    spending a think budget.
  * Plain rows (no `prepare_for_thinking` flag) fall through to the
    unchanged build_example path.
"""
from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from experiments.sft_code import (
    build_example,
    build_example_with_cot_thinking,
    load_distilled_jsonl_with_cot,
)


THINK_ID = 9999


class _FakeTokenizer:
    """Whitespace tokenizer producing fixed integer ids. Avoids HF
    download in a unit test. Ids are stable across calls (so a token
    that appears in both the CoT and the solution tokenizes to the
    same id, matching real tokenizer behaviour).

    eos_token_id is a constant in the "real" range so it's
    distinguishable from -100 and from THINK_ID.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.eos_token_id = 1

    def _id(self, s: str) -> int:
        if s not in self._vocab:
            # Reserve 0..9 for special-ish ids (0=pad, 1=eos, etc).
            self._vocab[s] = 10 + len(self._vocab)
        return self._vocab[s]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(tok) for tok in text.split()]


def test_cot_example_length_is_prompt_plus_cot_plus_solution_plus_eos() -> None:
    tok = _FakeTokenizer()
    prompt = "compute the sum of two numbers"
    cot = "first read both then add them"
    sol = "def add ( a , b ) : return a + b"

    full, labels = build_example_with_cot_thinking(
        prompt, cot, sol, tok, THINK_ID, max_len=512,
    )

    n_prompt = len(tok.encode(f"# {prompt}\n"))
    n_cot = len(tok.encode(cot))
    n_sol = len(tok.encode(sol + "\n")) + 1  # +1 eos
    assert len(full) == n_prompt + n_cot + n_sol, (
        f"got {len(full)}, want {n_prompt} + {n_cot} + {n_sol}")
    assert len(labels) == len(full)


def test_cot_positions_are_all_thinking_token_id() -> None:
    tok = _FakeTokenizer()
    prompt = "p1 p2 p3"
    cot = "c1 c2 c3 c4 c5"
    sol = "s1 s2"
    full, _ = build_example_with_cot_thinking(
        prompt, cot, sol, tok, THINK_ID, max_len=512)
    n_prompt = len(tok.encode(f"# {prompt}\n"))
    n_cot = len(tok.encode(cot))
    # The N tokens AFTER the prompt and BEFORE the solution must all be
    # the think id.
    cot_slice = full[n_prompt:n_prompt + n_cot]
    assert cot_slice == [THINK_ID] * n_cot, cot_slice


def test_labels_mask_prompt_and_cot_keep_solution_and_eos() -> None:
    tok = _FakeTokenizer()
    prompt = "p1 p2 p3"
    cot = "c1 c2 c3"
    sol = "s1 s2 s3"
    full, labels = build_example_with_cot_thinking(
        prompt, cot, sol, tok, THINK_ID, max_len=512)
    n_prompt = len(tok.encode(f"# {prompt}\n"))
    n_cot = len(tok.encode(cot))
    n_sol_with_eos = len(tok.encode(sol + "\n")) + 1
    # Prompt + CoT positions must be -100.
    for i in range(n_prompt + n_cot):
        assert labels[i] == -100, (i, labels[i])
    # Solution positions must equal input ids verbatim.
    for i in range(n_prompt + n_cot, n_prompt + n_cot + n_sol_with_eos):
        assert labels[i] == full[i], (i, labels[i], full[i])
    # The very last label is the eos token id (the only label in the
    # span that comes from `eos_token_id` rather than a content token).
    assert labels[-1] == tok.eos_token_id


def test_roundtrip_recover_solution_from_unmasked_positions() -> None:
    """Stage the canonical use case: write a CoT-shaped JSONL row,
    load it, build the example, and verify the solution tokens are
    exactly recoverable from the non-masked labels."""
    tok = _FakeTokenizer()
    prompt = "compute factorial recursively"
    cot = "base case is n equal to one then recurse on n minus one"
    sol = "def fact ( n ) : return 1 if n <= 1 else n * fact ( n - 1 )"

    row = {
        "task_id": "test/1",
        "sample_idx": 0,
        "problem_prompt": prompt,
        "qwen_completion": f"{cot}\n\n```python\n{sol}\n```\n",
        "extracted_code": sol,
        "has_tests": False,
        "tier": "pass",
        "score": 1.0,
        "source_tier": "cot",
        "source_score": 1.0,
        "prepare_for_thinking": True,
        "cot_text": cot,
    }

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(row) + "\n")
        path = f.name

    try:
        loaded = load_distilled_jsonl_with_cot(path)
    finally:
        pathlib.Path(path).unlink()

    assert len(loaded) == 1
    p_back, sol_back, cot_back = loaded[0]
    assert p_back == prompt
    # For CoT rows the loader returns extracted_code as the solution
    # (the CoT becomes the think span; including it in the solution
    # target would double-count it).
    assert sol_back == sol
    assert cot_back == cot

    full, labels = build_example_with_cot_thinking(
        p_back, cot_back, sol_back, tok, THINK_ID, max_len=512)

    # Recover the solution from the labels: take every label != -100.
    recovered_ids = [x for x in labels if x != -100]
    expected_ids = tok.encode(sol + "\n") + [tok.eos_token_id]
    assert recovered_ids == expected_ids


def test_plain_row_without_prepare_for_thinking_passes_cot_text_none() -> None:
    """A row WITHOUT the prepare_for_thinking flag must come back from
    the loader with cot_text=None and route through build_example
    unchanged."""
    tok = _FakeTokenizer()
    row = {
        "task_id": "plain/1",
        "sample_idx": 0,
        "problem_prompt": "p1 p2 p3",
        "qwen_completion": "s1 s2 s3",
        "extracted_code": "s1 s2 s3",
        "has_tests": False,
        "tier": "pass",
        "score": 1.0,
    }

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(row) + "\n")
        path = f.name

    try:
        loaded = load_distilled_jsonl_with_cot(path)
    finally:
        pathlib.Path(path).unlink()

    assert len(loaded) == 1
    p, sol, cot = loaded[0]
    assert cot is None
    # The plain row, given to build_example, must produce an output
    # IDENTICAL to feeding the original (prompt, solution) — proving
    # the CoT-thinking path didn't sneak any change into the plain
    # path. No think tokens, no -100 leak past the prompt.
    full, labels = build_example(p, sol, tok, max_len=512)
    assert THINK_ID not in full
    n_prompt = len(tok.encode(f"# {p}\n"))
    # Prompt positions are -100; solution positions are real.
    assert all(l == -100 for l in labels[:n_prompt])
    assert all(l != -100 for l in labels[n_prompt:])


def test_min_and_max_cot_thinks_clamp_burst_length() -> None:
    """The CoT think-count is clamped to [min_cot_thinks,
    max_cot_thinks] — handy for budgeting (very short or very long
    CoTs)."""
    tok = _FakeTokenizer()
    short_cot = "a"               # tokenizes to 1
    long_cot = " ".join(f"w{i}" for i in range(50))  # 50 tokens

    full_short, _ = build_example_with_cot_thinking(
        "p", short_cot, "s", tok, THINK_ID, max_len=512,
        min_cot_thinks=5,
    )
    assert sum(1 for x in full_short if x == THINK_ID) == 5

    full_long, _ = build_example_with_cot_thinking(
        "p", long_cot, "s", tok, THINK_ID, max_len=512,
        max_cot_thinks=10,
    )
    assert sum(1 for x in full_long if x == THINK_ID) == 10


def test_truncation_preserves_solution_signal() -> None:
    """When the example exceeds max_len, the loss-bearing solution
    tail must still be present (else the example is structurally
    useless — all labels become -100)."""
    tok = _FakeTokenizer()
    prompt = "p"
    cot = " ".join(f"c{i}" for i in range(200))
    sol = " ".join(f"s{i}" for i in range(20))
    max_len = 30

    full, labels = build_example_with_cot_thinking(
        prompt, cot, sol, tok, THINK_ID, max_len=max_len)

    assert len(full) <= max_len
    assert len(labels) == len(full)
    # The CRITICAL property: at least one non-masked label must remain,
    # i.e. the example carries gradient.
    assert any(l != -100 for l in labels), (
        "truncation killed all solution tokens — example is inert")


def test_input_label_length_match() -> None:
    """input_ids and labels must always be the same length."""
    tok = _FakeTokenizer()
    for cot_len in (0, 1, 5, 20):
        cot = " ".join(f"c{i}" for i in range(cot_len)) or "x"
        full, labels = build_example_with_cot_thinking(
            "p1 p2", cot, "s1 s2", tok, THINK_ID, max_len=256,
        )
        assert len(full) == len(labels), (cot_len, len(full), len(labels))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
