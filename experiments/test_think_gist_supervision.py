"""Tests for Phase 1a — gist-at-think compression SFT
(THINKING_PLAN.md Phase 1a).

The CONTRACT these tests pin:
  * build_example_with_cot_compression returns a STUDENT sequence with
    N_think = max(min_thinks, ceil(N_cot / K)) [THINKING] tokens, a
    TEACHER sequence with the full CoT prose, AND a gist_meta dict that
    maps each student think position to one teacher CoT-chunk-end
    position.
  * think_gist_loss returns 0 when student exactly matches teacher
    (cosine = 1), positive otherwise.
  * After ~50 gradient steps on a tiny synthetic batch the gist loss
    drops measurably — proving the signal is trainable end-to-end
    (the LOAD-BEARING test).
"""
from __future__ import annotations

import math

import pytest
import torch

from experiments.gist_loss import (
    build_think_gist_head, think_gist_loss,
)
from experiments.sft_code import (
    build_example_with_cot_compression,
)


THINK_ID = 9999


class _FakeTokenizer:
    """Whitespace tokenizer with stable per-word ids. Matches the helper
    in test_sft_cot_thinking.py so the two files agree on token shapes."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.eos_token_id = 1

    def _id(self, s: str) -> int:
        if s not in self._vocab:
            self._vocab[s] = 10 + len(self._vocab)
        return self._vocab[s]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(t) for t in text.split()]


# --------------------------------------------------------------------
# build_example_with_cot_compression
# --------------------------------------------------------------------
def test_n_think_is_ceil_n_cot_div_k() -> None:
    tok = _FakeTokenizer()
    prompt = "p"
    # 12 CoT tokens, K=5 → ceil(12/5) = 3 thinks. min_thinks=1 so 3 wins.
    cot = " ".join(f"c{i}" for i in range(12))
    sol = "s1 s2"
    s_ids, s_labs, t_ids, gm = build_example_with_cot_compression(
        prompt, cot, sol, tok, THINK_ID, max_len=512,
        compression_k=5, min_thinks=1,
    )
    assert sum(1 for x in s_ids if x == THINK_ID) == 3
    assert gm["n_think"] == 3
    assert gm["n_cot"] == 12


def test_min_thinks_lower_bound() -> None:
    tok = _FakeTokenizer()
    cot = "c1 c2 c3"     # 3 tokens, K=5 → ceil = 1; min_thinks=4 → 4
    s_ids, _, _, gm = build_example_with_cot_compression(
        "p", cot, "s1", tok, THINK_ID, max_len=128,
        compression_k=5, min_thinks=4,
    )
    assert sum(1 for x in s_ids if x == THINK_ID) == 4
    assert gm["n_think"] == 4


def test_student_layout_prompt_thinks_solution_eos() -> None:
    """Student input: comment_ids + [THINK]*N_think + sol_ids + [eos]."""
    tok = _FakeTokenizer()
    s_ids, s_labs, _, gm = build_example_with_cot_compression(
        "p1 p2", "c1 c2 c3 c4 c5", "s1 s2", tok, THINK_ID, max_len=128,
        compression_k=5, min_thinks=1,
    )
    n_prompt = len(tok.encode("# p1 p2\n"))
    n_think = gm["n_think"]
    # The N_think positions right after the prompt must all be the think id.
    assert s_ids[n_prompt:n_prompt + n_think] == [THINK_ID] * n_think
    # And labels at prompt + think positions must be -100.
    for i in range(n_prompt + n_think):
        assert s_labs[i] == -100, i
    # Solution tail must have real labels (cross-entropy gradient targets).
    assert any(l != -100 for l in s_labs[n_prompt + n_think:])


def test_teacher_layout_prompt_cot_solution_eos() -> None:
    tok = _FakeTokenizer()
    prompt, cot, sol = "p1 p2", "c1 c2 c3 c4 c5", "s1 s2"
    _, _, t_ids, gm = build_example_with_cot_compression(
        prompt, cot, sol, tok, THINK_ID, max_len=128,
        compression_k=5, min_thinks=1,
    )
    n_prompt = len(tok.encode("# p1 p2\n"))
    n_cot = len(tok.encode(cot))
    # Teacher CoT span = the verbatim CoT tokenization.
    expected_cot_ids = tok.encode(cot)
    assert t_ids[n_prompt:n_prompt + n_cot] == expected_cot_ids
    assert gm["n_cot"] == n_cot


def test_gist_meta_maps_think_to_chunk_end_in_teacher() -> None:
    """think i must map to teacher position
       n_prompt + min((i+1)*K - 1, n_cot - 1)."""
    tok = _FakeTokenizer()
    prompt = "p"
    cot = " ".join(f"c{i}" for i in range(15))   # 15 tokens
    K = 5
    s_ids, _, t_ids, gm = build_example_with_cot_compression(
        prompt, cot, "s", tok, THINK_ID, max_len=128,
        compression_k=K, min_thinks=1,
    )
    n_prompt = len(tok.encode("# p\n"))
    n_cot = len(tok.encode(cot))
    assert gm["n_think"] == 3
    for i, (sp, tp) in enumerate(
            zip(gm["think_positions"], gm["teacher_cot_positions"])):
        # student think i sits at n_prompt + i
        assert sp == n_prompt + i
        # teacher CoT chunk i ends at n_prompt + (i+1)*K - 1 (clamped)
        expected = n_prompt + min((i + 1) * K - 1, n_cot - 1)
        assert tp == expected, (i, tp, expected)


def test_gist_meta_handles_uneven_division() -> None:
    """When N_cot doesn't divide evenly by K, the last think's chunk-end
    clamps to N_cot - 1 (so the supervision is on the last visible CoT
    position, not past it)."""
    tok = _FakeTokenizer()
    cot = " ".join(f"c{i}" for i in range(12))  # 12 tokens
    s_ids, _, _, gm = build_example_with_cot_compression(
        "p", cot, "s", tok, THINK_ID, max_len=128,
        compression_k=5, min_thinks=1,
    )
    n_prompt = len(tok.encode("# p\n"))
    # 3 thinks, K=5: positions 4, 9, 11 (last clamps from 14 to 11).
    expected_offsets = [4, 9, 11]
    assert gm["teacher_cot_positions"] == [
        n_prompt + off for off in expected_offsets]


def test_truncation_drops_pairs_past_teacher_window() -> None:
    """If the teacher CoT got truncated, the gist meta must not point
    past the surviving CoT window."""
    tok = _FakeTokenizer()
    prompt = "p"
    cot = " ".join(f"c{i}" for i in range(80))     # large CoT
    sol = " ".join(f"s{i}" for i in range(8))
    max_len = 30
    s_ids, _, t_ids, gm = build_example_with_cot_compression(
        prompt, cot, sol, tok, THINK_ID, max_len=max_len,
        compression_k=5, min_thinks=1,
    )
    assert len(s_ids) <= max_len
    assert len(t_ids) <= max_len
    for tp in gm["teacher_cot_positions"]:
        assert tp < len(t_ids), (tp, len(t_ids))
    for sp in gm["think_positions"]:
        assert sp < len(s_ids), (sp, len(s_ids))


def test_input_label_length_match_student() -> None:
    tok = _FakeTokenizer()
    s_ids, s_labs, _, _ = build_example_with_cot_compression(
        "p1 p2", "c1 c2 c3", "s1 s2 s3", tok, THINK_ID,
        max_len=128, compression_k=2, min_thinks=1,
    )
    assert len(s_ids) == len(s_labs)


# --------------------------------------------------------------------
# think_gist_loss
# --------------------------------------------------------------------
def test_gist_loss_zero_when_student_matches_teacher() -> None:
    """Identity head + identical student/teacher hidden states at the
    mapped positions → cosine = 1 → loss = 0."""
    torch.manual_seed(0)
    B, sT, tT, d = 2, 6, 10, 8
    h = torch.randn(B, max(sT, tT), d)
    # Construct student/teacher so the chosen positions ARE equal.
    student_h = h[:, :sT].clone()
    teacher_h = h[:, :tT].clone()
    # Make student_h[0, 2] == teacher_h[0, 5] etc.
    teacher_h[0, 5] = student_h[0, 2]
    teacher_h[1, 7] = student_h[1, 3]
    head = build_think_gist_head(d)
    head.weight.data = torch.eye(d)
    loss = think_gist_loss(
        student_h, teacher_h,
        think_positions=[[2], [3]],
        teacher_cot_positions=[[5], [7]],
        head=head,
    )
    assert loss.item() < 1e-5, loss.item()


def test_gist_loss_positive_with_random_head() -> None:
    torch.manual_seed(1)
    B, sT, tT, d = 1, 8, 12, 16
    student_h = torch.randn(B, sT, d)
    teacher_h = torch.randn(B, tT, d)
    head = build_think_gist_head(d)
    loss = think_gist_loss(
        student_h, teacher_h,
        think_positions=[[1, 3, 5]],
        teacher_cot_positions=[[2, 6, 10]],
        head=head,
    )
    assert loss.item() > 0.1, loss.item()


def test_gist_loss_returns_zero_when_no_pairs() -> None:
    student_h = torch.randn(2, 5, 8)
    teacher_h = torch.randn(2, 5, 8)
    head = build_think_gist_head(8)
    loss = think_gist_loss(
        student_h, teacher_h, [[], []], [[], []], head,
    )
    assert loss.item() == 0.0


def test_gist_loss_gradient_reaches_student_and_head() -> None:
    torch.manual_seed(2)
    d = 8
    student_h = torch.randn(1, 6, d, requires_grad=True)
    teacher_h = torch.randn(1, 6, d)
    head = build_think_gist_head(d)
    loss = think_gist_loss(
        student_h, teacher_h, [[1, 3]], [[2, 4]], head)
    loss.backward()
    assert student_h.grad is not None
    assert student_h.grad.abs().sum().item() > 0
    assert head.weight.grad is not None
    assert head.weight.grad.abs().sum().item() > 0


def test_gist_loss_does_not_propagate_to_teacher() -> None:
    """The teacher hidden states are detached inside think_gist_loss
    (the function call doesn't even need requires_grad on teacher_h)."""
    torch.manual_seed(3)
    d = 8
    student_h = torch.randn(1, 6, d, requires_grad=True)
    teacher_h = torch.randn(1, 6, d, requires_grad=True)
    head = build_think_gist_head(d)
    loss = think_gist_loss(
        student_h, teacher_h, [[1, 3]], [[2, 4]], head)
    loss.backward()
    # Teacher should have grad of zero (or None) because the function
    # detaches; we don't require strict None, but the magnitude must be 0.
    if teacher_h.grad is not None:
        assert teacher_h.grad.abs().sum().item() == 0.0


def test_gist_loss_mse_path() -> None:
    torch.manual_seed(4)
    d = 8
    student_h = torch.randn(1, 4, d)
    teacher_h = torch.randn(1, 4, d)
    head = build_think_gist_head(d)
    loss = think_gist_loss(
        student_h, teacher_h, [[0, 2]], [[1, 3]], head,
        loss_type="mse",
    )
    assert loss.item() > 0


def test_gist_loss_rejects_unknown_loss_type() -> None:
    head = build_think_gist_head(4)
    with pytest.raises(ValueError):
        think_gist_loss(
            torch.zeros(1, 2, 4), torch.zeros(1, 2, 4),
            [[0]], [[1]], head, loss_type="kl",
        )


# --------------------------------------------------------------------
# LOAD-BEARING: end-to-end micro-train, gist loss must drop
# --------------------------------------------------------------------
def test_gist_loss_demonstrably_trains_end_to_end() -> None:
    """LOAD-BEARING: simulate the SFT inner loop with a tiny
    proxy-"student" + frozen-"teacher" pair. After 50 gradient steps
    the gist loss must drop by at least 50 % from its initial value.

    Setup mirrors the real trainer: synthetic batched hidden states
    play the role of `student_h` (gradient flows through them via a
    learnable linear proxy); teacher hidden states are stop-grad'd
    inside think_gist_loss. The think_positions / teacher_cot_positions
    use the same chunk-end mapping that
    build_example_with_cot_compression produces (K=5 → think i maps to
    teacher position 5i+4)."""
    torch.manual_seed(0)
    B, sT, tT, d = 2, 16, 64, 32

    # Frozen teacher hidden states + a frozen base "trunk" that maps an
    # arbitrary student input to a hidden state. The student gradient
    # path is the proxy `student_proj` + the gist head — together they
    # have to learn to map the proxy's randomly-initialised hidden to
    # the teacher target.
    teacher_h = torch.randn(B, tT, d)
    student_input = torch.randn(B, sT, d)
    student_proj = torch.nn.Linear(d, d, bias=False)
    head = build_think_gist_head(d)
    opt = torch.optim.AdamW(
        list(student_proj.parameters()) + list(head.parameters()),
        lr=1e-2,
    )
    # K=5 chunk-end mapping for 3 thinks → teacher positions 4, 9, 14.
    think_positions = [[0, 1, 2]] * B
    teacher_cot_positions = [[4, 9, 14]] * B

    initial_loss = None
    for step in range(50):
        s_h = student_proj(student_input)
        loss = think_gist_loss(
            s_h, teacher_h, think_positions, teacher_cot_positions,
            head, loss_type="cosine")
        if step == 0:
            initial_loss = float(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = float(loss.item())
    # LOAD-BEARING assertion: gist loss must drop substantially. A
    # measured ~10x improvement is typical in the synthetic harness;
    # we require at least 50% (catches "the head is structurally
    # disconnected" without flaking on optimizer noise).
    assert final_loss < 0.5 * initial_loss, (
        f"gist loss did not train: initial={initial_loss:.4f}, "
        f"final={final_loss:.4f}")
    print(f"  LOAD-BEARING ok: gist {initial_loss:.4f} -> {final_loss:.4f}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
