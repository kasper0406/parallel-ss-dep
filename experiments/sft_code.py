"""Supervised fine-tune of a distilled TinyLM on (problem, solution) pairs.

Goal: bridge the gap from "produces code-shaped text" (the distilled base)
to "produces code that passes unit tests" by training on instruction-shaped
data: MBPP train split + CodeAlpaca.

Format: we present each (problem, solution) as a Python comment followed by
the solution code, so the SFT distribution matches plain-code distribution
at HumanEval inference time:

    # <problem text, one-line>
    <solution code>
    <eos>

The loss is computed only on the solution tokens (everything from the
newline after the comment to <eos>). Problem-comment tokens are masked.

Phase-4 CoT-thinking SFT (THINKING_PLAN.md Phase 4, Option A): when an
input row has `prepare_for_thinking: true` plus a `cot_text` field
(produced by `experiments/build_cot_sft_data.py`), the example is built
by `build_example_with_cot_thinking` instead of `build_example`. The CoT
prose is *replaced* with N consecutive `[THINKING]` tokens (where N =
the CoT's tokenized length), and those positions are masked (-100) from
the loss. The model thus learns "given the problem, spend ~N think
tokens, then emit the solution" — directly teaching the gate to fire on
hard problems without us needing to assign per-think-step targets.

Why Design A (replace, not interleave): keeps the change to a single
helper + a per-row dispatch in the main loop. Designs B (per-CoT-token
interleaving) and C (retrieval-as-input from CoT embeddings) require new
trunk plumbing; A re-uses everything already in place. The CoT *content*
is not directly modeled, but the temporal "think-before-emit" structure
is — which is what we need before Phase 5 can give thinking process
reward (see THINKING_DECISIONS.md 2026-05-26).

Usage:
    CUDA_VISIBLE_DEVICES=0 python experiments/sft_code.py \\
        --load_ckpt checkpoints/distill_qwen36_dn217_mem.pt \\
        --save_ckpt checkpoints/sft_dn217_mem.pt \\
        --epochs 2 --batch 4 --lr 3e-5
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.gist_loss import (
    build_gist_heads, trunk_gist_loss, parse_horizons,
    build_think_gist_head, think_gist_loss,
)
from experiments.process_reward import compute_process_reward_loss


def _flatten_to_oneline(s: str) -> str:
    """Squash multi-line problem text into a one-line `# ...` comment-safe string."""
    return " ".join(s.split())


def load_pairs(max_codealpaca: int | None = None) -> list[tuple[str, str]]:
    """Return a list of (problem, solution) string pairs.

    Sources combined:
      - MBPP train split (~370 problems with `text` and `code` fields).
      - CodeAlpaca-20k (filtered to Python-looking outputs).
    """
    from datasets import load_dataset

    pairs: list[tuple[str, str]] = []

    print("loading MBPP train split...")
    mbpp = load_dataset("mbpp", split="train")
    for x in mbpp:
        pairs.append((x["text"], x["code"]))

    print("loading CodeAlpaca-20k...")
    try:
        ca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception as e:
        print(f"  CodeAlpaca load failed ({e}); proceeding with MBPP only.")
        return pairs

    n_added = 0
    for x in ca:
        if max_codealpaca is not None and n_added >= max_codealpaca:
            break
        instr = x["instruction"]
        inp = x.get("input") or ""
        out = x["output"]
        # Filter to Python-looking outputs (heuristic): must contain `def `
        # OR start with python keywords / imports.
        if not any(t in out for t in ("def ", "import ", "for ", "while ",
                                       "if ", "return ", "print(")):
            continue
        prompt = instr if not inp else f"{instr}\n{inp}"
        pairs.append((prompt, out))
        n_added += 1
    print(f"  added {n_added} CodeAlpaca examples")
    return pairs


def load_distilled_jsonl(path: str, *,
                          prefer_full_completion: bool = True,
                          require_extracted_code: bool = True,
                          keep_only_passing: bool = False,
                          ) -> list[tuple[str, str]]:
    """Load (problem, solution) pairs from a distill_solutions.py JSONL.

    Each JSONL row has {task_id, problem_prompt, qwen_completion,
    extracted_code, has_tests, tier, score, sample_idx}.

      prefer_full_completion:
        True  (default) → solution = qwen_completion (CoT + code block).
                          Student learns to reason before emitting code,
                          which exercises the thinking gate during SFT.
        False → solution = extracted_code only (cleaner but no reasoning
                signal).

      require_extracted_code:
        Drop rows where Qwen ran out of tokens before producing a
        ```python``` block. Default True — those rows are noisy.

      keep_only_passing:
        If True, drop rows where has_tests and tier != "pass" (rejection
        sampling: train only on solutions known to work). Distillation-
        only sources (no tests) are kept regardless.
    """
    import json
    pairs: list[tuple[str, str]] = []
    n_total = n_dropped_no_code = n_dropped_failed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if require_extracted_code and not r.get("extracted_code"):
                n_dropped_no_code += 1
                continue
            if keep_only_passing and r.get("has_tests") and r.get("tier") != "pass":
                n_dropped_failed += 1
                continue
            problem = r["problem_prompt"]
            solution = (r["qwen_completion"] if prefer_full_completion
                        else r["extracted_code"])
            if not solution:
                continue
            pairs.append((problem, solution))
    print(f"  loaded {len(pairs)} pairs from {path} "
          f"(total rows={n_total}, dropped no-code={n_dropped_no_code}, "
          f"dropped failed={n_dropped_failed})")
    return pairs


def load_distilled_jsonl_with_cot(
    path: str,
    *,
    prefer_full_completion: bool = True,
    require_extracted_code: bool = True,
    keep_only_passing: bool = False,
) -> list[tuple[str, str, str | None]]:
    """Like `load_distilled_jsonl` but returns 3-tuples
    `(problem, solution, cot_text_or_None)` so the SFT main loop can
    dispatch per row to `build_example_with_cot_thinking` when
    `prepare_for_thinking=True`.

    `cot_text` is `None` for rows without `prepare_for_thinking=True` —
    those rows train via the unchanged `build_example` path.
    """
    import json
    rows: list[tuple[str, str, str | None]] = []
    n_total = n_dropped_no_code = n_dropped_failed = n_cot = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if require_extracted_code and not r.get("extracted_code"):
                n_dropped_no_code += 1
                continue
            if keep_only_passing and r.get("has_tests") and r.get("tier") != "pass":
                n_dropped_failed += 1
                continue
            problem = r["problem_prompt"]
            # For CoT-flagged rows, the solution target is just the
            # extracted code: the CoT is already going to be materialised
            # as the think span, so including it in qwen_completion would
            # double-count it (model would emit it as real text after
            # already having thought it).
            is_cot = bool(r.get("prepare_for_thinking")) and r.get("cot_text")
            if is_cot:
                solution = r.get("extracted_code") or ""
                cot_text = r["cot_text"]
            else:
                solution = (r["qwen_completion"] if prefer_full_completion
                            else r.get("extracted_code") or "")
                cot_text = None
            if not solution:
                continue
            rows.append((problem, solution, cot_text))
            if is_cot:
                n_cot += 1
    print(f"  loaded {len(rows)} pairs from {path} "
          f"(total rows={n_total}, dropped no-code={n_dropped_no_code}, "
          f"dropped failed={n_dropped_failed}, cot-thinking rows={n_cot})")
    return rows


def build_example(prompt: str, solution: str, tokenizer,
                  max_len: int) -> tuple[list[int], list[int]]:
    """Tokenize one example to (input_ids, labels) where labels are -100 on
    the prompt-comment tokens and the actual token ids on the solution
    tokens (causal LM convention: predict each token given prefix).

    If `prompt` already starts with "# " AND is multi-line (a pre-formatted
    multi-comment block like the iterative-repair prompt), use it verbatim;
    otherwise flatten to a single `# {text}\n` line for the legacy
    one-line-instruction format used by CodeAlpaca / single-line distill
    problems.
    """
    if prompt.startswith("# ") and "\n" in prompt.rstrip("\n"):
        comment_line = prompt if prompt.endswith("\n") else prompt + "\n"
    else:
        comment_line = f"# {_flatten_to_oneline(prompt)}\n"
    comment_ids = tokenizer.encode(comment_line, add_special_tokens=False)
    solution_text = solution + ("\n" if not solution.endswith("\n") else "")
    sol_ids = tokenizer.encode(solution_text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        sol_ids = sol_ids + [int(eos)]
    full = comment_ids + sol_ids
    # Truncate from the right if needed (keep the comment intact).
    if len(full) > max_len:
        full = full[:max_len]
        sol_len = max(0, len(full) - len(comment_ids))
    else:
        sol_len = len(sol_ids)
    # Labels: -100 on the comment portion, real ids on the solution.
    labels = [-100] * (len(full) - sol_len) + full[len(full) - sol_len:]
    return full, labels


def build_example_with_cot_thinking(
    prompt: str,
    cot_text: str,
    solution: str,
    tokenizer,
    thinking_token_id: int,
    max_len: int,
    *,
    min_cot_thinks: int = 1,
    max_cot_thinks: int | None = None,
) -> tuple[list[int], list[int]]:
    """Phase-4 Option-A SFT example builder.

    Layout (prompt | CoT-as-thinks | solution | eos):
        comment_ids + [THINKING] * N_cot + solution_ids + [eos]
    where N_cot = len(tokenize(cot_text)) (clamped to
    [min_cot_thinks, max_cot_thinks]).

    Label mask:
        comment positions  → -100
        think positions    → -100
        solution positions → real token ids
        eos position       → real eos id

    The CoT content is intentionally NOT taught; only the budget (length
    of the think span) and the temporal ordering "think-before-emit" are.
    Per the Option-A rationale in the module docstring.

    Truncation: if the full sequence exceeds `max_len`, the solution
    tail is preserved (we keep the comment intact and truncate the CoT
    span, then the solution from the right). If even the
    comment+solution don't fit, the solution is right-truncated; the
    comment is preserved up to `max_len`. This mirrors `build_example`'s
    bias toward keeping the *target* signal (comment + solution) when
    space is tight.
    """
    if prompt.startswith("# ") and "\n" in prompt.rstrip("\n"):
        comment_line = prompt if prompt.endswith("\n") else prompt + "\n"
    else:
        comment_line = f"# {_flatten_to_oneline(prompt)}\n"
    comment_ids = tokenizer.encode(comment_line, add_special_tokens=False)

    cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
    n_cot = len(cot_ids)
    if min_cot_thinks is not None:
        n_cot = max(n_cot, int(min_cot_thinks))
    if max_cot_thinks is not None:
        n_cot = min(n_cot, int(max_cot_thinks))
    think_ids = [int(thinking_token_id)] * n_cot

    solution_text = solution + ("\n" if not solution.endswith("\n") else "")
    sol_ids = tokenizer.encode(solution_text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        sol_ids = sol_ids + [int(eos)]

    full = comment_ids + think_ids + sol_ids
    if len(full) <= max_len:
        sol_len = len(sol_ids)
        think_len = len(think_ids)
        comment_len = len(comment_ids)
    else:
        # Over budget. Preserve comment + (truncated) solution; trim the
        # CoT think span first since its content is not graded.
        comment_len = min(len(comment_ids), max_len)
        room_after_comment = max_len - comment_len
        # Always keep at least one solution token so the loss is non-empty.
        sol_len = min(len(sol_ids), max(1, room_after_comment - 0))
        think_len = max(0, room_after_comment - sol_len)
        if think_len > len(think_ids):
            think_len = len(think_ids)
            sol_len = min(len(sol_ids), room_after_comment - think_len)
        full = (
            comment_ids[:comment_len]
            + [int(thinking_token_id)] * think_len
            + sol_ids[:sol_len]
        )

    labels = (
        [-100] * comment_len
        + [-100] * think_len
        + list(full[comment_len + think_len:])
    )
    assert len(labels) == len(full), (len(labels), len(full))
    return full, labels


def build_example_with_cot_compression(
    prompt: str,
    cot_text: str,
    solution: str,
    tokenizer,
    thinking_token_id: int,
    max_len: int,
    *,
    compression_k: int = 5,
    min_thinks: int = 4,
):
    """Phase-1a SFT example builder (THINKING_PLAN.md): build a STUDENT
    sequence whose CoT span is COMPRESSED into N_think = ceil(N_cot / K)
    consecutive [THINKING] tokens, plus a parallel TEACHER sequence that
    keeps the full CoT prose. The trainer runs the teacher in no_grad to
    extract hidden-state targets for the student's think positions
    (gist-at-think supervision via experiments.gist_loss.think_gist_loss).

    Compression mapping (think -> teacher CoT position):
      think i -> teacher position (within the CoT span) of the LAST
      token of the i-th K-chunk: min((i+1)*K - 1, N_cot - 1). This is
      the natural "summary point" — at that hidden state the causal
      teacher has seen every token in chunk i.

    Both sequences share `comment_ids` as prefix and `sol_ids + [eos]`
    as suffix; ONLY the middle differs.

    Returns:
      student_input_ids: list[int]
      student_labels:    list[int]   (-100 on prompt + thinks; real on sol)
      teacher_input_ids: list[int]
      gist_meta:         dict with keys
        "think_positions":         list[int] positions in student_input_ids
        "teacher_cot_positions":   list[int] positions in teacher_input_ids
        "comment_len":             int
        "n_cot":                   int (teacher CoT token count, post-truncation)
        "n_think":                 int

    Truncation policy: same as `build_example_with_cot_thinking` — keep
    the comment intact, then keep the solution tail, then fill any
    remaining budget with think (student) / CoT (teacher) tokens. If
    teacher truncation drops some CoT positions a think would otherwise
    point to, those think -> cot pairs are dropped from gist_meta so the
    loss only sees valid pairs.
    """
    if prompt.startswith("# ") and "\n" in prompt.rstrip("\n"):
        comment_line = prompt if prompt.endswith("\n") else prompt + "\n"
    else:
        comment_line = f"# {_flatten_to_oneline(prompt)}\n"
    comment_ids = tokenizer.encode(comment_line, add_special_tokens=False)

    cot_ids = tokenizer.encode(cot_text, add_special_tokens=False)
    n_cot_full = len(cot_ids)
    if n_cot_full == 0:
        # No CoT tokens — fall back to plain example layout via the same
        # contract (still produce min_thinks for budget consistency).
        cot_ids = []
        n_cot_full = 0

    K = max(1, int(compression_k))
    # N_think = ceil(N_cot / K), at least `min_thinks`. When n_cot is
    # zero, force at least min_thinks.
    if n_cot_full == 0:
        n_think_full = max(1, int(min_thinks))
    else:
        n_think_full = max(int(min_thinks), (n_cot_full + K - 1) // K)
    think_ids_full = [int(thinking_token_id)] * n_think_full

    solution_text = solution + ("\n" if not solution.endswith("\n") else "")
    sol_ids = tokenizer.encode(solution_text, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is not None:
        sol_ids = sol_ids + [int(eos)]

    # --- STUDENT sequence (with think compression) --------------------
    student_full = comment_ids + think_ids_full + sol_ids
    if len(student_full) <= max_len:
        s_comment_len = len(comment_ids)
        s_think_len = n_think_full
        s_sol_len = len(sol_ids)
    else:
        s_comment_len = min(len(comment_ids), max_len)
        room = max_len - s_comment_len
        s_sol_len = min(len(sol_ids), max(1, room))
        s_think_len = max(0, room - s_sol_len)
        if s_think_len > n_think_full:
            s_think_len = n_think_full
            s_sol_len = min(len(sol_ids), room - s_think_len)
        student_full = (
            comment_ids[:s_comment_len]
            + [int(thinking_token_id)] * s_think_len
            + sol_ids[:s_sol_len]
        )
    student_labels = (
        [-100] * s_comment_len
        + [-100] * s_think_len
        + list(student_full[s_comment_len + s_think_len:])
    )
    assert len(student_labels) == len(student_full)

    # --- TEACHER sequence (with full CoT prose) -----------------------
    teacher_full = comment_ids + cot_ids + sol_ids
    if len(teacher_full) <= max_len:
        t_comment_len = len(comment_ids)
        t_cot_len = n_cot_full
        t_sol_len = len(sol_ids)
    else:
        t_comment_len = min(len(comment_ids), max_len)
        room = max_len - t_comment_len
        t_sol_len = min(len(sol_ids), max(1, room))
        t_cot_len = max(0, room - t_sol_len)
        if t_cot_len > n_cot_full:
            t_cot_len = n_cot_full
            t_sol_len = min(len(sol_ids), room - t_cot_len)
        teacher_full = (
            comment_ids[:t_comment_len]
            + cot_ids[:t_cot_len]
            + sol_ids[:t_sol_len]
        )

    # --- Gist mapping: student think i -> teacher CoT position --------
    # think i represents CoT chunk [i*K, (i+1)*K); supervise on the last
    # token of that chunk (clamped to the last valid CoT position).
    # Pairs that fall outside the (post-truncation) teacher CoT window
    # are dropped — partial supervision is better than wrong.
    think_positions: list[int] = []
    teacher_cot_positions: list[int] = []
    if s_think_len > 0 and t_cot_len > 0:
        # The COMMON comment prefix has length min(s_comment_len, t_comment_len)
        # under normal use (both equal len(comment_ids)). When that
        # invariant holds, student think index i sits at position
        # s_comment_len + i; teacher CoT chunk i ends at
        # t_comment_len + min((i+1)*K - 1, t_cot_len - 1).
        for i in range(s_think_len):
            target_chunk_end = (i + 1) * K - 1
            if target_chunk_end >= t_cot_len:
                # The teacher CoT got truncated past this chunk's end —
                # clamp to last available CoT position; if that's already
                # been used, drop the pair (no duplicate supervision).
                target_chunk_end = t_cot_len - 1
            if target_chunk_end < 0:
                break
            sp = s_comment_len + i
            tp = t_comment_len + target_chunk_end
            # Drop duplicate teacher positions (multiple thinks pointing
            # at the same already-clamped tail token would over-weight it).
            if teacher_cot_positions and teacher_cot_positions[-1] == tp:
                continue
            think_positions.append(sp)
            teacher_cot_positions.append(tp)

    gist_meta = {
        "think_positions": think_positions,
        "teacher_cot_positions": teacher_cot_positions,
        "comment_len": int(s_comment_len),
        "n_cot": int(t_cot_len),
        "n_think": int(s_think_len),
    }
    return student_full, student_labels, teacher_full, gist_meta


def insert_think_bursts(
    input_ids: list[int], labels: list[int],
    thinking_token_id: int, max_len: int,
    max_bursts: int = 3, max_burst_depth: int = 8,
    rng: torch.Generator | None = None,
    aligned: list[int] | None = None,
) -> tuple[list[int], list[int]] | tuple[list[int], list[int], list[int]]:
    """Insert random-depth thinking-token bursts into a (input_ids, labels) pair.

    Each burst is a run of `depth` think tokens; depth ~ U[1, max_burst_depth].
    Labels at inserted think positions are set to -100 (no loss). The result
    is truncated to `max_len` if too long.

    Purpose: give the think-token embedding, the gate head, and the working-
    memory module dense supervised gradient *before* RL ever sees them. The
    SFT loss is still next-token CE on real tokens (think positions don't
    contribute), but every real-token prediction that follows a think burst
    requires the model to have processed those think tokens gracefully, so
    the think-embedding and memory weights have to learn to be useful.

    `aligned`: an optional per-token array (same length as `input_ids`, e.g.
    document ids) that must stay aligned through the same insertions. Each
    inserted think token copies the doc id of the preceding real token, so a
    think burst belongs to the document it sits inside. When given, the
    return is a 3-tuple `(new_ids, new_labels, new_aligned)`; otherwise the
    2-tuple `(new_ids, new_labels)` — existing callers are unaffected.
    """
    def _ret(ids, labs, al):
        return (ids, labs, al) if aligned is not None else (ids, labs)

    if rng is None:
        rng = torch.Generator().manual_seed(0)
    if max_bursts <= 0:
        return _ret(input_ids, labels, aligned)
    n = len(input_ids)
    if n < 4:
        return _ret(input_ids, labels, aligned)
    n_bursts = int(torch.randint(0, max_bursts + 1, (1,), generator=rng).item())
    if n_bursts == 0:
        return _ret(input_ids, labels, aligned)
    burst_positions = sorted(
        torch.randperm(n, generator=rng)[:n_bursts].tolist()
    )
    new_ids: list[int] = []
    new_labels: list[int] = []
    new_aligned: list[int] = []
    last = 0
    for p in burst_positions:
        new_ids.extend(input_ids[last:p])
        new_labels.extend(labels[last:p])
        depth = int(
            torch.randint(1, max_burst_depth + 1, (1,), generator=rng).item()
        )
        new_ids.extend([int(thinking_token_id)] * depth)
        new_labels.extend([-100] * depth)
        if aligned is not None:
            new_aligned.extend(aligned[last:p])
            # Inserted think tokens belong to the document of the preceding
            # real token (or the first document, at position 0).
            fill = aligned[p - 1] if p > 0 else aligned[0]
            new_aligned.extend([fill] * depth)
        last = p
    new_ids.extend(input_ids[last:])
    new_labels.extend(labels[last:])
    if aligned is not None:
        new_aligned.extend(aligned[last:])
    if len(new_ids) > max_len:
        new_ids = new_ids[:max_len]
        new_labels = new_labels[:max_len]
        new_aligned = new_aligned[:max_len]
    return _ret(new_ids, new_labels, new_aligned)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--load_ckpt", type=str, required=True)
    p.add_argument("--save_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_codealpaca", type=int, default=10000,
                   help="Cap on CodeAlpaca samples (full set is 20k).")
    p.add_argument("--distilled_jsonl", type=str, default=None,
                   help="If set, load (problem, solution) pairs from a "
                        "distill_solutions.py JSONL output INSTEAD OF "
                        "MBPP+CodeAlpaca. Each row contributes one pair where "
                        "the solution is Qwen's full completion (CoT + code).")
    p.add_argument("--distilled_keep_only_passing", action="store_true",
                   help="If set with --distilled_jsonl, drop rows where the "
                        "problem had tests and Qwen's sample didn't pass "
                        "(rejection sampling). Distillation-only rows "
                        "(magicoder/codefeedback, no tests) are always kept.")
    p.add_argument("--distilled_code_only", action="store_true",
                   help="If set with --distilled_jsonl, use ONLY the extracted "
                        "code block as the solution target (drop Qwen's "
                        "reasoning prose). Default keeps the full completion.")
    p.add_argument("--seed", type=int, default=0)
    # --- Thinking-during-SFT: forces the model to handle think tokens ---
    p.add_argument("--with_thinking", action="store_true",
                   help="Enable working memory + output gate + random think "
                        "burst insertion during SFT. This trains the "
                        "think-embedding, memory weights, and trunk to handle "
                        "think tokens gracefully BEFORE RL evaluates the "
                        "gate's emit/think decision. Without this, RL has to "
                        "bootstrap thinking from random init — which it "
                        "rationally refuses (see WORKING_MEMORY_FINDINGS.md).")
    p.add_argument("--think_max_bursts", type=int, default=3,
                   help="Max think-token bursts inserted per example.")
    p.add_argument("--think_max_depth", type=int, default=8,
                   help="Max depth per inserted burst.")
    p.add_argument("--mem_size", type=int, default=1024)
    p.add_argument("--mem_dim", type=int, default=0)
    p.add_argument("--mem_write_only_at_think", action="store_true",
                   help="FIX A: force WorkingMemory writes to come only "
                        "from think positions. See train_lm_args.py for "
                        "full rationale. When the loaded ckpt was trained "
                        "without this flag, retraining a few epochs with "
                        "it on lets the model adapt its write-gate to "
                        "actually be selective at think positions.")
    p.add_argument("--retrieval_as_input_thinking", action="store_true",
                   help="Replace the discrete [THINKING] token's input "
                        "embedding with the WorkingMemory retrieval at the "
                        "previous position. Solves the think-position "
                        "homogeneity that caused FIX A to fail (the "
                        "[THINKING] token has one embedding so successive "
                        "thinks are highly correlated). With this flag, "
                        "each think step's input is the model's own "
                        "retrieval → diverse per-step → diverse buffer → "
                        "useful sharp reads. See GEMINI.md "
                        "'retrieval-as-input' for the architectural "
                        "rationale.")
    p.add_argument("--disable_wm_during_sft", action="store_true",
                   help="Zero WorkingMemory.W_proj weight and freeze it, "
                        "so WM injections are always zero during this SFT "
                        "run. Diagnostic flag — not currently used in "
                        "production; the milestone calls for fixing WM, "
                        "not removing it.")
    # --- v7 trunk multi-horizon GIST loss (Fix C, 2026-05-20) ----------
    # The trunk's job is "high-level direction": at position t each head
    # predicts the GIST of the upcoming window — the mean-pooled hidden
    # state over h[t+1 : t+1+K], stop-grad'd. The trunk is causal so
    # each h[t] is a running contextualised summary; the windowed mean
    # is a genuine "where this is going" vector. Multi-horizon (K in
    # {16,64,256}) gives local tactic + mid plan + global direction.
    #
    # History: v5 supervised the WM read to predict embed(input_ids[t+4])
    # (context-free lexical). v6 supervised the WM read to predict this
    # gist — but routing a blurry gist through WM broke precise recall
    # (longctx eval 2026-05-20: 99%→61%). v7 supervises the TRUNK with
    # the gist and leaves WM free to learn precise retrieval.
    p.add_argument("--future_emb_loss_weight", type=float, default=0.0,
                   help="Weight for the v7 trunk multi-horizon gist "
                        "loss. 0 = disabled. Recommended 0.1.")
    p.add_argument("--wm_gist_horizons", type=str, default="16,64,256",
                   help="Comma-separated future-window sizes K for the "
                        "trunk gist loss (one head per horizon).")
    # Deprecated flags — kept so older launchers still parse.
    p.add_argument("--wm_future_pred_weight", type=float, default=0.0,
                   help="DEPRECATED since v7 (WM gist supervision "
                        "removed). Ignored.")
    p.add_argument("--wm_future_pred_T", type=int, default=4,
                   help="DEPRECATED (v5 single-offset embed target). "
                        "Ignored.")
    p.add_argument("--future_emb_T_max", type=int, default=8,
                   help="DEPRECATED (v5 lexical-target ramp). Ignored.")
    p.add_argument("--future_emb_T_ramp_frac", type=float, default=0.3,
                   help="DEPRECATED (v5 lexical-target ramp). Ignored.")
    # --- Phase 1a: gist-at-think compression (THINKING_PLAN.md) -------
    # CoT-thinking rows route through build_example_with_cot_compression
    # (instead of build_example_with_cot_thinking) when these are set:
    # student gets N_think = ceil(N_cot / K) thinks; the trainer also
    # runs a no_grad teacher forward over the full (prompt + CoT + code)
    # sequence and supervises each student think against the teacher
    # hidden state at the corresponding CoT chunk end via
    # think_gist_loss. This tests the "1 think ≈ K CoT tokens"
    # compression claim — see THINKING_PROBE_RESULTS.md for why we
    # pivoted from Design A (loss-masked padding, no gradient signal).
    p.add_argument("--cot_compression_k", type=int, default=0,
                   help="Phase 1a compression factor K. >0 enables "
                        "gist-at-think SFT: CoT-thinking rows produce "
                        "N_think = ceil(N_cot / K) thinks with a "
                        "per-think hidden-state target from a no_grad "
                        "teacher forward over the full CoT. 0 (default) "
                        "= keep legacy Design A behaviour "
                        "(build_example_with_cot_thinking).")
    p.add_argument("--cot_min_thinks", type=int, default=4,
                   help="Lower bound on N_think per CoT row when "
                        "--cot_compression_k > 0.")
    p.add_argument("--think_gist_weight", type=float, default=0.0,
                   help="Auxiliary loss weight for the Phase 1a "
                        "gist-at-think supervision. Recommended 0.1. "
                        "0 disables the gist loss (still routes through "
                        "the compression builder if --cot_compression_k > 0; "
                        "useful for ablating the gist signal vs the "
                        "compressed-think layout alone).")
    p.add_argument("--think_gist_loss_type", type=str, default="cosine",
                   choices=["cosine", "mse"],
                   help="Distance for think_gist_loss.")
    # Phase-2/3 toggles (defaults inherit from cfg). These flags FORCE
    # the flag on at SFT time even if the loaded ckpt's cfg had it off
    # (allows turning the architectural fix on for an SFT continuation).
    p.add_argument("--state_readonly_at_think", action="store_true",
                   help="Phase 2: force DeltaNet β=0 at think positions "
                        "so thinking can't corrupt the recurrent state. "
                        "Inherits from ckpt cfg when omitted; pass this "
                        "to flip ON for a continuation SFT.")
    p.add_argument("--think_index_emb_size", type=int, default=-1,
                   help="Phase 3: per-position think-index embedding "
                        "table size (breaks the multi-think input-"
                        "homogenization that motivated FIX A). -1 (default) "
                        "= inherit from ckpt cfg; >=0 forces that size "
                        "(0 disables the embedding, >0 builds it).")
    # Phase B (THINKING_PLAN v5): small MLP that fires only at think
    # positions, giving the trunk dedicated parameters for thinking-
    # time computation. Default = inherit from ckpt (auto-detected
    # from state_dict). Pass --use_think_adapter to ATTACH a fresh
    # adapter to a ckpt that didn't have one (alpha init 0 →
    # byte-identical at start, learns during SFT).
    p.add_argument("--use_think_adapter", action="store_true",
                   help="Phase B: enable per-Block ThinkAdapter MLP. "
                        "Omit = inherit from ckpt; pass = force ON.")
    p.add_argument("--think_adapter_hidden_mult", type=int, default=2,
                   help="Phase B: hidden width multiplier for the "
                        "ThinkAdapter MLP (default 2 → d_hidden=2*d_model).")
    # --- Phase D: RefinementHead (THINKING_PLAN v5, 2026-05-27) -----------
    p.add_argument("--use_refinement_head", action="store_true",
                   help="Phase D: attach a RefinementHead (windowed local "
                        "attention + MLP, soft-mixed with trunk via σ). "
                        "Omit = inherit from ckpt; pass = force ON.")
    p.add_argument("--refinement_head_window", type=int, default=128,
                   help="Phase D: local-attention window (default 128).")
    p.add_argument("--refinement_head_n_heads", type=int, default=8,
                   help="Phase D: refinement-head attention heads (default 8).")
    p.add_argument("--refinement_head_mlp_mult", type=int, default=2,
                   help="Phase D: refinement-head MLP hidden mult (default 2).")
    # --- Phase A: process-reward auxiliary loss (THINKING_PLAN.md v5) -----
    # On a sampled subset of positions where the gate already wants to
    # think, do a SECOND forward over [prefix, K * THINK] and ask whether
    # the K-think prediction puts more probability mass on the true
    # next token than the original (no-think) main-forward prediction.
    # Loss = mean(log p_before - log p_after); the optimiser is pushed
    # to make thinks actually reduce next-token error. Default off
    # → byte-identical training.
    p.add_argument("--process_reward_weight", type=float, default=0.0,
                   help="Phase A: weight on the process-reward aux loss. "
                        "0 disables (default). Try 0.1.")
    p.add_argument("--process_reward_K", type=int, default=4,
                   help="Number of think tokens to insert before the "
                        "sampled position for the 'after' forward.")
    p.add_argument("--process_reward_apply_min_sigma", type=float,
                   default=0.3,
                   help="Only apply the loss at positions where σ(gate) "
                        "exceeds this threshold (avoids wasting compute "
                        "on positions the gate clearly didn't want to "
                        "think at).")
    p.add_argument("--process_reward_sample_frac", type=float, default=0.1,
                   help="Fraction of qualifying positions to evaluate "
                        "per batch (bounds compute).")
    p.add_argument("--process_reward_max_positions", type=int, default=128,
                   help="Hard cap on sampled positions per batch — extra "
                        "guard so the after-forward stays bounded even "
                        "on a huge batch.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    # --- 1. Build model from ckpt ----------------------------------------
    print(f"loading checkpoint: {args.load_ckpt}")
    if args.with_thinking:
        # Detect whether the loaded ckpt ALREADY has thinking infrastructure
        # (memory + output gate + thinking-token vocab slot). v5-pkm and later
        # ckpts have all of this baked in from pretrain; older distilled
        # ckpts (the original SFT use case) do not, and we have to add it.
        import torch as _t
        raw_ckpt = _t.load(args.load_ckpt, map_location="cpu", weights_only=False)
        sd_keys = set(raw_ckpt["state_dict"].keys())
        ckpt_has_memory = any(k.startswith("memory.") for k in sd_keys)
        ckpt_has_gate = any(k.startswith("gate_head.") for k in sd_keys)
        if ckpt_has_memory and ckpt_has_gate:
            # Modern path: ckpt already has memory + gate (and possibly PKM).
            # Use build_model_from_ckpt — it autodetects all three and gets
            # the architecture exactly right. Skip the "expand vocab" code
            # because the thinking-token slot is already in the saved vocab.
            from experiments.eval_bracket_structure import build_model_from_ckpt
            # Phase B: if --use_think_adapter is passed and the ckpt
            # didn't carry adapter weights, attach a fresh adapter
            # (alpha init 0 → byte-identical at start). Auto-detect
            # path still wins if the flag isn't passed.
            force_adapter = True if args.use_think_adapter else None
            # Phase D: same override pattern — attach a fresh refinement
            # head when --use_refinement_head is passed on a ckpt that
            # didn't have one (alpha init 0 → byte-identical at start).
            force_refinement = True if args.use_refinement_head else None
            model, cfg = build_model_from_ckpt(
                args.load_ckpt,
                force_use_think_adapter=force_adapter,
                force_think_adapter_hidden_mult=(
                    int(args.think_adapter_hidden_mult)
                    if args.use_think_adapter else None),
                force_use_refinement_head=force_refinement,
                force_refinement_head_window=(
                    int(args.refinement_head_window)
                    if args.use_refinement_head else None),
                force_refinement_head_n_heads=(
                    int(args.refinement_head_n_heads)
                    if args.use_refinement_head else None),
                force_refinement_head_mlp_mult=(
                    int(args.refinement_head_mlp_mult)
                    if args.use_refinement_head else None),
            )
            if args.use_think_adapter:
                cfg["use_think_adapter"] = True
                cfg["think_adapter_hidden_mult"] = int(
                    args.think_adapter_hidden_mult)
                print(f"  Phase B: ThinkAdapter ON "
                      f"(hidden_mult={args.think_adapter_hidden_mult}, "
                      "alpha init 0).")
            if args.use_refinement_head:
                cfg["use_refinement_head"] = True
                cfg["refinement_head_window"] = int(
                    args.refinement_head_window)
                cfg["refinement_head_n_heads"] = int(
                    args.refinement_head_n_heads)
                cfg["refinement_head_mlp_mult"] = int(
                    args.refinement_head_mlp_mult)
                print(f"  Phase D: RefinementHead ON "
                      f"(window={args.refinement_head_window}, "
                      f"n_heads={args.refinement_head_n_heads}, "
                      f"mlp_mult={args.refinement_head_mlp_mult}, "
                      "alpha init 0).")
            thinking_token_id = cfg.get("thinking_token_id")
            if thinking_token_id is None:
                # Fall back to "last vocab slot" if cfg didn't store it.
                thinking_token_id = int(cfg["vocab_size"]) - 1
            # FIX A: honour the new flag even on a pre-built model — flip
            # the bit on the already-constructed WorkingMemory module.
            if bool(args.mem_write_only_at_think):
                model.memory.write_only_at_think = True
                cfg["mem_write_only_at_think"] = True
                print(f"  FIX A enabled: WorkingMemory.write_only_at_think = True "
                      "(non-think positions masked to -1.0 before topk)")
            # Phase-2 override: force state_readonly_at_think on (safe to
            # flip post-construction — it's a plain bool the forward
            # checks each step). The Block-level hook is installed in
            # TinyLM.__init__ if the flag was on at build time, so we
            # also walk Blocks here to install the hook for late-on.
            if bool(getattr(args, "state_readonly_at_think", False)):
                model.state_readonly_at_think = True
                cfg["state_readonly_at_think"] = True
                for block in model.blocks:
                    attn = getattr(block, "attn", None)
                    if attn is not None and hasattr(
                            attn, "enable_state_readonly_at_think"):
                        attn.enable_state_readonly_at_think()
                print("  Phase 2 enabled: state_readonly_at_think = True "
                      "(DeltaNet β forced to 0 at think positions)")
            # Phase-3 override: think_index_emb_size. -1 (default) =
            # leave as-built from cfg. A nonzero value mismatching the
            # existing embedding triggers a hard error rather than a
            # silent shape mismatch on save/load round-trip.
            req_size = int(getattr(args, "think_index_emb_size", -1))
            cur_size = int(model.think_index_emb_size)
            if req_size >= 0 and req_size != cur_size:
                raise RuntimeError(
                    f"Requested --think_index_emb_size={req_size} but "
                    f"loaded model has think_index_emb_size={cur_size}. "
                    "Rebuild the ckpt from scratch with the new size "
                    "instead (live resize would discard learned weights).")
            print(f"  with-thinking + ckpt-already-has-thinking: loaded as-is "
                  f"(memory + gate {'+ pkm ' if any(k.startswith('pkm_layer.') for k in sd_keys) else ''}"
                  f"think_id={thinking_token_id})")
            cfg["sft_with_thinking"] = True
            base_vocab_for_loss = int(cfg["vocab_size"]) - 1
            model.train()
            # Skip the rest of the with-thinking branch below.
            args_with_thinking_done = True
        else:
            args_with_thinking_done = False
    else:
        args_with_thinking_done = False

    # Three branches: modern (already done above), legacy (build fresh +
    # expand vocab), or original (no thinking).
    if args_with_thinking_done:
        # Modern path already loaded `model` and `cfg` above and applied
        # FIX A if requested. Don't re-load here — that would silently
        # overwrite the FIX A flag and any other modern-path setup.
        pass
    elif args.with_thinking and not args_with_thinking_done:
        # Legacy path: ckpt has no memory + gate. Build the model directly
        # with thinking + memory ON, then load the ckpt state with
        # strict=False so memory + gate heads stay freshly-initialised. The
        # think-token embedding gets fresh init at the new last vocab slot
        # (later over-written to embed-mean inside TinyLM.__init__ when
        # use_memory=True).
        from experiments.model import TinyLM
        from experiments.layers import DeltaNetAttention
        cfg = dict(raw_ckpt["config"])  # copy
        sd = raw_ckpt["state_dict"]
        base_vocab = int(cfg["vocab_size"])
        new_vocab = base_vocab + 1
        thinking_token_id = base_vocab
        # Expand embed + lm_head rows if needed (ckpt had output_gate=False,
        # so no extra slot was reserved).
        for key in ("embed.weight", "lm_head.weight"):
            if key in sd and sd[key].shape[0] < new_vocab:
                old = sd[key]
                pad = _t.zeros(new_vocab - old.shape[0], old.shape[1], dtype=old.dtype)
                sd[key] = _t.cat([old, pad], dim=0)
        fb_pairs = tuple(tuple(p) for p in cfg.get("feedback_pairs", ()) or ())
        model = TinyLM(
            vocab_size=new_vocab,
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]),
            d_head=int(cfg["d_head"]),
            max_T=int(cfg.get("max_T", 0)),
            feedback_mode=str(cfg.get("feedback_mode", "none")),
            feedback_pairs=fb_pairs,
            feedback_self_k=int(cfg.get("feedback_self_k", 0)),
            tie_embeddings=bool(cfg.get("tie_embeddings", True)),
            output_gate=True,
            use_memory=True,
            mem_size=int(args.mem_size),
            mem_dim=int(args.mem_dim) if args.mem_dim > 0 else int(cfg["d_model"]),
            thinking_token_id=thinking_token_id,
            mem_write_only_at_think=bool(args.mem_write_only_at_think),
            use_think_adapter=bool(args.use_think_adapter),
            think_adapter_hidden_mult=int(args.think_adapter_hidden_mult),
            attention_cls=DeltaNetAttention,
        ).cuda()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  with-thinking build: +1 vocab slot for think token "
              f"({thinking_token_id=}), missing={len(missing)} unexpected={len(unexpected)}")
        cfg["use_memory"] = True
        cfg["output_gate"] = True
        cfg["thinking_token_id"] = thinking_token_id
        cfg["vocab_size"] = new_vocab
        cfg["mem_size"] = int(args.mem_size)
        cfg["mem_dim"] = int(args.mem_dim) if args.mem_dim > 0 else int(cfg["d_model"])
        cfg["mem_write_only_at_think"] = bool(args.mem_write_only_at_think)
        cfg["sft_with_thinking"] = True
    else:
        # Original path: load whatever was saved, leave memory inert if present.
        from experiments.eval_bracket_structure import build_model_from_ckpt
        model, cfg = build_model_from_ckpt(args.load_ckpt)
        thinking_token_id = cfg.get("thinking_token_id")
    # --- Optional: disable WorkingMemory injection for this run ----------
    # Motivated by the 2026-05-19 ablation on v1: wm_off scored 12/164 vs
    # baseline 11/164, suggesting WM is at best neutral, slightly hurting.
    # Zero W_proj.weight (the output projection) and freeze it — every
    # injection becomes 0, so WM is structurally inert. The gate still
    # fires and think tokens are still inserted (so the rest of the
    # think-burst infrastructure works), but the WM-read path contributes
    # nothing to h.
    if (getattr(args, "disable_wm_during_sft", False)
            and hasattr(model, "memory")):
        with torch.no_grad():
            model.memory.W_proj.weight.zero_()
        model.memory.W_proj.weight.requires_grad_(False)
        cfg["disable_wm_during_sft"] = True
        print("  disable_wm_during_sft: WM.W_proj zeroed + frozen "
              "(injection = 0 for all positions)")
    model.train()
    # base_vocab_for_loss = index BELOW which targets are valid emit tokens.
    # Slicing logits[..., :base_vocab_for_loss] removes the thinking-token
    # slot (and any kernel-alignment padding above it) from the CE softmax.
    if args.with_thinking and thinking_token_id is not None:
        base_vocab_for_loss = int(thinking_token_id)
    else:
        base_vocab_for_loss = int(cfg["vocab_size"])
    print(f"  model: {cfg['n_layers']}L  d_model={cfg['d_model']}  "
          f"params={model.num_params() / 1e6:.1f}M  "
          f"with_thinking={args.with_thinking}  "
          f"vocab={cfg['vocab_size']} loss_slice=:{base_vocab_for_loss}")

    # --- 2. Tokenizer ------------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer",
                                                "HuggingFaceTB/SmolLM2-135M"))
    print(f"  tokenizer: {cfg.get('tokenizer')}  vocab={tok.vocab_size}")

    # --- 3. Data ----------------------------------------------------------
    # Rows are 3-tuples (problem, solution, cot_text_or_None); the CoT
    # field is non-None ONLY for Phase-4 prepare_for_thinking rows
    # produced by experiments/build_cot_sft_data.py.
    if args.distilled_jsonl:
        rows_with_cot = load_distilled_jsonl_with_cot(
            args.distilled_jsonl,
            prefer_full_completion=not args.distilled_code_only,
            require_extracted_code=True,
            keep_only_passing=args.distilled_keep_only_passing,
        )
    else:
        rows_with_cot = [(p, s, None)
                         for p, s in load_pairs(max_codealpaca=args.max_codealpaca)]
    print(f"  total pairs: {len(rows_with_cot)}")
    n_cot_rows = sum(1 for _, _, c in rows_with_cot if c)
    if n_cot_rows > 0 and (not args.with_thinking or thinking_token_id is None):
        # CoT-thinking rows REQUIRE the thinking machinery — the
        # think-token id has to exist in the vocab so the materialised
        # span is interpretable. Refuse to silently drop the CoT span.
        raise RuntimeError(
            f"{n_cot_rows} rows have prepare_for_thinking=True but "
            "--with_thinking is off or the ckpt has no thinking_token_id. "
            "Pass --with_thinking and use a ckpt that has the thinking "
            "vocab slot.")
    print(f"  tokenizing... ({n_cot_rows} cot-thinking rows, "
          f"{len(rows_with_cot) - n_cot_rows} plain rows)")
    # Each encoded entry is a 4-tuple
    #   (student_ids, student_labels, teacher_ids_or_None, gist_meta_or_None)
    # Phase 1a rows have teacher_ids + gist_meta; everything else has None,
    # None and the train loop treats them as a single-forward batch (or
    # mixed with thinking-gate burst insertion as before).
    encoded: list[tuple] = []
    skipped = 0
    use_phase_1a = (args.cot_compression_k or 0) > 0
    if use_phase_1a:
        print(f"  Phase 1a active: compression_k={args.cot_compression_k}, "
              f"min_thinks={args.cot_min_thinks}, "
              f"gist_weight={args.think_gist_weight} "
              f"({args.think_gist_loss_type})")
    for prompt, sol, cot_text in rows_with_cot:
        if cot_text and use_phase_1a:
            student_ids, labels, teacher_ids, gist_meta = (
                build_example_with_cot_compression(
                    prompt, cot_text, sol, tok,
                    int(thinking_token_id), args.max_len,
                    compression_k=int(args.cot_compression_k),
                    min_thinks=int(args.cot_min_thinks),
                )
            )
            if len(student_ids) < 8 or all(l == -100 for l in labels):
                skipped += 1
                continue
            encoded.append((student_ids, labels, teacher_ids, gist_meta))
        elif cot_text:
            full, labels = build_example_with_cot_thinking(
                prompt, cot_text, sol, tok,
                int(thinking_token_id), args.max_len,
            )
            if len(full) < 8 or all(l == -100 for l in labels):
                skipped += 1
                continue
            encoded.append((full, labels, None, None))
        else:
            full, labels = build_example(prompt, sol, tok, args.max_len)
            if len(full) < 8 or all(l == -100 for l in labels):
                skipped += 1
                continue
            encoded.append((full, labels, None, None))
    print(f"  encoded: {len(encoded)} (skipped {skipped})")
    pad_id = int(tok.eos_token_id) if tok.eos_token_id is not None else 0

    def make_batch(rows):
        """rows: list of (ids, labels) 2-tuples."""
        max_t = max(len(ids) for ids, _ in rows)
        max_t = min(max_t, args.max_len)
        bsz = len(rows)
        x = torch.full((bsz, max_t), pad_id, dtype=torch.long, device=device)
        y = torch.full((bsz, max_t), -100, dtype=torch.long, device=device)
        for i, (ids, labels) in enumerate(rows):
            ids = ids[:max_t]; labels = labels[:max_t]
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
        return x, y

    def make_teacher_batch(teacher_ids_list):
        """rows: list[list[int]]. Pads to the max-length teacher sequence."""
        max_t = max(len(ids) for ids in teacher_ids_list)
        max_t = min(max_t, args.max_len)
        bsz = len(teacher_ids_list)
        x = torch.full((bsz, max_t), pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(teacher_ids_list):
            ids = ids[:max_t]
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        return x

    # --- 4. Optim + train loop -------------------------------------------
    # v7 trunk multi-horizon gist heads (Fix C, 2026-05-20). The trunk's
    # job is "high-level direction": at position t each head predicts the
    # mean-pooled future hidden state over h[t+1:t+1+K] (the windowed
    # gist), one head per horizon K. This REPLACES both the old lexical
    # future-emb target (embed(input_ids[t+T]) — context-free) and the
    # v6 WM-injection gist supervision (which forced WM to emit blurry
    # gists and broke precise recall). WM is now left to learn precise
    # retrieval via the LM loss alone; "direction" lives in the trunk.
    d_model = int(cfg["d_model"])
    gist_horizons = parse_horizons(args.wm_gist_horizons)
    future_gist_heads = None
    if args.future_emb_loss_weight > 0:
        future_gist_heads = build_gist_heads(d_model, gist_horizons).cuda()
        _ck = locals().get("raw_ckpt")
        if _ck is None:
            _ck = torch.load(args.load_ckpt, map_location="cpu",
                              weights_only=False)
        if "future_gist_heads_state_dict" in _ck:
            try:
                future_gist_heads.load_state_dict(
                    _ck["future_gist_heads_state_dict"])
                print("  trunk-gist heads: restored from ckpt")
            except (RuntimeError, KeyError):
                print("  trunk-gist heads: ckpt horizons differ — fresh")
        print(f"  trunk-gist heads (v7 Fix C): d_model={d_model}, "
              f"horizons={gist_horizons}, "
              f"weight={args.future_emb_loss_weight}")
    if args.wm_future_pred_weight > 0:
        print("  NOTE: --wm_future_pred_weight is deprecated since v7 "
              "(WM gist supervision removed) — ignored.")

    # Phase 1a: per-think gist head (separate from the v7 trunk gist
    # heads). Single Linear(d_model, d_model) that projects student
    # think hidden states into the teacher's hidden-state space. Built
    # whenever Phase 1a is active (so the head exists for `loss_type`
    # ablations even if --think_gist_weight is 0 at this run).
    think_gist_head = None
    if use_phase_1a:
        think_gist_head = build_think_gist_head(d_model).cuda()
        _ck = locals().get("raw_ckpt")
        if _ck is None:
            _ck = torch.load(args.load_ckpt, map_location="cpu",
                              weights_only=False)
        if "think_gist_head_state_dict" in _ck:
            try:
                think_gist_head.load_state_dict(
                    _ck["think_gist_head_state_dict"])
                print("  Phase 1a think_gist_head: restored from ckpt")
            except (RuntimeError, KeyError):
                print("  Phase 1a think_gist_head: ckpt shape differs — fresh")
        print(f"  Phase 1a think_gist_head: d_model={d_model}, "
              f"weight={args.think_gist_weight}, "
              f"loss_type={args.think_gist_loss_type}")

    # Optimizer: retrieval_input_alpha gets NO weight decay (the FiLM-α
    # lesson — WD on a gate scalar manufactures a false low equilibrium).
    alpha_params, decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (alpha_params if n.endswith("retrieval_input_alpha")
         else decay_params).append(p)
    head_params = (list(future_gist_heads.parameters())
                   if future_gist_heads is not None else [])
    if think_gist_head is not None:
        head_params = head_params + list(think_gist_head.parameters())
    opt = torch.optim.AdamW(
        # WD=0.01 is the project default since 2026-05-14 (the v3a
        # residual-stream-collapse finding — see GEMINI.md). WD=0.1
        # was a Moonlight-scale (5.7 T-token) setting; at our
        # ~10 tok/param it acts as pure brake on the residual stream.
        [{"params": decay_params + head_params, "weight_decay": 0.01},
         {"params": alpha_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    n_steps = (len(encoded) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, n_steps), eta_min=args.lr * 0.1,
    )
    print(f"  total train steps: {n_steps}")
    print()

    rng = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    losses: list[float] = []
    step = 0
    # Phase 1a metric: track how the per-step gist loss trends. A real
    # signal will show this dropping monotonically over the first
    # ~hundred steps; a frozen/dead signal stays flat. Printed every
    # log_every steps when use_phase_1a is on.
    last_gist_loss = None
    # Phase A: most-recent process-reward stats for logging.
    last_pr_stats = None
    use_process_reward = (
        args.process_reward_weight > 0.0
        and args.with_thinking
        and thinking_token_id is not None
    )
    if args.process_reward_weight > 0.0 and not use_process_reward:
        print("  [process-reward] requested but disabled: requires "
              "--with_thinking AND a thinking_token_id in the loaded ckpt.")
    if use_process_reward:
        print(f"  [process-reward] ON: weight={args.process_reward_weight}, "
              f"K={args.process_reward_K}, "
              f"min_sigma={args.process_reward_apply_min_sigma}, "
              f"sample_frac={args.process_reward_sample_frac}, "
              f"max_positions={args.process_reward_max_positions}")
    for epoch in range(args.epochs):
        # Shuffle each epoch.
        idx = torch.randperm(len(encoded), generator=rng).tolist()
        for i in range(0, len(encoded) - args.batch + 1, args.batch):
            rows = [encoded[idx[j]] for j in range(i, i + args.batch)]
            # Partition into Phase 1a rows (have teacher_ids) and
            # legacy rows (None). Only legacy rows get
            # insert_think_bursts (the random-burst augmentation would
            # corrupt the structured think positions Phase 1a maps to
            # teacher CoT positions).
            phase_1a_rows = [(r[0], r[1], r[2], r[3]) for r in rows
                              if r[2] is not None]
            legacy_rows = [(r[0], r[1]) for r in rows if r[2] is None]

            if legacy_rows and args.with_thinking and thinking_token_id is not None:
                legacy_rows = [
                    insert_think_bursts(ids, lbls, int(thinking_token_id),
                                          args.max_len,
                                          args.think_max_bursts,
                                          args.think_max_depth, rng)
                    for ids, lbls in legacy_rows
                ]
            # If the whole batch is Phase 1a, skip the legacy forward; if
            # mixed, both forwards run and contribute additively.
            x, y = (make_batch(legacy_rows)
                    if legacy_rows else (None, None))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Need hidden states for the trunk gist aux loss.
                # Phase 1a ALSO needs hidden states (for the per-think gist).
                want_hidden = (future_gist_heads is not None) or bool(
                    phase_1a_rows)
                loss = torch.zeros((), device=device)
                # ====== Legacy path: only runs when there are non-Phase-1a rows
                if x is not None:
                    # --- Retrieval-as-input thinking (v7 additive, Fix B) ---
                    if (args.retrieval_as_input_thinking
                            and args.with_thinking
                            and thinking_token_id is not None):
                        with torch.no_grad():
                            _ = model(x)
                        inj = model.memory._last_injection  # (B,T,d) detached
                        base_emb = model.embed(x)
                        is_think = (x == int(thinking_token_id)).unsqueeze(-1)
                        shifted_inj = torch.cat(
                            [torch.zeros_like(inj[:, :1]), inj[:, :-1]],
                            dim=1,
                        )
                        alpha = model.retrieval_input_alpha
                        inputs_embeds = (
                            base_emb
                            + is_think.to(base_emb.dtype)
                            * alpha
                            * shifted_inj.to(base_emb.dtype)
                        )
                        if want_hidden:
                            logits, h = model(x, inputs_embeds=inputs_embeds,
                                              return_hidden=True)
                        else:
                            logits = model(x, inputs_embeds=inputs_embeds)
                    elif want_hidden:
                        logits, h = model(x, return_hidden=True)
                    else:
                        logits = model(x)
                    if args.with_thinking:
                        logits = logits[..., :base_vocab_for_loss]
                    shift_logits = logits[:, :-1].contiguous()
                    shift_labels = y[:, 1:].contiguous()
                    lm_loss = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    loss = loss + lm_loss
                    if future_gist_heads is not None:
                        loss = loss + args.future_emb_loss_weight * (
                            trunk_gist_loss(h, future_gist_heads, gist_horizons))
                    # --- Phase A: process-reward auxiliary loss ----------
                    # Reads the main-forward gate via `model._last_gate`
                    # (side-effect populated by TinyLM.forward when
                    # output_gate=True); reuses the just-computed `logits`
                    # for the "before" log-probabilities (no extra
                    # forward). The "after" forward runs over a small
                    # synthesised batch where K think tokens are inserted
                    # before each sampled position. Bounded by
                    # --process_reward_sample_frac and
                    # --process_reward_max_positions.
                    main_gate = getattr(model, "_last_gate", None)
                    if use_process_reward and main_gate is not None:
                        # PAD must NOT equal thinking_token_id — when
                        # state_readonly_at_think / mem_write_only_at_think
                        # are on, pad-as-think silently corrupts the
                        # after-forward's recurrent state. Use 0 (a
                        # bytefallback / safe non-think id).
                        pad_id = 0
                        pr_loss, pr_stats = compute_process_reward_loss(
                            model, x, y,
                            gate=main_gate,
                            main_logits=logits,
                            thinking_token_id=int(thinking_token_id),
                            K=int(args.process_reward_K),
                            apply_min_sigma=float(
                                args.process_reward_apply_min_sigma),
                            sample_frac=float(
                                args.process_reward_sample_frac),
                            rng=rng,
                            pad_token_id=pad_id,
                            retrieval_as_input=bool(
                                args.retrieval_as_input_thinking),
                            base_vocab_for_loss=(
                                int(base_vocab_for_loss)
                                if args.with_thinking else None),
                            max_positions=int(
                                args.process_reward_max_positions),
                        )
                        last_pr_stats = pr_stats
                        if pr_stats.n_sampled > 0:
                            loss = loss + (
                                args.process_reward_weight * pr_loss)
                # ====== Phase 1a path: gist-at-think supervision ============
                # Run the student forward over the compressed sequence and
                # the teacher forward (no_grad) over the full CoT sequence.
                # Student hidden states at think positions are projected
                # via think_gist_head and supervised against teacher hidden
                # states at the corresponding CoT chunk-end positions.
                if phase_1a_rows:
                    student_ids_list = [(r[0], r[1]) for r in phase_1a_rows]
                    teacher_ids_list = [r[2] for r in phase_1a_rows]
                    gist_metas = [r[3] for r in phase_1a_rows]
                    sx, sy = make_batch(student_ids_list)
                    tx = make_teacher_batch(teacher_ids_list)

                    # Teacher: no_grad forward, capture hidden states.
                    with torch.no_grad():
                        _t_logits, t_h = model(tx, return_hidden=True)
                    # Student: grad-enabled forward.
                    s_logits, s_h = model(sx, return_hidden=True)
                    if args.with_thinking:
                        s_logits = s_logits[..., :base_vocab_for_loss]
                    s_shift_logits = s_logits[:, :-1].contiguous()
                    s_shift_labels = sy[:, 1:].contiguous()
                    student_lm_loss = F.cross_entropy(
                        s_shift_logits.reshape(-1, s_shift_logits.size(-1)),
                        s_shift_labels.reshape(-1),
                        ignore_index=-100,
                    )
                    loss = loss + student_lm_loss
                    # Compose batched think-position / teacher-position
                    # lists, then call think_gist_loss. Truncate any
                    # positions that landed past the (post-truncation)
                    # padded sequence length to be safe.
                    sT = s_h.shape[1]
                    tT = t_h.shape[1]
                    think_positions = []
                    teacher_cot_positions = []
                    for gm in gist_metas:
                        sp = [p for p in gm["think_positions"] if p < sT]
                        tp = [p for q, p in zip(gm["think_positions"],
                                                gm["teacher_cot_positions"])
                              if q < sT and p < tT]
                        # Keep parallel: clip sp to len(tp) (drop tail
                        # mismatches caused by truncation).
                        n = min(len(sp), len(tp))
                        think_positions.append(sp[:n])
                        teacher_cot_positions.append(tp[:n])
                    if any(len(p) > 0 for p in think_positions):
                        g_loss = think_gist_loss(
                            s_h, t_h.detach(),
                            think_positions, teacher_cot_positions,
                            think_gist_head,
                            loss_type=args.think_gist_loss_type,
                        )
                        last_gist_loss = float(g_loss.detach().item())
                        if args.think_gist_weight > 0:
                            loss = loss + args.think_gist_weight * g_loss
                    if future_gist_heads is not None:
                        loss = loss + args.future_emb_loss_weight * (
                            trunk_gist_loss(s_h, future_gist_heads,
                                             gist_horizons))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                ppl = math.exp(min(loss.item(), 20))
                tok_s = (step * args.batch * args.max_len) / max(1, time.time() - t0)
                extra = ""
                if use_phase_1a and last_gist_loss is not None:
                    extra = f"  gist={last_gist_loss:.4f}"
                if use_process_reward and last_pr_stats is not None:
                    extra += (f"  pr(n={last_pr_stats.n_sampled}/"
                              f"{last_pr_stats.n_candidates}, "
                              f"Δlogp={last_pr_stats.mean_log_ratio:+.3f}, "
                              f"%pos={100 * last_pr_stats.frac_positive:.0f})")
                print(f"  step {step:>5}/{n_steps}  loss={loss.item():.4f}  "
                      f"ppl={ppl:.2f}  lr={lr:.2e}  tok/s={tok_s:.0f}{extra}")

    print(f"\nDone in {time.time() - t0:.0f}s.  Final loss: {losses[-1]:.4f}")

    # --- 5. Save ---------------------------------------------------------
    pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
    new_cfg = dict(cfg)
    new_cfg["sft_source"] = "MBPP+CodeAlpaca"
    new_cfg["sft_epochs"] = args.epochs
    # v7: record that retrieval-as-input used the additive α-gated form
    # so the eval generator picks the matching injection mode.
    new_cfg["retrieval_input_additive"] = bool(
        args.retrieval_as_input_thinking)
    if future_gist_heads is not None:
        new_cfg["future_emb_loss_weight"] = float(args.future_emb_loss_weight)
        new_cfg["wm_gist_horizons"] = list(gist_horizons)
    if think_gist_head is not None:
        new_cfg["cot_compression_k"] = int(args.cot_compression_k)
        new_cfg["cot_min_thinks"] = int(args.cot_min_thinks)
        new_cfg["think_gist_weight"] = float(args.think_gist_weight)
        new_cfg["think_gist_loss_type"] = str(args.think_gist_loss_type)
    if use_process_reward:
        new_cfg["process_reward_weight"] = float(args.process_reward_weight)
        new_cfg["process_reward_K"] = int(args.process_reward_K)
        new_cfg["process_reward_apply_min_sigma"] = float(
            args.process_reward_apply_min_sigma)
        new_cfg["process_reward_sample_frac"] = float(
            args.process_reward_sample_frac)
    ckpt_dict = {"state_dict": model.state_dict(), "config": new_cfg}
    if future_gist_heads is not None:
        ckpt_dict["future_gist_heads_state_dict"] = \
            future_gist_heads.state_dict()
    if think_gist_head is not None:
        ckpt_dict["think_gist_head_state_dict"] = think_gist_head.state_dict()
    torch.save(ckpt_dict, args.save_ckpt)
    print(f"saved: {args.save_ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
