"""Iterative repair helpers for grader-RL.

When a rollout fails, do a second rollout with the grader's `error_text`
as repair context. Both rollouts contribute to the GRPO group, which
turns some zero-variance groups into variance-bearing ones (the model's
gradient signal is concentrated on the "sometimes-works" zone; if every
attempt on a problem fails, repair gives us a second chance to find
the working completion that produces a non-zero advantage).

The repair prompt matches the SFT comment-style format
(`# {description}\n` header, used by `sft_code.build_example` and
`eval_humaneval.py --prompt_style sft_comment`).
"""
from __future__ import annotations

REPAIR_HEADER = "# The previous attempt failed. Fix the code below.\n"


def build_repair_prompt(original_prompt: str, failed_code: str,
                         error_text: str) -> str:
    """Build the augmented prompt fed back into a repair rollout.

    Format (matches the comment-style SFT prompt convention):

        # The previous attempt failed. Fix the code below.
        {original_prompt}
        # Attempted solution:
        {failed_code}
        # Error:
        # {error line 1}
        # {error line 2}
        # Fix the code:

    The error block is prefixed `# ` per line so the whole concatenation
    stays inside Python comments (parses as a no-op if the model's
    completion is appended raw).
    """
    err = (error_text or "").strip() or "(no error text)"
    err_lines = err.splitlines() or [""]
    err_block = "\n".join(f"# {line}" for line in err_lines)
    code = (failed_code or "").strip()
    parts = [
        REPAIR_HEADER.rstrip("\n"),
        original_prompt.rstrip("\n"),
        "# Attempted solution:",
        code,
        "# Error:",
        err_block,
        "# Fix the code:",
        "",
    ]
    return "\n".join(parts)


def select_repair_targets(
    rewards: list[float], *, max_per_group: int, min_failed: int,
    pass_threshold: float = 0.5,
) -> list[int]:
    """Return indices of failed rollouts that should be repaired.

    Returns at most `max_per_group` indices. Returns an empty list when
    fewer than `min_failed` rollouts are below `pass_threshold` (the
    group is already variance-bearing; repair is wasted there).
    """
    failed = [i for i, r in enumerate(rewards) if r < pass_threshold]
    if len(failed) < min_failed:
        return []
    return failed[:max_per_group]


def group_became_variance_bearing(
    original_rewards: list[float], repair_rewards: list[float],
    *, pass_threshold: float = 0.5,
) -> bool:
    """True iff the original group had zero variance (all-pass or all-fail)
    but the combined group has at least one rollout on the other side
    of `pass_threshold` after repair.
    """
    if not original_rewards or not repair_rewards:
        return False
    orig_passes = sum(1 for r in original_rewards if r >= pass_threshold)
    orig_zero_var = orig_passes == 0 or orig_passes == len(original_rewards)
    if not orig_zero_var:
        return False
    combined_passes = sum(1 for r in (original_rewards + repair_rewards)
                          if r >= pass_threshold)
    return 0 < combined_passes < len(original_rewards) + len(repair_rewards)
