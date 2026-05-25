"""Convert `data/repair_triples_v3.jsonl` (5247 broken→error→fix triples) to
a flat plain-text corpus consumable by `experiments.data_mix` as one more
pretrain source.

DeepSeek-Coder and similar models include debugging triples as NATURAL
CODE TEXT in pretrain. The previous attempt to use these triples as an
SFT target was a clean negative; folding them into pretrain at low
weight sidesteps the distribution-shift problem.

Input row (from `gen_repair_triples.py`):
    {
      "task_id":        "repair/mbpp_train/601/0",
      "problem_prompt": "# The previous attempt failed. Fix the code below.\\n"
                        "# {desc line 1}\\n# {desc line 2}\\n"
                        "# Attempted solution:\\n{failed_code}\\n"
                        "# Error:\\n# {err line 1}\\n# {err line 2}\\n"
                        "# Fix the code:\\n",
      "extracted_code": "{canonical_solution}",
      ...
    }

Output row (one per input row, written to
`data/pretrain_repair_corpus.jsonl`):
    {
      "task_id": "repair/{source_task_id}",
      "text":    "# Original problem:\\n"
                 "# {desc}\\n"
                 "# Attempted solution:\\n{failed_code}\\n\\n"
                 "# Got this error:\\n# {err}\\n\\n"
                 "# Fixed version:\\n{canonical_solution}\\n",
      "source":  "self_debug"
    }

The blank line at the end is intentional — gives a natural document
boundary in `data_mix.MixedSourceStream` (which appends EOS after each
document).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import NamedTuple


_HEADER = "# The previous attempt failed. Fix the code below."
_ATTEMPT_MARKER = "# Attempted solution:"
_ERROR_MARKER = "# Error:"
_FIX_MARKER = "# Fix the code:"


class ParsedRepair(NamedTuple):
    description: str       # multi-line if needed, no `# ` prefix
    failed_code: str
    error_text: str        # multi-line if needed, no `# ` prefix


def _strip_comment_prefix(lines: list[str]) -> str:
    """Drop the leading `# ` (or `#`) from each line, preserving content."""
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(line[2:])
        elif line.startswith("#"):
            out.append(line[1:])
        else:
            out.append(line)
    return "\n".join(out)


def parse_problem_prompt(prompt: str) -> ParsedRepair:
    """Pull (description, failed_code, error_text) out of `problem_prompt`.

    The format is the one produced by `iterative_repair.build_repair_prompt`:
    header line, description (`# `-prefixed lines), `# Attempted solution:`,
    failed code (no prefix), `# Error:`, error block (`# `-prefixed lines),
    `# Fix the code:` trailer.
    """
    lines = prompt.splitlines()
    if not lines or not lines[0].startswith(_HEADER.split("\n")[0]):
        raise ValueError(
            f"problem_prompt does not start with expected header: "
            f"{lines[0]!r}")

    try:
        i_attempt = lines.index(_ATTEMPT_MARKER)
    except ValueError as e:
        raise ValueError("missing '# Attempted solution:' marker") from e
    try:
        i_error = lines.index(_ERROR_MARKER, i_attempt + 1)
    except ValueError as e:
        raise ValueError("missing '# Error:' marker") from e
    try:
        i_fix = lines.index(_FIX_MARKER, i_error + 1)
    except ValueError as e:
        raise ValueError("missing '# Fix the code:' marker") from e

    desc_lines = lines[1:i_attempt]
    code_lines = lines[i_attempt + 1:i_error]
    err_lines = lines[i_error + 1:i_fix]

    description = _strip_comment_prefix(desc_lines).strip()
    failed_code = "\n".join(code_lines).rstrip()
    error_text = _strip_comment_prefix(err_lines).strip()

    if not description:
        description = "(no description)"
    if not failed_code:
        failed_code = "# (empty attempted solution)"
    if not error_text:
        error_text = "(no error text)"
    return ParsedRepair(description=description, failed_code=failed_code,
                         error_text=error_text)


def render_corpus_text(parsed: ParsedRepair, canonical_solution: str) -> str:
    """Build the flat-text training snippet from a parsed repair triple."""
    desc_block = "\n".join(f"# {line}" if line else "#"
                            for line in parsed.description.splitlines())
    err_block = "\n".join(f"# {line}" if line else "#"
                           for line in parsed.error_text.splitlines())
    canonical = canonical_solution.rstrip()
    parts = [
        "# Original problem:",
        desc_block,
        "# Attempted solution:",
        parsed.failed_code,
        "",
        "# Got this error:",
        err_block,
        "",
        "# Fixed version:",
        canonical,
        "",
    ]
    return "\n".join(parts)


def convert_row(row: dict) -> dict | None:
    """Convert one input JSONL row to the output row format."""
    prompt = row.get("problem_prompt")
    canonical = row.get("extracted_code") or ""
    src_task_id = row.get("task_id", "")
    if not prompt or not isinstance(prompt, str):
        return None
    if not canonical.strip():
        return None
    try:
        parsed = parse_problem_prompt(prompt)
    except ValueError:
        return None

    canonical = canonical.replace("\r\n", "\n").replace("\r", "\n")
    text = render_corpus_text(parsed, canonical)
    return {"task_id": src_task_id, "text": text, "source": "self_debug"}


def build_corpus(in_path: pathlib.Path, out_path: pathlib.Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    n_tokens_approx = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out_row = convert_row(row)
            if out_row is None:
                continue
            n_out += 1
            n_tokens_approx += len(out_row["text"].split())
            fout.write(json.dumps(out_row) + "\n")
    return {"n_in": n_in, "n_out": n_out, "n_tokens_approx": n_tokens_approx,
            "out_path": str(out_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in_path", default="data/repair_triples_v3.jsonl")
    p.add_argument("--out_path", default="data/pretrain_repair_corpus.jsonl")
    args = p.parse_args()
    in_path = pathlib.Path(args.in_path)
    out_path = pathlib.Path(args.out_path)
    if not in_path.exists():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 1
    stats = build_corpus(in_path, out_path)
    print(f"converted {stats['n_out']}/{stats['n_in']} rows -> "
          f"{stats['out_path']}")
    print(f"approx tokens (whitespace-split): {stats['n_tokens_approx']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
