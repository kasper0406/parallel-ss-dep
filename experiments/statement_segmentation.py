"""
Statement-level segmentation for Python code, used by the structural-surprise
oracle and eval pipeline.

We adapt the design doc's "sentence boundaries" to "Python statement
boundaries" via `ast.parse`. Each leaf statement (Assign, Return, Expr, If,
For, While, FunctionDef, ClassDef body items, etc.) is mapped from line/col
offsets to a token range using the HuggingFace tokenizer's offset_mapping.

Conventions:
- We collect *all* statement-like AST nodes recursively, so a function body
  contributes one statement per assign/return/etc. (not just the FunctionDef
  itself). This matches the design doc's intuition of "statements as the
  unit of structural surprise" — a function definition framing shift, an
  internal return, a conditional branch are all candidates.
- For each statement we record (start_token_idx, end_token_idx_exclusive)
  in the token stream of the file. End is exclusive in the usual Python
  convention.
- We skip statements that map to fewer than 1 token (rare edge cases like
  blank docstrings).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterator


# AST node classes that we treat as a "statement" for the purposes of
# pooling and scoring. All Python statement-like nodes have a `lineno`
# and `end_lineno`. We deliberately include FunctionDef and ClassDef
# (the *header* surprise, not just the body) and walk into their bodies
# to also pick up inner statements separately.
_STMT_TYPES = (
    ast.Assign, ast.AnnAssign, ast.AugAssign,
    ast.Return, ast.Expr, ast.Raise, ast.Assert,
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
    ast.Try, ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Pass, ast.Break, ast.Continue, ast.Delete,
    ast.Global, ast.Nonlocal,
)


@dataclass
class Statement:
    """A single statement within a tokenized file.

    start_tok_idx, end_tok_idx are inclusive/exclusive bounds into the
    file's token stream. `text` is the raw source text (may span multiple
    lines for compound statements).
    """
    start_tok_idx: int
    end_tok_idx: int            # exclusive
    text: str
    kind: str                   # AST node class name (e.g. "Assign", "FunctionDef")


def _walk_stmts(node: ast.AST) -> Iterator[ast.AST]:
    """Yield all statement-like nodes (recursive).

    A FunctionDef is yielded once for its 'header' (which we'll trim to
    only span the def-line, not the body), and we then recurse into its
    body.
    """
    body = getattr(node, "body", None)
    if body is None:
        return
    for child in body:
        if isinstance(child, _STMT_TYPES):
            yield child
            # Recurse into nested bodies (function bodies, class bodies,
            # if-branches, try-blocks, etc.) so inner statements are
            # collected too.
            if hasattr(child, "body"):
                yield from _walk_stmts(child)
            for attr in ("orelse", "finalbody", "handlers"):
                sub = getattr(child, attr, None)
                if sub is not None:
                    for h in sub:
                        if isinstance(h, _STMT_TYPES):
                            yield h
                            yield from _walk_stmts(h)
                        elif isinstance(h, ast.ExceptHandler):
                            yield from _walk_stmts(h)


def line_col_to_byte_offset(text: str, line: int, col: int) -> int:
    """Convert (line, col) — both 1-indexed line, 0-indexed col — to a
    byte offset into `text`.

    The Python AST uses 1-indexed `lineno` and 0-indexed `col_offset`.
    """
    if line < 1:
        return 0
    # Walk lines and accumulate.
    pos = 0
    cur_line = 1
    while cur_line < line and pos < len(text):
        nl = text.find("\n", pos)
        if nl < 0:
            break
        pos = nl + 1
        cur_line += 1
    return pos + col


def find_token_range(offsets: list[tuple[int, int]],
                     start_byte: int, end_byte: int) -> tuple[int, int]:
    """Map a [start_byte, end_byte) range to a token range
    [start_tok, end_tok) in the offsets list.

    A token is INCLUDED if any of its bytes overlap [start_byte, end_byte).
    Returns (start_tok, end_tok) where end_tok is exclusive. Returns
    (-1, -1) if no token overlaps.
    """
    n = len(offsets)
    start_tok = -1
    end_tok = -1
    for i, (lo, hi) in enumerate(offsets):
        # Tokens with (0, 0) are usually special tokens — skip.
        if lo == 0 and hi == 0:
            continue
        if hi <= start_byte:
            continue
        if lo >= end_byte:
            break
        if start_tok < 0:
            start_tok = i
        end_tok = i + 1
    return (start_tok, end_tok)


def parse_file_statements(
    text: str,
    tokenizer,
    *,
    header_only_for_compound: bool = True,
    max_stmt_lines: int | None = None,
) -> tuple[list[int], list[Statement]]:
    """Tokenize `text` and segment into statements with token ranges.

    Returns:
        (token_ids, statements) where statements is a list of `Statement`
        with token ranges. `token_ids` has length n_tokens; the
        statements' end_tok_idx values are bounded by n_tokens.

    Args:
        header_only_for_compound: If True, FunctionDef/ClassDef/If/For/While
            nodes are trimmed to only span their header line (the line they
            start on, up to but not including the body). This matches the
            design doc's intent of "function-def is one statement, internal
            return is another statement" without double-counting bytes.
        max_stmt_lines: If set, drops statements spanning more lines than
            this (extremely long compound bodies). None = no limit.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    offsets = list(enc["offset_mapping"])
    if not text.strip() or not token_ids:
        return token_ids, []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        # If we can't parse, return empty statement list (caller will
        # fall back to the whole file as one "statement").
        return token_ids, []

    statements: list[Statement] = []
    seen_ranges: set[tuple[int, int]] = set()
    for node in _walk_stmts(tree):
        line_start = node.lineno
        col_start = node.col_offset
        line_end = node.end_lineno or line_start
        col_end = node.end_col_offset or 0

        # For compound statements (def, class, if, for, while, try, with),
        # we only want the *header* line — the colon or the line up to it —
        # so the structural surprise is computed on the framing token, not
        # on the entire body's text. Walk to end of the header line.
        if (header_only_for_compound and
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef, ast.If, ast.For,
                                   ast.AsyncFor, ast.While, ast.With,
                                   ast.AsyncWith, ast.Try))):
            # End at the end of the header's line.
            line_end = line_start
            # Find the end of this line in the text.
            # Bytes-into-line: find position of newline after header start.
            byte_start_of_header = line_col_to_byte_offset(text, line_start, 0)
            nl = text.find("\n", byte_start_of_header)
            if nl < 0:
                col_end = len(text) - byte_start_of_header
            else:
                col_end = nl - byte_start_of_header

        if max_stmt_lines is not None and (line_end - line_start + 1) > max_stmt_lines:
            continue

        start_byte = line_col_to_byte_offset(text, line_start, col_start)
        end_byte = line_col_to_byte_offset(text, line_end, col_end)
        if end_byte <= start_byte:
            continue
        start_tok, end_tok = find_token_range(offsets, start_byte, end_byte)
        if start_tok < 0 or end_tok <= start_tok:
            continue
        # Dedupe (header-only trim can produce coincident ranges).
        key = (start_tok, end_tok)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        stmt_text = text[start_byte:end_byte]
        statements.append(Statement(
            start_tok_idx=start_tok, end_tok_idx=end_tok,
            text=stmt_text, kind=type(node).__name__,
        ))
    # Sort by start position, stable.
    statements.sort(key=lambda s: (s.start_tok_idx, s.end_tok_idx))
    return token_ids, statements
