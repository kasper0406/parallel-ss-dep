"""Cross-file usage-prediction episode generator (Meta-TTT Phase P0).

Builds `[repo context → cross-file usage task]` episodes for the repo-adaptive
(meta-TTT) kill-test (`META_TTT_PLAN_2026_07_13.md`). Each episode: an
identifier is DEFINED in one file (placed early in the context) and CALLED in a
different file (placed last); the supervision target is the usage line. The
cross-file def→use link is the only signal that "reading the repo" provides —
exactly the dep-distance-stratified probe scaled across files.

Repo source
-----------
codeparrot/codeparrot-clean (streaming). It is the ONLY multi-file-repo source
usable here: it exposes `repo_name` + `path` + `content` and streams without
auth. (bigcode/the-stack-smol is gated; codeparrot/github-code ships a dataset
script `datasets` no longer supports.) Files from one repo are scattered across
the (deduped) stream, so we GROUP by `repo_name` over a bounded scan.

CONTAMINATION NOTE: codeparrot IS in our pretrain mix (weight 0.30). Every eval
repo was therefore potentially seen in pretrain. The kill-test metric
lift(real − shuffled) is ROBUST to this (memorising the def helps the real AND
shuffled AND none arms roughly equally, so it cancels in real − shuffled); the
real − none lift is the one memorisation deflates. `--exclude_pretrain_overlap`
is a best-effort de-contaminator (see main()).

Tokenization / control fairness
-------------------------------
The shuffled-repo control must be EXACTLY length-matched to the real repo
(same token count) so the only variable is structure. Whole-file BPE glues a
newline to the following indentation (`:\n    ` is one token), so naive
text-level line-shuffle drifts the token count AND leaves indented blocks
un-shufflable. Instead, CONTEXT files are tokenised PER PHYSICAL LINE (each
`splitlines(keepends)` line encoded independently; see `perline_ids`). Per-line
tokenization is permutation-invariant at the TEXT level: shuffling file order or
line order is a pure reordering of the same per-line chunks → identical token
count, with text-only storage (no token_ids needed). Files are normalised to
end in a newline first, so a final line without a trailing newline cannot glue
to another line when permuted. The CE-target `task_line` is tokenised
canonically (single line → identical either way).

Usage
-----
  # generate the corpus (train + eval split, controls for the eval split):
  PYTHONPATH=. .venv/bin/python experiments/gen_repo_episodes.py \\
      --scan_rows 400000 --n_train 500 --n_eval 150 \\
      --out_dir data/repo_episodes --make_controls

  # tiny smoke (few repos):
  PYTHONPATH=. .venv/bin/python experiments/gen_repo_episodes.py \\
      --smoke --out_dir /tmp/repo_ep_smoke
"""
from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
_BUILTIN_NAMES = set(dir(builtins))

# Generic / throwaway identifier names that are almost always coincidental
# cross-file name collisions (test fixtures, closures) rather than a real
# def→use dependency. Skipping them raises link precision for the kill-test.
_GENERIC_NAMES = {
    "func", "foo", "bar", "baz", "test", "main", "run", "setup", "teardown",
    "wrapper", "inner", "fn", "cb", "callback", "handler", "func2", "f1", "f2",
    "tmp", "dummy", "sample", "example", "target", "source",
}

# Token-budget bucket edges (n_ctx_tokens). >MAX rejected, <MIN rejected.
BUCKETS = [("4-8k", 4000, 8000), ("8-16k", 8000, 16000), ("16-32k", 16000, 32000)]
MIN_CTX_TOKENS = 4000
MAX_CTX_TOKENS = 32000


# --------------------------------------------------------------------------- #
# AST cross-file link mining.
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Link:
    identifier: str
    def_path: str
    def_line: int          # 1-based lineno of the definition
    use_path: str
    use_line: int          # 1-based lineno of the usage
    use_col: int           # char offset of the identifier within use_line
    use_end_col: int
    imported: bool = False  # use file has `from ... import <identifier>` etc.


def _def_quality(node: ast.AST) -> bool:
    """A definition is "quality" (knowing it informs the usage) if it has a
    real signature/body: >=2 args OR a docstring OR >=3 body statements."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        n_args = (len(a.args) + len(a.posonlyargs) + len(a.kwonlyargs)
                  + (1 if a.vararg else 0) + (1 if a.kwarg else 0))
        has_doc = ast.get_docstring(node) is not None
        body_lines = len(node.body)
        return n_args >= 2 or has_doc or body_lines >= 3
    if isinstance(node, ast.ClassDef):
        has_doc = ast.get_docstring(node) is not None
        # count methods / body statements
        return has_doc or len(node.body) >= 3
    return False


def _collect_defs(tree: ast.AST) -> dict[str, tuple[int, ast.AST]]:
    """name -> (lineno, node) for every FunctionDef/AsyncFunctionDef/ClassDef
    anywhere in the module. First occurrence wins on duplicate names."""
    out: dict[str, tuple[int, ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name not in out:
                out[node.name] = (node.lineno, node)
    return out


def _collect_name_calls(tree: ast.AST) -> list[tuple[str, int, int, int]]:
    """Direct-call sites `name(...)` (ast.Call with ast.Name func) — the
    cleanest cross-file link with an exact identifier span. Returns
    (name, lineno, col_offset, end_col_offset)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            f = node.func
            out.append((f.id, f.lineno, f.col_offset, f.end_col_offset))
    return out


def _collect_imported_names(tree: ast.AST) -> set[str]:
    """Names brought into scope by `import`/`from ... import` (the bound local
    name — alias if present, else the imported symbol / top-level module)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def _parse(text: str) -> ast.AST | None:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None


def mine_cross_file_links(files: dict[str, str], skip_generic: bool = True,
                          require_import: bool = False,
                          min_name_len: int = 3) -> list[Link]:
    """Find identifiers DEFINED in one file and CALLED in a different file.

    Rules (pinned by tests):
      * definition must pass `_def_quality`.
      * the name must be defined in EXACTLY ONE file (unambiguous attribution).
      * the use file must NOT itself define the name (ignore same-file uses).
      * builtins are ignored.
      * the used identifier must actually appear at the recorded span.
    Precision heuristics (default on): drop generic/throwaway names and
    names shorter than `min_name_len` (coincidental collisions). Links whose
    use file IMPORTS the identifier are marked `imported=True` and sorted
    first; `require_import` keeps only those.
    """
    trees = {p: _parse(t) for p, t in files.items()}
    trees = {p: tr for p, tr in trees.items() if tr is not None}

    # name -> list of (path, lineno, node)  (across files)
    defs_by_name: dict[str, list[tuple[str, int, ast.AST]]] = defaultdict(list)
    defs_per_file: dict[str, dict[str, tuple[int, ast.AST]]] = {}
    imports_per_file: dict[str, set[str]] = {}
    for p, tr in trees.items():
        d = _collect_defs(tr)
        defs_per_file[p] = d
        imports_per_file[p] = _collect_imported_names(tr)
        for name, (ln, node) in d.items():
            defs_by_name[name].append((p, ln, node))

    links: list[Link] = []
    seen: set[tuple[str, str, str]] = set()  # (identifier, def_path, use_path)
    for use_path, tr in trees.items():
        local_defs = defs_per_file.get(use_path, {})
        use_imports = imports_per_file.get(use_path, set())
        lines = files[use_path].splitlines()
        for name, lineno, col, end_col in _collect_name_calls(tr):
            if name in _BUILTIN_NAMES:
                continue
            if skip_generic and (name in _GENERIC_NAMES
                                 or len(name) < min_name_len):
                continue
            if name in local_defs:            # same-file use — skip
                continue
            hits = defs_by_name.get(name)
            if not hits:
                continue
            if len(hits) != 1:                # ambiguous attribution — skip
                continue
            def_path, def_line, def_node = hits[0]
            if def_path == use_path:
                continue
            if not _def_quality(def_node):
                continue
            imported = name in use_imports
            if require_import and not imported:
                continue
            # validate the span on the source line
            if not (1 <= lineno <= len(lines)):
                continue
            line = lines[lineno - 1]
            c0, c1 = col, end_col
            if not (0 <= c0 < c1 <= len(line) and line[c0:c1] == name):
                idx = line.find(name)
                if idx < 0:
                    continue
                c0, c1 = idx, idx + len(name)
            key = (name, def_path, use_path)
            if key in seen:
                continue
            seen.add(key)
            links.append(Link(name, def_path, def_line, use_path, lineno,
                              c0, c1, imported=imported))
    # deterministic order: imported first, then by use_path / line / identifier
    links.sort(key=lambda L: (not L.imported, L.use_path, L.use_line,
                              L.identifier))
    return links


# --------------------------------------------------------------------------- #
# Token helpers.
#
# CONTEXT files are tokenised PER PHYSICAL LINE (each `splitlines(keepends)`
# line encoded independently, ids concatenated). Two reasons:
#   1. It is permutation-invariant at the TEXT level — shuffling file order or
#      line order preserves the total token count EXACTLY (the shuffled control
#      is length-matched with text-only storage; no token_ids needed).
#   2. It cleanly separates every logical line. Whole-file BPE merges a
#      newline with the following indentation (`:\n    ` is one token), which
#      would glue an indented function body into a single un-shufflable chunk
#      and let the definition survive the "shuffled" control.
# The mild non-canonicality (no cross-newline merges) applies EQUALLY to all
# arms, so arm comparisons stay fair; absolute CE is on this tokenisation.
# The CE-target `task_line` is tokenised canonically (single line → identical).
# --------------------------------------------------------------------------- #

def split_lines_nl(text: str) -> list[str]:
    """Split `text` into lines on '\\n' ONLY (never on the other boundary chars
    `str.splitlines` also cuts — `\\r`, `\\x0c` form-feed, unicode separators —
    which are common in Python files and would break the length-match invariant:
    they leave a chunk NOT ending in '\\n', and normalising it by appending '\\n'
    changes its tokenization, e.g. `\\r` → `\\r\\n`). Every returned line ends in
    '\\n', so any permutation re-joins and re-splits to the SAME chunks → exact
    token-count preservation. `\\r`/`\\x0c` stay as ordinary in-line characters."""
    if text == "":
        return []
    t = text if text.endswith("\n") else text + "\n"
    return [ln + "\n" for ln in t.split("\n")[:-1]]


def perline_ids(text: str, tok) -> list[int]:
    """Token ids of `text`, each physical line (split on '\\n') encoded
    independently. Permutation-invariant in total token count."""
    ids: list[int] = []
    for line in split_lines_nl(text):
        ids.extend(tok(line, add_special_tokens=False)["input_ids"])
    return ids


# Bounded cache of per-file per-line token counts: assemble_context/build_episode
# re-tokenize every repo file once per mined link, so the same file text repeats.
# Keyed on text (the tokenizer is fixed within a run). Correctness-neutral.
_PERLINE_LEN_CACHE: dict[str, int] = {}


def perline_len(text: str, tok) -> int:
    v = _PERLINE_LEN_CACHE.get(text)
    if v is None:
        v = len(perline_ids(text, tok))
        if len(_PERLINE_LEN_CACHE) < 200000:
            _PERLINE_LEN_CACHE[text] = v
    return v


# --------------------------------------------------------------------------- #
# Episode assembly.
# --------------------------------------------------------------------------- #

def _split_prefix_line(text: str, lineno: int) -> tuple[str, str]:
    """Return (prefix, line) where prefix is `text` up to (not incl.) the
    1-based `lineno`, and line is that physical line (with its trailing
    newline if present). `prefix + line` == text up through that line."""
    keep = text.splitlines(keepends=True)
    prefix = "".join(keep[:lineno - 1])
    line = keep[lineno - 1] if lineno - 1 < len(keep) else ""
    return prefix, line


def assemble_context(files: dict[str, str], def_path: str, use_path: str,
                     tok, max_ctx_tokens: int = MAX_CTX_TOKENS,
                     budget: int | None = None) -> list[str] | None:
    """Choose an ordered file list: def_path FIRST (first half), use_path LAST,
    other files greedily in between until the fill `budget` is hit (def+use are
    always included). Returns ordered paths or None if def+use alone exceed the
    hard `max_ctx_tokens` cap."""
    if budget is None:
        budget = max_ctx_tokens
    budget = min(budget, max_ctx_tokens)
    tok_len = {p: perline_len(t, tok) for p, t in files.items()}
    # task-file contributes only its PREFIX to n_ctx, but for budget purposes
    # use the full-file length as a conservative upper bound.
    base = tok_len[def_path] + tok_len[use_path]
    if base > max_ctx_tokens:
        return None
    others = sorted(p for p in files if p not in (def_path, use_path))
    chosen = []
    total = base
    for p in others:
        if total + tok_len[p] > budget:
            continue
        chosen.append(p)
        total += tok_len[p]
    ordered = [def_path] + chosen + [use_path]
    return ordered


def build_episode(repo_name: str, files: dict[str, str], link: Link, tok,
                  max_ctx_tokens: int = MAX_CTX_TOKENS,
                  min_ctx_tokens: int = MIN_CTX_TOKENS,
                  vary_budget: bool = True, seed: int = 0) -> dict | None:
    """Assemble one episode dict from a mined link. Returns None if it cannot
    be built within the token budget or fails validation. When `vary_budget`,
    the fill target is sampled deterministically in [min,max] per (repo,link)
    so episodes spread across the n_ctx buckets (needed for the curve-bending
    eval) rather than all saturating the 32k cap."""
    if link.def_path not in files or link.use_path not in files:
        return None
    budget = max_ctx_tokens
    if vary_budget:
        hb = int(hashlib.sha1(
            f"{seed}:{repo_name}:{link.identifier}:budget".encode()
        ).hexdigest()[:12], 16)
        budget = min_ctx_tokens + hb % max(1, (max_ctx_tokens - min_ctx_tokens))
    ordered = assemble_context(files, link.def_path, link.use_path, tok,
                               max_ctx_tokens, budget=budget)
    if ordered is None:
        return None
    use_text = files[link.use_path]
    task_prefix, task_line = _split_prefix_line(use_text, link.use_line)
    if not task_line.strip():
        return None
    # a usage on line 1 gives an empty task_prefix → the `none` arm would have
    # no predictor for the first target token. Require a non-empty prefix.
    if not task_prefix:
        return None
    # char span of the identifier within task_line (link cols are on the
    # source line == the stripped-of-prefix physical line).
    span0, span1 = link.use_col, link.use_end_col
    if not (0 <= span0 < span1 <= len(task_line) and
            task_line[span0:span1] == link.identifier):
        idx = task_line.find(link.identifier)
        if idx < 0:
            return None
        span0, span1 = idx, idx + len(link.identifier)

    # context = ordered files, but the LAST (use) file is represented by its
    # prefix at eval time; store full text for repo fidelity.
    context_files = [{"path": p, "text": files[p]} for p in ordered]

    # n_ctx_tokens = all context files EXCEPT the task file (whose prefix is
    # counted instead) + task_prefix.  (per-line tokenization; see header)
    n_ctx = 0
    for p in ordered[:-1]:
        n_ctx += perline_len(files[p], tok)
    n_ctx += perline_len(task_prefix, tok)

    if n_ctx < min_ctx_tokens or n_ctx > max_ctx_tokens:
        return None

    identifier_in_prefix = link.identifier in task_prefix
    # finer signal for the none-arm stratification: does the identifier appear
    # in the prefix OUTSIDE of import statements (a genuine local usage the none
    # arm can copy a pattern from) vs only via `from ... import <id>` (name-only
    # token hint)? The import case dominates for imported links, so the
    # non-import flag is the discriminating cut for "none arm truly blind".
    id_nonimport = any(
        link.identifier in ln and not ln.lstrip().startswith(("import ", "from "))
        for ln in task_prefix.splitlines())

    # unique per (repo, def-site, use-site) — the SAME identifier can be used in
    # multiple files within a repo, producing several episodes that share
    # (repo, identifier); a unique id keeps episode↔control matching 1:1.
    episode_id = hashlib.sha1(
        f"{repo_name}\x00{link.def_path}\x00{link.def_line}\x00{link.use_path}"
        f"\x00{link.use_line}\x00{link.identifier}".encode()).hexdigest()[:16]

    ep = {
        "episode_id": episode_id,
        "repo_name": repo_name,
        "context_files": context_files,
        "task_file": link.use_path,
        "task_prefix": task_prefix,
        "task_line": task_line,
        "task_char_span": [span0, span1],
        "link": {
            "identifier": link.identifier,
            "def_path": link.def_path,
            "def_line": link.def_line,
            "use_path": link.use_path,
            "use_line": link.use_line,
            "imported": link.imported,
        },
        "n_ctx_tokens": n_ctx,
        "identifier_in_task_prefix": identifier_in_prefix,
        "identifier_in_prefix_nonimport": id_nonimport,
        "bucket": bucket_of(n_ctx),
        "is_control": False,
    }
    return ep


def _build_episode_relaxed(files: dict[str, str], link: Link, tok) -> dict:
    """Test helper: build an episode ignoring the 4k min-token floor (synthetic
    test repos are tiny). Not used by the corpus builder."""
    ep = build_episode("test/repo", files, link, tok, min_ctx_tokens=0)
    assert ep is not None, "relaxed episode build failed"
    return ep


def bucket_of(n_ctx: int) -> str | None:
    for name, lo, hi in BUCKETS:
        if lo <= n_ctx < hi:
            return name
    if n_ctx == MAX_CTX_TOKENS:
        return BUCKETS[-1][0]
    return None


# --------------------------------------------------------------------------- #
# Shuffled-repo control builder.
# --------------------------------------------------------------------------- #

def _shuffle_nonidentity(rng: random.Random, n: int) -> list[int]:
    """A permutation of range(n) guaranteed != identity for n > 1."""
    if n <= 1:
        return list(range(n))
    ident = list(range(n))
    order = ident[:]
    for _ in range(8):
        rng.shuffle(order)
        if order != ident:
            return order
    # deterministic fallback: rotate by 1 (never identity for n > 1)
    return ident[1:] + ident[:1]


def build_control(episode: dict, tok, seed: int = 0) -> dict:
    """Build the shuffled-repo control for an eval episode:
      (a) shuffle the ORDER of context files except the task file (kept LAST);
      (b) permute each non-task file's physical LINES (destroys the definition's
          coherence). Per-line tokenization (see header) makes this EXACTLY
          length-matched to the real file with text-only storage.
    The task file (last) is untouched. Deterministic under `seed`."""
    ctx = episode["context_files"]
    task_path = episode["task_file"]
    non_task = ctx[:-1]
    task_file = ctx[-1]
    assert task_file["path"] == task_path, "task file must be last in context"

    h = hashlib.sha1(
        f"{seed}:{episode['repo_name']}:{episode['link']['identifier']}"
        .encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))

    # (a) shuffle order of non-task files
    order = _shuffle_nonidentity(rng, len(non_task))
    shuffled_non_task = []
    for i in order:
        f = non_task[i]
        # split on '\n' only (see split_lines_nl): every chunk ends in '\n', so
        # the permutation is EXACTLY length-preserving under the eval's per-line
        # re-tokenization — for all files incl. those with no trailing newline
        # or embedded \r/\x0c.
        lines = split_lines_nl(f["text"])
        lorder = _shuffle_nonidentity(rng, len(lines))
        permuted = "".join(lines[j] for j in lorder)
        shuffled_non_task.append({"path": f["path"], "text": permuted})
    control_ctx = shuffled_non_task + [task_file]  # task file untouched, last

    ctrl = dict(episode)
    ctrl["context_files"] = control_ctx
    ctrl["is_control"] = True
    ctrl["control_of"] = {
        "repo_name": episode["repo_name"],
        "identifier": episode["link"]["identifier"],
    }
    return ctrl


# --------------------------------------------------------------------------- #
# Repo split.
# --------------------------------------------------------------------------- #

def repo_split(repo_name: str, eval_frac: float = 0.23, seed: int = 0) -> str:
    """Deterministic disjoint train/eval split by repo hash."""
    h = int(hashlib.sha1(f"{seed}:{repo_name}".encode()).hexdigest(), 16)
    return "eval" if (h % 1000) < int(eval_frac * 1000) else "train"


# --------------------------------------------------------------------------- #
# Streaming corpus builder.
# --------------------------------------------------------------------------- #

def _iter_repos(scan_rows: int, min_files: int, seed: int,
                exclude_names: set[str] | None = None,
                max_copies: int | None = None,
                verbose: bool = True):
    """Group codeparrot-clean rows by repo_name over a bounded scan; yield
    (repo_name, {path: content}) for repos with >= min_files python files."""
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train",
                      streaming=True)
    repo_files: dict[str, dict[str, str]] = defaultdict(dict)
    t0 = time.time()
    for i, row in enumerate(ds):
        if i >= scan_rows:
            break
        rn = row["repo_name"]
        if exclude_names is not None and rn in exclude_names:
            continue
        if max_copies is not None:
            try:
                if int(row.get("copies", "1")) > max_copies:
                    continue
            except (ValueError, TypeError):
                pass
        path = row["path"]
        if not path.endswith(".py"):
            continue
        # avoid pathological duplicate paths
        if path not in repo_files[rn]:
            repo_files[rn][path] = row["content"]
        if verbose and (i + 1) % 100000 == 0:
            print(f"  scanned {i+1} rows, {len(repo_files)} repos, "
                  f"{time.time()-t0:.0f}s", file=sys.stderr)
    # deterministic repo order
    names = sorted(repo_files.keys(),
                   key=lambda n: hashlib.sha1(f"{seed}:{n}".encode()).hexdigest())
    for rn in names:
        files = repo_files[rn]
        if len(files) >= min_files:
            yield rn, files


def build_corpus(scan_rows: int, n_train: int, n_eval: int, tok,
                 min_files: int = 3, max_per_repo: int = 3,
                 eval_frac: float = 0.23, seed: int = 0,
                 exclude_names: set[str] | None = None,
                 max_copies: int | None = None,
                 require_import: bool = False,
                 make_controls: bool = True, verbose: bool = True):
    """Stream repos, mine links, assemble episodes, split by repo. Returns
    (train_eps, eval_eps, eval_controls, stats)."""
    train_eps: list[dict] = []
    eval_eps: list[dict] = []
    stats = {
        "repos_scanned": 0, "repos_with_links": 0, "links_found": 0,
        "episodes_built": 0, "rejected_budget": 0, "imported_episodes": 0,
        "train_buckets": defaultdict(int), "eval_buckets": defaultdict(int),
        "example_links": [],
    }
    for rn, files in _iter_repos(scan_rows, min_files, seed, exclude_names,
                                 max_copies, verbose):
        if len(train_eps) >= n_train and len(eval_eps) >= n_eval:
            break
        stats["repos_scanned"] += 1
        links = mine_cross_file_links(files, require_import=require_import)
        if not links:
            continue
        split = repo_split(rn, eval_frac, seed)
        target = eval_eps if split == "eval" else train_eps
        limit = n_eval if split == "eval" else n_train
        if len(target) >= limit:
            continue
        stats["links_found"] += len(links)
        built_this_repo = 0
        got_link = False
        for link in links:
            if built_this_repo >= max_per_repo:
                break
            ep = build_episode(rn, files, link, tok, seed=seed)
            if ep is None:
                stats["rejected_budget"] += 1
                continue
            if link.imported:
                stats["imported_episodes"] += 1
            ep["split"] = split
            target.append(ep)
            built_this_repo += 1
            got_link = True
            stats["episodes_built"] += 1
            stats[f"{split}_buckets"][ep["bucket"]] += 1
            if len(stats["example_links"]) < 12:
                stats["example_links"].append({
                    "repo": rn, "identifier": link.identifier,
                    "def": link.def_path, "use": link.use_path,
                    "n_ctx": ep["n_ctx_tokens"], "bucket": ep["bucket"],
                })
            if len(target) >= limit:
                break
        if got_link:
            stats["repos_with_links"] += 1

    eval_controls: list[dict] = []
    if make_controls:
        for ep in eval_eps:
            eval_controls.append(build_control(ep, tok, seed))

    stats["train_buckets"] = dict(stats["train_buckets"])
    stats["eval_buckets"] = dict(stats["eval_buckets"])
    stats["n_train"] = len(train_eps)
    stats["n_eval"] = len(eval_eps)
    stats["n_eval_controls"] = len(eval_controls)
    stats["imported_frac"] = (
        round(stats["imported_episodes"] / stats["episodes_built"], 3)
        if stats["episodes_built"] else 0.0)
    return train_eps, eval_eps, eval_controls, stats


def _scan_pretrain_repo_names(scan_rows: int, verbose: bool = True) -> set[str]:
    """Best-effort de-contamination: collect repo_names appearing in a bounded
    scan of the codeparrot stream (a proxy for "seen in pretrain"). Pretrain
    consumed only a fraction of codeparrot, so this reduces — not eliminates —
    overlap."""
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train",
                      streaming=True)
    names: set[str] = set()
    for i, row in enumerate(ds):
        if i >= scan_rows:
            break
        names.add(row["repo_name"])
    if verbose:
        print(f"pretrain-overlap scan: {len(names)} repo_names over "
              f"{scan_rows} rows", file=sys.stderr)
    return names


def _write_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/repo_episodes")
    ap.add_argument("--scan_rows", type=int, default=400000,
                    help="rows of codeparrot to scan/group")
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=150)
    ap.add_argument("--min_files", type=int, default=3)
    ap.add_argument("--max_per_repo", type=int, default=3)
    ap.add_argument("--eval_frac", type=float, default=0.23)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--make_controls", action="store_true",
                    help="also emit shuffled-repo controls for the eval split")
    ap.add_argument("--require_import", action="store_true",
                    help="keep only links whose use file imports the identifier "
                         "(higher precision, lower yield)")
    ap.add_argument("--exclude_pretrain_overlap", action="store_true",
                    help="best-effort: drop repos seen in a pretrain-scan")
    ap.add_argument("--pretrain_scan_rows", type=int, default=300000)
    ap.add_argument("--max_copies", type=int, default=None,
                    help="drop files whose codeparrot 'copies' > this "
                         "(widely-duplicated → more memorised); de-contam knob")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.scan_rows = min(args.scan_rows, 40000)
        args.n_train = 6
        args.n_eval = 4
        args.make_controls = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    exclude = None
    if args.exclude_pretrain_overlap:
        exclude = _scan_pretrain_repo_names(args.pretrain_scan_rows)

    t0 = time.time()
    train_eps, eval_eps, eval_controls, stats = build_corpus(
        args.scan_rows, args.n_train, args.n_eval, tok,
        min_files=args.min_files, max_per_repo=args.max_per_repo,
        eval_frac=args.eval_frac, seed=args.seed, exclude_names=exclude,
        max_copies=args.max_copies, require_import=args.require_import,
        make_controls=args.make_controls)
    dt = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train_eps)
    _write_jsonl(os.path.join(args.out_dir, "eval.jsonl"), eval_eps)
    if args.make_controls:
        _write_jsonl(os.path.join(args.out_dir, "eval_controls.jsonl"),
                     eval_controls)
    stats["elapsed_s"] = round(dt, 1)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps({k: v for k, v in stats.items()
                      if k != "example_links"}, indent=2))
    print("\nexample mined links:")
    for e in stats["example_links"]:
        print(f"  {e['identifier']:<24} {e['def']}  ->  {e['use']}  "
              f"(n_ctx={e['n_ctx']}, {e['bucket']})  [{e['repo']}]")
    print(f"\nwrote {len(train_eps)} train / {len(eval_eps)} eval / "
          f"{len(eval_controls)} controls to {args.out_dir}  ({dt:.0f}s)")


if __name__ == "__main__":
    main()
