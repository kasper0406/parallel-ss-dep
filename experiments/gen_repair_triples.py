"""Build the on-policy repair-triple corpus for repair-native training.

Produces (task, failed_attempt, error_text, verified_fix) tuples serialized
both as structured fields AND as a rendered single-`text` training document
(house style = `build_pretrain_repair_corpus.py`'s "# Original problem: /
# Attempted solution: / # Got this error: / # Fixed version:" format).

This is the corpus for the "hindsight repair-trace midtraining" idea
(`ideas_2026_07_13/08_agentic_native.md` idea 1, `04_test_time_compute.md`
idea 1): make edit->run->read-error->fix a pretrain FORMAT, with every
diagnostic produced by ACTUALLY EXECUTING the attempt through
`code_grader.grade()` (subprocess-sandboxed), never templated.

Two sources, distinguished by a mandatory `provenance` field (the ideation
report warns synthetic-bug distributions transfer worse, so training must be
able to weight them differently):

  * ``on_policy`` — mined from existing rollout dumps
    (`gen_rejection_data.py` output, e.g.
    `data/rejection_v2_step300_all.jsonl`: 10.8k rollouts / 1352
    mbpp_combined problems, 10.2k failing). The dump rows carry tier+score
    but NOT error_text, so every kept attempt is re-graded here to recover
    the real diagnostics. The fix is a PASSING SIBLING rollout for the same
    problem when one exists (238 problems have both), else the dataset gold
    solution; either way the fix is re-verified through the grader before
    the triple is emitted.

  * ``synthetic_mutant`` — gold solutions broken by AST-level semantic
    mutations (off-by-one, operator swap, wrong variable, dropped
    condition, wrong initial value). Every mutant is executed through the
    grader; only mutants that FAIL with a non-empty diagnostic are kept.
    The fix is the (verified) gold solution.

Train/heldout are DISJOINT BY PROBLEM ID (hash split on the problem key,
shared across both sources), and the syntax_error tier is capped at
``--syntax_cap`` (default 30%) per split so the corpus is not dominated by
trivially-diagnosable failures.

Regeneration (CPU-only, deterministic given --seed):

    PYTHONPATH=. .venv/bin/python experiments/gen_repair_triples.py \
        --source both --seed 0 --workers 16

Note: an earlier, unrelated scratch script of the same name produced
`data/repair_triples_v3.jsonl` (Qwen-self-debug format consumed by
`build_pretrain_repair_corpus.py`). This file supersedes it with the
structured-triple schema; the v3 artifact is left untouched.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
import pathlib
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.code_grader import Problem, grade  # noqa: E402

MUTATION_FAMILIES = (
    "off_by_one",
    "operator_swap",
    "wrong_variable",
    "dropped_condition",
    "wrong_init",
)

# Tiers that make a KEEPABLE failed attempt. "pass" is not a failure;
# "grader_error" means the dataset's own test block is broken (not the
# attempt's fault) so its diagnostic would teach the wrong lesson.
_KEEPABLE_FAIL_TIERS = frozenset(
    {"syntax_error", "exec_error", "runtime_error", "partial", "timeout"})

_DEFAULT_DUMPS = (
    "data/rejection_v2_step300_all.jsonl",
    "data/rejection_cal_t06.jsonl",
    "data/rejection_nothink_t06.jsonl",
    "data/rejection_sighint_t06.jsonl",
    "data/rejection_greedy.jsonl",
    "data/rejection_sighint_greedy.jsonl",
)


# ---------------------------------------------------------------------------
# Document rendering (house style: build_pretrain_repair_corpus.py output).
# ---------------------------------------------------------------------------

def _comment_block(text: str) -> str:
    """Prefix every line with `# ` (bare `#` for empty lines), stripping any
    existing single leading comment marker so both raw-NL (loader) and
    already-commented (dump) problem statements render identically."""
    out = []
    for line in (text or "").strip().splitlines():
        if line.startswith("# "):
            line = line[2:]
        elif line.startswith("#"):
            line = line[1:]
        out.append(f"# {line}" if line.strip() else "#")
    return "\n".join(out)


def render_document(problem: str, attempt: str, error_text: str,
                    fix: str) -> str:
    """Render one repair triple as a flat pretrain text document.

    Format matches `data/pretrain_repair_corpus.jsonl` (the repo's existing
    repair-as-pretrain-text house style); the trailing newline gives
    `data_mix.MixedSourceStream` a natural document boundary.
    """
    return (
        "# Original problem:\n"
        f"{_comment_block(problem)}\n"
        "# Attempted solution:\n"
        f"{(attempt or '').strip()}\n"
        "\n"
        "# Got this error:\n"
        f"{_comment_block(error_text or '(no error text)')}\n"
        "\n"
        "# Fixed version:\n"
        f"{(fix or '').strip()}\n"
    )


def _first_assert(tests: str) -> str | None:
    for line in (tests or "").splitlines():
        s = line.strip()
        if s.startswith("assert"):
            return s
    return None


def build_problem_text(prompt: str, tests: str) -> str:
    """NL problem statement + one example assert (the sft_comment-style
    problem framing used by the rollout dumps), un-commented."""
    desc = (prompt or "").strip()
    ex = _first_assert(tests)
    return f"{desc}\nExample: {ex}" if ex else desc


# ---------------------------------------------------------------------------
# AST semantic mutator.
# ---------------------------------------------------------------------------

_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE,
             ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
             ast.In: ast.NotIn, ast.NotIn: ast.In}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add,
             ast.Div: ast.Mult, ast.FloorDiv: ast.Div, ast.Mod: ast.FloorDiv}


def _is_int_const(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant)
            and type(node.value) is int)  # excludes bool


def _is_len_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "len" and not node.keywords)


class _SiteCollector(ast.NodeVisitor):
    """Enumerate every (family, site_index) mutation site in a tree.

    Site indices are per-family, in `ast.walk`-free deterministic visit
    order, so `(family, idx)` uniquely and reproducibly names a mutation.
    """

    def __init__(self):
        self.counts = collections.Counter()
        self.sites: list[tuple[str, int]] = []
        self._assigned: set[str] = set()
        self._loaded: list[str] = []

    def _add(self, family: str) -> int:
        idx = self.counts[family]
        self.counts[family] += 1
        self.sites.append((family, idx))
        return idx

    # -- off_by_one + wrong_init share Constant nodes; disambiguated by
    #    context (wrong_init = the value of a plain Assign).
    def visit_Constant(self, node):
        if _is_int_const(node):
            self._add("off_by_one")
        self.generic_visit(node)

    def visit_Call(self, node):
        if _is_len_call(node):
            self._add("off_by_one")   # len(x) -> len(x) - 1
        self.generic_visit(node)

    def visit_Assign(self, node):
        v = node.value
        if (_is_int_const(v)
                or (isinstance(v, ast.List) and not v.elts)
                or (isinstance(v, ast.Constant) and v.value == "")):
            self._add("wrong_init")
        self.generic_visit(node)

    def visit_Compare(self, node):
        if len(node.ops) == 1 and type(node.ops[0]) in _CMP_SWAP:
            self._add("operator_swap")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if type(node.op) in _BIN_SWAP:
            self._add("operator_swap")
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self._add("operator_swap")        # and <-> or
        if len(node.values) >= 2:
            self._add("dropped_condition")  # drop one clause
        self.generic_visit(node)

    def visit_If(self, node):
        self._add("dropped_condition")      # test -> True
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self._assigned.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self._loaded.append(node.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        self._assigned.add(node.arg)
        self.generic_visit(node)


def _collect_sites(tree: ast.AST) -> _SiteCollector:
    col = _SiteCollector()
    col.visit(tree)
    # wrong_variable sites: every Load of a locally-bound name (so the
    # replacement pool is non-trivial and the mutant stays "plausible code").
    if len(col._assigned) >= 2:
        n = sum(1 for nm in col._loaded if nm in col._assigned)
        col.counts["wrong_variable"] = n
        col.sites.extend(("wrong_variable", i) for i in range(n))
    return col


class _Mutator(ast.NodeTransformer):
    """Apply exactly ONE mutation: the `target_idx`-th site of `family`."""

    def __init__(self, family: str, target_idx: int, rng: random.Random,
                 assigned: set[str]):
        self.family = family
        self.target = target_idx
        self.rng = rng
        self.assigned = sorted(assigned)
        self._seen = 0
        self.applied = False

    def _hit(self) -> bool:
        hit = self._seen == self.target
        self._seen += 1
        if hit:
            self.applied = True
        return hit

    # ---- off_by_one ----
    def visit_Constant(self, node):
        if self.family == "off_by_one" and _is_int_const(node):
            if self._hit():
                delta = self.rng.choice((1, -1))
                return ast.copy_location(
                    ast.Constant(value=node.value + delta), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        # NOTE: the hit-check runs BEFORE generic_visit so the site index
        # ordering matches _SiteCollector's pre-order traversal exactly.
        if self.family == "off_by_one" and _is_len_call(node):
            if self._hit():
                op = self.rng.choice((ast.Sub(), ast.Add()))
                return ast.copy_location(
                    ast.BinOp(left=node, op=op, right=ast.Constant(1)), node)
        return self.generic_visit(node)

    # ---- wrong_init ----
    def visit_Assign(self, node):
        if self.family == "wrong_init":
            v = node.value
            applicable = (_is_int_const(v)
                          or (isinstance(v, ast.List) and not v.elts)
                          or (isinstance(v, ast.Constant) and v.value == ""))
            if applicable and self._hit():
                if _is_int_const(v):
                    new = ast.Constant(0 if v.value != 0 else 1)
                elif isinstance(v, ast.List):
                    new = ast.List(elts=[ast.Constant(0)], ctx=ast.Load())
                else:
                    new = ast.Constant(" ")
                node.value = ast.copy_location(new, v)
                return node
        return self.generic_visit(node)

    # ---- operator_swap ----
    def visit_Compare(self, node):
        if (self.family == "operator_swap" and len(node.ops) == 1
                and type(node.ops[0]) in _CMP_SWAP):
            if self._hit():
                node.ops = [_CMP_SWAP[type(node.ops[0])]()]
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if self.family == "operator_swap" and type(node.op) in _BIN_SWAP:
            if self._hit():
                node.op = _BIN_SWAP[type(node.op)]()
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        if self.family == "operator_swap":
            if self._hit():
                node.op = ast.Or() if isinstance(node.op, ast.And) \
                    else ast.And()
        elif self.family == "dropped_condition" and len(node.values) >= 2:
            if self._hit():
                keep = list(node.values)
                keep.pop(self.rng.randrange(len(keep)))
                if len(keep) == 1:
                    return self.generic_visit(keep[0])
                node.values = keep
        return self.generic_visit(node)

    # ---- dropped_condition ----
    def visit_If(self, node):
        if self.family == "dropped_condition":
            if self._hit():
                node.test = ast.copy_location(ast.Constant(True), node.test)
                self.generic_visit(node)   # still recurse into the body
                return node
        return self.generic_visit(node)

    # ---- wrong_variable ----
    def visit_Name(self, node):
        if (self.family == "wrong_variable" and isinstance(node.ctx, ast.Load)
                and node.id in self.assigned and len(self.assigned) >= 2):
            if self._hit():
                others = [n for n in self.assigned if n != node.id]
                node.id = self.rng.choice(others)
        return node


def enumerate_mutants(gold: str, seed_key: str,
                      max_candidates: int = 48) -> list[tuple[str, str]]:
    """Deterministically enumerate up to `max_candidates` UNIQUE, parseable
    single-site semantic mutants of `gold`.

    Returns [(mutation_family, mutant_code)]. All results are guaranteed to
    `ast.parse` (mutations are AST->unparse) and to differ textually from
    the (unparse-normalized) gold. They are NOT guaranteed to fail the
    tests — the caller must grade them and keep only real failures.
    """
    src = (gold or "").replace("\r\n", "\n")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    col = _collect_sites(tree)
    rng = random.Random(f"repair_triples:{seed_key}")
    # Round-robin across mutation families (shuffled within each family) so
    # site-rich families (wrong_variable on long solutions) don't dominate
    # the kept-mutant distribution.
    by_family: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for fam, idx in col.sites:
        by_family[fam].append((fam, idx))
    for fam in by_family:
        rng.shuffle(by_family[fam])
    fam_lists = [by_family[f] for f in MUTATION_FAMILIES if f in by_family]
    sites = []
    i = 0
    while any(i < len(lst) for lst in fam_lists):
        for lst in fam_lists:
            if i < len(lst):
                sites.append(lst[i])
        i += 1
    gold_norm = ast.unparse(tree)
    seen: set[str] = {gold_norm}
    out: list[tuple[str, str]] = []
    for family, idx in sites:
        if len(out) >= max_candidates:
            break
        # Per-site rng so a site's mutation doesn't depend on iteration order.
        site_rng = random.Random(f"repair_triples:{seed_key}:{family}:{idx}")
        mut = _Mutator(family, idx, site_rng, col._assigned)
        try:
            new_tree = mut.visit(copy.deepcopy(tree))
            ast.fix_missing_locations(new_tree)
            code = ast.unparse(new_tree)
            compile(code, "<mutant>", "exec")
        except (SyntaxError, ValueError, RecursionError):
            continue
        if not mut.applied or code in seen:
            continue
        seen.add(code)
        out.append((family, code))
    return out


# ---------------------------------------------------------------------------
# Per-problem workers (run in a ProcessPoolExecutor; problems passed as
# plain dicts so nothing heavyweight crosses the pickle boundary).
# ---------------------------------------------------------------------------

def _problem_from_dict(d: dict) -> Problem:
    return Problem(task_id=d["task_id"], prompt=d["prompt"],
                   tests=d["tests"], entry_point=d["entry_point"],
                   prompt_is_code=d.get("prompt_is_code", False),
                   gold_solution=d.get("gold_solution"))


def _verify_fix(prob: Problem, code: str, timeout_s: int) -> bool:
    if not (code or "").strip():
        return False
    return grade(prob, code, timeout_s=timeout_s).tier == "pass"


def _make_record(problem_key: str, problem_text: str, attempt: str,
                 error_text: str, fix: str, provenance: str, tier: str,
                 idx: int, mutation: str | None = None,
                 fix_source: str = "gold_solution") -> dict:
    attempt = (attempt or "").replace("\r\n", "\n").strip()
    fix = (fix or "").replace("\r\n", "\n").strip()
    error_text = (error_text or "").strip()
    return {
        "task_id": f"repair_triple/{problem_key}/{provenance}/{idx}",
        "problem_key": problem_key,
        "problem": problem_text,
        "attempt": attempt,
        "error_text": error_text,
        "fix": fix,
        "provenance": provenance,
        "tier": tier,
        "mutation": mutation,
        "fix_source": fix_source,
        "text": render_document(problem_text, attempt, error_text, fix),
    }


def mutate_problem_worker(prob_dict: dict, seed: int, max_keep: int,
                          max_candidates: int, timeout_s: int) -> list[dict]:
    """Generate + execute-grade semantic mutants of one problem's gold
    solution. Keeps only mutants that FAIL with a non-empty diagnostic.
    The fix is the gold solution, verified to pass first."""
    prob = _problem_from_dict(prob_dict)
    key = prob_dict["problem_key"]
    gold = (prob.gold_solution or "").replace("\r\n", "\n")
    if not gold.strip():
        return []
    if not _verify_fix(prob, gold, timeout_s):
        return []   # gold itself doesn't pass -> no verified fix available
    problem_text = build_problem_text(prob.prompt, prob.tests)
    records: list[dict] = []
    for family, code in enumerate_mutants(gold, f"{seed}:{key}",
                                          max_candidates=max_candidates):
        if len(records) >= max_keep:
            break
        res = grade(prob, code, timeout_s=timeout_s)
        if res.tier not in _KEEPABLE_FAIL_TIERS or not res.error_text:
            continue
        records.append(_make_record(
            key, problem_text, code, res.error_text, gold,
            provenance="synthetic_mutant", tier=res.tier,
            idx=len(records), mutation=family))
    return records


def mine_dump_problem_worker(prob_dict: dict, candidates: list[dict],
                             max_keep: int, timeout_s: int) -> list[dict]:
    """Mine one problem's rollout-dump candidates into repair triples.

    `candidates` = the dump rows for this problem (dicts with at least
    `extracted_code` and `tier`). Failing candidates are RE-GRADED to
    recover real error_text (the dumps don't store it). The fix is a
    verified passing sibling rollout when available, else the verified
    gold solution.
    """
    prob = _problem_from_dict(prob_dict)
    key = prob_dict["problem_key"]
    problem_text = build_problem_text(prob.prompt, prob.tests)

    # 1) resolve the verified fix: passing sibling first, then gold.
    fix, fix_source = None, None
    for row in candidates:
        if row.get("tier") == "pass" and (row.get("extracted_code") or "").strip():
            code = row["extracted_code"].replace("\r\n", "\n")
            if _verify_fix(prob, code, timeout_s):
                fix, fix_source = code, "sibling_rollout"
                break
    if fix is None:
        gold = (prob.gold_solution or "").replace("\r\n", "\n")
        if gold.strip() and _verify_fix(prob, gold, timeout_s):
            fix, fix_source = gold, "gold_solution"
    if fix is None:
        return []

    # 2) round-robin the failing candidates across their dump tiers so the
    #    per-problem cap doesn't collapse onto one failure mode.
    by_tier: dict[str, list[dict]] = collections.defaultdict(list)
    for row in candidates:
        code = (row.get("extracted_code") or "").strip()
        if row.get("tier") == "pass" or not code:
            continue
        if code.replace("\r\n", "\n").strip() == fix.strip():
            continue
        by_tier[row.get("tier", "?")].append(row)
    order: list[dict] = []
    tier_lists = [by_tier[t] for t in sorted(by_tier)]
    i = 0
    while any(tier_lists):
        for lst in tier_lists:
            if i < len(lst):
                order.append(lst[i])
        i += 1
        if all(i >= len(lst) for lst in tier_lists):
            break

    records: list[dict] = []
    for row in order:
        if len(records) >= max_keep:
            break
        attempt = row["extracted_code"].replace("\r\n", "\n")
        res = grade(prob, attempt, timeout_s=timeout_s)
        if res.tier not in _KEEPABLE_FAIL_TIERS or not res.error_text:
            continue   # re-grade passed / grader_error / no diagnostic
        records.append(_make_record(
            key, problem_text, attempt, res.error_text, fix,
            provenance="on_policy", tier=res.tier,
            idx=len(records), fix_source=fix_source))
    return records


# ---------------------------------------------------------------------------
# Split / tier-cap / stats.
# ---------------------------------------------------------------------------

def is_heldout_problem(problem_key: str, heldout_frac: float,
                       seed: int) -> bool:
    """Deterministic hash split, disjoint by problem id, shared across
    sources (a heldout problem is heldout for BOTH dumps and mutants)."""
    h = hashlib.md5(f"{seed}:{problem_key}".encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < heldout_frac


def split_records(records: list[dict], heldout_frac: float,
                  seed: int) -> tuple[list[dict], list[dict]]:
    train, heldout = [], []
    for r in records:
        (heldout if is_heldout_problem(r["problem_key"], heldout_frac, seed)
         else train).append(r)
    return train, heldout


def cap_tier(records: list[dict], tier: str, cap_frac: float,
             seed: int) -> list[dict]:
    """Downsample `tier` records so they are <= cap_frac of the result.
    Deterministic given seed; preserves input order of the survivors."""
    tier_recs = [r for r in records if r["tier"] == tier]
    others = [r for r in records if r["tier"] != tier]
    if not tier_recs or not others:
        return records
    # n_tier <= cap * (n_tier + n_others)  =>  n_tier <= cap/(1-cap)*n_others
    max_tier = int(cap_frac / (1.0 - cap_frac) * len(others))
    if len(tier_recs) <= max_tier:
        return records
    rng = random.Random(f"tiercap:{seed}:{tier}")
    keep_ids = {id(tier_recs[i])
                for i in rng.sample(range(len(tier_recs)), max_tier)}
    return [r for r in records
            if r["tier"] != tier or id(r) in keep_ids]


def corpus_stats(records: list[dict]) -> dict:
    tiers = collections.Counter(r["tier"] for r in records)
    prov = collections.Counter(r["provenance"] for r in records)
    mut = collections.Counter(r["mutation"] for r in records
                              if r.get("mutation"))
    fix_src = collections.Counter(r["fix_source"] for r in records)
    lens = sorted(len(r["text"]) for r in records) or [0]

    def pct(p):
        return lens[min(len(lens) - 1, int(p * len(lens)))]
    return {
        "n": len(records),
        "n_problems": len({r["problem_key"] for r in records}),
        "tiers": dict(tiers),
        "provenance": dict(prov),
        "mutations": dict(mut),
        "fix_source": dict(fix_src),
        "text_chars": {"p10": pct(.10), "p50": pct(.50), "p90": pct(.90),
                       "mean": sum(lens) // max(1, len(lens))},
    }


# ---------------------------------------------------------------------------
# Drivers.
# ---------------------------------------------------------------------------

def _load_problems(dataset: str) -> dict[str, dict]:
    from experiments.code_grader import LOADERS
    problems = LOADERS[dataset]()
    out = {}
    for p in problems:
        out[p.task_id] = {
            "problem_key": p.task_id, "task_id": p.task_id,
            "prompt": p.prompt, "tests": p.tests,
            "entry_point": p.entry_point, "prompt_is_code": p.prompt_is_code,
            "gold_solution": p.gold_solution,
        }
    return out


def load_dump_rows(paths: list[str]) -> dict[str, list[dict]]:
    """Read rollout dumps, group rows by problem key, dedupe exact
    (problem, code) duplicates across dumps."""
    by_prob: dict[str, list[dict]] = collections.defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for path in paths:
        p = pathlib.Path(path)
        if not p.exists():
            print(f"[dumps] missing (skipped): {path}")
            continue
        n = 0
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("task_id", "")
            if not tid.startswith("reject/"):
                continue
            key = tid[len("reject/"):].rsplit("/", 1)[0]
            code = (row.get("extracted_code") or "").strip()
            sig = (key, code)
            if code and sig in seen:
                continue
            seen.add(sig)
            by_prob[key].append(row)
            n += 1
        print(f"[dumps] {path}: {n} rows")
    return by_prob


def generate(args) -> tuple[list[dict], list[dict]]:
    t0 = time.time()
    problems = _load_problems(args.dataset)
    keys = sorted(problems)
    if args.limit_problems:
        keys = keys[:args.limit_problems]
        problems = {k: problems[k] for k in keys}
    print(f"[load] {len(problems)} problems from {args.dataset} "
          f"({time.time()-t0:.1f}s)")

    all_records: list[dict] = []
    counts = {}

    if args.source in ("dumps", "both"):
        by_prob = load_dump_rows(args.dumps)
        jobs = [(k, by_prob[k]) for k in sorted(by_prob) if k in problems]
        print(f"[dumps] {len(jobs)} problems with rollouts; re-grading with "
              f"{args.workers} workers ...")
        recs = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(mine_dump_problem_worker, problems[k], rows,
                              args.max_per_problem_dumps, args.timeout_s): k
                    for k, rows in jobs}
            for i, f in enumerate(as_completed(futs)):
                recs.extend(f.result())
                if (i + 1) % 200 == 0:
                    print(f"  [dumps] {i+1}/{len(jobs)} problems, "
                          f"{len(recs)} triples, {time.time()-t0:.0f}s")
        recs.sort(key=lambda r: r["task_id"])
        counts["on_policy"] = len(recs)
        all_records.extend(recs)
        print(f"[dumps] mined {len(recs)} on-policy triples")

    if args.source in ("mutants", "both"):
        jobs = sorted(k for k in problems
                      if (problems[k].get("gold_solution") or "").strip())
        print(f"[mutants] mutating {len(jobs)} gold solutions with "
              f"{args.workers} workers ...")
        recs = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(mutate_problem_worker, problems[k], args.seed,
                              args.max_mutants_per_problem,
                              args.mutation_attempts, args.timeout_s): k
                    for k in jobs}
            for i, f in enumerate(as_completed(futs)):
                recs.extend(f.result())
                if (i + 1) % 200 == 0:
                    print(f"  [mutants] {i+1}/{len(jobs)} problems, "
                          f"{len(recs)} triples, {time.time()-t0:.0f}s")
        recs.sort(key=lambda r: r["task_id"])
        counts["synthetic_mutant"] = len(recs)
        all_records.extend(recs)
        print(f"[mutants] kept {len(recs)} failing mutants")

    # Cross-source dedupe on (problem, attempt).
    seen: set[tuple[str, str]] = set()
    deduped = []
    for r in all_records:
        sig = (r["problem_key"], r["attempt"])
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(r)
    if len(deduped) != len(all_records):
        print(f"[dedupe] dropped {len(all_records)-len(deduped)} exact "
              f"duplicate attempts")
    all_records = deduped

    train, heldout = split_records(all_records, args.heldout_frac, args.seed)
    train = cap_tier(train, "syntax_error", args.syntax_cap, args.seed)
    heldout = cap_tier(heldout, "syntax_error", args.syntax_cap, args.seed)

    print(f"[counts] per-source: {counts}")
    print(f"[train]   {json.dumps(corpus_stats(train), indent=2)}")
    print(f"[heldout] {json.dumps(corpus_stats(heldout), indent=2)}")
    print(f"[done] total {time.time()-t0:.0f}s")
    return train, heldout


def _write_jsonl(path: str, records: list[dict]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {path}: {len(records)} records")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=("dumps", "mutants", "both"),
                    default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="mbpp_combined",
                    help="code_grader LOADERS key for the problem pool")
    ap.add_argument("--dumps", nargs="*", default=list(_DEFAULT_DUMPS),
                    help="rejection-dump jsonl paths (gen_rejection_data.py "
                         "output)")
    ap.add_argument("--out_train", default="data/repair_triples_train.jsonl")
    ap.add_argument("--out_heldout",
                    default="data/repair_triples_heldout.jsonl")
    ap.add_argument("--heldout_frac", type=float, default=0.10)
    ap.add_argument("--syntax_cap", type=float, default=0.30,
                    help="max fraction of syntax_error-tier records per split")
    ap.add_argument("--max_mutants_per_problem", type=int, default=24)
    ap.add_argument("--mutation_attempts", type=int, default=64,
                    help="max mutant CANDIDATES enumerated per problem "
                         "(before grading filters to failures)")
    ap.add_argument("--max_per_problem_dumps", type=int, default=12)
    ap.add_argument("--timeout_s", type=int, default=4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit_problems", type=int, default=0,
                    help="smoke mode: only the first N problems")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    train, heldout = generate(args)
    _write_jsonl(args.out_train, train)
    _write_jsonl(args.out_heldout, heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
