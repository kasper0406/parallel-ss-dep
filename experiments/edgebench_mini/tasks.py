"""EdgeBench-mini task generation.

~20 deterministic, executable, multi-step tasks. Each task is a small Python
workspace (2-5 files) generated PROGRAMMATICALLY from a seed (no hand-written
one-offs) plus 3-8 SEQUENTIAL dependent milestones. Milestones are graded by
HIDDEN tests kept in the Task object (never written into the workspace the
agent sees — the anti-gaming separation; enforcement in harness.py).

Two families, each seeded many times with a difficulty bucket:
  pipeline  — a numeric transform/accumulate/summarize chain across
              config.py / utils.py / core.py / feature.py + a visible test.
              Milestones: implement a stubbed helper -> use it to fix a bug in
              a dependent file -> make the failing visible test pass -> add
              features that reuse an earlier config decision.
  registry  — a parse/registry chain across config.py / records.py / store.py
              + a visible test. Same milestone shape on a different surface so
              the suite is not a single template a model can overfit.

Difficulty buckets control milestone count AND cross-milestone dependency
distance:
  easy  (floor)   : 3 milestones, short deps — a weak model can partially solve.
  med             : 4-5 milestones.
  hard  (stressor): 6-7 milestones, a late milestone depends on a config value
                    decided at task creation (distance 5-7) — the state/memory
                    stressor.

Determinism: `build_task(family, bucket, seed)` is a pure function of its args
(seeded `random.Random`); `to_dict`/`from_dict` round-trip losslessly.

CLI (materialize the suite to data/edgebench_mini/):
  PYTHONPATH=. .venv/bin/python -m experiments.edgebench_mini.tasks \\
      --out_dir data/edgebench_mini
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

VISIBLE_TEST = "tests/test_visible.py"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Milestone:
    index: int                        # 1-based
    name: str                         # short slug, e.g. "implement_transform"
    description: str                  # what the agent must do (NO hidden literals)
    hidden_test: str                  # python source; prints MILESTONE_OK on pass
    dependency_distance: int          # index - earliest required step (0 = config/initial)
    depends_on_files: tuple[str, ...] # earlier-milestone files whose CORRECT
                                      # content this milestone needs (for the
                                      # file-revert dependency check)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["depends_on_files"] = list(self.depends_on_files)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Milestone":
        return Milestone(
            index=d["index"], name=d["name"], description=d["description"],
            hidden_test=d["hidden_test"],
            dependency_distance=d["dependency_distance"],
            depends_on_files=tuple(d["depends_on_files"]),
        )


@dataclasses.dataclass
class Task:
    task_id: str
    family: str
    bucket: str                       # "easy" | "med" | "hard"
    seed: int
    description: str                  # agent-visible task + milestone list
    initial_workspace: dict[str, str] # relpath -> content (what the agent sees)
    reference_edits: list[dict[str, str]]  # per-milestone correct file contents
    milestones: list[Milestone]
    visible_test: str = VISIBLE_TEST

    @property
    def dep_distance_profile(self) -> dict:
        dists = [m.dependency_distance for m in self.milestones]
        return {
            "max": max(dists), "mean": round(sum(dists) / len(dists), 3),
            "per_milestone": dists,
        }

    def reference_workspace(self, up_to: int | None = None) -> dict[str, str]:
        """Workspace after applying reference edits for milestones 1..up_to
        (default: all). WRITE-replaces whole files, so later edits win."""
        up_to = len(self.milestones) if up_to is None else up_to
        ws = copy.deepcopy(self.initial_workspace)
        for i in range(min(up_to, len(self.reference_edits))):
            ws.update(self.reference_edits[i])
        return ws

    def scripted_solution_turns(self) -> list[str]:
        """Protocol turn-strings that solve the task milestone-by-milestone —
        the script for a `harness.ReplayAgent`. Edit milestones emit a WRITE;
        the integration milestone emits a RUN of the visible test; a final
        DONE closes the episode."""
        turns: list[str] = []
        for i, ms in enumerate(self.milestones):
            edit = self.reference_edits[i] if i < len(self.reference_edits) else {}
            if edit:
                for path, content in edit.items():
                    turns.append(f"WRITE {path}\n```python\n{content}```")
            else:
                turns.append(f"RUN {self.visible_test}")
        turns.append("DONE")
        return turns

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "family": self.family,
            "bucket": self.bucket, "seed": self.seed,
            "description": self.description,
            "initial_workspace": self.initial_workspace,
            "reference_edits": self.reference_edits,
            "milestones": [m.to_dict() for m in self.milestones],
            "visible_test": self.visible_test,
            "dep_distance_profile": self.dep_distance_profile,
        }

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(
            task_id=d["task_id"], family=d["family"], bucket=d["bucket"],
            seed=d["seed"], description=d["description"],
            initial_workspace=dict(d["initial_workspace"]),
            reference_edits=[dict(e) for e in d["reference_edits"]],
            milestones=[Milestone.from_dict(m) for m in d["milestones"]],
            visible_test=d.get("visible_test", VISIBLE_TEST),
        )


# --------------------------------------------------------------------------- #
# Name pools (seeded selection gives per-task variety)
# --------------------------------------------------------------------------- #

_TRANSFORM_NAMES = ["scale_value", "encode", "adjust", "warp", "remap", "temper"]
_PIPELINE_NAMES = ["accumulate", "run_series", "fold_scan", "scan_sum", "sweep"]
_SUMM_NAMES = ["summarize", "digest", "reduce_all", "collapse", "condense"]
_DESCRIBE_NAMES = ["describe", "profile", "snapshot", "characterize"]
_REPORT_NAMES = ["report", "render", "format_out", "headline"]
_COMBINE_NAMES = ["combine", "bundle", "compose_all", "assemble"]
_LABELS = ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA", "SIGMA", "KAPPA"]

_PARSE_NAMES = ["parse_record", "split_entry", "read_pair", "decode_line"]
_REG_NAMES = ["Registry", "Store", "Ledger", "Catalog", "Index"]
_TOTAL_NAMES = ["total", "grand_total", "sum_values", "aggregate"]
_COUNT_NAMES = ["count_over", "count_above", "num_exceeding"]
_EXPORT_NAMES = ["export", "dump_rows", "serialize"]
_TOPK_NAMES = ["top_k", "largest", "leaders"]
_MERGE_NAMES = ["merge", "combine_with", "union"]
_WORDS = ["apple", "brick", "cloud", "delta", "ember", "frost", "grove",
          "hazel", "ivory", "jolt", "koala", "lunar", "maple", "nova",
          "onyx", "pearl", "quartz", "raven", "sable", "topaz"]
_DELIMS = [":", "|", ";", "/", "#", "="]
_TAGS = ["REC", "ENT", "ITM", "ROW", "KV", "OBJ"]


def _milestone_count(bucket: str, rng: random.Random) -> int:
    if bucket == "easy":
        return 3
    if bucket == "med":
        return rng.choice([4, 5])
    if bucket == "hard":
        return rng.choice([6, 7])
    raise ValueError(f"unknown bucket {bucket!r}")


# --------------------------------------------------------------------------- #
# Family: pipeline
# --------------------------------------------------------------------------- #

def _pipeline_config_py(scale, offset, mod, label) -> str:
    return (f"SCALE = {scale}\n"
            f"OFFSET = {offset}\n"
            f"MODULUS = {mod}\n"
            f'LABEL = "{label}"\n')


def _pipeline_utils_py(transform, implemented: bool) -> str:
    body = ("    return (x * SCALE + OFFSET) % MODULUS\n" if implemented
            else "    raise NotImplementedError  # milestone 1: implement this\n")
    return (
        "from config import SCALE, OFFSET, MODULUS\n\n\n"
        "def clamp(x, lo, hi):\n"
        "    return max(lo, min(hi, x))\n\n\n"
        f"def {transform}(x):\n"
        '    """Return (x * SCALE + OFFSET) % MODULUS for an integer x."""\n'
        f"{body}"
    )


def _pipeline_core_py(transform, pipeline, fixed: bool) -> str:
    acc = ("        total = total + " + transform + "(x)\n" if fixed
           else "        total = " + transform
                + "(x)   # BUG: should accumulate (total = total + ...)\n")
    return (
        f"from utils import {transform}, clamp\n\n\n"
        f"def {pipeline}(data):\n"
        f'    """Return the running (cumulative) sum of {transform} applied to\n'
        f'    each item of `data`."""\n'
        "    total = 0\n"
        "    out = []\n"
        "    for x in data:\n"
        f"{acc}"
        "        out.append(total)\n"
        "    return out\n"
    )


def _pipeline_feature_py(transform, summ, describe, report, combine,
                         included: list[str]) -> str:
    """Build feature.py with the extra functions in `included` (subset, in
    fixed order)."""
    parts = [
        "from config import SCALE, OFFSET, MODULUS, LABEL\n"
        f"from utils import {transform}\n"
    ]
    if not included:
        parts.append("\n# Add feature functions here (later milestones).\n")
    if "summ" in included:
        parts.append(
            f"\n\ndef {summ}(data):\n"
            f'    """Sum of {transform}(x) over data, taken modulo MODULUS."""\n'
            f"    return sum({transform}(x) for x in data) % MODULUS\n")
    if "describe" in included:
        parts.append(
            f"\n\ndef {describe}(data):\n"
            "    \"\"\"Return {'scale': SCALE, 'offset': OFFSET,"
            f" 'summary': {summ}(data)}}.\"\"\"\n"
            "    return {\"scale\": SCALE, \"offset\": OFFSET,"
            f" \"summary\": {summ}(data)}}\n")
    if "report" in included:
        parts.append(
            f"\n\ndef {report}(data):\n"
            f'    """Return "<LABEL>:<{summ}(data)>"."""\n'
            f'    return "%s:%d" % (LABEL, {summ}(data))\n')
    if "combine" in included:
        parts.append(
            f"\n\ndef {combine}(data):\n"
            f'    """Return ({describe}(data), {report}(data))."""\n'
            f"    return ({describe}(data), {report}(data))\n")
    return "".join(parts)


def _pipeline_visible_test(pipeline, data, expected) -> str:
    return (
        f"from core import {pipeline}\n\n\n"
        "def test_running_sum():\n"
        f"    assert {pipeline}({data!r}) == {expected!r}\n\n\n"
        'if __name__ == "__main__":\n'
        "    test_running_sum()\n"
        '    print("VISIBLE_OK")\n'
    )


def build_pipeline_task(bucket: str, seed: int) -> Task:
    rng = random.Random(seed)
    n_ms = _milestone_count(bucket, rng)

    transform = rng.choice(_TRANSFORM_NAMES)
    pipeline = rng.choice(_PIPELINE_NAMES)
    summ = rng.choice(_SUMM_NAMES)
    describe = rng.choice(_DESCRIBE_NAMES)
    report = rng.choice(_REPORT_NAMES)
    combine = rng.choice(_COMBINE_NAMES)
    label = rng.choice(_LABELS)
    S = rng.randint(2, 9)
    O = rng.randint(1, 20)
    M = rng.randint(50, 200)

    def tf(x):
        return (x * S + O) % M

    def run_sum(data):
        tot, out = 0, []
        for x in data:
            tot += tf(x)
            out.append(tot)
        return out

    def summarize(data):
        return sum(tf(x) for x in data) % M

    def describe_val(data):
        return {"scale": S, "offset": O, "summary": summarize(data)}

    def report_val(data):
        return "%s:%d" % (label, summarize(data))

    # Distinct data for each milestone (so hidden literals != docstrings).
    vis_data = [rng.randint(0, 30) for _ in range(4)]
    d1 = [rng.randint(1, 40), rng.randint(1, 40), 0]
    d2 = [rng.randint(0, 30) for _ in range(5)]
    d4 = [rng.randint(0, 30) for _ in range(4)]
    d5 = [rng.randint(0, 30) for _ in range(3)]
    d6 = [rng.randint(0, 30) for _ in range(4)]
    d7 = [rng.randint(0, 30) for _ in range(3)]

    initial = {
        "config.py": _pipeline_config_py(S, O, M, label),
        "utils.py": _pipeline_utils_py(transform, implemented=False),
        "core.py": _pipeline_core_py(transform, pipeline, fixed=False),
        "feature.py": _pipeline_feature_py(transform, summ, describe, report,
                                           combine, included=[]),
        VISIBLE_TEST: _pipeline_visible_test(pipeline, vis_data,
                                             run_sum(vis_data)),
    }

    all_ms = [
        Milestone(
            1, "implement_transform",
            f"Implement `{transform}(x)` in utils.py per its docstring "
            f"((x * SCALE + OFFSET) % MODULUS). Use the constants from config.py.",
            f"from utils import {transform}\n"
            f"assert {transform}({d1[0]}) == {tf(d1[0])}\n"
            f"assert {transform}({d1[1]}) == {tf(d1[1])}\n"
            f"assert {transform}(0) == {tf(0)}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=1, depends_on_files=()),
        Milestone(
            2, "fix_pipeline",
            f"Fix `{pipeline}(data)` in core.py so it returns the RUNNING "
            f"cumulative sum of `{transform}` over the items (it currently "
            "does not accumulate). It depends on your milestone-1 work.",
            f"from core import {pipeline}\n"
            f"assert {pipeline}({d2!r}) == {run_sum(d2)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=1, depends_on_files=("utils.py",)),
        Milestone(
            3, "pass_visible_test",
            f"Make the failing test in {VISIBLE_TEST} pass "
            f"(RUN it to see the current failure). Requires milestones 1 and 2.",
            f"from core import {pipeline}\n"
            f"assert {pipeline}({vis_data!r}) == {run_sum(vis_data)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=2, depends_on_files=("utils.py", "core.py")),
        Milestone(
            4, "add_summarize",
            f"Add `{summ}(data)` to feature.py: the sum of `{transform}(x)` "
            "over data, taken modulo MODULUS. Reuse config + your transform.",
            f"from feature import {summ}\n"
            f"assert {summ}({d4!r}) == {summarize(d4)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=4, depends_on_files=("utils.py",)),
        Milestone(
            5, "add_describe",
            f"Add `{describe}(data)` to feature.py returning "
            "{'scale': SCALE, 'offset': OFFSET, 'summary': "
            f"{summ}(data)}}. It reuses the config constants chosen at the "
            f"start and your `{summ}`.",
            f"from feature import {describe}\n"
            f"assert {describe}({d5!r}) == {describe_val(d5)!r}\n"
            "print('MILESTONE_OK')\n",
            # depends_on_files names the GENUINE earlier-milestone artifact
            # (milestone 1's transform in utils.py, distance 4-6) — NOT
            # feature.py, which holds this milestone's own function and would
            # make verify_task's revert-check tautological (review 2026-07-13).
            dependency_distance=5, depends_on_files=("utils.py",)),
        Milestone(
            6, "add_report",
            f"Add `{report}(data)` to feature.py returning the string "
            f'"<LABEL>:<{summ}(data)>" using the LABEL from config.py '
            "(decided at task start).",
            f"from feature import {report}\n"
            f"assert {report}({d6!r}) == {report_val(d6)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=6, depends_on_files=("utils.py",)),
        Milestone(
            7, "add_combine",
            f"Add `{combine}(data)` to feature.py returning the tuple "
            f"({describe}(data), {report}(data)) — it composes milestones 5 "
            "and 6.",
            f"from feature import {combine}\n"
            f"assert {combine}({d7!r}) == {(describe_val(d7), report_val(d7))!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=7, depends_on_files=("utils.py",)),
    ]
    milestones = all_ms[:n_ms]

    # Reference edits per milestone (full correct file content; WRITE replaces).
    feat_after = {  # which extra fns exist in feature.py after milestone k
        4: ["summ"], 5: ["summ", "describe"],
        6: ["summ", "describe", "report"],
        7: ["summ", "describe", "report", "combine"],
    }
    ref_edits: list[dict[str, str]] = []
    for k in range(1, n_ms + 1):
        if k == 1:
            ref_edits.append({"utils.py": _pipeline_utils_py(transform, True)})
        elif k == 2:
            ref_edits.append({"core.py": _pipeline_core_py(transform, pipeline,
                                                           fixed=True)})
        elif k == 3:
            ref_edits.append({})  # integration — no new file
        else:
            ref_edits.append({"feature.py": _pipeline_feature_py(
                transform, summ, describe, report, combine, feat_after[k])})

    return Task(
        task_id=f"pipeline/{bucket}/{seed}", family="pipeline", bucket=bucket,
        seed=seed, description=_render_description("pipeline", bucket, milestones),
        initial_workspace=initial, reference_edits=ref_edits,
        milestones=milestones)


# --------------------------------------------------------------------------- #
# Family: registry
# --------------------------------------------------------------------------- #

def _registry_config_py(delim, tag) -> str:
    return f'DELIM = "{delim}"\nTAG = "{tag}"\n'


def _registry_records_py(parse, delim, implemented: bool) -> str:
    body = ("    parts = line.split(DELIM)\n"
            "    return parts[0], int(parts[1])\n" if implemented
            else "    raise NotImplementedError  # milestone 1: implement this\n")
    return (
        "from config import DELIM\n\n\n"
        f"def {parse}(line):\n"
        f'    """Split `line` on DELIM into (name, int(value)), e.g. '
        f'"foo{delim}3" -> ("foo", 3)."""\n'
        f"{body}"
    )


def _registry_store_py(reg, parse, total, count, export, topk, merge,
                       total_fixed: bool, included: list[str]) -> str:
    total_body = ("        return sum(self.items.values())\n" if total_fixed
                  else "        return 0  # BUG: should sum stored values\n")
    s = (
        "from config import DELIM, TAG\n"
        f"from records import {parse}\n\n\n"
        f"class {reg}:\n"
        "    def __init__(self):\n"
        "        self.items = {}\n\n"
        "    def add(self, line):\n"
        f"        name, value = {parse}(line)\n"
        "        self.items[name] = value\n\n"
        f"    def {total}(self):\n"
        f"{total_body}"
    )
    if "count" in included:
        s += (f"\n    def {count}(self, threshold):\n"
              "        return sum(1 for v in self.items.values() if v > threshold)\n")
    if "export" in included:
        s += (f"\n    def {export}(self):\n"
              "        return [TAG + DELIM + n + DELIM + str(v)\n"
              "                for n, v in self.items.items()]\n")
    if "topk" in included:
        s += (f"\n    def {topk}(self, k):\n"
              "        ordered = sorted(self.items.items(),"
              " key=lambda kv: (-kv[1], kv[0]))\n"
              "        return [n for n, _ in ordered[:k]]\n")
    if "merge" in included:
        s += (f"\n    def {merge}(self, other):\n"
              f"        out = {reg}()\n"
              "        out.items = dict(self.items)\n"
              "        out.items.update(other.items)\n"
              "        return out\n")
    return s


def _registry_visible_test(reg, total, lines, expected) -> str:
    return (
        f"from store import {reg}\n\n\n"
        "def test_total():\n"
        f"    r = {reg}()\n"
        f"    for line in {lines!r}:\n"
        "        r.add(line)\n"
        f"    assert r.{total}() == {expected!r}\n\n\n"
        'if __name__ == "__main__":\n'
        "    test_total()\n"
        '    print("VISIBLE_OK")\n'
    )


def _pairs(rng: random.Random, n: int, used: set[str]) -> list[tuple[str, int]]:
    out = []
    for _ in range(n):
        w = rng.choice([x for x in _WORDS if x not in used] or _WORDS)
        used.add(w)
        out.append((w, rng.randint(1, 99)))
    return out


def build_registry_task(bucket: str, seed: int) -> Task:
    rng = random.Random(seed)
    n_ms = _milestone_count(bucket, rng)

    parse = rng.choice(_PARSE_NAMES)
    reg = rng.choice(_REG_NAMES)
    total = rng.choice(_TOTAL_NAMES)
    count = rng.choice(_COUNT_NAMES)
    export = rng.choice(_EXPORT_NAMES)
    topk = rng.choice(_TOPK_NAMES)
    merge = rng.choice(_MERGE_NAMES)
    delim = rng.choice(_DELIMS)
    tag = rng.choice(_TAGS)

    used: set[str] = set()
    P2 = _pairs(rng, 4, used)
    Pvis = _pairs(rng, 3, used)
    P4 = _pairs(rng, 5, used)
    P5 = _pairs(rng, 3, used)
    P6 = _pairs(rng, 5, used)
    P7a = _pairs(rng, 2, used)
    P7b = _pairs(rng, 2, used)
    thr = rng.randint(20, 60)
    k_top = rng.randint(2, 3)

    def lines_of(pairs):
        return [f"{n}{delim}{v}" for n, v in pairs]

    def total_of(pairs):
        return sum(v for _, v in dict(pairs).items())

    def count_of(pairs, t):
        return sum(1 for v in dict(pairs).values() if v > t)

    def export_of(pairs):
        return [tag + delim + n + delim + str(v) for n, v in dict(pairs).items()]

    def topk_of(pairs, k):
        ordered = sorted(dict(pairs).items(), key=lambda kv: (-kv[1], kv[0]))
        return [n for n, _ in ordered[:k]]

    def merged_items(pa, pb):
        d = dict(pa)
        d.update(dict(pb))
        return d

    initial = {
        "config.py": _registry_config_py(delim, tag),
        "records.py": _registry_records_py(parse, delim, implemented=False),
        "store.py": _registry_store_py(reg, parse, total, count, export, topk,
                                       merge, total_fixed=False, included=[]),
        VISIBLE_TEST: _registry_visible_test(reg, total, lines_of(Pvis),
                                             total_of(Pvis)),
    }

    all_ms = [
        Milestone(
            1, "implement_parse",
            f'Implement `{parse}(line)` in records.py: split `line` on the '
            f'DELIM from config.py into (name, int(value)), e.g. '
            f'"foo{delim}3" -> ("foo", 3).',
            f"from records import {parse}\n"
            f"assert {parse}('x{delim}5') == ('x', 5)\n"
            f"assert {parse}('name{delim}42') == ('name', 42)\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=1, depends_on_files=()),
        Milestone(
            2, "fix_total",
            f"Fix `{reg}.{total}()` in store.py so it returns the sum of all "
            "stored values (it currently returns 0). `add` relies on your "
            "milestone-1 parser.",
            f"from store import {reg}\n"
            f"r = {reg}()\n"
            f"for line in {lines_of(P2)!r}:\n"
            "    r.add(line)\n"
            f"assert r.{total}() == {total_of(P2)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=1, depends_on_files=("records.py",)),
        Milestone(
            3, "pass_visible_test",
            f"Make {VISIBLE_TEST} pass (RUN it first). Requires milestones 1-2.",
            f"from store import {reg}\n"
            f"r = {reg}()\n"
            f"for line in {lines_of(Pvis)!r}:\n"
            "    r.add(line)\n"
            f"assert r.{total}() == {total_of(Pvis)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=2, depends_on_files=("records.py", "store.py")),
        Milestone(
            4, "add_count_over",
            f"Add a method `{reg}.{count}(self, threshold)` to store.py "
            "returning the number of stored values strictly greater than "
            "`threshold`.",
            f"from store import {reg}\n"
            f"r = {reg}()\n"
            f"for line in {lines_of(P4)!r}:\n"
            "    r.add(line)\n"
            f"assert r.{count}({thr}) == {count_of(P4, thr)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=3, depends_on_files=("records.py",)),
        Milestone(
            5, "add_export",
            f"Add a method `{reg}.{export}(self)` returning a list of "
            f'"TAG+DELIM+name+DELIM+value" strings, using the TAG and DELIM '
            "from config.py (chosen at task start).",
            f"from store import {reg}\n"
            f"r = {reg}()\n"
            f"for line in {lines_of(P5)!r}:\n"
            "    r.add(line)\n"
            f"assert r.{export}() == {export_of(P5)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=5, depends_on_files=("records.py",)),
        Milestone(
            6, "add_topk",
            f"Add a method `{reg}.{topk}(self, k)` returning the names of the "
            "top-k items by value (ties broken by name ascending).",
            f"from store import {reg}\n"
            f"r = {reg}()\n"
            f"for line in {lines_of(P6)!r}:\n"
            "    r.add(line)\n"
            f"assert r.{topk}({k_top}) == {topk_of(P6, k_top)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=5, depends_on_files=("records.py",)),
        Milestone(
            7, "add_merge",
            f"Add a method `{reg}.{merge}(self, other)` returning a NEW {reg} "
            "whose items are self's updated with other's.",
            f"from store import {reg}\n"
            f"a = {reg}()\n"
            f"for line in {lines_of(P7a)!r}:\n"
            "    a.add(line)\n"
            f"b = {reg}()\n"
            f"for line in {lines_of(P7b)!r}:\n"
            "    b.add(line)\n"
            f"m = a.{merge}(b)\n"
            f"assert isinstance(m, {reg})\n"
            f"assert dict(m.items) == {merged_items(P7a, P7b)!r}\n"
            "print('MILESTONE_OK')\n",
            dependency_distance=6, depends_on_files=("records.py",)),
    ]
    milestones = all_ms[:n_ms]

    store_after = {  # which methods store.py has after milestone k
        4: ["count"], 5: ["count", "export"],
        6: ["count", "export", "topk"],
        7: ["count", "export", "topk", "merge"],
    }
    ref_edits: list[dict[str, str]] = []
    for k in range(1, n_ms + 1):
        if k == 1:
            ref_edits.append({"records.py": _registry_records_py(parse, delim,
                                                                 True)})
        elif k == 2:
            ref_edits.append({"store.py": _registry_store_py(
                reg, parse, total, count, export, topk, merge,
                total_fixed=True, included=[])})
        elif k == 3:
            ref_edits.append({})
        else:
            ref_edits.append({"store.py": _registry_store_py(
                reg, parse, total, count, export, topk, merge,
                total_fixed=True, included=store_after[k])})

    return Task(
        task_id=f"registry/{bucket}/{seed}", family="registry", bucket=bucket,
        seed=seed, description=_render_description("registry", bucket, milestones),
        initial_workspace=initial, reference_edits=ref_edits,
        milestones=milestones)


# --------------------------------------------------------------------------- #
# Description rendering (agent-visible)
# --------------------------------------------------------------------------- #

_FAMILY_BLURB = {
    "pipeline": ("A small numeric-processing library. `config.py` holds the "
                 "constants; `utils.py`/`core.py`/`feature.py` build on each "
                 "other. Work through the milestones IN ORDER — each depends "
                 "on earlier ones."),
    "registry": ("A small record-registry library. `config.py` holds the "
                 "format constants; `records.py` parses lines and `store.py` "
                 "aggregates them. Work through the milestones IN ORDER — each "
                 "depends on earlier ones."),
}


def _render_description(family: str, bucket: str,
                        milestones: list[Milestone]) -> str:
    lines = [_FAMILY_BLURB[family], "", "Milestones:"]
    for m in milestones:
        lines.append(f"  {m.index}. {m.description}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #

_BUILDERS = {"pipeline": build_pipeline_task, "registry": build_registry_task}


def build_task(family: str, bucket: str, seed: int) -> Task:
    return _BUILDERS[family](bucket, seed)


# 20 tasks: 7 easy (floor), 5 med, 8 hard (long-dependency stressor); each
# difficulty split across both families. Fixed spec => reproducible suite.
SUITE_SPEC: list[tuple[str, str, int]] = [
    # easy floor (7)
    ("pipeline", "easy", 101), ("pipeline", "easy", 102), ("pipeline", "easy", 103),
    ("registry", "easy", 111), ("registry", "easy", 112), ("registry", "easy", 113),
    ("pipeline", "easy", 104),
    # med (5)
    ("pipeline", "med", 201), ("pipeline", "med", 202),
    ("registry", "med", 211), ("registry", "med", 212), ("registry", "med", 213),
    # hard stressor (8)
    ("pipeline", "hard", 301), ("pipeline", "hard", 302), ("pipeline", "hard", 303),
    ("pipeline", "hard", 304),
    ("registry", "hard", 311), ("registry", "hard", 312), ("registry", "hard", 313),
    ("registry", "hard", 314),
]


def build_suite() -> list[Task]:
    return [build_task(fam, bucket, seed) for fam, bucket, seed in SUITE_SPEC]


# --------------------------------------------------------------------------- #
# Self-check / gold verification (also a build-time correctness gate)
# --------------------------------------------------------------------------- #

def verify_task(task: Task, timeout: float = 5.0) -> dict:
    """Sanity gate for a generated task:
      - the FULL reference solution passes every milestone;
      - the INITIAL workspace passes none;
      - each milestone with `depends_on_files` FAILS when those files are
        reverted to their initial (stub/buggy) content within the otherwise
        complete reference workspace (proves the sequential dependency).
    Returns a dict of the outcomes."""
    from experiments.edgebench_mini.harness import grade_milestones_on_files
    ref = task.reference_workspace()
    ref_scores = grade_milestones_on_files(task, ref, timeout)
    init_scores = grade_milestones_on_files(task, task.initial_workspace, timeout)

    dep_checks = []
    for i, ms in enumerate(task.milestones):
        if not ms.depends_on_files:
            continue
        broken = copy.deepcopy(ref)
        for f in ms.depends_on_files:
            if f in task.initial_workspace:
                broken[f] = task.initial_workspace[f]
        broke_scores = grade_milestones_on_files(task, broken, timeout)
        dep_checks.append({
            "milestone": ms.index, "name": ms.name,
            "reverted": list(ms.depends_on_files),
            "passes_with_deps": bool(ref_scores[i]),
            "fails_without_deps": bool(not broke_scores[i]),
        })
    return {
        "task_id": task.task_id,
        "reference_all_pass": all(ref_scores),
        "initial_none_pass": not any(init_scores),
        "reference_scores": ref_scores,
        "initial_scores": init_scores,
        "dependency_checks": dep_checks,
        "dependency_ok": all(d["passes_with_deps"] and d["fails_without_deps"]
                             for d in dep_checks),
    }


# --------------------------------------------------------------------------- #
# CLI — materialize the suite to data/edgebench_mini/
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/edgebench_mini")
    ap.add_argument("--verify", action="store_true",
                    help="run the gold self-check on every task (subprocess "
                         "grading — a few seconds).")
    args = ap.parse_args()

    tasks = build_suite()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = pathlib.Path(__file__).resolve().parents[2] / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_path = out_dir / "suite.jsonl"
    with open(suite_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")

    manifest = {
        "n_tasks": len(tasks),
        "families": sorted({t.family for t in tasks}),
        "buckets": {b: sum(1 for t in tasks if t.bucket == b)
                    for b in ("easy", "med", "hard")},
        "tasks": [
            {"task_id": t.task_id, "family": t.family, "bucket": t.bucket,
             "n_milestones": len(t.milestones),
             "dep_distance_profile": t.dep_distance_profile,
             "files": sorted(t.initial_workspace)}
            for t in tasks
        ],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"wrote {len(tasks)} tasks -> {suite_path}")
    print(f"buckets: {manifest['buckets']}")

    if args.verify:
        print("\nverifying (gold self-check)...")
        n_ok = 0
        for t in tasks:
            v = verify_task(t)
            ok = (v["reference_all_pass"] and v["initial_none_pass"]
                  and v["dependency_ok"])
            n_ok += int(ok)
            flag = "ok " if ok else "FAIL"
            print(f"  [{flag}] {t.task_id:<24} ref_all={v['reference_all_pass']} "
                  f"init_none={v['initial_none_pass']} "
                  f"dep_ok={v['dependency_ok']} "
                  f"maxdist={t.dep_distance_profile['max']}")
        print(f"\n{n_ok}/{len(tasks)} tasks verified")
        return 0 if n_ok == len(tasks) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
