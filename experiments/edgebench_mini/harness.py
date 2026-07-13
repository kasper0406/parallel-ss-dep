"""EdgeBench-mini agent loop + subprocess sandbox + milestone grading.

Model-agnostic by construction: `run_episode` takes any `generate(prompt) -> str`
callable (or one returning a `GenOutput` for exact generated-token accounting)
and drives it through a small text action protocol against a sandboxed copy of
the task workspace, grading the task's HIDDEN milestone tests after every turn.

Design invariants (the anti-gaming separation the benchmark rests on):
  - Hidden tests live in the in-memory Task object and are executed via
    `python -c <src>` with cwd set to the live workspace. Their source is
    NEVER written into the workspace the agent reads/lists, and their output
    is NEVER fed back to the agent. The agent's only feedback comes from its
    own READ / RUN actions on the *visible* files.
  - Path actions are confined to the workspace root: absolute paths, `..`
    components, and post-realpath escapes are rejected (`is_safe_path` /
    `_resolve_in_root`).

Security posture: the same as `experiments/code_grader.py` — untrusted code
(both the task workspace and the model's edits) runs in a subprocess with a
hard wall-clock timeout. This is NOT a full sandbox (no network / fs isolation
beyond cwd); it is intended for trusted synthetic tasks + a small model's
outputs, offline.

This module imports NO torch at module load (so tasks/tests import it without
CUDA). `CkptAgent` lazy-imports the decode stack in its constructor.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable, Union

# Truncation cap for observations fed back to the agent — long enough to
# diagnose (mirrors code_grader._ERROR_TEXT_CAP), short enough not to bloat the
# next prompt.
_OBS_CAP = 1200
_MILESTONE_OK = "MILESTONE_OK"


# --------------------------------------------------------------------------- #
# Generated-token accounting
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class GenOutput:
    """A `generate` return value carrying an EXACT generated-token count.

    A generate callable may return a plain `str` (the harness then estimates
    tokens via `estimate_tokens`) or a `GenOutput` with the true count — the
    latter is what `CkptAgent` and `ReplayAgent` provide so the cost axis is
    exact, not a heuristic."""
    text: str
    n_gen_tokens: int


def estimate_tokens(text: str) -> int:
    """Heuristic generated-token count for a plain-string generate callable
    (~4 chars/token, the standard rough BPE rate). Only used when a callable
    does not report its own count."""
    return max(1, (len(text) + 3) // 4)


GenerateFn = Callable[[str], Union[str, GenOutput]]


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #

def is_safe_path(rel: str) -> bool:
    """Pure (no-fs) check that `rel` is a workspace-relative path that cannot
    escape the root: non-empty, not absolute, no `~`, and no `..` path
    component. `_resolve_in_root` does the second, fs-level realpath check."""
    if not isinstance(rel, str):
        return False
    rel = rel.strip()
    if not rel or rel.startswith("~"):
        return False
    p = pathlib.PurePosixPath(rel)
    if p.is_absolute():
        return False
    # Windows-drive / backslash absolute forms also rejected.
    if re.match(r"^[A-Za-z]:", rel) or rel.startswith("\\"):
        return False
    for part in p.parts:
        if part == ".." or part == "":
            return False
    return True


def _resolve_in_root(root: pathlib.Path, rel: str) -> pathlib.Path | None:
    """Resolve `rel` under `root`, returning the absolute path only if it is
    genuinely contained (realpath-based, so a symlink escape is caught too).
    Returns None on any escape / unsafe path."""
    if not is_safe_path(rel):
        return None
    root_r = pathlib.Path(os.path.realpath(root))
    cand = pathlib.Path(os.path.realpath(root / rel))
    try:
        cand.relative_to(root_r)
    except ValueError:
        return None
    return cand


# --------------------------------------------------------------------------- #
# Subprocess sandbox
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class RunResult:
    ok: bool                 # returncode == 0 and not timed out
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_s: float


def _run(argv: list[str], cwd: pathlib.Path, timeout: float) -> RunResult:
    env = dict(os.environ)
    # Make workspace modules importable; keep the child from re-importing the
    # heavy repo (PYTHONPATH is the workspace only).
    env["PYTHONPATH"] = str(cwd)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        return RunResult(
            ok=(proc.returncode == 0), returncode=proc.returncode,
            stdout=proc.stdout, stderr=proc.stderr, timed_out=False,
            elapsed_s=time.perf_counter() - t0,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            ok=False, returncode=None,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode("utf-8", "replace"),
            stderr="TIMEOUT: execution exceeded the time limit "
                   "(possible infinite loop)",
            timed_out=True, elapsed_s=time.perf_counter() - t0,
        )


def run_script_src(workspace_dir: pathlib.Path, script_src: str,
                   timeout: float = 5.0) -> RunResult:
    """Run `script_src` via `python -c` with cwd=workspace_dir. Used for HIDDEN
    milestone tests — the source is passed on argv and never written into the
    workspace, preserving the visible/hidden separation."""
    return _run([sys.executable, "-c", script_src], workspace_dir, timeout)


def run_workspace_file(workspace_dir: pathlib.Path, rel_path: str,
                       timeout: float = 5.0) -> RunResult:
    """Run a VISIBLE workspace .py file (a RUN action)."""
    resolved = _resolve_in_root(workspace_dir, rel_path)
    if resolved is None:
        return RunResult(False, None, "", f"unsafe or out-of-workspace path: "
                         f"{rel_path!r}", False, 0.0)
    if not resolved.exists():
        return RunResult(False, None, "", f"no such file: {rel_path}", False, 0.0)
    return _run([sys.executable, str(resolved.name)]
                if resolved.parent == workspace_dir
                else [sys.executable, os.path.relpath(resolved, workspace_dir)],
                workspace_dir, timeout)


# --------------------------------------------------------------------------- #
# Workspace materialization
# --------------------------------------------------------------------------- #

def materialize_workspace(files: dict[str, str], dir: pathlib.Path) -> None:
    """Write a {relpath: content} dict into `dir` (creating parents)."""
    for rel, content in files.items():
        if not is_safe_path(rel):
            raise ValueError(f"refusing to materialize unsafe path {rel!r}")
        dest = dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)


def workspace_hash(dir: pathlib.Path) -> str:
    """Content hash of all *.py files under `dir` (used to skip re-grading an
    unchanged workspace)."""
    h = hashlib.sha1()
    for p in sorted(dir.rglob("*.py")):
        h.update(str(p.relative_to(dir)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Milestone grading (duck-typed on the Task object from tasks.py)
# --------------------------------------------------------------------------- #

def grade_milestones_on_dir(task, workspace_dir: pathlib.Path,
                            timeout: float = 5.0) -> list[bool]:
    """Run every milestone's HIDDEN test against the current workspace state.
    Returns a per-milestone pass/fail list. Test source is executed via
    `python -c` (never written to disk in the workspace)."""
    out: list[bool] = []
    for ms in task.milestones:
        res = run_script_src(workspace_dir, ms.hidden_test, timeout)
        out.append(bool(res.ok and _MILESTONE_OK in res.stdout))
    return out


def grade_milestones_on_files(task, files: dict[str, str],
                              timeout: float = 5.0) -> list[bool]:
    """Grade a workspace given as a {relpath: content} dict (materialize into a
    throwaway tempdir first). Used by the task self-check and the dependency
    verification test."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="edgebench_grade_"))
    try:
        materialize_workspace(files, tmp)
        return grade_milestones_on_dir(task, tmp, timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Action protocol parsing
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Action:
    kind: str                 # "read" | "run" | "write" | "done" | "error"
    path: str | None = None
    content: str | None = None    # for write
    reason: str | None = None     # for error / malformed


_FENCE_RE = re.compile(r"^\s*```")


def parse_actions(text: str) -> list[Action]:
    """Parse the agent's turn text into a list of Actions. Tolerant of the
    freeform output a small model produces (leading prose, extra blank lines,
    ```python or plain ``` fences).

    Grammar (case-insensitive keyword at line start; everything else is
    ignored as prose):
        READ <path>
        RUN <path>
        DONE
        WRITE <path>
        ```[python]
        <file content...>
        ```
    A WRITE whose fenced block never opens/closes yields an `error` Action
    (surfaced to the agent as an observation, never a crash). A WRITE/READ/RUN
    with an unsafe path is still parsed here (kind set normally); the harness
    rejects it at apply time so `is_safe_path` has a single enforcement point.
    """
    lines = text.splitlines()
    actions: list[Action] = []
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()
        low = stripped.lower()
        m = re.match(r"^(read|run|write|done)\b(.*)$", low)
        if not m:
            i += 1
            continue
        kw = m.group(1)
        # Recover the original-case argument (path) from the raw line.
        arg = raw.strip()[len(kw):].strip()
        if kw == "done":
            # Only a standalone DONE ends the episode — a prose line like
            # "Done, that works." (trailing words) is NOT a terminator, so the
            # agent can't accidentally quit by narrating.
            if arg.strip(".!:,; ") == "":
                actions.append(Action(kind="done"))
            i += 1
            continue
        if kw in ("read", "run"):
            if not arg:
                actions.append(Action(kind="error", reason=f"{kw.upper()} "
                                      "requires a path"))
            else:
                actions.append(Action(kind=kw, path=arg.split()[0]))
            i += 1
            continue
        # kw == "write": path on this line, then a fenced block.
        path = arg.split()[0] if arg else None
        if not path:
            actions.append(Action(kind="error",
                                   reason="WRITE requires a path"))
            i += 1
            continue
        # Find the opening fence (allow up to a couple of prose/blank lines).
        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        if j >= n or not _FENCE_RE.match(lines[j]):
            actions.append(Action(kind="error", path=path,
                                   reason=f"WRITE {path}: expected an opening "
                                   "``` fence with the file content"))
            i = j
            continue
        # Collect until the closing fence.
        body: list[str] = []
        k = j + 1
        closed = False
        while k < n:
            if _FENCE_RE.match(lines[k]):
                closed = True
                break
            body.append(lines[k])
            k += 1
        if not closed:
            actions.append(Action(kind="error", path=path,
                                   reason=f"WRITE {path}: unterminated code "
                                   "fence (no closing ```)"))
            i = k
            continue
        actions.append(Action(kind="write", path=path,
                              content="\n".join(body) + "\n"))
        i = k + 1
    return actions


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

_PROTOCOL = """\
You are a coding agent working inside a small Python workspace. Respond ONLY
with actions in this protocol (one or more per turn):

  READ <path>          show the current contents of a file
  RUN <path>           execute a python file, see its stdout/stderr
  WRITE <path>         replace/create a file; follow it immediately with the
  ```python            FULL new file content inside a fenced block
  ...file content...
  ```
  DONE                 you believe every milestone is complete; ends the task

Rules: paths are workspace-relative (no absolute paths, no '..'). WRITE
replaces the whole file, so include everything. Work through the milestones in
order — later milestones depend on earlier ones."""


def _file_tree(files: dict[str, str]) -> str:
    return "\n".join(f"  {p}" for p in sorted(files))


def build_initial_prompt(task, files: dict[str, str]) -> str:
    return (
        f"{_PROTOCOL}\n\n"
        f"=== TASK: {task.task_id} ({task.bucket}) ===\n"
        f"{task.description}\n\n"
        f"=== WORKSPACE FILES ===\n{_file_tree(files)}\n\n"
        f"Begin. Emit your first action(s)."
    )


def build_followup_prompt(task, observations: list[str],
                          gen_left: int, calls_left: int) -> str:
    obs = "\n".join(observations) if observations else "(no output)"
    return (
        f"=== OBSERVATIONS ===\n{obs[:_OBS_CAP * 3]}\n\n"
        f"Budget left: ~{gen_left} generated tokens, {calls_left} tool calls.\n"
        f"Continue with the next action(s), or DONE if finished."
    )


# --------------------------------------------------------------------------- #
# Episode driver
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Budgets:
    max_iters: int = 12
    max_tool_calls: int = 24
    max_gen_tokens: int = 4096


@dataclasses.dataclass
class TrajectoryPoint:
    iter: int
    gen_tokens_cumulative: int
    tool_calls_cumulative: int
    milestone_scores: list[int]     # per-milestone 0/1 at this iteration

    @property
    def milestone_frac(self) -> float:
        return (sum(self.milestone_scores) / len(self.milestone_scores)
                if self.milestone_scores else 0.0)

    def to_dict(self) -> dict:
        return {
            "iter": self.iter,
            "gen_tokens_cumulative": self.gen_tokens_cumulative,
            "tool_calls_cumulative": self.tool_calls_cumulative,
            "milestone_scores": list(self.milestone_scores),
            "milestone_frac": self.milestone_frac,
        }


@dataclasses.dataclass
class EpisodeResult:
    task_id: str
    bucket: str
    n_milestones: int
    trajectory: list[TrajectoryPoint]
    transcript: list[dict]          # per-turn {prompt, output, actions, obs}
    finished_reason: str            # "done" | "max_iters" | "gen_budget" | "call_budget"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "bucket": self.bucket,
            "n_milestones": self.n_milestones,
            "points": [p.to_dict() for p in self.trajectory],
            "finished_reason": self.finished_reason,
        }


def _normalize_gen(out: Union[str, GenOutput]) -> tuple[str, int]:
    if isinstance(out, GenOutput):
        return out.text, int(out.n_gen_tokens)
    return out, estimate_tokens(out)


def run_episode(task, generate: GenerateFn, budgets: Budgets | None = None,
                timeout: float = 5.0, keep_transcript: bool = True,
                verbose: bool = False) -> EpisodeResult:
    """Drive `generate` through the action protocol against a fresh sandboxed
    copy of `task`'s workspace. Grades all milestones after each turn and
    records a trajectory point (cumulative generated tokens + tool calls +
    per-milestone scores). Stops on DONE or on any exhausted budget."""
    budgets = budgets or Budgets()
    files = dict(task.initial_workspace)
    ws = pathlib.Path(tempfile.mkdtemp(prefix=f"edgebench_{task.task_id.replace('/', '_')}_"))
    transcript: list[dict] = []
    try:
        materialize_workspace(files, ws)
        gen_cum = 0
        calls_cum = 0
        # iter 0 baseline: grade the untouched workspace.
        base_scores = grade_milestones_on_dir(task, ws, timeout)
        traj = [TrajectoryPoint(0, 0, 0, base_scores)]
        last_hash = workspace_hash(ws)
        last_scores = base_scores

        prompt = build_initial_prompt(task, files)
        finished_reason = "max_iters"
        for it in range(1, budgets.max_iters + 1):
            out = generate(prompt)
            text, n_gen = _normalize_gen(out)
            gen_cum += n_gen
            actions = parse_actions(text)
            observations: list[str] = []
            done = False
            for act in actions:
                if calls_cum >= budgets.max_tool_calls:
                    observations.append("[tool-call budget exhausted]")
                    break
                if act.kind == "done":
                    done = True
                    continue
                if act.kind == "error":
                    observations.append(f"[protocol error] {act.reason}")
                    continue
                # read / run / write each consume one tool call.
                resolved = _resolve_in_root(ws, act.path) if act.path else None
                if act.kind in ("read", "run", "write") and resolved is None:
                    observations.append(
                        f"[rejected] unsafe/out-of-workspace path: {act.path!r}")
                    continue
                calls_cum += 1
                # A model's freeform output can name a DIRECTORY ("READ tests",
                # "READ .") or hit any other OSError — that must become an
                # observation, never an exception out of the episode (review
                # 2026-07-13: an unguarded IsADirectoryError here killed the
                # whole suite run on one malformed action).
                try:
                    if act.kind == "read":
                        if resolved.is_file():
                            observations.append(
                                f"--- {act.path} ---\n"
                                f"{resolved.read_text()[:_OBS_CAP]}")
                        elif resolved.is_dir():
                            listing = "\n".join(
                                sorted(p.name for p in resolved.iterdir()))
                            observations.append(
                                f"[read] {act.path} is a directory; contents:\n"
                                f"{listing[:_OBS_CAP]}")
                        else:
                            observations.append(
                                f"[read] no such file: {act.path}")
                    elif act.kind == "run":
                        r = run_workspace_file(ws, act.path, timeout)
                        tail = (r.stdout
                                + ("\n" + r.stderr if r.stderr else ""))[-_OBS_CAP:]
                        observations.append(
                            f"[run {act.path}] exit={r.returncode} "
                            f"timed_out={r.timed_out}\n{tail}")
                    elif act.kind == "write":
                        if resolved.is_dir():
                            observations.append(
                                f"[write rejected] {act.path} is a directory")
                        else:
                            resolved.parent.mkdir(parents=True, exist_ok=True)
                            resolved.write_text(act.content or "")
                            observations.append(
                                f"[wrote {act.path}] "
                                f"({len((act.content or '').splitlines())} lines)")
                except OSError as exc:
                    observations.append(f"[{act.kind} failed] {act.path}: "
                                        f"{type(exc).__name__}: {exc}")

            # Re-grade only if the workspace changed (grading is the hot cost).
            cur_hash = workspace_hash(ws)
            if cur_hash != last_hash:
                last_scores = grade_milestones_on_dir(task, ws, timeout)
                last_hash = cur_hash
            traj.append(TrajectoryPoint(it, gen_cum, calls_cum, last_scores))

            if keep_transcript:
                transcript.append({
                    "iter": it, "prompt": prompt, "output": text,
                    "n_gen_tokens": n_gen,
                    "actions": [dataclasses.asdict(a) for a in actions],
                    "observations": observations,
                    "milestone_scores": list(last_scores),
                })
            if verbose:
                print(f"  [{task.task_id}] iter {it}: "
                      f"{sum(last_scores)}/{len(last_scores)} milestones, "
                      f"gen={gen_cum} calls={calls_cum}", flush=True)

            if done:
                finished_reason = "done"
                break
            if gen_cum >= budgets.max_gen_tokens:
                finished_reason = "gen_budget"
                break
            if calls_cum >= budgets.max_tool_calls:
                finished_reason = "call_budget"
                break
            gen_left = max(0, budgets.max_gen_tokens - gen_cum)
            calls_left = max(0, budgets.max_tool_calls - calls_cum)
            prompt = build_followup_prompt(task, observations, gen_left, calls_left)

        return EpisodeResult(
            task_id=task.task_id, bucket=task.bucket,
            n_milestones=len(task.milestones), trajectory=traj,
            transcript=transcript, finished_reason=finished_reason)
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def run_suite(tasks, generate: GenerateFn, budgets: Budgets | None = None,
              timeout: float = 5.0, verbose: bool = False) -> list[EpisodeResult]:
    """Run `generate` over a whole task suite. Returns one EpisodeResult per
    task (call `.to_dict()` on each to feed `scoring.suite_summary`)."""
    results = []
    for task in tasks:
        res = run_episode(task, generate, budgets, timeout,
                          keep_transcript=False, verbose=verbose)
        results.append(res)
        if verbose:
            best = max((p.milestone_frac for p in res.trajectory), default=0.0)
            print(f"[suite] {task.task_id}: best={best:.2f} "
                  f"({res.finished_reason})", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Built-in agents
# --------------------------------------------------------------------------- #

class ReplayAgent:
    """A scripted `generate` callable: yields a fixed list of turn strings in
    order, ignoring the prompt. When the list is exhausted it emits `DONE`.
    Used by the tests (deterministic end-to-end coverage) and by the task
    dependency-verification utility. Reports exact generated-token counts via
    `estimate_tokens` so the cost axis is well-defined in tests."""

    def __init__(self, turns: list[str]):
        self.turns = list(turns)
        self.i = 0
        self.calls = 0

    def __call__(self, prompt: str) -> GenOutput:
        self.calls += 1
        if self.i < len(self.turns):
            text = self.turns[self.i]
            self.i += 1
        else:
            text = "DONE"
        return GenOutput(text=text, n_gen_tokens=estimate_tokens(text))


class CkptAgent:
    """Adapter turning one of the repo's checkpoints into a `generate` callable
    via the TRUE incremental prefill/forward_step decode path (mirrors
    `eval_exec_trace_text.greedy_generate`).

    UNTESTED until a GPU frees — model-in-the-loop validation is deferred (both
    GPUs busy at build time). Requires CUDA + the FLA stack. No test exercises
    this class; the harness itself is validated with `ReplayAgent`.
    """

    def __init__(self, ckpt_path: str, max_gen: int = 384,
                 device: str | None = None):
        # Lazy heavy imports so `import harness` stays torch-free.
        import torch  # noqa: F401
        from experiments.decode_bench import load_ours
        from transformers import AutoTokenizer
        self._torch = torch
        model, cfg = load_ours(ckpt_path)
        if getattr(model, "use_memory", False):
            model.use_memory = False
        self.model = model
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
        self.eos_id = self.tok.eos_token_id
        self.max_gen = max_gen

    def __call__(self, prompt: str) -> GenOutput:
        torch = self._torch
        ids = self.tok.encode(prompt, add_special_tokens=False)
        device = next(self.model.parameters()).device
        t = torch.tensor([ids], dtype=torch.long, device=device)
        gen_ids: list[int] = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cache, last_logits = self.model.prefill(t)
            nxt = last_logits[:, -1, :].float().argmax(-1, keepdim=True)
            for _ in range(self.max_gen):
                tid = int(nxt.item())
                if tid == self.eos_id:
                    break
                gen_ids.append(tid)
                logits, cache = self.model.forward_step(nxt, cache)
                nxt = logits[:, -1, :].float().argmax(-1, keepdim=True)
        text = self.tok.decode(gen_ids, skip_special_tokens=False)
        return GenOutput(text=text, n_gen_tokens=len(gen_ids))
