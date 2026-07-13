"""Tests for EdgeBench-mini (experiments/edgebench_mini/*).

CPU-only. The scoring/parsing tests are pure Python (no subprocess); the
grading / episode tests spawn short subprocesses (the sandbox) but need no GPU,
no FLA, no HF. `CkptAgent` / `validate_discrimination` are intentionally NOT
tested here — they are the deferred model-in-the-loop pieces (both GPUs busy at
build time); the harness is validated end-to-end with the scripted ReplayAgent.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from experiments.edgebench_mini import scoring
from experiments.edgebench_mini.harness import (
    Action,
    Budgets,
    GenOutput,
    ReplayAgent,
    _resolve_in_root,
    estimate_tokens,
    grade_milestones_on_files,
    is_safe_path,
    materialize_workspace,
    parse_actions,
    run_episode,
    run_script_src,
    run_workspace_file,
)
from experiments.edgebench_mini.tasks import (
    SUITE_SPEC,
    VISIBLE_TEST,
    build_suite,
    build_task,
    verify_task,
)
from experiments.edgebench_mini.scoring import CALLS_KEY, TOKENS_KEY


# --------------------------------------------------------------------------- #
# 1. Task generation determinism
# --------------------------------------------------------------------------- #

def test_build_task_deterministic_dict():
    a = build_task("pipeline", "hard", 301).to_dict()
    b = build_task("pipeline", "hard", 301).to_dict()
    assert a == b


def test_build_task_deterministic_across_families_and_buckets():
    for fam in ("pipeline", "registry"):
        for bucket in ("easy", "med", "hard"):
            t1 = build_task(fam, bucket, 7).to_dict()
            t2 = build_task(fam, bucket, 7).to_dict()
            assert t1 == t2, f"{fam}/{bucket} not deterministic"


def test_task_roundtrip_to_from_dict():
    from experiments.edgebench_mini.tasks import Task
    t = build_task("registry", "hard", 311)
    assert Task.from_dict(t.to_dict()).to_dict() == t.to_dict()


def test_different_seeds_give_different_tasks():
    a = build_task("pipeline", "hard", 301).to_dict()
    b = build_task("pipeline", "hard", 302).to_dict()
    assert a["initial_workspace"] != b["initial_workspace"]


# --------------------------------------------------------------------------- #
# 2. Suite composition / difficulty spread
# --------------------------------------------------------------------------- #

def test_suite_has_20_tasks():
    assert len(build_suite()) == len(SUITE_SPEC) == 20


def test_suite_difficulty_spread():
    tasks = build_suite()
    easy = [t for t in tasks if t.bucket == "easy"]
    hard = [t for t in tasks if t.bucket == "hard"]
    assert len(easy) >= 5, "need >=5 floor tasks"
    assert len(hard) >= 5, "need >=5 long-dependency tasks"
    # hard tasks are the state/memory stressor: long cross-milestone deps.
    assert all(t.dep_distance_profile["max"] >= 4 for t in hard)
    # milestone counts within the documented 3-8 range.
    assert all(3 <= len(t.milestones) <= 8 for t in tasks)


def test_suite_dep_distance_profile_metadata():
    t = build_task("pipeline", "hard", 301)
    prof = t.dep_distance_profile
    assert prof["max"] == max(m.dependency_distance for m in t.milestones)
    assert len(prof["per_milestone"]) == len(t.milestones)


# --------------------------------------------------------------------------- #
# 3. Hidden / visible test separation (anti-gaming)
# --------------------------------------------------------------------------- #

def test_hidden_tests_not_in_workspace():
    # The MILESTONE_OK sentinel lives ONLY in hidden tests; it must never
    # appear in any file the agent can read.
    for fam, bucket, seed in SUITE_SPEC[:6]:
        t = build_task(fam, bucket, seed)
        for path, content in t.initial_workspace.items():
            assert "MILESTONE_OK" not in content, path
        for ms in t.milestones:
            assert "MILESTONE_OK" in ms.hidden_test


def test_visible_test_is_present_in_workspace():
    t = build_task("pipeline", "med", 201)
    assert VISIBLE_TEST in t.initial_workspace
    assert "VISIBLE_OK" in t.initial_workspace[VISIBLE_TEST]


def test_episode_never_leaks_hidden_test_output_to_agent():
    t = build_task("registry", "easy", 111)
    res = run_episode(t, ReplayAgent(t.scripted_solution_turns()),
                      Budgets(max_iters=20, max_tool_calls=40,
                              max_gen_tokens=10 ** 6))
    for turn in res.transcript:
        for obs in turn["observations"]:
            assert "MILESTONE_OK" not in obs


# --------------------------------------------------------------------------- #
# 4. Milestone sequential dependency
# --------------------------------------------------------------------------- #

def test_reference_solves_all_initial_solves_none():
    t = build_task("pipeline", "hard", 302)
    ref = t.reference_workspace()
    assert all(grade_milestones_on_files(t, ref))
    assert not any(grade_milestones_on_files(t, t.initial_workspace))


def test_dependency_revert_makes_milestone_fail():
    # milestone k's hidden test fails when the earlier artifact it depends on
    # is reverted to its initial (stub/buggy) content.
    t = build_task("pipeline", "hard", 301)
    v = verify_task(t)
    assert v["reference_all_pass"]
    assert v["initial_none_pass"]
    assert v["dependency_ok"]
    assert v["dependency_checks"], "hard task must have dependency checks"


def test_dependency_verified_with_scripted_agent():
    # Drive the scripted agent but OMIT milestone 1's edit -> milestone 2 can
    # never pass (its hidden test calls transform, still a stub).
    t = build_task("pipeline", "med", 201)
    turns = t.scripted_solution_turns()
    # turn 0 is M1's WRITE (utils.py); drop it.
    assert turns[0].startswith("WRITE utils.py")
    res = run_episode(t, ReplayAgent(turns[1:]),
                      Budgets(max_iters=20, max_tool_calls=40,
                              max_gen_tokens=10 ** 6))
    # milestone index 1 (0-based) = "fix_pipeline" must stay failed everywhere.
    assert all(p.milestone_scores[1] == 0 for p in res.trajectory)


def test_registry_dependency_ok():
    assert verify_task(build_task("registry", "hard", 311))["dependency_ok"]


# --------------------------------------------------------------------------- #
# 5. Edit-protocol parsing
# --------------------------------------------------------------------------- #

def test_parse_write_python_fence():
    txt = "WRITE utils.py\n```python\nx = 1\ny = 2\n```"
    acts = parse_actions(txt)
    assert len(acts) == 1 and acts[0].kind == "write"
    assert acts[0].path == "utils.py"
    assert acts[0].content == "x = 1\ny = 2\n"


def test_parse_write_plain_fence():
    txt = "WRITE a/b.py\n```\nprint(1)\n```"
    acts = parse_actions(txt)
    assert acts[0].kind == "write" and acts[0].path == "a/b.py"
    assert acts[0].content == "print(1)\n"


def test_parse_read_run_done_and_prose():
    txt = ("Let me look first.\nREAD core.py\nnow run it\nRUN "
           "tests/test_visible.py\nDONE")
    acts = parse_actions(txt)
    kinds = [(a.kind, a.path) for a in acts]
    assert kinds == [("read", "core.py"), ("run", "tests/test_visible.py"),
                     ("done", None)]


def test_parse_multiple_actions_one_turn():
    txt = ("WRITE u.py\n```python\na=1\n```\nRUN u.py")
    acts = parse_actions(txt)
    assert [a.kind for a in acts] == ["write", "run"]


def test_parse_malformed_write_no_fence():
    acts = parse_actions("WRITE u.py\nprint(1)\n")
    assert acts[0].kind == "error" and acts[0].path == "u.py"


def test_parse_malformed_write_unterminated_fence():
    acts = parse_actions("WRITE u.py\n```python\nprint(1)\n")
    assert acts[0].kind == "error"
    assert "unterminated" in acts[0].reason.lower()


def test_parse_read_missing_path_is_error():
    acts = parse_actions("READ\n")
    assert acts[0].kind == "error"


def test_parse_empty_and_prose_only():
    assert parse_actions("just thinking out loud, no actions") == []


def test_parse_done_requires_standalone():
    # A standalone DONE terminates; a prose line that merely starts with
    # "done" does not (so the agent can't quit by narrating).
    assert [a.kind for a in parse_actions("DONE")] == ["done"]
    assert [a.kind for a in parse_actions("done.")] == ["done"]
    assert parse_actions("Done, that works and I'm confident.") == []


# --------------------------------------------------------------------------- #
# 6. Path safety / escape rejection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    "../evil.py", "/etc/passwd", "~/x.py", "a/../../b.py", "", "..",
    "sub/../../out.py", "\\\\server\\share", "C:/x.py",
])
def test_is_safe_path_rejects(bad):
    assert not is_safe_path(bad)


@pytest.mark.parametrize("good", ["utils.py", "a/b.py", "tests/test_visible.py",
                                  "pkg/sub/mod.py"])
def test_is_safe_path_accepts(good):
    assert is_safe_path(good)


def test_resolve_in_root_blocks_escape(tmp_path):
    assert _resolve_in_root(tmp_path, "../x.py") is None
    resolved = _resolve_in_root(tmp_path, "ok.py")   # containment, not existence
    assert resolved is not None
    root_real = pathlib.Path(str(pathlib.Path(tmp_path))).resolve()
    assert str(resolved).startswith(str(root_real))


def test_episode_survives_directory_read_and_write():
    # Review 2026-07-13: "READ tests" / "READ ." / "WRITE tests" name a
    # DIRECTORY; an unguarded read_text()/write_text() raised
    # IsADirectoryError out of run_episode and killed the whole suite run.
    # Must become observations, never an exception.
    t = build_task("pipeline", "easy", 101)
    turns = [
        "READ tests\nREAD .\nWRITE tests\n```python\nBAD=1\n```",
        "Run the tests thoroughly now.",   # prose keyword false-positive ->
        "DONE",                            # RUN path="the" (no such file)
    ]
    res = run_episode(t, ReplayAgent(turns),
                      Budgets(max_iters=6, max_tool_calls=20,
                              max_gen_tokens=10 ** 6))
    assert res.finished_reason == "done"
    joined = "\n".join(o for turn in res.transcript
                       for o in turn["observations"])
    assert "is a directory" in joined
    # the directory-write was rejected, not applied.
    assert "[write rejected] tests is a directory" in joined


def test_pipeline_dependency_check_not_tautological():
    # Review 2026-07-13: pipeline milestones 5-7 previously reverted their OWN
    # file (feature.py) in verify_task, making the dependency check trivially
    # an ImportError. They must name a genuinely EARLIER artifact (utils.py).
    t = build_task("pipeline", "hard", 301)
    for ms in t.milestones:
        for dep in ms.depends_on_files:
            assert dep != "feature.py", (
                f"milestone {ms.index} reverts its own file — tautological")
    assert verify_task(t)["dependency_ok"]


def test_episode_rejects_path_escape_write():
    t = build_task("pipeline", "easy", 101)
    escape = "WRITE ../pwned.py\n```python\nBAD=1\n```"
    res = run_episode(t, ReplayAgent([escape, "DONE"]),
                      Budgets(max_iters=4, max_tool_calls=10,
                              max_gen_tokens=10 ** 6))
    joined = "\n".join(o for turn in res.transcript for o in turn["observations"])
    assert "rejected" in joined.lower()
    # And nothing got written outside the sandbox.
    assert not pathlib.Path(tempfile.gettempdir(), "pwned.py").exists()


# --------------------------------------------------------------------------- #
# 7. Sandbox execution + timeout
# --------------------------------------------------------------------------- #

def test_run_script_src_ok_and_fail(tmp_path):
    materialize_workspace({"m.py": "V = 5\n"}, tmp_path)
    ok = run_script_src(tmp_path, "import m; assert m.V == 5; print('MILESTONE_OK')")
    assert ok.ok and "MILESTONE_OK" in ok.stdout
    bad = run_script_src(tmp_path, "import m; assert m.V == 6")
    assert not bad.ok


def test_run_script_src_timeout(tmp_path):
    r = run_script_src(tmp_path, "while True:\n    pass\n", timeout=1.0)
    assert r.timed_out and not r.ok


def test_run_workspace_file_captures_output(tmp_path):
    materialize_workspace({"go.py": "print('hi'); print('bye')\n"}, tmp_path)
    r = run_workspace_file(tmp_path, "go.py")
    assert r.ok and "hi" in r.stdout and "bye" in r.stdout


def test_run_workspace_file_rejects_escape(tmp_path):
    r = run_workspace_file(tmp_path, "../x.py")
    assert not r.ok and "out-of-workspace" in r.stderr


# --------------------------------------------------------------------------- #
# 8. ReplayAgent end-to-end
# --------------------------------------------------------------------------- #

def test_replay_agent_solves_pipeline_easy():
    t = build_task("pipeline", "easy", 101)
    res = run_episode(t, ReplayAgent(t.scripted_solution_turns()),
                      Budgets(max_iters=20, max_tool_calls=40,
                              max_gen_tokens=10 ** 6))
    assert res.finished_reason == "done"
    assert max(p.milestone_frac for p in res.trajectory) == 1.0


def test_replay_agent_solves_registry_hard():
    t = build_task("registry", "hard", 311)
    res = run_episode(t, ReplayAgent(t.scripted_solution_turns()),
                      Budgets(max_iters=20, max_tool_calls=40,
                              max_gen_tokens=10 ** 6))
    assert res.finished_reason == "done"
    assert res.trajectory[-1].milestone_frac == 1.0
    # monotone progress: never loses a solved milestone.
    seen = 0
    for p in res.trajectory:
        s = sum(p.milestone_scores)
        assert s >= seen
        seen = s


def test_episode_records_point_per_iteration_and_costs_monotone():
    t = build_task("pipeline", "easy", 102)
    res = run_episode(t, ReplayAgent(t.scripted_solution_turns()),
                      Budgets(max_iters=20, max_tool_calls=40,
                              max_gen_tokens=10 ** 6))
    pts = res.trajectory
    assert pts[0].iter == 0
    for a, b in zip(pts, pts[1:]):
        assert b.gen_tokens_cumulative >= a.gen_tokens_cumulative
        assert b.tool_calls_cumulative >= a.tool_calls_cumulative


def test_budget_stops_episode():
    t = build_task("pipeline", "hard", 301)
    # A generous script but a tiny tool-call budget -> stop on call_budget.
    res = run_episode(t, ReplayAgent(t.scripted_solution_turns()),
                      Budgets(max_iters=20, max_tool_calls=2,
                              max_gen_tokens=10 ** 6))
    assert res.finished_reason in ("call_budget", "max_iters")
    assert res.trajectory[-1].tool_calls_cumulative <= 2


def test_gen_output_and_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd" * 10) == 10
    a = ReplayAgent(["READ x"])
    out = a("prompt")
    assert isinstance(out, GenOutput) and out.n_gen_tokens >= 1


# --------------------------------------------------------------------------- #
# 9. Scoring math vs hand-computed references
# --------------------------------------------------------------------------- #

def _traj():
    return {
        "task_id": "t", "bucket": "easy", "n_milestones": 2,
        "finished_reason": "done",
        "points": [
            {"iter": 0, TOKENS_KEY: 0, CALLS_KEY: 0,
             "milestone_scores": [0, 0], "milestone_frac": 0.0},
            {"iter": 1, TOKENS_KEY: 100, CALLS_KEY: 1,
             "milestone_scores": [1, 0], "milestone_frac": 0.5},
            {"iter": 2, TOKENS_KEY: 300, CALLS_KEY: 2,
             "milestone_scores": [1, 1], "milestone_frac": 1.0},
        ],
    }


def test_best_score_under_budget():
    pts = _traj()["points"]
    assert scoring.best_score_under_budget(pts, 0, TOKENS_KEY) == 0.0
    assert scoring.best_score_under_budget(pts, 100, TOKENS_KEY) == 0.5
    assert scoring.best_score_under_budget(pts, 250, TOKENS_KEY) == 0.5
    assert scoring.best_score_under_budget(pts, 300, TOKENS_KEY) == 1.0
    assert scoring.best_score_under_budget(pts, 50, TOKENS_KEY) == 0.0


def test_score_curve_and_auc_hand_computed():
    pts = _traj()["points"]
    assert scoring.score_curve(pts, [100, 300], TOKENS_KEY) == [0.5, 1.0]
    # trapezoid over log-x of [0.5,1.0] between log100,log300 / span = 0.75.
    auc = scoring.auc_normalized(pts, [100, 300], TOKENS_KEY, log_x=True)
    assert abs(auc - 0.75) < 1e-9


def test_task_summary_hand_computed():
    s = scoring.task_summary(_traj(), token_budgets=[100, 300],
                             call_budgets=[1, 2])
    assert s["best_score"] == 1.0 and s["final_score"] == 1.0
    assert s["gen_tokens_total"] == 300 and s["tool_calls_total"] == 2
    assert abs(s["score_per_1k_tokens"] - (1.0 / 0.3)) < 1e-9
    assert abs(s["auc_tokens"] - 0.75) < 1e-9


def test_auc_requires_two_budgets():
    with pytest.raises(ValueError):
        scoring.auc_normalized(_traj()["points"], [100], TOKENS_KEY)


# --------------------------------------------------------------------------- #
# 10. Bootstrap CI sanity
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_constant_zero_width():
    ci = scoring.bootstrap_ci([0.5, 0.5, 0.5], seed=0)
    assert ci["point"] == 0.5 and ci["lo"] == 0.5 and ci["hi"] == 0.5


def test_bootstrap_ci_deterministic():
    v = [0.1, 0.4, 0.9, 0.2, 0.7]
    assert scoring.bootstrap_ci(v, seed=3) == scoring.bootstrap_ci(v, seed=3)


def test_bootstrap_ci_edge_cases():
    assert scoring.bootstrap_ci([]) == {"point": None, "lo": None,
                                        "hi": None, "n": 0}
    one = scoring.bootstrap_ci([0.7])
    assert one["point"] == one["lo"] == one["hi"] == 0.7


def test_bootstrap_ci_brackets_point_and_orders():
    v = [0.0, 0.25, 0.5, 0.75, 1.0]
    ci = scoring.bootstrap_ci(v, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["lo"] < ci["hi"]


def test_bootstrap_diff_ci_separates_and_ties():
    sep = scoring.bootstrap_diff_ci([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], seed=0)
    assert sep["point"] == 1.0 and sep["excludes_zero"]
    tie = scoring.bootstrap_diff_ci([0.5, 0.5], [0.5, 0.5], seed=0)
    assert tie["point"] == 0.0 and not tie["excludes_zero"]


# --------------------------------------------------------------------------- #
# 11. Checkpoint comparison + monotonic separation gate
# --------------------------------------------------------------------------- #

def test_compare_checkpoints_orders_and_separates():
    named = {
        "hi": [{"auc_tokens": 0.9}, {"auc_tokens": 0.95}, {"auc_tokens": 0.92}],
        "lo": [{"auc_tokens": 0.1}, {"auc_tokens": 0.05}, {"auc_tokens": 0.12}],
    }
    comp = scoring.compare_checkpoints(named, headline_metric="auc_tokens",
                                       seed=0)
    assert comp["ranking"] == ["hi", "lo"]
    pair = comp["pairs"][0]
    assert pair["ci_disjoint"] and pair["paired_diff_excludes_zero"]


def test_check_monotonic_separation_pass_and_fail():
    named = {
        "a": [{"m": 1.0}] * 5, "b": [{"m": 0.5}] * 5, "c": [{"m": 0.0}] * 5,
    }
    comp = scoring.compare_checkpoints(named, headline_metric="m", seed=0)
    good = scoring.check_monotonic_separation(comp, ["a", "b", "c"])
    assert good["PASS"] and good["order_matches"]
    # wrong expected order -> fail on ordering.
    bad = scoring.check_monotonic_separation(comp, ["c", "b", "a"])
    assert not bad["PASS"]


def test_render_comparison_table_smoke():
    named = {"hi": [{"auc_tokens": 0.8}] * 3, "lo": [{"auc_tokens": 0.2}] * 3}
    comp = scoring.compare_checkpoints(named, headline_metric="auc_tokens")
    txt = scoring.render_comparison_table(comp)
    assert "checkpoint comparison" in txt and "hi" in txt and "lo" in txt


def test_suite_summary_shape():
    t = build_task("pipeline", "easy", 101)
    res = run_episode(t, ReplayAgent(t.scripted_solution_turns()),
                      Budgets(max_iters=20, max_tool_calls=40,
                              max_gen_tokens=10 ** 6))
    summ = scoring.suite_summary([scoring.task_summary(res.to_dict())])
    assert summ["n_tasks"] == 1
    assert "auc_tokens" in summ["metrics"]
    assert "easy" in summ["by_bucket"]


# --------------------------------------------------------------------------- #
# 12. Log-sigmoid interaction-time law (descriptive)
# --------------------------------------------------------------------------- #

def test_fit_log_sigmoid_degenerate_flat_curve():
    fit = scoring.fit_log_sigmoid([100, 200, 400], [0.5, 0.5, 0.5])
    assert fit["degenerate"] and abs(fit["L"] - 0.5) < 1e-9


def test_fit_log_sigmoid_rising_curve():
    budgets = [64, 128, 256, 512, 1024, 2048]
    scores = [0.0, 0.0, 0.2, 0.6, 0.9, 1.0]
    fit = scoring.fit_log_sigmoid(budgets, scores)
    assert not fit["degenerate"] and fit["k"] > 0 and fit["rmse"] < 0.2
