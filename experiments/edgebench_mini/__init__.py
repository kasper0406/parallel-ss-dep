"""EdgeBench-mini — a cost-normalized agentic trajectory benchmark for small
(sub-1B) bounded-state code models.

Adapts the ByteDance EdgeBench methodology (day-scale multi-step tasks scored
by a score-vs-interaction-time trajectory, hidden-judge separation) sized DOWN
for ~402M models used as a *dev signal*: greedy HumanEval-164 was shown too
noisy to order checkpoints (a 13-17 band swallowed a whole RL arc — see memory
`project_humaneval_config_artifact`). The committed north star is a cheap
bounded-state long-context coding AGENT scored on cost + adaptivity, not
benchmark rank; this harness is the metric family where an O(1)-decode model
can be *ordered* against itself rather than trailing on a point score.

Submodules:
  tasks    — deterministic, seeded, executable multi-file task generation with
             sequential dependent milestones graded by HIDDEN tests.
  harness  — the agent loop (text edit/read/run protocol), subprocess sandbox,
             milestone grading, and two built-in agents (a scripted ReplayAgent
             for tests, and an untested-until-GPU CkptAgent for the repo's
             checkpoints).
  scoring  — trajectory metrics (score@budget curves in generated-token AND
             tool-call units, AUC, cost-normalized score, bootstrap CIs) and a
             checkpoint comparison table with non-overlapping-CI detection.
  validate_discrimination — the acceptance-gate script (GPU-required, untested):
             does the harness order base < SFT < RL monotonically with
             non-overlapping CIs where HumanEval-164 greedy cannot?
"""
