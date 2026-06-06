"""Offline analysis of eval_humaneval --diagnose_json dumps (no GPU).

Answers the two pre-RL questions:
  (1) FAILURE MODES — re-grade each dumped output via code_grader to get tiers
      (syntax/exec/runtime/partial/pass) + sample diagnoses. Obvious or just hard?
  (2) THINKING QUANTIFICATION — latent steps/problem, pass-vs-fail correlation.
Plus the canonical FLIP: which problems thinking gains vs loses (think vs nothink).
"""
import json, sys, statistics
sys.path.insert(0, ".")
from experiments.eval_humaneval import _extract_code_block
import experiments.code_grader as CG

think = json.load(open("/tmp/diag_think.json"))
nothink = json.load(open("/tmp/diag_nothink.json"))
think = think[list(think)[0]]["per_problem"]
nothink = nothink[list(nothink)[0]]["per_problem"]
probs = {p.task_id: p for p in CG.load_humaneval()}


def tier_of(rec):
    gen = rec["gen"]
    code = _extract_code_block(gen)
    full = code if code is not None else gen
    p = probs[rec["task_id"]]
    gp = CG.Problem(task_id=p.task_id, prompt="", tests=p.tests, entry_point=p.entry_point)
    try:
        return CG.grade(gp, full)
    except Exception as e:
        return None


def summarize(name, recs):
    tiers, samples = {}, {}
    npass = 0
    for r in recs:
        res = tier_of(r)
        t = (res.tier if res else "grade_err")
        if r["passed"]:
            t = "pass"; npass += 1
        tiers[t] = tiers.get(t, 0) + 1
        if not r["passed"] and t not in samples:
            samples[t] = (r["task_id"], (getattr(res, "error_text", None) or "")[:160],
                          r["gen"][:140].replace("\n", "\\n"))
    print(f"\n===== {name}: pass(canonical)={npass}/{len(recs)} =====")
    print(f"  TIERS: {dict(sorted(tiers.items(), key=lambda x:-x[1]))}")
    th = [r["think_total"] for r in recs]
    if any(th):
        tp = [r["think_total"] for r in recs if r["passed"]]
        tf = [r["think_total"] for r in recs if not r["passed"]]
        print(f"  THINK steps/problem: mean={statistics.mean(th):.2f} median={statistics.median(th)} "
              f"max={max(th)} total={sum(th)} | nonzero on {sum(1 for x in th if x>0)}/{len(th)} problems")
        if tp and tf:
            print(f"  THINK on PASS: mean={statistics.mean(tp):.2f} | on FAIL: mean={statistics.mean(tf):.2f}")
    for t, (tid, err, cd) in samples.items():
        print(f"   [{t}] {tid}: {err!r}\n        gen: {cd!r}")
    return {r["task_id"]: r["passed"] for r in recs}


tp = summarize("THINKING-ON @0.3", think)
np_ = summarize("NO-THINK", nothink)
gained = [k for k in tp if tp[k] and not np_.get(k)]
lost = [k for k in tp if not tp[k] and np_.get(k)]
print(f"\n===== FLIP (canonical) =====")
print(f"GAINED (think solved, no-think failed) [{len(gained)}]: {gained}")
print(f"LOST   (think broke, no-think passed)  [{len(lost)}]: {lost}")
print(f"NET = {len(gained)-len(lost):+d}")
