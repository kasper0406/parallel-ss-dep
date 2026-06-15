"""Realistic long-trajectory AGENTIC recall tasks — the WM probe on agent traces.

WHY. The project's deployment target is coding + AGENTIC workflows. An agent
must recall, thousands of tokens into a trajectory, an EARLIER tool output / set
variable / file content / user instruction. This generator builds plausible
ReAct-style tool-use transcripts that plant N competing facts early, pad with
realistic distractor steps to a controlled distance, then query ONE — the same
capacity-exceeding + non-memorizable regime that gives WM headroom (see
project_recall_no_headroom + project_recall_discrete_key_direction), but in
agent-transcript form.

REALISM (flagged honestly).
  - This is SEMI-SYNTHETIC. The transcript STRUCTURE (system/user/Thought/Action/
    Observation steps), the planted facts, and the distractor steps are templated.
    A real agent trace (Claude Code / SWE-agent logs) would be messier: free-form
    reasoning, variable tool schemas, multi-line file contents, errors + retries.
    We document the gap; the controlled version is what lets us ANNOTATE the
    binding/query/answer spans and SCORE recall. Transfer to real traces is the
    open next step (would need to mine + annotate real logs — out of scope here,
    no GPU/data-mining budget).
  - The recall mechanism mapping is identical to the code case: KEY = the queried
    field/instruction/variable NAME at the query (name_emb_key addressing); VALUE =
    the concrete answer token span at the binding (copy/pointer readout);
    mem_read_mask over the answer span.

FAMILIES (addressing recorded per-record):
  toolout   : recall a JSON field value from an early tool Observation. addressing="name"
  userinstr : recall a user-specified setting/filename/flag.            addressing="name"
  setvar    : recall a variable an earlier step assigned.               addressing="name"

SCHEMA: identical superset to gen_code_recall_tasks.py (drops into data_mix
`text_field: [problem_prompt, qwen_completion]` + eval_code_recall).

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_agentic_recall_tasks.py \
      --out data/agentic_recall_train.jsonl --n_examples 4000 --seed 0
  PYTHONPATH=. .venv/bin/python experiments/gen_agentic_recall_tasks.py \
      --out data/agentic_recall_heldout.jsonl --seed 7 \
      --distance_buckets 256,512,768,1024,1536 --per_bucket 60
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import string
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# ------------------------------------------------------------------- vocab pools
_CFG_KEYS = ["max_workers", "retry_limit", "cache_ttl", "batch_size", "timeout_ms",
             "pool_size", "shard_count", "queue_depth", "page_limit", "rate_cap",
             "buffer_kb", "max_redirects", "backlog", "warmup_steps", "grace_sec"]
_FILES = ["results_final.csv", "report_v2.json", "summary.parquet", "model_best.pt",
          "metrics.tsv", "config.lock", "output_clean.txt", "manifest.yaml",
          "train_split.idx", "eval_dump.jsonl", "checkpoint_42.bin", "log_full.txt"]
_FLAGS = ["--no-cache", "--strict", "--dry-run", "--verbose", "--fail-fast",
          "--reuse-env", "--profile", "--shuffle", "--deterministic", "--amp"]
_VARNAMES = ["session_id", "run_token", "job_uid", "trace_id", "build_hash",
             "deploy_key", "lease_id", "tenant_ref", "commit_sha", "request_id"]
_TASKS = [
    "Investigate the failing integration tests and propose a fix.",
    "Refactor the data-loading pipeline and re-run the benchmark.",
    "Audit the service configuration and reconcile it with the deployment spec.",
    "Reproduce the reported memory regression and isolate the cause.",
    "Migrate the legacy config files to the new schema and validate.",
    "Triage the open CI failures and summarize the root causes.",
]

_DISTRACTOR_ACTIONS = [
    ('list_dir(path="{p}")',
     'src/  tests/  README.md  pyproject.toml  {f1}  {f2}'),
    ('grep(pattern="{w}", path="src/")',
     'src/core.py:{n1}: def {w}_handler(ctx):\nsrc/util.py:{n2}: # see {w}'),
    ('read_file(path="src/{f1}")',
     'def main():\n    cfg = load()\n    return run(cfg)  # {n1} lines elided'),
    ('run_tests(target="tests/test_{w}.py")',
     '{n1} passed, {n2} failed in {n3}.{n4}s'),
    ('git_status()',
     'On branch main\nChanges not staged:\n  modified: src/{w}.py\n  modified: {f1}'),
    ('python(code="import sys; print(sys.version_info[:2])")',
     '({n1}, {n2})'),
    ('search_web(query="{w} best practice")',
     'top result: a blog post on {w}; {n1} related links omitted'),
    ('describe_table(name="{w}_events")',
     'columns: id INT, {w}_ts TIMESTAMP, payload JSONB, {n1} rows'),
]


def _rand_hash(rng, k=8):
    return "".join(rng.choice(string.hexdigits.lower()[:16]) for _ in range(k))


def _distractor_step(rng, i):
    tmpl_a, tmpl_o = rng.choice(_DISTRACTOR_ACTIONS)
    sub = dict(p=rng.choice([".", "src", "tests", "configs"]),
               w=rng.choice(["loader", "cache", "router", "worker", "schema",
                             "parser", "session", "metric", "queue", "shard"]),
               f1=rng.choice(_FILES), f2=rng.choice(_FILES),
               n1=rng.randint(2, 400), n2=rng.randint(0, 30),
               n3=rng.randint(0, 9), n4=rng.randint(10, 99))
    action = tmpl_a.format(**sub)
    obs = tmpl_o.format(**sub)
    return (f"Step {i}\n"
            f"Thought: Continuing the investigation.\n"
            f"Action: {action}\n"
            f"Observation: {obs}")


def _pad_steps(rng, tok, target_tokens, start_i, forbidden):
    steps, ntok, i = [], 0, start_i
    guard = 0
    while ntok < target_tokens and guard < 4000:
        guard += 1
        s = _distractor_step(rng, i)
        if any(fb and fb in s for fb in forbidden):
            continue
        steps.append(s)
        ntok += len(tok.encode("\n\n" + s, add_special_tokens=False))
        i += 1
    return "\n\n".join(steps), i


def _span(haystack, needle, start=0):
    j = haystack.find(needle, start)
    return None if j < 0 else (j, j + len(needle))


_HEADER = ("System: You are an autonomous coding agent. You call tools and must "
           "remember facts from earlier steps.\n\nUser: {task}")


def _make_toolout(rng, tok, dist, n):
    keys = rng.sample(_CFG_KEYS, min(n, len(_CFG_KEYS)))
    vals, seen = [], set()
    while len(vals) < len(keys):
        v = rng.randint(1000, 9999)
        if str(v) not in seen:
            seen.add(str(v)); vals.append(v)
    qi = rng.randrange(len(keys))
    qkey, qval = keys[qi], vals[qi]
    forbidden = {str(v) for v in vals}
    obs = "{" + ", ".join(f'"{k}": {v}' for k, v in zip(keys, vals)) + "}"
    step1 = (f"Step 1\nThought: First I will read the service settings.\n"
             f"Action: read_file(path=\"config/settings.json\")\n"
             f"Observation: {obs}")
    pad, ni = _pad_steps(rng, tok, dist, 2, forbidden | set(keys))
    query = (f"Step {ni}\nThought: I now need a value from the settings I read "
             f"earlier.\nQuestion: What was the value of \"{qkey}\" in the "
             f"settings file?")
    task = rng.choice(_TASKS)
    prompt = _HEADER.format(task=task) + "\n\n" + step1 + "\n\n" + pad + "\n\n" + query
    completion = (f"Back in Step 1, the settings file reported \"{qkey}\": "
                  f"{qval}.\n\nAnswer: {qval}")
    bind = _span(prompt, f'"{qkey}": {qval}')
    bind = (bind[0] + len(f'"{qkey}": '), bind[1]) if bind else None  # the value
    qspan = _span(prompt, f'"{qkey}"', prompt.rfind("Question:"))
    qspan = (qspan[0] + 1, qspan[1] - 1) if qspan else None
    ans = _span(completion, str(qval), completion.rfind("Answer:"))
    return dict(problem_prompt=prompt, qwen_completion=completion,
                answer=str(qval), extracted_code=f"print({qval})",
                family="toolout", addressing="name", n_bindings=len(keys),
                recall_key=qkey, source_value_text=str(qval), query_key_text=qkey,
                binding_char_span=bind, query_key_char_span=qspan,
                answer_char_span=ans)


def _make_userinstr(rng, tok, dist, n):
    # N user constraints; recall ONE (the output filename).
    k = min(n, 6)
    fname = rng.choice(_FILES)
    flags = rng.sample(_FLAGS, min(3, len(_FLAGS)))
    extras = [
        f"name the output file `{fname}`",
        f"pass {flags[0]} to the runner",
        "use 4-space indentation",
        f"cap concurrency at {rng.randint(2, 16)}",
        "write logs in JSON lines",
        f"set the random seed to {rng.randint(1, 999)}",
    ][:max(3, k)]
    rng.shuffle(extras)
    instr = "; ".join(extras)
    forbidden = {fname}
    user_line = (f"User: {rng.choice(_TASKS)} Please follow these requirements: "
                 f"{instr}.")
    pad, ni = _pad_steps(rng, tok, dist, 1, forbidden)
    query = (f"Step {ni}\nThought: I am about to save results and must follow the "
             f"user's naming requirement.\nQuestion: What output filename did the "
             f"user ask for?")
    prompt = (user_line + "\n\n" + pad + "\n\n" + query)
    completion = (f"The user's requirements specified to name the output file "
                  f"`{fname}`.\n\nAnswer: {fname}")
    bind = _span(prompt, f"`{fname}`")
    bind = (bind[0] + 1, bind[1] - 1) if bind else None
    # query key here is semantic ("output filename") — addressing on the role; we
    # record the role phrase. This is closer to content-addressing than name.
    qrole = "output filename"
    qspan = _span(prompt, qrole, prompt.rfind("Question:"))
    ans = _span(completion, fname, completion.rfind("Answer:"))
    return dict(problem_prompt=prompt, qwen_completion=completion,
                answer=fname, extracted_code=f"print('{fname}')",
                family="userinstr", addressing="content", n_bindings=len(extras),
                recall_key=qrole, source_value_text=fname, query_key_text=qrole,
                binding_char_span=bind, query_key_char_span=qspan,
                answer_char_span=ans)


def _make_setvar(rng, tok, dist, n):
    names = rng.sample(_VARNAMES, min(n, len(_VARNAMES)))
    vals = [_rand_hash(rng, 8) for _ in names]
    qi = rng.randrange(len(names))
    qname, qval = names[qi], vals[qi]
    forbidden = set(vals)
    assigns = "\n".join(f"{nm} = \"{v}\"" for nm, v in zip(names, vals))
    step1 = (f"Step 1\nThought: I will establish the run context variables.\n"
             f"Action: python(code=\"\"\"\n{assigns}\n\"\"\")\n"
             f"Observation: variables set ({len(names)} identifiers)")
    pad, ni = _pad_steps(rng, tok, dist, 2, forbidden | set(names))
    query = (f"Step {ni}\nThought: I need to reference a run-context variable I "
             f"set earlier.\nQuestion: What value was assigned to `{qname}`?")
    prompt = (_HEADER.format(task=rng.choice(_TASKS)) + "\n\n" + step1
              + "\n\n" + pad + "\n\n" + query)
    completion = (f"In Step 1 the variable `{qname}` was assigned the value "
                  f"\"{qval}\".\n\nAnswer: {qval}")
    bind = _span(prompt, f'{qname} = "{qval}"')
    bind = (bind[0] + len(f'{qname} = "'), bind[0] + len(f'{qname} = "') + len(qval)) if bind else None
    qspan = _span(prompt, f"`{qname}`", prompt.rfind("Question:"))
    qspan = (qspan[0] + 1, qspan[1] - 1) if qspan else None
    ans = _span(completion, qval, completion.rfind("Answer:"))
    return dict(problem_prompt=prompt, qwen_completion=completion,
                answer=qval, extracted_code=f"print('{qval}')",
                family="setvar", addressing="name", n_bindings=len(names),
                recall_key=qname, source_value_text=qval, query_key_text=qname,
                binding_char_span=bind, query_key_char_span=qspan,
                answer_char_span=ans)


_BUILDERS = {"toolout": _make_toolout, "userinstr": _make_userinstr,
             "setvar": _make_setvar}


def _measured_distance(prompt, rec, tok):
    b = rec.get("binding_char_span"); q = rec.get("query_key_char_span")
    if not b or not q:
        return 0
    lo, hi = (b[1], q[0]) if b[1] <= q[0] else (q[1], b[0])
    return len(tok.encode(prompt[lo:hi], add_special_tokens=False))


def build_one(rng, tok, family, dist, n):
    rec = _BUILDERS[family](rng, tok, dist, n)
    rec["approx_distance_tokens"] = _measured_distance(rec["problem_prompt"], rec, tok)
    rec["has_tests"] = False
    rec["score"] = 1.0
    rec["tier"] = f"agentic_recall_{family}"
    return rec


def _buckets_floor(d):
    for b in (1536, 1024, 768, 512, 256):
        if d >= b:
            return b
    return 256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n_examples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", type=str, default="toolout,userinstr,setvar")
    p.add_argument("--n_bindings_choices", type=str, default="6,10,15")
    p.add_argument("--distance_min", type=int, default=256)
    p.add_argument("--distance_max", type=int, default=1536)
    p.add_argument("--distance_buckets", type=str, default="")
    p.add_argument("--per_bucket", type=int, default=60)
    p.add_argument("--max_total_tokens", type=int, default=1850)
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    families = [f for f in args.families.split(",") if f.strip()]
    n_choices = sorted(int(x) for x in args.n_bindings_choices.split(",") if x.strip())
    rng = random.Random(args.seed)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _PER_BIND = {"toolout": 11, "userinstr": 12, "setvar": 12}
    _OVERHEAD = 200

    def _feasible_n(family, dist):
        budget = args.max_total_tokens - dist - _OVERHEAD
        per = _PER_BIND[family]
        feas = [n for n in n_choices if n * per <= budget]
        return feas or [n_choices[0]]

    def emit(f, family, dist, nominal, idx):
        feas = _feasible_n(family, dist)
        for _ in range(3):
            n = rng.choice(feas)
            rec = build_one(rng, tok, family, dist, n)
            full = rec["problem_prompt"] + "\n\n" + rec["qwen_completion"]
            ntok = len(tok.encode(full, add_special_tokens=False))
            if ntok <= args.max_total_tokens and rec.get("binding_char_span") \
                    and rec.get("query_key_char_span") and rec.get("answer_char_span"):
                rec["task_id"] = f"{family}/d{nominal}/{idx}"
                rec["sample_idx"] = idx
                rec["total_tokens"] = ntok
                f.write(json.dumps(rec) + "\n")
                return True
        return False

    written = 0
    dist_hist, fam_hist = {}, {fam: 0 for fam in families}
    with open(out_path, "w") as f:
        if args.distance_buckets.strip():
            buckets = [int(b) for b in args.distance_buckets.split(",") if b.strip()]
            idx = 0
            for fam in families:
                for b in buckets:
                    got = 0
                    tries = 0
                    while got < args.per_bucket and tries < args.per_bucket * 30:
                        tries += 1
                        if emit(f, fam, b, b, idx):
                            written += 1; got += 1; fam_hist[fam] += 1
                            dist_hist[b] = dist_hist.get(b, 0) + 1
                        idx += 1
        else:
            idx = 0
            while written < args.n_examples and idx < args.n_examples * 30:
                fam = rng.choice(families)
                dist = rng.randint(args.distance_min, args.distance_max)
                nominal = _buckets_floor(dist)
                if emit(f, fam, dist, nominal, idx):
                    written += 1; fam_hist[fam] += 1
                    dist_hist[nominal] = dist_hist.get(nominal, 0) + 1
                idx += 1
    print(f"wrote {written} -> {out_path}")
    print(f"  families: {fam_hist}")
    print(f"  nominal-distance histogram: {dict(sorted(dist_hist.items()))}")


if __name__ == "__main__":
    sys.exit(main())
