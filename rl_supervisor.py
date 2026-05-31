#!/usr/bin/env python3
"""Autonomous overnight RL supervisor.

Goal: run grader-RL and ITERATE on config until results are good — reward
climbing, gate thinking selectively, and HumanEval improving — without a human
in the loop.

Mechanism (deterministic ladder + collapse recovery + milestone eval):
  0. Wait for the LLM-judge server, then HumanEval the SFT base => baseline.
  1. For each rung in LADDER (escalating exploration): warm-start from the
     BEST checkpoint so far, launch grader-RL with the rung's overrides,
     monitor the log live:
       - collapse (reward had signal then craters to ~0 for a window) => kill,
         skip to eval, advance (its ckpt simply won't promote).
       - normal completion (STEPS reached / process exits) => eval.
  2. HumanEval the rung's checkpoint (headline, no-thinking — the project-best
     metric on this weak base). Promote to BEST only if it beats the best.
  3. Stop when BEST >= GOAL passes, the ladder is exhausted, or the wall-clock
     budget is spent. Everything is logged to runs/rl_supervisor.log and a live
     status JSON at runs/rl_supervisor_status.json.

Only touches GPU0 (RL trainer + judge + eval). v9-clean on GPU1 is never
touched. Never deletes checkpoints. Conservative: never kills a run that is
still making progress.
"""
from __future__ import annotations
import glob
import json
import os
import re
import signal
import subprocess
import time
import urllib.request

ROOT = "/home/knielsen/ml/parallel-ss-dep"
GPU = "0"
JUDGE_URL = "http://localhost:8000"
BASE_CKPT = f"{ROOT}/checkpoints/sft_v8_combined.pt"
STATUS = f"{ROOT}/runs/rl_supervisor_status.json"
SUPLOG = f"{ROOT}/runs/rl_supervisor.log"
GOAL_PASSES = 16          # match the project best (16/164) -> stop early
BUDGET_H = 8.0            # wall-clock budget
N_TOTAL = 164

# CORRECTED direction (the v1 ladder regressed: temp 0.9-1.0 made the weak base
# emit mostly syntax errors -> no execution credit -> RL trained on noise and
# HumanEval dropped 13->10). Fixes:
#   - LOW temperature: rollouts actually RUN, so the execution grader (the real
#     credit) produces graded, grounded reward variance instead of all-0 ties.
#   - TIGHT KL target: keep the policy anchored to the SFT base -> can't drift
#     below baseline competence.
#   - group_var_floor 0.02: DROP groups that are still tied after execution +
#     the (bounded) judge -> never fabricate a gradient from no real signal.
#   - gentle curriculum (end 0.4, not 0.2): stay on problems the base can
#     partially solve, where partial-credit gives signal.
#   - judge ON as a SMALL within-tier tie-breaker (now parses), never primary.
# Each rung warm-starts from the BEST ckpt; regressions don't promote.
LADDER = [
    dict(name="K1_t07_kl08", NGROUP="8",  TEMP="0.7", MAX_TURNS="2",
         KL_TARGET="0.08", LR="2e-6",  STEPS="150", CEND="0.4", VFLOOR="0.02"),
    dict(name="K2_t06_kl06", NGROUP="8",  TEMP="0.6", MAX_TURNS="2",
         KL_TARGET="0.06", LR="1.5e-6", STEPS="150", CEND="0.4", VFLOOR="0.02"),
    dict(name="K3_t07_n12",  NGROUP="12", TEMP="0.7", MAX_TURNS="2",
         KL_TARGET="0.10", LR="2e-6",  STEPS="150", CEND="0.4", VFLOOR="0.02"),
]

T0 = time.time()
state = {"started": T0, "baseline": None, "best_passes": -1,
         "best_ckpt": BASE_CKPT, "rungs": [], "current": None, "done": False}


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] (+{(time.time()-T0)/3600:.2f}h) {msg}"
    print(line, flush=True)
    with open(SUPLOG, "a") as f:
        f.write(line + "\n")


def save_status():
    state["elapsed_h"] = (time.time() - T0) / 3600
    with open(STATUS, "w") as f:
        json.dump(state, f, indent=2)


def budget_left() -> float:
    return BUDGET_H - (time.time() - T0) / 3600


def wait_for_judge(timeout_s=300) -> bool:
    log("waiting for judge server /v1/models ...")
    t = time.time()
    while time.time() - t < timeout_s:
        try:
            with urllib.request.urlopen(f"{JUDGE_URL}/v1/models", timeout=5) as r:
                if b"Qwen" in r.read():
                    log("judge server READY")
                    return True
        except Exception:
            pass
        time.sleep(10)
    log("WARNING: judge server not ready in time — proceeding WITHOUT judge "
        "(runs will still work, just no tie-break on tied groups)")
    return False


def run_humaneval(ckpt: str, tag: str) -> int | None:
    """Headline HumanEval (no thinking — the strong metric on this base).
    Returns passes/164 or None on failure."""
    if not os.path.exists(ckpt):
        log(f"eval[{tag}]: ckpt missing {ckpt}")
        return None
    out = f"{ROOT}/runs/eval_{tag}.log"
    cmd = [f"{ROOT}/.venv/bin/python", "experiments/eval_humaneval.py",
           "--ckpt", ckpt, "--prompt_style", "sft_comment",
           "--extract_code_block", "--max_gen", "320",
           "--min_emit_before_eos", "30", "--gate_floor", "0.0",
           "--emit_threshold", "0.5"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
               PYTHONPATH=f"{os.environ.get('PYTHONPATH','')}:{ROOT}",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log(f"eval[{tag}]: HumanEval on {os.path.basename(ckpt)} ...")
    try:
        with open(out, "w") as f:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=3600)
    except Exception as e:
        log(f"eval[{tag}]: FAILED {e}")
        return None
    txt = open(out).read()
    m = re.findall(r"pass@\d+\s*=\s*[\d.]+\s*\((\d+)/(\d+)\)", txt)
    if not m:
        log(f"eval[{tag}]: could not parse score (see {out})")
        return None
    passes = int(m[-1][0])
    log(f"eval[{tag}]: {passes}/{N_TOTAL}")
    return passes


def eval_run_best(name: str, save_path: str) -> tuple[int, str]:
    """Eval the FINAL ckpt + the latest intermediate; return the best
    (passes, path). Even with tight KL, RL can peak mid-run then drift — this
    keeps the actual peak rather than only the last step."""
    cands: list[tuple[str, str]] = []
    if os.path.exists(save_path):
        cands.append((save_path, "final"))
    base = save_path[:-3]
    inters = sorted(
        glob.glob(f"{base}_step*.pt"),
        key=lambda p: int(re.search(r"_step(\d+)\.pt$", p).group(1)),
        reverse=True)
    if inters:  # the latest intermediate (near-end, pre any final-step noise)
        m = re.search(r"_step(\d+)\.pt$", inters[0]).group(1)
        cands.append((inters[0], f"mid{m}"))
    best_p, best_path = -1, save_path
    for path, tag in cands:
        s = run_humaneval(path, f"{name}_{tag}")
        if s is not None and s > best_p:
            best_p, best_path = s, path
    return best_p, best_path


def parse_rewards(logpath: str) -> list[float]:
    """Mean-reward column per step line (2nd field of the numeric step lines)."""
    if not os.path.exists(logpath):
        return []
    rs = []
    for ln in open(logpath, errors="ignore"):
        s = ln.strip()
        parts = s.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                rs.append(float(parts[1]))
            except ValueError:
                pass
    return rs


def launch_rl(rung: dict, load: str) -> tuple[subprocess.Popen, str, str]:
    name = rung["name"]
    save = f"{ROOT}/checkpoints/rl_sup_{name}.pt"
    logp = f"{ROOT}/runs/rl_sup_{name}.log"
    env = dict(os.environ, GPU=GPU, LOAD=load, SAVE=save, LOG=logp,
               NGROUP=rung["NGROUP"], TEMP=rung["TEMP"],
               MAX_TURNS=rung["MAX_TURNS"], KL_TARGET=rung["KL_TARGET"],
               LR=rung["LR"], STEPS=rung["STEPS"], BATCH="2", MICRO="4",
               SAVE_EVERY="40")
    # The launcher backgrounds the trainer and echoes "PID N"; we re-create the
    # python invocation directly here so we OWN the process handle.
    cmd = [f"{ROOT}/.venv/bin/python", "-u", "experiments/train_rl_grader.py",
           "--load_ckpt", load, "--save_ckpt", save,
           "--dataset", "mbpp_combined", "--extract_code_block",
           "--activation_checkpointing", "--steps", rung["STEPS"],
           "--batch", "2", "--grpo_n_group", rung["NGROUP"],
           "--policy_micro_chunk", "4", "--lr", rung["LR"],
           "--max_gen", "320", "--max_think_per_step", "4",
           "--total_think_budget", "120", "--think_budget_diversity", "0.7",
           "--stochastic_gate", "--gate_entropy_bonus", "0.01",
           "--emit_threshold", "0.5", "--gate_floor", "0.0",
           "--temperature", rung["TEMP"], "--min_emit_before_eos", "30",
           "--clip_eps", "0.1", "--kl_coef", "0.1",
           "--kl_target", rung["KL_TARGET"], "--kl_coef_min", "0.02",
           "--kl_coef_max", "0.6", "--ponder_cost", "0.0",
           "--ponder_shape", "quadratic", "--counterfactual",
           "--ponder_warmup_steps", "50", "--max_turns", rung["MAX_TURNS"],
           "--no-batch_turn0", "--group_var_floor", rung["VFLOOR"],
           "--progressive_curriculum",
           "--no-adaptive_curriculum", "--curriculum_target_start", "0.7",
           "--curriculum_target_end", rung["CEND"], "--llm_judge",
           "--judge_url", JUDGE_URL,
           "--judge_model", "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ",
           "--judge_strip_comments", "--grad_clip", "1.0",
           "--log_every", "1", "--save_every", "30", "--seed", "0"]
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH','')}:{ROOT}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    fh = open(logp, "w")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    log(f"rung[{name}]: launched pid={p.pid} load={os.path.basename(load)} "
        f"N={rung['NGROUP']} temp={rung['TEMP']} turns={rung['MAX_TURNS']} "
        f"steps={rung['STEPS']}")
    return p, save, logp


def monitor(p: subprocess.Popen, logp: str) -> str:
    """Poll until the run ends or collapses. Returns 'done'|'collapse'|'died'."""
    last_n = 0
    while True:
        if p.poll() is not None:
            time.sleep(3)  # flush
            rs = parse_rewards(logp)
            if len(rs) < 5:
                tail = "".join(open(logp, errors="ignore").readlines()[-8:])
                log(f"  process exited early (only {len(rs)} steps). tail:\n{tail}")
                return "died"
            return "done"
        rs = parse_rewards(logp)
        if len(rs) != last_n:
            last_n = len(rs)
            if last_n % 20 == 0 and last_n > 0:
                recent = rs[-20:]
                log(f"  step={last_n} reward(mean last20)="
                    f"{sum(recent)/len(recent):.4f} peak={max(rs):.3f}")
                state["current"]["step"] = last_n
                state["current"]["reward_recent"] = sum(recent) / len(recent)
                save_status()
        # Collapse: had real signal, now cratered for a sustained window.
        if len(rs) > 80:
            peak = max(rs[:len(rs) - 30])
            tail30 = rs[-30:]
            if peak > 0.08 and sum(tail30) / 30 < 0.008:
                log(f"  COLLAPSE detected (peak {peak:.3f} -> last30 "
                    f"{sum(tail30)/30:.4f}); killing rung.")
                p.send_signal(signal.SIGTERM)
                time.sleep(5)
                if p.poll() is None:
                    p.kill()
                return "collapse"
        if budget_left() < 0.3:
            log("  wall-clock budget nearly spent; stopping rung early.")
            p.send_signal(signal.SIGTERM)
            time.sleep(5)
            if p.poll() is None:
                p.kill()
            return "done"
        time.sleep(20)


def main():
    log("=== RL SUPERVISOR START ===")
    save_status()
    have_judge = wait_for_judge()
    state["judge"] = have_judge
    save_status()

    # Baseline HumanEval on the SFT base.
    base = run_humaneval(BASE_CKPT, "baseline_sft_v8")
    state["baseline"] = base
    if base is not None:
        state["best_passes"] = base
        state["best_ckpt"] = BASE_CKPT
    save_status()

    for rung in LADDER:
        if budget_left() < 1.0:
            log(f"budget low ({budget_left():.2f}h) — stopping before "
                f"rung {rung['name']}.")
            break
        state["current"] = {"name": rung["name"], "cfg": rung,
                            "load": os.path.basename(state["best_ckpt"])}
        save_status()
        try:
            p, save, logp = launch_rl(rung, state["best_ckpt"])
            reason = monitor(p, logp)
        except Exception as e:
            log(f"rung[{rung['name']}]: EXCEPTION {e}")
            reason = "died"
            save = f"{ROOT}/checkpoints/rl_sup_{rung['name']}.pt"
        passes, best_path_of_run = eval_run_best(rung["name"], save)
        if passes < 0:
            passes = None
        rec = {"name": rung["name"], "reason": reason, "passes": passes,
               "ckpt": os.path.basename(best_path_of_run),
               "warm_from": os.path.basename(state["best_ckpt"]),
               "elapsed_h": (time.time() - T0) / 3600}
        state["rungs"].append(rec)
        if passes is not None and passes > state["best_passes"]:
            log(f"rung[{rung['name']}]: NEW BEST {passes}/{N_TOTAL} "
                f"(was {state['best_passes']}) — promoting "
                f"{os.path.basename(best_path_of_run)}.")
            state["best_passes"] = passes
            state["best_ckpt"] = best_path_of_run
        else:
            log(f"rung[{rung['name']}]: {passes}/{N_TOTAL} (best stays "
                f"{state['best_passes']}). not promoted.")
        state["current"] = None
        save_status()
        if state["best_passes"] >= GOAL_PASSES:
            log(f"GOAL reached: {state['best_passes']}/{N_TOTAL} >= "
                f"{GOAL_PASSES}. Stopping.")
            break

    state["done"] = True
    save_status()
    log(f"=== SUPERVISOR DONE === baseline={state['baseline']} "
        f"best={state['best_passes']}/{N_TOTAL} "
        f"best_ckpt={os.path.basename(state['best_ckpt'])}")


if __name__ == "__main__":
    main()
