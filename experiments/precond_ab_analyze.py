"""Analyze the muon-vs-fused-DeltaNet-NS production A/B.

Reads the two arm logs (muon.log, fused.log) + per-arm nvidia-smi peak-memory
samples, emits:
  - loss_curves.csv  (step, muon_tloss, muon_tloss_ema, fused_tloss,
                      fused_tloss_ema, gap_ema)
  - val_curves.csv   (step, muon_val_ppl, fused_val_ppl)
  - DELTANET_PRECONDITIONER_AB.md  (aligned matched-step table, tok/s + ms/step,
                                    peak GPU memory, exclusive opt-step timing,
                                    honest verdict)

Run:  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/precond_ab_analyze.py runs/precond_ab
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import pathlib


_LINE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+-]+)")
_VAL = re.compile(r"VAL\s+loss=([\d.]+)\s+ppl=([\d.]+)")


def parse_log(path: pathlib.Path):
    """Return (steps->tloss, steps->tok_per_sec, steps->val_ppl)."""
    tloss, toks, val = {}, {}, {}
    cur_step = 0
    for ln in path.read_text(errors="ignore").splitlines():
        m = _LINE.match(ln)
        if m:
            s = int(m.group(1))
            cur_step = s
            toks[s] = float(m.group(2))
            tloss[s] = float(m.group(3))
            continue
        v = _VAL.search(ln)
        if v:
            val[cur_step] = float(v.group(2))
    return tloss, toks, val


def ema(series: dict, alpha=0.1):
    out, prev = {}, None
    for s in sorted(series):
        x = series[s]
        prev = x if prev is None else alpha * x + (1 - alpha) * prev
        out[s] = prev
    return out


def peak_mem_mib(path: pathlib.Path):
    if not path.exists():
        return None
    vals = []
    for ln in path.read_text(errors="ignore").splitlines():
        ln = ln.strip()
        if ln.isdigit():
            vals.append(int(ln))
    return max(vals) if vals else None


def median_toks(toks: dict, min_step: int):
    xs = [v for s, v in toks.items() if s >= min_step and v > 0]
    return statistics.median(xs) if xs else None


def exclusive_opt_timing():
    """Clean exclusive opt-step + memory numbers on the now-free GPU 1."""
    out = {}
    base = ["--d_model", "896", "--n_layers", "10", "--n_heads", "14",
            "--d_head", "64"]
    try:
        r = subprocess.run(
            [".venv/bin/python", "experiments/exp_deltanet_precond_fused.py",
             "--mode", "profile", "--warmup", "30", "--iters", "30",
             "--reps", "12"] + base,
            capture_output=True, text=True, timeout=900,
            env={**_ENV})
        out["profile"] = r.stdout
    except Exception as e:
        out["profile"] = f"(skipped: {e})"
    try:
        r = subprocess.run(
            [".venv/bin/python", "experiments/exp_deltanet_precond_fused.py",
             "--mode", "memory"] + base,
            capture_output=True, text=True, timeout=900, env={**_ENV})
        out["memory"] = r.stdout
    except Exception as e:
        out["memory"] = f"(skipped: {e})"
    return out


import os
_ENV = dict(os.environ)
_ENV["PYTHONPATH"] = _ENV.get("PYTHONPATH", "") + ":."
_ENV["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/precond_ab")
    meta = {}
    mp = out_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    B = int(meta.get("batch", 12)); T = int(meta.get("T", 2048))
    GA = int(meta.get("grad_accum", 6)); STEPS = int(meta.get("steps", 2500))
    kwarm = int(meta.get("k_warmup", 200))
    tok_per_step = B * T * GA

    m_t, m_tok, m_val = parse_log(out_dir / "muon.log")
    f_t, f_tok, f_val = parse_log(out_dir / "fused.log")
    m_ema, f_ema = ema(m_t), ema(f_t)

    # ---- CSVs ----
    common = sorted(set(m_t) & set(f_t))
    with open(out_dir / "loss_curves.csv", "w") as fh:
        fh.write("step,muon_tloss,muon_tloss_ema,fused_tloss,"
                 "fused_tloss_ema,gap_ema\n")
        for s in common:
            fh.write(f"{s},{m_t[s]:.5f},{m_ema[s]:.5f},{f_t[s]:.5f},"
                     f"{f_ema[s]:.5f},{m_ema[s]-f_ema[s]:+.5f}\n")
    vcommon = sorted(set(m_val) & set(f_val))
    with open(out_dir / "val_curves.csv", "w") as fh:
        fh.write("step,muon_val_ppl,fused_val_ppl,muon_minus_fused\n")
        for s in vcommon:
            fh.write(f"{s},{m_val[s]:.4f},{f_val[s]:.4f},"
                     f"{m_val[s]-f_val[s]:+.4f}\n")

    # ---- wall-clock + memory ----
    m_med = median_toks(m_tok, kwarm + 100)
    f_med = median_toks(f_tok, kwarm + 100)
    m_ms = (tok_per_step / m_med * 1e3) if m_med else None
    f_ms = (tok_per_step / f_med * 1e3) if f_med else None
    m_peak = peak_mem_mib(out_dir / "muon.mem")
    f_peak = peak_mem_mib(out_dir / "fused.mem")

    excl = exclusive_opt_timing()

    # ---- matched-step table (EMA) ----
    def at(d, s):
        ks = [k for k in d if k <= s]
        return d[max(ks)] if ks else None
    rows = []
    grid = list(range(200, max(common or [0]) + 1, 200))
    for s in grid:
        mm, ff = at(m_ema, s), at(f_ema, s)
        if mm is None or ff is None:
            continue
        rows.append((s, mm, ff, mm - ff))

    last = common[-1] if common else None
    final_gap = (m_ema[last] - f_ema[last]) if last else None
    val_gaps = [m_val[s] - f_val[s] for s in vcommon]
    mean_val_gap = statistics.mean(val_gaps) if val_gaps else None

    # ---- write the report ----
    md = []
    md.append("# Fused per-head DeltaNet-NS vs Muon — production pretrain A/B\n")
    md.append(f"**Config (BOTH arms identical):** real production trunk "
              f"10L × d896 × 14h, FiLM(0,5;1,6;2,7;3,8;4,9) K=3 (warmup "
              f"{kwarm}), `--bf16 --tf32 --bf16_optim_state "
              f"--activation_checkpointing --no-compile`, data "
              f"`configs/pretrain_mix_v4.yaml`, T={T}, batch={B}, "
              f"grad_accum={GA} (= {tok_per_step:,} tok/step), WSD "
              f"lr=1.4e-3 / lr_muon=5e-3, seed=0, {STEPS} steps. GPU 1 only, "
              f"sequential. The ONLY difference between arms is the q/k/v/b "
              f"orthogonalization (`--matrix_optimizer`).\n")
    md.append("## Wall-clock + peak memory (the clean deltas)\n")
    md.append("| arm | median tok/s | ms/step | peak GPU mem (MiB) |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| muon | {m_med:.0f} | {m_ms:.0f} | {m_peak} |"
              if m_med else "| muon | n/a | n/a | n/a |")
    md.append(f"| fused | {f_med:.0f} | {f_ms:.0f} | {f_peak} |"
              if f_med else "| fused | n/a | n/a | n/a |")
    if m_med and f_med:
        md.append(f"\nFused throughput vs muon: "
                  f"**{f_med/m_med:.3f}×** (>1 ⇒ fused faster). "
                  f"Peak-mem Δ (fused−muon): "
                  f"**{(f_peak-m_peak) if (m_peak and f_peak) else 'n/a'} MiB**.\n")
    md.append("> Note: at grad_accum %d the optimizer step is ~0.5%% of a full "
              "training step (perf doc §5), so full-step tok/s is expected to be "
              "near-identical between arms; the isolated opt-step timing below "
              "is where the per-head-NS cost difference actually shows.\n" % GA)

    md.append("## Loss — matched-step EMA (α=0.1) of train CE\n")
    md.append("gap = muon_ema − fused_ema (positive ⇒ fused ahead).\n")
    md.append("| step | muon CE (ema) | fused CE (ema) | gap |")
    md.append("|---:|---:|---:|---:|")
    for s, mm, ff, g in rows:
        md.append(f"| {s} | {mm:.4f} | {ff:.4f} | {g:+.4f} |")
    if final_gap is not None:
        md.append(f"\nFinal (step {last}) EMA train-CE gap (muon−fused): "
                  f"**{final_gap:+.4f}**.\n")

    if vcommon:
        md.append("## VAL ppl at matched steps\n")
        md.append("| step | muon VAL ppl | fused VAL ppl | muon−fused |")
        md.append("|---:|---:|---:|---:|")
        for s in vcommon:
            md.append(f"| {s} | {m_val[s]:.3f} | {f_val[s]:.3f} | "
                      f"{m_val[s]-f_val[s]:+.4f} |")
        md.append(f"\nMean VAL-ppl(muon−fused) over {len(vcommon)} evals: "
                  f"**{mean_val_gap:+.4f}** (positive ⇒ fused lower ppl).\n")

    md.append("## Exclusive opt-step timing + memory (GPU 1 free after the run)\n")
    md.append("```\n" + (excl.get("profile") or "") + "\n"
              + (excl.get("memory") or "") + "```\n")

    md.append("## Verdict\n")
    md.append("_(auto-generated draft — fill in the honest read below; the "
              "numbers above are the source of truth)_\n")
    md.append("- (a) lower loss at matched steps? see the EMA gap column + "
              "final gap; if |gap| ≲ per-eval noise, this is WITHIN single-seed "
              "noise at this budget (a 2nd seed / more steps would be needed to "
              "resolve it — consistent with the perf-doc prediction of a "
              "MODEST, convergence-speed win on dense text).\n")
    md.append("- (b) faster wall-clock? full-step tok/s ratio above (expected "
              "≈1.0 at this grad_accum); the isolated opt-step block shows the "
              "true per-head-NS vs Muon step-cost delta.\n")
    md.append("- (c) ≤ memory? peak-mem Δ above + the exclusive state/transient "
              "table.\n")

    (pathlib.Path("DELTANET_PRECONDITIONER_AB.md")).write_text("\n".join(md))
    print("Wrote DELTANET_PRECONDITIONER_AB.md, "
          f"{out_dir}/loss_curves.csv, {out_dir}/val_curves.csv")
    print(f"  muon: tok/s={m_med}, ms/step={m_ms}, peak={m_peak} MiB")
    print(f"  fused: tok/s={f_med}, ms/step={f_ms}, peak={f_peak} MiB")
    print(f"  final EMA train-CE gap (muon-fused)={final_gap}")
    print(f"  mean VAL-ppl gap (muon-fused)={mean_val_gap}")


if __name__ == "__main__":
    main()
