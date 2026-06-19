"""Analyze the embedding-optimizer A/B vs the reused precond_ab baselines.

Compares each embedding arm (runs/embed_ab/{lr2,lr5,lr10,rownorm}.log) against:
  - the Muon baseline  runs/precond_ab/muon.log   (shared-LR AdamW embeddings)
  - the per-head-NS arm runs/precond_ab/fused.log  (the matrix-optimizer win)
restricted to the matched first-N-step window (N = embed_ab steps), since the
embed arms run warmup->constant-peak with no decay tail = exactly the baseline's
first N steps (apples-to-apples; same seed/init/data/schedule in-window).

Emits:
  - runs/embed_ab/val_curves.csv   (step, VAL ppl per arm + both baselines)
  - runs/embed_ab/loss_curves.csv  (step, train-CE EMA per arm + both baselines)
  - EMBED_OPTIMIZER_AB.md          (matched-step tables + verdict)

Run:  .venv/bin/python experiments/embed_ab_analyze.py runs/embed_ab
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
import pathlib


_LINE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+-]+)")
_VAL = re.compile(r"VAL\s+loss=([\d.]+)\s+ppl=([\d.]+)")

ARMS = ["lr2", "lr5", "lr10", "rownorm"]
ARM_LABEL = {
    "lr2": "embed_lr 2x", "lr5": "embed_lr 5x", "lr10": "embed_lr 10x",
    "rownorm": "rownorm dualizer",
}


def parse_log(path: pathlib.Path):
    """Return (step->tloss, step->tok/s, step->val_ppl)."""
    tloss, toks, val = {}, {}, {}
    cur = 0
    if not path.exists():
        return tloss, toks, val
    for ln in path.read_text(errors="ignore").splitlines():
        m = _LINE.match(ln)
        if m:
            s = int(m.group(1)); cur = s
            toks[s] = float(m.group(2)); tloss[s] = float(m.group(3))
            continue
        v = _VAL.search(ln)
        if v:
            val[cur] = float(v.group(2))
    return tloss, toks, val


def ema(series, alpha=0.1):
    out, prev = {}, None
    for s in sorted(series):
        x = series[s]
        prev = x if prev is None else alpha * x + (1 - alpha) * prev
        out[s] = prev
    return out


def median_toks(toks, min_step):
    xs = [v for s, v in toks.items() if s >= min_step and v > 0]
    return statistics.median(xs) if xs else None


def main():
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/embed_ab")
    meta = json.loads((out_dir / "meta.json").read_text())
    N = int(meta["steps"]); B = int(meta["batch"]); T = int(meta["T"])
    GA = int(meta["grad_accum"]); kwarm = int(meta["k_warmup"])
    tok_per_step = B * T * GA

    base = pathlib.Path("runs/precond_ab")
    m_t, m_tok, m_val = parse_log(base / "muon.log")
    f_t, f_tok, f_val = parse_log(base / "fused.log")
    # truncate baselines to the embed-arm window
    m_t = {s: v for s, v in m_t.items() if s <= N}
    f_t = {s: v for s, v in f_t.items() if s <= N}
    m_val = {s: v for s, v in m_val.items() if s <= N}
    f_val = {s: v for s, v in f_val.items() if s <= N}
    m_ema, f_ema = ema(m_t), ema(f_t)

    arms = {}
    for a in ARMS:
        t, tok, val = parse_log(out_dir / f"{a}.log")
        arms[a] = {"t": t, "tok": tok, "val": val, "ema": ema(t),
                   "med": median_toks(tok, kwarm + 100),
                   "last": (max(t) if t else 0)}

    # ---------- CSVs ----------
    val_steps = sorted(set(m_val) | set().union(*[set(arms[a]["val"]) for a in ARMS]))
    with open(out_dir / "val_curves.csv", "w") as fh:
        fh.write("step,muon,fused," + ",".join(ARMS) + "\n")
        for s in val_steps:
            row = [str(s), f"{m_val.get(s,'')}", f"{f_val.get(s,'')}"]
            row += [f"{arms[a]['val'].get(s,'')}" for a in ARMS]
            fh.write(",".join(row) + "\n")
    loss_steps = sorted(set(m_ema) | set().union(*[set(arms[a]["ema"]) for a in ARMS]))
    with open(out_dir / "loss_curves.csv", "w") as fh:
        fh.write("step,muon_ema,fused_ema," + ",".join(f"{a}_ema" for a in ARMS) + "\n")
        for s in loss_steps:
            row = [str(s), f"{m_ema.get(s,''):.5f}" if s in m_ema else "",
                   f"{f_ema.get(s,''):.5f}" if s in f_ema else ""]
            row += [f"{arms[a]['ema'].get(s):.5f}" if s in arms[a]['ema'] else ""
                    for a in ARMS]
            fh.write(",".join(row) + "\n")

    # ---------- summary stats ----------
    def val_gap_stats(arm_val, ref_val):
        """ref - arm at common eval steps. Positive ppl gap => arm better.
        ln-gap = mean(ln(ref)-ln(arm)) = mean VAL-CE improvement (scale-stable).
        mid = mean ppl gap over steps >= N*0.66 (stable single-digit region)."""
        common = sorted(set(arm_val) & set(ref_val))
        if not common:
            return None
        ppl = [ref_val[s] - arm_val[s] for s in common]
        ln = [math.log(ref_val[s]) - math.log(arm_val[s]) for s in common]
        mid_steps = [s for s in common if s >= 0.66 * N]
        mid = ([ref_val[s] - arm_val[s] for s in mid_steps]) if mid_steps else []
        return {
            "common": common,
            "mean_ppl": statistics.mean(ppl),
            "mean_ln": statistics.mean(ln),
            "mid_mean_ppl": statistics.mean(mid) if mid else float("nan"),
            "mid_pct": (statistics.mean(
                [(ref_val[s]-arm_val[s])/ref_val[s] for s in mid_steps])*100
                if mid_steps else float("nan")),
            "final_step": common[-1],
            "final_ppl_gap": ref_val[common[-1]] - arm_val[common[-1]],
        }

    def ema_gap(arm_ema, ref_ema):
        common = sorted(set(arm_ema) & set(ref_ema))
        if not common:
            return None
        last = common[-1]
        return {"final_step": last, "final": ref_ema[last] - arm_ema[last],
                "mean": statistics.mean(ref_ema[s]-arm_ema[s] for s in common)}

    # reference: per-head-NS (fused) vs muon over the SAME window
    fused_vs_muon = val_gap_stats(f_val, m_val)
    fused_vs_muon_ema = ema_gap(f_ema, m_ema)

    # ---------- markdown ----------
    md = []
    md.append("# Better embedding optimizer vs shared-LR Adam — 287M DeltaNet "
              "pretrain A/B\n")
    md.append(f"**Config (every arm + both baselines identical):** real "
              f"production trunk 10L × d896 × 14h, FiLM(0,5;1,6;2,7;3,8;4,9) K=3 "
              f"(warmup {kwarm}), `--bf16 --tf32 --bf16_optim_state "
              f"--activation_checkpointing --no-compile`, data "
              f"`configs/pretrain_mix_v4.yaml`, T={T}, batch={B}, "
              f"grad_accum={GA} (= {tok_per_step:,} tok/step), WSD "
              f"lr=1.4e-3 / lr_muon=5e-3, **Muon matrix optimizer in EVERY arm**, "
              f"seed=0. Embedding arms run {N} steps at constant-peak LR "
              f"(decay_frac 0) = exactly the baseline's first {N} steps "
              f"(its decay starts at 2125), so matched-step comparison is "
              f"apples-to-apples. The ONLY thing varied across arms is the "
              f"embedding/lm_head treatment.\n")
    md.append("Baselines reused (NOT re-run): `runs/precond_ab/muon.log` "
              "(shared-LR AdamW embeddings — the thing we try to beat) and "
              "`runs/precond_ab/fused.log` (per-head Newton-Schulz matrix "
              "optimizer — the ~3% reference win), both seed-0, identical "
              "config.\n")

    # throughput
    md.append("## Throughput (median tok/s, steady-state)\n")
    md.append("| arm | median tok/s |")
    md.append("|---|---:|")
    md.append(f"| muon (baseline) | {median_toks(m_tok, kwarm+100):.0f} |")
    md.append(f"| fused (per-head NS) | {median_toks(f_tok, kwarm+100):.0f} |")
    for a in ARMS:
        md.append(f"| {ARM_LABEL[a]} | "
                  + (f"{arms[a]['med']:.0f}" if arms[a]['med'] else "n/a") + " |")
    md.append("")

    # VAL ppl matched-step table
    md.append("## VAL ppl at matched steps\n")
    hdr = "| step | muon | fused | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " |"
    md.append(hdr)
    md.append("|---:" * (3 + len(ARMS)) + "|")
    for s in val_steps:
        cells = [f"{m_val.get(s, float('nan')):.2f}" if s in m_val else "—",
                 f"{f_val.get(s, float('nan')):.2f}" if s in f_val else "—"]
        cells += [f"{arms[a]['val'][s]:.2f}" if s in arms[a]['val'] else "—"
                  for a in ARMS]
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md.append("")

    # gap-vs-muon summary
    md.append("## Each embedding arm vs the Muon baseline (positive ⇒ arm is "
              "BETTER)\n")
    md.append("`mean ppl gap` = mean over matched eval steps of "
              "(muon_ppl − arm_ppl); dominated by the steep early region. "
              "`mean VAL-CE gap` = mean(ln muon_ppl − ln arm_ppl) "
              "(scale-stable, the cleaner aggregate). `mid` = the stable "
              f"region (steps ≥ {int(0.66*N)}): mean ppl gap and mean %.\n")
    md.append("| arm | mean ppl gap | mean VAL-CE gap | mid ppl gap | mid % | "
              "final-step ppl gap |")
    md.append("|---|---:|---:|---:|---:|---:|")
    arm_stats = {}
    for a in ARMS:
        st = val_gap_stats(arms[a]["val"], m_val)
        arm_stats[a] = st
        if st is None:
            md.append(f"| {ARM_LABEL[a]} | n/a (no overlap / arm failed) | | | | |")
            continue
        md.append(f"| {ARM_LABEL[a]} | {st['mean_ppl']:+.3f} | "
                  f"{st['mean_ln']:+.4f} | {st['mid_mean_ppl']:+.3f} | "
                  f"{st['mid_pct']:+.2f}% | {st['final_ppl_gap']:+.3f} "
                  f"(step {st['final_step']}) |")
    # reference row: per-head NS vs muon on the same window
    if fused_vs_muon:
        md.append(f"| _ref: per-head NS (fused)_ | {fused_vs_muon['mean_ppl']:+.3f} "
                  f"| {fused_vs_muon['mean_ln']:+.4f} | "
                  f"{fused_vs_muon['mid_mean_ppl']:+.3f} | "
                  f"{fused_vs_muon['mid_pct']:+.2f}% | "
                  f"{fused_vs_muon['final_ppl_gap']:+.3f} "
                  f"(step {fused_vs_muon['final_step']}) |")
    md.append("")

    # train-CE EMA summary
    md.append("## Train-CE EMA (α=0.1) gap vs Muon baseline (positive ⇒ arm "
              "lower CE)\n")
    md.append("| arm | mean EMA gap | final-step EMA gap |")
    md.append("|---|---:|---:|")
    for a in ARMS:
        g = ema_gap(arms[a]["ema"], m_ema)
        if g is None:
            md.append(f"| {ARM_LABEL[a]} | n/a | n/a |")
            continue
        md.append(f"| {ARM_LABEL[a]} | {g['mean']:+.4f} | "
                  f"{g['final']:+.4f} (step {g['final_step']}) |")
    if fused_vs_muon_ema:
        md.append(f"| _ref: per-head NS (fused)_ | {fused_vs_muon_ema['mean']:+.4f} "
                  f"| {fused_vs_muon_ema['final']:+.4f} "
                  f"(step {fused_vs_muon_ema['final_step']}) |")
    md.append("")

    # verdict scaffold (numbers are the source of truth; prose filled by author)
    md.append("## Verdict\n")
    best = None
    for a in ARMS:
        st = arm_stats.get(a)
        if st is None:
            continue
        key = st["mean_ln"]
        if best is None or key > best[1]:
            best = (a, key)
    if best is not None:
        md.append(f"- Best embedding arm by mean VAL-CE gap: **{ARM_LABEL[best[0]]}** "
                  f"(mean VAL-CE gap {best[1]:+.4f} vs muon).")
    if fused_vs_muon:
        md.append(f"- Per-head-NS reference win over the SAME window: mean VAL-CE "
                  f"gap {fused_vs_muon['mean_ln']:+.4f}, mid {fused_vs_muon['mid_pct']:+.2f}%.")
    md.append("- Compare the best embedding arm's gap to the per-head-NS row: "
              "bigger ⇒ embedding optimizer is the larger lever; comparable ⇒ "
              "similar magnitude; (a combined arm would be needed to test "
              "additivity).")
    md.append("- HONESTY: single seed. The comparison is paired (shared "
              "init/data/schedule in-window) so the matched-step offset removes "
              "the bulk of run-to-run variance, but a gap smaller than the "
              "step-to-step wiggle in the VAL table is WITHIN noise — call it "
              "explicitly when it is.\n")

    pathlib.Path("EMBED_OPTIMIZER_AB.md").write_text("\n".join(md))
    print("Wrote EMBED_OPTIMIZER_AB.md + val_curves.csv + loss_curves.csv")
    for a in ARMS:
        st = arm_stats.get(a)
        if st:
            print(f"  {a}: mean VAL-CE gap vs muon={st['mean_ln']:+.4f}, "
                  f"mid%={st['mid_pct']:+.2f}, last_step={arms[a]['last']}")
        else:
            print(f"  {a}: NO DATA (failed?) last_step={arms[a]['last']}")
    if fused_vs_muon:
        print(f"  [ref] fused vs muon: mean VAL-CE gap={fused_vs_muon['mean_ln']:+.4f}, "
              f"mid%={fused_vs_muon['mid_pct']:+.2f}")


if __name__ == "__main__":
    main()
