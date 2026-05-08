"""
Aggregate Phase 22b ablation eval JSONs into a single table.

Usage:
    python experiments/aggregate_phase22b.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROW_ORDER = [
    ("Plain DN baseline", "bench_stmt_ppl_dn_baseline_v2.json"),
    ("K=3 self-feed (Phase 21c)", "bench_stmt_ppl_k3_v2.json"),
    ("K=3 + uniform L_sem β=1.0 (Phase 22, REF)",
     "bench_stmt_ppl_lsem_b10_uniform.json"),
    ("Ablation 1: random-frozen encoder",
     "bench_stmt_ppl_lsem_uniform_random_enc.json"),
    ("Ablation 2: per-token L_sem (no AST)",
     "bench_stmt_ppl_lsem_uniform_pertoken.json"),
    ("Ablation 3: KL-on-logits β=0.5",
     "bench_stmt_ppl_kl_b05.json"),
    ("Ablation 3: KL-on-logits β=1.0",
     "bench_stmt_ppl_kl_b10.json"),
    ("Ablation 4: past-K=3-ckpt encoder",
     "bench_stmt_ppl_lsem_uniform_pastckpt_enc.json"),
]


def main():
    base = pathlib.Path("/home/knielsen/ml/parallel-ss-dep")
    print(f"{'Variant':<48} {'PPL':>8} {'Top10':>8} {'Bot10':>8}")
    print("-" * 80)
    for label, fname in ROW_ORDER:
        path = base / fname
        if not path.exists():
            print(f"{label:<48} {'MISS':>8} {'MISS':>8} {'MISS':>8}")
            continue
        d = json.loads(path.read_text())
        s = d["summary"]
        ppl = s["overall"]["ppl"]
        top = s.get("top_decile", {}).get("ppl", float("nan"))
        bot = s.get("bottom_decile", {}).get("ppl", float("nan"))
        print(f"{label:<48} {ppl:>8.2f} {top:>8.2f} {bot:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
