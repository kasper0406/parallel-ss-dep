"""Aggregate MQAR boundary-sweep logs into a markdown table.

Each input log file is named like `T256_K64_memon.log` / `..._memoff.log`
and ends with a line of the form:

    deltanet       256    64    0.996    0.0116    186,689    60.0

We parse that line and produce a side-by-side comparison.
"""
from __future__ import annotations

import re
import sys
import pathlib
from collections import defaultdict


FNAME_RE = re.compile(r"T(?P<T>\d+)_K(?P<K>\d+)_mem(?P<mem>on|off)\.log$")
RESULT_RE = re.compile(
    r"^\s*deltanet\s+(?P<T>\d+)\s+(?P<K>\d+)\s+"
    r"(?P<recall>\d+\.\d+)\s+(?P<loss>\d+\.\d+)"
)


def parse_log(path: pathlib.Path) -> tuple[float, float] | None:
    """Return (recall, loss) from the bottom-of-log results table."""
    try:
        with open(path) as fh:
            for line in fh:
                m = RESULT_RE.match(line)
                if m:
                    return float(m["recall"]), float(m["loss"])
    except FileNotFoundError:
        return None
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate_mqar_sweep.py <log_dir>")
        return 1
    log_dir = pathlib.Path(sys.argv[1])
    if not log_dir.is_dir():
        print(f"no such dir: {log_dir}")
        return 1

    by_TK: dict[tuple[int, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    for log_path in sorted(log_dir.glob("T*_K*_mem*.log")):
        m = FNAME_RE.search(log_path.name)
        if not m:
            continue
        T = int(m["T"])
        K = int(m["K"])
        mem = m["mem"]
        parsed = parse_log(log_path)
        if parsed is None:
            continue
        by_TK[(T, K)][mem] = parsed

    print(f"{'T':>4}  {'K':>4}  "
          f"{'recall_off':>10}  {'recall_on':>9}  {'Δrecall':>8}  "
          f"{'loss_off':>9}  {'loss_on':>8}  verdict")
    print("-" * 78)
    for (T, K) in sorted(by_TK):
        cell = by_TK[(T, K)]
        if "off" not in cell or "on" not in cell:
            print(f"{T:>4}  {K:>4}    (incomplete: have {list(cell)})")
            continue
        roff, loff = cell["off"]
        ron, lon = cell["on"]
        delta = ron - roff
        if delta >= 0.02:
            verdict = "MEM helps"
        elif delta <= -0.02:
            verdict = "MEM hurts"
        else:
            verdict = "MEM neutral"
        print(f"{T:>4}  {K:>4}  {roff:>10.3f}  {ron:>9.3f}  {delta:>+8.3f}  "
              f"{loff:>9.4f}  {lon:>8.4f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
