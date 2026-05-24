"""Aggregate long-context recall sweep logs.

Reads `runs/longctx_recall/T*_*.log` and prints a recall vs T table
broken down by architecture.
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import defaultdict


FNAME_RE = re.compile(r"T(?P<T>\d+)_(?P<arch>.+)\.log$")
RESULT_RE = re.compile(
    r"^\s*\S+\s+\d+\s+\d+\s+(?P<recall>\d+\.\d+)\s+(?P<loss>\d+\.\d+)"
)


def parse_log(path: pathlib.Path) -> tuple[float, float] | None:
    try:
        for line in path.read_text().splitlines():
            m = RESULT_RE.match(line)
            if m:
                return float(m["recall"]), float(m["loss"])
    except FileNotFoundError:
        return None
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate_longctx_recall.py <log_dir>")
        return 1
    log_dir = pathlib.Path(sys.argv[1])
    by_T: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    for p in sorted(log_dir.glob("T*_*.log")):
        m = FNAME_RE.search(p.name)
        if not m:
            continue
        T = int(m["T"])
        arch = m["arch"]
        parsed = parse_log(p)
        if parsed is not None:
            by_T[T][arch] = parsed

    archs = sorted({a for cell in by_T.values() for a in cell})
    print(f"{'T':>6}  " + "  ".join(f"{a:>20}" for a in archs))
    print("-" * (6 + 2 + 22 * len(archs)))
    for T in sorted(by_T):
        cells = []
        for a in archs:
            if a in by_T[T]:
                rec, loss = by_T[T][a]
                cells.append(f"{rec:>13.3f} ({loss:>4.2f})")
            else:
                cells.append(f"{'—':>20}")
        print(f"{T:>6}  " + "  ".join(cells))
    print()
    print("(recall@1, val_loss in parentheses; '—' = config not run / OOM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
