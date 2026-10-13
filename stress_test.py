#!/usr/bin/env python3
"""Stress test: load every .gxz in the date-named measurement folders and sanity-check.

Usage:  python stress_test.py [root_dir]
Exit code is non-zero if any file fails to load or a sanity check trips.
"""
from __future__ import annotations

import re
import sys
import warnings as _warnings
from pathlib import Path

import numpy as np

import felixgx

# folders like 17-11-2025, 30_10_2026, 28-04-2026
DATE_DIR = re.compile(r"^\d{2}[-_]\d{2}[-_]\d{4}$")


def check(f: felixgx.FelixFile) -> list[str]:
    problems = []
    for m in f.measurements:
        if not m.traces:
            problems.append(f"{m.name!r}: measurement with no traces")
        for t in m.traces:
            if t.x.shape != t.y.shape:
                problems.append(f"{m.name!r}/{t.name!r}: x/y length mismatch "
                                f"{t.x.shape} vs {t.y.shape}")
            if len(t) and not np.any(np.isfinite(t.y)):
                problems.append(f"{m.name!r}/{t.name!r}: no finite y values")
        # a raw/corrected pair must share the same x axis
        if m.raw is not None and m.corrected is not None and len(m.raw) == len(m.corrected):
            if not np.array_equal(m.raw.x, m.corrected.x):
                problems.append(f"{m.name!r}: raw/corrected x axes differ")
    return problems


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    files = sorted(p for d in root.iterdir()
                   if d.is_dir() and DATE_DIR.match(d.name)
                   for p in d.rglob("*.gxz"))
    if not files:
        print(f"No .gxz files under date folders in {root}")
        return 1

    n_fail = n_warn = n_meas = 0
    for p in files:
        rel = p.relative_to(root)
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("error")   # numpy runtime warnings -> failure
                f = felixgx.load(p)
                problems = check(f)
        except Exception as e:
            n_fail += 1
            print(f"FAIL  {rel}: {type(e).__name__}: {e}")
            continue
        n_meas += len(f)
        n_warn += len(f.warnings)
        kinds = (len(f.decays), len(f.excitations), len(f.emissions), len(f.other))
        status = "ok  " if not problems and not f.warnings else "warn"
        if problems:
            n_fail += 1
            status = "BAD "
        print(f"{status} {rel}: {len(f)} measurements "
              f"(dec={kinds[0]}, exc={kinds[1]}, emi={kinds[2]}, other={kinds[3]}), "
              f"{len(f.warnings)} warning(s), {len(f.unassigned_traces)} unassigned trace(s)")
        for w in f.warnings:
            print(f"       warning: {w}")
        for pr in problems:
            print(f"       PROBLEM: {pr}")

    print(f"\n{len(files)} file(s), {n_meas} measurement(s), "
          f"{n_fail} failure(s), {n_warn} warning(s) total")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
