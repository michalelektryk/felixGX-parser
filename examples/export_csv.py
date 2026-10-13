#!/usr/bin/env python3
"""Export every curve of a .gxz file as CSV files.

    python examples/export_csv.py --in pomiary.gxz [--out measurements]

Layout of the output folder:

    <out>/
      metadata.csv                     file-level information
      decay/<measurement>/data.csv     raw curve
                       /corrected.csv  real-time corrected curve
                       /RCQCSignal.csv reference (ROC) signal
                       /metadata.csv   acquisition settings
      excitation/...   emission/...    (one folder per measurement)

A curve that is not in the file is simply not written.  --out defaults to the
input file's name (without extension) in the current directory.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import felixgx
from felixgx import FelixFile, Measurement, Quantity, Trace

# curve file names, per role
FILES = {"raw": "data.csv", "corrected": "corrected.csv", "roc": "RCQCSignal.csv"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def snake(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", s.strip()).strip("_").lower() or "unnamed"


def sanitize(name: str) -> str:
    """A measurement name that is safe to use as a folder name."""
    name = re.sub(r'[\\/:*?"<>|]+', "_", name).strip(" .")
    return name[:120] or "unnamed"


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def fmt(value: Any) -> str:
    """One metadata value -> a single CSV cell."""
    if isinstance(value, Quantity):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, Quantity) for v in value):     # a scan range
            return " -> ".join(str(v) for v in value)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def csv_text(rows: List[List[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


def trace_rows(trace: Trace) -> List[List[Any]]:
    def col(label: str, unit: str, axis: str) -> str:
        name = snake(label.split("(")[0]) if label else axis
        return f"{name}_{snake(unit)}" if unit else name

    header = [col(trace.x_label, trace.x_unit, "x"), col(trace.y_label, trace.y_unit, "y")]
    # repr() keeps full double precision and writes NaN as "nan"
    return [header, *[[repr(float(x)), repr(float(y))] for x, y in zip(trace.x, trace.y)]]


# --------------------------------------------------------------------------- #
# Metadata files
# --------------------------------------------------------------------------- #
def file_metadata_rows(f: FelixFile) -> List[List[Any]]:
    m = f.metadata
    rows: List[List[Any]] = [
        ["key", "value"],
        ["path", f.path or ""],
        ["format_version", m.get("format_version", "")],
        ["program_version", m.get("program_version", "")],
        ["converter_version", m.get("converter_version", "")],
        ["original_filename", m.get("original_filename", "")],
        ["decays", len(f.decays)],
        ["excitations", len(f.excitations)],
        ["emissions", len(f.emissions)],
        ["other", len(f.other)],
        ["unassigned_traces", len(f.unassigned_traces)],
        ["warnings", json.dumps(f.warnings, ensure_ascii=False)],
        ["sessions", json.dumps(m.get("sessions", []), ensure_ascii=False)],
    ]
    return rows


def measurement_metadata_rows(m: Measurement) -> List[List[Any]]:
    rows: List[List[Any]] = [
        ["key", "value"],
        ["name", m.name],
        ["kind", m.kind.value],
        ["index", m.index],
        ["type", m.type_name],
        ["session", m.session_name],
        ["detector", m.detector],
        ["executed", fmt(m.executed)],
        ["raw", "yes" if m.raw else "no"],
        ["corrected", "yes" if m.corrected else "no"],
        ["roc", "yes" if m.roc else "no"],
    ]
    skip = {"properties", "executed", "session", "detector", "traces"}
    for key, val in sorted(flatten(m.metadata).items()):
        if key in skip or key.startswith("properties."):
            continue
        rows.append([key, fmt(val)])
    for key, val in sorted(m.properties.items()):   # acq.* instrument values
        rows.append([f"properties.{key}", fmt(val)])
    return rows


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def unique_folder(base: Path, name: str, used: Dict[str, int]) -> Path:
    """``base/<name>``, disambiguated with " (2)", " (3)", ... on collisions."""
    stem = sanitize(name)
    n = used.get(stem, 0) + 1
    used[stem] = n
    return base / (stem if n == 1 else f"{stem} ({n})")


def export(f: FelixFile, out: Path) -> int:
    """Write the CSV tree; returns the number of files written."""
    files: Dict[Path, str] = {Path("metadata.csv"): csv_text(file_metadata_rows(f))}

    groups = {"decay": f.decays, "excitation": f.excitations, "emission": f.emissions}
    if f.other:
        groups["other"] = f.other                     # never drop data

    for kind, measurements in groups.items():
        used: Dict[str, int] = {}
        for m in measurements:
            folder = unique_folder(Path(kind), m.name, used)
            for role, trace in (("raw", m.raw), ("corrected", m.corrected), ("roc", m.roc)):
                if trace is not None:
                    files[folder / FILES[role]] = csv_text(trace_rows(trace))
            files[folder / "metadata.csv"] = csv_text(measurement_metadata_rows(m))

    for rel, text in files.items():                   # write lazily: folders that
        path = out / rel                              # got no curve are not created
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return len(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="input", required=True, metavar="FILE",
                    help="input .gxz (or plain XML) Felix GX file")
    ap.add_argument("--out", metavar="DIR",
                    help="output folder (default: input file name without extension, in the current directory)")
    args = ap.parse_args(argv)

    try:
        f = felixgx.load(args.input)
    except felixgx.FelixParseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else Path.cwd() / Path(args.input).stem
    n = export(f, out)
    for w in f.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"{len(f)} measurement(s) -> {n} file(s) in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
