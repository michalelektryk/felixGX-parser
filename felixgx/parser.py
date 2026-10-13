"""Reading ``.gxz`` files (gzipped Felix GX session XML)."""
from __future__ import annotations

import gzip
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ._description import parse_description, snake
from .models import (
    DecayMeasurement,
    EmissionMeasurement,
    ExcitationMeasurement,
    FelixFile,
    Measurement,
    MeasurementKind,
    Trace,
)

__all__ = ["load", "loads", "FelixParseError", "DEFAULT_TIME_TOLERANCE"]

log = logging.getLogger("felixgx")

#: Format versions this library was developed against.  Other versions are
#: still parsed, but a warning is recorded.
KNOWN_FORMAT_VERSIONS = {"3.5.1.0"}

_GZIP_MAGIC = b"\x1f\x8b"

# Trace tags that are handled explicitly; every *other* leaf tag lands in Trace.extra.
_TRACE_KNOWN_TAGS = {
    "Name", "Id", "XLabel", "YLabel", "TimeExecuted", "datapoints", "properties",
    "eventmarkers", "Acquisition_Description", "Axes_Description",
    "XAxis_Description", "YAxis_Description", "ZAxis_Description",
}
_FILE_TAG_NAMES = {
    "FelixSessionDataVersion": "format_version",
    "ProgramVersion": "program_version",
    "ConverterVersion": "converter_version",
    "OriginalFilename": "original_filename",
}
_REFERENCE_GROUP = re.compile(r"rcqc|\broc\b|reference", re.IGNORECASE)
_COR_NAME = re.compile(r"(\[COR\]|\(COR\)|\bCOR\b|\bcorrected\b)", re.IGNORECASE)


class FelixParseError(ValueError):
    """The file is not a readable Felix GX session file."""


# --------------------------------------------------------------------------- #
# Small XML / value helpers
# --------------------------------------------------------------------------- #
def _strip_namespaces(root: ET.Element) -> None:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.rsplit("}", 1)[1]


def _text(el: Optional[ET.Element], tag: str, default: str = "") -> str:
    child = el.find(tag) if el is not None else None
    return (child.text or "").strip() if child is not None and child.text else default


def _parse_dt(s: str) -> Optional[datetime]:
    """ISO-8601 with up to 7 fractional digits (Python wants <= 6)."""
    if not s:
        return None
    s = re.sub(r"(\.\d{6})\d+", r"\1", s.strip())
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _split_label(label: str) -> Tuple[str, str]:
    """``"Wavelength (Nanometers)"`` -> ``("Wavelength", "Nanometers")``."""
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", label)
    return (m.group(1), m.group(2)) if m else (label, "")


def _to_float(s: Optional[str]) -> float:
    try:
        return float(s)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


_PROP_CONVERTERS = {
    "double": float, "single": float, "float": float, "decimal": float,
    "int16": int, "int32": int, "int64": int, "int": int, "byte": int, "long": int,
    "boolean": lambda s: s.strip().lower() == "true",
    "datetime": lambda s: _parse_dt(s) or s,
}


def _parse_properties(el: Optional[ET.Element]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    if el is None:
        return props
    for child in el:
        name = child.get("Name") or child.tag
        text = (child.text or "").strip()
        conv = _PROP_CONVERTERS.get(child.tag.lower())
        try:
            props[name] = conv(text) if conv else text
        except (ValueError, TypeError):
            props[name] = text
    return props


# --------------------------------------------------------------------------- #
# Internal record: a trace plus what we need for grouping
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class _Rec:
    trace: Trace
    description: str
    time_key: str                 # raw TimeExecuted text (100 ns precision)
    ts: Optional[float]           # parsed execution time (seconds), None if unknown
    session_idx: int
    session_name: str
    session_id: str
    order: int                    # position in the file
    type_name: str = ""
    used: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


#: Traces saved a few ms apart belong to the same acquisition: Felix stamps
#: each curve (raw / corrected / reference) at its own save time, so the
#: timestamps of one acquisition are only equal up to a few ms.  Separate
#: acquisitions are never closer than seconds, so this window is safe.
DEFAULT_TIME_TOLERANCE = 1.0  # seconds


class _Reader:
    def __init__(self, strict: bool, time_tolerance: float = DEFAULT_TIME_TOLERANCE) -> None:
        self.strict = strict
        self.time_tolerance = max(0.0, float(time_tolerance))
        self.warnings: List[str] = []

    def warn(self, msg: str) -> None:
        if self.strict:
            raise FelixParseError(msg)
        log.warning(msg)
        self.warnings.append(msg)

    # ---- top level --------------------------------------------------- #
    def read(self, data: bytes, source: Optional[str]) -> FelixFile:
        root = self._xml(data)
        meta = self._file_metadata(root)
        version = meta.get("format_version")
        if version not in KNOWN_FORMAT_VERSIONS:
            self.warn(f"Untested file format version {version!r} "
                      f"(developed against {sorted(KNOWN_FORMAT_VERSIONS)}); results may be incomplete.")

        sessions_el = root.find("sessions")
        if sessions_el is None:
            raise FelixParseError("No <sessions> element: not a Felix GX session file.")

        detector_recs: List[_Rec] = []
        ref_recs: List[_Rec] = []
        meta["sessions"] = []
        order = 0
        for s_idx, sess in enumerate(sessions_el):
            s_name, s_id = _text(sess, "Name"), _text(sess, "Id")
            s_info: Dict[str, Any] = {"name": s_name, "id": s_id, "detectors": [], "reference_groups": []}
            meta["sessions"].append(s_info)
            groups = sess.find("groups")
            for grp in (groups if groups is not None else []):
                g_name = _text(grp, "Name")
                is_ref = bool(_REFERENCE_GROUP.search(g_name))
                s_info["reference_groups" if is_ref else "detectors"].append(g_name)
                traces_el = grp.find("traces")
                for t_el in (traces_el if traces_el is not None else []):
                    rec = self._trace(t_el, g_name, s_idx, s_name, s_id, order)
                    order += 1
                    if rec is not None:
                        (ref_recs if is_ref else detector_recs).append(rec)

        if not meta["sessions"]:
            self.warn("The file contains no sessions.")
        elif not detector_recs and not ref_recs:
            self.warn("The file contains no traces.")

        result = FelixFile(path=source, metadata=meta)
        self._assemble(result, detector_recs, ref_recs)
        result.warnings = self.warnings
        return result

    # ---- XML --------------------------------------------------------- #
    def _xml(self, data: bytes) -> ET.Element:
        if data[:2] == _GZIP_MAGIC:
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError) as e:
                raise FelixParseError(f"Corrupt gzip data: {e}") from e
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            raise FelixParseError(f"Not valid XML: {e}") from e
        _strip_namespaces(root)
        if root.tag != "FelixSessionData":
            raise FelixParseError(f"Unexpected root element <{root.tag}>; not a Felix GX session file.")
        return root

    def _file_metadata(self, root: ET.Element) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        for child in root:
            if child.tag == "sessions":
                continue
            if len(child) == 0:  # simple leaf: keep it, even if unknown to us
                meta[_FILE_TAG_NAMES.get(child.tag, snake(child.tag))] = (child.text or "").strip()
        return meta

    # ---- one trace --------------------------------------------------- #
    def _trace(self, el: ET.Element, group: str, s_idx: int, s_name: str, s_id: str,
               order: int) -> Optional[_Rec]:
        name = _text(el, "Name")
        try:
            xs, ys, skipped = [], [], 0
            dps = el.find("datapoints")
            for pt in (dps if dps is not None else []):
                x_el, y_el = pt.find("X"), pt.find("Y")
                if x_el is None or not (x_el.text or "").strip():
                    skipped += 1
                    continue
                xs.append(_to_float(x_el.text))
                ys.append(_to_float(y_el.text if y_el is not None else None))
        except Exception as e:  # defensive: one bad trace must not kill the file
            self.warn(f"Trace {name!r} could not be read ({e!r}); skipped.")
            return None
        if skipped:
            self.warn(f"Trace {name!r}: skipped {skipped} data point(s) without an X value.")

        x_label, y_label = _text(el, "XLabel"), _text(el, "YLabel")
        props = _parse_properties(el.find("properties"))
        time_text = _text(el, "TimeExecuted")
        executed = _parse_dt(time_text) or props.get("acq.DateTime")
        if not isinstance(executed, datetime):
            executed = None
        extra = {c.tag: (c.text or "").strip() for c in el
                 if c.tag not in _TRACE_KNOWN_TAGS and len(c) == 0}

        trace = Trace(
            name=name, id=_text(el, "Id"),
            x=np.asarray(xs, dtype=float), y=np.asarray(ys, dtype=float),
            x_label=x_label, y_label=y_label,
            x_unit=_split_label(x_label)[1], y_unit=_split_label(y_label)[1],
            executed=executed, group=group, properties=props, extra=extra,
        )
        # Pairing key: prefer the exact text; fall back to the acq.DateTime property.
        key = time_text or (str(props.get("acq.DateTime")) if "acq.DateTime" in props else "")
        ts = executed.timestamp() if executed is not None else None
        description = _text(el, "Acquisition_Description")
        parsed = parse_description(description)
        return _Rec(trace=trace, description=description, time_key=key, ts=ts,
                    session_idx=s_idx, session_name=s_name, session_id=s_id, order=order,
                    type_name=str(parsed.get("type", "")), meta=parsed)

    # ---- grouping ---------------------------------------------------- #
    def _assemble(self, result: FelixFile, detector_recs: List[_Rec], ref_recs: List[_Rec]) -> None:
        # Detector traces of one acquisition: same session, detector and type,
        # with execution times within `time_tolerance` of each other.  Felix
        # stamps every saved curve at its own save time, so the raw / corrected
        # curves of one acquisition differ by a few ms.
        buckets: Dict[Tuple, List[_Rec]] = defaultdict(list)
        for r in detector_recs:
            buckets[(r.session_idx, r.trace.group, r.type_name)].append(r)

        clusters: List[List[_Rec]] = []
        tol = self.time_tolerance
        for recs in buckets.values():
            timed = sorted((r for r in recs if r.ts is not None), key=lambda r: (r.ts, r.order))
            untimed = [r for r in recs if r.ts is None]
            cur: List[_Rec] = []
            for r in timed:
                if cur and (r.ts - cur[-1].ts) > tol:
                    clusters.append(cur)
                    cur = []
                cur.append(r)
            if cur:
                clusters.append(cur)
            # Without a timestamp only the exact text can pair traces up;
            # a trace with no time at all stays on its own.
            by_key: Dict[str, List[_Rec]] = defaultdict(list)
            for r in untimed:
                if r.time_key:
                    by_key[r.time_key].append(r)
                else:
                    clusters.append([r])
            clusters.extend(by_key.values())

        # Reference (ROC) signals of one acquisition: same session, timestamp
        # within `time_tolerance` of the acquisition's traces.
        timed_refs: Dict[int, List[_Rec]] = defaultdict(list)
        for r in ref_recs:
            if r.ts is not None:
                timed_refs[r.session_idx].append(r)
        for refs in timed_refs.values():
            refs.sort(key=lambda r: r.ts)

        measurements: List[Measurement] = []
        for recs in clusters:
            raw, cor, extras = self._assign_roles(recs)
            first = recs[0]
            roc, roc_extras = None, []
            refs = timed_refs.get(first.session_idx, [])
            if first.ts is not None and refs:
                t0 = min(r.ts for r in recs)
                t1 = max(r.ts for r in recs)
                candidates = [c for c in refs if t0 - tol <= c.ts <= t1 + tol]
                for c in candidates:
                    c.used = True
                if candidates:
                    candidates.sort(key=lambda c: min(abs(c.ts - r.ts) for r in recs))
                    roc, roc_extras = candidates[0], candidates[1:]
            all_extras = [*extras, *roc_extras]
            main = raw or cor or first
            kind = self._kind(main)
            cls = {MeasurementKind.DECAY: DecayMeasurement,
                   MeasurementKind.EMISSION: EmissionMeasurement,
                   MeasurementKind.EXCITATION: ExcitationMeasurement}.get(kind, Measurement)

            metadata = dict(main.meta)
            props: Dict[str, Any] = {}
            for r in (cor, raw):          # raw wins on conflicts
                if r is not None:
                    props.update(r.trace.properties)
            metadata["properties"] = props
            metadata["executed"] = main.trace.executed
            metadata["session"] = main.session_name
            metadata["detector"] = main.trace.group

            if not main.description:
                self.warn(f"Trace {main.trace.name!r} has no acquisition description; "
                          f"type inferred as {kind.value!r} from its data.")
            elif kind is MeasurementKind.OTHER:
                self.warn(f"Unsupported measurement type {main.type_name!r} "
                          f"(trace {main.trace.name!r}); stored in FelixFile.other.")
            if all_extras:
                self.warn(f"{main.trace.name!r}: {len(all_extras)} additional trace(s) with the same "
                          f"acquisition time could not be assigned; see Measurement.extra_traces.")

            measurements.append(cls(
                raw=raw.trace if raw else None,
                corrected=cor.trace if cor else None,
                roc=roc.trace if roc else None,
                metadata=metadata, type_name=main.type_name,
                session_name=main.session_name, session_id=main.session_id,
                detector=main.trace.group,
                extra_traces=[r.trace for r in all_extras],
                index=min(r.order for r in recs),
            ))

        measurements.sort(key=lambda m: m.index)
        for i, m in enumerate(measurements):
            m.index = i
            if isinstance(m, DecayMeasurement):
                result.decays.append(m)
            elif isinstance(m, EmissionMeasurement):
                result.emissions.append(m)
            elif isinstance(m, ExcitationMeasurement):
                result.excitations.append(m)
            else:
                result.other.append(m)

        result.unassigned_traces = [r.trace for r in ref_recs if not r.used]
        if result.unassigned_traces:
            log.info("%d reference trace(s) not matched to any measurement: %s",
                     len(result.unassigned_traces),
                     ", ".join(repr(t.name) for t in result.unassigned_traces))

    # ---- classification ---------------------------------------------- #
    @staticmethod
    def _corrected_score(r: _Rec) -> int:
        """Higher = more likely the corrected (derived) curve.

        The *raw* curve carries the instrument ``acq.*`` properties (excitation /
        emission wavelengths, ...); a derived curve only has ``acq.DateTime``.
        A ``COR`` marker in the name is a weaker hint (users can rename traces).
        """
        score = 0
        instrument = any(k.startswith("acq.") and k != "acq.DateTime" for k in r.trace.properties)
        score += -3 if instrument else 3
        if _COR_NAME.search(r.trace.name):
            score += 2
        return score

    @staticmethod
    def _expects_corrected(recs: List[_Rec]) -> Optional[bool]:
        """Does the acquisition's trace manifest list a derived (corrected) curve?

        ``None`` when the description has no (parsable) manifest.
        """
        for r in recs:
            manifest = r.meta.get("traces")
            if isinstance(manifest, list) and manifest:
                return any(e.get("kind") == "derived" for e in manifest)
        return None

    def _assign_roles(self, recs: List[_Rec]) -> Tuple[Optional[_Rec], Optional[_Rec], List[_Rec]]:
        """Pick the raw and corrected trace out of the traces of one acquisition.

        Returns ``(raw, corrected, leftovers)``.  Either of the first two may be None.
        """
        if self._expects_corrected(recs) is False:
            # Correction was not configured, so nothing here can be a corrected curve
            # (more than one trace means duplicates; keep them as leftovers).
            ordered = sorted(recs, key=lambda r: r.order)
            return ordered[0], None, ordered[1:]
        if len(recs) == 1:
            r = recs[0]
            return (None, r, []) if self._corrected_score(r) > 0 else (r, None, [])
        ranked = sorted(recs, key=lambda r: (self._corrected_score(r), r.order))
        return ranked[0], ranked[-1], ranked[1:-1]

    @staticmethod
    def _kind(r: _Rec) -> MeasurementKind:
        t = r.type_name.lower()
        if t:
            has_dec = "decay" in t or "tcspc" in t or "lifetime" in t
            has_exc, has_emi = "excitation" in t, "emission" in t
            if has_dec and not (has_exc or has_emi):
                return MeasurementKind.DECAY
            if has_exc and not has_emi and not has_dec:
                return MeasurementKind.EXCITATION
            if has_emi and not has_exc and not has_dec:
                return MeasurementKind.EMISSION
            return MeasurementKind.OTHER
        # No usable description: infer from the data.
        props = r.trace.properties
        if "nano" in r.trace.x_unit.lower() and "second" in r.trace.x_unit.lower():
            return MeasurementKind.DECAY
        if "acq.EmiStartWave" in props:
            return MeasurementKind.EMISSION
        if "acq.ExcStartWave" in props:
            return MeasurementKind.EXCITATION
        return MeasurementKind.OTHER


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def loads(data: bytes, *, strict: bool = False, source: Optional[str] = None,
          time_tolerance: float = DEFAULT_TIME_TOLERANCE) -> FelixFile:
    """Parse the *contents* of a ``.gxz`` file (gzipped or plain XML bytes).

    ``time_tolerance`` (seconds) is the window within which trace timestamps
    still count as the same acquisition; see :func:`load`.
    """
    return _Reader(strict, time_tolerance).read(data, source)


def load(path: Union[str, "os.PathLike[str]"], *, strict: bool = False,
         time_tolerance: float = DEFAULT_TIME_TOLERANCE) -> FelixFile:
    """Read a Felix GX ``.gxz`` file.

    Parameters
    ----------
    path:
        File to read.
    strict:
        If ``True`` any inconsistency raises :class:`FelixParseError`; by default
        problems are logged and collected in ``FelixFile.warnings``.
    time_tolerance:
        Traces whose execution times differ by no more than this many seconds
        (same session, detector and measurement type) are treated as one
        acquisition.  Felix stamps the raw, corrected and reference curves of
        one acquisition a few milliseconds apart, so an exact match would split
        them.  Real acquisitions are never this close together; set to ``0``
        for exact-timestamp matching.

    Returns
    -------
    FelixFile
        With ``decays``, ``excitations`` and ``emissions`` lists of measurements,
        each holding ``raw`` / ``corrected`` / ``roc`` traces (any may be ``None``).
    """
    path = os.fspath(path)
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        raise FelixParseError(f"Cannot read {path!r}: {e}") from e
    return loads(data, strict=strict, source=path, time_tolerance=time_tolerance)
