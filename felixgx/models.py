"""Data model returned by :func:`felixgx.load`."""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np

__all__ = [
    "Quantity",
    "MeasurementKind",
    "Trace",
    "Measurement",
    "DecayMeasurement",
    "ExcitationMeasurement",
    "EmissionMeasurement",
    "FelixFile",
]


class Quantity(NamedTuple):
    """A number with a unit, e.g. ``Quantity(5.0, 'nm')`` for ``"5 nm"``."""

    value: float
    unit: str

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}"


class MeasurementKind(str, enum.Enum):
    DECAY = "decay"
    EXCITATION = "excitation"
    EMISSION = "emission"
    OTHER = "other"


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Trace:
    """A single curve (x/y arrays plus everything the file says about it)."""

    name: str
    id: str
    x: np.ndarray
    y: np.ndarray
    x_label: str = ""          # as in the file, e.g. "Wavelength (Nanometers)"
    y_label: str = ""          # e.g. "Intensity (Counts/s)"
    x_unit: str = ""           # "Nanometers"
    y_unit: str = ""           # "Counts/s"
    executed: Optional[datetime] = None
    group: str = ""            # name of the group in the file, e.g. "Detector1"
    properties: Dict[str, Any] = field(default_factory=dict)  # typed <properties>
    extra: Dict[str, str] = field(default_factory=dict)       # any other leaf tags

    def __len__(self) -> int:
        return len(self.x)

    def as_array(self) -> np.ndarray:
        """Return the data as a ``(2, N)`` array: ``[x, y]``."""
        return np.vstack([self.x, self.y])

    def __repr__(self) -> str:
        return (f"Trace({self.name!r}, {len(self)} points, "
                f"x=[{self.x_unit}], y=[{self.y_unit}])")


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
_COR_SUFFIX = re.compile(r"[\s_-]*(\[COR\]|\(COR\)|COR)\s*$", re.IGNORECASE)


@dataclass(eq=False)
class Measurement:
    """One acquisition: the raw curve, the corrected curve and the ROC signal.

    Any of ``raw`` / ``corrected`` / ``roc`` may be ``None`` (e.g. real-time
    correction was disabled, or the reference signal was not saved).

    ``metadata`` is a nested dict parsed from the acquisition description
    (slits, integration time, correction settings, ...) plus the keys
    ``executed``, ``session``, ``detector`` and ``properties`` (the typed
    ``acq.*`` values stored next to the curve).
    """

    KIND: ClassVar[MeasurementKind] = MeasurementKind.OTHER

    raw: Optional[Trace] = None
    corrected: Optional[Trace] = None
    roc: Optional[Trace] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    type_name: str = ""                 # the file's own type string, e.g. "EmissionScan"
    session_name: str = ""
    session_id: str = ""
    detector: str = ""
    extra_traces: List[Trace] = field(default_factory=list)  # duplicates we could not place
    index: int = -1                     # position among all measurements in the file

    # ---- identity ---------------------------------------------------- #
    @property
    def kind(self) -> MeasurementKind:
        return self.KIND

    @property
    def name(self) -> str:
        """The user-visible name (the raw trace's name, without any ``COR`` suffix)."""
        if self.raw is not None:
            return self.raw.name
        if self.corrected is not None:
            return _COR_SUFFIX.sub("", self.corrected.name) or self.corrected.name
        if self.roc is not None:
            return self.roc.name
        return ""

    @property
    def executed(self) -> Optional[datetime]:
        for t in (self.raw, self.corrected, self.roc):
            if t is not None and t.executed is not None:
                return t.executed
        return None

    # ---- traces ------------------------------------------------------ #
    @property
    def traces(self) -> List[Trace]:
        """All traces that are present, in the order raw, corrected, roc."""
        return [t for t in (self.raw, self.corrected, self.roc) if t is not None]

    @property
    def best(self) -> Optional[Trace]:
        """The corrected curve if available, otherwise the raw one."""
        return self.corrected if self.corrected is not None else self.raw

    @property
    def properties(self) -> Dict[str, Any]:
        return self.metadata.get("properties", {})

    # ---- helpers for subclasses ------------------------------------- #
    def _prop(self, name: str) -> Optional[float]:
        """Numeric ``acq.*`` property; negative values mean 'not applicable'."""
        v = self.properties.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
            return float(v)
        return None

    def _meta_quantity(self, key: str) -> Optional[float]:
        """Value of a ``Quantity`` entry of the parsed description (nm assumed
        for wavelengths); used when the ``acq.*`` properties are missing."""
        v = self.metadata.get(key)
        if isinstance(v, Quantity) and v.value >= 0:
            return float(v.value)
        return None

    def _x_range(self) -> Optional[Tuple[float, float]]:
        t = self.best or self.roc
        if t is None or len(t) == 0:
            return None
        return float(np.nanmin(t.x)), float(np.nanmax(t.x))

    def __repr__(self) -> str:
        have = "".join(
            f"{tag}{'+' if t is not None else '-'} "
            for tag, t in (("raw", self.raw), ("cor", self.corrected), ("roc", self.roc))
        ).strip()
        return f"{type(self).__name__}({self.name!r}, {have}{self._repr_extra()})"

    def _repr_extra(self) -> str:
        return ""


@dataclass(eq=False, repr=False)
class DecayMeasurement(Measurement):
    """TCSPC fluorescence decay histogram (x = time, y = counts)."""

    KIND: ClassVar[MeasurementKind] = MeasurementKind.DECAY

    @property
    def excitation_wavelength(self) -> Optional[float]:
        """Excitation wavelength in nm, ``None`` if not stored (e.g. LED source)."""
        return self._prop("acq.ExcWave")

    @property
    def emission_wavelength(self) -> Optional[float]:
        """Detection wavelength in nm."""
        v = self._prop("acq.EmiWave")
        if v is None and self.raw is not None:
            try:
                v = float(self.raw.extra["EmissionWavelengthDASTRES"])
            except (KeyError, ValueError):
                v = None
        return v

    @property
    def time_range(self) -> Optional[Quantity]:
        v = self.metadata.get("time_range")
        return v if isinstance(v, Quantity) else None

    def _repr_extra(self) -> str:
        em = self.emission_wavelength
        return f", em={em:g} nm" if em is not None else ""


@dataclass(eq=False, repr=False)
class EmissionMeasurement(Measurement):
    """Emission scan (fixed excitation, x = emission wavelength)."""

    KIND: ClassVar[MeasurementKind] = MeasurementKind.EMISSION

    @property
    def excitation_wavelength(self) -> Optional[float]:
        v = self._prop("acq.ExcWave")
        if v is None:
            # converted/re-saved files lose the acq.* properties; the
            # acquisition description still states "Excitation: 490 nm"
            v = self._meta_quantity("excitation")
        return v

    @property
    def emission_range(self) -> Optional[Tuple[float, float]]:
        """(start, end) in nm; falls back to the extent of the data."""
        a, b = self._prop("acq.EmiStartWave"), self._prop("acq.EmiEndWave")
        return (a, b) if a is not None and b is not None else self._x_range()

    @property
    def step(self) -> Optional[float]:
        return self._prop("acq.WaveStepSize")

    def _repr_extra(self) -> str:
        parts = []
        if self.excitation_wavelength is not None:
            parts.append(f"ex={self.excitation_wavelength:g} nm")
        r = self.emission_range
        if r:
            parts.append(f"em={r[0]:g}-{r[1]:g} nm")
        return (", " + ", ".join(parts)) if parts else ""


@dataclass(eq=False, repr=False)
class ExcitationMeasurement(Measurement):
    """Excitation scan (fixed emission, x = excitation wavelength).

    Note: the ``acq.*`` property names used here are the obvious mirror of the
    emission-scan ones; if they are absent the range falls back to the data.
    """

    KIND: ClassVar[MeasurementKind] = MeasurementKind.EXCITATION

    @property
    def emission_wavelength(self) -> Optional[float]:
        v = self._prop("acq.EmiWave")
        if v is None:
            # description fallback ("Emission1: 590 nm"); a *range* here would
            # be an emission scan's setting, so only a plain Quantity matches
            v = self._meta_quantity("emission1")
        return v

    @property
    def excitation_range(self) -> Optional[Tuple[float, float]]:
        a, b = self._prop("acq.ExcStartWave"), self._prop("acq.ExcEndWave")
        if a is not None and b is not None:
            return (a, b)
        rng = self.metadata.get("excitation")     # description: "Excitation: 400 nm -> 510 nm"
        if isinstance(rng, tuple) and len(rng) == 2 and all(isinstance(v, Quantity) for v in rng):
            return (float(rng[0].value), float(rng[1].value))
        return self._x_range()

    @property
    def step(self) -> Optional[float]:
        return self._prop("acq.WaveStepSize")

    def _repr_extra(self) -> str:
        parts = []
        if self.emission_wavelength is not None:
            parts.append(f"em={self.emission_wavelength:g} nm")
        r = self.excitation_range
        if r:
            parts.append(f"ex={r[0]:g}-{r[1]:g} nm")
        return (", " + ", ".join(parts)) if parts else ""


# --------------------------------------------------------------------------- #
# Whole file
# --------------------------------------------------------------------------- #
def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, path))
        else:
            out[path] = v
    return out


# Per-measurement bookkeeping that is never "common" settings.
_NOT_SETTINGS = {"executed", "session", "detector", "acquisition_name", "created",
                 "type", "traces", "properties"}


@dataclass(eq=False)
class FelixFile:
    """Everything read from one ``.gxz`` file.

    * ``decays`` / ``excitations`` / ``emissions`` - lists of measurements,
      in the order they appear in the file.
    * ``other`` - measurements of a type this library does not know yet.
    * ``metadata`` - file-level information (versions, original filename,
      sessions).  Settings shared by *all* measurements are available through
      :attr:`common_metadata`.
    * ``unassigned_traces`` - traces that could not be attached to any
      measurement (typically ROC signals of aborted scans).
    * ``warnings`` - human-readable notes about anything odd in the file.
    """

    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    decays: List[DecayMeasurement] = field(default_factory=list)
    excitations: List[ExcitationMeasurement] = field(default_factory=list)
    emissions: List[EmissionMeasurement] = field(default_factory=list)
    other: List[Measurement] = field(default_factory=list)
    unassigned_traces: List[Trace] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # ---- access ------------------------------------------------------ #
    @property
    def measurements(self) -> List[Measurement]:
        """All measurements (every kind) in the order they appear in the file."""
        allm: List[Measurement] = [*self.decays, *self.excitations, *self.emissions, *self.other]
        return sorted(allm, key=lambda m: m.index)

    def __iter__(self) -> Iterator[Measurement]:
        return iter(self.measurements)

    def __len__(self) -> int:
        return len(self.decays) + len(self.excitations) + len(self.emissions) + len(self.other)

    def find(self, pattern: str, *, regex: bool = False) -> List[Measurement]:
        """Measurements whose name contains ``pattern`` (case-insensitive) or matches a regex."""
        if regex:
            rx = re.compile(pattern)
            return [m for m in self.measurements if rx.search(m.name)]
        p = pattern.lower()
        return [m for m in self.measurements if p in m.name.lower()]

    # ---- metadata ---------------------------------------------------- #
    @property
    def common_metadata(self) -> Dict[str, Any]:
        """Settings with the *same value in every measurement* (flattened, dotted keys).

        Handy for seeing what is constant in a session (titrator, preferences,
        slit widths, ...) without repeating it per measurement.
        """
        ms = self.measurements
        if not ms:
            return {}
        flats = [_flatten({k: v for k, v in m.metadata.items() if k not in _NOT_SETTINGS}) for m in ms]
        common = {}
        for key, val in flats[0].items():
            if all(key in f and f[key] == val for f in flats[1:]):
                common[key] = val
        return common

    def __repr__(self) -> str:
        return (f"FelixFile({self.path!r}: {len(self.decays)} decays, "
                f"{len(self.excitations)} excitation, {len(self.emissions)} emission"
                + (f", {len(self.other)} other" if self.other else "") + ")")

    def summary(self) -> str:
        """A readable multi-line overview of the file."""
        lines = [repr(self)]
        for m in self.measurements:
            when = m.executed.strftime("%Y-%m-%d %H:%M:%S") if m.executed else "?"
            lines.append(f"  {when}  {m!r}")
        if self.unassigned_traces:
            lines.append(f"  ({len(self.unassigned_traces)} unassigned trace(s))")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)
