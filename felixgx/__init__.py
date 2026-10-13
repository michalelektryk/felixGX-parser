"""felixgx - read Felix GX (``.gxz``) fluorescence spectrometer files.

>>> import felixgx
>>> f = felixgx.load("pomiary.gxz")
>>> f.emissions[0].corrected.x, f.emissions[0].corrected.y   # doctest: +SKIP
"""
from .models import (
    DecayMeasurement,
    EmissionMeasurement,
    ExcitationMeasurement,
    FelixFile,
    Measurement,
    MeasurementKind,
    Quantity,
    Trace,
)
from .parser import FelixParseError, load, loads

__version__ = "0.2.0"

__all__ = [
    "load",
    "loads",
    "FelixFile",
    "Measurement",
    "DecayMeasurement",
    "EmissionMeasurement",
    "ExcitationMeasurement",
    "MeasurementKind",
    "Trace",
    "Quantity",
    "FelixParseError",
    "__version__",
]
