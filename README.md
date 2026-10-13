# felixgx

Reader for Felix GX (`.gxz`) fluorescence-spectrometer session files
(gzipped XML). Only dependency: NumPy. Python ≥ 3.9.

```
pip install -e .          # from this directory
pytest                    # synthetic + real-file tests (real ones run when the
                          # measurement folders are next to this directory)
python stress_test.py ..  # load every .gxz in the date-named folders and
                          # sanity-check the result
```

## Quick start

```python
import felixgx

f = felixgx.load("pomiary.gxz")
print(f.summary())

f.decays          # list[DecayMeasurement]       (TCSPC histograms)
f.emissions       # list[EmissionMeasurement]    (fixed excitation, scan emission)
f.excitations     # list[ExcitationMeasurement]  (fixed emission, scan excitation)
f.other           # measurements of a type the library doesn't know (never dropped)

m = f.emissions[0]
m.raw             # Trace | None   - not corrected
m.corrected       # Trace | None   - real-time corrected ("... COR")
m.roc             # Trace | None   - reference (RCQCSignal) signal
m.best            # corrected if present, else raw

m.corrected.x, m.corrected.y          # numpy arrays
m.corrected.x_unit, m.corrected.y_unit
m.name, m.executed, m.excitation_wavelength, m.emission_range, m.step
```

`raw`, `corrected` and `roc` are each `None` when that curve is not in the file
(correction disabled, reference not saved, curve deleted, ...). Nothing else
changes in that case.

## Exporting to CSV

```bash
python examples/export_csv.py --in pomiary.gxz [--out measurements]
```

Writes `decay/`, `excitation/` and `emission/` folders, one sub-folder per
measurement with `data.csv` (raw), `corrected.csv`, `RCQCSignal.csv` (only the
curves that exist) and a `metadata.csv`; a file-level `metadata.csv` goes to
the top. `--out` defaults to the input file name without extension in the
current directory.

## Metadata

| where | what |
|---|---|
| `f.metadata` | file level: `format_version`, `program_version`, `converter_version`, `original_filename`, `sessions` |
| `f.common_metadata` | settings with the *same value in every measurement* (dotted keys) |
| `m.metadata` | everything about one measurement: slits, integration, correction settings, trace manifest, comments, … plus `properties` (typed `acq.*` values), `executed`, `session`, `detector` |
| `Trace.properties`, `Trace.extra` | per-curve XML properties / any other tags |

Values are typed where possible: `bool`, `int`, `float`,
`Quantity(value, unit)` (e.g. `Quantity(5.0, 'nm')`), ranges as a pair of
`Quantity`, otherwise `str`. Example:

```python
m.metadata["slit_settings"]["emission"]["exit"]   # Quantity(1.0, 'nm')
m.metadata["integration"]                          # Quantity(1.0, 'sec')
m.metadata["real_time_correction_settings"]        # {'excitation_correction': 'Disabled', ...}
```

`created` is kept as the text Felix wrote (its format depends on the PC's
locale); use `executed` (ISO timestamp) for dates.

## How curves are matched (and why not by name)

Trace names are user-editable, so they are only a weak hint.

* **Grouping**: raw + corrected curves of one acquisition share session,
  detector and type, and execution timestamps within `time_tolerance`
  (default 1 s) — Felix stamps each saved curve a few milliseconds apart.
  The ROC signal lives in the separate `RCQCSignal` group and is matched on
  the same timestamps. Real acquisitions are never closer than seconds;
  pass `time_tolerance=0` to require exact timestamps.
* **Raw vs corrected**: the raw curve carries the instrument `acq.*`
  properties (wavelengths, step, …); a derived curve only has `acq.DateTime`.
  A `COR` marker in the name and the file order are tie-breakers. If the
  acquisition's trace manifest lists no derived curve, nothing is treated as
  corrected.
* **Type** comes from the `Type:` line of the acquisition description
  (`EmissionScan`, `ExcitationScan`, `TM_ibhTCSPC_Decay`); without a
  description it is inferred from the axes/properties.
* ROC traces matching no measurement (e.g. aborted scans) are kept in
  `f.unassigned_traces`. Duplicates that cannot be placed go to
  `Measurement.extra_traces`.

## Robustness

* Unknown/changed content never aborts loading: problems are logged
  (`logging.getLogger("felixgx")`) and collected in `f.warnings`.
  `felixgx.load(path, strict=True)` raises instead.
* Corrupt gzip / invalid XML / not a Felix file → `felixgx.FelixParseError`.
* Missing or non-numeric `Y` values become `NaN`; points without `X` are dropped
  (with a warning).
* Plain (un-gzipped) XML is accepted too; `felixgx.loads(bytes)` parses in-memory data.

## Caveat

Developed against format `3.5.1.0` / Felix GX `4.9.0.10321`. Emission scans,
TCSPC decays and **excitation scans** are verified against real measurement
files (emission and excitation spectra from 24-11-2025, 28-04-2026 and
31-01-2026 sessions). Excitation scans use the mirrored property names
(`acq.ExcStartWave`, `acq.ExcEndWave`, `acq.EmiWave`); when a file was
re-saved by Felix GX's converter and those properties were stripped, the
wavelengths fall back to the `Excitation:` / `Emission1:` lines of the
acquisition description.

Files merged/re-saved by Felix GX repeat the acquisition name after
converter header lines (`Original File:`, `FelixGX Version:`, …); this is
handled, but such traces keep only `acq.DateTime`, so raw/corrected
detection there relies on the `COR` name marker and file order.
