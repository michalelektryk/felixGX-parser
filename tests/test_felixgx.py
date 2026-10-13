"""Tests. Synthetic files exercise the awkward cases; real files are used if present."""
import gzip
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pytest

import felixgx
from felixgx import FelixParseError, Quantity
from felixgx._description import parse_description

# --------------------------------------------------------------------------- #
# Synthetic file builder
# --------------------------------------------------------------------------- #
EMISSION_DESC = """Acq (Copy)

Type: EmissionScan

Created: 19/11/2025 19:33:40

Emission Scan Settings

Repeats: 0
Background: (Not Used)

Excitation: 490 nm
Emission1: 505 nm -> 510 nm

Slit Settings
   Ex: Entr: 5 nm, Exit: 5 nm   Em: Entr: 1 nm, Exit: 1 nm

Step Size: 1 nm
Integration: 1 sec

Traces

  Raw: Name='D1 490:505-510', Src=Detector1, Bgrd=0, Vis=<Graph 1>, Grp=''
  Raw: Name='ExCorr 490:505-510', Src=RCQCSignal, Bgrd=0, Vis=<None>, Grp=''
  Derived: Name='D1 490:505-510 [COR]', Src1=, Func=RealTimeCorrection,   Vis=<None>, Grp=''

Comments

my comment: here

AutoSave Session
C:\\Users\\x\\autosave.gxz
"""

EXCITATION_DESC = EMISSION_DESC.replace("EmissionScan", "ExcitationScan")
DECAY_DESC = """Acq

Type: TM_ibhTCSPC_Decay

Created: 19/11/2025 19:28:07

DeltaHub TCSPC Fluorescence Decay Settings

Time Range:\t\t100 ns

Traces

  Raw: Name='T1 LED:570', Src=Detector1, Bgrd=0, Vis=<Graph 1>, Grp=''
"""


def _points(xs, ys):
    return "".join(f"<Pt_D><X>{x}</X><Y>{y}</Y></Pt_D>" for x, y in zip(xs, ys))


def trace(name, t="2025-11-19T19:34:02.5299893+01:00", *, kind="emission", raw=True,
          desc=None, ys=(1, 2, 3), xs=None, props=None, extra=""):
    if kind == "decay":
        xl, yl, xs = "Time (Nanoseconds)", "Counts (Counts)", xs or (0, 1, 2)
        desc = DECAY_DESC if desc is None else desc
        pr = props if props is not None else {"acq.ExcWave": -1, "acq.EmiWave": 570}
    else:
        xl, yl, xs = "Wavelength (Nanometers)", "Intensity (Counts/s)", xs or (505, 506, 507)
        base = EXCITATION_DESC if kind == "excitation" else EMISSION_DESC
        desc = base if desc is None else desc
        pr = props if props is not None else {"acq.ExcWave": 490, "acq.EmiStartWave": 505,
                                              "acq.EmiEndWave": 507, "acq.WaveStepSize": 1}
    if not raw:  # the derived curve only carries acq.DateTime
        pr = {}
    prop_xml = "".join(f'<Double Name="{k}">{v}</Double>' for k, v in pr.items())
    prop_xml += f'<DateTime Name="acq.DateTime">{t}</DateTime>'
    return (f"<FelixTrace><Name>{escape(name)}</Name><Id>id-{escape(name)}</Id>"
            f"<Acquisition_Description>{escape(desc)}</Acquisition_Description>"
            f"<XLabel>{xl}</XLabel><YLabel>{yl}</YLabel><Color>NamedColor:Red</Color>"
            f"<datapoints>{_points(xs, ys)}</datapoints><eventmarkers/>"
            f"<properties>{prop_xml}</properties><TimeExecuted>{t}</TimeExecuted>{extra}</FelixTrace>")


def roc(name, t="2025-11-19T19:34:02.5299893+01:00"):
    return (f"<FelixTrace><Name>{name}</Name><Id>r-{name}</Id>"
            f"<XLabel>Wavelength (Nanometers)</XLabel><YLabel>Intensity (Volts)</YLabel>"
            f"<datapoints>{_points((505, 506, 507), (.5, .6, .7))}</datapoints>"
            f"<properties/><TimeExecuted>{t}</TimeExecuted></FelixTrace>")


def build(det=(), refs=(), *, version="3.5.1.0", gz=True, sessions_tag=True, root="FelixSessionData"):
    sess = ""
    if sessions_tag:
        sess = ("<sessions><FelixSession><Name>S1</Name><Id>sid</Id><groups>"
                f"<FelixGroup><Name>Detector1</Name><Id>g1</Id><traces>{''.join(det)}</traces></FelixGroup>"
                f"<FelixGroup><Name>RCQCSignal</Name><Id>g2</Id><traces>{''.join(refs)}</traces></FelixGroup>"
                "</groups></FelixSession></sessions>")
    xml = ("\ufeff<?xml version=\"1.0\" encoding=\"utf-8\"?>"
           f"<{root} xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
           f"<FelixSessionDataVersion>{version}</FelixSessionDataVersion>"
           "<ProgramVersion>4.9</ProgramVersion><OriginalFilename>C:\\a.gxz</OriginalFilename>"
           f"{sess}</{root}>").encode("utf-8")
    return gzip.compress(xml) if gz else xml


def parse(*a, **kw):
    strict = kw.pop("strict", False)
    return felixgx.loads(build(*a, **kw), strict=strict)


# --------------------------------------------------------------------------- #
# Grouping & missing curves
# --------------------------------------------------------------------------- #
def test_full_emission_scan():
    f = parse([trace("S1"), trace("S1 COR", raw=False, ys=(10, 20, 30))], [roc("ExCorr")])
    assert len(f.emissions) == 1 and not f.decays and not f.excitations
    m = f.emissions[0]
    assert m.raw.name == "S1" and m.corrected.name == "S1 COR" and m.roc.name == "ExCorr"
    assert list(m.corrected.y) == [10, 20, 30]
    assert m.name == "S1"
    assert m.excitation_wavelength == 490 and m.emission_range == (505, 507) and m.step == 1
    assert m.best is m.corrected
    assert [t.name for t in m.traces] == ["S1", "S1 COR", "ExCorr"]


def test_roles_do_not_depend_on_names():
    # user renamed both curves to something without "COR", and the corrected one comes FIRST
    f = parse([trace("zzz", raw=False, ys=(9, 9, 9)), trace("aaa")], [])
    m = f.emissions[0]
    assert m.raw.name == "aaa" and m.corrected.name == "zzz"


def test_only_raw_when_correction_disabled():
    f = parse([trace("S1")], [roc("ExCorr")])
    m = f.emissions[0]
    assert m.raw is not None and m.corrected is None and m.roc is not None
    assert m.best is m.raw


def test_only_corrected_when_raw_deleted():
    f = parse([trace("whatever", raw=False)], [])
    m = f.emissions[0]
    assert m.raw is None and m.corrected is not None and m.roc is None
    assert m.name == "whatever"
    assert m.excitation_wavelength == 490           # no acq.* props; from the description
    assert m.emission_range == (505, 507)           # no acq.Emi*Wave; falls back to the data extent


def test_lone_trace_without_properties_is_raw_if_no_corrected_was_configured():
    # e.g. a decay: the manifest lists only a Raw trace, so this cannot be a corrected curve
    f = parse([trace("d", kind="decay", props={})], [])
    assert f.decays[0].raw is not None and f.decays[0].corrected is None


def test_uncorrected_in_name_is_not_a_cor_hint():
    f = parse([trace("sample uncorrected", raw=False)], [])
    assert f.emissions[0].corrected is not None     # still derived by its (missing) properties


def test_corrected_only_name_suffix_is_stripped():
    f = parse([trace("sample 1 COR", raw=False)], [])
    assert f.emissions[0].name == "sample 1"


def test_missing_roc_and_orphan_roc():
    f = parse([trace("S1"), trace("S1 COR", raw=False)],
              [roc("ExCorr(5)", t="2025-01-01T00:00:00+01:00")])      # different time
    assert f.emissions[0].roc is None
    assert [t.name for t in f.unassigned_traces] == ["ExCorr(5)"]


def test_measurements_do_not_get_mixed_up():
    t1, t2 = "2025-11-19T19:34:02.5299893+01:00", "2025-11-19T19:45:51.4752972+01:00"
    f = parse([trace("A", t1), trace("A COR", t1, raw=False), trace("B", t2), trace("B COR", t2, raw=False)],
              [roc("r2", t2), roc("r1", t1)])
    a, b = f.emissions
    assert (a.name, a.corrected.name, a.roc.name) == ("A", "A COR", "r1")
    assert (b.name, b.corrected.name, b.roc.name) == ("B", "B COR", "r2")


def test_three_kinds_and_file_order():
    t = ["2025-11-19T10:00:00+01:00", "2025-11-19T10:01:00+01:00", "2025-11-19T10:02:00+01:00"]
    f = parse([trace("em", t[0]), trace("dec", t[1], kind="decay"),
               trace("exc", t[2], kind="excitation", props={"acq.EmiWave": 600})], [])
    assert [m.name for m in f.emissions] == ["em"]
    assert [m.name for m in f.decays] == ["dec"]
    assert [m.name for m in f.excitations] == ["exc"]
    assert [m.name for m in f] == ["em", "dec", "exc"]
    assert f.excitations[0].emission_wavelength == 600
    assert f.decays[0].emission_wavelength == 570 and f.decays[0].excitation_wavelength is None
    assert f.decays[0].time_range == Quantity(100.0, "ns")
    assert len(f) == 3


def test_unknown_type_goes_to_other_with_warning():
    f = parse([trace("sync", desc=EMISSION_DESC.replace("EmissionScan", "SynchronousScan"))], [])
    assert not f.emissions and len(f.other) == 1
    assert any("SynchronousScan" in w for w in f.warnings)


def test_no_description_kind_is_inferred():
    f = parse([trace("x", desc=" "), trace("d", "2025-01-01T00:00:00+01:00", kind="decay", desc=" ")], [])
    assert len(f.emissions) == 1 and len(f.decays) == 1
    assert len(f.warnings) == 2


def test_duplicate_traces_are_kept_not_dropped():
    f = parse([trace("a"), trace("a COR", raw=False), trace("a COR copy", raw=False)], [])
    m = f.emissions[0]
    assert len(m.extra_traces) == 1
    assert f.warnings


# --------------------------------------------------------------------------- #
# Converter headers & timestamp jitter (bugs seen in real files)
# --------------------------------------------------------------------------- #
CONVERTER_HEADER = ("Original File: C:\\x.gxz\nFelixGX Version: 4.9.0.10321\n"
                    "Converter Version: Felix GX Version 4 4.9.0.10321\n\n")


def test_converter_header_does_not_swallow_type():
    # A re-saved/merged session repeats the acquisition name *after* the
    # converter header; "Type:" must stay top-level or the measurement loses
    # its type and lands in FelixFile.other.
    f = parse([trace("s", desc=CONVERTER_HEADER + EMISSION_DESC)], [])
    assert len(f.emissions) == 1 and not f.other and not f.warnings
    m = f.emissions[0]
    assert m.type_name == "EmissionScan" and m.metadata["type"] == "EmissionScan"
    assert m.metadata["acquisition_name"] == "Acq (Copy)"
    assert m.excitation_wavelength == 490


def test_converter_header_excitation_scan():
    f = parse([trace("e", kind="excitation",
                     desc=CONVERTER_HEADER + EXCITATION_DESC,
                     props={"acq.ExcStartWave": 400, "acq.ExcEndWave": 510, "acq.EmiWave": 590})], [])
    assert len(f.excitations) == 1 and not f.warnings
    assert f.excitations[0].excitation_range == (400, 510)


def _shift(ts, ms):
    """Timestamp text with the last digits replaced to shift by *ms* milliseconds."""
    from datetime import datetime, timedelta
    from felixgx.parser import _parse_dt
    return (_parse_dt(ts) + timedelta(milliseconds=ms)).isoformat()


def test_millisecond_jitter_groups_one_acquisition():
    # Felix stamps the raw / ROC / corrected curves of one acquisition a few ms
    # apart; they must not be split into separate measurements.
    t = "2025-11-19T19:34:02.5299893+01:00"
    f = parse([trace("S1", t), trace("S1 COR", _shift(t, 2), raw=False)],
              [roc("ExCorr", _shift(t, 1))])
    assert len(f.emissions) == 1 and not f.warnings
    m = f.emissions[0]
    assert m.raw is not None and m.corrected is not None and m.roc is not None


def test_acquisitions_a_second_apart_stay_separate():
    t = "2025-11-19T19:34:02.5299893+01:00"
    f = parse([trace("A", t), trace("B", _shift(t, 1500))], [])
    assert len(f.emissions) == 2


def test_zero_time_tolerance_restores_exact_matching():
    t = "2025-11-19T19:34:02.5299893+01:00"
    data = build([trace("S1", t), trace("S1 COR", _shift(t, 2), raw=False)], [])
    f = felixgx.loads(data, time_tolerance=0)
    assert len(f.emissions) == 2


# --------------------------------------------------------------------------- #
# CSV export example
# --------------------------------------------------------------------------- #
def test_export_csv_tree(tmp_path):
    import csv as _csv
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "export_csv", _P(__file__).resolve().parent.parent / "examples" / "export_csv.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    export = mod.export

    t = "2025-11-19T19:34:02.5299893+01:00"
    f = parse([trace("s/A", t), trace("s/A COR", t, raw=False),
               trace("d", "2025-11-19T20:30:00+01:00", kind="decay"),
               trace("e", "2025-11-19T21:00:00+01:00", kind="excitation")],
              [roc("ExCorr", t)])
    n = export(f, tmp_path)
    assert n > 0
    files = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.csv"))
    # raw+corrected+ROC of the scan share the (sanitized, deduplicated) folder
    assert "emission/s_A/data.csv" in files and "emission/s_A/corrected.csv" in files
    assert "emission/s_A/RCQCSignal.csv" in files and "emission/s_A/metadata.csv" in files
    # decay has no corrected/ROC -> no such files
    assert "decay/d/data.csv" in files and "decay/d/metadata.csv" in files
    assert not any(p.startswith("decay/d/") and p.rsplit("/", 1)[-1] not in
                   ("data.csv", "metadata.csv") for p in files)
    assert "excitation/e/data.csv" in files and (tmp_path / "metadata.csv").is_file()
    with open(tmp_path / "emission" / "s_A" / "data.csv") as fh:
        rows = list(_csv.reader(fh))
    assert rows[0] == ["wavelength_nanometers", "intensity_counts_s"] and len(rows) == 4


# --------------------------------------------------------------------------- #
# Data robustness
# --------------------------------------------------------------------------- #
def test_bad_and_missing_values():
    t = ("<FelixTrace><Name>n</Name><XLabel>Wavelength (Nanometers)</XLabel><YLabel>I (c)</YLabel>"
         "<datapoints><Pt_D><X>1</X><Y>abc</Y></Pt_D><Pt_D><X>2</X></Pt_D><Pt_D><Y>5</Y></Pt_D>"
         "<Pt_D><X>3</X><Y>Infinity</Y></Pt_D></datapoints></FelixTrace>")
    f = parse([t], [])
    tr = (f.emissions + f.other)[0].best or (f.emissions + f.other)[0].raw
    assert list(tr.x) == [1, 2, 3]
    assert np.isnan(tr.y[0]) and np.isnan(tr.y[1]) and np.isinf(tr.y[2])
    assert any("without an X" in w for w in f.warnings)


def test_empty_trace():
    t = "<FelixTrace><Name>e</Name><datapoints/></FelixTrace>"
    f = parse([t], [])
    assert len(f) == 1
    assert len((f.other + f.emissions)[0].traces[0]) == 0


def test_trace_extras_and_units():
    f = parse([trace("d", kind="decay", extra="<EmissionWavelengthDASTRES>570</EmissionWavelengthDASTRES>")], [])
    tr = f.decays[0].raw
    assert tr.extra["EmissionWavelengthDASTRES"] == "570" and tr.extra["Color"] == "NamedColor:Red"
    assert (tr.x_unit, tr.y_unit) == ("Nanoseconds", "Counts")
    assert tr.executed.year == 2025 and tr.as_array().shape == (2, 3)


# --------------------------------------------------------------------------- #
# File-level behaviour
# --------------------------------------------------------------------------- #
def test_plain_xml_accepted():
    assert len(parse([trace("a")], [], gz=False).emissions) == 1


def test_file_metadata():
    f = parse([trace("a")], [])
    assert f.metadata["format_version"] == "3.5.1.0"
    assert f.metadata["program_version"] == "4.9"
    assert f.metadata["original_filename"] == "C:\\a.gxz"
    assert f.metadata["sessions"] == [{"name": "S1", "id": "sid", "detectors": ["Detector1"],
                                       "reference_groups": ["RCQCSignal"]}]
    assert f.emissions[0].session_name == "S1" and f.emissions[0].detector == "Detector1"


def test_unknown_version_warns_and_strict_raises():
    f = parse([trace("a")], [], version="9.9")
    assert any("9.9" in w for w in f.warnings) and len(f.emissions) == 1
    with pytest.raises(FelixParseError):
        parse([trace("a")], [], version="9.9", strict=True)


def test_errors():
    with pytest.raises(FelixParseError):
        felixgx.loads(b"definitely not xml")
    with pytest.raises(FelixParseError):
        felixgx.loads(b"\x1f\x8b garbage")
    with pytest.raises(FelixParseError):
        felixgx.loads(build(root="Something"))
    with pytest.raises(FelixParseError):
        felixgx.loads(build(sessions_tag=False))
    with pytest.raises(FelixParseError):
        felixgx.load("/nonexistent/file.gxz")


def test_empty_session_is_ok():
    f = parse([], [])
    assert len(f) == 0 and f.warnings


def test_load_from_disk(tmp_path):
    p = tmp_path / "x.gxz"
    p.write_bytes(build([trace("a")], []))
    f = felixgx.load(p)                       # Path accepted
    assert f.path == str(p) and len(f.emissions) == 1


def test_find_and_summary():
    f = parse([trace("Alpha"), trace("Alpha COR", raw=False)], [])
    assert f.find("alp") == [f.emissions[0]] and f.find(r"^A", regex=True) and not f.find("zzz")
    assert "Alpha" in f.summary()


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_emission_metadata_parsed():
    m = parse([trace("a")], []).emissions[0]
    md = m.metadata
    assert md["type"] == "EmissionScan"
    assert md["excitation"] == Quantity(490.0, "nm")
    assert md["emission1"] == (Quantity(505.0, "nm"), Quantity(510.0, "nm"))
    assert md["step_size"] == Quantity(1.0, "nm") and md["integration"] == Quantity(1.0, "sec")
    assert md["slit_settings"] == {"excitation": {"entrance": Quantity(5.0, "nm"), "exit": Quantity(5.0, "nm")},
                                   "emission": {"entrance": Quantity(1.0, "nm"), "exit": Quantity(1.0, "nm")}}
    assert md["comments"] == "my comment: here"
    assert md["autosave_session"] == "C:\\Users\\x\\autosave.gxz"
    assert [t["kind"] for t in md["traces"]] == ["raw", "raw", "derived"]
    assert md["traces"][2]["func"] == "RealTimeCorrection"
    assert md["properties"]["acq.ExcWave"] == 490.0
    assert md["session"] == "S1" and md["detector"] == "Detector1" and md["executed"] is not None


def test_common_metadata():
    t1, t2 = "2025-11-19T19:34:02+01:00", "2025-11-19T19:45:51+01:00"
    other = EMISSION_DESC.replace("Excitation: 490 nm", "Excitation: 250 nm")
    f = parse([trace("A", t1), trace("B", t2, desc=other)], [])
    c = f.common_metadata
    assert c["integration"] == Quantity(1.0, "sec")
    assert "excitation" not in c                      # differs between the two
    assert "acquisition_name" not in c and "executed" not in c


def test_description_parser_is_lenient():
    assert parse_description(None) == {} and parse_description("   \n ") == {}
    d = parse_description("just a name")
    assert d == {"acquisition_name": "just a name"}
    d = parse_description("n\n\nType: X\n\nSome Heading\n   a: 1\n   Nested\n      b: 2.5 V\n\nweird line: : :")
    assert d["type"] == "X" and d["some_heading"]["a"] == 1
    assert d["some_heading"]["nested"]["b"] == Quantity(2.5, "V")
    # duplicate keys do not overwrite each other
    d = parse_description("n\n\nK: 1\nK: 2")
    assert d["k"] == 1 and d["k_2"] == 2


# --------------------------------------------------------------------------- #
# Real files (optional)
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent            # the measurements directory (next to felixgx_project/)
_DATE_DIR = re.compile(r"^\d{2}[-_]\d{2}[-_]\d{4}$")
_CANDIDATES = [Path(os.environ.get("FELIXGX_TEST_DATA", "")), _HERE, _HERE.parent,
               Path("/mnt/user-data/uploads"),
               *sorted(d for d in _ROOT.iterdir() if d.is_dir() and _DATE_DIR.match(d.name))]


def _real(name):
    for d in _CANDIDATES:
        if (d / name).is_file():
            return d / name
    pytest.skip(f"{name} not available")


def _real_in(folder, name):
    p = _ROOT / folder / name
    if not p.is_file():
        pytest.skip(f"{folder}/{name} not available")
    return p


def test_real_pomiary():
    f = felixgx.load(_real_in("19-11-2025", "pomiary.gxz"))
    assert (len(f.decays), len(f.excitations), len(f.emissions)) == (10, 0, 8)
    assert all(m.raw and m.corrected and m.roc for m in f.emissions)
    assert all(m.raw and not m.corrected and not m.roc for m in f.decays)
    assert len(f.unassigned_traces) == 4 and not f.warnings
    assert all(len(m.raw) == 4096 for m in f.decays)
    for m in f.emissions:     # the three curves of a scan share one x axis
        assert np.array_equal(m.raw.x, m.corrected.x) and np.array_equal(m.raw.x, m.roc.x)
        assert m.corrected.name == m.raw.name + " COR" or "COR" in m.corrected.name


def test_real_pmma_has_scans_without_roc():
    f = felixgx.load(_real("pomiary_w_PMMA.gxz"))
    assert (len(f.decays), len(f.emissions)) == (3, 3)
    assert [m.roc is not None for m in f.emissions] == [False, False, True]
    assert all(m.corrected is not None for m in f.emissions) and not f.warnings


def test_real_excitation_scans_barwniki():
    f = felixgx.load(_real_in("28-04-2026", "pomiary barwnikow ciala stalego.gxz"))
    assert not f.warnings and not f.other
    assert (len(f.decays), len(f.excitations), len(f.emissions)) == (5, 3, 5)
    for m in f.excitations:
        assert m.type_name == "ExcitationScan"
        assert m.raw is not None and len(m.raw) == 111
        assert m.emission_wavelength == 590 and m.excitation_range == (400, 510)
        assert m.step == 1 and (m.raw.x.min(), m.raw.x.max()) == (400, 510)
    # ROC signals of two excitation scans are matched despite ms jitter
    assert [m.roc is not None for m in f.excitations] == [False, True, True]
    # emission scans from the merged/converter-header session keep their type
    assert all(m.type_name == "EmissionScan" for m in f.emissions)
    assert all(m.excitation_wavelength == 490 for m in f.emissions)


def test_real_merged_file_no_ms_jitter_splits():
    f = felixgx.load(_real_in("20-11-2025", "pomiary.gxz"))
    b = f.find("B - 13:35")
    assert len(b) == 1 and b[0].raw is not None and b[0].corrected is not None
