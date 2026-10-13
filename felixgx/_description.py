"""Best-effort parser for the free-text ``Acquisition_Description`` blob.

The description is human-oriented text, e.g.::

    Acquisition 2 (Copy)

    Type: EmissionScan

    Excitation: 490 nm
    Emission1: 505 nm -> 750 nm

    Slit Settings
       Ex: Entr: 5 nm, Exit: 5 nm   Em: Entr: 1 nm, Exit: 1 nm
    ...

Rules (kept deliberately simple so unknown content degrades gracefully):

* Blank lines separate *blocks*.
* ``Key: value`` lines become dict entries (keys are snake_cased, values are
  converted to ``bool`` / ``int`` / ``float`` / :class:`Quantity` / a
  ``(Quantity, Quantity)`` range when they look like one, else kept as ``str``).
* A block starting with a non-``key: value`` line is a *section* titled by that
  line.  A section title on its own adopts the following block.  The main
  "... Settings" title before the trace list is not a section (it is stored as
  ``settings_title`` and its entries stay at top level).
* Indentation only matters for value-less sub-headings
  (``Emission 1 Correction`` -> nested dict).

Nothing here raises on odd input; see :func:`parse_description`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .models import Quantity

_KV = re.compile(r"^(?P<k>[^:]+?):(?:\s+|$)(?P<v>.*)$")
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_INT = re.compile(r"^[-+]?\d+$")
_FLOAT = re.compile(rf"^{_NUM}$")
_QTY = re.compile(rf"^({_NUM})\s*([A-Za-zµ°%][\w/%µ°·^-]*)$")
_PERCENT = re.compile(rf"^({_NUM})\s*(%)$")
_TRACE_ATTR = re.compile(r"(\w+)=('[^']*'|<[^>]*>|[^,]*)")

Line = Tuple[int, str]  # (indent, stripped text)


def snake(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", s.strip()).strip("_").lower() or "unnamed"


def convert(value: str) -> Any:
    """Turn a textual value into bool / int / float / Quantity / range / str."""
    s = value.strip()
    if s in ("True", "False"):
        return s == "True"
    if _INT.match(s):
        return int(s)
    if _FLOAT.match(s):
        return float(s)
    m = _QTY.match(s) or _PERCENT.match(s)
    if m:
        return Quantity(float(m.group(1)), m.group(2))
    if "->" in s:
        parts = [p.strip() for p in s.split("->")]
        if len(parts) == 2:
            a, b = convert(parts[0]), convert(parts[1])
            if all(isinstance(v, (int, float, Quantity)) for v in (a, b)):
                return (a, b)
    return s


def _indent(line: str) -> int:
    n = 0
    for ch in line:
        if ch == " ":
            n += 1
        elif ch == "\t":
            n += 4
        else:
            break
    return n


def _put(d: Dict[str, Any], key: str, value: Any) -> None:
    k = snake(key)
    if k in d:
        i = 2
        while f"{k}_{i}" in d:
            i += 1
        k = f"{k}_{i}"
    d[k] = value


def _parse_body(lines: List[Line]) -> Any:
    """Parse the lines of one section. Returns a dict, or a str for plain text."""
    out: Dict[str, Any] = {}
    texts: List[str] = []
    i = 0
    while i < len(lines):
        indent, text = lines[i]
        m = _KV.match(text)
        if m:
            _put(out, m.group("k"), convert(m.group("v")))
            i += 1
            continue
        # value-less line: sub-heading if deeper-indented lines follow, else plain text
        j = i + 1
        while j < len(lines) and lines[j][0] > indent:
            j += 1
        child = lines[i + 1:j]
        if child:
            _put(out, text, _parse_body(child))
        else:
            texts.append(text)
        i = j
    if not out:
        return "\n".join(texts)
    if texts:
        out["text"] = "\n".join(texts)
    return out


def _unquote(v: str) -> Any:
    v = v.strip()
    if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
        return v[1:-1]
    if len(v) >= 2 and v[0] == "<" and v[-1] == ">":
        inner = v[1:-1]
        return None if inner == "None" else inner
    return convert(v) if v else ""


def _parse_trace_lines(lines: List[Line]) -> List[Dict[str, Any]]:
    """``Raw: Name='x', Src=Detector1, ...`` -> ``[{'kind': 'raw', 'name': 'x', 'src': ...}]``"""
    result = []
    for _, text in lines:
        m = _KV.match(text)
        if not m:
            continue
        entry: Dict[str, Any] = {"kind": m.group("k").strip().lower()}
        for k, v in _TRACE_ATTR.findall(m.group("v")):
            entry[snake(k)] = _unquote(v)
        result.append(entry)
    return result


def _parse_slits(lines: List[Line]) -> Optional[Dict[str, Any]]:
    """``Ex: Entr: 5 nm, Exit: 5 nm   Em: Entr: 1 nm, Exit: 1 nm`` -> nested dict."""
    names = {"ex": "excitation", "em": "emission"}
    out: Dict[str, Any] = {}
    for _, text in lines:
        for part in re.split(r"\s{2,}", text):
            m = re.match(r"(\w+):\s*Entr:\s*(.+?),\s*Exit:\s*(.+)$", part.strip())
            if not m:
                return None
            label = names.get(m.group(1).lower(), snake(m.group(1)))
            out[label] = {"entrance": convert(m.group(2)), "exit": convert(m.group(3))}
    return out or None


def _section_value(title: str, body: List[Line]) -> Any:
    key = snake(title)
    if key == "traces":
        return _parse_trace_lines(body)
    if key == "slit_settings":
        slits = _parse_slits(body)
        if slits is not None:
            return slits
    return _parse_body(body)


def _split_blocks(text: str) -> List[List[Line]]:
    blocks: List[List[Line]] = []
    cur: List[Line] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append((_indent(line), line.strip()))
    if cur:
        blocks.append(cur)
    return blocks


def _is_kv(line: Line) -> bool:
    return _KV.match(line[1]) is not None


def _parse(text: str) -> Dict[str, Any]:
    blocks = _split_blocks(text)
    out: Dict[str, Any] = {}
    seen_traces = False
    i = 0

    # The very first line is the acquisition's name (unless it is already "Type: ...").
    if blocks and not _is_kv(blocks[0][0]):
        out["acquisition_name"] = blocks[0][0][1]
        blocks[0] = blocks[0][1:]
        if not blocks[0]:
            blocks.pop(0)

    while i < len(blocks):
        block = blocks[i]
        i += 1
        if _is_kv(block[0]):                      # headless block: top-level entries
            parsed = _parse_body(block)
            for k, v in (parsed.items() if isinstance(parsed, dict) else []):
                _put(out, k, v)
            continue

        title, body = block[0][1], block[1:]

        if title.lower() == "comments":           # free text until the AutoSave block
            texts = [t for _, t in body]
            while i < len(blocks) and not blocks[i][0][1].lower().startswith("autosave session"):
                texts.extend(t for _, t in blocks[i])
                i += 1
            if texts:
                out["comments"] = "\n".join(texts)
            continue

        if not body:
            if title.endswith("Settings") and not seen_traces and "settings_title" not in out:
                out["settings_title"] = title      # e.g. "Emission Scan Settings"
                continue
            if i < len(blocks) and _is_kv(blocks[i][0]):   # adopt the following block
                m = _KV.match(blocks[i][0][1])
                if m and m.group("k").strip().lower() == "type" and "acquisition_name" not in out:
                    # The acquisition's name, repeated after a converter
                    # header ("Original File: ...", "FelixGX Version: ..."):
                    # not a section title - the following "Type:" line stays
                    # top-level so the measurement keeps its type.
                    out["acquisition_name"] = title
                    continue
                body = blocks[i]
                i += 1
            else:
                continue                           # empty heading, nothing to store

        _put(out, title, _section_value(title, body))
        if snake(title) == "traces":
            seen_traces = True
    return out


def parse_description(text: Optional[str]) -> Dict[str, Any]:
    """Parse an ``Acquisition_Description`` into a nested dict. Never raises.

    Returns ``{}`` for empty input.  If something unexpected happens while
    parsing, whatever was understood is lost but the raw text is kept under the
    ``"description_text"`` key so no information is thrown away.
    """
    if not text or not text.strip():
        return {}
    try:
        return _parse(text)
    except Exception:  # pragma: no cover - defensive, the parser is lenient by design
        return {"description_text": text}
