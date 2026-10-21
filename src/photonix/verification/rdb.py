"""KLayout report databases (``.lyrdb``): write photonix results as markers, read KLayout's back.

The ``.lyrdb`` format is KLayout's XML marker database (``rdb`` module), which
its Marker Browser opens next to the layout. One *category* is written per
rule (or LVS issue kind), one *item* per violation, with the violation region
as a ``polygon: (x,y;x,y;...)`` value in µm. The category description carries
the rule's layer and value (``"layer 1/0, value 0.15: ..."``) so that
:func:`read_lyrdb` restores a :class:`~photonix.layout.DrcReport` with
identical polygons. Reading also accepts the ``box``, ``edge``, and
``edge-pair`` values that KLayout's own DRC writes (edges and edge pairs
become their spanning quadrilaterals).
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from photonix.layout.drc import DrcReport, DrcViolation

__all__ = ["write_lyrdb", "read_lyrdb"]

_DESC_RE = re.compile(r"^layer (-?\d+)/(-?\d+), value ([^:]+?)(?::|$)")


def _num(v: float) -> str:
    r = repr(float(v))
    return "0" if r in ("0.0", "-0.0") else (r[:-2] if r.endswith(".0") else r)


def _poly_value(poly: np.ndarray) -> str:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return "polygon: (" + ";".join(f"{_num(x)},{_num(y)}" for x, y in pts) + ")"


def _quote(name: str) -> str:
    return "'" + name.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _unquote(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        return re.sub(r"\\(.)", r"\1", t[1:-1])
    return t


def _write_items(
    items: Sequence[tuple[str, np.ndarray]],
    path: str | os.PathLike[str],
    *,
    top: str = "TOP",
    descriptions: Mapping[str, str] | None = None,
    generator: str = "photonix",
    description: str = "photonix verification",
) -> str:
    descriptions = dict(descriptions or {})
    cats: list[str] = []
    for cat, _poly in items:
        if cat not in cats:
            cats.append(cat)
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<report-database>",
        f" <description>{escape(description)}</description>",
        " <original-file/>",
        f" <generator>{escape(generator)}</generator>",
        f" <top-cell>{escape(top)}</top-cell>",
        " <tags/>",
        " <categories>",
    ]
    for cat in cats:
        lines += ["  <category>", f"   <name>{escape(cat)}</name>",
                  f"   <description>{escape(descriptions.get(cat, ''))}</description>",
                  "   <categories/>", "  </category>"]
    lines += [" </categories>", " <cells>", "  <cell>", f"   <name>{escape(top)}</name>", "   <variant/>",
              "   <references/>", "  </cell>", " </cells>", " <items>"]
    for cat, poly in items:
        lines += ["  <item>", "   <tags/>", f"   <category>{escape(_quote(cat))}</category>",
                  f"   <cell>{escape(top)}</cell>", "   <visited>false</visited>", "   <multiplicity>1</multiplicity>",
                  "   <values>", f"    <value>{escape(_poly_value(poly))}</value>", "   </values>", "  </item>"]
    lines += [" </items>", "</report-database>", ""]
    out = Path(path)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def write_lyrdb(report: DrcReport, path: str | os.PathLike[str], *, top: str = "TOP",
                descriptions: Mapping[str, str] | None = None) -> str:
    """Write a DRC report as a KLayout ``.lyrdb`` marker database.

    Parameters
    ----------
    report
        Any :class:`~photonix.layout.DrcReport` (pre-check or rule deck).
    path
        Output file.
    top
        Cell name the markers belong to (the layout's top cell).
    descriptions
        Optional rule name -> text appended to each category description.

    Returns
    -------
    str
        The path written.

    Examples
    --------
    >>> import os, tempfile
    >>> from photonix.layout import Cell, run_drc_precheck
    >>> c = Cell("t")
    >>> _ = c.add_polygon([(0, 0), (10, 0), (10, 0.1), (0, 0.1)])
    >>> rep = run_drc_precheck(c, {(1, 0): {"min_width": 0.15}})
    >>> path = write_lyrdb(rep, os.path.join(tempfile.mkdtemp(), "drc.lyrdb"))
    >>> back = read_lyrdb(path)
    >>> [(v.rule, v.layer, v.value) for v in back.violations]
    [('min_width', (1, 0), 0.15)]
    """
    extra = dict(descriptions or {})
    desc: dict[str, str] = {}
    items: list[tuple[str, np.ndarray]] = []
    for v in report.violations:
        cat = v.rule
        text = f"layer {v.layer[0]}/{v.layer[1]}, value {_num(v.value)}"
        if extra.get(v.rule):
            text += f": {extra[v.rule]}"
        if cat in desc and desc[cat] != text:
            cat = f"{v.rule}@{v.layer[0]}/{v.layer[1]}/{_num(v.value)}"
        desc.setdefault(cat, text)
        items.append((cat, v.polygon))
    return _write_items(items, path, top=top, descriptions=desc, generator="photonix.verification.write_lyrdb",
                        description="photonix DRC")


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _points(text: str) -> np.ndarray:
    pairs = re.findall(rf"({_NUM})\s*,\s*({_NUM})", text)
    return np.array([(float(x), float(y)) for x, y in pairs], dtype=float).reshape(-1, 2)


def _value_polygon(value: str) -> np.ndarray | None:
    kind, _, body = value.partition(":")
    kind = kind.strip().lower()
    body = body.strip()
    if kind == "polygon":
        outline = body.strip("()").split("/")[0]
        return _points(outline)
    if kind == "box":
        p = _points(body)
        if len(p) != 2:
            return None
        (x0, y0), (x1, y1) = p
        return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    if kind == "edge-pair":
        parts = re.split(r"[/|]", body)
        if len(parts) != 2:
            return None
        a, b = _points(parts[0]), _points(parts[1])
        return np.array([a[0], a[1], b[0], b[1]]) if len(a) == 2 and len(b) == 2 else None
    if kind == "edge":
        p = _points(body)
        return p if len(p) == 2 else None
    return None


def read_lyrdb(path: str | os.PathLike[str], *, default_layer: tuple[int, int] = (0, 0)) -> DrcReport:
    """Read a ``.lyrdb`` marker database into a :class:`~photonix.layout.DrcReport`.

    Category names become rule names. Layer and value are taken from a
    ``"layer L/D, value V"`` description (as written by :func:`write_lyrdb`),
    otherwise ``default_layer`` and ``nan``. Items whose values are not
    polygons, boxes, edges or edge pairs (texts, numbers) are skipped.
    ``report.rules`` maps every category name to its description.
    """
    root = ET.parse(os.fspath(path)).getroot()
    meta: dict[str, tuple[tuple[int, int], float, str]] = {}

    def walk(cats: ET.Element | None, prefix: str = "") -> None:
        if cats is None:
            return
        for cat in cats.findall("category"):
            name = (cat.findtext("name") or "").strip()
            full = f"{prefix}{name}"
            text = (cat.findtext("description") or "").strip()
            m = _DESC_RE.match(text)
            if m:
                layer = (int(m.group(1)), int(m.group(2)))
                try:
                    value = float(m.group(3))
                except ValueError:  # pragma: no cover - malformed description
                    value = float("nan")
            else:
                layer, value = default_layer, float("nan")
            meta[full] = (layer, value, text)
            walk(cat.find("categories"), f"{full}.")

    walk(root.find("categories"))
    report = DrcReport(rules={k: v[2] for k, v in meta.items()})
    items = root.find("items")
    for item in ([] if items is None else items.findall("item")):
        cat = _unquote(item.findtext("category") or "")
        layer, value, _text = meta.get(cat, (default_layer, float("nan"), ""))
        rule = cat.split("@", 1)[0]
        vals = item.find("values")
        for v in ([] if vals is None else vals.findall("value")):
            poly = _value_polygon(v.text or "")
            if poly is not None and len(poly) >= 2:
                report.violations.append(DrcViolation(rule, layer, value, poly))
    return report

