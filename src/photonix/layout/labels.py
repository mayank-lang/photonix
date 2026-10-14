"""Text: polygon lettering for masks, and GDS TEXT labels.

Two different things are called "text" in a layout:

* **Drawn lettering** -- device names, die IDs, wafer marks -- is *geometry*:
  polygons that are etched or plated like any other shape. :func:`text` draws
  it with gdstk's stroke font and can enlarge it until every stroke meets a
  minimum-width design rule.
* **Labels** are GDSII ``TEXT`` elements: a string at a point, not printed on
  the mask. Port names, automated-test labels (``opt_in_TE_1550_device_...``)
  and pin names are labels. A :class:`Label` lives on a :class:`LabeledCell`,
  a :class:`~photonix.layout.Cell` subclass that
  :func:`~photonix.layout.write_layout` writes as ``TEXT`` and
  :func:`~photonix.layout.read_layout` reads back.

Font geometry
-------------
gdstk's font is drawn on a grid of ``size / 16``: every vertex of every
printable ASCII glyph lies on it, and the narrowest stroke (and the thinnest
serif) is exactly one grid step, :data:`TEXT_STROKE_RATIO` ``* size``. This is
measured, not assumed: ``tests/test_layout_packaging.py`` recomputes the
smallest interior vertex-to-edge distance over all 94 printable glyphs and runs
:func:`~photonix.layout.check_min_width` at the ratio. ``min_stroke`` therefore
scales ``size`` up to ``16 * min_stroke`` when needed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any  # noqa: F401  (inherited dataclass field annotations)

import numpy as np

from .cell import Cell, Port, Reference  # noqa: F401  (Port: inherited dataclass field annotations)
from .gds import _gdstk

__all__ = ["Label", "LabeledCell", "text", "TEXT_STROKE_RATIO", "TEXT_LAYER", "ANCHORS"]

#: Narrowest stroke of the gdstk font as a fraction of the text ``size``.
TEXT_STROKE_RATIO = 1.0 / 16.0
#: Default layer for drawn lettering (a common "text" layer in open PDKs).
TEXT_LAYER = (66, 0)
#: Anchor names accepted by :func:`text` (compass points of the glyph bounding box).
ANCHORS = ("sw", "s", "se", "w", "c", "e", "nw", "n", "ne")
_GDS_ANCHORS = ("o", "n", "s", "e", "w", "ne", "nw", "se", "sw")


def _layer(layer) -> tuple[int, int]:
    return (int(layer[0]), int(layer[1]))


@dataclass(frozen=True)
class Label:
    """A GDSII ``TEXT`` element: a string at a point, not drawn on the mask.

    Attributes
    ----------
    text
        The string (non-empty; GDSII allows up to 512 characters).
    position
        Text origin in µm.
    layer
        GDS ``(layer, texttype)``.
    rotation
        Presentation rotation in degrees (GDSII only; OASIS ``TEXT`` records
        have no rotation and drop it).
    magnification
        Presentation magnification (GDSII only, as ``rotation``).
    anchor
        gdstk anchor: ``"o"`` (origin), ``"n"``, ``"s"``, ``"e"``, ``"w"``,
        ``"ne"``, ``"nw"``, ``"se"`` or ``"sw"``.

    Examples
    --------
    >>> Label("die_A1", (10.0, 20.0), (10, 0)).position
    (10.0, 20.0)
    """

    text: str
    position: tuple[float, float] = (0.0, 0.0)
    layer: tuple[int, int] = (10, 0)
    rotation: float = 0.0
    magnification: float = 1.0
    anchor: str = "o"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Label text must be a non-empty string")
        if len(self.text.encode()) > 512:
            raise ValueError("GDSII TEXT strings are limited to 512 bytes")
        pos = (float(self.position[0]), float(self.position[1]))
        if not all(math.isfinite(v) for v in pos):
            raise ValueError(f"Label position must be finite, got {self.position!r}")
        if self.anchor not in _GDS_ANCHORS:
            raise ValueError(f"anchor must be one of {_GDS_ANCHORS}, got {self.anchor!r}")
        if not math.isfinite(self.magnification) or self.magnification <= 0:
            raise ValueError("Label magnification must be positive and finite")
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "layer", _layer(self.layer))
        object.__setattr__(self, "rotation", float(self.rotation) % 360.0)
        object.__setattr__(self, "magnification", float(self.magnification))

    def moved(self, ref: Reference) -> Label:
        """This label in the parent coordinates of ``ref`` (mirror -> rotate -> translate).

        A mirrored reference negates the rotation but keeps the string
        readable (labels are not drawn, so there is nothing to mirror).
        """
        x, y = self.position
        rot = self.rotation
        if ref.mirror:
            y, rot = -y, -rot
        th = math.radians(ref.rotation)
        xr = x * math.cos(th) - y * math.sin(th) + ref.origin[0]
        yr = x * math.sin(th) + y * math.cos(th) + ref.origin[1]
        return Label(self.text, (xr, yr), self.layer, rot + ref.rotation, self.magnification, self.anchor)


@dataclass(repr=False)
class LabeledCell(Cell):
    """A :class:`~photonix.layout.Cell` that also carries GDS ``TEXT`` labels.

    It behaves exactly like a :class:`Cell` (extraction, routing, DRC ignore
    the labels); :func:`~photonix.layout.write_layout` writes ``labels`` as
    ``TEXT`` elements and :func:`~photonix.layout.read_layout` returns every
    structure as a ``LabeledCell``.

    Examples
    --------
    >>> c = LabeledCell("chip")
    >>> _ = c.add_label("opt_in_TE_1550_device_me_ring1", (0.0, 0.0), (10, 0))
    >>> [lbl.text for lbl in c.labels]
    ['opt_in_TE_1550_device_me_ring1']
    """

    labels: list[Label] = field(default_factory=list)

    def add_label(self, text: str, position=(0.0, 0.0), layer=(10, 0), *, rotation: float = 0.0,
                  magnification: float = 1.0, anchor: str = "o") -> LabeledCell:
        """Append a :class:`Label` and return ``self``."""
        self.labels.append(Label(text, tuple(position), _layer(layer), rotation, magnification, anchor))
        return self

    def get_labels(self) -> list[Label]:
        """Labels of this cell and of every :class:`LabeledCell` below it, in this cell's coordinates."""
        out = list(self.labels)
        for ref in self.references:
            if isinstance(ref.cell, LabeledCell):
                out.extend(lbl.moved(ref) for lbl in ref.cell.get_labels())
        return out

    def flatten(self, name: str | None = None) -> LabeledCell:
        """As :meth:`Cell.flatten`, keeping every label (transformed) as well."""
        flat = super().flatten(name)
        return LabeledCell(flat.name, flat.polygons, [], flat.ports, flat.model, flat.model_settings,
                           self.get_labels())

    @classmethod
    def from_cell(cls, cell: Cell, labels: Iterable[Label] = ()) -> LabeledCell:
        """A ``LabeledCell`` with ``cell``'s contents (lists copied, geometry shared) plus ``labels``."""
        existing = list(cell.labels) if isinstance(cell, LabeledCell) else []
        return cls(cell.name, list(cell.polygons), list(cell.references), dict(cell.ports), cell.model,
                   dict(cell.model_settings), existing + list(labels))


def _anchor_point(bbox: tuple[np.ndarray, np.ndarray], anchor: str) -> np.ndarray:
    (x0, y0), (x1, y1) = bbox
    xs = {"w": x0, "c": 0.5 * (x0 + x1), "e": x1}
    ys = {"s": y0, "c": 0.5 * (y0 + y1), "n": y1}
    if anchor == "c":
        return np.array([xs["c"], ys["c"]])
    vert = anchor[0] if anchor[0] in "ns" else "c"
    horiz = anchor[-1] if anchor[-1] in "ew" else "c"
    return np.array([xs[horiz], ys[vert]])


def text(string: str, *, size: float = 10.0, layer: Sequence[int] = TEXT_LAYER,
         position: Sequence[float] = (0.0, 0.0), anchor: str = "sw", min_stroke: float | None = None,
         name: str = "text") -> Cell:
    """Polygon lettering (gdstk stroke font) as a portless :class:`Cell`.

    Parameters
    ----------
    string
        Text to draw; ``"\\n"`` starts a new line.
    size
        Full glyph height in µm (descender to ascender). Characters advance by
        ``9/16 size`` and lines by ``5/4 size``.
    layer
        GDS layer of the glyph polygons.
    position, anchor
        The ``anchor`` point of the glyphs' bounding box (``"sw"``, ``"s"``,
        ``"se"``, ``"w"``, ``"c"``, ``"e"``, ``"nw"``, ``"n"``, ``"ne"``) is
        placed at ``position``.
    min_stroke
        Minimum stroke width in µm (a min-width design rule). ``size`` is
        raised to ``min_stroke / TEXT_STROKE_RATIO`` when it would draw a
        narrower stroke; it is never reduced.

    Notes
    -----
    Requires gdstk (``pip install photonix[layout]``). Characters outside
    printable ASCII are not in the font and draw nothing.

    Examples
    --------
    >>> text("A1", size=16.0).bbox()                     # cap height 11/16 of size
    ((0.0, 0.0), (15.0, 11.0))
    >>> text("A1", size=16.0, min_stroke=2.0).bbox()     # 2 um strokes need size 32
    ((0.0, 0.0), (30.0, 22.0))
    """
    if not isinstance(string, str):
        raise TypeError("text() needs a string")
    if not math.isfinite(size) or size <= 0:
        raise ValueError(f"size must be positive and finite, got {size!r}")
    if min_stroke is not None:
        if not math.isfinite(min_stroke) or min_stroke <= 0:
            raise ValueError(f"min_stroke must be positive and finite, got {min_stroke!r}")
        size = max(float(size), float(min_stroke) / TEXT_STROKE_RATIO)
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    gdstk = _gdstk()
    lay = _layer(layer)
    polys = [np.asarray(p.points, dtype=float) for p in gdstk.text(string, float(size), (0.0, 0.0))]
    c = Cell(name)
    if not polys:
        return c
    allp = np.concatenate(polys)
    shift = np.asarray(position, dtype=float) - _anchor_point((allp.min(axis=0), allp.max(axis=0)), anchor)
    for p in polys:
        c.add_polygon(p + shift, lay)
    return c
