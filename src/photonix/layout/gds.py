"""GDSII import/export via gdstk."""
from __future__ import annotations

import numpy as np

from .cell import Cell

__all__ = ["write_gds", "read_gds"]


def _to_gdstk(cell: Cell, lib, built: dict):
    import gdstk

    if cell.name in built:
        return built[cell.name]
    gcell = lib.new_cell(cell.name)
    for pts, (layer, dt) in cell.polygons:
        gcell.add(gdstk.Polygon(np.asarray(pts), layer=layer, datatype=dt))
    for ref in cell.references:
        child = _to_gdstk(ref.cell, lib, built)
        gref = gdstk.Reference(
            child, origin=ref.origin, rotation=np.deg2rad(ref.rotation), x_reflection=ref.mirror
        )
        gcell.add(gref)
    # store ports as labels for round-trip readability
    for prt in cell.ports.values():
        gcell.add(gdstk.Label(prt.name, prt.center, layer=prt.layer[0]))
    built[cell.name] = gcell
    return gcell


def write_gds(cell: Cell, path: str) -> str:
    """Write ``cell`` (and its reference tree) to a GDSII file.

    Returns the path written.

    Examples
    --------
    >>> import tempfile, os
    >>> from photonix.layout import components, write_gds
    >>> p = os.path.join(tempfile.mkdtemp(), "wg.gds")
    >>> _ = write_gds(components.straight(10.0), p)
    >>> os.path.getsize(p) > 0
    True
    """
    import gdstk

    lib = gdstk.Library()
    _to_gdstk(cell, lib, {})
    lib.write_gds(path)
    return path


def read_gds(path: str) -> Cell:
    """Read a GDSII file into a (best-effort) :class:`Cell` of the top cell.

    Polygons and labels (as ports) of the top cell are imported; nested
    references are flattened into polygons.
    """
    import gdstk

    lib = gdstk.read_gds(path)
    tops = lib.top_level()
    gcell = tops[0] if tops else lib.cells[0]
    out = Cell(gcell.name)
    for poly in gcell.get_polygons():
        out.add_polygon(np.asarray(poly.points), (int(poly.layer), int(poly.datatype)))
    for lbl in getattr(gcell, "labels", []):
        out.add_port(str(lbl.text), tuple(np.asarray(lbl.origin)))
    return out
