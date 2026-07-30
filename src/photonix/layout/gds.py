"""Optional GDSII and OASIS import/export via gdstk."""
from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from .cell import Cell

__all__ = ["gdstk_available", "write_gds", "read_gds", "write_oas", "read_oas"]


def gdstk_available() -> bool:
    """Return whether the optional gdstk serializer can be imported."""
    try:
        importlib.import_module("gdstk")
    except (ImportError, OSError):
        return False
    return True


def _gdstk():
    try:
        return importlib.import_module("gdstk")
    except ImportError as exc:
        raise ImportError(
            "GDSII/OASIS I/O requires gdstk. Install it with `pip install photonix[layout]`."
        ) from exc


def _to_gdstk(cell: Cell, lib, built: dict):
    gdstk = _gdstk()

    if cell.name in built:
        return built[cell.name]
    gcell = lib.new_cell(cell.name)
    for pts, (layer, dt) in cell.polygons:
        polygon_points = cast(Sequence[tuple[float, float] | complex], np.asarray(pts))
        gcell.add(gdstk.Polygon(polygon_points, layer=layer, datatype=dt))
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


def write_gds(cell: Cell, path: str | os.PathLike[str]) -> str:
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
    gdstk = _gdstk()
    target = os.fspath(path)
    lib = gdstk.Library()
    _to_gdstk(cell, lib, {})
    lib.write_gds(target)
    return target


def _from_gdstk(lib) -> Cell:
    """Convert the top cell of a gdstk library to a flattened Photonix cell."""
    tops = lib.top_level()
    gcell = cast(Any, tops[0] if tops else lib.cells[0])
    out = Cell(gcell.name)
    for poly in gcell.get_polygons():
        out.add_polygon(np.asarray(poly.points), (int(poly.layer), int(poly.datatype)))
    for lbl in getattr(gcell, "labels", []):
        out.add_port(str(lbl.text), tuple(np.asarray(lbl.origin)))
    return out


def read_gds(path: str | os.PathLike[str]) -> Cell:
    """Read a GDSII file into a (best-effort) :class:`Cell` of the top cell.

    Polygons and labels (as ports) of the top cell are imported; nested
    references are flattened into polygons.
    """
    return _from_gdstk(_gdstk().read_gds(os.fspath(path)))


def write_oas(
    cell: Cell,
    path: str | os.PathLike[str],
    *,
    compression_level: int = 6,
    validation: str | None = None,
) -> str:
    """Write ``cell`` and its reference tree to an OASIS stream.

    ``validation`` may be ``None``, ``"crc32"``, or ``"checksum32"``.
    OASIS support is provided by the optional ``gdstk`` layout dependency.
    """
    if not isinstance(compression_level, int) or isinstance(compression_level, bool):
        raise ValueError("compression_level must be an integer from 0 to 9")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer from 0 to 9")
    if validation not in (None, "crc32", "checksum32"):
        raise ValueError("validation must be None, 'crc32', or 'checksum32'")
    gdstk = _gdstk()
    target = os.fspath(path)
    lib = gdstk.Library()
    _to_gdstk(cell, lib, {})
    lib.write_oas(target, compression_level=compression_level, validation=validation)
    return target


def read_oas(path: str | os.PathLike[str]) -> Cell:
    """Read an OASIS stream into a flattened Photonix top-level :class:`Cell`."""
    return _from_gdstk(_gdstk().read_oas(os.fspath(path)))
