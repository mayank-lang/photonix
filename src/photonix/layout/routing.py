"""Simple manhattan routing between two ports."""
from __future__ import annotations

import numpy as np

from .cell import Cell, Port

__all__ = ["route"]


def _path_polygon(points, width: float):
    """Build a filled polygon for a centerline path of constant ``width``."""
    pts = np.asarray(points, dtype=float)
    left, right = [], []
    h = width / 2
    for i in range(len(pts)):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == len(pts) - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        d = d / (np.linalg.norm(d) + 1e-12)
        nrm = np.array([-d[1], d[0]])
        left.append(pts[i] + h * nrm)
        right.append(pts[i] - h * nrm)
    return np.array(left + right[::-1])


def route(port1: Port, port2: Port, *, width: float | None = None, layer=(1, 0)) -> Cell:
    """Route an L/Z manhattan waveguide between two ports.

    Produces a :class:`Cell` whose polygon connects ``port1.center`` to
    ``port2.center`` via a midpoint dogleg. Ports ``o1``/``o2`` mirror the inputs.

    Examples
    --------
    >>> from photonix.layout import Port, route
    >>> c = route(Port("a", (0, 0), 0), Port("b", (20, 10), 180))
    >>> len(c.polygons)
    1
    """
    w = width or port1.width
    p1 = np.asarray(port1.center, dtype=float)
    p2 = np.asarray(port2.center, dtype=float)
    mid_x = (p1[0] + p2[0]) / 2.0
    points = [p1, (mid_x, p1[1]), (mid_x, p2[1]), p2]
    c = Cell("route")
    c.add_polygon(_path_polygon(points, w), layer)
    c.add_port("o1", tuple(p1), port1.orientation, w, layer)
    c.add_port("o2", tuple(p2), port2.orientation, w, layer)
    return c
