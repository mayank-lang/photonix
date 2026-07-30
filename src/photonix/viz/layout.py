"""Render layout cells (polygons + ports)."""
from __future__ import annotations

__all__ = ["plot_cell"]


def plot_cell(cell, *, ax=None, show_ports: bool = True):
    """Render a layout :class:`~photonix.layout.Cell` (or polygon list).

    Accepts any object exposing a ``polygons`` attribute/property (an iterable of
    Nx2 coordinate arrays) and optionally ``ports`` (objects with ``center`` and
    ``orientation``). A bare list of Nx2 arrays is also accepted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    get_polygons = getattr(cell, "get_polygons", None)
    polys = get_polygons() if callable(get_polygons) else getattr(cell, "polygons", cell)
    # cell.polygons may be a callable returning the list
    if callable(polys):
        polys = polys()
    for poly in polys:
        # Native Cell polygons are stored as (points, layer) records.
        if isinstance(poly, tuple) and len(poly) == 2:
            poly = poly[0]
        pts = np.asarray(poly)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.add_patch(MplPolygon(pts, closed=True, alpha=0.6, edgecolor="k", linewidth=0.5))

    get_ports = getattr(cell, "get_ports", None)
    ports = get_ports() if callable(get_ports) else getattr(cell, "ports", None)
    if show_ports and ports:
        items = ports.values() if hasattr(ports, "values") else ports
        for prt in items:
            c = np.asarray(getattr(prt, "center", (0, 0)))
            ax.plot(c[0], c[1], "r>", markersize=6)
            name = getattr(prt, "name", "")
            ax.annotate(str(name), (c[0], c[1]), fontsize=7, color="r")

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    return ax
