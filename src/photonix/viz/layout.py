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

    polys = getattr(cell, "polygons", cell)
    # cell.polygons may be a callable returning the list
    if callable(polys):
        polys = polys()
    for poly in polys:
        pts = np.asarray(poly)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.add_patch(MplPolygon(pts, closed=True, alpha=0.6, edgecolor="k", linewidth=0.5))

    ports = getattr(cell, "ports", None)
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
