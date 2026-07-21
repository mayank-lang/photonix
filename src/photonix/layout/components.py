"""Parametric layout generators returning :class:`~photonix.layout.Cell`s.

Geometry conventions match the component models: ports are named ``o1``, ``o2``,
... Lengths in µm. Default layer is the waveguide layer ``(1, 0)``.
"""
from __future__ import annotations

import numpy as np

from .cell import Cell

__all__ = ["straight", "bend_circular", "taper", "ring", "mmi1x2", "grating_coupler"]

WG = (1, 0)


def straight(length: float = 10.0, width: float = 0.5, layer=WG) -> Cell:
    """Straight waveguide rectangle with ports ``o1`` (left) and ``o2`` (right)."""
    c = Cell("straight")
    h = width / 2
    c.add_polygon([(0, -h), (length, -h), (length, h), (0, h)], layer)
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    c.add_port("o2", (length, 0.0), 0.0, width, layer)
    return c


def bend_circular(radius: float = 5.0, angle: float = 90.0, width: float = 0.5, layer=WG, npts: int = 60) -> Cell:
    """Circular 90°-style bend as a swept annulus; ports ``o1`` and ``o2``."""
    c = Cell("bend")
    th = np.deg2rad(np.linspace(0, angle, npts))
    h = width / 2
    # center of curvature at (0, radius); start at origin heading +x
    cx, cy = 0.0, radius
    outer = np.stack([(radius + h) * np.sin(th) + cx, cy - (radius + h) * np.cos(th)], axis=1)
    inner = np.stack([(radius - h) * np.sin(th) + cx, cy - (radius - h) * np.cos(th)], axis=1)
    poly = np.concatenate([outer, inner[::-1]], axis=0)
    c.add_polygon(poly, layer)
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    end = (radius * np.sin(th[-1]), cy - radius * np.cos(th[-1]))
    c.add_port("o2", (float(end[0]), float(end[1])), float(angle), width, layer)
    return c


def taper(length: float = 10.0, width1: float = 0.5, width2: float = 1.0, layer=WG) -> Cell:
    """Linear taper from ``width1`` (o1) to ``width2`` (o2)."""
    c = Cell("taper")
    c.add_polygon(
        [(0, -width1 / 2), (length, -width2 / 2), (length, width2 / 2), (0, width1 / 2)], layer
    )
    c.add_port("o1", (0.0, 0.0), 180.0, width1, layer)
    c.add_port("o2", (length, 0.0), 0.0, width2, layer)
    return c


def ring(radius: float = 10.0, width: float = 0.5, layer=WG, npts: int = 200) -> Cell:
    """Closed ring (annulus) of given ``radius``. No optical ports (closed loop)."""
    c = Cell("ring")
    th = np.linspace(0, 2 * np.pi, npts)
    h = width / 2
    outer = np.stack([(radius + h) * np.cos(th), (radius + h) * np.sin(th)], axis=1)
    inner = np.stack([(radius - h) * np.cos(th), (radius - h) * np.sin(th)], axis=1)
    c.add_polygon(np.concatenate([outer, inner[::-1]], axis=0), layer)
    return c


def mmi1x2(length: float = 4.0, mmi_width: float = 2.0, wg_width: float = 0.5, gap: float = 0.5, layer=WG) -> Cell:
    """Schematic 1x2 MMI body with one input (o1) and two outputs (o2, o3)."""
    c = Cell("mmi1x2")
    h = mmi_width / 2
    c.add_polygon([(0, -h), (length, -h), (length, h), (0, h)], layer)
    off = (gap + wg_width) / 2
    c.add_port("o1", (0.0, 0.0), 180.0, wg_width, layer)
    c.add_port("o2", (length, off), 0.0, wg_width, layer)
    c.add_port("o3", (length, -off), 0.0, wg_width, layer)
    return c


def grating_coupler(length: float = 20.0, width: float = 12.0, period: float = 0.63, ff: float = 0.5, layer=WG) -> Cell:
    """Schematic grating coupler footprint with a single port ``o1``."""
    c = Cell("grating_coupler")
    n = max(int(length / period), 1)
    for i in range(n):
        x0 = i * period
        c.add_polygon([(x0, -width / 2), (x0 + ff * period, -width / 2),
                       (x0 + ff * period, width / 2), (x0, width / 2)], layer)
    c.add_port("o1", (length, 0.0), 0.0, 0.5, layer)
    return c
