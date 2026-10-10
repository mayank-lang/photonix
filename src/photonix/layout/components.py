"""Parametric layout generators returning :class:`~photonix.layout.Cell`s.

Geometry conventions match the component models: optical ports are named
``o1``, ``o2``, ... and electrical ports ``e1``, ``e2``, ... Lengths in µm.
Default layer is the waveguide layer ``(1, 0)``.

Every generator records the compact model its geometry represents
(:attr:`Cell.model`) together with the geometry-derived settings
(:attr:`Cell.model_settings`), so netlist extraction simulates what was drawn.
Settings that geometry alone cannot determine (a grating's index contrast, a
heater's ``P_pi``) are left to the model's defaults or the PDK; add them with
``cell.model_settings.update(...)``.
"""
from __future__ import annotations

import math

import numpy as np

from ._geometry import arc_curve, euler_curve, path_polygon, polyline_length, sbend_curve
from .cell import ELECTRICAL, Cell

__all__ = [
    "straight", "bend_circular", "bend_euler", "sbend", "taper", "ring",
    "ring_single", "ring_double", "directional_coupler", "coupler",
    "mmi1x2", "mmi2x2", "crossing", "grating_coupler", "edge_coupler",
    "bragg_grating", "heater_straight", "pn_modulator", "pad", "mzi",
]

WG = (1, 0)
SLAB = (2, 0)
METAL = (11, 0)
N_IMPLANT = (20, 0)
P_IMPLANT = (21, 0)
FIBER_PORT = (203, 0)


def straight(length: float = 10.0, width: float = 0.5, layer=WG) -> Cell:
    """Straight waveguide rectangle with ports ``o1`` (left) and ``o2`` (right)."""
    c = Cell("straight", model="straight", model_settings={"length": float(length)})
    h = width / 2
    c.add_polygon([(0, -h), (length, -h), (length, h), (0, h)], layer)
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    c.add_port("o2", (length, 0.0), 0.0, width, layer)
    return c


def _curve_cell(name, curve, width, layer, model, settings) -> Cell:
    pts, hd = curve
    c = Cell(name, model=model, model_settings=settings)
    c.add_polygon(path_polygon(pts, width, hd), layer)
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    end = pts[-1]
    c.add_port("o2", (float(end[0]), float(end[1])), float(np.rad2deg(hd[-1])) % 360.0, width, layer)
    return c


def bend_circular(radius: float = 5.0, angle: float = 90.0, width: float = 0.5, layer=WG, npts: int = 60) -> Cell:
    """Circular bend starting at the origin heading +x; ports ``o1`` and ``o2``.

    Positive ``angle`` turns counter-clockwise, negative clockwise.
    """
    return _curve_cell("bend", arc_curve(radius, angle, npts), width, layer, "bend",
                       {"radius": float(radius), "angle": float(angle)})


def bend_euler(radius: float = 5.0, angle: float = 90.0, p: float = 0.5, width: float = 0.5,
               layer=WG, npts: int = 80) -> Cell:
    """Euler (clothoid) bend with minimum radius ``radius``; ports ``o1``/``o2``.

    Examples
    --------
    >>> c = bend_euler(radius=5.0, angle=90.0, p=1.0)
    >>> x, y = c.ports["o2"].center
    >>> bool(abs(x - y) < 1e-6 and x > 5.0)     # symmetric, larger footprint than an arc
    True
    """
    return _curve_cell("bend_euler", euler_curve(radius, angle, p, npts), width, layer, "bend_euler",
                       {"radius": float(radius), "angle": float(angle), "p": float(p)})


def sbend(length: float = 10.0, offset: float = 2.0, width: float = 0.5, layer=WG, npts: int = 60) -> Cell:
    """Cosine S-bend moving ``offset`` µm sideways over ``length`` µm.

    Simulated as a straight waveguide of the drawn path length. The minimum
    radius of curvature is ``2 length^2 / (pi^2 |offset|)``.
    """
    pts, hd = sbend_curve(length, offset, npts)
    c = _curve_cell("sbend", (pts, hd), width, layer, "straight", {"length": polyline_length(pts)})
    # A fine resample gives the converged arc length independent of npts.
    fine, _ = sbend_curve(length, offset, 4000)
    c.model_settings["length"] = polyline_length(fine)
    return c


def taper(length: float = 10.0, width1: float = 0.5, width2: float = 1.0, layer=WG) -> Cell:
    """Linear taper from ``width1`` (o1) to ``width2`` (o2)."""
    c = Cell("taper", model="taper",
             model_settings={"length": float(length), "width1": float(width1), "width2": float(width2)})
    c.add_polygon(
        [(0, -width1 / 2), (length, -width2 / 2), (length, width2 / 2), (0, width1 / 2)], layer
    )
    c.add_port("o1", (0.0, 0.0), 180.0, width1, layer)
    c.add_port("o2", (length, 0.0), 0.0, width2, layer)
    return c


def _annulus(radius: float, width: float, npts: int, center=(0.0, 0.0)) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, npts)
    h = width / 2
    outer = np.stack([(radius + h) * np.cos(th), (radius + h) * np.sin(th)], axis=1)
    inner = np.stack([(radius - h) * np.cos(th), (radius - h) * np.sin(th)], axis=1)
    return np.concatenate([outer, inner[::-1]], axis=0) + np.asarray(center)


def _racetrack(radius: float, length_x: float, width: float, npts: int, center) -> np.ndarray:
    """Closed racetrack centre-line polygon band (straights along x)."""
    if length_x <= 0:
        return _annulus(radius, width, npts, center)
    half = length_x / 2
    th_r = np.linspace(-np.pi / 2, np.pi / 2, npts // 2)
    th_l = np.linspace(np.pi / 2, 3 * np.pi / 2, npts // 2)
    cl = np.concatenate([
        np.stack([half + radius * np.cos(th_r), radius * np.sin(th_r)], axis=1),
        np.stack([-half + radius * np.cos(th_l), radius * np.sin(th_l)], axis=1),
    ])
    cl = np.concatenate([cl, cl[:1]])
    poly = path_polygon(cl, width)
    return poly + np.asarray(center)


def ring(radius: float = 10.0, width: float = 0.5, layer=WG, npts: int = 200) -> Cell:
    """Closed ring (annulus) of given ``radius``. No optical ports (closed loop)."""
    c = Cell("ring")
    c.add_polygon(_annulus(radius, width, npts), layer)
    return c


def ring_single(radius: float = 10.0, gap: float = 0.2, length_x: float = 0.0, width: float = 0.5,
                layer=WG, npts: int = 240) -> Cell:
    """All-pass ring or racetrack above a bus waveguide; ports ``o1``/``o2`` on the bus.

    Simulated with :func:`photonix.components.ring_single`, which derives the
    coupling from ``gap``, ``radius`` and ``length_x``.
    """
    c = Cell("ring_single", model="ring_single",
             model_settings={"radius": float(radius), "gap": float(gap), "length_x": float(length_x)})
    half_bus = radius + length_x / 2 + 2.0
    h = width / 2
    c.add_polygon([(-half_bus, -h), (half_bus, -h), (half_bus, h), (-half_bus, h)], layer)
    c.add_polygon(_racetrack(radius, length_x, width, npts, (0.0, width + gap + radius)), layer)
    c.add_port("o1", (-half_bus, 0.0), 180.0, width, layer)
    c.add_port("o2", (half_bus, 0.0), 0.0, width, layer)
    return c


def ring_double(radius: float = 10.0, gap: float = 0.2, gap2: float | None = None, length_x: float = 0.0,
                width: float = 0.5, layer=WG, npts: int = 240) -> Cell:
    """Add-drop ring between two buses.

    Ports: ``o1`` input and ``o2`` through on the lower bus; ``o3`` drop and
    ``o4`` add on the upper bus (``o3`` is above ``o1``), matching
    :func:`photonix.components.add_drop_ring`.
    """
    g2 = gap if gap2 is None else gap2
    c = Cell("ring_double", model="ring_double",
             model_settings={"radius": float(radius), "gap": float(gap), "gap2": float(g2),
                             "length_x": float(length_x)})
    half_bus = radius + length_x / 2 + 2.0
    h = width / 2
    yc = width + gap + radius
    y2 = yc + radius + g2 + width
    for y in (0.0, y2):
        c.add_polygon([(-half_bus, y - h), (half_bus, y - h), (half_bus, y + h), (-half_bus, y + h)], layer)
    c.add_polygon(_racetrack(radius, length_x, width, npts, (0.0, yc)), layer)
    c.add_port("o1", (-half_bus, 0.0), 180.0, width, layer)
    c.add_port("o2", (half_bus, 0.0), 0.0, width, layer)
    c.add_port("o3", (-half_bus, y2), 180.0, width, layer)
    c.add_port("o4", (half_bus, y2), 0.0, width, layer)
    return c


def directional_coupler(
    length: float = 10.0,
    gap: float = 0.2,
    width: float = 0.5,
    layer=WG,
) -> Cell:
    """Schematic straight-arm directional coupler with canonical four ports.

    Simulated with :func:`photonix.components.evanescent_coupler` from the
    drawn ``gap`` and ``length``.
    """
    c = Cell("directional_coupler", model="evanescent_coupler",
             model_settings={"gap": float(gap), "length": float(length)})
    offset = (gap + width) / 2.0
    half_width = width / 2.0
    for y in (offset, -offset):
        c.add_polygon(
            [(0, y - half_width), (length, y - half_width),
             (length, y + half_width), (0, y + half_width)],
            layer,
        )
    c.add_port("o1", (0.0, offset), 180.0, width, layer)
    c.add_port("o2", (0.0, -offset), 180.0, width, layer)
    c.add_port("o4", (length, offset), 0.0, width, layer)
    c.add_port("o3", (length, -offset), 0.0, width, layer)
    return c


def coupler(length: float = 10.0, gap: float = 0.2, dx: float = 10.0, dy: float = 4.0,
            width: float = 0.5, layer=WG, npts: int = 60) -> Cell:
    """Symmetric directional coupler with cosine S-bend arms.

    The straight coupling section is ``length`` long; the arms fan out over
    ``dx`` to a port pitch of ``dy`` on each side. The extra coupling in the
    S-bends uses their curvature at the junction (``R = 2 dx^2 / (pi^2 a)``
    for an arm offset ``a``), and their path length is included in the phase.

    Examples
    --------
    >>> c = coupler(length=5.0, gap=0.2, dx=10.0, dy=4.0)
    >>> round(c.ports["o1"].center[1] - c.ports["o2"].center[1], 9)
    4.0
    """
    arm_offset = 0.5 * (dy - gap - width)
    if arm_offset <= 0:
        raise ValueError("dy must exceed gap + width")
    fine, _ = sbend_curve(dx, arm_offset, 4000)
    r_eq = 2.0 * dx * dx / (math.pi ** 2 * arm_offset)
    c = Cell("coupler", model="evanescent_coupler",
             model_settings={"gap": float(gap), "length": float(length), "bend_radius": float(r_eq),
                             "curved_arms": 2, "extra_length": 2.0 * polyline_length(fine)})
    y0 = 0.5 * (gap + width)
    yp = y0 + arm_offset
    xr = 2.0 * dx + length
    for sgn in (1.0, -1.0):
        # in: port -> coupling section; out: coupling section -> port
        p_in, h_in = sbend_curve(dx, -sgn * arm_offset, npts)
        p_out, h_out = sbend_curve(dx, sgn * arm_offset, npts)
        p_in = p_in + np.array([0.0, sgn * yp])
        p_out = p_out + np.array([dx + length, sgn * y0])
        if length > 0:
            cl = np.concatenate([p_in, p_out])
            hh = np.concatenate([h_in, h_out])
        else:
            cl = np.concatenate([p_in, p_out[1:]])
            hh = np.concatenate([h_in, h_out[1:]])
        c.add_polygon(path_polygon(cl, width, hh), layer)
    c.add_port("o1", (0.0, yp), 180.0, width, layer)
    c.add_port("o2", (0.0, -yp), 180.0, width, layer)
    c.add_port("o3", (xr, -yp), 0.0, width, layer)
    c.add_port("o4", (xr, yp), 0.0, width, layer)
    return c


def mmi1x2(length: float = 4.0, mmi_width: float = 2.0, wg_width: float = 0.5, gap: float = 0.5, layer=WG) -> Cell:
    """Schematic 1x2 MMI body with one input (o1) and two outputs (o2 top, o3 bottom)."""
    c = Cell("mmi1x2", model="mmi1x2")
    h = mmi_width / 2
    c.add_polygon([(0, -h), (length, -h), (length, h), (0, h)], layer)
    off = (gap + wg_width) / 2
    c.add_port("o1", (0.0, 0.0), 180.0, wg_width, layer)
    c.add_port("o2", (length, off), 0.0, wg_width, layer)
    c.add_port("o3", (length, -off), 0.0, wg_width, layer)
    return c


def mmi2x2(length: float = 6.0, mmi_width: float = 2.5, wg_width: float = 0.5, gap: float = 0.5, layer=WG) -> Cell:
    """Schematic 2x2 MMI with ports ``o1``/``o2`` (left) and ``o3``/``o4`` (right)."""
    c = Cell("mmi2x2", model="mmi2x2")
    h = mmi_width / 2
    c.add_polygon([(0, -h), (length, -h), (length, h), (0, h)], layer)
    off = (gap + wg_width) / 2
    c.add_port("o1", (0.0, off), 180.0, wg_width, layer)
    c.add_port("o2", (0.0, -off), 180.0, wg_width, layer)
    c.add_port("o3", (length, -off), 0.0, wg_width, layer)
    c.add_port("o4", (length, off), 0.0, wg_width, layer)
    return c


def crossing(arm_length: float = 3.0, width: float = 0.5, layer=WG) -> Cell:
    """Plus-shaped waveguide crossing; ports ``o1`` W, ``o2`` N, ``o3`` E, ``o4`` S."""
    c = Cell("crossing", model="crossing")
    a, h = arm_length, width / 2
    c.add_polygon([
        (-a, -h), (-h, -h), (-h, -a), (h, -a), (h, -h), (a, -h),
        (a, h), (h, h), (h, a), (-h, a), (-h, h), (-a, h),
    ], layer)
    c.add_port("o1", (-a, 0.0), 180.0, width, layer)
    c.add_port("o2", (0.0, a), 90.0, width, layer)
    c.add_port("o3", (a, 0.0), 0.0, width, layer)
    c.add_port("o4", (0.0, -a), 270.0, width, layer)
    return c


def grating_coupler(length: float = 20.0, width: float = 12.0, period: float = 0.63, ff: float = 0.5,
                    layer=WG) -> Cell:
    """Schematic grating coupler footprint.

    ``o1`` is the waveguide port; ``o2`` marks the fiber at the grating centre
    on layer ``(203, 0)`` so it surfaces as an external circuit port.
    """
    c = Cell("grating_coupler", model="grating_coupler")
    n = max(int(length / period), 1)
    for i in range(n):
        x0 = i * period
        c.add_polygon([(x0, -width / 2), (x0 + ff * period, -width / 2),
                       (x0 + ff * period, width / 2), (x0, width / 2)], layer)
    c.add_port("o1", (length, 0.0), 0.0, 0.5, layer)
    c.add_port("o2", (length / 2, 0.0), 90.0, 10.4, FIBER_PORT)
    return c


def edge_coupler(length: float = 200.0, width: float = 0.5, tip_width: float = 0.18, layer=WG) -> Cell:
    """Inverse-taper edge coupler: ``o1`` on the waveguide side, ``o2`` at the facet tip."""
    c = Cell("edge_coupler", model="edge_coupler")
    c.add_polygon([(0, -width / 2), (length, -tip_width / 2), (length, tip_width / 2), (0, width / 2)], layer)
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    c.add_port("o2", (length, 0.0), 0.0, tip_width, layer)
    return c


def bragg_grating(period: float = 0.3229, n_periods: int = 300, width: float = 0.5, dw: float = 0.05,
                  layer=WG) -> Cell:
    """Sidewall-corrugated Bragg grating (50 % duty, ``width +/- dw/2``).

    The model records ``period`` and ``n_periods``; the effective-index
    contrast of the corrugation must come from a mode solve or the PDK
    (``cell.model_settings["dn"] = ...``).
    """
    c = Cell("bragg_grating", model="bragg_grating",
             model_settings={"period": float(period), "n_periods": int(n_periods)})
    top = []
    for i in range(n_periods):
        x0 = i * period
        wide, narrow = (width + dw) / 2, (width - dw) / 2
        top += [(x0, wide), (x0 + period / 2, wide), (x0 + period / 2, narrow), (x0 + period, narrow)]
    top_arr = np.asarray(top)
    bottom = top_arr[::-1] * np.array([1.0, -1.0])
    c.add_polygon(np.concatenate([top_arr, bottom]), layer)
    length = period * n_periods
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    c.add_port("o2", (length, 0.0), 0.0, width, layer)
    return c


def heater_straight(length: float = 100.0, width: float = 0.5, heater_width: float = 2.0,
                    layer=WG, heater_layer=METAL) -> Cell:
    """Waveguide with a resistive heater strip; electrical ports ``e1``/``e2``.

    Simulated with :func:`photonix.components.heater_phase_shifter` of the
    drawn ``length``.
    """
    c = Cell("heater_straight", model="heater_phase_shifter", model_settings={"length": float(length)})
    h, hh = width / 2, heater_width / 2
    c.add_polygon([(0, -h), (length, -h), (length, h), (0, h)], layer)
    c.add_polygon([(0, -hh), (length, -hh), (length, hh), (0, hh)], heater_layer)
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    c.add_port("o2", (length, 0.0), 0.0, width, layer)
    c.add_port("e1", (0.0, hh), 90.0, heater_width, heater_layer, ELECTRICAL)
    c.add_port("e2", (length, hh), 90.0, heater_width, heater_layer, ELECTRICAL)
    return c


def pn_modulator(length: float = 2000.0, width: float = 0.5, slab_width: float = 6.0, layer=WG,
                 slab_layer=SLAB) -> Cell:
    """Rib waveguide with lateral p (bottom) / n (top) implants; ports o1/o2, e1 (p), e2 (n)."""
    c = Cell("pn_modulator", model="pn_phase_shifter", model_settings={"length": float(length)})
    h, s = width / 2, slab_width / 2
    c.add_polygon([(0, -h), (length, -h), (length, h), (0, h)], layer)
    c.add_polygon([(0, -s), (length, -s), (length, s), (0, s)], slab_layer)
    c.add_polygon([(0, -s), (length, -s), (length, 0), (0, 0)], P_IMPLANT)
    c.add_polygon([(0, 0), (length, 0), (length, s), (0, s)], N_IMPLANT)
    c.add_port("o1", (0.0, 0.0), 180.0, width, layer)
    c.add_port("o2", (length, 0.0), 0.0, width, layer)
    c.add_port("e1", (length / 2, -s), 270.0, length, slab_layer, ELECTRICAL)
    c.add_port("e2", (length / 2, s), 90.0, length, slab_layer, ELECTRICAL)
    return c


def pad(size: float = 80.0, layer=METAL) -> Cell:
    """Square bond pad centred at the origin with electrical port ``e1``."""
    c = Cell("pad")
    h = size / 2
    c.add_polygon([(-h, -h), (h, -h), (h, h), (-h, h)], layer)
    c.add_port("e1", (0.0, 0.0), 90.0, size, layer, ELECTRICAL)
    return c


def mzi(delta_length: float = 20.0, length: float = 10.0, radius: float = 5.0, arm_spacing: float = 10.0,
        width: float = 0.5, layer=WG) -> Cell:
    """Hierarchical Mach-Zehnder interferometer built from placed sub-cells.

    An ``mmi1x2`` splitter and combiner joined by two arms of bends and
    straights; the top arm is ``delta_length`` µm longer. The cell carries no
    model of its own, so :func:`~photonix.layout.extract_netlist` descends into
    it and simulates every drawn section. Ports ``o1`` (input) and ``o2``
    (output).

    Examples
    --------
    >>> import photonix as px
    >>> from photonix.layout import Cell, extract_netlist
    >>> nl = extract_netlist(mzi(delta_length=20.0))
    >>> sorted(nl.ports)
    ['o1', 'o2']
    """
    if delta_length < 0:
        raise ValueError("delta_length must be non-negative (the top arm is the long one)")
    c = Cell("mzi")
    split = c.add_ref(mmi1x2(wg_width=width, layer=layer), name="splitter")
    off = split.port("o2").center[1]
    # Horizontal arm sections sit at +/-(off + 2R + v): v_bot from the spacing.
    v_bot = max(arm_spacing / 2 - off - 2 * radius, 0.0)
    v_top = v_bot + delta_length / 2

    def arm(tag: str, start_port, sign: float, vertical: float):
        sections = [("b1", bend_circular(radius, 90 * sign, width, layer)),
                    ("s1", straight(vertical, width, layer) if vertical > 0 else None),
                    ("b2", bend_circular(radius, -90 * sign, width, layer)),
                    ("s2", straight(length, width, layer) if length > 0 else None),
                    ("b3", bend_circular(radius, -90 * sign, width, layer)),
                    ("s3", straight(vertical, width, layer) if vertical > 0 else None),
                    ("b4", bend_circular(radius, 90 * sign, width, layer))]
        port = start_port
        for name, cell in sections:
            if cell is None:
                continue
            ref = c.add_ref(cell, name=f"{tag}_{name}").connect("o1", port)
            port = ref.port("o2")
        return port

    top_end = arm("top", split.port("o2"), +1.0, v_top)
    bot_end = arm("bot", split.port("o3"), -1.0, v_bot)
    comb = c.add_ref(mmi1x2(wg_width=width, layer=layer), name="combiner").connect("o3", top_end)
    if np.linalg.norm(np.subtract(comb.port("o2").center, bot_end.center)) > 1e-6:
        raise RuntimeError("internal error: MZI arms do not close")  # pragma: no cover
    c.expose_port(split, "o1", "o1")
    c.expose_port(comb, "o1", "o2")
    return c

