"""S-bend routes between parallel, facing ports, and closed-form S-bend geometry.

Three S-bend shapes join two ports that face each other across an axial span
``L`` with a lateral offset ``D``:

``"cosine"``
    ``y = (D/2) (1 - cos(pi x / L))``. Its curvature peaks at the two ends,
    ``kappa_max = pi^2 |D| / (2 L^2)``, so a minimum radius ``R`` needs
    ``L >= pi sqrt(R |D| / 2)``. The curvature jumps from 0 to ``kappa_max`` at
    the joints. The centre-line length is the complete elliptic integral
    ``(2 L / pi) E(-A^2)`` with ``A = pi D / (2 L)``.
``"circular"``
    Two opposite circular arcs of angle ``theta = 2 atan(|D| / L)`` and radius
    ``R = (L^2 + D^2) / (4 |D|)``; length ``2 R theta``.
``"euler"`` (default for routing)
    Two opposite Euler (clothoid) bends of the same ``theta`` with minimum
    radius ``R = L / (2 T_1(theta) (1 + cos theta))``, where ``T_1`` is the
    tangent length of a unit-radius Euler bend; length ``2 R theta (1 + p)``.
    Curvature is continuous everywhere, which avoids the junction mismatch of
    the other two shapes.

The two-bend shapes follow from one polyline identity: back-to-back bends of
tangent length ``T`` turning ``+theta`` then ``-theta`` advance
``2 T (1 + cos theta)`` and offset ``2 T sin theta``, so
``tan(theta / 2) = |D| / L`` for *every* bend profile.

:func:`sbend_arc_length` and :func:`sbend_min_radius` use the backend array
module and are differentiable under JAX (the cosine length by an exponentially
convergent trapezoid rule over one period, the Euler tangent by 32-point
Gauss-Legendre quadrature of the clothoid). :func:`route_sbend` draws the
route; like every router it returns a ``model="route"`` cell recording the
exact centre-line ``length`` and ``n_bends`` (an S-bend counts as two bends).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from photonix.core.backend import xp

from ._geometry import place_curve, sbend_curve
from ._route_common import (
    TOL,
    fillet_path,
    join_pieces,
    make_bend,
    resolve_style,
    rot90,
    route_cell,
    same_angle,
)
from .cell import Cell, Port
from .routing import RoutingError

__all__ = ["route_sbend", "sbend_arc_length", "sbend_min_radius", "sbend_min_span"]

_KINDS = ("cosine", "circular", "euler")
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
_TRAPZ_N = 256


def _check_kind(kind: str, p: float) -> None:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"Euler fraction p must lie in [0, 1], got {p!r}")


def _euler_unit_end(theta, p: float):
    """End point ``(X, Y)`` of an Euler bend of unit minimum radius turning ``theta`` (radians).

    Ramp ``s in [0, L_c]`` has heading ``s^2 / (2 L_c)`` with ``L_c = p theta``;
    the circular middle is integrated in closed form and the second ramp is
    the first one mirrored about the bend bisector.
    """
    if p == 0.0:
        return xp.sin(theta), 1.0 - xp.cos(theta)
    xg = xp.asarray(_GL_X)
    wg = xp.asarray(_GL_W)
    lc = p * theta
    phi1 = lc * (xg + 1.0) ** 2 / 8.0          # heading at the Gauss nodes of ramp 1
    a0 = 0.5 * p * theta
    a1 = theta - a0
    x_ramp = 0.5 * lc * (xp.sum(wg * xp.cos(phi1)) + xp.sum(wg * xp.cos(theta - phi1)))
    y_ramp = 0.5 * lc * (xp.sum(wg * xp.sin(phi1)) + xp.sum(wg * xp.sin(theta - phi1)))
    return x_ramp + xp.sin(a1) - xp.sin(a0), y_ramp + xp.cos(a0) - xp.cos(a1)


def _euler_unit_tangent(theta, p: float):
    """Tangent length ``T_1(theta)`` of a unit-radius Euler bend (start to corner)."""
    _x, y = _euler_unit_end(theta, p)
    return y / xp.sin(theta)


def sbend_arc_length(length=10.0, offset=2.0, *, kind: str = "cosine", p: float = 0.5):
    """Centre-line length of an S-bend spanning ``length`` axially with lateral ``offset`` (µm).

    Differentiable in ``length`` and ``offset`` under JAX. ``offset = 0`` gives
    ``length``.

    Parameters
    ----------
    length, offset
        Axial span ``L`` and lateral offset ``D`` of the S-bend.
    kind
        ``"cosine"``, ``"circular"`` or ``"euler"`` (see the module docstring).
    p
        Euler clothoid fraction (``kind="euler"`` only).

    Examples
    --------
    >>> import math
    >>> from scipy.special import ellipe
    >>> L, D = 30.0, 3.0
    >>> A = math.pi * D / (2 * L)
    >>> bool(abs(float(sbend_arc_length(L, D)) - 2 * L / math.pi * ellipe(-A * A)) < 1e-12)
    True
    >>> th = 2 * math.atan(D / L)                     # circular: two arcs of radius (L^2 + D^2) / (4 D)
    >>> abs(float(sbend_arc_length(L, D, kind="circular")) - (L * L + D * D) / (2 * D) * th) < 1e-12
    True
    """
    _check_kind(kind, p)
    p = float(p)
    L = xp.asarray(length, dtype=float)
    D = xp.abs(xp.asarray(offset, dtype=float))
    nz = D > 0
    Ds = xp.where(nz, D, 1.0)  # safe denominator: keeps gradients finite at offset = 0
    if kind == "cosine":
        t = xp.asarray(np.pi * np.arange(_TRAPZ_N) / _TRAPZ_N)
        a = np.pi * D / (2.0 * L)
        return L * xp.mean(xp.sqrt(1.0 + (a * xp.sin(t)) ** 2))
    theta = 2.0 * xp.arctan(Ds / L)
    if kind == "circular":
        val = (L * L + Ds * Ds) / (2.0 * Ds) * theta
    else:
        radius = L / (2.0 * _euler_unit_tangent(theta, p) * (1.0 + xp.cos(theta)))
        val = 2.0 * radius * theta * (1.0 + p)
    return xp.where(nz, val, L)


def sbend_min_radius(length=10.0, offset=2.0, *, kind: str = "cosine", p: float = 0.5):
    """Minimum radius of curvature of an S-bend spanning ``length`` with lateral ``offset`` (µm).

    ``inf`` for ``offset = 0``. Differentiable in both arguments under JAX.

    Examples
    --------
    >>> import math
    >>> round(float(sbend_min_radius(30.0, 3.0)), 9) == round(2 * 30.0**2 / (math.pi**2 * 3.0), 9)
    True
    >>> float(sbend_min_radius(30.0, 3.0, kind="circular")) == (30.0**2 + 3.0**2) / 12.0
    True
    """
    _check_kind(kind, p)
    p = float(p)
    L = xp.asarray(length, dtype=float)
    D = xp.abs(xp.asarray(offset, dtype=float))
    nz = D > 0
    Ds = xp.where(nz, D, 1.0)
    if kind == "cosine":
        val = 2.0 * L * L / (np.pi**2 * Ds)
    elif kind == "circular":
        val = (L * L + Ds * Ds) / (4.0 * Ds)
    else:
        theta = 2.0 * xp.arctan(Ds / L)
        val = L / (2.0 * _euler_unit_tangent(theta, p) * (1.0 + xp.cos(theta)))
    return xp.where(nz, val, xp.inf)


def sbend_min_span(offset: float = 2.0, min_radius: float = 5.0, *, kind: str = "cosine",
                   p: float = 0.5) -> float:
    """Shortest axial span of an S-bend with lateral ``offset`` whose radius stays >= ``min_radius``.

    Cosine: ``pi sqrt(R |D| / 2)``. Circular: ``sqrt(4 R |D| - D^2)`` for
    ``|D| <= 2R``. Euler: the turn angle solving ``2 R T_1(theta) sin theta = |D|``
    is found by bisection. Two-bend shapes cannot offset by more than two 90-degree
    bends; a :class:`~photonix.layout.RoutingError` says so.

    Examples
    --------
    >>> import math
    >>> abs(sbend_min_span(3.0, 10.0) - math.pi * math.sqrt(10.0 * 3.0 / 2)) < 1e-12
    True
    >>> L = sbend_min_span(3.0, 10.0, kind="euler")
    >>> abs(float(sbend_min_radius(L, 3.0, kind="euler")) - 10.0) < 1e-9
    True
    """
    _check_kind(kind, p)
    D = abs(float(offset))
    R = float(min_radius)
    if R <= 0:
        raise ValueError("min_radius must be positive")
    if D == 0.0:
        return 0.0
    if kind == "cosine":
        return math.pi * math.sqrt(R * D / 2.0)
    if kind == "circular" or p == 0.0:
        if D > 2.0 * R * (1 + 1e-12):
            raise RoutingError(f"a circular S-bend of radius {R:g} offsets at most {2 * R:g} um (asked {D:g})")
        return math.sqrt(max(4.0 * R * D - D * D, 0.0))

    def lateral(th: float) -> float:
        return 2.0 * R * float(_euler_unit_tangent(th, p)) * math.sin(th)

    hi = 0.5 * math.pi
    if lateral(hi) < D * (1 - 1e-12):
        raise RoutingError(f"an Euler S-bend of radius {R:g} offsets at most {lateral(hi):.6g} um (asked {D:g})")
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lateral(mid) < D:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    th = 0.5 * (lo + hi)
    return 2.0 * R * float(_euler_unit_tangent(th, p)) * (1.0 + math.cos(th))


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def _facing_frame(port1: Port, port2: Port, what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """``(start, u, n, axial, lateral)`` for ports that face each other."""
    if not same_angle(port2.orientation, port1.orientation + 180.0):
        raise RoutingError(
            f"{what} needs port2 to face port1 (orientations {port1.orientation:g} and "
            f"{port2.orientation:g} deg are not antiparallel); use route_auto"
        )
    s = np.asarray(port1.center, dtype=float)
    e = np.asarray(port2.center, dtype=float)
    u = port1.direction
    n = rot90(u)
    return s, u, n, float((e - s) @ u), float((e - s) @ n)


def sbend_path(s: np.ndarray, e: np.ndarray, u: np.ndarray, *, kind: str, p: float, min_radius: float,
               start_straight: float, span: float, npts: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Centre line ``(points, headings, length, n_bends)`` of straight + S-bend + straight from ``s`` to ``e``.

    ``u`` is the common heading; the S-bend starts ``start_straight`` after
    ``s`` and spans ``span`` along ``u``. Raises :class:`RoutingError` if the
    S-bend would be tighter than ``min_radius`` or does not fit.
    """
    n = rot90(u)
    axial = float((e - s) @ u)
    D = float((e - s) @ n)
    if start_straight < -TOL or span <= TOL:
        raise RoutingError("the S-bend needs a positive span and a non-negative start straight")
    if start_straight + span > axial + TOL * max(1.0, axial):
        raise RoutingError(
            f"the S-bend ({start_straight:.6g} + {span:.6g} um) does not fit in the {axial:.6g} um axial distance"
        )
    if abs(D) < TOL:
        return np.array([s, e]), np.full(2, math.atan2(u[1], u[0])), axial, 0
    r_shape = float(sbend_min_radius(span, D, kind=kind, p=p))
    if r_shape < min_radius * (1.0 - 1e-12):
        try:
            advice = f"it needs a span of at least {sbend_min_span(D, min_radius, kind=kind, p=p):.6g} um"
        except RoutingError as exc:
            advice = f"{exc}; use route_auto for a Z route"
        raise RoutingError(
            f"a {kind} S-bend offsetting {D:.6g} um over {span:.6g} um has radius {r_shape:.6g} um "
            f"< min_radius {min_radius:g}; {advice}"
        )
    b0 = s + start_straight * u
    heading = math.atan2(u[1], u[0])
    if kind == "cosine":
        m = max(int(npts), 2 * int(math.ceil(npts * math.atan(math.pi * abs(D) / (2 * span)) / (0.5 * math.pi))), 16)
        cp, ch = place_curve(sbend_curve(span, D, m), b0, heading)
        end_s = cp[-1]
        pieces = [(np.array([s]), np.array([heading])), (cp, ch), (np.array([e]), np.array([heading]))]
        pts, hds = join_pieces(pieces)
        pts[0], pts[-1] = s, e
        length = start_straight + float(sbend_arc_length(span, D, kind="cosine")) + float((e - end_s) @ u)
        return pts, hds, length, 2
    if abs(D) > span * (1 + 1e-12):
        raise RoutingError(f"a two-bend S-bend cannot offset {abs(D):.6g} um over only {span:.6g} um")
    theta = 2.0 * math.atan(abs(D) / span)
    unit_bend = make_bend(1.0, theta, kind, p, npts)
    radius = span / ((unit_bend.t_in + unit_bend.t_out) * (1.0 + math.cos(theta)))
    sgn = 1.0 if D > 0 else -1.0
    b1 = make_bend(radius, sgn * theta, kind, p, npts)
    b2 = make_bend(radius, -sgn * theta, kind, p, npts)
    direction = math.cos(theta) * u + sgn * math.sin(theta) * n
    c1 = b0 + b1.t_in * u
    c2 = c1 + (b1.t_out + b2.t_in) * direction
    path = fillet_path([s, c1, c2, e], [radius, radius], bend=kind, p=p, npts=npts, bends=[b1, b2])
    return path.points, path.headings, path.length, path.n_bends


def route_sbend(
    port1: Port,
    port2: Port,
    *,
    min_radius: float | None = None,
    kind: str = "euler",
    p: float = 0.5,
    start_straight: float = 0.0,
    sbend_length: float | None = None,
    width: float | None = None,
    layer: Any = None,
    xs: Any = None,
    npts: int = 48,
) -> Cell:
    """Route an S-bend between two ports that face each other (any common orientation).

    The route leaves ``port1`` along its orientation, runs ``start_straight``
    µm straight, makes an S-bend of axial span ``sbend_length`` (default: all
    of the remaining axial distance, which gives the largest radius), and runs
    straight into ``port2``. Raises :class:`~photonix.layout.RoutingError` if the
    S-bend radius would fall below ``min_radius`` (the message gives the
    shortest span that works) or if the ports do not face each other.

    Parameters
    ----------
    min_radius
        Smallest allowed radius of curvature (default: ``xs.radius`` or 5 µm).
    kind
        ``"euler"`` (continuous curvature, default), ``"circular"`` or
        ``"cosine"`` -- see the module docstring.
    xs
        Optional cross-section (any object with ``width``, ``layer`` and
        ``radius``) supplying defaults for ``width``, ``layer`` and ``min_radius``.

    Returns
    -------
    Cell
        ``model="route"`` with the exact centre-line ``length`` and
        ``n_bends`` (2, or 0 for a straight route).

    Examples
    --------
    >>> from photonix.layout import Port, route_sbend
    >>> r = route_sbend(Port("a", (0, 0), 0), Port("b", (30, 3), 180), min_radius=10.0, kind="cosine")
    >>> r.model_settings["n_bends"], round(r.model_settings["length"], 6)
    (2, 30.184208)
    >>> r.ports["o2"].center
    (30.0, 3.0)
    """
    _check_kind(kind, p)
    w, lay, r_min = resolve_style(port1, width=width, layer=layer, radius=min_radius, xs=xs)
    s, u, _n, axial, _lat = _facing_frame(port1, port2, "route_sbend")
    if axial <= TOL:
        raise RoutingError("port2 must lie ahead of port1 for an S-bend; use route_auto")
    start = float(start_straight)
    span = axial - start if sbend_length is None else float(sbend_length)
    e = np.asarray(port2.center, dtype=float)
    pts, hds, length, n_bends = sbend_path(s, e, u, kind=kind, p=float(p), min_radius=r_min,
                                           start_straight=start, span=span, npts=npts)
    return route_cell(pts, hds, port1, port2, w, lay, length, n_bends, name="route_sbend")
