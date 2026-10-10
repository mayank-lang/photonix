"""Validation of the advanced routers: S-bends, route_auto, bundles, fan-outs, A*, length matching.

Every geometric claim is checked on the *drawn* polygon (centre line recovered
from the two offset sides), against an independent reference: closed forms,
scipy quadrature, Fresnel integrals, gdstk booleans, the DRC pre-check,
Dijkstra's algorithm, and circuit simulation of the extracted layout.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc
from photonix.pdk import CrossSection

W = 0.5


# --------------------------------------------------------------------------- #
# Independent geometry helpers (deliberately not the routers' own code)
# --------------------------------------------------------------------------- #
def _centerline(cell: lay.Cell) -> np.ndarray:
    poly = np.asarray(cell.polygons[0][0])
    n = len(poly) // 2
    return 0.5 * (poly[:n] + poly[n:][::-1])


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _max_curvature(line: np.ndarray) -> float:
    seg = np.diff(line, axis=0)
    ell = np.linalg.norm(seg, axis=1)
    phi = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    kappa = np.abs(np.diff(phi)) / (0.5 * (ell[:-1] + ell[1:]))
    return float(kappa.max()) if len(kappa) else 0.0


def _segments_cross(a0, a1, b0, b1) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return (orient(a0, a1, b0) * orient(a0, a1, b1) < 0) and (orient(b0, b1, a0) * orient(b0, b1, a1) < 0)


def _is_simple(poly: np.ndarray) -> bool:
    """No two non-adjacent edges of the closed polygon properly intersect."""
    n = len(poly)
    edges = [(poly[i], poly[(i + 1) % n]) for i in range(n)]
    a0 = np.array([e[0] for e in edges])
    a1 = np.array([e[1] for e in edges])
    # vectorised proper-intersection test over all edge pairs
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    A0, A1 = a0[:, None], a1[:, None]
    B0, B1 = a0[None, :], a1[None, :]
    cross = (orient(A0, A1, B0) * orient(A0, A1, B1) < -1e-18) & (orient(B0, B1, A0) * orient(B0, B1, A1) < -1e-18)
    idx = np.arange(n)
    adjacent = (np.abs(idx[:, None] - idx[None, :]) <= 1) | (np.abs(idx[:, None] - idx[None, :]) == n - 1)
    return not np.any(cross & ~adjacent)


def _check_route(cell, p1, p2, *, radius=None, width=W, area_rtol=2e-3, kappa_rtol=1e-3):
    """Endpoints on the ports, mating port conventions, constant width, min radius, simple polygon."""
    assert cell.model == "route" and set(cell.model_settings) == {"length", "n_bends"}
    line = _centerline(cell)
    assert np.allclose(line[0], p1.center, atol=1e-9) and np.allclose(line[-1], p2.center, atol=1e-9)
    assert np.allclose(cell.ports["o1"].center, p1.center) and np.allclose(cell.ports["o2"].center, p2.center)
    assert ((cell.ports["o1"].orientation - p1.orientation) % 360) == pytest.approx(180.0)
    assert ((cell.ports["o2"].orientation - p2.orientation) % 360) == pytest.approx(180.0)
    poly = cell.polygons[0][0]
    n = len(poly) // 2
    rung = np.linalg.norm(poly[:n] - poly[n:][::-1], axis=1)
    assert np.allclose(rung, width, atol=1e-9)                       # constant width everywhere
    length = cell.model_settings["length"]
    assert _area(poly) / width == pytest.approx(length, rel=area_rtol)
    if radius is not None:
        assert _max_curvature(line) <= (1.0 + kappa_rtol) / radius
    assert _is_simple(poly)
    return line


# --------------------------------------------------------------------------- #
# S-bends: closed forms, quadrature, curvature
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("L, D", [(30.0, 3.0), (10.0, 6.0), (200.0, 0.4), (5.0, 8.0)])
def test_cosine_sbend_length_matches_quadrature_and_elliptic_integral(L, D):
    from scipy.integrate import quad
    from scipy.special import ellipe

    ref, _err = quad(lambda x: math.sqrt(1 + (math.pi * D / (2 * L) * math.sin(math.pi * x / L)) ** 2),
                     0.0, L, epsabs=1e-13, epsrel=1e-13)
    A = math.pi * D / (2 * L)
    assert float(lay.sbend_arc_length(L, D)) == pytest.approx(ref, abs=1e-9)
    assert float(lay.sbend_arc_length(L, D)) == pytest.approx(2 * L / math.pi * ellipe(-A * A), rel=1e-13)
    assert float(lay.sbend_arc_length(L, -D)) == pytest.approx(ref, abs=1e-9)
    assert float(lay.sbend_arc_length(30.0, 0.0)) == 30.0
    assert float(lay.sbend_arc_length(30.0, 0.0, kind="euler")) == 30.0


def test_euler_tangent_quadrature_matches_fresnel_closed_form():
    """The differentiable Gauss-Legendre clothoid equals the Fresnel-integral solution."""
    from scipy.special import fresnel

    from photonix.layout.routing_sbend import _euler_unit_end

    for p, deg in itertools.product((0.2, 0.5, 1.0), (5.0, 37.0, 90.0, 150.0)):
        th = math.radians(deg)
        lc_ = p * th                                            # ramp length (unit radius)
        S, C = fresnel(math.sqrt(lc_ / math.pi))
        xr, yr = math.sqrt(math.pi * lc_) * C, math.sqrt(math.pi * lc_) * S   # ramp 1 end point
        a0 = 0.5 * p * th
        # ramp 2 is ramp 1 mirrored about the bisector: rotate (xr, -yr) by theta
        x3 = math.cos(th) * xr + math.sin(th) * yr
        y3 = math.sin(th) * xr - math.cos(th) * yr
        X = xr + (math.sin(th - a0) - math.sin(a0)) + x3
        Y = yr + (math.cos(a0) - math.cos(th - a0)) + y3
        x, y = _euler_unit_end(th, p)
        assert float(x) == pytest.approx(X, abs=1e-13) and float(y) == pytest.approx(Y, abs=1e-13)


def test_drawn_euler_curve_is_accurate_to_the_documented_level():
    """`_geometry.euler_curve` integrates numerically; it agrees with the exact clothoid to ~1e-7."""
    from photonix.layout._geometry import euler_curve
    from photonix.layout.routing_sbend import _euler_unit_end

    pts, _ = euler_curve(1.0, 90.0, 0.5, 48)
    x, y = _euler_unit_end(math.pi / 2, 0.5)
    assert abs(pts[-1, 0] - float(x)) < 1e-6 and abs(pts[-1, 1] - float(y)) < 1e-6


@pytest.mark.parametrize("kind", ["cosine", "circular", "euler"])
def test_sbend_min_span_inverts_min_radius(kind):
    for D, R in ((3.0, 10.0), (0.5, 50.0), (12.0, 7.0)):
        L = lay.sbend_min_span(D, R, kind=kind)
        assert float(lay.sbend_min_radius(L, D, kind=kind)) == pytest.approx(R, rel=1e-9)
        assert float(lay.sbend_min_radius(1.01 * L, D, kind=kind)) > R      # longer is gentler
    assert math.isinf(float(lay.sbend_min_radius(10.0, 0.0)))
    with pytest.raises(lay.RoutingError):
        lay.sbend_min_span(25.0, 5.0, kind="circular")                     # beyond two 90-degree arcs


@pytest.mark.parametrize("kind", ["cosine", "circular", "euler"])
def test_route_sbend_geometry_at_minimum_radius(kind):
    """At the shortest allowed span the drawn curvature reaches, and never exceeds, 1/R_min."""
    R, D = 10.0, 3.0
    L = lay.sbend_min_span(D, R, kind=kind) * (1 + 1e-9)
    p1, p2 = lay.Port("a", (5.0, -2.0), 0), lay.Port("b", (5.0 + L + 7.0, 1.0), 180)
    r = lay.route_sbend(p1, p2, min_radius=R, kind=kind, sbend_length=L, npts=192)
    line = _check_route(r, p1, p2, radius=R, area_rtol=1e-4)
    assert _max_curvature(line) >= (1 - 2e-2) / R                     # the bound is tight
    assert r.model_settings["n_bends"] == 2
    expected = float(lay.sbend_arc_length(L, D, kind=kind)) + 7.0
    tol = 1e-12 if kind != "euler" else 2e-6                           # euler: drawn-curve integration error
    assert r.model_settings["length"] == pytest.approx(expected, rel=tol)
    with pytest.raises(lay.RoutingError, match="span of at least"):
        lay.route_sbend(p1, p2, min_radius=R, kind=kind, sbend_length=0.9 * L)


def test_route_sbend_length_converges_to_area_over_width():
    """The tube formula area/width -> length with second-order convergence in npts."""
    p1, p2 = lay.Port("a", (0, 0), 0), lay.Port("b", (40, 6), 180)
    errs = []
    for npts in (24, 48, 96):
        r = lay.route_sbend(p1, p2, min_radius=10.0, kind="euler", npts=npts)
        errs.append(abs(_area(r.polygons[0][0]) / W - r.model_settings["length"]) / r.model_settings["length"])
    assert errs[2] < 1e-4
    assert 3.0 < errs[0] / errs[1] < 5.0 and 3.0 < errs[1] / errs[2] < 5.0


def test_route_sbend_any_orientation_and_cross_section():
    th = math.radians(33.0)
    u, n = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
    start = np.array([3.0, 4.0])
    end = start + 50.0 * u + 4.0 * n
    p1, p2 = lay.Port("a", tuple(start), 33.0), lay.Port("b", tuple(end), 213.0)
    xs = CrossSection("strip", (2, 0), width=0.8, radius=20.0)
    r = lay.route_sbend(p1, p2, xs=xs)
    _check_route(r, p1, p2, radius=20.0, width=0.8)
    assert r.polygons[0][1] == (2, 0)
    with pytest.raises(lay.RoutingError, match="antiparallel"):
        lay.route_sbend(p1, lay.Port("b", tuple(end), 180.0))


@requires_jax
@pytest.mark.parametrize("kind", ["cosine", "circular", "euler"])
@pytest.mark.parametrize("L0, D0", [(30.0, 3.0), (12.0, 5.0)])
def test_sbend_length_gradients_match_finite_differences(kind, L0, D0):
    gD = float(px.grad(lambda D: lay.sbend_arc_length(L0, D, kind=kind))(D0))
    gL = float(px.grad(lambda L: lay.sbend_arc_length(L, D0, kind=kind))(L0))
    fD = finite_diff(lambda D: float(lay.sbend_arc_length(L0, D, kind=kind)), D0, h=1e-5)
    fL = finite_diff(lambda L: float(lay.sbend_arc_length(L, D0, kind=kind)), L0, h=1e-5)
    assert gD == pytest.approx(fD, rel=1e-7) and gL == pytest.approx(fL, rel=1e-7)
    gR = float(px.grad(lambda D: lay.sbend_min_radius(L0, D, kind=kind))(D0))
    fR = finite_diff(lambda D: float(lay.sbend_min_radius(L0, D, kind=kind)), D0, h=1e-5)
    assert gR == pytest.approx(fR, rel=1e-6)
    # finite gradient at zero offset (safe-where), where the length is stationary
    g0 = float(px.grad(lambda D: lay.sbend_arc_length(30.0, D, kind=kind))(0.0))
    assert math.isfinite(g0)


# --------------------------------------------------------------------------- #
# route_auto
# --------------------------------------------------------------------------- #
def test_route_auto_routes_the_cases_route_manhattan_rejects():
    a = lay.Port("a", (0, 0), 0)
    cases = {
        "3um offset over 30um": lay.Port("b", (30, 3), 180),
        "port2 behind port1": lay.Port("b", (-20, 0), 180),
        "behind and offset": lay.Port("b", (-25, 4), 180),
        "too close for an L": lay.Port("b", (3, 30), 270),
        "facing away, small offset": lay.Port("b", (30, 4), 0),
    }
    for name, b in cases.items():
        with pytest.raises(lay.RoutingError):
            lay.route_manhattan(a, b, radius=5.0)
        for bend in ("circular", "euler"):
            r = lay.route_auto(a, b, radius=5.0, bend=bend)
            _check_route(r, a, b, radius=5.0)
            assert px.layout._route_common.self_clearance(_centerline(r), 2 * W) is None, name
    assert lay.route_auto(a, cases["3um offset over 30um"], radius=10.0).name == "route_sbend"
    assert lay.route_auto(a, cases["port2 behind port1"], radius=5.0).model_settings["n_bends"] == 4
    assert lay.route_auto(a, cases["too close for an L"], radius=5.0).model_settings["n_bends"] == 3


def test_route_auto_non_manhattan_ports():
    for o1, o2, end in ((30.0, 180.0, (60, 40)), (-60.0, 90.0, (40, -50)), (10.0, 225.0, (80, 70))):
        a, b = lay.Port("a", (0, 0), o1), lay.Port("b", end, o2)
        with pytest.raises(lay.RoutingError):
            lay.route_manhattan(a, b)
        r = lay.route_auto(a, b, radius=5.0)
        _check_route(r, a, b, radius=5.0)
    # Ports sharing an orientation modulo 90 degrees route Manhattan in their own frame (no fix-up bends).
    th = math.radians(30.0)
    a = lay.Port("a", (0, 0), 30.0)
    b = lay.Port("b", (40 * math.cos(th) - 30 * math.sin(th), 40 * math.sin(th) + 30 * math.cos(th)), 300.0)
    r = lay.route_auto(a, b, radius=5.0, bend="circular")
    _check_route(r, a, b, radius=5.0)
    assert r.model_settings["n_bends"] == 1
    assert r.model_settings["length"] == pytest.approx(40 + 30 - 2 * 5 + 2.5 * math.pi, rel=1e-12)


@pytest.mark.parametrize("bend", ["circular", "euler"])
def test_route_auto_reproduces_route_manhattan(bend):
    a = lay.Port("a", (0, 0), 0)
    for b in (lay.Port("b", (40, 0), 180), lay.Port("b", (30, 25), 270), lay.Port("b", (40, -30), 180),
              lay.Port("b", (-10, 30), 0), lay.Port("b", (25, -40), 90)):
        ref = lay.route_manhattan(a, b, radius=5.0, bend=bend)
        got = lay.route_auto(a, b, radius=5.0, bend=bend, allow_sbend=False)
        assert got.model_settings == ref.model_settings
        assert np.allclose(got.polygons[0][0], ref.polygons[0][0])


def test_route_auto_topology_search_matches_grid_dijkstra(monkeypatch):
    """Closed-form topology search == exhaustive Dijkstra over an obstacle-free grid (same cost).

    When the grid optimum crosses itself, ``route_astar`` falls back to the
    closed-form candidates; those cases are not independent and are counted
    separately (the fallback is replaced by a sentinel here).
    """
    import photonix.layout.routing_astar as ra

    class Looping(Exception):
        pass

    def sentinel(*args, **kwargs):
        raise Looping

    monkeypatch.setattr(ra, "_closed_form_fallback", sentinel)
    R = 5.0
    d, bl = 5.0, 2.5 * math.pi
    bp_auto = lay.default_bend_penalty(R)               # cost = routed length + bp_auto * n
    bp_grid = bp_auto - (2 * d - bl)                    # same cost in corner-polyline terms
    a = lay.Port("a", (0, 0), 0)
    compared = looping = 0
    for (x, y), o in itertools.product(((-20, 0), (-20, 30), (3, 30), (40, 30), (30, -40), (12, 8), (-15, -25)),
                                       (0, 90, 180, 270)):
        b = lay.Port("b", (x, y), o)
        try:
            grid = lay.route_astar(a, b, obstacles=[], keepout=0.0, grid=1.0, radius=R, bend_penalty=bp_grid,
                                   heuristic=False)
        except Looping:
            looping += 1
            continue
        auto = lay.route_auto(a, b, radius=R, bend="circular", allow_sbend=False, max_bends=6, min_space=W)
        g_cost = grid.model_settings["length"] + bp_auto * grid.model_settings["n_bends"]
        a_cost = auto.model_settings["length"] + bp_auto * auto.model_settings["n_bends"]
        assert a_cost == pytest.approx(g_cost, abs=1e-9), (x, y, o)
        compared += 1
    assert compared >= 24 and looping <= 4


def test_route_auto_property_sweep():
    """Every placement either routes with valid geometry or raises RoutingError -- nothing else."""
    a = lay.Port("a", (0, 0), 0)
    ok = 0
    total = 0
    for x, y, o in itertools.product((-30, -8, 0, 6, 25, 60), (-35, -4, 0, 2, 12, 40), (0, 45, 90, 180, 270, 300)):
        if (x, y) == (0, 0):
            continue
        total += 1
        b = lay.Port("b", (x, y), o)
        try:
            r = lay.route_auto(a, b, radius=5.0)
        except lay.RoutingError:
            continue
        ok += 1
        _check_route(r, a, b, radius=5.0, area_rtol=3e-3)
    assert ok / total > 0.85


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #
def _l_bundle(N=8, pitch=2.0):
    ins = [lay.Port(f"i{k}", (0.0, pitch * k), 0) for k in range(N)]
    outs = [lay.Port(f"o{k}", (100.0 - pitch * k, 80.0), 270) for k in range(N)]
    return ins, outs


def _partner(route, ports):
    """The port of ``ports`` that ``route``'s o2 lands on."""
    hits = [q for q in ports if np.allclose(q.center, route.ports["o2"].center)]
    assert len(hits) == 1
    return hits[0]


def _bundle_cell(routes):
    top = lay.Cell("bundle")
    for k, r in enumerate(routes):
        top.add_ref(r, name=f"r{k}")
    return top


def test_l_bundle_spec_validation():
    """N=8, pitch=2, R=10: disjoint polygons, min space clean, L_k = L_0 - 2kp, extraction connects all."""
    gdstk = pytest.importorskip("gdstk")
    N, pitch, R = 8, 2.0, 10.0
    ins, outs = _l_bundle(N, pitch)
    routes = lay.route_bundle(ins, outs, pitch=pitch, radius=R, bend="euler")
    for r, a, b in zip(routes, ins, outs, strict=True):
        _check_route(r, a, b, radius=R, area_rtol=3e-4)
    polys = [gdstk.Polygon(r.polygons[0][0]) for r in routes]
    for i, j in itertools.combinations(range(N), 2):
        assert gdstk.boolean(polys[i], polys[j], "and") == []
    report = lay.run_drc_precheck(_bundle_cell(routes), {(1, 0): {"min_space": pitch - W}})
    assert report.clean, str(report)
    L = [r.model_settings["length"] for r in routes]
    for k in range(N):
        assert L[k] == pytest.approx(L[0] - 2 * k * pitch, abs=1e-9)
    # extraction: devices at every port, every route joins its pair
    top = _bundle_cell(routes)
    for k, (a, b) in enumerate(zip(ins, outs, strict=True)):
        top.add_ref(lc.straight(5.0), name=f"in{k}").connect("o2", a.flipped())
        top.add_ref(lc.straight(5.0), name=f"out{k}").connect("o1", b.flipped())
    nl = lay.extract_netlist(top)
    conn = {frozenset([x, y]) for x, y in nl.connections.items()}
    for k in range(N):
        assert frozenset([(f"in{k}", "o2"), (f"r{k}", "o1")]) in conn
        assert frozenset([(f"r{k}", "o2"), (f"out{k}", "o1")]) in conn
    s = px.circuit.circuit_from_netlist(nl)(wl=1.55)
    for k in range(N):
        ref = px.components.straight(wl=1.55, length=10.0 + L[k])[("o1", "o2")]
        assert complex(s[(f"in{k}_o1", f"out{k}_o2")]) == pytest.approx(complex(ref), abs=1e-12)


def test_bundle_sorting_and_planarity_check():
    ins, outs = _l_bundle(4)
    shuffled = [ins[2], ins[0], ins[3], ins[1]]
    routes = lay.route_bundle(shuffled, outs, radius=10.0)
    # cell i belongs to shuffled[i] and lands on its rank partner
    for r, a in zip(routes, shuffled, strict=True):
        assert np.allclose(r.ports["o1"].center, a.center)
        k = ins.index(a)
        assert np.allclose(r.ports["o2"].center, outs[k].center)
    with pytest.raises(lay.RoutingError, match="non-planar"):
        lay.route_bundle(ins, outs[::-1], radius=10.0, sort=False)
    assert len(lay.route_bundle(ins, outs, radius=10.0, sort=False)) == 4


@pytest.mark.parametrize("bend", ["circular", "euler"])
def test_z_s_u_and_backbone_bundles_keep_pitch_and_never_touch(bend):
    from photonix.layout._route_common import polyline_distance

    N, pitch, R = 6, 2.0, 10.0
    ins = [lay.Port(f"i{k}", (0.0, pitch * k), 0) for k in range(N)]
    scenes = {
        "Z up": [lay.Port(f"o{k}", (160.0, 60.0 + pitch * k), 180) for k in range(N)],
        "Z down": [lay.Port(f"o{k}", (160.0, -60.0 + 3.0 * k), 180) for k in range(N)],
        "S": [lay.Port(f"o{k}", (160.0, 5.0 + pitch * k), 180) for k in range(N)],
        "U": [lay.Port(f"o{k}", (20.0, 60.0 + pitch * k), 0) for k in range(N)],
    }
    for name, outs in scenes.items():
        routes = lay.route_bundle(ins, outs, pitch=pitch, radius=R, bend=bend)
        lines = [_check_route(r, a, _partner(r, outs), radius=R) for r, a in zip(routes, ins, strict=True)]
        if name == "U":       # nested: the lowest input reaches the highest output
            assert _partner(routes[0], outs) is outs[-1]
        dmin = min(polyline_distance(lines[i], lines[j]) for i, j in itertools.combinations(range(N), 2))
        assert dmin >= (pitch if name != "S" else 0.99 * pitch) - 1e-9, name
    # Z jog columns sit exactly one pitch apart, the highest route jogging first
    routes = lay.route_bundle(ins, scenes["Z up"], pitch=pitch, radius=R, bend="circular")
    cols = []
    for r in routes:
        line = _centerline(r)
        vertical = np.abs(np.diff(line[:, 0])) < 1e-12
        cols.append(float(line[1:][vertical][0, 0]))
    assert np.allclose(np.diff(cols), -pitch)
    # backbone
    outs = [lay.Port(f"o{k}", (300.0, 200.0 + pitch * k), 180) for k in range(N)]
    routes = lay.route_bundle(ins, outs, pitch=3.0, radius=R, bend=bend,
                              waypoints=[(60, 5), (60, 120), (200, 120), (200, 205)])
    lens = []
    for r, a, b in zip(routes, ins, outs, strict=True):
        _check_route(r, a, b, radius=R)
        lens.append(r.model_settings["length"])
        assert r.model_settings["n_bends"] == 4
    lines = [_centerline(r) for r in routes]
    assert min(polyline_distance(lines[i], lines[i + 1]) for i in range(N - 1)) >= pitch - 1e-9


def test_fan_in_mixing_jogs_and_sbends_is_planar():
    """Wide-pitch inputs converging on a narrow output group: inner routes S-bend first, outer ones jog."""
    gdstk = pytest.importorskip("gdstk")
    N, R = 8, 10.0
    ins = [lay.Port(f"i{k}", (0.0, 30.0 * k), 0) for k in range(N)]
    outs = [lay.Port(f"o{k}", (300.0, 100.0 + 2.0 * k), 180) for k in range(N)]
    routes = lay.route_bundle(ins, outs, pitch=2.0, radius=R, min_space=1.0)
    for r, a, b in zip(routes, ins, outs, strict=True):
        _check_route(r, a, b, radius=R)
    polys = [gdstk.Polygon(r.polygons[0][0]) for r in routes]
    for i, j in itertools.combinations(range(N), 2):
        assert gdstk.boolean(polys[i], polys[j], "and") == []
    assert lay.run_drc_precheck(_bundle_cell(routes), {(1, 0): {"min_space": 1.0}}).clean


def test_bundle_errors_are_explicit():
    ins, outs = _l_bundle(4)
    with pytest.raises(lay.RoutingError, match="min_space"):
        lay.route_bundle(ins, outs, radius=10.0, pitch=2.0, min_space=1.6)
    with pytest.raises(lay.RoutingError, match="share one orientation"):
        lay.route_bundle([*ins[:3], lay.Port("x", (0, 9), 90)], outs, radius=10.0)
    with pytest.raises(lay.RoutingError, match="L bundle"):
        lay.route_bundle(ins, [lay.Port(f"o{k}", (5.0 - 2 * k, 20.0), 270) for k in range(4)], radius=10.0)
    with pytest.raises(lay.RoutingError, match="turns back"):
        lay.route_bundle([lay.Port("a", (0, 0), 0), lay.Port("b", (0, 40), 0)],
                         [lay.Port("c", (0, 60), 0), lay.Port("d", (0, 30), 0)], radius=5.0)
    with pytest.raises(lay.RoutingError, match="axial room"):
        lay.route_bundle(ins, [lay.Port(f"o{k}", (30.0, 50.0 + 2 * k), 180) for k in range(4)], radius=10.0)


def test_bundle_clearance_prefilter_matches_exhaustive_check():
    """The KD-tree prefiltered clearance check never misses what the O(N^2) exact check finds."""
    from photonix.layout._route_common import _check_bundle_clearance_exhaustive, check_bundle_clearance

    def raises(fn, *args) -> bool:
        try:
            fn(*args)
        except lay.RoutingError:
            return True
        return False

    rng = np.random.default_rng(3)
    outcomes = set()
    for _ in range(80):
        lines = [np.cumsum(rng.normal(size=(int(rng.integers(2, 10)), 2)) * rng.uniform(1.0, 20.0), axis=0)
                 + rng.uniform(-40, 40, 2) for _ in range(4)]
        thr = float(rng.uniform(0.3, 4.0))
        fast = raises(check_bundle_clearance, lines, thr)
        assert fast == raises(_check_bundle_clearance_exhaustive, lines, thr)
        outcomes.add(fast)
    assert outcomes == {True, False}


def test_fanout_sbends():
    ports = [lay.Port(f"p{k}", (0.0 - 3.0 * (k % 2), 127.0 * k), 0) for k in range(4)]
    routes, new = lay.fanout(ports, pitch=10.0, dx=600.0, min_radius=50.0, kind="cosine", min_space=5.0)
    ys = [q.center[1] for q in new]
    assert np.allclose(np.diff(ys), 10.0) and np.mean(ys) == pytest.approx(190.5)
    assert all(q.center[0] == pytest.approx(600.0) and q.orientation == 0.0 for q in new)
    assert [q.name for q in new] == [q.name for q in ports]
    for r, a, b in zip(routes, ports, new, strict=True):
        _check_route(r, a, b.flipped(), radius=50.0)
        D = b.center[1] - a.center[1]
        lead = 0.0 - a.center[0]
        assert r.model_settings["length"] == pytest.approx(lead + float(lay.sbend_arc_length(600.0, D)), abs=1e-9)
    with pytest.raises(lay.RoutingError, match="span of at least"):
        lay.fanout(ports, pitch=10.0, dx=100.0, min_radius=50.0, kind="cosine")
    with pytest.raises(lay.RoutingError, match="offsets at most"):
        lay.fanout(ports, pitch=10.0, dx=100.0, min_radius=50.0, kind="euler")


# --------------------------------------------------------------------------- #
# A*
# --------------------------------------------------------------------------- #
WALL = np.array([(20, -10), (40, -10), (40, 10), (20, 10)], dtype=float)


@pytest.mark.parametrize("heuristic", [True, False])
def test_astar_detour_around_rectangle_is_the_analytic_optimum(heuristic):
    """Optimal detour: up to the hugging line y = 10 + keepout + w/2 + grid/2, across, back down."""
    keepout, grid, R = 1.0, 1.0, 5.0
    a, b = lay.Port("a", (0, 0), 0), lay.Port("b", (60, 0), 180)
    r = lay.route_astar(a, b, obstacles=[WALL], keepout=keepout, grid=grid, radius=R, heuristic=heuristic)
    yc = 10 + keepout + W / 2 + grid / 2
    d, bl = R, 0.5 * math.pi * R
    assert r.model_settings["n_bends"] == 4
    assert r.model_settings["length"] == pytest.approx(60 + 2 * yc - 4 * (2 * d - bl), abs=1e-9)
    _check_route(r, a, b, radius=R)


def test_astar_equals_dijkstra_and_keeps_clearance_in_a_cluttered_scene():
    gdstk = pytest.importorskip("gdstk")
    rng = np.random.default_rng(7)
    scene = lay.Cell("scene")
    scene.add_ref(lc.ring_single(radius=10.0), origin=(45, 0), name="ring")
    for k in range(6):
        x, y = rng.uniform(15, 150), rng.uniform(-50, 50)
        w_, h_ = rng.uniform(4, 15), rng.uniform(4, 30)
        blk = lay.Cell(f"b{k}")
        blk.add_polygon([(0, 0), (w_, 0), (w_, h_), (0, h_)])
        scene.add_ref(blk, origin=(x, y), name=f"blk{k}")
    a, b = lay.Port("a", (0, 0), 0), lay.Port("b", (180, 12), 180)
    kw = dict(obstacles=scene, keepout=2.0, grid=1.0, radius=5.0, bend="euler")
    fast = lay.route_astar(a, b, **kw)
    slow = lay.route_astar(a, b, heuristic=False, **kw)
    bp = lay.default_bend_penalty(5.0)
    cost = [r.model_settings["length"] + bp * r.model_settings["n_bends"] for r in (fast, slow)]
    assert cost[0] == pytest.approx(cost[1], abs=1e-9)
    _check_route(fast, a, b, radius=5.0)
    # independent clearance check: the route polygon never enters the obstacles grown by the keepout
    grown = gdstk.offset([gdstk.Polygon(p) for p, _ in scene.get_polygons()], 2.0 - 1e-4, join="round")
    route_poly = gdstk.Polygon(fast.polygons[0][0])
    assert gdstk.boolean(route_poly, grown, "and") == []
    assert not any(gdstk.inside(fast.polygons[0][0], grown))


def test_astar_routes_between_devices_of_the_obstacle_cell():
    """Ports on obstacle devices: the leads may abut their own device; everything else keeps clear."""
    top = lay.Cell("chip")
    src = top.add_ref(lc.straight(10.0), origin=(-10, 0), name="src")
    dst = top.add_ref(lc.straight(10.0), origin=(100, 30), name="dst")
    blk = lay.Cell("blk")
    blk.add_polygon([(0, -20), (20, -20), (20, 60), (0, 60)])
    top.add_ref(blk, origin=(40, 0), name="blk")
    r = lay.route_astar(src.port("o2"), dst.port("o1"), obstacles=top, keepout=1.5, grid=0.5, radius=5.0)
    _check_route(r, src.port("o2"), dst.port("o1"), radius=5.0)
    top.add_ref(r, name="route")
    nl = lay.extract_netlist(top)
    conn = {frozenset([x, y]) for x, y in nl.connections.items()}
    assert conn == {frozenset([("src", "o2"), ("route", "o1")]), frozenset([("route", "o2"), ("dst", "o1")])}


def test_astar_off_grid_ports_layers_and_failures():
    a, b = lay.Port("a", (0.37, 0.21), 0), lay.Port("b", (60.13, 25.9), 180)     # off the uniform grid
    r = lay.route_astar(a, b, obstacles=[WALL + np.array([0.0, 10.0])], radius=5.0, keepout=1.0)
    _check_route(r, a, b, radius=5.0)
    assert r.model_settings["n_bends"] == 2
    gdstk = pytest.importorskip("gdstk")
    wall = gdstk.Polygon(WALL + np.array([0.0, 10.0]))
    assert gdstk.boolean(gdstk.Polygon(r.polygons[0][0]), gdstk.offset(wall, 1.0 - 1e-4), "and") == []
    # an obstacle on a layer outside obstacle_layers is ignored
    on_metal = lay.Cell("m")
    on_metal.add_polygon(WALL, (11, 0))
    c, f = lay.Port("c", (0.37, 0.21), 0), lay.Port("f", (60.13, 0.21), 180)
    ignored = lay.route_astar(c, f, obstacles=on_metal, obstacle_layers=[(1, 0)], radius=5.0)
    avoided = lay.route_astar(c, f, obstacles=on_metal, radius=5.0)
    assert ignored.model_settings["n_bends"] == 0 and avoided.model_settings["n_bends"] == 4
    _check_route(avoided, c, f, radius=5.0)
    box = [np.array([(50, 10), (70, 10), (70, 40), (50, 40)], dtype=float)]  # swallows port2's lead-in
    with pytest.raises(lay.RoutingError, match="lead-in"):
        lay.route_astar(a, b, obstacles=box, radius=5.0)
    dam = [np.array([(30, -100), (32, -100), (32, 100), (30, 100)], dtype=float)]   # splits the window
    with pytest.raises(lay.RoutingError, match="no obstacle-free route"):
        lay.route_astar(a, b, obstacles=dam, radius=5.0, bounds=((-20, -50), (80, 50)))
    with pytest.raises(lay.RoutingError, match="max_nodes"):
        lay.route_astar(a, b, obstacles=[WALL], radius=5.0, max_nodes=10)


# --------------------------------------------------------------------------- #
# Length matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bend", ["circular", "euler"])
def test_length_matched_routes_between_staggered_ports(bend):
    N, target = 6, 330.0
    cells = []
    for k in range(N):
        a = lay.Port(f"a{k}", (0.0 + 4.0 * k, 40.0 * k), 0)
        b = lay.Port(f"b{k}", (220.0 - 9.0 * k, 40.0 * k + 6.0 * k - 2.0), 180)
        r = lay.route_length_matched(a, b, target_length=target, radius=5.0, bend=bend, npts=96)
        assert abs(r.model_settings["length"] - target) < 1e-3 * 1e-3          # 1 nm
        _check_route(r, a, b, radius=5.0, area_rtol=1e-4)
        assert px.layout._route_common.self_clearance(_centerline(r), 2 * W) is None
        cells.append(r)
    # the extracted circuit gives every route the same phase
    phases = []
    for k, r in enumerate(cells):
        top = lay.Cell(f"t{k}")
        top.add_ref(r, name="r")
        s = px.circuit.circuit_from_netlist(lay.extract_netlist(top))(wl=1.55)
        phases.append(complex(s[("r_o1", "r_o2")]))
    assert np.allclose(phases, phases[0], atol=1e-10)


def test_trombone_height_obeys_the_closed_form():
    R = 5.0
    d, bl = R, 0.5 * math.pi * R
    a, b = lay.Port("a", (0, 0), 0), lay.Port("b", (200, 0), 180)
    for n, dL in ((1, 40.0), (2, 90.0), (3, 300.0)):
        r = lay.route_length_matched(a, b, target_length=200.0 + dL, radius=R, n_meanders=n)
        h = 0.5 * (dL / n + 4 * (2 * d - bl))
        assert r.model_settings["n_bends"] == 4 * n
        ymax = float(_centerline(r)[:, 1].max())
        assert ymax == pytest.approx(h, abs=1e-9)                              # side=+1: left of +x
        right = lay.route_length_matched(a, b, target_length=200.0 + dL, radius=R, n_meanders=n, side=-1)
        assert float(_centerline(right)[:, 1].min()) == pytest.approx(-h, abs=1e-9)
    capped = lay.route_length_matched(a, b, target_length=500.0, radius=R, max_height=40.0)
    assert float(_centerline(capped)[:, 1].max()) <= 40.0 + 1e-9
    assert capped.model_settings["length"] == pytest.approx(500.0, abs=1e-9)


def test_small_extra_length_uses_a_shallow_bump():
    a, b = lay.Port("a", (0, 0), 0), lay.Port("b", (100, 0), 180)
    for dL in (1e-3, 0.05, 1.0, 11.0):          # all below the minimum trombone 4 (L_bend - d) = 11.42 um
        r = lay.route_length_matched(a, b, target_length=100.0 + dL, radius=5.0, npts=96)
        assert r.model_settings["length"] == pytest.approx(100.0 + dL, abs=1e-9)
        assert r.model_settings["n_bends"] == 4
        _check_route(r, a, b, radius=5.0, area_rtol=1e-4)
    # continuity at the switch-over to trombones
    lo = lay.route_length_matched(a, b, target_length=100.0 + 4 * (2.5 * math.pi - 5.0) - 1e-7, radius=5.0)
    hi = lay.route_length_matched(a, b, target_length=100.0 + 4 * (2.5 * math.pi - 5.0) + 1e-7, radius=5.0)
    assert np.max(np.abs(_centerline(lo)[:, 1].max() - _centerline(hi)[:, 1].max())) < 1e-4


def test_length_matched_spreads_meanders_and_reports_errors():
    a, b = lay.Port("a", (0, 0), 0), lay.Port("b", (150, 40), 180)
    base = lay.route_auto(a, b, radius=5.0, bend="circular", allow_sbend=False).model_settings["length"]
    r = lay.route_length_matched(a, b, target_length=base + 300.0, radius=5.0, max_height=30.0)
    assert r.model_settings["length"] == pytest.approx(base + 300.0, abs=1e-9)
    _check_route(r, a, b, radius=5.0)
    with pytest.raises(lay.RoutingError, match="shorter than the shortest"):
        lay.route_length_matched(a, b, target_length=base - 1.0, radius=5.0)
    with pytest.raises(lay.RoutingError, match="could not place"):
        lay.route_length_matched(a, b, target_length=base + 300.0, radius=5.0, n_meanders=40)
    # waypoints and non-Manhattan ports are accepted as the base route
    w = lay.route_length_matched(a, b, target_length=base + 100.0, radius=5.0,
                                 waypoints=[(20, 0), (20, 40)])
    assert w.model_settings["length"] == pytest.approx(base + 100.0, abs=1e-9)
    c = lay.Port("c", (0, 0), 20.0)
    q = lay.route_length_matched(c, b, target_length=400.0, radius=5.0)
    _check_route(q, c, b, radius=5.0)
    assert q.model_settings["length"] == pytest.approx(400.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Whole flow: every router's cells extract and simulate
# --------------------------------------------------------------------------- #
def test_all_routers_extract_and_simulate_their_drawn_length():
    top = lay.Cell("chip")
    gc = [top.add_ref(lc.straight(5.0), origin=(-5.0, 30.0 * k), name=f"gc{k}") for k in range(3)]
    sink = [top.add_ref(lc.straight(5.0), origin=(400.0, 50.0 + 2.0 * k), name=f"sink{k}") for k in range(3)]
    routes1, mid = lay.fanout([g.port("o2") for g in gc], pitch=2.0, dx=150.0, min_radius=20.0)
    for k, r in enumerate(routes1):
        top.add_ref(r, name=f"fan{k}")
    bundle = lay.route_bundle(mid, [s.port("o1") for s in sink], radius=10.0)
    for k, r in enumerate(bundle):
        top.add_ref(r, name=f"bus{k}")
    nl = lay.extract_netlist(top)
    s = px.circuit.circuit_from_netlist(nl)(wl=1.55)
    for k in range(3):
        L = 10.0 + routes1[k].model_settings["length"] + bundle[k].model_settings["length"]
        ref = px.components.straight(wl=1.55, length=L)[("o1", "o2")]
        assert complex(s[(f"gc{k}_o1", f"sink{k}_o2")]) == pytest.approx(complex(ref), abs=1e-12)
