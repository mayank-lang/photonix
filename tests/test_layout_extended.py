"""Layout -> netlist -> simulation fidelity, routing, placement and DRC pre-checks."""
from __future__ import annotations

import inspect
import math
import os
import warnings

import numpy as np
import pytest

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc


def _simulate(cell, wl=1.55):
    top = lay.Cell("top")
    top.add_ref(cell, name="dut")
    return px.circuit.circuit_from_netlist(lay.extract_netlist(top))(wl=wl)


def _polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --------------------------------------------------------------------------- #
# Every generator simulates what it draws
# --------------------------------------------------------------------------- #
GENERATORS = [name for name, fn in inspect.getmembers(lc, inspect.isfunction)
              if name in lc.__all__]


@pytest.mark.parametrize("name", GENERATORS)
def test_every_generator_extracts_and_simulates(name):
    cell = getattr(lc, name)()
    top = lay.Cell("top")
    top.add_ref(cell, name="dut")
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # well-formed generators raise no placement warnings
        nl = lay.extract_netlist(top)
    if not nl.instances:      # portless (ring) or electrical-only (pad) cells
        assert not [p for p in cell.ports.values() if p.port_type == lay.OPTICAL]
        return
    s = px.circuit.circuit_from_netlist(nl)(wl=px.linspace(1.54, 1.56, 7))
    S, _ = px.as_sdense(s)
    assert np.all(np.linalg.svd(np.asarray(S), compute_uv=False) <= 1 + 1e-9)


def test_drawn_length_reaches_the_model():
    s = _simulate(lc.straight(25.0))
    ref = px.components.straight(length=25.0)
    assert complex(s[("dut_o1", "dut_o2")]) == pytest.approx(complex(ref[("o1", "o2")]), abs=1e-12)


def test_bend_and_taper_settings_round_trip():
    nl = lay.extract_netlist(_wrap(lc.bend_circular(radius=7.0, angle=-45.0)))
    assert nl.instances == {"dut": "bend"} and nl.settings["dut"] == {"radius": 7.0, "angle": -45.0}
    nl = lay.extract_netlist(_wrap(lc.taper(12.0, 0.4, 0.9)))
    assert nl.settings["dut"] == {"length": 12.0, "width1": 0.4, "width2": 0.9}


def _wrap(cell):
    top = lay.Cell("top")
    top.add_ref(cell, name="dut")
    return top


def test_clockwise_bend_geometry():
    c = lc.bend_circular(radius=5.0, angle=-90.0)
    assert np.allclose(c.ports["o2"].center, (5.0, -5.0), atol=1e-12)
    assert c.ports["o2"].orientation == pytest.approx(270.0)


def test_euler_bend_layout_matches_model_length_and_min_radius():
    c = lc.bend_euler(radius=4.0, angle=90.0, p=0.7, npts=4000)
    from photonix.layout._geometry import euler_curve, polyline_length

    pts, hd = euler_curve(4.0, 90.0, 0.7, npts=8000)
    assert polyline_length(pts) == pytest.approx(float(px.components.euler_bend_length(4.0, 90.0, 0.7)), rel=1e-6)
    curvature = np.diff(hd) / np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert curvature.max() <= 1 / 4.0 * (1 + 1e-3)
    assert c.ports["o2"].orientation == pytest.approx(90.0)


def test_sbend_offset_and_length():
    c = lc.sbend(length=20.0, offset=-3.0)
    assert np.allclose(c.ports["o2"].center, (20.0, -3.0))
    # numeric arc length of the cosine profile
    x = np.linspace(0, 20, 200001)
    dy = -1.5 * np.pi / 20 * np.sin(np.pi * x / 20)
    assert c.model_settings["length"] == pytest.approx(np.trapezoid(np.sqrt(1 + dy**2), x), rel=1e-6)


# --------------------------------------------------------------------------- #
# GDS integrity
# --------------------------------------------------------------------------- #
def test_gds_keeps_distinct_cells_that_share_a_name(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    top = lay.Cell("top")
    top.add_ref(lc.straight(25.0), origin=(0, 0), name="a")
    top.add_ref(lc.straight(10.0), origin=(25, 0), name="b")
    top.add_ref(lc.straight(10.0), origin=(35, 0), name="c")
    path = os.fspath(tmp_path / "t.gds")
    lay.write_gds(top, path)
    lib = gdstk.read_gds(path)
    sub = {c.name: c for c in lib.cells if c.name != "top"}
    assert set(sub) == {"straight", "straight$1"}            # 10 um straights deduplicated
    widths = sorted(round(c.bounding_box()[1][0], 6) for c in sub.values())
    assert widths == [10.0, 25.0]
    # Flattened geometry is exactly the native geometry.
    flat = lay.read_gds(path)
    assert sum(_polygon_area(p) for p, _ in flat.polygons) == pytest.approx(45.0 * 0.5, rel=1e-9)


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("orientation", [0.0, 90.0, 135.0, 270.0])
def test_reference_connect_mates_ports(orientation, mirror):
    target = lay.Port("t", (3.0, -2.0), orientation)
    top = lay.Cell("top")
    ref = top.add_ref(lc.bend_euler(radius=3.0, angle=60.0), name="b").connect("o2", target, mirror=mirror)
    mated = ref.port("o2")
    assert np.allclose(mated.center, target.center, atol=1e-9)
    assert ((mated.orientation - target.orientation) % 360.0) == pytest.approx(180.0)
    # polygons follow the same transform as ports
    flat = top.get_polygons()[0][0]
    assert np.min(np.linalg.norm(flat - np.asarray(target.center), axis=1)) < 0.3


# --------------------------------------------------------------------------- #
# Hierarchical extraction
# --------------------------------------------------------------------------- #
def test_hierarchical_mzi_matches_interferometer_physics():
    dl = 37.0
    cell = lc.mzi(delta_length=dl, length=15.0)
    nl = lay.extract_netlist(cell)
    assert any(name.startswith("top_") for name in nl.instances)
    wl = px.linspace(1.50, 1.60, 801)
    s = px.circuit.circuit_from_netlist(nl)(wl=wl)
    T = np.asarray(px.power(s[("o1", "o2")]))
    n = px.components.neff_linear(wl, 2.4, 4.2, 1.55)
    expected = np.cos(np.pi * np.asarray(n) * dl / np.asarray(wl)) ** 2
    assert np.allclose(T, expected, atol=1e-9)


def test_nested_cells_compose_transforms_and_names():
    inner = lay.Cell("pair")
    a = inner.add_ref(lc.straight(5.0), name="a")
    inner.add_ref(lc.straight(7.0), name="b").connect("o1", a.port("o2"))
    top = lay.Cell("top")
    top.add_ref(inner, origin=(100.0, 50.0), rotation=90.0, name="p1")
    top.add_port("in", (100.0, 50.0), 270.0)
    nl = lay.extract_netlist(top)
    assert set(nl.instances) == {"p1.a", "p1.b"}
    assert nl.connections == {("p1.a", "o2"): ("p1.b", "o1")}
    assert nl.ports["in"] == ("p1.a", "o1")
    s = px.circuit.circuit_from_netlist(nl)(wl=1.55)
    ref = px.components.straight(length=12.0)
    assert complex(s[("in", "p1.b_o2")]) == pytest.approx(complex(ref[("o1", "o2")]), abs=1e-12)


def test_electrical_ports_and_pads_are_not_optical_terminals():
    top = lay.Cell("top")
    top.add_ref(lc.heater_straight(50.0), name="h")
    top.add_ref(lc.pad(), origin=(0, 50), name="p")
    nl = lay.extract_netlist(top)
    assert set(nl.instances) == {"h"}
    assert set(nl.ports) == {"h_o1", "h_o2"}


def test_misplaced_device_is_reported():
    rotated = lay.Cell("rotated")
    rotated.add_ref(lc.straight(10.0), name="a")
    rotated.add_ref(lc.straight(10.0), origin=(10.0, 0.0), rotation=90.0, name="b")  # o1 faces 270
    with pytest.warns(UserWarning, match="not antiparallel"):
        lay.extract_netlist(rotated)
    wide = lay.Cell("wide")
    wide.add_ref(lc.straight(10.0), name="a")
    wide.add_ref(lc.straight(5.0, width=0.8), origin=(10.0, 0.0), name="c")
    with pytest.warns(UserWarning, match="different widths"):
        lay.extract_netlist(wide)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def _route_checks(r, p1, p2, width=0.5):
    assert np.allclose(r.ports["o1"].center, p1.center) and np.allclose(r.ports["o2"].center, p2.center)
    assert ((r.ports["o1"].orientation - p1.orientation) % 360) == pytest.approx(180.0)
    assert ((r.ports["o2"].orientation - p2.orientation) % 360) == pytest.approx(180.0)
    area = _polygon_area(r.polygons[0][0])
    assert area == pytest.approx(r.model_settings["length"] * width, rel=2e-3)


@pytest.mark.parametrize("bend", ["circular", "euler"])
def test_route_manhattan_topologies(bend):
    radius = 5.0
    cases = [
        (lay.Port("a", (0, 0), 0), lay.Port("b", (40, 0), 180), 0),     # straight
        (lay.Port("a", (0, 0), 0), lay.Port("b", (30, 25), 270), 1),    # L
        (lay.Port("a", (0, 0), 0), lay.Port("b", (40, -30), 180), 2),   # Z
        (lay.Port("a", (0, 0), 0), lay.Port("b", (-10, 30), 0), 2),     # U
    ]
    for p1, p2, n_bends in cases:
        r = lay.route_manhattan(p1, p2, radius=radius, bend=bend)
        assert r.model_settings["n_bends"] == n_bends
        _route_checks(r, p1, p2)


def test_route_manhattan_length_formula_circular_and_euler():
    p1, p2 = lay.Port("a", (0, 0), 0), lay.Port("b", (30, 25), 270)
    circ = lay.route_manhattan(p1, p2, radius=5.0)
    assert circ.model_settings["length"] == pytest.approx(30 + 25 - 2 * 5 + math.pi / 2 * 5)
    # Same minimum radius: the Euler bend is longer but has a larger corner offset,
    # which shortens the adjoining straights.
    from photonix.layout._geometry import corner_offset, euler_curve

    d = corner_offset(euler_curve(5.0, 90.0, 0.5, 48))
    assert d > 5.0
    eul = lay.route_manhattan(p1, p2, radius=5.0, bend="euler", p=0.5)
    expected = 30 + 25 - 2 * d + float(px.components.euler_bend_length(5.0, 90.0, 0.5))
    assert eul.model_settings["length"] == pytest.approx(expected, rel=1e-12)


def test_route_manhattan_waypoints_and_errors():
    p1, p2 = lay.Port("a", (0, 0), 0), lay.Port("b", (0, 40), 180)
    r = lay.route_manhattan(p1, p2, radius=5.0, waypoints=[(20, 0), (20, 20), (-15, 20), (-15, 40)])
    assert r.model_settings["n_bends"] == 4
    _route_checks(r, p1, p2)
    with pytest.raises(lay.RoutingError, match="Z route"):
        lay.route_manhattan(lay.Port("a", (0, 0), 0), lay.Port("b", (40, 3), 180), radius=5.0)
    with pytest.raises(lay.RoutingError, match="axis-aligned"):
        lay.route_manhattan(lay.Port("a", (0, 0), 45), lay.Port("b", (40, 3), 180))
    with pytest.raises(lay.RoutingError, match="bends need"):
        lay.route_manhattan(p1, p2, radius=15.0, waypoints=[(20, 0), (20, 20), (-15, 20), (-15, 40)])


def test_routed_layout_simulates_the_route_length():
    top = lay.Cell("top")
    a = top.add_ref(lc.straight(10.0), name="a")
    b = top.add_ref(lc.straight(10.0), origin=(60.0, 30.0), name="b")
    top.add_ref(lay.route_manhattan(a.port("o2"), b.port("o1"), radius=5.0), name="r")
    nl = lay.extract_netlist(top)
    L = nl.settings["r"]["length"]
    s = px.circuit.circuit_from_netlist(nl)(wl=1.55)
    ref = px.components.straight(length=20.0 + L)
    assert complex(s[("a_o1", "b_o2")]) == pytest.approx(complex(ref[("o1", "o2")]), abs=1e-12)


# --------------------------------------------------------------------------- #
# Geometry-driven resonator models
# --------------------------------------------------------------------------- #
def test_ring_single_matches_circuit_assembly():
    wl = px.linspace(1.545, 1.555, 401)
    R, gap, lx = 8.0, 0.18, 4.0
    cpl = px.components.evanescent_coupler(wl=wl, gap=gap, length=lx, bend_radius=R, loss_db_cm=3.0)
    ring = px.components.straight(wl=wl, length=2 * np.pi * R + lx, loss_db_cm=3.0)
    s_circ = px.circuit.evaluate_circuit(
        {"c": cpl, "w": ring},
        {("c", "o3"): ("w", "o1"), ("w", "o2"): ("c", "o2")},
        {"in": ("c", "o1"), "out": ("c", "o4")},
    )
    s = px.components.ring_single(wl=wl, radius=R, gap=gap, length_x=lx, loss_db_cm=3.0)
    assert np.allclose(np.asarray(s[("o1", "o2")]), np.asarray(s_circ[("in", "out")]), atol=1e-12)


def test_ring_double_matches_analytic_add_drop():
    wl = px.linspace(1.545, 1.555, 801)
    R, gap = 10.0, 0.22
    cpl = px.components.evanescent_coupler(wl=1.55, gap=gap, length=0.0, bend_radius=R)
    kappa2 = float(px.power(cpl[("o1", "o3")]))
    ours = px.components.ring_double(wl=wl, radius=R, gap=gap, loss_db_cm=2.0)
    # The analytic model takes a wavelength-independent coupling; compare near 1.55 um.
    ref = px.components.add_drop_ring(wl=wl, coupling1=kappa2, coupling2=kappa2, radius=R, loss_db_cm=2.0)
    near = np.abs(np.asarray(wl) - 1.55) < 2e-3
    for key in (("o1", "o2"), ("o1", "o3"), ("o4", "o3")):
        assert np.allclose(np.asarray(px.power(ours[key]))[near], np.asarray(px.power(ref[key]))[near], atol=2e-3)


# --------------------------------------------------------------------------- #
# DRC pre-check
# --------------------------------------------------------------------------- #
def test_drc_precheck_flags_narrow_gap_only_in_coupling_region():
    pytest.importorskip("gdstk")
    rules = {(1, 0): {"min_width": 0.15, "min_space": 0.15}}
    assert lay.run_drc_precheck(lc.ring_single(gap=0.2), rules).clean
    report = lay.run_drc_precheck(lc.ring_single(gap=0.1, radius=10.0), rules)
    assert not report.clean
    assert {v.rule for v in report.violations} == {"min_space"}
    for v in report.violations:
        (x0, y0), (x1, y1) = v.bbox
        assert 0.2 < y0 and y1 < 0.65 and abs(x0 + x1) < 1.0   # between bus and ring, centred


def test_drc_precheck_width_and_area():
    pytest.importorskip("gdstk")
    c = lay.Cell("t")
    c.add_polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])
    c.add_polygon([(20, 0), (20.3, 0), (20.3, 0.3), (20, 0.3)])
    c.add_polygon([(0, 5), (10, 5), (10, 5.08), (0, 5.08)])
    report = lay.run_drc_precheck(c, {(1, 0): {"min_width": 0.1, "min_area": 0.2}})
    # the 80 nm sliver breaks min_width; only the 0.09 um^2 square breaks min_area
    assert report.summary() == {("min_width", (1, 0)): 1, ("min_area", (1, 0)): 1}
    assert "violation" in str(report)
    with pytest.raises(ValueError, match="unknown rule"):
        lay.run_drc_precheck(c, {(1, 0): {"min_widht": 0.1}})


def test_generated_components_pass_sane_rules():
    pytest.importorskip("gdstk")
    rules = {(1, 0): {"min_width": 0.15, "min_space": 0.15}}
    for cell in (lc.straight(), lc.bend_euler(), lc.coupler(gap=0.2), lc.mmi2x2(), lc.crossing(), lc.mzi()):
        report = lay.run_drc_precheck(cell, rules)
        assert report.clean, (cell.name, report.summary())
