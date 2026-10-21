"""LVS-lite: geometric connectivity, parameter re-extraction and netlist comparison.

Every defect layout below has an exact expected answer (gap, overlap area,
offset), and every library generator at its defaults must be clean.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("gdstk")

from photonix.circuit import Netlist  # noqa: E402
from photonix.layout import Cell, extract_netlist, route_manhattan  # noqa: E402
from photonix.layout import components as lc  # noqa: E402
from photonix.verification import (  # noqa: E402
    ISSUE_KINDS,
    LvsIssue,
    check_optical_connectivity,
    check_parameters,
    check_ports_on_geometry,
    compare_netlists,
    extract_parameters,
    run_lvs_lite,
)


def _pair(dx: float, dy: float = 0.0, w2: float = 0.5) -> Cell:
    top = Cell("pair")
    top.add_ref(lc.straight(10.0), name="a")
    top.add_ref(lc.straight(10.0, width=w2), origin=(10.0 + dx, dy), name="b")
    return top


def _wrapped(cell: Cell, rotation: float, mirror: bool) -> Cell:
    outer = Cell("outer")
    outer.add_ref(cell, origin=(3.0, -7.0), rotation=rotation, mirror=mirror, name="u")
    return outer


def _key(issues):
    return sorted((i.kind, tuple(o.split(".", 1)[-1] if o.startswith("u.") else o for o in i.objects),
                   None if i.value is None else round(i.value, 5)) for i in issues)


def _old_grating_coupler(length=20.0, width=12.0, period=0.63, ff=0.5) -> Cell:
    """The pre-fix generator: o1 sat at x=length, beyond the last tooth, with no taper."""
    c = Cell("gc_old", model="grating_coupler")
    n = int(np.floor(length / period + 1e-9))
    for i in range(n):
        x0 = i * period
        c.add_polygon([(x0, -width / 2), (x0 + ff * period, -width / 2),
                       (x0 + ff * period, width / 2), (x0, width / 2)])
    c.add_port("o1", (length, 0.0), 0.0, 0.5)
    c.add_port("o2", (0.5 * n * period, 0.0), 90.0, 10.4, (203, 0))
    return c


# --------------------------------------------------------------------------- #
# synthetic defects with exact answers
# --------------------------------------------------------------------------- #
def test_five_nm_gap_is_exactly_one_near_open():
    issues = run_lvs_lite(_pair(0.005)).issues
    assert [(i.kind, i.objects) for i in issues] == [("near_open", ("a.o2", "b.o1"))]
    assert issues[0].value == pytest.approx(0.005, abs=1e-12)
    assert extract_netlist(_pair(0.005), warn=False).connections == {}   # what extraction alone sees


def test_overlap_is_one_short_with_exact_area():
    top = Cell("ov")
    top.add_ref(lc.straight(10.0), name="a")
    top.add_ref(lc.straight(10.0), origin=(7.0, 0.0), name="b")       # 3 um overlap
    issues = run_lvs_lite(top).issues
    assert [(i.kind, i.objects) for i in issues] == [("short", ("a", "b"))]
    assert issues[0].value == pytest.approx(3.0 * 0.5, abs=1e-6)
    assert issues[0].marker is not None


def test_right_angle_crossing_is_a_short_of_area_w_squared():
    top = Cell("x")
    top.add_ref(lc.straight(10.0), origin=(-5.0, 0.0), name="h")
    top.add_ref(lc.straight(10.0), origin=(0.0, -5.0), rotation=90.0, name="v")
    issues = run_lvs_lite(top).issues
    assert [(i.kind, i.objects) for i in issues] == [("short", ("h", "v"))]
    assert issues[0].value == pytest.approx(0.25, abs=1e-6)
    assert np.allclose(issues[0].location, (0.0, 0.0), atol=1e-9)


def test_route_drawn_across_two_connected_straights_shorts_both():
    top = Cell("t")
    a = top.add_ref(lc.straight(10.0), name="a")
    top.add_ref(lc.straight(10.0), name="b").connect("o1", a.port("o2"))
    top.add_ref(lc.straight(4.0), origin=(10.0, -2.0), rotation=90.0, name="c")   # through the joint
    kinds = sorted((i.kind, i.objects) for i in check_optical_connectivity(top))
    assert kinds == [("short", ("a", "c")), ("short", ("b", "c"))]


def test_connected_overlap_inside_the_joint_disc_is_allowed():
    top = Cell("t")
    a = top.add_ref(lc.straight(10.0), name="a")
    top.add_ref(lc.bend_circular(5.0), name="b").connect("o1", a.port("o2"))
    assert check_optical_connectivity(top) == []


def test_lateral_misalignment_is_a_near_open_with_its_offset():
    issues = check_optical_connectivity(_pair(0.0, dy=0.1))
    assert [(i.kind, i.objects) for i in issues] == [("near_open", ("a.o2", "b.o1"))]
    assert issues[0].value == pytest.approx(0.1, abs=1e-12)


def test_connected_ports_of_different_width_are_reported():
    issues = run_lvs_lite(_pair(0.0, w2=0.45)).issues
    assert [(i.kind, i.objects) for i in issues] == [("port_width_mismatch", ("a.o2", "b.o1"))]
    assert issues[0].value == pytest.approx(0.05)


def test_coincident_but_not_facing_ports_are_an_angle_mismatch():
    top = Cell("t")
    top.add_ref(lc.straight(10.0), name="a")
    top.add_ref(lc.straight(10.0), origin=(10.0, 0.0), rotation=90.0, name="b")
    kinds = {i.kind for i in check_optical_connectivity(top)}
    assert "angle_mismatch" in kinds


def test_three_coincident_ports_are_ambiguous():
    top = Cell("t")
    top.add_ref(lc.straight(10.0), name="a")
    top.add_ref(lc.straight(10.0), origin=(10.0, 0.0), name="b")
    top.add_ref(lc.straight(10.0), origin=(10.0, 0.0), rotation=90.0, name="c")
    issues = check_optical_connectivity(top)
    amb = [i for i in issues if i.kind == "ambiguous_connection"]
    assert len(amb) == 1 and len(amb[0].objects) == 3


def test_unconnected_internal_port_is_an_open_when_top_declares_its_ports():
    top = Cell("t")
    a = top.add_ref(lc.straight(10.0), name="a")
    b = top.add_ref(lc.straight(10.0), origin=(30.0, 0.0), name="b")
    top.expose_port(a, "o1", "o1")
    top.expose_port(b, "o2", "o2")
    issues = check_optical_connectivity(top)
    assert sorted((i.kind, i.objects) for i in issues) == [("open", ("a.o2",)), ("open", ("b.o1",))]
    # without declared top ports the external ports are unknown: no opens
    bare = Cell("bare")
    bare.add_ref(lc.straight(10.0), name="a")
    bare.add_ref(lc.straight(10.0), origin=(30.0, 0.0), name="b")
    assert check_optical_connectivity(bare) == []


def test_old_grating_coupler_port_is_off_geometry_by_0_785_um():
    issues = check_ports_on_geometry(_old_grating_coupler())
    assert [(i.kind, i.objects) for i in issues] == [("port_off_geometry", ("o1",))]
    assert issues[0].value == pytest.approx(20.0 - (30 * 0.63 + 0.5 * 0.63), abs=1e-9)   # 0.785
    assert check_ports_on_geometry(lc.grating_coupler()) == []        # the current generator has its taper


def test_port_inside_geometry_and_port_overhang_are_flagged():
    c = Cell("c")
    c.add_polygon([(0, -0.25), (10, -0.25), (10, 0.25), (0, 0.25)])
    c.add_port("o1", (1.0, 0.0), 180.0, 0.5)          # 1 um inside the waveguide
    c.add_port("o2", (10.0, 0.2), 0.0, 0.5)           # 0.2 um off axis: overhangs the face
    got = {i.objects[0]: (i.kind, i.value) for i in check_ports_on_geometry(c)}
    assert got["o1"][0] == "port_off_geometry"
    assert got["o2"][0] == "port_off_geometry" and got["o2"][1] == pytest.approx(0.2, abs=1e-6)


def test_drawn_face_narrower_than_port_is_a_width_mismatch():
    c = lc.straight(10.0, width=0.45)
    c.ports["o2"].width = 0.5
    issues = check_ports_on_geometry(c)
    assert [(i.kind, i.objects) for i in issues] == [("port_width_mismatch", ("o2",))]
    assert issues[0].value == pytest.approx(0.45, abs=1e-6)
    wide = lc.straight(10.0, width=0.6)
    wide.ports["o2"].width = 0.5
    assert check_ports_on_geometry(wide) == []                      # wider faces (MMI bodies) are allowed
    assert [i.kind for i in check_ports_on_geometry(wide, exact_width=True)] == ["port_width_mismatch"]


@pytest.mark.parametrize("name", lc.__all__)
def test_every_library_generator_is_clean_at_defaults(name):
    cell = getattr(lc, name)()
    assert check_ports_on_geometry(cell) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert run_lvs_lite(cell).issues == []


def test_issue_kind_is_validated():
    with pytest.raises(ValueError, match="unknown issue kind"):
        LvsIssue("bogus", (0.0, 0.0), (), "")
    assert "near_open" in ISSUE_KINDS


@pytest.mark.parametrize("rotation,mirror", [(90.0, False), (0.0, True), (90.0, True), (217.0, True)])
def test_issues_are_invariant_under_rotation_and_mirroring(rotation, mirror):
    defects = Cell("defects")
    defects.add_ref(_pair(0.005), name="gap")
    ov = Cell("ov")
    ov.add_ref(lc.straight(10.0), name="a")
    ov.add_ref(lc.straight(10.0), origin=(7.0, 0.0), name="b")
    defects.add_ref(ov, origin=(0.0, 20.0), name="ov")
    defects.add_ref(_old_grating_coupler(), origin=(0.0, 50.0), name="gc")
    base = run_lvs_lite(defects).issues
    moved = run_lvs_lite(_wrapped(defects, rotation, mirror)).issues
    assert len(base) == 3
    assert _key(moved) == _key(base)


# --------------------------------------------------------------------------- #
# parameter re-extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gap", [0.1, 0.2, 0.35])
def test_directional_coupler_gap_is_recovered_exactly(gap):
    assert extract_parameters(lc.directional_coupler(gap=gap))["gap_geo"] == pytest.approx(gap, abs=1e-9)
    assert extract_parameters(lc.coupler(gap=gap))["gap_geo"] == pytest.approx(gap, abs=1e-9)


@pytest.mark.parametrize("radius,length_x", [(5.0, 0.0), (10.0, 0.0), (8.0, 6.0)])
def test_ring_radius_and_racetrack_length_are_recovered(radius, length_x):
    p = extract_parameters(lc.ring_single(radius=radius, gap=0.2, length_x=length_x))
    assert p["radius_geo"] == pytest.approx(radius, abs=1e-4)
    assert p["length_x_geo"] == pytest.approx(length_x, abs=1e-4)
    # the drawn gap exceeds the circle's by at most the polygon sagitta (R + w/2)(1 - cos(dtheta/2))
    dtheta = 2 * np.pi / (240 - 1) if length_x == 0 else np.pi / (240 // 2 - 1)
    bound = (radius + 0.25) * (1 - np.cos(dtheta / 2))
    assert 0.2 - 1e-9 <= p["gap_geo"] <= 0.2 + bound + 1e-9


def test_ring_double_recovers_both_gaps():
    p = extract_parameters(lc.ring_double(radius=10.0, gap=0.2, gap2=0.3))
    bound = 10.25 * (1 - np.cos(np.pi / 239))
    assert 0.2 <= p["gap_geo"] <= 0.2 + bound + 1e-9
    assert 0.3 <= p["gap2_geo"] <= 0.3 + bound + 1e-9


@pytest.mark.parametrize("npts", [20, 60, 200])
def test_bend_length_is_the_exact_polygonal_area_over_width(npts):
    radius, angle = 5.0, 90.0
    p = extract_parameters(lc.bend_circular(radius, angle, npts=npts))
    dth = np.deg2rad(angle) / (npts - 1)
    assert p["length_geo"] == pytest.approx(radius * (npts - 1) * np.sin(dth), rel=1e-9)   # Steiner, polygonal
    true = radius * np.deg2rad(angle)
    assert abs(p["length_geo"] - true) <= true * dth ** 2 / 6 * 1.0001
    assert p["radius_geo"] == pytest.approx(radius, abs=1e-6)


def test_taper_and_sbend_lengths():
    assert extract_parameters(lc.taper(12.0, 0.5, 2.0))["length_geo"] == pytest.approx(12.0, rel=1e-9)
    sb = lc.sbend(20.0, 3.0)
    assert extract_parameters(sb)["length_geo"] == pytest.approx(sb.model_settings["length"], rel=2e-4)


def test_routed_lengths_match_their_settings_within_the_chord_error():
    a, b = lc.straight(5.0), lc.straight(5.0)
    top = Cell("r")
    ra = top.add_ref(a, name="a")
    rb = top.add_ref(b, origin=(40.0, 30.0), rotation=180.0, name="b")
    for bend in ("circular", "euler"):
        r = route_manhattan(ra.port("o2"), rb.port("o2"), radius=5.0, bend=bend)
        got = extract_parameters(r)["length_geo"]
        assert got == pytest.approx(r.model_settings["length"], rel=2e-3)
        assert got <= r.model_settings["length"] + 1e-9        # chords cut corners: never longer


def test_check_parameters_is_clean_for_the_mzi_and_flags_stale_settings():
    assert check_parameters(lc.mzi(delta_length=20.0)) == []
    top = Cell("t")
    ring = lc.ring_single(radius=10.0, gap=0.2)
    ring.model_settings["gap"] = 0.25
    top.add_ref(ring, name="r")
    dc = lc.directional_coupler(gap=0.2)
    dc.model_settings["gap"] = 0.2 + 1e-3        # within atol: accepted
    top.add_ref(dc, origin=(0, 40), name="dc")
    issues = check_parameters(top)
    assert [(i.kind, i.objects) for i in issues] == [("parameter_mismatch", ("r",))]
    assert issues[0].value == pytest.approx(0.2, abs=1e-3)


# --------------------------------------------------------------------------- #
# layout versus schematic
# --------------------------------------------------------------------------- #
def _mzi_schematic(prefix: str = "") -> Netlist:
    """Hand-written netlist of ``lc.mzi()`` defaults (delta_length 20, R 5, length 10, spacing 10)."""
    inst: dict[str, str] = {f"{prefix}splitter": "mmi1x2", f"{prefix}combiner": "mmi1x2"}
    settings: dict[str, dict] = {}
    conns: dict = {}
    for arm, sign, vertical in (("top", 1.0, 10.0), ("bot", -1.0, 0.0)):
        chain = [("b1", "bend", {"radius": 5.0, "angle": 90.0 * sign})]
        if vertical:
            chain.append(("s1", "straight", {"length": vertical}))
        chain += [("b2", "bend", {"radius": 5.0, "angle": -90.0 * sign}), ("s2", "straight", {"length": 10.0}),
                  ("b3", "bend", {"radius": 5.0, "angle": -90.0 * sign})]
        if vertical:
            chain.append(("s3", "straight", {"length": vertical}))
        chain.append(("b4", "bend", {"radius": 5.0, "angle": 90.0 * sign}))
        prev = (f"{prefix}splitter", "o2" if sign > 0 else "o3")
        for tag, model, st in chain:
            name = f"{prefix}{arm}_{tag}"
            inst[name] = model
            settings[name] = st
            conns[prev] = (name, "o1")
            prev = (name, "o2")
        conns[prev] = (f"{prefix}combiner", "o3" if sign > 0 else "o2")
    return Netlist(instances=inst, connections=conns, settings=settings,
                   ports={"o1": (f"{prefix}splitter", "o1"), "o2": (f"{prefix}combiner", "o1")})


def test_mzi_layout_matches_hand_written_schematic():
    rep = run_lvs_lite(lc.mzi(), _mzi_schematic())
    assert rep.clean, str(rep)
    assert rep.mapping == {k: k for k in _mzi_schematic().instances}


def test_mzi_matches_a_renamed_schematic_through_ports_and_propagation():
    rep = compare_netlists(extract_netlist(lc.mzi()), _mzi_schematic("X_"))
    assert rep.clean, str(rep)
    assert rep.mapping["top_s2"] == "X_top_s2" and rep.mapping["combiner"] == "X_combiner"


def test_one_changed_setting_gives_exactly_one_setting_mismatch():
    sch = _mzi_schematic()
    sch.settings["top_s2"] = {"length": 11.0}
    rep = run_lvs_lite(lc.mzi(), sch)
    assert [(i.kind, i.objects) for i in rep.issues] == [("setting_mismatch", ("top_s2", "top_s2"))]
    assert rep.issues[0].value == pytest.approx(10.0)


def test_swapped_connections_are_connection_mismatches():
    sch = _mzi_schematic()
    conns = dict(sch.connections)
    # swap the splitter outputs: top arm on o3, bottom arm on o2
    conns[("splitter", "o2")], conns[("splitter", "o3")] = conns[("splitter", "o3")], conns[("splitter", "o2")]
    swapped = Netlist(instances=sch.instances, connections=conns, ports=sch.ports, settings=sch.settings)
    rep = run_lvs_lite(lc.mzi(), swapped)
    assert rep.issues and {i.kind for i in rep.issues} == {"connection_mismatch"}
    assert len(rep.issues) == 4    # two drawn connections absent from the schematic, and vice versa


def test_missing_extra_model_and_port_mismatches():
    sch = Netlist(instances={"a": "straight", "b": "bend", "c": "straight"},
                  connections={("a", "o2"): ("b", "o1")},
                  ports={"in": ("a", "o1"), "out": ("b", "o2"), "x": ("c", "o1")})
    lay = Netlist(instances={"a": "straight", "b": "straight", "d": "mmi1x2"},
                  connections={("a", "o2"): ("b", "o1")},
                  ports={"in": ("a", "o1"), "out": ("b", "o2"), "y": ("d", "o1")})
    kinds = compare_netlists(lay, sch).summary()
    assert kinds["model_mismatch"] == 1               # b: straight vs bend (anchored by port "out")
    assert kinds["missing_instance"] == 1 and kinds["extra_instance"] == 1
    assert kinds["external_port_mismatch"] == 2       # x and y


def test_routes_are_transparent():
    top = Cell("routed")
    a = top.add_ref(lc.mmi1x2(), name="a")
    b = top.add_ref(lc.mmi1x2(), origin=(60.0, 25.0), rotation=180.0, name="b")
    top.add_ref(route_manhattan(a.port("o2"), b.port("o3"), radius=5.0), name="r1")
    top.expose_port(a, "o1", "in")
    top.expose_port(b, "o1", "out")
    top.expose_port(a, "o3", "a_o3")
    top.expose_port(b, "o2", "b_o2")
    sch = Netlist(instances={"a": "mmi1x2", "b": "mmi1x2"}, connections={("a", "o2"): ("b", "o3")},
                  ports={"in": ("a", "o1"), "out": ("b", "o1"), "a_o3": ("a", "o3"), "b_o2": ("b", "o2")})
    rep = run_lvs_lite(top, sch)
    assert rep.clean, str(rep)
    with_route = compare_netlists(extract_netlist(top), sch, transparent_models=())
    assert with_route.summary().get("extra_instance") == 1


def test_report_summary_and_str():
    rep = run_lvs_lite(_pair(0.005))
    assert rep.summary() == {"near_open": 1} and not rep.clean
    assert "near_open" in str(rep) and len(rep.of_kind("near_open")) == 1
    assert str(run_lvs_lite(lc.straight())) == "LVS-lite: clean"


def test_large_layout_runs_in_reasonable_time():
    import time

    top = Cell("big")
    prev = None
    for k in range(300):
        r = top.add_ref(lc.straight(2.0), name=f"w{k}")
        if prev is not None:
            r.connect("o1", prev.port("o2"))
        prev = r
    t0 = time.perf_counter()
    assert run_lvs_lite(top).clean
    assert time.perf_counter() - t0 < 30.0
