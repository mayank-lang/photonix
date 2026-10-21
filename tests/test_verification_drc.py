"""DRC rule deck: threshold toggles on analytic geometry, metrics, invariance, markers and KLayout decks.

Each rule is exercised at ``value - delta`` (must fail) and ``value + delta``
(must pass) with ``delta`` = 2 grid units (2 nm), so the band that lets a
feature drawn exactly at the rule pass is also shown not to hide a real
violation.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("gdstk")

from photonix.layout import Cell, klayout_available, run_drc_precheck  # noqa: E402
from photonix.layout import components as lc  # noqa: E402
from photonix.pdk import demo_pdk  # noqa: E402
from photonix.verification import (  # noqa: E402
    METRICS,
    DeckReport,
    LvsReport,
    Rule,
    RuleDeck,
    Waiver,
    apply_waivers,
    check_enclosure,
    check_extension,
    check_hole_area,
    check_inside,
    check_max_width,
    check_min_angle,
    check_not_overlap,
    check_offgrid_vertices,
    check_separation,
    check_space,
    compare_with_klayout,
    read_lyrdb,
    run_lvs_lite,
    write_lyrdb,
)

D = 0.002   # two 1 nm grid units
METAL, VIA, WG, SLAB = (11, 0), (12, 0), (1, 0), (2, 0)


def rect(c: Cell, x0, y0, x1, y1, layer=WG) -> None:
    c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer)


def area(vs) -> float:
    return float(sum(v.area for v in vs))


# --------------------------------------------------------------------------- #
# threshold toggles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("e", [0.1, 0.5])
def test_enclosure_toggles_at_the_rule(metric, e):
    for margin, fails in ((e - D, True), (e, False), (e + D, False)):
        c = Cell("v")
        rect(c, 0, 0, 2 + 2 * margin, 2 + 2 * margin, METAL)
        rect(c, margin, margin, margin + 2, margin + 2, VIA)
        assert bool(check_enclosure(c, VIA, METAL, e, metric=metric)) is fails, (margin, metric)


def test_inner_outside_outer_is_an_enclosure_violation():
    c = Cell("v")
    rect(c, 0, 0, 2, 2, METAL)
    rect(c, 5, 5, 6, 6, VIA)
    v = check_enclosure(c, VIA, METAL, 0.1)
    assert len(v) == 1 and area(v) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("s", [0.15, 1.0])
def test_separation_toggles_at_the_rule(metric, s):
    for gap, fails in ((s - D, True), (s, False), (s + D, False)):
        c = Cell("s")
        rect(c, 0, 0, 10, 0.5, WG)
        rect(c, 0, 0.5 + gap, 10, 2.5 + gap, METAL)
        assert bool(check_separation(c, METAL, WG, s, metric=metric)) is fails, (gap, metric)
        same = Cell("ss")
        rect(same, 0, 0, 10, 0.5)
        rect(same, 0, 0.5 + gap, 10, 1.0 + gap)
        assert bool(check_space(same, WG, s, metric=metric)) is fails, (gap, metric)


@pytest.mark.parametrize("offset,expected", [
    (0.6, {"euclidean": True, "square": True, "projection": False}),     # diagonal 0.85 s
    (0.8, {"euclidean": False, "square": True, "projection": False}),    # diagonal 1.13 s, max(dx, dy) 0.8 s
])
def test_corner_to_corner_spacing_follows_each_metric(offset, expected):
    s = 1.0
    c = Cell("diag")
    rect(c, 0, 0, 2, 2, WG)
    rect(c, 2 + offset * s, 2 + offset * s, 4, 4, METAL)
    got = {m: bool(check_separation(c, WG, METAL, s, metric=m)) for m in METRICS}
    assert got == expected
    same = Cell("diag1")
    rect(same, 0, 0, 2, 2)
    rect(same, 2 + offset * s, 2 + offset * s, 4, 4)
    assert {m: bool(check_space(same, WG, s, metric=m)) for m in METRICS} == expected


def test_facing_edges_fail_projection_but_offset_edges_do_not():
    c = Cell("p")
    rect(c, 0, 0, 2, 2)
    rect(c, 2.5, 1.0, 4, 3)          # facing, 0.5 apart, projections overlap over 1 um
    v = check_space(c, WG, 1.0, metric="projection")
    assert len(v) == 1
    # the half-growths (0.5 - 2e-4 each) overlap by (1 - 4e-4) - 0.5 across the 1 um of facing edges
    assert area(v) == pytest.approx((1.0 - 4e-4 - 0.5) * 1.0, abs=1e-6)


@pytest.mark.parametrize("metric", METRICS)
def test_extension_toggles_and_ignores_where_the_core_leaves(metric):
    x = 1.0
    for ext, fails in ((x - D, True), (x, False), (x + D, False)):
        c = Cell("rib")
        rect(c, -10, -0.25, 10, 0.25, WG)                     # leaves the slab at both ends
        rect(c, -5, -0.25 - ext, 5, 0.25 + ext, SLAB)
        assert bool(check_extension(c, WG, SLAB, x, metric=metric)) is fails, (ext, metric)
    capped = Cell("cap")                                       # core ends inside: slab must pass its end too
    rect(capped, -4.5, -0.25, 4.5, 0.25, WG)
    rect(capped, -5, -1.5, 5, 1.5, SLAB)
    assert check_extension(capped, WG, SLAB, 1.0)              # 0.5 um end caps < 1 um


def test_not_overlap_and_inside():
    c = Cell("o")
    rect(c, 0, 0, 4, 1, WG)
    rect(c, 3, 0, 6, 1, METAL)
    assert area(check_not_overlap(c, WG, METAL)) == pytest.approx(1.0, abs=1e-6)
    assert area(check_inside(c, WG, METAL)) == pytest.approx(3.0, abs=1e-6)


def test_min_angle_flags_acute_tips_only():
    c = Cell("a")
    t30 = np.tan(np.deg2rad(30.0))
    c.add_polygon([(0, 0), (4, 0), (4, 4 * t30)])                                   # 30-60-90
    rect(c, 10, 0, 12, 2)                                                            # square
    c.add_polygon([(20, 0), (23, 0), (24, 1), (24, 3), (20, 3)])                     # 135 deg chamfer
    frame = Cell("f")                                                                # hole: 270 deg material corners
    rect(frame, 30, 0, 36, 6)
    frame.add_polygon([(31, 1), (35, 1), (35, 5), (31, 5)], (1, 1))
    c.add_ref(frame)
    acute = check_min_angle(c, WG, 45.0)
    assert len(acute) == 1
    tip = acute[0].polygon[0]
    assert np.allclose(tip, (0, 0), atol=1e-3)
    assert len(check_min_angle(c, WG, 25.0)) == 0
    assert len(check_min_angle(c, WG, 61.0)) == 2                                    # 30 and 60 deg corners
    merged_hole = Cell("mh")                                                         # a real hole, via booleans
    for x0, y0, x1, y1 in ((0, 0, 10, 2), (0, 8, 10, 10), (0, 2, 2, 8), (8, 2, 10, 8)):
        rect(merged_hole, x0, y0, x1, y1)
    assert check_min_angle(merged_hole, WG, 89.0) == []


def test_translated_layout_has_every_vertex_off_grid():
    c = lc.mmi2x2()
    assert check_offgrid_vertices(c, WG, 0.001) == []
    top = Cell("shift")
    top.add_ref(c, origin=(0.0005, 0.0))
    v = check_offgrid_vertices(top, WG, 0.001)
    assert len(v) == len({tuple(p) for p, _l in c.polygons for p in np.round(p, 9)})


def test_max_width_reports_the_wide_feature():
    length = 40.0
    for w, fails in ((5.0 - D, False), (5.0, False), (5.0 + D, True), (10.0, True)):
        c = Cell("m")
        rect(c, 0, 0, length, w, METAL)
        v = check_max_width(c, METAL, 5.0)
        assert bool(v) is fails, w
        if fails:
            assert area(v) == pytest.approx(w * length, rel=1e-6)


def test_hole_area_toggle():
    for hole, fails in ((0.3 - D, True), (0.3 + D, False)):
        c = Cell("h")
        rect(c, 0, 0, 10, 3)
        rect(c, 0, 3 + hole, 10, 10)
        rect(c, 0, 3, 3, 3 + hole)
        rect(c, 3 + hole, 3, 10, 3 + hole)
        v = check_hole_area(c, WG, 0.09)
        assert bool(v) is fails
        if fails:
            assert area(v) == pytest.approx(hole * hole, rel=1e-3)


# --------------------------------------------------------------------------- #
# decks, invariance, waivers
# --------------------------------------------------------------------------- #
def _test_layout() -> Cell:
    c = Cell("chip")
    rect(c, 0, 0, 30, 0.5, WG)
    rect(c, 0, 0.6, 30, 1.1, WG)                     # 0.1 um gap: space
    rect(c, 0, 1.3, 30, 4.0, METAL)                  # metal 0.2 um from the waveguide: separation
    rect(c, 1.0, 1.4, 2.0, 2.4, VIA)                 # via with 0.1 um enclosure on one side
    rect(c, 40, 0, 60, 12, METAL)                    # 12 um wide: max width
    c.add_polygon([(70, 0), (74, 0), (74, 1)], WG)   # 14 deg tip
    rect(c, -5, -1.25, 35, 1.85, SLAB)
    return c


def _deck() -> RuleDeck:
    return RuleDeck([
        Rule("WG.W", "width", WG, value=0.15), Rule("WG.S", "space", WG, value=0.15),
        Rule("WG.A", "area", WG, value=1.0), Rule("WG.ANG", "min_angle", WG, value=30.0),
        Rule("M.EN.V", "enclosure", VIA, METAL, 0.2), Rule("M.SEP.WG", "separation", METAL, WG, 1.0),
        Rule("SLAB.EXT", "extension", WG, SLAB, 0.8), Rule("M.MAXW", "max_width", METAL, value=10.0),
        Rule("WG.GRID", "grid", WG, value=0.001), Rule("M.HOLE", "hole_area", METAL, value=0.5),
    ])


def _signature(report) -> dict:
    out: dict = {}
    for v in report.violations:
        n, a = out.get(v.rule, (0, 0.0))
        out[v.rule] = (n + 1, a + v.area)
    return {k: (n, round(a, 3)) for k, (n, a) in out.items()}


def test_deck_on_a_test_layout():
    sig = _signature(_deck().run(_test_layout()))
    # the 14 degree tip is also narrower than the minimum width near its apex
    assert set(sig) == {"WG.S", "WG.W", "WG.ANG", "M.EN.V", "M.SEP.WG", "SLAB.EXT", "M.MAXW"}
    assert sig["WG.ANG"][0] == 1
    assert sig["M.MAXW"] == (1, 240.0)


@pytest.mark.parametrize("rotation,mirror", [(90.0, False), (0.0, True), (270.0, True)])
def test_counts_and_areas_are_invariant_under_rotation_and_mirroring(rotation, mirror):
    base = _signature(_deck().run(_test_layout()))
    top = Cell("rot")
    top.add_ref(_test_layout(), origin=(13.0, -4.0), rotation=rotation, mirror=mirror)
    assert _signature(_deck().run(top)) == base


def test_width_space_area_rules_agree_with_the_precheck():
    c = lc.directional_coupler(gap=0.1)
    deck = RuleDeck([Rule("w", "width", WG, value=0.15), Rule("s", "space", WG, value=0.15),
                     Rule("a", "area", WG, value=10.0)])
    pre = run_drc_precheck(c, {WG: {"min_width": 0.15, "min_space": 0.15, "min_area": 10.0}})
    ours = deck.run(c)
    assert sorted(round(v.area, 9) for v in ours.violations) == sorted(round(v.area, 9) for v in pre.violations)


def test_pdk_deck_and_json_roundtrip(tmp_path):
    deck = RuleDeck.from_pdk(demo_pdk())
    assert [r.kind for r in deck.rules] == ["width", "space", "width", "space"]
    full = _deck()
    back = RuleDeck.from_json(full.to_json())
    assert back.rules == full.rules
    path = tmp_path / "deck.json"
    full.to_json(path)
    assert RuleDeck.from_json(path).rules == full.rules
    assert json.loads(path.read_text())["rules"][0]["name"] == "WG.W"


def test_rule_validation():
    with pytest.raises(ValueError, match="unknown rule kind"):
        Rule("x", "bogus", WG, value=1.0)
    with pytest.raises(ValueError, match="second layer"):
        Rule("x", "enclosure", WG, value=1.0)
    with pytest.raises(ValueError, match="takes no second layer"):
        Rule("x", "width", WG, METAL, 1.0)
    with pytest.raises(ValueError, match="metric"):
        Rule("x", "space", WG, value=1.0, metric="manhattan")
    with pytest.raises(ValueError, match="positive"):
        Rule("x", "space", WG, value=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Rule("x", "min_density", WG, value=1.5)
    with pytest.raises(ValueError, match="duplicate"):
        RuleDeck([Rule("x", "width", WG, value=1.0), Rule("x", "space", WG, value=1.0)])


def test_density_rules_in_a_deck():
    c = Cell("d")
    rect(c, 0, 0, 100, 10)                 # bottom 50 x 50 windows: density 10 * 50 / 2500 = 0.2
    rect(c, 0, 90, 100, 100, METAL)        # sets the region to 100 x 100; top windows: density 0
    deck = RuleDeck([Rule("WG.DMIN", "min_density", WG, value=0.2, window=50.0),
                     Rule("WG.DMAX", "max_density", WG, value=0.15, window=50.0)])
    assert _signature(deck.run(c)) == {"WG.DMIN": (2, 5000.0), "WG.DMAX": (2, 5000.0)}


def test_waivers_move_violations_and_survive_str():
    c = _test_layout()
    rep = _deck().run(c)
    n = len(rep.violations)
    angle = [v for v in rep.violations if v.rule == "WG.ANG"]
    box = np.array([(65, -5), (80, -5), (80, 5), (65, 5)])
    out = _deck().run(c, waivers=[Waiver("WG.ANG", box, "tip of a test taper")])
    assert isinstance(out, DeckReport)
    assert len(out.violations) == n - len(angle) and len(out.waived) == len(angle)
    assert "waived" in str(out)
    other_rule = apply_waivers(rep, [Waiver("WG.S", box, "wrong rule")])
    assert len(other_rule.violations) == n
    star = apply_waivers(rep, [Waiver("*", np.array([(-100, -100), (200, -100), (200, 100), (-100, 100)]), "all")])
    assert star.clean and len(star.waived) == n


# --------------------------------------------------------------------------- #
# marker databases and KLayout decks
# --------------------------------------------------------------------------- #
def test_lyrdb_roundtrip_gives_identical_polygons(tmp_path):
    rep = _deck().run(_test_layout())
    path = write_lyrdb(rep, tmp_path / "drc.lyrdb", top="chip")
    back = read_lyrdb(path)
    assert len(back.violations) == len(rep.violations)
    for a, b in zip(rep.violations, back.violations, strict=True):
        assert (a.rule, a.layer, a.value) == (b.rule, b.layer, b.value)
        assert np.array_equal(a.polygon, b.polygon)
    text = (tmp_path / "drc.lyrdb").read_text()
    assert "<report-database>" in text and "<top-cell>chip</top-cell>" in text and "polygon: (" in text


def test_lyrdb_reads_klayout_value_types(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<report-database><categories>
 <category><name>W</name><description>min width</description><categories>
  <category><name>sub</name><description/></category></categories></category>
</categories><cells><cell><name>TOP</name></cell></cells><items>
 <item><category>W</category><cell>TOP</cell><values><value>edge-pair: (0,0;10,0)/(10,0.1;0,0.1)</value></values></item>
 <item><category>'W.sub'</category><cell>TOP</cell><values><value>box: (0,0;2,1)</value></values></item>
 <item><category>W</category><cell>TOP</cell><values><value>text: 'hello'</value></values></item>
</items></report-database>"""
    path = tmp_path / "kl.lyrdb"
    path.write_text(xml)
    rep = read_lyrdb(path)
    assert [v.rule for v in rep.violations] == ["W", "W.sub"]
    assert rep.violations[0].area == pytest.approx(1.0)
    assert rep.violations[1].area == pytest.approx(2.0)
    assert set(rep.rules) == {"W", "W.sub"}


def test_lvs_report_writes_lyrdb(tmp_path):
    top = Cell("ov")
    top.add_ref(lc.straight(10.0), name="a")
    top.add_ref(lc.straight(10.0), origin=(7.0, 0.0), name="b")
    rep = run_lvs_lite(top)
    assert isinstance(rep, LvsReport)
    back = read_lyrdb(rep.to_lyrdb(tmp_path / "lvs.lyrdb", top="ov"))
    assert [v.rule for v in back.violations] == ["short"]
    assert back.violations[0].area == pytest.approx(1.5, abs=1e-5)


def test_generated_klayout_deck_text():
    text = _deck().to_klayout_drc()
    assert text.startswith("# KLayout DRC deck generated by photonix")
    for line in (
        "source($input)", 'report("photonix", $report)', "l1_0 = input(1, 0)",
        'l1_0.width(0.15.um, euclidian).polygons.output("WG.W", "width 0.15")',
        'l1_0.space(0.15.um, euclidian).polygons.output("WG.S", "space 0.15")',
        'l1_0.with_area(nil, 1.0.um2).output("WG.A", "area 1")',
        'l11_0.enclosing(l12_0, 0.2.um, euclidian).polygons.output("M.EN.V", "enclosure 0.2")',
        'l11_0.separation(l1_0, 1.0.um, euclidian).polygons.output("M.SEP.WG", "separation 1")',
        'l1_0.ongrid(0.001.um).output("WG.GRID", "grid 0.001")',
        'l11_0.holes.with_area(nil, 0.5.um2).output("M.HOLE", "hole_area 0.5")',
        "# WG.ANG: min_angle 30 is not translated (check it with photonix)",
    ):
        assert line in text.splitlines(), line
    custom = RuleDeck([Rule("A", "width", WG, value=0.2)]).to_klayout_drc(report_var="rpt", input_var="gds")
    assert "source($gds)" in custom and 'report("photonix", $rpt)' in custom


@pytest.mark.skipif(not klayout_available(), reason="requires a KLayout executable")
def test_klayout_cross_check(tmp_path):    # pragma: no cover - needs KLayout
    deck = RuleDeck([Rule("WG.W", "width", WG, value=0.15), Rule("WG.S", "space", WG, value=0.15),
                     Rule("WG.A", "area", WG, value=1.0), Rule("M.MAXW", "max_width", METAL, value=10.0)])
    res = compare_with_klayout(_test_layout(), deck, workdir=tmp_path)
    assert all(r["presence"] for r in res.values())
    for name in ("WG.A", "M.MAXW"):
        assert res[name]["area_agrees"], res[name]
