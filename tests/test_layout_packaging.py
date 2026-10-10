"""Die-level assembly: text, labels, packing/placement, fibre arrays, pads, die frames, floorplan checks."""
from __future__ import annotations

import math
import os
import string
import warnings

import numpy as np
import pytest

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc
from photonix.layout.labels import TEXT_STROKE_RATIO
from photonix.layout.library import (
    add_test_labels,
    die_frame,
    fiber_array,
    loopback_length,
    opt_in_label,
    pad_array,
)

gdstk = pytest.importorskip("gdstk")

PRINTABLE = "".join(ch for ch in string.printable if not ch.isspace())


def _square(side, name="sq"):
    return lay.Cell(name).add_polygon([(0, 0), (side, 0), (side, side), (0, side)])


def _rect(w, h, name="r", x0=0.0, y0=0.0):
    return lay.Cell(name).add_polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])


def _union_area(polys):
    if not polys:
        return 0.0
    return sum(p.area() for p in gdstk.boolean([gdstk.Polygon(p) for p in polys], [], "or", precision=1e-6))


# --------------------------------------------------------------------------- #
# Text and labels
# --------------------------------------------------------------------------- #
def test_font_stroke_ratio_from_independent_geometry():
    """Smallest interior vertex-to-edge distance over every printable glyph is exactly size/16."""
    size = 16.0
    worst = math.inf
    for ch in PRINTABLE:
        for pts, _layer in lay.text(ch, size=size).polygons:
            q = np.roll(pts, -1, axis=0)
            poly = gdstk.Polygon(pts)
            for i, v in enumerate(pts):
                ab = q - pts
                t = np.clip(np.einsum("ij,ij->i", v - pts, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
                proj = pts + t[:, None] * ab
                dist = np.linalg.norm(proj - v, axis=1)
                for j in np.argsort(dist):
                    if j in (i, (i - 1) % len(pts)) or dist[j] < 1e-12 or dist[j] >= worst:
                        continue
                    if gdstk.inside([tuple(0.5 * (v + proj[j]))], [poly])[0]:
                        worst = dist[j]
                        break
    assert worst == pytest.approx(size * TEXT_STROKE_RATIO, abs=1e-12)


def test_min_stroke_text_passes_width_rule():
    rule = 0.5
    small = lay.text(PRINTABLE, size=4.0)                    # strokes of 0.25 um
    assert lay.check_min_width(small, lay.labels.TEXT_LAYER, rule)
    big = lay.text(PRINTABLE, size=4.0, min_stroke=rule)     # enlarged to size 8
    assert not lay.check_min_width(big, lay.labels.TEXT_LAYER, rule)
    (x0, y0), (x1, y1) = big.bbox()
    assert y1 - y0 == pytest.approx(2.0 * (small.bbox()[1][1] - small.bbox()[0][1]))


@pytest.mark.parametrize("anchor", lay.labels.ANCHORS)
def test_text_anchor_lands_on_position(anchor):
    c = lay.text("Die-07", size=10.0, position=(3.0, -4.0), anchor=anchor)
    (x0, y0), (x1, y1) = c.bbox()
    x = {"w": x0, "e": x1}.get(anchor[-1], 0.5 * (x0 + x1))
    y = {"s": y0, "n": y1}.get(anchor[0], 0.5 * (y0 + y1))
    assert (x, y) == pytest.approx((3.0, -4.0), abs=1e-12)
    assert not c.ports and c.model is None


def test_label_transforms_with_references():
    child = lay.LabeledCell("child").add_label("x", (1.0, 0.0), rotation=10.0)
    top = lay.LabeledCell("top")
    top.add_ref(child, origin=(10.0, 0.0), rotation=90.0)
    top.add_ref(child, origin=(0.0, 5.0), mirror=True)
    got = [(lbl.position, lbl.rotation) for lbl in top.get_labels()]
    assert got[0][0] == pytest.approx((10.0, 1.0)) and got[0][1] == pytest.approx(100.0)
    assert got[1][0] == pytest.approx((1.0, 5.0)) and got[1][1] == pytest.approx(350.0)
    assert [lb.text for lb in top.flatten().labels] == ["x", "x"]
    with pytest.raises(ValueError):
        lay.Label("")


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("heuristic", ["bssf", "blsf", "baf", "bl"])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_equal_squares_fill_the_bin_completely(n, heuristic):
    side = 7.0
    bins = lay.pack([_square(side)] * (n * n), max_size=(n * side, n * side), spacing=0.0, heuristic=heuristic)
    assert len(bins) == 1 and len(bins[0].references) == n * n
    corners = sorted(lay.reference_bbox(r)[0] for r in bins[0].references)
    assert corners == sorted((i * side, j * side) for i in range(n) for j in range(n))


def test_maxrects_finds_the_perfect_tiling_of_dominoes():
    bins = lay.pack([_rect(10.0, 20.0)] * 8, max_size=(40.0, 40.0), spacing=0.0)
    assert len(bins) == 1 and len(bins[0].references) == 8


def _check_packing(bins, cells, max_size, spacing):
    placed = [(b, r) for b in bins for r in b.references]
    assert len(placed) == len(cells)
    for b in bins:
        boxes = [lay.reference_bbox(r) for r in b.references]
        for (lo, hi) in boxes:
            assert lo[0] >= -1e-9 and lo[1] >= -1e-9 and hi[0] <= max_size[0] + 1e-9 and hi[1] <= max_size[1] + 1e-9
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (a0, a1), (b0, b1) = boxes[i], boxes[j]
                gap_x = max(b0[0] - a1[0], a0[0] - b1[0])
                gap_y = max(b0[1] - a1[1], a0[1] - b1[1])
                assert max(gap_x, gap_y) >= spacing - 1e-9, (i, j, boxes[i], boxes[j])


@pytest.mark.parametrize("heuristic", ["bssf", "blsf", "baf", "bl"])
def test_random_rectangles_pack_without_overlap_and_deterministically(heuristic):
    rng = np.random.default_rng(7)
    cells = [_rect(float(w), float(h), name=f"c{k}", x0=float(x), y0=float(y))
             for k, (w, h, x, y) in enumerate(zip(rng.uniform(5, 60, 40), rng.uniform(5, 60, 40),
                                                  rng.uniform(-50, 50, 40), rng.uniform(-50, 50, 40), strict=True))]
    size, spacing = (150.0, 120.0), 3.0
    bins = lay.pack(cells, max_size=size, spacing=spacing, heuristic=heuristic)
    _check_packing(bins, cells, size, spacing)
    again = lay.pack(cells, max_size=size, spacing=spacing, heuristic=heuristic)
    assert [[(r.name, r.origin) for r in b.references] for b in bins] == \
        [[(r.name, r.origin) for r in b.references] for b in again]
    # quality: every bin but the last is at least 70 % full (inflated boxes), in line with the
    # 80-90 % occupancy Jylanki reports for MaxRects on random boxes
    def fill(b):
        boxes = [lay.reference_bbox(r) for r in b.references]
        return sum((hi[0] - lo[0] + spacing) * (hi[1] - lo[1] + spacing) for lo, hi in boxes) / (
            (size[0] + spacing) * (size[1] + spacing))

    assert all(fill(b) >= 0.7 for b in bins[:-1])


def test_rotation_is_opt_in_and_quarter_turn_only():
    tall_bin = (12.0, 40.0)
    with pytest.raises(ValueError, match="does not fit"):
        lay.pack([_rect(30.0, 10.0)], max_size=tall_bin, spacing=0.0)
    bins = lay.pack([_rect(30.0, 10.0)], max_size=tall_bin, spacing=0.0, allow_rotation=True)
    ref = bins[0].references[0]
    assert ref.rotation == 90.0
    lo, hi = lay.reference_bbox(ref)
    assert lo[0] >= -1e-9 and hi[0] <= 12.0 + 1e-9 and hi[1] <= 40.0 + 1e-9


def test_overflow_opens_new_bins():
    bins = lay.pack([_square(10.0)] * 10, max_size=(20.0, 20.0), spacing=0.0)
    assert [len(b.references) for b in bins] == [4, 4, 2]
    assert [b.name for b in bins] == ["pack_0", "pack_1", "pack_2"]


# --------------------------------------------------------------------------- #
# Grid, align, distribute
# --------------------------------------------------------------------------- #
def test_grid_centres_lie_on_the_exact_lattice():
    cells = [lc.straight(L) for L in (10.0, 30.0, 5.0, 12.0, 7.0)] + [lc.ring_single(radius=8.0)]
    g = lay.grid(cells, spacing=(4.0, 6.0))
    wmax = max(c.bbox()[1][0] - c.bbox()[0][0] for c in cells)
    hmax = max(c.bbox()[1][1] - c.bbox()[0][1] for c in cells)
    for k, ref in enumerate(g.references):
        r, c = divmod(k, 3)
        (x0, y0), (x1, y1) = lay.reference_bbox(ref)
        assert (0.5 * (x0 + x1), 0.5 * (y0 + y1)) == pytest.approx((c * (wmax + 4.0), -r * (hmax + 6.0)), abs=1e-12)
    assert [r.name for r in g.references][:2] == ["straight_0_0", "straight_0_1"]


def test_grid_alignment_and_shape():
    cells = [_rect(10.0, 4.0), _rect(4.0, 10.0)]
    g = lay.grid(cells, shape=(2, 1), spacing=2.0, align="sw")
    assert [lay.reference_bbox(r)[0] for r in g.references] == [(-5.0, -5.0), (-5.0, -17.0)]
    with pytest.raises(ValueError, match="cannot hold"):
        lay.grid(cells * 3, shape=(2, 2))


def test_align_and_distribute_with_rotated_references():
    top = lay.Cell("top")
    a = top.add_ref(lc.straight(10.0), origin=(3.0, 1.0))
    b = top.add_ref(lc.straight(20.0), origin=(-4.0, 9.0), rotation=90.0)
    lay.align([a, b], "bottom")
    assert lay.reference_bbox(a)[0][1] == pytest.approx(lay.reference_bbox(b)[0][1])
    lay.align([a, b], "x", value=50.0)
    assert [0.5 * (lay.reference_bbox(r)[0][0] + lay.reference_bbox(r)[1][0]) for r in (a, b)] == \
        pytest.approx([50.0, 50.0])
    lay.distribute([a, b], "y", spacing=3.0)
    assert lay.reference_bbox(b)[0][1] - lay.reference_bbox(a)[1][1] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Fibre array
# --------------------------------------------------------------------------- #
def _analytic_loop(n, pitch=127.0, r=10.0, clearance=5.0, w=0.5):
    """Loop geometry of the default coupler from first principles (see fiber_array's docs)."""
    gc = lc.grating_coupler()
    dx = gc.ports["o1"].center[0] - gc.ports["o2"].center[0]      # waveguide port 19.45 um above the fibre
    half = 0.5 * (gc.bbox()[1][1] - gc.bbox()[0][1])                # 6 um half width
    below = gc.ports["o2"].center[0] - gc.bbox()[0][0]               # body extends 9.765 um below the fibre
    x0, xn, yw = 0.0, (n - 1) * pitch, dx
    y_top, y_bot = yw + r, -below - clearance - w / 2
    x_left = min(x0 - 2 * r, x0 - half - clearance - w / 2 - r)
    x_right = max(xn + 2 * r, xn + half + clearance + w / 2 + r)
    corners = [(x0, y_top), (x_left, y_top), (x_left, y_bot), (x_right, y_bot), (x_right, y_top), (xn, y_top)]
    segs = np.diff(np.array([(x0, yw), *corners, (xn, yw)]), axis=0)
    return float(np.abs(segs).sum() - 12 * r + 6 * math.pi * r / 2), corners


@pytest.mark.parametrize("n", [2, 4, 8])
def test_fiber_array_pitch_ports_and_loopback(n):
    fa = fiber_array(n)
    ports = fa.get_ports()
    for k in range(n):
        assert ports[f"gc{k}/o2"].center == pytest.approx((k * 127.0, 0.0), abs=1e-9)
    assert list(fa.ports) == [f"o{i}" for i in range(1, n - 1)]
    for i, p in enumerate(fa.ports.values(), start=1):
        assert p.orientation == 90.0 and p.center[0] == pytest.approx(i * 127.0, abs=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # LVS-lite: no misplaced or mismatched ports
        nl = lay.extract_netlist(fa)
    assert nl.connections == {("gc0", "o1"): ("loopback", "o1"), (f"gc{n - 1}", "o1"): ("loopback", "o2")}
    assert sorted(nl.ports) == sorted([f"gc{k}_o2" for k in range(n)] + list(fa.ports))
    loop = next(r for r in fa.references if r.name == "loopback").cell
    want, corners = _analytic_loop(n)
    assert loop.model_settings == {"length": pytest.approx(want, rel=1e-12), "n_bends": 6}
    assert loopback_length(corners, (0, 19.45), ((n - 1) * 127.0, 19.45)) == pytest.approx(want, rel=1e-12)
    assert _union_area([p for p, _ in loop.polygons]) / 0.5 == pytest.approx(want, rel=1e-4)


def _corners_of(route_cell):
    """Corner points of a filleted Manhattan route, from its drawn centre line's straight runs."""
    (lo_x, lo_y), (hi_x, hi_y) = route_cell.bbox()
    w = 0.5
    xs = sorted({round(lo_x + w / 2, 9), round(hi_x - w / 2, 9)})
    ys = sorted({round(lo_y + w / 2, 9), round(hi_y - w / 2, 9)})
    p0 = route_cell.ports["o1"].center
    p1 = route_cell.ports["o2"].center
    top = max(ys)
    return [(p0[0], top), (xs[0], top), (xs[0], ys[0]), (xs[-1], ys[0]), (xs[-1], top), (p1[0], top)]


def test_loopback_keeps_clear_of_every_coupler():
    fa = fiber_array(5, clearance=5.0)
    loop = next(r for r in fa.references if r.name == "loopback")
    loop_poly = [gdstk.Polygon(p) for p, _ in loop.cell.polygons]
    for ref in fa.references:
        if not ref.name.startswith("gc"):
            continue
        body = [gdstk.Polygon(pts) for pts, _ in lay.Cell("w", references=[ref]).get_polygons()]
        grown = gdstk.offset(body, 5.0 - 1e-3, join="round")
        hits = gdstk.boolean(loop_poly, grown, "and")
        port = np.asarray(ref.port("o1").center)
        for h in hits:                                          # only where the loop joins its own coupler
            assert ref.name in ("gc0", "gc4")
            assert np.max(np.linalg.norm(np.asarray(h.points) - port, axis=1)) < 5.0 + 1e-3 + 0.5


def test_loopback_transmission_is_two_couplers_and_the_drawn_waveguide():
    fa = fiber_array(4)
    wl = px.linspace(1.53, 1.57, 5)
    s = px.circuit.circuit_from_netlist(lay.extract_netlist(fa))(wl=wl)
    loop = next(r for r in fa.references if r.name == "loopback").cell
    g = px.components.MODELS["grating_coupler"](wl=wl)
    route = px.components.MODELS["route"](wl=wl, **loop.model_settings)
    want = px.power(g[("o2", "o1")]) * px.power(route[("o1", "o2")]) * px.power(g[("o1", "o2")])
    np.testing.assert_allclose(np.asarray(px.power(s[("gc0_o2", "gc3_o2")])), np.asarray(want), rtol=1e-10)


def test_fiber_array_variants():
    euler = fiber_array(3, bend="euler")
    loop = next(r for r in euler.references if r.name == "loopback").cell
    assert loop.model_settings["n_bends"] == 6
    from photonix.layout._geometry import corner_offset, euler_curve
    d, arc = corner_offset(euler_curve(10.0, 90.0, 0.5, 48)), 0.5 * math.pi * 10.0 * 1.5
    corners = [tuple(c) for c in _corners_of(loop)]
    start, end = loop.ports["o1"].center, loop.ports["o2"].center
    manhattan = np.abs(np.diff(np.array([start, *corners, end]), axis=0)).sum()
    assert loop.model_settings["length"] == pytest.approx(manhattan - 12 * d + 6 * arc, rel=1e-11)
    side = fiber_array(3, orientation=0.0, loopback=False)
    ports = side.get_ports()
    np.testing.assert_allclose([ports[f"gc{k}/o2"].center for k in range(3)],
                               [(0.0, 0.0), (0.0, -127.0), (0.0, -254.0)], atol=1e-9)
    assert [p.orientation for p in side.ports.values()] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="n >= 2"):
        fiber_array(1)
    with pytest.raises(ValueError, match="no port"):
        fiber_array(2, fiber_port="fibre")


def test_pad_array():
    pa = pad_array(4, pitch=120.0, size=90.0)
    assert [p.center for p in pa.ports.values()] == [(k * 120.0, 0.0) for k in range(4)]
    assert all(p.port_type == lay.ELECTRICAL and p.width == 90.0 for p in pa.ports.values())
    assert _union_area([p for p, _ in pa.get_polygons()]) == pytest.approx(4 * 90.0 ** 2)
    with pytest.raises(ValueError):
        pad_array(2, pitch=50.0, size=80.0)


# --------------------------------------------------------------------------- #
# Die frame and floorplan checks
# --------------------------------------------------------------------------- #
def _layer_polys(cell, layer):
    return [p for p, lay_ in cell.get_polygons() if lay_ == layer]


def test_die_frame_areas_are_exact():
    W, H, d, k = 3000.0, 2000.0, 100.0, 50.0
    die = die_frame((W, H), dicing_lane=d, keepout=k, die_id="A7")
    assert isinstance(die, lay.DieCell)
    assert _union_area(_layer_polys(die, (99, 0))) == W * H - (W - 2 * d) * (H - 2 * d)
    assert _union_area(_layer_polys(die, (98, 0))) == (W - 2 * d) * (H - 2 * d) - (W - 2 * d - 2 * k) * (
        H - 2 * d - 2 * k)
    assert die.usable_box() == ((150.0, 150.0), (2850.0, 1850.0))
    for layer in ((99, 1), (66, 0)):                       # corner marks and die ID stay out of the usable area
        for pts in _layer_polys(die, layer):
            inner = gdstk.boolean([gdstk.Polygon(pts)], [gdstk.rectangle(*die.usable_box())], "and")
            assert not inner
    assert len(_layer_polys(die, (99, 1))) == 4 and _layer_polys(die, (66, 0))


def test_facet_trench_and_facet_position():
    die = die_frame((1000.0, 800.0), dicing_lane=50.0, facet_sides=("east", "W"), facet_trench_layer=(97, 0),
                    facet_trench_width=20.0)
    assert die.facet_sides == ("E", "W")
    assert (die.facet_position("E"), die.facet_position("W")) == (980.0, 20.0)
    trenches = _layer_polys(die, (97, 0))
    assert len(trenches) == 2 and sum(_union_area([t]) for t in trenches) == 2 * 20.0 * 800.0
    plain = die_frame((1000.0, 800.0), facet_sides=("E",))
    assert plain.facet_position("E") == 1000.0 and not _layer_polys(plain, (97, 0))


def _chip(die, *extra):
    top = lay.Cell("chip")
    top.add_ref(die, name="frame")
    for name, cell, origin, rot in extra:
        top.add_ref(cell, origin=origin, rotation=rot, name=name)
    return top


def test_check_floorplan_clean_and_keepout():
    die = die_frame((2000.0, 2000.0), dicing_lane=100.0, keepout=50.0)
    ok = _chip(die, ("mzi", lc.mzi(), (500.0, 1000.0), 0.0), ("wg", lc.straight(100.0), (150.0, 300.0), 0.0))
    assert lay.check_floorplan(ok, die) == []
    intruding = _chip(die, ("wg", lc.straight(100.0), (149.0, 300.0), 0.0))       # 1 um into the keep-out
    issues = lay.check_floorplan(intruding, die)
    assert [(i.kind, i.instances) for i in issues] == [("keepout", ("wg",))]
    assert lay.check_floorplan(intruding, die, keepout=49.0) == []
    nested = _chip(die, ("blk", lc.mzi(), (151.0, 1000.0), 270.0))                 # a bend of the MZI pokes out
    kinds = {(i.kind, i.instances[0].split(".")[0]) for i in lay.check_floorplan(nested, die)}
    assert kinds == {("keepout", "blk")}
    assert all("." in i.instances[0] for i in lay.check_floorplan(nested, die))


def test_check_floorplan_sees_polygons_drawn_in_composites():
    die = die_frame((1000.0, 1000.0), dicing_lane=50.0, keepout=50.0)
    block = lay.Cell("block")
    block.add_ref(lc.straight(10.0), name="wg")
    block.add_polygon([(0, 0), (5, 0), (5, 5), (0, 5)], (11, 0))         # metal drawn in the composite itself
    issues = lay.check_floorplan(_chip(die, ("blk", block, (500.0, 97.0), 0.0)), die)
    assert [(i.kind, i.instances) for i in issues] == [("keepout", ("blk",)), ("keepout", ("blk.wg",))]


def test_check_floorplan_outside_and_overlap():
    die = die_frame((1000.0, 1000.0), dicing_lane=50.0, keepout=0.0)
    top = _chip(die, ("a", _rect(100.0, 100.0), (200.0, 200.0), 0.0), ("b", _rect(100.0, 100.0), (250.0, 250.0), 0.0),
                ("far", _rect(10.0, 10.0), (995.0, 500.0), 0.0))
    a, b = top.references[1], top.references[2]
    route = lay.route_manhattan(lay.Port("p", (300.0, 210.0), 0.0), lay.Port("q", (600.0, 600.0), 180.0))
    top.add_ref(route, name="r")
    issues = lay.check_floorplan(top, die)
    assert {(i.kind, i.instances) for i in issues} == {("overlap", ("a", "b")), ("outside_die", ("far",)),
                                                      ("keepout", ("far",))}
    ov = next(i for i in issues if i.kind == "overlap")
    assert ov.bbox == ((250.0, 250.0), (300.0, 300.0))
    assert ("overlap", ("b", "r")) in {(i.kind, i.instances) for i in lay.check_floorplan(top, die,
                                                                                         include_routes=True)}
    assert a is not b


def test_check_floorplan_edge_coupler_facets():
    die = die_frame((1000.0, 1000.0), dicing_lane=50.0, keepout=50.0, facet_sides=("E",))
    ec = lc.edge_coupler(200.0)

    def issues(x, y=500.0, rot=0.0, tol=1.0, d=die):
        return [(i.kind, i.instances) for i in lay.check_floorplan(_chip(d, ("ec", ec, (x, y), rot)), d,
                                                                    facet_tol=tol)]

    assert issues(800.0) == []                                  # tip on the facet
    assert issues(799.5) == []                                  # 0.5 um short, within 1 um
    assert issues(798.0) == [("facet", ("ec",))]                # 2 um inside the edge
    assert issues(798.0, tol=2.5) == []
    assert issues(200.0, rot=180.0) == [("facet", ("ec",))]     # west side is not a facet side
    trench = die_frame((1000.0, 1000.0), dicing_lane=50.0, keepout=50.0, facet_sides=("E",),
                       facet_trench_layer=(97, 0), facet_trench_width=30.0)
    assert issues(770.0, d=trench) == [] and issues(800.0, d=trench) == [("facet", ("ec",))]


# --------------------------------------------------------------------------- #
# Automated-test labels
# --------------------------------------------------------------------------- #
def test_opt_in_label_format_and_validation():
    assert opt_in_label("ring_10um", "LukasC", pol="TM", wl_nm=1310) == "opt_in_TM_1310_device_LukasC_ring_10um"
    for bad in (dict(user="lukas_c"), dict(user=""), dict(device="my ring"), dict(pol="XX"), dict(wl_nm=1.55)):
        kwargs = {"device": "d1", "user": "me", **bad}
        with pytest.raises(ValueError):
            opt_in_label(kwargs.pop("device"), kwargs.pop("user"), **kwargs)


def test_add_test_labels_on_a_device_with_couplers(tmp_path):
    top = lay.Cell("dev")
    gcs = [top.add_ref(lc.grating_coupler(), origin=(0.0, -127.0 * k), rotation=180.0, name=f"gc{k}")
           for k in range(3)]
    labeled = add_test_labels(top, gc_refs=gcs, device="mzi1", user="me")
    lbl = labeled.labels[0]
    assert lbl.text == "opt_in_TE_1550_device_me_mzi1" and lbl.layer == (10, 0)
    assert lbl.position == pytest.approx(gcs[0].port("o2").center)
    assert labeled.references == top.references and labeled is not top
    again = add_test_labels(labeled, gc_refs=gcs[:1], device="mzi2", user="me")
    assert again is labeled and len(labeled.labels) == 2
    path = os.fspath(tmp_path / "dev.gds")
    lay.write_layout(labeled, path)
    back = lay.read_layout(path)
    assert [(lb.text, lb.position, lb.layer) for lb in back.labels] == [(lb.text, lb.position, lb.layer)
                                                                        for lb in lay.snap_to_grid(labeled).labels]
    off_pitch = top.add_ref(lc.grating_coupler(), origin=(0.0, -300.0), rotation=180.0, name="gcx")
    with pytest.raises(ValueError, match="pitch"):
        add_test_labels(top, gc_refs=[gcs[0], off_pitch], device="x", user="me")
    assert add_test_labels(top, gc_refs=[gcs[0], off_pitch], device="x", user="me", pitch=None).labels
    stray = lay.Cell("other").add_ref(lc.grating_coupler())
    with pytest.raises(ValueError, match="directly in top"):
        add_test_labels(top, gc_refs=[stray], device="x", user="me")
