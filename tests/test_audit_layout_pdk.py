"""Regression tests for layout/PDK bugs found by the round-2 audit."""
from __future__ import annotations

import os

import numpy as np
import pytest

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc
from photonix.pdk import CrossSection, LayerLevel, LayerStack, Pdk

gdstk = pytest.importorskip("gdstk")


def _rect(c, x0, y0, x1, y1, layer=(1, 0)):
    c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer)


@pytest.mark.parametrize("w, flagged", [(0.5, 0), (0.4999, 0), (0.4995, 1), (0.49, 1)])
def test_drc_width_at_rule_value_passes(w, flagged):
    c = lay.Cell("w")
    _rect(c, 0, 0, 10, w)
    assert len(lay.check_min_width(c, (1, 0), 0.5)) == flagged


@pytest.mark.parametrize("g, flagged", [(0.2, 0), (0.1999, 0), (0.1995, 1)])
def test_drc_space_at_rule_value_passes(g, flagged):
    c = lay.Cell("s")
    _rect(c, 0, 0, 10, 0.5)
    _rect(c, 0, 0.5 + g, 10, 1.0 + g)
    assert len(lay.check_min_space(c, (1, 0), 0.2)) == flagged


def test_demo_pdk_heater_passes_its_own_metal_rule():
    pdk = px.pdk.demo_pdk()
    assert pdk.drc_precheck(pdk.get_layout("heater")).clean


@pytest.mark.parametrize("d, flagged", [(0.05, True), (0.1, True), (0.11, False)])
def test_drc_catches_diagonal_corner_gaps_euclidean(d, flagged):
    c = lay.Cell("d")
    _rect(c, 0, 0, 1, 1)
    _rect(c, 1 + d, 1 + d, 2, 2)
    assert bool(lay.check_min_space(c, (1, 0), 0.15)) is flagged    # distance d*sqrt(2) vs 0.15


def test_drc_flags_acute_tips_but_not_valid_small_bends():
    tri = lay.Cell("t")
    tri.add_polygon([(0, 0), (4, 0), (0, 1.07)])                         # ~15 degree tip
    assert lay.check_min_width(tri, (1, 0), 0.15)
    for radius, npts in ((1.0, 60), (1.0, 200), (1.5, 30)):
        assert not lay.check_min_width(lc.bend_circular(radius=radius, npts=npts), (1, 0), 0.15)


@pytest.mark.parametrize("length_x", [4.0, 10.0])
def test_racetrack_polygon_is_simple_and_drc_clean(length_x):
    c = lc.ring_single(gap=0.2, length_x=length_x)
    rules = {(1, 0): {"min_width": 0.15, "min_space": 0.15}}
    assert lay.run_drc_precheck(c, rules).clean
    ring_poly = gdstk.Polygon(c.polygons[1][0])
    merged = gdstk.boolean([ring_poly], [], "or")
    assert sum(m.area() for m in merged) == pytest.approx(ring_poly.area(), abs=2e-3)


def test_sharp_route_keeps_full_width():
    r = lay.route(lay.Port("a", (0, 0), 0), lay.Port("b", (100, 3), 180))
    assert gdstk.Polygon(r.polygons[0][0]).area() == pytest.approx(r.model_settings["length"] * 0.5, rel=1e-9)
    assert not lay.check_min_width(r, (1, 0), 0.45)


def test_stack_without_cover_raises_and_air_clad_solves_correctly():
    bad = LayerStack([LayerLevel("sub", None, 1.0, -3.0, 3.48), LayerLevel("box", None, 2.0, -2.0, 1.444),
                      LayerLevel("core", (1, 0), 0.22, 0.0, 3.4757)])
    with pytest.raises(ValueError, match="no unpatterned level covers"):
        bad.cross_section({(1, 0): 0.5})
    a = LayerStack([LayerLevel("box", None, 3, -3, 1.444), LayerLevel("air", None, 3, 0, 1.0),
                    LayerLevel("core", (1, 0), 0.22, 0, 3.4757)])
    b = LayerStack([LayerLevel("air", None, 3, 0, 1.0), LayerLevel("box", None, 3, -3, 1.444),
                    LayerLevel("core", (1, 0), 0.22, 0, 3.4757)])
    assert np.allclose(a.cross_section({(1, 0): 0.5}).eps, b.cross_section({(1, 0): 0.5}).eps, atol=1e-12)
    neff = a.solve_modes({(1, 0): 0.5}, resolution=30).neff0
    assert 2.3 < neff < 2.45            # below the fully oxide-clad 2.45, far from a Si slab (~3.4)


def test_stack_overlapping_levels_use_exact_area_fractions():
    # Two partially overlapping patterned levels: last one wins, areas exact.
    st = LayerStack([LayerLevel("clad", None, 10, -5, 1.0), LayerLevel("slab", (2, 0), 0.1, 0.0, 2.0),
                     LayerLevel("core", (1, 0), 0.2, 0.0, 3.0)])
    cs = st.cross_section({(1, 0): 0.5, (2, 0): 2.0}, resolution=20)
    h = cs.dx
    total = float(np.sum(cs.eps - 1.0) * h * h)
    expected = (2.0**2 - 1) * 2.0 * 0.1 - (2.0**2 - 1) * 0.5 * 0.1 + (3.0**2 - 1) * 0.5 * 0.2
    assert total == pytest.approx(expected, rel=1e-9)


def test_flattened_and_read_back_cells_refuse_to_simulate(tmp_path):
    flat = lc.mzi(delta_length=50.0).flatten()
    assert flat.model == lay.GEOMETRY_ONLY
    top = lay.Cell("top")
    top.add_ref(flat, name="m")
    with pytest.raises(ValueError, match="geometry-only"):
        lay.extract_netlist(top)
    path = os.fspath(tmp_path / "wg.gds")
    lay.write_gds(lc.straight(250.0), path)
    with pytest.raises(ValueError, match="geometry-only"):
        lay.extract_netlist(lay.read_gds(path))


def test_hand_drawn_cell_with_library_name_warns():
    c = lay.Cell("straight")
    _rect(c, 0, -0.25, 25, 0.25)
    c.add_port("o1", (0, 0), 180)
    c.add_port("o2", (25, 0), 0)
    with pytest.warns(UserWarning, match="defaults"):
        lay.extract_netlist(c)


def test_leaf_component_extracts_as_single_instance():
    pdk = px.pdk.demo_pdk()
    s = pdk.simulate(pdk.get_layout("straight", length=1e4))
    assert -10 * np.log10(float(px.power(s[("o1", "o2")]))) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="nothing to extract"):
        lay.extract_netlist(lay.Cell("empty"))


def test_add_model_binding_errors_and_kwargs_models():
    pdk = Pdk("x")

    def free(*, wl=1.55, **kw):
        return {("o1", "o2"): kw.get("neff", 0.0) + 0j}

    pdk.add_model("free", free, neff=3.0)
    assert complex(pdk.models()["free"](wl=1.55)[("o1", "o2")]) == 3.0
    with pytest.raises(TypeError, match="does not accept"):
        pdk.add_model("s", px.components.straight, not_a_param=1.0)
    with pytest.raises(KeyError):
        pdk.models("rib")
    pdk.add_cross_section(CrossSection("strip", (1, 0), 0.5))
    with pytest.raises(KeyError):
        pdk.models("rib")


def test_gds_names_layers_and_fingerprint_injectivity(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        lay.write_gds(lay.Cell(""), os.fspath(tmp_path / "e.gds"))
    c = lay.Cell("np")
    c.add_polygon([(0, 0), (1, 0), (1, 1)], (np.int64(3), np.int64(0)))
    lay.write_gds(c, os.fspath(tmp_path / "np.gds"))
    from photonix.layout.gds import _fingerprint

    a = lay.Cell("a").add_port("o1", (0, 0))
    b = lay.Cell("a" + repr(("o1", [0.0, 0.0], 0.0, 0.5, (1, 0), "optical")))
    assert _fingerprint(a, {}) != _fingerprint(b, {})


def test_grating_coupler_port_on_geometry_and_tooth_count():
    g = lc.grating_coupler()
    xs = np.concatenate([p[:, 0] for p, _ in g.polygons])
    assert g.ports["o1"].center[0] == pytest.approx(xs.max())
    assert len(lc.grating_coupler(length=0.6, period=0.2).polygons) == 3 + 1   # teeth + taper


def test_demo_pdk_mmi_layout_matches_model_geometry():
    pdk = px.pdk.demo_pdk()
    cell = pdk.get_layout("mmi1x2")
    (x0, y0), (x1, y1) = cell.bbox()
    ms = pdk.model_settings("mmi1x2")
    assert x1 - x0 == pytest.approx(ms["length_mmi"]) and y1 - y0 == pytest.approx(ms["width_mmi"])
