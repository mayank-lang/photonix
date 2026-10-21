"""Density maps, density rules, dummy fill and the optical keep-out.

Densities are checked against analytic patterns and brute-force booleans,
fill against exact lattice counts and exact distances, and the keep-out
against full-vector mode solves of a strip waveguide with a silicon block
beside it.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

pytest.importorskip("gdstk")

import photonix as px  # noqa: E402
from photonix.layout import Cell, extract_netlist, fill  # noqa: E402
from photonix.layout import components as lc  # noqa: E402
from photonix.verification import (  # noqa: E402
    check_density,
    density_map,
    evanescent_keepout,
    run_lvs_lite,
)
from photonix.verification._geom import all_loops, merge, polygon_distance  # noqa: E402

WG = (1, 0)


def rect(c: Cell, x0, y0, x1, y1, layer=WG) -> None:
    c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer)


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# --------------------------------------------------------------------------- #
# density
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,p", [(1.0, 4.0), (0.5, 2.5), (3.0, 5.0)])
def test_stripes_have_density_a_over_p_in_every_window(a, p):
    c = Cell("stripes")
    n = 40
    for k in range(n):
        rect(c, k * p, 0, k * p + a, 200)
    span = n * p
    dm = density_map(c, WG, window=(4 * p, 50.0), step=(2 * p, 25.0), region=square(0, 0, span, 200))
    assert np.all(np.abs(dm.density - a / p) < 1e-12)


def test_checkerboard_and_empty_region():
    c = Cell("cb")
    for i in range(20):
        for j in range(20):
            if (i + j) % 2 == 0:
                rect(c, 5 * i, 5 * j, 5 * i + 5, 5 * j + 5)
    dm = density_map(c, WG, window=20.0, step=10.0, region=square(0, 0, 100, 100))
    assert np.allclose(dm.density, 0.5, atol=1e-12)
    empty = density_map(c, (2, 0), window=20.0, region=square(0, 0, 100, 100))
    assert np.all(empty.density == 0.0)


def test_box_sums_equal_brute_force_booleans_for_random_windows():
    gdstk = pytest.importorskip("gdstk")
    rng = np.random.default_rng(7)
    c = Cell("rand")
    for _ in range(120):
        x, y = rng.uniform(0, 290, 2)
        w, h = rng.uniform(0.3, 12.0, 2)
        th = rng.uniform(0, np.pi)
        pts = np.array([(0, 0), (w, 0), (w, h), (0, h)]) @ np.array([[np.cos(th), np.sin(th)],
                                                                    [-np.sin(th), np.cos(th)]]) + (x, y)
        c.add_polygon(pts)
    c.add_ref(lc.ring(radius=20.0), origin=(150, 150))
    reg = square(0, 0, 300, 300)
    dm = density_map(c, WG, window=(30.0, 40.0), step=(10.0, 20.0), region=reg)
    merged = gdstk.boolean([gdstk.Polygon(p) for p, _l in c.get_polygons()], [], "or", precision=1e-9)
    for _ in range(50):
        iy, ix = rng.integers(dm.density.shape[0]), rng.integers(dm.density.shape[1])
        win = dm.window_polygon(iy, ix)
        clip = gdstk.boolean(merged, gdstk.boolean(gdstk.Polygon(win), gdstk.Polygon(np.array(reg, float)), "and"),
                             "and", precision=1e-9)
        win_area = 30.0 * 40.0 - max(0.0, (dm.x[ix] + 15 - 300)) * 40.0 - max(0.0, dm.y[iy] + 20 - 300) * 30.0 \
            + max(0.0, dm.x[ix] + 15 - 300) * max(0.0, dm.y[iy] + 20 - 300)
        brute = sum(p.area() for p in clip) / win_area
        assert dm.density[iy, ix] == pytest.approx(brute, abs=1e-9)


def test_windows_leaving_the_region_are_normalised_by_their_inside_area():
    c = Cell("c")
    rect(c, 0, 0, 25, 25)
    dm = density_map(c, WG, window=20.0, region=square(0, 0, 25, 25))   # windows overhang by 15 um
    assert dm.density.shape == (2, 2)
    assert np.allclose(dm.density, 1.0)


def test_check_density_reports_windows():
    c = Cell("c")
    rect(c, 0, 0, 100, 10)
    v = check_density(c, WG, min_density=0.25, max_density=0.8, window=50.0, region=square(0, 0, 100, 100))
    assert len(v) == 4 and {x.rule for x in v} == {"min_density"}
    v = check_density(c, WG, max_density=0.15, window=50.0, region=square(0, 0, 100, 100))
    assert [x.rule for x in v] == ["max_density", "max_density"]
    with pytest.raises(ValueError):
        check_density(c, WG, min_density=0.9, max_density=0.1)


# --------------------------------------------------------------------------- #
# fill
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("L,t,s", [(100.0, 2.0, 2.0), (50.0, 1.5, 0.7), (37.0, 3.0, 1.0)])
def test_fill_of_an_empty_region_has_the_exact_lattice_count(L, t, s):
    res = fill(Cell("empty"), layer=WG, region=square(0, 0, L, L), tile_size=t, tile_space=s)
    assert res.n_tiles == (int(np.floor((L - t) / (t + s))) + 1) ** 2
    x = res.sites[:, 0]
    assert x.min() == pytest.approx(0.0) and x.max() + t <= L + 1e-9


def _photonic_chip() -> Cell:
    chip = Cell("chip")
    a = chip.add_ref(lc.straight(30.0), origin=(5.0, 10.0), name="a")
    b = chip.add_ref(lc.bend_circular(10.0), name="b").connect("o1", a.port("o2"))
    chip.add_ref(lc.ring_single(radius=8.0), origin=(40.0, 60.0), name="r")
    chip.add_ref(lc.straight(20.0), name="c").connect("o1", b.port("o2"))
    return chip


def _tile_loops(res):
    t = res.tile_size
    return [np.array(square(x, y, x + t, y + t), float) for x, y in res.sites]


@pytest.mark.parametrize("stagger", [False, True])
def test_fill_respects_keepout_exactly(stagger):
    chip = _photonic_chip()
    keep = 1.3
    res = fill(chip, layer=WG, region=square(0, 0, 90, 90), tile_size=1.0, tile_space=1.0,
               keepout={WG: keep}, stagger=stagger)
    assert res.n_tiles > 500
    wg_loops = all_loops(merge([p for p, _l in chip.get_polygons()]))
    tiles = _tile_loops(res)
    dist = polygon_distance(tiles, wg_loops)
    assert dist >= keep - 1e-9
    assert dist < keep + 1.0 + 1e-6          # and the fill does come close (not just conservative)
    # nothing overlaps: the filled layout is free of optical shorts and simulates the same circuit
    top = Cell("filled")
    top.add_ref(chip, name="chip")
    top.add_ref(res.cell, name="fill")
    assert run_lvs_lite(top).clean
    assert extract_netlist(top).instances == {f"chip.{k}": v for k, v in extract_netlist(chip).instances.items()}


def test_fill_avoids_every_listed_layer_and_its_own_layer():
    c = Cell("m")
    rect(c, 10, 10, 30, 30, (11, 0))        # metal
    rect(c, 50, 50, 60, 60, WG)
    res = fill(c, layer=WG, region=square(0, 0, 80, 80), keepout={(11, 0): 3.0})
    tiles = _tile_loops(res)
    assert polygon_distance(tiles, [np.array(square(10, 10, 30, 30), float)]) >= 3.0 - 1e-9
    assert polygon_distance(tiles, [np.array(square(50, 50, 60, 60), float)]) >= 2.0 - 1e-9   # tile_space


def test_fill_to_target_density():
    chip = _photonic_chip()
    reg = square(0, 0, 100, 100)
    res = fill(chip, layer=WG, region=reg, tile_size=1.0, tile_space=1.0, keepout={WG: 1.0},
               target_density=0.2, window=50.0)
    assert res.density_before.max < 0.2
    assert res.density_after.min >= 0.2 - 1e-12 and res.residual == []
    assert res.density_after.max < 0.25 + 0.05        # filled to the target, not to the maximum
    full = fill(chip, layer=WG, region=reg, tile_size=1.0, tile_space=1.0, keepout={WG: 1.0})
    assert full.n_tiles > res.n_tiles


def test_unreachable_target_is_reported_as_residual():
    c = Cell("c")
    rect(c, 0, 0, 1, 1)
    res = fill(c, layer=WG, region=square(0, 0, 50, 50), tile_size=1.0, tile_space=3.0,
               target_density=0.5, window=50.0)                   # max lattice density 1/16
    assert len(res.residual) == 1
    assert res.density_after.max < 0.1


def test_fill_validation():
    with pytest.raises(ValueError):
        fill(Cell("e"), layer=WG, region=square(0, 0, 10, 10), tile_size=0.0)
    with pytest.raises(ValueError):
        fill(Cell("e"), layer=WG, region=square(0, 0, 10, 10), keepout={WG: -1.0})
    with pytest.raises(ValueError):
        fill(Cell("e"), layer=WG, region=square(0, 0, 10, 10), target_density=1.5)


# --------------------------------------------------------------------------- #
# optical keep-out
# --------------------------------------------------------------------------- #
def test_keepout_formula():
    neff, nc, wl, target = 2.44, 1.444, 1.55, 1e-4
    gamma = 2 * np.pi / wl * np.sqrt(neff ** 2 - nc ** 2)
    d = float(evanescent_keepout(neff=neff, n_clad=nc, wl=wl, target=target))
    assert d == pytest.approx(np.log(1 / target) / (2 * gamma), rel=1e-14)
    assert np.exp(-2 * gamma * d) == pytest.approx(target, rel=1e-12)
    with pytest.raises(ValueError):
        evanescent_keepout(neff=1.4, n_clad=1.444)
    with pytest.raises(ValueError):
        evanescent_keepout(neff=2.4, n_clad=1.444, target=2.0)


@requires_jax
def test_keepout_gradient_matches_finite_differences():
    for wrt in ("neff", "n_clad", "wl"):
        base = {"neff": 2.44, "n_clad": 1.444, "wl": 1.55}

        def f(v, wrt=wrt, base=base):
            return evanescent_keepout(**{**base, wrt: v}, target=1e-4)

        g = float(px.grad(f)(base[wrt]))
        assert g == pytest.approx(float(finite_diff(f, base[wrt], h=1e-5)), rel=1e-6)


def test_fill_block_shift_decays_at_twice_the_evanescent_rate():
    """Full-vector FDE: a 0.2 um Si line at distance d beside a 500 x 220 nm strip.

    In the homogeneous cladding the field decays as K0(gamma r), so the
    first-order n_eff shift scales as exp(-2 gamma d) / d; the fitted slope of
    ln(dn d) over d must be -2 gamma within 5 %.
    """
    from photonix.xsection import XSection, n_eff, uniform_grid
    from photonix.xsection import rect as xrect
    from photonix.xsection.library import N_SI, N_SIO2

    grid = uniform_grid((-3.0, 3.0, 0.0, 0.22), 20, 1.0)     # shared, faces on grid lines
    core = xrect(-0.25, 0.25, 0.0, 0.22, N_SI, tags={"core"})
    n0 = n_eff(XSection([core], background=N_SIO2), grid=grid)
    gamma = 2 * np.pi / 1.55 * np.sqrt(n0 ** 2 - N_SIO2 ** 2)
    ds = np.array([0.4, 0.6, 0.8, 1.0])
    dn = np.array([n_eff(XSection([core, xrect(0.25 + d, 0.45 + d, 0.0, 0.22, N_SI)], background=N_SIO2),
                         grid=grid) - n0 for d in ds])
    assert np.all(dn > 0) and np.all(np.diff(dn) < 0)
    slope = np.polyfit(ds, np.log(dn * ds), 1)[0]
    assert slope == pytest.approx(-2 * gamma, rel=0.05)
    keep = float(evanescent_keepout(neff=n0, n_clad=N_SIO2, target=1e-4))
    assert 0.55 < keep < 0.62
    # at the keep-out distance the shift of this line is below 1e-5
    at_keep = n_eff(XSection([core, xrect(0.25 + 0.6, 0.45 + 0.6, 0.0, 0.22, N_SI)], background=N_SIO2),
                    grid=grid) - n0
    assert 0 < at_keep < 1e-5
