"""Validation of photonix.xsection: exact rasterization, smoothing, adapters and modes.

Evidence types used here (see docs/xsection.md):

* exact identities -- polygon areas (shoelace), cell areas from an independent
  Sutherland-Hodgman clipper, painter's order vs boolean differences, and the
  existing exact rectangle rasterizers (``em.rectangular_waveguide``,
  ``LayerStack.cross_section``);
* exact symmetries of Maxwell's equations -- rotation invariance of n_eff and
  x <-> y transposition (TE <-> TM with swapped anisotropy);
* grid-convergence studies (Richardson / GCI from ``em.estimate_convergence``);
* an independent prior calculation -- ``SOI_STRIP_SPLITTING_FIT`` of the
  coupler supermode splitting (80 px/um legacy full-vector solves);
* JAX gradients against central finite differences.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import requires_jax

import photonix as px
import photonix.em as em
import photonix.xsection as xsec
from photonix.em.convergence import estimate_convergence
from photonix.xsection import (
    TensorGrid,
    XSection,
    band,
    coupler,
    disk,
    from_layer_stack,
    graded_grid,
    heater_over_waveguide,
    n_eff,
    polygon,
    rect,
    resolve_index,
    resolve_property,
    rib_waveguide,
    solve_modes,
    strip,
    trapezoid,
    uniform_grid,
)

N_SI, N_OX = 3.4757, 1.444


# --------------------------------------------------------------------------- #
# Independent reference: Sutherland-Hodgman clipping of a polygon to a convex window
# --------------------------------------------------------------------------- #
def _clip(poly, inside, intersect):
    out = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        pin, qin = inside(p), inside(q)
        if pin and qin:
            out.append(q)
        elif pin and not qin:
            out.append(intersect(p, q))
        elif not pin and qin:
            out.append(intersect(p, q))
            out.append(q)
    return out


def _clip_convex(subject, clip_poly):
    """Sutherland-Hodgman: ``subject`` (any simple polygon) clipped by convex CCW ``clip_poly``."""
    out = [tuple(p) for p in subject]
    m = len(clip_poly)
    for k in range(m):
        a, b = np.asarray(clip_poly[k], float), np.asarray(clip_poly[(k + 1) % m], float)
        if not out:
            break

        def inside(p, a=a, b=b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

        def intersect(p, q, a=a, b=b):
            p, q = np.asarray(p, float), np.asarray(q, float)
            d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            d2 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
            t = d1 / (d1 - d2)
            return tuple(p + t * (q - p))

        out = _clip(out, inside, intersect)
    return out


def _area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _cell_areas_reference(poly, grid):
    out = np.zeros(grid.shape)
    X, Y = grid.x_edges, grid.y_edges
    for j in range(grid.ny):
        for i in range(grid.nx):
            box = [(X[i], Y[j]), (X[i + 1], Y[j]), (X[i + 1], Y[j + 1]), (X[i], Y[j + 1])]
            out[j, i] = _area(_clip_convex(poly, box))
    return out


def _star(n=9, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(0.25, 1.0, n)
    return np.stack([rad * np.cos(ang) + 0.07, rad * np.sin(ang) - 0.03], axis=1)


# --------------------------------------------------------------------------- #
# Exact area rasterization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_area_conservation_random_nonconvex_polygons(seed):
    """sum(fraction * cell area) equals the shoelace area to rounding; fractions sum to one."""
    reg = polygon(_star(11, seed), 2.0)
    grid = uniform_grid((-1, 1, -1, 1), resolution=13.7, margin=0.3)
    cov = XSection([reg], background=1.0).rasterize(grid)
    assert float((cov.fractions[0] * grid.areas).sum()) == pytest.approx(reg.area, abs=1e-13)
    assert np.allclose(cov.fractions.sum(axis=0), 1.0, atol=1e-14, rtol=0)
    assert cov.fractions.min() > -1e-15 and cov.fractions.max() < 1 + 1e-15


def test_area_conservation_rotated_rectangle_sliver_and_disk():
    grid = uniform_grid((-1, 1, -1, 1), resolution=10, margin=0.2)
    shapes = [
        rect(-0.6, 0.5, -0.07, 0.11, 2.0).rotated(23.0, (0.05, 0.02)),
        rect(0.0101, 0.0104, 0.0203, 0.0207, 2.0),                        # sliver << one cell
        disk(0.03, -0.02, 0.7, 2.0, n_vertices=200),
    ]
    for reg in shapes:
        cov = XSection([reg], background=1.0).rasterize(grid)
        assert float((cov.fractions[0] * grid.areas).sum()) == pytest.approx(reg.area, rel=1e-12, abs=1e-18)


def test_cell_areas_match_independent_clipper_on_graded_grid():
    """Per-cell areas equal Sutherland-Hodgman cell clipping (an independent algorithm)."""
    poly = _star(9, 4)
    grid = graded_grid((-1.1, 1.1, -1.1, 1.1), refine=[(-0.3, 0.4, -0.2, 0.3, 0.07)], h_max=0.3, ratio=1.4)
    cov = XSection([polygon(poly, 2.0)], background=1.0).rasterize(grid)
    ref = _cell_areas_reference(polygon(poly, 2.0).polygon, grid)
    assert np.allclose(cov.fractions[0] * grid.areas, ref, atol=1e-14, rtol=0)


def test_painter_order_equals_boolean_difference():
    """Visible area of a lower region = its area minus its overlap with the region painted over it."""
    lower = polygon(_star(9, 5), 2.0)
    upper = trapezoid(0.1, -0.3, 0.5, 0.6, material=3.0, sidewall_deg=70.0)  # convex
    grid = uniform_grid((-1, 1, -1, 1), resolution=9.3, margin=0.2)
    cov = XSection([lower, upper], background=1.0).rasterize(grid)
    a_low = _cell_areas_reference(lower.polygon, grid)
    a_up = _cell_areas_reference(upper.polygon, grid)
    overlap = _cell_areas_reference(_clip_convex(lower.polygon, upper.polygon), grid)
    assert np.allclose(cov.fractions[1] * grid.areas, a_up, atol=1e-14, rtol=0)
    assert np.allclose(cov.fractions[0] * grid.areas, a_low - overlap, atol=1e-14, rtol=0)
    # Explicit order beats list order.
    cov2 = XSection([lower.translated(0, 0), upper], background=1.0)
    cov2 = XSection([polygon(lower.polygon, 2.0, order=5), upper], background=1.0).rasterize(grid)
    assert np.allclose(cov2.fractions[0] * grid.areas, a_low, atol=1e-14, rtol=0)


def test_face_fractions_match_segment_overlaps():
    reg = rect(-0.33, 0.21, -0.17, 0.26, 2.0)
    grid = uniform_grid(reg.bbox, resolution=10, margin=0.4)
    cov = XSection([reg], background=1.0).rasterize(grid)
    X, Y = grid.x_edges, grid.y_edges

    def overlap(lo, hi, a, b):
        return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None) / (hi - lo)

    on_x = (X > -0.33) & (X < 0.21)
    fy = overlap(Y[:-1], Y[1:], -0.17, 0.26)
    assert np.allclose(cov.face_x[0], fy[:, None] * on_x[None, :], atol=1e-14)
    on_y = (Y > -0.17) & (Y < 0.26)
    fx = overlap(X[:-1], X[1:], -0.33, 0.21)
    assert np.allclose(cov.face_y[0], on_y[:, None] * fx[None, :], atol=1e-14)


# --------------------------------------------------------------------------- #
# Regression against the existing exact rectangle rasterizers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("resolution", [20, 33, 40])
def test_rectangle_reproduces_rectangular_waveguide(resolution):
    cs = em.rectangular_waveguide(width=0.5, thickness=0.22, resolution=resolution)
    xs = XSection([rect(-0.25, 0.25, -0.11, 0.11, N_SI)], background=N_OX)
    grid = uniform_grid(xs.bbox(), resolution, 1.5)
    assert np.allclose(grid.xc, cs.x, atol=1e-15) and np.allclose(grid.yc, cs.y, atol=1e-15)
    assert np.max(np.abs(xs.eps(grid) - cs.eps)) < 1e-13
    # ... and the library strip (core bottom at y = 0) is the same raster on its own grid.
    st = strip(0.5, 0.22).to_em(resolution=resolution)
    assert np.max(np.abs(st.eps - cs.eps)) < 1e-13


def test_trapezoid_at_90_degrees_is_the_rectangle():
    t = trapezoid(0.1, 0.0, 0.22, 0.5, material=N_SI)
    r = rect(-0.15, 0.35, 0.0, 0.22, N_SI)
    assert np.array_equal(np.sort(t.polygon, axis=0), np.sort(r.polygon, axis=0))
    g = uniform_grid(r.bbox, 25, 0.5)
    assert np.array_equal(XSection([t], background=N_OX).eps(g), XSection([r], background=N_OX).eps(g))


@pytest.mark.parametrize("slab_width", [2.0, 3.3])
def test_rib_from_layer_stack_matches_layer_stack_cross_section(slab_width):
    pdk = px.pdk.demo_pdk()
    stack = pdk.layer_stack
    wg, slab = (1, 0), (2, 0)
    ref = stack.cross_section({wg: 0.5, slab: slab_width}, wl=1.55, resolution=30, margin=1.0)
    xs = from_layer_stack(stack, [(wg, 0.0, 0.5, 90.0), (slab, 0.0, slab_width)])
    grid = uniform_grid(xs.bbox(), 30, 1.0)
    assert np.allclose(grid.xc, ref.x, atol=1e-14) and np.allclose(grid.yc, ref.y, atol=1e-14)
    # Identical up to rounding of the cell geometry (observed <= 3e-14 relative).
    assert np.max(np.abs(xs.eps(grid, 1.55) - ref.eps) / ref.eps) < 1e-13
    # The mapping form of the LayerStack API is accepted too.
    xs2 = from_layer_stack(stack, {wg: 0.5, slab: slab_width})
    assert np.max(np.abs(xs2.eps(grid, 1.55) - ref.eps) / ref.eps) < 1e-13


def test_from_layer_stack_supports_two_cores_and_validates():
    stack = px.pdk.demo_pdk().layer_stack
    xs = from_layer_stack(stack, [((1, 0), -0.35, 0.5), ((1, 0), 0.35, 0.5, 85.0)])
    cores = [r for r in xs.regions if r.name == "core"]
    assert len(cores) == 2 and xs.bbox()[0] < -0.59 and xs.bbox()[1] > 0.6
    with pytest.raises(KeyError):
        from_layer_stack(stack, {(9, 9): 0.5})
    with pytest.raises(ValueError, match="no region covers"):
        thin = px.pdk.LayerStack([px.pdk.LayerLevel("ox", None, 0.5, -0.25, 1.444),
                                  px.pdk.LayerLevel("core", (1, 0), 0.22, 0.0, 3.4757)])
        xs = from_layer_stack(thin, {(1, 0): 0.5})
        xs.rasterize(uniform_grid(xs.bbox(), 20, 1.0))


# --------------------------------------------------------------------------- #
# Normals and tensor smoothing
# --------------------------------------------------------------------------- #
def test_normals_of_a_slanted_interface_are_exact():
    th = np.deg2rad(27.0)
    n_true = np.array([np.cos(th), np.sin(th)])
    # Half-plane n . r <= 0.013, clipped to a big square.
    big = 5.0
    t = np.array([-n_true[1], n_true[0]])
    c = 0.013 * n_true
    poly = [c - big * t, c + big * t, c + big * t - big * n_true, c - big * t - big * n_true]
    grid = uniform_grid((-0.5, 0.5, -0.5, 0.5), 10, 0.0)
    xs = XSection([polygon(poly, 3.0)], background=1.0)
    cov = xs.rasterize(grid)
    nrm = xs.normals(grid)
    cut = (cov.fractions[0] > 1e-9) & (cov.fractions[0] < 1 - 1e-9)
    assert cut.sum() > 5
    dots = np.abs(nrm[cut] @ n_true)
    assert np.allclose(dots, 1.0, atol=1e-12)


def test_tensor_smoothing_is_exact_laminate_homogenization():
    """Vertical interface: xx is the harmonic mean, yy/zz the arithmetic mean, xy = 0 (Kottke 2008)."""
    grid = TensorGrid([0.0, 1.0], [0.0, 1.0])
    f = 0.3
    xs = XSection([rect(-1.0, f, -1.0, 2.0, 3.0)], background=1.5)
    e = xs.eps(grid, smoothing="tensor", offdiagonal=True)[:, 0, 0]
    harm = 1.0 / (f / 9.0 + (1 - f) / 2.25)
    arith = f * 9.0 + (1 - f) * 2.25
    assert e == pytest.approx([harm, arith, arith, 0.0], abs=1e-12)
    # Anisotropic media: normal component harmonic in eps_xx, tangential arithmetic in eps_yy.
    xa = XSection([rect(-1.0, f, -1.0, 2.0, (2.2, 2.1, 2.0))], background=(1.5, 1.4, 1.3))
    ea = xa.eps(grid, smoothing="tensor", offdiagonal=True)[:, 0, 0]
    assert ea[0] == pytest.approx(1.0 / (f / 2.2**2 + (1 - f) / 1.5**2), abs=1e-12)
    assert ea[1] == pytest.approx(f * 2.1**2 + (1 - f) * 1.4**2, abs=1e-12)
    assert ea[2] == pytest.approx(f * 2.0**2 + (1 - f) * 1.3**2, abs=1e-12)
    assert ea[3] == pytest.approx(0.0, abs=1e-12)


def test_isotropic_tensor_equals_projector_formula_on_slanted_cells():
    """General tau average reduces to P <1/eps>^-1 + (1 - P) <eps> with P = n n^T."""
    reg = trapezoid(0.0, -0.11, 0.22, 0.5, material=N_SI, sidewall_deg=63.0).rotated(11.0)
    xs = XSection([reg], background=N_OX)
    grid = uniform_grid(xs.bbox(), 23, 0.2)
    cov = xs.rasterize(grid)
    e = cov.eps(smoothing="tensor", offdiagonal=True)
    f = cov.fractions[0]
    nrm = xs.normals(grid)
    tau = 1.0 / (f / N_SI**2 + (1 - f) / N_OX**2)
    alpha = f * N_SI**2 + (1 - f) * N_OX**2
    has = np.sum(nrm**2, axis=-1) > 0.5
    nx_, ny_ = nrm[..., 0], nrm[..., 1]
    ref_xx = np.where(has, alpha + nx_**2 * (tau - alpha), alpha)
    ref_yy = np.where(has, alpha + ny_**2 * (tau - alpha), alpha)
    ref_xy = np.where(has, nx_ * ny_ * (tau - alpha), 0.0)
    assert np.allclose(e[0], ref_xx, atol=1e-12) and np.allclose(e[1], ref_yy, atol=1e-12)
    assert np.allclose(e[2], alpha, atol=1e-12) and np.allclose(e[3], ref_xy, atol=1e-12)
    assert np.max(np.abs(ref_xy)) > 0.5                                     # slanted cells present


def test_coefficients_masks_and_properties():
    props = {"core": {"k_th": 148.0, "rho_cp": 1.63e6}, "box": {"k_th": 1.38, "rho_cp": 1.6e6},
             "substrate": {"k_th": 148.0, "rho_cp": 1.63e6}, "clad": {"k_th": 1.38, "rho_cp": 1.6e6},
             "heater": {"k_th": lambda T: 20.0 + 0.01 * (T - 300.0), "rho_cp": 2.5e6}}
    xs = heater_over_waveguide(heater_offset=0.7, metal=2.0 - 4.0j, properties=props)
    grid = uniform_grid(xs.bbox(), 20, 1.0)
    heater = xs.mask(grid, "heater")
    assert float((heater * grid.areas).sum()) == pytest.approx(2.0 * 0.12, rel=1e-12)
    k_ar = xs.coefficient(grid, "k_th", 350.0)
    k_h = xs.coefficient(grid, "k_th", 350.0, average="harmonic")
    k_t = xs.coefficient(grid, "k_th", 350.0, average="tensor")
    assert k_t.shape == (3,) + grid.shape
    assert np.all(k_h <= k_ar + 1e-9)                                       # harmonic <= arithmetic
    assert np.all(k_t[0] <= k_ar + 1e-9) and np.all(k_t[0] >= k_h - 1e-9)   # tensor between the bounds
    j, i = np.unravel_index(np.argmax(heater), heater.shape)
    assert k_ar[j, i] == pytest.approx(20.5)                                # callable of temperature
    with pytest.raises(ValueError, match="temperature"):
        xs.coefficient(grid, "k_th")                                        # heater needs T
    with pytest.raises(KeyError, match="sigma"):
        xs.coefficient(grid, "sigma", 300.0)
    # Regions absent from the window need not define the quantity.
    small = uniform_grid((-0.3, 0.3, 0.0, 0.22), 20, 0.05)
    assert np.allclose(xs.coefficient(small, "rho_cp", 300.0)[5, 6], 1.63e6)


def test_material_resolution_protocols():
    class Mat:
        def index(self, wl, temperature=None):
            return 2.0 + 1e-4 * (0.0 if temperature is None else temperature - 300.0)

    class NoT:
        def index(self, wl):
            return 2.0

    assert float(resolve_index(Mat(), 1.55, 400.0)) == pytest.approx(2.01)
    assert float(resolve_index(em.Material("Si", em.silicon), 1.55)) == pytest.approx(float(em.silicon(1.55)))
    assert resolve_index((2.2, 2.1, em.silica), 1.55).shape == (3,)
    with pytest.raises(TypeError, match="temperature"):
        resolve_index(NoT(), 1.55, 300.0)
    with pytest.raises(TypeError, match="temperature"):
        resolve_index(em.silicon, 1.55, 300.0)
    assert float(resolve_index(3.0, 1.55, 300.0)) == 3.0                    # constants are T independent
    r = rect(0, 1, 0, 1, Mat(), name="m")
    with pytest.raises(KeyError, match="k_th"):
        resolve_property(r, "k_th", 300.0)
    with pytest.raises(TypeError):
        rect(0, 1, 0, 1, "silicon")


def test_photonix_materials_physics_sign_becomes_engineering_loss():
    """photonix.materials returns n + i k (k >= 0 loss); xsection treats Im(n) < 0 as loss.  A lossy
    material object must stay lossy: MaterialSpec.index is called with convention="engineering",
    other photonix.materials objects (LorentzFit, bound methods) are conjugated, and plain numbers /
    callables / foreign objects keep the xsection engineering convention unchanged."""
    from photonix.materials import DrudeTerm, LorentzFit, doped_si, get_material

    dsi = doped_si(electrons_cm3=1e19)
    phys = complex(np.asarray(dsi.index(1.55)))
    assert phys.imag > 0                                                   # physics sign: k >= 0
    assert complex(resolve_index(dsi, 1.55)) == pytest.approx(phys.conjugate(), abs=1e-15)
    assert complex(resolve_index(dsi.index, 1.55)) == pytest.approx(phys.conjugate(), abs=1e-15)
    si = get_material("Si")
    assert complex(resolve_index(si, 1.55, 300.0)) == pytest.approx(
        complex(si.index(1.55, 300.0, convention="engineering")), abs=1e-15)
    fit = LorentzFit(eps_inf=1.0, poles=(), drude=DrudeTerm(plasma_frequency=5.0, gamma=0.1),
                     wl_range_um=(1.0, 2.0), max_abs_error=0.0, max_rel_error=0.0)
    n_fit = complex(np.asarray(fit.index(1.55)))
    assert n_fit.imag > 0
    assert complex(resolve_index(fit, 1.55)) == pytest.approx(n_fit.conjugate(), abs=1e-15)
    assert complex(resolve_index(fit.index, 1.55)) == pytest.approx(n_fit.conjugate(), abs=1e-15)
    aniso = resolve_index((dsi, fit, 2.0 - 0.1j), 1.55)
    assert np.all(aniso.imag < 0)
    # Numbers, plain callables and foreign objects: already engineering sign, untouched.
    assert complex(resolve_index(2.0 - 0.1j, 1.55)) == 2.0 - 0.1j
    assert complex(resolve_index(lambda wl: 2.0 - 0.1j, 1.55)) == 2.0 - 0.1j
    assert complex(resolve_index(em.Material("m", lambda wl: 2.0 - 0.1j), 1.55)) == 2.0 - 0.1j

    class Aware:                                       # any convention-aware index gets "engineering"
        def index(self, wl, *, convention="physics"):
            return 2.0 + (0.1j if convention == "physics" else -0.1j)

    assert complex(resolve_index(Aware(), 1.55)) == 2.0 - 0.1j
    # End to end: a doped-Si core gives a decaying mode (Im n_eff < 0).
    m = solve_modes(XSection([rect(-0.25, 0.25, 0.0, 0.22, dsi)], background=N_OX), resolution=15,
                    margin=0.8, smoothing="arithmetic")
    assert m.n_eff[0].imag < 0


def test_region_validation():
    with pytest.raises(ValueError, match="not simple"):
        polygon([(0, 0), (2, 2), (2, 0), (0, 1)], 2.0)                      # crossed edges
    with pytest.raises(ValueError, match="zero area"):
        polygon([(0, 0), (1, 0), (2, 0)], 2.0)
    cw = polygon([(0, 0), (0, 1), (1, 1), (1, 0)], 2.0)
    assert xsec.polygon_area(cw.polygon) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="sidewall"):
        trapezoid(0, 0, 0.5, 0.2, material=2.0, sidewall_deg=30.0, reference="bottom")
    with pytest.raises(ValueError, match="gap"):
        coupler(0.5, 0.05, 0.22, sidewall_deg=60.0)
    with pytest.raises(ValueError):
        band(0.0, 1.0, 1.444).rotated(10.0)


def test_graded_grid_spacing_rules():
    g = graded_grid((-4, 4, -3, 3), refine=[(-0.3, 0.3, -0.11, 0.11, 0.01)], h_max=0.4, ratio=1.25)
    for edges, (a, b) in ((g.x_edges, (-0.3, 0.3)), (g.y_edges, (-0.11, 0.11))):
        h = np.diff(edges)
        assert edges[0] == pytest.approx(-4 if a == -0.3 else -3) and np.all(h > 0)
        assert h.max() <= 0.4 + 1e-12
        mid = 0.5 * (edges[1:] + edges[:-1])
        assert h[(mid > a) & (mid < b)].max() <= 0.01 + 1e-12
        assert np.any(np.isclose(edges, a)) and np.any(np.isclose(edges, b))   # zone faces are grid lines
        ratios = h[1:] / h[:-1]
        assert np.max(np.maximum(ratios, 1 / ratios)) < 1.25 * 1.15


def test_heater_geometry_and_painting():
    xs = heater_over_waveguide(heater_offset=1.0, spacing=0.8, metal=2.0 - 4.0j, top_clad_thickness=0.5)
    grid = uniform_grid(xs.bbox(), 25, 0.8)
    for tag, area in (("core", 0.5 * 0.22), ("heater", 2.0 * 0.12)):
        assert float((xs.mask(grid, tag) * grid.areas).sum()) == pytest.approx(area, rel=1e-12)
    # Every cell is fully accounted for by substrate/box/core/clad(bg)/heater/top.
    assert np.allclose(xs.fractions(grid).sum(axis=0), 1.0, atol=1e-13)
    assert float(xs.mask(grid, "top").max()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# JAX differentiability of the rasterization
# --------------------------------------------------------------------------- #
@requires_jax
@pytest.mark.parametrize("smoothing", ["arithmetic", "tensor"])
def test_rasterized_eps_gradient_matches_finite_difference_cellwise(smoothing):
    """Per-cell d eps / d width equals central differences, including cells where a slanted edge
    ends exactly on a grid line (top corners at x = +-0.25 with 18 px/um)."""
    import jax

    grid = uniform_grid(strip(0.5, 0.22, sidewall_deg=82.0).bbox(), 18, 1.0)
    assert np.any(np.isclose(grid.x_edges, 0.25, atol=1e-12))

    def eps(w):
        e = strip(w, 0.22, sidewall_deg=82.0).eps(grid, smoothing=smoothing, offdiagonal=True)
        return e if smoothing == "tensor" else e[None]

    _, jvp = jax.jvp(eps, (0.5,), (1.0,))
    h = 1e-7
    fd = (np.asarray(eps(0.5 + h)) - np.asarray(eps(0.5 - h))) / (2 * h)
    assert np.max(np.abs(np.asarray(jvp) - fd)) < 1e-5 * np.max(np.abs(fd))


# --------------------------------------------------------------------------- #
# Mode solving
# --------------------------------------------------------------------------- #
def test_arithmetic_path_reproduces_legacy_fullvector_solver():
    xs = strip(0.5, 0.22)
    cs = xs.to_em(resolution=20)
    legacy = em.solve_modes_fullvector(eps=cs.eps, grid=(cs.x, cs.y), num_modes=2)
    mine = solve_modes(xs, resolution=20, smoothing="arithmetic", num_modes=2)
    assert np.allclose(np.real(mine.n_eff), np.real(legacy.n_eff), atol=1e-10, rtol=0)
    assert np.allclose(mine.te_fraction, legacy.te_fraction, atol=1e-8)


def test_sparse_operator_matches_matrix_free_operator():
    from photonix.xsection.modes import _apply, _assemble

    rng = np.random.default_rng(0)
    ny, nx = 7, 9
    comps = tuple(rng.uniform(2, 12, (ny, nx)) for _ in range(3)) + (rng.uniform(-1, 1, (ny, nx)),)
    for c in (comps[:3], comps):
        A = _assemble(c, 0.05, 0.04, 4.0)
        f = rng.standard_normal(2 * nx * ny)
        got = np.asarray(_apply(tuple(px.xp.asarray(e.reshape(-1)) for e in c), px.xp.asarray(f),
                                (ny, nx), 0.05, 0.04, 4.0))
        assert np.allclose(A @ f, got, rtol=1e-12, atol=1e-10)


def _aligned_grid(r, x=1.5, y0=-1.2, y1=1.4):
    """Uniform grid whose lines include y = 0 and y = 0.2 at every resolution in {20, 30, 40, 60}."""
    h = 1.0 / r
    return TensorGrid(np.arange(-round(x * r), round(x * r) + 1) * h, np.arange(round(y0 * r), round(y1 * r) + 1) * h)


def test_rotation_invariance_requires_tensor_smoothing():
    """n_eff of a rotated core is angle independent; only tensor smoothing (with eps_xy) honours that.

    Reference: Richardson limit (p = 2) of the axis-aligned 500 x 200 nm core from the *arithmetic*
    (legacy-equivalent) operator on grids aligned with its faces, where the scheme is second order;
    the design study's three-level limit is 2.36506 (60/80/120 px/um, both smoothings within 1e-5).
    """
    base = XSection([rect(-0.25, 0.25, 0.0, 0.2, N_SI)], background=N_OX)
    n40, n80 = (n_eff(base, grid=_aligned_grid(r), smoothing="arithmetic") for r in (40, 80))
    limit = (4.0 * n80 - n40) / 3.0
    assert limit == pytest.approx(2.36506, abs=2e-4)
    rotated = XSection([rect(-0.25, 0.25, -0.1, 0.1, N_SI).rotated(45.0)], background=N_OX)
    grid = rotated.uniform_grid(40, 1.2)
    err_t = abs(n_eff(rotated, grid=grid, smoothing="tensor") - limit)
    err_d = abs(n_eff(rotated, grid=grid, smoothing="tensor", offdiagonal=False) - limit)
    err_a = abs(n_eff(rotated, grid=grid, smoothing="arithmetic") - limit)
    assert err_t < 3.5e-3
    assert err_t < 0.2 * err_a and err_t < 0.2 * err_d


def test_trapezoid_convergence_order_and_common_limit():
    """80-degree trapezoid on aligned grids: tensor order ~2, arithmetic lower, same continuum limit."""
    xs = strip(0.5, 0.2, sidewall_deg=80.0)
    ress = [20, 30, 40, 60]
    studies = {}
    for sm in ("arithmetic", "tensor"):
        vals = [n_eff(xs, grid=_aligned_grid(r), smoothing=sm) for r in ress]
        studies[sm] = estimate_convergence(ress, np.array(vals))
    t, a = studies["tensor"], studies["arithmetic"]
    assert 1.5 < t.observed_order < 2.8
    assert a.observed_order < t.observed_order
    assert t.grid_convergence_index < a.grid_convergence_index
    gap = abs(float(t.extrapolated) - float(a.extrapolated))
    assert gap < (t.grid_convergence_index + a.grid_convergence_index) * abs(float(t.extrapolated))


def test_lateral_subcell_translation_within_gci():
    """Moving the slanted sidewalls through the cells changes n_eff by less than the GCI."""
    xs = strip(0.5, 0.22, sidewall_deg=80.0)
    ress = [20, 30, 40]
    vals = [n_eff(xs, resolution=r, margin=1.2) for r in ress]
    gci = estimate_convergence(ress, np.array(vals)).grid_convergence_index * abs(vals[-1])
    grid = xs.uniform_grid(40, 1.2)
    shifted = [n_eff(xs.translated(s / 40.0, 0.0), grid=grid) for s in (0.25, 0.5)]
    assert np.max(np.abs(np.array(shifted) - vals[-1])) < gci


def test_physically_ordered_effective_indices():
    """More silicon -> higher TE0 index: rib > strip; shallower sidewall (fixed top width) > vertical."""
    kw = {"resolution": 30, "margin": 1.2}
    n_strip = n_eff(strip(0.5, 0.22), **kw)
    n_rib = n_eff(rib_waveguide(0.5, 0.22, 0.09, slab_width=3.0), **kw)
    n85 = n_eff(strip(0.5, 0.22, sidewall_deg=85.0), **kw)
    n75 = n_eff(strip(0.5, 0.22, sidewall_deg=75.0), **kw)
    n75_bottom = n_eff(strip(0.5, 0.22, sidewall_deg=75.0, reference="bottom"), **kw)
    assert n_rib > n_strip + 0.02
    assert n75 > n85 > n_strip
    assert n75_bottom < n_strip


def test_transposition_symmetry_with_uniaxial_core():
    """Swapping x <-> y maps TE0 of (n_x, n_y, n_z) onto TM0 of (n_y, n_x, n_z) exactly."""
    mat = (2.21, 2.14, 2.17)                                         # LN-like uniaxial values
    core = rect(-0.6, 0.6, 0.0, 0.4, mat)
    xs = XSection([core], background=N_OX)
    grid = uniform_grid(xs.bbox(), 22, 1.0)
    swapped = XSection([polygon(core.polygon[:, ::-1], (mat[1], mat[0], mat[2]))], background=N_OX)
    gridT = TensorGrid(grid.y_edges, grid.x_edges)
    for sm in ("arithmetic", "tensor"):
        a = solve_modes(xs, grid=grid, smoothing=sm, num_modes=2, offdiagonal=False)
        b = solve_modes(swapped, grid=gridT, smoothing=sm, num_modes=2, offdiagonal=False)
        assert a.te_fraction[0] > 0.5 and b.te_fraction[0] < 0.5
        assert np.allclose(np.sort(np.real(a.n_eff)), np.sort(np.real(b.n_eff)), atol=1e-9, rtol=0)
    # The TE-like mode is dominated by eps_xx: raising n_x moves it more than raising n_y.
    up_x = solve_modes(XSection([rect(-0.6, 0.6, 0.0, 0.4, (2.31, 2.14, 2.17))], background=N_OX), grid=grid)
    up_y = solve_modes(XSection([rect(-0.6, 0.6, 0.0, 0.4, (2.21, 2.24, 2.17))], background=N_OX), grid=grid)
    base = solve_modes(xs, grid=grid)
    assert up_x.neff0 - base.neff0 > 3 * (up_y.neff0 - base.neff0) > 0


def test_lossy_heater_gives_decaying_complex_index():
    near = heater_over_waveguide(heater_width=1.0, spacing=0.25, metal=2.0 - 4.0j)
    far = heater_over_waveguide(heater_width=1.0, spacing=0.6, metal=2.0 - 4.0j)
    m_near = solve_modes(near, resolution=20, margin=0.8, smoothing="arithmetic")
    m_far = solve_modes(far, resolution=20, margin=0.8, smoothing="arithmetic")
    k_near, k_far = -m_near.n_eff[0].imag, -m_far.n_eff[0].imag
    assert k_near > 10 * k_far > 0                                   # Im(n) < 0: loss, decays with spacing
    assert 2.3 < m_near.neff0 < 2.6 and m_near.te_fraction[0] > 0.9


@pytest.mark.parametrize(("gap", "kernel"), [(0.1, "box"), (0.2, "box"), (0.3, "box"), (0.4, "box"),
                                             (0.2, "tent")])
def test_coupler_splitting_reproduces_soi_fit(gap, kernel):
    """Even/odd TE splitting of library.coupler vs SOI_STRIP_SPLITTING_FIT.

    The fit declares a 1.3 % residual against its 80 px/um legacy solves plus a
    +/-1.3 % resolution spread of those solves, so it is trusted to ~2 % of the
    continuum.  Tensor smoothing at 60 px/um lands within 1.31 % at every gap
    (gap 0.2 is the worst; 1.67 % with the tent kernel); the legacy arithmetic
    discretization at the fit's own 80 px/um, solved through this geometry path,
    lands within 1.26 % (design study).
    """
    from photonix.components.couplers import supermode_splitting

    m = solve_modes(coupler(0.5, gap, 0.22), resolution=60, margin=1.0, num_modes=3, kernel=kernel)
    te = np.flatnonzero(m.te_fraction > 0.5)[:2]
    dn = float(np.real(m.n_eff[te[0]] - m.n_eff[te[1]]))
    assert dn == pytest.approx(float(supermode_splitting(gap, 1.55)), rel=0.02)


@requires_jax
@pytest.mark.parametrize("smoothing", ["arithmetic", "tensor"])
def test_n_eff_gradient_through_geometry_matches_finite_difference(smoothing):
    """d n_eff / d width: exact rasterization derivative + eigenvalue adjoint vs central differences.

    At 18 px/um the 0.5 um top face ends exactly on grid lines (x = +-0.25), the degenerate case
    where a naive clip derivative mis-assigns area between neighbouring cells.
    """
    import jax

    def f(w):
        return n_eff(strip(w, 0.22, sidewall_deg=82.0), resolution=18, margin=1.0, smoothing=smoothing)

    g = float(jax.grad(f)(0.5))
    # n_eff(w) is C1 but not C2 where the corners sit on grid lines, so the central difference
    # carries an O(h) error (2.5e-5 relative at h = 1e-4); h = 1e-6 keeps it below 1e-6.
    h = 1e-6
    fd = (f(0.5 + h) - f(0.5 - h)) / (2 * h)
    assert g == pytest.approx(fd, rel=5e-6)
    assert 0.5 < g < 2.0                                             # wider core -> higher index


def test_solver_rejects_graded_grid_and_bad_arguments():
    xs = strip()
    with pytest.raises(ValueError, match="uniform"):
        solve_modes(xs, grid=graded_grid((-2, 2, -2, 2), refine=[(-0.3, 0.3, 0, 0.22, 0.02)]))
    with pytest.raises(ValueError, match="num_modes"):
        solve_modes(xs, num_modes=0)
    with pytest.raises(ValueError, match="smoothing"):
        xs.eps(xs.uniform_grid(10), smoothing="harmonic")


def test_math_constants_used_in_docs():
    assert math.isclose(disk(0, 0, 1.0, 1.0, n_vertices=64).area, math.pi, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
# Tent (bilinear hat) kernel
# --------------------------------------------------------------------------- #
def _poly_moments(poly):
    """Exact (int 1, int x, int y, int xy) dA of a simple polygon by Green's theorem."""
    if len(poly) < 3:
        return np.zeros(4)
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    return np.array([c.sum() / 2, ((x + x1) * c).sum() / 6, ((y + y1) * c).sum() / 6,
                     ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * c).sum() / 24])


def _tent_reference(pieces, grid):
    """Tent-weighted visible fractions from per-dual-cell clipping (independent of the sweep).

    ``pieces`` is a list of (polygon, sign) whose signed sum is the visible region.  The tent of
    cell centre (i, j) is the bilinear nodal function on the lines through the cell centres
    (plus one mirrored ghost line per side).
    """
    xn = np.concatenate([[2 * grid.xc[0] - grid.xc[1]], grid.xc, [2 * grid.xc[-1] - grid.xc[-2]]])
    yn = np.concatenate([[2 * grid.yc[0] - grid.yc[1]], grid.yc, [2 * grid.yc[-1] - grid.yc[-2]]])
    w = np.zeros((grid.ny + 2, grid.nx + 2))
    for b in range(yn.size - 1):
        for a in range(xn.size - 1):
            X0, X1, Y0, Y1 = xn[a], xn[a + 1], yn[b], yn[b + 1]
            box = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]
            m = sum(sign * _poly_moments(_clip_convex(poly, box)) for poly, sign in pieces)
            W, H = X1 - X0, Y1 - Y0
            # Corner functions (ax + bx x)(ay + by y) / (W H) in global coordinates.
            for cx, (ax, bx) in ((a + 1, (-X0, 1.0)), (a, (X1, -1.0))):
                for cy, (ay, by) in ((b + 1, (-Y0, 1.0)), (b, (Y1, -1.0))):
                    w[cy, cx] += (ax * ay * m[0] + bx * ay * m[1] + ax * by * m[2] + bx * by * m[3]) / (W * H)
    mass = (0.5 * (yn[2:] - yn[:-2]))[:, None] * (0.5 * (xn[2:] - xn[:-2]))[None, :]
    return w[1:-1, 1:-1] / mass


def test_tent_fractions_match_independent_clipper_moments_with_painting():
    """Tent fractions = per-dual-cell Sutherland-Hodgman clipping + exact polygon moments, on a
    graded grid, for a non-convex polygon partly painted over by a slanted trapezoid."""
    lower = polygon(_star(9, 6), 2.0)
    upper = trapezoid(0.15, -0.35, 0.55, 0.5, material=3.0, sidewall_deg=67.0)
    grid = graded_grid((-1.1, 1.1, -1.1, 1.1), refine=[(-0.3, 0.4, -0.2, 0.3, 0.09)], h_max=0.3, ratio=1.4)
    cov = XSection([lower, upper], background=1.0).rasterize(grid, kernel="tent")
    assert cov.kernel == "tent" and cov.face_x is None
    overlap = _clip_convex(lower.polygon, upper.polygon)
    ref_up = _tent_reference([(upper.polygon, 1.0)], grid)
    ref_low = _tent_reference([(lower.polygon, 1.0), (overlap, -1.0)], grid)
    assert np.allclose(cov.fractions[1], ref_up, atol=1e-13, rtol=0)
    assert np.allclose(cov.fractions[0], ref_low, atol=1e-13, rtol=0)
    assert np.allclose(cov.fractions.sum(axis=0), 1.0, atol=1e-14, rtol=0)


def test_tent_reproduces_bilinear_moments_but_box_does_not():
    """The defining property: sum_k f_k A_k (1, x, y, xy)_k equals the exact region moments."""
    reg = polygon(_star(11, 3) * 0.8, 2.0)
    grid = uniform_grid((-1, 1, -1, 1), resolution=11.3, margin=0.3)
    exact = _poly_moments(reg.polygon)
    X, Y = np.meshgrid(grid.xc, grid.yc)
    basis = np.stack([np.ones_like(X), X, Y, X * Y])
    got = {}
    for kernel in ("tent", "box"):
        cov = XSection([reg], background=1.0).rasterize(grid, kernel=kernel)
        got[kernel] = (basis * (cov.fractions[0] * cov.sample_areas)).reshape(4, -1).sum(axis=1)
    assert np.allclose(got["tent"], exact, atol=1e-14, rtol=0)
    assert abs(got["box"][0] - exact[0]) < 1e-14                             # box conserves area only
    assert np.max(np.abs(got["box"][1:] - exact[1:])) > 1e-6


def test_tent_normals_and_laminate_are_exact():
    # Vertical interface x = s through the centre sample of a 3 x 3 grid: the tent fraction is the hat CDF.
    grid = TensorGrid([-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5])
    s = 0.3
    xs = XSection([rect(-5.0, s, -5.0, 5.0, 3.0)], background=1.5)
    f = 1.0 - (1.0 - s) ** 2 / 2.0
    e = xs.eps(grid, smoothing="tensor", offdiagonal=True, kernel="tent")[:, 1, 1]
    assert e == pytest.approx([1.0 / (f / 9.0 + (1 - f) / 2.25), f * 9.0 + (1 - f) * 2.25,
                               f * 9.0 + (1 - f) * 2.25, 0.0], abs=1e-12)
    # Slanted half-plane: every sample whose tent sees the interface gets the exact normal.
    th = np.deg2rad(27.0)
    n_true = np.array([np.cos(th), np.sin(th)])
    t = np.array([-n_true[1], n_true[0]])
    c = 0.013 * n_true
    big = 5.0
    poly = [c - big * t, c + big * t, c + big * t - big * n_true, c - big * t - big * n_true]
    g = uniform_grid((-0.5, 0.5, -0.5, 0.5), 10, 0.0)
    hp = XSection([polygon(poly, 3.0)], background=1.0)
    fr = hp.fractions(g, kernel="tent")[0]
    nrm = hp.normals(g, kernel="tent")
    cut = (fr > 1e-9) & (fr < 1 - 1e-9)
    assert cut.sum() > 10
    assert np.allclose(np.abs(nrm[cut] @ n_true), 1.0, atol=1e-12)


def test_tent_kernel_api_and_validation():
    xs = heater_over_waveguide(heater_offset=0.7, metal=2.0 - 4.0j, properties={
        "core": {"k_th": 148.0}, "box": {"k_th": 1.38}, "substrate": {"k_th": 148.0},
        "clad": {"k_th": 1.38}, "heater": {"k_th": 20.0}})
    grid = uniform_grid(xs.bbox(), 20, 1.0)
    cov = xs.rasterize(grid, kernel="tent")
    assert np.allclose(cov.sample_areas, grid.areas, rtol=1e-12)
    for tag, area in (("core", 0.5 * 0.22), ("heater", 2.0 * 0.12)):
        assert float((xs.mask(grid, tag, kernel="tent") * grid.areas).sum()) == pytest.approx(area, rel=1e-12)
    k = xs.coefficient(grid, "k_th", average="tensor", kernel="tent")
    assert k.shape == (3,) + grid.shape and np.all(k > 1.3) and np.all(k < 148.1)
    cs = strip().to_em(resolution=20, kernel="tent")
    assert cs.eps.shape == strip().to_em(resolution=20).eps.shape
    with pytest.raises(ValueError, match="kernel"):
        xs.rasterize(grid, kernel="gauss")


@requires_jax
@pytest.mark.parametrize("resolution", [18, 36])
def test_tent_eps_gradient_matches_finite_difference_cellwise(resolution):
    """d eps / d width per sample, tent kernel.  At 36 px/um the top corners (x = +-0.25) sit exactly
    on tent nodes, where eps(w) is C1 but not C2; a Richardson-corrected central difference removes
    the resulting O(h) finite-difference error."""
    import jax

    grid = uniform_grid(strip(0.5, 0.22, sidewall_deg=82.0).bbox(), resolution, 1.0)
    assert np.any(np.isclose(grid.xc if resolution == 36 else grid.x_edges, 0.25, atol=1e-12))

    def eps(w):
        return strip(w, 0.22, sidewall_deg=82.0).eps(grid, smoothing="tensor", offdiagonal=True, kernel="tent")

    _, jvp = jax.jvp(eps, (0.5,), (1.0,))

    def fd(h):
        return (np.asarray(eps(0.5 + h)) - np.asarray(eps(0.5 - h))) / (2 * h)

    rich = 2 * fd(5e-7) - fd(1e-6)
    assert np.max(np.abs(np.asarray(jvp) - rich)) < 1e-7 * np.max(np.abs(rich))


def test_tent_kernel_translation_invariance_within_gci():
    """Spec check: sub-cell shifts (vertical, lateral, diagonal) of an 80-degree trapezoid change the
    tensor/tent n_eff by much less than the GCI of a 20/30/40 px/um study.  The box kernel moves the
    same core by ~5.6e-3 under vertical shifts -- as much as its GCI -- because its O(h^2) interface
    error depends on where the horizontal faces cut their cells (as for the 1-D TE slab)."""
    xs = strip(0.5, 0.22, sidewall_deg=80.0)
    study = xsec.n_eff_convergence(xs, (20, 30, 40), margin=1.2)
    assert study.asymptotic
    grid = xs.uniform_grid(40, 1.2)
    shifts = [(0.0, 0.25), (0.0, 0.5), (0.0, 0.75), (0.5, 0.0), (0.3, 0.6)]
    spread = {}
    for kernel in ("tent", "box"):
        ref = n_eff(xs, grid=grid, kernel=kernel)
        dev = [n_eff(xs.translated(a / 40.0, b / 40.0), grid=grid, kernel=kernel) - ref for a, b in shifts]
        spread[kernel] = float(np.max(np.abs(dev)))
    assert spread["tent"] < 0.1 * study.estimated_absolute_error
    assert spread["tent"] < 1e-3                                       # observed 4.0e-4
    assert spread["box"] > 5 * spread["tent"]


@pytest.mark.parametrize("angle", [0.0, 45.0])
def test_tent_extrapolation_recovers_aligned_limit_on_unaligned_grids(angle):
    """Richardson limit of tent/tensor n_eff on ordinary (face-unaligned) grids, for the axis-aligned
    and the 45-degree-rotated 500 x 200 nm core, equals the aligned-grid continuum value 2.36506
    (see test_rotation_invariance_requires_tensor_smoothing) within the study's own error bound;
    observed deviations 2.2e-4 and 2.9e-4 against bounds of 3.5e-3 and 4.2e-3."""
    core = rect(-0.25, 0.25, -0.1, 0.1, N_SI).rotated(angle) if angle else rect(-0.25, 0.25, 0.0, 0.2, N_SI)
    study = xsec.n_eff_convergence(XSection([core], background=N_OX), (30, 40, 60), margin=1.2)
    assert study.asymptotic and 1.3 < study.observed_order < 2.7
    err = abs(float(study.extrapolated) - 2.36506)
    assert err < study.estimated_absolute_error and err < 6e-4


@requires_jax
def test_tent_n_eff_gradient_through_geometry_matches_finite_difference():
    import jax

    def f(w):
        return n_eff(strip(w, 0.22, sidewall_deg=82.0), resolution=18, margin=1.0, kernel="tent")

    g = float(jax.grad(f)(0.5))
    h = 1e-6
    fd = (f(0.5 + h) - f(0.5 - h)) / (2 * h)
    assert g == pytest.approx(fd, rel=5e-6)
    assert 0.5 < g < 2.0


@requires_jax
def test_traced_translation_and_rotation_are_differentiable():
    """Region.translated / rotated accept JAX values: d n_eff / d(position) of a dielectric rail
    0.15 um beside a strip matches central differences (moving the rail away lowers the index), and
    the rasterized area of a rotated core has zero derivative with respect to the angle (both
    kernels).  The rail is moved rather than the strip: at 18 px/um the strip's own lattice
    artefact under lateral shifts (~4e-3 per cell) would swamp the physical coupling."""
    import jax

    grid = uniform_grid((-0.8, 1.2, 0.0, 0.22), 18, 1.0)

    def f(dx):
        rail = rect(0.4, 0.8, 0.0, 0.22, 2.0).translated(dx, 0.0)
        return n_eff(XSection([strip(0.5, 0.22).regions[0], rail], background=N_OX), grid=grid, kernel="tent")

    g = float(jax.grad(f)(0.013))
    h = 1e-6
    assert g == pytest.approx((f(0.013 + h) - f(0.013 - h)) / (2 * h), rel=1e-5)
    assert g < 0

    g2 = uniform_grid((-0.6, 0.6, -0.6, 0.6), 17, 0.2)
    for kernel in ("box", "tent"):
        def area(theta, kernel=kernel):
            reg = rect(-0.25, 0.25, -0.11, 0.11, N_SI).rotated(theta, (0.02, -0.01))
            cov = XSection([reg], background=N_OX).rasterize(g2, kernel=kernel)
            return (cov.fractions[0] * cov.sample_areas).sum()

        assert float(area(31.0)) == pytest.approx(0.11, abs=1e-13)
        assert abs(float(jax.grad(area)(31.0))) < 1e-12


def test_n_eff_convergence_validates_resolutions():
    with pytest.raises(ValueError, match="integers"):
        xsec.n_eff_convergence(strip(), (10.5, 20, 30))
    with pytest.raises(ValueError):
        xsec.n_eff_convergence(strip(), (10, 20))
    with pytest.raises(ValueError, match="increasing"):
        xsec.n_eff_convergence(strip(), (20, 10, 30))
