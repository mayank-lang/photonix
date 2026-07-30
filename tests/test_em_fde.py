"""Tests for the FDE mode solver: analytic-anchored accuracy + differentiability."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_jax
from scipy.optimize import brentq

import photonix as px
import photonix.em as em
from photonix.em import fde


def _slab_analytic(t, n1, n2, wl):
    k0 = 2 * np.pi / wl

    def f(ne):
        u = (t / 2) * np.sqrt(max(k0**2 * (n1**2 - ne**2), 0.0))
        w = (t / 2) * np.sqrt(max(k0**2 * (ne**2 - n2**2), 0.0))
        return u * np.tan(u) - w

    return brentq(f, n2 + 1e-9, n1 - 1e-9, xtol=1e-13)


def test_slab_anchor_under_0p1pct():
    wl, t, n1, n2 = 1.55, 0.22, 3.4757, 1.444
    ana = _slab_analytic(t, n1, n2, wl)
    ne = em.slab_neff(thickness=t, n_core=n1, n_clad=n2, wl=wl, resolution=40, richardson=True)
    assert abs(ne - ana) / ana < 1e-3  # < 0.1%


def test_slab_convergence():
    wl, t, n1, n2 = 1.55, 0.22, 3.4757, 1.444
    ana = _slab_analytic(t, n1, n2, wl)
    # plain (non-Richardson) errors show the clean O(h^2) decrease; Richardson
    # values are already so small that their residual is no longer monotone.
    e_coarse = abs(em.slab_neff(thickness=t, wl=wl, resolution=20, richardson=False) - ana)
    e_fine = abs(em.slab_neff(thickness=t, wl=wl, resolution=40, richardson=False) - ana)
    assert e_fine < e_coarse


def test_strip_guided():
    result = em.solve_modes(wl=1.55, width=0.5, thickness=0.22, resolution=35)
    assert 1.444 < result.neff0 < 3.4757
    assert result.n_guided == 1
    assert result.guided.tolist() == [True]


def test_custom_grid_guided_cutoff_uses_highest_exterior_index(monkeypatch):
    """A substrate index, not the grid-wide minimum, defines the light line."""
    import photonix.em.fde as fde

    eps = np.ones((4, 4))
    eps[-1] = 1.5**2

    def fake_solve(*_args):
        return np.array([1.6, 1.4]), np.zeros((2, 4, 4)), np.zeros(16)

    monkeypatch.setattr(fde, "_solve_eps", fake_solve)
    result = fde.solve_modes(eps=eps, num_modes=2)
    assert result.guided.tolist() == [True, False]
    assert result.n_guided == 1


@pytest.mark.parametrize(
    "grid, match",
    [
        (([0.0, 0.1, 0.3, 0.4], [0.0, 0.2, 0.4]), "uniformly spaced"),
        (([0.0, 0.1, 0.2], [0.0, 0.2, 0.4]), "length 4"),
        (([0.0, 0.1, 0.2, 0.3], [0.0, 0.2, 0.1]), "strictly increasing"),
    ],
)
def test_custom_grid_validation(grid, match):
    eps = np.ones((3, 4))
    with pytest.raises(ValueError, match=match):
        em.solve_modes(eps=eps, grid=grid)


def test_mode_solver_rejects_nonpositive_mode_count():
    import pytest

    with pytest.raises(ValueError, match="num_modes"):
        em.solve_modes(num_modes=0)


def test_group_index_exceeds_neff():
    kw = dict(width=0.5, thickness=0.22, resolution=25)
    assert em.group_index(wl=1.55, **kw) > em.n_eff(wl=1.55, **kw)


@pytest.mark.parametrize("wl, dwl", [(1.55, 0.0), (1.55, -0.1), (1.55, 1.55), (0.0, 0.1)])
def test_group_index_rejects_invalid_difference_interval(wl, dwl):
    with pytest.raises(ValueError):
        em.group_index(wl=wl, dwl=dwl, resolution=10)


@requires_jax
def test_differentiable_neff_matches_fd():
    import jax

    from photonix.em.geometry import rectangular_waveguide

    cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=20)
    eps0 = px.xp.asarray(cs.eps.reshape(-1))
    shape, k0 = cs.eps.shape, 2 * np.pi / 1.55
    f = lambda e: em.n_eff_eps(e, shape, cs.dy, cs.dx, k0)  # noqa: E731
    g = np.asarray(jax.grad(f)(eps0))
    kmax = int(np.argmax(np.abs(g)))
    h = 1e-6
    fd = (float(f(eps0.at[kmax].add(h))) - float(f(eps0.at[kmax].add(-h)))) / (2 * h)
    assert abs(g[kmax] - fd) / abs(fd) < 1e-4


def test_slab_te_and_tm_under_0p1pct():
    """Polarization-resolved slab solver matches analytic TE and TM to <0.1%."""
    import photonix.em as em

    for pol in ("te", "tm"):
        num = em.slab.slab_neff(thickness=0.22, wl=1.55, resolution=40, polarization=pol)
        ana = em.slab.slab_neff_analytic(thickness=0.22, wl=1.55, polarization=pol)
        assert abs(num - ana) / ana < 1e-3, (pol, num, ana)


def test_slab_resolution_is_exact_points_per_micrometre():
    """Adjacent resolutions must not collapse to the same core-aligned mesh."""
    n40 = em.slab.slab_neff(resolution=40, richardson=False)
    n41 = em.slab.slab_neff(resolution=41, richardson=False)
    assert n40 != n41


def test_scalar_solver_preserves_evanescent_propagation_constants(monkeypatch):
    """Negative beta^2 roots remain distinct decaying modes, not zeros."""
    import scipy.sparse as sp

    values = np.array([4.0, -1.0, -9.0])

    def fake_eigsh(_operator, **_kwargs):
        return values, np.eye(3)

    monkeypatch.setattr(fde, "helmholtz_operator", lambda *_args: sp.eye(3))
    monkeypatch.setattr(fde.spla, "eigsh", fake_eigsh)
    neff, _fields, _v0 = fde._solve_eps(np.ones((1, 3)), 1.0, 1.0, 2.0, 3)
    assert neff[0] == pytest.approx(1.0)
    assert neff[1] == pytest.approx(-0.5j)
    assert neff[2] == pytest.approx(-1.5j)


def test_group_index_can_include_material_dispersion(monkeypatch):
    monkeypatch.setattr(fde, "n_eff", lambda *, n_core, **_kwargs: n_core)
    material = lambda wavelength: 3.5 - 0.1 * (wavelength - 1.55)  # noqa: E731
    ng = fde.group_index(wl=1.55, dwl=1e-3, core_material=material)
    assert ng == pytest.approx(3.5 + 1.55 * 0.1)


def test_slab_analytic_returns_fundamental_for_multimode_slab():
    """Thick (multimode) slab: the analytic root must stay on the fundamental
    branch instead of jumping across a tan() pole (it used to return ~1.455
    for a 0.5 um Si TE slab whose true fundamental is ~3.27)."""
    import photonix.em as em

    for pol in ("te", "tm"):
        ana = em.slab.slab_neff_analytic(thickness=0.5, wl=1.55, polarization=pol)
        num = em.slab.slab_neff(thickness=0.5, wl=1.55, resolution=40, polarization=pol)
        assert abs(ana - num) / num < 1e-3, (pol, ana, num)
        assert ana > 3.0  # fundamental of a thick Si slab, not a higher branch


def test_slab_analytic_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="polarization"):
        em.slab.slab_neff_analytic(polarization="x")
    with pytest.raises(ValueError, match="n_core"):
        em.slab.slab_neff_analytic(n_core=1.4, n_clad=1.5)


def test_te_more_confined_than_tm():
    import photonix.em as em

    te = em.slab.slab_neff(polarization="te")
    tm = em.slab.slab_neff(polarization="tm")
    assert te > tm > 1.444


def test_eim_reduces_to_slab_at_wide_width():
    """EIM must reproduce the vertical slab in the wide-width limit (<0.1%)."""
    import photonix.em as em

    nv = em.slab.slab_neff(thickness=0.22, wl=1.55, polarization="te")
    wide = em.eim.neff(width=8.0, thickness=0.22, polarization="te")
    assert abs(wide - nv) / nv < 1e-3


def test_eim_strip_in_literature_range():
    """500x220 Si strip: EIM TE0/TM0 in the expected physical range, TE>TM."""
    import photonix.em as em

    te = em.eim.neff(width=0.5, thickness=0.22, polarization="te")
    tm = em.eim.neff(width=0.5, thickness=0.22, polarization="tm")
    assert 2.3 < te < 2.6 and 1.7 < tm < 2.0
    assert te > tm
