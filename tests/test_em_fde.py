"""Tests for the FDE mode solver: analytic-anchored accuracy + differentiability."""
from __future__ import annotations

import numpy as np
from conftest import requires_jax
from scipy.optimize import brentq

import photonix as px
import photonix.em as em


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
    ne = em.solve_modes(wl=1.55, width=0.5, thickness=0.22, resolution=35).neff0
    assert 1.444 < ne < 3.4757


def test_group_index_exceeds_neff():
    kw = dict(width=0.5, thickness=0.22, resolution=25)
    assert em.group_index(wl=1.55, **kw) > em.n_eff(wl=1.55, **kw)


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
