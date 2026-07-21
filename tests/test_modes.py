"""Tests for the FDFD mode solver (analytic validation + convergence)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import photonix.modes as m


def _slab_neff(t, n1, n2, wl):
    """Analytic scalar symmetric-slab fundamental effective index."""
    k0 = 2 * np.pi / wl

    def f(ne):
        u = (t / 2) * np.sqrt(max(k0**2 * (n1**2 - ne**2), 0.0))
        w = (t / 2) * np.sqrt(max(k0**2 * (ne**2 - n2**2), 0.0))
        return u * np.tan(u) - w

    return brentq(f, n2 + 1e-6, n1 - 1e-6, xtol=1e-10)


def test_matches_analytic_slab():
    wl, t, n1, n2 = 1.55, 0.22, 3.4757, 1.444
    ana = _slab_neff(t, n1, n2, wl)
    num = m.n_eff(wl=wl, width=8.0, thickness=t, n_core=n1, n_clad=n2, resolution=60)
    assert abs(num - ana) < 1e-2


def test_mode_is_guided():
    ne = m.n_eff(wl=1.55, width=0.5, thickness=0.22, resolution=35)
    assert 1.444 < ne < 3.4757


def test_normal_dispersion():
    kw = dict(width=0.5, thickness=0.22, resolution=30)
    assert m.n_eff(wl=1.50, **kw) > m.n_eff(wl=1.60, **kw)


def test_group_index_exceeds_neff():
    kw = dict(width=0.5, thickness=0.22, resolution=30)
    assert m.group_index(wl=1.55, **kw) > m.n_eff(wl=1.55, **kw)


def test_grid_convergence():
    wl, t, n1, n2 = 1.55, 0.22, 3.4757, 1.444
    ana = _slab_neff(t, n1, n2, wl)
    e_coarse = abs(m.n_eff(wl=wl, width=8.0, thickness=t, n_core=n1, n_clad=n2, resolution=25) - ana)
    e_fine = abs(m.n_eff(wl=wl, width=8.0, thickness=t, n_core=n1, n_clad=n2, resolution=55) - ana)
    assert e_fine < e_coarse


def test_materials_literature_values():
    assert abs(float(m.silicon(1.55)) - 3.4757) < 3e-3
    assert abs(float(m.silica(1.55)) - 1.444) < 2e-3
    assert abs(float(m.silicon_nitride(1.55)) - 1.996) < 3e-3
