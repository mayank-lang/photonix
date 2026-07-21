"""Tests for the FDFD solver: PML, exact adjoint gradient, topology optimization."""
from __future__ import annotations

import numpy as np

from photonix.em.fdfd import FDFD, focus_objective, optimize_focus, point_source

WL, NCLAD, NCORE = 1.55, 1.444, 3.4757


def test_pml_absorbs():
    ny = nx = 70
    eps = np.full((ny, nx), NCLAD**2)
    sim = FDFD(eps, 0.05, 0.05, WL, npml=12).factor()
    e = sim.solve(point_source(ny, nx, ny // 2, nx // 2))
    center = abs(e[ny // 2, nx // 2])
    edge = np.mean(np.abs(e[1, :]))
    assert center / edge > 50  # field absorbed before the boundary


def test_adjoint_matches_finite_difference():
    ny = nx = 60
    dx = dy = 0.05
    eps = np.full((ny, nx), NCLAD**2)
    b = point_source(ny, nx, ny // 2, 8)
    target = (ny // 2, nx - 10)
    f0, grad, _ = focus_objective(eps, dx=dx, dy=dy, wl=WL, source=b, target=target)
    ky, kx = ny // 2, nx // 2
    h = 1e-3
    ep = eps.copy()
    ep[ky, kx] += h
    em = eps.copy()
    em[ky, kx] -= h
    fp = focus_objective(ep, dx=dx, dy=dy, wl=WL, source=b, target=target)[0]
    fm = focus_objective(em, dx=dx, dy=dy, wl=WL, source=b, target=target)[0]
    fd = (fp - fm) / (2 * h)
    assert abs(grad[ky, kx] - fd) / abs(fd) < 1e-4


def test_topology_optimization_improves_objective():
    ny = nx = 60
    dx = dy = 0.05
    eps0 = np.full((ny, nx), NCLAD**2)
    src = point_source(ny, nx, ny // 2, 8)
    target = (ny // 2, nx - 10)
    mask = np.zeros((ny, nx), bool)
    mask[ny // 2 - 12:ny // 2 + 12, 18:nx - 14] = True
    _eps, hist, _E = optimize_focus(eps0, mask, dx=dx, dy=dy, wl=WL, source=src, target=target,
                                    eps_lo=NCLAD**2, eps_hi=NCORE**2, steps=12)
    assert hist[-1] > 2.0 * hist[0]  # adjoint optimization improves the objective


def test_waveguide_sparams_straight():
    """Straight waveguide: near-unity transmission, negligible reflection."""
    from photonix.em.fdfd import waveguide_sparams

    ny, nx = 80, 150
    dx = dy = 0.04
    eps = np.full((ny, nx), 1.444**2)
    eps[ny // 2 - 6:ny // 2 + 6, :] = 2.85**2
    s = waveguide_sparams(eps, dx=dx, dy=dy, wl=1.55, src_col=15, in_mon_col=30, out_mon_col=nx - 30)
    t = abs(s[("o1", "o2")]) ** 2
    r = abs(s[("o1", "o1")]) ** 2
    assert 0.95 < t <= 1.01
    assert r < 1e-2
    assert abs(t + r - 1.0) < 0.05
    # S22 now computed (second RHS on the same factorization); symmetric guide
    r22 = abs(s[("o2", "o2")]) ** 2
    assert r22 < 1e-2


def test_waveguide_sparams_rejects_complex_eps():
    """Complex eps must raise, not silently drop absorption."""
    import pytest

    from photonix.em.fdfd import waveguide_sparams

    eps = np.full((40, 60), 2.0 + 0.1j)
    with pytest.raises(ValueError, match="imaginary"):
        waveguide_sparams(eps, dx=0.05, dy=0.05, wl=1.55, src_col=8,
                          in_mon_col=16, out_mon_col=44)
