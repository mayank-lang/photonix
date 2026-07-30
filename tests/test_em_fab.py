"""Tests for fabrication constraints (filter + projection) and their gradient."""
from __future__ import annotations

import numpy as np

from photonix.em import fabrication as fab


def test_projection_binarizes():
    rho = np.array([0.2, 0.45, 0.55, 0.8])
    p = fab.tanh_projection(rho, beta=12.0)
    assert p[0] < 0.1 and p[3] > 0.9          # extremes pushed to 0/1
    assert p[1] < rho[1] and p[2] > rho[2]    # below/above threshold sharpened


def test_filter_enforces_smoothness():
    rho = np.zeros((21, 21))
    rho[10, 10] = 1.0                          # single hot pixel
    out = fab.conic_filter(rho, radius_cells=3.0)
    assert out[10, 10] < 0.5                   # spread out -> no single-pixel feature
    assert out[10, 11] > 0                     # neighbors picked up weight


def test_filter_handles_integer_density_and_zero_radius():
    rho = np.zeros((7, 7), dtype=int)
    rho[3, 3] = 1
    out = fab.conic_filter(rho, radius_cells=2.0)
    assert out.dtype.kind == "f"
    assert 0 < out[3, 3] < 1
    assert np.array_equal(fab.conic_filter(rho, radius_cells=0), rho)
    assert np.array_equal(fab.conic_filter_adjoint(rho, radius_cells=0), rho)


def test_zero_beta_projection_is_identity():
    rho = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(fab.tanh_projection(rho, beta=0), rho)
    assert np.array_equal(fab.tanh_projection_deriv(rho, beta=0), np.ones_like(rho))


def test_conic_filter_adjoint_identity():
    """<c, A x> == <A^T c, x> to machine precision, incl. non-integer radii."""
    rng = np.random.default_rng(1)
    x = rng.random((13, 17))
    c = rng.random((13, 17))
    for radius in (2.0, 2.5, 3.7):
        lhs = float(np.sum(c * fab.conic_filter(x, radius)))
        rhs = float(np.sum(fab.conic_filter_adjoint(c, radius) * x))
        assert abs(lhs - rhs) / abs(lhs) < 1e-12


def test_density_to_eps_vjp_matches_fd_at_corner():
    """Boundary pixels: the naive self-adjoint VJP was ~13% wrong at corners."""
    rng = np.random.default_rng(2)
    R = rng.random((12, 12))
    emn, emx = 1.444**2, 3.4757**2
    _eps, cache = fab.density_to_eps(R, eps_min=emn, eps_max=emx, radius_cells=2.5, beta=8.0)
    w = rng.random((12, 12))
    g = fab.density_to_eps_vjp(w, cache)
    h = 1e-6
    for (i, j) in [(0, 0), (0, 5), (11, 11), (6, 6)]:
        Rp = R.copy()
        Rp[i, j] += h
        Rm = R.copy()
        Rm[i, j] -= h
        ep = fab.density_to_eps(Rp, eps_min=emn, eps_max=emx, radius_cells=2.5, beta=8.0)[0]
        em = fab.density_to_eps(Rm, eps_min=emn, eps_max=emx, radius_cells=2.5, beta=8.0)[0]
        fd = (np.sum(w * ep) - np.sum(w * em)) / (2 * h)
        assert abs(g[i, j] - fd) / abs(fd) < 1e-5


def test_density_to_eps_vjp_matches_fd():
    rng = np.random.default_rng(0)
    R = rng.random((16, 16))
    emn, emx = 1.444**2, 3.4757**2
    _eps, cache = fab.density_to_eps(R, eps_min=emn, eps_max=emx, radius_cells=2.0, beta=8.0)
    w = rng.random((16, 16))
    g = fab.density_to_eps_vjp(w, cache)
    i, j, h = 6, 9, 1e-6
    Rp = R.copy()
    Rp[i, j] += h
    Rm = R.copy()
    Rm[i, j] -= h
    ep = fab.density_to_eps(Rp, eps_min=emn, eps_max=emx, radius_cells=2.0, beta=8.0)[0]
    em = fab.density_to_eps(Rm, eps_min=emn, eps_max=emx, radius_cells=2.0, beta=8.0)[0]
    fd = (np.sum(w * ep) - np.sum(w * em)) / (2 * h)
    assert abs(g[i, j] - fd) / abs(fd) < 1e-4
