"""Tests for the EME propagator: lossless/reciprocal/energy-conserving + adiabatic."""
from __future__ import annotations

import numpy as np

from photonix.em.eme import Section, eme_smatrix

WL, N1, N2 = 1.55, 3.4757, 1.444
X = np.linspace(-3, 3, 301)
DX = X[1] - X[0]


def _strip(w):
    return np.where(np.abs(X) < w / 2, N1**2, N2**2)


def _taper(nsteps, ltot=20.0):
    ws = np.linspace(0.5, 1.2, nsteps)
    return eme_smatrix([Section(_strip(w), ltot / nsteps) for w in ws], DX, WL, num_modes=6)


def test_uniform_lossless_no_reflection():
    r = eme_smatrix([Section(_strip(0.5), 7.3)], DX, WL, num_modes=4)
    assert abs(abs(r.Tf[0, 0]) - 1.0) < 1e-9
    assert abs(r.Rf[0, 0]) < 1e-9


def test_cascade_consistency():
    whole = eme_smatrix([Section(_strip(0.5), 7.3)], DX, WL, num_modes=4)
    split = eme_smatrix([Section(_strip(0.5), 3.0), Section(_strip(0.5), 4.3)], DX, WL, num_modes=4)
    assert abs(split.Tf[0, 0] - whole.Tf[0, 0]) < 1e-10


def test_energy_conserved_and_reciprocal():
    r = _taper(20)
    energy = np.sum(np.abs(r.Tf[:, 0]) ** 2) + np.sum(np.abs(r.Rf[:, 0]) ** 2)
    assert abs(energy - 1.0) < 1e-9
    assert np.max(np.abs(r.Tf - r.Tb.T)) < 1e-10


def test_taper_more_adiabatic_higher_transmission():
    t_coarse = abs(_taper(3).Tf[0, 0]) ** 2
    t_fine = abs(_taper(40).Tf[0, 0]) ** 2
    assert t_fine > t_coarse


def test_sdict_export():
    r = eme_smatrix([Section(_strip(0.5), 5.0)], DX, WL, num_modes=4)
    s = r.sdict(n_in=1, n_out=1)
    assert ("o1", "o2") in s and abs(abs(s[("o1", "o2")]) - 1.0) < 1e-9


def test_slab_operator_has_both_dirichlet_faces():
    """The modal Laplacian must not silently use Neumann on the left edge."""
    from photonix.em.eme import _d_faces

    n, h = 7, 0.2
    d = _d_faces(n, h)
    lap = (-d.T @ d).toarray() * h**2
    expected = np.diag(-2.0 * np.ones(n))
    expected += np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    assert d.shape == (n + 1, n)
    assert np.allclose(lap, expected)


def test_absorber_modes_remain_eigenmodes_without_parity_projection():
    """Symmetry comes from the operator, without modifying ARPACK eigenvectors."""
    import scipy.sparse as sp

    from photonix.em.eme import _d_faces, slab_modes, transverse_pml

    eps = _strip(0.5)
    k0 = 2 * np.pi / WL
    d = _d_faces(len(eps), DX)
    d_eps = transverse_pml(len(eps), DX, k0, (0.8, 1.0), eps_edge=float(eps[0]))
    operator = -d.T @ d + sp.diags(k0**2 * (eps + d_eps))
    beta, fields, _ = slab_modes(eps, DX, WL, 6, pml=(0.8, 1.0))

    residuals = []
    parities = []
    for mode_beta, field in zip(beta, fields.T, strict=True):
        lhs = operator @ field
        rhs = mode_beta**2 * field
        residuals.append(np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + np.linalg.norm(rhs)))
        parities.append(
            min(np.linalg.norm(field - field[::-1]), np.linalg.norm(field + field[::-1]))
            / np.linalg.norm(field)
        )
    assert max(residuals) < 1e-8
    assert max(parities) < 1e-8


def test_invalid_eme_configuration_is_rejected():
    import pytest

    from photonix.em.eme import slab_modes, transverse_pml

    with pytest.raises(ValueError, match="sections"):
        eme_smatrix([], DX, WL)
    with pytest.raises(ValueError, match="num_modes"):
        slab_modes(_strip(0.5), DX, WL, 0)
    with pytest.raises(ValueError, match="thickness"):
        transverse_pml(len(X), DX, 2 * np.pi / WL, (-1.0, 1.0))
    with pytest.raises(ValueError, match="length"):
        Section(_strip(0.5), -1.0)


# --------------------------------------------------------------------------- #
# Vectorial TM EME (1/eps-weighted power overlap)
# --------------------------------------------------------------------------- #
def _core(x, w):
    return np.where(np.abs(x) < w / 2, 3.4757**2, 1.444**2)


def test_tm_transparent_interface():
    """Identical TM sections -> transparent interface (T=I, R=0) to machine eps."""
    from photonix.em.eme import _interface, slab_modes

    x = np.linspace(-3, 3, 481)
    dx = x[1] - x[0]
    b, f, w = slab_modes(_core(x, 0.5), dx, 1.55, 6, "tm")
    Rf, Tf, Tb, Rb = _interface(b, f, b, f, dx, w)
    assert np.max(np.abs(Tf - np.eye(6))) < 1e-12
    assert np.max(np.abs(Rf)) < 1e-12


def test_tm_straight_lossless():
    """A uniform TM section is lossless."""
    from photonix.em.eme import Section, eme_smatrix

    x = np.linspace(-3, 3, 481)
    dx = x[1] - x[0]
    r = eme_smatrix([Section(_core(x, 0.5), 5.0)], dx, 1.55, 4, "tm")
    assert abs(abs(r.Tf[0, 0]) - 1.0) < 1e-6


def test_tm_step_energy_and_reciprocity():
    """TM mode-mismatch step conserves energy and is reciprocal."""
    from photonix.em.eme import Section, eme_smatrix

    x = np.linspace(-3, 3, 481)
    dx = x[1] - x[0]
    r = eme_smatrix([Section(_core(x, 0.5), 2.0), Section(_core(x, 0.7), 2.0)],
                    dx, 1.55, 5, "tm")
    power = np.sum(np.abs(r.Tf[:, 0]) ** 2) + np.sum(np.abs(r.Rf[:, 0]) ** 2)
    assert abs(power - 1.0) < 1e-9
    assert np.max(np.abs(r.Tf - r.Tb.T)) < 1e-9


def test_tm_taper_more_adiabatic_than_abrupt():
    """A smooth TM taper transmits more into the fundamental than an abrupt step."""
    from photonix.em.eme import Section, eme_smatrix

    x = np.linspace(-3, 3, 481)
    dx = x[1] - x[0]
    abrupt = eme_smatrix([Section(_core(x, 0.5), 0.01), Section(_core(x, 0.7), 0.01)],
                         dx, 1.55, 5, "tm")
    ws = np.linspace(0.5, 0.7, 40)
    smooth = eme_smatrix([Section(_core(x, w), 20.0 / 40) for w in ws], dx, 1.55, 5, "tm")
    assert abs(smooth.Tf[0, 0]) ** 2 > abs(abrupt.Tf[0, 0]) ** 2
    assert abs(smooth.Tf[0, 0]) ** 2 > 0.99
