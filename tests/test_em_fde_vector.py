"""Tests for the semivectorial (polarization-resolved) 2-D FDE solver.

Anchors:
* the canonical 500x220 nm SOI strip lands TE0/TM0 in the accepted physical
  window, with TE more confined than TM (the qualitative split scalar cannot do);
* the differentiable xp matvec reproduces the scipy eigenproblem operator
  exactly (so the adjoint differentiates the right thing);
* the non-symmetric (left-eigenvector) gradient matches finite differences.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_jax

import photonix as px
import photonix.em as em
from photonix.em.fde_vector import _apply, _assemble


def test_soi_strip_te_tm_physical_range():
    """500x220 Si strip: semivectorial TE0/TM0 in range, TE0 > TM0."""
    te = em.n_eff_vector(width=0.5, thickness=0.22, resolution=30, polarization="te")
    tm = em.n_eff_vector(width=0.5, thickness=0.22, resolution=30, polarization="tm")
    assert 2.2 < te < 2.6, te
    assert 1.6 < tm < 2.05, tm
    assert te > tm


def test_vector_te_below_scalar():
    """Scalar FDE over-estimates the strip TE index; semivector should sit below it."""
    scalar = em.solve_modes(width=0.5, thickness=0.22, resolution=30).neff0
    vector_te = em.n_eff_vector(width=0.5, thickness=0.22, resolution=30, polarization="te")
    assert vector_te < scalar + 1e-6
    assert vector_te > 1.444


def test_te_widens_toward_slab():
    """As width grows the quasi-TE index approaches the vertical-slab TE index."""
    nv = em.slab.slab_neff(thickness=0.22, wl=1.55, polarization="te")
    wide = em.n_eff_vector(width=6.0, thickness=0.22, resolution=20, polarization="te")
    assert abs(wide - nv) / nv < 0.02  # within 2% in the wide-width limit


def test_operator_consistency_te_and_tm():
    """xp matvec == scipy sparse operator (guarantees the adjoint is exact)."""
    from photonix.em.geometry import rectangular_waveguide

    cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=18)
    k0 = 2 * np.pi / 1.55
    rng = np.random.default_rng(0)
    field = rng.standard_normal(cs.eps.size)
    for pol in ("te", "tm"):
        A = _assemble(cs.eps, cs.dx, cs.dy, k0, pol)
        ref = A @ field
        got = np.asarray(
            _apply(px.xp.asarray(cs.eps.reshape(-1)), px.xp.asarray(field),
                   cs.eps.shape, cs.dx, cs.dy, k0, pol)
        )
        assert np.allclose(ref, got, atol=1e-9, rtol=1e-9), pol


@requires_jax
def test_vector_gradient_matches_fd():
    """Non-symmetric left-eigenvector adjoint vs central finite differences."""
    import jax

    from photonix.em.geometry import rectangular_waveguide

    cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=16)
    eps0 = px.xp.asarray(cs.eps.reshape(-1))
    shape, k0 = cs.eps.shape, 2 * np.pi / 1.55
    f = lambda e: em.n_eff_eps_vector(e, shape, cs.dx, cs.dy, k0, "te")  # noqa: E731
    g = np.asarray(jax.grad(f)(eps0))
    kmax = int(np.argmax(np.abs(g)))
    h = 1e-6
    fd = (float(f(eps0.at[kmax].add(h))) - float(f(eps0.at[kmax].add(-h)))) / (2 * h)
    assert abs(g[kmax] - fd) / abs(fd) < 1e-3, (g[kmax], fd)


# --------------------------------------------------------------------------- #
# Full-vector (Yee-grid) solver
# --------------------------------------------------------------------------- #
def test_fullvector_soi_strip_literature():
    """500x220 Si strip: full-vector TE0 ~ 2.45 (literature), TM0 ~ 1.81."""
    r = em.solve_modes_fullvector(width=0.5, thickness=0.22, resolution=30, num_modes=2)
    te0 = r.neff0
    tm0 = float(np.real(r.n_eff[1]))
    assert 2.40 < te0 < 2.50, te0
    assert 1.70 < tm0 < 1.90, tm0
    assert te0 > tm0


def test_fullvector_below_semivector_below_scalar():
    """Each refinement lowers the strip TE index: full < semivector < scalar."""
    full = em.n_eff_fullvector(width=0.5, thickness=0.22, resolution=30)
    semi = em.n_eff_vector(width=0.5, thickness=0.22, resolution=30, polarization="te")
    scalar = em.solve_modes(width=0.5, thickness=0.22, resolution=30).neff0
    assert full < semi < scalar
    assert full > 1.444


def test_fullvector_polarization_fractions():
    """Fundamental is strongly Ex (TE-like); second mode is Ey (TM-like)."""
    r = em.solve_modes_fullvector(width=0.5, thickness=0.22, resolution=30, num_modes=2)
    assert r.te_fraction[0] > 0.9       # TE-like
    assert r.te_fraction[1] < 0.1       # TM-like (the hybrid Ey mode)


def test_vector_custom_grid_guided_cutoff_uses_highest_exterior_index(monkeypatch):
    """Vector mode labels use the highest-index exterior material as cutoff."""
    import photonix.em.fde_vector as fde_vector

    eps = np.ones((4, 4))
    eps[-1] = 1.5**2

    def fake_solve(*_args):
        return np.array([1.6, 1.4]), np.zeros((2, 4, 4)), np.zeros(16)

    monkeypatch.setattr(fde_vector, "_solve", fake_solve)
    result = fde_vector.solve_modes_vector(eps=eps, num_modes=2)
    assert result.guided.tolist() == [True, False]
    assert result.n_guided == 1


@pytest.mark.parametrize("solver", [em.solve_modes_vector, em.solve_modes_fullvector])
def test_vector_custom_grid_rejects_nonuniform_coordinates(solver):
    eps = np.ones((3, 4))
    grid = ([0.0, 0.1, 0.3, 0.4], [0.0, 0.2, 0.4])
    with pytest.raises(ValueError, match="uniformly spaced"):
        solver(eps=eps, grid=grid)


def test_vector_solvers_reject_nonpositive_mode_count():
    import pytest

    with pytest.raises(ValueError, match="num_modes"):
        em.solve_modes_vector(num_modes=0)
    with pytest.raises(ValueError, match="num_modes"):
        em.solve_modes_fullvector(num_modes=-1)


def test_fullvector_convergence():
    """TE0 settles (changes shrink) under grid refinement."""
    n20 = em.n_eff_fullvector(width=0.5, thickness=0.22, resolution=20)
    n30 = em.n_eff_fullvector(width=0.5, thickness=0.22, resolution=30)
    n40 = em.n_eff_fullvector(width=0.5, thickness=0.22, resolution=40)
    assert abs(n40 - n30) < abs(n30 - n20)


def test_fullvector_operator_consistency():
    """xp matvec == scipy sparse Omega (guarantees the adjoint is exact)."""
    from photonix.em.fde_vector import _apply_fullvector, _assemble_fullvector
    from photonix.em.geometry import rectangular_waveguide

    cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=14)
    k0 = 2 * np.pi / 1.55
    A = _assemble_fullvector(cs.eps, cs.dx, cs.dy, k0)
    rng = np.random.default_rng(1)
    field = rng.standard_normal(2 * cs.eps.size)
    ref = A @ field
    got = np.asarray(
        _apply_fullvector(px.xp.asarray(cs.eps.reshape(-1)), px.xp.asarray(field),
                          cs.eps.shape, cs.dx, cs.dy, k0)
    )
    assert np.allclose(ref, got, atol=1e-9, rtol=1e-9)


@requires_jax
def test_fullvector_gradient_matches_fd():
    """Full-vector left-eigenvector adjoint (lambda = -neff^2) vs finite differences."""
    import jax

    from photonix.em.geometry import rectangular_waveguide

    cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=14)
    eps0 = px.xp.asarray(cs.eps.reshape(-1))
    shape, k0 = cs.eps.shape, 2 * np.pi / 1.55
    f = lambda e: em.n_eff_eps_fullvector(e, shape, cs.dx, cs.dy, k0)  # noqa: E731
    g = np.asarray(jax.grad(f)(eps0))
    kmax = int(np.argmax(np.abs(g)))
    h = 1e-6
    fd = (float(f(eps0.at[kmax].add(h))) - float(f(eps0.at[kmax].add(-h)))) / (2 * h)
    assert abs(g[kmax] - fd) / abs(fd) < 1e-3, (g[kmax], fd)


# --------------------------------------------------------------------------- #
# PML + bend (radiation) loss
# --------------------------------------------------------------------------- #
def test_pml_straight_nonperturbing():
    """A straight guide with PML keeps Im(n_eff) ~ 0 and a physical Re(n_eff).

    Tight tolerance on purpose: with the default grid the inner PML must not
    overlap the core (an overlapping absorber used to fake Im(n_eff) ~ 3e-5).
    """
    m = em.bend_loss_fullvector(bend_radius=None, resolution=24)
    assert abs(m.n_eff.imag) < 1e-6        # PML does not invent loss
    assert 1.444 < m.n_eff.real < 3.4757


def test_bend_loss_increases_as_radius_tightens():
    """Radiation loss grows as the bend tightens, with the physical mode tracked."""
    # A reduced inner gap keeps the full grid on the physical R+x > 0 side of
    # the conformal map for these deliberately tight radii.
    r12 = em.bend_loss_fullvector(bend_radius=1.2, resolution=24, inner=0.1)
    r10 = em.bend_loss_fullvector(bend_radius=1.0, resolution=24, inner=0.1)
    assert r10.loss_db_per_90deg > r12.loss_db_per_90deg > 0.0
    assert r10.loss_db_per_90deg > 1e-3          # appreciable at a tight bend
    assert r10.overlap > 0.9 and r12.overlap > 0.9  # tracked the guided mode
    assert r10.n_eff.imag != 0.0                 # complex n_eff (loss) resolved


def test_bend_grid_rejects_conformal_singularity_crossing():
    with pytest.raises(ValueError, match="does not cross x=-R"):
        em.bend_loss_fullvector(bend_radius=1.0, resolution=24)


def test_fullvector_pml_uses_distinct_integer_and_half_grid_samples():
    from photonix.em.fde_vector import _pml_stretch

    coord = np.linspace(-2.0, 2.0, 17)
    h = coord[1] - coord[0]
    bounds = (coord[0] - h / 2, coord[-1] + h / 2)
    integer = _pml_stretch(coord, 2 * np.pi / 1.55, 0.75, 4.0, bounds=bounds)
    half = _pml_stretch(coord + h / 2, 2 * np.pi / 1.55, 0.75, 4.0, bounds=bounds)
    assert not np.allclose(integer, half)
    assert integer[0].imag == pytest.approx(integer[-1].imag)


# --------------------------------------------------------------------------- #
# 2-D vectorial mode/overlap foundation (for a future 2-D hybrid EME)
# --------------------------------------------------------------------------- #
def test_fullvector_modes_biorthonormal():
    """2-D full-vector modes are power-orthonormal: integral(E x H).z = identity."""
    from photonix.em.geometry import rectangular_waveguide

    cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=22)
    k0 = 2 * np.pi / 1.55
    neff, et, ht = em.fullvector_transverse_fields(cs.eps, cs.dx, cs.dy, k0, num_modes=4)
    M = em.power_overlap(et, ht, cs.dx * cs.dy)
    assert np.max(np.abs(M - np.eye(4))) < 1e-10      # bi-orthonormal to machine eps
    assert 2.3 < float(neff[0].real) < 2.5            # fundamental in range


def test_fullvector_magnetic_fields_include_modal_impedance():
    """The first-order equation is Q E_t = n_eff H_t, not Q E_t = H_t."""
    from photonix.em.fde_vector import _pq_operators
    from photonix.em.geometry import rectangular_waveguide

    cs = rectangular_waveguide(width=0.5, thickness=0.22, resolution=18)
    k0 = 2 * np.pi / 1.55
    neff, et, ht = em.fullvector_transverse_fields(
        cs.eps, cs.dx, cs.dy, k0, num_modes=2
    )
    _p, q = _pq_operators(cs.eps, cs.dx, cs.dy, k0)
    assert np.allclose(ht, (q @ et) / neff[None, :], atol=1e-10, rtol=1e-10)
