"""Tests for the FDFD solver: PML, exact adjoint gradient, topology optimization."""
from __future__ import annotations

import numpy as np

from photonix.em.fdfd import FDFD, focus_objective, optimize_focus, point_source, scpml_stretch

WL, NCLAD, NCORE = 1.55, 1.444, 3.4757


def test_zero_pml_is_identity_and_overlapping_pml_is_rejected():
    import pytest

    si, sh = scpml_stretch(12, 0.05, 0, 2 * np.pi / WL)
    assert np.array_equal(si, np.ones(12))
    assert np.array_equal(sh, np.ones(13))
    with pytest.raises(ValueError, match="non-PML"):
        scpml_stretch(12, 0.05, 6, 2 * np.pi / WL)
    with pytest.raises(ValueError, match="non-PML"):
        FDFD(np.ones((12, 20)), 0.05, 0.05, WL, npml=6)
    with pytest.raises(ValueError, match="m must"):
        scpml_stretch(12, 0.05, 2, 2 * np.pi / WL, m=-1)
    with pytest.raises(ValueError, match="log_R"):
        scpml_stretch(12, 0.05, 2, 2 * np.pi / WL, log_R=1)


def test_pml_and_dirichlet_stencil_are_symmetric_at_both_boundaries():
    """Both exterior faces must contribute equally to the scalar stencil."""
    from photonix.em.fdfd import _d_forward

    n, h = 7, 0.2
    d = _d_forward(n, h)
    lap = (-d.T @ d).toarray() * h**2
    expected = np.diag(-2.0 * np.ones(n))
    expected += np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    assert d.shape == (n + 1, n)
    assert np.allclose(lap, expected)

    si, sh = scpml_stretch(20, h, 4, 2 * np.pi / WL)
    assert si.shape == (20,)
    assert sh.shape == (21,)
    assert np.allclose(si, si[::-1])
    assert np.allclose(sh, sh[::-1])


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
    # S12 is independently extracted from the reverse solve, yet reciprocity
    # still makes it agree with S21 for this lossless scalar structure.
    assert abs(s[("o1", "o2")] - s[("o2", "o1")]) < 1e-8


def test_port_phase_uses_longitudinal_grid_dispersion_and_checks_nyquist():
    """Port launch/extraction must use the phase represented by the x stencil."""
    import pytest

    from photonix.em.fdfd import _decompose, mode_source

    ny, nx, col = 3, 7, 2
    beta, dx, dy = 4.0, 0.25, 0.1
    qdx = 2.0 * np.arcsin(beta * dx / 2.0)
    profile = np.array([0.5, 1.0, -0.25])

    source = mode_source(ny, nx, col, profile, beta, dx)
    assert np.allclose(source[:, col + 1], profile * np.exp(-1j * qdx))
    assert not np.allclose(source[:, col + 1], profile * np.exp(-1j * beta * dx))

    forward, backward = 1.2 - 0.3j, -0.15 + 0.4j
    field = np.zeros((ny, nx), complex)
    field[:, col] = profile * (forward + backward)
    field[:, col + 1] = profile * (
        forward * np.exp(-1j * qdx) + backward * np.exp(1j * qdx)
    )
    got_forward, got_backward = _decompose(field, col, profile, beta, dy, dx)
    assert np.allclose(got_forward, forward)
    assert np.allclose(got_backward, backward)

    with pytest.raises(ValueError, match="Nyquist"):
        mode_source(ny, nx, col, profile, beta=2.0 / dx, dx=dx)


def test_waveguide_sparams_assigns_forward_and_reverse_to_input_output_keys(monkeypatch):
    """S21/S12 use SDict's (input, output) keys and discrete power flux."""
    import photonix.em.fdfd as fdfd

    beta_in, beta_out, dx = 2.0, 6.0, 0.2
    modes = iter([
        (beta_in, np.ones(4)),
        (beta_out, np.ones(4)),
    ])
    monkeypatch.setattr(fdfd, "waveguide_mode", lambda *args, **kwargs: next(modes))

    # Forward input, forward output, reverse input/reflection at o2, then
    # reverse transmitted output at o1. The deliberately non-reciprocal values
    # make copying S21 into S12 observable in this extraction-level unit test.
    decompositions = iter([
        (2.0 + 0.0j, 0.2 + 0.0j),
        (1.0 + 0.0j, 9.0 + 0.0j),
        (0.4 + 0.0j, 4.0 + 0.0j),
        (8.0 + 0.0j, 3.0 + 0.0j),
    ])
    monkeypatch.setattr(fdfd, "_decompose", lambda *args, **kwargs: next(decompositions))

    class FakeFDFD:
        def __init__(self, eps, dx, dy, wl, npml=12, polarization="te"):
            self.shape = eps.shape

        def factor(self):
            return self

        def solve(self, source):
            return np.zeros(self.shape, complex)

    monkeypatch.setattr(fdfd, "FDFD", FakeFDFD)
    s = fdfd.waveguide_sparams(
        np.ones((4, 20)), dx=dx, dy=0.1, wl=1.55,
        src_col=2, in_mon_col=5, out_mon_col=14, npml=0,
    )

    def grid_flux(beta):
        qdx = 2.0 * np.arcsin(beta * dx / 2.0)
        return np.sin(qdx) / dx

    flux_in, flux_out = grid_flux(beta_in), grid_flux(beta_out)
    expected_s21 = (1.0 / 2.0) * np.sqrt(flux_out / flux_in)
    expected_s12 = (3.0 / 4.0) * np.sqrt(flux_in / flux_out)
    # Conventional S21 is output 2 due to input 1, hence Photonix key
    # (input=o1, output=o2).  S12 is the independently simulated reverse path.
    assert np.allclose(s[("o1", "o2")], expected_s21)
    assert np.allclose(s[("o2", "o1")], expected_s12)
    assert not np.allclose(s[("o1", "o2")], s[("o2", "o1")])
    assert np.allclose(s[("o2", "o2")], 0.1)


def test_waveguide_sparams_rejects_complex_eps():
    """Complex eps must raise, not silently drop absorption."""
    import pytest

    from photonix.em.fdfd import waveguide_sparams

    eps = np.full((40, 60), 2.0 + 0.1j)
    with pytest.raises(ValueError, match="imaginary"):
        waveguide_sparams(eps, dx=0.05, dy=0.05, wl=1.55, src_col=8,
                          in_mon_col=16, out_mon_col=44)
