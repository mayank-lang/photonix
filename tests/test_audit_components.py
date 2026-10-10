"""Regression tests for component bugs found by the round-2 audit."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

import photonix as px
import photonix.components as c


def _mmi_T(n):
    s = c.mmi_nxn(n=n)
    return np.array([[complex(s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])


def _modal_mmi(n, modes=60, npts=4000):
    """Facet-referenced modal self-imaging matrix (exp(-j beta z), parabolic beta)."""
    x = np.linspace(0.0, 1.0, npts)
    dx = x[1] - x[0]
    nu = np.arange(modes)
    psi = np.sin((nu[:, None] + 1) * np.pi * x[None, :])
    psi /= np.sqrt((psi**2).sum(1, keepdims=True) * dx)
    xs = ((2 * np.arange(1, n + 1) - 1) / (2 * n))[::-1]
    g = np.exp(-(((x[None, :] - xs[:, None]) * 6 * n) ** 2))
    g /= np.sqrt((g**2).sum(1, keepdims=True) * dx)
    cpl = g @ psi.T * dx
    return cpl @ np.diag(np.exp(1j * nu * (nu + 2) * np.pi / n)) @ cpl.T


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_cascaded_mmis_form_a_mirrored_image(n):
    T = _mmi_T(n)
    assert np.allclose(np.abs(np.linalg.matrix_power(T, n)) ** 2, np.fliplr(np.eye(n)), atol=1e-12)


@pytest.mark.parametrize("n", [3, 5])
def test_mmi_matches_modal_self_imaging_up_to_port_gauge(n):
    T, M = _mmi_T(n), _modal_mmi(n)
    worst = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for m in range(n):
                    a = T[i, j] * T[k, m] / (T[i, m] * T[k, j])
                    b = M[i, j] * M[k, m] / (M[i, m] * M[k, j])
                    worst = max(worst, abs(np.angle(a / b)))
    assert worst < 0.05


def test_raised_cosine_apodization_peaks_at_centre_and_suppresses_sidelobes():
    from photonix.components.bragg import _apodization_profile

    prof = _apodization_profile("raised_cosine", 41, 1.0)
    assert prof[20] == pytest.approx(1.0, abs=2e-3) and prof[0] < 0.01
    wl_b = c.bragg_wavelength(0.3229)
    wl = np.linspace(wl_b - 0.04, wl_b + 0.04, 8001)
    off = np.abs(wl - wl_b) > 0.012
    uni = np.asarray(px.power(c.bragg_grating(wl=wl, dn=0.02, n_periods=300)[("o1", "o1")]))
    rc = np.asarray(px.power(c.bragg_grating(wl=wl, dn=0.02, n_periods=300, n_sections=40,
                                             apodization="raised_cosine", apodization_strength=1.0)[("o1", "o1")]))
    assert rc[off].max() < 0.01 * uni[off].max()


def test_coarse_apodization_warns():
    with pytest.warns(UserWarning, match="coarsely"):
        c.bragg_grating(apodization="gaussian", n_sections=1)


@pytest.mark.parametrize("n, spacing", [(8, 0.02), (4, 0.0125), (4, 0.016), (8, 0.0008), (16, 0.0004)])
def test_awg_fsr_never_below_channel_plan(n, spacing):
    d = c.awg_design(n_channels=n, channel_spacing=spacing)
    assert d["fsr"] >= n * spacing * (1 - 1e-12)


def test_cwdm_awg_edge_channels_do_not_alias():
    n, sp = 8, 0.02
    d = c.awg_design(n_channels=n, channel_spacing=sp)
    s = c.awg(wl=float(d["channel_wavelengths"][-1]), n_channels=n, channel_spacing=sp)
    assert 10 * math.log10(float(px.power(s[("o1", "o2")]))) < -30.0


def test_awg_needs_at_least_as_many_arms_as_channels():
    with pytest.raises(ValueError, match="n_arms"):
        c.awg_design(n_channels=8, n_arms=5)


@requires_jax
@pytest.mark.parametrize("name", sorted(n for n, f in c.MODELS.items() if hasattr(f, "waveguide_parameters")))
def test_waveguide_models_differentiate_and_jit_in_neff(name):
    import jax

    model = c.MODELS[name]
    port = sorted(model(wl=1.55))[0]
    kwargs = {"n_channels": 4, "channel_spacing": 0.0016} if name == "awg" else {}
    if name == "awg":
        def f(n):
            return px.power(model(wl=1.5506, delta_neff=n - 2.4, **kwargs)[("o1", "o3")])
    else:
        def f(n):
            return px.power(model(wl=1.5506, neff=n, **kwargs)[port]) + px.xp.real(model(wl=1.5506, neff=n)[port])
    g = float(jax.grad(f)(2.4))
    assert np.isfinite(g)
    assert g == pytest.approx(float(finite_diff(f, 2.4, 1e-6)), rel=1e-4, abs=1e-6)
    assert np.isfinite(float(jax.jit(f)(2.4)))


@requires_jax
def test_edge_coupler_differentiable_in_facet_index():
    import jax

    def f(n):
        return px.power(c.edge_coupler(n_chip=n)[("o1", "o2")])

    assert float(jax.grad(f)(3.0)) == pytest.approx(float(finite_diff(f, 3.0, 1e-6)), rel=1e-6)


@requires_jax
def test_bragg_gradient_finite_at_gamma_zero():
    import jax

    wl_b = c.bragg_wavelength(0.3229)

    def f(k):
        return px.power(c.bragg_grating(wl=wl_b, kappa=k, n_periods=200)[("o1", "o1")])

    g = float(jax.grad(f)(0.0))
    assert np.isfinite(g)
    g1 = float(jax.grad(f)(0.01))
    assert g1 == pytest.approx(float(finite_diff(f, 0.01, 1e-7)), rel=1e-5)


def test_evanescent_coupler_warns_outside_calibrated_wavelengths_and_for_arrays():
    with pytest.warns(UserWarning, match="wl spans"):
        c.evanescent_coupler(wl=1.31)
    with pytest.warns(UserWarning, match="gap spans"):
        c.evanescent_coupler(gap=np.array([0.2, 0.5]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c.evanescent_coupler(wl=np.linspace(1.5, 1.6, 5), gap=np.array(0.2))


@requires_jax
def test_jit_with_traced_reflection_and_zero_radius():
    import jax
    import jax.numpy as jnp

    out = jax.jit(lambda r: px.power(c.crossing(reflection_db=r)[("o1", "o3")]))(-40.0)
    assert np.isfinite(float(out))

    def f(gap):
        return px.power(c.evanescent_coupler(gap=gap, bend_radius=jnp.asarray(0.0))[("o1", "o3")])

    g = float(jax.grad(f)(0.2))
    assert np.isfinite(g) and g == pytest.approx(float(finite_diff(f, 0.2, 1e-6)), rel=1e-5)


def test_built_in_voltage_falls_with_temperature():
    v300 = float(c.built_in_voltage(5e17, 5e17, 300.0))
    v310 = float(c.built_in_voltage(5e17, 5e17, 310.0))
    slope_mv_k = (v310 - v300) / 10.0 * 1e3
    assert -1.6 < slope_mv_k < -0.9
    assert float(c.intrinsic_density(300.0)) == pytest.approx(c.NI_SI_300K)
    # explicit ni restores the fixed-density behaviour
    assert float(c.built_in_voltage(5e17, 5e17, 310.0, ni=c.NI_SI_300K)) > v300
