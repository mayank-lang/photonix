"""Validation of the extended component library against analytic/independent results."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax
from scipy import integrate

import photonix as px
import photonix.components as c


def _dense(s):
    S, pm = px.as_sdense(s)
    return np.asarray(S), pm


def _max_singular(s):
    S, _ = _dense(s)
    return float(np.max(np.linalg.svd(S, compute_uv=False)))


# --------------------------------------------------------------------------- #
# Registry-wide contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(c.MODELS))
def test_every_registered_model_has_defaults_and_is_passive(name):
    s = c.MODELS[name](wl=px.linspace(1.53, 1.57, 5))
    px.core.validate_sdict(s)
    S, _ = px.as_sdense(s)
    sv = np.linalg.svd(np.asarray(S), compute_uv=False)
    assert np.all(sv <= 1.0 + 1e-9), f"{name} is not passive"


def test_models_run_under_numpy_backend(tmp_path):
    """Every registered model evaluates without JAX (CI may run the fallback)."""
    import subprocess
    import sys

    code = (
        "import photonix as px, numpy as np\n"
        "assert px.backend_name() == 'numpy'\n"
        "for name, model in px.components.MODELS.items():\n"
        "    s = model(wl=np.linspace(1.54, 1.56, 3))\n"
        "    px.core.validate_sdict(s)\n"
        "print('ok')\n"
    )
    env = {**__import__("os").environ, "PHOTONIX_BACKEND": "numpy"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


# --------------------------------------------------------------------------- #
# Waveguides
# --------------------------------------------------------------------------- #
def test_euler_length_matches_integrated_curvature():
    radius, angle, p = 4.0, 90.0, 0.6
    theta = math.radians(angle)
    lc = p * theta * radius                     # clothoid ramp length
    # curvature: ramp 0 -> 1/R, arc at 1/R, ramp 1/R -> 0; integrate angle
    arc = radius * theta * (1 - p)
    turned = 2 * integrate.quad(lambda s: s / (radius * lc), 0, lc)[0] + arc / radius
    assert turned == pytest.approx(theta, rel=1e-12)
    assert float(c.euler_bend_length(radius, angle, p)) == pytest.approx(2 * lc + arc, rel=1e-12)


def test_euler_bend_phase_equals_straight_of_same_length():
    L = float(c.euler_bend_length(5.0, 90.0, 0.5))
    a = complex(c.bend_euler(radius=5.0, angle=90.0, p=0.5)[("o1", "o2")])
    b = complex(c.straight(length=L)[("o1", "o2")])
    assert a == pytest.approx(b, abs=1e-12)


def test_taper_quadrature_integrates_width_dependent_index():
    def neff(wl, w):
        return 1.9 + 0.8 * w - 0.3 * w**2   # quadratic: quadrature must be exact

    w1, w2, L = 0.45, 1.2, 25.0
    mean = integrate.quad(lambda z: neff(1.55, w1 + (w2 - w1) * z / L), 0, L)[0] / L
    s = c.taper(length=L, width1=w1, width2=w2, neff=neff)
    expected = np.exp(-1j * 2 * np.pi / 1.55 * mean * L)
    assert complex(s[("o1", "o2")]) == pytest.approx(expected, abs=1e-10)


# --------------------------------------------------------------------------- #
# Couplers
# --------------------------------------------------------------------------- #
def test_evanescent_coupler_unitary_over_band(wl):
    s = c.evanescent_coupler(wl=wl, gap=0.18, length=23.0)
    S, _ = _dense(s)
    eye = np.eye(4)
    for i in range(0, len(wl), 50):
        assert np.allclose(S[i].conj().T @ S[i], eye, atol=1e-12)


def test_evanescent_coupler_cross_length_and_ideal_equivalence():
    dn = float(c.supermode_splitting(0.25, 1.55))
    L_c = 1.55 / (2 * dn)
    assert float(px.power(c.evanescent_coupler(gap=0.25, length=L_c)[("o1", "o3")])) == pytest.approx(1.0)
    # Same power split as the ideal coupler with coupling = sin^2(kappa L)
    L = 0.3 * L_c
    k2 = math.sin(math.pi / 1.55 * dn * L) ** 2
    phys = c.evanescent_coupler(gap=0.25, length=L)
    ideal = c.directional_coupler(coupling=k2)
    for key in (("o1", "o3"), ("o1", "o4")):
        assert float(px.power(phys[key])) == pytest.approx(float(px.power(ideal[key])), abs=1e-12)


def test_evanescent_fit_reproduces_calibration_points():
    # Values from the full-vector FDE calibration table (80 px/um).
    table = {(0.10, 1.55): 5.325972e-02, (0.20, 1.50): 1.612974e-02,
             (0.30, 1.60): 1.138730e-02, (0.40, 1.55): 3.610633e-03}
    for (gap, wl), dn in table.items():
        assert float(c.supermode_splitting(gap, wl)) == pytest.approx(dn, rel=0.014)


@pytest.mark.parametrize("arms", [1, 2])
def test_curved_coupling_length_matches_numerical_gap_integral(arms):
    """The Gaussian closed form for a curved gap agrees with direct quadrature."""
    gap, radius, wl = 0.2, 10.0, 1.55

    def dn_at(z):
        return float(c.supermode_splitting(gap + arms * z * z / (2 * radius), wl))

    numeric = integrate.quad(dn_at, -radius, radius, limit=200)[0]
    phase_numeric = math.pi / wl * numeric
    s = c.evanescent_coupler(gap=gap, length=0.0, bend_radius=radius, curved_arms=arms)
    phase_model = math.asin(math.sqrt(float(px.power(s[("o1", "o3")]))))
    # The model drops the (small) curvature of ln(dn) in the gap: <2 % in practice.
    assert phase_model == pytest.approx(phase_numeric, rel=0.025)


def test_splitting_extrapolation_decays_monotonically():
    gaps = np.linspace(0.02, 2.0, 400)
    dn = np.asarray(c.supermode_splitting(gaps, 1.55))
    assert np.all(np.diff(dn) < 0)
    assert dn[-1] < 1e-8


def test_evanescent_coupler_warns_outside_calibration():
    with pytest.warns(UserWarning, match="outside the calibrated range"):
        c.evanescent_coupler(gap=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c.evanescent_coupler(gap=0.6, splitting=lambda g, wl: 0.01 * np.exp(-g / 0.1))


@requires_jax
def test_evanescent_coupler_gradient_wrt_gap_matches_fd():
    def f(gap):
        return px.power(c.evanescent_coupler(gap=gap, length=12.0, bend_radius=5.0)[("o1", "o3")])

    g = float(px.grad(f)(0.22))
    assert g == pytest.approx(float(finite_diff(f, 0.22, 1e-5)), rel=1e-5)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
def test_mmi_nxn_unitary_reciprocal_equal_split(n):
    s = c.mmi_nxn(n=n)
    S, _ = _dense(s)
    assert np.allclose(S.conj().T @ S, np.eye(2 * n), atol=1e-12)
    assert np.allclose(S, S.T)
    for i in range(1, n + 1):
        for j in range(n + 1, 2 * n + 1):
            assert float(px.power(s[(f"o{i}", f"o{j}")])) == pytest.approx(1.0 / n)


def test_mmi_nxn_two_by_two_matches_mmi2x2():
    a, b = c.mmi_nxn(n=2), c.mmi2x2()
    for key, val in b.items():
        assert complex(a[key]) == pytest.approx(complex(val), abs=1e-14)


def test_mmi_nxn_imbalance_keeps_passivity():
    s = c.mmi_nxn(n=6, loss_db=0.3, imbalance_db=1.0)
    assert _max_singular(s) <= 1.0
    centre = float(px.power(s[("o1", "o9")]))   # output index 3 of 6 (top to bottom)
    edge = float(px.power(s[("o1", "o12")]))    # top output
    assert centre > edge


def test_crossing_passivity_bound_is_tight():
    # |t|^2 + 4|x|^2 = 1 exactly: largest singular value is 1.
    x_db = 10 * math.log10(0.25 * (1 - 10 ** (-0.05)))
    s = c.crossing(loss_db=0.5, crosstalk_db=x_db)
    assert _max_singular(s) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="not passive"):
        c.crossing(loss_db=0.5, crosstalk_db=x_db + 0.1)
    with pytest.raises(ValueError, match="not passive"):
        c.crossing(loss_db=0.0, crosstalk_db=-60.0, reflection_db=-30.0)
    s = c.crossing(loss_db=0.5, crosstalk_db=-40.0, reflection_db=-30.0)
    assert _max_singular(s) <= 1.0


# --------------------------------------------------------------------------- #
# Bragg gratings
# --------------------------------------------------------------------------- #
def test_bragg_lossless_unitary_and_symmetric(wl):
    s = c.bragg_grating(wl=wl, dn=0.04, n_periods=250)
    S, _ = _dense(s)
    for i in range(0, len(wl), 25):
        assert np.allclose(S[i].conj().T @ S[i], np.eye(2), atol=1e-10)
    assert np.allclose(np.asarray(s[("o1", "o1")]), np.asarray(s[("o2", "o2")]), atol=1e-10)


def test_bragg_peak_reflectance_and_null_bandwidth():
    period, n_per, dn = 0.3229, 200, 0.03
    wl_b = c.bragg_wavelength(period)
    L = period * n_per
    kappa = float(c.grating_kappa(dn, wl_b))
    wl = np.linspace(wl_b - 0.03, wl_b + 0.03, 60001)
    R = np.asarray(px.power(c.bragg_grating(wl=wl, period=period, n_periods=n_per, dn=dn)[("o1", "o1")]))
    i0 = int(np.argmax(R))
    assert wl[i0] == pytest.approx(wl_b, abs=2e-5)
    assert R[i0] == pytest.approx(math.tanh(kappa * L) ** 2, rel=1e-4)
    # First reflection nulls on either side of the stop band.
    left = i0 - np.argmax(np.diff(R[:i0 + 1][::-1]) > 0)
    right = i0 + np.argmax(np.diff(R[i0:]) > 0)
    measured = wl[right] - wl[left]
    expected = wl_b**2 / (math.pi * 4.2) * math.sqrt(kappa**2 + (math.pi / L) ** 2)
    # kappa itself is wavelength dependent (2 dn / wl): allow 1 %
    assert measured == pytest.approx(expected, rel=0.01)


def test_bragg_sectioning_is_exact_for_uniform_grating(wl):
    a = c.bragg_grating(wl=wl, n_sections=1)
    b = c.bragg_grating(wl=wl, n_sections=12)
    for key in a:
        assert np.allclose(np.asarray(a[key]), np.asarray(b[key]), atol=1e-10)


def test_quarter_wave_shift_opens_transmission_peak_at_bragg():
    wl_b = c.bragg_wavelength(0.3229)
    uniform = c.bragg_grating(wl=wl_b, dn=0.03, n_periods=400)
    shifted = c.bragg_grating(wl=wl_b, dn=0.03, n_periods=400, phase_shift=math.pi)
    kL = float(c.grating_kappa(0.03, wl_b)) * 0.3229 * 400
    assert float(px.power(uniform[("o1", "o2")])) == pytest.approx(1 / math.cosh(kL) ** 2, rel=1e-6)
    assert float(px.power(shifted[("o1", "o2")])) == pytest.approx(1.0, abs=1e-8)


def test_apodization_suppresses_sidelobes():
    wl_b = c.bragg_wavelength(0.3229)
    wl = np.linspace(wl_b - 0.04, wl_b + 0.04, 8001)
    off = np.abs(wl - wl_b) > 0.012
    uni = np.asarray(px.power(c.bragg_grating(wl=wl, dn=0.02, n_periods=300)[("o1", "o1")]))
    apo = np.asarray(px.power(c.bragg_grating(wl=wl, dn=0.02, n_periods=300, n_sections=40,
                                              apodization="gaussian", apodization_strength=0.5)[("o1", "o1")]))
    assert apo[off].max() < 0.1 * uni[off].max()


def test_bragg_loss_and_reciprocity(wl):
    s = c.bragg_grating(wl=wl, loss_db_cm=20.0, n_sections=20, apodization="raised_cosine",
                        apodization_strength=1.0)
    total = np.asarray(px.power(s[("o1", "o1")]) + px.power(s[("o1", "o2")]))
    assert np.all(total < 1.0)
    assert np.allclose(np.asarray(s[("o1", "o2")]), np.asarray(s[("o2", "o1")]))


@requires_jax
def test_bragg_gradient_matches_fd():
    wl_b = c.bragg_wavelength(0.3229)

    def f(dn):
        return px.power(c.bragg_grating(wl=wl_b + 0.002, dn=dn, n_periods=150, n_sections=3)[("o1", "o1")])

    assert float(px.grad(f)(0.025)) == pytest.approx(float(finite_diff(f, 0.025, 1e-6)), rel=1e-5)


# --------------------------------------------------------------------------- #
# Active devices
# --------------------------------------------------------------------------- #
def _phase(s):
    return np.angle(complex(s[("o1", "o2")]))


def test_heater_phase_is_linear_in_power():
    p0 = _phase(c.heater_phase_shifter(power=0.0))
    for power in (2.5, 7.0, 12.5):
        dphi = (p0 - _phase(c.heater_phase_shifter(power=power, p_pi=25.0))) % (2 * np.pi)
        assert dphi == pytest.approx(np.pi * power / 25.0, abs=1e-12)


def test_depletion_physics_matches_hand_formulas():
    na, nd = 3e17, 8e17
    kT_q = 1.380649e-23 * 300 / 1.602176634e-19
    vbi = kT_q * math.log(na * nd / c.NI_SI_300K**2)
    assert float(c.built_in_voltage(na, nd)) == pytest.approx(vbi, rel=1e-12)
    eps = 11.7 * 8.8541878128e-14
    W = math.sqrt(2 * eps * (vbi + 1.5) / 1.602176634e-19 * (na + nd) / (na * nd)) * 1e4
    x_p, x_n = c.depletion_width(1.5, na, nd)
    assert float(x_p + x_n) == pytest.approx(W, rel=1e-12)
    # charge neutrality: Na x_p = Nd x_n
    assert na * float(x_p) == pytest.approx(nd * float(x_n), rel=1e-12)


def test_pn_phase_shifter_vpil_in_physical_range():
    L = 3000.0  # um
    phi0 = _phase(c.pn_phase_shifter(length=L, voltage=0.0))
    phi2 = _phase(c.pn_phase_shifter(length=L, voltage=2.0))
    dphi = (phi0 - phi2) % (2 * np.pi)
    vpil = 2.0 * np.pi / dphi * (L * 1e-4)  # V*cm
    assert 1.0 < vpil < 4.0
    il0 = -10 * np.log10(float(px.power(c.pn_phase_shifter(length=10000.0, voltage=0.0)[("o1", "o2")])))
    assert 5.0 < il0 < 25.0   # dB/cm for 5e17 cm^-3 doping


@requires_jax
def test_pn_phase_shifter_gradient_wrt_voltage_matches_fd():
    def f(v):
        return xp_real(c.pn_phase_shifter(voltage=v, length=500.0)[("o1", "o2")])

    assert float(px.grad(f)(1.2)) == pytest.approx(float(finite_diff(f, 1.2, 1e-5)), rel=1e-5)


def xp_real(z):
    return px.xp.real(z)


def test_responsivity_and_photocurrent():
    assert float(c.responsivity(1.31, 1.0)) == pytest.approx(1.31 / 1.23984198, rel=1e-6)
    assert c.photocurrent(2e-3, 0.9, 5e-9) == pytest.approx(1.800005e-3, rel=1e-12)


# --------------------------------------------------------------------------- #
# Fiber coupling
# --------------------------------------------------------------------------- #
def test_gaussian_overlap_against_numerical_integral():
    w1, w2, d, theta, wl, n = 1.6, 2.4, 0.7, math.radians(3.0), 1.55, 1.0
    k = 2 * math.pi * n / wl
    x = np.linspace(-15, 15, 20001)
    f = np.exp(-((x - d) ** 2) / w1**2)
    g = np.exp(-(x**2) / w2**2) * np.exp(1j * k * theta * x)
    eta_x = abs(np.trapezoid(f * g.conj(), x)) ** 2 / (np.trapezoid(f**2, x) * np.trapezoid(abs(g) ** 2, x))
    eta_y = 2 * w1 * w2 / (w1**2 + w2**2)
    model = float(c.gaussian_overlap(wl=wl, mfd1=2 * w1, mfd2=2 * w2, offset_x=d, tilt_deg=3.0))
    assert model == pytest.approx(eta_x * eta_y, rel=1e-8)


def test_edge_coupler_facet_is_unitary_when_mode_matched():
    s = c.edge_coupler(n_chip=3.0, n_medium=1.0)
    S, _ = _dense(s)
    assert np.allclose(S.conj().T @ S, np.eye(2), atol=1e-12)
    assert float(px.power(s[("o1", "o2")])) == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# AWG
# --------------------------------------------------------------------------- #
def _awg_spectra(wl, **kw):
    s = c.awg(wl=wl, **kw)
    n = kw.get("n_channels", 8)
    return np.stack([np.asarray(px.power(s[("o1", f"o{k + 2}")])) for k in range(n)])


def test_awg_is_passive_everywhere():
    wl = np.linspace(1.53, 1.57, 4001)
    T = _awg_spectra(wl, n_channels=8, channel_spacing=0.0008, passband_fwhm=0.0006)
    assert T.sum(axis=0).max() <= 1.0 + 1e-12


def test_awg_channel_centres_bandwidth_and_crosstalk():
    spacing, fwhm = 0.0008, 0.0003
    d = c.awg_design(n_channels=8, channel_spacing=spacing, passband_fwhm=fwhm)
    wl = np.linspace(1.545, 1.555, 20001)
    T = _awg_spectra(wl, n_channels=8, channel_spacing=spacing, passband_fwhm=fwhm)
    for k, wl_k in enumerate(d["channel_wavelengths"]):
        band = np.abs(wl - wl_k) < 0.4 * spacing
        peak = int(np.argmax(np.where(band, T[k], 0)))
        assert wl[peak] == pytest.approx(wl_k, abs=0.02 * spacing)
        above = wl[band][T[k][band] >= 0.5 * T[k][peak]]
        assert above[-1] - above[0] == pytest.approx(fwhm, rel=0.1)
        # adjacent-channel crosstalk at this channel's centre
        for j in (k - 1, k + 1):
            if 0 <= j < 8:
                assert T[j][peak] < 1e-3 * T[k][peak]


def test_awg_fsr_periodicity_and_index_shift():
    d = c.awg_design(n_channels=4, channel_spacing=0.0016)
    wl_k = d["channel_wavelengths"][2]
    wl = np.linspace(wl_k - 0.0006, wl_k + 0.0006, 4001)
    base = _awg_spectra(wl, n_channels=4, channel_spacing=0.0016)[2]
    shifted = _awg_spectra(wl, n_channels=4, channel_spacing=0.0016, delta_neff=2e-4)[2]
    expected_shift = 2e-4 * wl_k / 4.2
    measured = wl[np.argmax(shifted)] - wl[np.argmax(base)]
    assert measured == pytest.approx(expected_shift, rel=0.05)
    # next diffraction order: the same port peaks again roughly one FSR away
    wl2 = np.linspace(wl_k + 0.97 * d["fsr"], wl_k + 1.03 * d["fsr"], 6001)
    T2 = _awg_spectra(wl2, n_channels=4, channel_spacing=0.0016)[2]
    assert T2.max() > 0.95
    assert wl2[np.argmax(T2)] - wl_k == pytest.approx(d["fsr"], rel=0.01)


def test_awg_more_arms_lower_crosstalk_floor():
    wl = np.linspace(1.545, 1.555, 4001)
    d = c.awg_design(n_channels=8, channel_spacing=0.0008)
    far = np.abs(wl - d["channel_wavelengths"][3]) > 0.0016
    few = _awg_spectra(wl, n_channels=8, channel_spacing=0.0008, n_arms=21)[3][far].max()
    many = _awg_spectra(wl, n_channels=8, channel_spacing=0.0008, n_arms=41)[3][far].max()
    assert many < few


def test_awg_phase_errors_raise_crosstalk_but_stay_passive():
    wl = np.linspace(1.545, 1.555, 4001)
    d = c.awg_design(n_channels=8, channel_spacing=0.0008)
    far = np.abs(wl - d["channel_wavelengths"][3]) > 0.0016
    ideal = _awg_spectra(wl, n_channels=8, channel_spacing=0.0008)
    noisy = _awg_spectra(wl, n_channels=8, channel_spacing=0.0008, phase_error_std=0.2, seed=4)
    assert noisy[3][far].max() > 100 * ideal[3][far].max()
    assert noisy.sum(axis=0).max() <= 1.0 + 1e-12
    again = _awg_spectra(wl, n_channels=8, channel_spacing=0.0008, phase_error_std=0.2, seed=4)
    assert np.array_equal(noisy, again)          # reproducible


@requires_jax
def test_awg_gradient_wrt_index_deviation():
    wl_k = c.awg_design(n_channels=4, channel_spacing=0.0016)["channel_wavelengths"][1]

    def f(dn):
        return px.power(c.awg(wl=wl_k + 0.0002, n_channels=4, channel_spacing=0.0016, delta_neff=dn)[("o1", "o3")])

    assert float(px.grad(f)(1e-4)) == pytest.approx(float(finite_diff(f, 1e-4, 1e-7)), rel=1e-4)


@requires_jax
def test_new_models_are_jittable():
    import jax

    wl = px.linspace(1.54, 1.56, 11)
    for fn in (c.bragg_grating, c.evanescent_coupler, c.pn_phase_shifter, c.awg, c.edge_coupler):
        out = jax.jit(lambda w, fn=fn: fn(wl=w))(wl)
        assert np.all(np.isfinite(np.asarray(next(iter(out.values())))))
