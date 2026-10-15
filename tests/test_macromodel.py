"""Rational macromodels: vector fitting, delays, realizations, passivity and causality.

Validation anchors (see docs/MACROMODEL.md):
* exact rational data are recovered (poles to 1e-9 relative);
* a straight waveguide is a pure delay ``n_g L / c`` plus a constant;
* a single ring resonance gives one pole whose Q equals the measured Q;
* fitted library spectra (rings, add-drop, Bragg) generalize between samples;
* the Hamiltonian violation bands equal the dense-sweep ones, and enforcement
  is certified passive while moving toward the original passive fit;
* generalized Kramers-Kronig relations pass every library model and fail a
  5 ps time advance;
* the exact time-domain simulator reproduces independent FFT pulse propagation.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax
from scipy.optimize import linear_sum_assignment

import photonix as px
from photonix import macromodel as mm
from photonix.analysis import find_resonances, impulse_response, pulse_response
from photonix.macromodel.passivity import _band_gram, _energy_gram

C_UM = 299792458e6  # um/s


def _dense(wl, sdict):
    return px.SParameterDataset.from_sdict(wl, sdict)


def _random_model(seed=0, n_poles=6, n_ports=3, f0=193.4e12, bandwidth=2e12, symmetric=False):
    rng = np.random.default_rng(seed)
    half = 2 * np.pi * bandwidth / 2
    poles = -rng.uniform(0.02, 0.1, n_poles) * half + 1j * rng.uniform(-0.8, 0.8, n_poles) * half
    res = (rng.normal(size=(n_poles, n_ports, n_ports)) + 1j * rng.normal(size=(n_poles, n_ports, n_ports)))
    res *= 0.05 * half
    d = (rng.normal(size=(n_ports, n_ports)) + 1j * rng.normal(size=(n_ports, n_ports))) * 0.1
    if symmetric:
        res = 0.5 * (res + np.swapaxes(res, 1, 2))
        d = 0.5 * (d + d.T)
    ports = tuple(f"o{i + 1}" for i in range(n_ports))
    return mm.RationalModel(ports, f0, poles, res, d)


def _match(a, b):
    cost = np.abs(a[:, None] - b[None, :])
    rows, cols = linear_sum_assignment(cost)
    return rows, cols, cost[rows, cols]


# --------------------------------------------------------------------------- #
# Vector fitting
# --------------------------------------------------------------------------- #
def test_exact_rational_data_recover_poles_and_residues():
    true = _random_model()
    f = np.linspace(true.carrier_hz - 1e12, true.carrier_hz + 1e12, 400)
    m = mm.vector_fit_frequency(f, true.response(f), carrier_hz=true.carrier_hz, n_poles=6, delays="none")
    rows, cols, dist = _match(true.poles, m.poles)
    assert np.max(dist / np.abs(true.poles[rows])) < 1e-9
    assert np.allclose(m.residues[cols], true.residues[rows], rtol=0, atol=1e-8 * np.abs(true.residues).max())
    assert np.allclose(m.d, true.d, atol=1e-10)
    assert m.metadata["rms_error"] < 1e-12
    # Complex baseband poles are not conjugate pairs: the fit must not force them to be.
    assert not m.is_real_system


def test_auto_order_stops_at_the_exact_order():
    true = _random_model(seed=3)
    f = np.linspace(true.carrier_hz - 1e12, true.carrier_hz + 1e12, 400)
    m = mm.vector_fit_frequency(f, true.response(f), carrier_hz=true.carrier_hz, delays="none")
    assert m.n_poles == 6 and m.metadata["target_met"]
    assert [p for p, _ in m.metadata["order_history"]][-1] == 6


def test_reciprocal_data_give_an_exactly_reciprocal_model():
    true = _random_model(seed=5, symmetric=True)
    f = np.linspace(true.carrier_hz - 1e12, true.carrier_hz + 1e12, 300)
    m = mm.vector_fit_frequency(f, true.response(f), carrier_hz=true.carrier_hz, n_poles=6, delays="none")
    assert m.metadata["symmetric"]
    assert np.array_equal(m.residues, np.swapaxes(m.residues, 1, 2))
    assert np.array_equal(m.d, m.d.T)


def test_straight_is_a_pure_delay_plus_constant():
    wl = np.linspace(1.50, 1.60, 2001)
    m = mm.vector_fit(wl, px.components.straight(wl=wl, length=1000.0, ng=4.2))
    tau = 4.2 * 1000.0 / C_UM  # n_g L / c = 14.0097 ps
    assert m.n_poles == 0
    assert np.allclose(m.delays[[0, 1], [1, 0]], tau, rtol=1e-6, atol=0)
    assert m.metadata["rms_error"] < 1e-10
    # Reflections are structural zeros and stay exactly zero.
    assert m.d[0, 0] == 0 and m.d[1, 1] == 0


def test_one_centimetre_straight_delay_extraction():
    wl = np.linspace(1.54, 1.56, 4001)  # resolves the 140 ps delay (0.55 rad per sample)
    m = mm.vector_fit(wl, px.components.straight(wl=wl, length=10_000.0, ng=4.2))
    tau = 4.2 * 10_000.0 / C_UM
    assert m.n_poles == 0
    assert m.delays[1, 0] == pytest.approx(tau, rel=1e-6)
    assert m.metadata["rms_error"] < 1e-9


def test_rational_model_fixes_the_interpolation_defect():
    """Linear Re/Im interpolation of a delay loses amplitude between samples."""
    wl = np.linspace(1.50, 1.60, 201)  # 0.5 nm steps, 200 um line
    ds = _dense(wl, px.components.straight(wl=wl, length=200.0))
    fine = np.linspace(1.50, 1.60, 2001)
    interpolated = np.abs(ds.interpolate(fine).s[:, 1, 0])
    assert interpolated.min() < 0.9  # the documented defect (|S21| drops to 0.83)
    m = mm.fit_dataset(ds)
    assert np.max(np.abs(np.abs(px.to_numpy(m(wl=fine)[("o1", "o2")])) - 1.0)) < 1e-9
    truth = px.to_numpy(px.components.straight(wl=fine, length=200.0)[("o1", "o2")])
    assert np.max(np.abs(px.to_numpy(m(wl=fine)[("o1", "o2")]) - truth)) < 1e-9


def test_single_ring_resonance_is_one_pole_with_the_measured_q():
    wl = np.linspace(1.5516, 1.5536, 3001)  # one resonance (1.55263 um), +-50 linewidths
    h = px.to_numpy(px.components.all_pass_ring(wl=wl, coupling=0.01)[("o1", "o2")])
    (res,) = find_resonances(wl, np.abs(h) ** 2)
    m = mm.vector_fit(wl, h, n_poles=1, delays="none")
    p = m.poles[0]
    f_res = m.carrier_hz + p.imag / (2 * np.pi)
    q_pole = 2 * np.pi * f_res / (2 * abs(p.real))  # Q_L = omega_0 tau_ph / 2, tau_ph = 1/|Re p|
    assert q_pole == pytest.approx(res.q_loaded, rel=0.01)
    fwhm_hz = C_UM / res.wavelength**2 * res.fwhm
    assert abs(f_res - C_UM / res.wavelength) < fwhm_hz / 100


def test_ring_over_five_fsrs_poles_on_resonances_and_monotone_error():
    wl = np.linspace(1.530, 1.5755, 4001)
    h = px.to_numpy(px.components.all_pass_ring(wl=wl, coupling=0.1)[("o1", "o2")])
    resonances = find_resonances(wl, np.abs(h) ** 2)
    assert len(resonances) == 5
    errors = []
    for order in (5, 6, 7, 8, 10):
        m = mm.vector_fit(wl, h, n_poles=order, delays="none")
        errors.append(m.metadata["rms_error"])
        if order == 8:
            pole_f = m.carrier_hz + m.poles.imag / (2 * np.pi)
            for r in resonances:
                fwhm_hz = C_UM / r.wavelength**2 * r.fwhm
                assert np.min(np.abs(pole_f - C_UM / r.wavelength)) < fwhm_hz / 100
    assert all(b < a for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 1e-6


@pytest.mark.parametrize(
    ("name", "model", "band"),
    [
        ("all_pass_ring", lambda w: px.components.all_pass_ring(wl=w, coupling=0.1), (1.53, 1.57)),
        ("add_drop_ring", lambda w: px.components.add_drop_ring(wl=w), (1.53, 1.57)),
        ("bragg_grating", lambda w: px.components.bragg_grating(wl=w), (1.53, 1.57)),
        ("apodized_bragg", lambda w: px.components.bragg_grating(wl=w, apodization="gaussian", n_sections=24),
         (1.53, 1.57)),
    ],
)
def test_library_spectra_fit_and_generalize_between_samples(name, model, band):
    wl = np.linspace(*band, 2001)
    ds = _dense(wl, model(wl))
    m = mm.fit_dataset(ds)
    assert m.metadata["target_met"], name
    assert m.metadata["relative_rms_error"] < 1e-3
    # Independent check between the fitted samples (4x denser, offset grid).
    check = np.linspace(band[0] + 1e-6, band[1] - 1e-6, 8003)
    truth = _dense(check, model(check)).s
    assert m.rms_error(check, truth) < 1e-3
    assert m.metadata["delays_mode"] == "auto"


def test_entry_delays_are_separate_per_path():
    wl = np.linspace(1.53, 1.57, 2001)
    s = px.components.bragg_grating(wl=wl)
    m = mm.vector_fit(wl, s, delays="entry")
    assert m.metadata["relative_rms_error"] < 1e-3
    assert m.delays[1, 0] > 0 and m.delays[0, 0] == 0  # transmission delayed, reflection starts at t = 0
    assert m.port_delays() is None  # not a reference-plane shift ...
    report = mm.passivity_report(m)
    assert report.method == "sampled" and not report.certified  # ... so passivity is not certified


def test_estimate_delays_is_conservative_for_resonators():
    wl = np.linspace(1.50, 1.60, 2001)
    f = C_UM / wl
    ring = px.to_numpy(px.components.all_pass_ring(wl=wl, coupling=0.1)[("o1", "o2")])
    # The minimum group delay of an all-pass ring is positive, but its impulse response starts at t = 0.
    assert mm.estimate_delays(f, ring)[0, 0] == 0.0
    line = px.to_numpy(px.components.straight(wl=wl, length=500.0)[("o1", "o2")])
    assert mm.estimate_delays(f, line * ring)[0, 0] < 4.2 * 500.0 / C_UM  # never more than the arrival
    assert mm.estimate_delays(f, line)[0, 0] == pytest.approx(4.2 * 500.0 / C_UM, rel=1e-9)


def test_measurement_noise_does_not_block_delay_extraction():
    """Noise-only (structurally zero) entries of measured data have no arrival time."""
    import warnings

    rng = np.random.default_rng(0)
    wl = np.linspace(1.53, 1.57, 2001)
    ds = _dense(wl, px.components.straight(wl=wl, length=1000.0))
    noise = 1e-3 * (rng.normal(size=ds.s.shape) + 1j * rng.normal(size=ds.s.shape)) / np.sqrt(2)
    tau_true = 4.2 * 1000.0 / C_UM
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no spurious aliasing warning from the noise
        tau = mm.estimate_delays(C_UM / wl, ds.s + noise)
    assert tau[0, 0] == 0.0 and tau[1, 1] == 0.0  # noise-like reflections have no arrival
    assert tau[1, 0] == pytest.approx(tau_true, rel=1e-5)  # delay-dominated despite the noise
    with pytest.warns(RuntimeWarning, match="target_rms"):  # 1e-4 is below the 1e-3 noise floor
        m = mm.vector_fit(wl, ds.s + noise, ports=ds.ports)
    assert m.delays[1, 0] == pytest.approx(tau_true, rel=1e-5)  # noisy reflections do not pin the delays
    assert m.n_poles == 0  # the most parsimonious order at the noise floor, not an overfit
    assert m.rms_error(wl, ds.s) < 1e-4  # closer to the truth than the noisy data (1e-3)


def test_noise_floor_order_selection_is_parsimonious():
    rng = np.random.default_rng(1)
    wl = np.linspace(1.53, 1.57, 2001)
    ds = _dense(wl, px.components.add_drop_ring(wl=wl))
    noise = 1e-3 * (rng.normal(size=ds.s.shape) + 1j * rng.normal(size=ds.s.shape)) / np.sqrt(2)
    with pytest.warns(RuntimeWarning, match="stagnated"):
        m = mm.vector_fit(wl, ds.s + noise, ports=ds.ports)
    assert m.n_poles <= 10 and not m.metadata["target_met"]
    assert m.rms_error(wl, ds.s) < 5e-4


def test_undersampled_delay_warns():
    wl = np.linspace(1.50, 1.60, 101)
    line = px.to_numpy(px.components.straight(wl=wl, length=2000.0)[("o1", "o2")])
    with pytest.warns(RuntimeWarning, match="aliased"):
        mm.estimate_delays(C_UM / wl, line)


def test_real_rf_fit_has_conjugate_pairs_and_a_real_realization():
    f = np.linspace(0.5e9, 40e9, 600)
    s = 2j * np.pi * f
    poles = np.array([-1e9 + 2j * np.pi * 10e9, -3e9 + 2j * np.pi * 25e9, -2e10])
    poles = np.concatenate([poles, np.conj(poles[:2])])
    res = np.array([2e9 + 1e9j, 1e9 - 5e8j, 5e9])
    res = np.concatenate([res, np.conj(res[:2])])
    h = (res[None, :] / (s[:, None] - poles[None, :])).sum(axis=1) + 0.2
    h = h * np.exp(-s * 50e-12)  # 50 ps transmission-line delay
    m = mm.vector_fit_frequency(f, h, real=True, n_poles=5, delays=50e-12)  # known delay: exact recovery
    assert m.is_real_system and m.carrier_hz == 0.0
    assert m.n_poles == 5 and m.metadata["rms_error"] < 1e-12
    rows, _, dist = _match(poles, m.poles)
    assert np.max(dist / np.abs(poles[rows])) < 1e-9
    auto = mm.vector_fit_frequency(f, h, real=True)  # estimated delay: conservative, then fitted
    assert auto.is_real_system and auto.metadata["target_met"]
    assert 0.0 <= auto.delays[0, 0] <= 50e-12  # 1/B = 25 ps here, so little can be removed safely
    ss = m.to_statespace(real=True)
    assert ss.A.dtype.kind == "f" and ss.B.dtype.kind == "f" and ss.C.dtype.kind == "f"
    probe = np.array([1e9, 9.7e9, 33e9, 80e9])
    assert np.allclose(ss.response(probe), m.response(probe), rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError):
        mm.vector_fit_frequency(f, h, real=True, carrier_hz=1e9)


# --------------------------------------------------------------------------- #
# Realizations, discretization and time domain
# --------------------------------------------------------------------------- #
def test_statespace_realizations_reproduce_the_model():
    m = _random_model(seed=7, n_ports=2)
    # A rank-one residue needs one state, not n.
    r = m.residues.copy()
    r[0] = np.outer([1.0, 2.0], [0.5, -1.0j]) * 1e11
    m = m.replace(residues=r)
    probe = m.carrier_hz + np.array([-7e11, 0.0, 1.3e11, 9e11])
    full = m.to_statespace(minimal=False)
    minimal = m.to_statespace()
    assert full.n_states == 2 * m.n_poles
    assert minimal.n_states == 2 * m.n_poles - 1
    for ss in (full, minimal):
        assert np.allclose(ss.response(probe), m.response(probe), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        m.to_statespace(real=True)  # not a real system


def test_zoh_discretization_is_exact_for_a_step():
    p = np.array([-2e11 + 5e11j, -1e11 - 3e11j])
    r = np.array([[[1e11 + 2e10j]], [[3e10 - 1e10j]]])
    m = mm.RationalModel(("o1",), 0.0, p, r, np.array([[0.1]]))
    dt = 1e-13
    k = np.arange(400)
    y = m.discretize(dt, "zoh").simulate(np.ones(k.size))[:, 0]
    t = k * dt
    exact = 0.1 + sum(r[i, 0, 0] / p[i] * (np.exp(p[i] * t) - 1) for i in range(2))
    assert np.allclose(y, exact, rtol=0, atol=1e-12)


def test_simulate_equals_discretized_foh_and_shifts_delays_exactly():
    m = _random_model(seed=11, n_ports=2, f0=0.0)
    dt = 2e-14
    t = np.arange(3000) * dt
    u = np.stack([np.exp(-((t - 5e-12) / 1e-12) ** 2), np.sin(2e12 * t) * (t < 3e-11)], axis=1)
    y_core = m.simulate(u, dt)
    assert np.allclose(m.discretize(dt, "foh").simulate(u), y_core, atol=1e-12)
    shift = 37
    delayed = m.replace(delays=np.full((2, 2), shift * dt))
    y_del = delayed.simulate(u, dt)
    assert np.allclose(y_del[shift:], y_core[:-shift], atol=1e-12)
    assert np.allclose(y_del[:shift], 0.0, atol=1e-15)


def test_impulse_response_matches_the_pole_residue_form():
    m = _random_model(seed=13, n_ports=1, f0=0.0).replace(delays=np.full((1, 1), 3e-12))
    t = np.linspace(0, 20e-12, 9)
    h = m.impulse_response(t)[:, 0, 0]
    shifted = t - 3e-12
    expected = np.where(shifted >= 0, (m.residues[:, 0, 0][None] * np.exp(np.outer(np.maximum(shifted, 0),
                                                                                 m.poles))).sum(1), 0)
    assert np.allclose(h, expected)
    assert np.all(h[t < 3e-12] == 0)


def test_simulated_pulse_matches_independent_fft_propagation():
    """State-space time stepping vs. analysis.pulse_response on the sampled data."""
    wl = np.linspace(1.53, 1.57, 4001)
    h = (px.to_numpy(px.components.straight(wl=wl, length=2000.0)[("o1", "o2")])
         * px.to_numpy(px.components.all_pass_ring(wl=wl, coupling=0.2)[("o1", "o2")]))
    S = np.zeros((wl.size, 2, 2), complex)
    S[:, 1, 0] = S[:, 0, 1] = h
    m = mm.vector_fit(wl, S)
    assert m.delays[1, 0] > 25e-12  # most of the 28 ps line delay is extracted
    fwhm = 5.0
    ref = pulse_response(wl, h, pulse_fwhm_ps=fwhm, center_wl=m.carrier_wl, n_samples=8192)
    dt = (ref.t_ps[1] - ref.t_ps[0]) * 1e-12
    e_in = np.exp(-2 * np.log(2) / fwhm**2 * (ref.t_ps - 4 * fwhm) ** 2)
    y = m.simulate(np.stack([e_in, np.zeros_like(e_in)], axis=1), dt)
    p_out = np.abs(y[:, 1]) ** 2
    assert np.max(np.abs(p_out - ref.output_power)) < 1e-3 * ref.output_power.max()


def test_discretized_impulse_response_matches_fft_impulse_response():
    """The FOH-discretized state space, driven by the band-limited (Hann-windowed)
    impulse, reproduces analysis.impulse_response of the sampled data."""
    wl = np.linspace(1.53, 1.57, 2001)
    ds = _dense(wl, px.components.add_drop_ring(wl=wl))
    m = mm.fit_dataset(ds)
    f = C_UM / wl
    f0, f1 = float(f.min()), float(f.max())
    bw, n, sub = f1 - f0, wl.size, 32
    for out in (1, 2):  # through and drop
        t_ps, h = impulse_response(wl, ds.s[:, out, 0], oversample=4)
        dt = (t_ps[1] - t_ps[0]) * 1e-12 / sub
        lead = round(40 / bw / dt)  # the window kernel is non-causal: launch it 40 / B early
        tt = (np.arange(lead + round(100e-12 / dt)) - lead) * dt
        # Inverse Fourier transform of the Hann window over the band (three sincs), about the carrier.
        kernel = np.exp(2j * np.pi * (0.5 * (f0 + f1) - m.carrier_hz) * tt) * bw * (
            0.5 * np.sinc(bw * tt) + 0.25 * (np.sinc(bw * tt + 1) + np.sinc(bw * tt - 1)))
        u = np.zeros((tt.size, 4), complex)
        u[:, 0] = kernel
        # analysis.impulse_response is (1/n) sum_k H_k w_k exp(...) = integral / (n df).
        sim = np.abs(m.discretize(dt, "foh").simulate(u)[lead::sub, out]) / (n * bw / (n - 1))
        ref = np.abs(h[:sim.size])
        # The same transform of the model's own response on an exactly uniform grid isolates the
        # time stepping (window-truncation and input-interpolation error only) ...
        spectrum = np.zeros(4 * n, complex)
        spectrum[:n] = m.response(np.linspace(f0, f1, n))[:, out, 0] * np.hanning(n)
        own = np.abs(np.fft.ifft(spectrum) * 4)[:sim.size]
        assert np.max(np.abs(sim - own)) < 1e-4 * own.max()
        # ... and the remaining difference is analysis's resampling of the wavelength sweep (5e-4).
        assert np.max(np.abs(sim - ref)) < 1e-3 * ref.max()


# --------------------------------------------------------------------------- #
# Passivity
# --------------------------------------------------------------------------- #
def test_band_gram_matrix_is_exact():
    a = np.array([-0.05 + 0.3j, -0.2 - 0.5j, -0.01 + 0.0j])
    G, h = _band_gram(a, -0.7, 1.1)
    x = np.linspace(-0.7, 1.1, 400_001)
    phi = 1 / (1j * x[:, None] - a[None, :])
    w = np.full(x.size, 1.0)
    w[[0, -1]] = 0.5
    G_num = (phi.conj().T * w) @ phi / w.sum()
    assert np.allclose(G, G_num, rtol=1e-6, atol=1e-6)
    assert np.allclose(h, (phi.conj() * w[:, None]).sum(0) / w.sum(), rtol=1e-6, atol=1e-6)


def test_hamiltonian_crossings_match_a_dense_sweep():
    m = _random_model(seed=2, n_ports=2)
    m = m.replace(residues=m.residues * 3.0, metadata={"fit_band_hz": [m.carrier_hz - 1e12, m.carrier_hz + 1e12]})
    report = mm.passivity_report(m)
    assert not report.passive and report.certified
    f = np.linspace(m.carrier_hz - 3e12, m.carrier_hz + 3e12, 200_001)
    sv = np.linalg.svd(m.response(f), compute_uv=False)[:, 0]
    edges = f[np.flatnonzero(np.diff((sv > 1).astype(int)))]
    reported = np.sort(report.crossing_freqs_hz)
    reported = reported[(reported > f[0]) & (reported < f[-1])]
    assert reported.size == edges.size
    assert np.max(np.abs(reported - edges)) < 2 * (f[1] - f[0])
    assert report.max_sigma == pytest.approx(sv.max(), rel=1e-6)


def test_passivity_enforcement_of_a_perturbed_add_drop_fit():
    wl = np.linspace(1.53, 1.57, 2001)
    ds = _dense(wl, px.components.add_drop_ring(wl=wl))
    fit = mm.fit_dataset(ds)
    assert mm.passivity_report(fit).passive
    bad = fit.replace(residues=fit.residues * 1.02)
    report = mm.passivity_report(bad)
    assert not report.passive and report.certified
    f0, f1 = fit.metadata["fit_band_hz"]
    f = np.linspace(f0, f1, 100_001)
    sv = np.linalg.svd(bad.response(f), compute_uv=False)[:, 0]
    inside = np.zeros(f.size, bool)
    for lo, hi in report.violation_bands_hz:
        inside |= (f >= lo) & (f <= hi)
    assert np.array_equal(sv > 1, inside)  # reported bands == dense-grid violations

    fixed = mm.enforce_passivity(bad)
    info = fixed.metadata["passivity_enforcement"]
    assert info["passive"] and info["certified"]
    dense = np.linspace(f0, f1, 20 * wl.size)
    assert np.linalg.svd(fixed.response(dense), compute_uv=False)[:, 0].max() <= 1.0
    ev = np.linalg.eigvals(mm.hamiltonian_matrix(fixed))
    assert np.min(np.abs(ev.real) / np.abs(ev)) > 1e-6  # no imaginary Hamiltonian eigenvalues
    assert mm.passivity_report(fixed).passive
    # Minimal change: within 3x the pointwise lower bound ||dS||_F >= sigma_max - 1 ...
    Hb, Hf, H0 = bad.response(dense), fixed.response(dense), fit.response(dense)
    sv_bad = np.linalg.svd(Hb, compute_uv=False)[:, 0]
    lower = np.sqrt(np.mean(np.maximum(sv_bad - 1, 0) ** 2))
    change = np.sqrt(np.mean(np.sum(np.abs(Hf - Hb) ** 2, axis=(1, 2))))
    assert lower < change < 3 * lower
    # ... and, as a projection onto the convex passive set, closer to the passive original fit.
    assert np.linalg.norm(Hf - H0) < np.linalg.norm(Hb - H0)
    # Reciprocity and structural zeros (o1 -> o4 and o2 -> o3 vanish) are preserved.
    assert np.allclose(fixed.residues, np.swapaxes(fixed.residues, 1, 2), atol=0)
    assert np.all(fixed.residues[:, 3, 0] == 0) and np.all(fixed.residues[:, 2, 1] == 0)


def test_lossless_fit_with_guard_bands_is_enforced_without_in_band_damage():
    """A lossless device's fit violates passivity just outside its band (the
    rational function extrapolates); fitting guard bands and minimizing the
    perturbation over the band of interest keeps the in-band error tiny."""
    interest = (1.53, 1.57)
    wl = np.linspace(interest[0] - 0.01, interest[1] + 0.01, 2501)
    fit = mm.fit_dataset(_dense(wl, px.components.bragg_grating(wl=wl)))
    assert not mm.passivity_report(fit).passive
    fixed = mm.enforce_passivity(fit, band_hz=(C_UM / interest[1], C_UM / interest[0]))
    assert mm.passivity_report(fixed).passive
    check = np.linspace(*interest, 4001)
    truth = _dense(check, px.components.bragg_grating(wl=check)).s
    assert fixed.rms_error(check, truth) < 1e-5


def test_enforcement_keeps_a_real_system_real():
    poles = np.array([-1e9 + 6e10j, -1e9 - 6e10j, -5e9])
    res = np.array([[[1.4e9 + 2e8j]], [[1.4e9 - 2e8j]], [[1e9]]])
    m = mm.RationalModel(("o1",), 0.0, poles, res, np.array([[0.05]]), metadata={"fit_band_hz": [1e9, 2e10]})
    assert not mm.passivity_report(m).passive
    fixed = mm.enforce_passivity(m)
    assert fixed.is_real_system
    assert mm.passivity_report(fixed).passive
    assert np.allclose(fixed.to_statespace(real=True).response([5e9]), fixed.response([5e9]))


def test_enforcement_clips_an_active_direct_term():
    m = mm.RationalModel(("o1", "o2"), 193e12, np.zeros(0), np.zeros((0, 2, 2)),
                         np.array([[0.0, 1.01], [1.01, 0.0]]))
    report = mm.passivity_report(m)
    assert not report.passive and report.violation_bands_hz == ((-np.inf, np.inf),)
    fixed = mm.enforce_passivity(m, margin=1e-6)
    assert np.linalg.svd(fixed.d, compute_uv=False).max() == pytest.approx(1 - 1e-6)
    assert mm.passivity_report(fixed).passive


def _quiet_fit(wl, sdict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mm.vector_fit(wl, {k: px.to_numpy(v) for k, v in sdict.items()})


def test_damped_enforcement_converges_where_the_full_step_diverged():
    """Regression: the undamped first-order step overshot and diverged (max_sigma 1.1e4 after 50 iterations
    for residues x 1.05, 5.5e5 for a 1 % gain on the data), and the diverged iterate was returned."""
    wl = np.linspace(1.54, 1.56, 801)
    ring = px.components.add_drop_ring(wl=wl, coupling1=0.1, coupling2=0.1)
    fit = _quiet_fit(wl, ring)
    bad = fit.replace(residues=fit.residues * 1.05)
    assert mm.passivity_report(bad).max_sigma > 1.07
    fixed = mm.enforce_passivity(bad)
    info = fixed.metadata["passivity_enforcement"]
    report = mm.passivity_report(fixed)
    assert info["passive"] and info["certified"] and report.passive and report.certified
    assert report.max_sigma <= 1.0 and info["relative_rms_change"] < 0.02
    # A 1 % gain on the data: sigma_max(D) > 1 is clipped first, and the 4-port's degenerate singular-value
    # pairs need the cluster (off-diagonal) constraints for the step to be a descent direction.
    gain = _quiet_fit(wl, {k: 1.01 * px.to_numpy(v) for k, v in ring.items()})
    assert mm.passivity_report(gain).sigma_max_d > 1.0
    fixed = mm.enforce_passivity(gain)
    assert fixed.metadata["passivity_enforcement"]["passive"] and mm.passivity_report(fixed).passive
    assert fixed.metadata["passivity_enforcement"]["relative_rms_change"] < 0.1


def test_failed_enforcement_never_returns_a_worse_model():
    wl = np.linspace(1.54, 1.56, 801)
    ring = px.components.add_drop_ring(wl=wl, coupling1=0.1, coupling2=0.1)
    bad = _quiet_fit(wl, ring)
    bad = bad.replace(residues=bad.residues * 1.05)
    before = mm.passivity_report(bad).max_sigma
    with pytest.warns(RuntimeWarning, match="least-violating"):
        partial = mm.enforce_passivity(bad, max_iter=1)
    info = partial.metadata["passivity_enforcement"]
    assert not info["passive"] and not info["returned_input"]
    assert mm.passivity_report(partial).max_sigma == pytest.approx(info["max_sigma"])
    assert info["max_sigma"] < before
    # Clipping sigma_max(D) of a 1 % gain fit raises the peak; with no iterations left the input is returned.
    gain = _quiet_fit(wl, {k: 1.01 * px.to_numpy(v) for k, v in ring.items()})
    with pytest.warns(RuntimeWarning, match="input model unchanged"):
        same = mm.enforce_passivity(gain, max_iter=0)
    assert same.metadata["passivity_enforcement"]["returned_input"]
    assert np.array_equal(same.residues, gain.residues) and np.array_equal(same.d, gain.d)


def _rf_two_port(scale):
    rng = np.random.default_rng(3)
    pp = -rng.uniform(0.2e9, 1e9, 3) + 2j * np.pi * rng.uniform(2e9, 15e9, 3)
    poles = np.concatenate([pp, pp.conj(), [-3e9 + 0j]])
    r = (rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))) * 1.5e9
    r = r + np.swapaxes(r, 1, 2)
    rr = rng.normal(size=(1, 2, 2))
    rr = (rr + np.swapaxes(rr, 1, 2)) * 2e9
    res = np.concatenate([r, r.conj(), rr.astype(complex)]) * scale
    d = np.array([[0.35, 0.4], [0.4, 0.1]])
    return mm.RationalModel(("o1", "o2"), 0.0, poles, res, d, metadata={"fit_band_hz": [1e9, 20e9]})


def test_real_system_energy_covers_the_mirror_band():
    """Regression: with the energy Gram on the positive band only, the poles with Im < 0 were nearly free
    and enforcement of a mildly active RF two-port diverged (max_sigma 7.8e72)."""
    a = np.array([-0.05 + 0.3j, -0.05 - 0.3j, -0.2 + 0j])
    G, h = _energy_gram(a, 0.1, 0.9, mirrored=True)
    G1, h1 = _band_gram(a, 0.1, 0.9)
    G2, h2 = _band_gram(a, -0.9, -0.1)
    assert np.allclose(G, 0.5 * (G1 + G2)) and np.allclose(h, 0.5 * (h1 + h2))
    assert np.allclose(_energy_gram(a, -0.2, 0.9, mirrored=True)[0], _band_gram(a, -0.9, 0.9)[0])
    for scale in (0.021, 0.03):
        m = _rf_two_port(scale)
        assert not mm.passivity_report(m).passive
        fixed = mm.enforce_passivity(m)
        info = fixed.metadata["passivity_enforcement"]
        assert info["passive"] and mm.passivity_report(fixed).passive
        assert fixed.is_real_system and info["relative_rms_change"] < 0.05


def test_direct_term_clipping_keeps_a_real_system_real():
    """Regression: the D-clipping compensation was not conjugate-symmetrized, so the output was no longer a
    real system and ``to_statespace(real=True)`` raised."""
    p = -1e9 + 2j * np.pi * 8e9
    m = mm.RationalModel(("o1",), 0.0, np.array([p, np.conj(p)]),
                         np.array([[[-0.3e9 + 0.1e9j]], [[-0.3e9 - 0.1e9j]]]), np.array([[1.02]]),
                         metadata={"fit_band_hz": [1e9, 20e9]})
    assert m.is_real_system
    fixed = mm.enforce_passivity(m)
    assert fixed.metadata["passivity_enforcement"]["d_clipped"]
    assert fixed.is_real_system and mm.passivity_report(fixed).passive
    f = np.linspace(1e9, 20e9, 7)
    assert np.allclose(fixed.response(-f), np.conj(fixed.response(f)))
    assert np.allclose(fixed.to_statespace(real=True).response(f), fixed.response(f))


def test_nearly_unitary_direct_term_is_not_certified_falsely():
    """Regression: with sigma_max(D) = 1 + 5e-12 (a lossless all-pass fit) the ill-conditioned Hamiltonian
    missed the crossings and the model was certified passive although sigma_max - 1 = 1.3e-8 > tol."""
    # Two all-pass sections -(s + conj p) / (s - p) = -1 - 2 Re(p) / (s - p), |S| = 1 on the axis, with
    # sigma_max(D) = 1 + 5e-12 and residues x (1 + 1e-8): sigma_max = 1 + 2e-8 at the resonances.
    p = np.array([-0.02 + 0.3j, -0.05 - 0.4j])
    res = np.zeros((2, 2, 2), complex)
    res[0, 0, 0], res[1, 1, 1] = -2 * p.real * (1 + 1e-8)
    m = mm.RationalModel(("o1", "o2"), 0.0, p, res, -np.eye(2) * (1 + 5e-12) + 0j,
                         metadata={"fit_band_hz": [-0.1, 0.1]})
    report = mm.passivity_report(m)
    assert not report.passive and not report.certified
    assert report.max_sigma == pytest.approx(1 + 2e-8, abs=1e-10)
    x = np.linspace(-1, 1, 20001) / (2 * np.pi)
    over = np.linalg.svd(m.response(x), compute_uv=False)[:, 0] > 1 + 1e-9
    inside = np.zeros(x.size, bool)
    for lo, hi in report.violation_bands_hz:
        inside |= (x >= lo) & (x <= hi)
    assert over.any() and np.all(inside[over])  # every swept violation lies in a reported band
    # A real lossless fit (D unitary to ~1e-11): the verdict agrees with max_sigma but is not certified,
    # and enforcement (which clips D to 1 - margin) yields a certified passive model at negligible cost.
    wl = np.linspace(1.54, 1.56, 801)
    fit = _quiet_fit(wl, px.components.all_pass_ring(wl=wl, coupling=0.1, loss_db_cm=0.0))
    report = mm.passivity_report(fit)
    assert abs(report.sigma_max_d - 1) < 1e-9 and not report.certified
    assert report.passive == (report.max_sigma <= 1 + 1e-9)
    fixed = mm.enforce_passivity(fit)
    report = mm.passivity_report(fixed)
    assert report.passive and report.certified  # sigma_max(D) is clipped to 1 - margin
    assert fixed.metadata["passivity_enforcement"]["relative_rms_change"] < 1e-5


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #
_LIBRARY = {
    "straight": lambda w: px.components.straight(wl=w),
    "long_straight": lambda w: px.components.straight(wl=w, length=200.0),
    "all_pass_ring": lambda w: px.components.all_pass_ring(wl=w, coupling=0.2),
    "add_drop_ring": lambda w: px.components.add_drop_ring(wl=w, coupling1=0.2, coupling2=0.2),
    "bragg_grating": lambda w: px.components.bragg_grating(wl=w),
    "directional_coupler": lambda w: px.components.directional_coupler(wl=w),
    "mmi2x2": lambda w: px.components.mmi2x2(wl=w),
    "mzi": lambda w: px.components.mzi(wl=w),
}


@pytest.mark.parametrize("name", sorted(_LIBRARY))
def test_library_models_are_causal_and_time_advance_is_detected(name):
    wl = np.linspace(1.50, 1.60, 1201)
    ds = _dense(wl, _LIBRARY[name](wl))
    assert mm.causality_report(wl, ds.s, ports=ds.ports).causal, name
    f = C_UM / wl
    advance = np.exp(2j * np.pi * (f - 0.5 * (f.min() + f.max())) * 5e-12)  # S * exp(+j Omega 5 ps)
    report = mm.causality_report(wl, ds.s * advance[:, None, None], ports=ds.ports)
    assert not report.causal, name
    assert report.worst_entry is not None and report.worst_band_hz is not None


def test_causality_of_analytic_poles():
    wl = np.linspace(1.50, 1.60, 1001)
    f = C_UM / wl
    half = np.pi * (f.max() - f.min())
    omega = 2 * np.pi * (f - 0.5 * (f.max() + f.min()))
    a = 0.05 * half
    causal = a / (1j * (omega - 0.2 * half) + a)  # left-half-plane pole
    anti = a / (-1j * (omega - 0.2 * half) + a)  # right-half-plane pole
    good = mm.causality_report(wl, causal)
    assert good.causal and np.all(good.violation_per_entry == 0)
    interior = np.isfinite(good.bound[:, 0, 0])
    assert np.median(good.bound[interior, 0, 0]) < 0.05  # the test is sensitive in the interior
    bad = mm.causality_report(wl, anti)
    assert not bad.causal
    lo, hi = bad.worst_band_hz
    assert lo <= 0.5 * (f.max() + f.min()) + 0.2 * half / (2 * np.pi) + 2e11 and hi > lo


@pytest.mark.parametrize("sigma", [1e-4, 1e-3, 1e-2])
def test_causality_tolerates_measurement_noise(sigma):
    """Noise is not Kramers-Kronig consistent; its propagated bound prevents false alarms."""
    rng = np.random.default_rng(3)
    wl = np.linspace(1.50, 1.60, 1201)
    ds = _dense(wl, px.components.add_drop_ring(wl=wl, coupling1=0.2, coupling2=0.2))
    noisy = ds.s + sigma * (rng.normal(size=ds.s.shape) + 1j * rng.normal(size=ds.s.shape)) / np.sqrt(2)
    assert mm.causality_report(wl, noisy, ports=ds.ports).causal  # noise level estimated from the data
    assert mm.causality_report(wl, noisy, ports=ds.ports, noise_std=sigma).causal
    assert not mm.causality_report(wl, noisy, ports=ds.ports, noise_std=0.0).causal  # noise alone looks acausal
    f = C_UM / wl
    advance = np.exp(2j * np.pi * (f - 0.5 * (f.min() + f.max())) * 5e-12)
    assert not mm.causality_report(wl, noisy * advance[:, None, None], ports=ds.ports).causal


def test_causality_input_validation():
    wl = np.linspace(1.5, 1.6, 101)
    with pytest.raises(ValueError):
        mm.causality_report(wl, np.ones(101), n_subtractions=0)
    with pytest.raises(ValueError):
        mm.causality_report(wl[:9], np.ones(9))


# --------------------------------------------------------------------------- #
# Model contract: SDict, circuits, gradients, I/O
# --------------------------------------------------------------------------- #
def test_model_is_a_photonix_component_and_composes_in_circuits():
    wl = np.linspace(1.545, 1.555, 801)
    ring = mm.vector_fit(wl, px.components.all_pass_ring(wl=wl, coupling=0.1))
    s = ring(wl=wl)
    px.core.validate_sdict(s)
    assert set(s) == {(a, b) for a in ("o1", "o2") for b in ("o1", "o2")}
    assert ring()  # default wl works
    assert {ring: 1}[ring] == 1 and ring != ring.replace()  # hashable, identity semantics
    netlist = px.circuit.Netlist(
        instances={"r1": "fitted", "w": "straight", "r2": "fitted"},
        connections={("r1", "o2"): ("w", "o1"), ("w", "o2"): ("r2", "o1")},
        ports={"in": ("r1", "o1"), "out": ("r2", "o2")},
    )
    models = {**px.components.MODELS, "fitted": ring}
    cascade = px.circuit.circuit_from_netlist(netlist, models)(wl=wl)[("in", "out")]
    true_ring = px.to_numpy(px.components.all_pass_ring(wl=wl, coupling=0.1)[("o1", "o2")])
    line = px.to_numpy(px.components.straight(wl=wl)[("o1", "o2")])
    assert np.max(np.abs(px.to_numpy(cascade) - true_ring**2 * line)) < 1e-3


@requires_jax
def test_gradient_with_respect_to_residues_matches_finite_differences():
    import jax.numpy as jnp

    wl = np.linspace(1.545, 1.555, 401)
    m = mm.vector_fit(wl, px.components.add_drop_ring(wl=wl))
    probe = jnp.linspace(1.548, 1.552, 21)
    r0 = jnp.asarray(m.residues)

    def drop_power(scale):
        s = m(wl=probe, residues=r0 * scale)
        return jnp.sum(jnp.abs(s[("o1", "o3")]) ** 2)

    g = float(px.grad(drop_power)(1.0))
    assert g == pytest.approx(float(finite_diff(drop_power, 1.0, h=1e-6)), rel=1e-6)
    full = px.grad(lambda r: jnp.sum(jnp.abs(m(wl=probe, residues=r)[("o1", "o3")]) ** 2))(r0)
    assert full.shape == m.residues.shape and bool(jnp.all(jnp.isfinite(full)))
    assert float(px.jit(lambda w: jnp.abs(m(wl=w)[("o1", "o2")]))(1.55)) == pytest.approx(
        float(np.abs(m.response([C_UM / 1.55])[0, 1, 0])), rel=1e-12)


def test_model_runs_under_numpy_backend():
    code = (
        "import numpy as np, photonix as px\n"
        "from photonix import macromodel as mm\n"
        "assert px.backend_name() == 'numpy'\n"
        "wl = np.linspace(1.54, 1.56, 401)\n"
        "m = mm.vector_fit(wl, px.components.all_pass_ring(wl=wl, coupling=0.2))\n"
        "s = m(wl=wl)\n"
        "assert np.max(np.abs(s[('o1', 'o2')] - m.response(299792458e6 / wl)[:, 1, 0])) < 1e-12\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PHOTONIX_BACKEND": "numpy"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_npz_round_trip_and_versioning(tmp_path):
    m = _random_model(seed=4).replace(delays=np.full((3, 3), 1e-12), metadata={"note": "x"})
    path = tmp_path / "model.npz"
    m.save_npz(path)
    back = mm.RationalModel.load_npz(path)
    for name in ("poles", "residues", "d", "delays"):
        assert np.array_equal(getattr(back, name), getattr(m, name))
    assert back.ports == m.ports and back.carrier_hz == m.carrier_hz and back.metadata == m.metadata
    with np.load(path) as data:
        content = dict(data)
    content["format_version"] = np.asarray(99)
    np.savez(tmp_path / "future.npz", **content)
    with pytest.raises(ValueError, match="version"):
        mm.RationalModel.load_npz(tmp_path / "future.npz")


def test_touchstone_round_trip_then_fit(tmp_path):
    wl = np.linspace(1.53, 1.57, 1001)
    ds = _dense(wl, px.components.add_drop_ring(wl=wl))
    ds.save_touchstone(tmp_path / "ring.s4p")
    loaded = px.SParameterDataset.load_touchstone(tmp_path / "ring.s4p")
    m = loaded.to_rational()
    assert m.ports == ds.ports
    assert m.rms_error(wl, ds.s) < 1e-4 * np.abs(ds.s).max()
    back = m.to_dataset(wl)
    assert np.max(np.abs(back.s - ds.s)) < 1e-3


def test_rational_model_validation():
    p = np.array([-1.0 + 0j])
    r = np.ones((1, 1, 1), complex)
    with pytest.raises(ValueError, match="stable"):
        mm.RationalModel(("o1",), 0.0, np.array([1.0 + 0j]), r, np.zeros((1, 1)))
    with pytest.raises(ValueError, match="shape"):
        mm.RationalModel(("o1",), 0.0, p, np.ones((2, 1, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        mm.RationalModel(("o1",), 0.0, p, r, np.zeros((1, 1)), delays=-np.ones((1, 1)))
    with pytest.raises(ValueError, match="ports"):
        mm.RationalModel(("o1", "o1"), 0.0, np.zeros(0), np.zeros((0, 2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="JSON"):
        mm.RationalModel(("o1",), 0.0, p, r, np.zeros((1, 1)), metadata={"x": object()})
    with pytest.raises(ValueError):
        mm.vector_fit(np.linspace(1.5, 1.6, 10), np.ones(10), n_poles=50)
    with pytest.raises(ValueError):
        mm.vector_fit(np.linspace(1.5, 1.6, 10), np.ones(10), weighting="bogus")
    with pytest.raises(ValueError, match="reciprocal"):
        mm.vector_fit(np.linspace(1.5, 1.6, 20), np.tile([[0, 1], [0.5, 0]], (20, 1, 1)), symmetric=True)


def test_inverse_magnitude_weighting_balances_relative_error():
    wl = np.linspace(1.53, 1.57, 2001)
    ds = _dense(wl, px.components.add_drop_ring(wl=wl, loss_db_cm=20.0))
    m = mm.fit_dataset(ds, weighting="inverse_magnitude", n_poles=12)
    drop = ds.s[:, 2, 0]
    err = np.abs(m.response(C_UM / wl)[:, 2, 0] - drop) / np.abs(drop)
    assert np.sqrt(np.mean(err**2)) < 1e-3
