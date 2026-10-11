"""Transceiver and signal-integrity library: validation against closed forms.

PRBS structure (ITU-T O.150), laser rate-equation threshold / slope /
relaxation oscillation / Henry linewidth, CW-laser linewidth and RIN,
shot / thermal noise variances, the 0.35 / f3dB rise-time rule, MZM transfer
and traveling-wave electrode, ring-modulator CMT against the analytic ring and
the Sacher-Poon small-signal response, eye height / ER / ISI closure,
the thermal-noise-limited sensitivity and the Gaussian BER law.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import finite_diff, requires_jax
from scipy.stats import chi2

import photonix as px
from photonix.core.constants import C0_UM_S, Q_E
from photonix.timedomain import (
    K_BOLTZMANN,
    AnalogFilter,
    LaserParams,
    RingModulator,
    TravelingWaveElectrode,
    Waveform,
    ber_count,
    ber_from_q,
    cw,
    cw_laser,
    drive_waveform,
    eam,
    eye_diagram,
    mzm,
    photodiode,
    prbs,
    push_pull,
    q_from_ber,
    rate_equation_laser,
    receiver_sensitivity,
    recover_clock,
    symbols_to_levels,
    tia,
)


def _chi2_ok(samples, sigma2, confidence=0.95):
    """``sigma2`` lies inside the chi-square confidence interval of the sample variance."""
    x = np.asarray(samples)
    n = x.size
    s2 = np.var(x, ddof=1)
    lo = (n - 1) * s2 / chi2.ppf(0.5 + confidence / 2, n - 1)
    hi = (n - 1) * s2 / chi2.ppf(0.5 - confidence / 2, n - 1)
    return lo <= sigma2 <= hi, (lo / sigma2, hi / sigma2)


# --------------------------------------------------------------------------- #
# (1) PRBS
# --------------------------------------------------------------------------- #
def _naive_lfsr(n, m, n_bits, seed_bits):
    reg = list(seed_bits)
    out = list(seed_bits)
    for k in range(n, n_bits):
        out.append(out[k - m] ^ out[k - n])
    del reg
    return np.array(out[:n_bits], dtype=np.uint8)


@pytest.mark.parametrize("order", [7, 9, 11, 15])
def test_prbs_maximal_length_properties(order):
    period = 2 ** order - 1
    s = prbs(order, 2 * period)
    assert np.array_equal(s[:period], s[period:])                     # period divides 2^n - 1 ...
    for p in {d for d in range(1, period) if period % d == 0}:        # ... and is no shorter
        assert not np.array_equal(s[:period - p], s[p:period])
    one = s[:period]
    assert int(one.sum()) == 2 ** (order - 1)                         # balance
    # run lengths over one (cyclic) period: 2^(n-k-1) runs of length k < n - 1
    rot = np.roll(one, -int(np.argmax(one != one[-1])))               # start at a run boundary
    edges = np.flatnonzero(np.diff(rot.astype(int)) != 0) + 1
    runs = np.diff(np.r_[0, edges, period])
    counts = np.bincount(runs, minlength=order + 1)
    for k in range(1, order - 1):
        assert counts[k] == 2 ** (order - k - 1), (k, counts[k])
    assert counts[order - 1] == 1 and counts[order] == 1
    # two-valued periodic autocorrelation
    x = 1.0 - 2.0 * one
    ac = np.real(np.fft.ifft(np.abs(np.fft.fft(x)) ** 2)) / period
    assert ac[0] == pytest.approx(1.0)
    assert np.allclose(ac[1:], -1.0 / period, atol=1e-12)


@pytest.mark.parametrize("order", [23, 31])
def test_long_prbs_obeys_the_generator_recurrence(order):
    m = px.timedomain.PRBS_POLYNOMIALS[order]
    s = prbs(order, 400000, seed=0x5A5A5)
    k = np.arange(order, s.size)
    assert np.array_equal(s[k], s[k - m] ^ s[k - order])               # every bit, incl. the doubled blocks
    ref = _naive_lfsr(order, m, 5000, [(0x5A5A5 >> i) & 1 for i in range(order)])
    assert np.array_equal(s[:5000], ref)
    if order == 23:
        full = prbs(23)
        assert full.size == 2 ** 23 - 1 and int(full.sum()) == 2 ** 22


def test_symbol_mapping():
    lv = symbols_to_levels([0, 0, 0, 1, 1, 1, 1, 0], "pam4")
    assert np.allclose(lv * 3, [0, 1, 2, 3])                           # Gray: 00 01 11 10
    lv = symbols_to_levels([0, 0, 0, 1, 1, 0, 1, 1], "pam4", gray=False)
    assert np.allclose(lv * 3, [0, 1, 2, 3])
    with pytest.raises(ValueError):
        symbols_to_levels([0, 1, 1], "pam4")


# --------------------------------------------------------------------------- #
# (2) Lasers
# --------------------------------------------------------------------------- #
def test_rate_equation_threshold_and_slope_efficiency():
    p = LaserParams(epsilon_cm3=0.0)          # beta_sp = 1e-5 (default): realistic, nearly ideal threshold
    ith = p.threshold_current_a
    q_v = Q_E * p.volume_cm3
    assert ith == pytest.approx(q_v * p.threshold_density_cm3 / (p.eta_i * p.tau_n_s))
    levels = np.array([1.5, 2.0, 2.5, 3.0]) * ith
    hold = 12000
    field, _ = rate_equation_laser(np.repeat(levels, hold), 1e-12, p, noise=False, substeps=2)
    power = np.asarray(field.power())
    pm = np.array([power[(k + 1) * hold - 1000:(k + 1) * hold].mean() for k in range(4)])
    slope, icpt = np.polyfit(levels, pm, 1)
    assert abs(slope / p.slope_efficiency_w_a - 1) < 1e-4           # eta_i eta_out h nu / q
    assert abs(-icpt / slope / ith - 1) < 1e-3                       # extrapolated threshold
    for lev, pw in zip(levels, pm, strict=True):                     # settles on the steady state
        assert pw == pytest.approx(float(p.output_power_w(p.steady_state(lev)[1])), rel=1e-4)


def test_relaxation_oscillation_frequency_and_small_signal_response():
    p = LaserParams(epsilon_cm3=0.0)
    i0 = 2.0 * p.threshold_current_a
    f_r = p.relaxation_frequency_hz(i0)
    _, s0 = p.steady_state(i0)
    assert f_r == pytest.approx(math.sqrt(p.v_g_cm_s * p.a_cm2 * s0 / p.tau_p_s) / (2 * math.pi))
    f = np.linspace(0.2, 2.0, 4001) * f_r
    peak = f[np.argmax(np.abs(p.small_signal_response(f, i0)))]
    assert abs(peak / f_r - 1) < 0.02                                # spec: within 2 %
    # time-domain small-signal modulation against the linearized response
    dt, n = 1e-12, 24000
    t = dt * np.arange(n)
    for fm in (0.5 * f_r, 1.5 * f_r):
        di = 1e-3 * i0
        fld, _ = rate_equation_laser(i0 + di * np.sin(2 * np.pi * fm * t), dt, p, noise=False, substeps=2)
        pw = np.asarray(fld.power())
        n_per = int(round(12000 * dt * fm))
        seg = slice(n - int(round(n_per / fm / dt)), n)
        c = 2 * np.mean((pw[seg] - pw[seg].mean()) * np.exp(-2j * np.pi * fm * t[seg]))
        h_td = c / (-1j * di)
        assert abs(h_td / complex(p.small_signal_response(fm, i0)) - 1) < 2e-3


def test_linearized_langevin_theory_gives_the_henry_linewidth():
    # DC frequency noise of the linearized rate equations = Delta nu_ST (1 + alpha^2) / pi (Henry 1982)
    for alpha in (0.0, 2.0, 4.0):
        for beta_sp in (1e-5, 1e-4):
            p = LaserParams(epsilon_cm3=0.0, alpha_h=alpha, beta_sp=beta_sp)
            for m in (1.5, 4.0):
                i0 = m * p.threshold_current_a
                s0 = float(p.noise_spectra(0.0, i0)["frequency_noise"])
                assert s0 * math.pi / p.linewidth_hz(i0) == pytest.approx(1.0, rel=1e-3)
    p0 = LaserParams(epsilon_cm3=0.0, alpha_h=0.0)
    p4 = LaserParams(epsilon_cm3=0.0, alpha_h=4.0)
    i0 = 3 * p0.threshold_current_a
    assert p4.linewidth_hz(i0) / p0.linewidth_hz(i0) == pytest.approx(17.0)


@requires_jax          # 2.4 M Runge-Kutta substeps: compiled lax.scan (the NumPy loop is minutes)
def test_rate_equation_noise_spectra_match_the_linearized_theory():
    """Simulated frequency noise and RIN against the independent linear-noise analysis.

    Statistical tolerances are ~4 sigma of the Welch estimates (checked over seeds 7-11). Bands avoid the
    relaxation peak and its second harmonic, where the strong RO fluctuations at beta_sp = 1e-4 make the
    (correct) nonlinear simulation depart from the linear theory.
    """
    from scipy.signal import welch

    dt, n = 1e-12, 600000
    p = LaserParams(epsilon_cm3=0.0, alpha_h=4.0, beta_sp=1e-4)
    i0 = 4 * p.threshold_current_a
    fld, _ = rate_equation_laser(np.full(n, i0), dt, p, noise=True, seed=7)
    e = np.asarray(fld.samples)
    f, s_nu = welch(np.diff(np.unwrap(np.angle(e))) / (2 * np.pi * dt), fs=1 / dt, nperseg=2 ** 15)
    pw = np.abs(e) ** 2
    _, rin = welch(pw / pw.mean(), fs=1 / dt, nperseg=2 ** 15)
    th = p.noise_spectra(f, i0)

    def band(x, lo, hi):
        return float(np.mean(x[(f >= lo) & (f <= hi)]))

    low, high = (0.03e9, 1.5e9), (50e9, 200e9)
    # white Schawlow-Townes level far above f_R (the carrier term has died out there)
    assert band(s_nu, *high) * math.pi / (p.linewidth_hz(i0) / 17.0) == pytest.approx(1.0, abs=0.03)
    for lo, hi in (low, (1.5e9, 4e9)):
        assert band(s_nu, lo, hi) / band(th["frequency_noise"], lo, hi) == pytest.approx(1.0, abs=0.08)
    # Henry enhancement seen in the simulation: low-frequency over Schawlow-Townes level ~ 1 + alpha^2
    henry_sim = band(s_nu, *low) / band(s_nu, *high)
    henry_th = band(th["frequency_noise"], *low) / band(th["frequency_noise"], *high)
    assert henry_th == pytest.approx(17.0, rel=0.05)
    assert henry_sim == pytest.approx(henry_th, rel=0.10)
    # RIN: level away from the resonance, and the relaxation-oscillation peak position
    assert band(rin, 0.03e9, 4e9) / band(th["rin"], 0.03e9, 4e9) == pytest.approx(1.0, abs=0.12)
    assert band(rin, 20e9, 100e9) / band(th["rin"], 20e9, 100e9) == pytest.approx(1.0, abs=0.05)
    above = f > 1e9
    f_peak_sim = f[np.argmax(np.where(above, rin, 0.0))]
    f_peak_th = f[np.argmax(np.where(above, th["rin"], 0.0))]
    assert f_peak_sim == pytest.approx(f_peak_th, rel=0.02)
    assert f_peak_th == pytest.approx(p.relaxation_frequency_hz(i0), rel=0.05)


def test_cw_laser_linewidth_and_rin():
    dnu = 1e9
    w = cw_laser(1e-3, 2 ** 20, 1e-12, linewidth_hz=dnu, seed=3)
    phi = np.unwrap(np.angle(np.asarray(w.samples)))
    ok, _ = _chi2_ok(np.diff(phi), 2 * np.pi * dnu * 1e-12)         # Wiener increments
    assert ok
    e = np.asarray(w.samples) / math.sqrt(1e-3)
    lags = np.arange(5, 300)
    ac = np.array([abs(np.mean(e[k:] * np.conj(e[:-k]))) for k in lags])
    fwhm = -np.polyfit(lags * 1e-12, np.log(ac), 1)[0] / np.pi       # |<E(t+tau)E*(t)>| = exp(-pi dnu tau)
    assert abs(fwhm / dnu - 1) < 0.05
    w = cw_laser(1e-3, 2 ** 18, 1e-12, rin_db_hz=-140.0, seed=4)
    pw = np.asarray(w.power())
    ok, _ = _chi2_ok(pw / 1e-3, 1e-14 * 0.5e12)                      # RIN * f_s / 2
    assert ok
    from scipy.signal import welch

    _, psd = welch(pw / 1e-3, fs=1e12, nperseg=4096)
    assert abs(10 * np.log10(np.mean(psd[10:-10])) + 140.0) < 0.2   # flat one-sided PSD at RIN


# --------------------------------------------------------------------------- #
# Electrical filters and drive waveforms
# --------------------------------------------------------------------------- #
def test_first_order_low_pass_rise_time_is_0_35_over_f3db():
    f3 = 10e9
    lp = AnalogFilter.first_order(f3)
    assert lp.rise_time_s() == pytest.approx(math.log(9.0) / (2 * math.pi * f3), rel=1e-9)   # = 0.3497 / f3dB
    dt = 0.05e-12
    y = np.asarray(lp.apply(np.ones(4000), dt, discretization="zoh"))
    t = dt * np.arange(4000)
    t10, t90 = np.interp([0.1, 0.9], y, t)
    assert abs((t90 - t10) * f3 - 0.3497) < 2e-3


def test_filter_prototypes():
    for n in (1, 2, 4, 6):
        bw = AnalogFilter.butterworth(n, 5e9)
        assert bw.f3db_hz() == pytest.approx(5e9, rel=1e-6)
        # B_n of an order-n Butterworth: f3dB (pi / 2n) / sin(pi / 2n)
        x = math.pi / (2 * n)
        assert bw.noise_bandwidth_hz() == pytest.approx(5e9 * x / math.sin(x), rel=1e-9)
        be = AnalogFilter.bessel(n, 5e9)
        assert be.f3db_hz() == pytest.approx(5e9, rel=1e-6) and be.dc_gain == pytest.approx(1.0)
    # exact discretization: the FOH step response of a 4th-order Bessel equals the analytic one
    be = AnalogFilter.bessel(4, 20e9)
    dt = 0.2e-12
    x = np.r_[0.0, np.ones(3000)]
    y = np.asarray(be.apply(x, dt))
    t = dt * np.arange(3001)
    # FOH reads the step as a ramp over [0, dt]: compare with the analytic ramp-step response
    ramp = (np.cumsum(be.step_response(t)) - 0.5 * (be.step_response(t) + be.step_response(0 * t))) * 0  # unused
    del ramp
    y_ref = np.array([_ramp_step(be, tk, dt) for tk in t])
    assert np.max(np.abs(y - y_ref)) < 1e-9


def test_real_prototypes_filter_real_signals_to_real_outputs():
    x = np.random.default_rng(1).normal(size=300)
    for order in range(1, 7):
        for flt in (AnalogFilter.butterworth(order, 7e9), AnalogFilter.bessel(order, 7e9)):
            y = np.asarray(flt.apply(x, 1e-12))
            assert not np.iscomplexobj(y), flt.name
    odd = AnalogFilter((-1e10 + 1e10j,), (1e10,))                       # an unpaired complex pole
    assert np.iscomplexobj(np.asarray(odd.apply(x, 1e-12)))


def _ramp_step(flt, t, dt):
    """Analytic response to x(t) = clip(t / dt, 0, 1) of a pole-residue filter."""
    p = np.asarray(flt.poles)
    r = np.asarray(flt.residues)

    def integ(tau):          # integral_0^tau of the step response
        tau = max(tau, 0.0)
        return float(np.real(np.sum(r / p * ((np.exp(p * tau) - 1.0) / p - tau))))

    return (integ(t) - integ(t - dt)) / dt


@pytest.mark.parametrize("shape", ["gaussian", "raised_cosine"])
def test_drive_waveform_rise_time(shape):
    tr = 20e-12
    w = drive_waveform([0, 1, 1, 1], baud=10e9, samples_per_symbol=400, rise_time_20_80_s=tr, shape=shape,
                       amplitude_v=2.0, offset_v=1.0)
    v = np.asarray(w.samples)
    seg = slice(200, 600)                    # the 0 -> 1 edge centred at t = 100 ps
    t20, t80 = np.interp([0.4, 1.6], v[seg], w.t[seg])
    assert abs((t80 - t20) / tr - 1) < 2e-3
    assert v[0] == pytest.approx(0.0) and v[-1] == pytest.approx(2.0)


def test_drive_jitter_and_clock_recovery():
    bits = prbs(9, 1000)
    w = drive_waveform(bits, baud=25e9, samples_per_symbol=32, rise_time_20_80_s=8e-12, jitter_rms_s=1e-12, seed=2)
    phase = recover_clock(w, 25e9)
    assert abs(phase - 20e-12) < 1e-12          # eye centre half a UI after the (nominal) edges


# --------------------------------------------------------------------------- #
# (3) Receiver noise
# --------------------------------------------------------------------------- #
def _filtered_variance_ok(x, sigma2, flt, dt, n_sigma=4.0):
    """Sample variance of filtered Gaussian noise against ``sigma2``, with its exact standard error.

    For a stationary Gaussian sequence ``Var(s^2) = (2 sigma^4 / n) sum_k rho_k^2`` (Bartlett), with the
    autocorrelation ``rho`` known from the filter's band-limited impulse response.
    """
    x = np.asarray(x)
    n = x.size
    f = np.fft.fftfreq(2 ** 16, dt)
    acf = np.real(np.fft.ifft(np.abs(flt.response(f)) ** 2))
    rho = acf / acf[0]
    se = math.sqrt(2.0 / n * np.sum(rho ** 2)) * sigma2
    return abs(np.var(x) - sigma2) <= n_sigma * se, (np.var(x) / sigma2, se / sigma2)


def test_shot_noise_variance_is_2qIB():
    dt, n = 1e-12, 2 ** 20
    current = 1e-3
    for flt in (AnalogFilter.bessel(4, 10e9), AnalogFilter.first_order(5e9), AnalogFilter.bessel(4, 80e9)):
        i = photodiode(cw(current, n, dt), responsivity_a_w=1.0, noise=("shot",), bandwidth=flt, seed=11)
        b_n = flt.noise_bandwidth_hz()
        # noise is band-limited to f_s / 2: the part of B_n beyond Nyquist is not simulated
        f = np.linspace(0.0, 0.5 / dt, 200001)
        b_sim = np.trapezoid(np.abs(flt.response(f)) ** 2, f)
        assert b_sim == pytest.approx(b_n, rel=0.02 if flt.name.startswith("RC") else 1e-5)
        x = np.asarray(i.samples)[5000:] - current              # past the filter's start-up transient
        assert not np.iscomplexobj(x)
        ok, stats = _filtered_variance_ok(x, 2 * Q_E * current * b_sim, flt, dt)
        assert ok, (flt.name, stats)
    raw = photodiode(cw(current, n, dt), responsivity_a_w=1.0, noise=("shot",), seed=12)
    ok, ci = _chi2_ok(np.asarray(raw.samples), 2 * Q_E * current * 0.5 / dt)   # white to Nyquist
    assert ok, ci


def test_thermal_and_tia_noise_variances():
    dt, n = 1e-12, 2 ** 20
    flt = AnalogFilter.butterworth(4, 8e9)
    i = photodiode(cw(0.0, n, dt), responsivity_a_w=1.0, noise=("thermal",), load_ohm=50.0, temperature_k=300.0,
                   bandwidth=flt, seed=5)
    ok, stats = _filtered_variance_ok(i.samples, 4 * K_BOLTZMANN * 300.0 / 50.0 * flt.noise_bandwidth_hz(), flt, dt)
    assert ok, stats
    v = tia(Waveform(np.zeros(n), dt, carrier_wl=None, unit="A"), gain_ohm=2000.0, bandwidth=flt,
            input_noise_a_rthz=15e-12, seed=6)
    ok, stats = _filtered_variance_ok(v.samples, (2000.0 * 15e-12) ** 2 * flt.noise_bandwidth_hz(), flt, dt)
    assert ok, stats
    # the output noise spectrum is N0 |H(f)|^2 (Welch estimate, averaged in bands)
    from scipy.signal import welch

    fw, psd = welch(np.asarray(i.samples), fs=1 / dt, nperseg=2 ** 12)
    n0 = 4 * K_BOLTZMANN * 300.0 / 50.0
    for lo, hi in ((0.5e9, 4e9), (6e9, 10e9)):
        sel = (fw >= lo) & (fw <= hi)
        assert np.mean(psd[sel]) / np.mean(n0 * np.abs(flt.response(fw[sel])) ** 2) == pytest.approx(1.0, abs=0.03)


# --------------------------------------------------------------------------- #
# (4) Thermal-noise-limited sensitivity and the BER law
# --------------------------------------------------------------------------- #
BAUD, SPS = 10e9, 32
DT = 1 / (BAUD * SPS)
RX = AnalogFilter.bessel(4, 2.5 * BAUD)
BITS = np.random.default_rng(21).integers(0, 2, 2 ** 13)


def _thermal_link(p_dbm, bits=BITS, seed=8):
    p_avg = 1e-3 * 10 ** (p_dbm / 10)
    power = np.repeat(bits, SPS) * 2 * p_avg            # infinite extinction ratio NRZ
    field = Waveform(np.sqrt(power).astype(complex), DT, carrier_wl=1.55)
    i = photodiode(field, responsivity_a_w=1.0, noise=("thermal",), bandwidth=RX, seed=seed)
    return eye_diagram(i, baud=BAUD, symbols=bits, skip_symbols=4)


def test_thermal_noise_limited_sensitivity_matches_agrawal():
    sigma_t = math.sqrt(4 * K_BOLTZMANN * 300.0 / 50.0 * RX.noise_bandwidth_hz())
    expected_dbm = 10 * math.log10(7.0 * sigma_t / 1.0 / 1e-3)        # P = Q sigma_T / R at Q = 7
    sens = receiver_sensitivity(_thermal_link, target_ber=float(ber_from_q(7.0)), bounds_dbm=(-30.0, 0.0))
    assert abs(sens - expected_dbm) < 0.1
    # the sensitivity drops straight into a link budget
    lb = px.analysis.LinkBudget(source_dbm=0.0, sensitivity_dbm=sens).add("coupler", 3.0).add("modulator", 4.0)
    eye = _thermal_link(lb.received_dbm)
    assert (lb.margin_db > 0) == (eye.ber_gaussian < ber_from_q(7.0))


def test_counted_ber_follows_the_gaussian_law_near_q3():
    sigma_t = math.sqrt(4 * K_BOLTZMANN * 300.0 / 50.0 * RX.noise_bandwidth_hz())
    p_dbm = 10 * math.log10(3.0 * sigma_t / 1e-3)
    bits = np.random.default_rng(22).integers(0, 2, 2 ** 16)
    eye = _thermal_link(p_dbm, bits=bits, seed=9)
    assert abs(eye.q_factor - 3.0) < 0.05
    ber, (lo, hi) = ber_count(eye.decisions(), eye.labels)
    assert lo <= float(ber_from_q(eye.q_factor)) <= hi
    assert float(q_from_ber(ber_from_q(3.0))) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# (5) Mach-Zehnder modulator and traveling-wave electrode
# --------------------------------------------------------------------------- #
def test_mzm_static_transfer_extinction_and_chirp():
    v = np.linspace(-6, 6, 241)
    out = mzm(cw(1.0, v.size, 1e-12), *push_pull(v), v_pi=2.5)
    assert np.max(np.abs(np.asarray(out.power()) - np.cos(np.pi * v / (2 * 2.5)) ** 2)) < 1e-12
    assert np.allclose(np.angle(np.asarray(out.samples))[np.cos(np.pi * v / 5) > 0.1], 0.0, atol=1e-12)   # no chirp
    out = mzm(cw(1.0, v.size, 1e-12), *push_pull(v), v_pi=2.5, extinction_db=25.0, insertion_loss_db=3.0)
    pw = np.asarray(out.power())
    assert 10 * np.log10(pw.max() / pw.min()) == pytest.approx(25.0, abs=1e-3)
    assert 10 * np.log10(pw.max()) == pytest.approx(-3.0, abs=1e-9)
    # small-signal chirp at quadrature equals the drive imbalance
    c = 0.4
    t = 1e-12 * np.arange(2000)
    vv = 1.25 + 0.01 * np.sin(2 * np.pi * 5e9 * t)
    out = mzm(cw(1.0, t.size, 1e-12), *push_pull(vv, chirp=c), v_pi=2.5)
    e = np.asarray(out.samples)
    pw = np.abs(e) ** 2
    dphi = np.gradient(np.unwrap(np.angle(e)), t)
    dpw = np.gradient(pw, t)
    steep = np.abs(dpw) > 0.5 * np.abs(dpw).max()            # away from the drive extrema (0 / 0)
    alpha = 2 * pw[steep] * dphi[steep] / dpw[steep]
    assert np.median(alpha) == pytest.approx(c, rel=0.02)


def test_traveling_wave_electrode_response():
    el = TravelingWaveElectrode(length_mm=10.0, n_m=2.6, n_g_opt=2.2)
    f = 1.3916 * 299792458.0 / (math.pi * 0.4 * 0.01)
    assert abs(abs(complex(el.response(f))) - 1 / math.sqrt(2)) < 0.01
    assert el.bandwidth_hz() == pytest.approx(f, rel=1e-3)
    # lossy, mismatched line: closed form vs direct quadrature of the standing wave seen by the light
    el = TravelingWaveElectrode(length_mm=4.0, n_m=3.9, n_g_opt=4.1, alpha_db_per_cm_sqrt_ghz=0.8,
                                z0_ohm=35.0, z_load_ohm=50.0, z_source_ohm=50.0)
    for f in (0.0, 7e9, 31e9):
        assert abs(complex(el.response(f)) - _electrode_quadrature(el, f)) < 1e-6
    # applied in the time domain, a sinusoidal drive is scaled by |H| (and delayed by its phase)
    dt, n = 0.5e-12, 2 ** 15
    t = dt * np.arange(n)
    for fm in (5e9, 20e9, 40e9):
        y = np.asarray(el.filter(np.sin(2 * np.pi * fm * t), dt))
        period = int(round(1 / (fm * dt)))
        seg = slice(n // 2, n // 2 + (n // 2 // period) * period)      # whole periods in the second half
        c = 2 * np.mean(y[seg] * np.exp(-2j * np.pi * fm * t[seg])) / -1j
        h = complex(el.response(fm)) * np.exp(-2j * np.pi * fm * el.latency_s)
        assert abs(c - h) < 2e-3 * abs(h), fm


def test_skin_effect_electrode_is_causal_with_the_sqrt_t_tail():
    el = TravelingWaveElectrode(length_mm=4.0, n_m=3.9, n_g_opt=4.1, alpha_db_per_cm_sqrt_ghz=0.8)
    dt, m = 1e-12, 2 ** 21                                   # a 2 us grid: aliasing << the tails probed
    k = el.kernel(dt, m)
    assert abs(k.sum() - 1.0) < 1e-12                        # H(0) = 1
    # causal: the negative-time part (the end of the circular kernel) holds only the alternating Gibbs
    # ringing of the sampled box edges (net ~1 % of the tail in the same window), not the t^-3/2 tail that
    # a loss without its skin-effect phase (non-causal) puts on both sides of t = 0 (> 30 %)
    neg, pos = slice(m - 1000, m - 100), slice(100, 1000)
    assert abs(np.sum(k[neg])) < 0.03 * np.sum(k[pos])
    acausal = TravelingWaveElectrode(length_mm=4.0, n_m=3.9, n_g_opt=4.1)      # loss without its phase
    f = np.fft.rfftfreq(m, dt)
    alpha = 0.8 * np.sqrt(f / 1e9) * 100 / (20 / math.log(10))
    x = (alpha + 1j * 2 * np.pi * f * (3.9 - 4.1) / 299792458.0) * 4e-3
    h_bad = np.where(np.abs(x) > 0, (1 - np.exp(-x)) / np.where(np.abs(x) > 0, x, 1), 1.0)
    k_bad = np.fft.irfft(h_bad * np.exp(-2j * np.pi * f * acausal.latency_s), m)
    assert abs(np.sum(k_bad[neg])) > 0.3 * np.sum(k_bad[pos])   # the test would catch a non-causal model
    step = np.cumsum(k)
    for t1, t2 in ((1e-9, 4e-9), (2e-9, 20e-9)):
        late = step[int(t2 / dt)] - step[int(t1 / dt)]      # = tail(t1) - tail(t2), aliasing-free
        assert late == pytest.approx(el.skin_tail(t1) - el.skin_tail(t2), rel=0.03)
    # a lossless line has a finite kernel: nothing (beyond decaying, alternating edge ringing) arrives late
    lossless = TravelingWaveElectrode(length_mm=4.0, n_m=3.9, n_g_opt=4.1)
    assert lossless.skin_tail(1e-9) == 0.0
    k0 = lossless.kernel(dt, 2 ** 16)
    assert abs(np.sum(k0[2000:-2000])) < 1e-4                              # ~ (edge jump) / (2 pi 2000)
    assert np.sum(k[2000:2 ** 15]) > 30 * abs(np.sum(k0[2000:-2000]))      # the lossy line's tail is there


def _electrode_quadrature(el, f):
    c = 299792458.0
    length = el.length_mm * 1e-3
    w = 2 * math.pi * f
    a = el.alpha_db_per_cm_sqrt_ghz * math.sqrt(f / 1e9) * 100 / (20 / math.log(10))
    g = a * (1 + 1j) + 1j * w * el.n_m / c                 # causal skin effect, Z_s = (1 + j) R_s
    rho = (el.z_load_ohm - el.z0_ohm) / (el.z_load_ohm + el.z0_ohm)
    z = np.linspace(0, length, 20001)
    vz = np.exp(-g * z) + rho * np.exp(-2 * g * length) * np.exp(g * z)          # V(z) / A
    v0 = 1 + rho * np.exp(-2 * g * length)
    z_in = el.z0_ohm * (1 + rho * np.exp(-2 * g * length)) / (1 - rho * np.exp(-2 * g * length))
    avg = np.trapezoid(vz * np.exp(1j * w * el.n_g_opt * z / c), z) / length / v0
    raw = avg * z_in / (z_in + el.z_source_ohm)
    return raw / el.dc_gain


def test_eam_transmission_and_henry_chirp():
    v = np.linspace(0, 2, 2001)
    t = 1e-12 * np.arange(v.size)
    out = eam(cw(1.0, v.size, 1e-12), v, absorption_db=lambda x: 3.0 * x ** 2, alpha_h=1.5)
    pw = np.asarray(out.power())
    assert np.allclose(pw, 10 ** (-0.3 * v ** 2), rtol=1e-12)
    e = np.asarray(out.samples)
    alpha = 2 * pw * np.gradient(np.unwrap(np.angle(e)), t) / np.gradient(pw, t)
    assert np.allclose(alpha[10:-10], 1.5, rtol=1e-3)


# --------------------------------------------------------------------------- #
# (6) Ring modulator
# --------------------------------------------------------------------------- #
def test_ring_modulator_cmt_matches_the_analytic_ring_in_the_high_q_limit():
    rm = RingModulator(radius=10.0, coupling=0.01, loss_db_cm=2.0, pn_fraction=0.0, dneff_dv=1e-4)
    t_rt = rm.round_trip_s
    _, inv_i, inv_e = (float(np.asarray(v)) for v in rm.rates(1.55, 0.0))
    dw0 = float(np.asarray(rm.rates(1.55, 0.0)[0]))
    lw = (inv_i + inv_e) / math.pi                      # FWHM in Hz
    offsets = dw0 / (2 * math.pi) + np.linspace(-4, 4, 161) * lw
    cmt = np.abs(np.asarray(rm.steady_transmission(1.55, 0.0, offsets))) ** 2
    ring = px.components.all_pass_ring(wl=C0_UM_S / (C0_UM_S / 1.55 + offsets), coupling=0.01, radius=10.0,
                                       loss_db_cm=2.0)
    analytic = np.abs(np.asarray(ring[("o1", "o2")])) ** 2
    assert np.max(np.abs(cmt - analytic)) < 0.01
    assert t_rt == pytest.approx(4.2 * 2 * math.pi * 10.0 / C0_UM_S)
    # the time-domain integration holds the steady state exactly at constant drive (carrier at the laser)
    wl_laser = C0_UM_S / (C0_UM_S / 1.55 + offsets[60])
    out = rm.simulate(cw(1e-3, 500, 1e-12, carrier_wl=wl_laser), np.zeros(500))
    s = complex(np.asarray(rm.steady_transmission(1.55, 0.0, offsets[60])))
    assert complex(np.asarray(rm.steady_transmission(wl_laser, 0.0))) == pytest.approx(s, abs=1e-9)
    assert np.allclose(np.asarray(out.samples) / math.sqrt(1e-3), s, atol=1e-12)
    # a step of the drive relaxes with the photon lifetime: field transient exp(-t / tau) toward the new state
    rm = RingModulator(radius=10.0, coupling=0.01, loss_db_cm=2.0, pn_fraction=0.5, dneff_dv=1e-4)
    v = np.r_[np.zeros(100), np.full(1900, 3.0)]
    out = np.asarray(rm.simulate(cw(1e-3, v.size, 0.1e-12, carrier_wl=wl_laser), v).samples) / math.sqrt(1e-3)
    s_new = complex(np.asarray(rm.steady_transmission(wl_laser, 3.0)))
    assert abs(s_new - complex(np.asarray(rm.steady_transmission(wl_laser, 0.0)))) > 0.1
    _, ii, ie = (float(np.asarray(x)) for x in rm.rates(wl_laser, 3.0))
    t = 0.1e-12 * (np.arange(v.size) - 100)
    late = t > 20e-12
    env = np.abs(out[late] - s_new)
    rate = -np.polyfit(t[late], np.log(env), 1)[0]
    assert rate == pytest.approx(ii + ie, rel=1e-3)


def test_ring_modulator_small_signal_matches_sacher_poon():
    rm = RingModulator(radius=7.5, coupling=0.02, loss_db_cm=5.0, dneff_dv=2e-5, pn_fraction=0.75)
    dw0, inv_i, inv_e = (float(np.asarray(v)) for v in rm.rates(1.55, 0.0))
    inv_tau = inv_i + inv_e
    offset = (dw0 + 0.6 * inv_tau) / (2 * math.pi)       # laser on the resonance slope
    dt, n = 0.5e-12, 2 ** 15
    t = dt * np.arange(n)
    for fm in (2e9, 15e9, 40e9):
        dv = 1e-3
        out = rm.simulate(cw(1e-3, n, dt, offset_hz=offset), dv * np.cos(2 * np.pi * fm * t))
        pw = np.asarray(out.power())
        n_per = int(n // 2 * dt * fm)
        seg = slice(n - int(round(n_per / fm / dt)), n)
        c = 2 * np.mean((pw[seg] - pw[seg].mean()) * np.exp(-2j * np.pi * fm * t[seg]))
        h = complex(rm.small_signal_response(fm, 0.0, power_w=1e-3, offset_hz=offset))
        assert abs(c / dv / h - 1) < 0.01, fm


def test_ring_modulator_with_pn_junction_and_rc():
    rm = RingModulator(radius=7.5, coupling=0.03, loss_db_cm=5.0, rc_time_s=5e-12)
    dw0_a, inv_i_a, _ = (np.asarray(v) for v in rm.rates(1.55, np.array([0.0, 2.0])))
    assert dw0_a[1] < dw0_a[0]                 # reverse bias raises n_eff: red shift
    assert inv_i_a[1] < inv_i_a[0]             # and depletes free carriers: less loss
    v = np.r_[np.zeros(200), np.full(800, 2.0)]
    vj = np.asarray(rm.drive(v, 1e-12))
    # RC charging; first-order hold reads the step as a ramp over the sample before it (exact solution)
    tau, dt = 5e-12, 1e-12
    t = dt * (np.arange(1000) - 200)
    exact = np.where(t >= 0, 2.0 * (1 - tau / dt * (1 - math.exp(-dt / tau)) * np.exp(-t / tau)), 0.0)
    assert np.max(np.abs(vj - exact)) < 1e-12


# --------------------------------------------------------------------------- #
# (7) Eye diagrams
# --------------------------------------------------------------------------- #
def test_ideal_eye_height_equals_oma_and_drive_extinction():
    bits = prbs(9, 1022)
    drive = drive_waveform(bits, baud=BAUD, samples_per_symbol=16, amplitude_v=2.5, offset_v=1.25)
    field = mzm(cw(1e-3, drive.n_steps, drive.dt), *push_pull(drive), v_pi=2.5, extinction_db=12.0)
    i = photodiode(field, responsivity_a_w=1.0, noise=())
    eye = eye_diagram(i, baud=BAUD, symbols=1 - bits)          # V = 0 is the bright state
    assert eye.height == pytest.approx(eye.oma, rel=1e-12)
    assert eye.extinction_ratio_db == pytest.approx(12.0, abs=1e-9)
    assert eye.crossing_pct == pytest.approx(50.0, abs=1.0)
    assert eye.width == pytest.approx(1 / BAUD, rel=0.07)


def test_first_order_channel_isi_closure_is_exact():
    bits = prbs(7, 1016)
    tau = 40e-12
    x = Waveform(np.repeat(bits, 16).astype(float), 1 / (BAUD * 16), carrier_wl=None, unit="V")
    y = AnalogFilter.first_order(1 / (2 * math.pi * tau)).apply(x, discretization="zoh")
    eye = eye_diagram(y, baud=BAUD, symbols=bits, skip_symbols=8)
    # exact bit-end samples: v_k = e^(-T/tau) v_{k-1} + (1 - e^(-T/tau)) b_k
    a = math.exp(-1 / BAUD / tau)
    v = np.zeros(bits.size)
    for k in range(bits.size):
        v[k] = a * (v[k - 1] if k else 0.0) + (1 - a) * bits[k]
    v, b = v[8:], bits[8:]
    assert eye.height == pytest.approx(v[b == 1].min() - v[b == 0].max(), abs=1e-12)


def test_pam4_eye_and_unaided_clustering():
    bits = prbs(11, 4000)
    levels = symbols_to_levels(bits, "pam4")
    sym = np.rint(levels * 3).astype(int)
    w = drive_waveform(levels, baud=25e9, samples_per_symbol=16, rise_time_20_80_s=6e-12, amplitude_v=1.5)
    eye = eye_diagram(w, baud=25e9, fmt="pam4", symbols=sym)
    assert eye.levels.size == 4 and eye.height == pytest.approx(0.5, rel=1e-3)
    assert eye.oma == pytest.approx(1.5, rel=1e-6)            # Gaussian edge tails: ~1e-8 at mid-UI
    blind = eye_diagram(w, baud=25e9, fmt="pam4")                  # k-means classification
    assert np.allclose(blind.levels, eye.levels, atol=1e-9)
    noisy = Waveform(np.asarray(w.samples) + np.random.default_rng(0).normal(0, 0.04, w.n_steps), w.dt,
                     carrier_wl=None, unit="V")
    # level spacing 0.5 V, sigma 0.04 V: Q = 0.5 / (2 * 0.04) per eye. A fixed sampling phase avoids the
    # upward bias of picking the best of 16 phases with independent noise.
    e2 = eye_diagram(noisy, baud=25e9, fmt="pam4", symbols=sym, sampling_index=eye.sampling_index)
    assert e2.q_factor == pytest.approx(6.25, rel=0.08)              # ~1000 samples per level
    assert np.allclose(e2.sigmas, 0.04, rtol=0.08)
    # Gray-coded PAM4: at the Q-optimal threshold both tails of eye e equal Qf(q_e) = erfc(q_e / sqrt 2) / 2,
    # so SER = sum_e (pi_e + pi_{e+1}) Qf(q_e) and BER = SER / 2 (-> (3/4) Qf(q) for equal eyes and priors)
    prior = np.bincount(e2.labels, minlength=4) / e2.labels.size
    ser = sum((prior[k] + prior[k + 1]) * float(ber_from_q(q)) for k, q in enumerate(e2.q_eyes))
    assert e2.ber_gaussian == pytest.approx(ser / 2, rel=1e-9)


def test_plot_eye_never_shows(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: (_ for _ in ()).throw(AssertionError("plt.show called")))
    bits = prbs(7, 254)
    eye = eye_diagram(drive_waveform(bits, baud=10e9, samples_per_symbol=16, rise_time_20_80_s=25e-12),
                      baud=10e9, symbols=bits)
    ax = eye.plot()
    assert ax.get_xlabel() == "Time (UI)"
    plt.close("all")


# --------------------------------------------------------------------------- #
# Gradients through the transceiver chain
# --------------------------------------------------------------------------- #
@requires_jax
def test_gradient_through_modulator_filter_and_ring():
    t = 1e-12 * np.arange(800)
    drive = np.sin(2 * np.pi * 10e9 * t)

    def energy(amp):
        v1, v2 = push_pull(amp * px.xp.asarray(drive), bias_v=1.25)
        f = mzm(cw(1e-3, t.size, 1e-12), v1, v2, v_pi=2.5, drive_filter=AnalogFilter.bessel(4, 30e9))
        rm = RingModulator(radius=7.5, coupling=0.03, dneff_dv=2e-5)
        out = rm.simulate(f, amp * px.xp.asarray(drive))
        return px.xp.sum(px.xp.abs(out.samples) ** 2)

    g = float(px.grad(energy)(0.8))
    fd = float(finite_diff(lambda a: float(energy(a)), 0.8, h=1e-5))
    assert abs(g - fd) <= 1e-6 * abs(fd) + 1e-12
