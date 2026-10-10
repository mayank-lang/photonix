"""Validation of photonix.analysis against closed-form results."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

import photonix as px
from photonix import analysis as an
from photonix.pdk import MonteCarloSpec, ProcessCorner, ProcessStudy


def _ring_through(wl, **kw):
    return np.asarray(px.power(px.components.all_pass_ring(wl=wl, **kw)[("o1", "o2")]))


# --------------------------------------------------------------------------- #
# Resonances
# --------------------------------------------------------------------------- #
def test_fsr_and_finesse_match_ring_theory():
    R, ng = 20.0, 4.2
    L = 2 * np.pi * R
    wl = np.linspace(1.54, 1.56, 400001)       # FSR ~4.6 nm: four resonances
    res = an.find_resonances(wl, _ring_through(wl, coupling=0.05, radius=R, ng=ng, loss_db_cm=1.0))
    assert len(res) >= 3
    fsr = an.free_spectral_range(res)
    assert fsr == pytest.approx(1.55**2 / (ng * L), rel=5e-3)
    assert an.finesse(res) == pytest.approx(fsr / np.mean([r.fwhm for r in res]))


def test_ring_q_split_recovers_physical_qs():
    R, ng, wl0, kappa2, loss = 30.0, 4.2, 1.55, 0.01, 2.0
    L = 2 * np.pi * R
    alpha_power = loss * math.log(10) / 10 * 1e-4         # 1/um
    qi = 2 * np.pi * ng / (wl0 * alpha_power)
    qc = 2 * np.pi * ng * L / (wl0 * kappa2)
    wl = np.linspace(1.5495, 1.5505, 200001)
    T = _ring_through(wl, coupling=kappa2, radius=R, ng=ng, loss_db_cm=loss)
    r = an.fit_lorentzian(wl, T, center=1.55, window=2e-4)
    assert r.q_loaded == pytest.approx(1 / (1 / qi + 1 / qc), rel=0.02)
    under, over = an.ring_q_factors(r)
    regime = under if qc > qi else over          # ground truth picks the regime
    assert regime.q_intrinsic == pytest.approx(qi, rel=0.03)
    assert regime.q_coupling == pytest.approx(qc, rel=0.03)


def test_drop_port_peaks_are_found():
    wl = np.linspace(1.54, 1.56, 40001)
    s = px.components.add_drop_ring(wl=wl, coupling1=0.1, coupling2=0.1)
    D = np.asarray(px.power(s[("o1", "o3")]))
    res = an.find_resonances(wl, D, kind="peak", prominence=0.1)
    assert res and all(r.extremum > 0.5 for r in res)
    with pytest.raises(ValueError, match="kind"):
        an.find_resonances(wl, D, kind="notch")


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_bandwidth_of_lorentzian_and_box_like_crow():
    wl = np.linspace(1.545, 1.555, 40001)
    single = np.asarray(px.power(px.circuit.crow(couplings=(0.1, 0.1), loss_db_cm=0.0)(wl=wl)[("o1", "o3")]))
    third = np.asarray(px.power(px.circuit.crow(couplings=(0.4, 0.06, 0.06, 0.4), loss_db_cm=0.0)(wl=wl)[("o1", "o3")]))
    centre = float(wl[np.argmax(third)])
    m1 = an.passband_metrics(wl, single, center=float(wl[np.argmax(single)]))
    m3 = an.passband_metrics(wl, third, center=centre)
    # A ring's drop port is an Airy line, D = 1 / (1 + 4 t^2 sin^2(phi/2) / (1 - t^2)^2):
    # the exact shape factor exceeds the Lorentzian sqrt(99 / 0.995) because the
    # -20 dB points sit far enough out for sin(phi/2) != phi/2.
    t2 = 0.9
    c = (1 - t2) ** 2 / (4 * t2)
    airy = math.asin(math.sqrt(c * 99)) / math.asin(math.sqrt(c * (10**0.3 - 1)))
    assert m1.shape_factor == pytest.approx(airy, rel=2e-3)
    assert m3.shape_factor < 0.4 * m1.shape_factor
    assert m3.insertion_loss_db < 0.01


def test_extinction_ratio_and_bandwidth_errors():
    assert an.extinction_ratio_db([1.0, 0.01]) == pytest.approx(20.0)
    wl = np.linspace(1.5, 1.6, 11)
    with pytest.raises(ValueError, match="does not fall"):
        an.bandwidth(wl, np.ones_like(wl))
    with pytest.raises(ValueError, match="negative"):
        an.bandwidth(wl, np.ones_like(wl), level_db=3.0)


# --------------------------------------------------------------------------- #
# Link budget
# --------------------------------------------------------------------------- #
def test_link_budget_arithmetic_and_report():
    s = px.components.edge_coupler(mfd_chip=3.0, mfd_fiber=10.4)
    stage = an.stage_from_sparameters("edge coupler", s, ("o1", "o2"), count=2)
    expected = -10 * math.log10(float(px.components.gaussian_overlap(mfd1=3.0, mfd2=10.4)))
    assert stage.loss_db == pytest.approx(expected)
    lb = an.LinkBudget(source_dbm=13.0, sensitivity_dbm=-20.0).extend([stage]).add("ring filter", 1.5)
    assert lb.total_loss_db == pytest.approx(2 * expected + 1.5)
    assert lb.margin_db == pytest.approx(13.0 - lb.total_loss_db + 20.0)
    assert lb.received_mw == pytest.approx(10 ** (lb.received_dbm / 10))
    assert lb.table()[-1]["power_dbm"] == pytest.approx(lb.received_dbm)
    assert "margin" in str(lb)
    with pytest.raises(ValueError):
        an.LinkStage("bad", float("nan"))


# --------------------------------------------------------------------------- #
# Yield
# --------------------------------------------------------------------------- #
def test_monte_carlo_yield_matches_gaussian_probability():
    sigma = 0.01
    study = ProcessStudy(
        ProcessCorner("tt", {"coupling": 0.5}),
        corners=(ProcessCorner("ff", {"coupling": 0.53}), ProcessCorner("ss", {"coupling": 0.47})),
        monte_carlo=MonteCarloSpec.independent({"coupling": 0.5}, {"coupling": sigma}),
        units={"coupling": "1"},
    )

    def evaluate(case):
        return px.components.directional_coupler(coupling=case.parameters["coupling"])

    spec = an.Spec("cross", lambda s: float(px.power(s[("o1", "o3")])), minimum=0.5 - 1.5 * sigma,
                   maximum=0.5 + 1.5 * sigma)
    res = an.yield_analysis(evaluate, [spec], study=study, samples=4000, seed=3)
    exact = norm.cdf(1.5) - norm.cdf(-1.5)
    lo, hi = res.yield_interval
    assert lo <= exact <= hi
    assert res.yield_fraction == pytest.approx(exact, abs=0.02)
    assert res.corner_pass == {"tt": True, "ff": False, "ss": False}
    stats = res.statistics("cross")
    assert stats["std"] == pytest.approx(sigma, rel=0.05)
    assert "yield" in str(res)


def test_wilson_interval_reference_values():
    # Reference: Wilson (1927) score interval, z = 1.959964
    lo, hi = an.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = an.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)


def test_yield_requires_exactly_one_case_source():
    spec = an.Spec("x", lambda r: r, maximum=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        an.yield_analysis(lambda c: 0.0, [spec])
    with pytest.raises(ValueError, match="minimum and/or"):
        an.Spec("y", lambda r: r)


# --------------------------------------------------------------------------- #
# Time domain
# --------------------------------------------------------------------------- #
def test_impulse_response_delay_is_group_delay():
    wl = np.linspace(1.53, 1.57, 8001)
    s = px.components.straight(wl=wl, length=3000.0, neff=2.4, ng=4.2)
    t, h = an.impulse_response(wl, np.asarray(s[("o1", "o2")]), oversample=16)
    tau = 4.2 * 3000.0 / (px.core.constants.C0_UM_S * 1e-12)
    assert t[np.argmax(np.abs(h))] == pytest.approx(tau, abs=t[1])


def test_gaussian_pulse_broadening_matches_gdd_formula():
    c_um_ps = px.core.constants.C0_UM_S * 1e-12
    wl = np.linspace(1.50, 1.60, 20001)
    f = c_um_ps / wl
    f0 = c_um_ps / 1.55
    phi2 = 0.8                                  # ps^2
    tau = 20.0                                  # ps delay keeps the pulse inside the window
    om = 2 * np.pi * (f - f0)
    H = np.exp(-1j * (om * tau + 0.5 * phi2 * om**2))
    tp = 1.0
    r = an.pulse_response(wl, H, pulse_fwhm_ps=tp, center_wl=1.55, n_samples=16384)
    expected = math.sqrt(1 + (4 * math.log(2) * phi2 / tp**2) ** 2)
    assert r.broadening == pytest.approx(expected, rel=1e-3)
    assert r.delay_ps == pytest.approx(tau, abs=1e-3)
    assert r.energy_transmission == pytest.approx(1.0, rel=1e-6)


def test_pulse_outside_band_is_rejected():
    wl = np.linspace(1.549, 1.551, 201)
    with pytest.raises(ValueError, match="beyond the sampled band"):
        an.pulse_response(wl, np.ones(201), pulse_fwhm_ps=0.5)
