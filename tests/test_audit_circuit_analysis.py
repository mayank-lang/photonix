"""Regression tests for circuit/analysis bugs found by the round-2 audit."""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest
from scipy.stats import unitary_group

import photonix as px
from photonix import analysis as an
from photonix.circuit import MeshProgram, Netlist, clements_decompose, mesh_matrix, mzi_mesh
from photonix.pdk import MonteCarloSpec, ProcessCorner, ProcessStudy

C_UM_PS = px.core.constants.C0_UM_S * 1e-12


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
def test_ripple_measures_in_band_ripple_not_the_band_edge():
    wl = np.linspace(1.54, 1.56, 40001)
    flat = 1 / (1 + ((wl - 1.55) / 0.004) ** 16)
    assert an.passband_metrics(wl, flat).ripple_db < 1e-6
    rip = flat * 10 ** (-0.1 * (1 - np.cos(2 * np.pi * (wl - 1.55) / 0.0015)) / 2 / 10)
    m = an.passband_metrics(wl, rip)
    assert m.ripple_db == pytest.approx(0.1, rel=0.05)


def test_center_selects_channel_peak_and_descending_sweeps_work():
    wl = np.linspace(1.54, 1.56, 20001)
    two = 0.5 * np.exp(-((wl - 1.545) / 0.001) ** 2) + 0.9 * np.exp(-((wl - 1.555) / 0.001) ** 2)
    m = an.passband_metrics(wl, two, center=1.5452)
    assert m.peak_wavelength == pytest.approx(1.545, abs=2e-6)
    assert m.insertion_loss_db == pytest.approx(-10 * np.log10(0.5), abs=1e-3)
    fwd = an.passband_metrics(wl, two)
    rev = an.passband_metrics(wl[::-1], two[::-1])
    assert rev.bandwidth_3db == pytest.approx(fwd.bandwidth_3db) and rev.bandwidth_3db > 0
    assert an.bandwidth(wl[::-1], two[::-1])[2] > 0


# --------------------------------------------------------------------------- #
# resonances
# --------------------------------------------------------------------------- #
def test_zero_extremum_and_zero_baseline_do_not_crash():
    wl = np.linspace(1.549, 1.551, 2001)
    dip = 1 - 1 / (1 + ((wl - 1.55) / 2e-5) ** 2)
    dip[np.argmin(np.abs(wl - 1.55))] = 0.0
    r = an.find_resonances(wl, dip)[0]
    assert r.extinction_db == math.inf
    peak = np.clip(1 / (1 + ((wl - 1.55) / 2e-5) ** 2) - 0.01, 0, None)
    r = an.find_resonances(wl, peak, kind="peak")[0]
    assert r.extinction_db == math.inf
    assert an.fit_lorentzian(wl, dip, window=2e-4).fwhm > 0


def test_doublet_widths_are_not_merged():
    wl = np.linspace(1.5495, 1.5505, 40001)
    w = 4e-6
    T = 1 - 0.9 / (1 + ((wl - 1.54999) / (w / 2)) ** 2) - 0.9 / (1 + ((wl - 1.55001) / (w / 2)) ** 2)
    res = an.find_resonances(wl, np.clip(T, 0, None))
    assert len(res) == 2
    for r in res:
        assert np.isnan(r.fwhm) or r.fwhm < 2 * w      # never the merged ~2.5x width


# --------------------------------------------------------------------------- #
# yield
# --------------------------------------------------------------------------- #
def test_duplicate_case_names_rejected_and_confidence_validated():
    mc = MonteCarloSpec.independent({"x": 0.0}, {"x": 1.0})
    cases = mc.sample(5, seed=1) + mc.sample(5, seed=2)            # same names mc_000000..
    spec = an.Spec("x", lambda r: r, maximum=1.0)
    with pytest.raises(ValueError, match="unique"):
        an.yield_analysis(lambda c: c.parameters["x"], [spec], cases=cases)
    with pytest.raises(ValueError, match="confidence"):
        an.wilson_interval(9, 10, confidence=95)
    with pytest.raises(ValueError, match="between"):
        an.wilson_interval(11, 10)
    study = ProcessStudy(ProcessCorner("tt", {"x": 0.0}), monte_carlo=mc, units={"x": "1"})
    with pytest.raises(ValueError, match="confidence"):
        an.yield_analysis(lambda c: c.parameters["x"], [spec], study=study, samples=5, confidence=1.0)


# --------------------------------------------------------------------------- #
# time domain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fraction", [0.3, 0.6, 0.8])
def test_long_delays_on_uniform_wavelength_sweeps(fraction):
    wl = np.asarray(px.linspace(1.54, 1.56, 2001))
    window = 1 / np.max(np.abs(np.diff(C_UM_PS / wl)))
    L = fraction * window * C_UM_PS / 4.2
    tau = 4.2 * L / C_UM_PS
    h = np.asarray(px.components.straight(wl=wl, length=L)[("o1", "o2")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # wrapped-delay notice
        t, ir = an.impulse_response(wl, h, oversample=8)
        r = an.pulse_response(wl, h, pulse_fwhm_ps=5.0)
    assert t[np.argmax(np.abs(ir))] == pytest.approx(tau, abs=2 * t[1])
    assert np.abs(ir).max() == pytest.approx(0.5, rel=0.02)
    assert r.delay_ps == pytest.approx(tau, rel=1e-4)
    assert r.broadening == pytest.approx(1.0, abs=1e-3)
    # an explicit hint gives the same branch without any warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r2 = an.pulse_response(wl, h, pulse_fwhm_ps=5.0, delay_hint_ps=tau * 1.01)
    assert r2.delay_ps == pytest.approx(tau, rel=1e-4)


@pytest.mark.parametrize("gdd", [0.8, 2.0, 5.0, 10.0])
def test_pure_dispersion_is_not_mistaken_for_a_delay(gdd):
    wl = np.linspace(1.50, 1.60, 20001)
    om = 2 * np.pi * (C_UM_PS / wl - C_UM_PS / 1.55)
    r = an.pulse_response(wl, np.exp(-0.5j * gdd * om**2), pulse_fwhm_ps=1.0, center_wl=1.55, n_samples=16384)
    assert r.broadening == pytest.approx(math.sqrt(1 + (4 * math.log(2) * gdd) ** 2), rel=1e-3)
    assert abs(r.delay_ps) < 1e-3


@pytest.mark.parametrize("tau", [-10.0, -3.0])
def test_advanced_pulses_wrap_correctly(tau):
    wl = np.linspace(1.50, 1.60, 20001)
    om = 2 * np.pi * (C_UM_PS / wl - C_UM_PS / 1.55)
    r = an.pulse_response(wl, np.exp(-1j * om * tau), pulse_fwhm_ps=1.0, center_wl=1.55, n_samples=16384)
    assert r.delay_ps == pytest.approx(tau, abs=1e-6)
    assert r.broadening == pytest.approx(1.0, abs=1e-6)


def test_short_time_window_gives_an_honest_error():
    wl = np.linspace(1.50, 1.60, 41)                                  # 1/df ~ 3 ps
    with pytest.raises(ValueError, match="time window"):
        an.pulse_response(wl, np.ones(41), pulse_fwhm_ps=1.0)


# --------------------------------------------------------------------------- #
# circuit
# --------------------------------------------------------------------------- #
def test_mesh_phase_loss_applies_to_every_shifter():
    n = 3
    prog = clements_decompose(np.eye(n))
    s = mzi_mesh(prog, phase_loss_db=0.1)(wl=1.55)
    # identity mesh: each mode crosses 2 shifters per MZI on its path plus the output screen
    S = np.array([[complex(s[(f"o{i + 1}", f"o{2 * n - j}")]) for i in range(n)] for j in range(n)])
    assert np.max(np.linalg.svd(S, compute_uv=False)) <= 10 ** (-0.1 / 20) + 1e-12


def test_clements_handles_subnormal_entries():
    a = 1e-310
    U = np.array([[math.sqrt(1 - a * a), a], [-a, math.sqrt(1 - a * a)]], dtype=complex)
    assert np.allclose(mesh_matrix(clements_decompose(U)), U, atol=1e-12)


def test_mesh_program_is_immutable_and_hashable():
    prog = clements_decompose(unitary_group.rvs(3, random_state=1))
    again = clements_decompose(unitary_group.rvs(3, random_state=1))
    assert prog == again and hash(prog) == hash(again)
    with pytest.raises(ValueError):
        prog.output_phases[0] = 1.0
    assert isinstance(MeshProgram(2, ((0, 1.0, 0.5),), [0.0, 0.0]).output_phases, np.ndarray)


def test_netlist_json_is_strict():
    nl = Netlist(name="x")
    nl.add("a", "straight", length=float("nan"))
    with pytest.raises(TypeError, match="not finite"):
        nl.to_json()
    nl = Netlist(name="y")
    nl.add("a", "straight", table={0: 1.0})
    with pytest.raises(TypeError, match="string mapping keys"):
        nl.to_dict()
    nl = Netlist(name="z")
    nl.add("a", "straight", length=5.0)
    nl.expose("i", ("a", "o1"))
    json.loads(nl.to_json())          # parses with the strict standard parser


def test_link_budget_count_and_band_average():
    lb = an.LinkBudget(source_dbm=0.0)
    with pytest.raises(ValueError, match="count"):
        lb.add("wg", 2.0, count=2.9)
    wl = np.linspace(1.54, 1.56, 2001)
    s = px.components.all_pass_ring(wl=wl, coupling=0.05, loss_db_cm=1.0)
    st = an.stage_from_sparameters("ring", s, ("o1", "o2"), worst_case=False)
    p = np.abs(np.asarray(s[("o1", "o2")])) ** 2
    assert st.loss_db == pytest.approx(-10 * np.log10(p.mean()))
