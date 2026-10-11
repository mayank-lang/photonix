"""Time-domain (sample-mode) circuit simulator: elements, conversion and network stepping.

Validation follows the work-package plan: CW steady state against the
frequency-domain solver, 4th-order dt convergence of Lagrange interpolation,
pulse responses against FFT propagation, exact energy conservation and
reciprocity, the Sagnac mirror, gradients through ``lax.scan`` and throughput.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

import photonix as px
from photonix.circuit import Netlist
from photonix.core.constants import C0_UM_S
from photonix.timedomain import (
    FIRTD,
    CompositeTD,
    DelayLineTD,
    MemorylessTD,
    PoleResidueTD,
    TimeDomainCircuit,
    TimeVaryingTD,
    carrier_frequency,
    centered_frequencies,
    cw,
    design_fir,
    fir_from_model,
    gaussian_pulse,
    lagrange_taps,
    thiran_coefficients,
    to_td,
    to_td_time_varying,
)
from photonix.timedomain.network import _run_numpy

FC = carrier_frequency(1.55)
MODELS = {
    "coupler": px.components.directional_coupler,
    "straight": px.components.straight,
    "phase_shifter": px.components.phase_shifter,
}


def _wl(nu):
    return C0_UM_S / (FC + np.asarray(nu, dtype=float))


# --------------------------------------------------------------------------- #
# Netlists mirroring the photonix.circuit builders (checked against them below)
# --------------------------------------------------------------------------- #
def ring_netlist(*, coupling=0.2, radius=10.0, neff=2.4, ng=4.2, loss_db_cm=2.0):
    nl = Netlist(name="ring")
    nl.add("cpl", "coupler", coupling=coupling)
    nl.add("rw", "straight", length=2 * math.pi * radius, neff=neff, ng=ng, loss_db_cm=loss_db_cm)
    nl.connect(("cpl", "o3"), ("rw", "o1"))
    nl.connect(("rw", "o2"), ("cpl", "o2"))
    nl.expose("o1", ("cpl", "o1"))
    nl.expose("o2", ("cpl", "o4"))
    return nl


def crow_netlist(couplings, *, radius=10.0, neff=2.4, ng=4.2, loss_db_cm=2.0, wl0=1.55):
    n_rings = len(couplings) - 1
    wg = {"length": math.pi * radius, "neff": neff, "ng": ng, "loss_db_cm": loss_db_cm, "wl0": wl0}
    nl = Netlist(name="crow")
    for k, kappa in enumerate(couplings):
        nl.add(f"c{k}", "coupler", coupling=kappa)
    for r in range(1, n_rings + 1):
        nl.add(f"r{r}_right", "straight", **wg)
        nl.add(f"r{r}_left", "straight", **wg)
        nl.connect((f"c{r - 1}", "o4"), (f"r{r}_right", "o1"))
        nl.connect((f"r{r}_right", "o2"), (f"c{r}", "o3"))
        nl.connect((f"c{r}", "o2"), (f"r{r}_left", "o1"))
        nl.connect((f"r{r}_left", "o2"), (f"c{r - 1}", "o1"))
    nl.expose("o1", ("c0", "o2"))
    nl.expose("o2", ("c0", "o3"))
    top = f"c{n_rings}"
    nl.expose("o3", (top, "o1") if n_rings % 2 else (top, "o4"))
    nl.expose("o4", (top, "o4") if n_rings % 2 else (top, "o1"))
    return nl


def lattice_netlist(couplings, delta_lengths, phases, *, length=10.0, neff=2.4, ng=4.2, loss_db_cm=0.0):
    nl = Netlist(name="lattice_filter")
    for i, k in enumerate(couplings):
        nl.add(f"c{i}", "coupler", coupling=k)
    wg = {"neff": neff, "ng": ng, "loss_db_cm": loss_db_cm}
    for i, (dl, ph) in enumerate(zip(delta_lengths, phases, strict=True)):
        nl.add(f"top{i}", "straight", length=length + dl, **wg)
        nl.add(f"bot{i}", "straight", length=length, **wg)
        nl.add(f"ps{i}", "phase_shifter", phase0=ph, length=0.0)
        nl.connect((f"c{i}", "o4"), (f"top{i}", "o1"))
        nl.connect((f"top{i}", "o2"), (f"ps{i}", "o1"))
        nl.connect((f"ps{i}", "o2"), (f"c{i + 1}", "o1"))
        nl.connect((f"c{i}", "o3"), (f"bot{i}", "o1"))
        nl.connect((f"bot{i}", "o2"), (f"c{i + 1}", "o2"))
    last = len(couplings) - 1
    nl.expose("o1", ("c0", "o1"))
    nl.expose("o2", ("c0", "o2"))
    nl.expose("o4", (f"c{last}", "o4"))
    nl.expose("o3", (f"c{last}", "o3"))
    return nl


def sagnac_netlist(*, coupling=0.5, loop_length=50.0, loss_db_cm=0.0):
    nl = Netlist(name="sagnac_mirror")
    nl.add("c", "coupler", coupling=coupling)
    nl.add("loop", "straight", length=loop_length, loss_db_cm=loss_db_cm)
    nl.connect(("c", "o4"), ("loop", "o1"))
    nl.connect(("loop", "o2"), ("c", "o3"))
    nl.expose("o1", ("c", "o1"))
    nl.expose("o2", ("c", "o2"))
    return nl


LATTICE = dict(couplings=(0.5, 0.3, 0.5), delta_lengths=(40.0, 80.0), phases=(0.3, 1.1))
CROW3 = (0.2, 0.03, 0.03, 0.2)

CASES = {
    # name: (netlist, frequency-domain reference model, port pair)
    "all_pass_ring": (lambda: ring_netlist(coupling=0.1, loss_db_cm=3.0),
                      lambda: px.circuit.ring(coupling=0.1, loss_db_cm=3.0), ("o1", "o2")),
    "add_drop_ring": (lambda: crow_netlist((0.1, 0.15)),
                      lambda: (lambda wl: px.components.add_drop_ring(wl=wl, coupling1=0.1, coupling2=0.15)),
                      ("o1", "o3")),
    "crow3": (lambda: crow_netlist(CROW3), lambda: px.circuit.crow(couplings=CROW3), ("o1", "o3")),
    "lattice": (lambda: lattice_netlist(**LATTICE), lambda: px.circuit.lattice_filter(**LATTICE), ("o1", "o4")),
}


def _reference(name, nu):
    _, ref, pair = CASES[name]
    model = ref()
    s = model(wl=_wl(nu)) if name != "add_drop_ring" else model(_wl(nu))
    return np.asarray(s[pair], dtype=complex)


@pytest.mark.parametrize("name", sorted(CASES))
def test_netlists_reproduce_the_frequency_domain_builders(name):
    nl_fn, _, pair = CASES[name]
    nu = np.linspace(-0.4e12, 0.4e12, 17)
    s = px.circuit.circuit_from_netlist(nl_fn(), MODELS)(wl=_wl(nu))
    assert np.allclose(np.asarray(s[pair]), _reference(name, nu), atol=1e-12)


# --------------------------------------------------------------------------- #
# (1) CW steady state and 4th-order convergence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(CASES))
def test_cw_steady_state_matches_frequency_domain(name):
    nl_fn, _, (p_in, p_out) = CASES[name]
    dt, df = 0.1e-12, 23e9
    tdc = TimeDomainCircuit.from_netlist(nl_fn(), MODELS, carrier_wl=1.55, dt=dt)
    n = 20000                                  # 2 ns: > 20 photon lifetimes of every case
    src = cw(1e-3, n, dt, offset_hz=df)
    out = tdc.simulate({p_in: src})[p_out]
    s_td = complex(out.samples[-1]) / complex(src.samples[-1])
    s_late = complex(out.samples[-500]) / complex(src.samples[-500])
    s_fd = complex(_reference(name, [df])[0])
    assert abs(s_td - s_late) < 1e-9              # truly steady
    assert abs(s_td - s_fd) < 1e-6


def test_lagrange3_converges_at_fourth_order_in_dt():
    nl = ring_netlist(coupling=0.15, loss_db_cm=3.0)
    tau = 4.2 * 2 * math.pi * 10.0 / C0_UM_S                     # ring round trip n_g L / c
    df = 0.25e12
    s_fd = complex(np.asarray(px.circuit.ring(coupling=0.15, loss_db_cm=3.0)(wl=_wl([df]))[("o1", "o2")])[0])
    errs, dts = [], []
    for n in (4, 8, 16, 32):
        dt = tau / (n + 1.5)                    # fixed fractional part: a clean convergence study
        tdc = TimeDomainCircuit.from_netlist(nl, MODELS, dt=dt)
        s = complex(tdc.discrete_response([df])[("o1", "o2")][0])
        errs.append(abs(s - s_fd))
        dts.append(dt)
    slopes = np.diff(np.log(errs)) / np.diff(np.log(dts))
    assert np.all(np.abs(slopes - 4.0) < 0.3), slopes


def test_simulation_reproduces_the_discrete_response_exactly():
    tdc = TimeDomainCircuit.from_netlist(crow_netlist((0.3, 0.2, 0.3), loss_db_cm=200.0), MODELS, dt=0.1e-12)
    n = 4096
    h = tdc.impulse_response("o1", n)
    nu = np.fft.fftfreq(n, tdc.dt)
    resp = tdc.discrete_response(nu)
    for port, pair in (("o2", ("o1", "o2")), ("o3", ("o1", "o3"))):
        spec = np.fft.fft(np.asarray(h[port]))
        assert np.max(np.abs(spec - resp[pair])) < 1e-10


def test_numpy_runner_matches_the_jax_runner():
    tdc = TimeDomainCircuit.from_netlist(crow_netlist((0.3, 0.1, 0.3)), MODELS, dt=0.1e-12, interp="thiran3")
    pulse = gaussian_pulse(1e-12, 3000, 0.1e-12)
    out = tdc.simulate({"o1": pulse}, record=["o2", "o3"])
    st, arrays, _ = tdc._prepare(None, [tdc._ports["o2"], tdc._ports["o3"]], ("o1",))
    u = np.zeros((3000, len(tdc._ports)), dtype=complex)
    u[:, list(tdc._ports).index("o1")] = np.asarray(pulse.samples)
    ref = _run_numpy(st, arrays, u, None)
    assert np.allclose(np.asarray(out["o2"].samples), ref[:, 0], atol=1e-14)
    assert np.allclose(np.asarray(out["o3"].samples), ref[:, 1], atol=1e-14)


# --------------------------------------------------------------------------- #
# (2) Pulse / impulse responses against FFT propagation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["crow3", "lattice", "all_pass_ring"])
def test_pulse_response_matches_fft_propagation(name):
    nl_fn, _, (p_in, p_out) = CASES[name]
    dt, n = 0.1e-12, 16384
    pulse = gaussian_pulse(2e-12, n, dt)
    out = TimeDomainCircuit.from_netlist(nl_fn(), MODELS, dt=dt).simulate({p_in: pulse})[p_out]
    nu = np.fft.fftfreq(n, dt)
    ref = np.fft.ifft(np.fft.fft(np.asarray(pulse.samples)) * _reference(name, nu))
    got = np.asarray(out.samples)
    assert np.max(np.abs(got - ref)) < 2e-3 * np.max(np.abs(ref))


def test_impulse_peaks_match_analysis_impulse_response():
    # a lattice's impulse response is a few delayed taps; their times are the arm delays
    dt = 0.05e-12
    tdc = TimeDomainCircuit.from_netlist(lattice_netlist(**LATTICE), MODELS, dt=dt)
    h = np.abs(np.asarray(tdc.impulse_response("o1", 2048)["o4"]))
    wl = np.linspace(1.50, 1.60, 8001)
    t_ps, h_fd = px.analysis.impulse_response(wl, np.asarray(px.circuit.lattice_filter(**LATTICE)(wl=wl)[("o1", "o4")]))
    peak_td = np.argmax(h) * dt * 1e12
    peak_fd = float(t_ps[np.argmax(np.abs(h_fd))])
    assert abs(peak_td - peak_fd) < 2 * float(t_ps[1] - t_ps[0]) + dt * 1e12


# --------------------------------------------------------------------------- #
# (3) Delay lines: exact shifts, the 28.02 ps straight, dispersion
# --------------------------------------------------------------------------- #
def test_pure_integer_delay_shifts_exactly():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500) + 1j * rng.normal(size=500)
    for interp in ("lagrange3", "thiran3", "nearest"):
        tdc = TimeDomainCircuit({"d": DelayLineTD(tau_s=37e-12, interp=interp)}, {},
                                {"in": ("d", "o1"), "out": ("d", "o2")}, dt=1e-12)
        y = np.asarray(tdc.simulate({"in": x})["out"].samples)
        assert np.allclose(y[37:], x[:-37], atol=1e-13) and np.allclose(y[:37], 0.0)


def test_fractional_delay_filters():
    p = np.linspace(1.0, 2.0, 11)
    for v in p:
        taps = np.asarray(lagrange_taps(v, 3))
        assert abs(taps.sum() - 1.0) < 1e-14                       # unit DC gain
        assert abs(np.sum(np.arange(4) * taps) - v) < 1e-13        # exact first moment (delay at DC)
    w = np.linspace(0, np.pi, 64)
    for d in (2.6, 3.0, 3.4):
        a = np.asarray(thiran_coefficients(d, 3))
        z = np.exp(1j * w)
        num = sum(a[::-1][k] * z ** -k for k in range(4))
        den = sum(a[k] * z ** -k for k in range(4))
        assert np.allclose(np.abs(num / den), 1.0, atol=1e-13)    # all-pass at every frequency
        gd = -np.gradient(np.unwrap(np.angle(num / den)), w)
        assert abs(gd[0] - d) < 1e-3                               # group delay d at DC


def test_gaussian_pulse_through_2mm_straight_is_delayed_28_02_ps():
    dt = 0.05e-12
    tdc = TimeDomainCircuit.from_netlist(_one("straight", length=2000.0), dt=dt)
    pulse = gaussian_pulse(1e-12, 2000, dt)
    out = tdc.simulate({"in": pulse})["out"]
    t = pulse.t

    def centroid(w):
        p = np.asarray(w.power())
        return float(np.sum(t * p) / np.sum(p))

    delay_ps = (centroid(out) - centroid(pulse)) * 1e12
    assert abs(delay_ps - 4.2 * 2000.0 / C0_UM_S * 1e12) < 1e-4
    assert round(delay_ps, 2) == 28.02                          # analysis.pulse_response doctest value
    assert abs(out.energy() / pulse.energy() - 1.0) < 1e-5      # Lagrange amplitude droop only


def test_gvd_broadening_follows_the_gaussian_law():
    dt, t0_fwhm, b2l = 0.1e-12, 1.665e-12, 1.0e-24      # T0 = 1 ps, beta2 L = 1 ps^2
    el = DelayLineTD(tau_s=60e-12, beta2_l_s2=b2l)
    tdc = TimeDomainCircuit({"wg": el}, {}, {"in": ("wg", "o1"), "out": ("wg", "o2")}, dt=dt)
    pulse = gaussian_pulse(t0_fwhm, 3000, dt, t_center=40e-12)
    out = tdc.simulate({"in": pulse})["out"]
    t = pulse.t

    def rms(w):
        p = np.asarray(w.power())
        m = np.sum(t * p) / np.sum(p)
        return math.sqrt(np.sum((t - m) ** 2 * p) / np.sum(p))

    T0 = t0_fwhm / (2 * math.sqrt(math.log(2)))
    expected = math.sqrt(1.0 + (b2l / T0 ** 2) ** 2)
    assert abs(rms(out) / rms(pulse) - expected) < 1e-4
    assert abs(out.energy() / pulse.energy() - 1.0) < 1e-5     # all-pass in band


# --------------------------------------------------------------------------- #
# (4)-(6) Energy conservation, reciprocity, Sagnac mirror
# --------------------------------------------------------------------------- #
def test_lossless_network_conserves_energy_exactly():
    tdc = TimeDomainCircuit.from_netlist(crow_netlist((0.3, 0.1, 0.1, 0.3), loss_db_cm=0.0), MODELS,
                                         dt=0.1e-12, interp="thiran3")
    pulse = gaussian_pulse(0.5e-12, 40000, 0.1e-12)
    out = tdc.simulate({"o1": pulse})
    e_out = sum(w.energy() for w in out.values())
    assert abs(e_out / pulse.energy() - 1.0) < 1e-9


@pytest.mark.parametrize("name, pair", [("crow3", ("o1", "o3")), ("lattice", ("o1", "o4"))])
def test_time_domain_reciprocity(name, pair):
    tdc = TimeDomainCircuit.from_netlist(CASES[name][0](), MODELS, dt=0.1e-12)
    h12 = np.asarray(tdc.impulse_response(pair[0], 3000)[pair[1]])
    h21 = np.asarray(tdc.impulse_response(pair[1], 3000)[pair[0]])
    assert np.max(np.abs(h12)) > 1e-3
    assert np.max(np.abs(h12 - h21)) < 1e-13


def test_sagnac_mirror_reflects_fully_in_the_time_domain():
    tdc = TimeDomainCircuit.from_netlist(sagnac_netlist(coupling=0.5), MODELS, dt=0.05e-12, interp="thiran3")
    pulse = gaussian_pulse(1e-12, 2000, 0.05e-12)
    out = tdc.simulate({"o1": pulse})
    assert abs(out["o1"].energy() / pulse.energy() - 1.0) < 1e-9
    assert out["o2"].energy() / pulse.energy() < 1e-20


# --------------------------------------------------------------------------- #
# (7) Gradients and (8) throughput
# --------------------------------------------------------------------------- #
@requires_jax
def test_gradient_of_output_energy_wrt_coupling_matches_finite_differences():
    tdc = TimeDomainCircuit.from_netlist(crow_netlist((0.2, 0.15)), MODELS, dt=0.1e-12)
    pulse = gaussian_pulse(3e-12, 1500, 0.1e-12)

    def drop_energy(k):
        out = tdc.simulate({"o1": pulse}, record=["o3"], overrides={"c0": {"coupling": k}})
        return px.xp.sum(px.xp.abs(out["o3"].samples) ** 2) * 0.1e-12

    g = float(px.grad(drop_energy)(0.2))
    fd = float(finite_diff(lambda k: float(drop_energy(k)), 0.2, h=1e-5))
    assert abs(g - fd) <= 1e-6 * abs(fd)
    assert abs(g) > 0


@requires_jax
def test_prbs_through_a_four_ring_crow_is_fast():
    baud, sps = 10e9, 16
    dt = 1.0 / (baud * sps)
    radius = 2.5 * dt * C0_UM_S / 4.2 / math.pi          # half-ring delay of 2.5 samples
    tdc = TimeDomainCircuit.from_netlist(crow_netlist((0.3, 0.05, 0.03, 0.05, 0.3), radius=radius), MODELS, dt=dt)
    bits = np.random.default_rng(3).integers(0, 2, 2 ** 15)
    field = np.repeat(bits, sps).astype(complex) * 1e-2
    t = time.perf_counter()
    out = tdc.simulate({"o1": field})
    np.asarray(out["o3"].samples)
    first = time.perf_counter() - t
    t = time.perf_counter()
    out = tdc.simulate({"o1": field})
    np.asarray(out["o3"].samples)
    second = time.perf_counter() - t
    assert out["o3"].n_steps == 2 ** 15 * sps
    assert second < 5.0, (first, second)          # 524288 steps (compiled); first call includes JIT
    assert first < 20.0, (first, second)


# --------------------------------------------------------------------------- #
# Conversion rules
# --------------------------------------------------------------------------- #
def _one(model, **settings):
    nl = Netlist()
    nl.add("x", model, **settings)
    nl.expose("in", ("x", "o1"))
    nl.expose("out", ("x", "o2"))
    return nl


def test_waveguide_models_become_exact_delay_lines():
    c = px.components
    cases = [
        (c.straight, {"length": 500.0}, 500.0),
        (c.bend, {"radius": 5.0, "angle": 90.0}, 5.0 * math.pi / 2),
        (c.bend_euler, {"radius": 5.0, "angle": 90.0, "p": 0.5}, float(c.euler_bend_length(5.0, 90.0, 0.5))),
        (c.waveguide_route, {"length": 300.0, "n_bends": 2, "bend_loss_db": 0.1, "loss_db_cm": 3.0}, 300.0),
        (c.taper, {"length": 40.0}, 40.0),
    ]
    for model, settings, length in cases:
        el = to_td(model, settings, dt=0.1e-12)
        assert isinstance(el, DelayLineTD), model.__name__
        assert el.tau_s == pytest.approx(4.2 * length / C0_UM_S, rel=1e-9)
        t = complex(model(wl=1.55, **settings)[("o1", "o2")])
        assert abs(complex(el.amplitude) - t) < 1e-12


def test_flat_devices_become_memoryless_and_rings_their_structure():
    c = px.components
    for model in (c.directional_coupler, c.mmi1x2, c.mmi2x2, c.y_branch, c.crossing, c.terminator):
        assert isinstance(to_td(model, dt=0.1e-12), MemorylessTD), model.__name__
    assert isinstance(to_td(c.phase_shifter, {"phase0": 0.4, "length": 0.0}, dt=0.1e-12), MemorylessTD)
    for model, settings, pairs in (
        (c.all_pass_ring, {"coupling": 0.1, "radius": 10.0}, [("o1", "o2")]),
        (c.add_drop_ring, {"coupling1": 0.1, "coupling2": 0.2}, [("o1", "o2"), ("o1", "o3"), ("o4", "o3")]),
    ):
        el = to_td(model, settings, dt=0.1e-12)
        assert isinstance(el, CompositeTD)
        ports = {p: ("r", p) for p in el.ports}
        tdc = TimeDomainCircuit({"r": el}, {}, ports, dt=0.1e-12)
        nu = np.array([0.0, 2e10, -5e10])
        resp = tdc.discrete_response(nu)
        ref = model(wl=_wl(nu), **settings)
        for pair in pairs:
            assert abs(resp[pair][0] - complex(np.asarray(ref[pair])[0])) < 1e-12      # exact at the carrier
            assert np.max(np.abs(resp[pair] - np.asarray(ref[pair]))) < 1e-5            # interpolation error


def test_generic_models_get_passive_fir_fits():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gc = fir_from_model(px.components.grating_coupler, {"bandwidth": 0.004}, dt=0.1e-12)
        bragg = to_td(px.components.bragg_grating, {}, dt=0.5e-12)
    assert gc.added_delay_s > 0 and gc.fit_error < 1e-4
    assert isinstance(bragg, FIRTD) and bragg.fit_error < 1e-4
    for el, model, settings, dt in ((gc, px.components.grating_coupler, {"bandwidth": 0.004}, 0.1e-12),
                                    (bragg, px.components.bragg_grating, {}, 0.5e-12)):
        tdc = TimeDomainCircuit({"x": el}, {}, {p: ("x", p) for p in el.ports}, dt=dt)
        nu = centered_frequencies(512, dt)
        resp = tdc.discrete_response(nu)
        ref = model(wl=_wl(nu), **settings)
        lat = np.exp(-2j * np.pi * nu * el.added_delay_s)
        inband = np.abs(nu) * 2 * dt <= 0.5
        for pair in el.taps:
            err = np.abs(resp[pair] - np.asarray(ref[pair]) * lat)
            assert np.max(err[inband]) < 1e-4
        # strict passivity over the whole band: the largest singular value never exceeds 1
        mats = np.stack([[resp.get((pi, po), np.zeros(nu.size)) for pi in el.ports] for po in el.ports])
        sv = np.linalg.svd(np.moveaxis(mats, -1, 0), compute_uv=False)
        assert sv.max() <= 1.0 + 1e-12


def test_design_fir_error_bound_and_gain_bound():
    dt = 1e-12
    nu = centered_frequencies(2048, dt)
    target = 0.9 * np.exp(-2j * np.pi * nu * 20.37 * dt) / (1 + 1j * nu / 0.1e12)     # delay plus a pole
    n0, taps, err = design_fir(target, dt, tol=1e-5, n_bulk=20)
    assert err < 0.5e-5
    h = np.fft.fft(np.r_[taps, np.zeros(8192)])
    assert np.max(np.abs(h)) < 0.9 + 0.5e-5


def test_pole_residue_low_pass_is_exact_for_piecewise_linear_input():
    tau, dt = 20e-12, 1e-12
    lp = PoleResidueTD(("o1", "o2"), poles=[-1 / tau], residues={("o1", "o2"): [1 / tau]})
    tdc = TimeDomainCircuit({"lp": lp}, {}, {"in": ("lp", "o1"), "out": ("lp", "o2")}, dt=dt)
    y = np.asarray(tdc.simulate({"in": np.ones(400)})["out"].samples).real
    t = dt * np.arange(400)
    # FOH reads the step as a ramp over [-dt, 0]: the exact continuous response
    exact = 1.0 - tau / dt * (1.0 - math.exp(-dt / tau)) * np.exp(-t / tau)
    assert np.max(np.abs(y - exact)) < 1e-12


# --------------------------------------------------------------------------- #
# Time-varying (quasi-static) elements
# --------------------------------------------------------------------------- #
def test_quasi_static_phase_modulator_in_an_mzi():
    dt, n = 0.1e-12, 3000
    nl = Netlist(name="mzi_mod")
    nl.add("c1", "coupler", coupling=0.5).add("c2", "coupler", coupling=0.5)
    nl.add("top", "straight", length=100.0).add("bot", "straight", length=100.0)
    nl.add("pm", "phase_shifter", length=0.0, phase0=0.0)
    nl.connect(("c1", "o4"), ("top", "o1")).connect(("top", "o2"), ("pm", "o1")).connect(("pm", "o2"), ("c2", "o1"))
    nl.connect(("c1", "o3"), ("bot", "o1")).connect(("bot", "o2"), ("c2", "o2"))
    nl.expose("o1", ("c1", "o1")).expose("o4", ("c2", "o4")).expose("o3", ("c2", "o3"))
    tdc = TimeDomainCircuit.from_netlist(nl, MODELS, dt=dt, time_varying={"pm": {"phase0": "phi"}})
    assert tdc.controls == ("phi",)
    phi = np.where(np.arange(n) < 1500, 0.4, 2.0)
    out = tdc.simulate({"o1": cw(1e-3, n, dt)}, controls={"phi": phi})
    p_bar = np.asarray(out["o4"].power()) / 1e-3
    k = np.r_[400:1490, 1600:n]                        # away from the start and the step
    # equal arms: the bar port carries sin^2(phi/2) with the library's coupler convention
    assert np.allclose(p_bar[k], np.sin(phi[k] / 2) ** 2, atol=1e-9)


def test_time_varying_pn_modulator_matches_the_static_model_at_constant_bias():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        el = to_td_time_varying(px.components.pn_phase_shifter, {"length": 500.0},
                                control_params={"voltage": "v"}, dt=0.5e-12)
    tdc = TimeDomainCircuit({"pm": el}, {}, {"in": ("pm", "o1"), "out": ("pm", "o2")}, dt=0.5e-12)
    n = 2000
    out = tdc.simulate({"in": cw(1e-3, n, 0.5e-12)}, controls={"v": np.full(n, 2.0)})
    s_td = complex(out["out"].samples[-1]) / math.sqrt(1e-3)
    s_fd = complex(px.components.pn_phase_shifter(wl=1.55, length=500.0, voltage=2.0)[("o1", "o2")])
    assert abs(s_td - s_fd) < 1e-9          # generic FIRs are exact at the carrier


def test_time_varying_element_direct_construction():
    pm = TimeVaryingTD(("o1", "o2"), lambda t, v: {("o1", "o2"): np.exp(-1j * v), ("o2", "o1"): np.exp(-1j * v)},
                       controls=("v",))
    tdc = TimeDomainCircuit({"pm": pm}, {}, {"in": ("pm", "o1"), "out": ("pm", "o2")}, dt=1e-12)
    v = np.linspace(0, 3, 50)
    out = tdc.simulate({"in": np.ones(50)}, controls={"v": v})["out"]
    assert np.allclose(np.asarray(out.samples), np.exp(-1j * v), atol=1e-14)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_zero_delay_lossless_loop_is_rejected():
    nl = Netlist()
    nl.add("c", "coupler", coupling=0.0)
    nl.connect(("c", "o3"), ("c", "o2"))          # bar path closed on itself with no delay
    nl.expose("in", ("c", "o1")).expose("out", ("c", "o4"))
    tdc = TimeDomainCircuit.from_netlist(nl, MODELS, dt=1e-12)
    with pytest.raises(ValueError, match="zero-delay"):
        tdc.simulate({"in": np.ones(4)})


def test_delay_shorter_than_a_step_warns():
    with pytest.warns(UserWarning, match="shorter than one time step"):
        TimeDomainCircuit.from_netlist(_one("straight", length=5.0), dt=1e-12)


def test_inputs_are_validated():
    tdc = TimeDomainCircuit.from_netlist(_one("straight", length=200.0), dt=0.1e-12)
    with pytest.raises(KeyError):
        tdc.simulate({"nope": np.ones(3)})
    with pytest.raises(ValueError, match="dt"):
        tdc.simulate({"in": cw(1e-3, 10, 0.2e-12)})
    with pytest.raises(ValueError, match="carrier"):
        tdc.simulate({"in": cw(1e-3, 10, 0.1e-12, carrier_wl=1.31)})
    recs = tdc.simulate({"in": cw(1e-3, 50, 0.1e-12)}, record="all")
    assert set(recs) == {"x,o1", "x,o2"}
