"""Sensitivity, FOSM, corners, worst-case distance and importance sampling."""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import requires_jax
from scipy.stats import norm

import photonix as px
from photonix import analysis as an
from photonix.pdk import MonteCarloSpec, ProcessCorner, ProcessStudy


def test_sensitivity_matches_analytic_phase_derivative():
    L, wl = 120.0, 1.55

    def phase(p):
        s = px.components.straight(wl=wl, length=L, neff=p["neff"], ng=p["neff"])
        return -px.xp.angle(s[("o1", "o2")] * px.xp.exp(1j * 2 * px.xp.pi / wl * 2.4 * L))

    t = an.sensitivity(phase, {"neff": 2.4})
    assert t.gradient[0] == pytest.approx(2 * math.pi * L / wl, rel=1e-6)


@requires_jax
def test_ad_and_fd_agree_on_an_interferometer_fom():
    def fom(p):
        return px.power(px.circuit.mzi(delta_length=p["dl"], neff=p["neff"], coupling=p["k"])(wl=1.551)[("o1", "o4")])

    params = {"dl": 37.0, "neff": 2.41, "k": 0.47}
    ad = an.sensitivity(fom, params, method="ad")
    fd = an.sensitivity(fom, params, method="fd", fd_step=1e-4)
    assert ad.method == "ad" and fd.method == "fd"
    assert np.allclose(ad.gradient, fd.gradient, rtol=1e-6, atol=1e-9)
    assert ad.ranked()[0][0] in params


def test_fosm_exact_for_linear_and_second_order_for_quadratic():
    a = np.array([2.0, -3.0, 0.5])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.25]])
    nominal = {"x": 1.0, "y": 0.5, "z": -2.0}
    m = an.fosm(lambda p: 2 * p["x"] - 3 * p["y"] + 0.5 * p["z"], nominal, cov)
    assert m.mean == pytest.approx(a @ [1.0, 0.5, -2.0], abs=1e-12)
    assert m.std == pytest.approx(math.sqrt(a @ cov @ a), rel=1e-10)
    q = an.fosm(lambda p: p["x"] ** 2 + 3 * p["x"] * p["y"], nominal, cov, second_order=True)
    hess = np.array([[2.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # JAX gives the exact Hessian; the NumPy backend falls back to a central
    # second difference (step 1e-5), whose round-off is ~eps/h^2 ~ 2e-6.
    rel = 1e-6 if px.HAS_JAX else 1e-4
    assert q.mean_shift_2nd == pytest.approx(0.5 * np.trace(hess @ cov), rel=rel)


def test_fosm_accepts_monte_carlo_spec():
    mc = MonteCarloSpec.independent({"a": 1.0, "b": 2.0}, {"a": 0.1, "b": 0.2})
    m = an.fosm(lambda p: p["a"] + p["b"], {"b": 2.0, "a": 1.0}, mc)
    assert m.std == pytest.approx(math.sqrt(0.01 + 0.04))


def test_worst_case_distance_linear_matches_monte_carlo_yield():
    cov = np.array([[1.0, 0.3], [0.3, 0.5]])
    a = np.array([1.0, 2.0])
    spec = an.Spec("lin", lambda v: v, maximum=4.0)
    wc = an.worst_case_distance(lambda p: p["u"] + 2 * p["v"], spec, {"u": 0.0, "v": 0.5}, cov)
    beta = (4.0 - a @ [0.0, 0.5]) / math.sqrt(a @ cov @ a)
    assert wc.converged and wc.beta == pytest.approx(beta, rel=1e-6)
    mc = MonteCarloSpec(("u", "v"), np.array([0.0, 0.5]), cov)
    study = ProcessStudy(ProcessCorner("tt", {"u": 0.0, "v": 0.5}), monte_carlo=mc, units={"u": "1", "v": "1"})
    res = an.yield_analysis(lambda c: c.parameters["u"] + 2 * c.parameters["v"], [spec], study=study,
                            samples=20000, seed=3)
    lo, hi = res.yield_interval
    assert lo <= wc.yield_linear <= hi


def test_worst_case_distance_nonlinear_and_failing_nominal():
    spec = an.Spec("radius", lambda v: v, maximum=4.0)
    wc = an.worst_case_distance(lambda p: p["x"] ** 2 + p["y"] ** 2, spec, {"x": 0.3, "y": 0.0}, [1.0, 1.0],
                                n_starts=6)
    assert wc.beta == pytest.approx(2.0 - 0.3, abs=1e-5)       # nearest point of the radius-2 circle
    fail = an.worst_case_distance(lambda p: p["x"], an.Spec("x", lambda v: v, maximum=-1.0), {"x": 0.0}, [4.0])
    assert fail.beta == pytest.approx(-0.5, abs=1e-8)


def test_importance_sampling_resolves_rare_failures():
    spec = [an.Spec("x", lambda v: v, maximum=3.719)]          # 1 - Phi(3.719) = 1e-4
    truth = 1 - norm.cdf(3.719)
    r = an.importance_sampling_yield(lambda p: p["x"] + 0 * p["y"], spec, {"x": 0.0, "y": 0.0},
                                     [1.0, 1.0], samples=2000, seed=11)
    assert abs(r.failure_probability - truth) / truth < 0.15
    lo, hi = r.ci
    assert lo <= 1 - truth <= hi
    assert r.ess > 100
    # crude Monte Carlo with the same budget almost never sees a failure
    crude = np.random.default_rng(11).standard_normal(2000)
    assert (crude > 3.719).sum() <= 2


def test_corner_designs():
    nominal = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    deltas = {"a": 0.1, "b": 0.2, "c": 0.3}
    ff = an.corners(nominal, deltas)
    signs = {tuple(np.sign(np.round([c.parameters[k] - nominal[k] for k in deltas], 9))) for c in ff}
    assert len(ff) == 8 and len(signs) == 8
    assert all(c.parameters["d"] == 4.0 for c in ff)
    pb = an.corners(nominal, deltas, design="plackett_burman")
    m = np.array([[np.sign(c.parameters[k] - nominal[k]) for k in deltas] for c in pb])
    assert np.allclose(m.T @ m, len(pb) * np.eye(3)) and np.allclose(m.sum(axis=0), 0)
    assert len(an.corners(nominal, deltas, design="axial")) == 6
    hi, lo = an.corners(nominal, deltas, design="gradient_sign", fn=lambda p: p["a"] - p["b"] + p["c"])
    assert hi.parameters["a"] > 1.0 and hi.parameters["b"] < 2.0 and lo.parameters["c"] < 3.0
    res = an.yield_analysis(lambda c: c.parameters["a"], [an.Spec("a", lambda v: v, maximum=1.05)], cases=ff)
    assert sum(res.corner_pass.values()) == 4


@pytest.mark.parametrize("k", [2e-6, 3e-5, 1e-3, 0.7, 40.0])
def test_fd_step_is_relative_for_small_parameters(k):
    # A weak coupling ~1e-6 must not be stepped below zero (sqrt -> nan) or by a
    # large fraction of itself (~1% error) -- the only gradient path under NumPy.
    t = an.sensitivity(lambda p: px.xp.sqrt(p["k"]), {"k": k}, method="fd")
    assert t.method == "fd"
    assert t.gradient[0] == pytest.approx(0.5 / math.sqrt(k), rel=1e-7)
    m = an.fosm(lambda p: px.xp.sqrt(p["k"]), {"k": k}, [(0.05 * k) ** 2], method="fd", second_order=True)
    assert m.std == pytest.approx(0.05 * k * 0.5 / math.sqrt(k), rel=1e-7)
    assert m.mean_shift_2nd == pytest.approx(0.5 * (0.05 * k) ** 2 * (-0.25 * k ** -1.5), rel=1e-3)


def test_fd_gradient_of_lengths_in_metres_and_per_parameter_steps():
    def fn(p):
        return px.xp.sin(2 * math.pi * p["L"] / 1e-6) + 3.0 * p["a"]

    t = an.sensitivity(fn, {"L": 0.3e-6, "a": 0.0}, method="fd")
    assert t.gradient[0] == pytest.approx(2 * math.pi / 1e-6 * math.cos(2 * math.pi * 0.3), rel=1e-7)
    assert t.gradient[1] == pytest.approx(3.0, rel=1e-9)            # x == 0: absolute step
    t2 = an.sensitivity(fn, {"L": 0.3e-6, "a": 0.0}, method="fd", fd_step={"L": 1e-4})
    assert np.allclose(t2.gradient, t.gradient, rtol=1e-6)
    with pytest.raises(KeyError):
        an.sensitivity(fn, {"L": 0.3e-6, "a": 0.0}, method="fd", fd_step={"bogus": 1e-4})
    with pytest.raises(ValueError):
        an.sensitivity(fn, {"L": 0.3e-6, "a": 0.0}, method="fd", fd_step=0.0)


def test_fd_gradient_falls_back_to_one_sided_where_f_is_not_finite():
    def fn(p):                     # defined only for x >= 1 (python branch: not traceable)
        x = float(p["x"])
        return x * x if x >= 1.0 else float("nan")

    with pytest.warns(RuntimeWarning, match="one-sided"):
        t = an.sensitivity(fn, {"x": 1.0}, method="fd")
    assert t.gradient[0] == pytest.approx(2.0, rel=1e-8)


@requires_jax
def test_fd_matches_ad_on_crow_with_tiny_interior_couplings():
    d = px.synthesis.synthesize_crow(4, 1e9, fsr_hz=1e12)
    wl_edge = d.wavelength(0.4e9)

    def edge_drop(p):
        k = (p["k0"], p["k1"], p["k2"], p["k1"], p["k0"])
        s = px.circuit.crow(couplings=k, radius=d.radius, neff=d.neff, ng=d.ng, loss_db_cm=0.0)
        return px.power(s(wl=wl_edge)[("o1", "o3")])

    nominal = dict(zip(["k0", "k1", "k2"], d.couplings[:3], strict=False))
    assert min(nominal.values()) < 1e-5
    ad = an.sensitivity(edge_drop, nominal, method="ad")
    fd = an.sensitivity(edge_drop, nominal, method="fd")
    assert np.all(np.isfinite(fd.gradient))
    assert np.allclose(fd.gradient, ad.gradient, rtol=1e-5)


def test_worst_case_distance_two_sided_spec_returns_the_nearest_bound():
    # f = exp(x): the min bound 0.3 is closer in f (0.7 vs 0.8) but the max bound
    # 1.8 is closer in sigma (ln 1.8 = 0.588 vs -ln 0.3 = 1.204).
    spec = an.Spec("s", lambda v: v, minimum=0.3, maximum=1.8)
    wc = an.worst_case_distance(lambda p: px.xp.exp(p["x"]), spec, {"x": 0.0}, [1.0])
    assert wc.converged and wc.side == "max" and wc.bound == 1.8
    assert wc.beta == pytest.approx(math.log(1.8), abs=1e-6)
    assert wc.point["x"] == pytest.approx(math.log(1.8), abs=1e-6)
    # mirrored in x, and a spec where the min bound really is the nearer one
    mirrored = an.worst_case_distance(lambda p: px.xp.exp(-p["x"]), spec, {"x": 0.0}, [1.0])
    assert mirrored.side == "max" and mirrored.point["x"] == pytest.approx(-math.log(1.8), abs=1e-6)
    near_min = an.worst_case_distance(lambda p: px.xp.exp(p["x"]), an.Spec("s", lambda v: v, minimum=0.7,
                                      maximum=1.8), {"x": 0.0}, [1.0])
    assert near_min.side == "min" and near_min.beta == pytest.approx(-math.log(0.7), abs=1e-6)
    # a failing nominal reports the violated bound, with negative beta
    fail = an.worst_case_distance(lambda p: px.xp.exp(p["x"]), spec, {"x": 1.0}, [1.0])
    assert fail.side == "max" and fail.beta == pytest.approx(-(1.0 - math.log(1.8)), abs=1e-6)
    fail_lo = an.worst_case_distance(lambda p: px.xp.exp(p["x"]), spec, {"x": -2.0}, [1.0])
    assert fail_lo.side == "min" and fail_lo.beta == pytest.approx(-(math.log(0.3) + 2.0), abs=1e-6)
