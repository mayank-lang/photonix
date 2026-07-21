"""Tests for the differentiable circuit solver (correctness vs analytic + autodiff)."""
from __future__ import annotations

import numpy as np
from conftest import finite_diff, requires_jax

import photonix as px
import photonix.components as comp


def test_cascade_equals_single_waveguide(wl):
    s = px.circuit.evaluate_circuit(
        {"a": comp.straight(wl=wl, length=30.0), "b": comp.straight(wl=wl, length=70.0)},
        {("a", "o2"): ("b", "o1")},
        {"in0": ("a", "o1"), "out0": ("b", "o2")},
    )
    single = comp.straight(wl=wl, length=100.0)
    assert np.allclose(np.asarray(s[("in0", "out0")]), np.asarray(single[("o1", "o2")]), atol=1e-10)


def test_ring_solver_matches_analytic(wl):
    """The feedback-loop solver must reproduce the closed-form all-pass ring."""
    rc = px.circuit.ring(radius=10.0, coupling=0.2, loss_db_cm=2.0, neff=2.4, ng=4.2)
    Tr = np.asarray(px.power(rc(wl=wl)[("in0", "out0")]))
    Ta = np.asarray(px.power(comp.all_pass_ring(wl=wl, coupling=0.2, radius=10.0, loss_db_cm=2.0)[("o1", "o2")]))
    assert np.max(np.abs(Tr - Ta)) < 1e-10


def test_mzi_energy_conserved(wl):
    mzi = px.circuit.mzi(delta_length=40.0, coupling=0.5)
    s = mzi(wl=wl)
    total = np.asarray(px.power(s[("in0", "out0")]) + px.power(s[("in0", "out1")]))
    assert np.allclose(total, 1.0, atol=1e-9)


def test_mzi_ports_match_analytic(wl):
    """circuit.mzi and components.mzi must agree on which port is bar/cross.

    Uses coupling != 0.5 so bar and cross fringes are distinguishable.
    """
    s_c = px.circuit.mzi(delta_length=40.0, coupling=0.3)(wl=wl)
    s_a = comp.mzi(wl=wl, delta_length=40.0, coupling=0.3)
    for pc, pa in [(("in0", "out0"), ("in0", "out0")), (("in0", "out1"), ("in0", "out1"))]:
        Tc = np.asarray(px.power(s_c[pc]))
        Ta = np.asarray(px.power(s_a[pa]))
        assert np.max(np.abs(Tc - Ta)) < 1e-10


def test_analytic_mzi_loss_matches_circuit_for_both_dl_signs(wl):
    """Lossy analytic MZI == circuit MZI normalized to the shorter arm, dL = +/-."""
    from photonix.core.units import db_per_cm_to_alpha_um

    cpl, L, loss = 0.3, 50.0, 500.0
    alpha = float(np.asarray(db_per_cm_to_alpha_um(loss)))
    for dl in (+40.0, -40.0):
        s_c = px.circuit.mzi(delta_length=dl, length=L, coupling=cpl, loss_db_cm=loss)(wl=wl)
        Tc = np.asarray(px.power(s_c[("in0", "out0")])) / np.exp(-2 * alpha * min(L, L + dl))
        Ta = np.asarray(px.power(comp.mzi(wl=wl, delta_length=dl, coupling=cpl, loss_db_cm=loss)[("in0", "out0")]))
        assert np.max(np.abs(Tc - Ta)) < 1e-10


@requires_jax
def test_solver_gradient_matches_fd():
    import jax

    def fom(dl):
        return px.power(px.circuit.mzi(delta_length=dl)(wl=1.55)[("in0", "out0")])

    g = float(jax.grad(fom)(40.0))
    assert abs(g - float(finite_diff(fom, 40.0))) < 1e-3


@requires_jax
def test_solver_is_jittable():
    import jax

    f = jax.jit(lambda dl: px.power(px.circuit.mzi(delta_length=dl)(wl=1.55)[("in0", "out0")]))
    assert np.isfinite(float(f(25.0)))
