"""Tests for inverse design: objectives, optimizers, differentiability."""
from __future__ import annotations

from conftest import finite_diff, requires_jax

import photonix as px
import photonix.components as comp
import photonix.optim as opt


def test_objective_positive():
    s = comp.directional_coupler(coupling=0.4)
    loss = float(opt.target_transmission(s, ("o1", "o3"), 0.5))
    assert loss > 0


@requires_jax
def test_objective_gradient_matches_fd():
    import jax

    def loss(k):
        s = comp.directional_coupler(coupling=k)
        return opt.target_transmission(s, ("o1", "o3"), 0.5)

    g = float(jax.grad(loss)(0.4))
    assert abs(g - float(finite_diff(loss, 0.4))) < 1e-3


@requires_jax
def test_adam_minimizes_convex():
    res = opt.adam(lambda p: (p["x"] - 3.0) ** 2, {"x": 0.0}, steps=300, lr=0.05)
    assert abs(float(res.params["x"]) - 3.0) < 1e-2
    assert res.history[-1] < res.history[0]


@requires_jax
def test_inverse_design_coupler():
    loss = opt.make_loss(comp.directional_coupler, opt.target_transmission,
                         wl=1.55, port=("o1", "o3"), target=0.25)
    res = opt.adam(loss, {"coupling": 0.5}, steps=300, lr=0.02)
    got = float(px.power(comp.directional_coupler(coupling=float(res.params["coupling"]))[("o1", "o3")]))
    assert abs(got - 0.25) < 1e-3
