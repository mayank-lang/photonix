"""Regression tests for the package-wide port convention and its legacy aliases.

photonix names optical ports `o1 … oN` in every component and circuit builder.
`components.mzi` (`in0`/`out0`/`out1`) and `add_drop_ring` (`i1`/`t1`/`d2`) used
to disagree, which is exactly the kind of drift that is invisible until a
circuit silently misbehaves. These tests pin the convention, and pin the two
properties that make the compatibility aliases safe:

* looking a legacy name up returns the canonical value, and
* aliases are *not* stored as extra entries -- otherwise `ports_of` would report
  phantom terminals and the solver would build an oversized S-matrix.
"""
from __future__ import annotations

import numpy as np
import pytest

import photonix as px
import photonix.components as c
from photonix.core import as_sdense, is_passive, is_reciprocal, ports_of
from photonix.core.types import AliasedSDict


@pytest.mark.parametrize("name", sorted(c.MODELS))
def test_every_model_uses_canonical_port_names(name):
    """Ports must be exactly o1..oN with no gaps and no alternative scheme."""
    ports = ports_of(c.MODELS[name](wl=1.55))
    assert ports == [f"o{i + 1}" for i in range(len(ports))], (name, ports)


@pytest.mark.parametrize("name", sorted(c.MODELS))
def test_every_model_is_reciprocal_and_passive(name):
    s = c.MODELS[name](wl=1.55)
    assert is_reciprocal(s), name
    assert is_passive(s), name


@pytest.mark.parametrize(
    ("model", "legacy", "canonical"),
    [
        (lambda wl: c.mzi(wl=wl), ("in0", "out0"), ("o1", "o2")),
        (lambda wl: c.mzi(wl=wl), ("in0", "out1"), ("o1", "o3")),
        (lambda wl: c.add_drop_ring(wl=wl), ("i1", "t1"), ("o1", "o2")),
        (lambda wl: c.add_drop_ring(wl=wl), ("i1", "d2"), ("o1", "o3")),
        (lambda wl: px.circuit.mzi()(wl=wl), ("in0", "out0"), ("o1", "o4")),
        (lambda wl: px.circuit.mzi()(wl=wl), ("in0", "out1"), ("o1", "o3")),
        (lambda wl: px.circuit.ring()(wl=wl), ("in0", "out0"), ("o1", "o2")),
    ],
)
def test_legacy_names_resolve_to_canonical(model, legacy, canonical):
    s = model(px.linspace(1.54, 1.56, 51))
    assert legacy in s
    assert np.allclose(px.to_numpy(s[legacy]), px.to_numpy(s[canonical]))


@pytest.mark.parametrize(
    ("model", "n_ports"),
    [
        (lambda: c.mzi(wl=1.55), 3),
        (lambda: c.add_drop_ring(wl=1.55), 4),
        (lambda: px.circuit.mzi()(wl=1.55), 4),
        (lambda: px.circuit.ring()(wl=1.55), 2),
    ],
)
def test_aliases_do_not_create_phantom_ports(model, n_ports):
    """Aliases resolve on lookup only -- they must never be stored as entries.

    If they were, the dense S-matrix would gain rows/columns for terminals that
    do not physically exist and every circuit using the model would be wrong.
    """
    s = model()
    assert len(ports_of(s)) == n_ports
    S, port_map = as_sdense(s)
    assert S.shape[-1] == n_ports
    assert set(port_map) == {f"o{i + 1}" for i in range(n_ports)}


def test_aliased_sdict_semantics():
    s = AliasedSDict({("o1", "o2"): 1.0 + 2.0j}, aliases={"in0": "o1", "out0": "o2"})
    assert s[("in0", "out0")] == s[("o1", "o2")]
    assert ("in0", "out0") in s
    assert s.get(("in0", "out0")) == 1.0 + 2.0j
    assert s.get(("nope", "o2")) is None
    assert list(s) == [("o1", "o2")]          # iteration stays canonical
    assert s.copy().aliases == s.aliases
    with pytest.raises(KeyError):
        s[("nope", "o2")]


def test_add_drop_ring_is_a_four_port():
    """It used to expose only three terminals, leaving the add port undefined."""
    s = c.add_drop_ring(wl=1.55)
    assert ports_of(s) == ["o1", "o2", "o3", "o4"]
    # Light entering o1 and light entering o4 circulate opposite ways round the
    # ring, so these paths must be absent rather than merely small.
    assert ("o1", "o4") not in s
    assert ("o2", "o3") not in s


def test_add_drop_ring_resonance_and_fsr():
    """Drop peaks where through dips, power is conserved, and the FSR is right."""
    radius, ng = 10.0, 4.2
    wl = px.linspace(1.540, 1.560, 40001)
    s = c.add_drop_ring(wl=wl, coupling1=0.15, coupling2=0.15, radius=radius,
                        ng=ng, loss_db_cm=3.0)
    T = np.asarray(px.to_numpy(px.power(s[("o1", "o2")])))
    D = np.asarray(px.to_numpy(px.power(s[("o1", "o3")])))

    assert (T + D).max() <= 1.0 + 1e-9        # passive
    assert D.max() > 0.9 and T.min() < 0.05   # strong resonant drop
    assert abs(int(np.argmax(D)) - int(np.argmin(T))) < 10

    peaks = np.asarray(px.to_numpy(wl))[1:-1][
        (D[1:-1] > D[:-2]) & (D[1:-1] > D[2:]) & (D[1:-1] > 0.5 * D.max())
    ]
    fsr = float(np.diff(peaks).mean())
    assert abs(fsr - 1.55**2 / (ng * 2 * np.pi * radius)) < 0.05 * fsr


def test_add_drop_add_branch_swaps_the_couplers():
    """`o4 -> o3` is the through response with t1 and t2 exchanged."""
    wl = px.linspace(1.54, 1.56, 501)
    sym = c.add_drop_ring(wl=wl, coupling1=0.2, coupling2=0.2)
    assert np.allclose(px.to_numpy(sym[("o1", "o2")]), px.to_numpy(sym[("o4", "o3")]))

    asym = c.add_drop_ring(wl=wl, coupling1=0.30, coupling2=0.05)
    assert not np.allclose(px.to_numpy(asym[("o1", "o2")]), px.to_numpy(asym[("o4", "o3")]))
    # ... but the drop path stays reciprocal.
    assert np.allclose(px.to_numpy(asym[("o4", "o2")]), px.to_numpy(asym[("o1", "o3")]))


def test_analytic_and_circuit_mzi_agree_on_canonical_ports():
    """The two MZI implementations must label their bar port the same way."""
    wl = px.linspace(1.50, 1.60, 401)
    analytic = px.to_numpy(px.power(c.mzi(wl=wl, delta_length=20.0)[("o1", "o2")]))
    circuit = px.to_numpy(px.power(px.circuit.mzi(delta_length=20.0)(wl=wl)[("o1", "o4")]))
    assert np.allclose(analytic, circuit, atol=1e-10)
