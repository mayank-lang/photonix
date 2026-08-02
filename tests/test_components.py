"""Tests for the component model library (physics validation + differentiability)."""
from __future__ import annotations

import numpy as np
import pytest

import photonix as px
import photonix.components as c


def test_directional_coupler_unitary(wl):
    s = c.directional_coupler(wl=wl, coupling=0.37)
    t = np.asarray(px.power(s[("o1", "o4")]))
    k = np.asarray(px.power(s[("o1", "o3")]))
    assert np.allclose(t + k, 1.0, atol=1e-12)


def test_directional_coupler_reciprocal():
    s = c.directional_coupler(coupling=0.5)
    assert np.allclose(np.asarray(s[("o1", "o3")]), np.asarray(s[("o3", "o1")]))


def test_straight_phase_and_loss():
    wl0, L, neff = 1.55, 100.0, 2.4
    s = c.straight(wl=wl0, length=L, neff=neff, ng=neff, loss_db_cm=0.0)
    expected_phase = (2 * np.pi / wl0 * neff * L) % (2 * np.pi)
    got = (-np.angle(complex(s[("o1", "o2")]))) % (2 * np.pi)
    assert abs(got - expected_phase) < 1e-6
    # 100 dB/cm over 100 um = 1 dB -> power 10^-0.1
    s2 = c.straight(wl=wl0, length=L, loss_db_cm=100.0)
    assert abs(float(px.power(s2[("o1", "o2")])) - 10 ** (-0.1)) < 1e-6


def test_solver_bend_loss_is_independent_of_handedness(monkeypatch):
    """Clockwise and counter-clockwise bends radiate equally; neither may gain."""
    from types import SimpleNamespace

    import photonix.em.fde_vector as fde_vector

    monkeypatch.setattr(
        fde_vector,
        "bend_loss_fullvector",
        lambda **_kwargs: SimpleNamespace(n_eff=2.4 + 1e-5j, loss_db_per_90deg=1.2),
    )
    ccw = c.bend_from_solver(angle=90.0)
    cw = c.bend_from_solver(angle=-90.0)
    p_ccw = float(px.power(ccw[("o1", "o2")]))
    p_cw = float(px.power(cw[("o1", "o2")]))
    assert p_cw == pytest.approx(p_ccw)
    assert p_cw == pytest.approx(10.0 ** (-1.2 / 10.0))


def test_grating_peak_wavelength(wl):
    s = c.grating_coupler(wl=wl, wl0=1.55)
    i = int(np.argmax(np.asarray(px.power(s[("o1", "o2")]))))
    assert abs(float(wl[i]) - 1.55) < 2e-3


def test_mmi1x2_energy():
    s = c.mmi1x2()
    assert abs(float(px.power(s[("o1", "o2")]) + px.power(s[("o1", "o3")])) - 1.0) < 1e-12


def test_ring_has_resonance():
    # Near-critical coupling (self-coupling ~ round-trip loss) gives a deep,
    # high-contrast all-pass notch. A dense sweep resolves the narrow resonance.
    wl = px.linspace(1.5, 1.6, 40001)
    T = np.asarray(px.power(c.all_pass_ring(wl=wl, coupling=0.05, radius=10.0, loss_db_cm=50.0)[("o1", "o2")]))
    assert T.min() < 0.1          # deep resonant dip
    assert T.max() - T.min() > 0.8


def test_uncoupled_lossless_rings_have_finite_transparent_bus_at_exact_resonance(monkeypatch):
    """A disconnected ring cannot affect either bus, even at its eigenfrequency."""
    import photonix.components.resonators as resonators

    # Pin the round trip to z=exp(-i*phi)=1 exactly.  Constructing phi from a
    # finite radius normally leaves a few ulps of trigonometric round-off, which
    # would hide the removable 0/0 in the old closed form.
    monkeypatch.setattr(resonators, "_round_trip", lambda *_args: (1.0, 0.0))

    all_pass = c.all_pass_ring(
        wl=1.55, coupling=0.0, radius=10.0, loss_db_cm=0.0,
    )
    assert np.isfinite(complex(all_pass[("o1", "o2")]))
    assert complex(all_pass[("o1", "o2")]) == 1.0 + 0.0j

    add_drop = c.add_drop_ring(
        wl=1.55, coupling1=0.0, coupling2=0.0, radius=10.0,
        loss_db_cm=0.0,
    )
    assert complex(add_drop[("o1", "o2")]) == 1.0 + 0.0j
    assert complex(add_drop[("o4", "o3")]) == 1.0 + 0.0j
    assert complex(add_drop[("o1", "o3")]) == 0.0 + 0.0j

    # Below machine epsilon, sqrt(1-c) rounds to one.  The linewidth must still
    # come from c itself: an infinitesimally coupled lossless all-pass has a pi
    # phase flip at exact resonance, and a symmetric add-drop transfers all
    # resonant power to the drop port.
    weak = 1e-20
    weak_all_pass = c.all_pass_ring(wl=1.55, coupling=weak)
    assert complex(weak_all_pass[("o1", "o2")]) == pytest.approx(-1.0 + 0.0j)
    weak_add_drop = c.add_drop_ring(wl=1.55, coupling1=weak, coupling2=weak)
    assert complex(weak_add_drop[("o1", "o2")]) == pytest.approx(0.0 + 0.0j, abs=1e-12)
    assert abs(complex(weak_add_drop[("o1", "o3")])) == pytest.approx(1.0)
