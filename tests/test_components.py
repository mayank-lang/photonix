"""Tests for the component model library (physics validation + differentiability)."""
from __future__ import annotations

import numpy as np

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