"""Tests for EME-backed component models (taper, 1x2 MMI)."""
from __future__ import annotations

import photonix as px
from photonix.em import components as emc


def test_taper_low_loss_and_reciprocal():
    s = emc.taper(width1=0.5, width2=1.2, length=20.0, num_sections=18, num_modes=5, points=201)
    assert 0.9 < float(px.power(s[("o1", "o2")])) <= 1.0
    assert abs(s[("o1", "o2")] - s[("o2", "o1")]) < 1e-9  # reciprocal


def test_taper_exports_both_reflections():
    """S11/S22 must be exported -- dropping them fakes matched ports."""
    s = emc.taper(width1=0.5, width2=1.2, length=20.0, num_sections=18, num_modes=5, points=201)
    assert ("o1", "o1") in s and ("o2", "o2") in s
    # adiabatic taper: reflections exist but are small
    assert 0.0 <= float(px.power(s[("o1", "o1")])) < 0.05
    assert 0.0 <= float(px.power(s[("o2", "o2")])) < 0.05


def test_mmi_exports_reflections():
    s = emc.mmi1x2(width_mmi=2.5, length_mmi=29.5, gap=1.0, num_modes=10, points=261, half_window=4.0)
    for key in [("o1", "o1"), ("o2", "o2"), ("o3", "o3"), ("o2", "o3"), ("o3", "o2")]:
        assert key in s
    assert float(px.power(s[("o1", "o1")])) < 0.1
    # symmetric structure: o2/o3 self-reflections equal, cross terms reciprocal
    assert abs(s[("o2", "o2")] - s[("o3", "o3")]) < 1e-9
    assert abs(s[("o2", "o3")] - s[("o3", "o2")]) < 1e-9


def test_mmi_balanced_and_passive():
    s = emc.mmi1x2(width_mmi=2.5, length_mmi=29.5, gap=1.0, num_modes=10, points=261, half_window=4.0)
    p2 = float(px.power(s[("o1", "o2")]))
    p3 = float(px.power(s[("o1", "o3")]))
    assert abs(p2 - p3) < 1e-6          # symmetric split
    assert 0.0 < p2 + p3 <= 1.0 + 1e-9  # passive
    assert p2 + p3 > 0.6                # near self-imaging length (lateral-TM physics)


def test_eme_taper_drops_into_circuit():
    """An EME taper SDict composes in the circuit solver (cascade with a WG)."""
    import photonix.components as comp

    t = emc.taper(width1=0.5, width2=0.5, length=5.0, num_sections=6, num_modes=4, points=161)
    s = px.circuit.evaluate_circuit(
        {"tp": t, "wg": comp.straight(wl=1.55, length=10.0)},
        {("tp", "o2"): ("wg", "o1")},
        {"in0": ("tp", "o1"), "out0": ("wg", "o2")},
    )
    assert ("in0", "out0") in s
