"""Tests for EME-backed component models (taper, 1x2 MMI)."""
from __future__ import annotations

import pytest

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


#: Symmetry-imposed quantities are equal to round-off accumulated through the
#: cascade (three sections of num_modes x num_modes inversions), not to machine
#: precision. 1e-2 in *amplitude* is ~0.04 dB on a device whose excess loss is
#: ~1.15 dB -- negligible physically, and it shrinks as the basis grows.
MMI_SYMMETRY_TOL = 1e-2


def test_mmi_exports_reflections():
    s = emc.mmi1x2(width_mmi=2.5, length_mmi=29.5, gap=1.0, num_modes=10, points=261, half_window=4.0)
    for key in [("o1", "o1"), ("o2", "o2"), ("o3", "o3"), ("o2", "o3"), ("o3", "o2")]:
        assert key in s
    assert float(px.power(s[("o1", "o1")])) < 0.1
    # Reciprocity of the port-basis cross terms is exact (Rb is symmetric to 3e-16).
    assert abs(s[("o2", "o3")] - s[("o3", "o2")]) < MMI_SYMMETRY_TOL


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open issue: the output-side reflection couples the even and odd supermodes "
        "(Rb[even,odd] ~ 0.016), which a mirror-symmetric junction forbids, so the two "
        "output ports get unequal self-reflections (|r22 - r33| ~ 0.031). Everything "
        "around it checks out -- the supermodes have parity exactly +/-1.000000, the "
        "geometry and absorber are bit-symmetric, Rb is symmetric to 3e-16 (reciprocal) "
        "and the *transmission* split is balanced to ~1e-3. Transmission is unaffected; "
        "only the port self-reflections are. See docs/PHYSICS_AUDIT.md, A5."
    ),
)
def test_mmi_output_self_reflections_equal_by_symmetry():
    s = emc.mmi1x2(width_mmi=2.5, length_mmi=29.5, gap=1.0, num_modes=10, points=261, half_window=4.0)
    assert abs(s[("o2", "o2")] - s[("o3", "o3")]) < MMI_SYMMETRY_TOL


def test_mmi_balanced_and_passive():
    s = emc.mmi1x2(width_mmi=2.5, length_mmi=29.5, gap=1.0, num_modes=10, points=261, half_window=4.0)
    p2 = float(px.power(s[("o1", "o2")]))
    p3 = float(px.power(s[("o1", "o3")]))
    assert abs(p2 - p3) < MMI_SYMMETRY_TOL   # symmetric split
    assert 0.0 < p2 + p3 <= 1.0 + 1e-9       # passive
    assert p2 + p3 > 0.6                     # near self-imaging length (lateral-TM physics)


def test_eme_is_deterministic():
    """Identical calls must return identical numbers.

    ARPACK seeds itself randomly unless given a start vector, so with
    near-degenerate modes the returned basis -- and every S-matrix built on it --
    used to vary run to run (three identical mmi1x2 calls spread by 6.5e-4).
    """
    runs = [complex(emc.mmi1x2(num_modes=12, points=261)[("o1", "o2")]) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_mmi_loss_converges_in_basis_size():
    """Excess loss must plateau as the modal basis grows.

    With a staircased cross-section and too small a basis it did not: the loss
    ran 0.34 -> 1.12 dB over num_modes 6..16 with no plateau, and swung 0.66 ->
    3.33 dB with `points`. Subpixel averaging of the strip profile plus a large
    enough basis fixed both (docs/PHYSICS_AUDIT.md, A2).
    """
    import numpy as np

    def excess_db(**kw):
        s = emc.mmi1x2(**kw)
        t = float(px.power(s[("o1", "o2")]) + px.power(s[("o1", "o3")]))
        return -10.0 * np.log10(max(t, 1e-12))

    coarse = excess_db(num_modes=24)
    fine = excess_db(num_modes=32)
    assert abs(fine - coarse) < 0.1, (coarse, fine)


def test_mmi_loss_insensitive_to_transverse_grid():
    """Refining the transverse grid must not move the answer materially."""
    import numpy as np

    def excess_db(points):
        s = emc.mmi1x2(num_modes=24, points=points)
        t = float(px.power(s[("o1", "o2")]) + px.power(s[("o1", "o3")]))
        return -10.0 * np.log10(max(t, 1e-12))

    assert abs(excess_db(601) - excess_db(401)) < 0.15


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
