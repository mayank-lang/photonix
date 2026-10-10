"""Filter synthesis: prototypes, exact CROW synthesis and coupling-to-gap mapping."""
from __future__ import annotations

import math

import numpy as np
import pytest

import photonix as px
from photonix import synthesis as sy

_C = 299_792_458.0


def _drop_db(design, df_hz):
    s = design.model()(wl=design.wavelength(df_hz))
    return 10 * np.log10(px.to_numpy(px.power(s[("o1", "o3")])))


@pytest.mark.parametrize(("kind", "order", "ripple", "expected"), [
    ("butterworth", 2, None, [1.0, 1.414214, 1.414214, 1.0]),
    ("chebyshev", 2, 0.5, [1.0, 1.4029, 0.7071, 1.9841]),           # Matthaei-Young-Jones table 4.05-2(a)
    ("chebyshev", 5, 0.1, [1.0, 1.1468, 1.3712, 1.9750, 1.3712, 1.1468, 1.0]),
])
def test_lowpass_prototype_matches_tables(kind, order, ripple, expected):
    assert np.allclose(sy.lowpass_prototype(kind, order, ripple), expected, atol=1e-4)


def test_prototype_input_validation():
    with pytest.raises(ValueError):
        sy.lowpass_prototype("butterworth", 0)
    with pytest.raises(ValueError):
        sy.lowpass_prototype("chebyshev", 3)
    with pytest.raises(ValueError):
        sy.lowpass_prototype("elliptic", 3)


def test_fast_transfer_matrix_matches_netlist_simulation():
    k = (0.3, 0.05, 0.02, 0.07, 0.25)
    d = sy.synthesize_crow(4, 50e9, radius=10.0, refine=False)      # only for its resonant radius / FSR
    s = px.circuit.crow(couplings=k, radius=d.radius, neff=d.neff, ng=d.ng, loss_db_cm=0.0)
    phi = np.linspace(-math.pi, math.pi, 201)
    sim = px.to_numpy(px.power(s(wl=d.wavelength(phi / (2 * math.pi) * d.fsr_hz))[("o1", "o3")]))
    assert np.allclose(sim, sy.crow_drop_power(k, phi), atol=1e-10)


def test_single_ring_reproduces_add_drop_linewidth():
    bw = 20e9
    d = sy.synthesize_crow(1, bw, radius=8.0)
    k = d.couplings[0]
    assert d.couplings[0] == pytest.approx(d.couplings[1], rel=1e-12)
    # exact lossless add-drop: FWHM phase = 4 asin(K / (2 sqrt(1 - K)))
    fwhm_phi = 4 * math.asin(k / (2 * math.sqrt(1 - k)))
    assert fwhm_phi / (2 * math.pi) * d.fsr_hz == pytest.approx(bw, rel=1e-9)
    assert _drop_db(d, np.array([bw / 2]))[0] == pytest.approx(-10 * math.log10(2), abs=1e-9)


@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_butterworth_crow_is_exact_symmetric_and_maximally_flat(order):
    bw = 100e9
    d = sy.synthesize_crow(order, bw, radius=10.0)
    assert d.max_error_db < 1e-8
    assert np.allclose(d.couplings, d.couplings[::-1], rtol=1e-12)          # K_i = K_{N-i}
    edge = _drop_db(d, np.array([-bw / 2, bw / 2]))
    assert np.allclose(edge, -10 * math.log10(2), atol=1e-8)                # -3 dB at the band edges
    df = np.linspace(-0.05, 0.05, 11) * bw                                  # flat to O(x^(2N)) near centre
    x = np.sin(math.pi * df / d.fsr_hz) / math.sin(math.pi * bw / (2 * d.fsr_hz))
    assert np.allclose(_drop_db(d, df), -10 * np.log10(1 + x ** (2 * order)), atol=1e-9)


@pytest.mark.parametrize(("order", "ripple"), [(3, 0.1), (4, 0.5)])
def test_chebyshev_crow_is_equiripple(order, ripple):
    bw = 100e9
    d = sy.synthesize_crow(order, bw, kind="chebyshev", ripple_db=ripple, radius=10.0)
    assert d.max_error_db < 1e-8
    assert np.allclose(d.couplings, d.couplings[::-1], rtol=1e-12)
    half = math.sin(math.pi * bw / (2 * d.fsr_hz))

    def at(x):                                                              # detuning where sin(phi/2) = x * half
        return d.fsr_hz / math.pi * np.arcsin(np.asarray(x) * half)

    peaks = np.cos((2 * np.arange(1, order + 1) - 1) * math.pi / (2 * order))
    valleys = np.cos(np.arange(order + 1) * math.pi / order)               # includes both band edges
    assert np.allclose(_drop_db(d, at(peaks)), 0.0, atol=1e-8)              # reflectionless peaks
    assert np.allclose(_drop_db(d, at(valleys)), -ripple, atol=1e-8)        # equal ripple to the edges
    band = _drop_db(d, np.linspace(-bw / 2, bw / 2, 2001))
    assert band.min() >= -ripple - 1e-8 and band.max() <= 1e-8


def test_refinement_corrects_the_narrow_band_closed_form():
    exact = sy.synthesize_crow(3, 200e9, radius=10.0)
    approx = sy.synthesize_crow(3, 200e9, radius=10.0, refine=False)
    assert approx.max_error_db > 1e3 * exact.max_error_db
    # Little et al. are accurate for narrow bands: couplings agree to a few per cent
    narrow = sy.synthesize_crow(3, 5e9, radius=10.0)
    narrow0 = sy.synthesize_crow(3, 5e9, radius=10.0, refine=False)
    assert np.allclose(narrow.couplings, narrow0.couplings, rtol=0.05)


def test_wide_band_design_by_continuation():
    d = sy.synthesize_crow(4, 400e9, kind="chebyshev", ripple_db=0.2, fsr_hz=1.2e12)
    assert d.max_error_db < 1e-7
    assert d.fsr_hz == pytest.approx(1.2e12, rel=0.01)
    assert all(0 < k < 1 for k in d.couplings)


def test_design_is_resonant_at_wl0_and_validates_inputs():
    d = sy.synthesize_crow(2, 50e9, radius=7.3, wl0=1.31)
    assert (d.neff * 2 * math.pi * d.radius / d.wl0) == pytest.approx(round(d.neff * 2 * math.pi * d.radius / d.wl0))
    assert _drop_db(d, np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(d.target(np.array([0.0, d.bandwidth_hz / 2])), [1.0, 0.5])
    with pytest.raises(ValueError):
        sy.synthesize_crow(2, 50e9)                                          # neither radius nor fsr
    with pytest.raises(ValueError):
        sy.synthesize_crow(2, 50e9, radius=10.0, fsr_hz=1e12)
    with pytest.raises(ValueError):
        sy.synthesize_crow(2, 2e12, radius=10.0)                             # wider than FSR/2


@pytest.mark.parametrize(("target", "length"), [(0.02, 0.0), (0.05, 0.0), (0.1, 2.0), (0.3, 5.0)])
def test_coupling_to_gap_round_trip(target, length):
    gap = sy.coupling_to_gap(target, radius=10.0, length=length)
    assert 0.1 <= gap <= 0.4
    s = px.components.evanescent_coupler(gap=gap, length=length, bend_radius=10.0)
    assert float(px.power(s[("o1", "o3")])) == pytest.approx(target, abs=1e-10)


def test_coupling_to_gap_maps_a_whole_design():
    d = sy.synthesize_crow(3, 100e9, radius=10.0)
    gaps = [sy.coupling_to_gap(k, radius=d.radius, length=6.0) for k in d.couplings]
    assert gaps[0] == pytest.approx(gaps[-1]) and gaps[1] > gaps[0]          # weaker inner couplings: wider gaps
    with pytest.raises(ValueError, match="achievable range"):
        sy.coupling_to_gap(0.5, radius=10.0)                                 # point coupler is too weak
    with pytest.raises(ValueError, match="not monotone"):
        sy.coupling_to_gap(0.5, radius=10.0, length=30.0)                    # over-coupled at small gaps
    with pytest.raises(ValueError):
        sy.coupling_to_gap(1.5)
