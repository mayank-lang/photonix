"""FIR lattice (cascaded-MZI) synthesis: peeling, targets, spectral factorization."""
from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import finite_diff, requires_jax

import photonix as px
from photonix import synthesis as sy


def _poly(coeffs, omega):
    """Evaluate coefficients in z^-1 at z^-1 = exp(-j omega)."""
    return np.polyval(np.asarray(coeffs)[::-1], np.exp(-1j * np.asarray(omega)))


def _laurent(h):
    h = np.asarray(h, dtype=complex)
    return np.convolve(h, np.conj(h[::-1]))


def _circuit(design, omega):
    """Netlist simulation of ``design.model()`` divided by the common arm delay (independent straights)."""
    wl = design.wavelength(np.asarray(omega) / (2 * np.pi) * design.fsr_hz)
    s = design.model()(wl=wl)
    t_l = px.components.straight(wl=wl, length=design.length, neff=design.neff, ng=design.ng,
                                 wl0=design.wl0)[("o1", "o2")]
    common = np.asarray(t_l) ** len(design.delays)
    return np.asarray(s[("o1", "o3")]) / common, np.asarray(s[("o1", "o4")]) / common


# --------------------------------------------------------------------------- #
# Forward models
# --------------------------------------------------------------------------- #
def test_coefficients_are_the_fourier_series_of_the_response():
    rng = np.random.default_rng(11)
    ks, phs, dls = rng.uniform(0.05, 0.95, 5), rng.uniform(0, 2 * np.pi, 4), (1, 2, 0, 3)
    pair = sy.lattice_coefficients(ks, phs, delays=dls)
    m = 16                                                            # > degree 6: no aliasing
    omega = 2 * np.pi * np.arange(m) / m
    resp = sy.lattice_response(ks, phs, omega, delays=dls)
    for got, want in zip(pair, resp, strict=True):
        series = np.fft.ifft(np.asarray(want))                       # coefficient of z^-n = exp(-j n omega)
        assert np.allclose(series[: got.size], got, atol=1e-14)
        assert np.allclose(series[got.size:], 0.0, atol=1e-14)
    assert np.allclose(np.abs(np.asarray(resp.cross)) ** 2 + np.abs(np.asarray(resp.bar)) ** 2, 1.0, atol=1e-14)


def test_single_stage_is_the_mzi():
    omega = np.linspace(-np.pi, np.pi, 101)
    cross, bar = sy.lattice_response([0.5, 0.5], [0.0], omega)
    assert np.allclose(np.abs(np.asarray(cross)) ** 2, np.cos(omega / 2) ** 2, atol=1e-14)
    assert np.allclose(np.abs(np.asarray(bar)) ** 2, np.sin(omega / 2) ** 2, atol=1e-14)
    d = sy.synthesize_lattice(*sy.maxflat_halfband(1), delta_length=30.0)
    assert d.couplings == pytest.approx((0.5, 0.5), abs=1e-15) and d.delays == (1,)


def test_forward_model_input_validation():
    with pytest.raises(ValueError, match="one more coupling"):
        sy.lattice_response([0.5, 0.5], [0.0, 1.0], 0.0)
    with pytest.raises(ValueError, match="one more coupling"):
        sy.lattice_coefficients([0.5, 0.5, 0.5], [0.0])
    with pytest.raises(ValueError, match="delays"):
        sy.lattice_coefficients([0.5, 0.5], [0.0], delays=(1.5,))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sy.lattice_coefficients([0.5, 1.2], [0.0])


# --------------------------------------------------------------------------- #
# Peeling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", range(1, 9))
def test_round_trip_recovers_couplings_and_phases(n):
    rng = np.random.default_rng(100 + n)
    ks, phs = rng.uniform(0.05, 0.95, n + 1), rng.uniform(0, 2 * np.pi, n)
    cross, bar = sy.lattice_coefficients(ks, phs)
    ga, gb = np.exp(1j * rng.uniform(-np.pi, np.pi, 2))               # arbitrary output phases on the target
    d = sy.synthesize_lattice(gb * cross, ga * bar, delta_length=25.0)
    # theta in (0, pi/2) with the a_0/b_0 phase convention makes the peel unique when 0 < K < 1:
    # the same couplings and stage phases come back; only the two output phases are gauge.
    assert d.delays == (1,) * n
    assert np.allclose(d.couplings, ks, atol=1e-12)
    assert np.allclose(np.exp(1j * np.array(d.phases)), np.exp(1j * phs), atol=1e-12)
    assert np.exp(-1j * d.common_phase) == pytest.approx(gb, abs=1e-12)
    assert np.exp(-1j * (d.common_phase + d.output_phase)) == pytest.approx(ga, abs=1e-12)
    assert d.max_error < 1e-12
    omega = np.linspace(-np.pi, np.pi, 257)
    real = d.response(omega)
    want = d.target(omega)
    assert np.allclose(np.abs(real.cross) ** 2, np.abs(want.cross) ** 2, atol=1e-12)
    assert np.allclose(np.abs(real.bar) ** 2, np.abs(want.bar) ** 2, atol=1e-12)
    assert np.allclose(np.exp(-1j * d.common_phase) * real.cross, want.cross, atol=1e-12)
    assert np.allclose(np.exp(-1j * (d.common_phase + d.output_phase)) * real.bar, want.bar, atol=1e-12)


def test_netlist_simulation_matches_the_synthesized_response_over_a_full_fsr():
    rng = np.random.default_rng(5)
    cross, bar = sy.lattice_coefficients(rng.uniform(0.1, 0.9, 6), rng.uniform(0, 2 * np.pi, 5))
    omega = np.linspace(-np.pi, np.pi, 401)
    for d in (sy.synthesize_lattice(cross, bar, fsr_hz=200e9, wl0=1.31, length=7.0),
              sy.interleaver(3, delta_length=40.0)):                  # includes merged 2 dL stages
        sim_cross, sim_bar = _circuit(d, omega)
        model_cross, model_bar = d.response(omega)
        assert np.allclose(sim_cross, model_cross, atol=1e-10)
        assert np.allclose(sim_bar, model_bar, atol=1e-10)
        assert np.allclose(np.exp(-1j * d.common_phase) * sim_cross, d.target(omega).cross, atol=1e-10)
        assert round(d.neff * d.delta_length / d.wl0) == pytest.approx(d.neff * d.delta_length / d.wl0, abs=1e-9)


def test_degenerate_pairs_use_null_or_full_couplers():
    # z^-1 times an MZI (a_0 = b_0 = 0): the first coupler sends everything through the delayed arm
    d = sy.synthesize_lattice([0.0, -0.5j, -0.5j], [0.0, -0.5, 0.5], delta_length=10.0)
    assert d.couplings[0] == pytest.approx(0.0, abs=1e-24) and d.max_error < 1e-15
    # a pure bar delay z^-2 (cross = 0 everywhere): every coupler is a straight-through
    d = sy.synthesize_lattice([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], delta_length=10.0)
    assert np.allclose(d.couplings, 0.0, atol=1e-24) and d.max_error < 1e-15
    # full cross with a delayed bar partner: K = 1 everywhere it matters
    d = sy.synthesize_lattice([0.0, 1.0], [0.0, 0.0], delta_length=10.0)
    assert d.max_error < 1e-15


def test_physical_unit_delay_and_fsr():
    d = sy.synthesize_lattice(*sy.maxflat_halfband(2), fsr_hz=100e9)
    assert d.fsr_hz == pytest.approx(100e9, rel=5e-4)                  # rounded to an integer neff dL / wl0
    assert d.fsr_hz == pytest.approx(299_792_458.0 / (d.ng * d.delta_length * 1e-6), rel=1e-14)
    assert d.delta_lengths == (2 * d.delta_length, d.delta_length)
    df = np.array([0.0, d.fsr_hz / 2, d.fsr_hz])
    assert np.allclose(d.omega(d.wavelength(df)), [0.0, np.pi, 2 * np.pi], atol=1e-9)


def test_synthesis_input_validation():
    cross, bar = sy.maxflat_halfband(2)
    with pytest.raises(ValueError, match="exactly one"):
        sy.synthesize_lattice(cross, bar)
    with pytest.raises(ValueError, match="exactly one"):
        sy.synthesize_lattice(cross, bar, fsr_hz=1e11, delta_length=10.0)
    with pytest.raises(ValueError, match="not power complementary"):
        sy.synthesize_lattice(cross, 1.01 * bar, delta_length=10.0)
    with pytest.raises(ValueError, match="length >= 2"):
        sy.synthesize_lattice([1.0], [0.0], delta_length=10.0)
    # common vanishing top coefficients are trimmed (the pair is really of degree 1)
    d = sy.synthesize_lattice([0.5, 0.5, 0.0], [0.5, -0.5, 0.0], delta_length=10.0)
    assert d.delays == (1,) and d.couplings == pytest.approx((0.5, 0.5)) and d.max_error < 1e-15
    with pytest.raises(ValueError, match="degree 0"):
        sy.synthesize_lattice([0.6, 0.0], [0.8, 0.0], delta_length=10.0)
    with pytest.raises(ValueError, match="positive"):
        sy.synthesize_lattice(cross, bar, delta_length=-5.0)


# --------------------------------------------------------------------------- #
# Maximally flat half-band (interleaver)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", range(1, 7))
def test_maxflat_halfband_is_the_daubechies_pair(order):
    cross, bar = sy.maxflat_halfband(order)
    assert cross.size == 2 * order and np.isrealobj(cross)
    omega = np.linspace(0, 2 * np.pi, 1001)
    pb, pa = np.abs(_poly(cross, omega)) ** 2, np.abs(_poly(bar, omega)) ** 2
    assert np.allclose(pa + pb, 1.0, atol=1e-12)
    x = np.sin(omega / 2) ** 2
    daub = (1 - x) ** order * sum(math.comb(order - 1 + k, k) * x**k for k in range(order))
    assert np.allclose(pb, daub, atol=1e-12)
    assert np.allclose(pb, np.abs(_poly(bar, omega + np.pi)) ** 2, atol=1e-12)      # half-band mirror
    # Maximal flatness: |B|^2 = r_0 + 2 sum r_k cos(k w) is even, and its derivatives of order
    # 2p at w = 0 are (-1)^p 2 sum_k k^(2p) r_k, which must vanish for p = 1 .. N - 1 (order 2N - 1 flat).
    r = _laurent(cross)[cross.size - 1:].real
    k = np.arange(r.size)
    for p in range(1, order):
        assert abs(np.sum(k ** (2 * p) * r)) < 1e-10 * np.sum(k ** (2 * p) * np.abs(r))
    if order > 1:                                                     # ... and the next one does not
        assert abs(np.sum(k ** (2 * order) * r)) > 1e-3
    d = sy.synthesize_lattice(cross, bar, delta_length=20.0)
    assert d.max_error < 1e-13
    assert d.delays == (2,) * (order - 1) + (1,)                      # every other coupler synthesizes to K = 0
    assert np.allclose(np.exp(1j * np.array(d.phases)), [1.0] + [-1.0] * (order - 1) if order > 1 else [1.0],
                       atol=1e-10)                                    # real taps: stage phases 0 or pi
    d2 = sy.synthesize_lattice(cross, delta_length=20.0)              # minimum-phase bar instead of B(-z)
    assert d2.max_error < 1e-12 and len(d2.couplings) == 2 * order
    assert np.allclose(np.abs(d2.response(omega).cross) ** 2, pb, atol=1e-12)


def test_maxflat_matches_published_daubechies_coefficients():
    # Daubechies, Ten Lectures on Wavelets (SIAM 1992), Table 6.1 (normalized sum h_n = sqrt 2).
    table = {
        2: [.4829629131445341, .8365163037378079, .2241438680420134, -.1294095225512604],
        3: [.3326705529500825, .8068915093110924, .4598775021184914, -.1350110200102546,
            -.0854412738820267, .0352262918857095],
        4: [.2303778133088964, .7148465705529154, .6308807679298587, -.0279837694168599,
            -.1870348117190931, .0308413818355607, .0328830116668852, -.0105974017850690],
    }
    for order, h in table.items():
        assert np.allclose(math.sqrt(2) * sy.maxflat_halfband(order).cross, h, atol=1e-14)
    # closed forms: D4 and D6 (Daubechies, Comm. Pure Appl. Math. 41, 909, 1988)
    s3 = math.sqrt(3)
    assert np.allclose(sy.maxflat_halfband(2).cross, np.array([1 + s3, 3 + s3, 3 - s3, 1 - s3]) / 8, atol=1e-15)
    r, s = math.sqrt(10), math.sqrt(5 + 2 * math.sqrt(10))
    d6 = np.array([1 + r + s, 5 + r + 3 * s, 10 - 2 * r + 2 * s, 10 - 2 * r - 2 * s, 5 + r - 3 * s, 1 + r - s]) / 32
    assert np.allclose(sy.maxflat_halfband(3).cross, d6, atol=1e-15)


def test_order_two_interleaver_closed_form():
    # Derivation: with couplers C(t0), C(t1), C(t2), stages w = z^-2 (phase 0) and v = -z^-1 (phase pi),
    #   B = -j [ c2 c1 s0 + s2 s1 s0 z^-1 + c2 s1 c0 z^-2 - s2 c1 c0 z^-3 ].
    # t0 = 75, t1 = 60, t2 = 45 degrees give B = -j [1+r3, 3+r3, 3-r3, 1-r3]/8 = -j D4/sqrt2 exactly
    # (e.g. c1 s0 / sqrt2 = (1/2)(r6+r2)/4/r2 = (1+r3)/8). So K = sin^2 = ((2+r3)/4, 3/4, 1/2).
    d = sy.interleaver(2, delta_length=50.0)
    assert d.couplings == pytest.approx(((2 + math.sqrt(3)) / 4, 0.75, 0.5), abs=1e-14)
    assert d.delays == (2, 1)
    assert np.allclose(np.exp(1j * np.array(d.phases)), [1.0, -1.0], atol=1e-13)
    cross, _ = sy.lattice_coefficients(d.couplings, [0.0, math.pi], delays=(2, 1))
    assert np.allclose(1j * cross, sy.maxflat_halfband(2).cross, atol=1e-15)


def test_interleaver_separates_odd_and_even_channels():
    d = sy.interleaver(4, fsr_hz=100e9, wl0=1.5501)
    df = np.array([0.0, 50e9, 100e9, -50e9])                          # channels on a 50 GHz grid
    s = d.model()(wl=d.wavelength(df))
    cross, bar = px.to_numpy(px.power(s[("o1", "o3")])), px.to_numpy(px.power(s[("o1", "o4")]))
    assert np.allclose(cross, [1, 0, 1, 0], atol=1e-12) and np.allclose(bar, [0, 1, 0, 1], atol=1e-12)
    # flat-top: within +-10 % of the channel spacing, crosstalk below -25 dB and insertion loss < 0.01 dB
    near = np.linspace(-5e9, 5e9, 41)
    s = d.model()(wl=d.wavelength(near))
    assert np.max(px.to_numpy(px.power(s[("o1", "o4")]))) < 10 ** (-25 / 10)
    assert np.min(px.to_numpy(px.power(s[("o1", "o3")]))) > 10 ** (-0.01 / 10)
    # the channel shape is the Daubechies polynomial, through the full netlist: crosstalk x^N P(1-x)
    x = np.sin(np.pi * near / d.fsr_hz) ** 2
    leak = x**4 * sum(math.comb(3 + k, k) * (1 - x) ** k for k in range(4))
    assert np.allclose(px.to_numpy(px.power(s[("o1", "o4")])), leak, atol=1e-13)
    mzi = sy.interleaver(1, fsr_hz=100e9)                             # a plain MZI: crosstalk sin^2(pi df / FSR)
    s1 = mzi.model()(wl=mzi.wavelength(near))
    assert np.allclose(px.to_numpy(px.power(s1[("o1", "o4")])), np.sin(np.pi * near / mzi.fsr_hz) ** 2, atol=1e-13)
    assert 10 * math.log10(np.max(px.to_numpy(px.power(s1[("o1", "o4")])))) > -17       # vs < -25 dB above


def test_maxflat_input_validation():
    with pytest.raises(ValueError):
        sy.maxflat_halfband(0)
    with pytest.raises(ValueError):
        sy.maxflat_halfband(2.5)


# --------------------------------------------------------------------------- #
# Spectral factorization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("n", "complex_taps"), [(1, False), (3, True), (8, False), (8, True), (15, True)])
def test_spectral_factor_recovers_minimum_phase_filters(n, complex_taps):
    rng = np.random.default_rng(n + 7 * complex_taps)
    if complex_taps:
        roots = rng.uniform(0.1, 0.95, n) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    else:                                                             # real filter: conjugate root pairs
        pairs = rng.uniform(0.1, 0.95, n // 2) * np.exp(1j * rng.uniform(0.1, np.pi - 0.1, n // 2))
        roots = np.concatenate([pairs, np.conj(pairs), rng.uniform(-0.9, 0.9, n % 2)])
    h_true = 1.7 * np.poly(roots)                                     # minimum phase, h_0 > 0
    if not complex_taps:
        h_true = np.real(h_true)
    h, err = sy.spectral_factor(_laurent(h_true))
    assert np.isrealobj(h) == (not complex_taps)
    assert err < 1e-10 * np.sum(np.abs(h_true) ** 2)
    assert np.allclose(h, h_true, atol=1e-10)                         # the unique minimum-phase factor


@pytest.mark.parametrize("w0", [0.3, 1.7, np.pi])
@pytest.mark.parametrize("mult", [1, 2])
def test_spectral_factor_with_zeros_on_the_unit_circle(w0, mult):
    rng = np.random.default_rng(int(10 * w0) + mult)
    g = np.poly(rng.uniform(0.2, 0.8, 3) * np.exp(1j * rng.uniform(-3, 3, 3)))
    h_true = g
    for _ in range(mult):                                             # R has a 2*mult-fold root at exp(j w0)
        h_true = np.convolve(h_true, [1.0, -np.exp(1j * w0)])
    h, err = sy.spectral_factor(_laurent(h_true))
    assert err < 1e-12
    assert np.allclose(h * np.exp(-1j * np.angle(h_true[0])), h_true, atol=1e-6)


def test_spectral_factor_of_the_maxflat_complement():
    # 1 - |B|^2 = x^N P(1 - x) has a 2N-fold zero at z = 1: the clustering/projection path.
    for order in range(1, 9):
        cross, bar = sy.maxflat_halfband(order)
        r = -_laurent(cross)
        r[cross.size - 1] += 1
        h, err = sy.spectral_factor(r)
        assert err < 1e-12
        omega = np.linspace(0, np.pi, 501)
        assert np.allclose(np.abs(_poly(h, omega)) ** 2, np.abs(_poly(bar, omega)) ** 2, atol=1e-12)
        assert np.allclose(sy.complementary_bar(cross), h, atol=1e-12)


def test_spectral_factor_from_a_callable_and_input_validation():
    h, err = sy.spectral_factor(lambda w: 2 - np.cos(w), 1)           # |Q|^2 of the D4 derivation
    assert np.allclose(h, [(1 + math.sqrt(3)) / 2, (1 - math.sqrt(3)) / 2], atol=1e-14) and err < 1e-14
    h, _ = sy.spectral_factor(lambda w: np.full(w.shape, 4.0), 0)
    assert np.allclose(h, [2.0])
    with pytest.raises(ValueError, match="degree"):
        sy.spectral_factor(lambda w: 2 - np.cos(w))
    with pytest.raises(ValueError, match="not a trigonometric polynomial"):
        sy.spectral_factor(lambda w: 1 / (2 - np.cos(w)), 3)
    with pytest.raises(ValueError, match="negative"):
        sy.spectral_factor([1.0, 1.0, 1.0])                            # 1 + 2 cos w dips below 0
    with pytest.raises(ValueError, match="Hermitian"):
        sy.spectral_factor([0.5j, 2.0, 0.5j])
    with pytest.raises(ValueError, match="odd length"):
        sy.spectral_factor([1.0, 2.0])
    with pytest.raises(ValueError, match="positive mean"):
        sy.spectral_factor([0.0, 0.0, 0.0])


def test_complementary_bar_of_an_arbitrary_cross():
    rng = np.random.default_rng(2)
    b = rng.normal(size=7) + 1j * rng.normal(size=7)
    omega = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    for peak in (0.9, 1.0):                                           # peak 1: a double zero of 1 - |B|^2
        fine = np.linspace(0, 2 * np.pi, 200001)
        cross = b * math.sqrt(peak) / np.max(np.abs(_poly(b, fine)))
        bar = sy.complementary_bar(cross)
        assert np.allclose(np.abs(_poly(bar, omega)) ** 2 + np.abs(_poly(cross, omega)) ** 2, 1.0, atol=1e-9)
        d = sy.synthesize_lattice(cross, delta_length=15.0)
        assert d.max_error < 1e-9 and len(d.couplings) == 7
    with pytest.raises(ValueError, match="> 1"):
        sy.complementary_bar([0.8, 0.8])
    assert np.array_equal(sy.complementary_bar([0.0, 1.0]), [0.0, 0.0])      # all-pass cross: no bar light
    d = sy.synthesize_lattice([0.0, 1.0], delta_length=10.0)
    assert d.couplings == pytest.approx((0.0, 1.0), abs=1e-15) and d.max_error < 1e-15     # delayed arm, then cross


# --------------------------------------------------------------------------- #
# Equiripple (Parks-McClellan) target
# --------------------------------------------------------------------------- #
def _alternation(err):
    """Local maxima of |err| on one band, band edges included, as (index, value) pairs."""
    a = np.abs(err)
    idx = [0, *(np.flatnonzero((a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:])) + 1), a.size - 1]
    return [(i, err[i]) for i in idx]


@pytest.mark.parametrize(("degree", "wp", "ws", "weight"), [(7, 0.4, 0.6, 1.0), (12, 0.3, 0.5, 4.0)])
def test_equiripple_lowpass_is_a_factored_minimax_design(degree, wp, ws, weight):
    from scipy.signal import remez

    wp, ws = wp * np.pi, ws * np.pi
    cross, bar = sy.equiripple_lowpass(degree, wp, ws, weight=weight)
    omega = np.linspace(0, np.pi, 100001)
    pb, pa = np.abs(_poly(cross, omega)) ** 2, np.abs(_poly(bar, omega)) ** 2
    assert np.allclose(pa + pb, 1.0, atol=1e-12)
    # touches 0 and 1 (double zeros of |B|^2 and |A|^2; the 3e-5 grid only resolves them to ~1e-9)
    assert pb.min() == pytest.approx(0.0, abs=1e-8) and pb.max() == pytest.approx(1.0, abs=1e-8)
    # |B|^2 is an affine image of an independent remez run: fit it, the residual is rounding only
    g = remez(2 * degree + 1, [0, wp, ws, np.pi], [1, 0], weight=[1, weight], fs=2 * np.pi, maxiter=200)
    big_g = np.real(np.exp(1j * degree * omega) * _poly(g, omega))
    (scale, offset), *_ = np.linalg.lstsq(np.stack([big_g, np.ones_like(big_g)], axis=1), pb, rcond=None)
    assert np.allclose(scale * big_g + offset, pb, atol=1e-12)
    # Chebyshev alternation: the weighted error W (D - G) of the minimax design reaches its maximum,
    # with alternating sign, at >= degree + 2 points of the two bands (equal to remez's ~1 % tolerance)
    stop, pas = omega >= ws, omega <= wp
    pts = _alternation(1.0 - big_g[pas]) + _alternation(-weight * big_g[stop])
    peak = max(abs(v) for _, v in pts)
    extremal = [v for _, v in pts if abs(v) > 0.99 * peak]
    assert len(extremal) >= degree + 2
    assert all(np.sign(extremal[i]) != np.sign(extremal[i + 1]) for i in range(len(extremal) - 1))
    d = sy.synthesize_lattice(cross, bar, fsr_hz=400e9)
    assert d.max_error < 1e-12
    sim_cross, _ = _circuit(d, omega[::1000])
    assert np.allclose(np.abs(sim_cross) ** 2, pb[::1000], atol=1e-10)


@pytest.mark.parametrize(("degree", "wp", "ws", "weight"), [
    (1, 0.2, 0.4, 1.0), (2, 0.2, 0.4, 10.0), (3, 0.05, 0.1, 1.0), (5, 0.05, 0.1, 10.0), (8, 0.05, 0.1, 10.0),
    (1, 0.7, 0.9, 1.0),
])
def test_equiripple_lowpass_without_overshoot(degree, wp, ws, weight):
    # For low degrees, narrow pass bands or heavy weights the minimax G does not ripple
    # about both 0 and 1 (G_max < 1 or G_min > 0): the affine map still gives a valid
    # [0, 1] response (it used to raise).
    from scipy.signal import remez

    g = remez(2 * degree + 1, [0, wp * np.pi, ws * np.pi, np.pi], [1, 0], weight=[1, weight], fs=2 * np.pi,
              maxiter=200)
    omega = np.linspace(0, np.pi, 4001)
    big_g = np.real(np.exp(1j * degree * omega) * _poly(g, omega))
    assert big_g.max() < 1.0 or big_g.min() > 0.0                     # the case under test
    cross, bar = sy.equiripple_lowpass(degree, wp * np.pi, ws * np.pi, weight=weight)
    pb, pa = np.abs(_poly(cross, omega)) ** 2, np.abs(_poly(bar, omega)) ** 2
    assert np.allclose(pa + pb, 1.0, atol=1e-12)
    assert pb.min() > -1e-12 and pb.max() < 1 + 1e-12 and pb.max() == pytest.approx(1.0, abs=1e-6)
    d = sy.synthesize_lattice(cross, bar, delta_length=20.0)
    assert d.max_error < 1e-10


@pytest.mark.parametrize("degree", [2, 4, 14])
@pytest.mark.parametrize(("wp", "ws"), [(0.4, 0.6), (0.45, 0.55)])
def test_half_band_equiripple_of_even_degree_synthesizes(degree, wp, ws):
    # Half-band (wp + ws = pi) designs of even degree have zero top taps in both
    # cross and bar; synthesize_lattice trims them instead of raising.
    cross, bar = sy.equiripple_lowpass(degree, wp * np.pi, ws * np.pi)
    assert abs(cross[-1]) + abs(bar[-1]) < 1e-14
    d = sy.synthesize_lattice(cross, bar, fsr_hz=100e9)
    assert sum(d.delays) < degree and d.max_error < 1e-12
    omega = np.linspace(0, 2 * np.pi, 257)
    rc, rb = d.response(omega)
    assert np.allclose(np.abs(rc) ** 2, np.abs(_poly(cross, omega)) ** 2, atol=1e-12)
    assert np.allclose(np.abs(rb) ** 2, np.abs(_poly(bar, omega)) ** 2, atol=1e-12)


@pytest.mark.parametrize("zeros", [
    [1, 1, 1, 0.98, 0.98],                                            # off-circle pair next to a triple zero
    [1, 1, 1, 0.9, 0.9],
    [-1, -1, -1, -0.97],
    [-1, -1, -1, -0.97, -0.97],
])
def test_spectral_factor_is_minimum_phase_next_to_unit_circle_zeros(zeros):
    # The roots of R scatter into one ring around the high-order unit-circle zero and
    # swallow the nearby off-circle zeros; the factor used to come back with 1/0.98
    # in place of 0.98 (right power, wrong phase). Now the unit-circle zeros are
    # divided out of R exactly and the rest factored separately.
    h_true = np.real(np.poly(zeros))
    h, err = sy.spectral_factor(_laurent(h_true))
    assert err < 1e-10
    assert h[0] == pytest.approx(1.0, abs=1e-6)                       # |h_0| is largest for minimum phase
    assert np.allclose(h, h_true, atol=1e-6)


def test_equiripple_input_validation():
    with pytest.raises(ValueError):
        sy.equiripple_lowpass(5, 0.6 * np.pi, 0.4 * np.pi)
    with pytest.raises(ValueError):
        sy.equiripple_lowpass(0, 0.4 * np.pi, 0.6 * np.pi)


# --------------------------------------------------------------------------- #
# Differentiability
# --------------------------------------------------------------------------- #
@requires_jax
def test_lattice_response_gradient_matches_finite_differences():
    import jax

    ks0 = np.array([0.3, 0.6, 0.45, 0.7])
    ph0 = np.array([0.4, 2.0, 5.1])
    omega = np.linspace(-1.0, 2.5, 9)

    def loss(ks, ph, w):
        cross, bar = sy.lattice_response(ks, ph, w, delays=(1, 2, 1))
        return px.xp.sum(px.xp.abs(cross) ** 2 * px.xp.cos(w)) + px.xp.sum(px.xp.real(bar) * px.xp.sin(2 * w))

    gk, gp, gw = jax.grad(loss, argnums=(0, 1, 2))(px.xp.asarray(ks0), px.xp.asarray(ph0), px.xp.asarray(omega))
    for i in range(ks0.size):
        e = np.eye(ks0.size)[i]
        fd = finite_diff(lambda t, e=e: float(loss(ks0 + t * e, ph0, omega)), 0.0, h=1e-6)
        assert float(gk[i]) == pytest.approx(fd, rel=1e-6, abs=1e-8)
    for i in range(ph0.size):
        e = np.eye(ph0.size)[i]
        fd = finite_diff(lambda t, e=e: float(loss(ks0, ph0 + t * e, omega)), 0.0, h=1e-6)
        assert float(gp[i]) == pytest.approx(fd, rel=1e-6, abs=1e-8)
    fd = finite_diff(lambda t: float(loss(ks0, ph0, omega + t)), 0.0, h=1e-6)
    assert float(np.sum(np.asarray(gw))) == pytest.approx(fd, rel=1e-6, abs=1e-8)
