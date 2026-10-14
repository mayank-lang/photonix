"""Exact synthesis of FIR lattice (cascaded-MZI) filters by polynomial peeling.

A lattice of ``N + 1`` lossless directional couplers joined by ``N`` stages,
whose upper arm is longer by ``dL`` (plus a phase ``phi_k``), is a lossless FIR
filter (Jinguji & Kawachi, JLT 13, 73, 1995; Madsen & Zhao, *Optical Filter
Design and Analysis*, Wiley 1999, ch. 4). With ``z^-1 = exp(-j omega)`` and
the normalized frequency ``omega = beta dL - 2 pi m = 2 pi df / FSR`` (``dL``
is rounded so ``neff dL / wl0 = m`` is an integer), the bar (``o1 -> o4``)
and cross (``o1 -> o3``) responses are degree-``N`` polynomials ``A(z)`` and
``B(z)`` with ``|A|^2 + |B|^2 = 1`` on the unit circle.

Conventions (those of :func:`photonix.circuit.lattice_filter`): in the
(upper, lower) basis a coupler of power coupling ``K = sin^2 theta`` is
``C = [[cos theta, -j sin theta], [-j sin theta, cos theta]]`` and stage ``k``
is ``diag(z^-1 exp(-j phi_k), 1)``, up to the common arm delay
``exp(-j beta L)`` that this module leaves out. The response is the first
column ``(A, B)`` of ``C_N D_N ... D_1 C_0``.

:func:`synthesize_lattice` inverts that product exactly: stage ``k`` is peeled
off by the coupler angle that clears the constant term of the bar and the top
term of the cross, using the relative phase between the arms to keep every
angle in ``[0, pi/2]``. Targets come from :func:`maxflat_halfband` (Daubechies
maximally flat half-band pair, the classic interleaver), :func:`equiripple_lowpass`
(Parks-McClellan power response, factored) or any cross response with
``|B| <= 1``, whose partner :func:`complementary_bar` supplies through the
minimum-phase :func:`spectral_factor`.

Evidence: every design reports ``max_error`` from a forward evaluation of the
synthesized lattice. The tests check the conventions against the netlist
simulation of :func:`photonix.circuit.lattice_filter` (1e-10 over a full FSR),
the maximally flat pairs against Daubechies' published filters and closed
forms, and the order-2 interleaver against a hand derivation; see
``docs/SYNTHESIS.md``. The synthesis is exact for lossless,
wavelength-independent couplers; coupler dispersion is not modelled.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from photonix.core.backend import xp

__all__ = [
    "CrossBar", "LatticeDesign",
    "lattice_response", "lattice_coefficients",
    "spectral_factor", "complementary_bar",
    "maxflat_halfband", "equiripple_lowpass",
    "synthesize_lattice", "interleaver",
]

_C = 299_792_458.0
_TRIM_TOL = 1e-14          # relative size below which common top coefficients are trimmed
_VANISH = 64 * np.finfo(float).eps   # relative size of a numerically vanishing derivative of R


class CrossBar(NamedTuple):
    """The cross (``o1 -> o3``) and bar (``o1 -> o4``) of a lattice filter.

    Holds either responses (arrays over frequency) or the coefficients of the
    polynomials in ``z^-1``, cross first. Unpacks as ``cross, bar = pair``.
    """

    cross: Any
    bar: Any


# --------------------------------------------------------------------------- #
# Forward models
# --------------------------------------------------------------------------- #
def _stage_delays(delays: Sequence[int] | None, n: int) -> list[int]:
    if delays is None:
        return [1] * n
    out = [int(d) for d in delays]
    if len(out) != n or any(d < 0 or d != v for d, v in zip(out, delays, strict=True)):
        raise ValueError("delays must hold one non-negative integer (unit delays) per stage")
    return out


def lattice_response(couplings, phases, omega, *, delays: Sequence[int] | None = None) -> CrossBar:
    """Cross and bar amplitudes of a lossless lattice at normalized frequency ``omega``.

    Chains the 2x2 matrices of the module conventions: couplers with power
    couplings ``couplings[i]`` and stages ``diag(exp(-j (d_k omega + phi_k)), 1)``
    with ``d_k = delays[k]`` unit delays (default 1). The common arm delay is
    omitted, so the result is periodic in ``omega`` with period ``2 pi``.
    Written with the backend array module, so it is differentiable (JAX) in
    ``couplings``, ``phases`` and ``omega`` (singular at ``K = 0`` or ``1``,
    where ``sqrt`` has an infinite slope).

    Examples
    --------
    >>> import numpy as np
    >>> cross, bar = lattice_response([0.5, 0.5], [0.0], np.array([0.0, np.pi / 2]))
    >>> np.round(np.abs(np.asarray(cross)) ** 2, 12).tolist()            # MZI: cos^2(omega/2)
    [1.0, 0.5]
    """
    k = xp.asarray(couplings)
    ph = xp.asarray(phases)
    n = int(k.shape[0]) - 1
    if k.ndim != 1 or n < 0 or tuple(ph.shape) != (n,):
        raise ValueError("a lattice needs one more coupling than phases (one phase per stage)")
    dl = _stage_delays(delays, n)
    s, c = xp.sqrt(k), xp.sqrt(1.0 - k)
    w = xp.asarray(omega)
    one = xp.ones_like(xp.real(w)) + 0j
    bar = c[0] * one
    cross = -1j * s[0] * one
    for i in range(n):
        bar = bar * xp.exp(-1j * (dl[i] * w + ph[i]))
        bar, cross = c[i + 1] * bar - 1j * s[i + 1] * cross, -1j * s[i + 1] * bar + c[i + 1] * cross
    return CrossBar(cross, bar)


def lattice_coefficients(couplings, phases, *, delays: Sequence[int] | None = None) -> CrossBar:
    """Coefficients (in ``z^-1``, constant term first) of a lattice's cross and bar.

    The polynomial form of :func:`lattice_response` (NumPy): evaluating them at
    ``z^-1 = exp(-j omega)`` gives the same amplitudes. The degree is
    ``sum(delays)``.

    Examples
    --------
    >>> import numpy as np
    >>> cross, bar = lattice_coefficients([0.5, 0.5], [0.0])
    >>> np.round(cross, 12).tolist(), np.round(bar, 12).tolist()     # -j (1 + z^-1)/2, (z^-1 - 1)/2
    ([-0.5j, -0.5j], [(-0.5+0j), (0.5+0j)])
    """
    k = np.asarray(couplings, dtype=float)
    ph = np.asarray(phases, dtype=float)
    n = k.size - 1
    if k.ndim != 1 or n < 0 or ph.shape != (n,):
        raise ValueError("a lattice needs one more coupling than phases (one phase per stage)")
    if np.any((k < 0) | (k > 1)):
        raise ValueError("couplings must lie in [0, 1]")
    dl = _stage_delays(delays, n)
    s, c = np.sqrt(k), np.sqrt(1.0 - k)
    bar = np.array([c[0]], dtype=complex)
    cross = np.array([-1j * s[0]], dtype=complex)
    for i in range(n):
        bar = np.concatenate([np.zeros(dl[i]), np.exp(-1j * ph[i]) * bar])
        cross = np.concatenate([cross, np.zeros(dl[i])])
        bar, cross = c[i + 1] * bar - 1j * s[i + 1] * cross, -1j * s[i + 1] * bar + c[i + 1] * cross
    return CrossBar(cross, bar)


# --------------------------------------------------------------------------- #
# Spectral factorization
# --------------------------------------------------------------------------- #
def _autocorr(h: np.ndarray) -> np.ndarray:
    """Laurent coefficients ``r_{-n..n}`` of ``|H(e^{j omega})|^2`` (``r_k = sum h_{m+k} h_m^*``)."""
    return np.convolve(h, np.conj(h[::-1]))


def _laurent(power, degree: int | None, tol: float) -> np.ndarray:
    if callable(power):
        if degree is None or int(degree) < 0:
            raise ValueError("a callable power response needs its (non-negative) degree")
        n = int(degree)
        m = 1 << max(6, (8 * (n + 1)).bit_length())
        vals = np.asarray(power(2 * np.pi * np.arange(m) / m), dtype=complex)
        if vals.shape != (m,):
            raise ValueError("the power callable must return one value per frequency")
        full = np.fft.ifft(vals)
        scale = max(float(np.max(np.abs(vals))), 1e-300)
        if np.max(np.abs(full[n + 1: m - n]), initial=0.0) > tol * scale:
            raise ValueError(f"the power response is not a trigonometric polynomial of degree {n}")
        return np.concatenate([full[m - n:], full[: n + 1]]) if n else full[:1]
    r = np.asarray(power, dtype=complex)
    if r.ndim != 1 or r.size % 2 == 0:
        raise ValueError("power coefficients must be a 1-D array r_{-n..n} of odd length")
    if degree is not None and int(degree) != (r.size - 1) // 2:
        raise ValueError("degree does not match the length of the power coefficients")
    return r


def _select_roots(roots: np.ndarray, n: int, tau: float) -> np.ndarray | None:
    """Pick ``n`` factor roots: inside the circle, plus half of each unit-circle cluster.

    Roots of a ``2m``-fold zero on the unit circle scatter numerically into a
    small ring around it (radius ~ eps^(1/2m)); their centroid stays accurate.
    Roots closer than ``tau`` are clustered; an even cluster centred on the
    circle contributes ``m`` roots at its centroid projected onto the circle.
    """
    from scipy.sparse.csgraph import connected_components

    if tau > 0 and roots.size:
        adj = np.abs(roots[:, None] - roots[None, :]) < tau
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(roots.size)
    chosen: list[complex] = []
    for lab in np.unique(labels):
        rs = roots[labels == lab]
        centre = complex(np.mean(rs))
        on_circle = (rs.size % 2 == 0 and abs(1.0 - abs(centre)) < max(tau, 1e-14)
                     and bool(np.all(np.abs(np.abs(rs) - 1.0) < 2 * tau + 1e-14)))
        if on_circle:
            chosen += [centre / abs(centre)] * (rs.size // 2)
        else:
            chosen += [complex(v) for v in rs[np.abs(rs) < 1.0]]
    return np.array(chosen, dtype=complex) if len(chosen) == n else None


def _polish(h: np.ndarray, r: np.ndarray, real: bool) -> np.ndarray:
    """Gauss-Newton (Levenberg-Marquardt) polish of ``h`` on the autocorrelation equations."""
    from scipy.optimize import least_squares

    n = h.size - 1
    target = r[n:]
    if real:
        def res_real(x):
            return np.real(_autocorr(x.astype(complex))[n:] - target)

        sol = least_squares(res_real, np.real(h), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
        return sol.x.astype(complex)
    h = h * np.exp(-1j * np.angle(h[0]))            # gauge: h_0 real, so unknowns = equations

    def unpack(x):
        return x[: n + 1] + 1j * np.concatenate([[0.0], x[n + 1:]])

    def res_complex(x):
        d = _autocorr(unpack(x))[n:] - target
        return np.concatenate([d.real, d.imag[1:]])

    sol = least_squares(res_complex, np.concatenate([h.real, h.imag[1:]]), method="lm",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return unpack(sol.x)


def _unit_circle_zeros(r: np.ndarray, peak: float, real: bool = False) -> list[tuple[complex, int]]:
    """Unit-circle zeros ``(exp(j theta), q)`` of ``R`` (a ``2q``-fold zero each), found from its derivatives.

    Candidate angles are the local minima of ``R`` on a dense grid; each is
    refined by Newton's method on ``d^(2q-1) R / d omega^(2q-1)`` and kept
    with the largest ``q`` for which the Taylor coefficients of
    ``R(theta + x)`` below order ``2q`` vanish (to 64 eps of their scale, the rounding level).
    Unlike the roots of ``R``, which scatter by ``eps^(1/2q)``, these
    derivatives are well conditioned: zeros just off the circle that hide
    in that scatter are not counted.
    """
    m = (r.size - 1) // 2
    k = np.arange(-m, m + 1)
    size = 64 * (m + 1)
    grid = 2 * np.pi * np.arange(size) / size
    vals = np.real(np.exp(1j * np.outer(grid, -k)) @ r)
    low = (vals <= np.roll(vals, 1)) & (vals <= np.roll(vals, -1)) & (vals < 1e-2 * peak)

    def taylor(theta: float, j: int) -> tuple[float, float]:
        terms = r * np.exp(-1j * k * theta) * (-1j * k) ** j / math.factorial(j)
        return float(np.real(np.sum(terms))), float(np.sum(np.abs(terms)))

    def vanish(theta: float, q: int) -> bool:
        return all(abs(c) <= _VANISH * s for c, s in (taylor(theta, j) for j in range(2 * q)))

    cands: list[tuple[int, float, float]] = []
    for theta0 in grid[low]:
        best = None
        theta = float(theta0)
        for q in range(1, m + 1):
            if not vanish(theta, q):
                trial = theta
                for _ in range(30):                               # Newton on the (2q-1)-th derivative
                    g, _s = taylor(trial, 2 * q - 1)
                    dg, _s = taylor(trial, 2 * q)
                    if dg == 0.0:
                        break
                    step = max(-0.05, min(0.05, g / (2 * q * dg)))  # c_j are Taylor coefficients
                    trial -= step
                    if abs(step) < 1e-15:
                        break
                snapped = round(trial / math.pi) * math.pi       # real R: prefer the symmetric angles 0, pi
                if real and abs(trial - snapped) < 1e-3 and vanish(snapped, q):
                    trial = snapped
                if not vanish(trial, q):
                    break
                theta = trial
            best = (theta, q)
        if best is not None:
            theta, q = best
            c2q, _s = taylor(theta, 2 * q)
            _c, s0 = taylor(theta, 0)
            # R stays below the rounding level for |x| < width around a 2q-fold zero
            width = (_VANISH * s0 / abs(c2q)) ** (1.0 / (2 * q)) if c2q else math.pi
            cands.append((q, theta, width))
    found: list[tuple[complex, int]] = []
    kept: list[tuple[float, float]] = []
    for q, theta, width in sorted(cands, key=lambda c: -c[0]):      # highest order first
        if all(abs(math.remainder(theta - t, 2 * math.pi)) > max(w, 1e-3) for t, w in kept):
            kept.append((theta, width))
            found.append((complex(np.exp(1j * theta)), q))
    return found


def spectral_factor(power, degree: int | None = None, *, tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """Minimum-phase spectral factor of a non-negative trigonometric polynomial.

    Finds ``h`` (coefficients of ``H(z) = sum h_n z^-n``, ``h_0 > 0``) with
    ``|H(e^{j omega})|^2 = R(omega) = sum_k r_k exp(-j k omega)``. ``power`` is
    either the Laurent coefficients ``r_{-n..n}`` (the autocorrelation
    ``np.convolve(h, np.conj(h[::-1]))``) or a callable ``R(omega)``, in which
    case ``degree`` is required and the coefficients are recovered by FFT.

    Method: the ``2n`` roots of ``z^n R(z)`` pair up as ``rho, 1/rho^*``; the
    factor takes those inside the unit circle. Zeros *on* the circle have even
    multiplicity ``2m`` and scatter numerically (radius ~ ``eps^(1/2m)``), so
    roots are clustered and each even, circle-centred cluster contributes
    ``m`` copies of its centroid projected onto the circle. The clustering
    radius is chosen (from 0 to 0.5) by the smallest reconstruction error, and
    the result is polished by Levenberg-Marquardt on the autocorrelation
    equations. Real ``r`` (a real filter) gives a real ``h``. The polish may
    nudge unit-circle zeros by about the reconstruction error, so "minimum
    phase" holds to that accuracy. A high-order unit-circle zero can hide
    nearby off-circle zeros in its numerical ring, which the selection may
    then put on the wrong side; so the unit-circle zeros are also located
    from the (well-conditioned) derivatives of ``R``, divided out exactly, and
    the rest factored separately, and the candidate with the larger ``|h_0|``
    (the minimum-phase one: reflecting a zero ``rho`` outside scales ``h_0``
    by ``|rho|``) is kept when it is accurate.

    Returns
    -------
    h, error
        ``error = sum_k |r_k - r_k(h)|``, a rigorous bound on
        ``max_omega |R - |H|^2|``. ``ValueError`` if ``R`` is negative
        somewhere, not Hermitian, or the error exceeds ``tol * max R``.

    Examples
    --------
    >>> import numpy as np
    >>> h, err = spectral_factor([0.5, 1.25, 0.5])           # |1 + 0.5 e^{-j w}|^2
    >>> np.round(h, 12).tolist(), err < 1e-15
    ([1.0, 0.5], True)
    """
    r = _laurent(power, degree, tol)
    n = (r.size - 1) // 2
    r0 = float(np.real(r[n]))
    if not r0 > 0:
        raise ValueError("the power response must have a positive mean (r_0 > 0)")
    if np.max(np.abs(r - np.conj(r[::-1]))) > tol * r0:
        raise ValueError("power coefficients are not Hermitian (r_{-k} = conj(r_k)): R(omega) is not real")
    r = 0.5 * (r + np.conj(r[::-1]))
    real = bool(np.max(np.abs(r.imag)) <= 1e-15 * r0)
    if real:
        r = r.real.astype(complex)
    grid = 2 * np.pi * np.arange(max(64, 16 * (n + 1))) / max(64, 16 * (n + 1))
    vals = np.real(np.exp(1j * n * grid) * np.polyval(r[::-1], np.exp(-1j * grid)))
    peak = float(np.max(vals))
    if np.min(vals) < -tol * peak:
        raise ValueError(f"the power response is negative (min {np.min(vals):.3g}); it has no spectral factor")
    if n == 0:
        return np.array([math.sqrt(r0)], dtype=float if real else complex), 0.0
    m = n                                                      # effective degree: drop vanishing end terms
    while m > 0 and abs(r[n + m]) <= 1e-15 * r0:
        m -= 1
    sub = r[n - m: n + m + 1]
    roots = np.roots(sub) if m else np.array([], dtype=complex)

    def finish(h: np.ndarray) -> tuple[np.ndarray, float]:
        hh = np.concatenate([h, np.zeros(n - m, dtype=complex)])
        return hh, float(np.sum(np.abs(_autocorr(hh) - r)))

    best: tuple[np.ndarray, float] | None = None
    for tau in (0.0, *np.geomspace(1e-12, 0.5, 24)):
        sel = _select_roots(roots, m, float(tau))
        if sel is None:
            continue
        h = np.poly(sel).astype(complex) if m else np.ones(1, dtype=complex)
        h = h * math.sqrt(r0 / float(np.sum(np.abs(h) ** 2)))
        cand = finish(h)
        if best is None or cand[1] < best[1]:
            best = cand
    if best is None:                                           # pragma: no cover - selection always succeeds at tau=0.5
        raise RuntimeError("spectral factorization could not split the roots")
    h0 = best[0].real.astype(complex) if real else best[0]
    polished = finish(_polish(h0[: m + 1], sub, real))
    h, err = min((finish(h0[: m + 1]), polished), key=lambda t: t[1])
    # A unit-circle cluster of R swallows nearby off-circle zeros (its roots
    # scatter by eps^(1/2m)), so the selection and the polish can land on a
    # factor with some of them outside the circle. Second candidate: divide the
    # unit-circle zeros (found from the well-conditioned derivatives of R) out
    # of R exactly and factor the well-separated rest. Reflecting a zero rho
    # outside the circle to 1/rho* multiplies |h_0| by |rho|, so the
    # minimum-phase factor is the one with the largest |h_0|.
    zeros = _unit_circle_zeros(sub, peak, real) if m else []
    if zeros:
        quo, on_circle, exact = sub, np.ones(1, dtype=complex), True
        for a, q in zeros:
            for _ in range(q):
                quo, rem = np.polydiv(quo, _autocorr(np.array([1.0, -a])))
                exact = exact and float(np.max(np.abs(rem), initial=0.0)) <= 1e-10 * float(np.sum(np.abs(sub)))
                on_circle = np.convolve(on_circle, [1.0, -a])
        rest = None
        if exact and quo.size < sub.size:
            try:
                rest, _e = spectral_factor(quo, tol=tol)
            except ValueError:
                pass
        if rest is not None:
            prod = np.convolve(rest, on_circle)
            alt = finish(prod.real.astype(complex) if real else prod)
            if alt[1] <= tol * peak and (abs(alt[0][0]) > abs(h[0]) * (1.0 + 1e-6) or err > tol * peak):
                h, err = alt
    if np.real(h[0]) < 0:
        h = -h
    if err > tol * peak:
        raise ValueError(f"spectral factorization error {err:.3g} exceeds tol * max R = {tol * peak:.3g}")
    return (np.real(h) if real else h), err


def complementary_bar(cross, *, tol: float = 1e-9) -> np.ndarray:
    """Minimum-phase bar coefficients ``A`` with ``|A|^2 = 1 - |B|^2`` for a cross ``B``.

    ``cross`` holds the coefficients of ``B(z)`` in ``z^-1``; ``|B|`` must not
    exceed 1 on the unit circle. The factor comes from :func:`spectral_factor`
    and is real when ``cross`` is.

    Examples
    --------
    >>> import numpy as np
    >>> bar = complementary_bar([0.5, 0.5])                      # MZI cross (1 + z^-1)/2
    >>> np.round(bar, 12).tolist()                               # (1 - z^-1)/2 up to a sign
    [0.5, -0.5]
    """
    b = np.asarray(cross, dtype=complex)
    if b.ndim != 1 or b.size < 1:
        raise ValueError("cross must be a 1-D coefficient array")
    r = -_autocorr(b)
    r[b.size - 1] += 1.0
    grid = 2 * np.pi * np.arange(max(64, 16 * b.size)) / max(64, 16 * b.size)
    peak = float(np.max(np.abs(np.polyval(b[::-1], np.exp(-1j * grid))) ** 2))
    if peak > 1.0 + tol:
        raise ValueError(f"|cross|^2 reaches {peak:.6g} > 1: no lossless complementary bar exists")
    if np.sum(np.abs(r)) <= tol:                               # all-pass cross: the bar carries no light
        return np.zeros(b.size, dtype=float if np.all(b.imag == 0) else complex)
    h, _ = spectral_factor(r, tol=tol)
    return np.real(h) if np.all(b.imag == 0) else h


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def maxflat_halfband(order: int) -> CrossBar:
    """Maximally flat, half-band, power-complementary pair (the classic interleaver).

    ``|B|^2 = (1 - x)^N sum_{k<N} C(N-1+k, k) x^k`` with ``x = sin^2(omega/2)``
    and ``N = order``: the Daubechies polynomial, with ``|B|^2 = 1 - O(omega^2N)``
    at ``omega = 0`` (cross passes) and ``|B|^2 = O((omega - pi)^2N)`` at
    ``omega = pi`` (bar passes), the most derivatives a degree ``2N - 1`` pair
    can null (Jinguji & Kawachi 1995; Madsen & Zhao 1999, ch. 4). Built
    without root-finding any unit-circle zero: ``B = ((1 + z^-1)/2)^N Q(z)``
    where ``Q`` is the minimum-phase factor of ``P(sin^2(omega/2))``, each root
    ``x_r`` of ``P`` giving the pair ``z + 1/z = 2 - 4 x_r``. ``B`` is the
    Daubechies D(2N) scaling filter divided by ``sqrt 2`` (Daubechies, Comm.
    Pure Appl. Math. 41, 909, 1988) and the bar is its half-band mirror
    ``A(z) = B(-z)``. Degree ``2N - 1``. Power complementarity holds to
    ~1e-13 through order 12 and degrades with the conditioning of the
    Daubechies roots (~2e-11 at order 20); orders that lose more than 1e-9 are
    refused.

    Examples
    --------
    >>> import numpy as np
    >>> cross, bar = maxflat_halfband(2)
    >>> bool(np.allclose(8 * cross, [1 + 3**0.5, 3 + 3**0.5, 3 - 3**0.5, 1 - 3**0.5]))   # D4
    True
    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    n = int(order)
    p = np.array([math.comb(n - 1 + k, k) for k in range(n)], dtype=float)
    zeros = []
    for x in (np.roots(p[::-1]) if n > 1 else []):
        pair = np.roots([1.0, -(2.0 - 4.0 * x), 1.0])
        zeros.append(pair[np.argmin(np.abs(pair))])
    q = np.real(np.poly(zeros)) if zeros else np.ones(1)
    q = q / np.sum(q)                                          # Q(1) = 1, so B(1) = 1
    binom = np.array([math.comb(n, k) for k in range(n + 1)], dtype=float) / 2.0**n
    cross = np.convolve(binom, q)
    bar = cross * (-1.0) ** np.arange(cross.size)
    unit = _autocorr(cross) + _autocorr(bar)
    unit[cross.size - 1] -= 1.0
    if np.sum(np.abs(unit)) > 1e-9:                          # pragma: no cover - only for very high orders
        raise ValueError(f"order {n} is too high: root-finding the Daubechies polynomial lost accuracy")
    return CrossBar(cross, bar)


def _cheb_extremes(g: np.ndarray) -> tuple[float, float]:
    """Exact min and max over omega of the zero-phase response with symmetric taps ``g``."""
    from numpy.polynomial import chebyshev as ch

    n = (g.size - 1) // 2
    c = np.concatenate([[g[n]], 2.0 * g[n + 1:]])               # G = sum c_k T_k(cos omega)
    xs = [-1.0, 1.0]
    if n > 1:
        crit = ch.chebroots(ch.chebder(c))
        xs += [float(v.real) for v in crit if abs(v.imag) < 1e-7 and -1.0 <= v.real <= 1.0]
    vals = ch.chebval(np.array(xs), c)
    return float(np.min(vals)), float(np.max(vals))


def equiripple_lowpass(degree: int, passband_edge: float, stopband_edge: float, *,
                       weight: float = 1.0, tol: float = 1e-9) -> CrossBar:
    """Equiripple (minimax) low-pass power response, factored into a lattice pair.

    A zero-phase FIR ``G`` of ``2 degree + 1`` taps is designed with the
    Parks-McClellan algorithm (:func:`scipy.signal.remez`) to approximate 1 on
    ``[0, passband_edge]`` and 0 on ``[stopband_edge, pi]`` (``weight``
    weights the stop band). Its exact extremes ``G_min`` and ``G_max`` (from
    the roots of ``G'`` in ``cos omega``) give the power response
    ``|B|^2 = (G - G_min) / (G_max - G_min)`` in ``[0, 1]``, which touches 0 in
    the stop band and 1 in the pass band (Herrmann & Schuessler, Electron.
    Lett. 6, 329, 1970). Usually ``G_min < 0 < 1 < G_max``; for low degrees,
    narrow pass bands or a heavy ``weight`` the minimax ``G`` may not reach 1
    (or 0), and the same affine map still yields a valid ``[0, 1]`` response
    (with a correspondingly larger pass-band ripple). ``B`` and ``A`` are the
    minimum-phase factors of ``|B|^2`` and ``1 - |B|^2``; for half-band
    designs of even degree both top coefficients vanish, which
    :func:`synthesize_lattice` trims. Edges are normalized frequencies
    ``omega = 2 pi df / FSR`` in ``(0, pi)``.

    Examples
    --------
    >>> import numpy as np
    >>> cross, bar = equiripple_lowpass(7, 0.4 * np.pi, 0.6 * np.pi)
    >>> w = np.linspace(0, np.pi, 501); z = np.exp(-1j * w)
    >>> total = np.abs(np.polyval(cross[::-1], z))**2 + np.abs(np.polyval(bar[::-1], z))**2
    >>> bool(np.allclose(total, 1.0, atol=1e-12))
    True
    """
    from scipy.signal import remez

    if not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if not 0.0 < passband_edge < stopband_edge < math.pi:
        raise ValueError("need 0 < passband_edge < stopband_edge < pi")
    n = int(degree)
    g = remez(2 * n + 1, [0.0, passband_edge, stopband_edge, math.pi], [1.0, 0.0], weight=[1.0, weight],
              fs=2 * math.pi, maxiter=200)
    g = 0.5 * (g + g[::-1])
    lo, hi = _cheb_extremes(g)
    if not hi > lo:
        raise ValueError("the minimax response is constant; widen the transition band or raise the degree")
    rb = g / (hi - lo)
    rb[n] -= lo / (hi - lo)
    ra = -rb
    ra[n] += 1.0
    cross, _ = spectral_factor(rb, tol=tol)
    bar, _ = spectral_factor(ra, tol=tol)
    return CrossBar(cross, bar)


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatticeDesign:
    """A synthesized FIR lattice filter.

    Attributes
    ----------
    couplings
        Power couplings of the couplers, input to output.
    phases
        Extra upper-arm phase of each stage (radians, the
        ``exp(-j phase)`` of :func:`~photonix.components.phase_shifter`).
    delays
        Unit delays per stage; the upper arm of stage ``k`` is longer by
        ``delays[k] * delta_length``. Couplers that synthesize to ``K = 0``
        are merged away, so e.g. half-band designs use ``2 dL`` stages.
    output_phase, common_phase
        The gauge between the realized lattice and the target: target cross
        ``= exp(-j common_phase) * cross``, target bar
        ``= exp(-j (common_phase + output_phase)) * bar``. A phase shifter of
        ``output_phase`` on the bar output (``o4``) reproduces the target's
        complex responses up to a common phase; powers need nothing.
    delta_length, length
        Unit delay ``dL`` (µm, rounded so ``neff dL / wl0`` is an integer)
        and the common arm length.
    fsr_hz
        Free spectral range of the unit delay, ``c / (ng dL)`` (exact: the
        phase of the linear-dispersion waveguide is linear in frequency).
    max_error
        Bound on the largest deviation of the realized complex responses
        (after the gauge) from the target over the full FSR,
        ``sum_n |coefficient error|``.
    cross, bar
        The target coefficients in ``z^-1``.
    """

    couplings: tuple[float, ...]
    phases: tuple[float, ...]
    delays: tuple[int, ...]
    output_phase: float
    common_phase: float
    delta_length: float
    length: float
    neff: float
    ng: float
    wl0: float
    fsr_hz: float
    max_error: float
    cross: tuple[complex, ...] = field(repr=False)
    bar: tuple[complex, ...] = field(repr=False)

    @property
    def delta_lengths(self) -> tuple[float, ...]:
        """Upper-arm length excess of each stage (µm)."""
        return tuple(d * self.delta_length for d in self.delays)

    def model(self, *, loss_db_cm: float = 0.0):
        """The filter as a :func:`photonix.circuit.lattice_filter` circuit model."""
        from photonix.circuit import lattice_filter

        return lattice_filter(couplings=self.couplings, delta_lengths=self.delta_lengths, phases=self.phases,
                              length=self.length, neff=self.neff, ng=self.ng, loss_db_cm=loss_db_cm, wl0=self.wl0)

    def wavelength(self, df_hz):
        """Wavelength (µm) at a frequency detuning ``df_hz`` from ``wl0``."""
        f0 = _C / (self.wl0 * 1e-6)
        return _C / (f0 + np.asarray(df_hz)) * 1e6

    def omega(self, wl):
        """Normalized frequency ``2 pi df / FSR`` at wavelength ``wl`` (µm)."""
        return 2 * np.pi * self.ng * self.delta_length * (1.0 / np.asarray(wl) - 1.0 / self.wl0)

    def response(self, omega) -> CrossBar:
        """Cross and bar of the realized lattice (without the common arm delay)."""
        return lattice_response(self.couplings, self.phases, omega, delays=self.delays)

    def target(self, omega) -> CrossBar:
        """The target cross and bar at normalized frequency ``omega``."""
        z = np.exp(-1j * np.asarray(omega))
        return CrossBar(np.polyval(np.asarray(self.cross)[::-1], z), np.polyval(np.asarray(self.bar)[::-1], z))


def _wrap(phase: float) -> float:
    """Phase in ``[0, 2 pi)``, with values within 1e-12 of ``2 pi`` snapped to 0."""
    v = float(phase) % (2 * math.pi)
    return 0.0 if v > 2 * math.pi - 1e-12 else v


def _phase_to_axis(a: np.ndarray, b: np.ndarray, angle: float) -> float:
    """Rotation of ``a`` that puts ``a_0 / b_0`` (and ``b_N / a_N``) on the ray at ``angle``.

    Power complementarity makes ``a_N b_N^*`` point opposite ``a_0 b_0^*``, so
    both ends contribute to a well-conditioned estimate. When both vanish
    (one arm carries no light at this stage) any rotation works.
    """
    u = a[0] * np.conj(b[0]) - (a[-1] * np.conj(b[-1]) if a.size > 1 else 0.0)
    return angle - float(np.angle(u)) if abs(u) > 0.0 else 0.0


def _peel(bar: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Couplings angles, stage phases, output and common phase of an exact unit-delay lattice."""
    a, b = bar.copy(), cross.copy()
    n = a.size - 1
    alpha = _phase_to_axis(a, b, -math.pi / 2)
    a = np.exp(1j * alpha) * a
    thetas, phis = [], []
    for k in range(n, 0, -1):
        theta = math.atan2(abs(a[0]) + abs(b[k]), abs(b[0]) + abs(a[k]))
        c, s = math.cos(theta), math.sin(theta)
        a, b = c * a + 1j * s * b, 1j * s * a + c * b          # C(theta)^-1
        a, b = a[1:], b[:k]                                    # cleared a_0 and b_k: remove the stage delay
        phi = _phase_to_axis(a, b, -math.pi / 2 if k > 1 else math.pi / 2)
        a = np.exp(1j * phi) * a
        thetas.append(theta)
        phis.append(phi)
    thetas.append(math.atan2(abs(b[0]), abs(a[0])))
    common = -float(np.angle(a[0] + 1j * b[0]))                 # (a_0, b_0) = e^{j g} (cos t0, -j sin t0)
    return np.array(thetas[::-1]), np.array(phis[::-1]), math.remainder(alpha, 2 * math.pi), \
        math.remainder(common, 2 * math.pi)


def _unit_delay(delta_length: float | None, fsr_hz: float | None, neff: float, ng: float, wl0: float) -> float:
    if (delta_length is None) == (fsr_hz is None):
        raise ValueError("give exactly one of delta_length or fsr_hz")
    if fsr_hz is not None:
        if not fsr_hz > 0:
            raise ValueError("fsr_hz must be positive")
        delta_length = _C / (ng * float(fsr_hz)) * 1e6
    assert delta_length is not None
    if not delta_length > 0:
        raise ValueError("delta_length must be positive")
    m = max(1, round(neff * delta_length / wl0))
    return m * wl0 / neff


def synthesize_lattice(cross, bar=None, *, fsr_hz: float | None = None, delta_length: float | None = None,
                       neff: float = 2.4, ng: float = 4.2, wl0: float = 1.55, length: float = 10.0,
                       merge_tol: float = 1e-12, tol: float = 1e-9) -> LatticeDesign:
    """Exact couplings and phases of a lattice with the given cross and bar.

    ``cross`` and ``bar`` are the coefficients of ``B(z)`` and ``A(z)`` in
    ``z^-1`` (constant term first); their common degree ``N`` sets the number
    of unit-delay stages. Omitting ``bar`` uses :func:`complementary_bar`.
    Give exactly one of ``delta_length`` (µm) or ``fsr_hz``; the unit delay is
    rounded so ``neff dL / wl0`` is an integer, which puts ``omega = 0`` at
    ``wl0``.

    Peeling (stage ``N`` down to 1) applies ``C(theta)^-1`` with
    ``tan theta = |a_0 / b_0| = |b_N / a_N|``, which clears the constant term
    of the bar and the top term of the cross so the stage delay comes off
    exactly; the relative phase between the arms (the output phase at the top,
    then the stage phases) keeps ``a_0 / b_0`` on the negative imaginary axis
    so ``theta`` lies in ``[0, pi/2]``. Interior couplers with
    ``theta < merge_tol`` are removed and their stages merged. ``ValueError``
    if the pair is not power complementary to ``tol``.

    Top coefficients that vanish in *both* arrays (``|a_k| + |b_k|`` below
    1e-14 of the largest) are trimmed first, lowering the degree: e.g. the
    even-degree half-band designs of :func:`equiripple_lowpass`. ``cross`` and
    ``bar`` of the design refer to the trimmed pair and ``max_error``
    includes the trimmed magnitudes.

    Examples
    --------
    >>> import numpy as np
    >>> d = synthesize_lattice(*maxflat_halfband(2), delta_length=20.0)
    >>> np.round(d.couplings, 6).tolist(), d.delays, d.max_error < 1e-14
    ([0.933013, 0.75, 0.5], (2, 1), True)
    """
    dl = _unit_delay(delta_length, fsr_hz, neff, ng, wl0)
    b = np.atleast_1d(np.asarray(cross, dtype=complex))
    if b.ndim != 1 or b.size < 2:
        raise ValueError("cross must be a 1-D coefficient array of length >= 2 (at least one stage)")
    a = complementary_bar(b, tol=tol).astype(complex) if bar is None else np.asarray(bar, dtype=complex)
    if a.ndim != 1:
        raise ValueError("bar must be a 1-D coefficient array")
    size = max(a.size, b.size)
    a = np.concatenate([a, np.zeros(size - a.size, dtype=complex)])
    b = np.concatenate([b, np.zeros(size - b.size, dtype=complex)])
    unit = _autocorr(a) + _autocorr(b)
    unit[size - 1] -= 1.0
    mismatch = float(np.sum(np.abs(unit)))
    if mismatch > tol:
        raise ValueError(f"cross and bar are not power complementary: | |A|^2 + |B|^2 - 1 | up to {mismatch:.3g}")
    # Trim common vanishing top coefficients: the pair has a lower degree. (Common
    # leading zeros need no special case: peeling realizes them with K = 0 couplers.)
    ends = np.abs(a) + np.abs(b)
    last = int(np.flatnonzero(ends >= _TRIM_TOL * float(np.max(ends)))[-1])
    trimmed = float(np.sum(ends[last + 1:]))
    a, b = a[: last + 1], b[: last + 1]
    if a.size < 2:
        raise ValueError("cross and bar have degree 0 after trimming vanishing top coefficients: a single "
                         "coupler, no lattice stage")
    thetas, phis, alpha, common = _peel(a, b)
    keep, delays, phases = [0], [], []                         # merge null interior couplers:
    run, acc = 0, 0.0                                           # C(0) = I joins two stages exactly
    for i in range(1, thetas.size):
        run, acc = run + 1, acc + float(phis[i - 1])
        if i < thetas.size - 1 and thetas[i] < merge_tol:
            continue
        keep.append(i)
        delays.append(run)
        phases.append(_wrap(acc))
        run, acc = 0, 0.0
    couplings = np.sin(thetas[keep]) ** 2
    got = lattice_coefficients(couplings, phases, delays=delays)
    err = max(float(np.sum(np.abs(np.exp(-1j * common) * got.cross - b))),
              float(np.sum(np.abs(np.exp(-1j * (common + alpha)) * got.bar - a)))) + trimmed
    return LatticeDesign(
        couplings=tuple(float(v) for v in couplings), phases=tuple(phases), delays=tuple(delays),
        output_phase=float(alpha), common_phase=float(common), delta_length=dl, length=float(length),
        neff=float(neff), ng=float(ng), wl0=float(wl0), fsr_hz=_C / (ng * dl * 1e-6), max_error=err,
        cross=tuple(complex(v) for v in b), bar=tuple(complex(v) for v in a))


def interleaver(order: int = 2, *, fsr_hz: float | None = None, delta_length: float | None = None,
                neff: float = 2.4, ng: float = 4.2, wl0: float = 1.55, length: float = 10.0) -> LatticeDesign:
    """Maximally flat lattice interleaver: :func:`maxflat_halfband` synthesized.

    The cross port (``o3``) passes the channels at ``wl0 + k FSR`` and the bar
    port (``o4``) those half an FSR away, so the channel spacing is
    ``FSR / 2``. ``order`` ``N`` gives ``N + 1`` couplers and stages of
    ``2 dL`` (``N - 1`` of them) and ``dL``; ``N = 1`` is a plain MZI.

    Examples
    --------
    >>> d = interleaver(3, fsr_hz=100e9)
    >>> len(d.couplings), d.delays, round(d.fsr_hz / 1e9, 1)       # FSR within 0.05 %: integer neff dL / wl0
    (4, (2, 2, 1), 100.0)
    """
    return synthesize_lattice(*maxflat_halfband(order), fsr_hz=fsr_hz, delta_length=delta_length, neff=neff,
                              ng=ng, wl0=wl0, length=length)

