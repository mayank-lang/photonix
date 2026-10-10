"""Resonance extraction from sampled spectra: wavelength, FWHM, Q, ER, FSR, finesse.

All functions take plain wavelength (µm) and power-transmission arrays, so they
work on simulated ``|S|^2`` traces and on measured data alike.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Resonance", "find_resonances", "fit_lorentzian", "free_spectral_range", "finesse",
    "CouplingSolution", "ring_q_factors",
]


@dataclass(frozen=True)
class Resonance:
    """One resonance of a power-transmission spectrum.

    Attributes
    ----------
    wavelength
        Centre wavelength in µm (parabolic refinement of the extremum).
    fwhm
        Full width at half depth (dips) or half height (peaks), in µm.
    q_loaded
        ``wavelength / fwhm``.
    extinction_db
        Contrast between the local baseline and the extremum, in dB.
    extremum
        Transmission at the extremum (minimum for a dip, maximum for a peak).
    baseline
        Local off-resonance transmission used for the half-depth level.
    index
        Sample index of the extremum.
    """

    wavelength: float
    fwhm: float
    q_loaded: float
    extinction_db: float
    extremum: float
    baseline: float
    index: int


def _crossing(x, y, i0, level, step, stop):
    """Position where ``y`` crosses ``level`` walking from ``i0`` towards ``stop``.

    The walk never passes ``stop`` (the neighbouring resonance), so the lines of
    a doublet are not merged into one width.
    """
    i = i0
    n = len(y)
    while 0 <= i + step < n and (i + step - stop) * step <= 0:
        a, b = y[i], y[i + step]
        if (a - level) * (b - level) <= 0 and a != b:
            return x[i] + (level - a) * (x[i + step] - x[i]) / (b - a)
        i += step
    return None


def _extinction_db(baseline: float, extremum: float, kind: str) -> float:
    """Contrast in dB: inf for a perfect zero, NaN for unphysical (negative) data."""
    num, den = (baseline, extremum) if kind == "dip" else (extremum, baseline)
    if num > 0 and den > 0:
        return float(10.0 * np.log10(num / den))
    if num > 0 and den == 0:
        return float("inf")
    return float("nan")


def _refine(x, y, i):
    if 0 < i < len(y) - 1:
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
            if abs(shift) <= 1:
                return float(x[i] + shift * (x[i + 1] - x[i - 1]) / 2)
    return float(x[i])


def find_resonances(
    wavelength,
    transmission,
    *,
    kind: str = "dip",
    prominence: float = 0.05,
    baseline_window: int | None = None,
) -> list[Resonance]:
    """Locate and characterize resonances in a power-transmission spectrum.

    Parameters
    ----------
    wavelength, transmission
        1-D arrays (µm, linear power). The grid must resolve each linewidth by
        a few samples for a meaningful FWHM.
    kind
        ``"dip"`` (all-pass through port, notch filter) or ``"peak"`` (drop port).
    prominence
        Minimum linear-power prominence passed to :func:`scipy.signal.find_peaks`.
    baseline_window
        Samples on each side used to estimate the local baseline (default: half
        the distance to the neighbouring resonance).

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.545, 1.555, 20001)
    >>> T = px.to_numpy(px.power(px.components.all_pass_ring(wl=wl, coupling=0.05)[("o1", "o2")]))
    >>> res = find_resonances(wl, T)
    >>> len(res) >= 1 and res[0].q_loaded > 1e3
    True
    """
    from scipy.signal import find_peaks

    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(transmission, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 5:
        raise ValueError("wavelength and transmission must be equal-length 1-D arrays (>= 5 samples)")
    if kind not in ("dip", "peak"):
        raise ValueError("kind must be 'dip' or 'peak'")
    order = np.argsort(x)
    x, y = x[order], y[order]
    signal = -y if kind == "dip" else y
    peaks, _ = find_peaks(signal, prominence=prominence)
    out: list[Resonance] = []
    for k, i in enumerate(peaks):
        lo = peaks[k - 1] if k > 0 else 0
        hi = peaks[k + 1] if k + 1 < len(peaks) else len(y) - 1
        if baseline_window is None:
            a, b = (lo + i) // 2 if k > 0 else 0, (i + hi) // 2 if k + 1 < len(peaks) else len(y) - 1
        else:
            a, b = max(0, i - baseline_window), min(len(y) - 1, i + baseline_window)
        segment = y[a:b + 1]
        baseline = float(segment.max() if kind == "dip" else segment.min())
        extremum = float(y[i])
        level = 0.5 * (baseline + extremum)
        left = _crossing(x, y, i, level, -1, lo)
        right = _crossing(x, y, i, level, +1, hi)
        centre = _refine(x, y, i)
        # NaN width when the half level is not reached before the neighbouring
        # line or the sweep edge (unresolved doublet, edge resonance): such a
        # line needs a multi-Lorentzian fit, and a merged width would be wrong.
        fwhm = float(right - left) if left is not None and right is not None else float("nan")
        er = _extinction_db(baseline, extremum, kind)
        out.append(Resonance(centre, fwhm, centre / fwhm, er, extremum, baseline, int(order[i])))
    return out


def fit_lorentzian(wavelength, transmission, *, kind: str = "dip", center=None, window=None) -> Resonance:
    """Least-squares Lorentzian fit of one resonance.

    Fits ``T = B -/+ A / (1 + ((wl - wl0) / (w / 2))^2)`` (``-`` for dips),
    which is the exact line shape of a high-Q ring near resonance. ``window``
    (µm) limits the fit to ``|wl - center| < window``.

    Examples
    --------
    >>> import numpy as np
    >>> wl = np.linspace(1.549, 1.551, 2001)
    >>> T = 1 - 0.8 / (1 + ((wl - 1.55) / 25e-6) ** 2)
    >>> r = fit_lorentzian(wl, T)
    >>> round(r.wavelength, 6), round(r.fwhm * 1e6, 3)
    (1.55, 50.0)
    """
    from scipy.optimize import curve_fit

    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(transmission, dtype=float)
    guess = find_resonances(x, y, kind=kind, prominence=0.0)
    if center is not None and guess:
        g = min(guess, key=lambda r: abs(r.wavelength - center))
    elif guess:
        g = max(guess, key=lambda r: abs(r.baseline - r.extremum))
    else:
        raise ValueError("no resonance found to fit")
    if window is not None:
        sel = np.abs(x - g.wavelength) < window
        x, y = x[sel], y[sel]
    sign = -1.0 if kind == "dip" else 1.0

    def model(w, base, amp, w0, width):
        return base + sign * amp / (1.0 + ((w - w0) / (0.5 * width)) ** 2)

    p0 = [g.baseline, abs(g.baseline - g.extremum), g.wavelength, g.fwhm]
    popt, _ = curve_fit(model, x, y, p0=p0, maxfev=20000)
    base, amp, w0, width = (float(v) for v in popt)
    width = abs(width)
    extremum = base + sign * amp
    return Resonance(w0, width, w0 / width, _extinction_db(base, extremum, kind), extremum, base, g.index)


def free_spectral_range(resonances: Sequence[Resonance]) -> float:
    """Mean spacing (µm) of consecutive resonances (needs at least two)."""
    wls = np.sort([r.wavelength for r in resonances])
    if len(wls) < 2:
        raise ValueError("need at least two resonances for an FSR")
    return float(np.mean(np.diff(wls)))


def finesse(resonances: Sequence[Resonance]) -> float:
    """``FSR / mean(FWHM)`` (unresolved widths, reported as NaN, are skipped)."""
    return free_spectral_range(resonances) / float(np.nanmean([r.fwhm for r in resonances]))


@dataclass(frozen=True)
class CouplingSolution:
    """Intrinsic and coupling Q of an all-pass resonance for one coupling regime."""

    regime: str
    q_intrinsic: float
    q_coupling: float


def ring_q_factors(resonance: Resonance) -> tuple[CouplingSolution, CouplingSolution]:
    """Split the loaded Q of an all-pass (notch) resonance into ``Q_i`` and ``Q_c``.

    For a high-Q all-pass resonator ``1/Q_L = 1/Q_i + 1/Q_c`` and the
    on-resonance transmission is ``T_min = ((1/Q_i - 1/Q_c) / (1/Q_i + 1/Q_c))^2``.
    A power spectrum cannot tell under- from over-coupling, so both solutions
    are returned as ``(under_coupled, over_coupled)``; a phase measurement or
    a coupling-gap sweep resolves the ambiguity.

    Examples
    --------
    >>> r = Resonance(1.55, 1.55e-4, 1e4, 6.0, 0.25, 1.0, 0)
    >>> under, over = ring_q_factors(r)
    >>> round(under.q_intrinsic), round(under.q_coupling)
    (13333, 40000)
    """
    x = float(np.sqrt(max(resonance.extremum / resonance.baseline, 0.0)))
    ql = resonance.q_loaded
    a, b = 2.0 * ql / (1.0 + x), 2.0 * ql / (1.0 - x) if x < 1 else np.inf
    return (CouplingSolution("under", a, b), CouplingSolution("over", b, a))
