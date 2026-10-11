"""Signal-integrity metrics: eye diagrams, Q factor, BER, extinction ratio, OMA, sensitivity.

An :class:`EyeDiagram` folds a received waveform on the unit interval, picks
the sampling phase that maximizes the Q factor, and classifies every sample
by the transmitted symbol (given, and aligned automatically) or by clustering.
With Gaussian noise on each level,

.. code-block:: text

    Q = (mu_1 - mu_0) / (sigma_1 + sigma_0),      BER = erfc(Q / sqrt 2) / 2

(Agrawal, *Fiber-Optic Communication Systems*, 4th ed., Sec. 4.5), evaluated
for NRZ and, eye by eye, for PAM4. Counted error rates carry a Wilson score
confidence interval (:func:`photonix.analysis.wilson_interval`).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .signal import Waveform

__all__ = [
    "ber_from_q", "q_from_ber", "EyeDiagram", "eye_diagram", "ber_count", "receiver_sensitivity",
    "KP4_BER_THRESHOLD",
]

#: Pre-FEC BER threshold of the IEEE 802.3 RS(544, 514) "KP4" FEC.
KP4_BER_THRESHOLD = 2.4e-4


def ber_from_q(q):
    """Gaussian bit-error ratio ``erfc(Q / sqrt 2) / 2``.

    Examples
    --------
    >>> f"{ber_from_q(7.0):.3e}"         # the classic 1e-12 operating point is Q ~ 7.03
    '1.280e-12'
    """
    from scipy.special import erfc

    return 0.5 * erfc(np.asarray(q, dtype=float) / math.sqrt(2.0))


def q_from_ber(ber):
    """Inverse of :func:`ber_from_q`.

    Examples
    --------
    >>> round(float(q_from_ber(1e-12)), 4)
    7.0345
    """
    from scipy.special import erfcinv

    return math.sqrt(2.0) * erfcinv(2.0 * np.asarray(ber, dtype=float))


def _kmeans_1d(x: np.ndarray, m: int, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    centers = np.quantile(x, (np.arange(m) + 0.5) / m)
    labels = np.zeros(x.size, dtype=int)
    for _ in range(iters):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new = np.array([x[labels == k].mean() if np.any(labels == k) else centers[k] for k in range(m)])
        if np.allclose(new, centers, rtol=0, atol=1e-15 * max(1.0, float(np.max(np.abs(x))))):
            break
        centers = new
    order = np.argsort(centers)
    remap = np.empty(m, dtype=int)
    remap[order] = np.arange(m)
    return centers[order], remap[labels]


def _stats(values: np.ndarray, labels: np.ndarray, m: int):
    mu = np.array([values[labels == k].mean() if np.any(labels == k) else np.nan for k in range(m)])
    sd = np.array([values[labels == k].std() if np.sum(labels == k) > 1 else 0.0 for k in range(m)])
    return mu, sd


def _q_eyes(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    gap = np.diff(mu)
    noise = sd[1:] + sd[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(noise > 0, gap / noise, np.inf)


@dataclass
class EyeDiagram:
    """A folded eye with level statistics at the chosen sampling instant.

    Attributes
    ----------
    traces
        ``(n_traces, n_ui * samples_per_symbol + 1)`` waveform segments, each
        starting half a unit interval before a sampling instant (for plots).
    t_ui
        Time axis of :attr:`traces` in unit intervals (sampling instants at
        ``0.5, 1.5, ...``).
    samples
        Received value at the sampling instant of every analysed symbol.
    labels
        Symbol index (``0..M-1``) of every analysed symbol.
    sampling_index
        Sample offset of the sampling instant within the unit interval.
    """

    traces: np.ndarray
    t_ui: np.ndarray
    samples: np.ndarray
    labels: np.ndarray
    sampling_index: int
    samples_per_symbol: int
    baud: float
    dt: float
    fmt: str = "nrz"
    unit: str = ""
    delay_symbols: int = 0
    _phase_values: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)

    @property
    def n_levels(self) -> int:
        return 2 if self.fmt == "nrz" else 4

    @property
    def levels(self) -> np.ndarray:
        """Mean value of each level (ascending)."""
        return _stats(self.samples, self.labels, self.n_levels)[0]

    @property
    def sigmas(self) -> np.ndarray:
        """Standard deviation of each level."""
        return _stats(self.samples, self.labels, self.n_levels)[1]

    @property
    def thresholds(self) -> np.ndarray:
        """Q-optimal decision thresholds ``(sigma_0 mu_1 + sigma_1 mu_0) / (sigma_0 + sigma_1)`` per eye."""
        mu, sd = self.levels, self.sigmas
        w = sd[1:] + sd[:-1]
        mid = 0.5 * (mu[1:] + mu[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(w > 0, (sd[:-1] * mu[1:] + sd[1:] * mu[:-1]) / w, mid)

    @property
    def q_eyes(self) -> np.ndarray:
        """Q factor of each eye (one for NRZ, three for PAM4)."""
        return _q_eyes(self.levels, self.sigmas)

    @property
    def q_factor(self) -> float:
        """``(mu_1 - mu_0) / (sigma_1 + sigma_0)`` (the worst eye for PAM4)."""
        return float(np.min(self.q_eyes))

    @property
    def ber_gaussian(self) -> float:
        """Gaussian-noise BER estimate.

        NRZ: ``erfc(Q / sqrt 2) / 2``. PAM4 (Gray coded, nearest-neighbour
        errors): the symbol error rate from the Gaussian tails across every
        threshold, divided by ``log2 M = 2``.
        """
        if self.fmt == "nrz":
            return float(ber_from_q(self.q_factor))
        from scipy.special import erfc

        mu, sd, th = self.levels, self.sigmas, self.thresholds
        m = self.n_levels
        counts = np.array([np.sum(self.labels == k) for k in range(m)], dtype=float)
        prior = counts / counts.sum()
        ser = 0.0
        for k in range(m):
            if sd[k] == 0:
                continue
            if k > 0:
                ser += prior[k] * 0.5 * erfc((mu[k] - th[k - 1]) / (sd[k] * math.sqrt(2)))
            if k < m - 1:
                ser += prior[k] * 0.5 * erfc((th[k] - mu[k]) / (sd[k] * math.sqrt(2)))
        return float(ser / math.log2(m))

    @property
    def height(self) -> float:
        """Inner (worst-case) eye opening at the sampling instant; the smallest eye for PAM4."""
        lo = [np.max(self.samples[self.labels == k]) for k in range(self.n_levels - 1)]
        hi = [np.min(self.samples[self.labels == k + 1]) for k in range(self.n_levels - 1)]
        return float(np.min(np.asarray(hi) - np.asarray(lo)))

    @property
    def oma(self) -> float:
        """Outer modulation amplitude: top minus bottom level mean."""
        mu = self.levels
        return float(mu[-1] - mu[0])

    @property
    def extinction_ratio_db(self) -> float:
        """``10 log10(mu_top / mu_bottom)`` (for a signal proportional to optical power)."""
        mu = self.levels
        if mu[0] <= 0:
            return math.inf
        return float(10.0 * math.log10(mu[-1] / mu[0]))

    def _around(self, offsets: np.ndarray, first: int, last: int) -> np.ndarray:
        """Values at sample offsets ``offsets`` from the sampling instant of symbols ``first..last-1``."""
        vals = self._phase_values
        sps, s = self.samples_per_symbol, self.sampling_index
        ph = (s + offsets) % sps
        shift = (s + offsets) // sps
        k = np.arange(first, last)
        return vals[ph[:, None], k[None, :] + shift[:, None]]

    @property
    def width(self) -> float:
        """Horizontal opening (s) of the (worst) eye: the span of sampling phases with an open eye.

        The worst-case inner opening is evaluated at every sample offset within
        half a unit interval of the sampling instant and the open span is
        refined by linear interpolation of the opening at its two ends.
        """
        sps = self.samples_per_symbol
        lab = self.labels
        if lab.size < 3:
            return 0.0
        offsets = np.arange(-(sps // 2), sps - sps // 2)
        v = self._around(offsets, 1, lab.size - 1)
        lk = lab[1:-1]
        op = np.array([min(np.min(row[lk == e + 1]) - np.max(row[lk == e]) for e in range(self.n_levels - 1))
                       for row in v])
        c = sps // 2
        if op[c] <= 0:
            return 0.0
        right = c
        while right + 1 < sps and op[right + 1] > 0:
            right += 1
        left = c
        while left - 1 >= 0 and op[left - 1] > 0:
            left -= 1
        span = float(right - left)
        if right + 1 < sps:
            span += op[right] / (op[right] - op[right + 1])
        if left - 1 >= 0:
            span += op[left] / (op[left] - op[left - 1])
        return span * self.dt

    @property
    def crossing_pct(self) -> float:
        """NRZ crossing level of the mean rising and falling edges, in % of the eye amplitude."""
        if self.fmt != "nrz":
            raise ValueError("crossing_pct is defined for NRZ eyes")
        sps = self.samples_per_symbol
        lab = self.labels
        # one UI from the sampling instant of symbol k-1 to that of symbol k
        seg = self._around(np.arange(sps + 1), 0, lab.size - 1)
        rise = (lab[:-1] == 0) & (lab[1:] == 1)
        fall = (lab[:-1] == 1) & (lab[1:] == 0)
        if not (np.any(rise) and np.any(fall)):
            raise ValueError("no rising and falling transitions to locate the crossing")
        r = seg[:, rise].mean(axis=1)
        f = seg[:, fall].mean(axis=1)
        d = r - f
        idx = np.nonzero(np.diff(np.sign(d)))[0]
        if idx.size == 0:
            raise ValueError("the mean edges do not cross")
        i = int(idx[np.argmin(np.abs(idx - sps / 2))])
        w = d[i] / (d[i] - d[i + 1])
        v_x = r[i] + w * (r[i + 1] - r[i])
        mu = self.levels
        return float(100.0 * (v_x - mu[0]) / (mu[1] - mu[0]))

    def decisions(self) -> np.ndarray:
        """Decided symbol index of every analysed symbol (Q-optimal thresholds)."""
        return np.searchsorted(self.thresholds, self.samples)

    def plot(self, ax=None, **kwargs: Any):
        """Draw the eye with :func:`photonix.viz.plot_eye` (never calls ``plt.show``)."""
        from photonix.viz.eye import plot_eye

        return plot_eye(self, ax=ax, **kwargs)


def _align(values: np.ndarray, ref_levels: np.ndarray, max_delay: int) -> tuple[int, float]:
    """Integer delay ``d`` maximizing the correlation of ``values[k]`` with ``ref_levels[k - d]``."""
    v = values - values.mean()
    best_d, best_c = 0, -np.inf
    for d in range(0, max_delay + 1):
        n = min(v.size, ref_levels.size + d) - d
        if n < 8:
            break
        r = ref_levels[:n] - ref_levels[:n].mean()
        c = float(np.dot(v[d:d + n], r) / (np.linalg.norm(v[d:d + n]) * np.linalg.norm(r) + 1e-300))
        if c > best_c:
            best_d, best_c = d, c
    return best_d, best_c


def eye_diagram(
    signal: Waveform | np.ndarray,
    *,
    baud: float,
    samples_per_symbol: int | None = None,
    dt: float | None = None,
    n_ui: int = 2,
    skip_symbols: int = 0,
    fmt: Literal["nrz", "pam4"] = "nrz",
    symbols: Sequence[int] | np.ndarray | None = None,
    sampling_index: int | None = None,
    max_delay_symbols: int = 64,
) -> EyeDiagram:
    """Fold ``signal`` into an eye and compute its level statistics.

    Parameters
    ----------
    signal
        A received electrical :class:`Waveform` (an optical one is analysed as
        power ``|E|^2``) or an array with ``dt``.
    baud
        Symbol rate; ``1 / (baud dt)`` must be an integer number of samples.
    skip_symbols
        Symbols discarded at the start (filter transients).
    fmt
        ``"nrz"`` or ``"pam4"``.
    symbols
        The transmitted symbol indices (``0/1`` for NRZ, ``0..3`` for PAM4)
        starting at the first sample of ``signal``. The channel delay is found
        by correlation (up to ``max_delay_symbols``) and the samples are
        classified by the known symbol, which is exact even for closed eyes.
        Without it, the levels are found by 1-D k-means clustering.
    sampling_index
        Force the sampling instant (sample offset within the UI); by default
        the one with the largest Q factor.

    Examples
    --------
    >>> from photonix.timedomain import drive_waveform, prbs
    >>> bits = prbs(7, 508)
    >>> w = drive_waveform(bits, baud=10e9, samples_per_symbol=16, amplitude_v=1.0, offset_v=0.5)
    >>> eye = eye_diagram(w, baud=10e9, symbols=bits)
    >>> round(eye.height, 12), round(eye.oma, 12), eye.delay_symbols
    (1.0, 1.0, 0)
    """
    if isinstance(signal, Waveform):
        x = np.asarray(signal.samples)
        x = np.abs(x) ** 2 if signal.is_optical else np.real(x)
        dt = signal.dt
        unit = "W" if signal.is_optical else signal.unit
    else:
        if dt is None:
            raise ValueError("give dt for an array signal")
        x = np.real(np.asarray(signal, dtype=complex))
        unit = ""
    sps = samples_per_symbol if samples_per_symbol is not None else int(round(1.0 / (baud * dt)))
    if abs(sps * baud * dt - 1.0) > 1e-6:
        raise ValueError(f"1 / (baud dt) = {1.0 / (baud * dt):.6g} is not an integer number of samples per symbol")
    m = 2 if fmt == "nrz" else 4
    if fmt not in ("nrz", "pam4"):
        raise ValueError("fmt must be 'nrz' or 'pam4'")
    n_sym = x.size // sps - skip_symbols - 1
    if n_sym < 8:
        raise ValueError("too few symbols for an eye")
    base = skip_symbols * sps
    phase_vals = np.stack([x[base + j: base + j + n_sym * sps: sps][:n_sym] for j in range(sps)])
    ref = None if symbols is None else np.asarray(symbols, dtype=int)
    if ref is not None and (ref.min() < 0 or ref.max() >= m):
        raise ValueError(f"symbols must lie in 0..{m - 1}")

    def classify(j: int):
        v = phase_vals[j]
        if ref is None:
            _, lab = _kmeans_1d(v, m)
            return lab, 0, None
        d, corr = _align(v, ref[skip_symbols:].astype(float), max_delay_symbols)
        lab = np.full(v.size, -1)
        idx = np.arange(v.size) + skip_symbols - d
        ok = (idx >= 0) & (idx < ref.size)
        lab[ok] = ref[idx[ok]]
        return lab, d, corr

    candidates = range(sps) if sampling_index is None else [int(sampling_index) % sps]
    best = None
    for j in candidates:
        lab, d, _ = classify(j)
        keep = lab >= 0
        if len(np.unique(lab[keep])) < m:
            continue
        mu, sd = _stats(phase_vals[j][keep], lab[keep], m)
        if np.any(np.diff(mu) <= 0):
            continue
        q = float(np.min(_q_eyes(mu, sd)))
        if best is None or q > best[0]:
            best = (q, j, lab, d)
    if best is None:
        raise ValueError("could not resolve all levels at any sampling phase (eye fully closed?)")
    _, j, lab, d = best
    keep = lab >= 0
    pv = phase_vals[:, keep]
    # traces: n_ui UIs starting half a UI before each sampling instant
    span = n_ui * sps
    starts = base + j - sps // 2 + sps * np.arange(n_sym)
    starts = starts[(starts >= 0) & (starts + span < x.size)]
    traces = np.stack([x[s:s + span + 1] for s in starts]) if starts.size else np.zeros((0, span + 1))
    t_ui = (np.arange(span + 1) - (sps // 2 - 0.5 * sps)) / sps
    return EyeDiagram(traces, t_ui, phase_vals[j][keep], lab[keep], int(j), int(sps), float(baud), float(dt),
                      fmt, unit, int(d), pv)


def ber_count(decided: Sequence[int] | np.ndarray, reference: Sequence[int] | np.ndarray,
              confidence: float = 0.95) -> tuple[float, tuple[float, float]]:
    """Counted error ratio with a Wilson score interval.

    Examples
    --------
    >>> ber, (lo, hi) = ber_count([0, 1, 1, 0] * 250, [0, 1, 0, 0] * 250)
    >>> ber, lo < 0.25 < hi
    (0.25, True)
    """
    from photonix.analysis import wilson_interval

    a = np.asarray(decided)
    b = np.asarray(reference)
    if a.shape != b.shape:
        raise ValueError("decided and reference must have the same length")
    errors = int(np.sum(a != b))
    lo, hi = wilson_interval(errors, a.size, confidence)
    return errors / a.size, (lo, hi)


def receiver_sensitivity(
    link: Callable[[float], EyeDiagram],
    *,
    target_ber: float = 1e-12,
    bounds_dbm: tuple[float, float] = (-40.0, 10.0),
    xtol_db: float = 1e-3,
) -> float:
    """Received power (dBm) at which ``link(p_dbm).ber_gaussian`` reaches ``target_ber``.

    ``link`` maps an average received optical power in dBm to the resulting
    :class:`EyeDiagram` (the whole receiver chain, with a fixed noise seed so
    the BER is a deterministic function of power). The root of
    ``log10 BER(P) - log10 target`` is found by Brent's method in
    ``bounds_dbm``. Feed the result to
    :class:`photonix.analysis.LinkBudget` ``(sensitivity_dbm=...)``.
    """
    from scipy.optimize import brentq

    def f(p_dbm: float) -> float:
        ber = link(p_dbm).ber_gaussian
        return math.log10(max(ber, 1e-300)) - math.log10(target_ber)

    lo, hi = bounds_dbm
    if f(lo) < 0 or f(hi) > 0:
        raise ValueError("the target BER is not bracketed by bounds_dbm")
    return float(brentq(f, lo, hi, xtol=xtol_db))
