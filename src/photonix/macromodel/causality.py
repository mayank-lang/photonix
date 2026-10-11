"""Causality test of sampled S-parameters via generalized Kramers-Kronig relations.

A causal response (``h(t) = 0`` for ``t < 0``) in Photonix's ``exp(+1j omega
t)`` convention is analytic in the lower half of the complex frequency plane,
so its real and imaginary parts are Hilbert-transform pairs. For band-limited
samples the Hilbert integral must be truncated, and the out-of-band part is
unknown. Following Triverio & Grivet-Talocia (2008), ``n`` *subtractions* at
in-band points ``x_k`` make the kernel decay fast enough that the truncation
error can be bounded using only ``|S| <= 1`` out of band (passivity)::

    g(x)     = (H(x) - P(x)) / Pi(x),   Pi(x) = prod_k (x - x_k),
               P = the polynomial interpolating H at the x_k
    H_rec(x) = P(x) + Pi(x) * (1j/pi) PV integral_band g(x') / (x' - x) dx'
    |H - H_rec| <= T(x) = |Pi(x)|/pi * integral_out (M + |P(x')|) / (|Pi(x')| |x' - x|) dx'

for causal data (``M`` bounds ``|H|`` out of band). Any excess of the
reconstruction error over ``T + D + E`` (``D``: discretization error,
estimated by Richardson comparison of the trapezoidal-linear quadrature on the
full and on every other sample; ``E``: the contribution of additive measurement
noise, which is not Kramers-Kronig consistent, propagated through the same
linear reconstruction by Monte Carlo) is evidence of non-causality:
time-advanced data, a wrong reference-plane sign, or a corrupted measurement. Several subtraction sets
placed across the band localize the test; a frequency is flagged if *any* set
exceeds its bound.

The test can prove non-causality but not causality: a ``causal=True`` result
means no violation larger than the bounds was found. Near the band edges the
truncation bound diverges, so violations there cannot be detected; inspect
``bound`` to see where the test is sensitive. The sweep must resolve the
response (several samples per phase cycle and per linewidth), otherwise ``D``
becomes large and the test loses sensitivity rather than giving false alarms.

Reference: P. Triverio and S. Grivet-Talocia, "Robust causality
characterization via generalized dispersion relations", IEEE Trans. Adv.
Packag. 31(3), 579 (2008).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._common import dense_data, frequency_grid, wavelengths_to_hz

__all__ = ["CausalityReport", "causality_report"]


@dataclass(frozen=True)
class CausalityReport:
    """Result of :func:`causality_report`.

    Attributes
    ----------
    ports
        Port names; arrays use ``[sample, out, in]`` order.
    frequency_hz
        Increasing sample frequencies (Hz).
    error
        ``(N, n, n)`` reconstruction error ``|S - S_rec|`` of the most
        discriminating subtraction set at each sample.
    bound
        ``(N, n, n)`` truncation + discretization bound for that set
        (``inf`` where the test is blind, e.g. at band edges).
    error_per_entry
        ``(n, n)`` largest reconstruction error of each entry.
    violation_per_entry
        ``(n, n)`` largest excess ``error - bound`` (0 where none was found).
    causal
        False if any sample of any entry exceeds its bound.
    worst_entry
        ``(in_port, out_port)`` of the largest violation, or ``None``.
    worst_band_hz
        Contiguous frequency interval around the largest violation where the
        bound is exceeded, or ``None``.
    """

    ports: tuple[str, ...]
    frequency_hz: np.ndarray
    error: np.ndarray
    bound: np.ndarray
    error_per_entry: np.ndarray
    violation_per_entry: np.ndarray
    causal: bool
    worst_entry: tuple[str, str] | None
    worst_band_hz: tuple[float, float] | None


def _pv_apply(x: np.ndarray, g: np.ndarray, chunk: int = 256) -> np.ndarray:
    """``PV integral g_lin(x') / (x' - x) dx'`` over ``[x_0, x_-1]`` at every node.

    ``g_lin`` is the piecewise-linear interpolant of the columns of ``g``; the
    integral of each linear piece is exact, and the logarithmic singularities
    of the two pieces adjacent to an evaluation node cancel in the principal
    value (their ``log 0`` coefficients are equal), so those terms are dropped.
    """
    n = x.size
    h = np.diff(x)
    out = np.empty((n, g.shape[1]), dtype=complex)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        xe = x[start:stop]
        dist = x[None, :] - xe[:, None]
        with np.errstate(divide="ignore"):
            logs = np.log(np.abs(dist))
        logs[dist == 0.0] = 0.0
        dlog = logs[:, 1:] - logs[:, :-1]
        beta = (xe[:, None] - x[None, :-1]) / h[None, :]
        alpha = 1.0 - beta
        kernel = np.zeros((stop - start, n))
        kernel[:, :-1] += dlog * alpha
        kernel[:, 1:] += dlog * beta
        kernel[:, -1] += 1.0
        kernel[:, 0] -= 1.0
        out[start:stop] = kernel @ g
    return out


def _subtraction_sets(x_coarse_idx: np.ndarray, x: np.ndarray, n_sub: int, n_support: int) -> list[np.ndarray]:
    """Indices (into the fine grid, all on the coarse grid) of each subtraction set."""
    sets = []
    xc = x[x_coarse_idx]
    for m in range(n_support):
        centre = -1.0 + (2 * m + 1) / n_support
        offsets = np.zeros(1) if n_sub == 1 else np.linspace(-1.0, 1.0, n_sub) * 0.5 / n_support
        chosen: list[int] = []
        for target in centre + offsets:
            order = np.argsort(np.abs(xc - target))
            for k in order:
                idx = int(x_coarse_idx[k])
                if idx not in chosen and 0 < idx < x.size - 1:
                    chosen.append(idx)
                    break
        if len(chosen) == n_sub:
            sets.append(np.sort(np.asarray(chosen)))
    return sets


def _subtracted(x, H, nodes):
    """Interpolant ``P``, subtracted function ``g = (H - P) / Pi`` and the set's data."""
    xk = x[nodes]
    wk = np.array([np.prod([xk[k] - xk[m] for m in range(xk.size) if m != k]) for k in range(xk.size)])
    hk = H[nodes]  # (n_sub, M)
    pi = np.prod(x[:, None] - xk[None, :], axis=1)  # (N,)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_over_pi = np.sum(hk[None, :, :] / ((x[:, None] - xk[None, :]) * wk[None, :])[:, :, None], axis=1)
        g = H / pi[:, None] - p_over_pi
    for idx in nodes:  # removable singularities: interpolate from the neighbours
        g[idx] = 0.5 * (g[idx - 1] + g[idx + 1])
    with np.errstate(invalid="ignore"):
        p = pi[:, None] * p_over_pi
    for k, idx in enumerate(nodes):
        p[idx] = hk[k]
    return p, g, pi, xk, wk, hk


def _truncation_bound(x, xk, wk, hk, pi, out_max) -> np.ndarray:
    """``|Pi(x)|/pi * integral_{|x'|>1} (M/|Pi| + |P/Pi|)(x') / |x' - x| dx'`` (N, M)."""
    u = np.linspace(-30.0, 25.0, 701)
    t = np.exp(u)
    w = np.full(u.size, u[1] - u[0]) * t
    w[0] *= 0.5
    w[-1] *= 0.5
    total = np.zeros((x.size, hk.shape[1]))
    for side in (1.0, -1.0):
        xq = side * (1.0 + t)  # (Q,)
        pi_q = np.prod(xq[:, None] - xk[None, :], axis=1)
        p_over_pi = np.sum(hk[None, :, :] / ((xq[:, None] - xk[None, :]) * wk[None, :])[:, :, None], axis=1)
        amp = out_max / np.abs(pi_q)[:, None] + np.abs(p_over_pi)  # (Q, M)
        kernel = w[None, :] / np.abs(xq[None, :] - x[:, None])  # (N, Q)
        total += kernel @ amp
    bound = np.abs(pi)[:, None] / np.pi * total * 1.02  # 2 % quadrature safety
    bound[np.abs(x) >= 1.0 - 1e-12] = np.inf  # the integral diverges (logarithmically) at the band edges
    return bound


_NOISE_REALIZATIONS = 16


def _noise_level(H: np.ndarray) -> np.ndarray:
    """Robust per-column estimate of additive white-noise standard deviation.

    The fourth difference annihilates cubic trends; for complex white noise of
    standard deviation ``s`` it has ``E|d4|^2 = 70 s^2`` and a Rayleigh
    magnitude, whose median is ``1.1774 sqrt(35) s``. The median ignores the
    few large differences at sharp resonances. Smooth but under-resolved data
    raise the estimate, which only makes the test less sensitive.
    """
    d4 = H[:-4] - 4 * H[1:-3] + 6 * H[2:-2] - 4 * H[3:-1] + H[4:]
    return np.median(np.abs(d4), axis=0) / (1.1774100225154747 * np.sqrt(35.0))


def causality_report(
    wavelengths,
    S,
    *,
    ports: Sequence[str] | None = None,
    n_subtractions: int = 2,
    n_support: int = 8,
    out_of_band_max: float = 1.0,
    noise_std: float | None = None,
    atol: float = 1e-6,
) -> CausalityReport:
    """Test sampled S-parameters for causality with generalized dispersion relations.

    Parameters
    ----------
    wavelengths
        Sample wavelengths in micrometres (any order, distinct).
    S
        ``(N, n, n)`` in ``S[:, out, in]`` order, ``(N,)`` for one trace, or
        an array-valued SDict.
    ports
        Port names (default ``o1..oN``).
    n_subtractions
        Subtraction points per set (at least 1; 2 is the usual choice).
    n_support
        Number of subtraction sets spread across the band.
    out_of_band_max
        Bound on ``|S|`` outside the band (1 for passive S-parameters).
    noise_std
        Standard deviation of additive complex measurement noise per sample
        (absolute). Noise is not Kramers-Kronig consistent, so it is propagated
        through the same (linear) reconstruction with 16 Monte-Carlo white-noise
        realizations and ``5 x`` their RMS is added to the bound. ``None``
        (default) estimates it for each entry from fourth differences of the
        samples (a robust median, so resonances do not inflate it); pass ``0``
        for noiseless simulated data to get the most sensitive test.
    atol
        Absolute tolerance relative to ``max|S|`` added to every bound.

    Returns
    -------
    CausalityReport

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.50, 1.60, 2001)
    >>> h = px.to_numpy(px.components.all_pass_ring(wl=wl, coupling=0.3)[("o1", "o2")])
    >>> causality_report(wl, h).causal
    True
    >>> f = 299792458e6 / wl
    >>> advanced = h * np.exp(2j * np.pi * (f - f.mean()) * 5e-12)     # 5 ps time advance
    >>> causality_report(wl, advanced).causal
    False
    """
    f_raw = wavelengths_to_hz(wavelengths)
    data, names = dense_data(S, f_raw.size, ports)
    f, data = frequency_grid(f_raw, data)
    if n_subtractions < 1:
        raise ValueError("n_subtractions must be at least 1 (the truncation bound diverges without subtraction)")
    if n_support < 1:
        raise ValueError("n_support must be at least 1")
    n = len(names)
    n_total = f.size
    usable = n_total if n_total % 2 == 1 else n_total - 1  # odd count: the coarse grid shares both ends
    if usable < 8 * n_subtractions + 1:
        raise ValueError("too few samples for the causality test")
    fu = f[:usable]
    centre = 0.5 * (fu[0] + fu[-1])
    half = 0.5 * (fu[-1] - fu[0])
    x = (fu - centre) / half
    H = data[:usable].reshape(usable, n * n)
    hmax = float(np.max(np.abs(H)))
    tol_abs = float(atol) * max(hmax, 1e-300)
    coarse = np.arange(0, usable, 2)
    m_cols = n * n
    if noise_std is None:
        sigma = _noise_level(H)
    else:
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and non-negative")
        sigma = np.full(m_cols, float(noise_std))
    # Unit white-noise realizations ride along as extra columns: the reconstruction is linear, so their
    # reconstruction error is the noise contribution to the error of the data.
    n_noise = _NOISE_REALIZATIONS if np.any(sigma > 0) else 0
    rng = np.random.default_rng(0)
    unit_noise = (rng.normal(size=(usable, n_noise)) + 1j * rng.normal(size=(usable, n_noise))) / np.sqrt(2)
    augmented = np.concatenate([H, unit_noise], axis=1)
    width = m_cols + n_noise

    sets = _subtraction_sets(coarse, x, int(n_subtractions), int(n_support))
    parts = [_subtracted(x, augmented, nodes) for nodes in sets]
    # The principal-value kernel depends only on the grid: apply it once to every set's g.
    g_all = np.concatenate([part[1] for part in parts], axis=1)
    pv_fine_all = _pv_apply(x, g_all)
    pv_coarse_all = _pv_apply(x[coarse], g_all[coarse][:, np.arange(g_all.shape[1]) % width < m_cols])
    best_excess = np.full((usable, m_cols), -np.inf)
    best_err = np.zeros((usable, m_cols))
    best_bound = np.full((usable, m_cols), np.inf)
    for s_index, (p_aug, _g, pi, xk, wk, hk_aug) in enumerate(parts):
        cols = slice(s_index * width, s_index * width + m_cols)
        noise_cols = slice(s_index * width + m_cols, (s_index + 1) * width)
        pv_fine = pv_fine_all[:, cols]
        pv_coarse = pv_coarse_all[:, s_index * m_cols:(s_index + 1) * m_cols]
        p = p_aug[:, :m_cols]
        hk = hk_aug[:, :m_cols]
        rec = p + pi[:, None] * (1j / np.pi) * pv_fine
        noise_bound = 0.0
        if n_noise:
            rec_noise = p_aug[:, m_cols:] + pi[:, None] * (1j / np.pi) * pv_fine_all[:, noise_cols]
            unit = np.sqrt(np.mean(np.abs(unit_noise - rec_noise) ** 2, axis=1))  # (N,) per unit sigma
            noise_bound = 5.0 * unit[:, None] * sigma[None, :]
        # Richardson: the trapezoidal-linear quadrature is O(h^2), so |I_h - I_2h| ~ 3x the error of I_h.
        diff = np.abs(pi[coarse, None]) / np.pi * np.abs(pv_fine[coarse] - pv_coarse)
        disc = np.empty((usable, m_cols))
        disc[coarse] = diff
        disc[1::2] = np.maximum(diff[:-1], diff[1:])
        # Local maximum over +-2 samples guards against accidental agreement of the two grids.
        padded = np.pad(disc, ((2, 2), (0, 0)), mode="edge")
        disc = np.max(np.stack([padded[k:k + usable] for k in range(5)]), axis=0)
        trunc = _truncation_bound(x, xk, wk, hk, pi, float(out_of_band_max))
        err = np.abs(H - rec)
        bound = trunc + disc + tol_abs + noise_bound
        excess = err - bound
        better = excess > best_excess
        best_excess = np.where(better, excess, best_excess)
        best_err = np.where(better, err, best_err)
        best_bound = np.where(better, bound, best_bound)

    error = np.zeros((n_total, n * n))
    bound = np.full((n_total, n * n), np.inf)
    excess = np.full((n_total, n * n), -np.inf)
    error[:usable], bound[:usable], excess[:usable] = best_err, best_bound, best_excess
    violation = np.maximum(excess, 0.0)
    causal = not bool(np.any(violation > 0))
    worst_entry = None
    worst_band = None
    if not causal:
        k, m = np.unravel_index(int(np.argmax(violation)), violation.shape)
        i, j = divmod(int(m), n)
        worst_entry = (names[j], names[i])
        lo = hi = int(k)
        while lo > 0 and violation[lo - 1, m] > 0:
            lo -= 1
        while hi + 1 < n_total and violation[hi + 1, m] > 0:
            hi += 1
        worst_band = (float(f[lo]), float(f[hi]))
    return CausalityReport(
        ports=names,
        frequency_hz=f,
        error=error.reshape(n_total, n, n),
        bound=bound.reshape(n_total, n, n),
        error_per_entry=np.max(error, axis=0).reshape(n, n),
        violation_per_entry=np.max(violation, axis=0).reshape(n, n),
        causal=causal,
        worst_entry=worst_entry,
        worst_band_hz=worst_band,
    )
