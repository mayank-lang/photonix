"""Vector fitting of sampled S-parameters to delayed rational macromodels.

Algorithm
---------
Each entry is modelled as ``exp(-1j Omega tau_ij) H~_ij(Omega)`` with a common
set of poles for the delay-free cores ``H~``:

1. **Delay extraction.** A long waveguide's phase winds many times across a
   sweep; fitting it directly needs hundreds of poles. Per-entry *arrival*
   delays are estimated conservatively (the least-squares linear-phase delay
   of a delay-dominated entry, otherwise the onset of the band-limited impulse
   response, see :func:`estimate_delays`) and removed,
   ``H~ = H exp(+1j Omega tau)``. Removing
   less delay than the true arrival time only costs poles; removing more would
   leave a non-causal core that no stable rational function represents.
   By default (``delays="auto"``) the delays are constrained to per-port
   reference-plane delays ``tau_ij = a_i + b_j`` (``a = b`` for reciprocal
   data), chosen by a linear program to remove as much delay as possible
   without exceeding any entry's estimate. Such delays leave the singular
   values of ``S`` unchanged, so the Hamiltonian passivity test of the core is
   an exact test of the whole model (:mod:`~photonix.macromodel.passivity`).
2. **Starting poles** ``-alpha_k + 1j beta_k`` with ``beta_k`` spread over the
   band and ``alpha_k = beta-spacing / 100`` (complex baseband), or conjugate
   pairs ``-beta_k/100 +- 1j beta_k`` (real systems).
3. **Relaxed pole relocation** (Gustavsen 2006) with the fast QR reduction of
   Deschrijver et al. (2008): per entry the numerator unknowns are eliminated
   by a QR factorization, the stacked reduced systems are solved for the
   scaling function ``sigma(s) = sum c~_k/(s - a_k) + d~`` under
   ``Re sum_n sigma(s_n) = N`` (enforced exactly), and the new poles are the
   zeros of ``sigma``, ``eig(diag(a) - 1 c~^T / d~)``. Unstable poles are
   reflected, ``Re p -> -Re p``. For a complex baseband response the poles are
   *not* constrained to conjugate pairs (Ye et al. 2019); with ``real=True``
   the classical real formulation keeps exact conjugate symmetry.
4. **Residues and D** by linear least squares with the poles fixed.
5. **Order selection** (``n_poles="auto"``): the order grows until the RMS
   error over all ``n**2`` entries falls below ``target_rms * max|S|`` or stops
   improving.

References
----------
B. Gustavsen and A. Semlyen, IEEE Trans. Power Delivery 14(3), 1052 (1999).
B. Gustavsen, IEEE Trans. Power Delivery 21(3), 1587 (2006).
D. Deschrijver et al., IEEE Microw. Wireless Compon. Lett. 18(6), 383 (2008).
Y. Ye, D. Spina, D. Deschrijver, W. Bogaerts, T. Dhaene, Photonics Research
7(7), 771 (2019).
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._common import dense_data, frequency_grid, wavelengths_to_hz
from .model import RationalModel

__all__ = ["vector_fit", "vector_fit_frequency", "fit_dataset", "estimate_delays"]

#: Orders tried by ``n_poles="auto"`` (capped by ``max_poles`` and the sample count).
_AUTO_ORDERS = (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 88, 104, 124, 148, 176, 200)
_ZERO_RTOL = 1e-13  # entries below this fraction of max|S| are treated as structural zeros
_SIGNIFICANT = 1e-3  # entries below this fraction of max|S| do not constrain port delays


# --------------------------------------------------------------------------- #
# Delay estimation
# --------------------------------------------------------------------------- #
_COHERENCE = 10.0  # envelope peak/median below this marks a noise-like entry (white noise gives ~4)


def _arrival_time(omega: np.ndarray, h: np.ndarray, threshold: float) -> tuple[float, float]:
    """Earliest significant time of the band-limited impulse response of ``h(omega)``.

    The response is first centred on its energy centroid (the |h|^2-weighted
    mean group delay), resampled onto a uniform grid, Hann-windowed and
    inverse-transformed with 4x zero padding. The arrival is the first time
    the envelope reaches ``threshold`` of its peak, minus one resolution cell
    ``1 / B``; the window's leading sidelobes add a further margin. Also
    returns the envelope's peak-to-median ratio: a deterministic response
    concentrates its energy in time (ratios of 50 to 1e8 for the library
    models), white measurement noise does not (about 4).
    """
    from scipy.interpolate import CubicSpline

    mag = np.abs(h)
    weight = mag[1:] * mag[:-1]
    step = np.angle(h[1:] * np.conj(h[:-1]))
    gd = -step / np.diff(omega)
    centroid = float(np.sum(weight * gd) / np.sum(weight)) if np.sum(weight) > 0 else 0.0
    centred = h * np.exp(1j * omega * centroid)
    n = omega.size
    grid = np.linspace(omega[0], omega[-1], n)
    if np.allclose(grid, omega, rtol=0.0, atol=1e-12 * float(np.max(np.abs(omega)))):
        uniform = centred
    else:
        uniform = CubicSpline(omega, centred.real)(grid) + 1j * CubicSpline(omega, centred.imag)(grid)
    size = 1 << int(np.ceil(np.log2(4 * n)))
    spectrum = np.zeros(size, dtype=complex)
    spectrum[:n] = uniform * np.hanning(n)
    envelope = np.abs(np.fft.ifft(spectrum))
    d_omega = grid[1] - grid[0]
    period = 2.0 * np.pi / d_omega
    times = np.arange(size) * period / size
    times = np.where(times >= 0.5 * period, times - period, times)
    first = float(np.min(times[envelope >= threshold * float(envelope.max())]))
    bandwidth_hz = float(omega[-1] - omega[0]) / (2.0 * np.pi)
    concentration = float(envelope.max() / max(float(np.median(envelope)), 1e-300))
    return centroid + first - 1.0 / bandwidth_hz, concentration


def estimate_delays(
    freq_hz,
    S,
    *,
    threshold: float = 1e-2,
    magnitude_floor: float = 1e-3,
) -> np.ndarray:
    """Estimate conservative per-entry arrival delays (seconds) from sampled data.

    The delay that can be removed from an entry without leaving a non-causal
    remainder is its *arrival* time, the start of its impulse response. Two
    estimates are combined:

    * **Delay-dominated entries.** The phase (accumulated from wrapped
      differences of adjacent samples, so no global unwrapping is needed) is
      fitted by a least-squares linear phase. If that leaves only a small
      residual (weighted RMS below 0.01 rad, maximum below 0.05 rad: rounding
      error, measurement noise or weak dispersion), the least-squares delay is
      returned; the remaining core is then nearly constant. A dispersion-free
      waveguide is recovered to rounding error, a noisy one to the noise.
    * **Otherwise**, the time-domain arrival: the response is centred on its
      energy centroid, Hann-windowed and inverse-transformed, and the delay is
      the first time its envelope reaches ``threshold`` of the peak, minus one
      resolution cell ``1/B``. Group delay alone is *not* conservative (the
      minimum group delay of an all-pass ring is positive although its
      impulse response starts at ``t = 0``).

    Negative values are clipped to zero, and so are *noise-like* entries
    (a band-limited impulse response whose envelope peak is less than 10 times
    its median, as for the measured noise floor of a structurally zero
    entry), which have no meaningful arrival. Such entries also do not
    constrain the port delays chosen by ``vector_fit(delays="auto")``.

    Parameters
    ----------
    freq_hz
        Sample frequencies in Hz (any order).
    S
        ``(N,)``, ``(N, n, n)`` (``S[:, out, in]``) or an array-valued SDict.
    threshold
        Relative envelope level that defines the arrival.
    magnitude_floor
        Samples below this fraction of an entry's peak are excluded from the
        pure-delay test (their phase is undefined).

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` delays in seconds (shape ``(1, 1)`` for a single trace).

    Notes
    -----
    The sweep must resolve the phase: adjacent samples may differ by less than
    ``pi`` (a delay below ``1 / (2 df)``). Beyond that the delay aliases and a
    :class:`RuntimeWarning` is issued; pass delays explicitly in that case.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.50, 1.60, 2001)
    >>> s = px.components.straight(wl=wl, length=1000.0, ng=4.2)
    >>> tau = estimate_delays(299792458e6 / wl, px.to_numpy(s[("o1", "o2")]))
    >>> round(float(tau[0, 0]) * 1e12, 4)              # n_g L / c = 14.0097 ps
    14.0097
    >>> ring = px.components.all_pass_ring(wl=wl, coupling=0.1)
    >>> float(estimate_delays(299792458e6 / wl, px.to_numpy(ring[("o1", "o2")]))[0, 0])  # arrives at t = 0
    0.0
    """
    f_raw = np.asarray(freq_hz, dtype=float)
    data, _ = dense_data(S, f_raw.size)
    f, data = frequency_grid(f_raw, data)
    tau, coherent = _entry_delays(f, data, threshold, magnitude_floor, stacklevel=3)
    return np.where(coherent, tau, 0.0)


def _entry_delays(f, data, threshold, magnitude_floor, stacklevel=2) -> tuple[np.ndarray, np.ndarray]:
    """Per-entry arrival delays and a mask of coherent (non-noise-like) entries."""
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must lie in (0, 1)")
    if f.size < 4:
        raise ValueError("delay estimation needs at least four samples")
    omega = 2.0 * np.pi * f
    n = data.shape[1]
    tau = np.zeros((n, n))
    coherent = np.zeros((n, n), dtype=bool)
    aliased = False
    global_peak = float(np.max(np.abs(data)))
    for i in range(n):
        for j in range(n):
            h = data[:, i, j]
            mag = np.abs(h)
            peak = float(mag.max())
            if peak == 0.0:
                continue
            ok = (mag[1:] > magnitude_floor * peak) & (mag[:-1] > magnitude_floor * peak)
            step = np.angle(h[1:] * np.conj(h[:-1]))
            if np.all(ok):
                # Least-squares linear phase. If it leaves only a small phase residual (rounding error or
                # measurement noise, < 0.01 rad RMS), the entry is a delay times a nearly constant core:
                # the precise delay is safe to remove and is used instead of the conservative arrival.
                phase = np.concatenate([[0.0], np.cumsum(step)])
                w = mag * mag
                centred = omega - np.sum(w * omega) / np.sum(w)
                slope = float(np.sum(w * centred * phase) / np.sum(w * centred * centred))
                residual = phase - np.sum(w * phase) / np.sum(w) - slope * centred
                if np.sqrt(np.sum(w * residual**2) / np.sum(w)) < 1e-2 and np.max(np.abs(residual)) < 5e-2:
                    tau[i, j] = -slope
                    coherent[i, j] = True
                    continue
            tau[i, j], concentration = _arrival_time(omega, h, threshold)
            coherent[i, j] = concentration >= _COHERENCE
            # Aliasing shows on strong samples of significant, coherent entries; a noise-like entry or
            # samples near a noise floor have random phase steps that say nothing about the sampling.
            strong = (mag[1:] > 0.1 * peak) & (mag[:-1] > 0.1 * peak)
            if coherent[i, j] and peak >= _SIGNIFICANT * global_peak and np.any(strong) \
                    and np.mean(np.abs(step[strong]) > 0.9 * np.pi) > 0.05:
                aliased = True
    if aliased:
        warnings.warn(
            "the phase advances by nearly pi between many adjacent samples; the delay estimate may be "
            "aliased -- refine the sweep or pass delays explicitly",
            RuntimeWarning,
            stacklevel=stacklevel,
        )
    return np.maximum(tau, 0.0), coherent


def _separable_delays(tau_hat: np.ndarray, significant: np.ndarray, symmetric: bool) -> np.ndarray:
    """Largest port-separable delays ``a_i + b_j <= tau_hat_ij`` on significant entries.

    Stage 1 maximizes the total removed delay; stage 2 picks, among the
    optimal solutions, the one with the smallest largest port delay (which
    splits a two-port's delay evenly between its ports).
    """
    from scipy.optimize import linprog

    n = tau_hat.shape[0]
    rows, cols = np.nonzero(significant)
    if rows.size == 0:
        return np.zeros((n, n))
    unit = float(np.max(tau_hat[rows, cols]))
    if unit <= 0.0:
        return np.zeros((n, n))
    t = tau_hat / unit
    nv = n if symmetric else 2 * n

    def column(j: int) -> int:
        return j if symmetric else n + j

    a_ub = np.zeros((rows.size, nv))
    for r, (i, j) in enumerate(zip(rows, cols, strict=True)):
        a_ub[r, i] += 1.0
        a_ub[r, column(j)] += 1.0
    b_ub = t[rows, cols]
    used = np.any(a_ub != 0, axis=0)
    bounds = [(0.0, 1.0) if used[k] else (0.0, 0.0) for k in range(nv)]
    options = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
    objective = -a_ub.sum(axis=0)
    stage1 = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
    if not stage1.success:  # pragma: no cover - the LP is always feasible (x = 0)
        return np.zeros((n, n))
    best = -float(stage1.fun)
    # Stage 2: minimize t_max subject to optimality of stage 1.
    a2 = np.zeros((rows.size + nv + 1, nv + 1))
    a2[:rows.size, :nv] = a_ub
    a2[rows.size:rows.size + nv, :nv] = np.eye(nv)
    a2[rows.size:rows.size + nv, nv] = -1.0
    a2[-1, :nv] = objective
    b2 = np.concatenate([b_ub, np.zeros(nv), [-best * (1.0 - 1e-12)]])
    c2 = np.zeros(nv + 1)
    c2[nv] = 1.0
    stage2 = linprog(c2, A_ub=a2, b_ub=b2, bounds=[*bounds, (0.0, 1.0)], method="highs", options=options)
    x = stage2.x[:nv] if stage2.success else stage1.x
    x = _polish_vertex(x, stage2.x[nv] if stage2.success else None, a_ub, b_ub, bounds, best)
    x = np.clip(x, 0.0, 1.0) * unit
    a = x[:n]
    b = x[:n] if symmetric else x[n:]
    return a[:, None] + b[None, :]


def _polish_vertex(x, t_max, a_ub, b_ub, bounds, best) -> np.ndarray:
    """Re-solve the LP vertex's active constraints exactly (removes solver tolerance).

    HiGHS solves to ~1e-10 of the delay scale, which leaves a visible residual
    phase on a long waveguide. The active set (constraints met with equality)
    determines the vertex, so a least-squares solve of just those equalities
    recovers it to rounding error. The polished point is used only if it is
    still feasible and optimal.
    """
    tol = 1e-7
    rows = [a_ub[k] for k in range(a_ub.shape[0]) if abs(a_ub[k] @ x - b_ub[k]) < tol]
    rhs = [b_ub[k] for k in range(a_ub.shape[0]) if abs(a_ub[k] @ x - b_ub[k]) < tol]
    nv = x.size
    for k in range(nv):
        lo, hi = bounds[k]
        e = np.zeros(nv)
        e[k] = 1.0
        if abs(x[k] - lo) < tol:
            rows.append(e)
            rhs.append(lo)
        elif abs(x[k] - hi) < tol:
            rows.append(e)
            rhs.append(hi)
        elif t_max is not None and abs(x[k] - t_max) < tol:
            # Equal to the minimax level: tie it to the other variables at that level.
            peers = [m for m in range(nv) if m != k and abs(x[m] - t_max) < tol]
            if peers:
                e[peers[0]] = -1.0
                rows.append(e)
                rhs.append(0.0)
    if not rows:
        return x
    sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    feasible = np.all(a_ub @ sol <= b_ub + 1e-12) and all(lo - 1e-12 <= v <= hi + 1e-12 for v, (lo, hi) in
                                                             zip(sol, bounds, strict=True))
    optimal = a_ub.sum(axis=0) @ sol >= best * (1.0 - 1e-9)
    return sol if feasible and optimal else x


def _resolve_delays(delays, f, data, active, symmetric) -> tuple[np.ndarray, str]:
    n = data.shape[1]
    if isinstance(delays, str):
        if delays == "none":
            return np.zeros((n, n)), "none"
        if delays not in ("auto", "entry"):
            raise ValueError("delays must be 'auto', 'entry', 'none' or an (n, n) array of seconds")
        tau_hat, coherent = _entry_delays(f, data, 1e-2, 1e-3, stacklevel=4)
        peak = np.max(np.abs(data), axis=0)
        # Weak or noise-like entries (e.g. structural zeros of measured data) must not pin the port delays.
        significant = (peak >= _SIGNIFICANT * float(np.max(peak))) & coherent
        if delays == "entry":
            if symmetric:
                tau_hat = np.minimum(tau_hat, tau_hat.T)
            return np.where(significant, tau_hat, 0.0), "entry"
        if symmetric:
            tau_hat = np.minimum(tau_hat, tau_hat.T)
        return _separable_delays(tau_hat, significant, symmetric), "auto"
    tau = np.asarray(delays, dtype=float)
    if tau.ndim == 0:
        tau = np.full((n, n), float(tau))
    if tau.shape != (n, n) or not np.all(np.isfinite(tau)) or np.any(tau < 0):
        raise ValueError(f"explicit delays must be a non-negative finite ({n}, {n}) array of seconds")
    return tau, "explicit"


# --------------------------------------------------------------------------- #
# Vector-fitting kernels (normalized units: s = 1j * Omega / scale)
# --------------------------------------------------------------------------- #
@dataclass
class _Fit:
    poles: np.ndarray
    residues: np.ndarray  # (P, M)
    d: np.ndarray  # (M,)
    rms: float
    iterations: int
    converged: bool


def _initial_poles(x_min: float, x_max: float, n_poles: int, real: bool) -> np.ndarray:
    if not real:
        if n_poles == 1:
            beta = np.array([0.5 * (x_min + x_max)])
            spacing = x_max - x_min
        else:
            beta = np.linspace(x_min, x_max, n_poles)
            spacing = (x_max - x_min) / (n_poles - 1)
        return -0.01 * spacing + 1j * beta
    n_pairs, n_real = divmod(n_poles, 2)
    poles = [complex(-0.5 * (x_min + x_max))] * n_real
    beta = np.linspace(x_min, x_max, n_pairs) if n_pairs > 1 else np.array([0.5 * (x_min + x_max)] * n_pairs)
    for b in beta:
        a = complex(-b / 100.0, b)
        poles.extend([a, np.conj(a)])
    return np.asarray(poles, dtype=complex)


def _real_structure(poles: np.ndarray) -> np.ndarray:
    """0 for a real pole, 1 for the first (Im > 0) and 2 for the second member of a pair."""
    kinds = np.zeros(poles.size, dtype=int)
    k = 0
    while k < poles.size:
        if poles[k].imag == 0.0:
            k += 1
            continue
        kinds[k], kinds[k + 1] = 1, 2
        k += 2
    return kinds


def _order_real_poles(eigenvalues: np.ndarray) -> np.ndarray:
    """Real poles first, then exact conjugate pairs ``(a, conj(a))`` with ``Im a > 0``."""
    real = np.sort(eigenvalues[eigenvalues.imag == 0.0].real).astype(complex)
    upper = eigenvalues[eigenvalues.imag > 0.0]
    upper = upper[np.argsort(upper.imag)]
    pairs = np.empty(2 * upper.size, dtype=complex)
    pairs[0::2] = upper
    pairs[1::2] = np.conj(upper)
    return np.concatenate([real, pairs])


def _basis(s: np.ndarray, poles: np.ndarray, real: bool) -> np.ndarray:
    """Basis columns (complex functions) multiplying *real* unknowns (real) or complex ones."""
    cauchy = 1.0 / (s[:, None] - poles[None, :]) if poles.size else np.zeros((s.size, 0), complex)
    if not real:
        return np.hstack([cauchy, np.ones((s.size, 1))])
    kinds = _real_structure(poles)
    cols = []
    for k, kind in enumerate(kinds):
        if kind == 0:
            cols.append(cauchy[:, k])
        elif kind == 1:
            cols.append(cauchy[:, k] + cauchy[:, k + 1])
            cols.append(1j * cauchy[:, k] - 1j * cauchy[:, k + 1])
    cols.append(np.ones(s.size))
    return np.stack(cols, axis=1)


def _unknowns_to_residues(y: np.ndarray, poles: np.ndarray, real: bool) -> tuple[np.ndarray, np.ndarray]:
    """Map basis coefficients ``y`` ``(Q, M)`` to complex residues ``(P, M)`` and ``d`` ``(M,)``."""
    if not real:
        return y[:-1], y[-1]
    kinds = _real_structure(poles)
    res = np.empty((poles.size, y.shape[1]), dtype=complex)
    q = 0
    for k, kind in enumerate(kinds):
        if kind == 0:
            res[k] = y[q]
            q += 1
        elif kind == 1:
            res[k] = y[q] + 1j * y[q + 1]
            res[k + 1] = y[q] - 1j * y[q + 1]
            q += 2
    return res, y[-1].real.astype(complex)


def _constrained_lstsq(K: np.ndarray, c: np.ndarray, value: float) -> np.ndarray:
    """``argmin ||K y||`` subject to ``c . y = value`` (real arrays)."""
    pivot = int(np.argmax(np.abs(c)))
    rest = np.delete(np.arange(c.size), pivot)
    reduced = K[:, rest] - np.outer(K[:, pivot], c[rest] / c[pivot])
    rhs = -K[:, pivot] * (value / c[pivot])
    y_rest, *_ = np.linalg.lstsq(reduced, rhs, rcond=None)
    y = np.empty(c.size)
    y[rest] = y_rest
    y[pivot] = (value - c[rest] @ y_rest) / c[pivot]
    return y


def _relocate(s, H, W, poles, real, relaxed) -> np.ndarray:
    """One relaxed (or classical) vector-fitting pole relocation step."""
    n_samples, n_entries = H.shape
    phi = _basis(s, poles, real)
    q = phi.shape[1]
    blocks = []
    for m in range(n_entries):
        w = np.ones(n_samples) if W is None else W[:, m]
        a_blk = w[:, None] * phi
        b_blk = -(w * H[:, m])[:, None] * phi
        mat = np.hstack([a_blk, b_blk])
        if real:
            mat = np.vstack([mat.real, mat.imag])
        r = np.linalg.qr(mat, mode="r")
        blocks.append(r[q:, q:])
    K = np.vstack(blocks)
    if real:
        Kr = K
        c = phi.sum(axis=0).real
    else:
        Kr = np.block([[K.real, -K.imag], [K.imag, K.real]])
        g = phi.sum(axis=0)
        c = np.concatenate([g.real, -g.imag])
    y = None
    if relaxed:
        y = _constrained_lstsq(Kr, c, float(n_samples))
        coeff = y if real else y[:q] + 1j * y[q:]
        d_tilde = coeff[-1]
        if abs(d_tilde) < 1e-8:
            y = None  # degenerate relaxation: fall back to the classical d~ = 1
    if y is None:
        if real:
            sol, *_ = np.linalg.lstsq(Kr[:, :-1], -Kr[:, -1], rcond=None)
            coeff = np.concatenate([sol, [1.0]])
        else:
            sol, *_ = np.linalg.lstsq(K[:, :-1], -K[:, -1], rcond=None)
            coeff = np.concatenate([sol, [1.0 + 0j]])
    d_tilde = coeff[-1]
    if real:
        kinds = _real_structure(poles)
        size = poles.size
        a_mat = np.zeros((size, size))
        b_vec = np.zeros(size)
        q_idx = 0
        for k, kind in enumerate(kinds):
            if kind == 0:
                a_mat[k, k] = poles[k].real
                b_vec[k] = 1.0
            elif kind == 1:
                a, b = poles[k].real, poles[k].imag
                a_mat[k:k + 2, k:k + 2] = [[a, -b], [b, a]]
                b_vec[k] = 1.0
        c_vec = np.zeros(size)
        for k, kind in enumerate(kinds):
            if kind == 0:
                c_vec[k] = coeff[q_idx]
                q_idx += 1
            elif kind == 1:
                c_vec[k] = 2.0 * coeff[q_idx]
                c_vec[k + 1] = -2.0 * coeff[q_idx + 1]
                q_idx += 2
        zeros = np.linalg.eigvals(a_mat - np.outer(b_vec, c_vec) / d_tilde)
        zeros = np.where(zeros.real > 0, -zeros.real + 1j * zeros.imag, zeros)
        zeros = np.where(zeros.real == 0, -1e-12 + 1j * zeros.imag, zeros)
        # Remove rounding asymmetry, then order as real poles followed by exact pairs.
        zeros = np.where(np.abs(zeros.imag) <= 1e-14 * max(1.0, float(np.max(np.abs(zeros)))), zeros.real, zeros)
        return _order_real_poles(zeros)
    c_tilde = coeff[:-1]
    zeros = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(poles.size), c_tilde) / d_tilde)
    zeros = np.where(zeros.real > 0, -zeros.real + 1j * zeros.imag, zeros)
    zeros = np.where(zeros.real == 0, -1e-12 + 1j * zeros.imag, zeros)
    return zeros[np.argsort(zeros.imag)]


def _identify(s, H, W, poles, real) -> tuple[np.ndarray, np.ndarray]:
    """Residues ``(P, M)`` and direct terms ``(M,)`` by least squares with fixed poles."""
    phi = _basis(s, poles, real)
    n_entries = H.shape[1]
    norms = np.linalg.norm(phi, axis=0)
    norms[norms == 0] = 1.0
    phi_n = phi / norms

    def solve(a_mat, rhs):
        if real:
            a_mat = np.vstack([a_mat.real, a_mat.imag])
            rhs = np.concatenate([rhs.real, rhs.imag], axis=0)
        sol, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
        return sol

    if W is None:
        y = solve(phi_n, H)
    else:
        y = np.empty((phi.shape[1], n_entries), dtype=float if real else complex)
        for m in range(n_entries):
            y[:, m] = solve(W[:, m, None] * phi_n, W[:, m] * H[:, m])
    y = y / norms[:, None]
    return _unknowns_to_residues(y, poles, real)


def _evaluate(s, poles, residues, d) -> np.ndarray:
    out = np.broadcast_to(d, (s.size, d.size)).astype(complex)
    if poles.size:
        out = out + (1.0 / (s[:, None] - poles[None, :])) @ residues
    return out


def _weighted_rms(err: np.ndarray, multiplicity: np.ndarray, n_total: int) -> float:
    return float(np.sqrt(np.sum(multiplicity * np.sum(np.abs(err) ** 2, axis=0)) / (n_total * err.shape[0])))


def _pole_change(old: np.ndarray, new: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    cost = np.abs(old[:, None] - new[None, :])
    rows, cols = linear_sum_assignment(cost)
    return float(np.max(cost[rows, cols]))


def _fit_order(s, H, W, n_poles, real, relaxed, n_iter, tol, multiplicity, n_total) -> _Fit:
    x = s.imag
    if n_poles == 0:
        if W is None:
            d = H.mean(axis=0)
        else:
            w2 = W**2
            d = np.sum(w2 * H, axis=0) / np.sum(w2, axis=0)
        if real:
            d = d.real.astype(complex)
        rms = _weighted_rms(H - d[None, :], multiplicity, n_total)
        return _Fit(np.zeros(0, complex), np.zeros((0, H.shape[1]), complex), d, rms, 0, True)
    poles = _initial_poles(float(x.min()), float(x.max()), n_poles, real)
    best: _Fit | None = None
    history: list[float] = []
    converged = False
    iteration = 0
    for iteration in range(1, n_iter + 1):
        new = _relocate(s, H, W, poles, real, relaxed)
        change = _pole_change(poles, new)
        poles = new
        res, d = _identify(s, H, W, poles, real)
        rms = _weighted_rms(H - _evaluate(s, poles, res, d), multiplicity, n_total)
        if best is None or rms < best.rms:
            best = _Fit(poles.copy(), res, d, rms, iteration, False)
        history.append(best.rms)
        if change < tol:
            converged = True
            break
        if len(history) >= 5 and history[-1] > (1.0 - 1e-3) * history[-5]:
            converged = True  # stagnated at the attainable accuracy for this order
            break
    assert best is not None
    best.iterations = iteration
    best.converged = converged
    return best


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def vector_fit_frequency(
    freq_hz,
    S,
    *,
    ports: Sequence[str] | None = None,
    carrier_hz: float | None = None,
    real: bool = False,
    n_poles: int | Literal["auto"] = "auto",
    delays: Literal["auto", "entry", "none"] | np.ndarray | float = "auto",
    relaxed: bool = True,
    weighting: Literal["uniform", "inverse_magnitude"] = "uniform",
    n_iter: int = 20,
    tol: float = 1e-12,
    target_rms: float = 1e-4,
    max_poles: int = 200,
    symmetric: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> RationalModel:
    """Fit a delayed rational model to S-parameters sampled at frequencies ``freq_hz`` (Hz).

    This is the frequency-domain core of :func:`vector_fit`; use it directly
    for RF data (``real=True``) or when a frequency grid is at hand.

    Parameters
    ----------
    freq_hz
        Sample frequencies in Hz (any order, distinct).
    S
        ``(N,)``, ``(N, n, n)`` in ``S[:, out, in]`` order, or an SDict.
    ports
        Port names (default ``o1..oN`` or the SDict's ports).
    carrier_hz
        Baseband carrier (default: the centre of the sampled band). Must be
        ``None`` or ``0`` with ``real=True``.
    real
        Fit a real (RF) system in absolute frequency: conjugate-pair poles and
        residues and a real ``D``. Requires positive frequencies.
    n_poles
        Number of common poles, or ``"auto"`` to grow the order until the RMS
        error (over all ``n**2`` entries) is below ``target_rms * max|S|`` or no
        longer improves.
    delays
        ``"auto"`` (per-port reference-plane delays, passivity-preserving),
        ``"entry"`` (independent per-entry delays: often fewer poles, but the
        Hamiltonian passivity test no longer applies), ``"none"``, or an
        explicit ``(n, n)`` array (or scalar) in seconds.
    relaxed
        Use the relaxed non-triviality constraint (Gustavsen 2006).
    weighting
        ``"uniform"`` (absolute error) or ``"inverse_magnitude"`` (relative
        error per sample, floored at 1e-3 of each entry's peak).
    n_iter, tol
        Maximum relocation iterations per order and the pole-change tolerance
        (normalized to the band half-width). Iteration also stops when the
        error stagnates.
    target_rms
        Relative RMS target for ``n_poles="auto"``.
    max_poles
        Largest order tried by ``n_poles="auto"``.
    symmetric
        Fit a reciprocal model (``S = S^T`` exactly). ``None`` detects it from
        the data (``max|S - S^T| <= 1e-9 max|S|``).
    metadata
        Extra JSON-serializable provenance stored in the model.

    Returns
    -------
    RationalModel
        Fit statistics are in ``model.metadata`` (``rms_error``,
        ``max_error``, ``target_met``, ``order_history``, ...).

    Examples
    --------
    >>> import numpy as np
    >>> f = np.linspace(1e9, 20e9, 300)
    >>> s = 1j * 2 * np.pi * f
    >>> p = -2e9 + 2j * np.pi * 8e9
    >>> h = 3e9 / (s - p) + 3e9 / (s - np.conj(p)) + 0.1             # a real resonance
    >>> m = vector_fit_frequency(f, h, real=True, n_poles=2, delays="none")
    >>> m.is_real_system, m.metadata["rms_error"] < 1e-12
    (True, True)
    """
    f_raw = np.asarray(freq_hz, dtype=float)
    data, names = dense_data(S, f_raw.size, ports)
    f, data = frequency_grid(f_raw, data)
    n = len(names)
    if weighting not in ("uniform", "inverse_magnitude"):
        raise ValueError("weighting must be 'uniform' or 'inverse_magnitude'")
    if real:
        if carrier_hz not in (None, 0, 0.0):
            raise ValueError("a real (RF) fit uses absolute frequency: carrier_hz must be None or 0")
        if np.any(f <= 0):
            raise ValueError("a real (RF) fit requires positive frequencies")
        carrier = 0.0
    else:
        carrier = 0.5 * (f[0] + f[-1]) if carrier_hz is None else float(carrier_hz)
        if not np.isfinite(carrier) or carrier < 0:
            raise ValueError("carrier_hz must be finite and non-negative")
    omega = 2.0 * np.pi * (f - carrier)
    scale = float(np.max(np.abs(omega)))
    if scale == 0.0:
        raise ValueError("the sampled band must not collapse onto the carrier")
    hmax = float(np.max(np.abs(data)))
    reciprocity = float(np.max(np.abs(data - np.swapaxes(data, 1, 2)))) if n > 1 else 0.0
    if symmetric is None:
        symmetric = n > 1 and reciprocity <= 1e-9 * max(hmax, 1e-300)
    elif symmetric and reciprocity > 1e-6 * max(hmax, 1e-300):
        raise ValueError(f"symmetric=True but the data are not reciprocal (max|S - S^T| = {reciprocity:.3g})")
    peak = np.max(np.abs(data), axis=0)
    active = peak > _ZERO_RTOL * hmax
    tau, delay_mode = _resolve_delays(delays, f, data, active, bool(symmetric))
    if symmetric and not np.allclose(tau, tau.T, rtol=0.0, atol=1e-18):
        symmetric = False
    core = data * np.exp(1j * omega[:, None, None] * tau[None])
    entries = [(i, j) for i in range(n) for j in range(n) if active[i, j] and (not symmetric or i <= j)]
    multiplicity = np.array([1.0 if (i == j or not symmetric) else 2.0 for i, j in entries])
    s = 1j * omega / scale
    n_samples = f.size

    order_history: list[list[float]] = []
    if not entries:
        fit: _Fit | None = _Fit(np.zeros(0, complex), np.zeros((0, 0), complex), np.zeros(0, complex), 0.0, 0, True)
        target_met = True
    else:
        H = np.stack([core[:, i, j] for i, j in entries], axis=1)
        W = None
        if weighting == "inverse_magnitude":
            mag = np.abs(H)
            W = 1.0 / np.maximum(mag, 1e-3 * mag.max(axis=0, keepdims=True))
        limit = max(n_samples - 2, 0)
        if n_poles == "auto":
            orders = [p for p in _AUTO_ORDERS if p <= min(max_poles, limit)]
            if min(max_poles, limit) not in orders:
                orders.append(min(max_poles, limit))
        else:
            orders = [int(n_poles)]
            if orders[0] < 0:
                raise ValueError("n_poles must be non-negative")
            if orders[0] > limit:
                raise ValueError(f"n_poles={orders[0]} needs more samples (at most {limit} for {n_samples} samples)")
        target = float(target_rms) * hmax
        best: _Fit | None = None
        stall = 0
        fit = None
        trials: list[_Fit] = []
        for order in orders:
            trial = _fit_order(s, H, W, order, real, relaxed, int(n_iter), float(tol), multiplicity, n * n)
            trials.append(trial)
            order_history.append([order, trial.rms / max(hmax, 1e-300)])
            if best is None or trial.rms < 0.9 * best.rms:
                stall = 0
            else:
                stall += 1
            if best is None or trial.rms < best.rms:
                best = trial
            if n_poles != "auto" or trial.rms <= target:
                fit = trial
                break
            if stall >= 5 and order >= 16:
                break
        target_met = fit is not None and fit.rms <= target
        if fit is None:
            assert best is not None
            # Target unreachable (e.g. a noise floor): keep the most parsimonious order whose error is
            # within 5 % of the best, rather than the largest order that merely fits more of the noise.
            fit = next(t for t in trials if t.rms <= 1.05 * best.rms)
            warnings.warn(
                f"vector fitting did not reach target_rms={target_rms:g}: the error stagnated at relative RMS "
                f"{best.rms / hmax:.3g} (noise floor or max_poles); using {fit.poles.size} poles",
                RuntimeWarning,
                stacklevel=2,
            )

    assert fit is not None
    n_p = fit.poles.size
    residues = np.zeros((n_p, n, n), dtype=complex)
    d = np.zeros((n, n), dtype=complex)
    for m, (i, j) in enumerate(entries):
        residues[:, i, j] = fit.residues[:, m] * scale
        d[i, j] = fit.d[m]
        if symmetric:
            residues[:, j, i] = residues[:, i, j]
            d[j, i] = d[i, j]
    info: dict[str, Any] = {
        "source": "photonix.macromodel.vector_fit",
        "method": "relaxed vector fitting" if relaxed else "vector fitting",
        "formulation": "real (conjugate pairs)" if real else "complex baseband",
        "fit_band_hz": [float(f[0]), float(f[-1])],
        "n_samples": int(n_samples),
        "weighting": weighting,
        "delays_mode": delay_mode,
        "symmetric": bool(symmetric),
        "iterations": int(fit.iterations),
        "converged": bool(fit.converged),
        "target_rms": float(target_rms),
        "order_history": order_history,
    }
    model = RationalModel(names, carrier, fit.poles * scale, residues, d, tau, {**info, **(metadata or {})})
    err = model.response(f) - data
    rms = float(np.sqrt(np.mean(np.abs(err) ** 2)))
    final = {
        "rms_error": rms,
        "relative_rms_error": rms / hmax if hmax > 0 else 0.0,
        "max_error": float(np.max(np.abs(err))),
        "target_met": bool(target_met) if n_poles == "auto" else bool(rms <= float(target_rms) * hmax),
    }
    return model.replace(metadata={**model.metadata, **final})


def vector_fit(
    wavelengths,
    S,
    *,
    ports: Sequence[str] | None = None,
    n_poles: int | Literal["auto"] = "auto",
    carrier_wl: float | None = None,
    delays: Literal["auto", "entry", "none"] | np.ndarray | float = "auto",
    relaxed: bool = True,
    weighting: Literal["uniform", "inverse_magnitude"] = "uniform",
    n_iter: int = 20,
    tol: float = 1e-12,
    target_rms: float = 1e-4,
    max_poles: int = 200,
    symmetric: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> RationalModel:
    """Fit optical S-parameters sampled over wavelength to a complex-baseband rational model.

    ``wavelengths`` are in micrometres (any order); ``S`` is ``(N, n, n)`` in
    ``S[:, out, in]`` order, ``(N,)`` for one trace, or an array-valued SDict.
    ``carrier_wl`` (um) defaults to the band centre in frequency. All other
    parameters are as in :func:`vector_fit_frequency`.

    Returns
    -------
    RationalModel
        Callable as a Photonix component model, ``model(wl=...) -> SDict``.

    Notes
    -----
    Evidence: exact rational data are recovered to rounding error, a straight
    waveguide is reproduced by its delay and a constant, and ring, add-drop and
    Bragg-grating spectra are fitted to below 1e-3 RMS
    (``tests/test_macromodel.py``). A fit is only qualified inside its band;
    the model extrapolates as a rational function outside it.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.50, 1.60, 201)                      # 0.5 nm steps
    >>> s = px.components.straight(wl=wl, length=200.0)
    >>> m = vector_fit(wl, s)                                  # the two-port SDict
    >>> m.n_poles, m.metadata["rms_error"] < 1e-10             # delay + constant
    (0, True)
    >>> fine = np.linspace(1.50, 1.60, 2001)                   # re-query between samples
    >>> bool(np.allclose(np.abs(px.to_numpy(m(wl=fine)[("o1", "o2")])), 1.0, atol=1e-9))
    True
    """
    f = wavelengths_to_hz(wavelengths)
    carrier_hz = None if carrier_wl is None else float(wavelengths_to_hz(np.array([carrier_wl]))[0])
    return vector_fit_frequency(
        f, S, ports=ports, carrier_hz=carrier_hz, real=False, n_poles=n_poles, delays=delays, relaxed=relaxed,
        weighting=weighting, n_iter=n_iter, tol=tol, target_rms=target_rms, max_poles=max_poles,
        symmetric=symmetric, metadata=metadata,
    )


def fit_dataset(dataset, **kwargs: Any) -> RationalModel:
    """Fit a :class:`~photonix.core.SParameterDataset` (keywords as :func:`vector_fit`).

    The dataset's ports are used and its metadata is kept under
    ``model.metadata["dataset_metadata"]``.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.54, 1.56, 401)
    >>> ds = px.SParameterDataset.from_sdict(wl, px.components.all_pass_ring(wl=wl, coupling=0.2))
    >>> m = fit_dataset(ds)
    >>> m.ports, m.metadata["target_met"]
    (('o1', 'o2'), True)
    """
    metadata = {"dataset_metadata": dict(dataset.metadata), **dict(kwargs.pop("metadata", None) or {})}
    return vector_fit(dataset.wavelengths, dataset.s, ports=dataset.ports, metadata=metadata, **kwargs)
