"""Passivity assessment and enforcement for rational scattering macromodels.

A linear multiport described by power-normalized scattering parameters is
passive exactly when ``sigma_max(S(1j Omega)) <= 1`` at *every* real frequency,
including outside the fitted band (a fitted model is evaluated there by any
time-domain or broadband simulation). Sampling cannot certify this; the
Hamiltonian test can.

Hamiltonian test (scattering form, valid for complex ``A, B, C, D``)::

    R = D^H D - I,   Q = D D^H - I
    M = [[A - B R^-1 D^H C,      -B R^-1 B^H              ],
         [C^H Q^-1 C,           -A^H + C^H D R^-1 B^H     ]]

``M`` has an eigenvalue ``1j Omega`` exactly when a singular value of
``S(1j Omega)`` equals one, so its purely imaginary eigenvalues are the
boundaries of the violation bands (Grivet-Talocia 2004). The test applies to
the delay-free rational core. It is exact for the whole model when the delays
are per-port reference-plane delays ``tau_ij = a_i + b_j`` (the default of
:func:`~photonix.macromodel.vector_fit`), because those multiply ``S`` by
diagonal unitary matrices. Other delay structures fall back to a dense
sampled test that is *not* a certificate.

Enforcement perturbs only the residues (the ``C`` matrix) with the poles
fixed: at every local maximum of each singular value above one, the
first-order change ``delta sigma = Re(u^H delta S v)`` is set to bring it to
``1 - margin``, and the perturbation of minimum in-band energy
``integral_band ||delta S||_F^2`` (an exact Gram matrix, no sampling) is
applied. This repeats until the Hamiltonian has no imaginary eigenvalues
(Grivet-Talocia 2004; Gustavsen 2008/2010 for the residue-perturbation
formulation). Reciprocity and real-system symmetry are preserved when the
input model has them, and structurally zero entries stay zero.

References
----------
S. Grivet-Talocia, IEEE Trans. Circuits Syst. I 51(9), 1755 (2004).
B. Gustavsen, IEEE Trans. Adv. Packag. 33(1), 257 (2010).
S. Grivet-Talocia and B. Gustavsen, *Passive Macromodeling* (Wiley, 2016).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import RationalModel

__all__ = ["PassivityReport", "passivity_report", "enforce_passivity", "hamiltonian_matrix"]


@dataclass(frozen=True)
class PassivityReport:
    """Result of :func:`passivity_report`.

    Attributes
    ----------
    passive
        True if no singular value exceeds one (within ``tol``) anywhere.
    certified
        True if the verdict covers *all* real frequencies (Hamiltonian test),
        False for the sampled fallback.
    method
        ``"hamiltonian"``, ``"no-states"`` (a constant model) or ``"sampled"``.
    crossing_freqs_hz
        Absolute frequencies (Hz) where a singular value equals one.
    violation_bands_hz
        ``(f_lo, f_hi)`` intervals where ``sigma_max > 1``; ``-inf``/``inf``
        when ``sigma_max(D) > 1``.
    peak_freqs_hz, peak_sigmas
        Local maxima of ``sigma_max`` inside the violation bands.
    max_sigma
        Largest singular value found (the worst violation peak, or the
        largest value on a band sweep for a passive model).
    sigma_max_d
        ``sigma_max(D)``, the response far from every pole.
    """

    passive: bool
    certified: bool
    method: str
    crossing_freqs_hz: np.ndarray
    violation_bands_hz: tuple[tuple[float, float], ...]
    peak_freqs_hz: np.ndarray
    peak_sigmas: np.ndarray
    max_sigma: float
    sigma_max_d: float


# --------------------------------------------------------------------------- #
# Normalized core realization
# --------------------------------------------------------------------------- #
@dataclass
class _Core:
    a: np.ndarray  # (S,) diagonal of A, normalized
    B: np.ndarray  # (S, n)
    C: np.ndarray  # (n, S), normalized
    D: np.ndarray  # (n, n)
    scale: float  # rad/s per normalized unit


def _scale_of(model: RationalModel) -> float:
    band = model.metadata.get("fit_band_hz")
    half = 0.0
    if isinstance(band, list | tuple) and len(band) == 2:
        half = 2.0 * np.pi * max(abs(band[0] - model.carrier_hz), abs(band[1] - model.carrier_hz))
    peak = float(np.max(np.abs(model.poles), initial=0.0))
    return max(half, peak, 1.0)


def _core(model: RationalModel, scale: float) -> _Core:
    ss = model.to_statespace(minimal=True)
    return _Core(np.diag(ss.A) / scale, ss.B, ss.C / scale, ss.D, scale)


def _core_matrix(core: _Core, x: float) -> np.ndarray:
    if core.a.size == 0:
        return core.D
    return core.C @ (core.B / (1j * x - core.a)[:, None]) + core.D


def _hamiltonian(core: _Core) -> np.ndarray:
    n = core.D.shape[0]
    eye = np.eye(n)
    A = np.diag(core.a)
    B, C, D = core.B, core.C, core.D
    R = D.conj().T @ D - eye
    Q = D @ D.conj().T - eye
    Ri = np.linalg.inv(R)
    Qi = np.linalg.inv(Q)
    return np.block([
        [A - B @ Ri @ D.conj().T @ C, -B @ Ri @ B.conj().T],
        [C.conj().T @ Qi @ C, -A.conj().T + C.conj().T @ D @ Ri @ B.conj().T],
    ])


def hamiltonian_matrix(model: RationalModel) -> np.ndarray:
    """Scattering Hamiltonian of the model's delay-free core, in rad/s.

    Its purely imaginary eigenvalues ``1j Omega`` are the baseband angular
    frequencies where a singular value of the core equals one. Requires
    ``sigma_max(D) != 1``.

    Examples
    --------
    >>> import numpy as np
    >>> m = RationalModel(("o1",), 0.0, np.array([-1.0 + 0j]), np.array([[[1.5 + 0j]]]), np.array([[0.0]]))
    >>> ev = np.linalg.eigvals(hamiltonian_matrix(m))    # |1.5/(1j w + 1)| = 1 at w = +-sqrt(1.25)
    >>> np.round(np.sort(ev.imag), 6)
    array([-1.118034,  1.118034])
    """
    ss = model.to_statespace(minimal=True)
    core = _Core(np.diag(ss.A), ss.B, ss.C, ss.D, 1.0)
    return _hamiltonian(core)


# --------------------------------------------------------------------------- #
# Violation detection
# --------------------------------------------------------------------------- #
@dataclass
class _Peak:
    x: float
    sigma: np.ndarray  # singular values above one
    u: np.ndarray  # (n, k)
    v: np.ndarray  # (n, k)


def _crossings(core: _Core) -> np.ndarray:
    """Imaginary parts of the (numerically) imaginary Hamiltonian eigenvalues.

    The threshold is deliberately generous: a near-tangency (an eigenvalue
    pair just off the axis, where a singular value approaches one without
    crossing) only adds an interval boundary, which the interval test below
    classifies correctly, whereas a missed crossing could hide a violation.
    """
    ev = np.linalg.eigvals(_hamiltonian(core))
    size = max(1.0, float(np.max(np.abs(ev), initial=0.0)))
    candidates = np.sort(ev[np.abs(ev.real) <= 1e-5 * size].imag)
    out: list[float] = []
    for x in candidates:
        if not out or abs(x - out[-1]) > 1e-12 * size:
            out.append(float(x))
    return np.asarray(out)


def _sigma_max(H: np.ndarray) -> float:
    return float(np.linalg.svd(H, compute_uv=False)[0])


_POLE_PROBES = np.array([-3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0])


def _peaks_in(fun, lo: float, hi: float, n_points: int = 41, poles: np.ndarray | None = None) -> list[float]:
    """Refined local maxima of ``fun`` on ``[lo, hi]``.

    A uniform grid can step over a resonance much narrower than the interval,
    so every pole inside it adds probes at its centre +- multiples of its
    half-width ``|Re p|``.
    """
    from scipy.optimize import minimize_scalar

    xs = np.linspace(lo, hi, n_points)
    if poles is not None and poles.size:
        probes = (poles.imag[:, None] + _POLE_PROBES[None, :] * np.abs(poles.real)[:, None]).ravel()
        xs = np.unique(np.concatenate([xs, probes[(probes > lo) & (probes < hi)]]))
        n_points = xs.size
    vals = np.array([fun(x) for x in xs])
    peaks = []
    for k in range(n_points):
        left = vals[k - 1] if k > 0 else -np.inf
        right = vals[k + 1] if k + 1 < n_points else -np.inf
        if vals[k] >= left and vals[k] >= right:
            a = xs[max(k - 1, 0)]
            b = xs[min(k + 1, n_points - 1)]
            if b > a:
                res = minimize_scalar(lambda t: -fun(t), bounds=(a, b), method="bounded",
                                      options={"xatol": 1e-12 * max(1.0, abs(b))})
                peaks.append(float(res.x) if -res.fun >= vals[k] else float(xs[k]))
            else:
                peaks.append(float(xs[k]))
    return peaks


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge touching intervals (sorted input)."""
    merged: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def _hamiltonian_violations(core: _Core, tol: float):
    """True crossings, merged violation bands (normalized) and violation peaks of the core."""
    candidates = _crossings(core)
    sd = _sigma_max(core.D)
    edges = np.concatenate([[-np.inf], candidates, [np.inf]])
    intervals: list[tuple[float, float]] = []
    peaks: list[_Peak] = []

    def smax(x: float) -> float:
        return _sigma_max(_core_matrix(core, x))

    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if np.isinf(lo) or np.isinf(hi):
            if sd > 1.0 + tol:
                intervals.append((float(lo), float(hi)))
            continue
        probes = (0.5, 0.25, 0.75)
        if all(smax(lo + t * (hi - lo)) <= 1.0 + tol for t in probes):
            continue
        intervals.append((float(lo), float(hi)))
        for x in _peaks_in(smax, float(lo), float(hi), poles=core.a):
            u, s, vh = np.linalg.svd(_core_matrix(core, x))
            keep = s > 1.0 + tol
            if np.any(keep):
                peaks.append(_Peak(x, s[keep], u[:, keep], vh[keep].conj().T))
    bands = _merge(intervals)
    # Only band edges are genuine crossings; other candidates were near-tangencies.
    crossings = np.asarray([e for band in bands for e in band if np.isfinite(e)])
    return crossings, bands, peaks


def _sample_grid(model: RationalModel, core: _Core) -> np.ndarray:
    """Normalized frequencies for the sampled fallback: band, extension and pole neighbourhoods."""
    band = model.metadata.get("fit_band_hz")
    if isinstance(band, list | tuple) and len(band) == 2:
        lo = 2 * np.pi * (band[0] - model.carrier_hz) / core.scale
        hi = 2 * np.pi * (band[1] - model.carrier_hz) / core.scale
    else:
        lo, hi = -1.0, 1.0
    width = hi - lo
    xs = [np.linspace(lo - 0.5 * width, hi + 0.5 * width, 4001)]
    a = model.poles / core.scale
    offsets = np.linspace(-6.0, 6.0, 49)
    for p in a:
        xs.append(p.imag + offsets * abs(p.real))
    return np.unique(np.concatenate(xs))


def _full_matrix(model: RationalModel, core: _Core, x: float) -> np.ndarray:
    omega = x * core.scale
    return _core_matrix(core, x) * np.exp(-1j * omega * model.delays)


def _sampled_violations(model: RationalModel, core: _Core, tol: float):
    xs = _sample_grid(model, core)
    smax = np.array([_sigma_max(_full_matrix(model, core, x)) for x in xs])
    over = smax > 1.0 + tol
    bands: list[tuple[float, float]] = []
    peaks: list[_Peak] = []
    k = 0
    while k < xs.size:
        if not over[k]:
            k += 1
            continue
        start = k
        while k + 1 < xs.size and over[k + 1]:
            k += 1
        bands.append((float(xs[start]), float(xs[k])))
        lo = xs[max(start - 1, 0)]
        hi = xs[min(k + 1, xs.size - 1)]
        for x in _peaks_in(lambda t: _sigma_max(_full_matrix(model, core, t)), float(lo), float(hi), 21,
                           poles=core.a):
            u, s, vh = np.linalg.svd(_full_matrix(model, core, x))
            keep = s > 1.0 + tol
            if np.any(keep):
                peaks.append(_Peak(x, s[keep], u[:, keep], vh[keep].conj().T))
        k += 1
    return np.zeros(0), bands, peaks, float(np.max(smax))


def _to_hz(model: RationalModel, core: _Core, x) -> Any:
    return model.carrier_hz + np.asarray(x, dtype=float) * core.scale / (2 * np.pi)


def passivity_report(model: RationalModel, *, tol: float = 1e-9) -> PassivityReport:
    """Assess passivity of a rational model over all real frequencies.

    Uses the Hamiltonian test of the delay-free core when the delays are
    port-separable (exact, ``certified=True``); otherwise falls back to a dense
    sweep of the full delayed response over the fitted band (extended by 50 %
    on each side) and every pole's neighbourhood (``certified=False``).

    Parameters
    ----------
    model
        The model to test (power-normalized scattering parameters).
    tol
        Singular values up to ``1 + tol`` are accepted as passive.

    Examples
    --------
    >>> import numpy as np
    >>> lossy = RationalModel(("o1",), 0.0, np.array([-1.0 + 0j]), np.array([[[0.9 + 0j]]]), np.array([[0.0]]))
    >>> passivity_report(lossy).passive
    True
    >>> gain = lossy.replace(residues=np.array([[[1.5 + 0j]]]))
    >>> r = passivity_report(gain)
    >>> r.passive, r.certified, np.round(np.array(r.violation_bands_hz[0]) * 2 * np.pi, 6)
    (False, True, array([-1.118034,  1.118034]))
    """
    scale = _scale_of(model)
    core = _core(model, scale)
    sd = _sigma_max(core.D)
    if model.port_delays() is None:
        crossings, bands, peaks, smax = _sampled_violations(model, core, tol)
        method, certified = "sampled", False
    elif core.a.size == 0:
        crossings, bands, peaks = np.zeros(0), ([(-np.inf, np.inf)] if sd > 1 + tol else []), []
        smax = sd
        method, certified = "no-states", True
    else:
        if abs(sd - 1.0) < 1e-12:
            core = _Core(core.a, core.B, core.C, core.D * (1.0 - 1e-12), scale)
        crossings, bands, peaks = _hamiltonian_violations(core, tol)
        method, certified = "hamiltonian", True
        if peaks:
            smax = max(float(p.sigma[0]) for p in peaks)
        else:
            smax = max(sd, max(_sigma_max(_core_matrix(core, x)) for x in _sample_grid(model, core)))
    if bands and any(np.isinf(b).any() for b in bands):
        smax = max(smax, sd)
    return PassivityReport(
        passive=not bands,
        certified=certified,
        method=method,
        crossing_freqs_hz=_to_hz(model, core, crossings),
        violation_bands_hz=tuple((float(_to_hz(model, core, lo)), float(_to_hz(model, core, hi))) for lo, hi in bands),
        peak_freqs_hz=_to_hz(model, core, [p.x for p in peaks]),
        peak_sigmas=np.asarray([float(p.sigma[0]) for p in peaks]),
        max_sigma=float(smax),
        sigma_max_d=sd,
    )


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #
def _band_gram(a: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact band-averaged Gram matrix ``G_kl = <phi_k, phi_l>`` and ``h_k = <phi_k, 1>``.

    ``phi_k(x) = 1 / (1j x - a_k)`` on ``[lo, hi]`` with ``Re a_k < 0``. The
    logarithms are continuous along the band because their arguments stay in
    the right half-plane.
    """
    width = hi - lo

    def log_plus(x):  # antiderivative pieces of 1/(1j x - a): (1/1j) log(1j x - a)
        return np.log(1j * x - a)

    def log_minus(x):  # of 1/(1j x + conj(a)): (1/1j) log(-1j x - conj(a))
        return np.log(-1j * x - np.conj(a))

    lp = log_plus(hi) - log_plus(lo)  # (P,) for index l
    lm = log_minus(hi) - log_minus(lo)  # (P,) for index k
    denom = a[None, :] + np.conj(a)[:, None]  # a_l + conj(a_k)
    G = -(lp[None, :] - lm[:, None]) / (1j * denom) / width
    G = 0.5 * (G + G.conj().T)
    h = 1j * lm / width  # <phi_k, 1> = mean of conj(phi_k)
    return G, h


def _band_limits(model: RationalModel, scale: float, band_hz) -> tuple[float, float]:
    band = band_hz if band_hz is not None else model.metadata.get("fit_band_hz")
    if band is None:
        a = model.poles / scale
        if a.size == 0:
            return -1.0, 1.0
        return float(np.min(a.imag - 3 * np.abs(a.real))), float(np.max(a.imag + 3 * np.abs(a.real)))
    lo, hi = (2 * np.pi * (float(b) - model.carrier_hz) / scale for b in band)
    if not hi > lo:
        raise ValueError("band_hz must be an increasing (f_lo, f_hi) pair")
    return lo, hi


def enforce_passivity(
    model: RationalModel,
    *,
    margin: float = 1e-6,
    max_iter: int = 50,
    tol: float = 1e-9,
    preserve_reciprocity: bool | None = None,
    band_hz: tuple[float, float] | None = None,
) -> RationalModel:
    """Return a passive model by minimum-energy perturbation of the residues.

    Parameters
    ----------
    model
        A stable rational model. Its poles and delays are kept.
    margin
        Violating singular values are pushed to ``1 - margin``; ``sigma_max(D)``
        is clipped to ``1 - margin`` first if needed (the residues are then
        re-fitted in band to compensate).
    max_iter
        Maximum perturbation iterations; a :class:`RuntimeWarning` is issued if
        the model is still not passive.
    tol
        Passivity tolerance used for detection.
    preserve_reciprocity
        Keep ``S = S^T``. ``None`` keeps it when the input model is reciprocal.
    band_hz
        Band ``(f_lo, f_hi)`` (Hz) over which the perturbation energy is
        minimized; defaults to the fitted band in ``model.metadata``.

    Returns
    -------
    RationalModel
        The enforced model; ``metadata["passivity_enforcement"]`` records the
        iterations, the in-band relative RMS change and the final certificate.

    Examples
    --------
    >>> import numpy as np
    >>> gain = RationalModel(("o1",), 0.0, np.array([-1.0 + 0j]), np.array([[[1.5 + 0j]]]), np.array([[0.0]]),
    ...                      metadata={"fit_band_hz": [-1.0, 1.0]})
    >>> fixed = enforce_passivity(gain)
    >>> passivity_report(fixed).passive
    True
    """
    scale = _scale_of(model)
    n = model.n_ports
    a = model.poles / scale
    residues = model.residues / scale  # normalized
    d = model.d.copy()
    reciprocal = (
        np.allclose(residues, np.swapaxes(residues, 1, 2), rtol=0, atol=1e-12 * max(1.0, np.abs(residues).max(
            initial=0.0)))
        and np.allclose(d, d.T, rtol=0, atol=1e-12)
        and np.allclose(model.delays, model.delays.T, rtol=0, atol=1e-18)
    )
    if preserve_reciprocity is None:
        preserve_reciprocity = reciprocal
    partners = model.conjugate_partners()
    mask = (np.max(np.abs(residues), axis=0, initial=0.0) > 0) | (np.abs(d) > 0)
    lo, hi = _band_limits(model, scale, band_hz)
    G, h = _band_gram(a, lo, hi) if a.size else (np.zeros((0, 0)), np.zeros(0))
    if a.size:
        # The Gram matrix of many nearby poles is numerically semi-definite: a small ridge keeps the
        # Cholesky factorization (and the minimum-norm solve) well defined.
        ridge = 1e-12 * float(np.max(np.abs(np.diag(G))))
        while True:
            try:
                chol = np.linalg.cholesky(G + ridge * np.eye(a.size))
                break
            except np.linalg.LinAlgError:
                ridge *= 1e3

        def g_solve(rhs):
            return np.linalg.solve(chol.conj().T, np.linalg.solve(chol, rhs))

    original = model
    sampled = model.port_delays() is None
    if sampled:
        warnings.warn(
            "the model's delays are not port-separable; passivity is enforced on a dense frequency sweep, "
            "which is not a certificate outside the sampled points",
            RuntimeWarning,
            stacklevel=2,
        )

    # 1. Asymptotic passivity: sigma_max(D) <= 1 - margin.
    u_d, s_d, vh_d = np.linalg.svd(d)
    d_changed = bool(s_d[0] > 1.0 - margin)
    if d_changed:
        d_new = (u_d * np.minimum(s_d, 1.0 - margin)) @ vh_d
        if partners is not None:
            d_new = d_new.real.astype(complex)
        if preserve_reciprocity:
            d_new = 0.5 * (d_new + d_new.T)
        delta_d = d_new - d
        if a.size:
            # Least-squares in-band compensation: minimize ||phi . dr + delta_d||^2 entry by entry.
            dr = -g_solve(h)[:, None, None] * delta_d[None]
            residues = residues + np.where(mask[None], dr, 0.0)
        d = d_new

    def build(res_n, d_mat) -> RationalModel:
        return model.replace(residues=res_n * scale, d=d_mat)

    def detect(candidate: RationalModel) -> list[_Peak]:
        core = _core(candidate, scale)
        if sampled:
            return _sampled_violations(candidate, core, tol)[2]
        if core.a.size == 0:
            return []
        return _hamiltonian_violations(core, tol)[2]

    steps = 0
    current = build(residues, d)
    peaks = detect(current)
    while peaks and a.size and steps < max_iter:
        rows_w: list[np.ndarray] = []
        rows_phi: list[np.ndarray] = []
        targets: list[float] = []
        if partners is not None:  # real system: use x >= 0 and add the mirror image explicitly
            peaks = [p for p in peaks if p.x >= 0]
        for peak in peaks:
            xs = [peak.x] if partners is None or peak.x == 0 else [peak.x, -peak.x]
            for sign_index, x in enumerate(xs):
                phase = np.exp(-1j * x * scale * current.delays) if sampled else np.ones((n, n))
                for q in range(peak.sigma.size):
                    u = peak.u[:, q]
                    v = peak.v[:, q]
                    if sign_index == 1:  # mirrored constraint of a real system: S(-x) = conj(S(x))
                        u, v = np.conj(u), np.conj(v)
                    w = np.conj(u)[:, None] * v[None, :] * phase
                    if preserve_reciprocity:
                        w = 0.5 * (w + w.T)
                    w = np.where(mask, w, 0.0)
                    rows_w.append(w)
                    rows_phi.append(1.0 / (1j * x - a))
                    targets.append((1.0 - margin) - float(peak.sigma[q]))
        W = np.asarray(rows_w)  # (L, n, n)
        Phi = np.asarray(rows_phi)  # (L, P)
        b = np.asarray(targets)
        g_phi = g_solve(np.conj(Phi).T)  # (P, L): G^-1 conj(phi_l)
        F = Phi @ g_phi  # F_ml = phi_m^T G^-1 conj(phi_l)
        Wm = np.einsum("mij,lij->ml", W, np.conj(W))
        M = np.real(Wm * F)
        lam, *_ = np.linalg.lstsq(0.5 * M, b, rcond=None)
        dr = 0.5 * np.einsum("l,lij,kl->kij", lam, np.conj(W), g_phi)
        if partners is not None:
            dr = 0.5 * (dr + np.conj(dr[partners]))
        residues = residues + dr
        current = build(residues, d)
        steps += 1
        peaks = detect(current)
    passive = not peaks and _sigma_max(current.d) <= 1.0 + tol
    if not passive:
        warnings.warn(
            f"passivity enforcement did not converge in {max_iter} iterations", RuntimeWarning, stacklevel=2
        )
    # In-band relative RMS change of the (delay-free) response.
    xs = np.linspace(lo, hi, 2001)
    original_core = _core(original, scale)
    before = np.array([_core_matrix(original_core, x) for x in xs])
    final_core = _core(current, scale)
    after = np.array([_core_matrix(final_core, x) for x in xs])
    change = float(np.sqrt(np.mean(np.abs(after - before) ** 2)) / max(np.sqrt(np.mean(np.abs(before) ** 2)), 1e-300))
    info = {
        "iterations": int(steps),
        "margin": float(margin),
        "d_clipped": d_changed,
        "relative_rms_change": change,
        "passive": bool(passive),
        "certified": not sampled,
        "preserve_reciprocity": bool(preserve_reciprocity),
    }
    return current.replace(metadata={**current.metadata, "passivity_enforcement": info})
