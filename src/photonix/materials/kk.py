"""Kramers-Kronig consistency check for tabulated optical constants.

For any causal medium ``N(w) = n + i k`` is analytic in the upper half plane
and tends to 1 at high frequency, so ``n`` follows from ``k``. Data are only
known over a finite band, and the *singly subtractive* form (R. K. Ahrenkiel,
"Modified Kramers-Kronig analysis of optical spectra", J. Opt. Soc. Am. 61,
1651-1655 (1971), doi:10.1364/JOSA.61.001651) suppresses the truncation error
by anchoring the result at one measured point ``n(w0)``::

    n(w) = n(w0) + (2/pi) P int w' k(w') [1/(w'^2 - w^2) - 1/(w'^2 - w0^2)] dw'

The bracket decays as ``w'^-4`` instead of ``w'^-2``, so absorption outside the
table matters much less. The principal values are evaluated by subtracting the
pole (``int (g(w') - g(a))/(w' - a) dw' + g(a) ln((w2 - a)/(a - w1))``) on a
fine uniform frequency grid with ``k`` interpolated linearly.

This is a diagnostic: a residual larger than the data's stated uncertainty
flags inconsistent n, k sets, but a small residual cannot prove correctness, and
the residual grows with the absorption that lies outside the tabulated band.
Pure NumPy (not in the differentiable modelling path).
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["kramers_kronig_n"]

# numpy >= 2.0 renamed trapz -> trapezoid; photonix supports numpy >= 1.24.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # noqa: B009


def _pv_over_diff(nu: np.ndarray, g: np.ndarray, a: float) -> float:
    """``P int g(v) / (v - a) dv`` over the grid ``nu`` (uniform), for ``nu[0] < a < nu[-1]``."""
    ga = float(np.interp(a, nu, g))
    dv = nu - a
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (g - ga) / dv
    near = np.abs(dv) < 1e-12 * max(abs(a), 1.0)
    if np.any(near):
        h[near] = np.interp(a, nu, np.gradient(g, nu))
    return float(_trapz(h, nu) + ga * np.log((nu[-1] - a) / (a - nu[0])))


def _pv(nu: np.ndarray, g: np.ndarray, a: float) -> float:
    """``P int g(v) / (v^2 - a^2) dv``."""
    return (_pv_over_diff(nu, g, a) - float(_trapz(g / (nu + a), nu))) / (2.0 * a)


def kramers_kronig_n(wl: Any, k: Any, *, wl_ref: float, n_ref: float, wl_eval: Any = None,
                     n_grid: int = 40001) -> np.ndarray:
    """Real index predicted from tabulated ``k`` by singly subtractive Kramers-Kronig.

    Parameters
    ----------
    wl, k
        Tabulated wavelengths [um] and extinction coefficients (``k >= 0``).
    wl_ref, n_ref
        Anchor: the measured real index ``n_ref`` at ``wl_ref``.
    wl_eval
        Wavelengths at which to predict ``n``; defaults to ``wl``. Every
        evaluation point and the anchor must lie strictly inside the table.
    n_grid
        Points of the uniform frequency grid used for the integrals.

    Examples
    --------
    A single Lorentz oscillator is causal, so its ``n`` is recovered from ``k``:

    >>> import numpy as np
    >>> f = np.geomspace(0.01, 100.0, 4000)                          # frequency, f0 = 1
    >>> N = np.sqrt(1 + 2.0 / (1 - f**2 - 0.2j * f))
    >>> n_kk = kramers_kronig_n(1 / f, N.imag, wl_ref=10.0, n_ref=float(np.sqrt(1 + 2.0 / (1 - 0.01 - 0.02j)).real),
    ...                         wl_eval=[2.0, 1.0, 0.5])
    >>> exact = np.sqrt(1 + 2.0 / (1 - np.array([0.5, 1.0, 2.0])**2 - 0.2j * np.array([0.5, 1.0, 2.0]))).real
    >>> bool(np.max(np.abs(n_kk - exact)) < 2e-3)
    True
    """
    wl = np.asarray(wl, dtype=float)
    k = np.asarray(k, dtype=float)
    if wl.shape != k.shape or wl.ndim != 1 or wl.size < 3:
        raise ValueError("wl and k must be 1-D arrays of equal length >= 3")
    if np.any(k < 0):
        raise ValueError("k must be >= 0 (physics sign)")
    nu_tab = 1.0 / wl
    order = np.argsort(nu_tab)
    nu_tab, k_tab = nu_tab[order], k[order]
    nu = np.linspace(nu_tab[0], nu_tab[-1], int(n_grid))
    g = nu * np.interp(nu, nu_tab, k_tab)
    targets = wl if wl_eval is None else np.atleast_1d(np.asarray(wl_eval, dtype=float))
    a0 = 1.0 / float(wl_ref)
    span = (nu[0], nu[-1])

    def inside(a: float) -> bool:
        return bool(span[0] < a < span[1])

    if not inside(a0):
        raise ValueError("wl_ref must lie strictly inside the tabulated range")
    i0 = _pv(nu, g, a0)
    out = np.empty(targets.shape, dtype=float)
    for j, wt in enumerate(targets):
        a = 1.0 / float(wt)
        if not inside(a):
            raise ValueError(f"wl_eval {wt} um is not strictly inside the tabulated range")
        out[j] = n_ref + (2.0 / np.pi) * (_pv(nu, g, a) - i0)
    return out
