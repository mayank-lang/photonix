"""Low-pass prototype filters and periodic all-pole targets for resonator filters.

A lossless chain of ``N`` identical, series-coupled rings has an all-pole drop
response in ``z = exp(j phi)`` (``phi`` the round-trip phase), so its power
response is ``1 / P(sin^2(phi/2))`` with ``P`` a polynomial of degree ``N``.
Butterworth and Chebyshev responses map onto that form exactly through
``x = sin(phi/2) / sin(phi_c/2)``, the periodic analogue of the normalized
low-pass frequency (Little et al., JLT 15, 998, 1997; Madsen & Zhao, *Optical
Filter Design and Analysis*, 1999).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["lowpass_prototype", "chebyshev_epsilon", "periodic_target"]


def lowpass_prototype(kind: str, order: int, ripple_db: float | None = None) -> np.ndarray:
    """Element values ``g_0, ..., g_{N+1}`` of the normalized low-pass prototype.

    Butterworth: ``g_k = 2 sin((2k - 1) pi / (2N))``. Chebyshev with
    ``ripple_db`` pass-band ripple: the recursion of Matthaei, Young & Jones,
    *Microwave Filters, Impedance-Matching Networks and Coupling Structures*
    (1964), section 4.05.

    Examples
    --------
    >>> np.round(lowpass_prototype("butterworth", 3), 6).tolist()
    [1.0, 1.0, 2.0, 1.0, 1.0]
    >>> np.round(lowpass_prototype("chebyshev", 3, 0.1), 4).tolist()     # MYJ table 4.05-2(a)
    [1.0, 1.0316, 1.1474, 1.0316, 1.0]
    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    n = int(order)
    g = np.ones(n + 2)
    a = np.array([math.sin((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1)])
    if kind == "butterworth":
        g[1:n + 1] = 2 * a
        return g
    if kind != "chebyshev":
        raise ValueError("kind must be 'butterworth' or 'chebyshev'")
    if ripple_db is None or ripple_db <= 0:
        raise ValueError("chebyshev needs a positive ripple_db")
    beta = math.log(1.0 / math.tanh(ripple_db / 17.37))
    gamma = math.sinh(beta / (2 * n))
    b = np.array([gamma**2 + math.sin(k * math.pi / n) ** 2 for k in range(1, n + 1)])
    g[1] = 2 * a[0] / gamma
    for k in range(2, n + 1):
        g[k] = 4 * a[k - 2] * a[k - 1] / (b[k - 2] * g[k - 1])
    g[n + 1] = 1.0 if n % 2 else 1.0 / math.tanh(beta / 4) ** 2
    return g


def chebyshev_epsilon(ripple_db: float) -> float:
    """``epsilon`` of a Chebyshev response with ``ripple_db`` peak-to-valley ripple."""
    return math.sqrt(10 ** (ripple_db / 10) - 1)


def periodic_target(phi, *, kind: str, order: int, phi_c: float, ripple_db: float | None = None):
    """Target drop power ``1 / (1 + eps^2 F_N(x)^2)`` with ``x = sin(phi/2)/sin(phi_c/2)``.

    ``phi`` is the round-trip phase detuning (``2 pi df / FSR``) and ``phi_c``
    its value at the band edge (half the bandwidth): the -3 dB edge for
    Butterworth, the ripple-band edge for Chebyshev.

    Examples
    --------
    >>> round(float(periodic_target(0.2, kind="butterworth", order=3, phi_c=0.2)), 12)
    0.5
    """
    x = np.sin(np.asarray(phi) / 2) / math.sin(phi_c / 2)
    if kind == "butterworth":
        return 1.0 / (1.0 + x ** (2 * order))
    eps = chebyshev_epsilon(ripple_db if ripple_db is not None else 0.0)
    cheb = np.where(np.abs(x) <= 1, np.cos(order * np.arccos(np.clip(x, -1, 1))),
                    np.cosh(order * np.arccosh(np.maximum(np.abs(x), 1.0))) * np.sign(x) ** order)
    return 1.0 / (1.0 + eps**2 * cheb**2)
