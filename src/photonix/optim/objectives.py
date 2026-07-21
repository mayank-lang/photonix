"""Differentiable figures of merit for inverse design.

Each objective maps a scattering dictionary (and optionally a wavelength array)
to a scalar *loss* to be **minimized**. They are pure functions of backend arrays
so ``grad`` flows through them.
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.sparams import power
from photonix.core.types import PortPair, SDict

__all__ = [
    "target_transmission",
    "match_spectrum",
    "insertion_loss",
    "extinction_ratio",
    "flatness",
]


def target_transmission(sdict: SDict, port: PortPair, target) -> float:
    """Mean squared error between ``|S[port]|**2`` and a scalar/array ``target``.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.directional_coupler(coupling=0.4)
    >>> float(px.optim.target_transmission(s, ("o1", "o3"), 0.5)) > 0
    True
    """
    t = power(sdict[port])
    return xp.mean((t - xp.asarray(target)) ** 2)


def match_spectrum(sdict: SDict, port: PortPair, target_curve) -> float:
    """MSE between a transmission spectrum and a target curve (same length)."""
    t = power(sdict[port])
    return xp.mean((t - xp.asarray(target_curve)) ** 2)


def insertion_loss(sdict: SDict, port: PortPair) -> float:
    """Mean insertion loss in dB for ``port`` (positive = lossy)."""
    t = power(sdict[port])
    return -10.0 * xp.mean(xp.log10(t + 1e-12))


def extinction_ratio(sdict: SDict, port: PortPair) -> float:
    """Negative extinction ratio (so minimizing increases contrast).

    Returns ``-(max - min)`` of the transmission over the sweep; minimizing it
    drives a deep, high-contrast response (e.g. a filter notch).
    """
    t = power(sdict[port])
    return -(xp.max(t) - xp.min(t))


def flatness(sdict: SDict, port: PortPair) -> float:
    """Passband flatness penalty: variance of the transmission over the sweep."""
    t = power(sdict[port])
    return xp.var(t)
