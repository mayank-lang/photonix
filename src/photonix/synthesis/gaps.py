"""Map designed power couplings to physical coupler gaps."""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["coupling_to_gap"]


def coupling_to_gap(target: float, *, wl0: float = 1.55, length: float = 0.0, radius: float | None = None,
                    curved_arms: int = 1, coupler=None, bracket: tuple[float, float] = (0.10, 0.40),
                    **coupler_kwargs) -> float:
    """Gap (µm) at which ``coupler`` reaches power coupling ``target`` at ``wl0``.

    ``coupler`` defaults to :func:`photonix.components.evanescent_coupler`
    (calibrated for 500 x 220 nm SOI strips); ``radius`` adds the curved-gap
    contribution of a ring or racetrack (``length`` is its straight coupling
    section). Root finding uses Brent's method on the monotone part of the
    coupling curve inside ``bracket``; a target outside the bracketed range
    raises ``ValueError``.

    Examples
    --------
    >>> import photonix as px
    >>> g = coupling_to_gap(0.05, radius=10.0)
    >>> k = px.power(px.components.evanescent_coupler(gap=g, length=0.0, bend_radius=10.0)[("o1", "o3")])
    >>> abs(float(k) - 0.05) < 1e-10
    True
    """
    from scipy.optimize import brentq

    if coupler is None:
        from photonix.components import evanescent_coupler as coupler
    if not 0 < target < 1:
        raise ValueError("target coupling must lie in (0, 1)")

    def k_of(gap):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            s = coupler(wl=wl0, gap=gap, length=length, bend_radius=0.0 if radius is None else radius,
                        curved_arms=curved_arms, **coupler_kwargs)
        return float(np.abs(np.asarray(s[("o1", "o3")])) ** 2)

    lo, hi = bracket
    k_lo, k_hi = k_of(lo), k_of(hi)
    gaps = np.linspace(lo, hi, 64)
    ks = np.array([k_of(g) for g in gaps])
    if np.any(np.diff(ks) > 0):
        raise ValueError("coupling is not monotone in gap over the bracket (over-coupled section); shorten the coupler")
    if not (min(k_lo, k_hi) <= target <= max(k_lo, k_hi)):
        raise ValueError(f"target {target} outside the achievable range {k_hi:.4g}..{k_lo:.4g} for gaps {bracket} um")
    return float(brentq(lambda g: k_of(g) - target, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200))
