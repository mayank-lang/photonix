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
    coupling curve inside ``bracket``: for a long coupler, whose ``k(gap)``
    is over-coupled (rises to a maximum) at small gaps, that is the segment
    from the last maximum to ``bracket[1]``. ``ValueError`` if the target is
    outside that segment's range, or is also reached in the over-coupled
    part (the gap would be ambiguous).

    Examples
    --------
    >>> import photonix as px
    >>> g = coupling_to_gap(0.05, radius=10.0)
    >>> k = px.power(px.components.evanescent_coupler(gap=g, length=0.0, bend_radius=10.0)[("o1", "o3")])
    >>> abs(float(k) - 0.05) < 1e-10
    True
    """
    from scipy.optimize import brentq, minimize_scalar

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
    gaps = np.linspace(lo, hi, 64)
    ks = np.array([k_of(g) for g in gaps])
    rising = np.flatnonzero(np.diff(ks) > 0)
    seg_lo, k_top = lo, float(ks[0])
    if rising.size:
        # over-coupled small gaps: k(gap) peaks, then falls monotonically up to hi.
        # Solve on that monotone large-gap segment, starting at the last maximum.
        j = int(rising[-1])
        a, b = gaps[max(j - 1, 0)], gaps[min(j + 2, gaps.size - 1)]
        peak = minimize_scalar(lambda g: -k_of(g), bounds=(a, b), method="bounded", options={"xatol": 1e-12})
        seg_lo, k_top = float(peak.x), float(-peak.fun)
        if k_top < ks[j + 1]:                                     # keep the best sampled maximum
            seg_lo, k_top = float(gaps[j + 1]), float(ks[j + 1])
    k_hi = k_of(hi)
    if not k_hi <= target <= k_top:
        raise ValueError(f"target {target} outside the monotone achievable range {k_hi:.4g}..{k_top:.4g} for gaps "
                         f"{seg_lo:.4g}..{hi} um")
    if rising.size and float(np.min(ks[gaps <= seg_lo], initial=k_top)) <= target:
        raise ValueError(f"target {target} is also reached in the over-coupled section (gaps below {seg_lo:.4g} um); "
                         "shorten the coupler or narrow the bracket")
    if target == k_top:
        return seg_lo
    return float(brentq(lambda g: k_of(g) - target, seg_lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200))
