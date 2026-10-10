"""Fiber-to-chip coupling from Gaussian mode overlap.

For two Gaussian beams with 1/e^2 intensity radii ``w1`` and ``w2`` (half the
mode-field diameter), lateral offset ``d`` and relative tilt ``theta`` in a
medium of index ``n``, the power coupling along one transverse axis is::

    eta = 2 w1 w2 / (w1^2 + w2^2)
          * exp(-2 d^2 / (w1^2 + w2^2))
          * exp(-(k theta)^2 w1^2 w2^2 / (2 (w1^2 + w2^2))),   k = 2 pi n / wl

and the 2-D efficiency is the product of the x and y factors. These closed
forms follow from the overlap integral of the two (separable) Gaussians.
"""
from __future__ import annotations

import math

from photonix.core.backend import xp
from photonix.core.types import SDict

__all__ = ["gaussian_overlap", "edge_coupler", "SMF28_MFD_1550"]

#: Nominal SMF-28 mode-field diameter at 1550 nm [µm] (Corning datasheet:
#: 10.4 +/- 0.5 µm).
SMF28_MFD_1550 = 10.4


def _axis_efficiency(w1, w2, offset, tilt_rad, k):
    s = w1 * w1 + w2 * w2
    return (2.0 * w1 * w2 / s) * xp.exp(-2.0 * offset * offset / s) \
        * xp.exp(-(k * tilt_rad) ** 2 * (w1 * w1) * (w2 * w2) / (2.0 * s))


def gaussian_overlap(
    *,
    wl=1.55,
    mfd1: float = SMF28_MFD_1550,
    mfd2: float = SMF28_MFD_1550,
    mfd1_y: float | None = None,
    mfd2_y: float | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    tilt_deg: float = 0.0,
    n_medium: float = 1.0,
):
    """Power coupling efficiency between two (elliptical) Gaussian modes.

    ``mfd*`` are mode-field diameters in µm along x (and optionally y; a
    missing y value means a circular mode). ``tilt_deg`` is an angular
    misalignment in the x-z plane.

    Examples
    --------
    >>> float(gaussian_overlap(mfd1=10.4, mfd2=10.4))
    1.0
    >>> # 3 µm chip mode into SMF-28: (2 * 1.5 * 5.2 / (1.5^2 + 5.2^2))^2
    >>> round(float(gaussian_overlap(mfd1=3.0, mfd2=10.4)), 4)
    0.2837
    """
    k = 2.0 * math.pi * n_medium / xp.asarray(wl)
    w1x, w2x = 0.5 * xp.asarray(mfd1), 0.5 * xp.asarray(mfd2)
    w1y = w1x if mfd1_y is None else 0.5 * xp.asarray(mfd1_y)
    w2y = w2x if mfd2_y is None else 0.5 * xp.asarray(mfd2_y)
    tilt = xp.asarray(tilt_deg) * math.pi / 180.0
    return _axis_efficiency(w1x, w2x, xp.asarray(offset_x), tilt, k) \
        * _axis_efficiency(w1y, w2y, xp.asarray(offset_y), 0.0, k)


def edge_coupler(
    *,
    wl=1.55,
    mfd_chip: float = SMF28_MFD_1550,
    mfd_fiber: float = SMF28_MFD_1550,
    mfd_chip_y: float | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    tilt_deg: float = 0.0,
    n_medium: float = 1.0,
    n_chip: float | None = None,
    excess_loss_db: float = 0.0,
) -> SDict:
    """Edge coupler / spot-size converter between ``o1`` (chip) and ``o2`` (fiber).

    The transmission is the Gaussian overlap (:func:`gaussian_overlap`) of
    the chip-side spot with the fiber mode, times an optional ``excess_loss_db``
    for the on-chip converter. If ``n_chip`` is given, a normal-incidence
    Fresnel facet between ``n_chip`` and ``n_medium`` adds reflection
    ``r = (n_chip - n_medium) / (n_chip + n_medium)`` on both ports and reduces
    the transmission by ``1 - r^2``; omit it for anti-reflection-coated or
    index-matched facets.

    The defaults describe an ideally mode-matched facet; set ``mfd_chip`` for
    a real converter (for example ~3 µm with a lensed fiber of matching MFD).

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.edge_coupler(mfd_chip=10.4, offset_x=1.0)
    >>> import math
    >>> abs(float(px.power(s[("o1", "o2")])) - math.exp(-2 / (2 * 5.2**2))) < 1e-12
    True
    """
    eta = gaussian_overlap(
        wl=wl, mfd1=mfd_chip, mfd2=mfd_fiber, mfd1_y=mfd_chip_y,
        offset_x=offset_x, offset_y=offset_y, tilt_deg=tilt_deg, n_medium=n_medium,
    )
    amp = xp.sqrt(eta) * 10.0 ** (-xp.asarray(excess_loss_db) / 20.0)
    wl_arr = xp.asarray(wl)
    ones = xp.ones_like(xp.real(wl_arr), dtype=float)
    out: SDict = {}
    if n_chip is not None:
        r = (xp.asarray(n_chip) - n_medium) / (xp.asarray(n_chip) + n_medium)
        amp = amp * xp.sqrt(1.0 - r * r)
        rr = xp.asarray(r * ones, dtype=complex)
        out[("o1", "o1")] = rr
        out[("o2", "o2")] = -rr
    t = xp.asarray(amp * ones, dtype=complex)
    out[("o1", "o2")] = t
    out[("o2", "o1")] = t
    return out
