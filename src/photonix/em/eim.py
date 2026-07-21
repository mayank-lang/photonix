"""Effective Index Method (EIM) for polarization-resolved 2-D waveguide modes.

EIM approximates a 2-D channel waveguide by two successive 1-D slab solves:

1. **Vertical** slab (the layer stack through the core) gives an effective index
   ``n_v`` for the guided region.
2. **Lateral** slab of the core width, with core index ``n_v`` and the side
   (cladding) index, gives the final ``n_eff``.

The polarization rotates between the two steps: a quasi-**TE** channel mode is
vertical-TE then lateral-TM, and quasi-**TM** is vertical-TM then lateral-TE.

Both 1-D solves use :mod:`photonix.em.slab`, which is validated to <0.1% against
the analytic transcendental. EIM is therefore **exact in the wide-width (slab)
limit** and is a well-established *approximation* for finite widths (typically a
few percent versus a full-vector solver for high-contrast strips). It correctly
captures the TE/TM splitting and trends, and is fast and robust on CPU.

For research requiring full-vector accuracy on strongly-confined or hybrid modes,
a full-vector finite-difference solver is the right tool; EIM is the fast,
validated, polarization-aware estimator that ships today.
"""
from __future__ import annotations

from .slab import slab_neff

__all__ = ["neff", "n_eff_te", "n_eff_tm"]


def neff(
    *,
    width: float = 0.5,
    thickness: float = 0.22,
    wl: float = 1.55,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    polarization: str = "te",
    lateral_clad: float | None = None,
    resolution: int = 60,
) -> float:
    """Effective index of a channel waveguide via the Effective Index Method.

    Parameters
    ----------
    width, thickness : float
        Core lateral width and vertical thickness (µm).
    polarization : str
        ``"te"`` (quasi-TE) or ``"tm"`` (quasi-TM).
    lateral_clad : float, optional
        Effective index beside the core. Defaults to ``n_clad`` (a fully-etched
        strip in uniform cladding); set to the slab index of the unetched region
        for a rib waveguide.
    resolution : int
        Grid resolution (points per µm) for each 1-D solve.

    Returns
    -------
    float
        Fundamental effective index.

    Examples
    --------
    >>> import photonix.em as em
    >>> te = em.eim.neff(width=0.5, thickness=0.22, polarization="te")
    >>> tm = em.eim.neff(width=0.5, thickness=0.22, polarization="tm")
    >>> te > tm > 1.444
    True
    """
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    vpol = polarization
    lpol = "tm" if polarization == "te" else "te"
    n_v = slab_neff(thickness=thickness, n_core=n_core, n_clad=n_clad, wl=wl,
                    resolution=resolution, polarization=vpol)
    lc = n_clad if lateral_clad is None else lateral_clad
    return slab_neff(thickness=width, n_core=n_v, n_clad=lc, wl=wl,
                     resolution=resolution, polarization=lpol)


def n_eff_te(**kwargs) -> float:
    """Quasi-TE effective index (see :func:`neff`)."""
    return neff(polarization="te", **kwargs)


def n_eff_tm(**kwargs) -> float:
    """Quasi-TM effective index (see :func:`neff`)."""
    return neff(polarization="tm", **kwargs)
