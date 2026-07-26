"""Scalar waveguide mode solving (thin layer over :mod:`photonix.em.fde`).

.. deprecated:: 0.2
   Prefer :mod:`photonix.em` directly. Everything here forwards to
   :mod:`photonix.em.fde`; this module exists so existing
   ``photonix.modes`` code keeps working.

Why this module no longer has its own solver
--------------------------------------------
photonix used to ship *two* independent scalar Helmholtz eigensolvers -- one
here and one in :mod:`photonix.em.fde` -- exporting the same three names
(``solve_modes``, ``n_eff``, ``group_index``) with the same signatures and
returning **different numbers**: 2.644 vs 2.612 for the standard 500x220 nm SOI
strip at 1.55 um. The difference came from the discretization, not the physics:
:mod:`photonix.em` uses subpixel permittivity averaging plus Richardson
extrapolation, this module used a staircased grid and neither.

There is now one implementation. ``photonix.modes.n_eff`` and
``photonix.em.n_eff`` are the same function, so they cannot drift apart again.

Choosing a solver
-----------------
The scalar model is fast and robust, and exact in the scalar (low-contrast)
limit, but it *overestimates* the effective index of a high-contrast SOI strip
(scalar ~2.61 vs full-vector ~2.45 for TE0 at 500x220 nm). For high-contrast
geometry use the vectorial solvers instead::

    photonix.em.n_eff_vector       # semivectorial quasi-TE / quasi-TM
    photonix.em.n_eff_fullvector   # full-vector hybrid modes + polarization fraction

Results (n_eff, n_g, field profiles) feed directly into component models::

    from photonix.modes import n_eff
    neff_fn = lambda wl: n_eff(wl=float(wl), width=0.5, thickness=0.22)
    s = photonix.components.straight(wl=1.55, length=100.0, neff=neff_fn)
"""
from __future__ import annotations

import numpy as np

from photonix.em.fde import ModeData as ModeResult
from photonix.em.fde import group_index, n_eff, solve_modes

__all__ = ["ModeResult", "solve_modes", "n_eff", "group_index", "overlap"]


def overlap(field_a, field_b) -> float:
    """Normalized scalar field overlap integral ``|<a|b>|^2 / (<a|a><b|b>)``.

    For the power overlap between *vectorial* modes (which weights by ``1/eps``)
    use :func:`photonix.em.power_overlap` instead.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.random.default_rng(0).random((10, 10))
    >>> bool(abs(overlap(f, f) - 1.0) < 1e-12)
    True
    """
    a = np.asarray(field_a).ravel()
    b = np.asarray(field_b).ravel()
    num = abs(np.vdot(a, b)) ** 2
    den = np.vdot(a, a).real * np.vdot(b, b).real
    return float(num / den)
