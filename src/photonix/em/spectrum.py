"""Wavelength sweeps: turn single-frequency solvers into broadband S-parameters.

EME and FDFD compute a scattering response at one wavelength. ``sweep`` evaluates
a model across a wavelength array and stacks the results into an ``SDict`` whose
values are arrays over wavelength -- exactly the broadband form the circuit
solver and :func:`photonix.viz.plot_spectrum` already consume. Sweeps are
embarrassingly parallel; here they run sequentially on CPU.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from photonix.core.types import SDict

__all__ = ["sweep"]


def sweep(model: Callable[..., SDict], wls, **kwargs) -> SDict:
    """Evaluate ``model(wl=w, **kwargs)`` over ``wls`` -> broadband ``SDict``.

    Parameters
    ----------
    model : callable
        Returns an ``SDict`` for a single ``wl`` keyword.
    wls : array
        Wavelengths (µm).
    **kwargs
        Forwarded to ``model`` at every wavelength.

    Returns
    -------
    SDict
        Each value is a complex array over ``wls``.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> from photonix.em import sweep
    >>> from photonix.em.components import taper
    >>> wls = np.linspace(1.5, 1.6, 5)
    >>> S = sweep(taper, wls, width1=0.5, width2=0.9, length=12.0,
    ...           num_sections=8, num_modes=4, points=141)
    >>> S[("o1", "o2")].shape
    (5,)
    """
    wls = np.asarray(wls, float)
    results = [model(wl=float(w), **kwargs) for w in wls]
    keys = set()
    for r in results:
        keys.update(r.keys())
    out: SDict = {}
    for k in keys:
        out[k] = np.array([complex(r.get(k, 0j)) for r in results])
    return out
