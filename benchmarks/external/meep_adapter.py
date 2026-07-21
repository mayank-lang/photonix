"""Meep adapter -- external reference numbers via the photonix Meep backend.

Implements the benchmark cases through :mod:`photonix.em.meep`, so the same code
path photonix users call for FDTD also produces the external-solver column of the
benchmark table. It stays optional: every entry point raises ``ImportError`` (not a
silent skip) when Meep is absent, which ``run.py`` reports as a clean skip.

Contract: ``run_all()`` returns ``{case_key: float}`` for the cases this solver can
do -- mode ``n_eff`` via MPB, transmission via Meep FDTD. Keys match
``benchmarks/cases.py``.
"""
from __future__ import annotations

import numpy as np


def _require_meep():
    from photonix.em.meep import require_meep

    require_meep()


def neff_soi_strip_te0() -> float:
    """500x220 nm Si strip TE0 effective index via MPB."""
    from photonix.em import meep

    return meep.n_eff(
        wl=1.55, width=0.5, thickness=0.22, n_core=3.4757, n_clad=1.444,
        resolution=40, num_modes=1,
    )


def width_step_te_T() -> float:
    """TE transmission of the 0.45->0.55 um width step via Meep FDTD."""
    from photonix.em import meep

    dx = dy = 0.02
    nco, ncl = 3.4757, 1.444
    sy = 3.0           # transverse window (um)
    lin = lout = 2.0   # input / output waveguide lengths (um)
    ny = int(round(sy / dy))
    nin = int(round(lin / dx))
    nout = int(round(lout / dx))
    nx = nin + nout
    y = (np.arange(ny) - ny / 2 + 0.5) * dy

    eps = np.full((ny, nx), ncl**2, float)
    half_in, half_out = 0.45 / 2, 0.55 / 2
    eps[np.abs(y) < half_in, :nin] = nco**2
    eps[np.abs(y) < half_out, nin:] = nco**2

    s = meep.waveguide_sparams(
        eps, dx=dx, dy=dy, wl=1.55,
        src_col=int(round(0.5 / dx)),
        in_mon_col=int(round(1.0 / dx)),
        out_mon_col=nx - int(round(0.5 / dx)),
        polarization="te",
    )
    return abs(s[("o1", "o2")]) ** 2


def run_all() -> dict[str, float]:
    _require_meep()
    out: dict[str, float] = {}
    for key, fn in {
        "soi_strip_te0_neff": neff_soi_strip_te0,
        "width_step_te_T": width_step_te_T,
    }.items():
        try:
            out[key] = float(fn())
        except NotImplementedError:
            pass
    return out
