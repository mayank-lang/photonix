"""Cross-solver benchmarks: independent methods must agree on the same structure.

This is the seed of the validation-as-a-product effort: rather than only checking
each solver against analytic limits, we check that *physically independent*
discretizations agree on a quantity of interest. Here EME (modal expansion +
Redheffer star product) and FDFD (full 2-D Helmholtz scattering with PML) compute
the transmission of the same planar width step and must agree to ~1%.
"""
from __future__ import annotations

import numpy as np

from photonix.em.eme import Section, eme_smatrix
from photonix.em.fdfd import waveguide_sparams

WL, NCO, NCL = 1.55, 3.4757, 1.444


def _col(y, w):
    return np.where(np.abs(y) < w / 2, NCO**2, NCL**2)


def test_eme_fdfd_transmission_agree_on_width_step():
    """EME and FDFD agree on the TE transmission of a gentle width step (<1%)."""
    dy = dx = 0.04
    y = np.arange(-1.5, 1.5 + 1e-9, dy)
    nx = int(3.0 / dx)
    w1, w2 = 0.45, 0.55

    # FDFD: 2-D (y transverse, x propagation), abrupt step at mid-domain
    eps = np.empty((len(y), nx))
    for ix in range(nx):
        eps[:, ix] = _col(y, w1) if ix / nx < 0.5 else _col(y, w2)
    s = waveguide_sparams(
        eps, dx=dx, dy=dy, wl=WL, src_col=15,
        in_mon_col=int(0.3 * nx), out_mon_col=int(0.7 * nx),
        in_eps_col=15, out_eps_col=nx - 15, npml=12, mode=0,
    )
    t_fdfd = abs(s[("o1", "o2")]) ** 2

    # EME: same step as two sections
    r = eme_smatrix([Section(_col(y, w1), 1.2), Section(_col(y, w2), 1.2)], dy, WL, 6, "te")
    t_eme = abs(r.Tf[0, 0]) ** 2

    assert abs(t_fdfd - t_eme) < 0.01, (t_fdfd, t_eme)
    assert 0.95 < t_eme < 1.0           # physically bounded, near-unity for a gentle step


def test_eme_no_nan_with_many_modes():
    """The beta~0 floor keeps the cascade finite even when modes are non-propagating."""
    dy = 0.04
    y = np.arange(-1.5, 1.5 + 1e-9, dy)
    r = eme_smatrix([Section(_col(y, 0.45), 1.0), Section(_col(y, 0.6), 1.0)], dy, WL, 12, "te")
    assert np.all(np.isfinite(np.abs(r.Tf)))


def test_tm_fdfd_straight_lossless():
    """The TM (Hz) FDFD reproduces a lossless straight guide (|S21|=1, |S11|=0)."""
    dy = dx = 0.035
    y = np.arange(-1.5, 1.5 + 1e-9, dy)
    nx = int(2.6 / dx)
    eps = np.tile(_col(y, 0.5)[:, None], (1, nx))
    s = waveguide_sparams(
        eps, dx=dx, dy=dy, wl=WL, src_col=12,
        in_mon_col=int(0.3 * nx), out_mon_col=int(0.7 * nx),
        in_eps_col=12, out_eps_col=nx - 12, npml=12, polarization="tm",
    )
    assert abs(abs(s[("o1", "o2")]) ** 2 - 1.0) < 5e-3
    assert abs(s[("o1", "o1")]) ** 2 < 5e-3


def test_tm_eme_fdfd_transmission_agree():
    """Independent TM solvers (EME modal vs FDFD full-wave) agree on a width step."""
    dy = dx = 0.04
    y = np.arange(-1.5, 1.5 + 1e-9, dy)
    nx = int(3.0 / dx)
    w1, w2 = 0.45, 0.55

    eps = np.empty((len(y), nx))
    for ix in range(nx):
        eps[:, ix] = _col(y, w1) if ix / nx < 0.5 else _col(y, w2)
    s = waveguide_sparams(
        eps, dx=dx, dy=dy, wl=WL, src_col=15,
        in_mon_col=int(0.3 * nx), out_mon_col=int(0.7 * nx),
        in_eps_col=15, out_eps_col=nx - 15, npml=12, polarization="tm",
    )
    t_fdfd = abs(s[("o1", "o2")]) ** 2

    r = eme_smatrix([Section(_col(y, w1), 1.2), Section(_col(y, w2), 1.2)], dy, WL, 6, "tm")
    t_eme = abs(r.Tf[0, 0]) ** 2

    assert abs(t_fdfd - t_eme) < 0.01, (t_fdfd, t_eme)
