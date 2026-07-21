"""Cross-section mode solving via MPB, returned as native photonix types.

This is the MPB half of the Meep backend: it solves the guided modes of a 2-D
cross-section and returns a :class:`photonix.em.fde_vector.VectorModeData`, so an
MPB result is a drop-in cross-check for the in-house full-vector FDE solver
(``solve_modes_fullvector``) and the source of truth for Meep eigenmode injection.

It accepts either explicit rectangular-waveguide parameters (matching
:func:`photonix.em.geometry.rectangular_waveguide`) or an arbitrary ``eps`` grid.
The grid is handed to MPB as a raw float64 epsilon array set as the solver's
``default_material`` (Meep's *epsilon-input* / ``MATERIAL_FILE`` mechanism, which
MPB bilinearly interpolates over the lattice). It is **not** passed as a
:class:`meep.MaterialGrid`: Meep's mode solver (``libpympb``) evaluates only
``Medium`` / epsilon-array / user-function / metal materials and hard-aborts on a
``MaterialGrid`` -- that class is supported by the FDTD engine only (see
:mod:`.materials`). Effective indices come from MPB's ``find_k`` (target
frequency -> propagation constant) via the unit bridge in :mod:`._guard`.
"""
from __future__ import annotations

import numpy as np

from ._guard import meep_frequency, n_eff_from_k, require_mpb

__all__ = ["solve_modes", "n_eff"]

_INF = 1.0e20  # Meep's "infinite" extent along the invariant (propagation) axis


def _build_solver(
    mpb, mp, *, wl, width, thickness, n_core, n_clad, margin, eps, grid, resolution,
    num_modes,
):
    """Construct an ``mpb.ModeSolver`` for the requested cross-section."""
    if eps is None:
        sx, sy = width + 2 * margin, thickness + 2 * margin
        lattice = mp.Lattice(size=mp.Vector3(sx, sy, 0))
        geometry = [
            mp.Block(
                size=mp.Vector3(width, thickness, _INF),
                center=mp.Vector3(),
                material=mp.Medium(epsilon=n_core**2),
            )
        ]
        default = mp.Medium(epsilon=n_clad**2)
        ny = max(int(round(sy * resolution)), 8)
        nx = max(int(round(sx * resolution)), 8)
        x = np.linspace(-sx / 2, sx / 2, nx)
        y = np.linspace(-sy / 2, sy / 2, ny)
    else:
        eps = np.asarray(eps, float)
        if grid is not None:
            x, y = np.asarray(grid[0], float), np.asarray(grid[1], float)
        else:
            ny_, nx_ = eps.shape
            x, y = np.arange(nx_, dtype=float), np.arange(ny_, dtype=float)
        sx = float(x[-1] - x[0])
        sy = float(y[-1] - y[0])
        lattice = mp.Lattice(size=mp.Vector3(sx, sy, 0))
        # Hand the grid to MPB as a raw float64 epsilon array used as the
        # *default material* over the whole lattice (Meep's epsilon-input /
        # MATERIAL_FILE path, bilinearly interpolated; dims[0] <-> x, dims[1]
        # <-> y, hence the transpose from photonix's (ny, nx)). A MaterialGrid
        # must NOT be used here: libpympb's get_material_pt()/material_epsmu()
        # switch has no MATERIAL_GRID case and calls meep::abort() on it --
        # MaterialGrid is an FDTD-only material in Meep.
        geometry = []
        default = np.ascontiguousarray(eps.T, dtype=np.float64)

    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=default,
        resolution=int(round(resolution)),
        num_bands=num_modes,
    )
    return ms, x, y


def solve_modes(
    *,
    wl: float = 1.55,
    width: float = 0.5,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    num_modes: int = 1,
    resolution: int = 40,
    margin: float = 1.5,
    eps: np.ndarray | None = None,
    grid: tuple | None = None,
    tol: float = 1e-6,
):
    """Solve cross-section modes with MPB; return a ``VectorModeData``.

    Mirrors the keyword surface of
    :func:`photonix.em.fde_vector.solve_modes_fullvector` so the two solvers are
    interchangeable at the call site. Effective indices are real (MPB is a
    lossless Hermitian eigensolver); ``te_fraction`` is computed from the MPB
    E-field when available, else ``None``.

    Requires Meep/MPB; raises :class:`ImportError` otherwise.
    """
    mp, mpb = require_mpb()
    from photonix.em.fde_vector import VectorModeData

    freq = meep_frequency(wl)
    ms, x, y = _build_solver(
        mpb, mp, wl=wl, width=width, thickness=thickness, n_core=n_core,
        n_clad=n_clad, margin=margin, eps=eps, grid=grid, resolution=resolution,
        num_modes=num_modes,
    )

    n_lo, n_hi = n_clad, n_core
    if eps is not None:
        idx = np.sqrt(np.asarray(eps, float))
        n_lo, n_hi = float(idx.min()), float(idx.max())
    kdir = mp.Vector3(0, 0, 1)
    kmag_guess = freq * 0.5 * (n_lo + n_hi)
    ks = ms.find_k(
        mp.NO_PARITY, freq, 1, num_modes, kdir, tol, kmag_guess,
        freq * n_lo * 0.5, freq * n_hi * 1.05,
    )
    n_eff_arr = np.array([n_eff_from_k(k, freq) for k in np.atleast_1d(ks)], dtype=float)

    fields, te_fraction = _extract_fields(ms, num_modes, x, y)
    return VectorModeData(
        n_eff=n_eff_arr,
        fields=fields,
        x=x,
        y=y,
        wl=float(wl),
        polarization="fullvector",
        te_fraction=te_fraction,
    )


def _extract_fields(ms, num_modes, x, y):
    """Best-effort E-field + te_fraction from a solved ModeSolver.

    MPB field layout varies across builds, so this degrades gracefully: on any
    failure it returns ``(empty_array, None)`` and the caller still gets n_eff.
    """
    try:
        fields = []
        te_frac = []
        for band in range(1, num_modes + 1):
            ef = np.asarray(ms.get_efield(band))  # (nx, ny, 1, 3) typically
            ef = np.squeeze(ef)
            ex, ey = ef[..., 0], ef[..., 1]
            px = float(np.sum(np.abs(ex) ** 2))
            py = float(np.sum(np.abs(ey) ** 2))
            te_frac.append(px / (px + py) if (px + py) > 0 else np.nan)
            fields.append(ef)
        return np.array(fields), np.array(te_frac)
    except Exception:  # noqa: BLE001 - fields are a best-effort extra
        return np.empty((0,)), None


def n_eff(**kwargs) -> float:
    """Fundamental-mode effective index via MPB (scalar convenience wrapper)."""
    kwargs.setdefault("num_modes", 1)
    return float(np.real(solve_modes(**kwargs).n_eff[0]))
