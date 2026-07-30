"""Cross-section mode solving via MPB, returned as native photonix types.

This is the MPB half of the Meep backend: it solves the guided modes of a 2-D
cross-section and returns a :class:`photonix.em.fde_vector.VectorModeData`, so an
MPB result is a drop-in cross-check for the in-house full-vector FDE solver
(``solve_modes_fullvector``) and the source of truth for Meep eigenmode injection.

It accepts either explicit rectangular-waveguide parameters (matching
:func:`photonix.em.geometry.rectangular_waveguide`) or an arbitrary ``eps`` grid.
The grid is handed to MPB through a supported position-dependent material on a
block spanning the lattice, preserving Photonix's cell-centred pixel semantics.
It is **not** passed as a
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


def _centres_and_extent(axis, n: int, resolution: float, name: str):
    """Return pixel centres and their edge-to-edge physical extent."""
    if axis is None:
        spacing = 1.0 / resolution
        centres = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * spacing
        return centres, n * spacing
    centres = np.asarray(axis, dtype=float)
    if centres.ndim != 1 or centres.size != n or n < 2:
        raise ValueError(f"{name} must be a one-dimensional array with {n} pixel centres")
    delta = np.diff(centres)
    if not np.all(np.isfinite(centres)) or np.any(delta <= 0):
        raise ValueError(f"{name} must contain finite, strictly increasing pixel centres")
    spacing = float(np.mean(delta))
    if not np.allclose(delta, spacing, rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced for MPB epsilon-array input")
    return centres, n * spacing


def _build_solver(
    mpb, mp, *, wl, width, thickness, n_core, n_clad, margin, eps, grid, resolution,
    num_modes,
):
    """Construct an ``mpb.ModeSolver`` for the requested cross-section."""
    if (not isinstance(num_modes, (int, np.integer))
            or isinstance(num_modes, (bool, np.bool_)) or num_modes <= 0):
        raise ValueError("num_modes must be a positive integer")
    if (not np.isfinite(resolution) or resolution <= 0):
        raise ValueError("resolution must be positive and finite")
    if eps is None:
        if any(not np.isfinite(v) or v <= 0 for v in (width, thickness, n_core, n_clad, margin)):
            raise ValueError("width, thickness, indices, and margin must be positive and finite")
        if n_core <= n_clad:
            raise ValueError("n_core must be greater than n_clad for a guided dielectric waveguide")
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
        x = (np.arange(nx) - 0.5 * (nx - 1)) * (sx / nx)
        y = (np.arange(ny) - 0.5 * (ny - 1)) * (sy / ny)
    else:
        eps = np.asarray(eps, float)
        if eps.ndim != 2 or min(eps.shape) < 2:
            raise ValueError("eps must be a two-dimensional array with at least two pixels per axis")
        if not np.all(np.isfinite(eps)) or np.any(eps <= 0):
            raise ValueError("eps must contain finite, positive relative permittivities")
        if grid is not None and (not isinstance(grid, tuple) or len(grid) != 2):
            raise ValueError("grid must be an (x, y) tuple of pixel-centre arrays")
        ny_, nx_ = eps.shape
        gx = None if grid is None else grid[0]
        gy = None if grid is None else grid[1]
        x, sx = _centres_and_extent(gx, nx_, resolution, "x")
        y, sy = _centres_and_extent(gy, ny_, resolution, "y")
        lattice = mp.Lattice(size=mp.Vector3(sx, sy, 0))
        dx, dy = sx / nx_, sy / ny_

        def sampled_medium(point):
            ix = int(np.clip(np.floor((float(point.x) + 0.5 * sx) / dx), 0, nx_ - 1))
            iy = int(np.clip(np.floor((float(point.y) + 0.5 * sy) / dy), 0, ny_ - 1))
            return mp.Medium(epsilon=float(eps[iy, ix]))

        # A callable material on a geometry object is supported by the Meep/MPB
        # Python interface. MaterialGrid is FDTD-only in MPB, while NumPy-array
        # ``default_material`` is not part of MPB's documented Python contract.
        geometry = [
            mp.Block(
                size=mp.Vector3(sx, sy, _INF),
                center=mp.Vector3(),
                material=sampled_medium,
            )
        ]
        default = mp.Medium(epsilon=float(np.min(eps)))

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
    with_fields: bool = True,
):
    """Solve cross-section modes with MPB; return a ``VectorModeData``.

    Mirrors the keyword surface of
    :func:`photonix.em.fde_vector.solve_modes_fullvector` so the two solvers are
    interchangeable at the call site. Effective indices are real (MPB is a
    lossless Hermitian eigensolver); ``te_fraction`` is computed from the MPB
    E-field when available, else ``None``.

    Requires Meep/MPB; raises :class:`ImportError` otherwise.
    """
    if not np.isfinite(wl) or wl <= 0:
        raise ValueError("wl must be positive and finite")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
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

    if with_fields:
        fields, te_fraction = _extract_fields(ms, num_modes, x, y)
        ny_field, nx_field = fields.shape[1:]
        sx = len(x) * float(x[1] - x[0])
        sy = len(y) * float(y[1] - y[0])
        x_out = (np.arange(nx_field) - 0.5 * (nx_field - 1)) * (sx / nx_field)
        y_out = (np.arange(ny_field) - 0.5 * (ny_field - 1)) * (sy / ny_field)
    else:
        fields, te_fraction = np.empty((0,)), None
        x_out, y_out = x, y
    if eps is None:
        exterior_index = float(n_clad)
    else:
        eps_arr = np.asarray(eps, dtype=float)
        boundary = np.concatenate(
            (eps_arr[0], eps_arr[-1], eps_arr[1:-1, 0], eps_arr[1:-1, -1])
        )
        exterior_index = float(np.sqrt(np.max(boundary)))
    return VectorModeData(
        n_eff=n_eff_arr,
        fields=fields,
        x=x_out,
        y=y_out,
        wl=float(wl),
        polarization="full",
        te_fraction=te_fraction,
        guided=np.real(n_eff_arr) > exterior_index,
    )


def _extract_fields(ms, num_modes, x, y):
    """Extract dominant E fields and TE fractions in native ``(mode,y,x)`` order."""
    fields = []
    te_frac = []
    for band in range(1, num_modes + 1):
        ef = np.asarray(ms.get_efield(band))  # (nx, ny, 1, 3) typically
        ef = np.squeeze(ef)
        if ef.ndim != 3 or ef.shape[-1] < 2:
            raise RuntimeError(
                f"MPB returned unexpected E-field shape {ef.shape}; "
                "pass with_fields=False to request effective indices only"
            )
        ex, ey = ef[..., 0], ef[..., 1]
        px = float(np.sum(np.abs(ex) ** 2))
        py = float(np.sum(np.abs(ey) ** 2))
        te_frac.append(px / (px + py) if (px + py) > 0 else np.nan)
        dominant = ex if px >= py else ey
        peak = dominant.reshape(-1)[np.argmax(np.abs(dominant))]
        if abs(peak) > 0:
            dominant = dominant * np.exp(-1j * np.angle(peak))
        # MPB stores [x, y, component]; photonix fields are [y, x].
        fields.append(np.real(dominant).T)
    array = np.asarray(fields)
    if array.ndim != 3 or array.shape[0] != num_modes:
        raise RuntimeError("MPB returned inconsistent field grids across bands")
    return array, np.asarray(te_frac)


def n_eff(**kwargs) -> float:
    """Fundamental-mode effective index via MPB (scalar convenience wrapper)."""
    kwargs.setdefault("num_modes", 1)
    return float(np.real(solve_modes(**kwargs).n_eff[0]))
