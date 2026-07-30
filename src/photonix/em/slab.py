"""Rigorous 1-D slab waveguide solver for TE and TM polarizations.

This is the analytic-anchored accuracy core of the vectorial FDE work. It solves
the slab waveguide to <0.1% versus the closed-form transcendental for *both*
polarizations **when the mode is well-confined** (e.g. SOI 220 nm, the default
parameters), using the correct interface treatment for each:

* **TE** (E parallel to the interfaces, continuous): scalar Helmholtz with
  arithmetic subpixel-averaged permittivity.
* **TM** (E normal to the interfaces, discontinuous D continuous): the magnetic
  field ``Hx`` is continuous, so we solve the symmetric *generalized* eigenproblem
  ``A Hx = beta^2 B Hx`` with ``A`` a finite-volume ``d/dy[(1/eps) d/dy] + k0^2``
  using face-sharp permittivity, and ``B = diag(<1/eps>_cell)``. This finite-
  volume scheme converges cleanly at O(h^2) where a naive average gives only O(h).

Richardson extrapolation across two resolutions cancels the leading O(h^2) term,
reaching <0.1% on CPU-friendly grids. These validated 1-D operators are the
building blocks for the 2-D polarization-resolved FDE.

**Margin caveat (B1)**: for weakly-confined modes (low contrast or thin cores)
the evanescent tail may extend beyond the default ``margin``, making domain
truncation the dominant error source. Richardson cannot help because the
truncation error is nearly identical at both resolutions. Increase ``margin``
until it covers >= 3 decay lengths (see PHYSICS_AUDIT §B1).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["slab_neff", "slab_neff_analytic"]


def slab_neff_analytic(*, thickness=0.22, n_core=3.4757, n_clad=1.444, wl=1.55, polarization="te") -> float:
    """Closed-form fundamental effective index of a symmetric slab.

    TE: ``u tan(u) = w``. TM: ``u tan(u) = (n_core/n_clad)^2 w``.

    The transcendental is solved on the **fundamental branch** ``u in (0,
    min(V, pi/2))`` (``u``, ``w`` the usual normalized transverse parameters,
    ``V`` the normalized frequency). Root-finding over ``n_eff`` directly would
    bracket across the poles of ``tan`` and can return a higher-order mode --
    or a spurious pole crossing -- for multimode (thick) slabs; the ``u``-space
    bracket always isolates the fundamental, for any thickness.
    """
    from scipy.optimize import brentq

    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    for name, value in (
        ("thickness", thickness), ("n_core", n_core),
        ("n_clad", n_clad), ("wl", wl),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if n_core <= n_clad:
        raise ValueError("n_core must be greater than n_clad")

    k0 = 2 * np.pi / wl
    ratio = (n_core**2 / n_clad**2) if polarization == "tm" else 1.0
    V = (thickness / 2) * k0 * np.sqrt(n_core**2 - n_clad**2)

    def g(u):
        w = np.sqrt(max(V**2 - u**2, 0.0))
        return u * np.tan(u) - ratio * w

    # g(0+) = -ratio*V < 0; g -> +inf at u = pi/2 (if V >= pi/2) or
    # g(V) = V tan(V) > 0 (if V < pi/2): the fundamental root is bracketed.
    u_hi = min(V, np.pi / 2) - 1e-12
    u = brentq(g, 1e-12, u_hi, xtol=1e-15)
    return float(np.sqrt(n_core**2 - (2.0 * u / (thickness * k0)) ** 2))


def _te_1d(thickness, n_core, n_clad, k0, h, margin):
    N = int(np.ceil((thickness / 2 + margin) / h))
    y = np.arange(-N, N + 1) * h
    # arithmetic subpixel-averaged eps (tangential field)
    lo, hi = y - h / 2, y + h / 2
    ov = np.clip(np.minimum(hi, thickness / 2) - np.maximum(lo, -thickness / 2), 0.0, h)
    frac = ov / h
    eps = frac * n_core**2 + (1 - frac) * n_clad**2
    e = np.ones(len(eps))
    A = (sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1]) / h**2 + sp.diags(k0**2 * eps)).tocsr()
    val, _ = spla.eigsh(A, k=1, sigma=(n_core * k0) ** 2 * 1.0001, which="LM")
    return float(np.sqrt(val[0]) / k0)


def _tm_1d(thickness, n_core, n_clad, k0, h, margin):
    N = int(np.ceil((thickness / 2 + margin) / h))
    y = np.arange(-N, N + 1) * h
    n = len(y)
    # Effective 1/eps on each segment between adjacent nodes. The TM flux is
    # constant across a cut segment, so its resistance is integral(eps dy): the
    # face coefficient is 1 / arithmetic-average(eps), including the exact
    # core fraction when an interface falls between grid points.
    seg_lo, seg_hi = y[:-1], y[1:]
    seg_overlap = np.clip(
        np.minimum(seg_hi, thickness / 2) - np.maximum(seg_lo, -thickness / 2),
        0.0,
        h,
    )
    seg_fraction = seg_overlap / h
    eps_segment = seg_fraction * n_core**2 + (1.0 - seg_fraction) * n_clad**2
    af = eps_segment
    # node cell-averaged 1/eps (subpixel)
    lo, hi = y - h / 2, y + h / 2
    ov = np.clip(np.minimum(hi, thickness / 2) - np.maximum(lo, -thickness / 2), 0.0, h)
    frac = ov / h
    inv = frac * (1.0 / n_core**2) + (1 - frac) * (1.0 / n_clad**2)
    ap = np.zeros(n)
    ap[:-1] = (1.0 / af) / h**2
    ap[-1] = (1.0 / n_clad**2) / h**2
    am = np.zeros(n)
    am[1:] = (1.0 / af) / h**2
    am[0] = (1.0 / n_clad**2) / h**2
    main = -(ap + am) + k0**2
    A = sp.diags([ap[:-1], main, ap[:-1]], [-1, 0, 1]).tocsr()
    B = sp.diags(inv).tocsr()
    val, _ = spla.eigsh(A, k=1, M=B, sigma=(n_core * k0) ** 2 * 1.0001, which="LM")
    return float(np.sqrt(val[0]) / k0)


def slab_neff(
    *,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    wl: float = 1.55,
    resolution: int = 40,
    margin: float = 2.0,
    polarization: str = "te",
    richardson: bool = True,
) -> float:
    """Fundamental slab effective index for ``polarization`` in {"te", "tm"}.

    Matches :func:`slab_neff_analytic` to <0.1% (Richardson, ``resolution>=30``).

    Raises
    ------
    ValueError
        If ``n_core <= n_clad`` (no guided mode can exist) or if the returned
        effective index is outside the physical range ``n_clad < n_eff < n_core``
        (indicating the eigensolver returned a box/cladding mode instead of a
        guided mode — see PHYSICS_AUDIT §A3).

    Examples
    --------
    >>> import photonix.em as em
    >>> te = em.slab.slab_neff(polarization="te")
    >>> tm = em.slab.slab_neff(polarization="tm")
    >>> te > tm > 1.444   # TE is more confined than TM
    True
    """
    for name, value in (
        ("thickness", thickness), ("n_core", n_core), ("n_clad", n_clad),
        ("wl", wl), ("resolution", resolution), ("margin", margin),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if n_core <= n_clad:
        raise ValueError(
            f"n_core ({n_core}) must be greater than n_clad ({n_clad}) for a "
            f"guided mode to exist."
        )
    k0 = 2 * np.pi / wl
    solver = _te_1d if polarization == "te" else _tm_1d
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    # ``resolution`` is exactly points per micrometre. The subpixel interface
    # coefficients in both solvers permit a core boundary between grid points;
    # there is no need to retune h so an integer number of cells fits a half-core.
    h = 1.0 / float(resolution)
    half_core_cells = (thickness / 2) / h
    if half_core_cells < 3:
        import warnings
        warnings.warn(
            f"slab_neff: resolution={resolution} gives only {half_core_cells:.2f} cells in the "
            f"half-core (thickness={thickness} um). Consider resolution >= "
            f"{int(np.ceil(3 / (thickness / 2)))} for this thickness.",
            stacklevel=2,
        )
    if not richardson:
        result = solver(thickness, n_core, n_clad, k0, h, margin)
    else:
        n_c = solver(thickness, n_core, n_clad, k0, h, margin)
        n_f = solver(thickness, n_core, n_clad, k0, h / 2.0, margin)
        result = (4.0 * n_f - n_c) / 3.0
    if not (n_clad < result < n_core):
        raise ValueError(
            f"Computed n_eff ({result:.6f}) is outside the physical range "
            f"({n_clad} < n_eff < {n_core}). No guided mode exists for the "
            f"given parameters (thickness={thickness}, wl={wl}, "
            f"polarization={polarization!r})."
        )
    return result
