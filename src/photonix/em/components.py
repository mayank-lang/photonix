"""EME-backed component models: rigorously simulated taper and 1x2 MMI.

These build a 2.5-D effective-index picture -- the vertical (thickness) dimension
is collapsed with the validated slab effective index (:mod:`photonix.em.slab`),
giving a 1-D in-plane permittivity profile that :mod:`photonix.em.eme` propagates.
The result is returned as a standard ``SDict`` so an EME-simulated taper or MMI
drops directly into :mod:`photonix.circuit` and :mod:`photonix.optim`.

These are slower than the analytic compact models (each section needs an
eigensolve), so they are opt-in: use them when you want rigorous, geometry-based
S-parameters rather than an idealized model.

Polarization
------------
``polarization`` labels the physical channel mode (quasi-TE / quasi-TM). Per the
effective-index method (see :mod:`photonix.em.eim`), the polarization *rotates*
between the two collapsed solves: quasi-TE uses the **vertical TE** slab index
and then the **lateral TM** (in-plane E, ``Hy``-continuous) EME propagation, and
quasi-TM the converse. The lateral EME polarization is therefore mapped
internally -- it is *not* the same label passed straight through.

Reflections
-----------
The EME cascade is bidirectional, so input- and output-side modal reflections
(``Rf``/``Rb``) come for free and are exported (``("o1","o1")``, ``("o2","o2")``,
...). Omitting them would make the circuit solver treat the ports as perfectly
matched, corrupting cascades and resonators built from these models.

Radiation
---------
Both models default to a transverse **absorber** (``absorber=(0.8, 1.0)``), which
is what makes the computed loss a property of the structure rather than of the
simulation window. With a closed window every non-guided basis mode is a lossless
box mode: radiated power reaches the far end and re-couples, so the answer tracks
``half_window``, ``points`` and ``num_modes`` instead of converging. Measured on
the 1x2 MMI before the absorber was added, excess loss ran 0.34 -> 1.12 dB across
``num_modes`` 6..16 and 0.66 -> 3.33 dB across ``points`` 401..801, with no
plateau anywhere (see ``docs/PHYSICS_AUDIT.md``, A2).

Pass ``absorber=None`` to recover the old closed-window behaviour -- appropriate
only for genuinely non-radiating structures, where it is cheaper.
"""
from __future__ import annotations

import numpy as np

from photonix.core.types import SDict

from .eme import Section, eme_smatrix, slab_modes
from .slab import slab_neff

__all__ = ["taper", "mmi1x2"]


def _index_at(value, wl: float, name: str) -> float:
    if hasattr(value, "index"):
        value = value.index(wl)
    elif callable(value):
        value = value(wl)
    arr = np.asarray(value)
    if arr.ndim != 0 or np.iscomplexobj(arr) or not np.isfinite(float(arr)) or float(arr) <= 0:
        raise ValueError(f"{name} must be a positive real index or wavelength-to-index material")
    return float(arr)


def _vertical_index(thickness, n_core, n_clad, wl, polarization):
    vpol = "te" if polarization == "te" else "tm"
    return slab_neff(thickness=thickness, n_core=n_core, n_clad=n_clad, wl=wl, polarization=vpol)


def _lateral_polarization(polarization: str) -> str:
    """EIM polarization rotation: quasi-TE -> lateral TM, quasi-TM -> lateral TE.

    The quasi-TE channel mode has in-plane E (``Ex``), which is *normal* to the
    lateral sidewalls, so the collapsed in-plane problem must use the
    ``Hy``-continuous (TM) formulation -- the same rule as
    :func:`photonix.em.eim.neff`.
    """
    if polarization not in ("te", "tm"):
        raise ValueError("polarization must be 'te' or 'tm'")
    return "tm" if polarization == "te" else "te"


def _strip(x, width, n_core_eff, n_clad, center=0.0):
    """Subpixel-averaged 1-D permittivity of a strip of ``width`` centred at ``center``.

    The sidewalls almost never land on a grid point, so a hard
    ``np.where(|x - c| < w/2, ...)`` staircases them onto the nearest cell. That
    makes the modal propagation constants jump discontinuously as the grid
    changes: measured on the 2.5 µm MMI body, the beat length
    ``L_pi = pi/(beta0 - beta1)`` wandered non-monotonically over 15.51-15.75 µm
    (±0.8 %) across ``points`` 301..1201. Over a ~30 µm MMI that moves the
    self-imaging point by ±0.4 µm, which is enough to slide a fixed-length device
    off its low-loss peak -- so the *excess loss* appeared not to converge even
    though the device physics had.

    Averaging the permittivity by the fraction of each cell inside the core is
    the same fix :mod:`photonix.em.geometry` already applies to the 2-D
    cross-sections, and it restores smooth dependence on the grid.
    """
    x = np.asarray(x, dtype=float)
    h = float(x[1] - x[0])
    lo, hi = center - width / 2.0, center + width / 2.0
    overlap = np.minimum(x + h / 2.0, hi) - np.maximum(x - h / 2.0, lo)
    frac = np.clip(overlap / h, 0.0, 1.0)
    return frac * n_core_eff**2 + (1.0 - frac) * n_clad**2


def _strip_union(x, strips, n_core_eff, n_clad):
    """Subpixel permittivity of the geometric union of 1-D strip intervals."""
    x = np.asarray(x, dtype=float)
    h = float(x[1] - x[0])
    fraction = np.zeros_like(x)
    for center, width in strips:
        lo, hi = center - width / 2.0, center + width / 2.0
        overlap = np.minimum(x + h / 2.0, hi) - np.maximum(x - h / 2.0, lo)
        fraction += np.clip(overlap / h, 0.0, 1.0)
    fraction = np.clip(fraction, 0.0, 1.0)
    return fraction * n_core_eff**2 + (1.0 - fraction) * n_clad**2


def _even_odd_indices(eps_out, dx, wl, num_modes, lat_pol, absorber):
    """Indices of the even and odd supermodes of a symmetric two-guide section.

    Parity is measured directly (``<psi | psi(-x)> / <psi | psi>``), so the
    identification survives the near-degeneracy of a weakly-coupled pair, where
    the eigensolver's ordering of the two is arbitrary. Among the modes of each
    parity the most-confined one (largest ``Re(beta)``, i.e. lowest index) is the
    supermode pair member.
    """
    betas, fields, _w = slab_modes(eps_out, dx, wl, num_modes, lat_pol, pml=absorber)
    order = np.argsort(-np.real(betas))
    i_even = i_odd = None
    for i in order:
        f = np.asarray(fields[:, i])
        denom = float(np.sum(np.abs(f) ** 2))
        if denom == 0:
            continue
        parity = float(np.real(np.sum(f * f[::-1]))) / denom
        if parity > 0.5 and i_even is None:
            i_even = int(i)
        elif parity < -0.5 and i_odd is None:
            i_odd = int(i)
        if i_even is not None and i_odd is not None:
            break
    if i_even is None or i_odd is None:
        raise RuntimeError(
            "Could not identify the even/odd supermode pair of the MMI output "
            "section; increase num_modes or check the geometry symmetry."
        )
    return i_even, i_odd


def taper(
    *,
    wl: float = 1.55,
    width1: float = 0.5,
    width2: float = 1.0,
    length: float = 20.0,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    polarization: str = "te",
    num_sections: int = 40,
    num_modes: int = 16,
    half_window: float = 3.0,
    points: int = 241,
    absorber: tuple | None = (0.8, 1.0),
) -> SDict:
    """Rigorous (EME) linear width taper. Ports ``o1`` (width1) -> ``o2`` (width2).

    ``absorber=(thickness_um, strength)`` puts a graded absorbing layer in the
    transverse cladding so radiated power leaves the simulation; ``None`` closes
    the window (see the module docstring).

    Examples
    --------
    >>> import photonix as px
    >>> s = px.em.components.taper(width1=0.5, width2=1.0, length=30.0, num_sections=20)
    >>> bool(0.0 < px.power(s[("o1", "o2")]) <= 1.0)
    True
    """
    n_core = _index_at(n_core, wl, "n_core")
    n_clad = _index_at(n_clad, wl, "n_clad")
    nve = _vertical_index(thickness, n_core, n_clad, wl, polarization)
    x = np.linspace(-half_window, half_window, points)
    dx = float(x[1] - x[0])
    widths = np.linspace(width1, width2, num_sections)
    secs = [Section(_strip(x, w, nve, n_clad), length / num_sections) for w in widths]
    r = eme_smatrix(secs, dx, wl, num_modes, _lateral_polarization(polarization),
                    pml=absorber)
    return {
        ("o1", "o2"): complex(r.Tf[0, 0]),
        ("o2", "o1"): complex(r.Tb[0, 0]),
        ("o1", "o1"): complex(r.Rf[0, 0]),   # input-side modal reflection
        ("o2", "o2"): complex(r.Rb[0, 0]),   # output-side modal reflection
    }


def mmi1x2(
    *,
    wl: float = 1.55,
    width_mmi: float = 2.5,
    length_mmi: float = 29.25,
    width_access: float = 0.5,
    gap: float = 1.0,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    polarization: str = "te",
    num_modes: int = 24,
    half_window: float = 4.0,
    points: int = 401,
    access_length: float = 1.0,
    absorber: tuple | None = (0.8, 1.0),
) -> SDict:
    """Rigorous (EME) 1x2 MMI splitter. Ports ``o1`` (in) -> ``o2``/``o3`` (out).

    The two outputs are formed from the even/odd supermodes of the output
    two-waveguide section. The split is balanced by *symmetry* -- a centred
    (even) input cannot excite an odd output supermode -- so ``|o1->o2|`` and
    ``|o1->o3|`` agreeing is a property of the grid, not evidence that the MMI
    self-imaging is right; the physics to check is the total transmission.

    ``length_mmi`` sets the self-imaging (low-loss) point; the default is the
    optimum found at converged settings (~1.15 dB excess loss). ``absorber`` puts
    a graded absorbing layer in the transverse cladding so radiated power leaves
    the simulation rather than re-coupling downstream.

    ``num_modes`` must be large enough to account for the scattered power: the
    excess loss climbs from 0.42 dB at 8 modes and only plateaus (~1.21 dB) above
    20, which is why the default is 24. Scale it up with ``half_window``, since a
    wider window puts more modes below any fixed cut.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.em.components.mmi1x2(length_mmi=30.0, num_modes=10, points=301)
    >>> t2, t3 = px.power(s[("o1", "o2")]), px.power(s[("o1", "o3")])
    >>> bool(abs(t2 - t3) < 1e-2)          # balanced to cascade round-off
    True
    >>> bool(0.0 < t2 + t3 <= 1.0)         # passive
    True
    """
    n_core = _index_at(n_core, wl, "n_core")
    n_clad = _index_at(n_clad, wl, "n_clad")
    nve = _vertical_index(thickness, n_core, n_clad, wl, polarization)
    x = np.linspace(-half_window, half_window, points)
    dx = float(x[1] - x[0])
    offset = (gap + width_access) / 2.0
    eps_in = _strip(x, width_access, nve, n_clad, center=0.0)
    eps_mmi = _strip(x, width_mmi, nve, n_clad, center=0.0)
    eps_out = _strip_union(
        x,
        ((+offset, width_access), (-offset, width_access)),
        nve,
        n_clad,
    )
    secs = [
        Section(eps_in, access_length),
        Section(eps_mmi, length_mmi),
        Section(eps_out, access_length),
    ]
    lat_pol = _lateral_polarization(polarization)
    r = eme_smatrix(secs, dx, wl, num_modes, lat_pol, pml=absorber)
    # Output two-waveguide supermodes. These must be identified by *parity*, not
    # by index: with a large modal basis the even and odd supermodes of a
    # weakly-coupled pair are nearly degenerate, so the eigensolver's ordering is
    # not stable and "mode 0 is even, mode 1 is odd" silently swaps.
    i_even, i_odd = _even_odd_indices(eps_out, dx, wl, num_modes, lat_pol, absorber)
    t_even = complex(r.Tf[i_even, 0])
    t_odd = complex(r.Tf[i_odd, 0])
    inv2 = 1.0 / np.sqrt(2.0)
    t_top = inv2 * (t_even + t_odd)
    t_bot = inv2 * (t_even - t_odd)
    # Reflections: input port directly (Rf00); output ports via the even/odd
    # supermode basis, o2 = (e + o)/sqrt(2), o3 = (e - o)/sqrt(2).
    Rb = np.asarray(r.Rb, complex)
    r11 = complex(r.Rf[0, 0])
    ee, eo, oe, oo = (complex(Rb[i_even, i_even]), complex(Rb[i_even, i_odd]),
                      complex(Rb[i_odd, i_even]), complex(Rb[i_odd, i_odd]))
    r22 = 0.5 * (ee + eo + oe + oo)
    r33 = 0.5 * (ee - eo - oe + oo)
    # (in, out) keys: value = amplitude at `out` due to unit input at `in`.
    r_2to3 = 0.5 * (ee + eo - oe - oo)
    r_3to2 = 0.5 * (ee - eo + oe - oo)
    return {
        ("o1", "o2"): t_top, ("o2", "o1"): t_top,
        ("o1", "o3"): t_bot, ("o3", "o1"): t_bot,
        ("o1", "o1"): r11,
        ("o2", "o2"): r22, ("o3", "o3"): r33,
        ("o2", "o3"): r_2to3, ("o3", "o2"): r_3to2,
    }
