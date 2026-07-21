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
"""
from __future__ import annotations

import numpy as np

from photonix.core.types import SDict

from .eme import Section, eme_smatrix
from .slab import slab_neff

__all__ = ["taper", "mmi1x2"]


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
    return np.where(np.abs(x - center) < width / 2, n_core_eff**2, n_clad**2)


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
    num_modes: int = 6,
    half_window: float = 3.0,
    points: int = 241,
) -> SDict:
    """Rigorous (EME) linear width taper. Ports ``o1`` (width1) -> ``o2`` (width2).

    Examples
    --------
    >>> import photonix as px
    >>> s = px.em.components.taper(width1=0.5, width2=1.0, length=30.0, num_sections=20)
    >>> 0.0 < px.power(s[("o1", "o2")]) <= 1.0
    True
    """
    nve = _vertical_index(thickness, n_core, n_clad, wl, polarization)
    x = np.linspace(-half_window, half_window, points)
    dx = float(x[1] - x[0])
    widths = np.linspace(width1, width2, num_sections)
    secs = [Section(_strip(x, w, nve, n_clad), length / num_sections) for w in widths]
    r = eme_smatrix(secs, dx, wl, num_modes, _lateral_polarization(polarization))
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
    length_mmi: float = 29.5,
    width_access: float = 0.5,
    gap: float = 1.0,
    thickness: float = 0.22,
    n_core: float = 3.4757,
    n_clad: float = 1.444,
    polarization: str = "te",
    num_modes: int = 12,
    half_window: float = 4.0,
    points: int = 401,
    access_length: float = 1.0,
) -> SDict:
    """Rigorous (EME) 1x2 MMI splitter. Ports ``o1`` (in) -> ``o2``/``o3`` (out).

    The two outputs are formed from the even/odd supermodes of the output
    two-waveguide section. By symmetry the split is exactly balanced
    (``|o1->o2| == |o1->o3|``); the MMI length sets the (low-)loss point. The
    default ``length_mmi`` is tuned for the quasi-TE (lateral-TM) propagation
    physics at the default geometry.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.em.components.mmi1x2(length_mmi=30.0, num_modes=10, points=301)
    >>> abs(px.power(s[("o1", "o2")]) - px.power(s[("o1", "o3")])) < 1e-6
    True
    """
    nve = _vertical_index(thickness, n_core, n_clad, wl, polarization)
    x = np.linspace(-half_window, half_window, points)
    dx = float(x[1] - x[0])
    offset = (gap + width_access) / 2.0
    eps_in = _strip(x, width_access, nve, n_clad, center=0.0)
    eps_mmi = _strip(x, width_mmi, nve, n_clad, center=0.0)
    eps_out = np.maximum(
        _strip(x, width_access, nve, n_clad, center=+offset),
        _strip(x, width_access, nve, n_clad, center=-offset),
    )
    secs = [
        Section(eps_in, access_length),
        Section(eps_mmi, length_mmi),
        Section(eps_out, access_length),
    ]
    r = eme_smatrix(secs, dx, wl, num_modes, _lateral_polarization(polarization))
    # output two-waveguide supermodes: mode0 = even, mode1 = odd
    t_even = complex(r.Tf[0, 0])
    t_odd = complex(r.Tf[1, 0])
    inv2 = 1.0 / np.sqrt(2.0)
    t_top = inv2 * (t_even + t_odd)
    t_bot = inv2 * (t_even - t_odd)
    # Reflections: input port directly (Rf00); output ports via the even/odd
    # supermode basis, o2 = (e + o)/sqrt(2), o3 = (e - o)/sqrt(2).
    Rb = np.asarray(r.Rb, complex)
    r11 = complex(r.Rf[0, 0])
    r22 = 0.5 * complex(Rb[0, 0] + Rb[0, 1] + Rb[1, 0] + Rb[1, 1])
    r33 = 0.5 * complex(Rb[0, 0] - Rb[0, 1] - Rb[1, 0] + Rb[1, 1])
    # (in, out) keys: value = amplitude at `out` due to unit input at `in`.
    r_2to3 = 0.5 * complex(Rb[0, 0] + Rb[0, 1] - Rb[1, 0] - Rb[1, 1])
    r_3to2 = 0.5 * complex(Rb[0, 0] - Rb[0, 1] + Rb[1, 0] - Rb[1, 1])
    return {
        ("o1", "o2"): t_top, ("o2", "o1"): t_top,
        ("o1", "o3"): t_bot, ("o3", "o1"): t_bot,
        ("o1", "o1"): r11,
        ("o2", "o2"): r22, ("o3", "o3"): r33,
        ("o2", "o3"): r_2to3, ("o3", "o2"): r_3to2,
    }
