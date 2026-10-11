"""Ready-made cross-sections: strip, rib, coupler pair, heater over waveguide.

Conventions: the bottom of the waveguide core is at ``y = 0`` and the buried
oxide (if a ``box`` material is given) fills ``y < 0``; the upper cladding is
the :class:`XSection` background.  Widths are measured at the trapezoid's
``reference`` plane (default ``"top"``, the mask width of a top-down etch),
and ``sidewall_deg < 90`` widens the core towards its bottom.  Region tags are
``"core"`` (``"core:1"``/``"core:2"`` for a coupler), ``"slab"``, ``"box"``,
``"substrate"``, ``"heater"``; ``properties`` maps a tag to per-region values
(``{"heater": {"k_th": 20.0}}``) for :meth:`XSection.coefficient`, with
``"clad"`` addressing the background.

Defaults are the photonix SOI convention (``n_Si = 3.4757``,
``n_SiO2 = 1.444`` at 1.55 µm, as in :func:`photonix.em.rectangular_waveguide`);
pass dispersive materials (``photonix.em.silicon``...) for wavelength sweeps.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from .raster import XSection
from .shapes import BAND_HALF_WIDTH, MaterialSpec, Region, _half_widths, band, rect, rib, trapezoid
from .shapes import _concrete as _c

__all__ = ["strip", "rib_waveguide", "coupler", "heater_over_waveguide"]

N_SI = 3.4757
N_SIO2 = 1.444

Reference = Literal["top", "middle", "bottom"]


def _props(properties: Mapping[str, Mapping[str, Any]] | None, tag: str) -> dict[str, Any]:
    return dict((properties or {}).get(tag, {}))


def _box_region(box: MaterialSpec | None, properties) -> list[Region]:
    if box is None:
        return []
    return [band(-BAND_HALF_WIDTH, 0.0, box, tags={"box"}, name="box", properties=_props(properties, "box"))]


def _xsection(regions: list[Region], clad: MaterialSpec, properties, name: str) -> XSection:
    return XSection(regions, background=clad, background_properties=_props(properties, "clad"), name=name)


def strip(
    width: float = 0.5,
    thickness: float = 0.22,
    *,
    sidewall_deg: float = 90.0,
    reference: Reference = "top",
    core: MaterialSpec = N_SI,
    clad: MaterialSpec = N_SIO2,
    box: MaterialSpec | None = None,
    properties: Mapping[str, Mapping[str, Any]] | None = None,
) -> XSection:
    """Strip (channel) waveguide, optionally trapezoidal and on a distinct BOX.

    With the defaults it is the geometry of
    :func:`photonix.em.rectangular_waveguide` (same permittivity on the same
    grid, up to rounding).

    Examples
    --------
    >>> xs = strip(0.5, 0.22, sidewall_deg=80.0)
    >>> [sorted(r.tags) for r in xs.regions]
    [['core']]
    """
    regions = _box_region(box, properties)
    regions.append(trapezoid(0.0, 0.0, thickness, width, material=core, sidewall_deg=sidewall_deg,
                             reference=reference, tags={"core"}, order=1, name="core",
                             properties=_props(properties, "core")))
    return _xsection(regions, clad, properties, "strip")


def rib_waveguide(
    width: float = 0.5,
    thickness: float = 0.22,
    slab_thickness: float = 0.09,
    slab_width: float | None = None,
    *,
    sidewall_deg: float = 90.0,
    reference: Reference = "top",
    core: MaterialSpec = N_SI,
    clad: MaterialSpec = N_SIO2,
    box: MaterialSpec | None = None,
    properties: Mapping[str, Mapping[str, Any]] | None = None,
) -> XSection:
    """Rib waveguide: a ridge of full ``thickness`` on a ``slab_thickness`` slab.

    ``slab_width=None`` makes the slab laterally unbounded.

    Examples
    --------
    >>> xs = rib_waveguide(0.5, 0.22, 0.09, slab_width=3.0)
    >>> [sorted(r.tags) for r in xs.regions]
    [['slab'], ['core']]
    """
    regions = _box_region(box, properties)
    slab, ridge = rib(0.0, width, 0.0, thickness, slab_thickness, slab_width, material=core,
                      sidewall_deg=sidewall_deg, reference=reference, order=1)
    regions.append(Region(slab.polygon, slab.material, order=slab.order, tags=slab.tags, name="slab",
                          properties=_props(properties, "slab"), band=slab.band))
    regions.append(Region(ridge.polygon, ridge.material, order=ridge.order, tags=ridge.tags, name="core",
                          properties=_props(properties, "core")))
    return _xsection(regions, clad, properties, "rib")


def coupler(
    width: float = 0.5,
    gap: float = 0.2,
    thickness: float = 0.22,
    *,
    width2: float | None = None,
    sidewall_deg: float = 90.0,
    reference: Reference = "top",
    core: MaterialSpec = N_SI,
    clad: MaterialSpec = N_SIO2,
    box: MaterialSpec | None = None,
    properties: Mapping[str, Mapping[str, Any]] | None = None,
) -> XSection:
    """Two parallel strips separated by ``gap`` (directional-coupler section).

    The gap is the edge-to-edge spacing at the ``reference`` plane and is
    centred on ``x = 0``; ``width2`` (default ``width``) sets the right core.
    Cores are tagged ``"core"`` plus ``"core:1"`` (left) / ``"core:2"``.

    Examples
    --------
    >>> xs = coupler(0.5, 0.2, 0.22)
    >>> xs.bbox()[:2]
    (-0.6, 0.6)
    """
    w2 = width if width2 is None else width2
    if not _c(gap) > 0:
        raise ValueError("gap must be positive")
    bot1, top1 = _half_widths(width, thickness, sidewall_deg, reference)
    bot2, top2 = _half_widths(w2, thickness, sidewall_deg, reference)
    ref1 = {"top": top1, "bottom": bot1, "middle": 0.5 * (top1 + bot1)}[reference]
    ref2 = {"top": top2, "bottom": bot2, "middle": 0.5 * (top2 + bot2)}[reference]
    xc1 = -(0.5 * gap + ref1)
    xc2 = 0.5 * gap + ref2
    if not _c(xc2 - bot2) > _c(xc1 + bot1) or not _c(xc2 - top2) > _c(xc1 + top1):
        raise ValueError("the sidewall angle closes the gap: the cores would touch")
    regions = _box_region(box, properties)
    for k, (xc, w) in enumerate(((xc1, width), (xc2, w2)), start=1):
        regions.append(trapezoid(xc, 0.0, thickness, w, material=core, sidewall_deg=sidewall_deg,
                                 reference=reference, tags={"core", f"core:{k}"}, order=1, name=f"core{k}",
                                 properties={**_props(properties, "core"), **_props(properties, f"core:{k}")}))
    return _xsection(regions, clad, properties, "coupler")


def heater_over_waveguide(
    wg_width: float = 0.5,
    wg_thickness: float = 0.22,
    heater_width: float = 2.0,
    heater_thickness: float = 0.12,
    heater_offset: float = 0.0,
    spacing: float = 1.0,
    box_thickness: float = 2.0,
    *,
    metal: MaterialSpec,
    core: MaterialSpec = N_SI,
    clad: MaterialSpec = N_SIO2,
    box: MaterialSpec | None = None,
    substrate: MaterialSpec = N_SI,
    top_clad_thickness: float | None = None,
    top: MaterialSpec = 1.0,
    sidewall_deg: float = 90.0,
    properties: Mapping[str, Mapping[str, Any]] | None = None,
) -> XSection:
    """Metal heater above a strip waveguide: substrate / BOX / core / cladding / heater.

    The heater's bottom face is ``spacing`` above the top of the core and its
    centre is offset laterally by ``heater_offset``.  ``box`` defaults to the
    cladding material.  With ``top_clad_thickness`` the cladding ends that far
    above the heater top and ``top`` (default air) fills the rest.  ``metal``
    is required: photonix ships no heater-metal data.

    Examples
    --------
    >>> xs = heater_over_waveguide(heater_offset=1.0, metal=2.0 - 4.0j)
    >>> h = xs.tagged("heater")[0]
    >>> [round(v, 6) for v in h.bbox]
    [0.0, 2.0, 1.22, 1.34]
    """
    for name, v in (("wg_width", wg_width), ("wg_thickness", wg_thickness), ("heater_width", heater_width),
                    ("heater_thickness", heater_thickness), ("box_thickness", box_thickness)):
        if not _c(v) > 0:
            raise ValueError(f"{name} must be positive")
    if not _c(spacing) > 0:
        raise ValueError("spacing must be positive (the heater may not touch the core)")
    box_mat = clad if box is None else box
    regions: list[Region] = [
        band(-BAND_HALF_WIDTH, -box_thickness, substrate, tags={"substrate"}, name="substrate",
             properties=_props(properties, "substrate")),
        band(-box_thickness, 0.0, box_mat, tags={"box"}, name="box", properties=_props(properties, "box")),
    ]
    h0 = wg_thickness + spacing
    if top_clad_thickness is not None:
        if not _c(top_clad_thickness) >= 0:
            raise ValueError("top_clad_thickness must be non-negative")
        regions.append(band(h0 + heater_thickness + top_clad_thickness, BAND_HALF_WIDTH, top, tags={"top"},
                            name="top", properties=_props(properties, "top")))
    regions.append(trapezoid(0.0, 0.0, wg_thickness, wg_width, material=core, sidewall_deg=sidewall_deg,
                             tags={"core"}, order=1, name="core", properties=_props(properties, "core")))
    regions.append(rect(heater_offset - 0.5 * heater_width, heater_offset + 0.5 * heater_width, h0,
                        h0 + heater_thickness, metal, tags={"heater"}, order=2, name="heater",
                        properties=_props(properties, "heater")))
    return _xsection(regions, clad, properties, "heater_over_waveguide")
