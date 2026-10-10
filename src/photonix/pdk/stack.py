"""Process description: layer stacks and waveguide cross-sections.

A :class:`LayerStack` records the vertical process -- each level's GDS layer,
thickness, base height and (dispersive) material -- so one description can drive
the mode solvers (:meth:`LayerStack.cross_section`), the Meep backend
(:meth:`LayerStack.to_meep_layer_specs`) and documentation.

A :class:`CrossSection` names a routable waveguide type: its layer, width,
default bend radius and the compact-model parameters (``neff``, ``ng``,
``loss_db_cm``, ``wl0``) that every waveguide-like model of a PDK should use.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["LayerLevel", "LayerStack", "CrossSection"]


def _index(material: Any, wl: float) -> float:
    if callable(material):
        return float(np.asarray(material(wl)))
    return float(material)


@dataclass(frozen=True)
class LayerLevel:
    """One level of the vertical stack.

    Attributes
    ----------
    name
        Level name (``"core"``, ``"box"``...).
    layer
        GDS layer patterned at this level, or ``None`` for an unpatterned
        background level (substrate, BOX, blanket cladding).
    thickness, z_min
        Vertical extent in µm.
    material
        Refractive index as a number or a callable ``n(wl)`` (e.g.
        :func:`photonix.em.silicon`).
    """

    name: str
    layer: tuple[int, int] | None
    thickness: float
    z_min: float
    material: float | Callable[[Any], Any]

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError(f"level {self.name!r}: thickness must be positive")
        if not np.isfinite(self.z_min):
            raise ValueError(f"level {self.name!r}: z_min must be finite")
        if self.layer is not None:
            object.__setattr__(self, "layer", (int(self.layer[0]), int(self.layer[1])))

    @property
    def z_max(self) -> float:
        return self.z_min + self.thickness


def _check_coverage(background, lo: float, hi: float) -> None:
    """Raise unless the unpatterned levels cover ``[lo, hi]`` without gaps."""
    spans = sorted((lev.z_min, lev.z_max) for lev in background)
    reach = lo
    for z0, z1 in spans:
        if z0 > reach + 1e-12:
            break
        reach = max(reach, z1)
    if reach < hi - 1e-12:
        gap_start = reach
        gap_end = min([z0 for z0, _ in spans if z0 > reach] + [hi])
        raise ValueError(
            f"no unpatterned level covers z = {gap_start:.4g}..{gap_end:.4g} um of the {lo:.4g}..{hi:.4g} um "
            "cross-section window; add an explicit background level (e.g. air or cladding) there"
        )


@dataclass
class LayerStack:
    """Ordered process levels; later levels are painted over earlier ones.

    Examples
    --------
    >>> import photonix as px
    >>> from photonix.pdk import LayerLevel, LayerStack
    >>> stack = LayerStack([
    ...     LayerLevel("oxide", None, 4.0, -2.0, px.em.silica),
    ...     LayerLevel("core", (1, 0), 0.22, 0.0, px.em.silicon),
    ... ])
    >>> cs = stack.cross_section({(1, 0): 0.5}, wl=1.55, resolution=20)
    >>> bool(cs.eps.max() > 11.0 and cs.eps.min() < 2.2)
    True
    """

    levels: list[LayerLevel] = field(default_factory=list)

    def level(self, name: str) -> LayerLevel:
        for lev in self.levels:
            if lev.name == name:
                return lev
        raise KeyError(f"no level named {name!r}")

    def by_layer(self, layer: tuple[int, int]) -> LayerLevel:
        for lev in self.levels:
            if lev.layer == tuple(layer):
                return lev
        raise KeyError(f"no level is patterned on layer {tuple(layer)}")

    def cross_section(self, widths: Mapping[tuple[int, int], float], *, wl: float = 1.55,
                      resolution: int = 40, margin: float = 1.5):
        """Subpixel-averaged permittivity of a waveguide cross-section.

        ``widths`` maps patterned layers to their (centred) drawn width; other
        patterned levels are absent from this cross-section. Unpatterned levels
        fill the whole window and, together, must cover its full height (a
        ``ValueError`` names any gap -- add an explicit air or cladding level).
        Where levels overlap the later one wins, and every pixel holds the exact
        area average of ``n^2`` over its sub-regions. The grid uses cell centres at ``1/resolution``
        spacing, centred on the patterned region (the same construction as
        :func:`photonix.em.rectangular_waveguide`), and the window extends
        ``margin`` beyond the widest feature and the patterned height.

        Returns
        -------
        photonix.em.CrossSection
            With ``eps`` of shape ``(ny, nx)`` and coordinates ``x``, ``y``.
        """
        from photonix.em.geometry import CrossSection as _Grid

        widths = {(int(k[0]), int(k[1])): float(v) for k, v in widths.items()}
        patterned = [lev for lev in self.levels if lev.layer is not None and lev.layer in widths]
        if not patterned:
            raise ValueError("widths must name at least one patterned layer of the stack")
        missing = set(widths) - {lev.layer for lev in patterned}
        if missing:
            raise KeyError(f"layers not in the stack: {sorted(missing)}")
        w_max = max(widths.values())
        z_lo = min(lev.z_min for lev in patterned)
        z_hi = max(lev.z_max for lev in patterned)
        h = 1.0 / float(resolution)
        nx = max(int(np.ceil((w_max + 2 * margin) * resolution)), 8)
        ny = max(int(np.ceil((z_hi - z_lo + 2 * margin) * resolution)), 8)
        x = (np.arange(nx) - 0.5 * (nx - 1)) * h
        y = (np.arange(ny) - 0.5 * (ny - 1)) * h + 0.5 * (z_lo + z_hi)
        background = [lev for lev in self.levels if lev.layer is None]
        if not background:
            raise ValueError("the stack needs at least one unpatterned background level")
        active = [lev for lev in self.levels if lev.layer is None or lev.layer in widths]
        y_lo, y_hi = float(y[0] - h / 2), float(y[-1] + h / 2)
        x_lo, x_hi = float(x[0] - h / 2), float(x[-1] + h / 2)
        _check_coverage(background, y_lo, y_hi)

        # Exact painter's-order area averaging of n^2: split the window at every
        # pixel edge and every level edge; each resulting sub-rectangle lies in
        # one pixel and one material (the last level covering its centre).
        def rect(lev):
            if lev.layer is None:
                return (x_lo - 1.0, x_hi + 1.0, lev.z_min, lev.z_max)
            half = 0.5 * widths[lev.layer]
            return (-half, half, lev.z_min, lev.z_max)

        rects = [rect(lev) for lev in active]
        xb = np.unique(np.clip(np.concatenate([x_lo + h * np.arange(nx + 1),
                                               [r[0] for r in rects], [r[1] for r in rects]]), x_lo, x_hi))
        yb = np.unique(np.clip(np.concatenate([y_lo + h * np.arange(ny + 1),
                                               [r[2] for r in rects], [r[3] for r in rects]]), y_lo, y_hi))
        xc, yc = 0.5 * (xb[1:] + xb[:-1]), 0.5 * (yb[1:] + yb[:-1])
        dxs, dys = np.diff(xb), np.diff(yb)
        sub = np.full((yc.size, xc.size), np.nan)
        for lev, (rx0, rx1, rz0, rz1) in zip(active, rects, strict=True):
            inside = ((yc >= rz0) & (yc <= rz1))[:, None] & ((xc >= rx0) & (xc <= rx1))[None, :]
            sub = np.where(inside, _index(lev.material, wl) ** 2, sub)
        if np.isnan(sub).any():  # pragma: no cover - _check_coverage guarantees full coverage
            raise RuntimeError("internal error: uncovered cross-section region")
        ix = np.minimum(((xc - x_lo) / h).astype(int), nx - 1)
        iy = np.minimum(((yc - y_lo) / h).astype(int), ny - 1)
        eps = np.zeros((ny, nx))
        np.add.at(eps, (iy[:, None], ix[None, :]), sub * dys[:, None] * dxs[None, :])
        eps /= h * h
        return _Grid(eps=eps, x=x, y=y)

    def solve_modes(self, widths: Mapping[tuple[int, int], float], *, wl: float = 1.55, num_modes: int = 1,
                    resolution: int = 40, margin: float = 1.5):
        """Full-vector modes of :meth:`cross_section` (see ``solve_modes_fullvector``).

        Examples
        --------
        >>> import photonix as px
        >>> pdk = px.pdk.demo_pdk()
        >>> m = pdk.layer_stack.solve_modes({(1, 0): 0.5}, resolution=30)
        >>> 2.3 < m.neff0 < 2.6
        True
        """
        from photonix.em.fde_vector import solve_modes_fullvector

        cs = self.cross_section(widths, wl=wl, resolution=resolution, margin=margin)
        return solve_modes_fullvector(wl=wl, eps=cs.eps, grid=(cs.x, cs.y), num_modes=num_modes)

    def to_meep_layer_specs(self, wl: float = 1.55) -> list:
        """Patterned levels as :class:`photonix.em.meep.LayerSpec` extrusions at ``wl``.

        Materials are frozen to their index at ``wl`` (non-dispersive). The
        unpatterned background levels are not included; set them as Meep's
        default material or blocks.
        """
        from photonix.em.meep.layout import LayerSpec

        return [
            LayerSpec(layer=lev.layer, epsilon=_index(lev.material, wl) ** 2,
                      thickness=lev.thickness, z_center=lev.z_min + 0.5 * lev.thickness)
            for lev in self.levels if lev.layer is not None
        ]


@dataclass(frozen=True)
class CrossSection:
    """A routable waveguide type and its compact-model parameters.

    ``model_settings`` (``neff``, ``ng``, ``loss_db_cm``, ``wl0``) are applied by
    :meth:`photonix.pdk.Pdk.models` to every model that declares them as
    waveguide parameters, so routed and generated waveguides simulate with the
    PDK's waveguide properties rather than library defaults.
    """

    name: str
    layer: tuple[int, int]
    width: float
    radius: float = 5.0
    model_settings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.radius <= 0:
            raise ValueError(f"cross-section {self.name!r}: width and radius must be positive")
        object.__setattr__(self, "layer", (int(self.layer[0]), int(self.layer[1])))
        object.__setattr__(self, "model_settings", dict(self.model_settings))
