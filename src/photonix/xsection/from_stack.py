"""Adapter from a :class:`photonix.pdk.LayerStack` to a polygonal :class:`XSection`.

Every unpatterned level becomes a laterally unbounded band over its
``[z_min, z_max]`` and every drawn feature a trapezoid on its level, all
painted in stack order (later levels over earlier ones) -- the rule of
:meth:`photonix.pdk.LayerStack.cross_section`.  Unlike that method, any number
of features may share a layer (coupler gaps, neighbouring waveguides), and
each may be off-centre and have its own sidewall angle.  The stack's ``z`` is
the cross-section ``y``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .raster import XSection
from .shapes import Region, band, trapezoid

__all__ = ["from_layer_stack"]


def _features(features: Any) -> list[tuple[tuple[int, int], float, float, float]]:
    out = []
    if isinstance(features, Mapping):
        items: Sequence[Any] = [(layer, 0.0, width) for layer, width in features.items()]
    else:
        items = list(features)
    for f in items:
        if len(f) not in (3, 4):
            raise ValueError("each feature must be (layer, x_center, width) or (layer, x_center, width, sidewall_deg)")
        layer = (int(f[0][0]), int(f[0][1]))
        side = float(f[3]) if len(f) == 4 else 90.0
        out.append((layer, f[1], f[2], side))
    return out


def from_layer_stack(
    stack: Any,
    features: Mapping[tuple[int, int], float] | Sequence[Sequence[Any]],
    *,
    background: Any = None,
    reference: Literal["top", "middle", "bottom"] = "top",
) -> XSection:
    """Polygonal cross-section of ``stack`` with the drawn ``features``.

    Parameters
    ----------
    stack
        A :class:`photonix.pdk.LayerStack` (anything with ``levels`` of
        ``name, layer, thickness, z_min, material``).
    features
        ``{layer: width}`` (centred, vertical sidewalls -- the
        ``LayerStack.cross_section`` argument) or a sequence of
        ``(layer, x_center, width[, sidewall_deg])``.
    background
        Material for anything no level covers; ``None`` (default) makes an
        uncovered window an error, as in ``LayerStack.cross_section``.
    reference
        Plane at which feature widths are measured for slanted sidewalls.

    Examples
    --------
    >>> import photonix as px
    >>> stack = px.pdk.demo_pdk().layer_stack
    >>> xs = from_layer_stack(stack, [((1, 0), -0.35, 0.5), ((1, 0), 0.35, 0.5)])   # two cores, 0.2 um gap
    >>> [r.name for r in xs.regions]
    ['oxide', 'core', 'core']
    """
    feats = _features(features)
    if not feats:
        raise ValueError("features must name at least one patterned layer")
    layers = {lev.layer for lev in stack.levels if lev.layer is not None}
    missing = {f[0] for f in feats} - layers
    if missing:
        raise KeyError(f"layers not in the stack: {sorted(missing)}")
    regions: list[Region] = []
    for i, lev in enumerate(stack.levels):
        tags = {lev.name}
        if lev.layer is None:
            regions.append(band(lev.z_min, lev.z_max, lev.material, order=i, name=lev.name, tags=tags))
            continue
        for layer, xc, width, side in feats:
            if layer == lev.layer:
                regions.append(trapezoid(xc, lev.z_min, lev.thickness, width, material=lev.material,
                                         sidewall_deg=side, reference=reference, order=i, name=lev.name,
                                         tags=tags | {f"layer:{layer[0]}/{layer[1]}"}))
    return XSection(regions, background=background, name="layer_stack")
