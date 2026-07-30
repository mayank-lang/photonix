"""Native Photonix layout to Meep geometry and port-plane conversion.

The adapter operates on :class:`photonix.layout.Cell`'s flattened polygons, so
hierarchical references, rotations, and mirrors are resolved exactly once by the
layout kernel.  Each GDS layer is mapped to a :class:`LayerSpec` and becomes a
Meep :class:`~meep.Prism`; no rasterisation is required.  A ``thickness=None``
spec creates an infinite-z prism for a 2-D simulation, while finite thicknesses
produce a 3-D stack.

The pure :func:`prepare_layout` stage deliberately has no Meep dependency.  This
makes geometry, units, origins, and port normals inspectable before an expensive
run and keeps the package testable on machines where Meep is unavailable.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._guard import require_meep

__all__ = [
    "LayerSpec",
    "PreparedPolygon",
    "PreparedPort",
    "PreparedLayout",
    "MeepLayout",
    "MeepPortRegion",
    "prepare_layout",
    "build_layout_geometry",
    "build_layout_simulation",
    "port_region",
    "port_regions",
]


@dataclass(frozen=True)
class LayerSpec:
    """Map one GDS ``layer`` to a Meep material and vertical extrusion.

    Pass exactly one of ``epsilon`` (a scalar, non-dispersive relative
    permittivity) or ``material`` (for example a dispersive ``meep.Medium``).
    ``thickness=None`` means infinite z extent and selects a 2-D simulation.
    """

    layer: tuple[int, int]
    epsilon: float | None = None
    material: Any | None = None
    thickness: float | None = None
    z_center: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", (int(self.layer[0]), int(self.layer[1])))
        if (self.epsilon is None) == (self.material is None):
            raise ValueError("LayerSpec requires exactly one of epsilon or material")
        if self.epsilon is not None and (not np.isfinite(self.epsilon) or self.epsilon <= 0):
            raise ValueError("LayerSpec epsilon must be positive and finite")
        if self.thickness is not None and (not np.isfinite(self.thickness) or self.thickness <= 0):
            raise ValueError("LayerSpec thickness must be positive and finite")
        if not np.isfinite(self.z_center):
            raise ValueError("LayerSpec z_center must be finite")


@dataclass(frozen=True)
class PreparedPolygon:
    """One centred layout polygon paired with its process layer."""

    vertices: np.ndarray
    layer: tuple[int, int]


@dataclass(frozen=True)
class PreparedPort:
    """A centred optical port and its outward normal/tangent vectors."""

    name: str
    center: tuple[float, float]
    orientation: float
    width: float
    layer: tuple[int, int]
    z_center: float
    outward_normal: tuple[float, float]
    tangent: tuple[float, float]


@dataclass(frozen=True)
class PreparedLayout:
    """Meep-free, centred polygon/port representation of a Photonix cell."""

    name: str
    polygons: tuple[PreparedPolygon, ...]
    ports: Mapping[str, PreparedPort]
    layer_specs: Mapping[tuple[int, int], LayerSpec]
    origin: tuple[float, float]
    z_origin: float
    bbox: tuple[tuple[float, float], tuple[float, float]]
    cell_size: tuple[float, float, float]
    margin: tuple[float, float, float]
    dimensions: int


@dataclass(frozen=True)
class MeepLayout:
    """Constructed Meep cell/geometry plus the pure layout metadata."""

    cell_size: Any
    geometry: tuple[Any, ...]
    prepared: PreparedLayout


@dataclass(frozen=True)
class MeepPortRegion:
    """A Meep mode plane with outward/inward wave-vector directions."""

    name: str
    region: Any
    outward_kpoint: Any
    inward_kpoint: Any
    port: PreparedPort


def _triple(value, *, z_default: float = 0.0) -> tuple[float, float, float]:
    if np.isscalar(value):
        v = float(value)  # type: ignore[arg-type]
        out = (v, v, z_default)
    else:
        vals = tuple(float(v) for v in value)
        if len(vals) == 2:
            out = (vals[0], vals[1], z_default)
        elif len(vals) == 3:
            out = vals
        else:
            raise ValueError("expected a scalar, (x, y), or (x, y, z) value")
    if any(not np.isfinite(v) or v < 0 for v in out):
        raise ValueError("margins must be non-negative and finite")
    return out


def prepare_layout(
    cell,
    layer_specs: Mapping[tuple[int, int], LayerSpec] | list[LayerSpec] | tuple[LayerSpec, ...],
    *,
    margin: float | tuple[float, float] | tuple[float, float, float] = 1.0,
    cell_z: float | None = None,
    strict_layers: bool = True,
) -> PreparedLayout:
    """Flatten and centre a native layout for 2-D or 3-D Meep simulation.

    ``margin`` is clear space between the selected polygon bounding box and the
    simulation-cell edge.  It should be larger than the PML thickness plus the
    desired decay space.  All finite-thickness specs select 3-D; mixing finite
    and infinite-z layer specs is rejected because it is almost always a process
    stack error.
    """
    specs_iter = layer_specs.values() if isinstance(layer_specs, Mapping) else layer_specs
    specs = {spec.layer: spec for spec in specs_iter}
    if not specs:
        raise ValueError("at least one LayerSpec is required")
    finite = [spec.thickness is not None for spec in specs.values()]
    if any(finite) and not all(finite):
        raise ValueError("do not mix finite-thickness 3-D layers with infinite-z 2-D layers")
    dimensions = 3 if all(finite) else 2
    mx, my, mz = _triple(margin, z_default=1.0 if dimensions == 3 else 0.0)

    selected: list[tuple[np.ndarray, tuple[int, int]]] = []
    missing: set[tuple[int, int]] = set()
    for vertices, layer in cell.get_polygons():
        key = (int(layer[0]), int(layer[1]))
        if key not in specs:
            missing.add(key)
            continue
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError(f"layer {key} contains a polygon with fewer than three 2-D vertices")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"layer {key} contains non-finite polygon coordinates")
        selected.append((points, key))
    if strict_layers and missing:
        raise KeyError(f"layout layers have no Meep LayerSpec: {sorted(missing)!r}")
    if not selected:
        raise ValueError("the selected layer map contains no layout polygons")

    all_points = np.concatenate([points for points, _layer in selected], axis=0)
    lo = np.min(all_points, axis=0)
    hi = np.max(all_points, axis=0)
    origin_arr = 0.5 * (lo + hi)
    sx, sy = (hi - lo) + 2 * np.array([mx, my])
    if sx <= 0 or sy <= 0:
        raise ValueError("prepared simulation cell must have positive x and y extents")

    if dimensions == 2:
        sz = 0.0
        z_origin = 0.0
    else:
        def finite_thickness(spec: LayerSpec) -> float:
            if spec.thickness is None:  # guarded by the all(finite) check above
                raise RuntimeError("internal mixed-dimensional layer stack")
            return spec.thickness

        zlo = min(spec.z_center - 0.5 * finite_thickness(spec) for spec in specs.values())
        zhi = max(spec.z_center + 0.5 * finite_thickness(spec) for spec in specs.values())
        z_origin = 0.5 * (zlo + zhi)
        required_z = zhi - zlo + 2 * mz
        sz = required_z if cell_z is None else float(cell_z)
        if not np.isfinite(sz) or sz < required_z:
            raise ValueError(f"cell_z must be finite and at least {required_z:g} um")

    polygons = tuple(
        PreparedPolygon(np.asarray(points - origin_arr, dtype=float), layer)
        for points, layer in selected
    )
    ports: dict[str, PreparedPort] = {}
    for name, native in cell.get_ports().items():
        orientation = float(native.orientation) % 360.0
        theta = np.deg2rad(orientation)
        normal = (float(np.cos(theta)), float(np.sin(theta)))
        tangent = (-normal[1], normal[0])
        width = float(native.width)
        port_layer = (int(native.layer[0]), int(native.layer[1]))
        if strict_layers and port_layer not in specs:
            raise KeyError(f"port {name!r} layer {port_layer} has no Meep LayerSpec")
        port_spec = specs.get(port_layer)
        port_z = 0.0 if port_spec is None else float(port_spec.z_center) - z_origin
        center = np.asarray(native.center, dtype=float) - origin_arr
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError(f"port {name!r} has an invalid centre")
        if not np.isfinite(width) or width <= 0:
            raise ValueError(f"port {name!r} width must be positive and finite")
        ports[name] = PreparedPort(
            name=name,
            center=(float(center[0]), float(center[1])),
            orientation=orientation,
            width=width,
            layer=port_layer,
            z_center=port_z,
            outward_normal=normal,
            tangent=tangent,
        )

    return PreparedLayout(
        name=str(cell.name),
        polygons=polygons,
        ports=ports,
        layer_specs=specs,
        origin=(float(origin_arr[0]), float(origin_arr[1])),
        z_origin=float(z_origin),
        bbox=((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1]))),
        cell_size=(float(sx), float(sy), float(sz)),
        margin=(mx, my, mz),
        dimensions=dimensions,
    )


def build_layout_geometry(prepared: PreparedLayout) -> MeepLayout:
    """Convert a :class:`PreparedLayout` to exact Meep prism geometry."""
    mp = require_meep()
    geometry = []
    for polygon in prepared.polygons:
        spec = prepared.layer_specs[polygon.layer]
        material = spec.material if spec.material is not None else mp.Medium(epsilon=spec.epsilon)
        thickness = spec.thickness
        base_z = 0.0 if thickness is None else float(spec.z_center) - prepared.z_origin - 0.5 * thickness
        vertices = [mp.Vector3(float(x), float(y), base_z) for x, y in polygon.vertices]
        height = mp.inf if thickness is None else thickness
        geometry.append(
            mp.Prism(
                vertices,
                height=height,
                axis=mp.Vector3(0, 0, 1),
                material=material,
            )
        )
    sx, sy, sz = prepared.cell_size
    return MeepLayout(mp.Vector3(sx, sy, sz), tuple(geometry), prepared)


def port_region(
    port: PreparedPort,
    *,
    span: float | None = None,
    z_span: float = 0.0,
    inward_offset: float = 0.0,
) -> MeepPortRegion:
    """Create a Meep mode-monitor plane from native port metadata.

    ``inward_offset`` moves the plane into the component along the negative
    outward normal.  For an eigenmode source, use the returned
    ``inward_kpoint`` with ``direction=mp.NO_DIRECTION``; for decomposition use
    ``outward_kpoint`` so forward coefficient 0 consistently means outgoing.
    """
    plane_span = port.width if span is None else float(span)
    if not np.isfinite(plane_span) or plane_span <= 0:
        raise ValueError("span must be positive and finite")
    if not np.isfinite(z_span) or z_span < 0 or not np.isfinite(inward_offset) or inward_offset < 0:
        raise ValueError("z_span and inward_offset must be non-negative and finite")
    nx, ny = port.outward_normal
    tx, ty = port.tangent
    if abs(nx * ny) > 1e-10:
        raise ValueError(
            f"Meep ModeRegion planes must be axis-aligned; port {port.name!r} "
            f"has orientation {port.orientation:g} degrees"
        )
    mp = require_meep()
    cx = port.center[0] - inward_offset * nx
    cy = port.center[1] - inward_offset * ny
    region = mp.ModeRegion(
        center=mp.Vector3(cx, cy, port.z_center),
        size=mp.Vector3(abs(tx) * plane_span, abs(ty) * plane_span, z_span),
    )
    return MeepPortRegion(
        name=port.name,
        region=region,
        outward_kpoint=mp.Vector3(nx, ny, 0),
        inward_kpoint=mp.Vector3(-nx, -ny, 0),
        port=port,
    )


def port_regions(prepared: PreparedLayout, **kwargs) -> dict[str, MeepPortRegion]:
    """Build axis-aligned mode regions for every named port in a prepared layout."""
    return {name: port_region(port, **kwargs) for name, port in prepared.ports.items()}


def build_layout_simulation(
    prepared: PreparedLayout,
    *,
    resolution: float,
    sources=(),
    pml: float = 1.0,
    default_material=None,
    **simulation_kwargs,
):
    """Build a Meep ``Simulation`` from native polygons; return ``(sim, layout)``.

    The PML lies inside the margin reserved by :func:`prepare_layout`, so each
    in-plane margin must be at least ``pml``.  Sources and any extra Meep
    ``Simulation`` keywords are passed through unchanged.
    """
    mp = require_meep()
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")
    if not np.isfinite(pml) or pml < 0:
        raise ValueError("pml must be non-negative and finite")
    active_margins = prepared.margin[: prepared.dimensions]
    if pml > 0 and any(m < pml for m in active_margins):
        raise ValueError("each simulation margin must be at least the PML thickness")
    layout = build_layout_geometry(prepared)
    kwargs = dict(
        cell_size=layout.cell_size,
        geometry=list(layout.geometry),
        sources=list(sources),
        boundary_layers=[mp.PML(pml)] if pml > 0 else [],
        resolution=int(round(resolution)),
        dimensions=prepared.dimensions,
    )
    if default_material is not None:
        kwargs["default_material"] = default_material
    kwargs.update(simulation_kwargs)
    return mp.Simulation(**kwargs), layout
