"""Polygonal regions of a 2-D waveguide cross-section.

A :class:`Region` is one simple polygon in the cross-section plane ``(x, y)``
(``x`` lateral, ``y`` vertical, both in µm; light propagates along ``z``) filled
with one material. Regions are painted in ``order`` (ties: later in the list
wins), exactly like :class:`photonix.pdk.LayerStack` levels, so a core may be
drawn over a laterally unbounded cladding :func:`band`.

Materials are duck-typed (see :func:`photonix.xsection.resolve_index`): a number
(refractive index, complex allowed), a callable ``n(wl)``, any object with an
``index(wl)`` method, or a 3-tuple of those for a diagonal anisotropic medium
whose principal axes are the cross-section axes ``(x, y, z)``.

The constructors here are pure geometry; nothing is rasterized until
:meth:`photonix.xsection.XSection.rasterize` is called.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Union

import numpy as np

__all__ = [
    "MaterialSpec",
    "Region",
    "polygon",
    "rect",
    "band",
    "trapezoid",
    "rib",
    "disk",
    "polygon_area",
    "BAND_HALF_WIDTH",
]

#: A material: index number, ``n(wl)`` callable, object with ``index(wl)``, or a
#: 3-tuple of those (diagonal anisotropy along x, y, z).
MaterialSpec = Union[float, complex, Callable[..., Any], Any]

#: Half-width (µm) of the polygon used for a laterally unbounded :func:`band`.
#: Rasterization windows must lie inside ``|x| < BAND_HALF_WIDTH``.
BAND_HALF_WIDTH = 1.0e4


def _concrete(values: Any) -> np.ndarray:
    """Host (NumPy) value of ``values``, also for JAX arrays under ``jax.grad``.

    Geometry topology (orientation, simplicity, sweep order) is decided on the
    concrete coordinates; only the area arithmetic is traced.  Under ``jax.jit``
    there is no concrete value and a clear error is raised.
    """
    try:
        return np.asarray(values, dtype=float)
    except Exception:  # noqa: BLE001 - jax TracerArrayConversionError
        pass
    try:
        import jax

        return np.asarray(jax.lax.stop_gradient(values), dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            "photonix.xsection needs concrete polygon coordinates to decide the "
            "rasterization topology; it supports jax.grad but not jax.jit tracing"
        ) from exc


def _host(values: Any) -> np.ndarray:
    """NumPy copy of ``values`` keeping its dtype (stop-gradient for JAX values)."""
    if _is_jax_array(values):
        import jax

        return np.asarray(jax.lax.stop_gradient(values))
    return np.asarray(values)


def _is_jax_array(values: Any) -> bool:
    try:
        import jax
    except Exception:  # noqa: BLE001
        return False
    return isinstance(values, jax.Array)


def polygon_area(points: Any) -> float:
    """Signed shoelace area of a closed polygon (positive when counter-clockwise).

    Examples
    --------
    >>> polygon_area([(0, 0), (2, 0), (2, 1), (0, 1)])
    2.0
    """
    p = _concrete(points)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _segments_intersect(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Pairwise closed-segment intersection test (vectorized, exact orientation)."""
    a, b = p[:, None, 0, :], p[:, None, 1, :]
    c, d = q[None, :, 0, :], q[None, :, 1, :]

    def orient(u, v, w):
        return np.sign((v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1])
                       - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0]))

    def on_segment(u, v, w):  # w on segment uv, given collinear
        return ((np.minimum(u[..., 0], v[..., 0]) <= w[..., 0]) & (w[..., 0] <= np.maximum(u[..., 0], v[..., 0]))
                & (np.minimum(u[..., 1], v[..., 1]) <= w[..., 1]) & (w[..., 1] <= np.maximum(u[..., 1], v[..., 1])))

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    general = (o1 != o2) & (o3 != o4)
    special = (((o1 == 0) & on_segment(a, b, c)) | ((o2 == 0) & on_segment(a, b, d))
               | ((o3 == 0) & on_segment(c, d, a)) | ((o4 == 0) & on_segment(c, d, b)))
    return general | special


def _check_simple(p: np.ndarray) -> None:
    """Raise if the closed polygon ``p`` (N, 2) self-intersects."""
    n = len(p)
    if n <= 3:
        return
    seg = np.stack([p, np.roll(p, -1, axis=0)], axis=1)  # (n, 2, 2)
    hit = _segments_intersect(seg, seg)
    i, j = np.triu_indices(n, k=2)
    adjacent = (j - i == 1) | ((i == 0) & (j == n - 1))
    bad = hit[i, j] & ~adjacent
    if np.any(bad):
        k = int(np.argmax(bad))
        raise ValueError(f"polygon is not simple: edges {int(i[k])} and {int(j[k])} intersect")


def _check_material(material: Any) -> None:
    if isinstance(material, (tuple, list)):
        if len(material) != 3:
            raise ValueError("an anisotropic material must be a 3-tuple (x, y, z) of material specs")
        for m in material:
            _check_material(m)
        return
    if isinstance(material, (bool, np.bool_, str, bytes)):
        raise TypeError("material must be an index, a callable n(wl), or an object with index(wl); "
                        "names are not looked up")
    if isinstance(material, (int, float, complex, np.number)):
        if not np.isfinite(complex(material)):
            raise ValueError("material index must be finite")
        return
    if callable(getattr(material, "index", None)) or callable(material):
        return
    raise TypeError(
        f"unsupported material {material!r}: use an index, a callable n(wl), or an object with index(wl)"
    )


@dataclass(frozen=True, eq=False)
class Region:
    """One filled simple polygon of a cross-section.

    Attributes
    ----------
    polygon
        ``(N, 2)`` vertices ``(x, y)`` in µm.  Stored counter-clockwise (a
        clockwise input is reversed); consecutive duplicates and a repeated
        closing vertex are dropped.  May hold JAX values, in which case
        rasterized quantities are differentiable with respect to them.
    material
        See :data:`MaterialSpec`.
    order
        Painter's priority: higher ``order`` paints over lower; equal orders
        paint in list order (later wins).
    tags
        Free-form labels (``"core"``, ``"heater"``, ``"contact:anode"``,
        ``"doping:p"``...) used by :meth:`XSection.mask` and friends.
    name
        Optional identifier, used in error messages and property look-ups.
    properties
        Optional per-region values (numbers or callables of temperature) for
        non-optical quantities such as ``"k_th"``; these take precedence over
        attributes of ``material`` (see :meth:`XSection.coefficient`).
    band
        ``True`` for a laterally unbounded layer (see :func:`band`); such
        regions do not contribute to :meth:`XSection.bbox`.

    Examples
    --------
    >>> r = Region([(0, 0), (0, 1), (1, 1), (1, 0)], 3.48)   # clockwise input
    >>> polygon_area(r.polygon)                             # stored CCW
    1.0
    """

    polygon: Any
    material: MaterialSpec
    order: int = 0
    tags: frozenset[str] = frozenset()
    name: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    band: bool = False

    def __post_init__(self) -> None:
        pts = self.polygon
        is_jax = _is_jax_array(pts)
        if not is_jax:
            if isinstance(pts, np.ndarray) or not any(_is_jax_array(v) for v in _flatten(pts)):
                pts = np.array(pts, dtype=float)
            else:
                import jax.numpy as jnp

                pts = jnp.stack([jnp.stack([jnp.asarray(c, dtype=float) for c in v]) for v in pts])
                is_jax = True
        conc = _concrete(pts)
        if conc.ndim != 2 or conc.shape[1] != 2:
            raise ValueError("polygon must be an (N, 2) array of (x, y) vertices")
        if not np.all(np.isfinite(conc)):
            raise ValueError("polygon vertices must be finite")
        keep = np.any(conc != np.roll(conc, -1, axis=0), axis=1)  # drop v_k == v_{k+1}
        if not np.all(keep):
            idx = np.flatnonzero(keep)
            conc = conc[idx]
            pts = pts[idx]
        if len(conc) < 3:
            raise ValueError("polygon needs at least 3 distinct vertices")
        area = polygon_area(conc)
        span = float(np.max(np.ptp(conc, axis=0)))
        if not abs(area) > 1e-14 * span * span:
            raise ValueError("polygon has zero area")
        _check_simple(conc)
        if area < 0:
            pts = pts[::-1]
        if not is_jax:
            pts = np.array(pts, dtype=float)
            pts.setflags(write=False)
        object.__setattr__(self, "polygon", pts)
        _check_material(self.material)
        if isinstance(self.material, list):
            object.__setattr__(self, "material", tuple(self.material))
        if isinstance(self.order, (bool, np.bool_)) or not isinstance(self.order, (int, np.integer)):
            raise TypeError("order must be an integer")
        object.__setattr__(self, "order", int(self.order))
        tags = self.tags
        if isinstance(tags, str):
            tags = (tags,)
        object.__setattr__(self, "tags", frozenset(str(t) for t in tags))
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))
        object.__setattr__(self, "band", bool(self.band))

    @property
    def area(self) -> float:
        """Polygon area in µm² (positive)."""
        return polygon_area(self.polygon)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` of the polygon."""
        p = _concrete(self.polygon)
        return (float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max()))

    def translated(self, dx: float = 0.0, dy: float = 0.0) -> Region:
        """Copy shifted by ``(dx, dy)`` µm (a band only moves vertically)."""
        shift = np.array([0.0 if self.band else dx, dy])
        return Region(self.polygon + shift, self.material, order=self.order, tags=self.tags,
                      name=self.name, properties=self.properties, band=self.band)

    def rotated(self, angle_deg: float, origin: tuple[float, float] = (0.0, 0.0)) -> Region:
        """Copy rotated counter-clockwise by ``angle_deg`` about ``origin`` (not for bands)."""
        if self.band:
            raise ValueError("a laterally unbounded band cannot be rotated")
        th = np.deg2rad(float(angle_deg))
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        o = np.asarray(origin, dtype=float)
        pts = (self.polygon - o) @ rot.T + o
        return Region(pts, self.material, order=self.order, tags=self.tags, name=self.name,
                      properties=self.properties)

    def has_tag(self, tag: str) -> bool:
        """``True`` if ``tag`` is one of the region's tags."""
        return tag in self.tags


def _flatten(points: Any) -> Iterable[Any]:
    try:
        for vertex in points:
            yield from vertex
    except TypeError:
        return


def _kw(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"order", "tags", "name", "properties"}
    bad = set(kwargs) - allowed
    if bad:
        raise TypeError(f"unexpected keyword argument(s): {sorted(bad)}")
    return dict(kwargs)


def polygon(points: Any, material: MaterialSpec, **kwargs: Any) -> Region:
    """Region from explicit vertices ``[(x, y), ...]`` (either orientation).

    Keyword arguments ``order``, ``tags``, ``name``, ``properties`` are passed
    to :class:`Region`.

    Examples
    --------
    >>> tri = polygon([(0, 0), (1, 0), (0, 1)], 2.0, tags={"core"})
    >>> tri.area
    0.5
    """
    return Region(points, material, **_kw(kwargs))


def rect(x0: float, x1: float, y0: float, y1: float, material: MaterialSpec, **kwargs: Any) -> Region:
    """Axis-aligned rectangle ``[x0, x1] x [y0, y1]``.

    Examples
    --------
    >>> rect(-0.25, 0.25, 0.0, 0.22, 3.4757).bbox
    (-0.25, 0.25, 0.0, 0.22)
    """
    if not (_concrete(x1) > _concrete(x0) and _concrete(y1) > _concrete(y0)):
        raise ValueError("rect needs x1 > x0 and y1 > y0")
    return Region(_stack_points([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), material, **_kw(kwargs))


def band(y0: float, y1: float, material: MaterialSpec, **kwargs: Any) -> Region:
    """Laterally unbounded layer ``y0 <= y <= y1`` (BOX, cladding, substrate...).

    Represented by a rectangle of half-width :data:`BAND_HALF_WIDTH`; it is
    excluded from :meth:`XSection.bbox` so it never inflates the window.

    Examples
    --------
    >>> b = band(-2.0, 0.0, 1.444, tags={"box"})
    >>> b.band, b.bbox[2:]
    (True, (-2.0, 0.0))
    """
    if not _concrete(y1) > _concrete(y0):
        raise ValueError("band needs y1 > y0")
    w = BAND_HALF_WIDTH
    return Region(_stack_points([(-w, y0), (w, y0), (w, y1), (-w, y1)]), material, band=True, **_kw(kwargs))


def _stack_points(points: list[tuple[Any, Any]]) -> Any:
    if any(_is_jax_array(c) for v in points for c in v):
        import jax.numpy as jnp

        return jnp.stack([jnp.stack([jnp.asarray(c, dtype=float) for c in v]) for v in points])
    return np.array(points, dtype=float)


def _half_widths(width: Any, height: Any, sidewall_deg: float, reference: str) -> tuple[Any, Any]:
    """Bottom and top half-widths of a trapezoid with the given sidewall angle."""
    if not 0.0 < float(sidewall_deg) < 180.0:
        raise ValueError("sidewall_deg must lie strictly between 0 and 180 degrees")
    run = 0.0 if float(sidewall_deg) == 90.0 else height / np.tan(np.deg2rad(float(sidewall_deg)))
    # ``run`` > 0 for sidewall < 90 deg: the bottom is wider than the top.
    if reference == "top":
        top = 0.5 * width
        bot = top + run
    elif reference == "bottom":
        bot = 0.5 * width
        top = bot - run
    elif reference == "middle":
        top = 0.5 * width - 0.5 * run
        bot = 0.5 * width + 0.5 * run
    else:
        raise ValueError("reference must be 'top', 'middle' or 'bottom'")
    if not (_concrete(top) > 0 and _concrete(bot) > 0):
        raise ValueError("sidewall angle too shallow for this width and height: a face would vanish")
    return bot, top


def trapezoid(
    x_center: float,
    y0: float,
    height: float,
    width: float,
    *,
    material: MaterialSpec,
    sidewall_deg: float = 90.0,
    reference: Literal["top", "middle", "bottom"] = "top",
    **kwargs: Any,
) -> Region:
    """Symmetric trapezoid (etched core) from ``y0`` to ``y0 + height``.

    ``sidewall_deg`` is the angle between the sidewall and the bottom face:
    ``90`` is vertical, ``< 90`` gives the usual etched profile that is wider at
    the bottom, ``> 90`` a re-entrant (undercut-like) profile.  ``width`` is
    measured at the ``reference`` plane (default the top face, i.e. the mask
    width of a top-down etch); ``"middle"`` keeps the area of the
    ``width x height`` rectangle.

    Examples
    --------
    >>> t = trapezoid(0.0, 0.0, 0.22, 0.5, material=3.4757, sidewall_deg=80.0)
    >>> round(t.bbox[1] - t.bbox[0], 4)          # bottom width = 0.5 + 2 * 0.22 / tan(80 deg)
    0.5776
    >>> trapezoid(0.0, 0.0, 0.22, 0.5, material=3.4757).area == rect(-0.25, 0.25, 0.0, 0.22, 3.4757).area
    True
    """
    if not (_concrete(height) > 0 and _concrete(width) > 0):
        raise ValueError("trapezoid needs positive width and height")
    bot, top = _half_widths(width, height, sidewall_deg, reference)
    y1 = y0 + height
    pts = [(x_center - bot, y0), (x_center + bot, y0), (x_center + top, y1), (x_center - top, y1)]
    return Region(_stack_points(pts), material, **_kw(kwargs))


def rib(
    x_center: float,
    width: float,
    y0: float,
    height: float,
    slab_height: float,
    slab_width: float | None = None,
    *,
    material: MaterialSpec,
    sidewall_deg: float = 90.0,
    reference: Literal["top", "middle", "bottom"] = "top",
    **kwargs: Any,
) -> tuple[Region, Region]:
    """Rib (partially etched) waveguide as ``(slab, ridge)`` regions.

    The ridge spans the full ``height`` from ``y0`` and is painted over a slab
    of thickness ``slab_height``; ``slab_width=None`` makes the slab a
    laterally unbounded :func:`band`.  Both regions get ``tags`` plus
    ``"slab"`` / ``"core"`` respectively.

    Examples
    --------
    >>> slab, ridge = rib(0.0, 0.5, 0.0, 0.22, 0.09, 3.0, material=3.4757)
    >>> sorted(slab.tags), sorted(ridge.tags), ridge.order > slab.order
    (['slab'], ['core'], True)
    """
    if not 0 < _concrete(slab_height) < _concrete(height):
        raise ValueError("rib needs 0 < slab_height < height")
    kw = _kw(kwargs)
    tags = frozenset(kw.pop("tags", ()) or ())
    order = int(kw.pop("order", 0))
    if slab_width is None:
        slab = band(y0, y0 + slab_height, material, tags=tags | {"slab"}, order=order, **kw)
    else:
        slab = rect(x_center - 0.5 * slab_width, x_center + 0.5 * slab_width, y0, y0 + slab_height,
                    material, tags=tags | {"slab"}, order=order, **kw)
    ridge = trapezoid(x_center, y0, height, width, material=material, sidewall_deg=sidewall_deg,
                      reference=reference, tags=tags | {"core"}, order=order + 1, **kw)
    return slab, ridge


def disk(
    x_center: float,
    y_center: float,
    radius: float,
    material: MaterialSpec,
    *,
    n_vertices: int = 128,
    preserve_area: bool = True,
    **kwargs: Any,
) -> Region:
    """Regular ``n_vertices``-gon approximating a circle (fibre cores, vias...).

    With ``preserve_area`` (default) the polygon circumradius is scaled so its
    area equals ``pi * radius**2`` exactly; the remaining shape error is
    ``O(radius / n_vertices**2)``.

    Examples
    --------
    >>> import math
    >>> abs(disk(0.0, 0.0, 1.0, 1.45, n_vertices=64).area - math.pi) < 1e-12
    True
    """
    if not isinstance(n_vertices, (int, np.integer)) or n_vertices < 8:
        raise ValueError("n_vertices must be an integer >= 8")
    if not _concrete(radius) > 0:
        raise ValueError("radius must be positive")
    th = 2.0 * np.pi * np.arange(n_vertices) / n_vertices
    scale = np.sqrt(2.0 * np.pi / (n_vertices * np.sin(2.0 * np.pi / n_vertices))) if preserve_area else 1.0
    r = radius * scale
    pts = [(x_center + r * np.cos(t), y_center + r * np.sin(t)) for t in th]
    return Region(_stack_points(pts), material, **_kw(kwargs))
