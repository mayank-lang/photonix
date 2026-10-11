"""Duck-typed material look-up for cross-section regions.

The cross-section code never imports a material database.  It accepts, for the
refractive index,

* a number (real, or complex with ``Im(n) < 0`` for absorption -- photonix
  uses ``exp(-j beta z)`` propagation, so loss means a negative imaginary part);
* a callable ``n(wl)`` such as :func:`photonix.em.silicon`;
* any object with an ``index(wl)`` method (for example
  :class:`photonix.em.Material`, or a future ``photonix.materials`` entry);
* a 3-tuple of the above for a diagonal anisotropic medium whose principal axes
  are the cross-section axes ``(x, y, z)``; an ``index`` returning a
  length-3 array is treated the same way.

A temperature is forwarded as ``temperature=T`` only when one is requested; a
callable or ``index`` method that cannot accept it raises instead of silently
returning the room-temperature value.  Plain numbers are temperature
independent by definition.

Non-optical quantities (``"k_th"``, ``"rho_cp"``, ``"sigma"``,
``"eps_static"``, or any other name) are resolved from the region's
``properties`` mapping first and then from an attribute of the same name on the
material; either may be a number or a callable of temperature.
"""
from __future__ import annotations

import inspect
from typing import Any

import numpy as np

__all__ = ["resolve_index", "resolve_property"]


def _accepts_temperature(fn: Any) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables: assume not
        return False
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD or p.name == "temperature":
            return True
    return False


def _call(fn: Any, arg: Any, temperature: float | None, what: str) -> Any:
    if temperature is None:
        return fn(arg)
    if not _accepts_temperature(fn):
        raise TypeError(
            f"{what} does not accept a temperature argument; pass temperature=None or use a "
            "temperature-dependent material (a callable or index method with a 'temperature' parameter)"
        )
    return fn(arg, temperature=temperature)


def _as_index_array(value: Any, what: str) -> Any:
    from .shapes import _is_jax_array

    if _is_jax_array(value):
        import jax.numpy as jnp

        arr = jnp.asarray(value)
    else:
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"{what} returned a non-numeric index")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{what} returned a non-finite index")
    if arr.shape not in ((), (3,)):
        raise ValueError(
            f"{what} returned an index of shape {arr.shape}; a scalar wavelength must give a scalar "
            "or a length-3 (x, y, z) principal-index array"
        )
    return arr


def resolve_index(material: Any, wl: float, temperature: float | None = None) -> Any:
    """Refractive index of ``material`` at scalar ``wl`` (µm): shape ``()`` or ``(3,)``.

    Examples
    --------
    >>> import photonix as px
    >>> float(resolve_index(1.444, 1.55))
    1.444
    >>> round(float(resolve_index(px.em.silicon, 1.55)), 4)
    3.4777
    >>> resolve_index((2.21, 2.14, 2.14), 1.55).shape            # uniaxial, x = extraordinary
    (3,)
    """
    if isinstance(material, (tuple, list)):
        if len(material) != 3:
            raise ValueError("an anisotropic material must be a 3-tuple (x, y, z)")
        parts = [resolve_index(m, wl, temperature) for m in material]
        if any(np.shape(p) != () for p in parts):
            raise ValueError("components of an anisotropic material must be scalar indices")
        from .shapes import _is_jax_array

        if any(_is_jax_array(p) for p in parts):
            import jax.numpy as jnp

            return jnp.stack([jnp.asarray(p) for p in parts])
        return np.array([complex(p) if np.iscomplexobj(p) else float(p) for p in parts])
    if isinstance(material, (int, float, complex, np.number)) and not isinstance(material, (bool, np.bool_)):
        return np.asarray(material)
    index = getattr(material, "index", None)
    if callable(index):
        return _as_index_array(_call(index, wl, temperature, f"{material!r}.index"), repr(material))
    if callable(material):
        return _as_index_array(_call(material, wl, temperature, repr(material)), repr(material))
    raise TypeError(f"unsupported material {material!r}")


def resolve_property(region: Any, quantity: str, temperature: float | None = None, *,
                     material: Any = None) -> Any:
    """Value of a non-optical ``quantity`` for ``region`` (or a bare ``material``).

    Looks in ``region.properties[quantity]``, then ``getattr(material,
    quantity)``.  A number is temperature independent; a callable is evaluated
    as ``value(temperature)`` and requires a temperature (K).  Raises
    ``KeyError`` naming the region when nothing is defined.

    Examples
    --------
    >>> from photonix.xsection import rect
    >>> r = rect(0, 1, 0, 1, 1.444, properties={"k_th": 1.38}, name="oxide")
    >>> resolve_property(r, "k_th")
    1.38
    """
    props = getattr(region, "properties", {}) if region is not None else {}
    mat = getattr(region, "material", None) if material is None else material
    if quantity in props:
        value = props[quantity]
        where = f"region {getattr(region, 'name', None)!r} property {quantity!r}"
    elif mat is not None and not isinstance(mat, (int, float, complex, tuple, list)) \
            and hasattr(mat, quantity):
        value = getattr(mat, quantity)
        where = f"material {mat!r} attribute {quantity!r}"
    else:
        label = getattr(region, "name", None) or (sorted(getattr(region, "tags", ())) or [repr(mat)])
        raise KeyError(
            f"no {quantity!r} defined for region {label}: add it to the region's properties or use a "
            f"material object with a {quantity!r} attribute"
        )
    if callable(value):
        if temperature is None:
            raise ValueError(f"{where} is temperature dependent: pass a temperature (K)")
        value = value(temperature)
    out = np.asarray(value)
    if out.shape != () or not np.isfinite(out):
        raise ValueError(f"{where} must be a finite scalar")
    return float(out) if not np.iscomplexobj(out) else complex(out)
