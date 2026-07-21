"""Numerical backend abstraction for photonix.

photonix is *differentiable-first*: the default backend is JAX, which provides
``jit`` compilation, ``grad``/``value_and_grad`` automatic differentiation, and
transparent CPU/GPU/TPU execution. When JAX is not installed the package
gracefully falls back to NumPy so that non-differentiable functionality (layout,
data structures, plotting) still works.

All numerical code in photonix should import the array module from here::

    from photonix.core.backend import xp, jit, grad

and use ``xp`` instead of importing ``numpy``/``jax.numpy`` directly. This keeps
the whole library backend-agnostic and GPU-ready.
"""
from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any

__all__ = [
    "HAS_JAX",
    "xp",
    "jit",
    "grad",
    "value_and_grad",
    "vmap",
    "jacfwd",
    "jacrev",
    "stop_gradient",
    "to_numpy",
    "asarray",
    "use_x64",
    "device_count",
    "backend_name",
]

# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #
_FORCE_NUMPY = os.environ.get("PHOTONIX_BACKEND", "").lower() == "numpy"

try:  # pragma: no cover - exercised by environment, not unit tests
    if _FORCE_NUMPY:
        raise ImportError("PHOTONIX_BACKEND=numpy")
    import jax
    import jax.numpy as _jnp

    HAS_JAX = True
except Exception:  # noqa: BLE001 - any import failure means no JAX
    HAS_JAX = False


if HAS_JAX:
    xp = _jnp
    jit = jax.jit
    grad = jax.grad
    value_and_grad = jax.value_and_grad
    vmap = jax.vmap
    jacfwd = jax.jacfwd
    jacrev = jax.jacrev
    stop_gradient = jax.lax.stop_gradient

    def use_x64(enabled: bool = True) -> None:
        """Enable/disable 64-bit precision (recommended for photonics accuracy)."""
        jax.config.update("jax_enable_x64", bool(enabled))

    def device_count() -> int:
        try:
            return jax.device_count()
        except Exception:  # noqa: BLE001
            return 1

    def backend_name() -> str:
        try:
            return f"jax:{jax.default_backend()}"
        except Exception:  # noqa: BLE001
            return "jax"

else:  # NumPy fallback ------------------------------------------------------ #
    import numpy as xp  # type: ignore[no-redef]

    def _no_jit(fn: Callable | None = None, *args: Any, **kwargs: Any):
        """``jit`` is a no-op under NumPy. Supports bare and parametrized use."""
        if fn is None:
            return lambda f: f
        return fn

    jit = _no_jit  # type: ignore[assignment]

    def _grad_unavailable(*_a: Any, **_k: Any):
        raise RuntimeError(
            "Automatic differentiation requires JAX. Install it with "
            "`pip install \"photonix[jax]\"` (or `pip install jax jaxlib`)."
        )

    grad = _grad_unavailable  # type: ignore[assignment]
    value_and_grad = _grad_unavailable  # type: ignore[assignment]
    jacfwd = _grad_unavailable  # type: ignore[assignment]
    jacrev = _grad_unavailable  # type: ignore[assignment]

    def vmap(fn: Callable, in_axes: Any = 0, out_axes: Any = 0) -> Callable:
        """Minimal NumPy ``vmap`` shim over the leading axis (eager fallback)."""

        @functools.wraps(fn)
        def wrapped(*args: Any):
            import numpy as _np

            n = None
            for a in args:
                arr = _np.asarray(a)
                if arr.ndim > 0:
                    n = arr.shape[0]
                    break
            if n is None:
                return fn(*args)
            return _np.stack([fn(*[_np.asarray(a)[i] for a in args]) for i in range(n)])

        return wrapped

    def stop_gradient(x: Any) -> Any:  # type: ignore[misc]
        return x

    def use_x64(enabled: bool = True) -> None:  # noqa: ARG001
        # NumPy already defaults to float64.
        return None

    def device_count() -> int:
        return 1

    def backend_name() -> str:
        return "numpy"


# --------------------------------------------------------------------------- #
# Helpers usable under either backend
# --------------------------------------------------------------------------- #
def asarray(x: Any, dtype: Any = None) -> Any:
    """Convert ``x`` to the active backend's array type."""
    return xp.asarray(x, dtype=dtype)


def to_numpy(x: Any):
    """Return a plain ``numpy.ndarray`` copy of ``x`` (host transfer if on GPU)."""
    import numpy as _np

    return _np.asarray(x)
