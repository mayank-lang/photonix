"""Gradient-based optimizers for inverse design.

``adam`` is a dependency-free JAX implementation operating on a pytree of
parameters. ``minimize_scipy`` bridges to ``scipy.optimize`` for quasi-Newton
methods using JAX gradients.
"""
from __future__ import annotations

from collections.abc import Callable

from photonix.core.backend import HAS_JAX, grad, to_numpy, xp

__all__ = ["adam", "minimize_scipy", "OptResult"]


class OptResult(dict):
    """Lightweight result container (dict with attribute access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def adam(
    loss_fn: Callable,
    params0: dict,
    *,
    steps: int = 200,
    lr: float = 0.05,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> OptResult:
    """Minimize ``loss_fn(params)`` with Adam over a dict of scalar/array params.

    Parameters
    ----------
    loss_fn
        Scalar loss as a function of a parameter dict.
    params0
        Initial parameters (dict of floats or arrays).
    steps, lr, b1, b2, eps
        Standard Adam hyper-parameters.

    Returns
    -------
    OptResult
        With ``params`` (optimized), ``loss`` (final), and ``history`` (list).

    Examples
    --------
    >>> import photonix as px
    >>> res = px.optim.adam(lambda p: (p["x"] - 3.0) ** 2, {"x": 0.0}, steps=200)  # doctest: +SKIP
    >>> abs(float(res.params["x"]) - 3.0) < 1e-2  # doctest: +SKIP
    True
    """
    if not HAS_JAX:
        raise RuntimeError("adam requires JAX. Install photonix[jax].")
    if not isinstance(steps, int) or steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}.")
    if lr <= 0 or eps <= 0:
        raise ValueError("lr and eps must be positive.")
    if not (0 <= b1 < 1 and 0 <= b2 < 1):
        raise ValueError("b1 and b2 must satisfy 0 <= beta < 1.")

    import jax

    params = {k: xp.asarray(v, dtype=float) for k, v in params0.items()}
    m = {k: xp.zeros_like(v) for k, v in params.items()}
    v = {k: xp.zeros_like(val) for k, val in params.items()}
    gfn = grad(loss_fn)
    history = []

    for t in range(1, steps + 1):
        g = gfn(params)
        history.append(float(loss_fn(params)))
        new_params, new_m, new_v = {}, {}, {}
        for k in params:
            mk = b1 * m[k] + (1 - b1) * g[k]
            vk = b2 * v[k] + (1 - b2) * g[k] ** 2
            mhat = mk / (1 - b1 ** t)
            vhat = vk / (1 - b2 ** t)
            new_params[k] = params[k] - lr * mhat / (jax.numpy.sqrt(vhat) + eps)
            new_m[k], new_v[k] = mk, vk
        params, m, v = new_params, new_m, new_v

    return OptResult(params=params, loss=float(loss_fn(params)), history=history)


def minimize_scipy(loss_fn: Callable, params0: dict, *, method: str = "L-BFGS-B", **kwargs) -> OptResult:
    """Minimize ``loss_fn`` with SciPy using JAX gradients over a flat vector.

    The parameter dict is flattened to a 1-D vector for SciPy and unflattened for
    ``loss_fn``. Requires JAX (for gradients) and SciPy.
    """
    if not HAS_JAX:
        raise RuntimeError("minimize_scipy requires JAX for gradients.")
    import numpy as np
    from scipy.optimize import minimize

    keys = list(params0)
    shapes = {k: xp.asarray(params0[k]).shape for k in keys}
    sizes = {k: int(np.prod(shapes[k]) or 1) for k in keys}

    def pack(d):
        return np.concatenate([to_numpy(xp.asarray(d[k]).ravel()) for k in keys])

    def unpack(x):
        out, i = {}, 0
        for k in keys:
            out[k] = xp.asarray(x[i : i + sizes[k]].reshape(shapes[k] or ()))
            i += sizes[k]
        return out

    gfn = grad(loss_fn)

    def f(x):
        return float(loss_fn(unpack(x)))

    def jac(x):
        g = gfn(unpack(x))
        return pack(g).astype(float)

    res = minimize(f, pack(params0), jac=jac, method=method, **kwargs)
    return OptResult(params=unpack(res.x), loss=float(res.fun), success=bool(res.success))
