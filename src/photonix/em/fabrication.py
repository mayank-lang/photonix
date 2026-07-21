"""Fabrication constraints for density-based topology optimization.

Free-form (pixel) optimization produces gray, single-pixel features that cannot
be fabricated. The standard remedy is a two-step map from a raw design density
``rho in [0, 1]`` to permittivity:

1. **Density filter** (``conic_filter``): convolve ``rho`` with a conic kernel of
   a chosen radius -> enforces a *minimum feature size*.
2. **Projection** (``tanh_projection``): a smoothed Heaviside that pushes values
   toward 0/1 as ``beta`` grows -> *binarizes* the design.

Both steps are differentiable, so the FDFD adjoint gradient w.r.t. permittivity
is backpropagated to the raw density via :func:`density_to_eps_vjp`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve

__all__ = [
    "conic_kernel", "conic_filter", "conic_filter_adjoint",
    "tanh_projection", "tanh_projection_deriv",
    "density_to_eps", "density_to_eps_vjp",
]


def conic_kernel(radius_cells: float) -> np.ndarray:
    """Normalized conic (linear-decay) filter kernel of the given radius (cells)."""
    r = int(np.ceil(radius_cells))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    dist = np.sqrt(x**2 + y**2)
    k = np.maximum(0.0, 1.0 - dist / radius_cells)
    return k / k.sum()


def conic_filter(rho: np.ndarray, radius_cells: float) -> np.ndarray:
    """Apply the conic density filter (enforces minimum feature size)."""
    return convolve(rho, conic_kernel(radius_cells), mode="nearest")


def conic_filter_adjoint(g: np.ndarray, radius_cells: float) -> np.ndarray:
    """Exact adjoint of :func:`conic_filter` (replicate-padded convolution).

    ``conic_filter`` is a 'valid' convolution of the replicate-padded input, so
    its adjoint is a *full* convolution (kernel is symmetric) followed by the
    adjoint of replicate padding: folding the padded border strips back onto the
    edge cells. The filter is **not** self-adjoint at the boundary -- reusing
    ``conic_filter`` as its own VJP gives O(10%) gradient errors on edge/corner
    pixels for non-integer radii (interior pixels are unaffected).
    """
    from scipy.signal import convolve2d

    K = conic_kernel(radius_cells)
    r = K.shape[0] // 2
    g = np.asarray(g, float)
    n0, n1 = g.shape
    gp = convolve2d(g, K, mode="full")            # adjoint of the 'valid' conv
    # Adjoint of replicate padding: fold pad strips onto the nearest edge cells,
    # axis by axis (index clipping is separable, so sequential folds are exact).
    h = gp[r:n0 + r, :].copy()
    if r > 0:
        h[0, :] += gp[:r, :].sum(axis=0)
        h[n0 - 1, :] += gp[n0 + r:, :].sum(axis=0)
    out = h[:, r:n1 + r].copy()
    if r > 0:
        out[:, 0] += h[:, :r].sum(axis=1)
        out[:, n1 - 1] += h[:, n1 + r:].sum(axis=1)
    return out


def tanh_projection(rho: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    """Smoothed Heaviside projection toward 0/1 (binarization sharpens with beta)."""
    num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def tanh_projection_deriv(rho: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    """d(tanh_projection)/d(rho)."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (rho - eta)) ** 2) / den


def density_to_eps(rho, *, eps_min, eps_max, radius_cells=2.0, beta=8.0, eta=0.5):
    """Map raw density ``rho in [0,1]`` -> manufacturable permittivity.

    Pipeline: ``rho -> conic_filter -> tanh_projection -> eps``.
    Returns ``(eps, cache)`` where ``cache`` is reused by
    :func:`density_to_eps_vjp` for the gradient.
    """
    rho_f = conic_filter(rho, radius_cells)
    rho_p = tanh_projection(rho_f, beta, eta)
    eps = eps_min + rho_p * (eps_max - eps_min)
    return eps, {"rho_f": rho_f, "radius": radius_cells, "beta": beta, "eta": eta,
                 "span": eps_max - eps_min}


def density_to_eps_vjp(grad_eps, cache):
    """Backpropagate an eps-gradient to the raw density ``rho``.

    Uses :func:`conic_filter_adjoint` -- the *exact* adjoint of the
    replicate-padded conic filter. (The filter is self-adjoint in the interior
    thanks to the symmetric kernel, but not at the domain boundary, where the
    replicate padding breaks the symmetry.)
    """
    span = cache["span"]
    g = grad_eps * span                                   # d eps / d rho_p
    g = g * tanh_projection_deriv(cache["rho_f"], cache["beta"], cache["eta"])
    g = conic_filter_adjoint(g, cache["radius"])          # exact filter^T
    return g
