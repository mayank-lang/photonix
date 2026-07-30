"""Inverse-design routines: epigraph (max-min) optimization + binarization.

A device must work across a *band*, not at one wavelength. The **epigraph**
formulation maximizes the worst-case performance:

    maximize_rho  min_lambda  f_lambda(rho)

We use the smooth lower bound ``softmin_p(f) = -1/p log sum exp(-p f)`` (the
differentiable epigraph relaxation; ``-> min`` as ``p -> inf``). Its gradient is a
softmax-weighted combination of the per-wavelength adjoint gradients, so effort
concentrates on the currently-worst wavelength. Combined with fabrication
constraints (filter + ``beta``-continuation projection), this encourages a binary,
length-scale-controlled design that is robust across the band. A density filter is
not, by itself, a mathematical guarantee on both minimum solid and void features.
"""
from __future__ import annotations

import numpy as np

from . import fabrication as fab
from .fdfd import focus_objective

__all__ = ["binarization", "softmin", "robust_focus_design"]


def binarization(rho_projected: np.ndarray, mask=None) -> float:
    """Binarization measure in [0, 1]: ``1 - 4*mean(rho*(1-rho))`` (1 = fully 0/1)."""
    rho_projected = np.asarray(rho_projected, dtype=float)
    r = rho_projected if mask is None else rho_projected[np.asarray(mask, dtype=bool)]
    if r.size == 0:
        raise ValueError("binarization requires at least one selected density")
    if not np.all(np.isfinite(r)) or np.any((r < 0) | (r > 1)):
        raise ValueError("projected densities must be finite and lie in [0, 1]")
    return float(1.0 - 4.0 * np.mean(r * (1.0 - r)))


def softmin(values: np.ndarray, p: float):
    """Smooth minimum and its softmax weights: returns ``(softmin, weights)``."""
    v = np.asarray(values, float)
    if v.size == 0:
        raise ValueError("softmin requires at least one value")
    if not np.all(np.isfinite(v)):
        raise ValueError("softmin values must be finite")
    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be positive and finite")
    w = np.exp(-p * (v - v.min()))      # stable softmax of -p v
    w = w / w.sum()
    # Deliberately use sum, not mean: -log(sum(exp(-p*v)))/p is the
    # differentiable *lower bound* promised by the epigraph formulation.
    # Normalising by len(v) instead puts the result above the true minimum.
    sm = v.min() - np.log(np.sum(np.exp(-p * (v - v.min())))) / p
    return sm, w


def robust_focus_design(
    wls, *, ny, nx, dx, dy, mask, source, target,
    eps_min, eps_max, radius_cells=2.0, steps=24, p=40.0,
    beta_schedule=(4, 8, 16, 32, 64), seed=1, npml=12,
):
    """Epigraph max-min topology optimization of a focusing element over ``wls``.

    Returns ``(rho, eps, history, perf)`` where ``history`` is the worst-case
    objective per iteration and ``perf`` is the final per-wavelength objective.
    """
    wls = np.asarray(wls, dtype=float)
    if wls.ndim != 1 or wls.size == 0 or not np.all(np.isfinite(wls)) or np.any(wls <= 0):
        raise ValueError("wls must be a non-empty 1-D array of positive finite wavelengths")
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, (bool, np.bool_)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if not np.isfinite(p) or p <= 0:
        raise ValueError("p must be positive and finite")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (ny, nx):
        raise ValueError(f"mask must have shape {(ny, nx)}, got {mask.shape}")
    if not np.any(mask):
        raise ValueError("mask must select at least one design pixel")
    source = np.asarray(source)
    if source.shape != (ny, nx):
        raise ValueError(f"source must have shape {(ny, nx)}, got {source.shape}")
    beta_schedule = np.asarray(beta_schedule, dtype=float)
    if (beta_schedule.ndim != 1 or beta_schedule.size == 0
            or not np.all(np.isfinite(beta_schedule)) or np.any(beta_schedule < 0)):
        raise ValueError("beta_schedule must contain non-negative finite values")

    rng = np.random.default_rng(seed)
    rho = 0.5 + 0.05 * rng.standard_normal((ny, nx))
    # Filtering couples neighbouring densities, so fixed pixels need a fixed
    # physical value before filtering. Otherwise invisible random values outside
    # the mask perturb the optimized permittivity along the design boundary.
    rho[~mask] = 0.0
    betas = np.concatenate([
        np.full(max(steps // len(beta_schedule), 1), b) for b in beta_schedule
    ])
    if len(betas) < steps:
        betas = np.concatenate([betas, np.full(steps - len(betas), beta_schedule[-1])])
    betas = betas[:steps]
    history = []
    for it in range(steps):
        beta = float(betas[it])
        eps_des, cache = fab.density_to_eps(rho, eps_min=eps_min, eps_max=eps_max,
                                            radius_cells=radius_cells, beta=beta)
        foms, grads = [], []
        for wl in wls:
            eps = np.where(mask, eps_des, eps_min)
            fom, geps, _ = focus_objective(eps, dx=dx, dy=dy, wl=float(wl),
                                           source=source, target=target, npml=npml)
            foms.append(fom)
            grads.append(fab.density_to_eps_vjp(geps * mask, cache))
        foms = np.array(foms)
        _sm, w = softmin(foms, p)
        history.append(float(foms.min()))
        grad = sum(wi * gi for wi, gi in zip(w, grads, strict=False))   # softmax-weighted (worst-case) grad
        lr = 0.08 / (np.max(np.abs(grad[mask])) + 1e-30)
        rho[mask] = np.clip(rho[mask] + lr * grad[mask], 0, 1)
    final_beta = float(betas[-1]) if len(betas) else float(beta_schedule[0])
    eps_des, _ = fab.density_to_eps(rho, eps_min=eps_min, eps_max=eps_max,
                                    radius_cells=radius_cells, beta=final_beta)
    eps = np.where(mask, eps_des, eps_min)
    perf = np.array([focus_objective(eps, dx=dx, dy=dy, wl=float(wl), source=source,
                                     target=target, npml=npml)[0] for wl in wls])
    return rho, eps, history, perf
