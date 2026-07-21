"""Inverse-design routines: epigraph (max-min) optimization + binarization.

A device must work across a *band*, not at one wavelength. The **epigraph**
formulation maximizes the worst-case performance:

    maximize_rho  min_lambda  f_lambda(rho)

We use the smooth lower bound ``softmin_p(f) = -1/p log mean exp(-p f)`` (the
differentiable epigraph relaxation; ``-> min`` as ``p -> inf``). Its gradient is a
softmax-weighted combination of the per-wavelength adjoint gradients, so effort
concentrates on the currently-worst wavelength. Combined with fabrication
constraints (filter + ``beta``-continuation projection), this yields a binary,
minimum-feature design that is robust across the band.
"""
from __future__ import annotations

import numpy as np

from . import fabrication as fab
from .fdfd import focus_objective

__all__ = ["binarization", "softmin", "robust_focus_design"]


def binarization(rho_projected: np.ndarray, mask=None) -> float:
    """Binarization measure in [0, 1]: ``1 - 4*mean(rho*(1-rho))`` (1 = fully 0/1)."""
    r = rho_projected if mask is None else rho_projected[mask]
    return float(1.0 - 4.0 * np.mean(r * (1.0 - r)))


def softmin(values: np.ndarray, p: float):
    """Smooth minimum and its softmax weights: returns ``(softmin, weights)``."""
    v = np.asarray(values, float)
    w = np.exp(-p * (v - v.min()))      # stable softmax of -p v
    w = w / w.sum()
    sm = -1.0 / p * np.log(np.mean(np.exp(-p * (v - v.min())))) + v.min()
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
    rng = np.random.default_rng(seed)
    rho = 0.5 + 0.05 * rng.standard_normal((ny, nx))
    betas = np.concatenate([np.full(max(steps // len(beta_schedule), 1), b) for b in beta_schedule])
    if len(betas) < steps:
        betas = np.concatenate([betas, np.full(steps - len(betas), beta_schedule[-1])])
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
        sm, w = softmin(foms, p)
        history.append(float(foms.min()))
        grad = sum(wi * gi for wi, gi in zip(w, grads, strict=False))   # softmax-weighted (worst-case) grad
        lr = 0.08 / (np.max(np.abs(grad[mask])) + 1e-30)
        rho[mask] = np.clip(rho[mask] + lr * grad[mask], 0, 1)
    eps_des, _ = fab.density_to_eps(rho, eps_min=eps_min, eps_max=eps_max,
                                    radius_cells=radius_cells, beta=float(betas[-1]))
    eps = np.where(mask, eps_des, eps_min)
    perf = np.array([focus_objective(eps, dx=dx, dy=dy, wl=float(wl), source=source,
                                     target=target, npml=npml)[0] for wl in wls])
    return rho, eps, history, perf
