"""Tests for epigraph (max-min) inverse design + binarization."""
from __future__ import annotations

import numpy as np

from photonix.em.fdfd import point_source
from photonix.em.inverse import binarization, robust_focus_design, softmin


def test_softmin_is_lower_bound_and_weights_worst():
    v = np.array([1.0, 2.0, 5.0])
    sm, w = softmin(v, p=30.0)
    assert abs(sm - v.min()) < 0.1          # smooth min approaches the true min
    assert sm < v.mean()                    # and sits below the mean
    assert abs(w.sum() - 1.0) < 1e-12
    assert w[0] == w.max()  # weight concentrates on the worst (smallest) value


def test_binarization_metric():
    assert binarization(np.full((8, 8), 0.5)) < 0.05      # gray -> ~0
    assert binarization(np.array([0.0, 1.0, 0.0, 1.0])) > 0.99  # binary -> 1


def test_robust_design_improves_worst_case():
    nclad, ncore = 1.444, 3.4757
    ny = nx = 48
    dx = dy = 0.05
    src = point_source(ny, nx, ny // 2, 7)
    target = (ny // 2, nx - 9)
    mask = np.zeros((ny, nx), bool)
    mask[ny // 2 - 9:ny // 2 + 9, 15:nx - 10] = True
    wls = np.array([1.52, 1.58])
    _rho, _eps, hist, _perf = robust_focus_design(
        wls, ny=ny, nx=nx, dx=dx, dy=dy, mask=mask, source=src, target=target,
        eps_min=nclad**2, eps_max=ncore**2, radius_cells=2.0, steps=10,
    )
    assert max(hist) > 1.2 * hist[0]  # worst-case objective improves
