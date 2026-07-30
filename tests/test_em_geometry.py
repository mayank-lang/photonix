"""Regression tests for EM grid geometry semantics."""
from __future__ import annotations

import numpy as np
import pytest

from photonix.em.geometry import rectangular_waveguide, slab_profile


@pytest.mark.parametrize("resolution", [20, 40, 80])
def test_rectangular_resolution_is_exact_points_per_um(resolution):
    cs = rectangular_waveguide(width=0.53, thickness=0.23, margin=1.1,
                               resolution=resolution)
    assert cs.dx == pytest.approx(1.0 / resolution)
    assert cs.dy == pytest.approx(1.0 / resolution)
    # Cell edges cover at least the requested physical window.
    assert len(cs.x) * cs.dx >= 0.53 + 2 * 1.1
    assert len(cs.y) * cs.dy >= 0.23 + 2 * 1.1
    assert np.allclose(cs.x, -cs.x[::-1])
    assert np.allclose(cs.y, -cs.y[::-1])


def test_slab_profile_resolution_is_exact_points_per_um():
    eps, y = slab_profile(thickness=0.23, margin=1.1, resolution=37)
    assert y[1] - y[0] == pytest.approx(1.0 / 37)
    assert len(eps) == len(y)
