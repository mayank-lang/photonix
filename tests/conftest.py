"""Shared pytest fixtures and markers for the photonix test suite."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless plotting for viz tests

import photonix as px  # noqa: E402

requires_jax = pytest.mark.skipif(not px.HAS_JAX, reason="requires the JAX backend")


@pytest.fixture
def wl():
    """A C-band wavelength sweep (µm)."""
    return px.linspace(1.50, 1.60, 401)


@pytest.fixture
def two_port():
    """A simple lossless 2-port SDict (a phase delay)."""
    return {("o1", "o2"): np.exp(-1j * 0.7), ("o2", "o1"): np.exp(-1j * 0.7)}


@pytest.fixture
def four_port():
    """A 50/50 directional coupler SDict."""
    return px.components.directional_coupler(coupling=0.5)


def finite_diff(f, x, h=1e-4):
    """Central finite difference of scalar ``f`` at ``x``."""
    return (f(x + h) - f(x - h)) / (2 * h)
