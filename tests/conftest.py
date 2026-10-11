"""Shared pytest fixtures and markers for the photonix test suite.

The suite must collect and pass (with skips) on a minimal NumPy/SciPy + pytest
installation, which CI checks in its ``minimal`` job. Optional dependencies are
therefore never imported at module level here or in test modules: use the
``requires_*`` markers below or ``pytest.importorskip`` inside a test.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

try:  # headless plotting for the viz tests, when matplotlib is installed
    import matplotlib
except ImportError:  # pragma: no cover - exercised by the minimal-install CI job
    matplotlib = None
else:
    matplotlib.use("Agg")

import photonix as px  # noqa: E402


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_GDSTK = _installed("gdstk")
HAS_MATPLOTLIB = matplotlib is not None

requires_jax = pytest.mark.skipif(not px.HAS_JAX, reason="requires the JAX backend")
requires_gdstk = pytest.mark.skipif(not HAS_GDSTK, reason="requires the optional gdstk layout extra")
requires_matplotlib = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="requires the optional matplotlib extra")


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
