"""Smoke tests for the visualization helpers (headless Agg)."""
from __future__ import annotations

import numpy as np

import photonix as px


def test_plot_spectrum_returns_axes(wl):
    s = px.circuit.mzi(delta_length=40.0)(wl=wl)
    ax = px.viz.plot_spectrum(s, wl, [("in0", "out0")])
    assert ax.get_xlabel() == "Wavelength (µm)"
    assert len(ax.lines) >= 1


def test_plot_phase_returns_axes(wl):
    s = px.circuit.mzi(delta_length=40.0)(wl=wl)
    ax = px.viz.plot_phase(s, wl, [("in0", "out0")])
    assert ax.get_ylabel() == "Phase (rad)"


def test_plot_mode_returns_axes():
    g = np.linspace(-2, 2, 40)
    f = np.exp(-(g[:, None] ** 2 + g[None, :] ** 2))
    ax = px.viz.plot_mode(f, g, g)
    assert ax.get_xlabel() == "x (µm)"


def test_plot_netlist_returns_axes():
    nl = px.circuit.Netlist(
        instances={"a": "straight", "b": "straight"},
        connections={("a", "o2"): ("b", "o1")},
        ports={"in0": ("a", "o1"), "out0": ("b", "o2")},
    )
    ax = px.viz.plot_netlist(nl)
    assert ax is not None


def _assert_keeps_backend(plot):
    """Call ``plot()`` under the svg backend and check it is still active."""
    import matplotlib

    original = matplotlib.get_backend()
    try:
        matplotlib.use("svg")
        plot()
        assert matplotlib.get_backend() == "svg"
    finally:
        matplotlib.use(original)


def test_plot_spectrum_keeps_the_users_matplotlib_backend(wl):
    s = px.circuit.mzi(delta_length=40.0)(wl=wl)
    _assert_keeps_backend(lambda: px.viz.plot_spectrum(s, wl, [("in0", "out0")]))


def test_plot_phase_keeps_the_users_matplotlib_backend(wl):
    s = px.circuit.mzi(delta_length=40.0)(wl=wl)
    _assert_keeps_backend(lambda: px.viz.plot_phase(s, wl, [("in0", "out0")]))


def test_plot_mode_keeps_the_users_matplotlib_backend():
    _assert_keeps_backend(lambda: px.viz.plot_mode(np.ones((3, 4))))


def test_plot_cell_keeps_the_users_matplotlib_backend():
    _assert_keeps_backend(lambda: px.viz.plot_cell(px.layout.components.straight()))


def test_plot_netlist_keeps_the_users_matplotlib_backend():
    nl = px.circuit.Netlist(
        instances={"a": "straight", "b": "straight"},
        connections={("a", "o2"): ("b", "o1")},
        ports={"in0": ("a", "o1"), "out0": ("b", "o2")},
    )
    _assert_keeps_backend(lambda: px.viz.plot_netlist(nl))
