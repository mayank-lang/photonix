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
