"""Tests for wavelength sweeps (broadband S-parameters)."""
from __future__ import annotations

import numpy as np

from photonix.em import sweep


def test_sweep_stacks_arrays():
    wls = np.linspace(1.5, 1.6, 6)
    s = sweep(lambda wl: {("o1", "o2"): 0.5 + 1j * wl}, wls)
    assert s[("o1", "o2")].shape == (6,)
    assert np.allclose(s[("o1", "o2")].imag, wls)


def test_sweep_taper_broadband_high_transmission():
    from photonix.em.components import taper

    wls = np.linspace(1.5, 1.6, 4)
    s = sweep(taper, wls, width1=0.5, width2=0.9, length=12.0,
              num_sections=8, num_modes=4, points=141)
    t = np.abs(s[("o1", "o2")]) ** 2
    assert t.shape == (4,)
    assert np.all(t > 0.85)  # adiabatic taper: high across the band
