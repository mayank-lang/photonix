"""Example: Bragg-grating variants and a carrier-depletion ring modulator.

Left: one coupled-mode transfer-matrix model covers uniform, apodized and
quarter-wave-shifted gratings. Right: a ring modulator assembled as a netlist
from the physical evanescent coupler and the p-n depletion phase shifter; the
reverse bias shifts the resonance, and the circuit is differentiable, so the
modulation efficiency d(T)/dV comes from one ``grad`` call.

Run:  python examples/bragg_and_ring_modulator.py
Saves: examples/outputs/bragg_and_ring_modulator.png
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px
from photonix import analysis as an
from photonix.circuit import Netlist, circuit_from_netlist


def ring_modulator():
    radius = 8.0
    nl = Netlist(name="ring_modulator")
    nl.add("cpl", "evanescent_coupler", gap=0.19, length=0.0, bend_radius=radius)   # ~critical
    nl.add("pn", "pn_phase_shifter", length=2 * math.pi * radius)
    nl.connect(("cpl", "o3"), ("pn", "o1"))
    nl.connect(("pn", "o2"), ("cpl", "o2"))
    nl.expose("in", ("cpl", "o1"))
    nl.expose("through", ("cpl", "o4"))
    return circuit_from_netlist(nl)


def main() -> None:
    wl_b = px.components.bragg_wavelength(0.3229)
    wl = np.linspace(wl_b - 0.02, wl_b + 0.02, 8001)
    variants = {
        "uniform": {},
        "Gaussian apodized": {"n_sections": 60, "apodization": "gaussian", "apodization_strength": 0.5},
        "quarter-wave shifted": {"phase_shift": math.pi, "n_sections": 2},
    }
    spectra = {}
    for name, kw in variants.items():
        s = px.components.bragg_grating(wl=wl, dn=0.03, n_periods=400, **kw)
        spectra[name] = (px.to_numpy(px.power(s[("o1", "o1")])), px.to_numpy(px.power(s[("o1", "o2")])))
    uni_r = spectra["uniform"][0]
    print(f"uniform grating: peak R = {uni_r.max():.4f}, "
          f"-30 dB stop-band width {an.bandwidth(wl, uni_r, -30.0)[2] * 1e3:.2f} nm")
    qw_t = spectra["quarter-wave shifted"][1]
    print(f"quarter-wave shift: T at Bragg = {qw_t[np.argmin(np.abs(wl - wl_b))]:.4f}")

    model = ring_modulator()
    wlr = px.linspace(1.5474, 1.5486, 24001)        # one resonance, 0.05 pm steps
    traces = {}
    for v in (0.0, 1.0, 2.0, 3.0):
        T = px.to_numpy(px.power(model(wl=wlr, pn={"voltage": v})[("in", "through")]))
        res = an.find_resonances(px.to_numpy(wlr), T, prominence=0.2)
        traces[v] = (T, res)
    centres = {v: r[0].wavelength for v, (_t, r) in traces.items() if r}
    shift_pm = (centres[2.0] - centres[0.0]) * 1e6
    print(f"resonance shift 0 -> 2 V reverse: {shift_pm:.1f} pm  "
          f"(Q_L = {traces[0.0][1][0].q_loaded:,.0f}, ER = {traces[0.0][1][0].extinction_db:.1f} dB)")

    if px.HAS_JAX:
        wl_bias = float(centres[0.0]) + 0.5 * traces[0.0][1][0].fwhm   # half-width point
        slope = px.grad(lambda v: px.power(model(wl=wl_bias, pn={"voltage": v})[("in", "through")]))(0.0)
        print(f"modulation efficiency at the half-width point: dT/dV = {float(slope):.3f} /V")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (_R, T) in spectra.items():
        ax0.plot((wl - wl_b) * 1e3, an.to_db(T), label=f"{name} (T)")
    ax0.plot((wl - wl_b) * 1e3, an.to_db(spectra["uniform"][0]), "k:", lw=1, label="uniform (R)")
    ax0.set_ylim(-60, 2)
    ax0.set_xlabel("detuning from Bragg wavelength [nm]")
    ax0.set_ylabel("power [dB]")
    ax0.set_title("Bragg gratings (CMT transfer matrix, 400 periods)")
    ax0.legend(fontsize=8)
    for v, (T, _r) in traces.items():
        ax1.plot(px.to_numpy(wlr) * 1e3, an.to_db(T), label=f"{v:.0f} V reverse")
    ax1.set_xlabel("wavelength [nm]")
    ax1.set_ylabel("through [dB]")
    ax1.set_title("Depletion ring modulator (R = 8 um, 5e17 cm^-3)")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    print("Saved", save(fig, "bragg_and_ring_modulator.png"))
    assert shift_pm > 0.0


if __name__ == "__main__":
    main()
