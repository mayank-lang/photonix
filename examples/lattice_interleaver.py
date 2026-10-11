"""Example: synthesize a flat-top lattice interleaver and compare it with an MZI.

A 50 GHz interleaver separates odd and even channels of a 50 GHz grid, so each
output has a 100 GHz FSR. ``photonix.synthesis.interleaver`` designs the
maximally flat half-band lattice exactly (couplings and heater phases by
polynomial peeling), and the result is simulated as a full component-level
netlist. An equiripple design with the same five unit delays trades the
maximally flat centre for a guaranteed isolation over a +-12.5 GHz band.

Run:  python examples/lattice_interleaver.py
Saves: examples/outputs/lattice_interleaver.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px
from photonix import synthesis as sy


def _cross_db(design, df):
    s = design.model()(wl=design.wavelength(df))
    return 10 * np.log10(np.maximum(px.to_numpy(px.power(s[("o1", "o3")])), 1e-12))


def main() -> None:
    fsr = 100e9
    designs = {
        "MZI": sy.interleaver(1, fsr_hz=fsr),
        "max-flat, N = 3": sy.interleaver(3, fsr_hz=fsr),
        "equiripple, 5 delays": sy.synthesize_lattice(*sy.equiripple_lowpass(5, 0.25 * np.pi, 0.75 * np.pi),
                                                      fsr_hz=fsr),
    }
    for name, d in designs.items():
        print(f"{name}: couplings {np.round(d.couplings, 4).tolist()}, delays {d.delays} x {d.delta_length:.2f} um,")
        print(f"    phases/pi {np.round(np.array(d.phases) / np.pi, 4).tolist()}, max error {d.max_error:.1e}")
        band = _cross_db(d, np.array([12.5e9, 37.5e9]))
        print(f"    at +-12.5 GHz: insertion loss {-band[0]:.3f} dB, crosstalk into the other port {band[1]:.1f} dB")

    df = np.linspace(-fsr, fsr, 4001)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for (name, d), style in zip(designs.items(), (":", "-", "--"), strict=True):
        db = _cross_db(d, df)
        ax1.plot(df / 1e9, db, style, label=name)
        ax2.plot(df / 1e9, db, style, label=name)
    ax1.set(xlabel="Detuning (GHz)", ylabel="o1 -> o3 (dB)", ylim=(-60, 2),
            title="Cross port (o4 is the same, shifted 50 GHz)")
    ax2.set(xlabel="Detuning (GHz)", ylabel="o1 -> o3 (dB)", xlim=(-20, 20), ylim=(-1.0, 0.05),
            title="Pass-band flatness")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    print("Saved", save(fig, "lattice_interleaver.png"))


if __name__ == "__main__":
    main()
