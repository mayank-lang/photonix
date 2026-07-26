"""Example: simulate a Mach-Zehnder interferometer and plot its spectrum.

Run:  python examples/mzi_sweep.py
Saves: examples/outputs/mzi_sweep.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _output import save

import photonix as px


def main() -> None:
    wl = px.linspace(1.50, 1.60, 1001)
    mzi = px.circuit.mzi(delta_length=40.0, coupling=0.5)
    s = mzi(wl=wl)

    fig, ax = plt.subplots(figsize=(7, 4))
    px.viz.plot_spectrum(s, wl, [("o1", "o4"), ("o1", "o3")], ax=ax)
    ax.set_title("MZI transmission (ΔL = 40 µm)")
    fig.tight_layout()
    print("Saved", save(fig, "mzi_sweep.png"))


if __name__ == "__main__":
    main()
