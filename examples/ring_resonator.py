"""Example: an all-pass ring resonator built as a real feedback circuit.

Compares the circuit-solver result against the closed-form all-pass transfer
function (they agree to machine precision) and plots the spectrum.

Run:  python examples/ring_resonator.py
Saves: ring_resonator.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import photonix as px


def main() -> None:
    wl = px.linspace(1.545, 1.555, 4001)

    ring = px.circuit.ring(radius=10.0, coupling=0.05, loss_db_cm=50.0, neff=2.4, ng=4.2)
    s_circuit = ring(wl=wl)
    s_analytic = px.components.all_pass_ring(wl=wl, coupling=0.05, radius=10.0, loss_db_cm=50.0)

    err = float(np.max(np.abs(np.asarray(px.power(s_circuit[("in0", "out0")]))
                              - np.asarray(px.power(s_analytic[("o1", "o2")])))))
    print(f"max |circuit - analytic| = {err:.2e} (should be ~1e-15)")

    fig, ax = plt.subplots(figsize=(7, 4))
    px.viz.plot_spectrum(s_circuit, wl, [("in0", "out0")], ax=ax, unit="dB")
    ax.set_title("All-pass ring resonator (R = 10 µm, near critical coupling)")
    fig.tight_layout()
    fig.savefig("ring_resonator.png", dpi=130)
    print("Saved ring_resonator.png")


if __name__ == "__main__":
    main()
