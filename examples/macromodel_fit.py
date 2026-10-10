"""Example: turn sampled S-parameters into a causal, passive rational macromodel.

A ring notch filter behind a 1 mm bus (a 14 ps delay) is "measured" on a coarse
0.1 nm grid and exchanged as a Touchstone file. The script

1. checks the raw data for causality (generalized Kramers-Kronig relations),
2. fits a delayed rational model (delay extraction + vector fitting),
3. certifies passivity with the Hamiltonian test (enforcing it if needed),
4. compares the model and linear interpolation against the exact response on
   a 20x finer grid, and
5. sends 5 ps pulses off and on resonance through the exact time-domain simulator.

Run:  python examples/macromodel_fit.py
Saves: examples/outputs/macromodel_fit.png
"""
from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px
from photonix import macromodel as mm

C_UM = 299792458e6  # speed of light in um/s


def device(wl):
    """Bus waveguide (1 mm) followed by an all-pass ring: a reciprocal two-port."""
    h = (px.to_numpy(px.components.straight(wl=wl, length=1000.0)[("o1", "o2")])
         * px.to_numpy(px.components.all_pass_ring(wl=wl, coupling=0.15, loss_db_cm=20.0)[("o1", "o2")]))
    return {("o1", "o2"): h, ("o2", "o1"): h}


def main() -> None:
    wl = np.linspace(1.545, 1.565, 201)  # 0.1 nm "measurement" grid
    measured = px.SParameterDataset.from_sdict(wl, device(wl), metadata={"source": "synthetic measurement"})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ring.s2p")
        measured.save_touchstone(path)
        data = px.SParameterDataset.load_touchstone(path)

    causal = mm.causality_report(data.wavelengths, data.s, ports=data.ports)
    print(f"raw data causal: {causal.causal}")

    model = data.to_rational()
    meta = model.metadata
    print(f"fit: {model.n_poles} poles, delay {model.delays[1, 0] * 1e12:.2f} ps, "
          f"relative RMS {meta['relative_rms_error']:.1e}")

    report = mm.passivity_report(model)
    if not report.passive:
        model = mm.enforce_passivity(model)
        report = mm.passivity_report(model)
    print(f"passive (Hamiltonian certificate): {report.passive}")

    fine = np.linspace(wl[0], wl[-1], 20 * wl.size)
    exact = device(fine)[("o1", "o2")]
    fitted = px.to_numpy(model(wl=fine)[("o1", "o2")])
    interpolated = data.interpolate(fine).s[:, 1, 0]
    print(f"max error on the fine grid: rational {np.max(np.abs(fitted - exact)):.1e}, "
          f"linear interpolation {np.max(np.abs(interpolated - exact)):.1e}")

    # Gaussian envelopes (5 ps FWHM) off resonance and on the 1.5526 um resonance. The model works in
    # baseband about its carrier, so a carrier offset is a complex exponential on the envelope.
    dt = 0.05e-12
    t = np.arange(4000) * dt
    pulse = np.exp(-2 * np.log(2) * ((t - 20e-12) / 5e-12) ** 2)
    outputs = {}
    for label, centre in (("off resonance", model.carrier_wl), ("on resonance", 1.55263)):
        offset_hz = C_UM / centre - model.carrier_hz
        u = pulse * np.exp(2j * np.pi * offset_hz * t)
        outputs[label] = model.simulate(np.stack([u, np.zeros_like(u)], axis=1), dt)[:, 1]
        energy = np.sum(np.abs(outputs[label]) ** 2) / np.sum(pulse**2)
        print(f"pulse {label}: transmitted energy {energy:.3f}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    ax0.plot(fine, 20 * np.log10(np.abs(exact)), "k", lw=2.5, label="exact")
    ax0.plot(fine, 20 * np.log10(np.abs(fitted)), "C1--", label=f"rational ({model.n_poles} poles + delay)")
    ax0.plot(fine, 20 * np.log10(np.abs(interpolated)), "C0:", label="linear interpolation")
    ax0.plot(wl, 20 * np.log10(np.abs(data.s[:, 1, 0])), "C0.", ms=3, label="samples")
    ax0.set_xlim(1.5516, 1.5536)
    ax0.set_xlabel("wavelength (um)")
    ax0.set_ylabel("|S21| (dB)")
    ax0.legend(fontsize=8)
    ax1.plot(t * 1e12, np.abs(pulse) ** 2, "k", label="input")
    for colour, (label, y) in zip(("C1", "C2"), outputs.items(), strict=True):
        ax1.plot(t * 1e12, np.abs(y) ** 2, colour, label=f"output, {label}")
    ax1.set_xlim(0, 80)
    ax1.set_xlabel("time (ps)")
    ax1.set_ylabel("power (a.u.)")
    ax1.legend(fontsize=8)
    fig.suptitle("Delayed rational macromodel of a ring behind a 1 mm bus")
    fig.tight_layout()
    print("Saved", save(fig, "macromodel_fit.png"))


if __name__ == "__main__":
    main()
