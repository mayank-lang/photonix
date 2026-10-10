"""Example: an 8-channel, 100 GHz AWG -- design, crosstalk, and parametric yield.

1. Derive the phased-array geometry from the channel plan (``awg_design``).
2. Simulate the demultiplexer spectra and extract pass-band metrics and the
   channel crosstalk matrix.
3. Estimate the yield of channel registration (centre within +/-0.1 nm of the
   ITU grid) when the arrayed-waveguide index varies wafer to wafer.

Run:  python examples/awg_demux.py
Saves: examples/outputs/awg_demux.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px
from photonix import analysis as an
from photonix.pdk import MonteCarloSpec, ProcessCorner, ProcessStudy

PLAN = {"n_channels": 8, "channel_spacing": 0.0008, "wl0": 1.55, "passband_fwhm": 0.00032}


def main() -> None:
    design = px.components.awg_design(**PLAN)
    print(f"grating order {design['order']}, dL = {design['delta_length']:.3f} um, "
          f"FSR = {design['fsr'] * 1e3:.3f} nm, {design['n_arms']} arms")

    wl = np.linspace(1.5455, 1.5545, 9001)
    ideal = px.components.awg(wl=wl, **PLAN)
    s = px.components.awg(wl=wl, insertion_loss_db=2.5, nonuniformity_db=1.0, phase_error_std=0.15, **PLAN)
    T = np.stack([px.to_numpy(px.power(s[("o1", f"o{k + 2}")])) for k in range(PLAN["n_channels"])])

    m = an.passband_metrics(wl, T[3], center=design["channel_wavelengths"][3])
    print(f"ch4: IL {m.insertion_loss_db:.2f} dB, 1-dB BW {m.bandwidth_1db * 1e3:.3f} nm, "
          f"3-dB BW {m.bandwidth_3db * 1e3:.3f} nm")
    T_ideal = np.stack([px.to_numpy(px.power(ideal[("o1", f"o{k + 2}")])) for k in range(PLAN["n_channels"])])
    for label, spectra in (("ideal", T_ideal), ("0.15 rad arm phase errors", T)):
        X = an.channel_crosstalk(wl, spectra, design["channel_wavelengths"])
        worst = max(X[k, j] for k in range(8) for j in range(8) if j != k)
        print(f"worst crosstalk ({label}): {worst:.1f} dB")

    # Yield: arrayed-waveguide index varies by sigma = 1e-4 wafer to wafer.
    study = ProcessStudy(
        ProcessCorner("nominal", {"delta_neff": 0.0}),
        corners=(ProcessCorner("fast", {"delta_neff": -3e-4}), ProcessCorner("slow", {"delta_neff": 3e-4})),
        monte_carlo=MonteCarloSpec.independent({"delta_neff": 0.0}, {"delta_neff": 1e-4}),
        units={"delta_neff": "1"},
    )
    wl_fine = np.linspace(1.5506, 1.5514, 801)   # around channel 6

    def evaluate(case):
        sc = px.components.awg(wl=wl_fine, delta_neff=case.parameters["delta_neff"], **PLAN)
        return px.to_numpy(px.power(sc[("o1", "o7")]))

    target = design["channel_wavelengths"][5]
    spec = an.Spec("centre error", lambda t: (wl_fine[np.argmax(t)] - target) * 1e3,
                   minimum=-0.1, maximum=0.1, unit="nm")
    result = an.yield_analysis(evaluate, [spec], study=study, samples=300, seed=7)
    print(result)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    for k in range(PLAN["n_channels"]):
        ax0.plot(wl * 1e3, an.to_db(T[k]), lw=1)
    ax0.set_ylim(-60, 1)
    ax0.set_xlabel("wavelength [nm]")
    ax0.set_ylabel("transmission [dB]")
    ax0.set_title("8 x 100 GHz AWG (phased-array model, 0.15 rad arm phase errors)")
    errors = [result.values["centre error"][c] for c in result.monte_carlo_cases]
    ax1.hist(errors, bins=25, color="tab:blue", alpha=0.8)
    for edge in (-0.1, 0.1):
        ax1.axvline(edge, color="k", ls="--")
    lo, hi = result.yield_interval
    ax1.set_title(f"channel registration: yield {result.yield_fraction:.1%} ({lo:.1%}-{hi:.1%})")
    ax1.set_xlabel("centre-wavelength error [nm]")
    fig.tight_layout()
    print("Saved", save(fig, "awg_demux.png"))


if __name__ == "__main__":
    main()
