"""Example: a routed ring filter from layout to spectrum, in one consistent flow.

1. Place grating couplers and a geometry-driven ring with the demo PDK.
2. Route them with Euler-bend Manhattan routes.
3. Run the DRC pre-check and write GDSII (if gdstk is installed).
4. Extract the netlist -- every cell carries its drawn length, radius and gap
   into the simulation -- and simulate it with the PDK's waveguide loss.
5. Extract resonance wavelength, Q and extinction from the through spectrum.

Run:  python examples/layout_to_circuit.py
Saves: examples/outputs/layout_to_circuit.png
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
import photonix.layout as lay


def build(pdk) -> lay.Cell:
    top = lay.Cell("ring_filter_chip")
    ring = top.add_ref(pdk.get_layout("ring_single", radius=12.0, gap=0.28), origin=(0.0, 0.0), name="ring")
    gc_in = top.add_ref(pdk.get_layout("grating_coupler"), name="gc_in")
    gc_out = top.add_ref(pdk.get_layout("grating_coupler"), name="gc_out")
    # Grating couplers 127 um apart (a standard fibre-array pitch), facing up.
    gc_in.connect("o1", lay.Port("p", (-63.5, -60.0), 270.0))
    gc_out.connect("o1", lay.Port("p", (63.5, -60.0), 270.0))
    top.add_ref(pdk.route(gc_in.port("o1"), ring.port("o1"), bend="euler"), name="r_in")
    top.add_ref(pdk.route(ring.port("o2"), gc_out.port("o1"), bend="euler"), name="r_out")
    return top


def main() -> None:
    pdk = px.pdk.demo_pdk()
    chip = build(pdk)

    report = pdk.drc_precheck(chip)
    print(report)
    if lay.gdstk_available():
        path = os.path.join(tempfile.mkdtemp(), "ring_filter_chip.gds")
        lay.write_gds(chip, path)
        print("GDSII written to", path)

    nl = lay.extract_netlist(chip)
    print("instances:", nl.instances)
    print("routed lengths [um]:", {k: round(v["length"], 2) for k, v in nl.settings.items() if "length" in v})

    wl = px.linspace(1.540, 1.560, 40001)   # ~0.5 pm: resolves Q ~ 2e5
    s = pdk.simulate(chip, wl=wl)
    fibre_in, fibre_out = "gc_in_o2", "gc_out_o2"
    T = px.to_numpy(px.power(s[(fibre_in, fibre_out)]))
    ring_only = px.to_numpy(px.power(pdk.evaluate("ring_single", wl=wl, radius=12.0, gap=0.28)[("o1", "o2")]))

    res = px.analysis.find_resonances(px.to_numpy(wl), ring_only, prominence=0.05)
    for r in res:
        print(f"resonance {r.wavelength * 1e3:.3f} nm  FWHM {r.fwhm * 1e3:.3f} nm  "
              f"Q_L {r.q_loaded:,.0f}  ER {r.extinction_db:.1f} dB")
    if len(res) > 1:
        print(f"FSR {px.analysis.free_spectral_range(res) * 1e3:.3f} nm")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.1, 1]})
    px.viz.plot_cell(chip, ax=ax0)
    ax0.set_title("Layout (demo PDK, Euler-bend routes)")
    ax1.plot(px.to_numpy(wl) * 1e3, px.analysis.to_db(T), label="fibre to fibre")
    ax1.plot(px.to_numpy(wl) * 1e3, px.analysis.to_db(ring_only), "--", label="ring only")
    ax1.set_xlabel("wavelength [nm]")
    ax1.set_ylabel("transmission [dB]")
    ax1.set_title("Extracted-layout simulation")
    ax1.legend()
    fig.tight_layout()
    print("Saved", save(fig, "layout_to_circuit.png"))
    assert report.clean and np.all(T <= 1.0)


if __name__ == "__main__":
    main()
