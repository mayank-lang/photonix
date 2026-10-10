"""Example: production routing -- fan-in, bundles, A* around obstacles, length-matched MZI arms.

1. Fan four fibre-array grating couplers (127 um pitch) in to a 4 um pitch with
   Euler S-bends, then turn the bus through 90 degrees as a planar bundle.
2. Build a Mach-Zehnder interferometer whose top arm must detour around a
   keep-out block: route it with A*, and draw the bottom arm with
   route_length_matched so both arms have exactly the same length.
3. Compare with the same MZI whose bottom arm is simply routed straight: the
   unbalanced arms produce interference fringes, the matched ones a flat
   spectrum.
4. DRC pre-check, netlist extraction and simulation of the drawn geometry.

Run:  python examples/routing_showcase.py
Saves: examples/outputs/routing_showcase.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc

BLOCK = np.array([(150.0, 8.0), (250.0, 8.0), (250.0, 60.0), (150.0, 60.0)])


def fibre_bus(top: lay.Cell) -> None:
    """Four grating couplers -> 4 um pitch fan-in -> 90-degree bundle into four waveguide stubs."""
    gcs = []
    for k in range(4):
        gc = top.add_ref(lc.grating_coupler(), name=f"gc{k}")
        gc.connect("o1", lay.Port("p", (-300.0, -200.0 - 127.0 * k), 180.0))
        gcs.append(gc)
    fan, bus = lay.fanout([g.port("o1") for g in gcs], pitch=4.0, dx=500.0, min_radius=30.0)
    for k, r in enumerate(fan):
        top.add_ref(r, name=f"gc_fan{k}")
    stubs = [top.add_ref(lc.straight(20.0), name=f"stub{k}") for k in range(4)]
    for k, s in enumerate(stubs):
        s.connect("o1", lay.Port("p", (300.0 + 4.0 * k, -100.0), 90.0))
    for k, r in enumerate(lay.route_bundle(bus, [s.port("o1") for s in stubs], radius=10.0, min_space=1.0)):
        top.add_ref(r, name=f"bus{k}")


def mzi(top: lay.Cell, *, balanced: bool, prefix: str = "") -> tuple[float, float]:
    """Splitter -> fan-out -> two arms -> fan-in -> combiner; returns the two arm lengths."""
    split = top.add_ref(lc.mmi1x2(), name=f"{prefix}split")
    comb = top.add_ref(lc.mmi1x2(), origin=(404.0, 0.0), rotation=180.0, name=f"{prefix}comb")
    out_fan, out_ports = lay.fanout([split.port("o2"), split.port("o3")], pitch=40.0, dx=60.0, min_radius=20.0)
    in_fan, in_ports = lay.fanout([comb.port("o3"), comb.port("o2")], pitch=40.0, dx=60.0, min_radius=20.0)
    for k, r in enumerate(out_fan + in_fan):
        top.add_ref(r, name=f"{prefix}fan{k}")
    hi = int(np.argmax([q.center[1] for q in out_ports]))
    lo = 1 - hi
    hi_in = int(np.argmax([q.center[1] for q in in_ports]))
    top_arm = lay.route_astar(out_ports[hi], in_ports[hi_in], obstacles=[BLOCK], keepout=3.0, grid=1.0,
                              radius=10.0)
    target = top_arm.model_settings["length"]
    if balanced:
        bottom_arm = lay.route_length_matched(out_ports[lo], in_ports[1 - hi_in], target_length=target,
                                              radius=10.0, side=-1, max_height=60.0)
    else:
        bottom_arm = lay.route_auto(out_ports[lo], in_ports[1 - hi_in], radius=10.0)
    top.add_ref(top_arm, name=f"{prefix}arm_top")
    top.add_ref(bottom_arm, name=f"{prefix}arm_bot")
    return target, bottom_arm.model_settings["length"]


def main() -> None:
    chip = lay.Cell("routing_showcase")
    fibre_bus(chip)
    blk = lay.Cell("keepout_block")
    blk.add_polygon(BLOCK, (11, 0))                      # e.g. a heater pad: an obstacle on another layer
    chip.add_ref(blk, name="block")
    L_top, L_bot = mzi(chip, balanced=True)
    print(f"balanced MZI arms: {L_top:.9f} um and {L_bot:.9f} um (difference {abs(L_top - L_bot):.2e} um)")
    print(px.pdk.demo_pdk().drc_precheck(chip))

    plain = lay.Cell("plain_mzi")
    L_top2, L_bot2 = mzi(plain, balanced=False)
    print(f"plain MZI arms:    {L_top2:.3f} um and {L_bot2:.3f} um (difference {L_top2 - L_bot2:.3f} um)")

    wl = px.linspace(1.50, 1.60, 801)
    spectra = {}
    for name, cell in (("length-matched arms", chip), ("unmatched arms", plain)):
        nl = lay.extract_netlist(cell)
        s = px.circuit.circuit_from_netlist(nl)(wl=wl)
        spectra[name] = px.to_numpy(px.power(s[("split_o1", "comb_o1")]))
    print(f"matched:   T in [{spectra['length-matched arms'].min():.6f}, {spectra['length-matched arms'].max():.6f}]")
    print(f"unmatched: T in [{spectra['unmatched arms'].min():.3f}, {spectra['unmatched arms'].max():.3f}]")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    px.viz.plot_cell(chip, ax=ax0, show_ports=False)
    ax0.set_title("Fan-in + bundle (bottom), A* + length-matched MZI (top)")
    for name, T in spectra.items():
        ax1.plot(px.to_numpy(wl) * 1e3, 10 * np.log10(np.maximum(T, 1e-12)), label=name)
    ax1.set_xlabel("wavelength [nm]")
    ax1.set_ylabel("MZI transmission [dB]")
    ax1.set_ylim(-40, 2)
    ax1.legend()
    ax1.set_title("Extracted-layout simulation")
    fig.tight_layout()
    print("Saved", save(fig, "routing_showcase.png"))


if __name__ == "__main__":
    main()
