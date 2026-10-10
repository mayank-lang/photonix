"""Example: assemble a test die, write it losslessly, read it back and simulate the read-back layout.

1. Draw a die frame (dicing lane, keep-out band, corner marks, die ID, one
   edge-coupled facet).
2. Place a 6-channel, 127 um-pitch grating-coupler fibre array with its
   alignment loopback, and route two MZIs onto the inner channels.
3. Add an edge-coupled test waveguide on the facet, bin-pack a set of ring
   test structures into a free area, and label the MZI for automated test.
4. Check the floorplan (keep-out, overlaps, facet alignment).
5. Write GDSII plus the JSON sidecar, read it back, check that the cell tree
   is unchanged, and simulate the *read-back* layout: the loopback and MZI
   spectra come from the drawn lengths stored in the file.

Run:  python examples/die_assembly.py
Saves: examples/outputs/die_assembly.png
"""
from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save
from matplotlib.patches import Polygon as MplPolygon

import photonix as px
import photonix.layout as lay
from photonix.layout import components as lc
from photonix.layout.gds import _fingerprint
from photonix.layout.library import add_test_labels, die_frame, fiber_array


def build() -> tuple[lay.LabeledCell, lay.DieCell]:
    die = die_frame((2500.0, 1500.0), dicing_lane=50.0, keepout=50.0, die_id="PX-01", facet_sides=("E",))
    top = lay.Cell("die_PX01")
    top.add_ref(die, name="frame")

    fa = top.add_ref(fiber_array(6, radius=10.0), origin=(300.0, 200.0), name="fa")
    for k, (dl, (a, b)) in enumerate(((20.0, ("o1", "o2")), (40.0, ("o3", "o4")))):
        p_in, p_out = fa.port(a), fa.port(b)
        mzi = top.add_ref(lc.mzi(delta_length=dl), name=f"mzi{k}")
        mzi.origin = (p_in.center[0] + 20.0, 450.0 + 120.0 * k)
        top.add_ref(lay.route_manhattan(p_in, mzi.port("o1"), radius=10.0), name=f"r_in{k}")
        top.add_ref(lay.route_manhattan(mzi.port("o2"), p_out, radius=10.0), name=f"r_out{k}")
        top.add_ref(lay.text(f"MZI dL={dl:g}", size=12.0, min_stroke=0.5), origin=(mzi.origin[0], mzi.origin[1] - 40.0),
                    name=f"txt{k}")

    # grating coupler -> straight -> edge coupler whose tip sits on the east facet
    gc = top.add_ref(lc.grating_coupler(), origin=(1900.0, 1000.0), name="gc_ec")
    ec = top.add_ref(lc.edge_coupler(200.0), origin=(2300.0, 1000.0), name="ec")
    top.add_ref(lay.route_manhattan(gc.port("o1"), ec.port("o1")), name="r_ec")

    rings = [lc.ring_single(radius=r, gap=0.2) for r in (5.0, 8.0, 12.0, 16.0, 20.0, 25.0)]
    (block,) = lay.pack(rings, max_size=(400.0, 250.0), spacing=25.0, name="rings")
    top.add_ref(block, origin=(1300.0, 300.0), name="rings")

    # automated-test label on the MZI's input coupler (channel 1), output on channel 2
    fibres = [top.get_ports()[f"fa/gc{k}/o2"] for k in (1, 2)]
    return add_test_labels(top, gc_refs=fibres, device="MZI20", user="photonix"), die


def main() -> None:
    chip, die = build()
    issues = lay.check_floorplan(chip, die)
    print("floorplan issues:", [(i.kind, i.instances) for i in issues] or "none")

    path = os.path.join(tempfile.mkdtemp(), "die_PX01.gds")
    art = lay.write_layout(chip, path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes, sha256 {art.sha256[:12]}...), "
          f"{len(art.structures)} structures, max snap {art.snap_error * 1e3:.2f} nm")
    back = lay.read_layout(path)
    same = _fingerprint(back, {}) == _fingerprint(lay.snap_to_grid(chip), {})
    print("read-back cell tree identical to the written one:", same)
    print("labels:", [lb.text for lb in back.get_labels()])

    wl = px.linspace(1.53, 1.57, 801)
    s = px.circuit.circuit_from_netlist(lay.extract_netlist(back))(wl=wl)

    def db(a: str, b: str) -> np.ndarray:
        return 10.0 * np.log10(px.to_numpy(px.power(s[(a, b)])))

    loop, mzi20, mzi40 = db("fa.gc0_o2", "fa.gc5_o2"), db("fa.gc1_o2", "fa.gc2_o2"), db("fa.gc3_o2", "fa.gc4_o2")
    print(f"loopback transmission at 1550 nm: {np.interp(1.55, px.to_numpy(wl), loop):.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.5, 1.2, 1]})
    colours = {(1, 0): "tab:blue", (99, 0): "0.55", (98, 0): "0.9", (99, 1): "0.3", (66, 0): "tab:green"}
    for ax, (xlim, ylim), title in ((axes[0], ((-20, 2520), (-20, 1520)), "Read-back die PX-01"),
                                    (axes[1], ((230, 1000), (130, 640)), "Fibre array, loopback and MZIs")):
        for pts, layer in back.get_polygons():
            colour = colours.get(layer, "tab:orange")
            ax.add_patch(MplPolygon(pts, closed=True, facecolor=colour, edgecolor=colour, linewidth=0.6))
        for lbl in back.get_labels():
            ax.plot(*lbl.position, "r*", ms=8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    axes[1].annotate("opt_in label", back.get_labels()[0].position, color="r", fontsize=8,
                     xytext=(8, -14), textcoords="offset points")
    lam = px.to_numpy(wl) * 1e3
    axes[2].plot(lam, loop, label="alignment loopback (gc0 - gc5)")
    axes[2].plot(lam, mzi20, label="MZI ΔL = 20 µm (gc1 - gc2)")
    axes[2].plot(lam, mzi40, label="MZI ΔL = 40 µm (gc3 - gc4)")
    axes[2].set_xlabel("wavelength (nm)")
    axes[2].set_ylabel("fibre-to-fibre transmission (dB)")
    axes[2].set_ylim(-45, 0)
    axes[2].legend(fontsize=8, loc="lower center")
    axes[2].set_title("Simulated from the GDS + sidecar")
    fig.tight_layout()
    print("saved", save(fig, "die_assembly.png"))


if __name__ == "__main__":
    main()
