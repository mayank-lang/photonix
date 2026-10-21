"""Example: verify a small photonic layout before the foundry run, then dummy-fill it.

1. Draw an MZI, a ring with a heater, and two deliberate mistakes: a straight
   left 5 nm short of its neighbour, and a stub drawn across a waveguide.
2. LVS-lite: geometric connectivity (near-opens, shorts), parameters
   re-extracted from the polygons, and comparison with a schematic netlist.
3. A DRC rule deck (width, space, heater-to-waveguide separation, metal
   enclosure of the via, acute angles) with one waived violation, written
   as a KLayout ``.lyrdb`` marker file and as a KLayout DRC script.
4. Density of the silicon layer in 50 um windows, then dummy fill to 25 %
   with a keep-out set by the evanescent decay of the strip mode.

Run:  python examples/layout_verification.py
Saves: examples/outputs/layout_verification.png
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

from photonix.circuit import Netlist
from photonix.layout import Cell, extract_netlist, fill
from photonix.layout import components as lc
from photonix.verification import (
    Rule,
    RuleDeck,
    Waiver,
    density_map,
    evanescent_keepout,
    run_lvs_lite,
    write_lyrdb,
)
from photonix.xsection import n_eff, strip
from photonix.xsection.library import N_SIO2

WG, METAL, VIA = (1, 0), (11, 0), (12, 0)

# -- 1. layout ----------------------------------------------------------------- #
chip = Cell("chip")
mzi = chip.add_ref(lc.mzi(delta_length=30.0), origin=(10.0, 60.0), name="mzi")
ring = chip.add_ref(lc.ring_single(radius=10.0, gap=0.2), origin=(80.0, 20.0), name="ring")
chip.add_ref(lc.straight(20.0), name="feed").connect("o2", ring.port("o1"))
heater = Cell("heater")
heater.add_polygon(np.array([(-12, 30.5), (12, 30.5), (12, 33.0), (-12, 33.0)]), METAL)   # over the ring
heater.add_polygon(np.array([(-11.8, 31.0), (-10.4, 31.0), (-10.4, 32.4), (-11.8, 32.4)]), VIA)
chip.add_ref(heater, origin=(80.0, 0.0), name="heater")
# mistake 1: the output straight is 5 nm short of the ring's through port
chip.add_ref(lc.straight(20.0), origin=(ring.port("o2").center[0] + 0.005, 20.0), name="out")
# mistake 2: a test stub drawn straight across the MZI's lower arm
chip.add_ref(lc.straight(8.0), origin=(39.0, 46.0), rotation=90.0, name="stub")

# -- 2. LVS-lite ----------------------------------------------------------------- #
nl = extract_netlist(chip, warn=False)
schematic = Netlist(instances=dict(nl.instances), connections=dict(nl.connections), ports=dict(nl.ports),
                    settings={k: dict(v) for k, v in nl.settings.items()})
schematic.connections[("ring", "o2")] = ("out", "o1")         # what the designer meant
schematic.settings["ring"]["gap"] = 0.25                      # and a stale schematic value
schematic.ports = {k: v for k, v in schematic.ports.items() if v not in (("ring", "o2"), ("out", "o1"))}
lvs = run_lvs_lite(chip, schematic)
print(lvs)

# -- 3. DRC ---------------------------------------------------------------------- #
deck = RuleDeck([
    Rule("WG.W", "width", WG, value=0.15, description="min waveguide width"),
    Rule("WG.S", "space", WG, value=0.15, description="min waveguide space"),
    Rule("M.SEP.WG", "separation", METAL, WG, 1.0, description="heater to waveguide, except over the ring"),
    Rule("M.EN.V", "enclosure", VIA, METAL, 0.3, description="metal over via"),
    Rule("WG.ANG", "min_angle", WG, value=30.0, description="no acute silicon tips"),
])
ring_zone = np.array([(65, 15), (95, 15), (95, 35), (65, 35)])
drc = deck.run(chip, waivers=[Waiver("M.SEP.WG", ring_zone, "heater over the ring by design")])
print(drc)
work = tempfile.mkdtemp()
print("markers:", write_lyrdb(drc, os.path.join(work, "chip.lyrdb"), top="chip"))
print("KLayout deck:", deck.to_klayout_drc(os.path.join(work, "chip.drc")).count("\n") + 1, "lines")

# -- 4. density and fill ------------------------------------------------------- #
neff = n_eff(strip(0.5, 0.22), resolution=20)
keep = float(evanescent_keepout(neff=neff, n_clad=N_SIO2, target=1e-4))
keep = max(keep, 1.0)   # and never closer than 1 um (lithography)
region = np.array([(0, 0), (150, 0), (150, 100), (0, 100)])
before = density_map(chip, WG, window=50.0, region=region)
res = fill(chip, layer=WG, region=region, tile_size=2.0, tile_space=1.0, keepout={WG: keep, METAL: 2.0},
           target_density=0.25, window=50.0)
filled = Cell("filled")
filled.add_ref(chip, name="chip")
filled.add_ref(res.cell, name="fill")
print(f"n_eff = {neff:.4f}, keep-out = {keep:.3f} um, {res.n_tiles} tiles; window density "
      f"{before.min:.3f}-{before.max:.3f} -> {res.density_after.min:.3f}-{res.density_after.max:.3f}")
print("windows below target:", len(res.residual))
print("filled layout still LVS-equivalent:", run_lvs_lite(filled).summary() == run_lvs_lite(chip).summary())

# -- figure ---------------------------------------------------------------------- #
fig, (ax, axd) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
colours = {WG: "#1f77b4", METAL: "#d4a017", VIA: "#555555"}
for pts, lay in chip.get_polygons():
    ax.add_patch(MplPolygon(pts, closed=True, fc=colours.get(lay, "#999999"), ec="none", alpha=0.9))
for x, y in res.sites:
    ax.add_patch(MplPolygon([(x, y), (x + 2, y), (x + 2, y + 2), (x, y + 2)], fc="#9ecae1", ec="none"))
for issue in lvs.issues:
    if issue.location != (0.0, 0.0):
        ax.plot(*issue.location, "rx", ms=10, mew=2)
        ax.annotate(issue.kind, issue.location, xytext=(4, 4), textcoords="offset points", color="r", fontsize=8)
for v in drc.violations:
    ax.add_patch(MplPolygon(v.polygon, closed=True, fc="none", ec="m", lw=1.5))
ax.set_xlim(0, 150)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.set_title("LVS-lite issues (red), DRC markers (magenta), fill (light blue)")
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
im = axd.imshow(res.density_after.density, origin="lower", vmin=0, vmax=0.5, cmap="viridis",
                extent=(0, 150, 0, 100))
for (iy, ix), d in np.ndenumerate(before.density):
    axd.text(before.x[ix], before.y[iy], f"{d:.2f} -> {res.density_after.density[iy, ix]:.2f}",
             ha="center", va="center", color="w", fontsize=9)
axd.set_title("Si density per 50 um window (target 0.25)")
fig.colorbar(im, ax=axd, shrink=0.8)
fig.tight_layout()
print("saved", save(fig, "layout_verification.png"))
