"""Example: a polarization-diversity ring filter built from hierarchical, modal netlists.

Light from a fibre with polarization-mode dispersion (PMD) arrives in an
unknown, wavelength-dependent polarization. A polarization splitter-rotator
(PSR) sends TE0 to one arm and the rotated TM0 to the other; each arm holds an
identical add-drop ring filter (one ``filter_arm`` subcircuit, solved once per
call and reused), and the two drop ports are detected together. The left panel
compares the dropped power for random input polarizations with and without the
diversity scheme (a single TE ring sees only the TE share); the right panel
shows the PMD fibre's DGD spectrum.

Run:  python examples/polarization_diversity.py
Saves: examples/outputs/polarization_diversity.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix as px
from photonix.analysis.polarization import dgd, jones_from_sdict
from photonix.circuit import HierarchicalNetlist, ModalNetlist, Netlist, Subcircuit, compile_hierarchical

WL = np.linspace(1.545, 1.555, 801)

# One arm: a bus waveguide into an add-drop ring (single-mode, TE0).
arm = Netlist(name="filter_arm")
arm.add("bus", "straight", length=20.0).add("ring", "add_drop_ring", radius="=radius", coupling1=0.08,
                                              coupling2=0.08, loss_db_cm=2.0)
arm.connect(("bus", "o2"), ("ring", "o1"))
arm.expose("in", ("bus", "o1")).expose("thru", ("ring", "o2")).expose("drop", ("ring", "o3"))
filter_arm = Subcircuit(arm, {"radius": 8.0}, doc="bus + add-drop ring")

top = Netlist(name="diversity_rx")
top.add("te_arm", "filter_arm").add("tm_arm", "filter_arm")
top.expose("te_in", ("te_arm", "in")).expose("tm_in", ("tm_arm", "in"))
top.expose("te_drop", ("te_arm", "drop")).expose("tm_drop", ("tm_arm", "drop"))
arms = compile_hierarchical(HierarchicalNetlist(top, {"filter_arm": filter_arm}))   # one ring solve per call

# Front end: fibre -> PSR (modal netlist, TE0/TM0 terminals).
front = Netlist(name="front_end")
front.add("fiber", "jones_fiber", length_m=20_000.0, pmd_ps_per_sqrt_km=0.5, seed=4)
front.add("psr", "polarization_splitter_rotator", loss_db=0.3, conversion_loss_db=0.2)
front.connect(("fiber", "o2"), ("psr", "o1"))
front.expose("fiber_in", ("fiber", "o1")).expose("te", ("psr", "o2")).expose("tm", ("psr", "o3"))
fe = ModalNetlist(front).compile()(wl=WL)

s_arms = arms(wl=WL)
H_drop = np.asarray(s_arms[("te_in", "te_drop")])           # identical in both arms
rng = np.random.default_rng(0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for k in range(6):
    e_in = rng.normal(size=2) + 1j * rng.normal(size=2)
    e_in /= np.linalg.norm(e_in)
    a_te = sum(e_in[i] * np.asarray(fe[(f"fiber_in:{m}", "te:TE0")]) for i, m in enumerate(("TE0", "TM0")))
    a_tm = sum(e_in[i] * np.asarray(fe[(f"fiber_in:{m}", "tm:TE0")]) for i, m in enumerate(("TE0", "TM0")))
    p_div = (np.abs(a_te) ** 2 + np.abs(a_tm) ** 2) * np.abs(H_drop) ** 2
    p_single = np.abs(a_te) ** 2 * np.abs(H_drop) ** 2
    ax1.plot(WL * 1e3, 10 * np.log10(p_div + 1e-12), color="C0", lw=1.2, label="diversity" if k == 0 else None)
    ax1.plot(WL * 1e3, 10 * np.log10(p_single + 1e-12), color="C3", lw=0.8, alpha=0.7,
             label="TE ring only" if k == 0 else None)
ax1.set(xlabel="wavelength (nm)", ylabel="dropped power (dB)", ylim=(-40, 1),
        title="6 random input polarizations")
ax1.legend()

J = jones_from_sdict(px.components.jones_fiber(wl=WL, length_m=20_000.0, pmd_ps_per_sqrt_km=0.5, seed=4))
tau = dgd(WL, J)
ax2.plot(0.5 * (WL[1:] + WL[:-1]) * 1e3, tau)
ax2.axhline(0.5 * np.sqrt(20.0), color="k", ls="--", lw=0.8, label="ensemble mean DGD")
ax2.set(xlabel="wavelength (nm)", ylabel="DGD (ps)", title="PMD emulator (50 sections)")
ax2.legend()
fig.tight_layout()
print(f"peak drop loss (diversity): {-10 * np.log10(np.max(p_div)):.2f} dB; "
      f"mean DGD over the band: {tau.mean():.2f} ps")
print("saved", save(fig, "polarization_diversity.png"))
