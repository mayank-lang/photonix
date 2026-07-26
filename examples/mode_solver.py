"""Example: solve and plot the fundamental mode of a silicon strip waveguide.

Uses the scalar FDE solver for the field profile, then reports the full-vector
index alongside it: for a high-contrast SOI strip the scalar model overestimates
the true quasi-TE index by ~7%, so the vectorial solver is the one to quote.

Run:  python examples/mode_solver.py
Saves: examples/outputs/mode_profile.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _output import save

import photonix as px
import photonix.em as em


def main() -> None:
    n_core, n_clad = float(em.silicon(1.55)), float(em.silica(1.55))
    res = em.solve_modes(
        wl=1.55, width=0.5, thickness=0.22,
        n_core=n_core, n_clad=n_clad, num_modes=1, resolution=50,
    )
    ng = em.group_index(wl=1.55, width=0.5, thickness=0.22, resolution=25)
    te0 = em.n_eff_fullvector(wl=1.55, width=0.5, thickness=0.22,
                              n_core=n_core, n_clad=n_clad, resolution=30)
    print(f"scalar      n_eff = {res.neff0:.4f}")
    print(f"full-vector n_eff = {te0:.4f}   (TE0, the physical value)")
    print(f"group index  n_g  = {ng:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    px.viz.plot_mode(res.fields[0], res.x, res.y, ax=ax)
    ax.set_title(f"Si strip 500×220 nm — TE0 (full-vector n_eff = {te0:.3f})")
    fig.tight_layout()
    print("Saved", save(fig, "mode_profile.png"))


if __name__ == "__main__":
    main()
