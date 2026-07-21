"""Example: solve and plot the fundamental mode of a silicon strip waveguide.

Run:  python examples/mode_solver.py
Saves: mode_profile.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import photonix as px
import photonix.modes as modes


def main() -> None:
    res = modes.solve_modes(
        wl=1.55, width=0.5, thickness=0.22,
        n_core=float(modes.silicon(1.55)), n_clad=float(modes.silica(1.55)),
        num_modes=1, resolution=50,
    )
    ng = modes.group_index(wl=1.55, width=0.5, thickness=0.22, resolution=40)
    print(f"fundamental n_eff = {res.neff0:.4f}")
    print(f"group index  n_g  = {ng:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    px.viz.plot_mode(res.fields[0], res.x, res.y, ax=ax)
    ax.set_title(f"Si strip 500×220 nm — TE0 (n_eff = {res.neff0:.3f})")
    fig.tight_layout()
    fig.savefig("mode_profile.png", dpi=130)
    print("Saved mode_profile.png")


if __name__ == "__main__":
    main()
