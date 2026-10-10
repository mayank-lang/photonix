"""Example: one polygonal cross-section feeding optical and thermal coefficients.

* a trapezoidal (80 deg) directional-coupler section rasterized with Kottke tensor
  smoothing, and its even/odd supermode splitting;
* the same kind of geometry object giving a thermal-conductivity map for a heater
  over a waveguide (illustrative bulk property values, not process data);
* the rotation-invariance check that motivates tensor smoothing: the n_eff of a
  45-degree-rotated core must equal that of the axis-aligned one.

Run:  python examples/xsection_multiphysics.py
Saves: examples/outputs/xsection_multiphysics.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix.xsection as xs


def _aligned_grid(r: int) -> xs.TensorGrid:
    h = 1.0 / r
    return xs.TensorGrid(np.arange(-round(1.5 * r), round(1.5 * r) + 1) * h,
                         np.arange(round(-1.2 * r), round(1.4 * r) + 1) * h)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    # (a) Trapezoidal coupler: tensor-smoothed eps_xx and the supermode splitting.
    cpl = xs.coupler(0.5, 0.2, 0.22, sidewall_deg=80.0)
    grid = cpl.uniform_grid(resolution=40, margin=0.6)
    exx = cpl.eps(grid, smoothing="tensor")[0]
    m = xs.solve_modes(cpl, resolution=40, margin=1.0, num_modes=3)
    te = np.flatnonzero(m.te_fraction > 0.5)[:2]
    dn = float(np.real(m.n_eff[te[0]] - m.n_eff[te[1]]))
    print(f"80-deg coupler, gap 0.2 um: n_even = {m.n_eff[te[0]].real:.5f}, dn = {dn:.5f}, "
          f"L_c = {1.55 / (2 * dn):.1f} um (40 px/um)")
    ext = (*grid.bbox[:2], *grid.bbox[2:])
    im = axes[0, 0].imshow(exx, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=axes[0, 0], label=r"$\varepsilon_{xx}$ (tensor smoothed)")
    axes[0, 0].set_title(f"80° coupler, gap 0.2 µm: Δn = {dn:.4f}")

    # (b) Heater over waveguide: thermal conductivity, laminate (tensor) average.
    props = {"core": {"k_th": 148.0}, "substrate": {"k_th": 148.0}, "box": {"k_th": 1.38},
             "clad": {"k_th": 1.38}, "heater": {"k_th": 25.0}}            # illustrative bulk values
    htr = xs.heater_over_waveguide(heater_offset=0.8, metal=2.0 - 4.0j, properties=props)
    g2 = htr.uniform_grid(resolution=25, margin=1.0)
    k = htr.coefficient(g2, "k_th", average="tensor")
    im = axes[0, 1].imshow(np.log10(k[0]), origin="lower", extent=g2.bbox, cmap="magma", aspect="equal")
    fig.colorbar(im, ax=axes[0, 1], label=r"$\log_{10} k_{xx}$ (W/m/K)")
    axes[0, 1].set_title("Heater over waveguide: thermal map")

    # (c) TE0 of a rib waveguide on a BOX.
    rib = xs.rib_waveguide(0.5, 0.22, 0.09, box=1.444)
    mr = xs.solve_modes(rib, resolution=30, margin=1.0)
    axes[1, 0].imshow(np.abs(mr.ex[0]), origin="lower", extent=mr.grid.bbox, cmap="inferno", aspect="equal")
    axes[1, 0].set_title(f"Rib 500/220/90 nm: TE0 |Ex|, n_eff = {mr.neff0:.4f}")

    # (d) Rotation invariance: error of a 45-degree-rotated core against the aligned limit.
    base = xs.XSection([xs.rect(-0.25, 0.25, 0.0, 0.2, 3.4757)], background=1.444)
    n40, n80 = (xs.n_eff(base, grid=_aligned_grid(r), smoothing="arithmetic") for r in (40, 80))
    limit = (4 * n80 - n40) / 3
    rot = xs.XSection([xs.rect(-0.25, 0.25, -0.1, 0.1, 3.4757).rotated(45.0)], background=1.444)
    ress = [20, 30, 40, 60]
    for sm, kw, style in (("arithmetic", {}, "o-"), ("tensor", {"offdiagonal": False}, "s--"),
                          ("tensor", {}, "^-")):
        err = [abs(xs.n_eff(rot, resolution=r, margin=1.0, smoothing=sm, **kw) - limit) for r in ress]
        label = sm + (" (diagonal only)" if kw else "")
        axes[1, 1].loglog(ress, err, style, label=label)
        print(f"45-deg core, {label:26s}: |n_eff - limit| =", " ".join(f"{e:.1e}" for e in err))
    axes[1, 1].set_xlabel("resolution (px/µm)")
    axes[1, 1].set_ylabel("|n_eff(45°) - n_eff(aligned limit)|")
    axes[1, 1].set_title("Rotation invariance needs tensor smoothing")
    axes[1, 1].legend()
    for ax in axes.flat[:3]:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    fig.tight_layout()
    print("Saved", save(fig, "xsection_multiphysics.png"))


if __name__ == "__main__":
    main()
