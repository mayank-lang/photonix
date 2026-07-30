"""Showcase: the full-vector EM stack on a 500x220 nm silicon strip.

Exercises the vectorial solvers end-to-end and renders a 2x2 figure:

  (a) full-vector fundamental mode (dominant Ex) with n_eff + polarization;
  (b) the accuracy ladder -- scalar -> semivector -> full-vector -> literature;
  (c) bend (radiation) loss vs radius from the conformal-map + PML solver;
  (d) EME<->FDFD cross-solver agreement on a width step, for TE and TM.

Run:   python examples/vector_em_showcase.py
Saves: examples/outputs/vector_em_showcase.png
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _output import save

import photonix.em as em
from photonix.em.eme import Section, eme_smatrix
from photonix.em.fdfd import waveguide_sparams

warnings.filterwarnings("ignore")
WL, NCO, NCL = 1.55, 3.4757, 1.444


def _step_T(y, dy, w1, w2, pol):
    """Transmission of a width step from EME and from FDFD (same structure)."""
    col = lambda w: np.where(np.abs(y) < w / 2, NCO**2, NCL**2)  # noqa: E731
    nx = int(3.0 / dy)
    eps = np.empty((len(y), nx))
    for ix in range(nx):
        eps[:, ix] = col(w1) if ix / nx < 0.5 else col(w2)
    s = waveguide_sparams(
        eps, dx=dy, dy=dy, wl=WL, src_col=15,
        in_mon_col=int(0.3 * nx), out_mon_col=int(0.7 * nx),
        in_eps_col=15, out_eps_col=nx - 15, npml=12, polarization=pol,
    )
    t_fdfd = abs(s[("o1", "o2")]) ** 2
    r = eme_smatrix([Section(col(w1), 1.2), Section(col(w2), 1.2)], dy, WL, 6, pol)
    return abs(r.Tf[0, 0]) ** 2, t_fdfd


def main() -> None:
    # (a) full-vector fundamental + (b) ladder ---------------------------------
    fv = em.solve_modes_fullvector(width=0.5, thickness=0.22, resolution=40, num_modes=2)
    te0, tm0 = fv.neff0, float(np.real(fv.n_eff[1]))
    scalar = em.solve_modes(width=0.5, thickness=0.22, resolution=40).neff0
    semi = em.n_eff_vector(width=0.5, thickness=0.22, resolution=40, polarization="te")
    print(f"full-vector TE0={te0:.4f} (te_frac={fv.te_fraction[0]:.3f})  TM0={tm0:.4f}")

    # (c) bend loss vs radius (resolved regime; gentler bends fall below the PML
    #     noise floor for this confined strip, so we show R <= 1.5 um)
    radii = [1.0, 1.2, 1.35, 1.5]
    loss = [em.bend_loss_fullvector(bend_radius=R, resolution=30, inner=0.1).loss_db_per_90deg
            for R in radii]
    print("bend loss dB/90:", [f"{r}:{v:.2e}" for r, v in zip(radii, loss, strict=False)])

    # (d) cross-solver agreement ----------------------------------------------
    y = np.arange(-1.5, 1.5 + 1e-9, 0.04)
    te_eme, te_fdfd = _step_T(y, 0.04, 0.45, 0.55, "te")
    tm_eme, tm_fdfd = _step_T(y, 0.04, 0.45, 0.55, "tm")
    print(f"TE step  EME={te_eme:.4f} FDFD={te_fdfd:.4f}   TM step  EME={tm_eme:.4f} FDFD={tm_fdfd:.4f}")

    # ---- render --------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) mode
    f = fv.fields[0]
    ext = [fv.x.min(), fv.x.max(), fv.y.min(), fv.y.max()]
    vmax = np.abs(f).max()
    ax[0, 0].imshow(f, extent=ext, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    aspect="auto")
    ax[0, 0].add_patch(plt.Rectangle((-0.25, -0.11), 0.5, 0.22, ec="k", fc="none", lw=1.2))
    ax[0, 0].set(title=f"(a) full-vector TE0  $n_{{eff}}$={te0:.3f}, TE frac={fv.te_fraction[0]:.2f}",
                 xlabel="x (µm)", ylabel="y (µm)", xlim=(-1.2, 1.2), ylim=(-0.8, 0.8))

    # (b) accuracy ladder
    names = ["scalar", "semivector", "full-vector", "literature"]
    vals = [scalar, semi, te0, 2.44]
    colors = ["#bbb", "#88b", "#2a6", "k"]
    ax[0, 1].bar(names, vals, color=colors)
    ax[0, 1].set_ylim(2.40, max(vals) + 0.02)
    for i, v in enumerate(vals):
        ax[0, 1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax[0, 1].set(title="(b) TE0 accuracy ladder (500×220 Si)", ylabel="$n_{eff}$")
    ax[0, 1].axhline(2.44, ls="--", c="k", lw=0.8, alpha=0.5)

    # (c) bend loss
    ax[1, 0].semilogy(radii, loss, "o-", color="#c33")
    ax[1, 0].set(title="(c) bend radiation loss (full-vector + PML)",
                 xlabel="bend radius (µm)", ylabel="loss (dB / 90°)")
    ax[1, 0].grid(True, which="both", alpha=0.3)

    # (d) cross-solver agreement
    xb = np.arange(2)
    ax[1, 1].bar(xb - 0.18, [te_eme, tm_eme], 0.36, label="EME (modal)", color="#48a")
    ax[1, 1].bar(xb + 0.18, [te_fdfd, tm_fdfd], 0.36, label="FDFD (full-wave)", color="#e93")
    ax[1, 1].set_xticks(xb, ["TE step", "TM step"])
    ax[1, 1].set_ylim(0.95, 1.01)
    ax[1, 1].set(title="(d) EME ↔ FDFD transmission agreement", ylabel="$|T_{00}|^2$")
    ax[1, 1].legend(fontsize=8)
    for xi, (a, b) in zip(xb, [(te_eme, te_fdfd), (tm_eme, tm_fdfd)], strict=False):
        ax[1, 1].text(xi, 0.952, f"Δ={abs(a-b):.4f}", ha="center", fontsize=8)

    fig.suptitle("photonix full-vector EM stack — 500×220 nm SOI strip @ 1.55 µm",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    print("Saved", save(fig, "vector_em_showcase.png"))


if __name__ == "__main__":
    main()
