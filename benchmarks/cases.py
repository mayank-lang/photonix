"""Benchmark case registry.

Each case is a small, reproducible structure with a single scalar quantity of
interest (n_eff, transmission, loss). A case knows how to compute its photonix
value; reference values (literature, or external solvers) live in
``references.json`` and are matched by ``key``. Adding a case = appending one
``Case`` here and (optionally) a reference entry in the JSON.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class Case:
    key: str
    description: str
    quantity: str          # human label for the number, e.g. "n_eff" or "|T00|^2"
    compute: Callable[[], float]


def _soi_te0() -> float:
    import photonix.em as em
    return em.n_eff_fullvector(width=0.5, thickness=0.22, resolution=40)


def _soi_tm0() -> float:
    import photonix.em as em
    r = em.solve_modes_fullvector(width=0.5, thickness=0.22, resolution=40, num_modes=2)
    return float(np.real(r.n_eff[1]))


def _bend_loss_r1() -> float:
    import photonix.em as em
    return em.bend_loss_fullvector(
        bend_radius=1.0, resolution=28, inner=0.1
    ).loss_db_per_90deg


def _step_T(pol: str) -> float:
    import photonix.em as em
    from photonix.em.eme import Section

    y = np.arange(-1.5, 1.5 + 1e-9, 0.04)
    col = lambda w: np.where(np.abs(y) < w / 2, 3.4757**2, 1.444**2)  # noqa: E731
    r = em.eme.eme_smatrix(
        [Section(col(0.45), 1.2), Section(col(0.55), 1.2)], 0.04, 1.55, 6, pol
    )
    return abs(r.Tf[0, 0]) ** 2


CASES: list[Case] = [
    Case("soi_strip_te0_neff",
         "500x220 nm Si strip (oxide clad), full-vector fundamental TE0", "n_eff", _soi_te0),
    Case("soi_strip_tm0_neff",
         "500x220 nm Si strip, full-vector TM0", "n_eff", _soi_tm0),
    Case("soi_bend_loss_r1_db90",
         "500x220 nm Si strip, 90-deg bend radiation loss at R=1.0 um", "dB/90deg",
         _bend_loss_r1),
    Case("width_step_te_T",
         "TE transmission of a 0.45->0.55 um width step (EME)", "|T00|^2",
         lambda: _step_T("te")),
    Case("width_step_tm_T",
         "TM transmission of a 0.45->0.55 um width step (EME)", "|T00|^2",
         lambda: _step_T("tm")),
]
