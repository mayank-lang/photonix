"""Passive Lorentz(-Drude) pole fits of a material's permittivity, with Meep export.

Time-domain solvers need a rational (pole) model of ``eps(omega)``. This module
fits, over a wavelength band,

    eps(f) = eps_inf + sum_n  d_n f_n^2 / (f_n^2 - f^2 - i f g_n)  -  f_p^2 / (f^2 + i f g_D)

with ``f = 1 / wl`` in 1/um (Meep's frequency unit for ``a = 1 um``) and the
physics sign of the material protocol (``Im eps >= 0`` is loss; Meep uses the
same ``exp(-i w t)`` convention, so the parameters map one-to-one onto
``LorentzianSusceptibility`` / ``DrudeSusceptibility``).

Algorithm: a non-negative least-squares fit (``scipy.optimize.nnls``) over a
fixed grid of candidate poles picks the dominant terms, then a bounded
nonlinear least-squares refinement (``scipy.optimize.least_squares``) adjusts
every amplitude, resonance and damping. The bounds ``d_n >= 0``, ``g_n >= 0``,
``f_p^2 >= 0``, ``g_D >= 0`` and ``eps_inf >= 1`` make each term a passive,
causal oscillator, so ``Im eps(f) >= 0`` at *every* positive frequency -- not
only at the samples -- and the model obeys Kramers-Kronig by construction.
The reported error is measured on a grid four times denser than the fit
samples.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import MaterialSpec

__all__ = ["LorentzPole", "DrudeTerm", "LorentzFit", "fit_lorentz"]


@dataclass(frozen=True)
class LorentzPole:
    """One Lorentz oscillator ``d f0^2 / (f0^2 - f^2 - i f gamma)`` (frequencies in 1/um)."""

    delta_eps: float
    frequency: float
    gamma: float

    @property
    def wavelength_um(self) -> float:
        """Resonance wavelength ``1 / frequency`` [um]."""
        return 1.0 / self.frequency


@dataclass(frozen=True)
class DrudeTerm:
    """Free-carrier term ``-f_p^2 / (f^2 + i f gamma)`` (frequencies in 1/um)."""

    plasma_frequency: float
    gamma: float


def _eval(f: np.ndarray, eps_inf: float, poles: tuple[LorentzPole, ...], drude: DrudeTerm | None) -> np.ndarray:
    eps = np.full(f.shape, eps_inf, dtype=complex)
    for p in poles:
        eps = eps + p.delta_eps * p.frequency ** 2 / (p.frequency ** 2 - f * f - 1j * f * p.gamma)
    if drude is not None:
        eps = eps - drude.plasma_frequency ** 2 / (f * f + 1j * f * drude.gamma)
    return eps


@dataclass(frozen=True)
class LorentzFit:
    """Result of :func:`fit_lorentz`.

    Attributes
    ----------
    eps_inf
        High-frequency permittivity (``>= 1``).
    poles
        Lorentz oscillators.
    drude
        Optional Drude term.
    wl_range_um
        Band the fit was made over.
    max_abs_error, max_rel_error
        ``max |eps_fit - eps|`` and ``max |eps_fit - eps| / |eps|`` over a dense
        check grid in the band.
    """

    eps_inf: float
    poles: tuple[LorentzPole, ...]
    drude: DrudeTerm | None
    wl_range_um: tuple[float, float]
    max_abs_error: float
    max_rel_error: float

    def epsilon(self, wl: Any, *, convention: str = "physics") -> np.ndarray:
        """Fitted permittivity at ``wl`` [um] (``convention="engineering"`` conjugates).

        Examples
        --------
        >>> from photonix.materials import fit_lorentz, get_material
        >>> fit = fit_lorentz(get_material("SiO2"), 1.2, 1.7, n_poles=2)
        >>> bool(abs(fit.epsilon(1.55) - get_material("SiO2").epsilon(1.55)) < 1e-6)
        True
        """
        f = 1.0 / np.asarray(wl, dtype=float)
        eps = _eval(f, self.eps_inf, self.poles, self.drude)
        if convention == "engineering":
            return np.conj(eps)
        if convention != "physics":
            raise ValueError("convention must be 'physics' or 'engineering'")
        return eps

    def index(self, wl: Any) -> np.ndarray:
        """Fitted complex index ``sqrt(eps)`` (principal branch, ``k >= 0``)."""
        return np.sqrt(self.epsilon(wl))

    def is_passive(self) -> bool:
        """All terms are passive oscillators (non-negative strengths and damping, ``eps_inf >= 1``)."""
        ok = self.eps_inf >= 1.0 and all(p.delta_eps >= 0 and p.gamma >= 0 and p.frequency > 0 for p in self.poles)
        if self.drude is not None:
            ok = ok and self.drude.gamma >= 0
        return bool(ok)

    def meep_parameters(self) -> dict[str, Any]:
        """Plain-data description of the equivalent Meep medium (length unit ``a = 1 um``).

        ``E_susceptibilities`` lists ``{"type", "frequency", "gamma", "sigma"}``
        dicts in Meep's own parametrisation: a Lorentzian contributes
        ``sigma f_n^2 / (f_n^2 - f^2 - i f gamma)`` and a Drude term
        ``sigma f_n^2 / (-f^2 - i f gamma)``.

        Examples
        --------
        >>> from photonix.materials import fit_lorentz, get_material
        >>> p = fit_lorentz(get_material("Au"), 1.3, 1.7, n_poles=1, drude=True).meep_parameters()
        >>> sorted(s["type"] for s in p["E_susceptibilities"])
        ['drude', 'lorentzian']
        """
        sus: list[dict[str, Any]] = [
            {"type": "lorentzian", "frequency": p.frequency, "gamma": p.gamma, "sigma": p.delta_eps}
            for p in self.poles
        ]
        if self.drude is not None:
            sus.append({"type": "drude", "frequency": self.drude.plasma_frequency, "gamma": self.drude.gamma,
                        "sigma": 1.0})
        return {"epsilon": self.eps_inf, "E_susceptibilities": sus,
                "valid_freq_range": (1.0 / self.wl_range_um[1], 1.0 / self.wl_range_um[0])}

    def to_meep_medium(self) -> Any:
        """Build a dispersive :class:`meep.Medium` (requires the optional Meep package)."""
        from photonix.em.meep._guard import require_meep

        mp = require_meep()
        params = self.meep_parameters()
        sus = []
        for s in params["E_susceptibilities"]:
            cls = mp.LorentzianSusceptibility if s["type"] == "lorentzian" else mp.DrudeSusceptibility
            sus.append(cls(frequency=s["frequency"], gamma=s["gamma"], sigma=s["sigma"]))
        kwargs: dict[str, Any] = {"epsilon": params["epsilon"], "E_susceptibilities": sus}
        if hasattr(mp, "FreqRange"):
            lo, hi = params["valid_freq_range"]
            kwargs["valid_freq_range"] = mp.FreqRange(min=lo, max=hi)
        return mp.Medium(**kwargs)


def _target(material: MaterialSpec | Callable[[Any], Any], wl: np.ndarray, temperature: Any,
            axis: str | None) -> np.ndarray:
    if isinstance(material, MaterialSpec):
        return np.asarray(material.epsilon(wl, temperature, axis=axis), dtype=complex)
    if temperature is not None or axis is not None:
        raise ValueError("temperature/axis apply only to a MaterialSpec target")
    return np.asarray(material(wl), dtype=complex)


def fit_lorentz(material: MaterialSpec | Callable[[Any], Any], wl_min: float, wl_max: float, n_poles: int = 3, *,
                temperature: Any = None, axis: str | None = None, drude: bool = False,
                n_samples: int = 120) -> LorentzFit:
    """Fit a passive Lorentz(-Drude) model to ``material``'s permittivity over ``[wl_min, wl_max]`` um.

    Parameters
    ----------
    material
        A :class:`MaterialSpec` or a callable ``eps(wl)`` returning the complex
        permittivity in the physics sign (``Im >= 0`` is loss).
    wl_min, wl_max
        Fit band in um.
    n_poles
        Number of Lorentz oscillators (>= 0).
    temperature, axis
        Passed to :meth:`MaterialSpec.epsilon`.
    drude
        Add a Drude (free-carrier) term -- needed for metals.
    n_samples
        Fit samples, uniformly spaced in frequency.

    Returns
    -------
    LorentzFit
        With ``max_rel_error`` measured on a 4x denser grid; check it against
        your accuracy budget (a :class:`UserWarning` is issued above 1e-2).

    Examples
    --------
    A Sellmeier dielectric is exactly a sum of lossless Lorentz poles:

    >>> from photonix.materials import fit_lorentz, get_material
    >>> fit = fit_lorentz(get_material("Si3N4"), 1.2, 2.0, n_poles=2)
    >>> fit.is_passive(), fit.max_rel_error < 1e-6
    (True, True)
    """
    from scipy.optimize import least_squares, nnls

    if not (0 < wl_min < wl_max):
        raise ValueError("need 0 < wl_min < wl_max")
    if n_poles < 0 or (n_poles == 0 and not drude):
        raise ValueError("need n_poles >= 1, or drude=True")
    f_lo, f_hi = 1.0 / wl_max, 1.0 / wl_min
    f = np.linspace(f_lo, f_hi, int(n_samples))
    eps = _target(material, 1.0 / f, temperature, axis)
    w = 1.0 / np.maximum(np.abs(eps), 1e-12)
    rhs = (eps - 1.0) * w

    # ---- candidate dictionary --------------------------------------------- #
    f_mid = 0.5 * (f_lo + f_hi)
    cand_f = np.geomspace(f_lo / 50.0, f_hi * 50.0, 48)
    cands: list[tuple[str, float, float]] = []
    for fn in cand_f:
        inside = 0.85 * f_lo < fn < 1.15 * f_hi
        for g_rel in ((0.03, 0.1, 0.3, 1.0, 3.0) if inside else (0.0, 0.03, 0.3, 3.0)):
            cands.append(("L", float(fn), float(g_rel * fn)))
    if drude:
        for gd in np.geomspace(1e-4, 3.0, 24) * f_mid:
            cands.append(("D", 0.0, float(gd)))

    def basis(kind: str, fn: float, g: float, ff: np.ndarray) -> np.ndarray:
        if kind == "L":
            return fn * fn / (fn * fn - ff * ff - 1j * ff * g)
        return -1.0 / (ff * ff + 1j * ff * g)

    def stacked(col: np.ndarray) -> np.ndarray:
        c = col * w
        return np.concatenate([c.real, c.imag])

    const = stacked(np.ones_like(f, dtype=complex))
    cols = [stacked(basis(kd, fn, g, f)) for kd, fn, g in cands]
    norms = [max(float(np.linalg.norm(c)), 1e-300) for c in cols]
    b_real = np.concatenate([rhs.real, rhs.imag])
    c_norm = float(np.linalg.norm(const))

    def solve(sel: list[int]) -> tuple[np.ndarray, float]:
        a = np.stack([const / c_norm] + [cols[i] / norms[i] for i in sel], axis=1)
        x, res = nnls(a, b_real, maxiter=100 * a.shape[1])
        return x / np.array([c_norm] + [norms[i] for i in sel]), float(res)

    # Greedy forward selection: add, one at a time, the candidate that most reduces the NNLS residual.
    # (Taking the largest coefficients of a single all-candidate NNLS instead lets neighbouring
    # candidates split one resonance between them and drops weaker but distinct poles.)
    selected: list[int] = []
    pools = ([("D", 1)] if drude else []) + [("L", n_poles)]
    for kind_wanted, count in pools:
        for _ in range(count):
            best, best_res = None, np.inf
            for i, c in enumerate(cands):
                if c[0] != kind_wanted or i in selected:
                    continue
                _, res = solve(selected + [i])
                if res < best_res:
                    best, best_res = i, res
            if best is None:
                break
            selected.append(best)
    coef_sel, _ = solve(selected)
    lor = [i for i in selected if cands[i][0] == "L"]
    dru = [i for i in selected if cands[i][0] == "D"]
    amp = {i: float(coef_sel[1 + j]) for j, i in enumerate(selected)}

    # ---- nonlinear refinement --------------------------------------------- #
    k = len(lor)
    x0 = [max(float(coef_sel[0]), 0.0)]
    x0 += [max(amp[i], 1e-12) for i in lor]
    x0 += [np.log(cands[i][1]) for i in lor]
    x0 += [cands[i][2] / f_mid for i in lor]
    if drude:
        d = dru[0]
        x0 += [max(amp[d], 1e-12), cands[d][2] / f_mid]
    lb = [0.0] + [0.0] * k + [np.log(f_lo * 1e-3)] * k + [0.0] * k + ([0.0, 0.0] if drude else [])
    ub = [np.inf] + [np.inf] * k + [np.log(f_hi * 1e3)] * k + [np.inf] * k + ([np.inf, np.inf] if drude else [])
    x0 = np.clip(np.asarray(x0, dtype=float), np.asarray(lb) + 0.0, np.asarray(ub))

    def unpack(x: np.ndarray) -> tuple[float, tuple[LorentzPole, ...], DrudeTerm | None]:
        poles = tuple(LorentzPole(float(x[1 + j]), float(np.exp(x[1 + k + j])), float(x[1 + 2 * k + j] * f_mid))
                      for j in range(k))
        dt = DrudeTerm(float(np.sqrt(x[1 + 3 * k])), float(x[2 + 3 * k] * f_mid)) if drude else None
        return 1.0 + float(x[0]), poles, dt

    def resid(x: np.ndarray) -> np.ndarray:
        e_inf, poles, dt = unpack(x)
        r = (_eval(f, e_inf, poles, dt) - eps) * w
        return np.concatenate([r.real, r.imag])

    def jac(x: np.ndarray) -> np.ndarray:
        cols_j = [np.ones_like(f, dtype=complex)]
        amps, fr = x[1:1 + k], np.exp(x[1 + k:1 + 2 * k])
        gam = x[1 + 2 * k:1 + 3 * k] * f_mid
        den = [fr[j] ** 2 - f * f - 1j * f * gam[j] for j in range(k)]
        cols_j += [fr[j] ** 2 / den[j] for j in range(k)]
        cols_j += [2.0 * amps[j] * fr[j] ** 2 * (-f * f - 1j * f * gam[j]) / den[j] ** 2 for j in range(k)]
        cols_j += [1j * amps[j] * fr[j] ** 2 * f * f_mid / den[j] ** 2 for j in range(k)]
        if drude:
            dd = f * f + 1j * f * x[2 + 3 * k] * f_mid
            cols_j += [-1.0 / dd, x[1 + 3 * k] * 1j * f * f_mid / dd ** 2]
        jm = np.stack(cols_j, axis=1) * w[:, None]
        return np.vstack([jm.real, jm.imag])

    sol = least_squares(resid, x0, jac=jac, bounds=(lb, ub), x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=1500)
    e_inf, poles, dt = unpack(sol.x)
    order = np.argsort([p.frequency for p in poles])
    poles = tuple(poles[i] for i in order)

    f_chk = np.linspace(f_lo, f_hi, 4 * int(n_samples) + 1)
    eps_chk = _target(material, 1.0 / f_chk, temperature, axis)
    err = np.abs(_eval(f_chk, e_inf, poles, dt) - eps_chk)
    fit = LorentzFit(e_inf, poles, dt, (float(wl_min), float(wl_max)), float(np.max(err)),
                     float(np.max(err / np.abs(eps_chk))))
    if fit.max_rel_error > 1e-2:
        warnings.warn(f"Lorentz fit relative error {fit.max_rel_error:.2e} over {wl_min}-{wl_max} um; "
                      "increase n_poles or narrow the band", UserWarning, stacklevel=2)
    return fit
