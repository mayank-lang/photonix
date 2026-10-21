"""Multimode waveguide models with modal terminals (``"o1:TE0"``, ``"o1:TE1"``, ...).

* :func:`multimode_straight` -- one straight per guided mode (each with its own
  ``neff``/``ng``/loss), optionally with uniform two-mode coupling.
* :func:`asymmetric_mode_converter` -- phase-matched asymmetric directional
  coupler between a single-mode bus (``TE0``) and a multimode guide
  (``TE1``), from coupled-mode theory.
* :func:`mode_mux` -- ``N`` single-mode inputs onto ``N`` modes of one
  multimode output (behavioural, with insertion loss and crosstalk).

Coupled-mode theory
-------------------
Two co-propagating modes with propagation constants ``beta_1, beta_2`` and
coupling ``kappa`` (Yariv, IEEE JQE 9(9):919, 1973; Huang, JOSA A 11(3):963,
1994) give, with ``beta_bar = (beta_1 + beta_2)/2``,
``delta = (beta_1 - beta_2)/2`` and ``S = sqrt(kappa^2 + delta^2)``::

    T11 = e^{-j beta_bar L} (cos SL - j (delta/S) sin SL)
    T22 = e^{-j beta_bar L} (cos SL + j (delta/S) sin SL)
    T12 = T21 = -j (kappa/S) sin(SL) e^{-j beta_bar L}

so the converted power is ``kappa^2/(kappa^2 + delta^2) sin^2(S L)``: full
conversion at ``L = pi/(2 kappa)`` when phase matched. The matrix is unitary
for real ``kappa``, ``delta`` and symmetric, so the device is lossless and
reciprocal. Propagation loss enters through complex ``beta = beta - j alpha``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from photonix.core.backend import xp
from photonix.core.types import SDict
from photonix.core.units import db_per_cm_to_alpha_um

from .waveguide import neff_linear

__all__ = ["multimode_straight", "asymmetric_mode_converter", "mode_mux", "coupled_mode_transfer",
           "DEFAULT_MULTIMODE_MODES"]

#: Default modes of :func:`multimode_straight`: a 220 nm x ~1 µm silicon strip (TE0, TE1), illustrative values.
DEFAULT_MULTIMODE_MODES: dict[str, dict[str, float]] = {
    "TE0": {"neff": 2.6, "ng": 4.0, "loss_db_cm": 0.0},
    "TE1": {"neff": 2.2, "ng": 4.3, "loss_db_cm": 0.0},
}


def _beta(wl, neff: float, ng: float, wl0: float, loss_db_cm: float):
    n = neff_linear(wl, neff, ng, wl0)
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    return 2.0 * xp.pi * n / wl - 1j * alpha


def coupled_mode_transfer(beta1, beta2, kappa, length):
    """Two-mode coupled-mode transfer ``(T11, T12, T21, T22)`` (module docs).

    ``beta1``/``beta2`` may be complex (loss), ``kappa`` real (per µm).

    Examples
    --------
    >>> import numpy as np
    >>> t11, t12, t21, t22 = coupled_mode_transfer(10.0, 10.0, 0.01, np.pi / (2 * 0.01))
    >>> round(float(abs(t21) ** 2), 12), round(float(abs(t11) ** 2), 12)
    (1.0, 0.0)
    """
    beta1 = xp.asarray(beta1, dtype=complex)
    beta2 = xp.asarray(beta2, dtype=complex)
    bbar = 0.5 * (beta1 + beta2)
    delta = 0.5 * (beta1 - beta2)
    S = xp.sqrt(kappa * kappa + delta * delta)
    tiny = xp.abs(S) < 1e-15
    S_safe = xp.where(tiny, 1.0, S)
    sinc_l = xp.where(tiny, length, xp.sin(S_safe * length) / S_safe)    # sin(S L)/S, -> L as S -> 0
    cos_l = xp.cos(S * length)
    ph = xp.exp(-1j * bbar * length)
    t11 = ph * (cos_l - 1j * delta * sinc_l)
    t22 = ph * (cos_l + 1j * delta * sinc_l)
    t12 = -1j * kappa * sinc_l * ph
    return t11, t12, t12, t22


def multimode_straight(*, wl=1.55, length: float = 10.0, modes: Mapping[str, Mapping[str, float]] | None = None,
                       coupling_per_um: float = 0.0, wl0: float = 1.55) -> SDict:
    """Straight multimode waveguide: one propagation constant per mode.

    Parameters
    ----------
    modes
        ``mode label -> {"neff", "ng", "loss_db_cm"}`` at ``wl0`` (defaults:
        :data:`DEFAULT_MULTIMODE_MODES`).
    coupling_per_um
        Uniform coupling ``kappa`` between the two modes (needs exactly two
        modes), e.g. from sidewall tilt or a perturbation; ``0`` keeps the
        modes independent (block-diagonal S-matrix).

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> s = multimode_straight(wl=1.55, length=20.0)
    >>> te1 = px.components.straight(wl=1.55, length=20.0, neff=2.2, ng=4.3)[("o1", "o2")]
    >>> bool(np.isclose(complex(s[("o1:TE1", "o2:TE1")]), complex(te1))), ("o1:TE0", "o2:TE1") in s
    (True, False)
    """
    specs = {k: dict(v) for k, v in (modes or DEFAULT_MULTIMODE_MODES).items()}
    if not specs:
        raise ValueError("modes must not be empty")
    wl = xp.asarray(wl)
    betas = {m: _beta(wl, float(p.get("neff", 2.4)), float(p.get("ng", 4.2)), wl0, float(p.get("loss_db_cm", 0.0)))
             for m, p in specs.items()}
    out: SDict = {}
    if isinstance(coupling_per_um, (int, float)) and coupling_per_um == 0.0:
        for m, b in betas.items():
            t = xp.exp(-1j * b * length)
            out[(f"o1:{m}", f"o2:{m}")] = t
            out[(f"o2:{m}", f"o1:{m}")] = t
        return out
    if len(betas) != 2:
        raise ValueError("coupling_per_um needs exactly two modes")
    (m1, b1), (m2, b2) = betas.items()
    t11, t12, t21, t22 = coupled_mode_transfer(b1, b2, coupling_per_um, length)
    for (a, b), v in {(m1, m1): t11, (m2, m1): t12, (m1, m2): t21, (m2, m2): t22}.items():
        out[(f"o1:{a}", f"o2:{b}")] = v
        out[(f"o2:{b}", f"o1:{a}")] = v
    return out


def asymmetric_mode_converter(*, wl=1.55, length: float = 20.0, kappa_per_um: float = math.pi / 40.0,
                              delta_beta_per_um: float = 0.0, neff: float = 2.4, ng: float = 4.2,
                              wl0: float = 1.55, bus_mode: str = "TE0", converted_mode: str = "TE1") -> SDict:
    """Asymmetric directional coupler: bus ``TE0`` <-> multimode-guide ``TE1``.

    Ports follow the 2x2 coupler: the single-mode bus runs ``o1`` -> ``o4``
    (mode ``bus_mode``), the multimode guide ``o2`` -> ``o3`` (mode
    ``converted_mode``). ``delta_beta_per_um = beta_bus - beta_TE1`` is the
    phase mismatch (wavelength-independent here) and ``neff``/``ng`` set the
    mean propagation phase. Coupled-mode theory (module docs): the converted
    power ``|S[o1:TE0 -> o3:TE1]|^2 = kappa^2/(kappa^2 + delta^2) sin^2(S L)``
    with ``delta = delta_beta/2``; the defaults give full conversion.

    Examples
    --------
    >>> s = asymmetric_mode_converter(length=20.0, kappa_per_um=3.141592653589793 / 40.0)
    >>> round(abs(complex(s[("o1:TE0", "o3:TE1")])) ** 2, 12)
    1.0
    """
    wl = xp.asarray(wl)
    bbar = 2.0 * xp.pi * neff_linear(wl, neff, ng, wl0) / wl
    half = 0.5 * xp.asarray(delta_beta_per_um, dtype=float)
    t11, t12, t21, t22 = coupled_mode_transfer(bbar + half, bbar - half, kappa_per_um, length)
    a, b = f"o1:{bus_mode}", f"o2:{converted_mode}"
    c, d = f"o4:{bus_mode}", f"o3:{converted_mode}"
    out: SDict = {}
    for (i, o), v in {(a, c): t11, (a, d): t21, (b, c): t12, (b, d): t22}.items():
        out[(i, o)] = v
        out[(o, i)] = v
    return out


def mode_mux(*, wl=1.55, n_modes: int = 2, loss_db: float = 0.5, crosstalk_db: float = -30.0,
             mode_labels: Sequence[str] | None = None) -> SDict:
    """Mode multiplexer: input ``o{k}:TE0`` -> mode ``k-1`` of the multimode port ``o{N+1}``.

    ``T = a [d I - j sqrt(x) (J - I)]`` with ``a = 10**(-loss_db/20)``,
    ``x = 10**(crosstalk_db/10)`` (power from each input into every other
    mode), ``d = sqrt(1 - (N - 1) x)`` and ``J`` the all-ones matrix. ``T`` is
    normal with singular values ``a sqrt(d^2 + x)`` and
    ``a sqrt(d^2 + x (N-1)^2)``; it is divided by ``max(1, sigma_max)`` so the
    device is passive for every setting. Reciprocal (demultiplexer in reverse).

    Examples
    --------
    >>> s = mode_mux(n_modes=3, loss_db=0.0, crosstalk_db=-200.0)
    >>> round(abs(complex(s[("o2:TE0", "o4:TE1")])), 12)
    1.0
    """
    del wl
    n = int(n_modes)
    if n < 1:
        raise ValueError("n_modes must be >= 1")
    labels = tuple(mode_labels) if mode_labels is not None else tuple(f"TE{k}" for k in range(n))
    if len(labels) != n:
        raise ValueError("mode_labels must have n_modes entries")
    a = 10.0 ** (-xp.asarray(loss_db, dtype=float) / 20.0)
    x = 10.0 ** (xp.asarray(crosstalk_db, dtype=float) / 10.0) if n > 1 else xp.asarray(0.0)
    d = xp.sqrt(1.0 - (n - 1) * x)
    sigma = a * xp.sqrt(d * d + x * max(1, (n - 1) ** 2))
    k = 1.0 / xp.maximum(sigma, 1.0)
    out_port = f"o{n + 1}"
    out: SDict = {}
    for i in range(n):
        for j, lab in enumerate(labels):
            v: Any = k * a * (d if i == j else -1j * xp.sqrt(x))
            out[(f"o{i + 1}:TE0", f"{out_port}:{lab}")] = v
            out[(f"{out_port}:{lab}", f"o{i + 1}:TE0")] = v
    return out
