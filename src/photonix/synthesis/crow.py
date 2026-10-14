"""Exact synthesis of coupled-resonator (CROW) band-pass filters.

Step 1 maps the low-pass prototype to ring couplings in closed form (Little et
al., JLT 15, 998, 1997): the bus couplings ``K_ext = 2 pi B / (g0 g1 FSR)`` and
the ring-to-ring couplings ``K_i = (pi B / FSR)^2 / (g_i g_{i+1})``. That
narrow-band approximation is several per cent off in bandwidth (and much more
for wide bands), so step 2 solves for the symmetric coupling set
(``K_i = K_{N-i}``) whose lossless drop response equals the periodic target
:func:`~photonix.synthesis.prototypes.periodic_target` exactly. The solve uses a
transfer-matrix model of the chain and is checked against the netlist
simulation of :func:`photonix.circuit.crow`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from photonix.core.types import SDict

from .prototypes import lowpass_prototype, periodic_target

__all__ = ["CrowDesign", "synthesize_crow", "crow_closed_form", "crow_drop_power"]

_C = 299_792_458.0


def crow_drop_power(couplings, phi) -> np.ndarray:
    """Lossless drop power of a CROW at round-trip phase detuning ``phi``.

    Chains the 2x2 transfer matrices ``(j/k)[[-1, t], [-t, 1]]`` of the
    couplers (``t^2 + k^2 = 1``) with the half-ring propagation
    ``diag(e^{-j phi/2}, e^{+j phi/2})``; the drop amplitude is ``1/M22``.

    Examples
    --------
    >>> round(float(crow_drop_power([0.1, 0.1], 0.0)), 12)          # matched add-drop ring
    1.0
    """
    k = np.sqrt(np.asarray(couplings, dtype=float))
    t = np.sqrt(1.0 - k**2)
    h = np.exp(-0.5j * np.asarray(phi, dtype=float))
    m11 = np.ones_like(h)
    m12 = np.zeros_like(h)
    m21 = np.zeros_like(h)
    m22 = np.ones_like(h)
    for i, (ki, ti) in enumerate(zip(k, t, strict=True)):
        if i:                                                 # half ring between couplers i-1 and i
            m11, m12, m21, m22 = h * m11, h * m12, m21 / h, m22 / h
        a, b = 1j / ki * -1.0, 1j / ki * ti                    # C = [[a, b], [-b, -a]]
        m11, m12, m21, m22 = a * m11 + b * m21, a * m12 + b * m22, -b * m11 - a * m21, -b * m12 - a * m22
    return np.abs(1.0 / m22) ** 2


@dataclass(frozen=True)
class CrowDesign:
    """A synthesized CROW filter.

    Attributes
    ----------
    couplings
        Power couplings, bus -> ring 1 -> ... -> ring N -> bus.
    radius
        Ring radius in µm, chosen so ``wl0`` is resonant.
    fsr_hz, bandwidth_hz
        Free spectral range and design bandwidth (Butterworth: -3 dB width;
        Chebyshev: ripple-band width).
    max_error_db
        Largest deviation of the simulated drop response (netlist model) from
        the target over the full FSR, counted where the target exceeds -40 dB
        (2001 points over half the FSR plus 2001 over four half-bandwidths,
        so narrow bands are resolved).
    """

    couplings: tuple[float, ...]
    radius: float
    neff: float
    ng: float
    wl0: float
    fsr_hz: float
    bandwidth_hz: float
    kind: str
    order: int
    ripple_db: float | None
    max_error_db: float

    def model(self, *, loss_db_cm: float = 0.0):
        """The filter as a :func:`photonix.circuit.crow` circuit model."""
        from photonix.circuit import crow

        return crow(couplings=self.couplings, radius=self.radius, neff=self.neff, ng=self.ng,
                    loss_db_cm=loss_db_cm, wl0=self.wl0)

    def wavelength(self, df_hz):
        """Wavelength (µm) at a frequency detuning ``df_hz`` from the centre."""
        f0 = _C / (self.wl0 * 1e-6)
        return _C / (f0 + np.asarray(df_hz)) * 1e6

    def target(self, df_hz):
        """Target drop power at frequency detuning ``df_hz``."""
        phi = 2 * np.pi * np.asarray(df_hz) / self.fsr_hz
        return periodic_target(phi, kind=self.kind, order=self.order,
                               phi_c=np.pi * self.bandwidth_hz / self.fsr_hz, ripple_db=self.ripple_db)


def crow_closed_form(g: np.ndarray, bandwidth_hz: float, fsr_hz: float) -> np.ndarray:
    """Little et al. (1997) narrow-band coupling formulas from prototype values ``g``.

    Examples
    --------
    >>> g = lowpass_prototype("butterworth", 1)
    >>> k = crow_closed_form(g, 10e9, 1e12)
    >>> bool(np.allclose(k, 2 * np.pi * 10e9 / (2 * 1e12)))
    True
    """
    n = len(g) - 2
    b = bandwidth_hz / fsr_hz
    k = np.empty(n + 1)
    k[0] = 2 * math.pi * b / (g[0] * g[1])
    k[n] = 2 * math.pi * b / (g[n] * g[n + 1])
    for i in range(1, n):
        k[i] = (math.pi * b) ** 2 / (g[i] * g[i + 1])
    return k


def _unique(k: np.ndarray) -> np.ndarray:
    """Symmetrize (geometric mean of mirror pairs) and keep the first half."""
    sym = np.sqrt(k * k[::-1])
    return sym[: (len(k) + 1) // 2]


def _full(u: np.ndarray, n_couplers: int) -> np.ndarray:
    half = 1.0 / (1.0 + np.exp(-u))
    return np.concatenate([half, half[: n_couplers // 2][::-1]])


def _solve(order, kind, ripple_db, phi_c, k_start, n_nodes):
    from scipy.optimize import least_squares

    nn = n_nodes or 8 * order + 16
    phis = np.concatenate([np.linspace(0.0, 1.5 * phi_c, nn // 2, endpoint=False),
                           np.linspace(1.5 * phi_c, math.pi, nn - nn // 2)])
    target = periodic_target(phis, kind=kind, order=order, phi_c=phi_c, ripple_db=ripple_db)

    def residual(u):
        return np.log(crow_drop_power(_full(u, order + 1), phis)) - np.log(target)

    h = np.clip(_unique(k_start), 1e-12, 1 - 1e-9)
    sol = least_squares(residual, np.log(h / (1 - h)), jac="3-point", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        max_nfev=400 * (order + 1))
    return _full(sol.x, order + 1)


def synthesize_crow(order: int, bandwidth_hz: float, *, kind: str = "butterworth", ripple_db: float | None = None,
                    radius: float | None = None, fsr_hz: float | None = None, neff: float = 2.4, ng: float = 4.2,
                    wl0: float = 1.55, refine: bool = True, n_nodes: int | None = None) -> CrowDesign:
    """Design an ``order``-ring CROW drop filter with the given bandwidth.

    Give exactly one of ``radius`` (µm) or ``fsr_hz``; the circumference is then
    rounded to an integer number of guided wavelengths so ``wl0`` is resonant.
    ``refine=False`` returns the closed-form couplings (useful to quantify the
    narrow-band approximation). Wide bands are reached by continuation from a
    narrow one, where the closed form is accurate.

    Examples
    --------
    >>> d = synthesize_crow(3, 100e9, radius=10.0)
    >>> [round(k, 4) for k in d.couplings], d.max_error_db < 1e-6
    ([0.4234, 0.0372, 0.0372, 0.4234], True)
    """
    from photonix.circuit import crow

    if (radius is None) == (fsr_hz is None):
        raise ValueError("give exactly one of radius or fsr_hz")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if radius is not None:
        length = 2 * math.pi * radius
    else:
        assert fsr_hz is not None
        length = _C / (ng * float(fsr_hz)) * 1e6
    m = max(1, round(neff * length / wl0))
    length = m * wl0 / neff
    radius_r = length / (2 * math.pi)
    fsr = _C / (ng * length * 1e-6)                              # exact: phase is linear in frequency
    if not 0 < bandwidth_hz < fsr / 2:
        raise ValueError(f"bandwidth must be positive and below half the FSR ({fsr / 2:.4g} Hz)")
    g = lowpass_prototype(kind, order, ripple_db)
    if not refine:
        k = np.clip(crow_closed_form(g, bandwidth_hz, fsr), 1e-12, 1 - 1e-12)
    else:
        start_b = min(bandwidth_hz, 0.01 * fsr)
        steps = max(1, math.ceil(math.log(bandwidth_hz / start_b) / math.log(1.5)))
        k = crow_closed_form(g, start_b, fsr)
        for b in np.geomspace(start_b, bandwidth_hz, steps + 1)[(1 if steps else 0):] if bandwidth_hz > start_b \
                else [bandwidth_hz]:
            k = _solve(order, kind, ripple_db, math.pi * b / fsr, k, n_nodes)
    f0 = _C / (wl0 * 1e-6)
    phi_c = math.pi * bandwidth_hz / fsr
    # full half-FSR plus a dense in-band grid: a narrow band holds few points of the first
    check = np.union1d(np.linspace(0.0, math.pi, 2001), np.linspace(0.0, min(math.pi, 4 * phi_c), 2001))
    s = cast(SDict, crow(couplings=tuple(float(v) for v in k), radius=radius_r, neff=neff, ng=ng, loss_db_cm=0.0,
                         wl0=wl0)(wl=_C / (f0 + check / (2 * math.pi) * fsr) * 1e6))
    got = np.asarray(np.abs(np.asarray(s[("o1", "o3")])) ** 2, dtype=float)
    want = periodic_target(check, kind=kind, order=order, phi_c=phi_c, ripple_db=ripple_db)
    sel = want > 1e-4
    err = float(np.max(np.abs(10 * np.log10(got[sel] / want[sel]))))
    return CrowDesign(tuple(float(v) for v in k), radius_r, neff, ng, wl0, fsr, float(bandwidth_hz), kind,
                      int(order), ripple_db, err)
