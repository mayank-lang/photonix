"""Coupler / splitter component models.

Conventions
-----------
A 2x2 coupler exposes four ports: ``o1``/``o2`` on the input (left) side and
``o3``/``o4`` on the output (right) side, arranged::

    o1 ──┐        ┌── o4
         │ couple │
    o2 ──┘        └── o3

with the *through* (bar) paths ``o1->o4`` and ``o2->o3`` and the *cross* paths
``o1->o3`` and ``o2->o4``. The coupler is lossless, reciprocal and unitary; the
cross path carries the canonical 90-degree phase (factor ``-j``).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np

from photonix.core.backend import xp
from photonix.core.types import SDict

__all__ = [
    "directional_coupler", "coupler", "mmi1x2", "mmi2x2",
    "supermode_splitting", "evanescent_coupler", "y_branch", "mmi_nxn", "crossing",
]


def directional_coupler(*, wl=1.55, coupling: float = 0.5) -> SDict:
    """Lossless reciprocal 2x2 directional coupler.

    Parameters
    ----------
    wl : float or array
        Wavelength(s) in µm. Used only for broadcasting; ``coupling`` may itself
        be an array over ``wl`` for a dispersive coupler.
    coupling : float or array
        Power cross-coupling coefficient in [0, 1]. ``coupling=0.5`` is a 50/50
        coupler.

    Returns
    -------
    SDict
        Through amplitude ``t = sqrt(1 - coupling)`` on ``o1<->o4``, ``o2<->o3``;
        cross amplitude ``k = -j*sqrt(coupling)`` on ``o1<->o3``, ``o2<->o4``.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.directional_coupler(coupling=0.5)
    >>> t = px.power(s[("o1","o4")]); k = px.power(s[("o1","o3")])
    >>> abs(float(t + k) - 1.0) < 1e-12
    True
    """
    wl = xp.asarray(wl)
    c = xp.asarray(coupling) * xp.ones_like(xp.real(wl)) if xp.ndim(wl) else xp.asarray(coupling)
    # NOTE: use xp.asarray(..., dtype=complex) rather than .astype: under the
    # NumPy backend a scalar expression like ``-1j * np.float64`` collapses to a
    # plain Python complex (np.float64 subclasses float), which has no .astype.
    t = xp.asarray(xp.sqrt(1.0 - c), dtype=complex)
    k = xp.asarray(-1j * xp.sqrt(c), dtype=complex)
    return {
        ("o1", "o4"): t, ("o4", "o1"): t,
        ("o2", "o3"): t, ("o3", "o2"): t,
        ("o1", "o3"): k, ("o3", "o1"): k,
        ("o2", "o4"): k, ("o4", "o2"): k,
    }


def coupler(*, wl=1.55, coupling: float = 0.5) -> SDict:
    """Alias of :func:`directional_coupler`."""
    return directional_coupler(wl=wl, coupling=coupling)


def mmi1x2(*, wl=1.55, loss_db: float = 0.0) -> SDict:
    """Ideal 1x2 multimode-interference splitter.

    Input ``o1`` splits equally to outputs ``o2`` and ``o3`` (reciprocal).

    Idealization note: a reciprocal, lossless 3-port cannot be matched at all
    ports, so this standard compact model is deliberately *sub-unitary*: the
    antisymmetric (odd) coherent combination of ``o2``/``o3`` inputs is fully
    dissipated even at ``loss_db=0`` (singular values 1, 1, 0). Physically that
    power leaves through the unguided/radiation modes of a real MMI, which a
    matched 3-port cannot represent. Use the EME-backed
    :func:`photonix.em.components.mmi1x2` when back-reflections and the odd-mode
    response matter.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.mmi1x2()
    >>> abs(float(px.power(s[("o1","o2")]) + px.power(s[("o1","o3")])) - 1.0) < 1e-12
    True
    """
    wl = xp.asarray(wl)
    amp = (10.0 ** (-loss_db / 20.0)) / xp.sqrt(2.0)
    a = (amp * xp.ones_like(xp.real(wl))).astype(complex)
    return {
        ("o1", "o2"): a, ("o2", "o1"): a,
        ("o1", "o3"): a, ("o3", "o1"): a,
    }


def mmi2x2(*, wl=1.55, loss_db: float = 0.0) -> SDict:
    """Ideal 2x2 50/50 MMI coupler (ports o1,o2 in; o3,o4 out).

    Implemented as a lossless 50/50 directional coupler with optional excess
    ``loss_db`` applied to every path.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.mmi2x2()
    >>> set(s) >= {("o1","o3"), ("o1","o4")}
    True
    """
    base = directional_coupler(wl=wl, coupling=0.5)
    amp = 10.0 ** (-loss_db / 20.0)
    return {k: amp * v for k, v in base.items()}


# --------------------------------------------------------------------------- #
# Physical (gap/length/wavelength-dependent) evanescent coupler
# --------------------------------------------------------------------------- #
#: Fit of ``ln(n_even - n_odd)`` for two identical 500 nm x 220 nm Si strips in
#: SiO2 (TE supermodes): ``c0 + c1*x + c2*x**2 + c3*y + c4*x*y`` with
#: ``x = gap - 0.2`` and ``y = wl - 1.55`` (µm). Derived from Photonix
#: full-vector FDE solves (``solve_modes_fullvector``, 80 px/µm, 1.5 µm
#: margin, n_Si = 3.4757, n_SiO2 = 1.444) on gaps 0.10-0.40 µm and wavelengths
#: 1.50-1.60 µm. Fit residual <= 1.3 %; resolution spread over 40-80 px/µm is
#: about +/-1.3 %. This is a solver-derived default, not foundry data.
SOI_STRIP_SPLITTING_FIT = (-3.88003, -9.15021, 2.1386, 4.80324, 10.28683)
SOI_STRIP_FIT_GAP_RANGE = (0.10, 0.40)
SOI_STRIP_FIT_WL_RANGE = (1.50, 1.60)


def _is_concrete(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _concrete_array(value) -> np.ndarray | None:
    """``value`` as a NumPy array, or ``None`` for a JAX tracer (no host value)."""
    try:
        return np.asarray(value, dtype=float)
    except Exception:  # noqa: BLE001 - jax.errors.TracerArrayConversionError and friends
        return None


def supermode_splitting(gap, wl=1.55):
    """Even/odd supermode index splitting ``n_even - n_odd`` of an SOI strip pair.

    Evaluates :data:`SOI_STRIP_SPLITTING_FIT` (two 500 x 220 nm strips, TE).
    Outside the calibrated gap range the fit is continued exponentially with
    its edge slope. Differentiable in ``gap`` and ``wl``.

    Examples
    --------
    >>> abs(float(supermode_splitting(0.2, 1.55)) - 0.0206) < 5e-4
    True
    """
    c0, c1, c2, c3, c4 = SOI_STRIP_SPLITTING_FIT
    x = xp.asarray(gap) - 0.2
    y = xp.asarray(wl) - 1.55
    # Outside the calibrated gaps the quadratic term would make the splitting
    # *grow* again. Continue with the local exponential slope at the nearest
    # calibrated edge instead (C1-continuous, monotonically decaying).
    lo, hi = SOI_STRIP_FIT_GAP_RANGE[0] - 0.2, SOI_STRIP_FIT_GAP_RANGE[1] - 0.2
    xc = xp.clip(x, lo, hi)
    slope = c1 + 2.0 * c2 * xc + c4 * y
    return xp.exp(c0 + c1 * xc + c2 * xc * xc + c3 * y + c4 * xc * y + slope * (x - xc))


def evanescent_coupler(
    *,
    wl=1.55,
    gap: float = 0.2,
    length: float = 10.0,
    bend_radius: float = 0.0,
    curved_arms: int = 1,
    splitting: Callable | None = None,
    extra_length: float = 0.0,
    neff: float = 2.4,
    ng: float = 4.2,
    wl0: float = 1.55,
    loss_db_cm: float = 0.0,
) -> SDict:
    """Physical 2x2 evanescent (directional) coupler from coupled-mode theory.

    The two lowest supermodes of the coupled pair beat with index splitting
    ``dn(gap, wl)``; power transfers as ``sin^2(kappa_L)`` with::

        kappa_L = pi / wl * dn(gap, wl) * L_eff
        L_eff   = length + sqrt(2*pi*bend_radius / (curved_arms * gamma))

    The second term is the exact Gaussian integral of an exponentially decaying
    coupling over a curved gap ``g(z) = gap + curved_arms * z**2 / (2 R)``, with
    ``gamma = -d ln(dn)/d gap`` evaluated at ``gap`` (within 2 % of direct
    quadrature of the default fit for 0.12-0.35 µm gaps). ``curved_arms=1`` is a
    ring/racetrack-to-bus coupler (one straight, one curved arm); ``2`` is a
    symmetric coupler whose arms both bend away with radius ``bend_radius``.
    ``length=0`` with a finite ``bend_radius`` is a ring point coupler.

    Ports and phase follow :func:`directional_coupler` (bar ``o1->o4``, cross
    ``o1->o3`` with ``-j``); the common phase accrues over ``length`` with the
    linear-dispersion index of :func:`photonix.components.straight`.

    Parameters
    ----------
    gap
        Edge-to-edge waveguide gap in µm.
    length
        Length of the straight, parallel coupling section in µm.
    splitting
        Optional callable ``splitting(gap, wl) -> n_even - n_odd`` for another
        cross-section, e.g. a PDK fit. Defaults to :func:`supermode_splitting`
        (500 x 220 nm SOI strips, calibrated for gaps 0.10-0.40 µm and
        wavelengths 1.50-1.60 µm; a warning is issued outside that range).
    extra_length
        Additional uncoupled path on every arm (e.g. drawn S-bend arms), which
        adds common phase and loss but no coupling.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.evanescent_coupler(gap=0.2, length=10.0)
    >>> t, k = px.power(s[("o1", "o4")]), px.power(s[("o1", "o3")])
    >>> abs(float(t + k) - 1.0) < 1e-12
    True
    >>> L_c = 1.55 / (2 * float(px.components.supermode_splitting(0.2, 1.55)))
    >>> s = px.components.evanescent_coupler(gap=0.2, length=L_c)
    >>> bool(float(px.power(s[("o1", "o3")])) > 1 - 1e-12)   # full cross-over
    True
    """
    from photonix.core.units import db_per_cm_to_alpha_um

    from .waveguide import neff_linear

    if splitting is None:
        splitting = supermode_splitting
        for name, value, (lo, hi) in (("gap", gap, SOI_STRIP_FIT_GAP_RANGE), ("wl", wl, SOI_STRIP_FIT_WL_RANGE)):
            arr = _concrete_array(value)
            if arr is not None and arr.size and (arr.min() < lo - 1e-9 or arr.max() > hi + 1e-9):
                warnings.warn(
                    f"{name} spans {arr.min():.4g}-{arr.max():.4g} um, outside the calibrated range "
                    f"{lo}-{hi} um of the default SOI supermode fit; pass `splitting=` for a validated "
                    "cross-section.",
                    stacklevel=2,
                )
    if not isinstance(curved_arms, (int, np.integer)) or curved_arms not in (1, 2):
        raise ValueError("curved_arms must be 1 (ring/bus) or 2 (symmetric coupler)")

    wl = xp.asarray(wl)
    dn = xp.asarray(splitting(gap, wl))
    l_eff = xp.asarray(length, dtype=float)
    if not (_is_concrete(bend_radius) and bend_radius == 0):
        h = 1e-3
        gamma = -(xp.log(splitting(gap + h, wl)) - xp.log(splitting(gap - h, wl))) / (2 * h)
        # Guard R = 0 for array/traced radii: d sqrt(x)/dx is infinite at 0 and
        # would make every gradient NaN even though the contribution is zero.
        positive = xp.asarray(bend_radius) > 0
        r_safe = xp.where(positive, bend_radius, 1.0)
        l_eff = l_eff + xp.where(positive, xp.sqrt(2.0 * xp.pi * r_safe / (curved_arms * gamma)), 0.0)
    kappa_l = xp.pi / wl * dn * l_eff
    n = neff_linear(wl, neff, ng, wl0)
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    path = length + extra_length
    common = xp.exp(-alpha * path) * xp.exp(-1j * 2.0 * xp.pi / wl * n * path)
    t = xp.asarray(xp.cos(kappa_l) * common, dtype=complex)
    k = xp.asarray(-1j * xp.sin(kappa_l) * common, dtype=complex)
    return {
        ("o1", "o4"): t, ("o4", "o1"): t,
        ("o2", "o3"): t, ("o3", "o2"): t,
        ("o1", "o3"): k, ("o3", "o1"): k,
        ("o2", "o4"): k, ("o4", "o2"): k,
    }


def y_branch(*, wl=1.55, loss_db: float = 0.0) -> SDict:
    """Ideal symmetric Y-branch splitter: ``o1`` -> ``o2``/``o3`` equally.

    Same compact physics as :func:`mmi1x2` (including its deliberately
    sub-unitary odd-mode response), registered under its own name so layout
    cells called ``y_branch`` extract directly.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.y_branch(loss_db=0.1)
    >>> abs(float(px.power(s[("o1", "o2")])) - 0.5 * 10 ** -0.01) < 1e-12
    True
    """
    return mmi1x2(wl=wl, loss_db=loss_db)


def _mmi_phases(n: int) -> np.ndarray:
    """General-interference NxN MMI phases (Bachmann et al., Appl. Opt. 33, 3905, 1994).

    ``phi[i, j]`` couples input ``i`` to output ``j`` (1-based in the paper,
    0-based here). The paper numbers inputs and outputs from *opposite* sides,
    so :func:`mmi_nxn` reads output ``j`` (counted top to bottom) as
    ``phi[i, n - 1 - j]``.
    """
    phi = np.zeros((n, n))
    phi0 = -math.pi
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i + j) % 2 == 0:
                phi[i - 1, j - 1] = phi0 + math.pi + math.pi / (4 * n) * (j - i) * (2 * n - j + i)
            else:
                phi[i - 1, j - 1] = phi0 + math.pi / (4 * n) * (i + j - 1) * (2 * n - j - i + 1)
    return phi


def mmi_nxn(*, wl=1.55, n: int = 4, loss_db: float = 0.0, imbalance_db: float = 0.0) -> SDict:
    """Ideal NxN general-interference MMI coupler (self-imaging at ``3 L_pi``).

    Inputs ``o1..oN`` are on the left (top to bottom); outputs ``o(N+1)..o(2N)``
    on the right (bottom to top), the same counter-clockwise numbering as the
    2x2 :func:`directional_coupler`. Every input images to every output with
    power ``1/N`` and the self-imaging phases of Bachmann et al. (1994), so the
    transfer matrix is unitary; ``n=2`` reproduces :func:`mmi2x2`.

    Reference planes are at the MMI facets. The per-port field references
    alternate in sign with port position (``(-1)**p``, identically on both
    sides) -- the convention under which ``n=2`` is the ``-j`` directional
    coupler; it is a consistent per-port gauge of the modal self-imaging
    matrix. Consequently ``N`` cascaded ``N x N`` MMIs (each ``3 L_pi / N``) form
    a ``3 L_pi`` MMI and image every input onto the mirrored output, as
    self-imaging theory requires.

    Parameters
    ----------
    n
        Number of inputs (= outputs), ``n >= 2``.
    loss_db
        Excess loss applied to every path.
    imbalance_db
        Power imbalance between the outermost and central outputs, applied as
        a symmetric quadratic taper of the output amplitudes (keeps passivity).

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> s = px.components.mmi_nxn(n=4)
    >>> S, pm = px.as_sdense(s)
    >>> bool(np.allclose(np.asarray(S).conj().T @ np.asarray(S), np.eye(8)))
    True
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")
    n = int(n)
    wl = xp.asarray(wl)
    ones = xp.ones_like(xp.real(wl), dtype=float)
    amp = 10.0 ** (-loss_db / 20.0) / math.sqrt(n)
    phases = _mmi_phases(n)
    # Bachmann's output index runs from the opposite side; pi/2 and (-1)^(i+j)
    # fix the global phase and the per-port sign references (n=2 == mmi2x2).
    gauge = math.pi / 2.0
    centre = 0.5 * (n - 1)
    out: SDict = {}
    for j in range(n):  # output index, top to bottom
        rel = ((j - centre) / centre) ** 2 if n > 1 else 0.0
        taper = 10.0 ** (-(imbalance_db * rel) / 20.0)
        out_port = f"o{2 * n - j}"
        for i in range(n):
            phi = phases[i, n - 1 - j] + gauge + math.pi * (i + j)
            val = xp.asarray(amp * taper * np.exp(1j * phi) * ones, dtype=complex)
            in_port = f"o{i + 1}"
            out[(in_port, out_port)] = val
            out[(out_port, in_port)] = val
    return out


def crossing(
    *,
    wl=1.55,
    loss_db: float = 0.1,
    crosstalk_db: float = -40.0,
    reflection_db: float | None = None,
) -> SDict:
    """Waveguide crossing. Ports ``o1`` (W), ``o2`` (N), ``o3`` (E), ``o4`` (S).

    Light entering a port crosses straight through (``o1<->o3``, ``o2<->o4``)
    with ``loss_db`` insertion loss, leaks ``crosstalk_db`` (power, relative to
    the input) into each perpendicular arm with a 90-degree phase, and is
    optionally reflected with ``reflection_db``.

    The model is a symmetric circulant matrix, so its singular values are
    ``|r + t +/- 2x|`` and ``|r - t|``. With the quadrature crosstalk phase the
    passivity condition is ``(|t| + |r|)^2 + 4|x|^2 <= 1``, which is checked for
    concrete inputs.

    Examples
    --------
    >>> import photonix as px
    >>> s = px.components.crossing(loss_db=0.2, crosstalk_db=-35.0)
    >>> round(float(10 * __import__("math").log10(px.power(s[("o1", "o3")]))), 6)
    -0.2
    >>> bool(px.core.is_passive(s))
    True
    """
    if _is_concrete(loss_db) and _is_concrete(crosstalk_db) and (
            reflection_db is None or _is_concrete(reflection_db)):
        t2 = 10.0 ** (-loss_db / 10.0)
        x2 = 10.0 ** (crosstalk_db / 10.0)
        r = 0.0 if reflection_db is None else 10.0 ** (reflection_db / 20.0)
        if (math.sqrt(t2) + r) ** 2 + 4 * x2 > 1.0 + 1e-12:
            raise ValueError("crossing parameters are not passive: need (|t| + |r|)^2 + 4|x|^2 <= 1")
    wl = xp.asarray(wl)
    ones = xp.ones_like(xp.real(wl), dtype=float)
    t = xp.asarray(10.0 ** (-xp.asarray(loss_db) / 20.0) * ones, dtype=complex)
    x = xp.asarray(-1j * 10.0 ** (xp.asarray(crosstalk_db) / 20.0) * ones, dtype=complex)
    out: SDict = {}
    for a in range(4):
        opposite = (a + 2) % 4
        out[(f"o{a + 1}", f"o{opposite + 1}")] = t
        for neighbour in ((a + 1) % 4, (a + 3) % 4):
            out[(f"o{a + 1}", f"o{neighbour + 1}")] = x
        if reflection_db is not None:
            r = xp.asarray(10.0 ** (xp.asarray(reflection_db) / 20.0) * ones, dtype=complex)
            out[(f"o{a + 1}", f"o{a + 1}")] = r
    return out
