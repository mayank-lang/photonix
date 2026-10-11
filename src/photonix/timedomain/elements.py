"""Time-domain circuit elements in the complex-envelope (sample-mode) picture.

Every element is linear in its incident waves and splits, at each time step
``k``, into an *instantaneous* part and a *strictly causal* part::

    b_k = D a_k + y_k,        y_k depends only on a_{k-1}, a_{k-2}, ...

``D`` (rows = outgoing port, columns = incident port, as in the frequency-domain
solver) goes into the algebraic part of the network solve; ``y`` is produced by
*channels*. A channel is a strictly causal rational filter from one incident
port to one outgoing port::

    w_k = sum_m b_m x_{k - n0 - m} - sum_p a_p w_{k - p},     n0 >= 1

which covers fractional delay lines (Lagrange FIR or Thiran all-pass), FIRs
fitted to a sampled frequency response, and exactly discretized poles of a
rational (pole-residue) model. An instantaneous tap that a design produces is
moved into ``D`` (:func:`split_instantaneous`), so ``n0 >= 1`` always holds and
feedback loops through any element with at least one sample of delay are
explicit and causal.

All coefficients are computed with the backend array module, so they are
differentiable with respect to delays, losses, phases and pole/residue values;
the integer parts of delays (``n0``, the number of taps) are static and are
fixed from concrete values when an element is first compiled.

Conventions: ``exp(+j omega t)``; a pure delay ``tau`` is
``H(nu) = exp(-j 2 pi nu tau)`` at frequency offset ``nu`` from the carrier.
Times are in seconds.

References
----------
* T. I. Laakso, V. Valimaki, M. Karjalainen, U. K. Laine, "Splitting the unit
  delay," IEEE Signal Process. Mag. 13(1), 30-60 (1996) -- Lagrange and Thiran
  fractional-delay filters.
* B. Gustavsen, A. Semlyen, "Rational approximation of frequency domain
  responses by vector fitting," IEEE Trans. Power Del. 14(3), 1052 (1999), and
  "Simulation of transmission line transients using vector fitting and modal
  decomposition," IEEE Trans. Power Del. 13(2), 605 (1998) -- recursive
  convolution of pole-residue models with a piecewise-linear input.
"""
from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from photonix.core.backend import xp

__all__ = [
    "Channel",
    "CompiledTD",
    "TDElement",
    "MemorylessTD",
    "DelayLineTD",
    "FIRTD",
    "PoleResidueTD",
    "TimeVaryingTD",
    "CompositeTD",
    "lagrange_taps",
    "thiran_coefficients",
    "split_instantaneous",
    "design_fir",
    "parse_interp",
]


# --------------------------------------------------------------------------- #
# Compiled representation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Channel:
    """A strictly causal rational path from incident port ``src`` to outgoing port ``dst``.

    ``w_k = sum_m b[m] x_{k-n0-m} - sum_p a[p-1] w_{k-p}`` with ``n0 >= 1``;
    ``b`` has ``M`` taps and ``a`` (``None`` for an FIR) has ``P`` feedback
    coefficients ``a_1..a_P``. ``n0`` and the lengths are static integers.
    """

    src: str
    dst: str
    n0: int
    b: Any
    a: Any = None

    def __post_init__(self) -> None:
        if int(self.n0) < 1:
            raise ValueError("a channel must be strictly causal (n0 >= 1); move instantaneous taps into D")
        object.__setattr__(self, "n0", int(self.n0))

    @property
    def n_taps(self) -> int:
        return int(np.shape(self.b)[0])

    @property
    def n_feedback(self) -> int:
        return 0 if self.a is None else int(np.shape(self.a)[0])

    @property
    def signature(self) -> tuple[int, int]:
        return (self.n_taps, self.n_feedback)


@dataclass(frozen=True)
class CompiledTD:
    """An element reduced to its instantaneous matrix, causal channels and time-varying part."""

    ports: tuple[str, ...]
    D: Any
    channels: tuple[Channel, ...] = ()
    time_varying: TimeVaryingTD | None = None
    #: Static layout (offsets and tap counts) used to keep structure fixed when
    #: coefficients are recomputed from traced parameters.
    structure: Any = None


class TDElement:
    """Base class of time-domain elements.

    Subclasses define :attr:`ports` and :meth:`compile`, which turns the
    element into a :class:`CompiledTD` for a given time step. :meth:`frozen`
    returns a copy whose static layout (integer delays, tap counts) is pinned,
    so the element can be rebuilt from traced parameters without changing
    shape.
    """

    ports: tuple[str, ...] = ()

    def compile(self, dt: float) -> CompiledTD:  # pragma: no cover - abstract
        raise NotImplementedError

    def frozen(self, dt: float) -> TDElement:
        """A copy whose static layout is pinned for time step ``dt``."""
        return self

    def describe(self, dt: float) -> dict[str, Any]:
        """A short, JSON-friendly description (used by circuit reports)."""
        return {"type": type(self).__name__, "ports": list(self.ports)}


def _complex(x):
    return xp.asarray(x, dtype=complex)


def split_instantaneous(n0: int, b, a=None):
    """Make a rational channel strictly causal.

    For ``H(z) = z^{-n0} B(z) / A(z)`` with ``n0 = 0`` the constant term
    ``b0 = b[0]`` is instantaneous: ``H = b0 + z^{-1} B'(z) / A(z)`` with
    ``B'(z) = (B(z) - b0 A(z)) / z^{-1}``. Returns ``(d, n0', b', a)`` where
    ``d`` is the instantaneous coefficient (``0`` when ``n0 >= 1``) and ``b'``
    may be empty (``None``) when nothing causal remains.

    Examples
    --------
    >>> import numpy as np
    >>> d, n0, b, a = split_instantaneous(0, np.array([0.5, 0.25]))
    >>> float(d), n0, [float(v) for v in b], a
    (0.5, 1, [0.25], None)
    """
    n0 = int(n0)
    if n0 >= 1:
        return 0.0, n0, b, a
    if n0 < 0:
        raise ValueError("a negative tap offset is non-causal")
    b = _complex(b)
    b0 = b[0]
    if a is None:
        rest = b[1:]
    else:
        a = _complex(a)
        p = int(a.shape[0])
        m = int(b.shape[0])
        width = max(m, p + 1)
        bb = xp.concatenate([b, xp.zeros(width - m, dtype=complex)])
        aa = xp.concatenate([xp.ones(1, dtype=complex), a, xp.zeros(width - p - 1, dtype=complex)])
        rest = (bb - b0 * aa)[1:]
    if int(rest.shape[0]) == 0:
        return b0, 1, None, None
    return b0, 1, rest, a


# --------------------------------------------------------------------------- #
# Fractional-delay designs
# --------------------------------------------------------------------------- #
_INTERP_RE = re.compile(r"^(lagrange|thiran)(\d+)$")


def parse_interp(interp: str) -> tuple[str, int]:
    """Parse ``"lagrange3"``, ``"thiran3"`` or ``"nearest"`` into ``(kind, order)``.

    Examples
    --------
    >>> parse_interp("lagrange5"), parse_interp("nearest")
    (('lagrange', 5), ('nearest', 0))
    """
    if interp == "nearest":
        return "nearest", 0
    m = _INTERP_RE.match(str(interp))
    if not m:
        raise ValueError(f"interp must be 'nearest', 'lagrangeN' (odd N) or 'thiranN', got {interp!r}")
    kind, order = m.group(1), int(m.group(2))
    if kind == "lagrange" and (order < 1 or order % 2 == 0 or order > 15):
        raise ValueError("Lagrange interpolation order must be odd, between 1 and 15")
    if kind == "thiran" and not 1 <= order <= 8:
        raise ValueError("Thiran all-pass order must be between 1 and 8")
    return kind, order


def lagrange_taps(p, order: int = 3):
    """Lagrange fractional-delay FIR ``h_m = prod_{k != m} (p - k) / (m - k)``, ``m = 0..N``.

    The filter delays by ``p`` samples; it is exact for integer ``p`` and has
    phase-delay error ``O((omega dt)^(N+1))``, smallest for
    ``(N-1)/2 <= p <= (N+1)/2`` (Laakso et al. 1996). Differentiable in ``p``.

    Examples
    --------
    >>> import numpy as np
    >>> [round(float(v), 12) for v in np.asarray(lagrange_taps(1.0, 3))]
    [0.0, 1.0, 0.0, 0.0]
    >>> bool(np.isclose(np.asarray(lagrange_taps(1.5, 3)).sum(), 1.0))
    True
    """
    p = xp.asarray(p, dtype=float)
    taps = []
    for m in range(order + 1):
        h = xp.ones_like(p)
        for k in range(order + 1):
            if k != m:
                h = h * (p - k) / (m - k)
        taps.append(h)
    return xp.stack(taps)


def thiran_coefficients(d, order: int = 3):
    """Thiran all-pass denominator ``a_0..a_N`` for a delay of ``d`` samples.

    ``a_k = (-1)^k C(N, k) prod_{n=0}^{N} (d - N + n) / (d - N + k + n)``; the
    all-pass ``H(z) = z^{-N} A(1/z) / A(z)`` has unit magnitude at every
    frequency and maximally flat group delay ``d`` at DC. It is stable for
    ``d > N - 1`` and most accurate for ``N - 1/2 <= d < N + 1/2``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.asarray(thiran_coefficients(3.0, 3))       # integer delay -> pure z^-3
    >>> [round(float(v), 12) for v in a]
    [1.0, 0.0, 0.0, 0.0]
    """
    d = xp.asarray(d, dtype=float)
    coeffs = []
    for k in range(order + 1):
        c = (-1.0) ** k * math.comb(order, k) * xp.ones_like(d)
        for n in range(order + 1):
            c = c * (d - order + n) / (d - order + k + n)
        coeffs.append(c)
    return xp.stack(coeffs)


def _concrete_float(value, what: str) -> float:
    try:
        return float(np.asarray(value))
    except Exception as exc:  # noqa: BLE001 - jax ConcretizationTypeError and friends
        raise ValueError(
            f"{what} must be concrete to choose the static (integer) part of the design; "
            "build the element from concrete values first and use .frozen(dt) before tracing"
        ) from exc


# --------------------------------------------------------------------------- #
# Least-squares FIR design
# --------------------------------------------------------------------------- #
def _fir_matrix(nu, dt, n0: int, n_taps: int):
    n = n0 + np.arange(n_taps)
    return np.exp(-2j * np.pi * np.outer(np.asarray(nu, dtype=float) * dt, n))


def _fir_pinv(nu, dt, n0: int, n_taps: int, weights) -> np.ndarray:
    """Weighted least-squares pseudo-inverse ``(A^H W A)^{-1} A^H W`` (``n_taps x n_freq``)."""
    a = _fir_matrix(nu, dt, n0, n_taps)
    w = np.asarray(weights, dtype=float)
    aw = a.conj().T * w
    return np.linalg.solve(aw @ a, aw)


def design_fir(nu, response, dt: float, *, n0: int, n_taps: int, band: float = 0.8,
               out_of_band_weight: float = 1e-3):
    """Least-squares FIR ``sum_m h_m exp(-j 2 pi nu (n0 + m) dt)`` matching ``response(nu)``.

    The fit is weighted 1 for ``|nu| <= band / (2 dt)`` and
    ``out_of_band_weight`` beyond, which keeps the out-of-band gain bounded
    without spending taps on it. ``response`` may be ``(n_freq,)`` or
    ``(n_freq, k)`` (several responses sharing one tap window); it may be a
    traced array, the design is linear in it.

    Returns
    -------
    taps, max_error
        Taps ``(n_taps,)`` or ``(n_taps, k)`` and the maximum in-band absolute
        error (``nan`` when ``response`` is traced).

    Examples
    --------
    >>> import numpy as np
    >>> dt = 1e-12
    >>> nu = np.linspace(-0.5, 0.5, 512, endpoint=False) / dt
    >>> h, err = design_fir(nu, np.exp(-2j * np.pi * nu * 7.25 * dt), dt, n0=0, n_taps=24)
    >>> bool(err < 1e-3)        # a 7.25-sample delay, in-band error
    True
    """
    nu = np.asarray(nu, dtype=float)
    inband = np.abs(nu) <= 0.5 * band / dt + 1e-12 / dt
    weights = np.where(inband, 1.0, out_of_band_weight)
    pinv = _fir_pinv(nu, dt, n0, n_taps, weights)
    taps = xp.asarray(pinv) @ _complex(response)
    try:
        resp = np.asarray(response, dtype=complex)
        fit = _fir_matrix(nu, dt, n0, n_taps) @ np.asarray(taps)
        err = float(np.max(np.abs(fit - resp)[inband])) if np.any(inband) else 0.0
    except Exception:  # noqa: BLE001 - traced response
        err = float("nan")
    return taps, err


# --------------------------------------------------------------------------- #
# Elements
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemorylessTD(TDElement):
    """An instantaneous (frequency-flat) linear element ``b = S a``.

    ``S`` is ``(n, n)`` with rows = outgoing port and columns = incident port,
    in the order of ``ports``. Couplers, MMIs, crossings and terminations are
    memoryless at the carrier.

    Examples
    --------
    >>> import photonix as px
    >>> el = MemorylessTD.from_sdict(px.components.directional_coupler(coupling=0.5))
    >>> el.ports
    ('o1', 'o2', 'o3', 'o4')
    >>> round(abs(complex(el.S[3, 0])) ** 2, 12)      # o1 -> o4 bar power
    0.5
    """

    ports: tuple[str, ...]
    S: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        shape = np.shape(self.S)
        n = len(self.ports)
        if shape != (n, n):
            raise ValueError(f"S must be ({n}, {n}) for ports {self.ports}, got {shape}")

    @classmethod
    def from_sdict(cls, sdict: Mapping, ports: Sequence[str] | None = None) -> MemorylessTD:
        """Build from a scattering dictionary evaluated at the carrier (scalar entries)."""
        from photonix.core.types import ports_of

        names = tuple(ports) if ports is not None else tuple(ports_of(sdict))
        idx = {p: i for i, p in enumerate(names)}
        n = len(names)
        S = xp.zeros((n, n), dtype=complex)
        for (p_in, p_out), v in sdict.items():
            if p_in not in idx or p_out not in idx:
                continue
            value = xp.reshape(_complex(v), ())
            S = S.at[idx[p_out], idx[p_in]].set(value) if hasattr(S, "at") else _np_set(S, idx[p_out], idx[p_in], value)
        return cls(names, S)

    def compile(self, dt: float) -> CompiledTD:
        return CompiledTD(self.ports, _complex(self.S))

    def describe(self, dt: float) -> dict[str, Any]:
        return {"type": "MemorylessTD", "ports": list(self.ports)}


def _np_set(S, i, j, v):
    S = np.array(S, copy=True)
    S[i, j] = v
    return S


@dataclass(frozen=True)
class DelayLineTD(TDElement):
    """A reciprocal waveguide delay: delay ``tau_s``, carrier phase, loss, optional GVD.

    The transfer function between ``ports[0]`` and ``ports[1]`` (both
    directions) is::

        H(nu) = loss_amp * exp(-j carrier_phase) * exp(-j 2 pi nu tau_s)
                * exp(-j beta2_l_s2 (2 pi nu)^2 / 2)

    which is the envelope response of ``exp(-j beta(omega) L)`` expanded about
    the carrier (``carrier_phase = beta(omega_c) L``, ``tau_s = beta_1 L``,
    ``beta2_l_s2 = beta_2 L`` in s^2). For the linear-dispersion waveguide
    models of photonix (``beta`` exactly linear in frequency) the first three
    factors are exact.

    Parameters
    ----------
    tau_s
        Group delay in seconds.
    carrier_phase
        Phase ``beta(omega_c) L`` in radians (the factor is ``exp(-j phase)``).
    loss_amp
        Amplitude transmission (``10**(-loss_dB / 20)``).
    beta2_l_s2
        Total group-velocity dispersion ``beta_2 L`` in s^2, or ``None`` for
        none. Implemented, together with the fractional delay, by a
        least-squares FIR over ``|nu| <= gvd_band / (2 dt)``.
    interp
        Fractional-delay design: ``"lagrangeN"`` (odd ``N``; FIR, phase error
        ``O((omega dt)^(N+1))``, magnitude slightly below one at high
        frequency), ``"thiranN"`` (IIR all-pass: lossless at every frequency,
        maximally flat group delay; needs ``tau >= (N - 1/2) dt``) or
        ``"nearest"`` (rounds to whole samples). Ignored when GVD is on.
    split
        Static ``(n0, n_taps)`` layout. ``None`` derives it from the concrete
        delay at compile time; :meth:`frozen` pins it.

    Examples
    --------
    >>> el = DelayLineTD(tau_s=10e-12)
    >>> c = el.compile(1e-12)            # an exact 10-sample delay
    >>> c.channels[0].n0, [round(float(abs(v)), 12) for v in c.channels[0].b]
    (9, [0.0, 1.0, 0.0, 0.0])
    """

    tau_s: Any
    carrier_phase: Any = 0.0
    loss_amp: Any = 1.0
    beta2_l_s2: Any = None
    interp: str = "lagrange3"
    ports: tuple[str, ...] = ("o1", "o2")
    split: tuple[int, int] | None = None
    gvd_band: float = 0.8
    gvd_tol: float = 1e-7

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        if len(self.ports) != 2:
            raise ValueError("a delay line has exactly two ports")
        parse_interp(self.interp)
        if not 0.0 < self.gvd_band <= 1.0:
            raise ValueError("gvd_band must be in (0, 1]")

    @property
    def amplitude(self):
        """Complex carrier transmission ``loss_amp * exp(-j carrier_phase)``."""
        return _complex(self.loss_amp) * xp.exp(-1j * xp.asarray(self.carrier_phase, dtype=float))

    # -- static layout ------------------------------------------------------ #
    def _layout(self, dt: float) -> tuple[int, int]:
        if self.split is not None:
            return int(self.split[0]), int(self.split[1])
        d = _concrete_float(self.tau_s, "tau_s") / dt
        if d < 0:
            raise ValueError("a delay line cannot have a negative delay")
        if self.beta2_l_s2 is not None:
            b2 = abs(_concrete_float(self.beta2_l_s2, "beta2_l_s2"))
            omega_edge = math.pi * self.gvd_band / dt
            half = int(math.ceil(b2 * omega_edge / dt)) + 12
            n0 = int(math.floor(d)) - half
            if n0 < 0:
                raise ValueError(
                    f"the dispersive spread ({b2 * omega_edge:.3g} s) exceeds the group delay ({d * dt:.3g} s); "
                    "a causal FIR cannot represent it -- reduce gvd_band or dt, or add delay"
                )
            return n0, 2 * half + 2
        kind, order = parse_interp(self.interp)
        if kind == "nearest":
            return int(round(d)), 1
        if kind == "lagrange":
            n0 = max(int(math.floor(d - (order - 1) / 2.0)), 0)
            return n0, order + 1
        n_int = int(math.floor(d - order + 0.5))
        if n_int < 0:
            raise ValueError(
                f"a Thiran order-{order} all-pass needs a delay of at least {order - 0.5} samples "
                f"(got {d:.3g}); use a Lagrange design or a smaller dt"
            )
        return n_int, order + 1

    def frozen(self, dt: float) -> DelayLineTD:
        return dataclasses.replace(self, split=self._layout(dt))

    def delay_samples(self, dt: float) -> float:
        """The delay in samples, ``tau_s / dt`` (concrete)."""
        return _concrete_float(self.tau_s, "tau_s") / dt

    # -- design -------------------------------------------------------------- #
    def compile(self, dt: float) -> CompiledTD:
        n0, n_taps = self._layout(dt)
        d = xp.asarray(self.tau_s, dtype=float) / dt
        amp = self.amplitude
        a_fb = None
        if self.beta2_l_s2 is not None:
            n_f = max(8 * n_taps, 256)
            nu = (np.arange(n_f) - n_f // 2) / (n_f * dt)
            omega = 2.0 * np.pi * nu
            target = xp.exp(-1j * (omega * (xp.asarray(self.tau_s, dtype=float) - n0 * dt)
                                   + 0.5 * xp.asarray(self.beta2_l_s2, dtype=float) * omega ** 2))
            taps, _ = design_fir(nu, target, dt, n0=0, n_taps=n_taps, band=self.gvd_band)
            b = amp * taps
        else:
            kind, order = parse_interp(self.interp)
            if kind == "nearest":
                b = xp.reshape(amp, (1,))
            elif kind == "lagrange":
                b = amp * lagrange_taps(d - n0, order).astype(complex)
            else:
                a_all = thiran_coefficients(d - n0, order)
                b = amp * a_all[::-1].astype(complex)
                a_fb = a_all[1:].astype(complex)
        d0, n0c, bc, ac = split_instantaneous(n0, b, a_fb)
        p1, p2 = self.ports
        D = xp.zeros((2, 2), dtype=complex)
        if n0 == 0:
            D = xp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex) * d0
        chans: tuple[Channel, ...] = ()
        if bc is not None:
            chans = (Channel(p1, p2, n0c, bc, ac), Channel(p2, p1, n0c, bc, ac))
        return CompiledTD(self.ports, D, chans, structure=(n0, n_taps))

    def describe(self, dt: float) -> dict[str, Any]:
        out = {"type": "DelayLineTD", "ports": list(self.ports), "interp": self.interp}
        try:
            out["delay_samples"] = self.delay_samples(dt)
        except ValueError:
            pass
        return out


@dataclass(frozen=True)
class FIRTD(TDElement):
    """A linear element given by causal FIR taps per port pair, designed for one ``dt``.

    ``taps[(in, out)] = (n0, h)`` means ``b_out,k += sum_m h[m] a_in,{k-n0-m}``.
    Built by :func:`photonix.timedomain.to_td` from a sampled frequency
    response (least-squares fit with the bulk delay carried by ``n0``).

    Attributes
    ----------
    fit_error
        Maximum in-band absolute error of the fit (``nan`` if unknown).
    added_delay_s
        Latency added to make a non-causal model response causal.

    Examples
    --------
    >>> import numpy as np
    >>> el = FIRTD(("o1", "o2"), 1e-12, {("o1", "o2"): (2, np.array([0.5, 0.5]))})
    >>> c = el.compile(1e-12)
    >>> c.channels[0].n0, c.channels[0].dst
    (2, 'o2')
    """

    ports: tuple[str, ...]
    dt: float
    taps: Mapping[tuple[str, str], tuple[int, Any]]
    fit_error: float = float("nan")
    added_delay_s: float = 0.0
    design: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        for (p_in, p_out), (n0, _h) in self.taps.items():
            if p_in not in self.ports or p_out not in self.ports:
                raise ValueError(f"tap entry {(p_in, p_out)} references an unknown port")
            if int(n0) < 0:
                raise ValueError("tap offsets must be non-negative (causal)")

    def compile(self, dt: float) -> CompiledTD:
        if not math.isclose(dt, self.dt, rel_tol=1e-9):
            raise ValueError(f"this FIR was designed for dt = {self.dt:g} s, not {dt:g} s")
        idx = {p: i for i, p in enumerate(self.ports)}
        n = len(self.ports)
        D = xp.zeros((n, n), dtype=complex)
        chans = []
        for (p_in, p_out), (n0, h) in self.taps.items():
            d0, n0c, bc, ac = split_instantaneous(int(n0), _complex(h))
            if int(n0) == 0:
                D = D.at[idx[p_out], idx[p_in]].add(d0) if hasattr(D, "at") else _np_add(D, idx[p_out], idx[p_in], d0)
            if bc is not None:
                chans.append(Channel(p_in, p_out, n0c, bc, ac))
        return CompiledTD(self.ports, D, tuple(chans))

    def describe(self, dt: float) -> dict[str, Any]:
        n = max((int(np.shape(h)[0]) for _, h in self.taps.values()), default=0)
        return {"type": "FIRTD", "ports": list(self.ports), "max_taps": n, "fit_error": self.fit_error,
                "added_delay_s": self.added_delay_s}


def _np_add(S, i, j, v):
    S = np.array(S, copy=True)
    S[i, j] += v
    return S


def _phi12(z):
    """``phi1 = (e^z - 1)/z`` and ``phi2 = (e^z - 1 - z)/z^2``, stable at small ``|z|``."""
    z = _complex(z)
    small = xp.abs(z) < 1e-3
    zs = xp.where(small, xp.ones_like(z), z)
    ez = xp.exp(zs)
    phi1 = xp.where(small, 1.0 + z / 2.0 + z * z / 6.0 + z ** 3 / 24.0, (ez - 1.0) / zs)
    phi2 = xp.where(small, 0.5 + z / 6.0 + z * z / 24.0 + z ** 3 / 120.0, (ez - 1.0 - zs) / (zs * zs))
    return phi1, phi2


@dataclass(frozen=True)
class PoleResidueTD(TDElement):
    """A rational (pole-residue) linear element, discretized exactly between samples.

    Each port pair has ``H_ij(s) = d_ij + sum_p r_ijp / (s - p_p)`` in the
    baseband Laplace variable ``s = j 2 pi nu`` (poles in rad/s, ``Re p < 0``),
    optionally followed by an integer-sample delay ``delays[(i, j)]`` (s,
    rounded to whole samples). Each pole term is integrated exactly for a
    piecewise-linear input (``"foh"``, the recursive convolution of Gustavsen
    and Semlyen, second order in ``dt``) or a sample-and-hold input
    (``"zoh"``, first order)::

        foh:  s_k = e^{p dt} s_{k-1} + dt phi2(p dt) x_k + dt (phi1 - phi2)(p dt) x_{k-1}
        zoh:  s_k = e^{p dt} s_{k-1} + dt phi1(p dt) x_{k-1}

    with ``phi1(z) = (e^z - 1)/z`` and ``phi2(z) = (e^z - 1 - z)/z^2``. This is
    the bridge to vector-fitted macromodels.

    Examples
    --------
    >>> import numpy as np
    >>> lp = PoleResidueTD(("o1", "o2"), poles=[-1e11], residues={("o1", "o2"): [1e11]})
    >>> c = lp.compile(1e-12)        # a 15.9 GHz single-pole low-pass
    >>> len(c.channels), round(abs(complex(c.D[1, 0])), 6) > 0
    (1, True)
    """

    ports: tuple[str, ...]
    poles: Any
    residues: Mapping[tuple[str, str], Any]
    direct: Mapping[tuple[str, str], Any] = field(default_factory=dict)
    delays: Mapping[tuple[str, str], float] = field(default_factory=dict)
    discretization: str = "foh"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        if self.discretization not in ("foh", "zoh"):
            raise ValueError("discretization must be 'foh' or 'zoh'")
        n_p = int(np.shape(self.poles)[0])
        for key, r in self.residues.items():
            if key[0] not in self.ports or key[1] not in self.ports:
                raise ValueError(f"residue entry {key} references an unknown port")
            if int(np.shape(r)[0]) != n_p:
                raise ValueError(f"residues for {key} must have one value per pole")
        for key in list(self.direct) + list(self.delays):
            if key[0] not in self.ports or key[1] not in self.ports:
                raise ValueError(f"entry {key} references an unknown port")

    def response(self, nu):
        """The continuous-time response ``H(j 2 pi nu)`` per port pair (dict)."""
        s = 2j * np.pi * np.asarray(nu, dtype=float)
        poles = np.asarray(self.poles, dtype=complex)
        out = {}
        for key in set(self.residues) | set(self.direct):
            h = np.zeros_like(s) + complex(np.asarray(self.direct.get(key, 0.0)))
            if key in self.residues:
                r = np.asarray(self.residues[key], dtype=complex)
                h = h + np.sum(r[:, None] / (s[None, :] - poles[:, None]), axis=0)
            tau = self.delays.get(key, 0.0)
            out[key] = h * np.exp(-s * tau)
        return out

    def compile(self, dt: float) -> CompiledTD:
        idx = {p: i for i, p in enumerate(self.ports)}
        n = len(self.ports)
        D = xp.zeros((n, n), dtype=complex)
        poles = _complex(self.poles)
        z = poles * dt
        lam = xp.exp(z)
        phi1, phi2 = _phi12(z)
        chans: list[Channel] = []
        for key in set(self.residues) | set(self.direct):
            p_in, p_out = key
            nd = int(round(float(self.delays.get(key, 0.0)) / dt))
            terms = []
            if key in self.direct:
                terms.append((nd, xp.reshape(_complex(self.direct[key]), (1,)), None))
            if key in self.residues:
                r = _complex(self.residues[key])
                for p in range(int(poles.shape[0])):
                    a_fb = xp.reshape(-lam[p], (1,))
                    if self.discretization == "foh":
                        b = xp.stack([r[p] * dt * phi2[p], r[p] * dt * (phi1[p] - phi2[p])])
                        terms.append((nd, b, a_fb))
                    else:
                        terms.append((nd + 1, xp.reshape(r[p] * dt * phi1[p], (1,)), a_fb))
            for n0, b, a_fb in terms:
                d0, n0c, bc, ac = split_instantaneous(n0, b, a_fb)
                if n0 == 0:
                    D = D.at[idx[p_out], idx[p_in]].add(d0) if hasattr(D, "at") else _np_add(D, idx[p_out], idx[p_in], d0)
                if bc is not None:
                    chans.append(Channel(p_in, p_out, n0c, bc, ac))
        return CompiledTD(self.ports, D, tuple(chans))


@dataclass(frozen=True)
class TimeVaryingTD(TDElement):
    """A memoryless element whose scattering matrix follows electrical controls in time.

    ``fn(t, **controls)`` receives the time axis ``t`` (s, shape ``(n_steps,)``)
    and the named control waveforms (same shape) and returns either an array
    ``(n_steps, n, n)`` (rows = outgoing port) or a scattering dictionary
    whose entries have shape ``(n_steps,)``. The matrix is evaluated once for
    the whole record before the time stepping, so ``fn`` may call any
    vectorized photonix model (the quasi-static approximation: the device
    responds instantly to its controls).

    Examples
    --------
    >>> import numpy as np
    >>> pm = TimeVaryingTD(("o1", "o2"), lambda t, v: {("o1", "o2"): np.exp(-1j * v), ("o2", "o1"): np.exp(-1j * v)},
    ...                    controls=("v",))
    >>> pm.matrices(np.zeros(3), {"v": np.array([0.0, np.pi, 0.0])}).shape
    (3, 2, 2)
    """

    ports: tuple[str, ...]
    fn: Callable[..., Any]
    controls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        object.__setattr__(self, "controls", tuple(self.controls))

    @classmethod
    def from_model(cls, model: Callable[..., Any], settings: Mapping[str, Any] | None = None, *,
                   control_params: Mapping[str, str], carrier_wl: float = 1.55,
                   ports: Sequence[str] | None = None) -> TimeVaryingTD:
        """Quasi-static element from any photonix model with control-driven settings.

        ``control_params`` maps model parameter names to control names, e.g.
        ``{"voltage": "v_rf"}``; every call evaluates ``model(wl=carrier_wl,
        **settings, voltage=v_rf(t))`` vectorized over time.
        """
        from photonix.core.types import ports_of

        base = dict(settings or {})
        names = tuple(ports) if ports is not None else tuple(ports_of(model(wl=carrier_wl, **base)))
        mapping = dict(control_params)

        def fn(t, **controls):
            kw = dict(base)
            for param, cname in mapping.items():
                kw[param] = controls[cname]
            return model(wl=carrier_wl, **kw)

        return cls(names, fn, tuple(mapping.values()))

    def matrices(self, t, controls: Mapping[str, Any]):
        """``(n_steps, n, n)`` scattering matrices over the time axis ``t``."""
        missing = [c for c in self.controls if c not in controls]
        if missing:
            raise KeyError(f"missing control waveform(s) {missing} for a time-varying element")
        out = self.fn(t, **{c: controls[c] for c in self.controls})
        n_steps = int(np.shape(t)[0])
        n = len(self.ports)
        if isinstance(out, Mapping):
            idx = {p: i for i, p in enumerate(self.ports)}
            rows = [[xp.zeros(n_steps, dtype=complex) for _ in range(n)] for _ in range(n)]
            for (p_in, p_out), v in out.items():
                if p_in in idx and p_out in idx:
                    rows[idx[p_out]][idx[p_in]] = xp.broadcast_to(_complex(v), (n_steps,))
            mat = xp.stack([xp.stack(r, axis=-1) for r in rows], axis=-2)
        else:
            mat = xp.broadcast_to(_complex(out), (n_steps, n, n))
        return mat

    def compile(self, dt: float) -> CompiledTD:
        n = len(self.ports)
        return CompiledTD(self.ports, xp.zeros((n, n), dtype=complex), (), time_varying=self)


@dataclass(frozen=True)
class CompositeTD(TDElement):
    """A sub-network of elements, flattened into the enclosing circuit.

    ``elements`` maps sub-element names to elements, ``connections`` joins
    ``(sub, port)`` terminals, and ``port_map`` names the composite's ports.
    Sub-elements appear in the circuit as ``"<instance>/<sub>"``.

    Examples
    --------
    >>> comp = CompositeTD({"a": DelayLineTD(1e-12), "b": DelayLineTD(2e-12)},
    ...                    {("a", "o2"): ("b", "o1")}, {"o1": ("a", "o1"), "o2": ("b", "o2")})
    >>> comp.ports
    ('o1', 'o2')
    """

    elements: Mapping[str, TDElement]
    connections: Mapping[tuple[str, str], tuple[str, str]]
    port_map: Mapping[str, tuple[str, str]]

    @property
    def ports(self) -> tuple[str, ...]:  # type: ignore[override]
        return tuple(self.port_map)

    def compile(self, dt: float) -> CompiledTD:  # pragma: no cover - flattened by the network
        raise TypeError("a CompositeTD is flattened by TimeDomainCircuit and is never compiled directly")

    def frozen(self, dt: float) -> CompositeTD:
        return dataclasses.replace(self, elements={k: v.frozen(dt) for k, v in self.elements.items()})
