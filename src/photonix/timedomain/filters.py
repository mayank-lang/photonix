"""Electrical (baseband) linear filters for transmitters and receivers.

An :class:`AnalogFilter` is a continuous-time rational transfer function in
pole-residue form,

.. code-block:: text

    H(s) = d + sum_i r_i / (s - p_i),        s = j 2 pi f,   Re p_i < 0,

with poles in rad/s. Prototypes cover the usual receiver and driver
bandwidth models: a first-order RC low-pass, Butterworth and Bessel-Thomson
(the reference receiver of IEEE 802.3 / ITU-T G.691 eye masks) of any order,
all normalized to a -3 dB frequency.

Filtering a sampled waveform integrates every pole term *exactly* for a
piecewise-linear input (first-order hold, the recursive convolution of
Gustavsen and Semlyen) or a sample-and-hold input (zero-order hold), so the
step and pulse responses of the discrete filter equal the analytic ones at
the sample instants. The implementation uses the backend array module and is
differentiable (JAX ``associative_scan``; SciPy ``lfilter`` under NumPy).

References
----------
* W. E. Thomson, "Delay networks having maximally flat frequency
  characteristics," Proc. IEE 96(44), 487 (1949) -- Bessel filters.
* B. Gustavsen, A. Semlyen, IEEE Trans. Power Del. 13(2), 605 (1998) --
  recursive convolution.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from photonix.core.backend import HAS_JAX, xp

from .signal import Waveform

__all__ = ["AnalogFilter", "lowpass", "linear_recurrence"]


def linear_recurrence(lam, v, y0=0.0):
    """Solve ``y_k = lam_k y_{k-1} + v_k`` (``y_{-1} = y0``) along the first axis.

    ``lam`` may be a scalar or an array broadcastable to ``v``. Under JAX this
    is a parallel :func:`jax.lax.associative_scan` (differentiable); under
    NumPy, :func:`scipy.signal.lfilter` for a constant ``lam`` and a chunked
    cumulative-product solve otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> [round(float(v), 12) for v in np.asarray(linear_recurrence(0.5, np.ones(4)))]
    [1.0, 1.5, 1.75, 1.875]
    """
    v = xp.asarray(v)
    dtype = xp.result_type(v, xp.asarray(lam), xp.asarray(y0))
    v = v.astype(dtype)
    lam = xp.broadcast_to(xp.asarray(lam, dtype=dtype), v.shape)
    v = v.at[0].add(lam[0] * y0) if hasattr(v, "at") else _np_first_add(v, lam[0] * y0)
    if HAS_JAX:
        from jax import lax

        def combine(a, b):
            return a[0] * b[0], b[0] * a[1] + b[1]

        return lax.associative_scan(combine, (lam, v), axis=0)[1]
    lam_n = np.asarray(lam)
    v_n = np.asarray(v)
    if lam_n.ndim == 1 and np.all(lam_n == lam_n[0]):
        from scipy.signal import lfilter

        return lfilter([1.0], [1.0, -lam_n[0]], v_n)
    out = np.empty_like(v_n)
    y = np.zeros(v_n.shape[1:], dtype=v_n.dtype)
    chunk = 64
    for start in range(0, v_n.shape[0], chunk):
        lam_c = lam_n[start:start + chunk]
        cp = np.cumprod(lam_c, axis=0)
        if np.min(np.abs(cp)) < 1e-250:
            for k in range(lam_c.shape[0]):
                y = lam_c[k] * y + v_n[start + k]
                out[start + k] = y
            continue
        out[start:start + chunk] = cp * (y + np.cumsum(v_n[start:start + chunk] / cp, axis=0))
        y = out[start + lam_c.shape[0] - 1]
    return out


def _np_first_add(v, add):
    v = np.array(v, copy=True)
    v[0] = v[0] + add
    return v


def _phi12(z: np.ndarray):
    """``phi1 = (e^z - 1)/z``, ``phi2 = (e^z - 1 - z)/z^2`` (series near 0)."""
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < 1e-3
    zs = np.where(small, 1.0, z)
    phi1 = np.where(small, 1 + z / 2 + z * z / 6 + z ** 3 / 24, (np.exp(zs) - 1) / zs)
    phi2 = np.where(small, 0.5 + z / 6 + z * z / 24 + z ** 3 / 120, (np.exp(zs) - 1 - zs) / (zs * zs))
    return phi1, phi2


def _bessel_poles(order: int) -> np.ndarray:
    from scipy.signal import bessel

    _, p, _ = bessel(order, 1.0, analog=True, output="zpk", norm="mag")
    return np.asarray(p, dtype=complex)


def _butter_poles(order: int) -> np.ndarray:
    k = np.arange(order)
    return np.exp(1j * np.pi * (2 * k + order + 1) / (2 * order))


def _all_pole_residues(poles: np.ndarray, gain: complex) -> np.ndarray:
    """Residues of ``gain / prod(s - p_i)`` (simple poles)."""
    out = []
    for i, p in enumerate(poles):
        others = np.delete(poles, i)
        out.append(gain / np.prod(p - others))
    return np.asarray(out, dtype=complex)


@dataclass(frozen=True)
class AnalogFilter:
    """A continuous-time rational filter ``H(s) = direct + sum r_i / (s - p_i)``.

    Parameters
    ----------
    poles
        Poles in rad/s (``Re p < 0``; complex poles in conjugate pairs for a
        real filter).
    residues
        One residue per pole (rad/s).
    direct
        Feed-through term ``d``.
    name
        Label used by :meth:`describe`.

    Examples
    --------
    >>> import numpy as np
    >>> lp = AnalogFilter.first_order(10e9)
    >>> round(abs(complex(lp.response(10e9))) ** 2, 12)       # -3 dB at f3db
    0.5
    >>> round(lp.noise_bandwidth_hz() / 10e9, 6)               # pi / 2 for one pole
    1.570796
    """

    poles: Any
    residues: Any
    direct: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        p = np.atleast_1d(np.asarray(self.poles, dtype=complex))
        r = np.atleast_1d(np.asarray(self.residues, dtype=complex))
        if p.shape != r.shape:
            raise ValueError("poles and residues must have the same length")
        if np.any(p.real >= 0):
            raise ValueError("every pole must lie in the left half plane (a stable filter)")
        object.__setattr__(self, "poles", tuple(complex(v) for v in p))
        object.__setattr__(self, "residues", tuple(complex(v) for v in r))

    # -- prototypes ------------------------------------------------------------ #
    @classmethod
    def first_order(cls, f3db_hz: float) -> AnalogFilter:
        """Single-pole (RC) low-pass with unit DC gain and -3 dB at ``f3db_hz``."""
        if f3db_hz <= 0:
            raise ValueError("f3db_hz must be positive")
        w = 2.0 * math.pi * f3db_hz
        return cls((-w,), (w,), 0.0, f"RC {f3db_hz:.3g} Hz")

    @classmethod
    def butterworth(cls, order: int, f3db_hz: float) -> AnalogFilter:
        """Maximally flat magnitude low-pass of ``order`` (unit DC gain, -3 dB at ``f3db_hz``)."""
        p = _butter_poles(int(order)) * 2.0 * math.pi * f3db_hz
        gain = np.real(np.prod(-p))
        return cls(p, _all_pole_residues(p, gain), 0.0, f"Butterworth-{order} {f3db_hz:.3g} Hz")

    @classmethod
    def bessel(cls, order: int, f3db_hz: float) -> AnalogFilter:
        """Bessel-Thomson (maximally flat group delay) low-pass, -3 dB at ``f3db_hz``.

        The 4th-order filter at ``0.75 x`` the bit rate is the standard
        reference receiver for NRZ eye measurements.
        """
        if int(order) < 1:
            raise ValueError("order must be at least 1")
        p = _bessel_poles(int(order)) * 2.0 * math.pi * f3db_hz
        gain = np.real(np.prod(-p))
        return cls(p, _all_pole_residues(p, gain), 0.0, f"Bessel-{order} {f3db_hz:.3g} Hz")

    # -- frequency domain -------------------------------------------------- #
    def response(self, f_hz):
        """``H(j 2 pi f)`` at frequencies ``f_hz`` (Hz)."""
        s = 2j * np.pi * np.asarray(f_hz, dtype=float)
        p = np.asarray(self.poles)
        r = np.asarray(self.residues)
        return self.direct + np.sum(r[:, None] / (np.atleast_1d(s)[None, :] - p[:, None]), axis=0).reshape(s.shape)

    @property
    def dc_gain(self) -> float:
        """``|H(0)|``."""
        return float(abs(complex(self.response(0.0))))

    def f3db_hz(self) -> float:
        """The -3 dB frequency found numerically (bisection on ``|H|^2 = |H(0)|^2 / 2``)."""
        from scipy.optimize import brentq

        g0 = self.dc_gain ** 2
        hi = max(abs(p) for p in self.poles) / (2 * math.pi) * 10.0
        return float(brentq(lambda f: abs(complex(self.response(f))) ** 2 - 0.5 * g0, 0.0, hi, xtol=1e-6))

    def noise_bandwidth_hz(self) -> float:
        """Noise-equivalent bandwidth ``int_0^inf |H(f)|^2 df / |H(0)|^2`` (closed form).

        Uses ``int_0^inf |h(t)|^2 dt = -sum_ij r_i r_j^* / (p_i + p_j^*)`` and
        Parseval; only strictly proper filters (``direct = 0``) have a finite
        noise bandwidth.
        """
        if self.direct != 0.0:
            raise ValueError("a filter with a direct term has an infinite noise bandwidth")
        p = np.asarray(self.poles)
        r = np.asarray(self.residues)
        energy = float(np.real(-np.sum(r[:, None] * np.conj(r)[None, :] / (p[:, None] + np.conj(p)[None, :]))))
        return 0.5 * energy / self.dc_gain ** 2

    def step_response(self, t):
        """Analytic unit-step response ``d + sum r_i (exp(p_i t) - 1) / p_i`` for ``t >= 0``."""
        t = np.asarray(t, dtype=float)
        p = np.asarray(self.poles)
        r = np.asarray(self.residues)
        y = self.direct + np.sum((r / p)[:, None] * (np.exp(np.outer(p, np.clip(t, 0, None)).reshape(p.size, -1))
                                                     - 1.0), axis=0)
        return np.where(t.reshape(-1) >= 0, y.real, 0.0).reshape(t.shape)

    def rise_time_s(self, lo: float = 0.1, hi: float = 0.9) -> float:
        """``lo``-``hi`` (default 10-90 %) rise time of the step response (s)."""
        from scipy.optimize import brentq

        final = self.dc_gain
        t_max = 50.0 / min(abs(p.real) for p in self.poles)

        def cross(level):
            return brentq(lambda t: float(self.step_response(t)) - level * final, 0.0, t_max, xtol=1e-18)

        return float(cross(hi) - cross(lo))

    def describe(self) -> dict[str, Any]:
        return {"type": "AnalogFilter", "name": self.name, "order": len(self.poles)}

    # -- time domain ------------------------------------------------------------ #
    def apply(self, signal, dt: float | None = None, *, discretization: Literal["foh", "zoh"] = "foh"):
        """Filter a sampled signal (a :class:`Waveform` or an array along axis 0).

        ``"foh"`` (default) is exact for a piecewise-linear input, ``"zoh"``
        for a piecewise-constant (sample-and-hold) input. The filter starts at
        rest; a real input to a real filter gives a real output.

        Examples
        --------
        >>> import numpy as np
        >>> lp = AnalogFilter.first_order(1e9)
        >>> y = lp.apply(np.ones(2000), 1e-12, discretization="zoh")
        >>> t = 1e-12 * np.arange(2000)                 # a step applied (and held) from t = 0
        >>> bool(np.allclose(np.asarray(y), lp.step_response(t), atol=1e-12))
        True
        """
        if discretization not in ("foh", "zoh"):
            raise ValueError("discretization must be 'foh' or 'zoh'")
        wf = signal if isinstance(signal, Waveform) else None
        if wf is not None:
            dt = wf.dt
            x = wf.samples
        else:
            x = signal
            if dt is None:
                raise ValueError("give dt for an array input")
        x = xp.asarray(x)
        real_in = not xp.iscomplexobj(x)
        xc = x.astype(complex)
        p = np.asarray(self.poles)
        r = np.asarray(self.residues)
        z = p * dt
        lam = np.exp(z)
        phi1, phi2 = _phi12(z)
        prev = xp.concatenate([xp.zeros_like(xc[:1]), xc[:-1]], axis=0)
        y = self.direct * xc
        for i in range(p.size):
            if discretization == "foh":
                v = r[i] * dt * (phi2[i] * xc + (phi1[i] - phi2[i]) * prev)
            else:                                # x_{k-1} held over [t_{k-1}, t_k)
                v = r[i] * dt * phi1[i] * prev
            y = y + linear_recurrence(lam[i], v)
        if real_in and self._is_real():
            y = xp.real(y)
        if wf is not None:
            return wf.with_samples(y)
        return y

    def _is_real(self) -> bool:
        p = sorted(self.poles, key=lambda c: (round(c.real, 6), round(c.imag, 6)))
        conj = sorted((c.conjugate() for c in self.poles), key=lambda c: (round(c.real, 6), round(c.imag, 6)))
        return bool(np.allclose(p, conj, rtol=1e-9, atol=0)) and abs(np.imag(self.direct)) == 0


def lowpass(signal, f3db_hz: float, *, kind: Literal["rc", "bessel", "butterworth"] = "bessel", order: int = 4,
            dt: float | None = None, discretization: Literal["foh", "zoh"] = "foh"):
    """Low-pass filter ``signal`` (a :class:`Waveform` or array) with -3 dB at ``f3db_hz``.

    Examples
    --------
    >>> import numpy as np
    >>> y = lowpass(np.ones(500), 20e9, kind="rc", dt=1e-12)
    >>> round(float(np.asarray(y)[-1]), 9)
    1.0
    """
    if kind == "rc":
        flt = AnalogFilter.first_order(f3db_hz)
    elif kind == "bessel":
        flt = AnalogFilter.bessel(order, f3db_hz)
    elif kind == "butterworth":
        flt = AnalogFilter.butterworth(order, f3db_hz)
    else:
        raise ValueError("kind must be 'rc', 'bessel' or 'butterworth'")
    return flt.apply(signal, dt, discretization=discretization)


def _as_filter(spec: AnalogFilter | float | None, kind: str = "rc", order: int = 1) -> AnalogFilter | None:
    if spec is None or isinstance(spec, AnalogFilter):
        return spec
    f = float(spec)
    if kind == "rc" or order == 1:
        return AnalogFilter.first_order(f)
    if kind == "bessel":
        return AnalogFilter.bessel(order, f)
    return AnalogFilter.butterworth(order, f)


def _check_sequence(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)
