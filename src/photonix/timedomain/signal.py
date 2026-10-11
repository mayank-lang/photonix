"""Sampled time-domain signals: optical complex envelopes and electrical waveforms.

An optical signal is the complex envelope ``A(t)`` of the field about a carrier
of wavelength ``carrier_wl`` (µm), normalized so that ``|A|^2`` is the
instantaneous power in watts. The physical field is ``Re[A(t) exp(+j w_c t)]``,
the ``exp(+j omega t)`` convention of the rest of photonix, so a component with
transfer function ``H(f)`` acts on the envelope spectrum at offsets
``nu = f - f_c`` and a pure delay ``tau`` is ``H = exp(-j 2 pi f tau)``.

An electrical signal (``carrier_wl=None``) is a real baseband waveform in the
unit named by :attr:`Waveform.unit` (``"V"`` or ``"A"``).

Samples are uniform with step ``dt`` seconds and start at ``t0``; the time axis
is ``t0 + k dt``. A sampled envelope represents optical frequencies within
``f_c +/- 1 / (2 dt)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from photonix.core.backend import HAS_JAX, xp
from photonix.core.constants import C0_UM_S

__all__ = ["Waveform", "cw", "gaussian_pulse", "carrier_frequency"]


def carrier_frequency(carrier_wl: float) -> float:
    """Optical frequency in Hz of a vacuum wavelength in µm.

    Examples
    --------
    >>> round(carrier_frequency(1.55) / 1e12, 4)
    193.4145
    """
    return float(C0_UM_S / float(carrier_wl))


@dataclass(frozen=True)
class Waveform:
    """A uniformly sampled signal.

    Parameters
    ----------
    samples
        Array of shape ``(n_steps,)`` (or ``(n_steps, ...)`` for several
        channels). For an optical signal, the complex envelope in sqrt(W); for
        an electrical signal, real values in :attr:`unit`.
    dt
        Sample spacing in seconds.
    t0
        Time of the first sample in seconds.
    carrier_wl
        Carrier wavelength in µm for an optical envelope; ``None`` for an
        electrical (real baseband) waveform.
    unit
        Unit of the samples: ``"sqrt(W)"`` (optical), ``"V"``, ``"A"`` or any
        label.

    Notes
    -----
    Under JAX the class is a pytree whose only leaf is :attr:`samples`, so a
    :class:`Waveform` can be returned from ``jit``/``grad``-transformed
    functions.

    Examples
    --------
    >>> w = cw(1e-3, 8, 1e-12)
    >>> w.n_steps, round(float(w.power()[0]) * 1e3, 12), round(w.energy() * 1e15, 9)
    (8, 1.0, 8.0)
    """

    samples: Any
    dt: float
    t0: float = 0.0
    carrier_wl: float | None = 1.55
    unit: str = "sqrt(W)"

    def __post_init__(self) -> None:
        if isinstance(self.samples, (list, tuple, int, float, complex)):
            object.__setattr__(self, "samples", xp.asarray(self.samples))
        if not (isinstance(self.dt, (int, float, np.floating)) and math.isfinite(self.dt) and self.dt > 0):
            raise ValueError(f"dt must be a positive finite number of seconds, got {self.dt!r}")
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "t0", float(self.t0))
        if self.carrier_wl is not None:
            if not (float(self.carrier_wl) > 0):
                raise ValueError("carrier_wl must be a positive wavelength in µm (or None)")
            object.__setattr__(self, "carrier_wl", float(self.carrier_wl))
        shape = getattr(self.samples, "shape", None)
        if shape is not None and len(shape) == 0:
            raise ValueError("samples must have at least one dimension (time)")

    # -- sampling ------------------------------------------------------------ #
    @property
    def n_steps(self) -> int:
        """Number of time samples."""
        return int(self.samples.shape[0])

    @property
    def fs(self) -> float:
        """Sample rate in Hz."""
        return 1.0 / self.dt

    @property
    def duration(self) -> float:
        """Record length ``n_steps * dt`` in seconds."""
        return self.n_steps * self.dt

    @property
    def t(self) -> np.ndarray:
        """Time axis ``t0 + k dt`` in seconds (NumPy)."""
        return self.t0 + self.dt * np.arange(self.n_steps)

    @property
    def is_optical(self) -> bool:
        """True for a complex optical envelope (a carrier is defined)."""
        return self.carrier_wl is not None

    @property
    def carrier_frequency(self) -> float:
        """Carrier frequency in Hz (optical signals only)."""
        if self.carrier_wl is None:
            raise ValueError("an electrical waveform has no optical carrier")
        return carrier_frequency(self.carrier_wl)

    # -- derived quantities ---------------------------------------------------- #
    def power(self):
        """Instantaneous power ``|A|^2`` in W (optical) or ``|x|^2`` in unit^2."""
        return xp.abs(self.samples) ** 2

    def energy(self) -> float:
        """``sum |A|^2 dt`` in J (optical) over the whole record."""
        return float(np.sum(np.abs(np.asarray(self.samples)) ** 2) * self.dt)

    def mean_power(self) -> float:
        """Time-averaged power in W."""
        return float(np.mean(np.abs(np.asarray(self.samples)) ** 2))

    def spectrum(self):
        """Centred spectrum ``(nu, X)`` of the samples.

        ``nu`` is the frequency offset from the carrier in Hz (ascending) and
        ``X(nu) = dt * sum_k x_k exp(-j 2 pi nu t_k)`` approximates the
        continuous Fourier transform ``int x(t) exp(-j 2 pi nu t) dt``, so a
        component's transfer function multiplies ``X`` directly.

        Examples
        --------
        >>> import numpy as np
        >>> w = cw(1.0, 64, 1e-12, offset_hz=1 / 64e-12 * 3)   # 3 bins above the carrier
        >>> nu, X = w.spectrum()
        >>> round(float(nu[np.argmax(np.abs(X))] * 64e-12), 6)
        3.0
        """
        x = np.asarray(self.samples)
        n = x.shape[0]
        nu = np.fft.fftshift(np.fft.fftfreq(n, self.dt))
        phase = np.exp(-2j * np.pi * nu * self.t0)
        shape = (n,) + (1,) * (x.ndim - 1)
        spec = np.fft.fftshift(np.fft.fft(x, axis=0), axes=0) * self.dt * phase.reshape(shape)
        return nu, spec

    def with_samples(self, samples, *, unit: str | None = None, carrier_wl: Any = "keep") -> Waveform:
        """A copy with new ``samples`` (same time axis)."""
        kw: dict[str, Any] = {"samples": samples}
        if unit is not None:
            kw["unit"] = unit
        if carrier_wl != "keep":
            kw["carrier_wl"] = carrier_wl
        return replace(self, **kw)

    def __len__(self) -> int:
        return self.n_steps


if HAS_JAX:  # pragma: no branch - registration only matters under JAX
    from jax import tree_util as _tree_util

    _tree_util.register_dataclass(Waveform, data_fields=["samples"], meta_fields=["dt", "t0", "carrier_wl", "unit"])


def cw(
    power_w: float,
    n_steps: int,
    dt: float,
    *,
    carrier_wl: float = 1.55,
    offset_hz: float = 0.0,
    phase: float = 0.0,
    t0: float = 0.0,
) -> Waveform:
    """A continuous-wave optical envelope of constant power.

    ``offset_hz`` detunes the tone from the carrier: the envelope is
    ``sqrt(P) exp(j (2 pi offset_hz t + phase))``, i.e. light at optical
    frequency ``f_c + offset_hz``.

    Examples
    --------
    >>> w = cw(2e-3, 4, 1e-12)
    >>> [round(float(p) * 1e3, 12) for p in w.power()]
    [2.0, 2.0, 2.0, 2.0]
    """
    if power_w < 0:
        raise ValueError("power_w must be non-negative")
    n = int(n_steps)
    if n < 1:
        raise ValueError("n_steps must be at least 1")
    t = t0 + dt * np.arange(n)
    samples = math.sqrt(power_w) * np.exp(1j * (2.0 * np.pi * offset_hz * t + phase))
    return Waveform(xp.asarray(samples), dt, t0, carrier_wl, "sqrt(W)")


def gaussian_pulse(
    fwhm_s: float,
    n_steps: int,
    dt: float,
    *,
    peak_power_w: float = 1e-3,
    t_center: float | None = None,
    chirp: float = 0.0,
    carrier_wl: float = 1.55,
    offset_hz: float = 0.0,
    t0: float = 0.0,
) -> Waveform:
    """A Gaussian optical pulse of intensity full width at half maximum ``fwhm_s``.

    The envelope is ``sqrt(P0) exp(-(1 + j C) (t - t_c)^2 / (2 T0^2))`` with
    ``T0 = fwhm / (2 sqrt(ln 2))`` the 1/e intensity half-width (Agrawal's
    convention, so ``T0`` is the quantity in the dispersive-broadening law) and
    ``C = chirp``. ``t_center`` defaults to ``t0 + 4 fwhm``.

    Examples
    --------
    >>> import numpy as np
    >>> p = gaussian_pulse(1e-12, 400, 0.025e-12)
    >>> P = np.asarray(p.power())
    >>> above = p.t[P >= 0.5 * P.max()]
    >>> bool(abs((above[-1] - above[0]) - 1e-12) < 0.05e-12)   # FWHM to one sample
    True
    """
    if fwhm_s <= 0:
        raise ValueError("fwhm_s must be positive")
    n = int(n_steps)
    t = t0 + dt * np.arange(n)
    tc = t0 + 4.0 * fwhm_s if t_center is None else float(t_center)
    t_e = fwhm_s / (2.0 * math.sqrt(math.log(2.0)))
    x = (t - tc) / t_e
    samples = math.sqrt(peak_power_w) * np.exp(-(1.0 + 1j * chirp) * x * x / 2.0 + 2j * np.pi * offset_hz * t)
    return Waveform(xp.asarray(samples), dt, t0, carrier_wl, "sqrt(W)")
