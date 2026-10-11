"""Optical receivers: photodiode with noise, transimpedance amplifier, clock recovery.

Noise is generated *before* the receiver filter, as white Gaussian samples
whose variance is the one-sided power spectral density times the simulation
bandwidth ``f_s / 2`` (band-limited white noise), and the filter then sets the
physical noise bandwidth. The noise sequence is filtered *exactly within the
simulation band* (multiplication by ``H(f)`` on the DFT grid), so its output
spectrum is ``N0 |H(f)|^2`` for ``|f| < f_s / 2``; the deterministic signal
goes through the exactly discretized filter
(:meth:`~photonix.timedomain.filters.AnalogFilter.apply`). (Passing white
samples through the first-order-hold recursion instead would read them as a
piecewise-linear signal and shape the noise by ``sinc^4(f dt)``, under-
estimating the noise bandwidth by ``~ (2/3) (pi f_3dB dt)^2``.) For a filter
with noise-equivalent bandwidth ``B_n``
(:meth:`~photonix.timedomain.filters.AnalogFilter.noise_bandwidth_hz`) the
output variances are the textbook ones (Agrawal, *Fiber-Optic Communication
Systems*, 4th ed., Sec. 4.4):

* shot noise ``sigma_s^2 = 2 q (I + I_d) B_n`` (signal dependent, evaluated
  sample by sample),
* thermal noise ``sigma_T^2 = (4 k_B T / R_L) B_n``,
* receiver-referred intensity noise ``RIN (R P)^2 B_n``,
* TIA input-referred noise ``i_n^2 B_n``,

less the (normally negligible) part of ``B_n`` beyond ``f_s / 2``: for a
first-order RC response that fraction is ``(2 / pi) f_3dB / (f_s / 2)``, so
sample at least ~50x the RC bandwidth or use a steeper receiver filter.
The DFT filtering is circular: the first ~filter-memory of noise samples is
correlated with the last ones (statistically harmless; there is no start-up
transient in the noise).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from photonix.core.backend import xp
from photonix.core.constants import Q_E

from .filters import AnalogFilter
from .signal import Waveform

__all__ = ["photodiode", "tia", "recover_clock", "K_BOLTZMANN"]

#: Boltzmann constant (J/K), exact (SI 2019).
K_BOLTZMANN = 1.380649e-23


def _band_limited_noise(std, flt: AnalogFilter | None, dt: float, rng: np.random.Generator):
    """White Gaussian samples of standard deviation ``std`` (per sample), filtered exactly in band by ``flt``."""
    n = int(np.shape(std)[0]) if np.ndim(std) else None
    if n is None:
        raise ValueError("std must be an array over time")
    noise = std * xp.asarray(rng.standard_normal(n))
    if flt is None:
        return noise
    f = np.fft.fftfreq(n, dt)
    return xp.real(xp.fft.ifft(xp.fft.fft(noise) * xp.asarray(flt.response(f))))


def _filter_from(spec: AnalogFilter | float | None, kind: str, order: int) -> AnalogFilter | None:
    if spec is None or isinstance(spec, AnalogFilter):
        return spec
    f = float(spec)
    if kind == "rc" or order == 1:
        return AnalogFilter.first_order(f)
    if kind == "bessel":
        return AnalogFilter.bessel(order, f)
    if kind == "butterworth":
        return AnalogFilter.butterworth(order, f)
    raise ValueError("kind must be 'rc', 'bessel' or 'butterworth'")


def photodiode(
    field_in: Waveform,
    *,
    responsivity_a_w: float | None = None,
    dark_current_a: float = 0.0,
    bandwidth: AnalogFilter | float | None = None,
    kind: Literal["rc", "bessel", "butterworth"] = "rc",
    order: int = 1,
    load_ohm: float = 50.0,
    temperature_k: float = 300.0,
    noise: Sequence[str] = ("shot", "thermal"),
    rin_db_hz: float | None = None,
    seed: Any = None,
) -> Waveform:
    """Photocurrent of a PIN photodiode, with shot / thermal / intensity noise.

    ``I = R |E|^2 + I_d`` plus the selected noise terms (``"shot"``,
    ``"thermal"``; ``rin_db_hz`` adds receiver-referred RIN), followed by the
    receiver filter ``bandwidth`` (an :class:`AnalogFilter`, or a -3 dB
    frequency for a ``kind``/``order`` prototype; first-order RC by default).
    The signal is filtered with the exact (first-order-hold) discretization
    and the white noise exactly within the simulation band, so the output
    noise variance is the textbook one for the filter's noise bandwidth (see
    the module notes). ``responsivity_a_w`` defaults to
    :func:`photonix.components.responsivity` at the carrier (80 % quantum
    efficiency).

    Returns
    -------
    Waveform
        Current in A (``unit="A"``, electrical).

    Examples
    --------
    >>> from photonix.timedomain import cw
    >>> i = photodiode(cw(1e-3, 8, 1e-12), responsivity_a_w=1.0, noise=())
    >>> i.unit, round(float(i.samples[0]) * 1e3, 12)
    ('A', 1.0)
    """
    if not isinstance(field_in, Waveform) or not field_in.is_optical:
        raise TypeError("photodiode expects an optical Waveform")
    unknown = set(noise) - {"shot", "thermal"}
    if unknown:
        raise ValueError(f"unknown noise sources {sorted(unknown)}; use 'shot' and/or 'thermal'")
    if responsivity_a_w is None:
        from photonix.components import responsivity

        responsivity_a_w = float(responsivity(field_in.carrier_wl))
    dt = field_in.dt
    p = xp.abs(field_in.samples) ** 2
    i_sig = responsivity_a_w * p
    current = i_sig + dark_current_a
    rng = np.random.default_rng(seed)
    n = field_in.n_steps
    half_fs = 0.5 / dt
    var = xp.zeros(n)
    if "shot" in noise:
        var = var + 2.0 * Q_E * xp.maximum(current, 0.0) * half_fs
    if "thermal" in noise:
        var = var + 4.0 * K_BOLTZMANN * temperature_k / load_ohm * half_fs
    if rin_db_hz is not None:
        var = var + 10.0 ** (rin_db_hz / 10.0) * i_sig ** 2 * half_fs
    flt = _filter_from(bandwidth, kind, order)
    if flt is not None:
        current = flt.apply(current, dt)
    if noise or rin_db_hz is not None:
        current = current + _band_limited_noise(xp.sqrt(var), flt, dt, rng)
    return Waveform(current, dt, field_in.t0, None, "A")


def tia(
    current: Waveform,
    *,
    gain_ohm: float = 1000.0,
    bandwidth: AnalogFilter | float | None = None,
    kind: Literal["rc", "bessel", "butterworth"] = "bessel",
    order: int = 4,
    input_noise_a_rthz: float = 0.0,
    seed: Any = None,
) -> Waveform:
    """Transimpedance amplifier: ``V = Z_T * filter(I + i_n)``.

    ``bandwidth`` is an :class:`AnalogFilter` or a -3 dB frequency for the
    ``kind``/``order`` prototype (default 4th-order Bessel-Thomson);
    ``input_noise_a_rthz`` is the input-referred noise current density
    (A/sqrt(Hz), one-sided, white).

    Examples
    --------
    >>> import numpy as np
    >>> v = tia(Waveform(np.full(4, 1e-4), 1e-12, carrier_wl=None, unit="A"), gain_ohm=500.0)
    >>> v.unit, round(float(v.samples[0]), 12)
    ('V', 0.05)
    """
    if not isinstance(current, Waveform) or current.is_optical:
        raise TypeError("tia expects an electrical (current) Waveform")
    x = current.samples
    flt = _filter_from(bandwidth, kind, order)
    if flt is not None:
        x = flt.apply(x, current.dt)
    if input_noise_a_rthz > 0:
        rng = np.random.default_rng(seed)
        sigma = input_noise_a_rthz * math.sqrt(0.5 / current.dt)
        x = x + _band_limited_noise(xp.full(current.n_steps, sigma), flt, current.dt, rng)
    return Waveform(gain_ohm * x, current.dt, current.t0, None, "V")


def recover_clock(signal: Waveform, baud: float) -> float:
    """Sampling phase (s, in ``[0, 1/baud)``) from the spectral line of the transition density.

    The squared slope of an NRZ/PAM signal peaks at the symbol transitions; its
    Fourier component at the baud rate has phase ``-2 pi baud t_tr``, so the
    transitions sit at ``t_tr`` and the eye centre half a unit interval later
    (the square-law timing recovery of Oerder & Meyr, IEEE Trans. Commun. 36,
    605, 1988, applied to the waveform slope).

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.timedomain import drive_waveform, prbs
    >>> w = drive_waveform(prbs(7), baud=10e9, samples_per_symbol=16, rise_time_20_80_s=20e-12)
    >>> round(recover_clock(w, 10e9) * 1e12, 1)       # eye centre half a UI after the edges
    50.0
    """
    x = np.real(np.asarray(signal.samples))
    if signal.is_optical:
        x = np.abs(np.asarray(signal.samples)) ** 2
    slope = np.diff(x)
    t_mid = signal.t0 + signal.dt * (np.arange(slope.size) + 0.5)
    c = np.sum(slope ** 2 * np.exp(-2j * np.pi * baud * t_mid))
    period = 1.0 / baud
    t_tr = (-np.angle(c) / (2 * np.pi * baud)) % period
    return float((t_tr + 0.5 * period) % period)
