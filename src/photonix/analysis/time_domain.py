"""Time-domain response of linear optical devices from their transfer function.

Photonix propagation uses ``exp(-1j * beta * L)`` with an ``exp(+1j * omega * t)``
time dependence, so a pure delay ``tau`` is ``H = exp(-1j * omega * tau)`` and
the inverse FFT (which uses ``exp(+1j ...)``) places it at ``t = +tau``.

Responses are computed in baseband about a carrier: frequencies are offsets
from the centre of the sampled band, the time window is ``1 / df``, and
anything delayed by more than the window aliases (wraps). Sample the band
finely enough that the window exceeds the device's longest delay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photonix.core.constants import C0_UM_S

__all__ = ["PulseResponse", "impulse_response", "pulse_response"]

_C_UM_PS = C0_UM_S * 1e-12  # µm / ps


def _transfer_on_frequency(wavelength, transfer):
    """Return ascending frequency [THz] and transfer, with unwrapped phase."""
    wl = np.asarray(wavelength, dtype=float)
    h = np.asarray(transfer, dtype=complex)
    if wl.ndim != 1 or wl.shape != h.shape or wl.size < 4:
        raise ValueError("wavelength and transfer must be equal-length 1-D arrays (>= 4 samples)")
    f = _C_UM_PS / wl  # THz (1/ps)
    order = np.argsort(f)
    f, h = f[order], h[order]
    if np.any(np.diff(f) <= 0):
        raise ValueError("wavelengths must be distinct")
    return f, np.abs(h), np.unwrap(np.angle(h))


def _interp(f_grid, f, mag, phase):
    inside = (f_grid >= f[0]) & (f_grid <= f[-1])
    m = np.where(inside, np.interp(f_grid, f, mag), 0.0)
    p = np.interp(f_grid, f, phase)
    return m * np.exp(1j * p), inside


def impulse_response(wavelength, transfer, *, oversample: int = 4, window: str | None = "hann"):
    """Baseband impulse response of a sampled transfer function.

    Parameters
    ----------
    wavelength, transfer
        Wavelength sweep (µm) and complex transfer ``S[(in, out)]``.
    oversample
        Zero-padding factor of the frequency record (finer time sampling).
    window
        ``"hann"`` tapers the band edges to suppress truncation ringing;
        ``None`` uses a rectangular window.

    Returns
    -------
    t_ps, h
        Time axis in ps (``0`` to the window length) and complex impulse
        response (arbitrary units; ``|h|`` peaks at the group delay).

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.54, 1.56, 2001)
    >>> s = px.components.straight(wl=wl, length=1000.0, neff=2.4, ng=4.2)
    >>> t, h = impulse_response(wl, px.to_numpy(s[("o1", "o2")]))
    >>> tau = 4.2 * 1000.0 / 299.792458                 # n_g L / c = 14.01 ps
    >>> bool(abs(float(t[np.argmax(np.abs(h))]) - tau) < t[1])   # within one time sample
    True
    """
    f, mag, phase = _transfer_on_frequency(wavelength, transfer)
    n = f.size
    f_grid = np.linspace(f[0], f[-1], n)
    h_f, _ = _interp(f_grid, f, mag, phase)
    if window == "hann":
        h_f = h_f * np.hanning(n)
    elif window is not None:
        raise ValueError("window must be 'hann' or None")
    df = f_grid[1] - f_grid[0]
    m = n * int(oversample)
    padded = np.zeros(m, dtype=complex)
    padded[:n] = h_f
    h_t = np.fft.ifft(padded) * m / n
    t = np.arange(m) / (m * df)
    return t, h_t


@dataclass(frozen=True)
class PulseResponse:
    """Input and output envelopes of a pulse sent through a device."""

    t_ps: np.ndarray
    input_power: np.ndarray
    output_power: np.ndarray

    @property
    def delay_ps(self) -> float:
        """Shift of the output intensity centroid relative to the input."""
        return float(np.sum(self.t_ps * self.output_power) / np.sum(self.output_power)
                     - np.sum(self.t_ps * self.input_power) / np.sum(self.input_power))

    @property
    def broadening(self) -> float:
        """Ratio of output to input RMS intensity width."""
        def rms(p):
            mu = np.sum(self.t_ps * p) / np.sum(p)
            return np.sqrt(np.sum((self.t_ps - mu) ** 2 * p) / np.sum(p))

        return float(rms(self.output_power) / rms(self.input_power))

    @property
    def energy_transmission(self) -> float:
        return float(np.sum(self.output_power) / np.sum(self.input_power))


def pulse_response(wavelength, transfer, *, pulse_fwhm_ps: float, center_wl: float | None = None,
                   n_samples: int = 4096, t0_ps: float | None = None) -> PulseResponse:
    """Send a transform-limited Gaussian pulse through a sampled transfer function.

    The pulse (intensity FWHM ``pulse_fwhm_ps``, carrier ``center_wl``,
    default the sweep centre) is launched at ``t0_ps`` (default
    ``4 * pulse_fwhm_ps``) on a time grid matching the sweep's frequency
    span; spectral content outside the sweep is discarded. The pulse
    spectrum should fit well inside the sweep.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.50, 1.60, 4001)
    >>> s = px.components.straight(wl=wl, length=2000.0)
    >>> r = pulse_response(wl, px.to_numpy(s[("o1", "o2")]), pulse_fwhm_ps=1.0)
    >>> round(r.delay_ps, 2), round(r.energy_transmission, 4)
    (28.02, 1.0)
    """
    f, mag, phase = _transfer_on_frequency(wavelength, transfer)
    fc = _C_UM_PS / (0.5 * (np.min(wavelength) + np.max(wavelength)) if center_wl is None else center_wl)
    span = f[-1] - f[0]
    df = span / (f.size - 1)
    m = max(int(n_samples), int(2 ** np.ceil(np.log2(f.size))))
    dt = 1.0 / (m * df)
    t = np.arange(m) * dt
    t0 = 4.0 * pulse_fwhm_ps if t0_ps is None else float(t0_ps)
    a = 2.0 * np.log(2.0) / pulse_fwhm_ps ** 2
    e_in = np.exp(-a * (t - t0) ** 2)
    spec = np.fft.fft(e_in)
    nu = np.fft.fftfreq(m, dt)
    h, inside = _interp(fc + nu, f, mag, phase)
    if np.sum(np.abs(spec[~inside]) ** 2) > 1e-6 * np.sum(np.abs(spec) ** 2):
        raise ValueError("the pulse spectrum extends beyond the sampled band; widen the sweep or lengthen the pulse")
    e_out = np.fft.ifft(spec * h)
    return PulseResponse(t, np.abs(e_in) ** 2, np.abs(e_out) ** 2)
