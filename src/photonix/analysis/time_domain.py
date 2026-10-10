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

import warnings
from dataclasses import dataclass

import numpy as np

from photonix.core.constants import C0_UM_S

__all__ = ["PulseResponse", "impulse_response", "pulse_response"]

_C_UM_PS = C0_UM_S * 1e-12  # µm / ps


def _bulk_delay(f, h, hint: float | None = None) -> float:
    """Bulk group delay [ps] of ``h(f)``, resolving phase-unwrapping aliases.

    Plain unwrapping is exact while the phase advances by less than pi per
    sample, i.e. for delays below half the window ``1 / df``; beyond that it
    returns an *alias* ``tau - j / df``. Every alias leaves wrapped residual
    phase steps ``2 pi j (df_k / mean(df) - 1)``, which vanish only for the true
    delay when the grid is not uniform in frequency (a sweep uniform in
    wavelength, say). An alias is adopted only if it leaves a residual at
    least 100 times smaller than plain unwrapping does, because on such a grid
    an alias also resembles a quadratic (GDD-like) phase: a genuinely wrapped
    long delay passes this test (its residual is essentially zero), ordinary
    dispersion does not. ``hint`` (e.g. the known ``n_g L / c``) selects the
    alias nearest to it instead. On a
    frequency-uniform grid the delay is genuinely ambiguous modulo the window
    and the plain-unwrap value is kept.
    """
    df = np.diff(f)
    steps = np.angle(h[1:] * np.conj(h[:-1]))
    weight = np.abs(h[1:]) * np.abs(h[:-1])
    if not np.any(weight > 0):
        return 0.0
    unwrapped = np.unwrap(np.angle(h))
    tau_plain = float(-(unwrapped[-1] - unwrapped[0]) / (2 * np.pi * (f[-1] - f[0])))
    if hint is not None:
        j = round((float(hint) - tau_plain) * df.mean())
        return tau_plain + j / df.mean()
    spread = (df.max() - df.min()) / df.mean()
    if spread < 1e-9:
        return tau_plain
    n_alias = int(min(np.ceil(1.0 / spread), 500))

    def cost(tau):
        resid = np.angle(np.exp(1j * (steps + 2 * np.pi * tau * df)))
        return float(np.sum(weight * resid * resid))

    plain_cost = cost(tau_plain)
    best, best_cost = tau_plain, plain_cost
    for j in range(-n_alias, n_alias + 1):
        if j:
            tau = tau_plain + j / df.mean()
            c = cost(tau)
            if c < best_cost:
                best, best_cost = tau, c
    # On a wavelength-uniform grid an alias also mimics a quadratic (GDD-like)
    # phase, so only accept one that explains the data dramatically better --
    # the signature of a genuinely wrapped long delay, not of dispersion.
    if best_cost < 1e-6 * plain_cost:
        if best != tau_plain:
            warnings.warn(
                f"resolved a wrapped bulk delay of {best:.4g} ps (plain unwrapping gives {tau_plain:.4g} ps); "
                "pass delay_hint_ps to make the choice explicit",
                stacklevel=4,
            )
        return best
    return tau_plain


def _transfer_on_frequency(wavelength, transfer, delay_hint_ps: float | None = None):
    """Return ascending frequency [THz], |H| and the correctly unwrapped phase.

    Unwrapping ``angle(H)`` directly fails as soon as the phase advances by more
    than pi per sample -- i.e. for delays beyond half the window on a sweep that
    is uniform in wavelength. The bulk delay is therefore removed first, the
    slowly varying residual is unwrapped, and the delay is added back.
    """
    wl = np.asarray(wavelength, dtype=float)
    h = np.asarray(transfer, dtype=complex)
    if wl.ndim != 1 or wl.shape != h.shape or wl.size < 4:
        raise ValueError("wavelength and transfer must be equal-length 1-D arrays (>= 4 samples)")
    f = _C_UM_PS / wl  # THz (1/ps)
    order = np.argsort(f)
    f, h = f[order], h[order]
    if np.any(np.diff(f) <= 0):
        raise ValueError("wavelengths must be distinct")
    tau0 = _bulk_delay(f, h, delay_hint_ps)
    ramp = 2.0 * np.pi * (f - f[0]) * tau0
    residual = np.unwrap(np.angle(h * np.exp(1j * ramp)))
    if np.max(np.abs(np.diff(residual))) > 0.9 * np.pi:
        warnings.warn(
            "the residual phase changes by more than 0.9*pi between samples after removing the bulk delay; "
            "the sweep under-samples the response (refine the wavelength grid)",
            stacklevel=3,
        )
    return f, np.abs(h), residual - ramp


def _interp(f_grid, f, mag, phase):
    inside = (f_grid >= f[0]) & (f_grid <= f[-1])
    m = np.where(inside, np.interp(f_grid, f, mag), 0.0)
    p = np.interp(f_grid, f, phase)
    return m * np.exp(1j * p), inside


def impulse_response(wavelength, transfer, *, oversample: int = 4, window: str | None = "hann",
                     delay_hint_ps: float | None = None):
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
    delay_hint_ps
        Approximate bulk delay (e.g. ``n_g L / c``). Delays beyond half the
        window ``1/df`` are ambiguous in sampled data; the hint picks the right
        unwrapping branch. Without it, a wrapped delay is resolved
        automatically only when the evidence is unambiguous.

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
    f, mag, phase = _transfer_on_frequency(wavelength, transfer, delay_hint_ps)
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
    def window_ps(self) -> float:
        return float(self.t_ps[-1] + (self.t_ps[1] - self.t_ps[0]))

    def _centroid(self, p) -> float:
        # Circular mean: the FFT window is periodic, so energy that wraps past
        # t = 0 (an advanced or strongly dispersed pulse) is handled correctly.
        w = self.window_ps
        return float((w / (2 * np.pi) * np.angle(np.sum(p * np.exp(2j * np.pi * self.t_ps / w)))) % w)

    def _rms(self, p) -> float:
        w = self.window_ps
        tt = (self.t_ps - self._centroid(p) + w / 2) % w - w / 2
        return float(np.sqrt(np.sum(tt * tt * p) / np.sum(p)))

    @property
    def delay_ps(self) -> float:
        """Output-minus-input intensity centroid, wrapped to ``[-W/8, 7W/8)``.

        ``W`` is the periodic time window (``1 / df``); delays are only defined
        modulo ``W``, and the chosen range favours the usual positive delays
        while still reporting small advances (negative delays) as negative.
        """
        w = self.window_ps
        d = (self._centroid(self.output_power) - self._centroid(self.input_power)) % w
        return float(d - w if d >= 0.875 * w else d)

    @property
    def broadening(self) -> float:
        """Ratio of output to input RMS intensity width (circular moments)."""
        return self._rms(self.output_power) / self._rms(self.input_power)

    @property
    def energy_transmission(self) -> float:
        return float(np.sum(self.output_power) / np.sum(self.input_power))


def pulse_response(wavelength, transfer, *, pulse_fwhm_ps: float, center_wl: float | None = None,
                   n_samples: int = 4096, t0_ps: float | None = None,
                   delay_hint_ps: float | None = None) -> PulseResponse:
    """Send a transform-limited Gaussian pulse through a sampled transfer function.

    The pulse (intensity FWHM ``pulse_fwhm_ps``, carrier ``center_wl``,
    default the sweep centre) is launched at ``t0_ps`` (default
    ``4 * pulse_fwhm_ps``) on a time grid matching the sweep's frequency
    span; spectral content outside the sweep is discarded. The pulse
    spectrum should fit well inside the sweep. ``delay_hint_ps`` is as for
    :func:`impulse_response`.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> wl = np.linspace(1.50, 1.60, 4001)
    >>> s = px.components.straight(wl=wl, length=2000.0)
    >>> r = pulse_response(wl, px.to_numpy(s[("o1", "o2")]), pulse_fwhm_ps=1.0)
    >>> round(r.delay_ps, 2), round(r.energy_transmission, 4)
    (28.02, 1.0)
    """
    f, mag, phase = _transfer_on_frequency(wavelength, transfer, delay_hint_ps)
    fc = _C_UM_PS / (0.5 * (np.min(wavelength) + np.max(wavelength)) if center_wl is None else center_wl)
    span = f[-1] - f[0]
    df = span / (f.size - 1)
    m = max(int(n_samples), int(2 ** np.ceil(np.log2(f.size))))
    dt = 1.0 / (m * df)
    t = np.arange(m) * dt
    t0 = 4.0 * pulse_fwhm_ps if t0_ps is None else float(t0_ps)
    window = m * dt
    if t0 - 3.0 * pulse_fwhm_ps < 0.0 or t0 + 3.0 * pulse_fwhm_ps > window:
        raise ValueError(
            f"the time window 1/df = {window:.4g} ps is too short for a {pulse_fwhm_ps:g} ps pulse launched at "
            f"t0 = {t0:.4g} ps; sample the sweep more finely or pass t0_ps"
        )
    a = 2.0 * np.log(2.0) / pulse_fwhm_ps ** 2
    e_in = np.exp(-a * (t - t0) ** 2)
    spec = np.fft.fft(e_in)
    nu = np.fft.fftfreq(m, dt)
    h, inside = _interp(fc + nu, f, mag, phase)
    if np.sum(np.abs(spec[~inside]) ** 2) > 1e-6 * np.sum(np.abs(spec) ** 2):
        raise ValueError("the pulse spectrum extends beyond the sampled band; widen the sweep or lengthen the pulse")
    e_out = np.fft.ifft(spec * h)
    result = PulseResponse(t, np.abs(e_in) ** 2, np.abs(e_out) ** 2)
    if result._rms(result.output_power) > window / 8:
        warnings.warn(
            "the output pulse fills a large part of the periodic time window; delay and broadening may be "
            "ambiguous -- sample the sweep more finely",
            stacklevel=2,
        )
    return result
