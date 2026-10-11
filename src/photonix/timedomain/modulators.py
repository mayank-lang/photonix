"""Electro-optic modulators acting on sampled optical envelopes.

* :func:`mzm` -- dual-drive Mach-Zehnder modulator (push-pull, finite
  extinction, chirp) with optional traveling-wave electrode or drive filter;
  :func:`push_pull` builds the arm voltages.
* :class:`TravelingWaveElectrode` -- the electro-optic response of a
  traveling-wave electrode with RF loss, velocity mismatch and source / load
  reflections.
* :func:`eam` -- electro-absorption modulator with Henry-factor chirp.
* :class:`RingModulator` / :func:`ring_modulator` -- carrier-depletion
  micro-ring modulator by temporal coupled-mode theory (photon-lifetime
  dynamics) with the junction physics of
  :func:`~photonix.components.pn_phase_shifter` and an RC-limited drive.

Sign conventions follow photonix: ``exp(+j omega t)``, propagation
``exp(-j phi)``; the optical envelope is in sqrt(W).
"""
from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from photonix.core.backend import xp
from photonix.core.constants import C0, C0_UM_S

from .filters import AnalogFilter, linear_recurrence
from .signal import Waveform

__all__ = ["mzm", "push_pull", "TravelingWaveElectrode", "eam", "RingModulator", "ring_modulator"]


def _samples(x, n: int | None = None, what: str = "signal"):
    data = x.samples if isinstance(x, Waveform) else x
    data = xp.asarray(data)
    if n is not None and data.ndim == 0:
        data = xp.full(n, data)
    if n is not None and int(data.shape[0]) != n:
        raise ValueError(f"{what} has {int(data.shape[0])} samples, expected {n}")
    return data


def _optical(field_in: Waveform) -> Waveform:
    if not isinstance(field_in, Waveform) or not field_in.is_optical:
        raise TypeError("expected an optical Waveform (complex envelope with a carrier)")
    return field_in


# --------------------------------------------------------------------------- #
# Traveling-wave electrode
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TravelingWaveElectrode:
    """Small-signal electro-optic response of a traveling-wave electrode.

    The RF wave ``V(z) = A [exp(-gamma z) + rho_L exp(-2 gamma L) exp(gamma z)]``
    (``gamma = alpha_m + j omega n_m / c``, load reflection ``rho_L``, source
    impedance ``z_source_ohm``) is seen by the light, which travels at
    ``c / n_g_opt``, as the average

    .. code-block:: text

        H(f) ~ [ F((gamma - j beta_o) L) + rho_L e^{-2 gamma L} G((gamma + j beta_o) L) ] / (1 + rho_L e^{-2 gamma L})
               * Z_in / (Z_in + Z_S),
        F(x) = (1 - e^{-x}) / x,    G(x) = (e^{x} - 1) / x,    beta_o = omega n_g_opt / c,

    normalized so that ``H(0) = 1`` (the drive waveform is the electrode
    voltage at DC). For a matched, lossless line this reduces to the
    velocity-mismatch factor ``|H| = |sin u / u|``, ``u = pi f (n_m - n_g) L / c``
    (Ghione, *Semiconductor Devices for High-Speed Optoelectronics*, 2009;
    Witzens, Proc. IEEE 106, 2158, 2018). ``alpha_db_per_cm_sqrt_ghz`` is the
    skin-effect loss ``alpha(f) = alpha0 sqrt(f / 1 GHz)`` in dB/cm.

    Examples
    --------
    >>> import math
    >>> el = TravelingWaveElectrode(length_mm=10.0, n_m=2.6, n_g_opt=2.2)
    >>> u = 1.3916                                   # sin(u)/u = 1/sqrt(2)
    >>> f = u * 299792458.0 / (math.pi * 0.4 * 0.01)
    >>> round(abs(complex(el.response(f))), 4)
    0.7071
    """

    length_mm: float = 3.0
    n_m: float = 3.8
    n_g_opt: float = 3.9
    alpha_db_per_cm_sqrt_ghz: float = 0.0
    z0_ohm: float = 50.0
    z_load_ohm: float = 50.0
    z_source_ohm: float = 50.0

    def _raw(self, f_hz):
        f = np.atleast_1d(np.asarray(f_hz, dtype=float))
        length = self.length_mm * 1e-3
        omega = 2.0 * np.pi * f
        alpha = self.alpha_db_per_cm_sqrt_ghz * np.sqrt(np.abs(f) / 1e9) * 100.0 / (20.0 / math.log(10.0))
        gamma = alpha + 1j * omega * self.n_m / C0
        beta_o = omega * self.n_g_opt / C0
        rho_l = (self.z_load_ohm - self.z0_ohm) / (self.z_load_ohm + self.z0_ohm)
        e2 = np.exp(-2.0 * gamma * length)
        z_in = self.z0_ohm * (1.0 + rho_l * e2) / (1.0 - rho_l * e2)
        xm = (gamma - 1j * beta_o) * length
        xpl = (gamma + 1j * beta_o) * length

        def f_(x):
            small = np.abs(x) < 1e-6
            xs = np.where(small, 1.0, x)
            return np.where(small, 1.0 - x / 2.0, (1.0 - np.exp(-xs)) / xs)

        def g_(x):
            small = np.abs(x) < 1e-6
            xs = np.where(small, 1.0, x)
            return np.where(small, 1.0 + x / 2.0, (np.exp(xs) - 1.0) / xs)

        avg = (f_(xm) + rho_l * e2 * g_(xpl)) / (1.0 + rho_l * e2)
        return (avg * z_in / (z_in + self.z_source_ohm)).reshape(np.shape(f_hz))

    @property
    def dc_gain(self) -> float:
        """Electrode voltage per open-circuit source voltage at DC, ``Z_L / (Z_L + Z_S)``."""
        return self.z_load_ohm / (self.z_load_ohm + self.z_source_ohm)

    def response(self, f_hz):
        """Normalized electro-optic response ``H(f)`` (``H(0) = 1``)."""
        return self._raw(f_hz) / self.dc_gain

    def bandwidth_hz(self, level: float = 1.0 / math.sqrt(2.0), f_max_hz: float = 1e12) -> float:
        """Lowest frequency where ``|H|`` falls to ``level`` (default the electrical -3 dB, 1/sqrt 2)."""
        from scipy.optimize import brentq

        f = np.linspace(0.0, f_max_hz, 20001)[1:]
        mag = np.abs(self.response(f))
        below = np.nonzero(mag < level)[0]
        if below.size == 0:
            return math.inf
        i = int(below[0])
        lo = 0.0 if i == 0 else f[i - 1]
        return float(brentq(lambda x: abs(complex(self.response(x))) - level, lo, f[i]))

    @property
    def latency_s(self) -> float:
        """Reference delay making the time-domain kernel causal (``max(0, L (n_g - n_m) / c)``)."""
        return max(0.0, self.length_mm * 1e-3 * (self.n_g_opt - self.n_m) / C0)

    def kernel(self, dt: float) -> np.ndarray:
        """Causal impulse response (per sample) of the response delayed by :attr:`latency_s`."""
        length = self.length_mm * 1e-3
        rt = 2.0 * length * max(self.n_m, self.n_g_opt) / C0
        rho_l = abs((self.z_load_ohm - self.z0_ohm) / (self.z_load_ohm + self.z0_ohm))
        rho_s = abs((self.z_source_ohm - self.z0_ohm) / (self.z_source_ohm + self.z0_ohm))
        bounce = rho_l * rho_s
        n_rt = 2 if bounce < 1e-12 else int(math.ceil(math.log(1e-9) / math.log(bounce))) + 2
        span = n_rt * rt + self.latency_s + 4 * dt
        n = int(2 ** math.ceil(math.log2(max(256.0, 4.0 * span / dt))))
        f = np.fft.rfftfreq(n, dt)
        h = self.response(f) * np.exp(-2j * np.pi * f * self.latency_s)
        return np.fft.irfft(h, n)

    def filter(self, v, dt: float | None = None):
        """Apply the electrode response to a drive voltage (array or electrical :class:`Waveform`).

        Linear (zero-padded FFT) convolution with :meth:`kernel`; the output
        is delayed by :attr:`latency_s` relative to the optical reference.
        """
        wf = v if isinstance(v, Waveform) else None
        if wf is not None:
            dt = wf.dt
        if dt is None:
            raise ValueError("give dt for an array input")
        x = _samples(v).astype(float)
        k = self.kernel(dt)
        n = int(x.shape[0])
        m = int(2 ** math.ceil(math.log2(n + k.size)))
        y = xp.fft.irfft(xp.fft.rfft(x, m) * xp.asarray(np.fft.rfft(k, m)), m)[:n]
        return wf.with_samples(y) if wf is not None else y


# --------------------------------------------------------------------------- #
# Mach-Zehnder modulator
# --------------------------------------------------------------------------- #
def push_pull(v, *, bias_v: float = 0.0, chirp: float = 0.0):
    """Arm voltages ``(v1, v2)`` of a push-pull drive with an optional drive imbalance.

    ``v1 = (1 + chirp) / 2 (v + bias_v)``, ``v2 = -(1 - chirp) / 2 (v + bias_v)``:
    the differential voltage is ``v + bias_v`` and ``chirp`` is the
    small-signal chirp parameter at quadrature (0 = chirp-free push-pull,
    +/-1 = single-arm drive).

    Examples
    --------
    >>> push_pull(2.0)
    (1.0, -1.0)
    """
    if isinstance(v, Waveform):
        x = v.samples + bias_v
        return v.with_samples(0.5 * (1.0 + chirp) * x), v.with_samples(-0.5 * (1.0 - chirp) * x)
    x = v + bias_v if not hasattr(v, "shape") else xp.asarray(v) + bias_v
    return 0.5 * (1.0 + chirp) * x, -0.5 * (1.0 - chirp) * x


def mzm(
    field_in: Waveform,
    v_arm1,
    v_arm2=0.0,
    *,
    v_pi: float = 3.0,
    insertion_loss_db: float = 0.0,
    extinction_db: float = math.inf,
    electrode: TravelingWaveElectrode | None = None,
    drive_filter: AnalogFilter | None = None,
) -> Waveform:
    """Dual-drive Mach-Zehnder modulator.

    .. code-block:: text

        E_out = E_in 10^(-IL/20) [a1 exp(-j pi V1 / V_pi) + a2 exp(-j pi V2 / V_pi)] / 2

    so a differential voltage ``V1 - V2 = V`` gives the power transfer
    ``cos^2(pi V / (2 V_pi))`` (``V_pi`` is the differential voltage for a
    pi phase difference) and the common-mode voltage sets the chirp. A finite
    extinction ratio unbalances the arms, ``a_{1,2} = 1 -/+ 10^(-ER/20)``
    (maximum transmission stays ``10^(-IL/10)``, the minimum is
    ``10^(-ER/10)`` below it). The arm voltages are optionally shaped by a
    traveling-wave ``electrode`` response and/or a ``drive_filter`` (e.g. the
    RC bandwidth of a lumped device); otherwise the modulator is quasi-static.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.timedomain import cw
    >>> v = np.linspace(0, 3.0, 4)
    >>> out = mzm(cw(1.0, 4, 1e-12), *push_pull(v), v_pi=3.0)
    >>> [round(float(p), 12) for p in out.power()]            # cos^2(pi V / (2 V_pi))
    [1.0, 0.75, 0.25, 0.0]
    """
    fin = _optical(field_in)
    n = fin.n_steps
    v1 = _samples(v_arm1, n, "v_arm1").astype(float)
    v2 = _samples(v_arm2, n, "v_arm2").astype(float)
    for flt in (electrode, drive_filter):
        if flt is not None:
            v1 = flt.filter(v1, fin.dt) if isinstance(flt, TravelingWaveElectrode) else flt.apply(v1, fin.dt)
            v2 = flt.filter(v2, fin.dt) if isinstance(flt, TravelingWaveElectrode) else flt.apply(v2, fin.dt)
    delta = 0.0 if math.isinf(extinction_db) else 10.0 ** (-extinction_db / 20.0)
    gain = 10.0 ** (-insertion_loss_db / 20.0)
    arm1 = (1.0 - delta) * xp.exp(-1j * math.pi * v1 / v_pi)
    arm2 = (1.0 + delta) * xp.exp(-1j * math.pi * v2 / v_pi)
    t = 0.5 * gain * (arm1 + arm2)
    return fin.with_samples(fin.samples * t)


# --------------------------------------------------------------------------- #
# Electro-absorption modulator
# --------------------------------------------------------------------------- #
def eam(
    field_in: Waveform,
    v,
    *,
    absorption_db: Callable[[Any], Any] | float = 10.0,
    alpha_h: float = 0.0,
    insertion_loss_db: float = 0.0,
    drive_filter: AnalogFilter | None = None,
) -> Waveform:
    """Electro-absorption modulator with linewidth-enhancement (Henry) chirp.

    ``E_out = E_in 10^(-IL/20) exp(-(1 + j alpha_h) kappa(V))`` with the field
    attenuation ``kappa = ln(10) / 20 * A_dB(V)``; ``absorption_db`` is a
    callable ``A_dB(V)`` or a slope in dB/V (``A_dB = slope * V``). The
    instantaneous phase follows the intensity, ``dphi/dt = (alpha_h / 2)
    d ln P / dt``, the chirp of Koyama & Iga (J. Lightwave Technol. 6, 87,
    1988).

    Examples
    --------
    >>> from photonix.timedomain import cw
    >>> out = eam(cw(1.0, 2, 1e-12), [0.0, 1.0], absorption_db=10.0)
    >>> [round(float(p), 12) for p in out.power()]
    [1.0, 0.1]
    """
    fin = _optical(field_in)
    vv = _samples(v, fin.n_steps, "v").astype(float)
    if drive_filter is not None:
        vv = drive_filter.apply(vv, fin.dt)
    a_db = absorption_db(vv) if callable(absorption_db) else absorption_db * vv
    kappa = math.log(10.0) / 20.0 * xp.asarray(a_db)
    t = 10.0 ** (-insertion_loss_db / 20.0) * xp.exp(-(1.0 + 1j * alpha_h) * kappa)
    return fin.with_samples(fin.samples * t)


# --------------------------------------------------------------------------- #
# Ring modulator (temporal coupled-mode theory)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RingModulator:
    """All-pass carrier-depletion micro-ring modulator by temporal coupled-mode theory.

    .. code-block:: text

        da/dt = (j dw0(V(t)) - 1/tau(V(t))) a - j mu s_in,      s_out = s_in - j mu a,
        1/tau = 1/tau_i + 1/tau_e,   mu^2 = 2 / tau_e

    in the envelope frame of the carrier, with ``|a|^2`` the stored energy (J)
    and ``|s|^2`` the bus power (W) (Haus 1984; Sacher & Poon, Opt. Express 16,
    15741, 2008). The parameters follow from the ring geometry at the carrier:
    round trip ``T = n_g L / c``; ``exp(-2 T / tau_e) = 1 - coupling``;
    ``exp(-T / tau_i) = |round-trip amplitude|``; ``dw0 = (2 pi m - phi(V)) / T``
    is the detuning of the resonance nearest the carrier from it, where the
    round-trip phase ``phi(V)`` and amplitude include a fraction
    ``pn_fraction`` of carrier-depletion junction evaluated with
    :func:`~photonix.components.pn_phase_shifter` (``pn`` forwards its doping
    and geometry). Alternatively ``dneff_dv`` gives a linear index slope with
    no loss change. The applied voltage passes through a first-order RC
    low-pass (``rc_time_s``) before acting on the junction. In the high-Q
    limit the steady state equals :func:`~photonix.components.all_pass_ring`.

    Examples
    --------
    >>> rm = RingModulator(radius=10.0, coupling=0.05, loss_db_cm=5.0)
    >>> dw0, inv_tau_i, inv_tau_e = rm.rates(1.55, 0.0)
    >>> bool(inv_tau_e > 0 and inv_tau_i > 0)
    True
    """

    radius: float = 7.5
    coupling: float = 0.03
    loss_db_cm: float = 5.0
    neff: float = 2.4
    ng: float = 4.2
    wl0: float = 1.55
    pn_fraction: float = 0.75
    pn: Mapping[str, Any] = field(default_factory=dict)
    dneff_dv: float | None = None
    rc_time_s: float = 0.0
    bias_v: float = 0.0

    @property
    def circumference(self) -> float:
        return 2.0 * math.pi * self.radius

    @property
    def round_trip_s(self) -> float:
        return self.ng * self.circumference / C0_UM_S

    def _round_trip(self, wl: float, v):
        """Complex round-trip transmission ``a(V) exp(-j phi(V))`` at the carrier (array over ``v``)."""
        from photonix import components as c

        length = self.circumference
        f = self.pn_fraction
        passive = c.straight(wl=wl, length=(1.0 - f) * length, neff=self.neff, ng=self.ng, wl0=self.wl0,
                             loss_db_cm=self.loss_db_cm)[("o1", "o2")]
        v = xp.asarray(v, dtype=float)
        if self.dneff_dv is not None:
            junction = c.straight(wl=wl, length=f * length, neff=self.neff, ng=self.ng, wl0=self.wl0,
                                  loss_db_cm=self.loss_db_cm)[("o1", "o2")]
            junction = junction * xp.exp(-1j * 2.0 * math.pi / wl * self.dneff_dv * v * f * length)
        else:
            junction = c.pn_phase_shifter(wl=wl, length=f * length, voltage=v, neff=self.neff, ng=self.ng,
                                          wl0=self.wl0, loss_db_cm=self.loss_db_cm, **dict(self.pn))[("o1", "o2")]
        return passive * junction

    def rates(self, wl: float, v):
        """``(dw0, 1/tau_i, 1/tau_e)`` in rad/s, 1/s, 1/s at carrier ``wl`` and voltage(s) ``v``."""
        t_rt = self.round_trip_s
        ref = complex(np.asarray(self._round_trip(wl, self.bias_v)))
        phi_ref = -cmath.phase(ref)                    # round-trip phase past the nearest resonance, (-pi, pi]
        r = self._round_trip(wl, v)
        dphi = -xp.angle(r / ref)                      # small and continuous in V
        dw0 = -(phi_ref + dphi) / t_rt
        inv_tau_i = -xp.log(xp.abs(r)) / t_rt
        inv_tau_e = -0.5 * math.log(1.0 - self.coupling) / t_rt
        return dw0, inv_tau_i, inv_tau_e

    def steady_transmission(self, wl: float, v, offset_hz=0.0):
        """CMT steady-state through transmission ``s_out / s_in`` for a tone at ``f_c + offset``."""
        dw0, inv_i, inv_e = self.rates(wl, v)
        dw = 2.0 * math.pi * xp.asarray(offset_hz)
        return 1.0 - 2.0 * inv_e / (inv_i + inv_e + 1j * (dw - dw0))

    def drive(self, v, dt: float):
        """The junction voltage after the RC low-pass (array)."""
        vv = xp.asarray(v, dtype=float)
        if self.rc_time_s > 0:
            vv = AnalogFilter.first_order(1.0 / (2.0 * math.pi * self.rc_time_s)).apply(vv, dt)
        return vv

    def simulate(self, field_in: Waveform, v, *, initial: str = "steady") -> Waveform:
        """Through-port envelope for the input envelope ``field_in`` and drive voltage ``v``.

        The coupled-mode equation is integrated exactly for piecewise-constant
        coefficients (the drive at the interval midpoint) and a
        piecewise-linear input (first-order hold), so a constant drive reproduces
        the steady state to round-off.
        """
        fin = _optical(field_in)
        n, dt = fin.n_steps, fin.dt
        vv = self.drive(_samples(v, n, "v"), dt)
        v_mid = 0.5 * (vv + xp.concatenate([vv[1:], vv[-1:]]))
        dw0, inv_i, inv_e = self.rates(fin.carrier_wl, v_mid)
        mu = math.sqrt(2.0 * float(inv_e))
        z = (1j * dw0 - (inv_i + inv_e)) * dt
        lam = xp.exp(z)
        zs = xp.where(xp.abs(z) < 1e-3, xp.ones_like(z), z)
        phi1 = xp.where(xp.abs(z) < 1e-3, 1 + z / 2 + z * z / 6 + z ** 3 / 24, (xp.exp(zs) - 1) / zs)
        phi2 = xp.where(xp.abs(z) < 1e-3, 0.5 + z / 6 + z * z / 24 + z ** 3 / 120, (xp.exp(zs) - 1 - zs) / (zs * zs))
        s = xp.asarray(fin.samples, dtype=complex)
        s_next = xp.concatenate([s[1:], s[-1:]])
        drive_term = -1j * mu * dt * ((phi1 - phi2) * s + phi2 * s_next)
        if initial == "steady":
            a0 = -1j * mu * s[0] / ((inv_i[0] + inv_e) - 1j * dw0[0])
        elif initial == "empty":
            a0 = 0.0 + 0.0j
        else:
            raise ValueError("initial must be 'steady' or 'empty'")
        nxt = linear_recurrence(lam, drive_term, a0)          # a_{k+1}
        a = xp.concatenate([xp.reshape(xp.asarray(a0, dtype=complex), (1,)), nxt[:-1]])
        return fin.with_samples(s - 1j * mu * a)

    def small_signal_response(self, f_hz, v0: float, *, carrier_wl: float = 1.55, power_w: float = 1.0,
                              offset_hz: float = 0.0):
        """Closed-form small-signal response ``dP_out / dV`` (W/V) at modulation frequency ``f_hz``.

        Linearizing the coupled-mode equation about the steady state ``a0`` for
        a resonance-frequency modulation ``d omega_0`` (Sacher & Poon 2008):

        .. code-block:: text

            dP(Omega) / d omega_0 = mu [ s_t0^* a0 g(Omega) + s_t0 a0^* g(-Omega)^* ],
            g(Omega) = 1 / (1/tau + j (Omega - (dw0 - dw_laser)))

        times the drive slope ``d omega_0 / dV`` and the RC response; the
        loss modulation of the junction is neglected (exact with ``dneff_dv``).
        The time-domain convention is ``P(t) = P0 + Re[H dV exp(j Omega t)]``.
        """
        f = np.asarray(f_hz, dtype=float)
        wl = carrier_wl
        v0 = float(v0)
        h = 1e-4
        dw_p, inv_i, inv_e = (np.asarray(x) for x in self.rates(wl, v0 + h))
        dw_m, _, _ = (np.asarray(x) for x in self.rates(wl, v0 - h))
        dw0, inv_i, inv_e = (np.asarray(x) for x in self.rates(wl, v0))
        slope = float((dw_p - dw_m) / (2 * h))
        inv_tau = float(inv_i) + float(inv_e)
        mu = math.sqrt(2.0 * float(inv_e))
        det = float(dw0) - 2.0 * math.pi * offset_hz
        s0 = math.sqrt(power_w)
        a0 = -1j * mu * s0 / (inv_tau - 1j * det)
        st0 = s0 - 1j * mu * a0
        omega = 2.0 * np.pi * f

        def g(w):
            return 1.0 / (inv_tau + 1j * (w - det))

        resp = mu * (np.conj(st0) * a0 * g(omega) + st0 * np.conj(a0) * np.conj(g(-omega)))
        rc = 1.0 / (1.0 + 1j * omega * self.rc_time_s)
        return resp * slope * rc


def ring_modulator(field_in: Waveform, v, **params: Any) -> Waveform:
    """Convenience wrapper: ``RingModulator(**params).simulate(field_in, v)``.

    Examples
    --------
    >>> import numpy as np
    >>> from photonix.timedomain import cw
    >>> out = ring_modulator(cw(1e-3, 100, 1e-12), np.zeros(100), coupling=0.05)
    >>> out.n_steps
    100
    """
    initial = params.pop("initial", "steady")
    return RingModulator(**params).simulate(field_in, v, initial=initial)
