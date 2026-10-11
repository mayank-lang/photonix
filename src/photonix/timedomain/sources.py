"""Transmitter sources: PRBS patterns, symbol mapping, drive waveforms and lasers.

* :func:`prbs` -- maximal-length pseudo-random binary sequences with the
  ITU-T O.150 generator polynomials (PRBS7/9/11/15/23/31).
* :func:`symbols_to_levels` -- NRZ and (Gray-coded) PAM4 level mapping.
* :func:`drive_waveform` -- electrical drive voltages with a controlled 20-80 %
  rise time (Gaussian or raised-cosine edges) and random jitter.
* :func:`cw_laser` -- a CW optical envelope with a Lorentzian linewidth (phase
  random walk) and white relative-intensity noise.
* :class:`LaserParams`, :func:`rate_equation_laser` -- the single-mode
  semiconductor-laser rate equations with gain compression, spontaneous
  emission, linewidth enhancement and Langevin noise.

Random sources take a ``seed`` (any value accepted by
:func:`numpy.random.default_rng`) so every result is reproducible.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from photonix.core.backend import HAS_JAX, xp
from photonix.core.constants import C0, H_PLANCK, Q_E

from .signal import Waveform

__all__ = [
    "PRBS_POLYNOMIALS", "prbs", "symbols_to_levels", "levels_to_symbols", "drive_waveform", "cw_laser",
    "LaserParams", "rate_equation_laser",
]

#: ITU-T O.150 PRBS generator polynomials ``x^n + x^m + 1`` as ``n: m``.
PRBS_POLYNOMIALS: dict[int, int] = {7: 6, 9: 5, 11: 9, 15: 14, 23: 18, 31: 28}


# --------------------------------------------------------------------------- #
# PRBS
# --------------------------------------------------------------------------- #
def prbs(order: Literal[7, 9, 11, 15, 23, 31] = 7, n_bits: int | None = None, *, seed: int | None = None,
         invert: bool = False) -> np.ndarray:
    """Maximal-length PRBS of the ITU-T O.150 polynomial ``x^n + x^m + 1``.

    A Fibonacci shift register with feedback from stages ``n`` and ``m``:
    ``s_k = s_{k-m} XOR s_{k-n}``. Its period is ``2^n - 1`` and every period
    holds ``2^(n-1)`` ones. The first ``n`` bits are the register seed
    (``seed`` as an ``n``-bit integer, least significant bit first; default
    all ones). Long sequences are generated in blocks using the squared
    polynomial ``x^(2n) + x^(2m) + 1`` (Frobenius), so PRBS31 runs at NumPy
    speed. ``invert`` complements the output (O.150 specifies the inverted
    PRBS15/23/31 for some test sets).

    Parameters
    ----------
    order
        7, 9, 11, 15, 23 or 31.
    n_bits
        Number of bits; defaults to one period ``2^n - 1``.

    Examples
    --------
    >>> b = prbs(7)
    >>> b.size, int(b.sum())               # one period, 2^(n-1) ones
    (127, 64)
    >>> bool((prbs(7, 254)[:127] == prbs(7, 254)[127:]).all())
    True
    """
    if order not in PRBS_POLYNOMIALS:
        raise ValueError(f"order must be one of {sorted(PRBS_POLYNOMIALS)}")
    n, m = int(order), PRBS_POLYNOMIALS[order]
    total = (2 ** n - 1) if n_bits is None else int(n_bits)
    if total < 0:
        raise ValueError("n_bits must be non-negative")
    seed_v = (2 ** n - 1) if seed is None else int(seed)
    if not 0 < seed_v < 2 ** n:
        raise ValueError(f"seed must be a non-zero {n}-bit integer")
    s = np.zeros(max(total, n), dtype=np.uint8)
    s[:n] = [(seed_v >> i) & 1 for i in range(n)]
    k, j = n, 0
    while k < total:
        if k >= 2 * (n << (j + 1)) and (m << (j + 1)) <= 1 << 20:
            j += 1                              # the doubled recurrence is valid from k >= 2^(j+1) n
            continue
        lag_m, lag_n = m << j, n << j
        b = min(lag_m, total - k)
        s[k:k + b] = s[k - lag_m:k - lag_m + b] ^ s[k - lag_n:k - lag_n + b]
        k += b
    out = s[:total]
    return (1 - out).astype(np.uint8) if invert else out


def symbols_to_levels(bits: Sequence[int] | np.ndarray, fmt: Literal["nrz", "pam4"] = "nrz", *,
                      gray: bool = True) -> np.ndarray:
    """Map bits to normalized amplitude levels in ``[0, 1]``.

    NRZ: one bit per symbol, levels ``{0, 1}``. PAM4: bit pairs (first bit
    most significant) to levels ``{0, 1/3, 2/3, 1}``, Gray-coded by default
    (``00, 01, 11, 10`` -> ``0, 1, 2, 3``) as in IEEE 802.3bs.

    Examples
    --------
    >>> [float(v) for v in symbols_to_levels([0, 0, 0, 1, 1, 1, 1, 0], "pam4") * 3]
    [0.0, 1.0, 2.0, 3.0]
    """
    b = np.asarray(bits, dtype=np.int64).ravel()
    if np.any((b != 0) & (b != 1)):
        raise ValueError("bits must be 0 or 1")
    if fmt == "nrz":
        return b.astype(float)
    if fmt != "pam4":
        raise ValueError("fmt must be 'nrz' or 'pam4'")
    if b.size % 2:
        raise ValueError("PAM4 needs an even number of bits")
    msb, lsb = b[0::2], b[1::2]
    sym = 2 * msb + lsb
    if gray:
        sym = np.array([0, 1, 3, 2])[sym]
    return sym / 3.0


def levels_to_symbols(levels, fmt: Literal["nrz", "pam4"] = "nrz") -> np.ndarray:
    """Symbol indices (``0..M-1``) of normalized levels (the inverse of :func:`symbols_to_levels`)."""
    m = 1 if fmt == "nrz" else 3
    return np.rint(np.asarray(levels, dtype=float) * m).astype(int)


# --------------------------------------------------------------------------- #
# Drive waveforms
# --------------------------------------------------------------------------- #
_PHI_INV_08 = 0.8416212335729143            # standard-normal 80 % quantile
_RC_2080 = 2.0 * math.asin(0.6) / math.pi    # 20-80 % width of a raised-cosine edge / its full width


def _edge(x: np.ndarray, shape: str, rise: float) -> np.ndarray:
    if rise <= 0.0:
        return (x >= 0.0).astype(float)
    if shape == "gaussian":
        from scipy.special import ndtr

        return ndtr(x / (rise / (2.0 * _PHI_INV_08)))
    full = rise / _RC_2080
    return np.where(x <= -0.5 * full, 0.0, np.where(x >= 0.5 * full, 1.0, 0.5 * (1.0 + np.sin(np.pi * x / full))))


def drive_waveform(
    levels: Sequence[float] | np.ndarray,
    *,
    baud: float,
    samples_per_symbol: int,
    amplitude_v: float = 1.0,
    offset_v: float = 0.0,
    rise_time_20_80_s: float = 0.0,
    shape: Literal["gaussian", "raised_cosine"] = "gaussian",
    jitter_rms_s: float = 0.0,
    seed: Any = None,
    t0: float = 0.0,
) -> Waveform:
    """Electrical drive voltage for a sequence of normalized symbol levels.

    The voltage is ``offset_v + amplitude_v * (L(t) - 1/2)``, so a level
    sequence in ``[0, 1]`` swings ``amplitude_v`` peak to peak about
    ``offset_v``. ``L(t)`` changes at the symbol boundaries ``k / baud`` (each
    displaced by Gaussian random jitter of standard deviation
    ``jitter_rms_s``) through an edge of 20-80 % rise time
    ``rise_time_20_80_s``: a Gaussian (error-function) edge, or a
    raised-cosine edge of finite duration. The edges are evaluated in closed
    form at every sample (no filtering, no wrap-around); a zero rise time gives
    ideal rectangular NRZ.

    Examples
    --------
    >>> import numpy as np
    >>> w = drive_waveform([0, 1, 1, 0], baud=10e9, samples_per_symbol=8, amplitude_v=2.0)
    >>> w.unit, w.n_steps, [float(v) for v in np.asarray(w.samples)[::8]]
    ('V', 32, [-1.0, 1.0, 1.0, -1.0])
    """
    lv = np.asarray(levels, dtype=float).ravel()
    if lv.size == 0:
        raise ValueError("levels is empty")
    sps = int(samples_per_symbol)
    if sps < 1 or baud <= 0:
        raise ValueError("baud and samples_per_symbol must be positive")
    if shape not in ("gaussian", "raised_cosine"):
        raise ValueError("shape must be 'gaussian' or 'raised_cosine'")
    big_t = 1.0 / baud
    dt = big_t / sps
    n_sym = lv.size
    n = n_sym * sps
    tau = dt * np.arange(n)
    if rise_time_20_80_s <= 0.0 and jitter_rms_s <= 0.0:
        lev = np.repeat(lv, sps)
    else:
        rng = np.random.default_rng(seed)
        jitter = rng.normal(0.0, jitter_rms_s, n_sym) if jitter_rms_s > 0 else np.zeros(n_sym)
        t_edge = np.arange(n_sym) * big_t + jitter                     # boundary i between symbols i-1 and i
        width = (3.5 * rise_time_20_80_s if shape == "gaussian" else 0.5 * rise_time_20_80_s / _RC_2080)
        r = int(math.ceil((width + 6.0 * jitter_rms_s) / big_t)) + 2
        sym = np.arange(n) // sps
        cand = sym[:, None] + np.arange(-r, r + 1)[None, :]
        valid = (cand >= 1) & (cand < n_sym)
        ci = np.clip(cand, 1, n_sym - 1)
        d_lev = lv[ci] - lv[ci - 1]
        contrib = np.where(valid, d_lev * _edge(tau[:, None] - t_edge[ci], shape, rise_time_20_80_s), 0.0)
        lev = lv[np.clip(sym - r - 1, 0, n_sym - 1)] + contrib.sum(axis=1)
    volts = offset_v + amplitude_v * (lev - 0.5)
    return Waveform(xp.asarray(volts), dt, t0, None, "V")


# --------------------------------------------------------------------------- #
# CW laser with phase noise and RIN
# --------------------------------------------------------------------------- #
def cw_laser(
    power_w: float,
    n_steps: int,
    dt: float,
    *,
    linewidth_hz: float = 0.0,
    rin_db_hz: float | None = None,
    carrier_wl: float = 1.55,
    offset_hz: float = 0.0,
    seed: Any = None,
) -> Waveform:
    """A CW laser envelope with a Lorentzian linewidth and white intensity noise.

    The phase performs a Wiener random walk,
    ``phi_{k+1} = phi_k + N(0, 2 pi linewidth_hz dt)``, whose field spectrum is
    a Lorentzian of full width at half maximum ``linewidth_hz`` (the field
    autocorrelation decays as ``exp(-pi linewidth |tau|)``). The relative
    intensity noise is white with one-sided PSD ``RIN`` (dB/Hz) up to the
    Nyquist frequency: per-sample relative variance ``RIN f_s / 2``. A
    measured RIN spectrum (relaxation-oscillation peak) is not modelled -- use
    :func:`rate_equation_laser` for that.

    Examples
    --------
    >>> w = cw_laser(1e-3, 4, 1e-12)
    >>> [round(float(p) * 1e3, 12) for p in w.power()]
    [1.0, 1.0, 1.0, 1.0]
    """
    if power_w < 0 or linewidth_hz < 0:
        raise ValueError("power_w and linewidth_hz must be non-negative")
    n = int(n_steps)
    rng = np.random.default_rng(seed)
    t = dt * np.arange(n)
    phase = 2.0 * np.pi * offset_hz * t
    if linewidth_hz > 0:
        steps = rng.normal(0.0, math.sqrt(2.0 * math.pi * linewidth_hz * dt), n)
        steps[0] = 0.0
        phase = phase + np.cumsum(steps)
    power = np.full(n, float(power_w))
    if rin_db_hz is not None:
        var = 10.0 ** (rin_db_hz / 10.0) * 0.5 / dt
        power = np.clip(power * (1.0 + rng.normal(0.0, math.sqrt(var), n)), 0.0, None)
    return Waveform(xp.asarray(np.sqrt(power) * np.exp(1j * phase)), dt, 0.0, carrier_wl, "sqrt(W)")


# --------------------------------------------------------------------------- #
# Rate-equation laser
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LaserParams:
    """Single-mode semiconductor laser parameters (CGS densities, SI times).

    The defaults are *illustrative* values of the magnitude used in textbook
    examples of 1.55-µm InGaAsP lasers (cf. Agrawal, *Fiber-Optic Communication
    Systems*, 4th ed., Ch. 3; Coldren, Corzine & Masanovic, *Diode Lasers and
    Photonic Integrated Circuits*, 2nd ed., Ch. 5). They are not a calibrated
    device: extract parameters from measured L-I, RIN and modulation-response
    data for design work.

    Attributes
    ----------
    volume_cm3
        Active-region volume ``V``.
    eta_i
        Injection efficiency.
    tau_n_s
        Carrier lifetime (linear recombination).
    tau_p_s
        Photon lifetime ``1 / (v_g (alpha_i + alpha_m))``.
    v_g_cm_s
        Group velocity.
    a_cm2
        Differential gain ``dg/dN``.
    n_tr_cm3
        Transparency carrier density.
    epsilon_cm3
        Gain-compression coefficient (``g / (1 + epsilon S)``).
    beta_sp
        Spontaneous-emission coupling factor.
    gamma
        Optical confinement factor.
    alpha_h
        Linewidth-enhancement (Henry) factor.
    eta_out
        Fraction of the cavity photon loss emitted through the output facet,
        ``alpha_m,out / (alpha_i + alpha_m)``; the differential quantum
        efficiency is ``eta_d = eta_i eta_out``.
    wl_um
        Emission wavelength.
    gain_model
        ``"linear"``: ``g = a (N - N_tr)``; ``"log"``:
        ``g = a N_tr ln(N / N_tr)`` (same differential gain at transparency).
    """

    volume_cm3: float = 1.0e-10
    eta_i: float = 0.8
    tau_n_s: float = 2.2e-9
    tau_p_s: float = 1.334e-12
    v_g_cm_s: float = 2.99792458e10 / 3.4
    a_cm2: float = 2.5e-16
    n_tr_cm3: float = 1.0e18
    epsilon_cm3: float = 1.5e-17
    beta_sp: float = 1.0e-5
    gamma: float = 0.3
    alpha_h: float = 5.0
    eta_out: float = 0.265
    wl_um: float = 1.55
    gain_model: Literal["linear", "log"] = "linear"
    _extra: dict = field(default_factory=dict, compare=False, repr=False)

    def gain(self, n):
        """Material gain ``g(N)`` in 1/cm."""
        n = xp.asarray(n)
        if self.gain_model == "log":
            return self.a_cm2 * self.n_tr_cm3 * xp.log(xp.maximum(n, 1e-30) / self.n_tr_cm3)
        return self.a_cm2 * (n - self.n_tr_cm3)

    # -- analytic below/above-threshold quantities (beta_sp -> 0, epsilon -> 0) -- #
    @property
    def photon_energy_j(self) -> float:
        return H_PLANCK * C0 / (self.wl_um * 1e-6)

    @property
    def threshold_gain_cm(self) -> float:
        """``g_th = 1 / (Gamma v_g tau_p)``."""
        return 1.0 / (self.gamma * self.v_g_cm_s * self.tau_p_s)

    @property
    def threshold_density_cm3(self) -> float:
        """Carrier density at threshold."""
        if self.gain_model == "log":
            return self.n_tr_cm3 * math.exp(self.threshold_gain_cm / (self.a_cm2 * self.n_tr_cm3))
        return self.n_tr_cm3 + self.threshold_gain_cm / self.a_cm2

    @property
    def threshold_current_a(self) -> float:
        """``I_th = q V N_th / (eta_i tau_n)``."""
        return Q_E * self.volume_cm3 * self.threshold_density_cm3 / (self.eta_i * self.tau_n_s)

    @property
    def slope_efficiency_w_a(self) -> float:
        """``dP/dI = eta_i eta_out h nu / q`` above threshold."""
        return self.eta_i * self.eta_out * self.photon_energy_j / Q_E

    def output_power_w(self, s):
        """Output power ``eta_out h nu V S / (Gamma tau_p)`` for photon density ``S``."""
        return self.eta_out * self.photon_energy_j * self.volume_cm3 * xp.asarray(s) / (self.gamma * self.tau_p_s)

    def steady_state(self, current_a: float) -> tuple[float, float]:
        """Steady ``(N, S)`` of the full rate equations (with ``beta_sp`` and ``epsilon``)."""
        from scipy.optimize import brentq

        q_v = Q_E * self.volume_cm3
        if self.beta_sp == 0.0:
            # No spontaneous emission into the mode: below threshold S = 0; above it the gain is
            # clamped at g_th (1 + eps S) and the carrier balance fixes S.
            pump = self.eta_i * current_a / q_v
            if current_a <= self.threshold_current_a:
                return float(pump * self.tau_n_s), 0.0

            def n_of(s):
                g = self.threshold_gain_cm * (1.0 + self.epsilon_cm3 * s)
                if self.gain_model == "log":
                    return self.n_tr_cm3 * math.exp(g / (self.a_cm2 * self.n_tr_cm3))
                return self.n_tr_cm3 + g / self.a_cm2

            def balance(s):
                return pump - n_of(s) / self.tau_n_s - s / (self.gamma * self.tau_p_s)

            s_max = self.gamma * self.tau_p_s * pump
            s_ss = brentq(balance, 0.0, s_max, xtol=1e-12 * s_max, rtol=1e-14)
            return float(n_of(s_ss)), float(s_ss)

        def photons(n):
            g = float(self.gain(n))
            src = self.gamma * self.beta_sp * n / self.tau_n_s
            a2 = self.epsilon_cm3 / self.tau_p_s
            a1 = 1.0 / self.tau_p_s - self.gamma * self.v_g_cm_s * g - self.epsilon_cm3 * src
            if a2 == 0.0:
                return src / a1 if a1 > 0 else math.inf
            disc = a1 * a1 + 4.0 * a2 * src
            return (-a1 + math.sqrt(disc)) / (2.0 * a2)

        def resid(n):
            s = photons(n)
            g = float(self.gain(n))
            stim = self.v_g_cm_s * g * s / (1.0 + self.epsilon_cm3 * s)
            return self.eta_i * current_a / q_v - n / self.tau_n_s - stim

        lo = 1e-6 * self.n_tr_cm3
        hi = self.threshold_density_cm3
        while resid(hi) > 0:            # epsilon > 0 lets N rise above N_th
            hi *= 1.05
        n_ss = brentq(resid, lo, hi, xtol=1e-6, rtol=1e-14)
        return float(n_ss), float(photons(n_ss))

    def relaxation_frequency_hz(self, current_a: float) -> float:
        """``f_R = sqrt(v_g a S0 / tau_p) / (2 pi)`` at the steady photon density ``S0``."""
        _, s0 = self.steady_state(current_a)
        return math.sqrt(self.v_g_cm_s * self.a_cm2 * s0 / self.tau_p_s) / (2.0 * math.pi)

    def small_signal_response(self, f_hz, current_a: float):
        """Linearized ``dP_out / dI`` (W/A) at ``f_hz`` about the steady state at ``current_a``.

        Obtained from the Jacobian of the full rate equations (a numerical
        linearization, independent of the time-domain integrator).
        """
        n0, s0 = self.steady_state(current_a)
        x0 = np.array([n0, s0])
        jac = np.zeros((2, 2))
        for i in range(2):
            h = 1e-6 * x0[i]
            xp_, xm = x0.copy(), x0.copy()
            xp_[i] += h
            xm[i] -= h
            jac[:, i] = (_rhs_np(self, xp_, current_a) - _rhs_np(self, xm, current_a)) / (2 * h)
        b = np.array([self.eta_i / (Q_E * self.volume_cm3), 0.0])
        w = 2.0 * np.pi * np.atleast_1d(np.asarray(f_hz, dtype=float))
        out = []
        for wk in w:
            x = np.linalg.solve(1j * wk * np.eye(2) - jac, b)
            out.append(complex(self.output_power_w(1.0)) * x[1])
        return np.asarray(out).reshape(np.shape(f_hz))

    def linewidth_hz(self, current_a: float) -> float:
        """Schawlow-Townes-Henry linewidth ``R_sp (1 + alpha_h^2) / (4 pi P_ph)`` at ``current_a``.

        ``R_sp = beta_sp N V / tau_n`` is the spontaneous emission rate into the
        mode and ``P_ph = S V / Gamma`` the cavity photon number (Henry 1982).
        """
        n0, s0 = self.steady_state(current_a)
        r_sp = self.beta_sp * n0 * self.volume_cm3 / self.tau_n_s
        p_ph = s0 * self.volume_cm3 / self.gamma
        return r_sp * (1.0 + self.alpha_h ** 2) / (4.0 * math.pi * p_ph)


def _rhs_np(p: LaserParams, x: np.ndarray, current: float) -> np.ndarray:
    n, s = x
    g = float(p.gain(n))
    comp = 1.0 + p.epsilon_cm3 * s
    dn = p.eta_i * current / (Q_E * p.volume_cm3) - n / p.tau_n_s - p.v_g_cm_s * g * s / comp
    ds = p.gamma * p.v_g_cm_s * g * s / comp - s / p.tau_p_s + p.gamma * p.beta_sp * n / p.tau_n_s
    return np.array([dn, ds])


def rate_equation_laser(
    current_a,
    dt: float,
    params: LaserParams | None = None,
    *,
    noise: bool = True,
    seed: Any = None,
    substeps: int | None = None,
    initial: Literal["steady", "off"] = "steady",
) -> tuple[Waveform, Any]:
    """Integrate the single-mode laser rate equations for a drive current waveform.

    .. code-block:: text

        dN/dt   = eta_i I / (q V) - N / tau_n - v_g g(N) S / (1 + eps S)              + F_N
        dS/dt   = Gamma v_g g(N) S / (1 + eps S) - S / tau_p + Gamma beta_sp N / tau_n + F_S
        dphi/dt = (alpha_H / 2) (Gamma v_g g(N) - 1 / tau_p)                            + F_phi

    (Coldren, Corzine & Masanovic 2012, Ch. 5; the phase equation is the
    Henry form, Agrawal 2010, Sec. 3.5). The output field is
    ``sqrt(P) exp(j phi)`` with ``P = eta_out h nu V S / (Gamma tau_p)``.

    Each sample interval is split into ``substeps`` classical RK4 steps (by
    default enough that the step is at most ``tau_p / 5``) with the current
    interpolated linearly. With ``noise=True`` the Langevin forces are added
    per substep (Euler-Maruyama) with the standard photon-number / carrier-
    number diffusion coefficients (Henry, IEEE JQE 18, 259, 1982; Agrawal
    Sec. 3.5): ``<F_P F_P> = 2 R_sp P``, ``<F_N F_N> = 2 (R_sp P + N V / tau_n)``,
    ``<F_P F_N> = -R_sp P``, ``<F_phi F_phi> = R_sp / (2 P)``, with
    ``R_sp = beta_sp N V / tau_n``; they produce the Schawlow-Townes-Henry
    linewidth and the relaxation-oscillation RIN peak.

    Parameters
    ----------
    current_a
        Drive current samples in A (array or electrical :class:`Waveform`).
    dt
        Sample spacing (s) of ``current_a`` and of the output.
    initial
        ``"steady"`` starts at the steady state of the first current sample
        (:meth:`LaserParams.steady_state`); ``"off"`` from an empty cavity.

    Returns
    -------
    field, carrier_density
        The optical envelope (carrier ``params.wl_um``) and ``N`` (cm^-3) per
        sample.

    Examples
    --------
    >>> import numpy as np
    >>> p = LaserParams(beta_sp=0.0, epsilon_cm3=0.0)
    >>> i = np.full(200, 2 * p.threshold_current_a)
    >>> field, n = rate_equation_laser(i, 2e-12, p, noise=False)
    >>> slope = float(np.asarray(field.power())[-1]) / p.threshold_current_a
    >>> bool(abs(slope / p.slope_efficiency_w_a - 1) < 1e-9)          # P = eta_d h nu / q (I - I_th)
    True
    """
    p = params or LaserParams()
    cur = current_a.samples if isinstance(current_a, Waveform) else current_a
    cur = xp.asarray(cur, dtype=float)
    n_steps = int(cur.shape[0])
    n_sub = int(substeps) if substeps is not None else max(1, int(math.ceil(dt / (p.tau_p_s / 5.0))))
    h = dt / n_sub
    q_v = Q_E * p.volume_cm3
    v_p = p.volume_cm3 / p.gamma
    if initial == "steady":
        n0, s0 = p.steady_state(float(np.asarray(cur[0])))
    elif initial == "off":
        n0, s0 = 0.0, 0.0
    else:
        raise ValueError("initial must be 'steady' or 'off'")
    rng = np.random.default_rng(seed)
    xi = xp.asarray(rng.standard_normal((n_steps, n_sub, 3))) if noise else None

    def rhs(n, s, i):
        g = p.gain(n)
        comp = 1.0 + p.epsilon_cm3 * s
        dn = p.eta_i * i / q_v - n / p.tau_n_s - p.v_g_cm_s * g * s / comp
        ds = p.gamma * p.v_g_cm_s * g * s / comp - s / p.tau_p_s + p.gamma * p.beta_sp * n / p.tau_n_s
        dph = 0.5 * p.alpha_h * (p.gamma * p.v_g_cm_s * g - 1.0 / p.tau_p_s)
        return dn, ds, dph

    def substep(state, i_a, i_b, frac0, xi_k):
        n, s, ph = state
        i0 = i_a + (i_b - i_a) * frac0
        im = i_a + (i_b - i_a) * (frac0 + 0.5 / n_sub)
        i1 = i_a + (i_b - i_a) * (frac0 + 1.0 / n_sub)
        k1 = rhs(n, s, i0)
        k2 = rhs(n + 0.5 * h * k1[0], s + 0.5 * h * k1[1], im)
        k3 = rhs(n + 0.5 * h * k2[0], s + 0.5 * h * k2[1], im)
        k4 = rhs(n + h * k3[0], s + h * k3[1], i1)
        n_new = n + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        s_new = s + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ph_new = ph + h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if xi_k is not None:
            pn = xp.maximum(s, 0.0) * v_p                     # photon number
            nc = xp.maximum(n, 0.0) * p.volume_cm3            # carrier number
            r_sp = p.beta_sp * nc / p.tau_n_s
            d_pp = 2.0 * r_sp * pn
            d_nn = 2.0 * (r_sp * pn + nc / p.tau_n_s)
            d_pn = -r_sp * pn
            # Cholesky of [[d_pp, d_pn], [d_pn, d_nn]] * h
            l11 = xp.sqrt(xp.maximum(d_pp * h, 0.0))
            l21 = xp.where(l11 > 0, d_pn * h / xp.where(l11 > 0, l11, 1.0), 0.0)
            l22 = xp.sqrt(xp.maximum(d_nn * h - l21 * l21, 0.0))
            dp = l11 * xi_k[0]
            dnc = l21 * xi_k[0] + l22 * xi_k[1]
            s_new = s_new + dp / v_p
            n_new = n_new + dnc / p.volume_cm3
            ph_new = ph_new + xp.sqrt(r_sp / (2.0 * xp.maximum(pn, 1.0)) * h) * xi_k[2]
        return n_new, xp.maximum(s_new, 0.0), ph_new

    i_next = xp.concatenate([cur[1:], cur[-1:]])
    if HAS_JAX:
        from jax import lax

        def body(state, inp):
            if noise:
                i_a, i_b, xi_s = inp
            else:
                i_a, i_b = inp
                xi_s = None
            for j in range(n_sub):
                state = substep(state, i_a, i_b, j / n_sub, None if xi_s is None else xi_s[j])
            return state, (state[0], state[1], state[2])

        xs = (cur, i_next, xi) if noise else (cur, i_next)
        init = (xp.asarray(n0, dtype=float), xp.asarray(s0, dtype=float), xp.asarray(0.0))
        _, (n_tr, s_tr, ph_tr) = lax.scan(body, init, xs)
    else:
        state = (float(n0), float(s0), 0.0)
        n_tr = np.empty(n_steps)
        s_tr = np.empty(n_steps)
        ph_tr = np.empty(n_steps)
        cur_n, nxt_n = np.asarray(cur), np.asarray(i_next)
        xi_n = None if xi is None else np.asarray(xi)
        for k in range(n_steps):
            for j in range(n_sub):
                state = substep(state, cur_n[k], nxt_n[k], j / n_sub, None if xi_n is None else xi_n[k, j])
            n_tr[k], s_tr[k], ph_tr[k] = state
    # sample k reports the state at the *start* of interval k (the drive sample k)
    n_out = xp.concatenate([xp.asarray([n0], dtype=float), xp.asarray(n_tr)[:-1]])
    s_out = xp.concatenate([xp.asarray([s0], dtype=float), xp.asarray(s_tr)[:-1]])
    ph_out = xp.concatenate([xp.zeros(1), xp.asarray(ph_tr)[:-1]])
    power = p.output_power_w(s_out)
    env = xp.sqrt(power) * xp.exp(1j * ph_out)
    return Waveform(env, dt, 0.0, p.wl_um, "sqrt(W)"), n_out
