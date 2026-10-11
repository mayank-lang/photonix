"""Convert photonix frequency-domain models into time-domain elements.

:func:`to_td` turns any model ``f(*, wl, **settings) -> SDict`` into a
:class:`~photonix.timedomain.elements.TDElement` for a given carrier and time
step. The rules, in order:

1. A model with a ``td_model`` attribute supplies its own element (an element
   instance, or a callable ``td_model(*, carrier_wl, dt, **settings)``).
2. Waveguides (:func:`~photonix.components.straight`, ``bend``,
   ``bend_euler``, ``taper``, ``waveguide_route``) and the waveguide-type
   phase shifters become a :class:`~photonix.timedomain.elements.DelayLineTD`
   with the model's exact carrier transmission ``S21(f_c)`` and group delay
   ``tau = -d arg S21 / d omega`` (``n_g L / c`` for the linear-dispersion
   models, whose phase is exactly linear in frequency, so the conversion is
   exact apart from the fractional-delay interpolation).
3. Frequency-flat devices (directional couplers, MMIs, Y-branches,
   crossings, attenuators, terminators, photodetector terminations) become a
   :class:`~photonix.timedomain.elements.MemorylessTD` of the carrier
   S-matrix.
4. Ring resonators (:func:`~photonix.components.all_pass_ring`,
   :func:`~photonix.components.add_drop_ring`) become their physical
   structure -- couplers plus ring waveguide delays -- which is exactly the
   circuit the closed forms describe.
5. Anything else is sampled over the envelope band ``f_c +/- 1/(2 dt)``.
   Port pairs whose response is a pure delay (constant magnitude, linear
   phase) get a fractional-delay interpolator; the rest a causal FIR, the
   truncated inverse DFT of the band-limited response with the bulk delay
   carried by the tap offset. A non-causal model response (e.g. a zero-phase
   Gaussian filter) is made causal by adding latency, which is reported and
   warned about.

Every conversion is validated when its inputs are concrete (flatness of
"memoryless" devices, linearity of the phase of "delay" devices, FIR fit
error); if a rule's assumption fails, the generic sampled path is used.
"""
from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from photonix.core.backend import xp
from photonix.core.constants import C0_UM_S

from .elements import (
    FIRTD,
    CompositeTD,
    DelayLineTD,
    MemorylessTD,
    TDElement,
    TimeVaryingTD,
    _concrete_float,
    centered_frequencies,
    design_fir,
    fir_response,
    lagrange_taps,
    parse_interp,
    window_taps,
)

__all__ = ["to_td", "to_td_time_varying", "fir_from_model", "FIRDesign"]


def _components():
    from photonix import components as c

    return c


def _model_params(model: Callable, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Keyword defaults of ``model`` updated with ``settings`` (``wl`` excluded)."""
    if "wl" in settings:
        raise ValueError("'wl' is set by the time-domain conversion (the carrier); remove it from the settings")
    out: dict[str, Any] = {}
    try:
        sig = inspect.signature(model)
    except (TypeError, ValueError):
        return dict(settings)
    for name, par in sig.parameters.items():
        if name != "wl" and par.default is not inspect.Parameter.empty:
            out[name] = par.default
    out.update(settings)
    return out


def _carrier_hz(carrier_wl: float) -> float:
    return C0_UM_S / float(carrier_wl)


def _evaluate(model: Callable, settings: Mapping[str, Any], nu, carrier_wl: float) -> dict:
    """``model`` at optical frequencies ``f_c + nu`` (``nu`` in Hz)."""
    f = _carrier_hz(carrier_wl) + xp.asarray(nu, dtype=float)
    return dict(model(wl=C0_UM_S / f, **settings))


def _ports_of(sdict: Mapping) -> tuple[str, ...]:
    from photonix.core.types import ports_of

    return tuple(ports_of(dict(sdict)))


def _is_concrete(*values) -> bool:
    try:
        for v in values:
            if isinstance(v, Mapping):
                for x in v.values():
                    np.asarray(x)
            else:
                np.asarray(v)
        return True
    except Exception:  # noqa: BLE001 - tracers
        return False


def _group_delay_at_carrier(model, settings, carrier_wl: float, pair: tuple[str, str]):
    """Group delay ``-d arg S / d omega`` of ``S[pair]`` at the carrier (two-stage, unwrapping-safe)."""
    def ratio(df):
        sp = _evaluate(model, settings, xp.asarray([df]), carrier_wl)[pair]
        sm = _evaluate(model, settings, xp.asarray([-df]), carrier_wl)[pair]
        return xp.reshape(xp.asarray(sp, dtype=complex) / xp.asarray(sm, dtype=complex), ())

    df1 = 1e5                                   # unambiguous for |tau| < 2.5 us
    tau0 = -xp.angle(ratio(df1)) / (4.0 * math.pi * df1)
    try:
        t0 = abs(float(np.asarray(tau0)))
    except Exception:  # noqa: BLE001 - traced: refine with a fixed step
        t0 = 1e-9
    df2 = min(1e9, 0.25 / max(t0, 1e-15))       # phase step <= pi/... for the refinement
    rot = xp.exp(1j * 4.0 * math.pi * df2 * tau0)
    return tau0 - xp.angle(ratio(df2) * rot) / (4.0 * math.pi * df2)


# --------------------------------------------------------------------------- #
# Rule: delay lines
# --------------------------------------------------------------------------- #
def _delay_line(model, settings, *, carrier_wl, dt, interp, like, check: bool, fir_tol: float):
    s_c = _evaluate(model, settings, xp.zeros(1), carrier_wl)
    keys = set(s_c)
    if keys != {("o1", "o2"), ("o2", "o1")}:
        return None
    t12 = xp.reshape(xp.asarray(s_c[("o1", "o2")], dtype=complex), ())
    t21 = xp.reshape(xp.asarray(s_c[("o2", "o1")], dtype=complex), ())
    tau = _group_delay_at_carrier(model, settings, carrier_wl, ("o1", "o2"))
    if check and _is_concrete(t12, t21, tau):
        if abs(complex(t12) - complex(t21)) > 1e-12 * max(abs(complex(t12)), 1e-300):
            return None                       # non-reciprocal: not a plain delay line
        n_f = 64
        nu = (np.arange(n_f) - n_f // 2) / (n_f * dt)
        s = np.asarray(_evaluate(model, settings, nu, carrier_wl)[("o1", "o2")], dtype=complex)
        ideal = complex(t12) * np.exp(-2j * np.pi * nu * float(tau))
        if np.max(np.abs(s - ideal)) > fir_tol * max(abs(complex(t12)), 1e-300):
            return None                       # not linear-phase over the band
    phase, amp = -xp.angle(t12), xp.abs(t12)
    if _is_concrete(tau, t12):
        tau, phase, amp = float(np.asarray(tau)), float(np.asarray(phase)), float(np.asarray(amp))
    if isinstance(like, DelayLineTD):
        return DelayLineTD(tau_s=tau, carrier_phase=phase, loss_amp=amp, interp=like.interp, split=like.split)
    return DelayLineTD(tau_s=tau, carrier_phase=phase, loss_amp=amp, interp=interp)


# --------------------------------------------------------------------------- #
# Rule: memoryless devices
# --------------------------------------------------------------------------- #
def _memoryless(model, settings, *, carrier_wl, dt, check: bool, fir_tol: float):
    s_c = model(wl=carrier_wl, **settings)
    ports = _ports_of(s_c)
    el = MemorylessTD.from_sdict(s_c, ports)
    if check and _is_concrete(el.S):
        nu = np.array([-0.5, -0.25, 0.25, 0.5]) / dt
        s = _evaluate(model, settings, nu, carrier_wl)
        for key, val in s.items():
            v = np.broadcast_to(np.asarray(val, dtype=complex), nu.shape)
            ref = np.asarray(el.S)[ports.index(key[1]), ports.index(key[0])]
            if np.max(np.abs(v - ref)) > fir_tol:
                return None
    return el


# --------------------------------------------------------------------------- #
# Rule: rings as structures
# --------------------------------------------------------------------------- #
def _sub_like(like, name):
    if isinstance(like, CompositeTD):
        return like.elements.get(name)
    return None


def _ring_waveguide(p, length, *, carrier_wl, dt, interp, like):
    c = _components()
    wg = {"length": length, "neff": p["neff"], "ng": p["ng"], "wl0": p["wl0"], "loss_db_cm": p["loss_db_cm"]}
    return _delay_line(c.straight, wg, carrier_wl=carrier_wl, dt=dt, interp=interp, like=like, check=False,
                       fir_tol=0.0)


def _all_pass_ring(model, settings, *, carrier_wl, dt, interp, like):
    c = _components()
    p = _model_params(model, settings)
    ring = _ring_waveguide(p, 2.0 * math.pi * p["radius"], carrier_wl=carrier_wl, dt=dt, interp=interp,
                           like=_sub_like(like, "ring"))
    cpl = MemorylessTD.from_sdict(c.directional_coupler(wl=carrier_wl, coupling=p["coupling"]))
    return CompositeTD(
        {"c": cpl, "ring": ring},
        {("c", "o3"): ("ring", "o1"), ("ring", "o2"): ("c", "o2")},
        {"o1": ("c", "o1"), "o2": ("c", "o4")},
    )


def _add_drop_ring(model, settings, *, carrier_wl, dt, interp, like):
    c = _components()
    p = _model_params(model, settings)
    half = math.pi * p["radius"]
    right = _ring_waveguide(p, half, carrier_wl=carrier_wl, dt=dt, interp=interp, like=_sub_like(like, "right"))
    left = _ring_waveguide(p, half, carrier_wl=carrier_wl, dt=dt, interp=interp, like=_sub_like(like, "left"))
    c0 = MemorylessTD.from_sdict(c.directional_coupler(wl=carrier_wl, coupling=p["coupling1"]))
    c1 = MemorylessTD.from_sdict(c.directional_coupler(wl=carrier_wl, coupling=p["coupling2"]))
    # Same wiring as photonix.circuit.crow with one ring (validated against the closed form).
    return CompositeTD(
        {"c0": c0, "c1": c1, "right": right, "left": left},
        {("c0", "o4"): ("right", "o1"), ("right", "o2"): ("c1", "o3"),
         ("c1", "o2"): ("left", "o1"), ("left", "o2"): ("c0", "o1")},
        {"o1": ("c0", "o2"), "o2": ("c0", "o3"), "o3": ("c1", "o1"), "o4": ("c1", "o4")},
    )


# --------------------------------------------------------------------------- #
# Generic sampled conversion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FIRDesign:
    """Static design of a sampled FIR element (kept so it can be re-fitted from traced responses).

    ``pairs[(in, out)]`` is ``("delay", n0, order, latency)`` for a
    fractional-delay pair or ``("fir", n0, n_taps, latency)`` for a
    tapered inverse-DFT pair (:func:`~photonix.timedomain.elements.design_fir`);
    ``n_freq`` is the DFT size and ``band`` the fraction of the Nyquist band
    reproduced exactly.
    """

    n_freq: int
    band: float
    pairs: Mapping[tuple[str, str], tuple]
    #: Common scale (<= 1) applied to every pair so that a passive model gives a strictly passive FIR.
    gain: float = 1.0


def fir_from_model(
    model: Callable,
    settings: Mapping[str, Any] | None = None,
    *,
    carrier_wl: float = 1.55,
    dt: float,
    fir_tol: float = 1e-4,
    max_taps: int = 4096,
    band: float = 0.5,
    interp: str = "lagrange3",
    like: FIRTD | None = None,
) -> FIRTD:
    """Causal FIR element from a model sampled over the envelope band ``f_c +/- 1/(2 dt)``.

    For each port pair:

    * a pure delay (constant magnitude, linear phase across the whole band)
      becomes a Lagrange fractional-delay filter (``interp``), as in
      :class:`~photonix.timedomain.elements.DelayLineTD`;
    * any other response is designed by
      :func:`~photonix.timedomain.elements.design_fir`: the response times a
      smooth taper (flat for ``|nu| <= band / (2 dt)``, zero at Nyquist) is
      inverse-transformed and truncated so the discarded taps sum to at most
      ``fir_tol / 2`` of the peak response. The in-band error is then below
      ``fir_tol / 2`` and the gain never exceeds the model's by more than
      that, so a passive model stays passive (stable in feedback loops).

    The DFT grid is lengthened until the response is time-limited within it
    (no aliasing). If any pair is non-causal, the same latency is added to
    every pair (keeping their relative timing) and reported as
    :attr:`FIRTD.added_delay_s`. Outside ``band`` the element rolls off.

    Examples
    --------
    A zero-phase Gaussian grating-coupler passband needs latency to be causal:

    >>> import warnings, photonix as px
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     el = fir_from_model(px.components.grating_coupler, {"bandwidth": 0.004}, dt=0.1e-12)
    >>> el.added_delay_s > 0, el.fit_error < 1e-4
    (True, True)
    """
    settings = dict(settings or {})
    fc = _carrier_hz(carrier_wl)
    if 0.5 / dt >= fc:
        raise ValueError("the envelope band f_c +/- 1/(2 dt) reaches zero frequency; use a smaller dt")
    if not 0.0 < band <= 1.0:
        raise ValueError("band must be in (0, 1]")
    if like is not None:
        if like.design is None:
            raise ValueError("'like' must be an FIRTD produced by fir_from_model (and not truncated)")
        return _fir_redesign(model, settings, carrier_wl=carrier_wl, dt=dt, like=like)
    kind, order = parse_interp(interp)
    if kind != "lagrange":
        order = 3                         # an FIR element interpolates pure delays with Lagrange taps
    ports = _ports_of(model(wl=carrier_wl, **settings))
    n_freq = 1024
    tol_design = fir_tol
    while True:
        design, s, scale = _fir_layout(model, settings, ports, carrier_wl, dt, n_freq, tol_design, band, order)
        n_freq = design.n_freq
        el = _fir_build(ports, dt, design, s, model, settings, carrier_wl)
        # Strict passivity: a passive model must not become (slightly) active anywhere in the band,
        # or a lossless feedback loop built from it would grow without bound over long runs.
        fir_resp = {k: fir_response(n0, np.asarray(b), n_freq) for k, (n0, b) in el.taps.items()}
        sigma_fir = _max_singular_value(ports, fir_resp)
        if _max_singular_value(ports, s) <= 1.0 + 1e-9 and sigma_fir > 1.0:
            design = FIRDesign(n_freq, float(band), design.pairs, gain=1.0 / sigma_fir)
            el = _fir_build(ports, dt, design, s, model, settings, carrier_wl)
        if el.fit_error <= fir_tol * max(scale, 1e-300) or tol_design < fir_tol / 60:
            break
        tol_design /= 4.0                     # the passivity scaling ate the margin: tighten and redo
    latency = max((spec[-1] for spec in design.pairs.values()), default=0)
    if latency:
        warnings.warn(
            f"model {getattr(model, '__name__', model)!r} has a non-causal response at this carrier; "
            f"added {latency} samples ({latency * dt:.3g} s) of latency to make it causal",
            stacklevel=2,
        )
    longest = max((int(np.shape(b)[0]) for _, b in el.taps.values()), default=0)
    if longest > max_taps:
        warnings.warn(
            f"model {getattr(model, '__name__', model)!r} needs {longest} > max_taps={max_taps} FIR taps at "
            f"dt = {dt:g} s (a long impulse response, e.g. a high-Q resonance); the fit was truncated. "
            "Model the device structurally (rings, delays) or with a rational (pole-residue) model.",
            stacklevel=2,
        )
        el = _truncate(el, max_taps, s, dt)
    if el.fit_error > fir_tol * max(scale, 1e-300):
        warnings.warn(
            f"FIR fit of {getattr(model, '__name__', model)!r} has a maximum in-band error of {el.fit_error:.2e} "
            f"(> fir_tol={fir_tol:g} of the peak response)",
            stacklevel=2,
        )
    return el


def _fir_layout(model, settings, ports, carrier_wl, dt, n_freq, tol, band, order):
    """Choose the DFT size and every pair's static tap window (see :func:`fir_from_model`)."""
    while True:
        nu = centered_frequencies(n_freq, dt)
        s = _evaluate(model, settings, nu, carrier_wl)
        s = {k: np.broadcast_to(np.asarray(v, dtype=complex), nu.shape) for k, v in s.items()}
        scale = max((float(np.max(np.abs(v))) for v in s.values()), default=0.0)
        tol_abs = tol * max(scale, 1e-300)
        pairs: dict[tuple[str, str], tuple] = {}
        aliased = False
        for key, v in s.items():
            if not np.any(np.abs(v) > 1e-14 * max(scale, 1e-300)):
                continue
            tau = _concrete_float(_group_delay_at_carrier(model, settings, carrier_wl, key), "group delay")
            a0 = complex(v[n_freq // 2])
            ideal = a0 * np.exp(-2j * np.pi * nu * tau)
            if abs(a0) > 0 and tau >= 0 and np.max(np.abs(v - ideal)) <= 0.5 * tol_abs:
                pairs[key] = ("delay", max(int(math.floor(tau / dt - (order - 1) / 2.0)), 0), order, 0)
                continue
            if abs(a0) > 1e-3 * scale:
                tau_b = tau                   # unambiguous carrier group delay
            else:                             # energy-weighted delay from the sampled phase steps
                w = np.abs(v[1:] * v[:-1])
                step = np.angle(v[1:] * np.conj(v[:-1]))
                tau_b = float(-np.sum(w * step) / max(float(np.sum(w)), 1e-300) * n_freq * dt / (2 * np.pi))
            n0, taps, _ = design_fir(v, dt, tol=tol_abs, band=band, n_bulk=int(round(tau_b / dt)))
            if taps.size > n_freq // 4 and n_freq < 2 ** 18:
                aliased = True                # not time-limited within the DFT window: refine
                break
            pairs[key] = ("fir", int(n0), int(taps.size), 0)
        if not aliased:
            break
        n_freq *= 2
    # One common latency makes every pair causal and keeps their relative timing.
    latency = max(0, -min((spec[1] for spec in pairs.values()), default=0))
    design = FIRDesign(n_freq, float(band), {k: (*spec[:-1], latency) for k, spec in pairs.items()})
    return design, s, scale


def _max_singular_value(ports, responses) -> float:
    """Largest singular value over frequency of the matrix built from ``{(in, out): response}``."""
    idx = {p: i for i, p in enumerate(ports)}
    n_f = int(np.shape(next(iter(responses.values())))[0]) if responses else 0
    if not n_f:
        return 0.0
    mat = np.zeros((n_f, len(ports), len(ports)), dtype=complex)
    for (pi, po), r in responses.items():
        mat[:, idx[po], idx[pi]] = np.asarray(r)
    return float(np.max(np.linalg.svd(mat, compute_uv=False)))


def _fir_taps(spec, v_centered, dt, band, tau, a0):
    if spec[0] == "delay":
        _, n0, order, latency = spec
        return n0 + latency, a0 * lagrange_taps(tau / dt - n0, order).astype(complex)
    _, n0, n_taps, latency = spec
    taps = window_taps(v_centered, dt, n0, n_taps, band)
    # Exact at the carrier (nu = 0): a complex gain correction of order fir_tol.
    s0 = v_centered[int(v_centered.shape[0]) // 2]
    h0 = xp.sum(taps)
    taps = xp.where(xp.abs(h0) > 0, s0 / xp.where(xp.abs(h0) > 0, h0, 1.0), 1.0) * taps
    return n0 + latency, taps


def _in_band_error(design: FIRDesign, taps, s, dt) -> float:
    nu = centered_frequencies(design.n_freq, dt)
    inband = np.abs(nu) * 2.0 * dt <= design.band + 1e-12
    err = 0.0
    for key, (n0, b) in taps.items():
        spec = design.pairs[key]
        if spec[0] == "delay":
            continue                      # delay pairs: interpolation error, as for DelayLineTD
        fit = fir_response(n0, np.asarray(b), design.n_freq)
        target = np.asarray(s[key]) * np.exp(-2j * np.pi * nu * spec[-1] * dt)    # including the added latency
        err = max(err, float(np.max(np.abs(fit - target)[inband])))
    return err


def _fir_build(ports, dt, design: FIRDesign, s, model, settings, carrier_wl) -> FIRTD:
    taps: dict[tuple[str, str], tuple[int, Any]] = {}
    s_c = None
    for key, spec in design.pairs.items():
        tau = a0 = None
        if spec[0] == "delay":
            tau = _group_delay_at_carrier(model, settings, carrier_wl, key)
            s_c = model(wl=carrier_wl, **settings) if s_c is None else s_c
            a0 = xp.reshape(xp.asarray(s_c[key], dtype=complex), ())
        v = xp.broadcast_to(xp.asarray(s[key], dtype=complex), (design.n_freq,))
        n0, b = _fir_taps(spec, v, dt, design.band, tau, a0)
        taps[key] = (n0, b * design.gain if design.gain != 1.0 else b)
    err = float("nan")
    if _is_concrete(*[t[1] for t in taps.values()]) and all(_is_concrete(s[k]) for k in design.pairs):
        err = _in_band_error(design, taps, s, dt)
    latency = max((spec[-1] for spec in design.pairs.values()), default=0)
    return FIRTD(ports, dt, taps, fit_error=err, added_delay_s=latency * dt, design=design)


def _truncate(el: FIRTD, max_taps: int, s, dt) -> FIRTD:
    taps = {}
    for key, (n0, b) in el.taps.items():
        b = np.asarray(b)
        if b.size > max_taps:
            energy = np.convolve(np.abs(b) ** 2, np.ones(max_taps), mode="valid")
            start = int(np.argmax(energy))
            b = b[start:start + max_taps]
            n0 = n0 + start
        taps[key] = (n0, b)
    err = _in_band_error(el.design, taps, s, dt) if el.design is not None else float("nan")
    return FIRTD(el.ports, el.dt, taps, fit_error=err, added_delay_s=el.added_delay_s, design=None)


def _fir_redesign(model, settings, *, carrier_wl, dt, like: FIRTD) -> FIRTD:
    design: FIRDesign = like.design
    nu = centered_frequencies(design.n_freq, dt)
    s = _evaluate(model, settings, nu, carrier_wl)
    missing = [k for k in design.pairs if k not in s]
    if missing:
        raise ValueError(f"the re-evaluated model lacks port pairs {missing}")
    return _fir_build(like.ports, dt, design, s, model, settings, carrier_wl)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def _rule_tables():
    c = _components()
    delay = (c.straight, c.bend, c.bend_euler, c.taper, c.waveguide_route, c.phase_shifter,
             c.heater_phase_shifter, c.thermo_optic_phase_shifter)
    flat = (c.directional_coupler, c.coupler, c.ring_coupler, c.mmi1x2, c.mmi2x2, c.mmi_nxn, c.y_branch,
            c.crossing, c.attenuator, c.terminator, c.photodetector)
    rings = {c.all_pass_ring: _all_pass_ring, c.add_drop_ring: _add_drop_ring}
    return delay, flat, rings


def to_td(
    model: Callable,
    settings: Mapping[str, Any] | None = None,
    *,
    carrier_wl: float = 1.55,
    dt: float,
    interp: str = "lagrange3",
    fir_tol: float = 1e-4,
    max_taps: int = 4096,
    like: TDElement | None = None,
) -> TDElement:
    """Time-domain element of ``model(wl=..., **settings)`` about ``carrier_wl``.

    Parameters
    ----------
    model
        Any photonix model; see the module docstring for the conversion rules.
    settings
        Model parameters (may be JAX tracers when ``like`` is given).
    carrier_wl, dt
        Carrier wavelength (µm) and time step (s).
    interp
        Fractional-delay interpolation for delay lines (``"lagrange3"``,
        ``"thiran3"``, ``"nearest"``, ...).
    fir_tol
        Tolerance of the rule checks and of generic FIR fits.
    max_taps
        Cap on the FIR length of a generic conversion.
    like
        A previously converted (frozen) element whose static layout -- integer
        delays, tap windows -- is reused, so the element can be rebuilt from
        traced settings with an unchanged structure.

    Examples
    --------
    >>> import photonix as px
    >>> el = to_td(px.components.straight, {"length": 2000.0}, dt=0.1e-12)
    >>> type(el).__name__, round(el.tau_s * 1e12, 2)     # n_g L / c = 4.2 * 2 mm / c
    ('DelayLineTD', 28.02)
    >>> type(to_td(px.components.mmi2x2, dt=0.1e-12)).__name__
    'MemorylessTD'
    """
    settings = dict(settings or {})
    if "wl" in settings:
        raise ValueError("'wl' is set by the conversion (the carrier); remove it from the settings")
    custom = getattr(model, "td_model", None)
    if isinstance(custom, TDElement):
        return custom
    if callable(custom):
        return custom(carrier_wl=carrier_wl, dt=dt, **settings)
    if 0.5 / dt >= _carrier_hz(carrier_wl):
        raise ValueError("the envelope band f_c +/- 1/(2 dt) reaches zero frequency; use a smaller dt")
    check = like is None
    delay, flat, rings = _rule_tables()
    if model in rings:
        if like is None or isinstance(like, CompositeTD):
            return rings[model](model, settings, carrier_wl=carrier_wl, dt=dt, interp=interp, like=like)
    if any(model is m for m in delay) and not callable(settings.get("neff")):
        if like is None or isinstance(like, DelayLineTD):
            el = _delay_line(model, settings, carrier_wl=carrier_wl, dt=dt, interp=interp, like=like, check=check,
                             fir_tol=fir_tol)
            if el is not None and like is None and isinstance(el.tau_s, float) and abs(el.tau_s) <= 1e-9 * dt:
                return MemorylessTD.from_sdict(model(wl=carrier_wl, **settings))   # a lumped phase / loss
            if el is not None:
                return el
        elif isinstance(like, MemorylessTD):
            return MemorylessTD.from_sdict(model(wl=carrier_wl, **settings), like.ports)
    if any(model is m for m in flat):
        if like is None or isinstance(like, MemorylessTD):
            el = _memoryless(model, settings, carrier_wl=carrier_wl, dt=dt, check=check, fir_tol=fir_tol)
            if el is not None:
                return el
    if like is not None:
        if isinstance(like, DelayLineTD):
            el = _delay_line(model, settings, carrier_wl=carrier_wl, dt=dt, interp=interp, like=like, check=False,
                             fir_tol=fir_tol)
            if el is None:
                raise ValueError("the re-evaluated model is no longer a two-port delay line")
            return el
        if isinstance(like, MemorylessTD):
            return MemorylessTD.from_sdict(model(wl=carrier_wl, **settings), like.ports)
        if isinstance(like, FIRTD):
            return fir_from_model(model, settings, carrier_wl=carrier_wl, dt=dt, like=like)
        raise TypeError(f"cannot rebuild a {type(like).__name__} from traced settings")
    # Unknown model: try a memoryless element and a delay line, then the sampled FIR.
    el = _memoryless(model, settings, carrier_wl=carrier_wl, dt=dt, check=True, fir_tol=fir_tol)
    if el is not None:
        return el
    el = _delay_line(model, settings, carrier_wl=carrier_wl, dt=dt, interp=interp, like=None, check=True,
                     fir_tol=fir_tol) if _is_two_port(model, settings, carrier_wl) else None
    if el is not None:
        return el
    return fir_from_model(model, settings, carrier_wl=carrier_wl, dt=dt, fir_tol=fir_tol, max_taps=max_taps,
                          interp=interp)


def _is_two_port(model, settings, carrier_wl) -> bool:
    try:
        return set(model(wl=carrier_wl, **settings)) == {("o1", "o2"), ("o2", "o1")}
    except Exception:  # noqa: BLE001
        return False


def to_td_time_varying(
    model: Callable,
    settings: Mapping[str, Any] | None = None,
    *,
    control_params: Mapping[str, str],
    carrier_wl: float = 1.55,
    dt: float,
    interp: str = "lagrange3",
    fir_tol: float = 1e-4,
    max_taps: int = 4096,
    like: TDElement | None = None,
) -> TDElement:
    """A quasi-static, control-driven element from a photonix model.

    ``control_params`` maps model parameters to control names, e.g.
    ``{"voltage": "v_rf"}``. The device responds instantly to its controls
    (quasi-static approximation: no RC or carrier dynamics -- filter the drive
    waveform for those).

    * A two-port delay line (e.g. :func:`~photonix.components.pn_phase_shifter`,
      :func:`~photonix.components.heater_phase_shifter`) becomes its passive
      delay at the reference settings followed by a lumped modulation
      ``S21(V(t)) / S21(V_ref)`` at the carrier: the modulation acts on the
      light as it leaves the section (a lumped, velocity-matched picture).
    * A frequency-flat device (e.g. a tunable coupler) becomes a
      :class:`~photonix.timedomain.elements.TimeVaryingTD` of its full carrier
      S-matrix.
    * Anything else becomes a carrier-only :class:`TimeVaryingTD` and a
      warning that its memory is neglected.

    Examples
    --------
    >>> import photonix as px
    >>> el = to_td_time_varying(px.components.pn_phase_shifter, {"length": 1000.0},
    ...                         control_params={"voltage": "v"}, dt=0.5e-12)
    >>> sorted(el.elements), el.ports
    (['delay', 'mod'], ('o1', 'o2'))
    """
    settings = dict(settings or {})
    for param in control_params:
        if param == "wl":
            raise ValueError("the wavelength cannot be a control")
    static = to_td(model, settings, carrier_wl=carrier_wl, dt=dt, interp=interp, fir_tol=fir_tol, max_taps=max_taps,
                   like=_sub_like(like, "delay") if isinstance(like, CompositeTD) else None)
    two_port = isinstance(static, DelayLineTD) or (
        isinstance(static, FIRTD) and set(static.taps) == {("o1", "o2"), ("o2", "o1")})
    if two_port:
        ref = model(wl=carrier_wl, **settings)
        ref12 = xp.asarray(ref[("o1", "o2")], dtype=complex)
        ref21 = xp.asarray(ref[("o2", "o1")], dtype=complex)
        mapping = dict(control_params)

        def fn(t, **controls):
            kw = dict(settings)
            for param, cname in mapping.items():
                kw[param] = controls[cname]
            s = model(wl=carrier_wl, **kw)
            return {("o1", "o2"): xp.asarray(s[("o1", "o2")], dtype=complex) / ref12,
                    ("o2", "o1"): xp.asarray(s[("o2", "o1")], dtype=complex) / ref21}

        mod = TimeVaryingTD(("o1", "o2"), fn, tuple(mapping.values()))
        return CompositeTD({"delay": static, "mod": mod}, {("delay", "o2"): ("mod", "o1")},
                           {"o1": ("delay", "o1"), "o2": ("mod", "o2")})
    if not isinstance(static, MemorylessTD):
        warnings.warn(
            f"{getattr(model, '__name__', model)!r} has memory (dispersion or delay) that the quasi-static "
            "time-varying element neglects",
            stacklevel=2,
        )
    return TimeVaryingTD.from_model(model, settings, control_params=control_params, carrier_wl=carrier_wl)
