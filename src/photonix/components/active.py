"""Electro-optic, thermo-optic and opto-electronic component models.

These are physics-based compact models with every material coefficient exposed
as a parameter. The defaults are textbook silicon values at 300 K, quoted with
their source; they are *not* a substitute for a foundry's measured modulator or
heater data.

Units: lengths in µm, doping in cm^-3, voltage in V, electrical power in mW,
temperature changes in K.
"""
from __future__ import annotations

from photonix.core.backend import xp
from photonix.core.constants import C0, EPS0, H_PLANCK, N_GROUP_SI_STRIP, Q_E, WL_DEFAULT
from photonix.core.types import SDict
from photonix.core.units import db_per_cm_to_alpha_um

__all__ = [
    "heater_phase_shifter", "thermo_optic_phase_shifter",
    "plasma_dispersion", "built_in_voltage", "depletion_width", "pn_phase_shifter",
    "photodetector", "responsivity", "photocurrent",
    "DN_DT_SI", "EPS_R_SI", "NI_SI_300K",
]

#: Thermo-optic coefficient of crystalline Si near 1.55 µm and 300 K [1/K]
#: (Komma et al., Appl. Phys. Lett. 101, 041905, 2012).
DN_DT_SI = 1.86e-4
#: Static relative permittivity of silicon.
EPS_R_SI = 11.7
#: Intrinsic carrier density of Si at 300 K [cm^-3] (Altermatt et al., J. Appl.
#: Phys. 93, 1598, 2003).
NI_SI_300K = 9.65e9
_K_B = 1.380649e-23  # Boltzmann constant [J/K]


def _neff_linear(wl, neff, ng, wl0):
    return neff - (ng - neff) * (wl - wl0) / wl0


def _two_port(t) -> SDict:
    t = xp.asarray(t, dtype=complex)
    return {("o1", "o2"): t, ("o2", "o1"): t}


def heater_phase_shifter(
    *,
    wl=WL_DEFAULT,
    length: float = 100.0,
    power: float = 0.0,
    p_pi: float = 25.0,
    neff: float = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    wl0: float = WL_DEFAULT,
    loss_db_cm: float = 0.0,
) -> SDict:
    """Thermo-optic phase shifter specified by its ``P_pi`` (mW). Ports o1/o2.

    The heater-induced phase is ``pi * power / p_pi * (wl0 / wl)``: the index
    change is proportional to dissipated power, and the phase of a fixed index
    change scales as ``1/wl``. The propagation phase of the ``length`` of
    waveguide is included, so the element is interchangeable with a
    :func:`~photonix.components.straight` arm in an interferometer.

    For a resistive heater, ``power = V**2 / R`` is in mW for ``V`` in volts and
    ``R`` in kilo-ohms.

    Examples
    --------
    >>> import cmath, math, photonix as px
    >>> s0 = px.components.heater_phase_shifter(power=0.0)
    >>> s1 = px.components.heater_phase_shifter(power=25.0, p_pi=25.0)
    >>> dphi = cmath.phase(complex(s0[("o1", "o2")]) / complex(s1[("o1", "o2")]))
    >>> abs(abs(dphi) - math.pi) < 1e-9
    True
    """
    wl = xp.asarray(wl)
    n = _neff_linear(wl, neff, ng, wl0)
    phi = 2.0 * xp.pi / wl * n * length + xp.pi * xp.asarray(power) / p_pi * (wl0 / wl)
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    return _two_port(xp.exp(-alpha * length) * xp.exp(-1j * phi))


def thermo_optic_phase_shifter(
    *,
    wl=WL_DEFAULT,
    length: float = 100.0,
    delta_t: float = 0.0,
    dneff_dt: float = DN_DT_SI,
    neff: float = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    wl0: float = WL_DEFAULT,
    loss_db_cm: float = 0.0,
) -> SDict:
    """Waveguide whose effective index shifts by ``dneff_dt * delta_t``. Ports o1/o2.

    ``dneff_dt`` is the *modal* thermo-optic coefficient. It defaults to the
    bulk silicon value :data:`DN_DT_SI`; the modal value of a high-confinement
    strip is close to it but should come from a mode solve or a measurement
    (silica cladding contributes only ~1e-5 /K). The length giving a pi shift
    at temperature rise ``delta_t`` is ``L_pi = wl / (2 * dneff_dt * delta_t)``.

    Examples
    --------
    >>> import cmath, math, photonix as px
    >>> L_pi = 1.55 / (2 * px.components.DN_DT_SI * 10.0)       # ~417 µm at dT = 10 K
    >>> a = px.components.thermo_optic_phase_shifter(length=L_pi, delta_t=0.0)
    >>> b = px.components.thermo_optic_phase_shifter(length=L_pi, delta_t=10.0)
    >>> abs(abs(cmath.phase(complex(a[("o1", "o2")]) / complex(b[("o1", "o2")]))) - math.pi) < 1e-9
    True
    """
    wl = xp.asarray(wl)
    n = _neff_linear(wl, neff, ng, wl0) + dneff_dt * xp.asarray(delta_t)
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    return _two_port(xp.exp(-alpha * length) * xp.exp(-1j * 2.0 * xp.pi / wl * n * length))


def plasma_dispersion(delta_ne, delta_nh, wl=WL_DEFAULT):
    """Free-carrier index and absorption changes in silicon (Soref & Bennett).

    Uses the 1.55 µm fits of Soref and Bennett (IEEE J. Quantum Electron. 23,
    123, 1987)::

        dn     = -(8.8e-22 * dNe + 8.5e-18 * dNh**0.8)
        dalpha =   8.5e-18 * dNe + 6.0e-18 * dNh          [1/cm]

    scaled by the Drude ``(wl / 1.55)**2`` dependence away from 1.55 µm.
    Carrier densities are in cm^-3; ``dalpha`` is a *power* absorption
    coefficient in 1/cm.

    Returns
    -------
    (dn, dalpha_per_cm)

    Examples
    --------
    >>> dn, da = plasma_dispersion(1e18, 1e18)
    >>> round(float(dn), 6), round(float(da), 2)
    (-0.003015, 14.5)
    """
    scale = (xp.asarray(wl) / 1.55) ** 2
    dne = xp.asarray(delta_ne)
    dnh = xp.asarray(delta_nh)
    dn = -(8.8e-22 * dne + 8.5e-18 * dnh ** 0.8) * scale
    dalpha = (8.5e-18 * dne + 6.0e-18 * dnh) * scale
    return dn, dalpha


def built_in_voltage(na: float, nd: float, temperature: float = 300.0, ni: float = NI_SI_300K):
    """Built-in potential ``(kT/q) ln(Na Nd / ni^2)`` of an abrupt p-n junction [V].

    Examples
    --------
    >>> round(float(built_in_voltage(5e17, 5e17)), 3)
    0.918
    """
    vt = _K_B * temperature / Q_E
    return vt * xp.log(xp.asarray(na) * xp.asarray(nd) / ni ** 2)


def depletion_width(voltage, na: float, nd: float, *, temperature: float = 300.0,
                    eps_r: float = EPS_R_SI, ni: float = NI_SI_300K):
    """Depletion widths ``(x_p, x_n)`` in µm of an abrupt junction at reverse bias.

    ``W = sqrt(2 eps (V_bi + V) / q * (Na + Nd) / (Na Nd))`` split as
    ``x_n = W Na / (Na + Nd)`` and ``x_p = W Nd / (Na + Nd)``. ``voltage`` is
    the reverse bias (positive depletes); forward bias is clamped below
    ``V_bi`` because carrier injection is outside this depletion-mode model.

    Examples
    --------
    >>> xp_, xn = depletion_width(0.0, 5e17, 5e17)
    >>> round(float(xp_ + xn) * 1e3, 1)     # nm
    68.9
    """
    eps = eps_r * EPS0 * 1e-2  # F/cm
    vbi = built_in_voltage(na, nd, temperature, ni)
    v_tot = xp.maximum(vbi + xp.asarray(voltage), 1e-3 * vbi)
    na = xp.asarray(na)
    nd = xp.asarray(nd)
    w_cm = xp.sqrt(2.0 * eps * v_tot / Q_E * (na + nd) / (na * nd))
    w_um = w_cm * 1e4
    return w_um * nd / (na + nd), w_um * na / (na + nd)


def pn_phase_shifter(
    *,
    wl=WL_DEFAULT,
    length: float = 2000.0,
    voltage: float = 0.0,
    na: float = 5e17,
    nd: float = 5e17,
    width: float = 0.5,
    overlap: float = 0.8,
    temperature: float = 300.0,
    neff: float = 2.4,
    ng: float = N_GROUP_SI_STRIP,
    wl0: float = WL_DEFAULT,
    loss_db_cm: float = 0.0,
) -> SDict:
    """Carrier-depletion phase shifter with a lateral p-n junction. Ports o1/o2.

    The junction sits at the centre of a core of ``width`` µm; the p and n
    halves are doped ``na`` and ``nd`` (cm^-3). At reverse bias ``voltage`` the
    depletion widths :func:`depletion_width` remove carriers, and the
    remaining free carriers set the index and loss through
    :func:`plasma_dispersion`. In the first-order, uniform-lateral-field
    approximation, with a fraction ``overlap`` of the modal power in the doped
    core::

        dn_eff  = overlap * [dn_h (w/2 - x_p) + dn_e (w/2 - x_n)] / w
        alpha   = overlap * [a_h  (w/2 - x_p) + a_e  (w/2 - x_n)] / w

    so the phase rises (index increases) and the loss falls as the depletion
    region widens. ``neff`` is the *undoped* waveguide index and
    ``loss_db_cm`` its intrinsic loss. The defaults give ``V_pi L`` of about
    2 V cm, typical of 5e17 cm^-3 lateral junctions.

    Notes
    -----
    A centred junction under the uniform-field assumption underestimates
    efficiency slightly (the real mode peaks at the junction). Forward-bias
    injection, velocity saturation and RC bandwidth are not modelled.

    Examples
    --------
    >>> import cmath, photonix as px
    >>> s0 = px.components.pn_phase_shifter(voltage=0.0)
    >>> s2 = px.components.pn_phase_shifter(voltage=2.0)
    >>> dphi = cmath.phase(complex(s0[("o1", "o2")]) / complex(s2[("o1", "o2")]))
    >>> bool(0 < dphi < 3.2)     # reverse bias retards the phase
    True
    >>> bool(px.power(s2[("o1", "o2")]) > px.power(s0[("o1", "o2")]))   # less loss
    True
    """
    wl = xp.asarray(wl)
    x_p, x_n = depletion_width(voltage, na, nd, temperature=temperature)
    half = 0.5 * width
    undepleted_p = half - xp.minimum(x_p, half)
    undepleted_n = half - xp.minimum(x_n, half)
    dn_e, a_e = plasma_dispersion(nd, 0.0, wl)
    dn_h, a_h = plasma_dispersion(0.0, na, wl)
    dn_eff = overlap * (dn_h * undepleted_p + dn_e * undepleted_n) / width
    alpha_power_cm = overlap * (a_h * undepleted_p + a_e * undepleted_n) / width
    alpha_fc = 0.5 * alpha_power_cm * 1e-4  # field attenuation [1/µm]
    n = _neff_linear(wl, neff, ng, wl0) + dn_eff
    alpha = db_per_cm_to_alpha_um(loss_db_cm) + alpha_fc
    return _two_port(xp.exp(-alpha * length) * xp.exp(-1j * 2.0 * xp.pi / wl * n * length))


def photodetector(*, wl=WL_DEFAULT, reflection: float = 0.0) -> SDict:
    """Optical termination of a photodetector (single port ``o1``).

    Absorbed power is converted to current outside the scattering model; use
    :func:`responsivity` and :func:`photocurrent` with the power delivered to
    this port.

    Examples
    --------
    >>> import photonix as px
    >>> complex(px.components.photodetector()[("o1", "o1")])
    0j
    """
    wl = xp.asarray(wl)
    r = xp.asarray(xp.asarray(reflection) * xp.ones_like(xp.real(wl)), dtype=complex)
    return {("o1", "o1"): r}


def responsivity(wl=WL_DEFAULT, quantum_efficiency: float = 0.8):
    """Photodiode responsivity ``eta * q * wl / (h c)`` in A/W (``wl`` in µm).

    Examples
    --------
    >>> round(float(responsivity(1.55, 1.0)), 4)
    1.2502
    """
    return quantum_efficiency * Q_E * xp.asarray(wl) * 1e-6 / (H_PLANCK * C0)


def photocurrent(optical_power_w, responsivity_a_w: float = 1.0, dark_current_a: float = 0.0):
    """Photocurrent ``R * P + I_dark`` in A for optical power in W.

    Examples
    --------
    >>> photocurrent(1e-3, 1.0, 1e-9)
    0.001000001
    """
    return responsivity_a_w * optical_power_w + dark_current_a

