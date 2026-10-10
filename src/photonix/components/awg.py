"""Arrayed-waveguide grating (AWG) demultiplexer from a phased-array model.

Physics
-------
An input star coupler illuminates ``M`` arrayed waveguides with a Gaussian
amplitude envelope ``a_m`` (``sum a_m^2 = 1``). Arm ``m`` is ``m * dL`` longer
than its neighbour, so at wavelength ``wl`` it carries the phase
``u_m * Phi(wl)`` with ``Phi = beta(wl) * dL`` (``u_m`` integer arm index about
the centre). The output star coupler focuses the array onto output waveguide
``k``, which accepts the tilted wavefront ``d_k = a * exp(-j u psi_k)``; its
amplitude is the overlap ``s_k = <d_k, c>`` with ``c = a * exp(-j u Phi)``.
Channel ``k`` therefore peaks where ``Phi = psi_k (mod 2 pi)``, and the passband
is the Fourier transform of the array illumination -- Gaussian, with sidelobes
and a crosstalk floor set by truncation to ``M`` arms.

The output modes ``d_k`` overlap slightly; they are made exactly orthonormal by
the symmetric (Löwdin) orthogonalization ``d' = d (D^H D)^{-1/2}``, the closest
orthonormal set. Bessel's inequality then gives ``sum_k |s_k|^2 <= |c|^2 <= 1``:
the model is passive for every wavelength by construction rather than by
approximation.

Design equations (all evaluated at the centre wavelength ``wl0``)::

    dL    = q * wl0 / neff              (q: grating order, chosen from the FSR)
    FSR   = wl0^2 / (ng * dL)
    sigma = FSR * sqrt(ln 2) / (pi * passband_fwhm)     (envelope width, arms)

The geometry (``dL``, output positions ``psi_k`` and the Gram matrix) is fixed
by these *nominal* parameters. Fabrication or thermal deviations of the arrayed
waveguides enter through ``delta_neff``, which shifts the response the way a
real device does instead of being designed away.
"""
from __future__ import annotations

import math

import numpy as np

from photonix.core.backend import xp
from photonix.core.types import SDict
from photonix.core.units import db_per_cm_to_alpha_um

__all__ = ["awg", "awg_design"]


def awg_design(
    *,
    n_channels: int = 8,
    channel_spacing: float = 0.0008,
    wl0: float = 1.55,
    fsr: float | None = None,
    passband_fwhm: float | None = None,
    n_arms: int | None = None,
    neff: float = 2.4,
    ng: float = 4.2,
) -> dict:
    """Derive AWG geometry from channel-plan specifications.

    Returns a dict with ``order``, ``delta_length`` (µm), ``fsr`` (µm),
    ``sigma_arms``, ``n_arms``, ``channel_wavelengths`` (µm) and the output
    phase tilts ``psi``. ``fsr`` defaults to ``n_channels * channel_spacing``
    (a cyclic AWG) and ``passband_fwhm`` to ``0.4 * channel_spacing``.

    Examples
    --------
    >>> d = awg_design(n_channels=8, channel_spacing=0.0008)
    >>> d["order"] > 0 and d["n_arms"] % 2 == 1
    True
    >>> abs(d["fsr"] - 8 * 0.0008) / (8 * 0.0008) < 0.02
    True
    """
    if not isinstance(n_channels, (int, np.integer)) or n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if channel_spacing <= 0:
        raise ValueError("channel_spacing must be positive")
    fsr_target = n_channels * channel_spacing if fsr is None else float(fsr)
    if fsr_target < n_channels * channel_spacing - 1e-15:
        raise ValueError("fsr must be at least n_channels * channel_spacing")
    fwhm = 0.4 * channel_spacing if passband_fwhm is None else float(passband_fwhm)
    if not 0 < fwhm < channel_spacing:
        raise ValueError("passband_fwhm must be positive and below channel_spacing")

    dl0 = wl0 ** 2 / (ng * fsr_target)
    order = max(1, round(neff * dl0 / wl0))
    dl = order * wl0 / neff
    fsr_actual = wl0 ** 2 / (ng * dl)
    sigma = fsr_actual * math.sqrt(math.log(2.0)) / (math.pi * fwhm)
    if n_arms is None:
        n_arms = 2 * math.ceil(3.0 * sigma) + 1
    if not isinstance(n_arms, (int, np.integer)) or n_arms < 3 or n_arms % 2 == 0:
        raise ValueError("n_arms must be an odd integer >= 3")

    k = np.arange(n_channels) - 0.5 * (n_channels - 1)
    wls = wl0 + k * channel_spacing
    n_k = neff - (ng - neff) * (wls - wl0) / wl0
    psi = 2.0 * math.pi * n_k / wls * dl
    psi = np.angle(np.exp(1j * psi))  # wrap; arms use integer indices
    return {
        "order": int(order),
        "delta_length": float(dl),
        "fsr": float(fsr_actual),
        "sigma_arms": float(sigma),
        "n_arms": int(n_arms),
        "channel_wavelengths": wls,
        "psi": psi,
    }


def _inverse_sqrt_hermitian(g: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(g)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.conj().T


def awg(
    *,
    wl=1.55,
    n_channels: int = 8,
    channel_spacing: float = 0.0008,
    wl0: float = 1.55,
    fsr: float | None = None,
    passband_fwhm: float | None = None,
    n_arms: int | None = None,
    neff: float = 2.4,
    ng: float = 4.2,
    delta_neff: float = 0.0,
    insertion_loss_db: float = 0.0,
    nonuniformity_db: float = 0.0,
    arm_length: float = 100.0,
    loss_db_cm: float = 0.0,
    phase_error_std: float = 0.0,
    seed: int | None = 0,
) -> SDict:
    """1xN arrayed-waveguide grating demultiplexer.

    Ports: input ``o1``; outputs ``o2 .. o(N+1)`` from the shortest to the
    longest channel wavelength ``wl0 + (k - (N-1)/2) * channel_spacing``.

    Parameters
    ----------
    n_channels, channel_spacing, wl0, fsr, passband_fwhm, n_arms, neff, ng
        Channel plan and nominal arrayed-waveguide index used to *design* the
        geometry; see :func:`awg_design`.
    delta_neff
        Deviation of the arrayed-waveguide effective index from ``neff``
        (fabrication, temperature). Differentiable; shifts all channels by
        about ``delta_neff * wl0 / ng``.
    insertion_loss_db
        Lumped on-peak loss of the two star couplers.
    nonuniformity_db
        Extra loss of the outermost channels relative to the central one,
        applied quadratically across the outputs (the far-field envelope of
        the array elements).
    arm_length, loss_db_cm
        Shortest arm length and propagation loss; longer arms lose more, which
        tilts the illumination.
    phase_error_std, seed
        Standard deviation (radians) of static random phase errors in the arms,
        drawn reproducibly from ``seed``. Phase errors are what limit real AWG
        crosstalk (typically -25 to -35 dB); without them the floor is set
        only by array truncation. They do not affect passivity.

    Examples
    --------
    >>> import numpy as np, photonix as px
    >>> d = px.components.awg_design(n_channels=4, channel_spacing=0.0016)
    >>> s = px.components.awg(wl=d["channel_wavelengths"][1], n_channels=4,
    ...                      channel_spacing=0.0016)
    >>> T = [float(px.power(s[("o1", f"o{k + 2}")])) for k in range(4)]
    >>> int(np.argmax(T)), T[1] > 0.99
    (1, True)
    """
    d = awg_design(n_channels=n_channels, channel_spacing=channel_spacing, wl0=wl0, fsr=fsr,
                   passband_fwhm=passband_fwhm, n_arms=n_arms, neff=neff, ng=ng)
    m_arms = d["n_arms"]
    u = np.arange(m_arms) - (m_arms - 1) // 2          # integer arm indices
    env = np.exp(-(u ** 2) / (4.0 * d["sigma_arms"] ** 2))
    env = env / np.sqrt(np.sum(env ** 2))              # sum a_m^2 = 1
    psi = d["psi"]
    # Output acceptance vectors d_k = a * exp(-j u psi_k) and their Gram matrix.
    dmat = env[:, None] * np.exp(-1j * np.outer(u, psi))      # (M, N)
    gram = dmat.conj().T @ dmat                               # (N, N), Hermitian
    w = _inverse_sqrt_hermitian(gram)                         # Löwdin correction

    wl = xp.asarray(wl)
    n = neff - (ng - neff) * (wl - wl0) / wl0 + delta_neff
    beta = 2.0 * xp.pi * n / wl
    dl = d["delta_length"]
    alpha = db_per_cm_to_alpha_um(loss_db_cm)
    # c_m = a_m exp(-alpha L_m) exp(-j u_m Phi), broadcast over the wavelength
    # batch; the common phase of the centre arm is applied once at the end.
    batch = (m_arms,) + (1,) * xp.ndim(beta)
    lengths = arm_length + (u - u[0]) * dl                    # L_m, m = 0 .. M-1
    arm_amp = xp.asarray(env).reshape(batch) * xp.exp(-alpha * xp.asarray(lengths).reshape(batch))
    errors = np.zeros(m_arms)
    if phase_error_std:
        errors = np.random.default_rng(seed).normal(0.0, float(phase_error_std), m_arms)
    arm_phase = xp.asarray(u, dtype=float).reshape(batch) * (beta * dl) + xp.asarray(errors).reshape(batch)
    c = arm_amp * xp.exp(-1j * arm_phase)
    s_raw = xp.tensordot(xp.asarray(dmat.conj().T), c, axes=1)  # (N, ...)
    s = xp.tensordot(xp.asarray(w), s_raw, axes=1)
    l_centre = arm_length + 0.5 * (m_arms - 1) * dl
    common = xp.exp(-1j * beta * l_centre) * 10.0 ** (-insertion_loss_db / 20.0)
    centre = 0.5 * (n_channels - 1)
    out: SDict = {}
    for k in range(n_channels):
        rel = ((k - centre) / centre) ** 2 if n_channels > 1 else 0.0
        uniform = 10.0 ** (-(nonuniformity_db * rel) / 20.0)
        val = xp.asarray(s[k] * common * uniform, dtype=complex)
        port = f"o{k + 2}"
        out[("o1", port)] = val
        out[(port, "o1")] = val
    return out
