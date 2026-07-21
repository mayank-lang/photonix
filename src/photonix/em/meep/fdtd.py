"""FDTD via Meep, returned as native photonix S-parameters.

This is the FDTD backend photonix delegates to instead of shipping its own
time-domain Maxwell solver. :func:`waveguide_sparams` mirrors the signature of the
in-house frequency-domain :func:`photonix.em.fdfd.waveguide_sparams` -- same
``eps`` grid, same ``src_col`` / ``in_mon_col`` / ``out_mon_col`` column indexing,
same ``{("o1","o2"): ...}`` :data:`~photonix.core.types.SDict` out -- so a caller
swaps frequency-domain for time-domain by changing the import alone.

PML convention
--------------
Like the FDFD routine, the supplied ``eps`` grid is treated as the **physical
device**: PML is added as *extra* padding *outside* the grid (the edge waveguide
cross-sections are replicated into the padding so the guide continues, unbroken,
through the absorber). Source/monitor column indices therefore always address the
physical region and never land inside the PML. This is the difference between a
straight guide reading transmission ~ 1 and reading ~ 0.

Method: a 2-D Meep simulation with an :class:`meep.EigenModeSource` injecting the
input waveguide mode, mode monitors at the input and output planes, and
``get_eigenmode_coefficients`` to recover the forward/backward modal amplitudes.
Meep's coefficients are power-normalised, so ``|S21|**2`` is the power
transmission directly (no ``sqrt(beta)`` rescaling, unlike the FDFD routine).
"""
from __future__ import annotations

import numpy as np

from photonix.core.types import SDict

from ._guard import meep_frequency, require_meep
from .geometry import DeviceGrid, build_block

__all__ = ["waveguide_sparams", "build_simulation", "parity_for", "pad_for_pml"]


def parity_for(mp, polarization: str):
    """Map a photonix polarization label to a Meep 2-D eigenmode parity.

    photonix's 2-D in-plane solvers label polarizations by the scalar field
    family (:mod:`photonix.em.fdfd` / :mod:`photonix.em.eme`):

    * ``"te"`` -- out-of-plane ``Ez``, continuous scalar Helmholtz -> ``ODD_Z``;
    * ``"tm"`` -- out-of-plane ``Hz`` (in-plane E) -> ``EVEN_Z``.

    This function follows *photonix's* labels so that
    :func:`waveguide_sparams` is a drop-in replacement for
    :func:`photonix.em.fdfd.waveguide_sparams` at equal ``polarization``.

    .. warning::
       Meep's own TE/TM naming is the **opposite** (Meep defines ``TE = EVEN_Z``,
       the ``Hz`` family, per the photonic-crystal convention). Pass the explicit
       field-family labels ``"ez"`` / ``"hz"`` to avoid any ambiguity.
    """
    p = polarization.lower()
    if p in ("te", "ez"):
        return mp.ODD_Z   # out-of-plane E_z scalar family (photonix "te")
    if p in ("tm", "hz"):
        return mp.EVEN_Z  # out-of-plane H_z family, in-plane E (photonix "tm")
    raise ValueError(f"polarization must be 'te', 'tm', 'ez' or 'hz', got {polarization!r}")


def pad_for_pml(eps: np.ndarray, dx: float, dy: float, dpml: float):
    """Replicate-pad ``eps`` by ``dpml`` on every side; return ``(eps_p, npx, npy)``.

    The padding columns/rows carry the edge cross-section (``mode="edge"``), so an
    input/output waveguide continues straight through the PML instead of meeting a
    spurious material interface. ``npx`` / ``npy`` are the pad widths in cells, used
    to offset the caller's column indices into the padded grid.
    """
    eps = np.asarray(eps, float)
    npx = max(int(round(dpml / dx)), 1)
    npy = max(int(round(dpml / dy)), 1)
    eps_p = np.pad(eps, ((npy, npy), (npx, npx)), mode="edge")
    return eps_p, npx, npy


def build_simulation(
    device: DeviceGrid,
    *,
    wl: float,
    src_col: int,
    polarization: str = "te",
    mode: int = 1,
    dpml: float = 1.0,
    fwidth_frac: float = 0.1,
    eig_parity=None,
    src_size_y: float | None = None,
):
    """Assemble a 2-D Meep ``Simulation`` with an eigenmode source.

    ``device`` is expected to be **already padded** for PML (see
    :func:`pad_for_pml`); ``src_col`` indexes the padded grid. ``src_size_y`` sets
    the transverse extent of the eigenmode source/monitor plane (default: full
    device height). Returns ``(sim, info)``. Requires Meep.
    """
    mp = require_meep()
    fcen = meep_frequency(wl)
    cell, geometry = build_block(device)
    sy = device.size[1] if src_size_y is None else float(src_size_y)
    parity = eig_parity if eig_parity is not None else parity_for(mp, polarization)

    source = mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=fwidth_frac * fcen),
        center=mp.Vector3(device.x_of_col(src_col), 0),
        size=mp.Vector3(0, sy, 0),
        eig_band=mode,
        eig_parity=parity,
        eig_match_freq=True,
        direction=mp.X,
    )
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[source],
        boundary_layers=[mp.PML(dpml)],
        resolution=int(round(device.resolution)),
        force_complex_fields=False,
    )
    info = {"fcen": fcen, "parity": parity, "mode": mode, "sy": sy, "mp": mp}
    return sim, info


def waveguide_sparams(
    eps,
    *,
    dx: float,
    dy: float,
    wl: float,
    src_col: int,
    in_mon_col: int,
    out_mon_col: int,
    polarization: str = "te",
    mode: int = 1,
    dpml: float = 1.0,
    decay_by: float = 1e-4,
    fwidth_frac: float = 0.1,
    eig_parity=None,
    pad_pml: bool = True,
) -> SDict:
    """2-port S-parameters of a planar device via Meep FDTD.

    Parameters mirror :func:`photonix.em.fdfd.waveguide_sparams`: ``eps`` is an
    ``(ny, nx)`` permittivity grid with propagation along ``x`` (columns) and the
    transverse axis ``y`` (rows); ``src_col`` / ``in_mon_col`` / ``out_mon_col`` are
    column indices **into that physical grid** (PML is added outside it; see the
    module docstring). Returns an :data:`~photonix.core.types.SDict` with ports
    ``o1`` (input) and ``o2`` (output); ``|S[("o1","o2")]|**2`` is the power
    transmission. ``polarization`` follows photonix's field-family labels (see
    :func:`parity_for`; pass ``"ez"``/``"hz"`` to be explicit).

    Limitation: only the input-side reflection ``("o1","o1")`` is computed; a
    right-side-incident S22 would require a second FDTD run. Call again with the
    mirrored structure if S22 matters (the frequency-domain
    :func:`photonix.em.fdfd.waveguide_sparams` returns S22 directly).

    Requires Meep; raises :class:`ImportError` otherwise.
    """
    eps = np.asarray(eps, float)
    ny, nx = eps.shape
    if pad_pml:
        eps_p, npx, _npy = pad_for_pml(eps, dx, dy, dpml)
    else:
        eps_p, npx, _npy = eps, 0, 0
    device = DeviceGrid(eps_p, float(dx), float(dy))
    src_size_y = ny * float(dy)  # physical (un-padded) transverse window, centred

    sim, info = build_simulation(
        device, wl=wl, src_col=src_col + npx, polarization=polarization, mode=mode,
        dpml=dpml, fwidth_frac=fwidth_frac, eig_parity=eig_parity,
        src_size_y=src_size_y,
    )
    mp = info["mp"]
    fcen, parity = info["fcen"], info["parity"]

    def mode_region(col):
        return mp.ModeRegion(
            center=mp.Vector3(device.x_of_col(col + npx), 0),
            size=mp.Vector3(0, src_size_y, 0),
        )

    in_mon = sim.add_mode_monitor(fcen, 0, 1, mode_region(in_mon_col))
    out_mon = sim.add_mode_monitor(fcen, 0, 1, mode_region(out_mon_col))

    decay_pt = mp.Vector3(device.x_of_col(out_mon_col + npx), 0)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez if parity == mp.ODD_Z else mp.Hz, decay_pt, decay_by))

    bands = [mode]
    a_in = sim.get_eigenmode_coefficients(in_mon, bands, eig_parity=parity)
    a_out = sim.get_eigenmode_coefficients(out_mon, bands, eig_parity=parity)
    # alpha[band, freq, direction]: 0 = +x (forward), 1 = -x (backward)
    fwd_in = a_in.alpha[0, 0, 0]
    bwd_in = a_in.alpha[0, 0, 1]
    fwd_out = a_out.alpha[0, 0, 0]

    s21 = complex(fwd_out / fwd_in)
    s11 = complex(bwd_in / fwd_in)
    return {
        ("o1", "o2"): s21,
        ("o2", "o1"): s21,
        ("o1", "o1"): s11,
    }
