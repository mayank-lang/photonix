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

from ._guard import meep_frequency, require_mpb
from .geometry import DeviceGrid, build_block, build_pixel_block

__all__ = [
    "waveguide_sparams",
    "waveguide_spectrum",
    "waveguide_dataset",
    "build_simulation",
    "parity_for",
    "pad_for_pml",
]


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
    if eps.ndim != 2 or 0 in eps.shape or not np.all(np.isfinite(eps)) or np.any(eps <= 0):
        raise ValueError("eps must be a non-empty 2-D array of finite, positive permittivities")
    dx, dy, dpml = float(dx), float(dy), float(dpml)
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be positive and finite")
    if not np.isfinite(dpml) or dpml < 0:
        raise ValueError("dpml must be non-negative and finite")
    npx = int(np.ceil(dpml / dx))
    npy = int(np.ceil(dpml / dy))
    if npx == 0 and npy == 0:
        return eps.copy(), 0, 0
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
    grid_kind: str = "cell",
):
    """Assemble a 2-D Meep ``Simulation`` with an eigenmode source.

    ``device`` is expected to be **already padded** for PML (see
    :func:`pad_for_pml`); ``src_col`` indexes the padded grid. ``src_size_y`` sets
    the transverse extent of the eigenmode source/monitor plane (default: full
    device height). Returns ``(sim, info)``. Requires Meep.
    """
    mp, _mpb = require_mpb()  # EigenModeSource and mode monitors require MPB.
    fcen = meep_frequency(wl)
    if (not isinstance(src_col, (int, np.integer)) or isinstance(src_col, (bool, np.bool_))
            or not 0 <= src_col < device.shape[1]):
        raise ValueError(f"src_col must be an integer in [0, {device.shape[1]})")
    if not isinstance(mode, (int, np.integer)) or isinstance(mode, (bool, np.bool_)) or mode <= 0:
        raise ValueError("mode must be a positive integer")
    if not np.isfinite(dpml) or dpml < 0:
        raise ValueError("dpml must be non-negative and finite")
    if not np.isfinite(fwidth_frac) or fwidth_frac <= 0:
        raise ValueError("fwidth_frac must be positive and finite")
    if grid_kind == "cell":
        cell, geometry = build_pixel_block(device)
    elif grid_kind == "density":
        cell, geometry = build_block(device)
    else:
        raise ValueError("grid_kind must be 'cell' or 'density'")
    sy = device.size[1] if src_size_y is None else float(src_size_y)
    if not np.isfinite(sy) or sy <= 0 or sy > device.size[1] * (1 + 1e-12):
        raise ValueError("src_size_y must be positive and no larger than the device height")
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
        boundary_layers=[mp.PML(dpml)] if dpml > 0 else [],
        resolution=int(round(device.resolution)),
        force_complex_fields=False,
    )
    decay_component = mp.Ez if polarization.lower() in ("te", "ez") else mp.Hz
    info = {
        "fcen": fcen,
        "parity": parity,
        "mode": mode,
        "sy": sy,
        "mp": mp,
        "decay_component": decay_component,
    }
    return sim, info


def _validate_columns(nx: int, src_col: int, in_mon_col: int, out_mon_col: int) -> None:
    values = (src_col, in_mon_col, out_mon_col)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError("source and monitor columns must be integers")
    if not (0 <= src_col < in_mon_col < out_mon_col < nx):
        raise ValueError(
            "left-incidence columns must satisfy "
            f"0 <= src_col < in_mon_col < out_mon_col < nx; got {values} for nx={nx}"
        )


def _one_way_sparams(
    eps: np.ndarray,
    *,
    dx: float,
    dy: float,
    wl: float,
    src_col: int,
    in_mon_col: int,
    out_mon_col: int,
    polarization: str,
    mode: int,
    dpml: float,
    decay_by: float,
    fwidth_frac: float,
    eig_parity,
    pad_pml: bool,
    normalization_eps: np.ndarray | None,
) -> tuple[complex, complex]:
    """Return ``(reflection, transmission)`` for left incidence."""
    ny, nx = eps.shape
    _validate_columns(nx, src_col, in_mon_col, out_mon_col)

    def run(grid: np.ndarray):
        if pad_pml:
            eps_p, npx, _npy = pad_for_pml(grid, dx, dy, dpml)
        else:
            eps_p, npx = grid, 0
        device = DeviceGrid(eps_p, dx, dy)
        src_size_y = ny * dy
        sim, info = build_simulation(
            device,
            wl=wl,
            src_col=src_col + npx,
            polarization=polarization,
            mode=mode,
            dpml=dpml,
            fwidth_frac=fwidth_frac,
            eig_parity=eig_parity,
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
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                50,
                info["decay_component"],
                decay_pt,
                decay_by,
            )
        )
        bands = [mode]
        a_in = sim.get_eigenmode_coefficients(in_mon, bands, eig_parity=parity)
        a_out = sim.get_eigenmode_coefficients(out_mon, bands, eig_parity=parity)
        # alpha[band, frequency, direction]: 0 = +x, 1 = -x.
        return a_in.alpha[0, 0, 0], a_in.alpha[0, 0, 1], a_out.alpha[0, 0, 0]

    fwd_in, bwd_in, fwd_out = run(eps)
    if normalization_eps is None:
        incident = fwd_in
        reflected_background = 0.0
    else:
        incident, reflected_background, _reference_out = run(normalization_eps)
    if abs(incident) <= np.finfo(float).tiny:
        raise RuntimeError("Meep returned zero incident-mode amplitude; check the source mode and port geometry")
    return complex((bwd_in - reflected_background) / incident), complex(fwd_out / incident)


def waveguide_sparams(
    eps,
    *,
    dx: float,
    dy: float,
    wl: float | np.ndarray,
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
    bidirectional: bool = True,
    right_src_col: int | None = None,
    right_in_mon_col: int | None = None,
    right_out_mon_col: int | None = None,
    normalization_eps: np.ndarray | None = None,
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

    By default a second, right-incident run computes the complete two-port matrix;
    no reciprocity or mirror-symmetry assumption is made. Set ``bidirectional=False``
    to perform only the left-incident run. An array-valued ``wl`` returns arrays of
    identical shape in the SDict (one narrow-band run per wavelength, which is more
    robust than interpreting a single very-wide pulse across strongly dispersive
    port modes).

    ``normalization_eps`` may supply a same-shaped straight-through reference. Its
    incident modal amplitude is used as the denominator, preserving the Meep phase
    reference and reducing source-mismatch bias. Without it, the forward amplitude
    measured in the device run is used; this is convenient but less accurate for
    strong reflections or multimode launches.

    Requires Meep; raises :class:`ImportError` otherwise.
    """
    wavelengths = np.asarray(wl, dtype=float)
    if wavelengths.ndim:
        if wavelengths.size == 0:
            raise ValueError("wl must not be empty")
        samples = [
            waveguide_sparams(
                eps,
                dx=dx,
                dy=dy,
                wl=float(w),
                src_col=src_col,
                in_mon_col=in_mon_col,
                out_mon_col=out_mon_col,
                polarization=polarization,
                mode=mode,
                dpml=dpml,
                decay_by=decay_by,
                fwidth_frac=fwidth_frac,
                eig_parity=eig_parity,
                pad_pml=pad_pml,
                bidirectional=bidirectional,
                right_src_col=right_src_col,
                right_in_mon_col=right_in_mon_col,
                right_out_mon_col=right_out_mon_col,
                normalization_eps=normalization_eps,
            )
            for w in wavelengths.reshape(-1)
        ]
        keys = samples[0].keys()
        return {key: np.asarray([sample[key] for sample in samples]).reshape(wavelengths.shape) for key in keys}

    wl_scalar = float(wavelengths)
    if not np.isfinite(wl_scalar) or wl_scalar <= 0:
        raise ValueError("wl must be positive and finite")
    eps = np.asarray(eps, float)
    if eps.ndim != 2 or 0 in eps.shape or not np.all(np.isfinite(eps)) or np.any(eps <= 0):
        raise ValueError("eps must be a non-empty 2-D array of finite, positive permittivities")
    ny, nx = eps.shape
    dx, dy = float(dx), float(dy)
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError("dx and dy must be positive and finite")
    if not np.isfinite(decay_by) or not 0 < decay_by < 1:
        raise ValueError("decay_by must lie strictly between zero and one")
    norm = None if normalization_eps is None else np.asarray(normalization_eps, dtype=float)
    if norm is not None and (
        norm.shape != eps.shape or not np.all(np.isfinite(norm)) or np.any(norm <= 0)
    ):
        raise ValueError("normalization_eps must have the same shape as eps and be finite and positive")

    s11, s21 = _one_way_sparams(
        eps,
        dx=dx,
        dy=dy,
        wl=wl_scalar,
        src_col=src_col,
        in_mon_col=in_mon_col,
        out_mon_col=out_mon_col,
        polarization=polarization,
        mode=mode,
        dpml=dpml,
        decay_by=decay_by,
        fwidth_frac=fwidth_frac,
        eig_parity=eig_parity,
        pad_pml=pad_pml,
        normalization_eps=norm,
    )
    result: SDict = {
        ("o1", "o2"): s21,
        ("o1", "o1"): s11,
    }
    if not bidirectional:
        return result

    rsrc = nx - 1 - src_col if right_src_col is None else right_src_col
    rin = nx - 1 - in_mon_col if right_in_mon_col is None else right_in_mon_col
    rout = nx - 1 - out_mon_col if right_out_mon_col is None else right_out_mon_col
    flipped_cols = (nx - 1 - rsrc, nx - 1 - rin, nx - 1 - rout)
    s22, s12 = _one_way_sparams(
        eps[:, ::-1],
        dx=dx,
        dy=dy,
        wl=wl_scalar,
        src_col=flipped_cols[0],
        in_mon_col=flipped_cols[1],
        out_mon_col=flipped_cols[2],
        polarization=polarization,
        mode=mode,
        dpml=dpml,
        decay_by=decay_by,
        fwidth_frac=fwidth_frac,
        eig_parity=eig_parity,
        pad_pml=pad_pml,
        normalization_eps=None if norm is None else norm[:, ::-1],
    )
    result[("o2", "o1")] = s12
    result[("o2", "o2")] = s22
    return result


def waveguide_spectrum(eps, *, wavelengths, **kwargs) -> SDict:
    """Broadband convenience wrapper around :func:`waveguide_sparams`.

    ``wavelengths`` may have any non-empty shape; each returned coefficient has
    that same shape. Separate narrow-band runs keep each eigenmode source and
    monitor locked to its requested frequency, which is conservative for
    dispersive or cutoff-adjacent port modes.
    """
    return waveguide_sparams(eps, wl=np.asarray(wavelengths, dtype=float), **kwargs)


def waveguide_dataset(eps, *, wavelengths, metadata=None, **kwargs):
    """Return a versioned :class:`~photonix.core.SParameterDataset` from Meep."""
    from photonix.core import SParameterDataset

    wavelengths = np.asarray(wavelengths, dtype=float)
    sdict = waveguide_spectrum(eps, wavelengths=wavelengths, **kwargs)
    provenance = {"solver": "meep-fdtd", "polarization": kwargs.get("polarization", "te")}
    provenance.update(dict(metadata or {}))
    return SParameterDataset.from_sdict(wavelengths, sdict, ports=("o1", "o2"), metadata=provenance)
