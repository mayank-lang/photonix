"""General multimode, multiport FDTD orchestration for native layouts.

The planning layer in this module is Meep-free.  It expands every physical
layout port into one or more modal terminals (for example ``o1:m1``) and makes
the number and direction of the required simulations inspectable before any
expensive work starts.  The execution layer performs one narrow-band Meep run
for every incident modal terminal and wavelength, producing a complete square
:class:`~photonix.core.SParameterDataset`.

All coefficients use a port-local convention: direction 0 is the wave travelling
along the port's outward normal and direction 1 is the incident wave travelling
into the component.  Consequently ``dataset.s[:, out, in]`` is independent of
whether a port lies on the left, right, top, or bottom of the layout.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from photonix.core import SParameterDataset

from ._guard import meep_frequency, require_mpb
from .layout import PreparedLayout, PreparedPort, build_layout_simulation, port_region

__all__ = [
    "ModalTerminal",
    "PortModeSpec",
    "MultiportPlan",
    "modal_terminal",
    "plan_multiport",
    "simulate_multiport_sparameters",
]

_RESERVED_SIMULATION_KWARGS = {
    "boundary_layers",
    "cell_size",
    "default_material",
    "dimensions",
    "geometry",
    "resolution",
    "sources",
}


@dataclass(frozen=True)
class PortModeSpec:
    """Eigenmode settings shared by the selected bands of one physical port.

    ``bands`` are one-based MPB band numbers.  ``span`` is the transverse size
    of the source/monitor plane (the native port width by default).  Source and
    monitor offsets are measured inward from the native port reference plane.
    A nonzero monitor offset is useful for keeping the DFT plane away from the
    equivalent-current source.

    ``eig_parity`` is intentionally typed as ``Any``: it is normally a Meep
    parity constant, but keeping it opaque preserves import safety when Meep is
    not installed.  With ``None``, Meep's ``NO_PARITY`` is used.
    """

    bands: tuple[int, ...] = (1,)
    span: float | None = None
    z_span: float = 0.0
    source_offset: float = 0.0
    monitor_offset: float = 0.0
    eig_parity: Any | None = None
    eig_resolution: int = 0
    eig_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        bands = tuple(self.bands)
        if not bands or len(bands) != len(set(bands)):
            raise ValueError("bands must contain unique, positive MPB band numbers")
        if any(
            not isinstance(band, (int, np.integer))
            or isinstance(band, (bool, np.bool_))
            or band <= 0
            for band in bands
        ):
            raise ValueError("bands must contain unique, positive MPB band numbers")
        object.__setattr__(self, "bands", tuple(int(band) for band in bands))
        if self.span is not None and (not np.isfinite(self.span) or self.span <= 0):
            raise ValueError("span must be positive and finite")
        for name, value in (
            ("z_span", self.z_span),
            ("source_offset", self.source_offset),
            ("monitor_offset", self.monitor_offset),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if (
            not isinstance(self.eig_resolution, (int, np.integer))
            or isinstance(self.eig_resolution, (bool, np.bool_))
            or self.eig_resolution < 0
        ):
            raise ValueError("eig_resolution must be a non-negative integer")
        if not np.isfinite(self.eig_tolerance) or self.eig_tolerance <= 0:
            raise ValueError("eig_tolerance must be positive and finite")


@dataclass(frozen=True)
class ModalTerminal:
    """One circuit terminal: a physical layout port and one MPB band."""

    name: str
    port_name: str
    band: int
    port: PreparedPort
    spec: PortModeSpec


@dataclass(frozen=True)
class MultiportPlan:
    """Validated, Meep-free execution plan for a complete modal S matrix."""

    prepared: PreparedLayout
    wavelengths: np.ndarray
    terminals: tuple[ModalTerminal, ...]

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """Return the deterministic dataset port order."""
        return tuple(terminal.name for terminal in self.terminals)

    @property
    def run_count(self) -> int:
        """Number of device runs (excluding optional reference runs)."""
        return int(self.wavelengths.size * len(self.terminals))


def modal_terminal(port_name: str, band: int) -> str:
    """Return the stable terminal name for a one-based MPB band."""
    port_name = str(port_name)
    if not port_name or ":m" in port_name:
        raise ValueError("port names must be non-empty and must not contain ':m'")
    if (
        not isinstance(band, (int, np.integer))
        or isinstance(band, (bool, np.bool_))
        or band <= 0
    ):
        raise ValueError("band must be a positive integer")
    return f"{port_name}:m{int(band)}"


def _coerce_spec(value: PortModeSpec | Iterable[int]) -> PortModeSpec:
    return value if isinstance(value, PortModeSpec) else PortModeSpec(tuple(value))


def plan_multiport(
    prepared: PreparedLayout,
    wavelengths,
    *,
    port_modes: Mapping[str, PortModeSpec | Iterable[int]] | None = None,
) -> MultiportPlan:
    """Validate layout ports and expand them into modal circuit terminals.

    By default every native layout port contributes MPB band 1.  Supplying
    ``port_modes`` selects a subset of physical ports and/or multiple bands.
    The insertion order of that mapping defines the terminal order; otherwise
    the native layout port order is retained.
    """
    wl = np.asarray(wavelengths, dtype=float)
    if wl.ndim != 1 or wl.size == 0:
        raise ValueError("wavelengths must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0) or np.any(np.diff(wl) <= 0):
        raise ValueError("wavelengths must be positive, finite, and strictly increasing")
    if not prepared.ports:
        raise ValueError("the prepared layout has no ports")

    if port_modes is None:
        specs = {name: PortModeSpec() for name in prepared.ports}
    else:
        if not port_modes:
            raise ValueError("port_modes must select at least one physical port")
        unknown = set(port_modes) - set(prepared.ports)
        if unknown:
            raise KeyError(f"port_modes contains unknown layout ports: {sorted(unknown)!r}")
        specs = {str(name): _coerce_spec(value) for name, value in port_modes.items()}

    terminals: list[ModalTerminal] = []
    for port_name, spec in specs.items():
        port = prepared.ports[port_name]
        nx, ny = port.outward_normal
        if abs(nx * ny) > 1e-10:
            raise ValueError(
                f"Meep mode planes must be axis-aligned; port {port_name!r} "
                f"has orientation {port.orientation:g} degrees"
            )
        for band in spec.bands:
            terminals.append(
                ModalTerminal(modal_terminal(port_name, band), port_name, band, port, spec)
            )
    return MultiportPlan(prepared, wl.copy(), tuple(terminals))


def _run_excitation(
    plan: MultiportPlan,
    incident: ModalTerminal,
    wl: float,
    *,
    resolution: float,
    pml: float,
    fwidth_frac: float,
    dft_decay_tol: float,
    minimum_run_time: float,
    maximum_run_time: float | None,
    default_material,
    simulation_kwargs: Mapping[str, Any],
) -> dict[str, tuple[complex, complex]]:
    """Run one modal excitation; return terminal -> (outgoing, incoming)."""
    mp, _mpb = require_mpb()
    fcen = meep_frequency(wl)

    source_plane = port_region(
        incident.port,
        span=incident.spec.span,
        z_span=incident.spec.z_span,
        inward_offset=incident.spec.source_offset,
    )
    parity = mp.NO_PARITY if incident.spec.eig_parity is None else incident.spec.eig_parity
    source = mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=fwidth_frac * fcen),
        center=source_plane.region.center,
        size=source_plane.region.size,
        direction=mp.NO_DIRECTION,
        eig_kpoint=source_plane.inward_kpoint,
        eig_band=incident.band,
        eig_match_freq=True,
        eig_parity=parity,
        eig_resolution=incident.spec.eig_resolution,
        eig_tolerance=incident.spec.eig_tolerance,
    )
    sim, _layout = build_layout_simulation(
        plan.prepared,
        resolution=resolution,
        sources=[source],
        pml=pml,
        default_material=default_material,
        **dict(simulation_kwargs),
    )

    # One DFT plane per physical port is sufficient for all requested bands.
    physical_specs: dict[str, PortModeSpec] = {}
    for terminal in plan.terminals:
        physical_specs[terminal.port_name] = terminal.spec
    monitor_regions = {}
    monitors = {}
    for port_name, spec in physical_specs.items():
        region = port_region(
            plan.prepared.ports[port_name],
            span=spec.span,
            z_span=spec.z_span,
            inward_offset=spec.monitor_offset,
        )
        monitor_regions[port_name] = region
        monitors[port_name] = sim.add_mode_monitor(fcen, 0, 1, region.region)

    stop_kwargs: dict[str, float] = {
        "tol": dft_decay_tol,
        "minimum_run_time": minimum_run_time,
    }
    if maximum_run_time is not None:
        stop_kwargs["maximum_run_time"] = maximum_run_time
    sim.run(until_after_sources=mp.stop_when_dft_decayed(**stop_kwargs))

    result: dict[str, tuple[complex, complex]] = {}
    for port_name, spec in physical_specs.items():
        region = monitor_regions[port_name]

        def outward_guess(_frequency, _band, *, _k=region.outward_kpoint):
            return _k

        port_parity = mp.NO_PARITY if spec.eig_parity is None else spec.eig_parity
        coeffs = sim.get_eigenmode_coefficients(
            monitors[port_name],
            list(spec.bands),
            eig_parity=port_parity,
            eig_resolution=spec.eig_resolution,
            eig_tolerance=spec.eig_tolerance,
            kpoint_func=outward_guess,
            direction=mp.NO_DIRECTION,
        )
        alpha = np.asarray(coeffs.alpha)
        expected = (len(spec.bands), 1, 2)
        if alpha.shape != expected:
            raise RuntimeError(
                f"Meep returned coefficient shape {alpha.shape} for port {port_name!r}; "
                f"expected {expected}"
            )
        for index, band in enumerate(spec.bands):
            result[modal_terminal(port_name, band)] = (
                complex(alpha[index, 0, 0]),
                complex(alpha[index, 0, 1]),
            )
    return result


def _validate_execution(
    resolution: float,
    pml: float,
    fwidth_frac: float,
    dft_decay_tol: float,
    minimum_run_time: float,
    maximum_run_time: float | None,
) -> None:
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite")
    if not np.isfinite(pml) or pml < 0:
        raise ValueError("pml must be non-negative and finite")
    if not np.isfinite(fwidth_frac) or fwidth_frac <= 0:
        raise ValueError("fwidth_frac must be positive and finite")
    if not np.isfinite(dft_decay_tol) or not 0 < dft_decay_tol < 1:
        raise ValueError("dft_decay_tol must lie strictly between zero and one")
    if not np.isfinite(minimum_run_time) or minimum_run_time < 0:
        raise ValueError("minimum_run_time must be non-negative and finite")
    if maximum_run_time is not None and (
        not np.isfinite(maximum_run_time) or maximum_run_time <= minimum_run_time
    ):
        raise ValueError("maximum_run_time must be finite and greater than minimum_run_time")


def simulate_multiport_sparameters(
    prepared: PreparedLayout,
    *,
    wavelengths,
    resolution: float,
    port_modes: Mapping[str, PortModeSpec | Iterable[int]] | None = None,
    reference: PreparedLayout | None = None,
    pml: float = 1.0,
    fwidth_frac: float = 0.1,
    dft_decay_tol: float = 1e-8,
    minimum_run_time: float = 0.0,
    maximum_run_time: float | None = None,
    default_material=None,
    simulation_kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SParameterDataset:
    """Compute a complete multimode/multiport S matrix with Meep/MPB.

    A separate narrow-band run is made for every wavelength and incident modal
    terminal.  This is deliberately conservative near modal cutoff and for
    dispersive ports.  If ``reference`` is supplied, it must expose the same
    physical ports; a matching run provides the incident denominator and the
    residual source-port reflection to subtract.  A straight reference with the
    same port cross-sections, resolution, cell margins, and monitor offsets is
    recommended for quantitative work.

    Without a reference, the incoming coefficient measured in the device run is
    used for normalization.  This is convenient, but can be biased when a strong
    reflection reaches a monitor colocated with the source.
    """
    _validate_execution(
        resolution,
        pml,
        fwidth_frac,
        dft_decay_tol,
        minimum_run_time,
        maximum_run_time,
    )
    plan = plan_multiport(prepared, wavelengths, port_modes=port_modes)
    ref_plan = None
    if reference is not None:
        missing = set(terminal.port_name for terminal in plan.terminals) - set(reference.ports)
        if missing:
            raise KeyError(f"reference layout is missing physical ports: {sorted(missing)!r}")
        ref_plan = plan_multiport(reference, plan.wavelengths, port_modes=port_modes)
        if ref_plan.terminal_names != plan.terminal_names:
            raise ValueError("reference layout modal terminals do not match the device")

    kwargs = dict(simulation_kwargs or {})
    reserved = _RESERVED_SIMULATION_KWARGS.intersection(kwargs)
    if reserved:
        raise ValueError(
            "simulation_kwargs cannot override orchestrator-owned settings: "
            + ", ".join(sorted(reserved))
        )
    nterm = len(plan.terminals)
    matrix = np.zeros((plan.wavelengths.size, nterm, nterm), dtype=complex)
    for wi, wl in enumerate(plan.wavelengths):
        for incoming_index, incident in enumerate(plan.terminals):
            device_coeffs = _run_excitation(
                plan,
                incident,
                float(wl),
                resolution=resolution,
                pml=pml,
                fwidth_frac=fwidth_frac,
                dft_decay_tol=dft_decay_tol,
                minimum_run_time=minimum_run_time,
                maximum_run_time=maximum_run_time,
                default_material=default_material,
                simulation_kwargs=kwargs,
            )
            if ref_plan is None:
                incident_amplitude = device_coeffs[incident.name][1]
                reference_coeffs = None
            else:
                ref_incident = ref_plan.terminals[incoming_index]
                reference_coeffs = _run_excitation(
                    ref_plan,
                    ref_incident,
                    float(wl),
                    resolution=resolution,
                    pml=pml,
                    fwidth_frac=fwidth_frac,
                    dft_decay_tol=dft_decay_tol,
                    minimum_run_time=minimum_run_time,
                    maximum_run_time=maximum_run_time,
                    default_material=default_material,
                    simulation_kwargs=kwargs,
                )
                incident_amplitude = reference_coeffs[incident.name][1]
            if abs(incident_amplitude) <= np.finfo(float).tiny:
                raise RuntimeError(
                    f"Meep returned zero incident amplitude for {incident.name!r} at {wl:g} um"
                )
            for outgoing_index, terminal in enumerate(plan.terminals):
                outgoing = device_coeffs[terminal.name][0]
                # A reference can contain residual launch reflection in every
                # monitored band at the incident physical port.  Subtract that
                # whole backward modal vector, not only its diagonal element.
                if reference_coeffs is not None and terminal.port_name == incident.port_name:
                    outgoing -= reference_coeffs[terminal.name][0]
                matrix[wi, outgoing_index, incoming_index] = outgoing / incident_amplitude

    provenance: dict[str, Any] = {
        "solver": "meep-fdtd",
        "orchestration": "narrowband-full-modal-matrix",
        "normalization": "reference" if reference is not None else "device-incident",
        "device_runs": plan.run_count,
        "reference_runs": 0 if ref_plan is None else ref_plan.run_count,
        "modal_ports": [
            {"terminal": terminal.name, "physical_port": terminal.port_name, "mpb_band": terminal.band}
            for terminal in plan.terminals
        ],
    }
    provenance.update(dict(metadata or {}))
    return SParameterDataset(plan.wavelengths, plan.terminal_names, matrix, provenance)
