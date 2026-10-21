"""photonix component model library.

Physics-based, differentiable, parametric models. Every model is a pure function
``f(*, wl=1.55, **params) -> SDict`` returning a scattering dictionary, so it is
``jit``-able and ``grad``-able and composes directly in :mod:`photonix.circuit`.

Families
--------
* waveguides -- :func:`straight`, :func:`bend`, :func:`bend_euler`,
  :func:`taper`, :func:`waveguide_route`;
* couplers and splitters -- ideal :func:`directional_coupler`, physical
  :func:`evanescent_coupler`, :func:`mmi1x2`, :func:`mmi2x2`, :func:`mmi_nxn`,
  :func:`y_branch`, :func:`crossing`;
* resonators and interferometers -- :func:`all_pass_ring`,
  :func:`add_drop_ring`, geometry-driven :func:`ring_single` and
  :func:`ring_double`, :func:`mzi`;
* filters -- :func:`bragg_grating`, :func:`awg`;
* active -- :func:`phase_shifter`, :func:`heater_phase_shifter`,
  :func:`thermo_optic_phase_shifter`, :func:`pn_phase_shifter`,
  :func:`photodetector`;
* I/O -- :func:`grating_coupler`, :func:`edge_coupler`;
* lumped -- :func:`attenuator`, :func:`terminator`;
* polarization and multimode (modal terminals ``"o1:TE0"``) --
  :func:`polarization_rotator`, :func:`polarization_splitter_rotator`,
  :func:`polarization_beam_splitter`, :func:`grating_coupler_2d`,
  :func:`jones_fiber`, :func:`multimode_straight`,
  :func:`asymmetric_mode_converter`, :func:`mode_mux`.
"""
from __future__ import annotations

from .active import (
    DN_DT_SI,
    EPS_R_SI,
    NI_SI_300K,
    built_in_voltage,
    depletion_width,
    heater_phase_shifter,
    intrinsic_density,
    photocurrent,
    photodetector,
    plasma_dispersion,
    pn_phase_shifter,
    responsivity,
    silicon_bandgap,
    thermo_optic_phase_shifter,
)
from .awg import awg, awg_design
from .bragg import bragg_grating, bragg_wavelength, grating_kappa
from .couplers import (
    coupler,
    crossing,
    directional_coupler,
    evanescent_coupler,
    mmi1x2,
    mmi2x2,
    mmi_nxn,
    supermode_splitting,
    y_branch,
)
from .fiber import SMF28_MFD_1550, edge_coupler, gaussian_overlap
from .gratings import attenuator, grating_coupler, phase_shifter, terminator
from .multimode import asymmetric_mode_converter, coupled_mode_transfer, mode_mux, multimode_straight
from .mzi import mzi
from .polarization import (
    grating_coupler_2d,
    jones_fiber,
    pmd_fiber_jones,
    polarization_beam_splitter,
    polarization_rotator,
    polarization_splitter_rotator,
)
from .resonators import add_drop_ring, all_pass_ring, ring_coupler, ring_double, ring_single
from .waveguide import (
    bend,
    bend_euler,
    bend_from_solver,
    euler_bend_length,
    neff_linear,
    straight,
    taper,
    waveguide_route,
)

#: Parameter names that mean "property of the guiding waveguide" (and nothing
#: else) in a model. :meth:`photonix.pdk.Pdk.models` binds a cross-section's
#: settings only into parameters a model declares here, so a PDK's ``wl0``
#: (dispersion reference) never overrides, say, a grating coupler's peak
#: wavelength or an AWG's channel-plan centre, which share the name.
WAVEGUIDE_PARAMETERS = ("neff", "ng", "loss_db_cm", "wl0")
for _fn in (straight, bend, bend_euler, taper, waveguide_route, evanescent_coupler,
            all_pass_ring, add_drop_ring, ring_single, ring_double, mzi,
            heater_phase_shifter, thermo_optic_phase_shifter, pn_phase_shifter, bragg_grating):
    _fn.waveguide_parameters = WAVEGUIDE_PARAMETERS  # type: ignore[union-attr]
awg.waveguide_parameters = ("neff", "ng", "loss_db_cm")  # type: ignore[attr-defined]  # wl0 = channel centre
del _fn

#: Registry of model name -> callable, used by PDKs and the circuit solver.
#: Layout cells extract to these names (see ``Cell.model``), so every layout
#: generator in :mod:`photonix.layout.components` has a model here.
MODELS = {
    "straight": straight,
    "bend": bend,
    "bend_euler": bend_euler,
    "taper": taper,
    "route": waveguide_route,
    "waveguide_route": waveguide_route,
    "directional_coupler": directional_coupler,
    "coupler": coupler,
    "evanescent_coupler": evanescent_coupler,
    "mmi1x2": mmi1x2,
    "mmi2x2": mmi2x2,
    "mmi_nxn": mmi_nxn,
    "y_branch": y_branch,
    "crossing": crossing,
    "grating_coupler": grating_coupler,
    "edge_coupler": edge_coupler,
    "phase_shifter": phase_shifter,
    "heater_phase_shifter": heater_phase_shifter,
    "thermo_optic_phase_shifter": thermo_optic_phase_shifter,
    "pn_phase_shifter": pn_phase_shifter,
    "photodetector": photodetector,
    "attenuator": attenuator,
    "terminator": terminator,
    "ring_coupler": ring_coupler,
    "all_pass_ring": all_pass_ring,
    "add_drop_ring": add_drop_ring,
    "ring_single": ring_single,
    "ring_double": ring_double,
    "mzi": mzi,
    "bragg_grating": bragg_grating,
    "awg": awg,
    "polarization_rotator": polarization_rotator,
    "polarization_splitter_rotator": polarization_splitter_rotator,
    "polarization_beam_splitter": polarization_beam_splitter,
    "grating_coupler_2d": grating_coupler_2d,
    "jones_fiber": jones_fiber,
    "multimode_straight": multimode_straight,
    "asymmetric_mode_converter": asymmetric_mode_converter,
    "mode_mux": mode_mux,
}

__all__ = [
    "straight", "bend", "bend_euler", "bend_from_solver", "euler_bend_length", "neff_linear",
    "taper", "waveguide_route",
    "directional_coupler", "coupler", "evanescent_coupler", "supermode_splitting",
    "mmi1x2", "mmi2x2", "mmi_nxn", "y_branch", "crossing",
    "grating_coupler", "edge_coupler", "gaussian_overlap", "SMF28_MFD_1550",
    "phase_shifter", "heater_phase_shifter", "thermo_optic_phase_shifter",
    "pn_phase_shifter", "plasma_dispersion", "built_in_voltage", "depletion_width",
    "silicon_bandgap", "intrinsic_density",
    "photodetector", "responsivity", "photocurrent", "DN_DT_SI", "EPS_R_SI", "NI_SI_300K",
    "attenuator", "terminator",
    "ring_coupler", "all_pass_ring", "add_drop_ring", "ring_single", "ring_double", "mzi",
    "bragg_grating", "bragg_wavelength", "grating_kappa",
    "awg", "awg_design",
    "polarization_rotator", "polarization_splitter_rotator", "polarization_beam_splitter",
    "grating_coupler_2d", "jones_fiber", "pmd_fiber_jones",
    "multimode_straight", "asymmetric_mode_converter", "mode_mux", "coupled_mode_transfer",
    "MODELS", "WAVEGUIDE_PARAMETERS",
]
