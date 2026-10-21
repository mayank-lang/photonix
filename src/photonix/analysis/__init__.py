"""Post-processing for photonic design: resonances, filters, link budgets, yield, time domain.

Everything here consumes plain arrays or scattering dictionaries, so it applies
equally to simulated responses and to measured data.

* :mod:`~photonix.analysis.resonances` -- resonance wavelength, FWHM, loaded Q,
  extinction, FSR, finesse, Lorentzian fits, intrinsic/coupling Q split.
* :mod:`~photonix.analysis.filters` -- 1/3/20-dB bandwidths, insertion loss,
  ripple, shape factor, channel crosstalk matrices.
* :mod:`~photonix.analysis.link_budget` -- optical power budgets with margins.
* :mod:`~photonix.analysis.yield_analysis` -- spec-based parametric yield over
  :class:`~photonix.pdk.ProcessStudy` corners and Monte Carlo samples.
* :mod:`~photonix.analysis.time_domain` -- impulse and pulse responses.
* :mod:`~photonix.analysis.sensitivity` -- AD/FD sensitivities, FOSM moments,
  corner designs, worst-case distance and importance-sampling yield.
* :mod:`~photonix.analysis.polarization` -- Jones/Mueller/Stokes, PDL,
  polarization extinction, DGD and principal states.
"""
from __future__ import annotations

from .filters import PassbandMetrics, bandwidth, channel_crosstalk, extinction_ratio_db, passband_metrics, to_db
from .link_budget import LinkBudget, LinkStage, stage_from_sparameters
from .polarization import (
    dgd,
    jones_from_sdict,
    jones_to_mueller,
    pdl_db,
    polarization_extinction_db,
    principal_states,
    stokes,
)
from .resonances import (
    CouplingSolution,
    Resonance,
    find_resonances,
    finesse,
    fit_lorentzian,
    free_spectral_range,
    ring_q_factors,
)
from .sensitivity import (
    ISYield,
    MomentEstimate,
    SensitivityTable,
    WorstCase,
    corners,
    fosm,
    importance_sampling_yield,
    sensitivity,
    worst_case_distance,
)
from .time_domain import PulseResponse, impulse_response, pulse_response
from .yield_analysis import Spec, YieldResult, wilson_interval, yield_analysis

__all__ = [
    "Resonance", "find_resonances", "fit_lorentzian", "free_spectral_range", "finesse",
    "CouplingSolution", "ring_q_factors",
    "to_db", "bandwidth", "PassbandMetrics", "passband_metrics", "channel_crosstalk", "extinction_ratio_db",
    "LinkStage", "LinkBudget", "stage_from_sparameters",
    "Spec", "YieldResult", "yield_analysis", "wilson_interval",
    "PulseResponse", "impulse_response", "pulse_response",
    "SensitivityTable", "sensitivity", "MomentEstimate", "fosm", "corners",
    "WorstCase", "worst_case_distance", "ISYield", "importance_sampling_yield",
    "jones_from_sdict", "pdl_db", "polarization_extinction_db", "stokes", "jones_to_mueller", "dgd",
    "principal_states",
]
