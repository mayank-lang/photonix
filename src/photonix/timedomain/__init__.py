"""Time-domain (sample-mode) circuit simulation and transceiver signal integrity.

Complex-envelope simulation about an optical carrier with a fixed time step:

* :mod:`~photonix.timedomain.signal` -- :class:`Waveform` (optical envelopes
  in sqrt(W) and electrical waveforms), :func:`cw`, :func:`gaussian_pulse`.
* :mod:`~photonix.timedomain.elements` -- time-domain elements: memoryless
  S-matrices, fractional-delay lines (Lagrange / Thiran, optional GVD), FIRs,
  exactly discretized pole-residue models, quasi-static time-varying devices.
* :mod:`~photonix.timedomain.convert` -- :func:`to_td` turns any photonix
  model into an element (analytic delay lines, structural rings, passive
  sampled-FIR fits).
* :mod:`~photonix.timedomain.network` -- :class:`TimeDomainCircuit`: steps a
  netlist with feedback loops, time-varying elements and long bit sequences
  (``jax.lax.scan``, differentiable).

Transceiver and signal-integrity library:

* :mod:`~photonix.timedomain.sources` -- PRBS (ITU-T O.150), NRZ/PAM4
  mapping, drive waveforms with rise time and jitter, CW laser with linewidth
  and RIN, single-mode rate-equation laser.
* :mod:`~photonix.timedomain.modulators` -- Mach-Zehnder (push-pull, finite
  extinction, chirp, traveling-wave electrode), electro-absorption and
  coupled-mode-theory ring modulators.
* :mod:`~photonix.timedomain.filters` -- RC, Butterworth and Bessel-Thomson
  electrical filters, exactly discretized.
* :mod:`~photonix.timedomain.receivers` -- photodiode with shot / thermal /
  RIN noise, TIA, clock recovery.
* :mod:`~photonix.timedomain.metrics` -- eye diagrams, Q factor, Gaussian and
  counted BER, extinction ratio, OMA, receiver sensitivity.
"""
from __future__ import annotations

from .convert import FIRDesign, fir_from_model, to_td, to_td_time_varying
from .elements import (
    FIRTD,
    Channel,
    CompiledTD,
    CompositeTD,
    DelayLineTD,
    MemorylessTD,
    PoleResidueTD,
    TDElement,
    TimeVaryingTD,
    centered_frequencies,
    design_fir,
    fir_response,
    lagrange_taps,
    parse_interp,
    planck_taper,
    split_instantaneous,
    tail_support,
    thiran_coefficients,
    window_taps,
)
from .filters import AnalogFilter, linear_recurrence, lowpass
from .metrics import (
    KP4_BER_THRESHOLD,
    EyeDiagram,
    ber_count,
    ber_from_q,
    eye_diagram,
    q_from_ber,
    receiver_sensitivity,
)
from .modulators import RingModulator, TravelingWaveElectrode, eam, mzm, push_pull, ring_modulator
from .network import TimeDomainCircuit, flatten_elements
from .receivers import K_BOLTZMANN, photodiode, recover_clock, tia
from .signal import Waveform, carrier_frequency, cw, gaussian_pulse
from .sources import (
    PRBS_POLYNOMIALS,
    LaserParams,
    cw_laser,
    drive_waveform,
    levels_to_symbols,
    prbs,
    rate_equation_laser,
    symbols_to_levels,
)

__all__ = [
    # signals
    "Waveform", "cw", "gaussian_pulse", "carrier_frequency",
    # elements and FIR design
    "TDElement", "CompiledTD", "Channel", "MemorylessTD", "DelayLineTD", "FIRTD", "PoleResidueTD",
    "TimeVaryingTD", "CompositeTD", "lagrange_taps", "thiran_coefficients", "split_instantaneous",
    "design_fir", "window_taps", "fir_response", "planck_taper", "tail_support", "centered_frequencies",
    "parse_interp",
    # conversion and network
    "to_td", "to_td_time_varying", "fir_from_model", "FIRDesign",
    "TimeDomainCircuit", "flatten_elements",
    # transceiver library
    "PRBS_POLYNOMIALS", "prbs", "symbols_to_levels", "levels_to_symbols", "drive_waveform", "cw_laser",
    "LaserParams", "rate_equation_laser",
    "mzm", "push_pull", "TravelingWaveElectrode", "eam", "RingModulator", "ring_modulator",
    "AnalogFilter", "lowpass", "linear_recurrence",
    "photodiode", "tia", "recover_clock", "K_BOLTZMANN",
    "EyeDiagram", "eye_diagram", "ber_from_q", "q_from_ber", "ber_count", "receiver_sensitivity",
    "KP4_BER_THRESHOLD",
]
