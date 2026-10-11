"""Time-domain (sample-mode) circuit simulation and transceiver signal integrity.

Complex-envelope simulation about an optical carrier with a fixed time step:

* :mod:`~photonix.timedomain.signal` -- :class:`Waveform` (optical envelopes
  in sqrt(W) and electrical waveforms), :func:`cw`, :func:`gaussian_pulse`.
* :mod:`~photonix.timedomain.elements` -- time-domain elements: memoryless
  S-matrices, fractional-delay lines (Lagrange / Thiran, optional GVD), FIRs,
  exactly discretized pole-residue models, quasi-static time-varying devices.
* :mod:`~photonix.timedomain.convert` -- :func:`to_td` turns any photonix
  model into an element (analytic delay lines, structural rings, sampled
  FIR fits).
* :mod:`~photonix.timedomain.network` -- :class:`TimeDomainCircuit`: steps a
  netlist with feedback loops, time-varying elements and long bit sequences
  (``jax.lax.scan``, differentiable).
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
    lagrange_taps,
    parse_interp,
    planck_taper,
    split_instantaneous,
    tail_support,
    thiran_coefficients,
    window_taps,
)
from .network import TimeDomainCircuit, flatten_elements
from .signal import Waveform, carrier_frequency, cw, gaussian_pulse

__all__ = [
    "Waveform", "cw", "gaussian_pulse", "carrier_frequency",
    "TDElement", "CompiledTD", "Channel", "MemorylessTD", "DelayLineTD", "FIRTD", "PoleResidueTD",
    "TimeVaryingTD", "CompositeTD", "lagrange_taps", "thiran_coefficients", "split_instantaneous",
    "design_fir", "window_taps", "planck_taper", "tail_support", "centered_frequencies", "parse_interp",
    "to_td", "to_td_time_varying", "fir_from_model", "FIRDesign",
    "TimeDomainCircuit", "flatten_elements",
]
