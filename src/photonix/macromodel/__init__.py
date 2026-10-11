"""Rational macromodels of S-parameters: vector fitting, passivity and causality.

Turns sampled S-parameters (field solves, measurements, Touchstone files) into
stable, delayed pole-residue models that evaluate exactly between samples, act
as ordinary Photonix component models, and realize as state-space systems for
time-domain simulation. See ``docs/MACROMODEL.md``.

* :mod:`~photonix.macromodel.model` -- :class:`RationalModel` (delayed
  pole-residue form) with SDict evaluation, state-space export,
  discretization and exact time-domain simulation.
* :mod:`~photonix.macromodel.vectfit` -- delay extraction and relaxed
  complex-baseband (or real RF) vector fitting with automatic order selection.
* :mod:`~photonix.macromodel.passivity` -- Hamiltonian passivity certificate and
  minimum-energy residue-perturbation enforcement.
* :mod:`~photonix.macromodel.causality` -- generalized Kramers-Kronig causality
  test of raw sampled data with rigorous truncation bounds.
"""
from __future__ import annotations

from .causality import CausalityReport, causality_report
from .model import DiscreteStateSpace, RationalModel, StateSpace
from .passivity import PassivityReport, enforce_passivity, hamiltonian_matrix, passivity_report
from .vectfit import estimate_delays, fit_dataset, vector_fit, vector_fit_frequency

__all__ = [
    "RationalModel", "StateSpace", "DiscreteStateSpace",
    "vector_fit", "vector_fit_frequency", "fit_dataset", "estimate_delays",
    "PassivityReport", "passivity_report", "enforce_passivity", "hamiltonian_matrix",
    "CausalityReport", "causality_report",
]
