"""External electrothermal and carrier-solver orchestration contracts.

This namespace has no dependency on Elmer, DEVSIM, or commercial solver Python
packages.  Availability checks are side-effect-free; execution happens only via
an explicit :func:`run_job` call.
"""
from __future__ import annotations

from .adapters import (
    DevsimAdapter,
    ElmerAdapter,
    ExternalSolverAdapter,
    LumericalDeviceAdapter,
    SolverAdapter,
    detect_capabilities,
)
from .contracts import MultiphysicsJob, Physics, SimulationResult, SolverCapability
from .fields import FieldDataset, LinearIndexModel, LinearResponseTerm, MeshCellBlock
from .runner import SolverExecutionError, SolverUnavailableError, run_job

__all__ = [
    "Physics",
    "MultiphysicsJob",
    "SolverCapability",
    "SimulationResult",
    "SolverAdapter",
    "ElmerAdapter",
    "DevsimAdapter",
    "LumericalDeviceAdapter",
    "ExternalSolverAdapter",
    "detect_capabilities",
    "SolverUnavailableError",
    "SolverExecutionError",
    "run_job",
    "MeshCellBlock",
    "FieldDataset",
    "LinearResponseTerm",
    "LinearIndexModel",
]
